//! Partial lane ORAM implementation.
//!
//! Only the block and bucket layout and wide-tree path I/O are implemented.
//! The lane ORAM access and eviction algorithms will be added incrementally.
//!
//! Slot `i` across every bucket forms lane `i`. Path buffers are currently
//! bucket-major: all slots of one bucket are contiguous before the next bucket.

use std::mem::{align_of, offset_of, size_of};

use bytemuck::{Pod, Zeroable};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m512i;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use core::arch::x86_64::{
  _mm512_cmpeq_epi32_mask, _mm512_cmpeq_epi64_mask, _mm512_load_si512, _mm512_mask_mov_epi32,
  _mm512_mask_mov_epi64, _mm512_permutexvar_epi32, _mm512_permutexvar_epi64, _mm512_store_si512,
};
use rostl_primitives::{
  cmov_body, cxchg_body, impl_cmov_for_pod,
  traits::{_Cmovbase, Cmov},
};

use crate::{prelude::PositionType, wide_heap_tree::WideHeapTree};

/// Default number of lanes (blocks per bucket).
pub const DEFAULT_Z: usize = 2;

/// Invalid position for a [`Block64`].
pub const DUMMY_POS64: u64 = u64::MAX;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const BLOCK32_KEY_INDICES: __m512i = unsafe { std::mem::transmute([1u32; 16]) };

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const BLOCK32_DUMMY_POS: __m512i = unsafe { std::mem::transmute([u32::MAX; 16]) };

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const BLOCK64_KEY_INDICES: __m512i = unsafe { std::mem::transmute([1u64; 8]) };

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const BLOCK64_DUMMY_POS: __m512i = unsafe { std::mem::transmute([u64::MAX; 8]) };

/// A one-cache-line block with a 32-bit position and key.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Block32 {
  /// Position assigned to this block. [`PositionType::MAX`] denotes an empty block.
  pub pos: PositionType,
  /// Logical block key.
  pub key: u32,
  /// Untyped payload filling the remainder of the cache line.
  pub data: [u8; 56],
}

unsafe impl Zeroable for Block32 {}
unsafe impl Pod for Block32 {}

impl_cmov_for_pod!(Block32);

impl Default for Block32 {
  fn default() -> Self {
    Self { pos: PositionType::MAX, key: 0, data: [0; 56] }
  }
}

impl Block32 {
  /// Returns whether this block is empty.
  pub fn is_empty(&self) -> bool {
    self.pos == PositionType::MAX
  }

  /// Loads the entire block into one aligned AVX-512 register.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline]
  pub fn load_avx512(&self) -> __m512i {
    // SAFETY: Block32 is exactly 64 bytes and has 64-byte alignment.
    unsafe { _mm512_load_si512(self as *const Self as *const __m512i) }
  }

  /// TODO: Non-AVX-512 load fallback.
  #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
  #[inline]
  pub fn load_avx512(&self) -> __m512i {
    todo!("implement the non-AVX-512 Block32 load")
  }

  /// Stores one AVX-512 register into the entire aligned block.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline]
  pub fn store_avx512(&mut self, value: __m512i) {
    // SAFETY: Block32 is exactly 64 bytes and has 64-byte alignment.
    unsafe { _mm512_store_si512(self as *mut Self as *mut __m512i, value) }
  }

  /// TODO: Non-AVX-512 store fallback.
  #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
  #[inline]
  pub fn store_avx512(&mut self, _value: __m512i) {
    todo!("implement the non-AVX-512 Block32 store")
  }

  /// Copies this block into `ret` and marks it empty when its key matches.
  ///
  /// Every 32-bit lane of `desired_key` must contain the desired key.
  ///
  /// TODO: Consider also requiring `pos != PositionType::MAX`. A dedicated
  /// dummy key can make a key-only comparison sufficient for now, but that
  /// invariant prevents callers from using the entire key space.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline]
  pub fn read_if_key_matches_and_set_dummy_pos(&mut self, desired_key: __m512i, ret: &mut __m512i) {
    // SAFETY: This method is only compiled when AVX-512F is enabled.
    unsafe {
      let block = self.load_avx512();
      let block_key = _mm512_permutexvar_epi32(BLOCK32_KEY_INDICES, block);
      let matches = _mm512_cmpeq_epi32_mask(block_key, desired_key);

      *ret = _mm512_mask_mov_epi32(*ret, matches, block);

      let block_with_dummy_pos = _mm512_mask_mov_epi32(block, matches & 1, BLOCK32_DUMMY_POS);
      self.store_avx512(block_with_dummy_pos);
    }
  }

  /// TODO: Non-AVX-512 conditional-read fallback.
  #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
  #[inline]
  pub fn read_if_key_matches_and_set_dummy_pos(
    &mut self,
    _desired_key: __m512i,
    _ret: &mut __m512i,
  ) {
    todo!("implement the non-AVX-512 Block32 conditional read")
  }
}

/// Reads and removes the block matching `block.key` from a combined stash and path.
///
/// `stash_and_path` is one contiguous array laid out as `[stash | loaded path]`.
/// The public stash offset is intentionally not needed here because every block
/// is scanned. `block` may contain only the lookup key or a complete block that
/// will be reinserted later. The returned block equals `block` when no key
/// matches and equals the matching full block when one does.
///
/// The caller must maintain the invariant that at most one non-dummy block has
/// the requested key.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
pub fn read_and_remove_path32(block: &Block32, stash_and_path: &mut [Block32]) -> Block32 {
  let mut result = block.load_avx512();

  // SAFETY: This function is only compiled when AVX-512F is enabled.
  let desired_key = unsafe { _mm512_permutexvar_epi32(BLOCK32_KEY_INDICES, result) };

  for candidate in stash_and_path {
    candidate.read_if_key_matches_and_set_dummy_pos(desired_key, &mut result);
  }

  let mut result_block = *block;
  result_block.store_avx512(result);
  result_block
}

/// TODO: Non-AVX-512 combined stash/path conditional-read fallback.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
#[inline]
pub fn read_and_remove_path32(_block: &Block32, _stash_and_path: &mut [Block32]) -> Block32 {
  todo!("implement the non-AVX-512 read_and_remove_path32")
}

/// A one-cache-line block with a 64-bit position and key.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Block64 {
  /// Position assigned to this block. [`DUMMY_POS64`] denotes an empty block.
  pub pos: u64,
  /// Logical block key.
  pub key: u64,
  /// Untyped payload filling the remainder of the cache line.
  pub data: [u8; 48],
}

unsafe impl Zeroable for Block64 {}
unsafe impl Pod for Block64 {}

impl_cmov_for_pod!(Block64);

impl Default for Block64 {
  fn default() -> Self {
    Self { pos: DUMMY_POS64, key: 0, data: [0; 48] }
  }
}

impl Block64 {
  /// Returns whether this block is empty.
  pub fn is_empty(&self) -> bool {
    self.pos == DUMMY_POS64
  }

  /// Loads the entire block into one aligned AVX-512 register.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline]
  pub fn load_avx512(&self) -> __m512i {
    // SAFETY: Block64 is exactly 64 bytes and has 64-byte alignment.
    unsafe { _mm512_load_si512(self as *const Self as *const __m512i) }
  }

  /// TODO: Non-AVX-512 load fallback.
  #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
  #[inline]
  pub fn load_avx512(&self) -> __m512i {
    todo!("implement the non-AVX-512 Block64 load")
  }

  /// Stores one AVX-512 register into the entire aligned block.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline]
  pub fn store_avx512(&mut self, value: __m512i) {
    // SAFETY: Block64 is exactly 64 bytes and has 64-byte alignment.
    unsafe { _mm512_store_si512(self as *mut Self as *mut __m512i, value) }
  }

  /// TODO: Non-AVX-512 store fallback.
  #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
  #[inline]
  pub fn store_avx512(&mut self, _value: __m512i) {
    todo!("implement the non-AVX-512 Block64 store")
  }

  /// Copies this block into `ret` and marks it empty when its key matches.
  ///
  /// Every 64-bit lane of `desired_key` must contain the desired key.
  ///
  /// TODO: Consider also requiring `pos != DUMMY_POS64`. A dedicated dummy
  /// key can make a key-only comparison sufficient for now, but that invariant
  /// prevents callers from using the entire key space.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline]
  pub fn read_if_key_matches_and_set_dummy_pos(&mut self, desired_key: __m512i, ret: &mut __m512i) {
    // SAFETY: This method is only compiled when AVX-512F is enabled.
    unsafe {
      let block = self.load_avx512();
      let block_key = _mm512_permutexvar_epi64(BLOCK64_KEY_INDICES, block);
      let matches = _mm512_cmpeq_epi64_mask(block_key, desired_key);

      *ret = _mm512_mask_mov_epi64(*ret, matches, block);

      let block_with_dummy_pos = _mm512_mask_mov_epi64(block, matches & 1, BLOCK64_DUMMY_POS);
      self.store_avx512(block_with_dummy_pos);
    }
  }

  /// TODO: Non-AVX-512 conditional-read fallback.
  #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
  #[inline]
  pub fn read_if_key_matches_and_set_dummy_pos(
    &mut self,
    _desired_key: __m512i,
    _ret: &mut __m512i,
  ) {
    todo!("implement the non-AVX-512 Block64 conditional read")
  }
}

/// Reads and removes the block matching `block.key` from a combined stash and path.
///
/// `stash_and_path` is one contiguous array laid out as `[stash | loaded path]`.
/// The public stash offset is intentionally not needed here because every block
/// is scanned. `block` may contain only the lookup key or a complete block that
/// will be reinserted later. The returned block equals `block` when no key
/// matches and equals the matching full block when one does.
///
/// The caller must maintain the invariant that at most one non-dummy block has
/// the requested key.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
pub fn read_and_remove_path64(block: &Block64, stash_and_path: &mut [Block64]) -> Block64 {
  let mut result = block.load_avx512();

  // SAFETY: This function is only compiled when AVX-512F is enabled.
  let desired_key = unsafe { _mm512_permutexvar_epi64(BLOCK64_KEY_INDICES, result) };

  for candidate in stash_and_path {
    candidate.read_if_key_matches_and_set_dummy_pos(desired_key, &mut result);
  }

  let mut result_block = *block;
  result_block.store_avx512(result);
  result_block
}

/// TODO: Non-AVX-512 combined stash/path conditional-read fallback.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
#[inline]
pub fn read_and_remove_path64(_block: &Block64, _stash_and_path: &mut [Block64]) -> Block64 {
  todo!("implement the non-AVX-512 read_and_remove_path64")
}

const _: () = assert!(size_of::<Block32>() == 64);
const _: () = assert!(align_of::<Block32>() == 64);
const _: () = assert!(offset_of!(Block32, pos) == 0);
const _: () = assert!(offset_of!(Block32, key) == 4);
const _: () = assert!(offset_of!(Block32, data) == 8);
const _: () = assert!(size_of::<Block64>() == 64);
const _: () = assert!(align_of::<Block64>() == 64);
const _: () = assert!(offset_of!(Block64, pos) == 0);
const _: () = assert!(offset_of!(Block64, key) == 8);
const _: () = assert!(offset_of!(Block64, data) == 16);

/// A bucket in the lane ORAM tree.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Bucket<const Z: usize = DEFAULT_Z>(pub [Block32; Z]);

impl<const Z: usize> Default for Bucket<Z> {
  fn default() -> Self {
    Self([Block32::default(); Z])
  }
}

impl<const Z: usize> WideHeapTree<Bucket<Z>> {
  /// Reads every bucket on `path` into `out`, ordered from root to leaf.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline]
  pub fn read_path(&self, path: PositionType, out: &mut [Block32]) {
    debug_assert!((path as usize) < self.branching_factor.pow((self.height - 1) as u32));
    debug_assert!(out.len() == self.height * Z);

    for depth in 0..self.height {
      let index = self.get_index(depth, path);
      let bucket = &self.tree[index];

      for slot in 0..Z {
        let block = bucket.0[slot].load_avx512();
        out[depth * Z + slot].store_avx512(block);
      }
    }
  }

  /// Reads every bucket on `path` without AVX-512 support.
  #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
  #[inline]
  pub fn read_path(&self, path: PositionType, out: &mut [Block32]) {
    debug_assert!((path as usize) < self.branching_factor.pow((self.height - 1) as u32));
    debug_assert!(out.len() == self.height * Z);

    for depth in 0..self.height {
      let index = self.get_index(depth, path);
      let bucket = &self.tree[index];
      out[depth * Z..(depth + 1) * Z].copy_from_slice(&bucket.0);
    }
  }

  /// Writes root-to-leaf block data from `input` to every bucket on `path`.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline]
  pub fn write_path(&mut self, path: PositionType, input: &[Block32]) {
    debug_assert!((path as usize) < self.branching_factor.pow((self.height - 1) as u32));
    debug_assert!(input.len() == self.height * Z);

    for depth in 0..self.height {
      let index = self.get_index(depth, path);
      let bucket = &mut self.tree[index];

      for slot in 0..Z {
        let block = input[depth * Z + slot].load_avx512();
        bucket.0[slot].store_avx512(block);
      }
    }
  }

  /// Writes root-to-leaf block data without AVX-512 support.
  #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
  #[inline]
  pub fn write_path(&mut self, path: PositionType, input: &[Block32]) {
    debug_assert!((path as usize) < self.branching_factor.pow((self.height - 1) as u32));
    debug_assert!(input.len() == self.height * Z);

    for depth in 0..self.height {
      let index = self.get_index(depth, path);
      let bucket = &mut self.tree[index];
      bucket.0.copy_from_slice(&input[depth * Z..(depth + 1) * Z]);
    }
  }
}

/// Skeleton for the 32-bit lane ORAM.
#[derive(Debug)]
pub struct LaneORAM<const Z: usize = DEFAULT_Z> {
  /// Wide tree holding lane ORAM buckets.
  pub tree: WideHeapTree<Bucket<Z>>,
}

impl<const Z: usize> LaneORAM<Z> {
  /// Creates the tree backing a lane ORAM.
  pub fn new(height: usize, branching_factor: usize) -> Self {
    Self { tree: WideHeapTree::new(height, branching_factor) }
  }
}

#[cfg(test)]
mod tests {
  use std::mem::{align_of, size_of};

  use super::{Block32, Block64, LaneORAM};

  #[test]
  fn blocks_are_exactly_one_cache_line() {
    assert_eq!(size_of::<Block32>(), 64);
    assert_eq!(align_of::<Block32>(), 64);
    assert_eq!(size_of::<Block64>(), 64);
    assert_eq!(align_of::<Block64>(), 64);
  }

  #[test]
  fn reads_and_writes_a_wide_tree_path() {
    const Z: usize = 3;
    let mut oram = LaneORAM::<Z>::new(3, 3);
    let mut input = [Block32::default(); 3 * Z];

    for index in 0..input.len() {
      input[index].pos = 7;
      input[index].key = index as u32;
      input[index].data[0] = index as u8 + 10;
    }

    oram.tree.write_path(7, &input);

    let mut output = [Block32::default(); 3 * Z];
    oram.tree.read_path(7, &mut output);

    for index in 0..output.len() {
      assert_eq!(output[index].pos, input[index].pos);
      assert_eq!(output[index].key, input[index].key);
      assert_eq!(output[index].data, input[index].data);
    }
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[test]
  fn reads_and_removes_matching_block_from_combined_array() {
    let mut requested = Block32::default();
    requested.key = 9;

    let mut stash_and_path = [Block32::default(); 4];
    stash_and_path[2].pos = 17;
    stash_and_path[2].key = 9;
    stash_and_path[2].data[0] = 42;

    let result = super::read_and_remove_path32(&requested, &mut stash_and_path);

    assert_eq!(result.pos, 17);
    assert_eq!(result.key, 9);
    assert_eq!(result.data[0], 42);
    assert_eq!(requested.pos, u32::MAX);
    assert!(stash_and_path[2].is_empty());
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[test]
  fn reads_and_removes_matching_block64_from_combined_array() {
    let mut requested = Block64::default();
    requested.key = 9;

    let mut stash_and_path = [Block64::default(); 4];
    stash_and_path[2].pos = 17;
    stash_and_path[2].key = 9;
    stash_and_path[2].data[0] = 42;

    let result = super::read_and_remove_path64(&requested, &mut stash_and_path);

    assert_eq!(result.pos, 17);
    assert_eq!(result.key, 9);
    assert_eq!(result.data[0], 42);
    assert_eq!(requested.pos, super::DUMMY_POS64);
    assert!(stash_and_path[2].is_empty());
  }
}
