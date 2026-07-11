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
  __mmask16, _mm512_cmpeq_epi32_mask, _mm512_cmpeq_epi64_mask, _mm512_load_si512,
  _mm512_mask_mov_epi32, _mm512_mask_mov_epi64, _mm512_permutexvar_epi32, _mm512_permutexvar_epi64,
  _mm512_store_si512,
};
use rostl_primitives::{
  cmov_body, cxchg_body, impl_cmov_for_pod,
  traits::{_Cmovbase, Cmov},
};

use crate::{prelude::PositionType, wide_heap_tree::WideHeapTree};

/// Default number of lanes (blocks per bucket).
pub const DEFAULT_Z: usize = 3;

/// Default number of blocks reserved for the stash.
pub const DEFAULT_S: usize = 20;

/// Default branching factor of the wide tree.
pub const DEFAULT_B: usize = 2;

/// Invalid position for a [`Block64`].
pub const DUMMY_POS64: u64 = u64::MAX;

/// Key reserved for dummy [`Block32`] values.
///
/// This is a temporary invariant. Supporting the full 32-bit key space will
/// require key matching to also reject blocks whose position is dummy.
pub const DUMMY_KEY32: u32 = u32::MAX;

/// Key reserved for dummy [`Block64`] values.
///
/// This is a temporary invariant. Supporting the full 64-bit key space will
/// require key matching to also reject blocks whose position is dummy.
pub const DUMMY_KEY64: u64 = u64::MAX;

/// Position type used by the active lane ORAM configuration.
pub type PosType = PositionType;
/// Key type used by the active lane ORAM configuration.
pub type KeyType = u32;
/// Cache-line block used by the active lane ORAM configuration.
pub type BlockType = Block32;
/// All-zero or all-one AVX-512 lane-movement mask.
pub type LaneMask = u16;
/// Dummy position used by the active lane ORAM configuration.
pub const DUMMY_POS: PosType = PosType::MAX;
/// Dummy key used by the active lane ORAM configuration.
pub const DUMMY_KEY: KeyType = DUMMY_KEY32;
/// Initial payload passed to `update` when a key is absent.
pub const EMPTY_BLOCK_DATA: [u8; 56] = [0; 56];

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const BLOCK32_KEY_INDICES: __m512i = unsafe { std::mem::transmute([1u32; 16]) };

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const BLOCK32_POS_INDICES: __m512i = unsafe { std::mem::transmute([0u32; 16]) };

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
    Self { pos: PositionType::MAX, key: DUMMY_KEY32, data: [u8::MAX; 56] }
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

      let block_with_dummy_pos = _mm512_mask_mov_epi32(block, matches, BLOCK32_DUMMY_POS);
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
    Self { pos: DUMMY_POS64, key: DUMMY_KEY64, data: [u8::MAX; 48] }
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

      let block_with_dummy_pos = _mm512_mask_mov_epi64(block, matches, BLOCK64_DUMMY_POS);
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
fn read_and_remove_configured(block: &BlockType, stash_and_path: &mut [BlockType]) -> BlockType {
  read_and_remove_path32(block, stash_and_path)
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
pub struct Bucket<const Z: usize = DEFAULT_Z>(pub [BlockType; Z]);

impl<const Z: usize> Default for Bucket<Z> {
  fn default() -> Self {
    Self([BlockType::default(); Z])
  }
}

impl<const Z: usize> WideHeapTree<Bucket<Z>> {
  /// Reads every bucket on `path` into `out`, ordered from root to leaf.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline]
  pub fn read_path(&self, path: PosType, out: &mut [BlockType]) {
    debug_assert!((path as usize) < self.path_count());
    debug_assert!(out.len() == self.height * Z);

    let mut out_index = 0;
    for depth in 0..self.height {
      let index = self.get_index(depth, path);
      let bucket = &self.tree[index];

      for slot in 0..Z {
        let block = bucket.0[slot].load_avx512();
        out[out_index].store_avx512(block);
        out_index += 1;
      }
    }
  }

  /// Reads every bucket on `path` without AVX-512 support.
  #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
  #[inline]
  pub fn read_path(&self, path: PosType, out: &mut [BlockType]) {
    debug_assert!((path as usize) < self.path_count());
    debug_assert!(out.len() == self.height * Z);

    let mut out_index = 0;
    for depth in 0..self.height {
      let index = self.get_index(depth, path);
      let bucket = &self.tree[index];
      out[out_index..out_index + Z].copy_from_slice(&bucket.0);
      out_index += Z;
    }
  }

  /// Writes root-to-leaf block data from `input` to every bucket on `path`.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline]
  pub fn write_path(&mut self, path: PosType, input: &[BlockType]) {
    debug_assert!((path as usize) < self.path_count());
    debug_assert!(input.len() == self.height * Z);

    let mut input_index = 0;
    for depth in 0..self.height {
      let index = self.get_index(depth, path);
      let bucket = &mut self.tree[index];

      for slot in 0..Z {
        let block = input[input_index].load_avx512();
        bucket.0[slot].store_avx512(block);
        input_index += 1;
      }
    }
  }

  /// Writes root-to-leaf block data without AVX-512 support.
  #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
  #[inline]
  pub fn write_path(&mut self, path: PosType, input: &[BlockType]) {
    debug_assert!((path as usize) < self.path_count());
    debug_assert!(input.len() == self.height * Z);

    let mut input_index = 0;
    for depth in 0..self.height {
      let index = self.get_index(depth, path);
      let bucket = &mut self.tree[index];
      bucket.0.copy_from_slice(&input[input_index..input_index + Z]);
      input_index += Z;
    }
  }
}

/// Partial 32-bit lane ORAM.
#[derive(Debug)]
pub struct LaneORAM<
  const Z: usize = DEFAULT_Z,
  const S: usize = DEFAULT_S,
  const B: usize = DEFAULT_B,
> {
  /// Logical capacity, rounded up to a power of `B`.
  pub max_n: usize,
  /// Height of the wide tree; a root-only tree has height 1.
  pub h: usize,
  /// Wide tree holding lane ORAM buckets.
  pub tree: WideHeapTree<Bucket<Z>>,
  /// Combined `[stash | loaded path]` buffer.
  ///
  /// The first `S` blocks are the stash. The remaining `tree.height * Z`
  /// blocks are scratch space for one loaded path.
  pub stash_and_path: Vec<BlockType>,
  /// Preallocated bucket-major masks used by lane eviction.
  pub lane_masks: Vec<LaneMask>,
}

impl<const Z: usize, const S: usize, const B: usize> LaneORAM<Z, S, B> {
  /// Creates an empty lane ORAM for at least `max_n` logical blocks.
  pub fn new(max_n: usize) -> Self {
    debug_assert!(max_n > 0);
    debug_assert!(Z > 0);
    debug_assert!(B >= 2);
    debug_assert!(B.is_power_of_two());

    let mut rounded_max_n = 1usize;
    let mut height = 1usize;
    while rounded_max_n < max_n {
      rounded_max_n *= B;
      height += 1;
    }

    let tree = WideHeapTree::new(height, B);
    let stash_and_path = vec![BlockType::default(); S + height * Z];
    let lane_masks = vec![0; height * Z];
    Self { max_n: rounded_max_n, h: height, tree, stash_and_path, lane_masks }
  }

  /// Moves the first non-dummy stash block into each free root lane.
  ///
  /// The path must already be loaded into `stash_and_path[S..]`. Every lane
  /// scans the entire stash, even after a block has been moved.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  pub fn move_from_stash_to_free_lanes(&mut self) {
    let mut root_index = S;

    // SAFETY: This method is only compiled when AVX-512F is enabled.
    unsafe {
      for _ in 0..Z {
        let mut root_register = self.stash_and_path[root_index].load_avx512();
        let root_pos = _mm512_permutexvar_epi32(BLOCK32_POS_INDICES, root_register);
        let root_empty_mask = _mm512_cmpeq_epi32_mask(root_pos, BLOCK32_DUMMY_POS);
        let mut moved_mask: __mmask16 = !root_empty_mask;

        for stash_index in 0..S {
          let stash_register = self.stash_and_path[stash_index].load_avx512();
          let stash_pos = _mm512_permutexvar_epi32(BLOCK32_POS_INDICES, stash_register);
          let stash_empty_mask = _mm512_cmpeq_epi32_mask(stash_pos, BLOCK32_DUMMY_POS);
          let stash_non_dummy_mask = !stash_empty_mask;
          let move_mask = stash_non_dummy_mask & !moved_mask;

          root_register = _mm512_mask_mov_epi32(root_register, move_mask, stash_register);
          let emptied_stash = _mm512_mask_mov_epi32(stash_register, move_mask, BLOCK32_DUMMY_POS);
          self.stash_and_path[stash_index].store_avx512(emptied_stash);
          moved_mask |= stash_non_dummy_mask;
        }

        self.stash_and_path[root_index].store_avx512(root_register);
        root_index += 1;
      }
    }
  }

  /// TODO: Non-AVX-512 stash-to-root-lanes fallback.
  #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
  pub fn move_from_stash_to_free_lanes(&mut self) {
    todo!("implement the non-AVX-512 stash-to-root-lanes move")
  }

  /// Moves blocks down each lane according to precomputed masks.
  ///
  /// The path must already be loaded into `stash_and_path[S..]`. `masks` uses
  /// the same bucket-major layout as the path: `[level * Z + lane]`. Each mask
  /// must be either all zeroes or all ones. One held register per lane starts
  /// as a dummy block.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  pub fn move_down_lane(&mut self) {
    debug_assert!(self.lane_masks.len() == self.h * Z);
    let mut held = [BLOCK32_DUMMY_POS; Z];
    let mut path_index = 0;
    let mut block_index = S;

    // SAFETY: This method is only compiled when AVX-512F is enabled.
    unsafe {
      for _ in 0..self.h {
        for lane in 0..Z {
          let mask = self.lane_masks[path_index];
          debug_assert!((mask == 0) | (mask == LaneMask::MAX));

          let block = self.stash_and_path[block_index].load_avx512();
          let next_held = _mm512_mask_mov_epi32(held[lane], mask, block);
          let next_block = _mm512_mask_mov_epi32(block, mask, held[lane]);
          held[lane] = next_held;
          self.stash_and_path[block_index].store_avx512(next_block);
          path_index += 1;
          block_index += 1;
        }
      }

      for lane in 0..Z {
        let held_pos = _mm512_permutexvar_epi32(BLOCK32_POS_INDICES, held[lane]);
        let held_empty_mask = _mm512_cmpeq_epi32_mask(held_pos, BLOCK32_DUMMY_POS);
        debug_assert!(held_empty_mask == LaneMask::MAX);
      }
    }
  }

  /// TODO: Non-AVX-512 lane movement fallback.
  #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
  pub fn move_down_lane(&mut self) {
    todo!("implement the non-AVX-512 lane movement")
  }

  /// Calculates the masked swaps used to evict blocks down each lane.
  ///
  /// The path must already be loaded into `stash_and_path[S..]`. Masks use the
  /// same bucket-major layout as the path. Levels are processed from leaf to
  /// root, while all lanes at one level are processed contiguously.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  pub fn calculate_masks(&mut self, path: PosType) {
    debug_assert!((path as usize) < self.max_n);
    debug_assert!(self.lane_masks.len() == self.h * Z);

    let bits_per_digit = B.trailing_zeros();
    let position_bits = ((self.h - 1) as u32) * bits_per_digit;
    debug_assert!(position_bits <= PosType::BITS);
    let unused_high_bits = PosType::BITS - position_bits;
    let mut target_bits = [u32::MAX; Z];
    let mut level_start = self.h * Z;

    for level in (0..self.h).rev() {
      level_start -= Z;
      let mut path_index = level_start;
      let mut block_index = S + level_start;

      for lane in 0..Z {
        let block = &self.stash_and_path[block_index];
        let pos = block.pos;
        let is_empty = pos == DUMMY_POS;

        let differing_bits = (pos ^ path).wrapping_shl(unused_high_bits);
        let matching_prefix_bits = differing_bits.leading_zeros();
        let can_reach_target = matching_prefix_bits >= target_bits[lane];
        let set_mask = is_empty | ((!is_empty) & can_reach_target);
        let current_target_bits = (level as u32) * bits_per_digit;
        target_bits[lane].cmov(&current_target_bits, set_mask);
        self.lane_masks[path_index] = 0u16.wrapping_sub(set_mask as u16);
        path_index += 1;
        block_index += 1;
      }
    }
  }

  /// TODO: Non-AVX-512 mask calculation fallback.
  #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
  pub fn calculate_masks(&mut self, _path: PosType) {
    todo!("implement the non-AVX-512 lane mask calculation")
  }

  /// Updates a block and assigns it `new_pos`, inserting a zeroed payload when absent.
  ///
  /// The closure may update the entire cache-line block, but changes to its
  /// position and key are discarded; `new_pos` and `key` always become the
  /// stored metadata.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  pub fn update<T, F>(
    &mut self,
    pos: PosType,
    new_pos: PosType,
    key: KeyType,
    update_func: F,
  ) -> (bool, T)
  where
    F: FnOnce(&mut BlockType) -> T,
  {
    debug_assert!((pos as usize) < self.max_n);
    debug_assert!((new_pos as usize) < self.max_n);
    debug_assert!(key != DUMMY_KEY);

    self.tree.read_path(pos, &mut self.stash_and_path[S..]);

    let lookup = BlockType { pos: DUMMY_POS, key, data: EMPTY_BLOCK_DATA };
    let mut block = read_and_remove_configured(&lookup, &mut self.stash_and_path);
    let found = !block.is_empty();
    let result = update_func(&mut block);
    block.pos = new_pos;
    block.key = key;

    let block_register = block.load_avx512();
    let mut previous_empty_mask: __mmask16 = 0;

    // SAFETY: This method is only compiled when AVX-512F is enabled.
    unsafe {
      for index in 0..S {
        let candidate = &mut self.stash_and_path[index];
        let candidate_register = candidate.load_avx512();
        let candidate_pos = _mm512_permutexvar_epi32(BLOCK32_POS_INDICES, candidate_register);
        let current_empty_mask = _mm512_cmpeq_epi32_mask(candidate_pos, BLOCK32_DUMMY_POS);
        let write_mask = current_empty_mask & !previous_empty_mask;
        let updated = _mm512_mask_mov_epi32(candidate_register, write_mask, block_register);
        candidate.store_avx512(updated);
        previous_empty_mask |= current_empty_mask;
      }
    }

    debug_assert!(previous_empty_mask != 0);

    self.move_from_stash_to_free_lanes();
    self.calculate_masks(pos);
    self.move_down_lane();
    self.tree.write_path(pos, &self.stash_and_path[S..]);
    (found, result)
  }

  /// TODO: Non-AVX-512 update fallback.
  #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
  pub fn update<T, F>(
    &mut self,
    _pos: PosType,
    _new_pos: PosType,
    _key: KeyType,
    _update_func: F,
  ) -> (bool, T)
  where
    F: FnOnce(&mut BlockType) -> T,
  {
    todo!("implement the non-AVX-512 update")
  }
}

#[cfg(test)]
mod tests {
  use std::mem::{align_of, size_of};

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  use rand::{rngs::StdRng, Rng, SeedableRng};

  use super::{Block32, Block64, LaneORAM};

  #[test]
  fn blocks_are_exactly_one_cache_line() {
    assert_eq!(size_of::<Block32>(), 64);
    assert_eq!(align_of::<Block32>(), 64);
    assert_eq!(size_of::<Block64>(), 64);
    assert_eq!(align_of::<Block64>(), 64);
  }

  #[test]
  fn default_blocks_set_every_simd_lane_to_dummy() {
    let block32 = Block32::default();
    let block64 = Block64::default();
    assert_eq!(bytemuck::bytes_of(&block32), &[u8::MAX; 64]);
    assert_eq!(bytemuck::bytes_of(&block64), &[u8::MAX; 64]);
  }

  #[test]
  fn reads_and_writes_a_wide_tree_path() {
    const Z: usize = 3;
    let mut oram = LaneORAM::<Z, 5, 4>::new(16);
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
    assert_eq!(stash_and_path[2].key, super::DUMMY_KEY32);
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
    assert_eq!(stash_and_path[2].key, super::DUMMY_KEY64);
  }

  #[test]
  fn lane_oram_allocates_stash_and_path() {
    let oram = LaneORAM::<3, 5, 4>::new(10);
    assert_eq!(oram.max_n, 16);
    assert_eq!(oram.h, 3);
    assert_eq!(oram.tree.height, 3);
    assert_eq!(oram.tree.branching_factor, 4);
    assert_eq!(oram.stash_and_path.len(), 5 + 3 * 3);
    assert_eq!(oram.lane_masks.len(), 3 * 3);
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[test]
  fn update_inserts_then_updates_block32() {
    let mut oram = LaneORAM::<2, 4, 4>::new(16);

    let (found, old) = oram.update(0, 3, 7, |block| {
      let old = block.data[0];
      block.data[0] = 41;
      old
    });
    assert!(!found);
    assert_eq!(old, 0);

    let (found, old) = oram.update(3, 5, 7, |block| {
      let old = block.data[0];
      block.data[0] = 42;
      old
    });
    assert!(found);
    assert_eq!(old, 41);

    let (found, old) = oram.update(5, 6, 7, |block| block.data[0]);
    assert!(found);
    assert_eq!(old, 42);
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[test]
  fn moves_first_stash_block_only_into_free_root_lane() {
    const Z: usize = 2;
    const S: usize = 4;
    let mut oram = LaneORAM::<Z, S, 4>::new(16);

    oram.stash_and_path[0].pos = 3;
    oram.stash_and_path[0].key = 10;
    oram.stash_and_path[1].pos = 7;
    oram.stash_and_path[1].key = 11;

    let root_lane_1 = S + 1;
    oram.stash_and_path[root_lane_1].pos = 12;
    oram.stash_and_path[root_lane_1].key = 20;

    oram.move_from_stash_to_free_lanes();

    let root_lane_0 = S;
    assert_eq!(oram.stash_and_path[root_lane_0].pos, 3);
    assert_eq!(oram.stash_and_path[root_lane_0].key, 10);
    assert!(oram.stash_and_path[0].is_empty());

    assert_eq!(oram.stash_and_path[root_lane_1].pos, 12);
    assert_eq!(oram.stash_and_path[root_lane_1].key, 20);
    assert_eq!(oram.stash_and_path[1].pos, 7);
    assert_eq!(oram.stash_and_path[1].key, 11);
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[test]
  fn moves_down_all_lanes_in_level_major_order() {
    const Z: usize = 2;
    const S: usize = 2;
    let mut oram = LaneORAM::<Z, S, 4>::new(16);

    oram.stash_and_path[S].pos = 1;
    oram.stash_and_path[S].key = 10;
    oram.stash_and_path[S + Z].pos = 2;
    oram.stash_and_path[S + Z].key = 11;

    oram.stash_and_path[S + 1].pos = 3;
    oram.stash_and_path[S + 1].key = 20;

    let masks = [u16::MAX, 0, u16::MAX, 0, u16::MAX, 0];
    oram.lane_masks.copy_from_slice(&masks);
    oram.move_down_lane();

    assert!(oram.stash_and_path[S].is_empty());
    assert_eq!(oram.stash_and_path[S + Z].key, 10);
    assert_eq!(oram.stash_and_path[S + 2 * Z].key, 11);

    assert_eq!(oram.stash_and_path[S + 1].key, 20);
    assert!(oram.stash_and_path[S + Z + 1].is_empty());
    assert!(oram.stash_and_path[S + 2 * Z + 1].is_empty());
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[test]
  fn calculates_lane_masks_from_leaf_to_root() {
    const Z: usize = 2;
    const S: usize = 2;
    let mut oram = LaneORAM::<Z, S, 4>::new(16);

    oram.stash_and_path[S].pos = 5;
    oram.stash_and_path[S + Z].pos = 5;

    oram.stash_and_path[S + 1].pos = 0;
    oram.stash_and_path[S + Z + 1].pos = 0;
    oram.stash_and_path[S + 2 * Z + 1].pos = 0;

    oram.calculate_masks(5);

    assert_eq!(oram.lane_masks, [u16::MAX, 0, u16::MAX, 0, u16::MAX, 0]);
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[test]
  fn randomized_updates_round_trip_cacheline_payloads() {
    const N: usize = 16;
    let mut oram = LaneORAM::<4, 20, 4>::new(N);
    let mut positions = [0u32; N];
    let mut values = [[u8::MAX; 56]; N];
    let mut rng = StdRng::seed_from_u64(1);

    for key in 1..N {
      let new_pos = rng.random_range(0..oram.max_n) as u32;
      let mut new_value = [0u8; 56];
      rng.fill(&mut new_value[..]);

      oram.update(positions[key], new_pos, key as u32, |block| {
        let old = block.data;
        block.data = new_value;
        old
      });

      positions[key] = new_pos;
      values[key] = new_value;
    }

    for round in 0..50 {
      for key in 1..N {
        let new_pos = rng.random_range(0..oram.max_n) as u32;
        let mut new_value = [0u8; 56];
        rng.fill(&mut new_value[..]);

        let (found, old) = oram.update(positions[key], new_pos, key as u32, |block| {
          let old = block.data;
          block.data = new_value;
          old
        });

        assert!(found, "missing key {key} in round {round} at position {}", positions[key]);
        assert_eq!(old, values[key]);
        positions[key] = new_pos;
        values[key] = new_value;
      }
    }
  }
}
