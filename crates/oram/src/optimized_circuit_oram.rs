//! Cache-line-specialized Circuit ORAM.
//!
//! Every bucket is exactly one aligned cache line containing two 32-byte
//! blocks. AVX-512F transfers a bucket in one register, while AVX-512VL moves
//! individual blocks in one 256-bit register.
#![allow(clippy::needless_bitwise_bool)]

use std::mem::{align_of, offset_of, size_of};

use bytemuck::{Pod, Zeroable};
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
use core::arch::x86_64::_mm_popcnt_epi32;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
use core::arch::x86_64::{
  __m256i, _mm256_load_si256, _mm256_mask_mov_epi32, _mm256_store_si256, _mm512_castsi512_si128,
  _mm_and_si128, _mm_cmpeq_epi32_mask, _mm_cmpgt_epi32_mask, _mm_mask_mov_epi32, _mm_set1_epi32,
  _mm_setzero_si128, _mm_sub_epi32, _mm_xor_si128,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use core::arch::x86_64::{
  __m512i, _mm512_cmpeq_epi32_mask, _mm512_load_si512, _mm512_mask_mov_epi32,
  _mm512_permutexvar_epi32, _mm512_set1_epi32, _mm512_store_si512,
};
use rostl_primitives::traits::Cmov;

use crate::{heap_tree::HeapTree, prelude::PositionType};

/// Blocks per Circuit ORAM bucket.
pub const Z: usize = 2;
/// Total blocks reserved for the stash.
pub const S: usize = 20;
const STASH_BUCKETS: usize = S / Z;
const EVICTIONS_PER_OP: usize = 2;
const MAX_HEIGHT: usize = 32;
/// Position marking a dummy block.
pub const DUMMY_POS: PositionType = PositionType::MAX;
/// Compact key type used by the optimized layout.
pub type Key = u32;
/// Payload bytes in one block.
pub const DATA_SIZE: usize = 24;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const POS_BROADCAST_INDICES: __m512i =
  unsafe { std::mem::transmute([0u32, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8]) };
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const KEY_BROADCAST_INDICES: __m512i =
  unsafe { std::mem::transmute([1u32, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9]) };
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const POS_PAIR_INDICES: __m512i =
  unsafe { std::mem::transmute([0u32, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8]) };

/// One half-cache-line Circuit ORAM block.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub struct Block {
  /// Assigned leaf; [`DUMMY_POS`] marks an empty block.
  pub pos: PositionType,
  /// Logical block key.
  pub key: Key,
  /// Untyped payload occupying the rest of the cache line.
  pub data: [u8; DATA_SIZE],
}

unsafe impl Zeroable for Block {}
unsafe impl Pod for Block {}

impl Default for Block {
  #[inline]
  fn default() -> Self {
    Self { pos: DUMMY_POS, key: Key::MAX, data: [u8::MAX; DATA_SIZE] }
  }
}

impl Block {
  /// Returns whether this block is a dummy.
  #[inline]
  pub const fn is_empty(&self) -> bool {
    self.pos == DUMMY_POS
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
  #[inline(always)]
  fn load(&self) -> __m256i {
    // SAFETY: Block is exactly 32 bytes with 32-byte alignment.
    unsafe { _mm256_load_si256(self as *const Self as *const __m256i) }
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
  #[inline(always)]
  fn store(&mut self, value: __m256i) {
    // SAFETY: Block is exactly 32 bytes with 32-byte alignment.
    unsafe { _mm256_store_si256(self as *mut Self as *mut __m256i, value) }
  }

  #[inline(always)]
  fn cmov(&mut self, source: &Self, choice: bool) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
    {
      let mask = 0u8.wrapping_sub(choice as u8);
      // SAFETY: AVX-512F/VL is enabled and the mask covers the whole block.
      let selected = unsafe { _mm256_mask_mov_epi32(self.load(), mask, source.load()) };
      self.store(selected);
    }
    #[cfg(not(all(
      target_arch = "x86_64",
      target_feature = "avx512f",
      target_feature = "avx512vl"
    )))]
    {
      self.pos.cmov(&source.pos, choice);
      self.key.cmov(&source.key, choice);
      for (dst, src) in self.data.iter_mut().zip(source.data.iter()) {
        dst.cmov(src, choice);
      }
    }
  }
}

const _: () = assert!(size_of::<Block>() == 32);
const _: () = assert!(align_of::<Block>() == 32);
const _: () = assert!(offset_of!(Block, pos) == 0);
const _: () = assert!(offset_of!(Block, key) == 4);
const _: () = assert!(offset_of!(Block, data) == 8);

/// A two-block bucket.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Bucket(pub [Block; Z]);

impl Default for Bucket {
  #[inline]
  fn default() -> Self {
    Self([Block::default(); Z])
  }
}

impl Bucket {
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline(always)]
  fn load(&self) -> __m512i {
    // SAFETY: Bucket is exactly 64 bytes with 64-byte alignment.
    unsafe { _mm512_load_si512(self as *const Self as *const __m512i) }
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[inline(always)]
  fn store(&mut self, value: __m512i) {
    // SAFETY: Bucket is exactly 64 bytes with 64-byte alignment.
    unsafe { _mm512_store_si512(self as *mut Self as *mut __m512i, value) }
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    not(all(target_feature = "avx512vl", target_feature = "avx512vpopcntdq"))
  ))]
  #[inline(always)]
  fn movement_mask(choice: [bool; Z]) -> u16 {
    let lane_0 = 0u16.wrapping_sub(choice[0] as u16) & 0x00ff;
    let lane_1 = 0u16.wrapping_sub(choice[1] as u16) & 0xff00;
    lane_0 | lane_1
  }

  #[inline(always)]
  #[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  )))]
  fn cmov_lanes(&mut self, source: &Self, choice: [bool; Z]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
      let mask = Self::movement_mask(choice);
      // SAFETY: AVX-512F is enabled and each mask byte selects one 32-byte lane.
      let selected = unsafe { _mm512_mask_mov_epi32(self.load(), mask, source.load()) };
      self.store(selected);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    for lane in 0..Z {
      self.0[lane].cmov(&source.0[lane], choice[lane]);
    }
  }

  #[inline(always)]
  #[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  )))]
  fn cxchg_lanes(&mut self, other: &mut Self, choice: [bool; Z]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
      let mask = Self::movement_mask(choice);
      let left = self.load();
      let right = other.load();
      // SAFETY: AVX-512F is enabled and each mask byte selects one 32-byte lane.
      let new_left = unsafe { _mm512_mask_mov_epi32(left, mask, right) };
      let new_right = unsafe { _mm512_mask_mov_epi32(right, mask, left) };
      self.store(new_left);
      other.store(new_right);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    for lane in 0..Z {
      let old = self.0[lane];
      self.0[lane].cmov(&other.0[lane], choice[lane]);
      other.0[lane].cmov(&old, choice[lane]);
    }
  }
}

impl HeapTree<Bucket> {
  #[inline]
  fn read_optimized_path(&self, path: PositionType, out: &mut [Bucket]) {
    debug_assert_eq!(out.len(), self.height);
    for depth in 0..self.height {
      let bucket = &self.tree[self.get_index(depth, path)];
      #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
      out[depth].store(bucket.load());
      #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
      {
        out[depth] = *bucket;
      }
    }
  }

  #[inline]
  fn write_optimized_path(&mut self, path: PositionType, input: &[Bucket]) {
    debug_assert_eq!(input.len(), self.height);
    for depth in 0..self.height {
      let index = self.get_index(depth, path);
      #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
      self.tree[index].store(input[depth].load());
      #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
      {
        self.tree[index] = input[depth];
      }
    }
  }
}

#[inline(always)]
#[cfg(not(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
)))]
const fn common_suffix_length(a: PositionType, b: PositionType) -> u32 {
  (a ^ b).trailing_zeros()
}

#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn read_and_remove(arr: &mut [Bucket], key: Key, result: &mut Block) -> bool {
  let mut found_mask = 0u16;
  let mut result_bucket = Bucket::default().load();
  // SAFETY: AVX-512F is enabled for this implementation.
  unsafe {
    let desired_key = _mm512_set1_epi32(key as i32);
    let dummy = _mm512_set1_epi32(DUMMY_POS as i32);
    for bucket in arr {
      let value = bucket.load();
      let keys = _mm512_permutexvar_epi32(KEY_BROADCAST_INDICES, value);
      let positions = _mm512_permutexvar_epi32(POS_BROADCAST_INDICES, value);
      let key_matches = _mm512_cmpeq_epi32_mask(keys, desired_key);
      let non_dummy = !_mm512_cmpeq_epi32_mask(positions, dummy);
      let matched = key_matches & non_dummy;
      debug_assert_eq!(found_mask & matched, 0);
      result_bucket = _mm512_mask_mov_epi32(result_bucket, matched, value);
      bucket.store(_mm512_mask_mov_epi32(value, matched & 0x0101, dummy));
      found_mask |= matched;
    }
  }

  let found = found_mask != 0;
  let matched_bucket = result_bucket_to_bucket(result_bucket);
  let mut matched_block = matched_bucket.0[1];
  let lane_0 = matched_bucket.0[0];
  matched_block.cmov(&lane_0, !lane_0.is_empty());
  result.cmov(&matched_block, found);
  found
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn result_bucket_to_bucket(value: __m512i) -> Bucket {
  let mut bucket = Bucket::default();
  bucket.store(value);
  bucket
}

#[inline]
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
fn read_and_remove(arr: &mut [Bucket], key: Key, result: &mut Block) -> bool {
  let mut found = false;
  for bucket in arr {
    for item in &mut bucket.0 {
      let matched = (!item.is_empty()) & (item.key == key);
      debug_assert!((!matched) | (!found));
      result.cmov(item, matched);
      item.pos.cmov(&DUMMY_POS, matched);
      found.cmov(&true, matched);
    }
  }
  found
}

#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn remove(arr: &mut [Bucket], key: Key) -> bool {
  let mut found_mask = 0u16;
  // SAFETY: AVX-512F is enabled for this implementation.
  unsafe {
    let desired_key = _mm512_set1_epi32(key as i32);
    let dummy = _mm512_set1_epi32(DUMMY_POS as i32);
    for bucket in arr {
      let value = bucket.load();
      let keys = _mm512_permutexvar_epi32(KEY_BROADCAST_INDICES, value);
      let positions = _mm512_permutexvar_epi32(POS_BROADCAST_INDICES, value);
      let matched =
        _mm512_cmpeq_epi32_mask(keys, desired_key) & !_mm512_cmpeq_epi32_mask(positions, dummy);
      debug_assert_eq!(found_mask & matched, 0);
      bucket.store(_mm512_mask_mov_epi32(value, matched & 0x0101, dummy));
      found_mask |= matched;
    }
  }
  found_mask != 0
}

#[inline]
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
fn remove(arr: &mut [Bucket], key: Key) -> bool {
  let mut found = false;
  for bucket in arr {
    for item in &mut bucket.0 {
      let matched = (!item.is_empty()) & (item.key == key);
      debug_assert!((!matched) | (!found));
      item.pos.cmov(&DUMMY_POS, matched);
      found.cmov(&true, matched);
    }
  }
  found
}

#[inline]
fn insert_first_empty(arr: &mut [Bucket], lane: usize, block: &Block) -> bool {
  let mut written = false;
  for bucket in arr {
    let item = &mut bucket.0[lane];
    let write = item.is_empty() & (!written);
    item.cmov(block, write);
    written.cmov(&true, write);
  }
  written
}

/// Circuit ORAM specialized for two 32-byte blocks per cache-line bucket.
#[derive(Debug)]
pub struct OptimizedCircuitORAM {
  /// Logical capacity, rounded up to a power of two.
  pub max_n: usize,
  /// Height of the tree.
  pub h: usize,
  /// Combined `[paired stash lanes | loaded path buckets]` storage.
  pub stash_and_path: Vec<Bucket>,
  /// Binary tree holding the ORAM buckets.
  pub tree: HeapTree<Bucket>,
  /// Next public path used by deterministic eviction.
  pub evict_counter: PositionType,
  /// Public round-robin lane used for the next stash insertion.
  pub insert_lane: usize,
}

impl OptimizedCircuitORAM {
  /// Creates an empty optimized Circuit ORAM.
  pub fn new(max_n: usize) -> Self {
    assert!(max_n > 1);
    assert!(max_n <= (1usize << 31));
    let h0 = max_n.ilog2() as usize;
    let h = h0 + 1 + usize::from((1usize << h0) < max_n);
    let max_n = 1usize << (h - 1);
    Self {
      max_n,
      h,
      stash_and_path: vec![Bucket::default(); STASH_BUCKETS + h],
      tree: HeapTree::new(h),
      evict_counter: 0,
      insert_lane: 0,
    }
  }

  /// Creates an ORAM and inserts the supplied keys, payloads, and positions.
  pub fn new_with_positions_and_values(
    max_n: usize,
    keys: &[Key],
    values: &[[u8; DATA_SIZE]],
    positions: &[PositionType],
  ) -> Self {
    assert_eq!(keys.len(), values.len());
    assert_eq!(keys.len(), positions.len());
    assert!(keys.len() <= max_n);
    let mut oram = Self::new(max_n);
    for ((&key, &data), &pos) in keys.iter().zip(values).zip(positions) {
      oram.write_or_insert(0, pos, key, data);
    }
    oram
  }

  #[inline]
  fn read_path(&mut self, pos: PositionType) {
    self.tree.read_optimized_path(pos, &mut self.stash_and_path[STASH_BUCKETS..]);
  }

  #[inline]
  fn write_path(&mut self, pos: PositionType) {
    self.tree.write_optimized_path(pos, &self.stash_and_path[STASH_BUCKETS..]);
  }

  #[inline]
  fn insert_into_stash(&mut self, block: &Block) -> bool {
    let inserted =
      insert_first_empty(&mut self.stash_and_path[..STASH_BUCKETS], self.insert_lane, block);
    self.insert_lane ^= 1;
    inserted
  }

  /// Algorithms 2--4 (`EvictOnceFast`) from the Circuit ORAM paper.
  pub fn evict_once_fast(&mut self, pos: PositionType) {
    #[cfg(all(
      target_arch = "x86_64",
      target_feature = "avx512f",
      target_feature = "avx512vl",
      target_feature = "avx512vpopcntdq"
    ))]
    self.evict_once_fast_avx512(pos);

    #[cfg(not(all(
      target_arch = "x86_64",
      target_feature = "avx512f",
      target_feature = "avx512vl",
      target_feature = "avx512vpopcntdq"
    )))]
    self.evict_once_fast_scalar(pos);
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  fn evict_once_fast_avx512(&mut self, pos: PositionType) {
    debug_assert!((pos as usize) < self.max_n);

    // SAFETY: Compiled only with AVX-512F/VL and VPOPCNTDQ.
    unsafe {
      const PAIR: u8 = 0x03;
      let lane_mask = |mask: u8| -> u16 {
        (0u16.wrapping_sub((mask & 1) as u16) & 0x00ff)
          | (0u16.wrapping_sub(((mask >> 1) & 1) as u16) & 0xff00)
      };
      let pos_mask = |mask: u8| -> u16 { (mask & 1) as u16 | (((mask & 2) as u16) << 7) };

      let zero = _mm_setzero_si128();
      let one = _mm_set1_epi32(1);
      let neg_one = _mm_set1_epi32(-1);
      let dummy = _mm_set1_epi32(DUMMY_POS as i32);
      let path = _mm_set1_epi32(pos as i32);
      let dummy_bucket = _mm512_set1_epi32(DUMMY_POS as i32);

      let mut deepest = [neg_one; MAX_HEIGHT];
      let mut deepest_idx = [zero; MAX_HEIGHT];
      let mut target = [neg_one; MAX_HEIGHT];
      let mut has_empty = [0u8; MAX_HEIGHT];
      debug_assert!(self.h <= deepest.len());

      let mut src = neg_one;
      let mut dst = neg_one;

      for index in 0..=STASH_BUCKETS {
        let value = self.stash_and_path[index].load();
        let positions = _mm512_castsi512_si128(_mm512_permutexvar_epi32(POS_PAIR_INDICES, value));
        let empty = _mm_cmpeq_epi32_mask(positions, dummy) & PAIR;
        let different = _mm_xor_si128(positions, path);
        let low_bit = _mm_and_si128(different, _mm_sub_epi32(zero, different));
        let depth = _mm_popcnt_epi32(_mm_sub_epi32(low_bit, one));
        let choose = _mm_cmpgt_epi32_mask(depth, dst) & !empty & PAIR;
        dst = _mm_mask_mov_epi32(dst, choose, depth);
        deepest_idx[0] = _mm_mask_mov_epi32(deepest_idx[0], choose, _mm_set1_epi32(index as i32));
      }
      src = _mm_mask_mov_epi32(src, !_mm_cmpeq_epi32_mask(dst, neg_one) & PAIR, zero);

      for level in 1..self.h {
        let level_value = _mm_set1_epi32(level as i32);
        let reaches = _mm_cmpgt_epi32_mask(dst, _mm_set1_epi32(level as i32 - 1)) & PAIR;
        deepest[level] = _mm_mask_mov_epi32(deepest[level], reaches, src);

        let index = STASH_BUCKETS + level;
        let value = self.stash_and_path[index].load();
        let positions = _mm512_castsi512_si128(_mm512_permutexvar_epi32(POS_PAIR_INDICES, value));
        let empty = _mm_cmpeq_epi32_mask(positions, dummy) & PAIR;
        has_empty[level] = empty;
        let different = _mm_xor_si128(positions, path);
        let low_bit = _mm_and_si128(different, _mm_sub_epi32(zero, different));
        let depth = _mm_popcnt_epi32(_mm_sub_epi32(low_bit, one));
        let non_empty = !empty & PAIR;
        let choose = _mm_cmpgt_epi32_mask(depth, dst) & non_empty;
        src = _mm_mask_mov_epi32(src, choose, level_value);
        dst = _mm_mask_mov_epi32(dst, choose, depth);
        deepest_idx[level] =
          _mm_mask_mov_epi32(deepest_idx[level], non_empty, _mm_set1_epi32(index as i32));
      }

      src = neg_one;
      dst = neg_one;
      for level in (1..self.h).rev() {
        let level_value = _mm_set1_epi32(level as i32);
        let is_source = _mm_cmpeq_epi32_mask(level_value, src) & PAIR;
        target[level] = _mm_mask_mov_epi32(target[level], is_source, dst);
        src = _mm_mask_mov_epi32(src, is_source, neg_one);
        dst = _mm_mask_mov_epi32(dst, is_source, neg_one);

        let dst_free = _mm_cmpeq_epi32_mask(dst, neg_one) & PAIR;
        let target_valid = !_mm_cmpeq_epi32_mask(target[level], neg_one) & PAIR;
        let deepest_valid = !_mm_cmpeq_epi32_mask(deepest[level], neg_one) & PAIR;
        let change = ((dst_free & has_empty[level]) | target_valid) & deepest_valid;
        src = _mm_mask_mov_epi32(src, change, deepest[level]);
        dst = _mm_mask_mov_epi32(dst, change, level_value);
      }
      let root_source = _mm_cmpeq_epi32_mask(src, zero) & PAIR;
      target[0] = _mm_mask_mov_epi32(target[0], root_source, dst);

      let mut held = dummy_bucket;
      let root_valid = !_mm_cmpeq_epi32_mask(target[0], neg_one) & PAIR;
      for index in 0..=STASH_BUCKETS {
        let take = _mm_cmpeq_epi32_mask(deepest_idx[0], _mm_set1_epi32(index as i32)) & root_valid;
        let value = self.stash_and_path[index].load();
        held = _mm512_mask_mov_epi32(held, lane_mask(take), value);
        self.stash_and_path[index].store(_mm512_mask_mov_epi32(
          value,
          pos_mask(take),
          dummy_bucket,
        ));
      }
      dst = target[0];

      for level in 1..self.h - 1 {
        let level_value = _mm_set1_epi32(level as i32);
        let index = STASH_BUCKETS + level;
        let value = self.stash_and_path[index].load();
        let positions = _mm512_castsi512_si128(_mm512_permutexvar_epi32(POS_PAIR_INDICES, value));
        let empty = _mm_cmpeq_epi32_mask(positions, dummy) & PAIR;
        let has_target = !_mm_cmpeq_epi32_mask(target[level], neg_one) & PAIR;
        let place = _mm_cmpeq_epi32_mask(level_value, dst) & !has_target & PAIR;
        let take =
          _mm_cmpeq_epi32_mask(deepest_idx[level], _mm_set1_epi32(index as i32)) & has_target;
        let swap = take | (empty & place);
        let movement = lane_mask(swap);
        let next_held = _mm512_mask_mov_epi32(held, movement, value);
        let next_value = _mm512_mask_mov_epi32(value, movement, held);
        held = next_held;
        self.stash_and_path[index].store(next_value);
        dst = _mm_mask_mov_epi32(dst, has_target | place, target[level]);
      }

      let level = self.h - 1;
      let index = STASH_BUCKETS + level;
      let value = self.stash_and_path[index].load();
      let positions = _mm512_castsi512_si128(_mm512_permutexvar_epi32(POS_PAIR_INDICES, value));
      let empty = _mm_cmpeq_epi32_mask(positions, dummy) & PAIR;
      let place = _mm_cmpeq_epi32_mask(_mm_set1_epi32(level as i32), dst) & PAIR;
      self.stash_and_path[index].store(_mm512_mask_mov_epi32(
        value,
        lane_mask(empty & place),
        held,
      ));
    }
  }

  #[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  )))]
  fn evict_once_fast_scalar(&mut self, pos: PositionType) {
    debug_assert!((pos as usize) < self.max_n);
    let mut deepest = [[-1i32; Z]; MAX_HEIGHT];
    let mut deepest_idx = [[0usize; Z]; MAX_HEIGHT];
    let mut target = [[-1i32; Z]; MAX_HEIGHT];
    let mut has_empty = [[false; Z]; MAX_HEIGHT];
    debug_assert!(self.h <= deepest.len());

    let mut src = [-1i32; Z];
    let mut dst = [-1i32; Z];

    // Each lane independently considers its stash and root slot.
    for index in 0..=STASH_BUCKETS {
      for lane in 0..Z {
        let block = &self.stash_and_path[index].0[lane];
        let level = common_suffix_length(block.pos, pos) as i32;
        let choose = (!block.is_empty()) & (level > dst[lane]);
        dst[lane].cmov(&level, choose);
        deepest_idx[0][lane].cmov(&index, choose);
      }
    }
    for lane in 0..Z {
      src[lane].cmov(&0, dst[lane] != -1);
    }

    // With Z=1 per lane, each remaining level contributes one candidate.
    for level in 1..self.h {
      let index = STASH_BUCKETS + level;
      for lane in 0..Z {
        deepest[level][lane].cmov(&src[lane], dst[lane] >= level as i32);
        let block = &self.stash_and_path[index].0[lane];
        let empty = block.is_empty();
        has_empty[level][lane] = empty;
        let block_level = common_suffix_length(block.pos, pos) as i32;
        let choose = (!empty) & (block_level > dst[lane]);
        src[lane].cmov(&(level as i32), choose);
        dst[lane].cmov(&block_level, choose);
        deepest_idx[level][lane].cmov(&index, !empty);
      }
    }

    src = [-1; Z];
    dst = [-1; Z];
    for level in (1..self.h).rev() {
      for lane in 0..Z {
        let is_source = level as i32 == src[lane];
        target[level][lane].cmov(&dst[lane], is_source);
        src[lane].cmov(&-1, is_source);
        dst[lane].cmov(&-1, is_source);
        let change = (((dst[lane] == -1) & has_empty[level][lane]) | (target[level][lane] != -1))
          & (deepest[level][lane] != -1);
        src[lane].cmov(&deepest[level][lane], change);
        dst[lane].cmov(&(level as i32), change);
      }
    }
    for lane in 0..Z {
      target[0][lane].cmov(&dst[lane], src[lane] == 0);
    }

    // Move both independent lanes with one masked AVX-512 operation per bucket.
    let mut held = Bucket::default();
    for index in 0..=STASH_BUCKETS {
      let mut take = [false; Z];
      for lane in 0..Z {
        take[lane] = (deepest_idx[0][lane] == index) & (target[0][lane] != -1);
      }
      held.cmov_lanes(&self.stash_and_path[index], take);
      for lane in 0..Z {
        self.stash_and_path[index].0[lane].pos.cmov(&DUMMY_POS, take[lane]);
      }
    }
    dst = target[0];

    for level in 1..self.h - 1 {
      let index = STASH_BUCKETS + level;
      let mut swap = [false; Z];
      for lane in 0..Z {
        let has_target = target[level][lane] != -1;
        let place = (level as i32 == dst[lane]) & (!has_target);
        let take = (deepest_idx[level][lane] == index) & has_target;
        let write = self.stash_and_path[index].0[lane].is_empty() & place;
        swap[lane] = take | write;
        dst[lane].cmov(&target[level][lane], has_target | place);
      }
      held.cxchg_lanes(&mut self.stash_and_path[index], swap);
    }

    let index = STASH_BUCKETS + self.h - 1;
    let mut write = [false; Z];
    for lane in 0..Z {
      write[lane] =
        self.stash_and_path[index].0[lane].is_empty() & ((self.h - 1) as i32 == dst[lane]);
    }
    self.stash_and_path[index].cmov_lanes(&held, write);
  }

  #[inline]
  fn perform_eviction(&mut self, pos: PositionType) {
    self.read_path(pos);
    self.evict_once_fast(pos);
    self.write_path(pos);
  }

  #[inline]
  fn perform_deterministic_evictions(&mut self) {
    for _ in 0..EVICTIONS_PER_OP {
      self.perform_eviction(self.evict_counter);
      self.evict_counter = (self.evict_counter + 1) % self.max_n as PositionType;
    }
    for lane in 0..Z {
      debug_assert!(self.stash_and_path[..STASH_BUCKETS]
        .iter()
        .any(|bucket| bucket.0[lane].is_empty()));
    }
  }

  #[inline]
  fn finish_access(&mut self, pos: PositionType) {
    self.evict_once_fast(pos);
    self.write_path(pos);
    self.perform_deterministic_evictions();
  }

  /// Reads and remaps a block. `out` is unchanged when the key is absent.
  pub fn read(
    &mut self,
    pos: PositionType,
    new_pos: PositionType,
    key: Key,
    out: &mut [u8; DATA_SIZE],
  ) -> bool {
    debug_assert!((pos as usize) < self.max_n);
    debug_assert!((new_pos as usize) < self.max_n || new_pos == DUMMY_POS);
    self.read_path(pos);
    let mut block = Block { pos: DUMMY_POS, key, data: *out };
    let found = read_and_remove(&mut self.stash_and_path, key, &mut block);
    out.copy_from_slice(&block.data);
    block.pos.cmov(&new_pos, found);
    let inserted = self.insert_into_stash(&block);
    debug_assert!(inserted);
    self.finish_access(pos);
    found
  }

  /// Updates an existing block, without inserting it when absent.
  pub fn write(
    &mut self,
    pos: PositionType,
    new_pos: PositionType,
    key: Key,
    data: [u8; DATA_SIZE],
  ) -> bool {
    debug_assert!((pos as usize) < self.max_n);
    debug_assert!((new_pos as usize) < self.max_n);
    self.read_path(pos);
    let found = remove(&mut self.stash_and_path, key);
    let mut block = Block { pos: DUMMY_POS, key, data };
    block.pos.cmov(&new_pos, found);
    let inserted = self.insert_into_stash(&block);
    debug_assert!(inserted);
    self.finish_access(pos);
    found
  }

  /// Updates a block or inserts it when absent.
  pub fn write_or_insert(
    &mut self,
    pos: PositionType,
    new_pos: PositionType,
    key: Key,
    data: [u8; DATA_SIZE],
  ) -> bool {
    debug_assert!((pos as usize) < self.max_n);
    debug_assert!((new_pos as usize) < self.max_n);
    self.read_path(pos);
    let found = remove(&mut self.stash_and_path, key);
    let inserted = self.insert_into_stash(&Block { pos: new_pos, key, data });
    debug_assert!(inserted);
    self.finish_access(pos);
    found
  }

  /// Reads or inserts a block, then applies `update` to its payload.
  pub fn update<T, F>(
    &mut self,
    pos: PositionType,
    new_pos: PositionType,
    key: Key,
    update: F,
  ) -> (bool, T)
  where
    F: FnOnce(&mut [u8; DATA_SIZE]) -> T,
  {
    debug_assert!((pos as usize) < self.max_n);
    debug_assert!((new_pos as usize) < self.max_n);
    self.read_path(pos);
    let mut block = Block { pos: DUMMY_POS, key, data: [0; DATA_SIZE] };
    let found = read_and_remove(&mut self.stash_and_path, key, &mut block);
    let result = update(&mut block.data);
    block.pos = new_pos;
    block.key = key;
    let inserted = self.insert_into_stash(&block);
    debug_assert!(inserted);
    self.finish_access(pos);
    (found, result)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn bucket_is_exactly_one_aligned_cache_line() {
    assert_eq!(size_of::<Block>(), 32);
    assert_eq!(align_of::<Block>(), 32);
    assert_eq!(size_of::<Bucket>(), 64);
    assert_eq!(align_of::<Bucket>(), 64);
    assert_eq!(offset_of!(Block, data), 8);
  }

  #[test]
  fn insert_read_update_and_missing_write() {
    let mut oram = OptimizedCircuitORAM::new(8);
    let mut initial = [0u8; DATA_SIZE];
    initial[0] = 17;
    assert!(!oram.write_or_insert(0, 3, 7, initial));
    let mut out = [0u8; DATA_SIZE];
    assert!(oram.read(3, 5, 7, &mut out));
    assert_eq!(out[0], 17);
    let (found, old) = oram.update(5, 2, 7, |data| {
      let old = data[0];
      data[0] = 29;
      old
    });
    assert!(found);
    assert_eq!(old, 17);
    assert!(!oram.write(0, 1, 99, [4; DATA_SIZE]));
    out.fill(0);
    assert!(oram.read(2, 6, 7, &mut out));
    assert_eq!(out[0], 29);
  }

  #[test]
  fn repetitive_accesses_preserve_values() {
    const N: usize = 16;
    let mut oram = OptimizedCircuitORAM::new(N);
    let mut positions = [0u32; N];
    for key in 0..N {
      let pos = ((key * 7) % N) as u32;
      let mut data = [0u8; DATA_SIZE];
      data[..4].copy_from_slice(&(key as u32).to_le_bytes());
      assert!(!oram.write_or_insert(positions[key], pos, key as Key, data));
      positions[key] = pos;
    }
    for round in 0..128 {
      let key = (round * 11) % N;
      let new_pos = ((round * 5 + 3) % N) as u32;
      let mut out = [0u8; DATA_SIZE];
      assert!(oram.read(positions[key], new_pos, key as Key, &mut out));
      assert_eq!(u32::from_le_bytes(out[..4].try_into().unwrap()), key as u32);
      positions[key] = new_pos;
    }
  }
}
