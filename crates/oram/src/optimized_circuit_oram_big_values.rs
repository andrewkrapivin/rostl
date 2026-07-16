//! Optimized Circuit ORAM for 56-byte payloads and exact cache-line-sized blocks.
//!
#![allow(clippy::needless_bitwise_bool)]

// UNDONE(git-8): This is needed to enforce the bitwise operations to not short circuit. Investigate if we should be using helper functions instead.
use bytemuck::{Pod, Zeroable};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use core::arch::x86_64::{
  __m256i, __m512i, __mmask16, __mmask8, _mm256_add_epi32, _mm256_and_si256,
  _mm256_cmpeq_epi32_mask, _mm256_cmpgt_epi32_mask, _mm256_mask_mov_epi32, _mm256_set1_epi32,
  _mm256_setr_epi32, _mm256_setzero_si256, _mm256_sub_epi32, _mm256_xor_si256, _mm512_and_si512,
  _mm512_castsi512_si256, _mm512_cmpeq_epi32_mask, _mm512_extracti64x4_epi64,
  _mm512_i32gather_epi32, _mm512_load_si512, _mm512_mask_i32gather_epi32, _mm512_mask_mov_epi32,
  _mm512_permutexvar_epi32, _mm512_reduce_max_epi32, _mm512_set1_epi32, _mm512_store_si512,
  _mm512_sub_epi32, _mm512_xor_si512,
};
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
use core::arch::x86_64::{_mm256_popcnt_epi32, _mm512_popcnt_epi32};
use rostl_primitives::{
  cmov_body, cxchg_body, impl_cmov_for_pod,
  traits::{_Cmovbase, Cmov},
};

use crate::heap_tree::HeapTree;
use crate::prelude::{PositionType, DUMMY_POS};
/// Slots in one independently addressed pooled lane.
pub const SLOTS_PER_POOLED_LANE: usize = 2;
/// Number of independent pooled lanes packed into one tree bucket.
pub const POOLED_LANES: usize = 8;
/// Total blocks in one wide tree bucket.
pub const BLOCKS_PER_BUCKET: usize = POOLED_LANES * SLOTS_PER_POOLED_LANE;
/// Blocks in the one shared stash.
pub const S: usize = 50;
/// Numerator of the amortized deterministic-eviction rate.
pub const EVICTIONS_PER_OP_NUMERATOR: usize = 3;
/// Denominator of the amortized deterministic-eviction rate.
pub const EVICTIONS_PER_OP_DENOMINATOR: usize = 1;
const EVICTIONS_PER_BATCH: usize = BLOCKS_PER_BUCKET;
const EVICTION_CREDITS_PER_BATCH: usize = EVICTIONS_PER_BATCH * EVICTIONS_PER_OP_DENOMINATOR;
/// Bytes stored in each block.
pub const DATA_SIZE: usize = 56;
/// Compact key stored alongside the position.
/// `Key::MAX` is reserved for dummy blocks.
pub type Key = u32;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const KEY_BROADCAST_INDICES: __m512i = unsafe { std::mem::transmute([1u32; 16]) };
/// Dword offsets of each block position, grouped by slot within a pooled lane.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const LEVEL_POSITION_GATHER_INDICES: __m512i = unsafe {
  std::mem::transmute([0i32, 32, 64, 96, 128, 160, 192, 224, 16, 48, 80, 112, 144, 176, 208, 240])
};
/// Dword offsets of the position and key fields in sixteen consecutive blocks.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const BLOCK_POSITION_GATHER_INDICES: __m512i = unsafe {
  std::mem::transmute([0i32, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240])
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const BLOCK_KEY_GATHER_INDICES: __m512i = unsafe {
  std::mem::transmute([1i32, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241])
};

/// A block in the ORAM tree
/// # Invariants
/// If `pos == DUMMY_POS`, the block is empty, there are no guarantees about the key of value in that case.
/// If `pos != DUMMY_POS`, the block is full and the key and value are valid.
///
/// # Note
/// * It is wrong to assume anything about the block being empty or not based on the key, please use pos.
///
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Block {
  /// The position of the block.
  pub pos: PositionType,
  /// The key of the block.
  pub key: Key,
  /// The data stored in the block.
  pub data: [u8; DATA_SIZE],
}

// SAFETY: The asserted field offsets and total size below prove that Block is
// exactly its three plain-data fields with no uninitialized padding bytes.
unsafe impl Zeroable for Block {}
unsafe impl Pod for Block {}

impl Default for Block {
  fn default() -> Self {
    Self { pos: DUMMY_POS, key: Key::MAX, data: [u8::MAX; DATA_SIZE] }
  }
}

impl_cmov_for_pod!(Block);

impl Block {
  /// Checks if the block is empty or not.
  pub const fn is_empty(&self) -> bool {
    self.pos == DUMMY_POS
  }
}

const _: () = assert!(std::mem::size_of::<Block>() == 64);
const _: () = assert!(std::mem::align_of::<Block>() == 64);
const _: () = assert!(std::mem::offset_of!(Block, pos) == 0);
const _: () = assert!(std::mem::offset_of!(Block, key) == 4);
const _: () = assert!(std::mem::offset_of!(Block, data) == 8);

/// A two-block pooled lane occupying two cache lines.
#[repr(C, align(64))]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct PooledLane([Block; SLOTS_PER_POOLED_LANE]);

const _: () = assert!(std::mem::size_of::<PooledLane>() == 128);
const _: () = assert!(std::mem::align_of::<PooledLane>() == 64);

/// A wide tree bucket containing sixteen independently addressed slots.
#[repr(C, align(64))]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct Bucket([Block; BLOCKS_PER_BUCKET]);

const _: () = assert!(std::mem::size_of::<Bucket>() == BLOCKS_PER_BUCKET * 64);
const _: () = assert!(std::mem::align_of::<Bucket>() == 64);

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
struct LevelMetadata {
  deepest: __m256i,
  source_slot_1: __mmask8,
  empty_slot_0: __mmask8,
  empty_slot_1: __mmask8,
}

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
struct BatchEvictionPlan {
  root_source: [i32; POOLED_LANES],
  target: [[i32; POOLED_LANES]; 64],
  source_slot_1: [__mmask8; 64],
  empty_slot_0: [__mmask8; 64],
  empty_slot_1: [__mmask8; 64],
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn gather_level_positions(bucket: &Bucket) -> (__m256i, __m256i) {
  // SAFETY: Bucket contains 16 contiguous Blocks. Every gather index names the
  // first dword of one Block, and SCALE converts dword offsets to byte offsets.
  let gathered = unsafe {
    _mm512_i32gather_epi32::<4>(LEVEL_POSITION_GATHER_INDICES, bucket.0.as_ptr().cast::<i32>())
  };
  // The low half is slot 0 across all pooled lanes; the high half is slot 1.
  unsafe { (_mm512_castsi512_si256(gathered), _mm512_extracti64x4_epi64::<1>(gathered)) }
}

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
#[inline(always)]
fn legal_depths(positions: __m256i, path: PositionType) -> __m256i {
  // trailing_zeros(pos ^ path) = popcount(((diff & -diff) - 1)).
  unsafe {
    let diff = _mm256_xor_si256(positions, _mm256_set1_epi32(path as i32));
    let low_bit = _mm256_and_si256(diff, _mm256_sub_epi32(_mm256_setzero_si256(), diff));
    _mm256_popcnt_epi32(_mm256_sub_epi32(low_bit, _mm256_set1_epi32(1)))
  }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vpopcntdq"))]
#[inline(always)]
fn legal_depths_512(positions: __m512i, path: PositionType) -> __m512i {
  unsafe {
    let diff = _mm512_xor_si512(positions, _mm512_set1_epi32(path as i32));
    let low_bit = _mm512_and_si512(diff, _mm512_sub_epi32(_mm512_set1_epi32(0), diff));
    _mm512_popcnt_epi32(_mm512_sub_epi32(low_bit, _mm512_set1_epi32(1)))
  }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vpopcntdq"))]
#[inline(always)]
fn find_root_source<const STASH_SIZE: usize>(
  stash: &AlignedStash,
  path: PositionType,
  pooled_lane: usize,
) -> (i32, i32) {
  unsafe {
    let no_source = _mm512_set1_epi32(-1);
    let desired_lane = _mm512_set1_epi32(pooled_lane as i32);
    let lane_mask = _mm512_set1_epi32((POOLED_LANES - 1) as i32);
    let mut best_depth = -1i32;
    let mut best_index = 0i32;

    let scanned_blocks = STASH_SIZE + SLOTS_PER_POOLED_LANE;
    for group in 0..scanned_blocks.div_ceil(16) {
      let remaining = scanned_blocks - group * 16;
      let valid: __mmask16 = if remaining >= 16 { 0xffff } else { (1u16 << remaining) - 1 };
      let base = stash.as_ptr().add(group * 16).cast::<i32>();
      let positions =
        _mm512_mask_i32gather_epi32::<4>(no_source, valid, BLOCK_POSITION_GATHER_INDICES, base);
      let keys = _mm512_mask_i32gather_epi32::<4>(no_source, valid, BLOCK_KEY_GATHER_INDICES, base);
      let belongs_to_lane =
        _mm512_cmpeq_epi32_mask(_mm512_and_si512(keys, lane_mask), desired_lane);
      let mut root_slots = 0u16;
      for root_slot in STASH_SIZE..scanned_blocks {
        if root_slot / 16 == group {
          root_slots |= 1 << (root_slot % 16);
        }
      }
      let nonempty = !_mm512_cmpeq_epi32_mask(positions, no_source);
      let eligible = valid & nonempty & (belongs_to_lane | root_slots);
      let depths = _mm512_mask_mov_epi32(no_source, eligible, legal_depths_512(positions, path));
      let group_best = _mm512_reduce_max_epi32(depths);
      let first_at_best = (_mm512_cmpeq_epi32_mask(depths, _mm512_set1_epi32(group_best)) & valid)
        .trailing_zeros() as i32;
      let group_index = group as i32 * 16 + first_at_best;
      let replaces_best = group_best > best_depth;
      best_depth.cmov(&group_best, replaces_best);
      best_index.cmov(&group_index, replaces_best);
    }

    (best_depth, best_index)
  }
}

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
#[inline(always)]
fn prepare_level_metadata(bucket: &Bucket, path: PositionType) -> LevelMetadata {
  unsafe {
    let (positions_0, positions_1) = gather_level_positions(bucket);
    let dummy = _mm256_set1_epi32(DUMMY_POS as i32);
    let empty_slot_0 = _mm256_cmpeq_epi32_mask(positions_0, dummy);
    let empty_slot_1 = _mm256_cmpeq_epi32_mask(positions_1, dummy);
    let no_source = _mm256_set1_epi32(-1);
    let depth_0 = _mm256_mask_mov_epi32(no_source, !empty_slot_0, legal_depths(positions_0, path));
    let depth_1 = _mm256_mask_mov_epi32(no_source, !empty_slot_1, legal_depths(positions_1, path));
    // Slot 0 wins ties, matching the scalar scan order.
    let source_slot_1 = _mm256_cmpgt_epi32_mask(depth_1, depth_0);
    let deepest = _mm256_mask_mov_epi32(depth_0, source_slot_1, depth_1);

    LevelMetadata { deepest, source_slot_1, empty_slot_0, empty_slot_1 }
  }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
#[inline(always)]
fn swap_register_with_block(mut held: __m512i, block: &mut Block, choice: bool) -> __m512i {
  // SAFETY: Block is exactly 64 bytes and 64-byte aligned. The held value stays
  // in vector form; only the scanned tree block is loaded and stored.
  unsafe {
    let mask = 0u16.wrapping_sub(choice as u16);
    let block_ptr = (block as *mut Block).cast::<__m512i>();
    let old_block = _mm512_load_si512(block_ptr);
    let new_block = _mm512_mask_mov_epi32(old_block, mask, held);
    held = _mm512_mask_mov_epi32(held, mask, old_block);
    _mm512_store_si512(block_ptr, new_block);
    held
  }
}

/// A flat block stash backed by 64-byte-aligned buckets.
#[derive(Debug)]
pub struct AlignedStash(Vec<PooledLane>);

impl AlignedStash {
  fn new(blocks: usize) -> Self {
    debug_assert_eq!(blocks % SLOTS_PER_POOLED_LANE, 0);
    Self(vec![PooledLane::default(); blocks / SLOTS_PER_POOLED_LANE])
  }
}

impl std::ops::Deref for AlignedStash {
  type Target = [Block];

  fn deref(&self) -> &Self::Target {
    bytemuck::cast_slice(&self.0)
  }
}

impl std::ops::DerefMut for AlignedStash {
  fn deref_mut(&mut self) -> &mut Self::Target {
    bytemuck::cast_slice_mut(&mut self.0)
  }
}

impl HeapTree<Bucket> {
  #[inline]
  fn read_lane_path(&mut self, path: PositionType, pooled_lane: usize, out: &mut [Block]) {
    debug_assert!((path as usize) < (1 << self.height));
    debug_assert!(pooled_lane < POOLED_LANES);
    debug_assert_eq!(out.len(), self.height * SLOTS_PER_POOLED_LANE);
    let lane_start = pooled_lane * SLOTS_PER_POOLED_LANE;
    for depth in 0..self.height {
      let index = self.get_index(depth, path);
      out[depth * SLOTS_PER_POOLED_LANE..(depth + 1) * SLOTS_PER_POOLED_LANE]
        .copy_from_slice(&self.tree[index].0[lane_start..lane_start + SLOTS_PER_POOLED_LANE]);
    }
  }

  #[inline]
  fn write_lane_path(&mut self, path: PositionType, pooled_lane: usize, input: &[Block]) {
    debug_assert!((path as usize) < (1 << self.height));
    debug_assert!(pooled_lane < POOLED_LANES);
    debug_assert_eq!(input.len(), self.height * SLOTS_PER_POOLED_LANE);
    let lane_start = pooled_lane * SLOTS_PER_POOLED_LANE;
    for depth in 0..self.height {
      let index = self.get_index(depth, path);
      self.tree[index].0[lane_start..lane_start + SLOTS_PER_POOLED_LANE].copy_from_slice(
        &input[depth * SLOTS_PER_POOLED_LANE..(depth + 1) * SLOTS_PER_POOLED_LANE],
      );
    }
  }
}

/// Circuit ORAMs packed into sixteen-slot buckets with one shared stash.
#[derive(Debug)]
pub struct OptimizedCircuitORAMBigValuesWithStash<const STASH_SIZE: usize> {
  /// Requested total logical capacity across all pooled-lane ORAMs.
  pub capacity: usize,
  /// Leaf count and valid position range for each pooled-lane ORAM.
  pub max_n: usize,
  /// Height of each packed binary tree.
  pub h: usize,
  /// One shared stash followed by scratch space for one two-lane path.
  pub stash: AlignedStash,
  /// The packed sixteen-slot tree.
  pub tree: HeapTree<Bucket>,
  /// Next deterministic path.
  pub evict_counter: PositionType,
  /// Accumulated numerator credits toward the next deterministic eviction batch.
  pub eviction_credit: usize,
}

#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn read_and_remove_element(arr: &mut AlignedStash, k: Key) -> Block {
  let mut found_mask = 0u16;
  // SAFETY: AVX-512F is enabled and every Block is exactly 64-byte aligned.
  let result = unsafe {
    let desired_key = _mm512_set1_epi32(k as i32);
    let dummy = _mm512_set1_epi32(-1);
    let mut held = dummy;

    for block in arr.iter_mut() {
      let block_ptr = (block as *mut Block).cast::<__m512i>();
      let value = _mm512_load_si512(block_ptr);
      let keys = _mm512_permutexvar_epi32(KEY_BROADCAST_INDICES, value);
      let matched = _mm512_cmpeq_epi32_mask(keys, desired_key);

      debug_assert!((found_mask == 0) | (matched == 0));
      let old_held = held;
      held = _mm512_mask_mov_epi32(held, matched, value);
      _mm512_store_si512(block_ptr, _mm512_mask_mov_epi32(value, matched, old_held));
      found_mask |= matched;
    }
    held
  };

  let mut result_block = Block::default();
  // SAFETY: Block is exactly 64 bytes and aligned to 64 bytes.
  unsafe {
    _mm512_store_si512((&mut result_block as *mut Block).cast::<__m512i>(), result);
  }
  result_block
}

#[inline]
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
fn read_and_remove_element(arr: &mut AlignedStash, k: Key) -> Block {
  let mut found = false;
  let mut result = Block::default();

  for item in arr.iter_mut() {
    let matched = item.key == k;
    debug_assert!((!matched) | (!found));

    result.cxchg(item, matched);
    found.cmov(&true, matched);
  }

  result
}

/// Writes a block to an empty slot in an array.
/// If there are no empty slots, nothing happens and returns false.
#[inline]
fn write_block_to_empty_slot(arr: &mut [Block], val: &Block) -> bool {
  let mut rv = false;

  for item in arr.iter_mut() {
    let matched = (item.is_empty()) & (!rv);
    debug_assert!((!matched) | (!rv));

    item.cmov(val, matched);
    rv.cmov(&true, matched);
  }

  rv
}

#[inline]
const fn common_suffix_length(a: PositionType, b: PositionType) -> u32 {
  let w = a ^ b;
  w.trailing_zeros()
}

/// Selects a two-slot pooled lane using the low three key bits.
#[inline]
const fn pooled_lane_for_key(key: Key) -> usize {
  (key as usize) & (POOLED_LANES - 1)
}

/// Big-value optimized Circuit ORAM with the default 50-block stash.
pub type OptimizedCircuitORAMBigValues = OptimizedCircuitORAMBigValuesWithStash<S>;
/// Big-value optimized Circuit ORAM with a 40-block stash.
pub type OptimizedCircuitORAMBigValuesS40 = OptimizedCircuitORAMBigValuesWithStash<40>;
/// Big-value optimized Circuit ORAM with a 56-block stash.
pub type OptimizedCircuitORAMBigValuesS56 = OptimizedCircuitORAMBigValuesWithStash<56>;
/// Big-value optimized Circuit ORAM with an 80-block stash.
pub type OptimizedCircuitORAMBigValuesS80 = OptimizedCircuitORAMBigValuesWithStash<80>;

impl<const STASH_SIZE: usize> OptimizedCircuitORAMBigValuesWithStash<STASH_SIZE> {
  /// Creates a new empty big-value optimized Circuit ORAM with the given capacity.
  ///
  /// # Arguments
  /// * `max_n` - The maximum number of blocks in the ORAM.
  ///
  /// # Returns
  /// A new instance of `OptimizedCircuitORAMBigValues`.
  ///
  /// # Preconditions
  /// * `0 < max_n < (2**33)`
  pub fn new(capacity: usize) -> Self {
    debug_assert!(capacity > 0);
    debug_assert!(capacity <= u32::MAX as usize);
    debug_assert!(POOLED_LANES.is_power_of_two());
    debug_assert!(EVICTIONS_PER_OP_NUMERATOR > 0);
    debug_assert!(EVICTIONS_PER_OP_DENOMINATOR > 0);

    debug_assert!(EVICTIONS_PER_OP_NUMERATOR <= EVICTION_CREDITS_PER_BATCH);

    // Each pooled-lane ORAM targets capacity / 8 real blocks. Its two bottom
    // slots per leaf are therefore at most 50% occupied.
    let blocks_per_lane = capacity.div_ceil(POOLED_LANES);
    let max_n = blocks_per_lane.max(2).next_power_of_two();
    let h = max_n.ilog2() as usize + 1;
    let tree = HeapTree::new(h);
    debug_assert!(STASH_SIZE > 0);
    debug_assert_eq!(STASH_SIZE % SLOTS_PER_POOLED_LANE, 0);
    let stash = AlignedStash::new(STASH_SIZE + h * SLOTS_PER_POOLED_LANE);

    Self { capacity, max_n, h, stash, tree, evict_counter: 0, eviction_credit: 0 }
  }

  /// Reads a path to the end of the stash
  /// Reads only one two-slot pooled lane along a path into the scratch tail.
  fn read_path_and_get_nodes(&mut self, pos: PositionType, pooled_lane: usize) {
    debug_assert!((pos as usize) < self.max_n);
    self.tree.read_lane_path(
      pos,
      pooled_lane,
      &mut self.stash[STASH_SIZE..STASH_SIZE + self.h * SLOTS_PER_POOLED_LANE],
    );
  }

  /// Writes one two-slot pooled lane from the scratch tail back to the tree.
  fn write_back_path(&mut self, pos: PositionType, pooled_lane: usize) {
    debug_assert!((pos as usize) < self.max_n);
    self.tree.write_lane_path(
      pos,
      pooled_lane,
      &self.stash[STASH_SIZE..STASH_SIZE + self.h * SLOTS_PER_POOLED_LANE],
    );
  }

  /// Alg. 4 - EvictOnceFast(path) in `OptimizedCircuitORAM` paper
  fn evict_once_fast(&mut self, pos: PositionType, pooled_lane: usize) {
    // UNDONE(git-10): Investigate using u8 and/or bitwise operations here instead of u32/bool cmov's
    // UNDONE(git-11): This only supports n<=32. Is it enough?
    //
    let mut deepest: [i32; 64] = [-1; 64];
    let mut deepest_idx: [i32; 64] = [0; 64];
    let mut target: [i32; 64] = [-1; 64];
    let mut has_empty: [bool; 64] = [false; 64];

    let mut src = -1;
    let mut dst: i32;

    // 1) First pass: (Alg 2 - PrepareDeepest in `OptimizedCircuitORAM` paper).
    // dst is the same as goal in the paper
    // First level (including the stash):
    //
    #[cfg(all(
      target_arch = "x86_64",
      target_feature = "avx512f",
      target_feature = "avx512vpopcntdq"
    ))]
    {
      (dst, deepest_idx[0]) = find_root_source::<STASH_SIZE>(&self.stash, pos, pooled_lane);
    }
    #[cfg(not(all(
      target_arch = "x86_64",
      target_feature = "avx512f",
      target_feature = "avx512vpopcntdq"
    )))]
    {
      dst = -1;
      for idx in 0..STASH_SIZE + SLOTS_PER_POOLED_LANE {
        let deepest_level = common_suffix_length(self.stash[idx].pos, pos) as i32;
        let belongs_to_lane =
          (idx >= STASH_SIZE) | (pooled_lane_for_key(self.stash[idx].key) == pooled_lane);
        let deeper_flag = (!self.stash[idx].is_empty()) & belongs_to_lane & (deepest_level > dst);
        dst.cmov(&deepest_level, deeper_flag);
        deepest_idx[0].cmov(&(idx as i32), deeper_flag);
      }
    }
    src.cmov(&0, dst != -1);

    let mut idx = STASH_SIZE + SLOTS_PER_POOLED_LANE;
    // Remaining levels:
    //
    for i in 1..self.h {
      deepest[i].cmov(&src, dst >= i as i32);
      let mut bucket_deepest_level: i32 = -1;
      for _ in 0..SLOTS_PER_POOLED_LANE {
        let deepest_level = common_suffix_length(self.stash[idx].pos, pos) as i32;
        let is_empty = self.stash[idx].is_empty();
        has_empty[i].cmov(&true, is_empty);

        let deeper_flag = (!is_empty) & (deepest_level > bucket_deepest_level);
        bucket_deepest_level.cmov(&deepest_level, deeper_flag);
        deepest_idx[i].cmov(&(idx as i32), deeper_flag);

        idx += 1;
      }

      let deepper_flag = bucket_deepest_level > dst;
      src.cmov(&(i as i32), deepper_flag);
      dst.cmov(&bucket_deepest_level, deepper_flag);
    }

    // 2) Second pass: (Alg 3 - PrepareTarget in OptimizedCircuitORAM paper).
    //
    src = -1;
    dst = -1;
    for i in (1..self.h).rev() {
      let is_src = (i as i32) == src;
      target[i].cmov(&dst, is_src);
      src.cmov(&-1, is_src);
      dst.cmov(&-1, is_src);
      let change_flag = (((dst == -1) & has_empty[i]) | (target[i] != -1)) & (deepest[i] != -1);
      src.cmov(&deepest[i], change_flag);
      dst.cmov(&(i as i32), change_flag);
    }
    target[0].cmov(&dst, src == 0);

    // 3) Third pass: Actually move the data (end of Alg 4 - EvictOnceFast in OptimizedCircuitORAM paper).
    //
    // First level (including the stash)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
    let mut hold = unsafe { _mm512_set1_epi32(-1) };
    #[cfg(not(all(
      target_arch = "x86_64",
      target_feature = "avx512f",
      target_feature = "avx512vl"
    )))]
    let mut hold = Block::default();
    for idx in 0..STASH_SIZE + SLOTS_PER_POOLED_LANE {
      let is_deepest = deepest_idx[0] == idx as i32;
      let read_and_remove_flag = is_deepest & (target[0] != -1);
      #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
      {
        hold = swap_register_with_block(hold, &mut self.stash[idx], read_and_remove_flag);
      }
      #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vl"
      )))]
      {
        hold.cxchg(&mut self.stash[idx], read_and_remove_flag);
      }
    }
    dst = target[0];

    // Remaining levels except the last
    let mut idx = STASH_SIZE + SLOTS_PER_POOLED_LANE;
    for i in 1..(self.h - 1) {
      let has_target_flag = target[i] != -1;
      let place_dummy_flag = (i as i32 == dst) & (!has_target_flag);
      for _ in 0..SLOTS_PER_POOLED_LANE {
        // case 0: level i is neither a dest and not a src
        //         hasTargetFlag = false, placeDummyFlag = false
        //         nothing will change
        // case 1: level i is a dest but not a src
        //         hasTargetFlag = false, placeDummyFlag = true
        //         hold will be swapped with each dummy slot
        //         after the first swap, hold will become dummy, and the
        //         subsequent swaps have no effect.
        // case 2: level i is a src but not a dest
        //         hasTargetFlag = true, placeDummyFlag = false
        //         hold must be dummy originally (eviction cannot carry two
        //         blocks). hold will be swapped with the slot that evicts to
        //         deepest.
        // case 3: level i is both a src and a dest
        //         hasTargetFlag = true, placeDummyFlag = false
        //         hold will be swapped with the slot that evicts to deepest,
        //         which fulfills both src and dest requirements.
        let is_deepest = deepest_idx[i] == idx as i32;
        let read_and_remove_flag = is_deepest & has_target_flag;
        let write_flag = (self.stash[idx].is_empty()) & place_dummy_flag;
        let swap_flag = read_and_remove_flag | write_flag;
        #[cfg(all(
          target_arch = "x86_64",
          target_feature = "avx512f",
          target_feature = "avx512vl"
        ))]
        {
          hold = swap_register_with_block(hold, &mut self.stash[idx], swap_flag);
        }
        #[cfg(not(all(
          target_arch = "x86_64",
          target_feature = "avx512f",
          target_feature = "avx512vl"
        )))]
        {
          hold.cxchg(&mut self.stash[idx], swap_flag);
        }
        idx += 1;
      }

      dst.cmov(&target[i], has_target_flag | place_dummy_flag);
    }

    // last level (this should not be called if h=1, but we just assert h>1)
    let place_dummy_flag = ((self.h - 1) as i32) == dst;
    let mut written = false;
    for _ in 0..SLOTS_PER_POOLED_LANE {
      let write_flag = (self.stash[idx].is_empty()) & place_dummy_flag & (!written);
      written |= write_flag;
      #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
      {
        hold = swap_register_with_block(hold, &mut self.stash[idx], write_flag);
      }
      #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vl"
      )))]
      {
        self.stash[idx].cmov(&hold, write_flag);
      }
      idx += 1;
    }
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  fn prepare_batch_eviction_plan(&self, path: PositionType) -> BatchEvictionPlan {
    unsafe {
      let no_source = _mm256_set1_epi32(-1);
      let zero = _mm256_setzero_si256();
      let one = _mm256_set1_epi32(1);
      let pooled_lane_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
      let mut bucket_deepest = [no_source; 64];
      let mut source_slot_1 = [0; 64];
      let mut empty_slot_0 = [0; 64];
      let mut empty_slot_1 = [0; 64];

      for depth in 0..self.h {
        let metadata = prepare_level_metadata(self.tree.get_path_at_depth(depth, path), path);
        bucket_deepest[depth] = metadata.deepest;
        source_slot_1[depth] = metadata.source_slot_1;
        empty_slot_0[depth] = metadata.empty_slot_0;
        empty_slot_1[depth] = metadata.empty_slot_1;
      }

      // PrepareDeepest starts with the unified stash. Each stash block updates
      // exactly one vector lane, selected by key & (POOLED_LANES - 1).
      let mut dst = no_source;
      let mut root_source_vector = no_source;
      for stash_index in 0..STASH_SIZE {
        let block = &self.stash[stash_index];
        let block_positions = _mm256_set1_epi32(block.pos as i32);
        let block_depth = legal_depths(block_positions, path);
        let block_pooled_lane = _mm256_and_si256(
          _mm256_set1_epi32(block.key as i32),
          _mm256_set1_epi32((POOLED_LANES - 1) as i32),
        );
        let belongs_to_lane = _mm256_cmpeq_epi32_mask(block_pooled_lane, pooled_lane_indices);
        let nonempty = !_mm256_cmpeq_epi32_mask(block_positions, no_source);
        let deeper = _mm256_cmpgt_epi32_mask(block_depth, dst) & belongs_to_lane & nonempty;
        dst = _mm256_mask_mov_epi32(dst, deeper, block_depth);
        root_source_vector =
          _mm256_mask_mov_epi32(root_source_vector, deeper, _mm256_set1_epi32(stash_index as i32));
      }

      // Root slots follow the stash in scan order. Strict comparison preserves
      // stash-over-root ties and the metadata helper preserves slot-0 ties.
      let root_is_deeper = _mm256_cmpgt_epi32_mask(bucket_deepest[0], dst);
      let root_slot = _mm256_mask_mov_epi32(zero, source_slot_1[0], one);
      let root_index = _mm256_add_epi32(_mm256_set1_epi32(STASH_SIZE as i32), root_slot);
      root_source_vector = _mm256_mask_mov_epi32(root_source_vector, root_is_deeper, root_index);
      dst = _mm256_mask_mov_epi32(dst, root_is_deeper, bucket_deepest[0]);

      let mut src =
        _mm256_mask_mov_epi32(no_source, !_mm256_cmpeq_epi32_mask(dst, no_source), zero);
      let mut deepest = [no_source; 64];

      // PrepareDeepest's level recurrence remains sequential across levels but
      // updates all eight independent pooled lanes in every instruction.
      for depth in 1..self.h {
        let depth_vector = _mm256_set1_epi32(depth as i32);
        let can_reach_level = _mm256_cmpgt_epi32_mask(dst, _mm256_set1_epi32(depth as i32 - 1));
        deepest[depth] = _mm256_mask_mov_epi32(no_source, can_reach_level, src);

        let level_is_deeper = _mm256_cmpgt_epi32_mask(bucket_deepest[depth], dst);
        src = _mm256_mask_mov_epi32(src, level_is_deeper, depth_vector);
        dst = _mm256_mask_mov_epi32(dst, level_is_deeper, bucket_deepest[depth]);
      }

      // PrepareTarget is the reverse vector recurrence. Target vectors are only
      // materialized because the later movement pass consumes one level at a time.
      src = no_source;
      dst = no_source;
      let mut target = [[-1; POOLED_LANES]; 64];
      for depth in (1..self.h).rev() {
        let depth_vector = _mm256_set1_epi32(depth as i32);
        let is_source = _mm256_cmpeq_epi32_mask(depth_vector, src);
        let target_at_depth = _mm256_mask_mov_epi32(no_source, is_source, dst);
        target[depth] = std::mem::transmute(target_at_depth);
        src = _mm256_mask_mov_epi32(src, is_source, no_source);
        dst = _mm256_mask_mov_epi32(dst, is_source, no_source);

        let destination_is_empty = _mm256_cmpeq_epi32_mask(dst, no_source);
        let target_exists = !_mm256_cmpeq_epi32_mask(target_at_depth, no_source);
        let deepest_exists = !_mm256_cmpeq_epi32_mask(deepest[depth], no_source);
        let has_empty = empty_slot_0[depth] | empty_slot_1[depth];
        let change = ((destination_is_empty & has_empty) | target_exists) & deepest_exists;
        src = _mm256_mask_mov_epi32(src, change, deepest[depth]);
        dst = _mm256_mask_mov_epi32(dst, change, depth_vector);
      }

      let root_has_target = _mm256_cmpeq_epi32_mask(src, zero);
      target[0] = std::mem::transmute(_mm256_mask_mov_epi32(no_source, root_has_target, dst));
      let root_source = std::mem::transmute(root_source_vector);

      BatchEvictionPlan { root_source, target, source_slot_1, empty_slot_0, empty_slot_1 }
    }
  }
  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  fn move_batch_breadth_first(&mut self, path: PositionType, plan: &BatchEvictionPlan) {
    // Each 64-byte block occupies one AVX-512 register. Scan the stash once,
    // routing the selected source into one of eight independent held chains.
    let dummy = unsafe { _mm512_set1_epi32(-1) };
    let mut held_0 = dummy;
    let mut held_1 = dummy;
    let mut held_2 = dummy;
    let mut held_3 = dummy;
    let mut held_4 = dummy;
    let mut held_5 = dummy;
    let mut held_6 = dummy;
    let mut held_7 = dummy;

    macro_rules! route_stash_source {
      ($pooled_lane:literal, $held:ident, $stash_index:ident, $stash_value:ident) => {{
        let selected = (plan.target[0][$pooled_lane] != -1)
          & (plan.root_source[$pooled_lane] == $stash_index as i32);
        let mask = 0u16.wrapping_sub(selected as u16);
        let old_held = $held;
        $held = unsafe { _mm512_mask_mov_epi32($held, mask, $stash_value) };
        $stash_value = unsafe { _mm512_mask_mov_epi32($stash_value, mask, old_held) };
      }};
    }

    for stash_index in 0..STASH_SIZE {
      let stash_ptr = (&mut self.stash[stash_index] as *mut Block).cast::<__m512i>();
      let mut stash_value = unsafe { _mm512_load_si512(stash_ptr) };
      route_stash_source!(0, held_0, stash_index, stash_value);
      route_stash_source!(1, held_1, stash_index, stash_value);
      route_stash_source!(2, held_2, stash_index, stash_value);
      route_stash_source!(3, held_3, stash_index, stash_value);
      route_stash_source!(4, held_4, stash_index, stash_value);
      route_stash_source!(5, held_5, stash_index, stash_value);
      route_stash_source!(6, held_6, stash_index, stash_value);
      route_stash_source!(7, held_7, stash_index, stash_value);
      unsafe { _mm512_store_si512(stash_ptr, stash_value) };
    }

    // A pooled lane whose level-0 source is in the root still holds dummy here.
    // Both root slots are scanned regardless of the secret source selector.
    let root = self.tree.get_path_at_depth_mut(0, path);
    macro_rules! move_root_lane {
      ($pooled_lane:literal, $held:ident) => {{
        let has_target = plan.target[0][$pooled_lane] != -1;
        for slot in 0..SLOTS_PER_POOLED_LANE {
          let selected =
            has_target & (plan.root_source[$pooled_lane] == (STASH_SIZE + slot) as i32);
          let block_index = $pooled_lane * SLOTS_PER_POOLED_LANE + slot;
          $held = swap_register_with_block($held, &mut root.0[block_index], selected);
        }
      }};
    }
    move_root_lane!(0, held_0);
    move_root_lane!(1, held_1);
    move_root_lane!(2, held_2);
    move_root_lane!(3, held_3);
    move_root_lane!(4, held_4);
    move_root_lane!(5, held_5);
    move_root_lane!(6, held_6);
    move_root_lane!(7, held_7);

    let mut destination_0 = plan.target[0][0];
    let mut destination_1 = plan.target[0][1];
    let mut destination_2 = plan.target[0][2];
    let mut destination_3 = plan.target[0][3];
    let mut destination_4 = plan.target[0][4];
    let mut destination_5 = plan.target[0][5];
    let mut destination_6 = plan.target[0][6];
    let mut destination_7 = plan.target[0][7];

    // Each macro expansion is an independent held-register dependency chain.
    macro_rules! move_lane_at_depth {
      ($pooled_lane:literal, $held:ident, $destination:ident, $depth:ident, $bucket:ident) => {{
        let target = plan.target[$depth][$pooled_lane];
        let has_target = target != -1;
        let place_held = ($destination == $depth as i32) & (!has_target);
        let source_is_slot_1 = ((plan.source_slot_1[$depth] >> $pooled_lane) & 1) != 0;
        let mut written = false;

        for slot in 0..SLOTS_PER_POOLED_LANE {
          let is_source_slot = source_is_slot_1 == (slot == 1);
          let read_source = has_target & is_source_slot;
          let slot_is_empty = if slot == 0 {
            ((plan.empty_slot_0[$depth] >> $pooled_lane) & 1) != 0
          } else {
            ((plan.empty_slot_1[$depth] >> $pooled_lane) & 1) != 0
          };
          let write_destination = place_held & slot_is_empty & (!written);
          let block_index = $pooled_lane * SLOTS_PER_POOLED_LANE + slot;
          $held = swap_register_with_block(
            $held,
            &mut $bucket.0[block_index],
            read_source | write_destination,
          );
          written |= write_destination;
        }

        $destination.cmov(&target, has_target | place_held);
      }};
    }

    // All eight chains process a level before any chain advances to the next.
    for depth in 1..self.h {
      let bucket = self.tree.get_path_at_depth_mut(depth, path);
      move_lane_at_depth!(0, held_0, destination_0, depth, bucket);
      move_lane_at_depth!(1, held_1, destination_1, depth, bucket);
      move_lane_at_depth!(2, held_2, destination_2, depth, bucket);
      move_lane_at_depth!(3, held_3, destination_3, depth, bucket);
      move_lane_at_depth!(4, held_4, destination_4, depth, bucket);
      move_lane_at_depth!(5, held_5, destination_5, depth, bucket);
      move_lane_at_depth!(6, held_6, destination_6, depth, bucket);
      move_lane_at_depth!(7, held_7, destination_7, depth, bucket);
    }

    let dummy = unsafe { _mm512_set1_epi32(-1) };
    let all_held_blocks_are_dummy = unsafe {
      _mm512_cmpeq_epi32_mask(held_0, dummy)
        & _mm512_cmpeq_epi32_mask(held_1, dummy)
        & _mm512_cmpeq_epi32_mask(held_2, dummy)
        & _mm512_cmpeq_epi32_mask(held_3, dummy)
        & _mm512_cmpeq_epi32_mask(held_4, dummy)
        & _mm512_cmpeq_epi32_mask(held_5, dummy)
        & _mm512_cmpeq_epi32_mask(held_6, dummy)
        & _mm512_cmpeq_epi32_mask(held_7, dummy)
    };
    debug_assert_eq!(all_held_blocks_are_dummy, 0xffff);
  }

  // Reads one pooled lane on a path, evicts it, and writes it back.
  #[allow(dead_code)]
  fn perform_eviction(&mut self, pos: PositionType, pooled_lane: usize) {
    debug_assert!((pos as usize) < self.max_n);
    self.read_path_and_get_nodes(pos, pooled_lane);
    self.evict_once_fast(pos, pooled_lane);
    self.write_back_path(pos, pooled_lane);
  }

  /// Evicts all eight pooled lanes on one shared path in level-major order.
  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  fn perform_deterministic_eviction_batch(&mut self) {
    let evict_pos = self.evict_counter;
    let plan = self.prepare_batch_eviction_plan(evict_pos);
    self.move_batch_breadth_first(evict_pos, &plan);
    self.evict_counter = (self.evict_counter + 1) % (self.max_n as PositionType);
  }

  /// Scalar fallback: evict each pooled lane separately on the same path.
  #[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  )))]
  fn perform_deterministic_eviction_batch(&mut self) {
    let evict_pos = self.evict_counter;
    for pooled_lane in 0..POOLED_LANES {
      self.perform_eviction(evict_pos, pooled_lane);
    }
    self.evict_counter = (self.evict_counter + 1) % (self.max_n as PositionType);
  }

  #[doc(hidden)]
  pub fn benchmark_deterministic_eviction_batch(&mut self) {
    self.perform_deterministic_eviction_batch();
  }

  /// Updates a value in the ORAM using a provided update function.
  /// If the element is not in the ORAM, the update function receives the all-ones dummy payload before insertion.
  /// # Arguments
  /// * `pos` - The current position of the block.
  /// * `new_pos` - The new position of the block, should be uniformly random on the size of the ORAM.
  /// * `key` - The key of the block.
  /// * `update_func` - The function to update the value.
  ///
  /// # Returns
  /// * A tuple containing a boolean indicating if the element was found and the result of the update function.
  pub fn update<T, F>(
    &mut self,
    pos: PositionType,
    new_pos: PositionType,
    key: Key,
    update_func: F,
  ) -> (bool, T)
  where
    F: FnOnce(&mut [u8; DATA_SIZE]) -> T,
  {
    debug_assert!((pos as usize) < self.max_n);
    debug_assert!((new_pos as usize) < self.max_n);

    debug_assert_ne!(key, Key::MAX);

    let pooled_lane = pooled_lane_for_key(key);
    self.read_path_and_get_nodes(pos, pooled_lane);

    let mut block = read_and_remove_element(&mut self.stash, key);
    let found = !block.is_empty();
    let rv = update_func(&mut block.data);
    block.pos = new_pos;
    block.key = key;

    let inserted = write_block_to_empty_slot(&mut self.stash[..STASH_SIZE], &block);
    debug_assert!(inserted); // Succeeds due to Inv1.

    self.evict_once_fast(pos, pooled_lane);
    self.write_back_path(pos, pooled_lane);

    self.eviction_credit += EVICTIONS_PER_OP_NUMERATOR;
    if self.eviction_credit >= EVICTION_CREDITS_PER_BATCH {
      self.perform_deterministic_eviction_batch();
      self.eviction_credit -= EVICTION_CREDITS_PER_BATCH;
    }

    (found, rv)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn real_block(pos: PositionType, key: Key, byte: u8) -> Block {
    Block { pos, key, data: [byte; DATA_SIZE] }
  }

  #[test]
  fn read_and_remove_finds_either_lane_in_a_later_bucket() {
    for lane in 0..SLOTS_PER_POOLED_LANE {
      let mut blocks = AlignedStash::new(6);
      blocks[0] = real_block(1, 10, 0x10);
      blocks[2 + lane] = real_block(7, 42, 0x42);
      blocks[5] = real_block(3, 11, 0x11);

      let result = read_and_remove_element(&mut blocks, 42);
      assert!(!result.is_empty());
      assert_eq!(result.data, [0x42; DATA_SIZE]);
      assert_eq!(blocks[2 + lane].pos, DUMMY_POS);
      assert_eq!(blocks[2 + lane].key, Key::MAX);
      assert_eq!(blocks[2 + lane].data, [u8::MAX; DATA_SIZE]);
      assert_eq!(blocks[0].data, [0x10; DATA_SIZE]);
      assert_eq!(blocks[5].data, [0x11; DATA_SIZE]);
    }
  }

  #[test]
  fn read_and_remove_leaves_input_and_result_unchanged_when_absent() {
    let mut blocks = AlignedStash::new(4);
    blocks[0] = real_block(1, 10, 0x10);
    blocks[3] = real_block(3, 11, 0x11);
    let before = blocks.to_vec();

    let result = read_and_remove_element(&mut blocks, 42);
    assert!(result.is_empty());
    assert_eq!(result.key, Key::MAX);
    assert_eq!(result.data, [u8::MAX; DATA_SIZE]);
    for (actual, expected) in blocks.iter().zip(before.iter()) {
      assert_eq!(actual.pos, expected.pos);
      assert_eq!(actual.key, expected.key);
      assert_eq!(actual.data, expected.data);
    }
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  #[test]
  fn level_position_gather_groups_slots_by_pooled_lane() {
    let mut bucket = Bucket::default();
    for pooled_lane in 0..POOLED_LANES {
      bucket.0[pooled_lane * SLOTS_PER_POOLED_LANE].pos = 100 + pooled_lane as PositionType;
      bucket.0[pooled_lane * SLOTS_PER_POOLED_LANE + 1].pos = 200 + pooled_lane as PositionType;
    }

    let (slot_0, slot_1) = gather_level_positions(&bucket);
    let slot_0: [PositionType; POOLED_LANES] = unsafe { std::mem::transmute(slot_0) };
    let slot_1: [PositionType; POOLED_LANES] = unsafe { std::mem::transmute(slot_1) };
    assert_eq!(slot_0, [100, 101, 102, 103, 104, 105, 106, 107]);
    assert_eq!(slot_1, [200, 201, 202, 203, 204, 205, 206, 207]);
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  #[test]
  fn level_metadata_selects_deeper_source_and_tracks_both_empty_slots() {
    let path = 0x5a5a_5a5a;
    let slot_0_depths = [0, 1, -1, 4, 6, 2, 5, -1];
    let slot_1_depths = [7, 3, 2, 4, 1, -1, 6, -1];
    let mut bucket = Bucket::default();

    for pooled_lane in 0..POOLED_LANES {
      for (slot, depths) in [slot_0_depths, slot_1_depths].iter().enumerate() {
        let depth = depths[pooled_lane];
        if depth >= 0 {
          let block_index = pooled_lane * SLOTS_PER_POOLED_LANE + slot;
          bucket.0[block_index] =
            real_block(path ^ (1 << depth), block_index as Key, block_index as u8);
        }
      }
    }

    let metadata = prepare_level_metadata(&bucket, path);
    let deepest: [i32; POOLED_LANES] = unsafe { std::mem::transmute(metadata.deepest) };
    assert_eq!(deepest, [7, 3, 2, 4, 6, 2, 6, -1]);
    // Slot 1 wins only strictly; pooled lane 3 is a tie and stays on slot 0.
    assert_eq!(metadata.source_slot_1, 0b0100_0111);
    assert_eq!(metadata.empty_slot_0, 0b1000_0100);
    assert_eq!(metadata.empty_slot_1, 0b1010_0000);
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vpopcntdq"
  ))]
  #[test]
  fn vector_root_source_matches_scalar_scan() {
    let mut stash = AlignedStash::new(S + SLOTS_PER_POOLED_LANE);
    let mut state = 0x9e37_79b9u32;

    for round in 0..64 {
      for index in 0..S + SLOTS_PER_POOLED_LANE {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        stash[index] = if state % 5 == 0 {
          Block::default()
        } else {
          real_block(state.rotate_left(7), state.rotate_right(11), index as u8)
        };
      }

      for pooled_lane in 0..POOLED_LANES {
        let path = state.wrapping_add(round * 0x1021).rotate_left(pooled_lane as u32);
        let mut scalar_depth = -1i32;
        let mut scalar_index = 0i32;
        for index in 0..S + SLOTS_PER_POOLED_LANE {
          let block = &stash[index];
          let depth = common_suffix_length(block.pos, path) as i32;
          let belongs_to_lane = (index >= S) | (pooled_lane_for_key(block.key) == pooled_lane);
          let replaces_best = !block.is_empty() & belongs_to_lane & (depth > scalar_depth);
          scalar_depth.cmov(&depth, replaces_best);
          scalar_index.cmov(&(index as i32), replaces_best);
        }

        assert_eq!(find_root_source::<S>(&stash, path, pooled_lane), (scalar_depth, scalar_index));
      }
    }
  }

  #[test]
  fn wide_bucket_layout_and_lane_routing_are_exact() {
    assert_eq!(std::mem::size_of::<Block>(), 64);
    assert_eq!(std::mem::size_of::<PooledLane>(), 128);
    assert_eq!(std::mem::size_of::<Bucket>(), 1024);
    assert_eq!(std::mem::align_of::<Bucket>(), 64);
    for key in 0..32 {
      assert_eq!(pooled_lane_for_key(key), key as usize & 7);
    }
  }

  #[test]
  fn each_pooled_lane_has_n_over_eight_leaves_at_half_bottom_load() {
    let oram = OptimizedCircuitORAMBigValues::new(1024);
    assert_eq!(oram.max_n, 1024 / POOLED_LANES);
    assert_eq!(oram.h, 8);
    let bottom_slots_per_pooled_lane = oram.max_n * SLOTS_PER_POOLED_LANE;
    let target_blocks_per_pooled_lane = oram.capacity / POOLED_LANES;
    assert_eq!(target_blocks_per_pooled_lane * 2, bottom_slots_per_pooled_lane);
    assert_eq!(oram.stash.len(), S + oram.h * SLOTS_PER_POOLED_LANE);
  }

  #[test]
  fn ordinary_update_touches_only_the_selected_pooled_lane_in_the_tree() {
    let mut oram = OptimizedCircuitORAMBigValues::new(64);
    oram.update(0, 1, 0, |data| data.fill(0x2a));

    for bucket in &oram.tree.tree {
      for block in &bucket.0[SLOTS_PER_POOLED_LANE..] {
        assert!(block.is_empty());
      }
    }
    assert_eq!(oram.eviction_credit, EVICTIONS_PER_OP_NUMERATOR);
    assert_eq!(oram.evict_counter, 0);
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  #[test]
  fn breadth_first_batch_matches_eight_scalar_evictions() {
    let mut batched = OptimizedCircuitORAMBigValues::new(256);
    let mut scalar = OptimizedCircuitORAMBigValues::new(256);

    for key in 0..32 {
      let new_pos = ((key * 13 + 7) % batched.max_n) as PositionType;
      for oram in [&mut batched, &mut scalar] {
        oram.update(0, new_pos, key as Key, |data| data.fill(key as u8));
        // Keep setup updates from triggering the deterministic batch under test.
        oram.eviction_credit = 0;
      }
    }

    for path in [0, 1, 3, 7, 11, 31] {
      let plan = batched.prepare_batch_eviction_plan(path);
      batched.move_batch_breadth_first(path, &plan);
      for pooled_lane in 0..POOLED_LANES {
        scalar.perform_eviction(path, pooled_lane);
      }

      for (actual, expected) in batched.stash[..S].iter().zip(&scalar.stash[..S]) {
        assert_eq!(bytemuck::bytes_of(actual), bytemuck::bytes_of(expected));
      }
      for (actual_bucket, expected_bucket) in batched.tree.tree.iter().zip(&scalar.tree.tree) {
        assert_eq!(bytemuck::bytes_of(actual_bucket), bytemuck::bytes_of(expected_bucket));
      }
    }
  }

  #[test]
  fn deterministic_evictions_run_as_one_all_lane_path_per_batch() {
    let mut oram = OptimizedCircuitORAMBigValues::new(512);
    assert_eq!(oram.max_n, 64);

    let mut expected_credit = 0;
    let mut expected_batches = 0;
    for key in 0..32 {
      oram.update(0, key, key, |data| data.fill(key as u8));

      expected_credit += EVICTIONS_PER_OP_NUMERATOR;
      if expected_credit >= EVICTION_CREDITS_PER_BATCH {
        expected_credit -= EVICTION_CREDITS_PER_BATCH;
        expected_batches += 1;
      }

      assert_eq!(oram.eviction_credit, expected_credit);
      assert_eq!(oram.evict_counter, (expected_batches % oram.max_n) as PositionType);
    }
  }
  #[test]
  fn updates_round_trip_across_all_pooled_lanes_and_batches() {
    const N: usize = 128;
    let mut oram = OptimizedCircuitORAMBigValues::new(N);
    let mut positions = [0; 64];

    for key in 0..positions.len() {
      let new_pos = ((key * 7 + 3) % oram.max_n) as PositionType;
      let (found, ()) = oram.update(positions[key], new_pos, key as Key, |data| {
        data.fill(0);
        data[..8].copy_from_slice(&(key as u64).to_le_bytes());
      });
      assert!(!found);
      positions[key] = new_pos;
    }

    for key in 0..positions.len() {
      let new_pos = ((key * 11 + 5) % oram.max_n) as PositionType;
      let (found, stored) = oram.update(positions[key], new_pos, key as Key, |data| {
        u64::from_le_bytes(data[..8].try_into().unwrap())
      });
      assert!(found, "missing key {key}");
      assert_eq!(stored, key as u64);
      positions[key] = new_pos;
    }
  }

  fn assert_stash_size_updates_round_trip<const STASH_SIZE: usize>() {
    const N: usize = 128;
    let mut oram = OptimizedCircuitORAMBigValuesWithStash::<STASH_SIZE>::new(N);
    let mut positions = [0; 64];
    assert_eq!(oram.stash.len(), STASH_SIZE + oram.h * SLOTS_PER_POOLED_LANE);

    for key in 0..positions.len() {
      let new_pos = ((key * 7 + 3) % oram.max_n) as PositionType;
      let (found, ()) = oram.update(positions[key], new_pos, key as Key, |data| {
        data.fill(0);
        data[..8].copy_from_slice(&(key as u64).to_le_bytes());
      });
      assert!(!found);
      positions[key] = new_pos;
    }

    for key in 0..positions.len() {
      let new_pos = ((key * 11 + 5) % oram.max_n) as PositionType;
      let (found, stored) = oram.update(positions[key], new_pos, key as Key, |data| {
        u64::from_le_bytes(data[..8].try_into().unwrap())
      });
      assert!(found, "missing key {key}");
      assert_eq!(stored, key as u64);
      positions[key] = new_pos;
    }
  }

  #[test]
  fn stash_40_updates_round_trip_across_all_pooled_lanes() {
    assert_stash_size_updates_round_trip::<40>();
  }

  #[test]
  fn stash_56_updates_round_trip_across_all_pooled_lanes() {
    assert_stash_size_updates_round_trip::<56>();
  }

  #[test]
  fn stash_80_updates_round_trip_across_all_pooled_lanes() {
    assert_stash_size_updates_round_trip::<80>();
  }
}
