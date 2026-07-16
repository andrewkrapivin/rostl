//! Pooled Circuit ORAM for 24-byte values with split 64-bit positions and
//! 32-byte key/value chunks.
//!
//! Each sixteen-slot bucket is 640 bytes: two slot-major cache lines of
//! positions followed by eight lane-major cache lines of key/value pairs.

#![allow(clippy::needless_bitwise_bool)]
#![allow(missing_docs)]

use bytemuck::{Pod, Zeroable};
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
use core::arch::x86_64::{
  __m256i, __m512i, __mmask8, _mm256_and_si256, _mm512_add_epi64, _mm512_and_si512,
  _mm512_castsi512_si256, _mm512_cmpeq_epi64_mask, _mm512_cmpgt_epi64_mask,
  _mm512_extracti64x4_epi64, _mm512_load_si512, _mm512_mask_mov_epi64, _mm512_permutexvar_epi64,
  _mm512_popcnt_epi64, _mm512_set1_epi64, _mm512_setr_epi64, _mm512_setzero_si512,
  _mm512_store_si512, _mm512_sub_epi64, _mm512_xor_si512,
};
use rostl_primitives::{
  cmov_body, cxchg_body, impl_cmov_for_pod,
  traits::{_Cmovbase, Cmov},
};

use crate::heap_tree::HeapTree;

pub const SLOTS_PER_POOLED_LANE: usize = 2;
pub const POOLED_LANES: usize = 8;
pub const BLOCKS_PER_BUCKET: usize = POOLED_LANES * SLOTS_PER_POOLED_LANE;
pub const DATA_SIZE: usize = 24;
pub const S: usize = 40;
pub const EVICTIONS_PER_OP_NUMERATOR: usize = 3;
pub const EVICTIONS_PER_OP_DENOMINATOR: usize = 1;
const EVICTIONS_PER_BATCH: usize = BLOCKS_PER_BUCKET;
const EVICTION_CREDITS_PER_BATCH: usize = EVICTIONS_PER_BATCH * EVICTIONS_PER_OP_DENOMINATOR;

pub type PositionType = u64;
pub type Key = u64;
pub const DUMMY_POS: PositionType = PositionType::MAX;
pub const DUMMY_KEY: Key = Key::MAX;

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
const KEY_BROADCAST_INDICES: __m512i = unsafe { std::mem::transmute([0i64, 0, 0, 0, 4, 4, 4, 4]) };

/// One naturally vector-sized key/value chunk.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct KeyValue {
  pub key: Key,
  pub data: [u8; DATA_SIZE],
}

impl Default for KeyValue {
  fn default() -> Self {
    Self { key: DUMMY_KEY, data: [u8::MAX; DATA_SIZE] }
  }
}

impl_cmov_for_pod!(KeyValue);

const _: () = assert!(std::mem::size_of::<KeyValue>() == 32);
const _: () = assert!(std::mem::align_of::<KeyValue>() == 8);
const _: () = assert!(std::mem::offset_of!(KeyValue, key) == 0);
const _: () = assert!(std::mem::offset_of!(KeyValue, data) == 8);

/// A logical block. Tree buckets store these fields in separate arrays.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Block {
  pub pos: PositionType,
  pub key_value: KeyValue,
}

impl Default for Block {
  fn default() -> Self {
    Self { pos: DUMMY_POS, key_value: KeyValue::default() }
  }
}

impl_cmov_for_pod!(Block);

impl Block {
  #[inline(always)]
  pub const fn is_empty(&self) -> bool {
    self.pos == DUMMY_POS
  }

  #[inline(always)]
  pub const fn key(&self) -> Key {
    self.key_value.key
  }
}

const _: () = assert!(std::mem::size_of::<Block>() == 40);
const _: () = assert!(std::mem::align_of::<Block>() == 8);
const _: () = assert!(std::mem::offset_of!(Block, pos) == 0);
const _: () = assert!(std::mem::offset_of!(Block, key_value) == 8);

/// Sixteen blocks in split position/key-value layout.
///
/// Positions are slot-major: positions 0..8 are slot zero across all pooled
/// lanes and positions 8..16 are slot one. Key/value chunks are lane-major so
/// both chunks for one pooled lane occupy exactly one cache line.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bucket {
  positions: [PositionType; BLOCKS_PER_BUCKET],
  key_values: [KeyValue; BLOCKS_PER_BUCKET],
}

unsafe impl Zeroable for Bucket {}
unsafe impl Pod for Bucket {}

impl Default for Bucket {
  fn default() -> Self {
    Self {
      positions: [DUMMY_POS; BLOCKS_PER_BUCKET],
      key_values: [KeyValue::default(); BLOCKS_PER_BUCKET],
    }
  }
}

impl Bucket {
  #[inline(always)]
  const fn position_index(pooled_lane: usize, slot: usize) -> usize {
    slot * POOLED_LANES + pooled_lane
  }

  #[inline(always)]
  const fn key_value_index(pooled_lane: usize, slot: usize) -> usize {
    pooled_lane * SLOTS_PER_POOLED_LANE + slot
  }

  #[inline(always)]
  fn block(&self, pooled_lane: usize, slot: usize) -> Block {
    Block {
      pos: self.positions[Self::position_index(pooled_lane, slot)],
      key_value: self.key_values[Self::key_value_index(pooled_lane, slot)],
    }
  }

  #[inline(always)]
  fn set_block(&mut self, pooled_lane: usize, slot: usize, block: Block) {
    self.positions[Self::position_index(pooled_lane, slot)] = block.pos;
    self.key_values[Self::key_value_index(pooled_lane, slot)] = block.key_value;
  }
}

const _: () = assert!(std::mem::size_of::<Bucket>() == 640);
const _: () = assert!(std::mem::align_of::<Bucket>() == 64);
const _: () = assert!(std::mem::offset_of!(Bucket, positions) == 0);
const _: () = assert!(std::mem::offset_of!(Bucket, key_values) == 128);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
struct PositionChunk([PositionType; 8]);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
struct KeyValuePair([KeyValue; 2]);

/// Split stash storage: all positions occupy aligned cacheline chunks, followed
/// logically by independently aligned 32-byte key/value pairs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitStash {
  len: usize,
  positions: Vec<PositionChunk>,
  key_values: Vec<KeyValuePair>,
}

impl SplitStash {
  fn new(blocks: usize) -> Self {
    let position_chunks = blocks.div_ceil(8);
    Self {
      len: blocks,
      positions: vec![PositionChunk([DUMMY_POS; 8]); position_chunks],
      key_values: vec![KeyValuePair([KeyValue::default(); 2]); position_chunks * 4],
    }
  }

  #[inline(always)]
  pub fn len(&self) -> usize {
    self.len
  }

  #[inline(always)]
  pub fn is_empty(&self) -> bool {
    self.len == 0
  }

  #[inline(always)]
  fn position(&self, index: usize) -> PositionType {
    self.positions[index / 8].0[index % 8]
  }

  #[inline(always)]
  fn set_position(&mut self, index: usize, position: PositionType) {
    self.positions[index / 8].0[index % 8] = position;
  }

  #[inline(always)]
  fn key_value(&self, index: usize) -> KeyValue {
    self.key_values[index / 2].0[index % 2]
  }

  #[inline(always)]
  fn set_key_value(&mut self, index: usize, key_value: KeyValue) {
    self.key_values[index / 2].0[index % 2] = key_value;
  }

  #[inline(always)]
  fn block(&self, index: usize) -> Block {
    Block { pos: self.position(index), key_value: self.key_value(index) }
  }

  #[inline(always)]
  fn set_block(&mut self, index: usize, block: Block) {
    self.set_position(index, block.pos);
    self.set_key_value(index, block.key_value);
  }

  #[inline(always)]
  fn swap_with_block(&mut self, index: usize, held: &mut Block, choice: bool) {
    let mut stash_block = self.block(index);
    held.cxchg(&mut stash_block, choice);
    self.set_block(index, stash_block);
  }
}

#[inline(always)]
const fn tree_index(depth: usize, path: PositionType) -> usize {
  let level_offset = (1usize << depth) - 1;
  level_offset + (path as usize & level_offset)
}

#[inline(always)]
const fn common_suffix_length(a: PositionType, b: PositionType) -> u32 {
  (a ^ b).trailing_zeros()
}

#[inline(always)]
const fn pooled_lane_for_key(key: Key) -> usize {
  key as usize & (POOLED_LANES - 1)
}

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
struct LevelMetadata {
  deepest: __m512i,
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
  root_source: [i64; POOLED_LANES],
  target: [[i64; POOLED_LANES]; 64],
  source_slot_1: [__mmask8; 64],
  empty_slot_0: [__mmask8; 64],
  empty_slot_1: [__mmask8; 64],
}

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
#[inline(always)]
fn legal_depths(positions: __m512i, path: PositionType) -> __m512i {
  unsafe {
    let diff = _mm512_xor_si512(positions, _mm512_set1_epi64(path as i64));
    let low_bit = _mm512_and_si512(diff, _mm512_sub_epi64(_mm512_setzero_si512(), diff));
    _mm512_popcnt_epi64(_mm512_sub_epi64(low_bit, _mm512_set1_epi64(1)))
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
    // These are the two position cache lines. No gather or permutation is
    // needed because the bucket stores positions slot-major.
    let slot_0 = _mm512_load_si512(bucket.positions.as_ptr().cast::<__m512i>());
    let slot_1 = _mm512_load_si512(bucket.positions.as_ptr().add(POOLED_LANES).cast::<__m512i>());
    let dummy = _mm512_set1_epi64(-1);
    let empty_slot_0 = _mm512_cmpeq_epi64_mask(slot_0, dummy);
    let empty_slot_1 = _mm512_cmpeq_epi64_mask(slot_1, dummy);
    let depth_0 = _mm512_mask_mov_epi64(dummy, !empty_slot_0, legal_depths(slot_0, path));
    let depth_1 = _mm512_mask_mov_epi64(dummy, !empty_slot_1, legal_depths(slot_1, path));
    let source_slot_1 = _mm512_cmpgt_epi64_mask(depth_1, depth_0);
    let deepest = _mm512_mask_mov_epi64(depth_0, source_slot_1, depth_1);
    LevelMetadata { deepest, source_slot_1, empty_slot_0, empty_slot_1 }
  }
}

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
#[inline(always)]
fn pack_block(block: Block) -> __m512i {
  let key_value: [u64; 4] = bytemuck::cast(block.key_value);
  unsafe {
    _mm512_setr_epi64(
      block.pos as i64,
      key_value[0] as i64,
      key_value[1] as i64,
      key_value[2] as i64,
      key_value[3] as i64,
      -1,
      -1,
      -1,
    )
  }
}

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
#[inline(always)]
fn unpack_block(block: __m512i) -> Block {
  let words: [u64; 8] = unsafe { std::mem::transmute(block) };
  Block { pos: words[0], key_value: bytemuck::cast([words[1], words[2], words[3], words[4]]) }
}

#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
#[inline(always)]
fn swap_register_with_bucket(
  bucket: &mut Bucket,
  pooled_lane: usize,
  slot: usize,
  held: __m512i,
  choice: bool,
) -> __m512i {
  unsafe {
    let tree_block = pack_block(bucket.block(pooled_lane, slot));
    let mask = 0u8.wrapping_sub(choice as u8);
    let new_tree_block = _mm512_mask_mov_epi64(tree_block, mask, held);
    let new_held = _mm512_mask_mov_epi64(held, mask, tree_block);
    bucket.set_block(pooled_lane, slot, unpack_block(new_tree_block));
    new_held
  }
}
#[inline]
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
))]
fn read_and_remove_element(stash: &mut SplitStash, key: Key) -> Block {
  let mut found_mask = 0u8;
  let key_value = unsafe {
    let desired_key = _mm512_set1_epi64(key as i64);
    let dummy = _mm512_set1_epi64(-1);
    let mut held = dummy;

    for group in 0..stash.positions.len() {
      let mut position_matches = 0u8;
      for pair_offset in 0..4 {
        let pair_index = group * 4 + pair_offset;
        let pair_ptr = (&mut stash.key_values[pair_index] as *mut KeyValuePair).cast::<__m512i>();
        let value = _mm512_load_si512(pair_ptr);
        let keys = _mm512_permutexvar_epi64(KEY_BROADCAST_INDICES, value);
        let matched = _mm512_cmpeq_epi64_mask(keys, desired_key);

        debug_assert!((found_mask == 0) | (matched == 0));
        let old_held = held;
        held = _mm512_mask_mov_epi64(held, matched, value);
        _mm512_store_si512(pair_ptr, _mm512_mask_mov_epi64(value, matched, old_held));
        found_mask |= matched;
        position_matches |= (((matched & 0x0f) != 0) as u8) << (pair_offset * 2);
        position_matches |= (((matched & 0xf0) != 0) as u8) << (pair_offset * 2 + 1);
      }

      let positions_ptr = (&mut stash.positions[group] as *mut PositionChunk).cast::<__m512i>();
      let positions = _mm512_load_si512(positions_ptr);
      _mm512_store_si512(positions_ptr, _mm512_mask_mov_epi64(positions, position_matches, dummy));
    }

    let low: __m256i = _mm512_castsi512_si256(held);
    let high = _mm512_extracti64x4_epi64::<1>(held);
    _mm256_and_si256(low, high)
  };

  Block { pos: DUMMY_POS, key_value: unsafe { std::mem::transmute(key_value) } }
}

#[inline]
#[cfg(not(all(
  target_arch = "x86_64",
  target_feature = "avx512f",
  target_feature = "avx512vl",
  target_feature = "avx512vpopcntdq"
)))]
fn read_and_remove_element(stash: &mut SplitStash, key: Key) -> Block {
  let mut result = Block::default();
  let mut found = false;
  for index in 0..stash.len() {
    let matched = stash.key_value(index).key == key;
    debug_assert!((!matched) | (!found));
    stash.swap_with_block(index, &mut result, matched);
    found.cmov(&true, matched);
  }
  result
}

#[inline]
fn write_block_to_empty_slot(stash: &mut SplitStash, length: usize, value: &Block) -> bool {
  let mut written = false;
  for index in 0..length {
    let choice = (stash.position(index) == DUMMY_POS) & (!written);
    let mut slot = stash.block(index);
    slot.cmov(value, choice);
    stash.set_block(index, slot);
    written.cmov(&true, choice);
  }
  written
}

/// Eight pooled two-slot ORAMs sharing one split stash and one split tree.
#[derive(Debug)]
pub struct BigKvOptimizedCircuitOramWithStash<const STASH_SIZE: usize> {
  pub capacity: usize,
  pub max_n: usize,
  pub h: usize,
  pub stash: SplitStash,
  pub tree: HeapTree<Bucket>,
  pub evict_counter: PositionType,
  pub eviction_credit: usize,
}

pub type BigKvOptimizedCircuitOram = BigKvOptimizedCircuitOramWithStash<S>;
pub type BigKvOptimizedCircuitOramS56 = BigKvOptimizedCircuitOramWithStash<56>;

impl<const STASH_SIZE: usize> BigKvOptimizedCircuitOramWithStash<STASH_SIZE> {
  pub fn new(capacity: usize) -> Self {
    debug_assert!(capacity > 0);
    debug_assert!(STASH_SIZE > 0);
    let blocks_per_lane = capacity.div_ceil(POOLED_LANES);
    let max_n = blocks_per_lane.max(2).next_power_of_two();
    let h = max_n.ilog2() as usize + 1;
    debug_assert!(h <= 64);
    let tree = HeapTree::new(h);
    let stash = SplitStash::new(STASH_SIZE + h * SLOTS_PER_POOLED_LANE);
    Self { capacity, max_n, h, stash, tree, evict_counter: 0, eviction_credit: 0 }
  }

  fn read_path_and_get_nodes(&mut self, path: PositionType, pooled_lane: usize) {
    for depth in 0..self.h {
      let bucket_index = tree_index(depth, path);
      for slot in 0..SLOTS_PER_POOLED_LANE {
        let block = self.tree.tree[bucket_index].block(pooled_lane, slot);
        self.stash.set_block(STASH_SIZE + depth * SLOTS_PER_POOLED_LANE + slot, block);
      }
    }
  }

  fn write_back_path(&mut self, path: PositionType, pooled_lane: usize) {
    for depth in 0..self.h {
      let bucket_index = tree_index(depth, path);
      for slot in 0..SLOTS_PER_POOLED_LANE {
        let block = self.stash.block(STASH_SIZE + depth * SLOTS_PER_POOLED_LANE + slot);
        self.tree.tree[bucket_index].set_block(pooled_lane, slot, block);
      }
    }
  }

  fn evict_once_fast(&mut self, path: PositionType, pooled_lane: usize) {
    let mut deepest = [-1i64; 64];
    let mut deepest_index = [0i64; 64];
    let mut target = [-1i64; 64];
    let mut has_empty = [false; 64];
    let mut source = -1i64;
    let mut destination = -1i64;

    for index in 0..STASH_SIZE + SLOTS_PER_POOLED_LANE {
      let block = self.stash.block(index);
      let depth = common_suffix_length(block.pos, path) as i64;
      let belongs_to_lane =
        (index >= STASH_SIZE) | (pooled_lane_for_key(block.key()) == pooled_lane);
      let deeper = (!block.is_empty()) & belongs_to_lane & (depth > destination);
      destination.cmov(&depth, deeper);
      deepest_index[0].cmov(&(index as i64), deeper);
    }
    source.cmov(&0, destination != -1);

    let mut index = STASH_SIZE + SLOTS_PER_POOLED_LANE;
    for level in 1..self.h {
      deepest[level].cmov(&source, destination >= level as i64);
      let mut level_deepest = -1i64;
      for _ in 0..SLOTS_PER_POOLED_LANE {
        let block = self.stash.block(index);
        let depth = common_suffix_length(block.pos, path) as i64;
        has_empty[level].cmov(&true, block.is_empty());
        let deeper = (!block.is_empty()) & (depth > level_deepest);
        level_deepest.cmov(&depth, deeper);
        deepest_index[level].cmov(&(index as i64), deeper);
        index += 1;
      }
      let deeper = level_deepest > destination;
      source.cmov(&(level as i64), deeper);
      destination.cmov(&level_deepest, deeper);
    }

    source = -1;
    destination = -1;
    for level in (1..self.h).rev() {
      let is_source = level as i64 == source;
      target[level].cmov(&destination, is_source);
      source.cmov(&-1, is_source);
      destination.cmov(&-1, is_source);
      let change =
        (((destination == -1) & has_empty[level]) | (target[level] != -1)) & (deepest[level] != -1);
      source.cmov(&deepest[level], change);
      destination.cmov(&(level as i64), change);
    }
    target[0].cmov(&destination, source == 0);

    let mut held = Block::default();
    for index in 0..STASH_SIZE + SLOTS_PER_POOLED_LANE {
      let take = (deepest_index[0] == index as i64) & (target[0] != -1);
      self.stash.swap_with_block(index, &mut held, take);
    }
    destination = target[0];

    let mut index = STASH_SIZE + SLOTS_PER_POOLED_LANE;
    for level in 1..self.h - 1 {
      let has_target = target[level] != -1;
      let place = (level as i64 == destination) & (!has_target);
      for _ in 0..SLOTS_PER_POOLED_LANE {
        let take = (deepest_index[level] == index as i64) & has_target;
        let put = (self.stash.position(index) == DUMMY_POS) & place;
        self.stash.swap_with_block(index, &mut held, take | put);
        index += 1;
      }
      destination.cmov(&target[level], has_target | place);
    }

    let place = self.h as i64 - 1 == destination;
    let mut written = false;
    for _ in 0..SLOTS_PER_POOLED_LANE {
      let put = (self.stash.position(index) == DUMMY_POS) & place & (!written);
      self.stash.swap_with_block(index, &mut held, put);
      written |= put;
      index += 1;
    }
    debug_assert!(held.is_empty());
  }

  #[allow(dead_code)]
  fn perform_eviction(&mut self, path: PositionType, pooled_lane: usize) {
    self.read_path_and_get_nodes(path, pooled_lane);
    self.evict_once_fast(path, pooled_lane);
    self.write_back_path(path, pooled_lane);
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  fn prepare_batch_eviction_plan(&self, path: PositionType) -> BatchEvictionPlan {
    unsafe {
      let none = _mm512_set1_epi64(-1);
      let zero = _mm512_setzero_si512();
      let one = _mm512_set1_epi64(1);
      let lane_indices = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
      let mut bucket_deepest = [none; 64];
      let mut source_slot_1 = [0; 64];
      let mut empty_slot_0 = [0; 64];
      let mut empty_slot_1 = [0; 64];

      for depth in 0..self.h {
        let bucket = &self.tree.tree[tree_index(depth, path)];
        let metadata = prepare_level_metadata(bucket, path);
        bucket_deepest[depth] = metadata.deepest;
        source_slot_1[depth] = metadata.source_slot_1;
        empty_slot_0[depth] = metadata.empty_slot_0;
        empty_slot_1[depth] = metadata.empty_slot_1;
      }

      let mut destination = none;
      let mut root_source = none;
      for chunk_index in 0..STASH_SIZE.div_ceil(8) {
        let positions = _mm512_load_si512(
          (&self.stash.positions[chunk_index] as *const PositionChunk).cast::<__m512i>(),
        );
        let depths: [i64; 8] = std::mem::transmute(legal_depths(positions, path));
        let position_values: [u64; 8] = std::mem::transmute(positions);

        for chunk_offset in 0..8 {
          let stash_index = chunk_index * 8 + chunk_offset;
          let valid = stash_index < STASH_SIZE;
          let depth = _mm512_set1_epi64(depths[chunk_offset]);
          let lane = _mm512_set1_epi64(
            (self.stash.key_value(stash_index).key & (POOLED_LANES as u64 - 1)) as i64,
          );
          let belongs = _mm512_cmpeq_epi64_mask(lane, lane_indices);
          let nonempty = valid & (position_values[chunk_offset] != DUMMY_POS);
          let eligible = 0u8.wrapping_sub(nonempty as u8);
          let deeper = _mm512_cmpgt_epi64_mask(depth, destination) & belongs & eligible;
          destination = _mm512_mask_mov_epi64(destination, deeper, depth);
          root_source =
            _mm512_mask_mov_epi64(root_source, deeper, _mm512_set1_epi64(stash_index as i64));
        }
      }

      let root_deeper = _mm512_cmpgt_epi64_mask(bucket_deepest[0], destination);
      let root_slot = _mm512_mask_mov_epi64(zero, source_slot_1[0], one);
      let root_index = _mm512_add_epi64(_mm512_set1_epi64(STASH_SIZE as i64), root_slot);
      root_source = _mm512_mask_mov_epi64(root_source, root_deeper, root_index);
      destination = _mm512_mask_mov_epi64(destination, root_deeper, bucket_deepest[0]);

      let mut source =
        _mm512_mask_mov_epi64(none, !_mm512_cmpeq_epi64_mask(destination, none), zero);
      let mut deepest = [none; 64];
      for depth in 1..self.h {
        let depth_vector = _mm512_set1_epi64(depth as i64);
        let reaches = _mm512_cmpgt_epi64_mask(destination, _mm512_set1_epi64(depth as i64 - 1));
        deepest[depth] = _mm512_mask_mov_epi64(none, reaches, source);
        let deeper = _mm512_cmpgt_epi64_mask(bucket_deepest[depth], destination);
        source = _mm512_mask_mov_epi64(source, deeper, depth_vector);
        destination = _mm512_mask_mov_epi64(destination, deeper, bucket_deepest[depth]);
      }

      source = none;
      destination = none;
      let mut target = [[-1i64; POOLED_LANES]; 64];
      for depth in (1..self.h).rev() {
        let depth_vector = _mm512_set1_epi64(depth as i64);
        let is_source = _mm512_cmpeq_epi64_mask(depth_vector, source);
        let target_here = _mm512_mask_mov_epi64(none, is_source, destination);
        target[depth] = std::mem::transmute(target_here);
        source = _mm512_mask_mov_epi64(source, is_source, none);
        destination = _mm512_mask_mov_epi64(destination, is_source, none);
        let destination_empty = _mm512_cmpeq_epi64_mask(destination, none);
        let target_exists = !_mm512_cmpeq_epi64_mask(target_here, none);
        let deepest_exists = !_mm512_cmpeq_epi64_mask(deepest[depth], none);
        let has_empty = empty_slot_0[depth] | empty_slot_1[depth];
        let change = ((destination_empty & has_empty) | target_exists) & deepest_exists;
        source = _mm512_mask_mov_epi64(source, change, deepest[depth]);
        destination = _mm512_mask_mov_epi64(destination, change, depth_vector);
      }
      let root_target = _mm512_cmpeq_epi64_mask(source, zero);
      target[0] = std::mem::transmute(_mm512_mask_mov_epi64(none, root_target, destination));

      BatchEvictionPlan {
        root_source: std::mem::transmute(root_source),
        target,
        source_slot_1,
        empty_slot_0,
        empty_slot_1,
      }
    }
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  fn move_batch_breadth_first(&mut self, path: PositionType, plan: &BatchEvictionPlan) {
    unsafe {
      // Eight independent padded logical blocks stay in eight zmm registers.
      // Only packing/unpacking at the split physical storage boundary is
      // needed; the held values themselves are never written to memory.
      let dummy = _mm512_set1_epi64(-1);
      let mut held_0 = dummy;
      let mut held_1 = dummy;
      let mut held_2 = dummy;
      let mut held_3 = dummy;
      let mut held_4 = dummy;
      let mut held_5 = dummy;
      let mut held_6 = dummy;
      let mut held_7 = dummy;

      macro_rules! route_stash_source {
        ($held:ident, $lane:literal, $stash_value:ident, $stash_index:ident) => {{
          let selected =
            (plan.target[0][$lane] != -1) & (plan.root_source[$lane] == $stash_index as i64);
          let mask = 0u8.wrapping_sub(selected as u8);
          let old_stash = $stash_value;
          $stash_value = _mm512_mask_mov_epi64(old_stash, mask, $held);
          $held = _mm512_mask_mov_epi64($held, mask, old_stash);
        }};
      }

      for stash_index in 0..STASH_SIZE {
        let mut stash_value = pack_block(self.stash.block(stash_index));
        route_stash_source!(held_0, 0, stash_value, stash_index);
        route_stash_source!(held_1, 1, stash_value, stash_index);
        route_stash_source!(held_2, 2, stash_value, stash_index);
        route_stash_source!(held_3, 3, stash_value, stash_index);
        route_stash_source!(held_4, 4, stash_value, stash_index);
        route_stash_source!(held_5, 5, stash_value, stash_index);
        route_stash_source!(held_6, 6, stash_value, stash_index);
        route_stash_source!(held_7, 7, stash_value, stash_index);
        self.stash.set_block(stash_index, unpack_block(stash_value));
      }

      macro_rules! move_root_lane {
        ($held:ident, $lane:literal, $bucket:ident) => {{
          let active = plan.target[0][$lane] != -1;
          let select_0 = active & (plan.root_source[$lane] == STASH_SIZE as i64);
          $held = swap_register_with_bucket($bucket, $lane, 0, $held, select_0);
          let select_1 = active & (plan.root_source[$lane] == (STASH_SIZE + 1) as i64);
          $held = swap_register_with_bucket($bucket, $lane, 1, $held, select_1);
        }};
      }

      let root_index = tree_index(0, path);
      let root = &mut self.tree.tree[root_index];
      move_root_lane!(held_0, 0, root);
      move_root_lane!(held_1, 1, root);
      move_root_lane!(held_2, 2, root);
      move_root_lane!(held_3, 3, root);
      move_root_lane!(held_4, 4, root);
      move_root_lane!(held_5, 5, root);
      move_root_lane!(held_6, 6, root);
      move_root_lane!(held_7, 7, root);

      let mut destination = plan.target[0];
      macro_rules! move_lane_at_depth {
        ($held:ident, $lane:literal, $bucket:ident, $depth:ident) => {{
          let target = plan.target[$depth][$lane];
          let has_target = target != -1;
          let place = (destination[$lane] == $depth as i64) & (!has_target);
          let source_slot_1 = ((plan.source_slot_1[$depth] >> $lane) & 1) != 0;
          let empty_0 = ((plan.empty_slot_0[$depth] >> $lane) & 1) != 0;
          let empty_1 = ((plan.empty_slot_1[$depth] >> $lane) & 1) != 0;
          let take_0 = has_target & (!source_slot_1);
          let put_0 = place & empty_0;
          $held = swap_register_with_bucket($bucket, $lane, 0, $held, take_0 | put_0);
          let take_1 = has_target & source_slot_1;
          let put_1 = place & empty_1 & (!put_0);
          $held = swap_register_with_bucket($bucket, $lane, 1, $held, take_1 | put_1);
          destination[$lane].cmov(&target, has_target | place);
        }};
      }

      for depth in 1..self.h {
        let bucket_index = tree_index(depth, path);
        let bucket = &mut self.tree.tree[bucket_index];
        move_lane_at_depth!(held_0, 0, bucket, depth);
        move_lane_at_depth!(held_1, 1, bucket, depth);
        move_lane_at_depth!(held_2, 2, bucket, depth);
        move_lane_at_depth!(held_3, 3, bucket, depth);
        move_lane_at_depth!(held_4, 4, bucket, depth);
        move_lane_at_depth!(held_5, 5, bucket, depth);
        move_lane_at_depth!(held_6, 6, bucket, depth);
        move_lane_at_depth!(held_7, 7, bucket, depth);
      }

      debug_assert_eq!(_mm512_cmpeq_epi64_mask(held_0, dummy), 0xff);
      debug_assert_eq!(_mm512_cmpeq_epi64_mask(held_1, dummy), 0xff);
      debug_assert_eq!(_mm512_cmpeq_epi64_mask(held_2, dummy), 0xff);
      debug_assert_eq!(_mm512_cmpeq_epi64_mask(held_3, dummy), 0xff);
      debug_assert_eq!(_mm512_cmpeq_epi64_mask(held_4, dummy), 0xff);
      debug_assert_eq!(_mm512_cmpeq_epi64_mask(held_5, dummy), 0xff);
      debug_assert_eq!(_mm512_cmpeq_epi64_mask(held_6, dummy), 0xff);
      debug_assert_eq!(_mm512_cmpeq_epi64_mask(held_7, dummy), 0xff);
    }
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  fn perform_deterministic_eviction_batch(&mut self) {
    let path = self.evict_counter;
    let plan = self.prepare_batch_eviction_plan(path);
    self.move_batch_breadth_first(path, &plan);
    self.evict_counter = (self.evict_counter + 1) % self.max_n as u64;
  }

  #[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  )))]
  fn perform_deterministic_eviction_batch(&mut self) {
    let path = self.evict_counter;
    for lane in 0..POOLED_LANES {
      self.perform_eviction(path, lane);
    }
    self.evict_counter = (self.evict_counter + 1) % self.max_n as u64;
  }

  #[doc(hidden)]
  pub fn benchmark_deterministic_eviction_batch(&mut self) {
    self.perform_deterministic_eviction_batch();
  }

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
    debug_assert_ne!(key, DUMMY_KEY);

    let pooled_lane = pooled_lane_for_key(key);
    self.read_path_and_get_nodes(pos, pooled_lane);
    let mut block = read_and_remove_element(&mut self.stash, key);
    let found = block.key() != DUMMY_KEY;
    let result = update_func(&mut block.key_value.data);
    block.pos = new_pos;
    block.key_value.key = key;
    let inserted = write_block_to_empty_slot(&mut self.stash, STASH_SIZE, &block);
    debug_assert!(inserted);
    self.evict_once_fast(pos, pooled_lane);
    self.write_back_path(pos, pooled_lane);

    self.eviction_credit += EVICTIONS_PER_OP_NUMERATOR;
    if self.eviction_credit >= EVICTION_CREDITS_PER_BATCH {
      self.perform_deterministic_eviction_batch();
      self.eviction_credit -= EVICTION_CREDITS_PER_BATCH;
    }
    (found, result)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn split_layout_is_exact() {
    assert_eq!(std::mem::size_of::<Block>(), 40);
    assert_eq!(std::mem::size_of::<KeyValue>(), 32);
    assert_eq!(std::mem::size_of::<Bucket>(), 640);
    assert_eq!(std::mem::offset_of!(Bucket, key_values), 128);
    assert_eq!(Bucket::position_index(3, 0), 3);
    assert_eq!(Bucket::position_index(3, 1), 11);
    assert_eq!(Bucket::key_value_index(3, 0), 6);
    assert_eq!(Bucket::key_value_index(3, 1), 7);
  }

  #[test]
  fn updates_round_trip_across_all_pooled_lanes() {
    let mut oram = BigKvOptimizedCircuitOram::new(128);
    let mut positions = [0u64; 64];
    for key in 0..64u64 {
      let new_pos = (key * 7 + 3) % oram.max_n as u64;
      let (found, ()) = oram.update(0, new_pos, key, |data| {
        data.fill(0);
        data[..8].copy_from_slice(&key.to_le_bytes());
      });
      assert!(!found);
      positions[key as usize] = new_pos;
    }
    for key in 0..64u64 {
      let new_pos = (key * 11 + 5) % oram.max_n as u64;
      let (found, stored) = oram.update(positions[key as usize], new_pos, key, |data| {
        u64::from_le_bytes(data[..8].try_into().unwrap())
      });
      assert!(found, "missing key {key}");
      assert_eq!(stored, key);
    }
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  #[test]
  fn direct_position_metadata_matches_scalar() {
    let mut bucket = Bucket::default();
    for lane in 0..POOLED_LANES {
      for slot in 0..SLOTS_PER_POOLED_LANE {
        if (lane + slot) % 3 != 0 {
          bucket.positions[Bucket::position_index(lane, slot)] = (lane * 17 + slot * 5) as u64;
        }
      }
    }
    let path = 37;
    let metadata = prepare_level_metadata(&bucket, path);
    let deepest: [i64; 8] = unsafe { std::mem::transmute(metadata.deepest) };
    for lane in 0..POOLED_LANES {
      let depths = [0, 1].map(|slot| {
        let pos = bucket.positions[Bucket::position_index(lane, slot)];
        if pos == DUMMY_POS {
          -1
        } else {
          common_suffix_length(pos, path) as i64
        }
      });
      assert_eq!(deepest[lane], depths[0].max(depths[1]));
      assert_eq!(((metadata.source_slot_1 >> lane) & 1) != 0, depths[1] > depths[0]);
    }
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512vpopcntdq"
  ))]
  #[test]
  fn breadth_first_batch_matches_scalar_lane_evictions() {
    let mut batched = BigKvOptimizedCircuitOram::new(256);
    let mut scalar = BigKvOptimizedCircuitOram::new(256);
    for key in 0..16u64 {
      let new_pos = (key * 13 + 7) % batched.max_n as u64;
      for oram in [&mut batched, &mut scalar] {
        oram.update(0, new_pos, key, |data| data.fill(key as u8));
        oram.eviction_credit = 0;
      }
    }

    for path in [0, 1, 7, 19] {
      let plan = batched.prepare_batch_eviction_plan(path);
      batched.move_batch_breadth_first(path, &plan);
      for lane in 0..POOLED_LANES {
        scalar.perform_eviction(path, lane);
      }
      for index in 0..S {
        assert_eq!(batched.stash.block(index), scalar.stash.block(index));
      }
      assert_eq!(batched.tree.tree, scalar.tree.tree);
    }
  }
}
