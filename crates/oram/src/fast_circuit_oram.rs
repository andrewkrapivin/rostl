//! Fast Circuit ORAM specialized for packed counters.
//!
//! The address space has `max_n` counters. Counter address `key` is interpreted
//! as `(prefix, suffix)`, where `suffix = key % 32` selects one counter inside a
//! `Counter_Block` and `prefix = key / 32` selects the logical block. To avoid
//! concentrating all large counters in one logical block, the ORAM stores the
//! block under `target_key = prefix ^ values[suffix]`, where `values` is a
//! fixed array of 32 random block-domain values sampled at construction time.
//! The suffix mask and counter slot are selected by scanning so that suffix does
//! not select a cache line or word directly.
//!
//! Public `read` and `read_and_incr` take the current and next position for the
//! shuffled block. The position map is caller-owned, as in `CircuitORAM`.
//! `position_map_key` exposes the shuffled block id the caller should use for
//! that external map.

#![allow(clippy::needless_bitwise_bool)]

use rand::{rng, rngs::ThreadRng, RngCore};
use rostl_primitives::traits::Cmov;

use crate::{
  circuit_oram::{S, Z},
  fast_buckets::{
    Cacheline_Counter_Bucket, Counter_Block, CACHELINE_COUNTER_BLOCK_COUNTERS,
    CACHELINE_COUNTER_BUCKET_BLOCKS,
  },
  heap_tree::HeapTree,
  prelude::PositionType,
};

const EVICTIONS_PER_OP: usize = 2;

/// Fast Circuit ORAM for an array of packed counters.
#[derive(Debug)]
pub struct FastCircuitCounterORAM {
  /// Logical counter capacity requested by the caller.
  pub max_n: usize,
  /// Number of logical `Counter_Block`s, rounded to a power of two.
  pub max_blocks: usize,
  /// Height of the underlying ORAM tree.
  pub h: usize,
  /// Stash plus path buffer. First `S` entries are the persistent stash.
  pub stash: Vec<Counter_Block>,
  /// ORAM tree, with two counter blocks packed into each tree bucket.
  pub tree: HeapTree<Cacheline_Counter_Bucket>,
  /// Scratch buffer for staging a packed path before splitting or writing it.
  pub path_buckets: Vec<Cacheline_Counter_Bucket>,
  /// Per-suffix xor mask for mapping `(prefix, suffix)` to a target block key.
  values: [PositionType; CACHELINE_COUNTER_BLOCK_COUNTERS],
  /// Deterministic eviction counter.
  pub evict_counter: PositionType,
}

impl FastCircuitCounterORAM {
  /// Creates a new counter ORAM with capacity for `max_n` counters.
  pub fn new(max_n: usize) -> Self {
    debug_assert!(max_n > 0);
    debug_assert!(max_n <= (u32::MAX as usize) * CACHELINE_COUNTER_BLOCK_COUNTERS);

    let requested_blocks = max_n.div_ceil(CACHELINE_COUNTER_BLOCK_COUNTERS);
    let max_blocks = requested_blocks.next_power_of_two().max(2);
    debug_assert!(max_blocks <= u32::MAX as usize);

    let h = max_blocks.ilog2() as usize + 1;
    let tree = HeapTree::new(h);
    let stash = vec![Counter_Block::default(); S + h * Z];
    let path_buckets = vec![Cacheline_Counter_Bucket::default(); h];

    let mut rng = rng();
    let mut values = [0; CACHELINE_COUNTER_BLOCK_COUNTERS];
    for mask in &mut values {
      *mask = random_position(&mut rng, max_blocks);
    }

    Self { max_n, max_blocks, h, stash, tree, path_buckets, values, evict_counter: 0 }
  }

  /// Returns the current value of counter `(prefix, suffix)`.
  #[inline]
  pub fn read(
    &mut self,
    pos: PositionType,
    new_pos: PositionType,
    prefix: usize,
    suffix: usize,
  ) -> u64 {
    self.access(pos, new_pos, prefix, suffix, false)
  }

  /// Returns the current value of counter `(prefix, suffix)`, then increments it.
  #[inline]
  pub fn read_and_incr(
    &mut self,
    pos: PositionType,
    new_pos: PositionType,
    prefix: usize,
    suffix: usize,
  ) -> u64 {
    self.access(pos, new_pos, prefix, suffix, true)
  }

  /// Returns the current value of a flat counter address.
  #[inline]
  pub fn read_key(&mut self, pos: PositionType, new_pos: PositionType, key: usize) -> u64 {
    let (prefix, suffix) = self.address_parts(key);
    self.read(pos, new_pos, prefix, suffix)
  }

  /// Returns the current value of a flat counter address, then increments it.
  #[inline]
  pub fn read_key_and_incr(&mut self, pos: PositionType, new_pos: PositionType, key: usize) -> u64 {
    let (prefix, suffix) = self.address_parts(key);
    self.read_and_incr(pos, new_pos, prefix, suffix)
  }

  /// Returns the external position-map index for counter `(prefix, suffix)`.
  #[inline]
  pub fn position_map_key(&self, prefix: usize, suffix: usize) -> usize {
    debug_assert!(suffix < CACHELINE_COUNTER_BLOCK_COUNTERS);
    self.shuffled_key(prefix, suffix)
  }

  /// Returns the external position-map index for a flat counter address.
  #[inline]
  pub fn position_map_key_for_key(&self, key: usize) -> usize {
    let (prefix, suffix) = self.address_parts(key);
    self.position_map_key(prefix, suffix)
  }

  #[inline]
  fn address_parts(&self, key: usize) -> (usize, usize) {
    debug_assert!(key < self.max_n);
    let prefix = key / CACHELINE_COUNTER_BLOCK_COUNTERS;
    let suffix = key & (CACHELINE_COUNTER_BLOCK_COUNTERS - 1);
    (prefix, suffix)
  }

  fn access(
    &mut self,
    pos: PositionType,
    new_pos: PositionType,
    prefix: usize,
    suffix: usize,
    increment: bool,
  ) -> u64 {
    debug_assert!(suffix < CACHELINE_COUNTER_BLOCK_COUNTERS);
    debug_assert!(prefix * CACHELINE_COUNTER_BLOCK_COUNTERS + suffix < self.max_n);
    debug_assert!((pos as usize) < self.max_blocks);
    debug_assert!((new_pos as usize) < self.max_blocks);

    let target_key = self.shuffled_key(prefix, suffix);
    debug_assert!(target_key < self.max_blocks);

    self.read_path_and_get_nodes(pos);

    let mut block = Counter_Block::default();
    let found = read_and_remove_block(&mut self.stash, target_key as u32, &mut block);
    block.set_key_pos(target_key as u32, new_pos);

    let value = block.access_counter_oblivious(suffix, increment);

    let written = write_block_to_empty_slot(&mut self.stash[..S], &block);
    debug_assert!(written);
    debug_assert!(found | !block.is_empty());

    self.evict_once_fast(pos);
    self.write_back_path(pos);
    self.perform_deterministic_evictions();

    value
  }

  #[inline]
  fn shuffled_key(&self, prefix: usize, suffix: usize) -> usize {
    let mut mask = 0;
    for (index, value) in self.values.iter().enumerate() {
      mask.cmov(value, index == suffix);
    }
    prefix ^ mask as usize
  }

  /// Reads a path to the end of the stash buffer.
  pub fn read_path_and_get_nodes(&mut self, pos: PositionType) {
    debug_assert!((pos as usize) < self.max_blocks);
    self.read_counter_path(pos);
  }

  /// Writes the path buffer back to the tree.
  pub fn write_back_path(&mut self, pos: PositionType) {
    debug_assert!((pos as usize) < self.max_blocks);
    self.write_counter_path(pos);
  }

  /// Reads all packed buckets on a path first, then expands them into blocks.
  fn read_counter_path(&mut self, path: PositionType) {
    debug_assert!((path as usize) < (1 << self.h));
    debug_assert_eq!(self.path_buckets.len(), self.h);

    for i in 0..self.h {
      let index = self.tree.get_index(i, path);
      self.path_buckets[i] = self.tree.tree[index];
    }

    let out = &mut self.stash[S..S + self.h * Z];
    for i in 0..self.h {
      let blocks = self.path_buckets[i].split_into_blocks();
      out[i * Z..(i + 1) * Z].copy_from_slice(&blocks);
    }
  }

  /// Merges all path blocks first, then writes packed buckets back to the tree.
  fn write_counter_path(&mut self, path: PositionType) {
    debug_assert!((path as usize) < (1 << self.h));
    debug_assert_eq!(self.path_buckets.len(), self.h);

    let in_ = &self.stash[S..S + self.h * Z];
    for i in 0..self.h {
      self.path_buckets[i] = Cacheline_Counter_Bucket::merge_blocks([in_[i * Z], in_[i * Z + 1]]);
    }

    for i in 0..self.h {
      let index = self.tree.get_index(i, path);
      self.tree.tree[index] = self.path_buckets[i];
    }
  }

  /// Alg. 4 - EvictOnceFast(path) specialized to `Counter_Block`.
  pub fn evict_once_fast(&mut self, pos: PositionType) {
    let mut deepest: [i32; 64] = [-1; 64];
    let mut deepest_idx: [i32; 64] = [0; 64];
    let mut target: [i32; 64] = [-1; 64];
    let mut has_empty: [bool; 64] = [false; 64];

    let mut src = -1;
    let mut dst: i32 = -1;

    for idx in 0..S + Z {
      let deepest_level = common_suffix_length(self.stash[idx].pos(), pos) as i32;
      let deeper_flag = (!self.stash[idx].is_empty()) & (deepest_level > dst);
      dst.cmov(&deepest_level, deeper_flag);
      deepest_idx[0].cmov(&(idx as i32), deeper_flag);
    }
    src.cmov(&0, dst != -1);

    let mut idx = S + Z;
    for i in 1..self.h {
      deepest[i].cmov(&src, dst >= i as i32);
      let mut bucket_deepest_level: i32 = -1;
      for _ in 0..Z {
        let deepest_level = common_suffix_length(self.stash[idx].pos(), pos) as i32;
        let is_empty = self.stash[idx].is_empty();
        has_empty[i].cmov(&true, is_empty);

        let deeper_flag = (!is_empty) & (deepest_level > bucket_deepest_level);
        bucket_deepest_level.cmov(&deepest_level, deeper_flag);
        deepest_idx[i].cmov(&(idx as i32), deeper_flag);

        idx += 1;
      }

      let deeper_flag = bucket_deepest_level > dst;
      src.cmov(&(i as i32), deeper_flag);
      dst.cmov(&bucket_deepest_level, deeper_flag);
    }

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

    let mut hold = Counter_Block::default();
    for idx in 0..S + Z {
      let is_deepest = deepest_idx[0] == idx as i32;
      let read_and_remove_flag = is_deepest & (target[0] != -1);
      hold.cmov(&self.stash[idx], read_and_remove_flag);
      self.stash[idx].cmov_empty(read_and_remove_flag);
    }
    dst = target[0];

    let mut idx = S + Z;
    for i in 1..(self.h - 1) {
      let has_target_flag = target[i] != -1;
      let place_dummy_flag = (i as i32 == dst) & (!has_target_flag);
      for _ in 0..Z {
        let is_deepest = deepest_idx[i] == idx as i32;
        let read_and_remove_flag = is_deepest & has_target_flag;
        let write_flag = self.stash[idx].is_empty() & place_dummy_flag;
        let swap_flag = read_and_remove_flag | write_flag;
        hold.cxchg(&mut self.stash[idx], swap_flag);
        idx += 1;
      }

      dst.cmov(&target[i], has_target_flag | place_dummy_flag);
    }

    let place_dummy_flag = ((self.h - 1) as i32) == dst;
    let mut written = false;
    for _ in 0..Z {
      let write_flag = self.stash[idx].is_empty() & place_dummy_flag & (!written);
      written |= write_flag;
      self.stash[idx].cmov(&hold, write_flag);
      idx += 1;
    }
  }

  fn perform_eviction(&mut self, pos: PositionType) {
    debug_assert!((pos as usize) < self.max_blocks);
    self.read_path_and_get_nodes(pos);
    self.evict_once_fast(pos);
    self.write_back_path(pos);
  }

  fn perform_deterministic_evictions(&mut self) {
    for _ in 0..EVICTIONS_PER_OP {
      let evict_pos = self.evict_counter;
      self.perform_eviction(evict_pos);
      self.evict_counter = (self.evict_counter + 1) % (self.max_blocks as PositionType);
    }

    let mut ok = false;
    for elem in &self.stash[..S] {
      ok.cmov(&true, elem.is_empty());
    }
    debug_assert!(ok);
  }
}

impl HeapTree<Cacheline_Counter_Bucket> {
  /// Reads all counter blocks in a path into `out`.
  #[inline]
  pub fn read_counter_path(&mut self, path: PositionType, out: &mut [Counter_Block]) {
    debug_assert!((path as usize) < (1 << self.height));
    debug_assert!(out.len() == self.height * Z);
    debug_assert_eq!(Z, CACHELINE_COUNTER_BUCKET_BLOCKS);

    for i in 0..self.height {
      let index = self.get_index(i, path);
      let blocks = self.tree[index].split_into_blocks();
      out[i * Z..(i + 1) * Z].copy_from_slice(&blocks);
    }
  }

  /// Writes all counter blocks in a path from `in_`.
  #[inline]
  pub fn write_counter_path(&mut self, path: PositionType, in_: &[Counter_Block]) {
    debug_assert!((path as usize) < (1 << self.height));
    debug_assert!(in_.len() == self.height * Z);
    debug_assert_eq!(Z, CACHELINE_COUNTER_BUCKET_BLOCKS);

    for i in 0..self.height {
      let index = self.get_index(i, path);
      self.tree[index] = Cacheline_Counter_Bucket::merge_blocks([in_[i * Z], in_[i * Z + 1]]);
    }
  }
}

#[inline]
fn read_and_remove_block(arr: &mut [Counter_Block], key: u32, ret: &mut Counter_Block) -> bool {
  let mut rv = false;

  for item in arr {
    let matched = (!item.is_empty()) & (item.key() == key);
    debug_assert!((!matched) | (!rv));

    ret.cmov(item, matched);
    item.cmov_empty(matched);
    rv.cmov(&true, matched);
  }

  rv
}

#[inline]
fn write_block_to_empty_slot(arr: &mut [Counter_Block], val: &Counter_Block) -> bool {
  let mut rv = false;

  for item in arr {
    let matched = item.is_empty() & (!rv);
    debug_assert!((!matched) | (!rv));

    item.cmov(val, matched);
    rv.cmov(&true, matched);
  }

  rv
}

#[inline]
fn random_position(rng: &mut ThreadRng, max_blocks: usize) -> PositionType {
  debug_assert!(max_blocks.is_power_of_two());
  (rng.next_u32() & (max_blocks as u32 - 1)) as PositionType
}

#[inline]
const fn common_suffix_length(a: PositionType, b: PositionType) -> u32 {
  let w = a ^ b;
  w.trailing_zeros()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn positions(oram: &FastCircuitCounterORAM) -> Vec<PositionType> {
    let mut rng = rng();
    (0..oram.max_blocks).map(|_| random_position(&mut rng, oram.max_blocks)).collect()
  }

  fn next_pos(_pos: PositionType, max_blocks: usize) -> PositionType {
    random_position(&mut rng(), max_blocks)
  }

  fn read(
    oram: &mut FastCircuitCounterORAM,
    positions: &mut [PositionType],
    prefix: usize,
    suffix: usize,
  ) -> u64 {
    let map_key = oram.position_map_key(prefix, suffix);
    let pos = positions[map_key];
    let new_pos = next_pos(pos, oram.max_blocks);
    let value = oram.read(pos, new_pos, prefix, suffix);
    positions[map_key] = new_pos;
    value
  }

  fn read_and_incr(
    oram: &mut FastCircuitCounterORAM,
    positions: &mut [PositionType],
    prefix: usize,
    suffix: usize,
  ) -> u64 {
    let map_key = oram.position_map_key(prefix, suffix);
    let pos = positions[map_key];
    let new_pos = next_pos(pos, oram.max_blocks);
    let value = oram.read_and_incr(pos, new_pos, prefix, suffix);
    positions[map_key] = new_pos;
    value
  }

  fn read_key(
    oram: &mut FastCircuitCounterORAM,
    positions: &mut [PositionType],
    key: usize,
  ) -> u64 {
    let map_key = oram.position_map_key_for_key(key);
    let pos = positions[map_key];
    let new_pos = next_pos(pos, oram.max_blocks);
    let value = oram.read_key(pos, new_pos, key);
    positions[map_key] = new_pos;
    value
  }

  fn read_key_and_incr(
    oram: &mut FastCircuitCounterORAM,
    positions: &mut [PositionType],
    key: usize,
  ) -> u64 {
    let map_key = oram.position_map_key_for_key(key);
    let pos = positions[map_key];
    let new_pos = next_pos(pos, oram.max_blocks);
    let value = oram.read_key_and_incr(pos, new_pos, key);
    positions[map_key] = new_pos;
    value
  }

  #[test]
  fn reads_start_at_zero() {
    let mut oram = FastCircuitCounterORAM::new(1024);
    let mut positions = positions(&oram);

    assert_eq!(read(&mut oram, &mut positions, 0, 0), 0);
    assert_eq!(read(&mut oram, &mut positions, 0, 31), 0);
    assert_eq!(read(&mut oram, &mut positions, 1, 0), 0);
  }

  #[test]
  fn supports_less_than_one_counter_block() {
    let mut oram = FastCircuitCounterORAM::new(8);
    let mut positions = positions(&oram);

    assert_eq!(read_and_incr(&mut oram, &mut positions, 0, 7), 0);
    assert_eq!(read(&mut oram, &mut positions, 0, 7), 1);
  }

  #[test]
  fn read_and_incr_returns_old_value() {
    let mut oram = FastCircuitCounterORAM::new(1024);
    let mut positions = positions(&oram);

    assert_eq!(read_and_incr(&mut oram, &mut positions, 0, 5), 0);
    assert_eq!(read(&mut oram, &mut positions, 0, 5), 1);
    assert_eq!(read_and_incr(&mut oram, &mut positions, 0, 5), 1);
    assert_eq!(read(&mut oram, &mut positions, 0, 5), 2);
  }

  #[test]
  fn suffixes_share_prefix_but_not_counter_slot() {
    let mut oram = FastCircuitCounterORAM::new(1024);
    let mut positions = positions(&oram);

    assert_eq!(read_and_incr(&mut oram, &mut positions, 2, 0), 0);
    assert_eq!(read_and_incr(&mut oram, &mut positions, 2, 1), 0);
    assert_eq!(read_and_incr(&mut oram, &mut positions, 2, 0), 1);
    assert_eq!(read(&mut oram, &mut positions, 2, 1), 1);
    assert_eq!(read(&mut oram, &mut positions, 2, 2), 0);
  }

  #[test]
  fn counters_match_reference_after_mixed_ops() {
    const N: usize = 2048;
    let mut oram = FastCircuitCounterORAM::new(N);
    let mut positions = positions(&oram);
    let mut reference = [0u64; N];

    for step in 0..256 {
      let key = (step * 37 + 11) % N;
      let old = read_key_and_incr(&mut oram, &mut positions, key);
      assert_eq!(old, reference[key]);
      reference[key] += 1;

      let key_to_read = (step * 19 + 7) % N;
      assert_eq!(read_key(&mut oram, &mut positions, key_to_read), reference[key_to_read]);
    }
  }
}
