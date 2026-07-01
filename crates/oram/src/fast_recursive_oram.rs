//! Simple recursive ORAM built from fast counter ORAM position maps.
//!
//! This is intentionally a first, not-yet-correct construction. It uses a
//! `LinearORAM<u64>` as the recursion base, then derives every lower ORAM
//! position as `H(secret, counter, key)`. The recursive position-map levels are
//! `FastCircuitCounterORAM15`; the final data level is a normal `CircuitORAM<V>`.

use bytemuck::Pod;
use rand::{rng, RngCore};
use rostl_primitives::traits::Cmov;
use sha2::{Digest, Sha256};

use crate::{
  circuit_oram::CircuitORAM,
  fast_buckets_15::COUNTER_15_BLOCK_COUNTERS,
  fast_circuit_oram_15::FastCircuitCounterORAM15,
  linear_oram::LinearORAM,
  prelude::{PositionType, K},
};

#[cfg(not(test))]
const ROOT_COUNTER_LIMIT: usize = 1024;
#[cfg(test)]
const ROOT_COUNTER_LIMIT: usize = 16;

const SECRET_BYTES: usize = 20;
const RESET_ALPHA: usize = 14;
const RESET_INTERVAL: usize = RESET_ALPHA + 1;

#[derive(Debug, Clone)]
struct LevelResetState {
  old_secret: [u8; SECRET_BYTES],
  new_secret: [u8; SECRET_BYTES],
  cursor: usize,
  size: usize,
}

/// A simple recursive ORAM whose position maps are fast counter ORAMs.
#[derive(Debug)]
pub struct FastRecursiveORAM<V: Cmov + Pod + Default + Clone + std::fmt::Debug> {
  /// Logical data capacity requested by the caller.
  pub n: usize,
  /// Fast counter position-map levels, ordered top-to-bottom.
  pub counter_orams: Vec<FastCircuitCounterORAM15>,
  /// Final data ORAM.
  pub data_oram: CircuitORAM<V>,
  /// Oblivious root counters for the top fast counter ORAM's external position map.
  base_counters: LinearORAM<u64>,
  /// Per positioned ORAM reset state, including the final data ORAM.
  reset_states: Vec<LevelResetState>,
  op_counter: usize,
}

impl<V: Cmov + Pod + Default + Clone + std::fmt::Debug> FastRecursiveORAM<V> {
  /// Creates an empty simple fast recursive ORAM.
  pub fn new(n: usize) -> Self {
    debug_assert!(n > 1);
    debug_assert!(n <= u32::MAX as usize);

    let data_oram = CircuitORAM::<V>::new(n);
    let counter_sizes = counter_level_sizes(data_oram.max_n);
    let counter_orams =
      counter_sizes.iter().copied().map(FastCircuitCounterORAM15::new).collect::<Vec<_>>();

    let root_len = counter_orams.first().map_or(1, |oram| oram.max_blocks);
    debug_assert!(root_len <= ROOT_COUNTER_LIMIT.max(2));

    let mut rng = rng();
    let mut level_sizes = counter_orams.iter().map(|oram| oram.max_blocks).collect::<Vec<_>>();
    level_sizes.push(data_oram.max_n);
    let reset_states = level_sizes
      .into_iter()
      .map(|size| {
        let mut old_secret = [0u8; SECRET_BYTES];
        let mut new_secret = [0u8; SECRET_BYTES];
        rng.fill_bytes(&mut old_secret);
        rng.fill_bytes(&mut new_secret);
        LevelResetState { old_secret, new_secret, cursor: 0, size }
      })
      .collect();

    Self {
      n,
      counter_orams,
      data_oram,
      base_counters: LinearORAM::new(root_len),
      reset_states,
      op_counter: 0,
    }
  }

  /// Reads `key` from the final data ORAM.
  pub fn read(&mut self, key: K, ret: &mut V) -> bool {
    debug_assert!((key as usize) < self.n);
    let (pos, new_pos) = self.access_position(key as usize);
    let found = self.data_oram.read(pos, new_pos, key, ret);
    self.finish_top_level_access();
    found
  }

  /// Writes `val` to `key` if it already exists.
  pub fn write(&mut self, key: K, val: V) -> bool {
    debug_assert!((key as usize) < self.n);
    let (pos, new_pos) = self.access_position(key as usize);
    let found = self.data_oram.write(pos, new_pos, key, val);
    self.finish_top_level_access();
    found
  }

  /// Writes `val` to `key`, inserting it if absent.
  pub fn write_or_insert(&mut self, key: K, val: V) -> bool {
    debug_assert!((key as usize) < self.n);
    let (pos, new_pos) = self.access_position(key as usize);
    let found = self.data_oram.write_or_insert(pos, new_pos, key, val);
    self.finish_top_level_access();
    found
  }

  /// Updates `key` in the final data ORAM.
  pub fn update<T, F>(&mut self, key: K, f: F) -> (bool, T)
  where
    F: FnOnce(&mut V) -> T,
  {
    debug_assert!((key as usize) < self.n);
    let (pos, new_pos) = self.access_position(key as usize);
    let result = self.data_oram.update(pos, new_pos, key, f);
    self.finish_top_level_access();
    result
  }

  fn access_position(&mut self, data_key: usize) -> (PositionType, PositionType) {
    let levels = self.counter_orams.len();
    debug_assert!(levels > 0);

    let counter = self.access_counter_level(levels - 1, data_key, CounterOp::Increment);
    let secret = self.selected_secret(levels, data_key);
    let pos = hash_position(&secret, counter, data_key as u32, self.data_oram.max_n);
    let new_pos =
      hash_position(&secret, counter.wrapping_add(1), data_key as u32, self.data_oram.max_n);

    (pos, new_pos)
  }

  fn access_counter_level(&mut self, level: usize, key: usize, op: CounterOp) -> u64 {
    debug_assert!(level < self.counter_orams.len());

    let parent_key = self.counter_orams[level].position_map_key_for_key(key);
    let parent_counter = if level == 0 {
      self.base_counter_access(parent_key, CounterOp::Increment)
    } else {
      self.access_counter_level(level - 1, parent_key, CounterOp::Increment)
    };

    let secret = self.selected_secret(level, parent_key);
    let pos = hash_position(
      &secret,
      parent_counter,
      parent_key as u32,
      self.counter_orams[level].max_blocks,
    );
    let new_pos = hash_position(
      &secret,
      parent_counter.wrapping_add(1),
      parent_key as u32,
      self.counter_orams[level].max_blocks,
    );

    match op {
      CounterOp::Increment => self.counter_orams[level].read_key_and_incr(pos, new_pos, key),
      CounterOp::Reset => self.counter_orams[level].read_key_and_reset(pos, new_pos, key),
    }
  }

  fn base_counter_access(&mut self, key: usize, op: CounterOp) -> u64 {
    debug_assert!(key < self.base_counters.data.len());

    let mut old = 0u64;
    for (index, counter) in self.base_counters.data.iter_mut().enumerate() {
      let matched = index == key;
      old.cmov(counter, matched);

      let incremented = counter.wrapping_add(1);
      let reset = 0u64;
      let mut new_value = incremented;
      new_value.cmov(&reset, matches!(op, CounterOp::Reset));
      counter.cmov(&new_value, matched);
    }
    old
  }

  fn finish_top_level_access(&mut self) {
    self.op_counter = self.op_counter.wrapping_add(1);
    if self.op_counter % RESET_INTERVAL == 0 {
      self.perform_reset_pass();
    }
  }

  fn perform_reset_pass(&mut self) {
    for level in 0..self.reset_states.len() {
      self.reset_one_level(level);
    }
  }

  fn reset_one_level(&mut self, level: usize) {
    let key = self.reset_states[level].cursor;
    let counter = if level == 0 {
      self.base_counter_access(key, CounterOp::Reset)
    } else {
      self.access_counter_level(level - 1, key, CounterOp::Reset)
    };

    let old_secret = self.reset_states[level].old_secret;
    let new_secret = self.reset_states[level].new_secret;
    let max_positions = self.level_position_count(level);
    let pos = hash_position(&old_secret, counter, key as u32, max_positions);
    let new_pos = hash_position(&new_secret, 0, key as u32, max_positions);

    if level < self.counter_orams.len() {
      self.counter_orams[level].remap_position_map_key(pos, new_pos, key);
    } else {
      let mut value = V::default();
      self.data_oram.read(pos, new_pos, key as K, &mut value);
    }

    self.advance_reset_cursor(level);
  }

  fn advance_reset_cursor(&mut self, level: usize) {
    let state = &mut self.reset_states[level];
    state.cursor += 1;
    if state.cursor == state.size {
      state.cursor = 0;
      state.old_secret = state.new_secret;
      rng().fill_bytes(&mut state.new_secret);
    }
  }

  fn selected_secret(&self, level: usize, key: usize) -> [u8; SECRET_BYTES] {
    let state = &self.reset_states[level];
    let use_new = key < state.cursor;
    select_secret(&state.old_secret, &state.new_secret, use_new)
  }

  fn level_position_count(&self, level: usize) -> usize {
    if level < self.counter_orams.len() {
      self.counter_orams[level].max_blocks
    } else {
      self.data_oram.max_n
    }
  }
}

#[derive(Clone, Copy)]
enum CounterOp {
  Increment,
  Reset,
}

fn counter_level_sizes(data_positions: usize) -> Vec<usize> {
  let mut sizes = vec![data_positions];
  let mut child_position_map_entries = counter_blocks_for_counter_space(data_positions);

  while child_position_map_entries > ROOT_COUNTER_LIMIT.max(2) {
    sizes.push(child_position_map_entries);
    child_position_map_entries = counter_blocks_for_counter_space(child_position_map_entries);
  }

  sizes.reverse();
  sizes
}

fn counter_blocks_for_counter_space(n: usize) -> usize {
  n.div_ceil(COUNTER_15_BLOCK_COUNTERS).next_power_of_two().max(2)
}

fn hash_position(
  secret: &[u8; SECRET_BYTES],
  counter: u64,
  key: u32,
  max_positions: usize,
) -> PositionType {
  debug_assert!(max_positions.is_power_of_two());
  debug_assert!(max_positions <= u32::MAX as usize);

  let mut input = [0u8; 32];
  input[..SECRET_BYTES].copy_from_slice(secret);
  input[SECRET_BYTES..SECRET_BYTES + 8].copy_from_slice(&counter.to_le_bytes());
  input[SECRET_BYTES + 8..SECRET_BYTES + 12].copy_from_slice(&key.to_le_bytes());

  let hash = Sha256::digest(input);
  let mut low = [0u8; 8];
  low.copy_from_slice(&hash[..8]);
  (u64::from_le_bytes(low) as usize & (max_positions - 1)) as PositionType
}

#[inline(always)]
fn select_secret(
  old_secret: &[u8; SECRET_BYTES],
  new_secret: &[u8; SECRET_BYTES],
  use_new: bool,
) -> [u8; SECRET_BYTES] {
  let mut selected = *old_secret;
  for i in 0..SECRET_BYTES {
    selected[i].cmov(&new_secret[i], use_new);
  }
  selected
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn counter_level_sizes_are_top_to_bottom() {
    let sizes = counter_level_sizes(1 << 16);
    assert!(!sizes.is_empty());
    for window in sizes.windows(2) {
      assert!(window[0] <= counter_blocks_for_counter_space(window[1]));
    }
  }

  #[test]
  fn write_or_insert_then_read_round_trips() {
    let mut oram = FastRecursiveORAM::<u64>::new(1024);
    assert!(!oram.write_or_insert(7, 11));

    let mut value = 0;
    assert!(oram.read(7, &mut value));
    assert_eq!(value, 11);

    assert!(oram.write(7, 19));
    assert!(oram.read(7, &mut value));
    assert_eq!(value, 19);
  }

  #[test]
  fn update_returns_old_value() {
    let mut oram = FastRecursiveORAM::<u64>::new(2048);
    assert!(!oram.write_or_insert(13, 5));

    let (found, old) = oram.update(13, |value| {
      let old = *value;
      *value = 8;
      old
    });
    assert!(found);
    assert_eq!(old, 5);

    let mut value = 0;
    assert!(oram.read(13, &mut value));
    assert_eq!(value, 8);
  }

  #[test]
  fn reads_survive_reset_passes() {
    let mut oram = FastRecursiveORAM::<u64>::new(1024);
    for key in 0..96 {
      assert!(!oram.write_or_insert(key, (key as u64) * 3 + 1));
    }

    for round in 0..64 {
      let key = (round * 17 + 9) % 96;
      let mut value = 0;
      assert!(oram.read(key as K, &mut value));
      assert_eq!(value, (key as u64) * 3 + 1);
    }
  }
}
