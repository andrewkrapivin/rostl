//! Fast metadata-only simulator for the current lane ORAM implementation.
//!
//! Each occupied slot stores only its key. The corresponding position is read
//! from `labels[key]`. This is equivalent to storing `(position, key)` because
//! the implementation always overwrites a block position with the fresh label
//! supplied by the caller, and payload bytes never affect placement.

const EMPTY_KEY: u32 = u32::MAX;
const NO_STASH_SLOT: usize = usize::MAX;

/// Outcome of one simulated lane ORAM update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimUpdate {
  /// Whether the key was already present.
  pub found: bool,
  /// Stash capacity required at the transient insertion point.
  pub insertion_demand: usize,
  /// Stash occupancy after root admission and lane eviction.
  pub stash_after: usize,
  /// Whether the configured simulator stash was too small.
  pub overflowed: bool,
}

#[derive(Clone, Copy, Debug)]
struct Level {
  offset: usize,
  path_mask: usize,
  path_shift: u32,
}

/// Control-flow-equivalent metadata simulator for [`crate::lane_oram::LaneORAM`].
#[derive(Debug)]
pub struct LaneOramSimulator {
  max_n: usize,
  height: usize,
  z: usize,
  b: usize,
  levels: Vec<Level>,
  tree: Vec<u32>,
  stash: Vec<u32>,
  stash_bits: Vec<u64>,
  stash_len: usize,
  key_to_stash: Vec<usize>,
  labels: Vec<u32>,
  initialized: Vec<bool>,
  path: Vec<u32>,
  masks: Vec<bool>,
  targets: Vec<u32>,
  held: Vec<u32>,
}

impl LaneOramSimulator {
  /// Creates an empty simulator with the requested public configuration.
  pub fn new(max_n: usize, z: usize, b: usize, stash_capacity: usize) -> Self {
    assert!(max_n > 0);
    assert!(z > 0);
    assert!(b >= 2 && b.is_power_of_two());
    assert!(stash_capacity > 0);

    let mut rounded_max_n = 1usize;
    let mut height = 1usize;
    while rounded_max_n < max_n {
      rounded_max_n *= b;
      height += 1;
    }

    let bits_per_digit = b.trailing_zeros();
    let mut levels = Vec::with_capacity(height);
    let mut node_count = 0usize;
    let mut level_width = 1usize;
    for depth in 0..height {
      levels.push(Level {
        offset: node_count,
        path_mask: level_width - 1,
        path_shift: ((height - 1 - depth) as u32) * bits_per_digit,
      });
      node_count += level_width;
      level_width *= b;
    }

    Self {
      max_n: rounded_max_n,
      height,
      z,
      b,
      levels,
      tree: vec![EMPTY_KEY; node_count * z],
      stash: vec![EMPTY_KEY; stash_capacity],
      stash_bits: vec![0; stash_capacity.div_ceil(64)],
      stash_len: 0,
      key_to_stash: vec![NO_STASH_SLOT; max_n],
      labels: vec![0; max_n],
      initialized: vec![false; max_n],
      path: vec![EMPTY_KEY; height * z],
      masks: vec![false; height * z],
      targets: vec![u32::MAX; z],
      held: vec![EMPTY_KEY; z],
    }
  }

  /// Rounded number of leaf labels used by this configuration.
  pub const fn max_n(&self) -> usize {
    self.max_n
  }

  /// Current post-eviction stash occupancy.
  pub const fn stash_len(&self) -> usize {
    self.stash_len
  }

  /// Ordered stash slots, with `u32::MAX` denoting an empty slot.
  pub fn stash_keys(&self) -> &[u32] {
    &self.stash
  }

  /// Current label of `key`.
  pub fn label(&self, key: usize) -> u32 {
    self.labels[key]
  }

  /// Copies keys on `path` into `out` in bucket-major order.
  pub fn read_path_keys(&self, path: u32, out: &mut [u32]) {
    assert!((path as usize) < self.max_n);
    assert_eq!(out.len(), self.height * self.z);
    for depth in 0..self.height {
      let node = self.node_index(depth, path);
      let source = &self.tree[node * self.z..(node + 1) * self.z];
      out[depth * self.z..(depth + 1) * self.z].copy_from_slice(source);
    }
  }

  /// Simulates one update using the caller-provided old and fresh labels.
  pub fn update(&mut self, old_pos: u32, new_pos: u32, key: u32) -> SimUpdate {
    self.update_with_background_paths(old_pos, new_pos, key, &[])
  }

  /// Simulates one update followed by public, deterministic background-path
  /// evictions using the same root-admission and lane-propagation transition.
  pub fn update_with_background_paths(
    &mut self,
    old_pos: u32,
    new_pos: u32,
    key: u32,
    background_paths: &[u32],
  ) -> SimUpdate {
    assert!((old_pos as usize) < self.max_n);
    assert!((new_pos as usize) < self.max_n);
    assert!((key as usize) < self.labels.len());
    if self.initialized[key as usize] {
      debug_assert_eq!(self.labels[key as usize], old_pos);
    }

    self.read_path(old_pos);
    let found = self.remove_key(key);
    self.labels[key as usize] = new_pos;
    self.initialized[key as usize] = true;

    let insertion_demand = self.stash_len + 1;
    let Some(slot) = self.first_empty_stash_slot() else {
      return SimUpdate { found, insertion_demand, stash_after: self.stash_len, overflowed: true };
    };
    self.put_in_stash(slot, key);

    self.move_from_stash_to_free_root_lanes();
    self.calculate_masks(old_pos);
    self.move_down_lanes();
    self.write_path(old_pos);

    for &path in background_paths {
      assert!((path as usize) < self.max_n);
      self.read_path(path);
      self.move_from_stash_to_free_root_lanes();
      self.calculate_masks(path);
      self.move_down_lanes();
      self.write_path(path);
    }

    SimUpdate { found, insertion_demand, stash_after: self.stash_len, overflowed: false }
  }

  fn node_index(&self, depth: usize, path: u32) -> usize {
    let level = self.levels[depth];
    level.offset + (((path as usize) >> level.path_shift) & level.path_mask)
  }

  fn read_path(&mut self, path: u32) {
    for depth in 0..self.height {
      let node = self.node_index(depth, path);
      let source_start = node * self.z;
      let target_start = depth * self.z;
      self.path[target_start..target_start + self.z]
        .copy_from_slice(&self.tree[source_start..source_start + self.z]);
    }
  }

  fn write_path(&mut self, path: u32) {
    for depth in 0..self.height {
      let node = self.node_index(depth, path);
      let source_start = depth * self.z;
      let target_start = node * self.z;
      self.tree[target_start..target_start + self.z]
        .copy_from_slice(&self.path[source_start..source_start + self.z]);
    }
  }

  fn remove_key(&mut self, key: u32) -> bool {
    let stash_slot = self.key_to_stash[key as usize];
    if stash_slot != NO_STASH_SLOT {
      self.take_from_stash(stash_slot);
      return true;
    }

    let mut found = false;
    for candidate in &mut self.path {
      if *candidate == key {
        *candidate = EMPTY_KEY;
        found = true;
      }
    }
    found
  }

  fn move_from_stash_to_free_root_lanes(&mut self) {
    for lane in 0..self.z {
      if self.path[lane] == EMPTY_KEY {
        let Some(slot) = self.first_occupied_stash_slot() else {
          break;
        };
        self.path[lane] = self.take_from_stash(slot);
      }
    }
  }

  fn calculate_masks(&mut self, path: u32) {
    let bits_per_digit = self.b.trailing_zeros();
    let position_bits = ((self.height - 1) as u32) * bits_per_digit;
    let unused_high_bits = u32::BITS - position_bits;
    self.targets.fill(u32::MAX);

    for level in (0..self.height).rev() {
      for lane in 0..self.z {
        let index = level * self.z + lane;
        let key = self.path[index];
        let selected = if key == EMPTY_KEY {
          true
        } else {
          let pos = self.labels[key as usize];
          let differing = (pos ^ path).wrapping_shl(unused_high_bits);
          differing.leading_zeros() >= self.targets[lane]
        };
        if selected {
          self.targets[lane] = (level as u32) * bits_per_digit;
        }
        self.masks[index] = selected;
      }
    }
  }

  fn move_down_lanes(&mut self) {
    self.held.fill(EMPTY_KEY);
    for level in 0..self.height {
      for lane in 0..self.z {
        let index = level * self.z + lane;
        if self.masks[index] {
          std::mem::swap(&mut self.held[lane], &mut self.path[index]);
        }
      }
    }
    debug_assert!(self.held.iter().all(|&key| key == EMPTY_KEY));
  }

  fn first_empty_stash_slot(&self) -> Option<usize> {
    for (word_index, &word) in self.stash_bits.iter().enumerate() {
      let mut available = !word;
      if word_index + 1 == self.stash_bits.len() && self.stash.len() % 64 != 0 {
        available &= (1u64 << (self.stash.len() % 64)) - 1;
      }
      if available != 0 {
        return Some(word_index * 64 + available.trailing_zeros() as usize);
      }
    }
    None
  }

  fn first_occupied_stash_slot(&self) -> Option<usize> {
    self.stash_bits.iter().enumerate().find_map(|(word_index, &word)| {
      (word != 0).then(|| word_index * 64 + word.trailing_zeros() as usize)
    })
  }

  fn put_in_stash(&mut self, slot: usize, key: u32) {
    debug_assert_eq!(self.stash[slot], EMPTY_KEY);
    self.stash[slot] = key;
    self.stash_bits[slot / 64] |= 1u64 << (slot % 64);
    self.key_to_stash[key as usize] = slot;
    self.stash_len += 1;
  }

  fn take_from_stash(&mut self, slot: usize) -> u32 {
    let key = std::mem::replace(&mut self.stash[slot], EMPTY_KEY);
    debug_assert_ne!(key, EMPTY_KEY);
    self.stash_bits[slot / 64] &= !(1u64 << (slot % 64));
    self.key_to_stash[key as usize] = NO_STASH_SLOT;
    self.stash_len -= 1;
    key
  }
}

#[cfg(all(test, target_arch = "x86_64", target_feature = "avx512f"))]
mod tests {
  use super::{LaneOramSimulator, EMPTY_KEY};
  use crate::lane_oram::{Block32, LaneORAM};

  fn os_random_position(n: usize) -> u32 {
    (getrandom::u32().expect("the operating system random source failed") as usize & (n - 1)) as u32
  }

  fn compare_configuration<const Z: usize, const B: usize>(n: usize) {
    const S: usize = 128;
    let mut real = LaneORAM::<Z, S, B>::new(n);
    let mut simulated = LaneOramSimulator::new(n, Z, B, S);
    let mut positions: Vec<u32> = (0..n).map(|_| os_random_position(n)).collect();

    for operation in 0..5_000usize {
      let key = if operation < n { operation } else { operation.wrapping_mul(17) & (n - 1) };
      let old_pos = positions[key];
      let new_pos = os_random_position(n);
      let (real_found, ()) = real.update(old_pos, new_pos, key as u32, |_| ());
      let simulated_result = simulated.update(old_pos, new_pos, key as u32);
      assert!(!simulated_result.overflowed);
      assert_eq!(real_found, simulated_result.found);
      positions[key] = new_pos;

      if operation % 97 == 0 {
        for (slot, block) in real.stash_and_path[..S].iter().enumerate() {
          let simulated_key = simulated.stash_keys()[slot];
          if block.is_empty() {
            assert_eq!(simulated_key, EMPTY_KEY);
          } else {
            assert_eq!(block.key, simulated_key);
            assert_eq!(block.pos, simulated.label(simulated_key as usize));
          }
        }

        let mut real_path = vec![Block32::default(); real.h * Z];
        let mut simulated_path = vec![EMPTY_KEY; real.h * Z];
        for path in 0..n as u32 {
          real.tree.read_path(path, &mut real_path);
          simulated.read_path_keys(path, &mut simulated_path);
          for (block, &simulated_key) in real_path.iter().zip(&simulated_path) {
            if block.is_empty() {
              assert_eq!(simulated_key, EMPTY_KEY);
            } else {
              assert_eq!(block.key, simulated_key);
              assert_eq!(block.pos, simulated.label(simulated_key as usize));
            }
          }
        }
      }
    }
  }

  #[test]
  fn simulator_matches_lane_oram_requested_configurations() {
    compare_configuration::<3, 2>(64);
    compare_configuration::<5, 4>(64);
    compare_configuration::<6, 4>(64);
    compare_configuration::<9, 8>(64);
    compare_configuration::<10, 8>(64);
  }

  #[test]
  fn background_paths_conserve_every_block() {
    const N: usize = 64;
    let mut simulated = LaneOramSimulator::new(N, 4, 2, N + 1);
    let mut positions = [0u32; N];

    for operation in 0..20_000usize {
      let key = operation & (N - 1);
      let old_pos = positions[key];
      let new_pos = os_random_position(N);
      let background =
        [(((operation / 2) & (N - 1)).reverse_bits() >> (usize::BITS - N.trailing_zeros())) as u32];
      let result = simulated.update_with_background_paths(
        old_pos,
        new_pos,
        key as u32,
        if operation & 1 == 0 { &[] } else { &background },
      );
      assert_eq!(result.found, operation >= N);
      assert!(!result.overflowed);
      positions[key] = new_pos;
      let blocks =
        simulated.stash_len + simulated.tree.iter().filter(|&&key| key != EMPTY_KEY).count();
      assert_eq!(blocks, N.min(operation + 1));
    }
  }

  #[test]
  fn simulator_predicts_real_debug_stash_failure() {
    const N: usize = 64;
    const S: usize = 20;
    let mut real = LaneORAM::<5, S, 4>::new(N);
    let mut simulated = LaneOramSimulator::new(N, 5, 4, S);
    let mut positions: [u32; N] = std::array::from_fn(|_| os_random_position(N));

    for operation in 0..1_000_000usize {
      let key = operation & (N - 1);
      let old_pos = positions[key];
      let new_pos = os_random_position(N);
      let simulated_result = simulated.update(old_pos, new_pos, key as u32);
      let real_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        real.update(old_pos, new_pos, key as u32, |_| ())
      }));

      if simulated_result.overflowed {
        assert!(real_result.is_err(), "simulator predicted an overflow at update {operation}");
        return;
      }

      let (real_found, ()) = real_result.expect("real implementation failed before simulator");
      assert_eq!(real_found, simulated_result.found);
      positions[key] = new_pos;
    }

    panic!("the selected trace did not overflow");
  }
}
