//! A mock lane ORAM used to prototype tree/queue eviction policies.
//!
//! This module is intentionally not an oblivious implementation. It is a small
//! simulation surface for experimenting with a perfect B-ary tree whose leaf
//! level is `levels`, whose bottom level has `branching_factor.pow(levels)`
//! nodes, and whose nodes each contain `slots_per_node` slots. Keys are in
//! `0..keys_per_leaf * branching_factor.pow(levels)`, where
//! `keys_per_leaf < slots_per_node`. A queue temporarily holds blocks removed
//! from the tree.

use std::collections::{BTreeMap, VecDeque};

use rand::{rng, Rng};

use crate::prelude::K;

/// A mock block stored in the test lane ORAM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestLaneBlock {
  /// Logical key used by the simulation to find a block.
  pub key: K,
  /// Leaf index assigned to this block.
  pub pos: usize,
}

impl TestLaneBlock {
  /// Creates a new mock block.
  pub const fn new(key: K, pos: usize) -> Self {
    Self { key, pos }
  }
}

/// One slot inside a tree node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Slot<T> {
  /// The slot does not currently hold a block.
  Empty,
  /// The slot currently holds a block.
  Full(T),
}

impl<T> Default for Slot<T> {
  fn default() -> Self {
    Self::Empty
  }
}

impl<T> Slot<T> {
  /// Returns true when the slot is empty.
  pub const fn is_empty(&self) -> bool {
    matches!(self, Self::Empty)
  }

  /// Returns true when the slot holds a block.
  pub const fn is_full(&self) -> bool {
    matches!(self, Self::Full(_))
  }

  /// Returns the item in this slot, if it is full.
  pub const fn as_ref(&self) -> Option<&T> {
    match self {
      Self::Empty => None,
      Self::Full(item) => Some(item),
    }
  }
}

/// A node in the mock B-ary tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node<T> {
  slots: Vec<Slot<T>>,
}

impl<T> Node<T> {
  /// Creates an empty node with `slot_count` slots.
  pub fn new(slot_count: usize) -> Self {
    let slots = std::iter::repeat_with(Slot::default).take(slot_count).collect();
    Self { slots }
  }

  /// Returns all slots in this node.
  pub fn slots(&self) -> &[Slot<T>] {
    &self.slots
  }

  /// Returns all slots in this node mutably.
  pub fn slots_mut(&mut self) -> &mut [Slot<T>] {
    &mut self.slots
  }

  /// Returns the index of the first empty slot in this node.
  pub fn first_empty_slot_index(&self) -> Option<usize> {
    self.slots.iter().position(Slot::is_empty)
  }

  /// Returns true when every slot in this node is full.
  pub fn is_full(&self) -> bool {
    self.first_empty_slot_index().is_none()
  }

  /// Removes and returns the first full slot in this node.
  pub fn take_first_full(&mut self) -> Option<(usize, T)> {
    for (slot_index, slot) in self.slots.iter_mut().enumerate() {
      if slot.is_full() {
        let Slot::Full(item) = std::mem::take(slot) else {
          unreachable!("slot was checked as full")
        };
        return Some((slot_index, item));
      }
    }

    None
  }

  /// Removes and returns the first slot whose item matches `predicate`.
  pub fn take_first_matching<F>(&mut self, mut predicate: F) -> Option<(usize, T)>
  where
    F: FnMut(&T) -> bool,
  {
    for (slot_index, slot) in self.slots.iter_mut().enumerate() {
      if let Slot::Full(item) = slot {
        if predicate(item) {
          let Slot::Full(item) = std::mem::take(slot) else {
            unreachable!("slot was checked as full")
          };
          return Some((slot_index, item));
        }
      }
    }

    None
  }

  /// Counts full slots in this node.
  pub fn occupied_slot_count(&self) -> usize {
    self.slots.iter().filter(|slot| slot.is_full()).count()
  }
}

/// A concrete slot location in the level-order tree layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotLocation {
  /// Index of the node in level-order storage.
  pub node_index: usize,
  /// Index of the slot inside the node.
  pub slot_index: usize,
}

/// Summary of one mock insert operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertResult {
  /// Previous leaf position read from the position map.
  pub old_pos: usize,
  /// Newly assigned leaf position for this key.
  pub new_pos: usize,
  /// Slot removed from the old path, if the key was already present there.
  pub removed_from: Option<SlotLocation>,
  /// Queue length after the insert and path eviction complete.
  pub queue_len: usize,
}

/// A perfect B-ary tree with a fixed number of slots per node.
///
/// The root is level 0 and `levels` is the leaf level, so the bottom level has
/// `branching_factor.pow(levels)` nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerfectBaryTree<T> {
  slots: Vec<Slot<T>>,
  levels: usize,
  branching_factor: usize,
  slots_per_node: usize,
  level_offsets: Vec<usize>,
  level_widths: Vec<usize>,
}

impl<T> PerfectBaryTree<T> {
  /// Creates an empty perfect B-ary tree.
  pub fn new(levels: usize, branching_factor: usize, slots_per_node: usize) -> Self {
    assert!(branching_factor > 0, "branching_factor must be non-zero");
    assert!(slots_per_node > 0, "slots_per_node must be non-zero");

    let (level_offsets, level_widths, node_count) = checked_tree_shape(levels, branching_factor);
    let slots = std::iter::repeat_with(Slot::default).take(node_count * slots_per_node).collect();

    Self { slots, levels, branching_factor, slots_per_node, level_offsets, level_widths }
  }

  /// Returns the leaf level `l`.
  pub const fn levels(&self) -> usize {
    self.levels
  }

  /// Returns the branching factor `B`.
  pub const fn branching_factor(&self) -> usize {
    self.branching_factor
  }

  /// Returns the number of slots `Z` in each node.
  pub const fn slots_per_node(&self) -> usize {
    self.slots_per_node
  }

  /// Returns the number of nodes in the tree.
  pub fn node_count(&self) -> usize {
    self.slots.len() / self.slots_per_node
  }

  /// Returns the number of nodes on the bottom level, `B^l`.
  pub fn leaf_count(&self) -> usize {
    self.level_widths[self.levels]
  }

  /// Returns the number of slots in the whole tree.
  pub fn capacity(&self) -> usize {
    self.node_count() * self.slots_per_node
  }

  /// Returns all slots in level-order node/slot order.
  pub fn all_slots(&self) -> &[Slot<T>] {
    &self.slots
  }

  /// Returns the number of nodes at `level`, if the level exists.
  pub fn level_width(&self, level: usize) -> Option<usize> {
    self.level_widths.get(level).copied()
  }

  /// Returns the node index for a `(level, offset_in_level)` pair.
  pub fn node_index(&self, level: usize, offset_in_level: usize) -> Option<usize> {
    let level_width = self.level_width(level)?;
    if offset_in_level >= level_width {
      return None;
    }

    Some(self.level_offsets[level] + offset_in_level)
  }

  /// Returns the offset at `level` on the root-to-leaf path for `leaf_pos`.
  pub fn path_offset_at_level(&self, level: usize, leaf_pos: usize) -> Option<usize> {
    if level > self.levels || leaf_pos >= self.leaf_count() {
      return None;
    }

    let divisor = self.level_widths[self.levels - level];
    Some(leaf_pos / divisor)
  }

  /// Returns the node index at `level` on the root-to-leaf path for `leaf_pos`.
  pub fn path_node_index(&self, level: usize, leaf_pos: usize) -> Option<usize> {
    let offset = self.path_offset_at_level(level, leaf_pos)?;
    self.node_index(level, offset)
  }

  /// Returns all node indices on the root-to-leaf path for `leaf_pos`.
  pub fn path_node_indices(&self, leaf_pos: usize) -> Option<Vec<usize>> {
    if leaf_pos >= self.leaf_count() {
      return None;
    }

    let mut path = Vec::with_capacity(self.levels + 1);
    self.path_node_indices_into(leaf_pos, &mut path);
    Some(path)
  }

  /// Writes all node indices on the root-to-leaf path for `leaf_pos` into `out`.
  pub fn path_node_indices_into(&self, leaf_pos: usize, out: &mut Vec<usize>) -> bool {
    if leaf_pos >= self.leaf_count() {
      return false;
    }

    out.clear();
    out.reserve(self.levels + 1);
    for level in 0..=self.levels {
      let offset = leaf_pos / self.level_widths[self.levels - level];
      out.push(self.level_offsets[level] + offset);
    }

    true
  }

  /// Returns the longest common base-B prefix length of two leaf positions.
  pub fn common_prefix_depth(&self, a: usize, b: usize) -> Option<usize> {
    if a >= self.leaf_count() || b >= self.leaf_count() {
      return None;
    }

    let mut common_depth = 0;
    for level in 1..=self.levels {
      if self.path_offset_at_level(level, a) == self.path_offset_at_level(level, b) {
        common_depth = level;
      } else {
        break;
      }
    }

    Some(common_depth)
  }

  /// Returns the parent node index, or `None` for the root/out-of-bounds nodes.
  pub fn parent_index(&self, node_index: usize) -> Option<usize> {
    if node_index == 0 || node_index >= self.node_count() {
      None
    } else {
      Some((node_index - 1) / self.branching_factor)
    }
  }

  /// Returns child node indices for an internal node.
  pub fn children_indices(&self, node_index: usize) -> Option<Vec<usize>> {
    if node_index >= self.node_count() {
      return None;
    }

    let first_child = node_index.checked_mul(self.branching_factor)?.checked_add(1)?;
    if first_child >= self.node_count() {
      return None;
    }

    let after_last_child = first_child + self.branching_factor;
    Some((first_child..after_last_child).collect())
  }

  /// Removes and returns the first full slot in level-order.
  pub fn take_first_full(&mut self) -> Option<(SlotLocation, T)> {
    for (node_index, node_slots) in self.slots.chunks_exact_mut(self.slots_per_node).enumerate() {
      if let Some(slot_index) = node_slots.iter().position(Slot::is_full) {
        let Slot::Full(item) = std::mem::take(&mut node_slots[slot_index]) else {
          unreachable!("slot was checked as full")
        };
        return Some((SlotLocation { node_index, slot_index }, item));
      }
    }

    None
  }

  /// Removes and returns the first item matching `predicate` in level-order.
  pub fn take_first_matching<F>(&mut self, mut predicate: F) -> Option<(SlotLocation, T)>
  where
    F: FnMut(&T) -> bool,
  {
    for (node_index, node_slots) in self.slots.chunks_exact_mut(self.slots_per_node).enumerate() {
      for (slot_index, slot) in node_slots.iter_mut().enumerate() {
        if slot.as_ref().is_some_and(&mut predicate) {
          let Slot::Full(item) = std::mem::take(slot) else {
            unreachable!("slot was checked as full")
          };
          return Some((SlotLocation { node_index, slot_index }, item));
        }
      }
    }

    None
  }

  /// Returns all slots for `node_index`.
  pub fn node_slots(&self, node_index: usize) -> Option<&[Slot<T>]> {
    let start = self.slot_start(node_index)?;
    Some(&self.slots[start..start + self.slots_per_node])
  }

  /// Returns all slots for `node_index` mutably.
  pub fn node_slots_mut(&mut self, node_index: usize) -> Option<&mut [Slot<T>]> {
    let start = self.slot_start(node_index)?;
    Some(&mut self.slots[start..start + self.slots_per_node])
  }

  /// Counts full slots in the tree.
  pub fn occupied_slot_count(&self) -> usize {
    self.slots.iter().filter(|slot| slot.is_full()).count()
  }

  /// Returns true when every slot in the tree is full.
  pub fn is_full(&self) -> bool {
    self.occupied_slot_count() == self.capacity()
  }

  fn slot_start(&self, node_index: usize) -> Option<usize> {
    if node_index < self.node_count() {
      Some(node_index * self.slots_per_node)
    } else {
      None
    }
  }
}

/// A mock lane ORAM with a perfect B-ary tree and a FIFO queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestLaneOram {
  /// Tree storage for mock blocks.
  pub tree: PerfectBaryTree<TestLaneBlock>,
  /// Position map from key to leaf index.
  pub position_map: Vec<usize>,
  /// Tracks whether a key has ever been inserted.
  pub inserted: Vec<bool>,
  /// FIFO queue used as the temporary holding area between find and eviction.
  pub queue: VecDeque<TestLaneBlock>,
  /// Number of keys per leaf, `Y`.
  keys_per_leaf: usize,
}

impl TestLaneOram {
  /// Creates an empty mock lane ORAM.
  pub fn new(
    levels: usize,
    branching_factor: usize,
    slots_per_node: usize,
    keys_per_leaf: usize,
  ) -> Self {
    assert!(keys_per_leaf > 0, "keys_per_leaf must be non-zero");
    assert!(keys_per_leaf <= slots_per_node, "keys_per_leaf must be <= slots_per_node");

    let tree = PerfectBaryTree::new(levels, branching_factor, slots_per_node);
    let key_count = checked_key_count(keys_per_leaf, tree.leaf_count());
    let mut rng = rng();
    let position_map = (0..key_count).map(|_| rng.random_range(0..tree.leaf_count())).collect();

    Self {
      tree,
      position_map,
      inserted: vec![false; key_count],
      queue: VecDeque::new(),
      keys_per_leaf,
    }
  }

  /// Returns the number of keys per leaf, `Y`.
  pub const fn keys_per_leaf(&self) -> usize {
    self.keys_per_leaf
  }

  /// Returns the valid exclusive upper bound for positions, `B^l`.
  pub fn pos_bound(&self) -> usize {
    self.tree.leaf_count()
  }

  /// Returns the valid exclusive upper bound for keys, `Y * B^l`.
  pub fn key_bound(&self) -> usize {
    self.position_map.len()
  }

  /// Returns the current position map value for `key`.
  pub fn position_for_key(&self, key: K) -> usize {
    self.assert_key_in_range(key);
    self.position_map[key]
  }

  /// Performs a mock insert operation with a random new position.
  pub fn insert(&mut self, key: K) -> InsertResult {
    self.assert_key_in_range(key);
    let new_pos = rng().random_range(0..self.pos_bound());
    self.insert_with_new_pos(key, new_pos)
  }

  /// Performs a mock insert operation with a caller-chosen new position.
  fn insert_with_new_pos(&mut self, key: K, new_pos: usize) -> InsertResult {
    let mut path = Vec::with_capacity(self.tree.levels() + 1);
    self.insert_with_new_pos_using_path(key, new_pos, &mut path)
  }

  fn insert_with_new_pos_using_path(
    &mut self,
    key: K,
    new_pos: usize,
    path: &mut Vec<usize>,
  ) -> InsertResult {
    self.assert_key_pos_in_range(key, new_pos);

    let old_pos = self.position_map[key];
    assert!(self.tree.path_node_indices_into(old_pos, path), "old_pos was checked");
    let removed_from = self.remove_key_from_path(key, path);
    let first_insert = !self.inserted[key];

    self.inserted[key] = true;
    self.position_map[key] = new_pos;
    if first_insert || removed_from.is_some() {
      self.queue.push_back(TestLaneBlock::new(key, new_pos));
    }
    self.evict_path_with_nodes(old_pos, path);

    InsertResult { old_pos, new_pos, removed_from, queue_len: self.queue.len() }
  }

  /// Evicts queued/root-path blocks on the path to `leaf_pos`.
  #[cfg(test)]
  fn evict_path(&mut self, leaf_pos: usize) {
    self.assert_pos_in_range(leaf_pos);
    let mut path = Vec::with_capacity(self.tree.levels() + 1);
    assert!(self.tree.path_node_indices_into(leaf_pos, &mut path), "leaf_pos was checked");
    self.evict_path_with_nodes(leaf_pos, &path);
  }

  fn evict_path_with_nodes(&mut self, leaf_pos: usize, path: &[usize]) {
    self.fill_root_from_queue();

    for lane in 0..self.tree.slots_per_node() {
      self.evict_lane(leaf_pos, lane, path);
    }
  }

  fn remove_key_from_path(&mut self, key: K, path: &[usize]) -> Option<SlotLocation> {
    let slots_per_node = self.tree.slots_per_node;
    for &node_index in path {
      let start = node_index * slots_per_node;
      for slot_index in 0..slots_per_node {
        let slot = &mut self.tree.slots[start + slot_index];
        if slot.as_ref().is_some_and(|block| block.key == key) {
          *slot = Slot::Empty;
          return Some(SlotLocation { node_index, slot_index });
        }
      }
    }

    None
  }

  fn fill_root_from_queue(&mut self) {
    for slot in &mut self.tree.slots[..self.tree.slots_per_node] {
      if !slot.is_empty() {
        continue;
      }

      let Some(block) = self.queue.pop_front() else {
        break;
      };
      let block = TestLaneBlock::new(block.key, self.position_map[block.key]);
      *slot = Slot::Full(block);
    }
  }

  fn evict_lane(&mut self, leaf_pos: usize, lane: usize, path: &[usize]) {
    let slots_per_node = self.tree.slots_per_node;
    let levels = self.tree.levels();
    let branching_factor = self.tree.branching_factor();

    let mut empty_path_index = None;
    for path_index in 0..path.len() {
      let slot_index = path[path_index] * slots_per_node + lane;
      if self.tree.slots[slot_index].is_empty() {
        empty_path_index = Some(path_index);
      }
    }

    let Some(mut empty_path_index) = empty_path_index else {
      return;
    };

    while empty_path_index > 0 {
      let target_level = empty_path_index + 1;
      let mut source_path_index = None;

      for candidate_path_index in (0..empty_path_index).rev() {
        let candidate_slot_index = path[candidate_path_index] * slots_per_node + lane;
        if let Slot::Full(block) = &self.tree.slots[candidate_slot_index] {
          let slot_depth = common_prefix_depth(block.pos, leaf_pos, levels, branching_factor) + 1;
          if slot_depth >= target_level {
            source_path_index = Some(candidate_path_index);
            break;
          }
        }
      }

      if let Some(source_path_index) = source_path_index {
        let empty_slot_index = path[empty_path_index] * slots_per_node + lane;
        let source_slot_index = path[source_path_index] * slots_per_node + lane;
        self.tree.slots[empty_slot_index] = std::mem::take(&mut self.tree.slots[source_slot_index]);
        empty_path_index = source_path_index;
      } else {
        empty_path_index -= 1;
      }
    }
  }

  fn assert_key_pos_in_range(&self, key: K, pos: usize) {
    self.assert_key_in_range(key);
    self.assert_pos_in_range(pos);
  }

  fn assert_key_in_range(&self, key: K) {
    assert!(key < self.key_bound(), "key must be in range [0, Y * B^l)");
  }

  fn assert_pos_in_range(&self, pos: usize) {
    assert!(pos < self.pos_bound(), "pos must be in range [0, B^l)");
  }
}

/// Simulates queue growth and returns queue-size frequencies.
///
/// Parameters are `B`, `l`, `Y`, `Z`, and `N`, respectively. Internally keys are
/// represented as `0..Y * B^l`, which covers the full mathematical key universe
/// of size `Y * B^l`.
pub fn queue_size_frequencies(
  branching_factor: usize,
  levels: usize,
  keys_per_leaf: usize,
  slots_per_node: usize,
  iterations: usize,
) -> BTreeMap<usize, usize> {
  let mut rng = rng();
  let mut oram = TestLaneOram::new(levels, branching_factor, slots_per_node, keys_per_leaf);
  let key_count = oram.key_bound();
  let pos_count = oram.pos_bound();
  let mut path = Vec::with_capacity(levels + 1);

  for key in 0..key_count {
    let new_pos = rng.random_range(0..pos_count);
    oram.insert_with_new_pos_using_path(key, new_pos, &mut path);
  }

  let mut frequencies = Vec::new();
  for iteration in 0..iterations {
    let key = iteration % key_count;
    let new_pos = rng.random_range(0..pos_count);
    let result = oram.insert_with_new_pos_using_path(key, new_pos, &mut path);
    if result.queue_len >= frequencies.len() {
      frequencies.resize(result.queue_len + 1, 0);
    }
    frequencies[result.queue_len] += 1;
  }

  frequencies.into_iter().enumerate().filter(|(_, frequency)| *frequency != 0).collect()
}

fn checked_tree_shape(levels: usize, branching_factor: usize) -> (Vec<usize>, Vec<usize>, usize) {
  let mut level_offsets = Vec::with_capacity(levels + 1);
  let mut level_widths = Vec::with_capacity(levels + 1);
  let mut width = 1usize;
  let mut node_count = 0usize;

  for level in 0..=levels {
    level_offsets.push(node_count);
    level_widths.push(width);
    node_count = node_count.checked_add(width).expect("B-ary tree node count overflow");
    if level < levels {
      width = width.checked_mul(branching_factor).expect("B-ary tree level width overflow");
    }
  }

  (level_offsets, level_widths, node_count)
}

fn checked_level_width(level: usize, branching_factor: usize) -> usize {
  let mut width = 1usize;

  for _ in 0..level {
    width = width.checked_mul(branching_factor).expect("B-ary tree level width overflow");
  }

  width
}

fn checked_key_count(keys_per_leaf: usize, leaf_count: usize) -> usize {
  keys_per_leaf.checked_mul(leaf_count).expect("key count overflow")
}

fn common_prefix_depth(a: usize, b: usize, levels: usize, branching_factor: usize) -> usize {
  let mut common_depth = 0;
  let mut divisor = checked_level_width(levels, branching_factor);

  for level in 1..=levels {
    divisor /= branching_factor;
    if a / divisor == b / divisor {
      common_depth = level;
    } else {
      break;
    }
  }

  common_depth
}

#[cfg(test)]
mod tests {
  use super::*;

  fn block_at(oram: &TestLaneOram, node_index: usize, slot_index: usize) -> Option<TestLaneBlock> {
    oram.tree.node_slots(node_index).unwrap()[slot_index].as_ref().cloned()
  }

  fn count_key(oram: &TestLaneOram, key: K) -> usize {
    oram
      .tree
      .all_slots()
      .iter()
      .filter(|slot| slot.as_ref().is_some_and(|block| block.key == key))
      .count()
  }

  #[test]
  fn test_perfect_bary_tree_shape() {
    let tree = PerfectBaryTree::<usize>::new(3, 3, 2);

    assert_eq!(tree.levels(), 3);
    assert_eq!(tree.branching_factor(), 3);
    assert_eq!(tree.slots_per_node(), 2);
    assert_eq!(tree.node_count(), 40);
    assert_eq!(tree.leaf_count(), 27);
    assert_eq!(tree.capacity(), 80);
    assert_eq!(tree.level_width(0), Some(1));
    assert_eq!(tree.level_width(1), Some(3));
    assert_eq!(tree.level_width(2), Some(9));
    assert_eq!(tree.level_width(3), Some(27));
    assert_eq!(tree.level_width(4), None);
    assert_eq!(tree.node_index(3, 26), Some(39));
    assert_eq!(tree.path_offset_at_level(2, 17), Some(5));
    assert_eq!(tree.path_node_index(3, 26), Some(39));
    assert_eq!(tree.path_node_indices(26), Some(vec![0, 3, 12, 39]));
    assert_eq!(tree.common_prefix_depth(26, 25), Some(2));
    assert_eq!(tree.parent_index(6), Some(1));
    assert_eq!(tree.children_indices(1), Some(vec![4, 5, 6]));
    assert_eq!(tree.children_indices(13), None);
  }

  #[test]
  fn test_insert_on_empty_tree_updates_position_map_and_fills_root() {
    let mut oram = TestLaneOram::new(2, 2, 3, 2);
    oram.position_map[7] = 0;

    let result = oram.insert_with_new_pos(7, 3);

    assert_eq!(result.old_pos, 0);
    assert_eq!(result.new_pos, 3);
    assert_eq!(result.removed_from, None);
    assert_eq!(result.queue_len, 0);
    assert_eq!(oram.key_bound(), 8);
    assert_eq!(oram.pos_bound(), 4);
    assert_eq!(oram.position_for_key(7), 3);
    assert_eq!(block_at(&oram, 0, 0), Some(TestLaneBlock::new(7, 3)));
  }

  #[test]
  fn test_insert_removes_existing_key_from_old_path_before_reinserting() {
    let mut oram = TestLaneOram::new(2, 2, 3, 2);
    oram.position_map[1] = 0;
    oram.insert_with_new_pos(1, 0);
    let leaf_on_old_path = oram.tree.path_node_index(2, 0).unwrap();

    let result = oram.insert_with_new_pos(1, 3);

    assert_eq!(result.old_pos, 0);
    assert_eq!(result.new_pos, 3);
    assert_eq!(
      result.removed_from,
      Some(SlotLocation { node_index: leaf_on_old_path, slot_index: 0 })
    );
    assert_eq!(result.queue_len, 0);
    assert_eq!(oram.position_for_key(1), 3);
    assert_eq!(count_key(&oram, 1), 1);
    assert_eq!(block_at(&oram, 0, 0), Some(TestLaneBlock::new(1, 3)));
  }

  #[test]
  fn test_evict_fills_empty_root_slots_from_queue_left_first() {
    let mut oram = TestLaneOram::new(2, 2, 3, 2);
    oram.position_map[0] = 3;
    oram.position_map[1] = 3;
    oram.queue.push_back(TestLaneBlock::new(0, 3));
    oram.queue.push_back(TestLaneBlock::new(1, 3));

    oram.evict_path(0);

    assert!(oram.queue.is_empty());
    assert_eq!(block_at(&oram, 0, 0), Some(TestLaneBlock::new(0, 3)));
    assert_eq!(block_at(&oram, 0, 1), Some(TestLaneBlock::new(1, 3)));
    assert_eq!(block_at(&oram, 0, 2), None);
  }

  #[test]
  fn test_evict_keeps_items_queued_when_root_is_full_and_no_lane_has_empty_slots() {
    let mut oram = TestLaneOram::new(0, 2, 2, 1);
    let root_slots = oram.tree.node_slots_mut(0).unwrap();
    root_slots[0] = Slot::Full(TestLaneBlock::new(0, 0));
    root_slots[1] = Slot::Full(TestLaneBlock::new(0, 0));
    oram.queue.push_back(TestLaneBlock::new(0, 0));

    oram.evict_path(0);

    assert_eq!(oram.queue.len(), 1);
    assert_eq!(oram.queue.front().map(|block| block.key), Some(0));
  }

  #[test]
  fn test_lane_eviction_moves_root_slot_down() {
    let mut oram = TestLaneOram::new(2, 2, 3, 1);
    let leaf_on_path = oram.tree.path_node_index(2, 0).unwrap();
    oram.tree.node_slots_mut(0).unwrap()[0] = Slot::Full(TestLaneBlock::new(0, 0));

    oram.evict_path(0);

    assert_eq!(block_at(&oram, 0, 0), None);
    assert_eq!(block_at(&oram, leaf_on_path, 0), Some(TestLaneBlock::new(0, 0)));
  }

  #[test]
  fn test_lane_eviction_shifts_eligible_chain_down() {
    let mut oram = TestLaneOram::new(2, 2, 3, 1);
    let child_on_path = oram.tree.path_node_index(1, 0).unwrap();
    let leaf_on_path = oram.tree.path_node_index(2, 0).unwrap();

    oram.tree.node_slots_mut(0).unwrap()[0] = Slot::Full(TestLaneBlock::new(0, 1));
    oram.tree.node_slots_mut(child_on_path).unwrap()[0] = Slot::Full(TestLaneBlock::new(1, 0));

    oram.evict_path(0);

    assert_eq!(block_at(&oram, 0, 0), None);
    assert_eq!(block_at(&oram, child_on_path, 0), Some(TestLaneBlock::new(0, 1)));
    assert_eq!(block_at(&oram, leaf_on_path, 0), Some(TestLaneBlock::new(1, 0)));
  }

  #[test]
  fn test_reinserting_queued_key_does_not_duplicate_queue_entry() {
    let mut oram = TestLaneOram::new(1, 2, 3, 1);
    oram.inserted[0] = true;
    oram.queue.push_back(TestLaneBlock::new(0, 0));

    for slot in oram.tree.node_slots_mut(0).unwrap() {
      *slot = Slot::Full(TestLaneBlock::new(1, 0));
    }

    let result = oram.insert_with_new_pos(0, 1);

    assert_eq!(result.removed_from, None);
    assert_eq!(oram.queue.len(), 1);
    assert_eq!(oram.position_for_key(0), 1);

    for slot in oram.tree.node_slots_mut(0).unwrap() {
      *slot = Slot::Empty;
    }
    oram.evict_path(0);

    assert!(oram.queue.is_empty());
    assert_eq!(block_at(&oram, 0, 0), Some(TestLaneBlock::new(0, 1)));
  }

  #[test]
  #[should_panic(expected = "pos must be in range [0, B^l)")]
  fn test_pos_must_be_in_bottom_level_range() {
    let mut oram = TestLaneOram::new(2, 2, 3, 2);

    oram.insert_with_new_pos(0, 4);
  }

  #[test]
  #[should_panic(expected = "key must be in range [0, Y * B^l)")]
  fn test_key_must_be_in_key_universe_range() {
    let mut oram = TestLaneOram::new(2, 2, 3, 2);

    oram.insert(8);
  }

  #[test]
  fn test_position_map_has_y_times_leaf_count_entries() {
    let mut oram = TestLaneOram::new(2, 2, 3, 2);

    assert_eq!(oram.keys_per_leaf(), 2);
    assert_eq!(oram.pos_bound(), 4);
    assert_eq!(oram.key_bound(), 8);
    assert_eq!(oram.position_map.len(), 8);
    assert!(oram.position_map.iter().all(|pos| *pos < oram.pos_bound()));

    oram.insert_with_new_pos(7, 3);

    assert_eq!(oram.position_for_key(7), 3);
  }

  #[test]
  fn test_queue_size_frequencies_records_each_iteration() {
    let iterations = 25;
    let frequencies = queue_size_frequencies(2, 2, 1, 2, iterations);

    assert_eq!(frequencies.values().sum::<usize>(), iterations);
    assert!(frequencies.keys().all(|queue_len| *queue_len <= iterations + 4));
    assert!(queue_size_frequencies(2, 2, 1, 2, 0).is_empty());
  }

  #[test]
  #[ignore = "long-running simulation; run with `cargo test -p rostl-oram --release test_queue_size_frequencies_large -- --ignored --nocapture`"]
  fn test_queue_size_frequencies_large() {
    let iterations = 10_000_000;
    let frequencies = queue_size_frequencies(4, 10, 4, 5, iterations);

    println!("{frequencies:?}");
  }

  #[test]
  #[should_panic(expected = "keys_per_leaf must be <= slots_per_node")]
  fn test_keys_per_leaf_must_be_less_than_slots_per_node() {
    TestLaneOram::new(2, 2, 2, 3);
  }
}
