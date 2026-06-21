use bytemuck::Pod;
use std::fmt::Debug;

use rand::{rng, Rng};
use rostl_oram::{circuit_oram::CircuitORAM, prelude::PositionType};
use rostl_primitives::{ooption::OOption, traits::Cmov, utils::min};
use rostl_sort::bitonic::bitonic_payload_sort;

use crate::b_tree_components::{BpTreeNode, BpValueNode, KeyValuePair, EXACT_MATCH};

/// A B+ tree. Big in the sense that it has at least two levels
/// Inline in the sense that values are stored inline with the keys at the last level
/// T is key type
/// V is value type
/// FB is the first level branching factor, FBM1 is FB-1
/// B is the branching factor for all subsequent levels, BM1 is B-1
/// VB is the number of K-V pairs to put in a node at the bottom level.
#[derive(Debug)]
pub struct BigInlineBpTree<
  T,
  V,
  const FBM1: usize,
  const FB: usize,
  const BM1: usize,
  const B: usize,
  const VB: usize,
> where
  T: Cmov + Pod + Ord + Debug,
  V: Cmov + Pod + Debug,
{
  first_level: BpTreeNode<T, FBM1, FB>,
  recursive_orams: Vec<CircuitORAM<BpTreeNode<T, BM1, B>>>,
  final_level: CircuitORAM<BpValueNode<T, V, VB>>
}

impl<T, V, const FBM1: usize, const FB: usize, const BM1: usize, const B: usize, const VB: usize>
  BigInlineBpTree<T, V, FBM1, FB, BM1, B, VB>
where
  T: Cmov + Pod + Ord + Debug,
  V: Cmov + Pod + Debug,
{
  /// Creates a new static Path ORAM b tree. Leaks the number of keys
  pub fn new(keys: &mut [T], initial_values: &mut [V]) -> Self {
    debug_assert!(keys.len() == initial_values.len());
    let n = keys.len();
    let mut rng = rng();
    bitonic_payload_sort(keys, initial_values);

    // Build largest level
    let largest_level_blocks = (n + VB - 1) / VB;
    let mut final_level = CircuitORAM::<BpValueNode<T, V, VB>>::new(largest_level_blocks);
    let mut final_level_positions = Vec::<PositionType>::with_capacity(largest_level_blocks);
    let mut promoted_keys = Vec::<T>::with_capacity(largest_level_blocks - 1);
    for i in 0..largest_level_blocks {
      let pos = rng.random_range(0..largest_level_blocks as PositionType);
      final_level_positions.push(pos);
      let start = i * VB;
      let end = min(n, (i + 1) * VB);
      let len = end - start;
      debug_assert!(len > 0);
      let block = BpValueNode::<T, V, VB>::new(&keys[start..end], &initial_values[start..end], len);
      final_level.write_or_insert(i as PositionType, pos, i, block);
      if i > 0 {
        promoted_keys.push(keys[start]);
      }
    }

    let mut recursive_orams = Vec::<CircuitORAM<BpTreeNode<T, BM1, B>>>::new();
    let mut current_keys = promoted_keys;
    // we take blocks of B keys, where B-1 keys are in a node and the B'th key gets promoted, except for the LAST block, where no key gets promoted so its only B-1.
    // Therefore, the number of blocks is ceil((current_keys.len() + 1) / B), with the +1 to get the final block to size B.
    let mut current_level_blocks = (current_keys.len() + B) / B;
    let mut prev_level_positions = final_level_positions;
    while prev_level_positions.len() > FB {
      debug_assert!(current_keys.len() == prev_level_positions.len() - 1);
      promoted_keys = Vec::<T>::with_capacity(current_level_blocks - 1);
      let mut current_level = CircuitORAM::<BpTreeNode<T, BM1, B>>::new(current_level_blocks);
      let mut current_level_positions = Vec::<PositionType>::with_capacity(current_level_blocks);
      for i in 0..current_level_blocks {
        let pos = rng.random_range(0..current_level_blocks as PositionType);
        current_level_positions.push(pos);
        let start = i * B;
        // we want to take B-1 keys in a block of B, then B'th key gets promoted
        let end = min(current_keys.len(), (i + 1) * B - 1);
        let len = end - start;
        debug_assert!(len > 0);
        let block = BpTreeNode::<T, BM1, B>::new(
          &current_keys[start..end],
          &prev_level_positions[start..(end + 1)],
          len,
        );
        current_level.write_or_insert(i as PositionType, pos, i, block);
        if i + 1 < current_level_blocks {
          promoted_keys.push(current_keys[end]);
        }
      }
      prev_level_positions = current_level_positions;
      current_keys = promoted_keys;
      current_level_blocks = (current_keys.len() + B) / B;
      recursive_orams.push(current_level);
    }

    // Added the orams biggest first to the vector, so let's reverse it
    recursive_orams.reverse();

    debug_assert!(current_keys.len() == prev_level_positions.len() - 1);

    // Build first level
    let first_level =
      BpTreeNode::<T, FBM1, FB>::new(&current_keys, &prev_level_positions, current_keys.len());

    Self { first_level, recursive_orams, final_level }
  }

  /// Returns Some key-value pair corresponding to key if key is there, otherwise returns None
  pub fn point_query_kv(&mut self, key: T) -> OOption<V> {
    let mut rng = rng();
    let first_child_max_n = if self.recursive_orams.is_empty() {
      self.final_level.max_n
    } else {
      self.recursive_orams[0].max_n
    };
    let mut current_new_pos = rng.random_range(0..first_child_max_n as PositionType);
    let (mut current_index, mut current_pos) =
      self.first_level.search_update_index(key, current_new_pos, EXACT_MATCH);

    for i in 0..self.recursive_orams.len() {
      let next_level_max_n = if i + 1 == self.recursive_orams.len() {
        self.final_level.max_n
      } else {
        self.recursive_orams[i + 1].max_n
      };
      let next_new_pos = rng.random_range(0..next_level_max_n as PositionType);
      let (_found, (child_index, next_pos)) =
        self.recursive_orams[i].update(current_pos, current_new_pos, current_index, |node| {
          node.search_update_index(key, next_new_pos, EXACT_MATCH)
        });
      debug_assert!(_found);
      current_index = current_index * B + child_index;
      current_pos = next_pos;
      current_new_pos = next_new_pos;
    }

    let (_found, ret) =
      self
        .final_level
        .update(current_pos, current_new_pos, current_index, |node| {
          node.search(key, EXACT_MATCH)
        });
    debug_assert!(_found);
    ret
  }

  /// Returns Some key corresponding to search type, otherwise returns None
  pub fn query_key(&mut self, key: T, search_type: usize) -> OOption<T> {
    let mut rng = rng();
    let first_child_max_n = if self.recursive_orams.is_empty() {
      self.final_level.max_n
    } else {
      self.recursive_orams[0].max_n
    };
    let mut current_new_pos = rng.random_range(0..first_child_max_n as PositionType);
    let (mut ret, mut current_index, mut current_pos) =
      self.first_level.search_update_key_index(key, current_new_pos, search_type);

    for i in 0..self.recursive_orams.len() {
      let next_level_max_n = if i + 1 == self.recursive_orams.len() {
        self.final_level.max_n
      } else {
        self.recursive_orams[i + 1].max_n
      };
      let next_new_pos = rng.random_range(0..next_level_max_n as PositionType);
      let (_found, (node_ret, child_index, next_pos)) =
        self.recursive_orams[i].update(current_pos, current_new_pos, current_index, |node| {
          node.search_update_key_index(key, next_new_pos, search_type)
        });
      debug_assert!(_found);
      ret.cmov(&node_ret, node_ret.is_some());
      current_index = current_index * B + child_index;
      current_pos = next_pos;
      current_new_pos = next_new_pos;
    }

    let (_found, leaf_ret) =
      self
        .final_level
        .update(current_pos, current_new_pos, current_index, |node| {
          node.search_key(key, search_type)
        });
    debug_assert!(_found);
    ret.cmov(&leaf_ret, leaf_ret.is_some());
    ret
  }
}






#[cfg(test)]
mod tests {
  use super::*;
  use crate::b_tree_components::{EXACT_MATCH, PREDECESSOR, SUCCESSOR};

  #[test]
  fn builds_small_tree_with_first_level_over_final_level() {
    let mut keys = [30_u64, 10, 60, 20, 50, 40];
    let mut values = [300_u64, 100, 600, 200, 500, 400];

    let tree = BigInlineBpTree::<u64, u64, 2, 3, 2, 3, 2>::new(&mut keys, &mut values);

    assert_eq!(keys, [10, 20, 30, 40, 50, 60]);
    assert_eq!(values, [100, 200, 300, 400, 500, 600]);
    assert_eq!(tree.recursive_orams.len(), 0);
  }

  #[test]
  fn lookup_small_tree_with_first_level_over_final_level() {
    let mut keys = [30_u64, 10, 60, 20, 50, 40];
    let mut values = [300_u64, 100, 600, 200, 500, 400];

    let mut tree = BigInlineBpTree::<u64, u64, 2, 3, 2, 3, 2>::new(&mut keys, &mut values);

    for i in 1..=6 {
      let ret = tree.point_query_kv(i * 10);
      assert!(ret.is_some());
      assert_eq!(ret.unwrap(), i * 100);
    }

    let ret = tree.point_query_kv(35);
    assert!(!ret.is_some());
  }

  #[test]
  fn search_key_small_tree_with_successor_and_predecessor() {
    let mut keys = [30_u64, 10, 60, 20, 50, 40];
    let mut values = [300_u64, 100, 600, 200, 500, 400];

    let mut tree = BigInlineBpTree::<u64, u64, 2, 3, 2, 3, 2>::new(&mut keys, &mut values);

    let successor = tree.query_key(35, SUCCESSOR);
    assert!(successor.is_some());
    assert_eq!(successor.unwrap(), 40);

    let predecessor = tree.query_key(35, PREDECESSOR);
    assert!(predecessor.is_some());
    assert_eq!(predecessor.unwrap(), 30);

    let boundary_successor = tree.query_key(20, SUCCESSOR);
    assert!(boundary_successor.is_some());
    assert_eq!(boundary_successor.unwrap(), 30);

    let boundary_predecessor = tree.query_key(30, PREDECESSOR);
    assert!(boundary_predecessor.is_some());
    assert_eq!(boundary_predecessor.unwrap(), 20);

    let exact = tree.query_key(40, EXACT_MATCH);
    assert!(exact.is_some());
    assert_eq!(exact.unwrap(), 40);

    let exact_min = tree.query_key(10, EXACT_MATCH);
    assert!(exact_min.is_some());
    assert_eq!(exact_min.unwrap(), 10);

    let exact_max = tree.query_key(60, EXACT_MATCH);
    assert!(exact_max.is_some());
    assert_eq!(exact_max.unwrap(), 60);

    let below_min_successor = tree.query_key(5, SUCCESSOR);
    assert!(below_min_successor.is_some());
    assert_eq!(below_min_successor.unwrap(), 10);

    let above_max_predecessor = tree.query_key(65, PREDECESSOR);
    assert!(above_max_predecessor.is_some());
    assert_eq!(above_max_predecessor.unwrap(), 60);

    assert!(!tree.query_key(60, SUCCESSOR).is_some());
    assert!(!tree.query_key(10, PREDECESSOR).is_some());
    assert!(!tree.query_key(35, EXACT_MATCH).is_some());
  }

  #[test]
  fn builds_small_tree_with_recursive_level() {
    let mut keys = [
      15_u64, 3, 29, 1, 18, 9, 22, 7, 30, 12, 5, 25, 16, 2, 27, 10, 20, 6, 24, 14, 8, 19, 4, 28,
      11, 21, 13, 23, 17, 26,
    ];
    let mut values = [
      150_u64, 30, 290, 10, 180, 90, 220, 70, 300, 120, 50, 250, 160, 20, 270, 100, 200, 60, 240,
      140, 80, 190, 40, 280, 110, 210, 130, 230, 170, 260,
    ];

    let tree = BigInlineBpTree::<u64, u64, 3, 4, 2, 3, 2>::new(&mut keys, &mut values);

    assert_eq!(
      keys,
      [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30,
      ]
    );
    assert_eq!(
      values,
      [
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
        210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
      ]
    );
    assert_eq!(tree.recursive_orams.len(), 2);
  }

  #[test]
  fn lookup_small_tree_with_recursive_level() {
    let mut keys = [
      15_u64, 3, 29, 1, 18, 9, 22, 7, 30, 12, 5, 25, 16, 2, 27, 10, 20, 6, 24, 14, 8, 19, 4, 28,
      11, 21, 13, 23, 17, 26,
    ];
    let mut values = [
      150_u64, 30, 290, 10, 180, 90, 220, 70, 300, 120, 50, 250, 160, 20, 270, 100, 200, 60, 240,
      140, 80, 190, 40, 280, 110, 210, 130, 230, 170, 260,
    ];

    let mut tree = BigInlineBpTree::<u64, u64, 3, 4, 2, 3, 2>::new(&mut keys, &mut values);

    for _ in 0..3 {
      for i in 1..25 {
        let ret = tree.point_query_kv(i);
        assert!(ret.is_some());
        assert_eq!(ret.unwrap(), i * 10);
      }
    }

    let ret = tree.point_query_kv(0);
    assert!(!ret.is_some());
  }

  #[test]
  fn search_key_recursive_tree_with_boundary_cases() {
    let mut keys = [
      15_u64, 3, 29, 1, 18, 9, 22, 7, 30, 12, 5, 25, 16, 2, 27, 10, 20, 6, 24, 14, 8, 19, 4, 28,
      11, 21, 13, 23, 17, 26,
    ];
    let mut values = [
      150_u64, 30, 290, 10, 180, 90, 220, 70, 300, 120, 50, 250, 160, 20, 270, 100, 200, 60, 240,
      140, 80, 190, 40, 280, 110, 210, 130, 230, 170, 260,
    ];

    let mut tree = BigInlineBpTree::<u64, u64, 3, 4, 2, 3, 2>::new(&mut keys, &mut values);

    let successor = tree.query_key(11, SUCCESSOR);
    assert!(successor.is_some());
    assert_eq!(successor.unwrap(), 12);

    let predecessor = tree.query_key(12, PREDECESSOR);
    assert!(predecessor.is_some());
    assert_eq!(predecessor.unwrap(), 11);

    let successor = tree.query_key(21, SUCCESSOR);
    assert!(successor.is_some());
    assert_eq!(successor.unwrap(), 22);

    let predecessor = tree.query_key(22, PREDECESSOR);
    assert!(predecessor.is_some());
    assert_eq!(predecessor.unwrap(), 21);

    let boundary_successor = tree.query_key(12, SUCCESSOR);
    assert!(boundary_successor.is_some());
    assert_eq!(boundary_successor.unwrap(), 13);

    let boundary_predecessor = tree.query_key(13, PREDECESSOR);
    assert!(boundary_predecessor.is_some());
    assert_eq!(boundary_predecessor.unwrap(), 12);

    let recursive_boundary_successor = tree.query_key(24, SUCCESSOR);
    assert!(recursive_boundary_successor.is_some());
    assert_eq!(recursive_boundary_successor.unwrap(), 25);

    let recursive_boundary_predecessor = tree.query_key(25, PREDECESSOR);
    assert!(recursive_boundary_predecessor.is_some());
    assert_eq!(recursive_boundary_predecessor.unwrap(), 24);

    let below_min_successor = tree.query_key(0, SUCCESSOR);
    assert!(below_min_successor.is_some());
    assert_eq!(below_min_successor.unwrap(), 1);

    let above_max_predecessor = tree.query_key(31, PREDECESSOR);
    assert!(above_max_predecessor.is_some());
    assert_eq!(above_max_predecessor.unwrap(), 30);

    assert!(!tree.query_key(30, SUCCESSOR).is_some());
    assert!(!tree.query_key(1, PREDECESSOR).is_some());
    assert!(!tree.query_key(31, EXACT_MATCH).is_some());
  }
}
