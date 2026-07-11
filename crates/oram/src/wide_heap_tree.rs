//! A heap tree with a configurable power-of-two branching factor.
//!
//! Paths use lexicographical prefix order. At depth `d`, the highest `d`
//! base-`branching_factor` digits of a path identify the node.

use crate::prelude::PositionType;

#[derive(Debug, Clone, Copy)]
struct Level {
  offset: usize,
  path_mask: usize,
  path_shift: u32,
}

/// An array-backed heap tree with a configurable branching factor.
#[derive(Debug)]
pub struct WideHeapTree<T> {
  /// Actual storage container.
  pub(crate) tree: Vec<T>,
  /// Public indexing metadata for each level.
  levels: Vec<Level>,
  /// Height of the tree (a tree containing only its root has height 1).
  pub height: usize,
  /// Number of children of each non-leaf node.
  pub branching_factor: usize,
}

impl<T> WideHeapTree<T>
where
  T: Default + Clone,
{
  /// Initializes an empty tree of `height` with the requested branching factor.
  pub fn new(height: usize, branching_factor: usize) -> Self {
    Self::new_with(height, branching_factor, T::default())
  }
}

impl<T> WideHeapTree<T>
where
  T: Clone,
{
  /// Initializes a tree whose nodes are copies of `default`.
  pub fn new_with(height: usize, branching_factor: usize, default: T) -> Self {
    debug_assert!(height > 0, "a tree must contain at least its root");
    debug_assert!(branching_factor >= 2, "the branching factor must be at least 2");
    debug_assert!(branching_factor.is_power_of_two(), "the branching factor must be a power of 2");

    let mut levels = Vec::with_capacity(height);
    let mut node_count = 0usize;
    let mut level_width = 1usize;
    let bits_per_digit = branching_factor.trailing_zeros();
    for depth in 0..height {
      let path_shift = ((height - 1 - depth) as u32) * bits_per_digit;
      levels.push(Level { offset: node_count, path_mask: level_width - 1, path_shift });
      node_count += level_width;
      level_width *= branching_factor;
    }

    let tree = vec![default; node_count];
    Self { tree, levels, height, branching_factor }
  }
}

impl<T> WideHeapTree<T> {
  /// Returns the array index of the node on `path` at `depth`.
  #[inline]
  pub fn get_index(&self, depth: usize, path: PositionType) -> usize {
    debug_assert!(depth < self.height, "depth is outside the tree");

    let level = self.levels[depth];
    level.offset + ((path as usize >> level.path_shift) & level.path_mask)
  }

  /// Returns the node on `path` at `depth`.
  #[inline]
  pub fn get_path_at_depth(&self, depth: usize, path: PositionType) -> &T {
    &self.tree[self.get_index(depth, path)]
  }

  /// Returns the mutable node on `path` at `depth`.
  #[inline]
  pub fn get_path_at_depth_mut(&mut self, depth: usize, path: PositionType) -> &mut T {
    let index = self.get_index(depth, path);
    &mut self.tree[index]
  }

  /// Returns the siblings before and after the node on `path` at `depth`.
  ///
  /// The two slices make the accessed memory ranges explicit and exclude the
  /// selected node without a filtering iterator or an allocation.
  pub fn get_siblings(&self, depth: usize, path: PositionType) -> (&[T], &[T]) {
    debug_assert!(depth > 0, "the root has no siblings");
    let index = self.get_index(depth, path);
    let level_offset = self.levels[depth].offset;
    let index_in_level = index - level_offset;
    let family_start = level_offset + (index_in_level & !(self.branching_factor - 1));
    let selected = index - family_start;
    let family = &self.tree[family_start..family_start + self.branching_factor];
    let (before, selected_and_after) = family.split_at(selected);
    (before, &selected_and_after[1..])
  }

  /// Returns the total number of nodes in the tree.
  pub fn len(&self) -> usize {
    self.tree.len()
  }

  /// Returns whether the tree contains no nodes.
  pub fn is_empty(&self) -> bool {
    self.tree.is_empty()
  }

  /// Returns the public number of root-to-leaf paths.
  pub fn path_count(&self) -> usize {
    self.levels[self.height - 1].path_mask + 1
  }
}

#[cfg(test)]
mod tests {
  use super::WideHeapTree;

  #[test]
  fn quaternary_tree_has_expected_shape_and_indices() {
    let tree = WideHeapTree::<u8>::new(3, 4);
    assert_eq!(tree.len(), 21);
    assert_eq!(tree.get_index(0, 15), 0);
    assert_eq!(tree.get_index(1, 15), 4);
    assert_eq!(tree.get_index(2, 15), 20);
  }

  #[test]
  fn mutable_path_access_and_siblings_work() {
    let mut tree = WideHeapTree::<u8>::new(3, 4);
    *tree.get_path_at_depth_mut(2, 7) = 42;
    assert_eq!(*tree.get_path_at_depth(2, 7), 42);

    let (before, after) = tree.get_siblings(2, 7);
    let siblings: Vec<_> = before.iter().chain(after).copied().collect();
    assert_eq!(siblings, vec![0, 0, 0]);
  }

  #[test]
  fn binary_indices_follow_high_order_prefixes() {
    let wide = WideHeapTree::<u8>::new(4, 2);
    assert_eq!(wide.get_index(0, 5), 0);
    assert_eq!(wide.get_index(1, 5), 2);
    assert_eq!(wide.get_index(2, 5), 5);
    assert_eq!(wide.get_index(3, 5), 12);
  }
}
