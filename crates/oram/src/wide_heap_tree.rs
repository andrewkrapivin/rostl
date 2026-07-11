//! A heap tree with a configurable branching factor.
//!
//! Like [`crate::heap_tree::HeapTree`], paths are stored in reverse
//! lexicographical order. At depth `d`, the lowest `d` base-`branching_factor`
//! digits of a path identify the node.

use crate::prelude::PositionType;

/// An array-backed heap tree with a configurable branching factor.
#[derive(Debug)]
pub struct WideHeapTree<T> {
  /// Actual storage container.
  pub(crate) tree: Vec<T>,
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
    debug_assert!(branching_factor > 1, "the branching factor must be at least 2");

    let node_count = geometric_sum(branching_factor, height);
    let tree = vec![default; node_count];
    Self { tree, height, branching_factor }
  }
}

impl<T> WideHeapTree<T> {
  /// Returns the array index of the node on `path` at `depth`.
  #[inline]
  pub fn get_index(&self, depth: usize, path: PositionType) -> usize {
    debug_assert!(depth < self.height, "depth is outside the tree");

    let level_width = self.branching_factor.pow(depth as u32);
    let level_offset = geometric_sum(self.branching_factor, depth);
    level_offset + (path as usize % level_width)
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
    let level_offset = geometric_sum(self.branching_factor, depth);
    let index_in_level = index - level_offset;
    let family_start =
      level_offset + (index_in_level / self.branching_factor) * self.branching_factor;
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
}

fn geometric_sum(branching_factor: usize, terms: usize) -> usize {
  let mut sum = 0usize;
  let mut width = 1usize;
  for _ in 0..terms {
    sum += width;
    width *= branching_factor;
  }
  sum
}

#[cfg(test)]
mod tests {
  use super::WideHeapTree;

  #[test]
  fn ternary_tree_has_expected_shape_and_indices() {
    let tree = WideHeapTree::<u8>::new(3, 3);
    assert_eq!(tree.len(), 13);
    assert_eq!(tree.get_index(0, 8), 0);
    assert_eq!(tree.get_index(1, 8), 3);
    assert_eq!(tree.get_index(2, 8), 12);
  }

  #[test]
  fn mutable_path_access_and_siblings_work() {
    let mut tree = WideHeapTree::<u8>::new(3, 3);
    *tree.get_path_at_depth_mut(2, 7) = 42;
    assert_eq!(*tree.get_path_at_depth(2, 7), 42);

    let (before, after) = tree.get_siblings(2, 7);
    let siblings: Vec<_> = before.iter().chain(after).copied().collect();
    assert_eq!(siblings, vec![0, 0]);
  }

  #[test]
  fn binary_indices_match_heap_tree() {
    let wide = WideHeapTree::<u8>::new(4, 2);
    let binary = crate::heap_tree::HeapTree::<u8>::new(4);
    for depth in 0..4 {
      for path in 0..8 {
        assert_eq!(wide.get_index(depth, path), binary.get_index(depth, path));
      }
    }
  }
}
