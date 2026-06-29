//! Represents a heap tree as an array and provides functions to access it.
//!
//! The tree is represented by a reverse lexicographical order heap tree.
//! For the default binary branching factor:
//!                            0
//!                     1             2
//!                3         5    4       6
//! Path:          0         2    1       3

use crate::prelude::PositionType;

/// Represents a heap tree structure.
#[derive(Debug)]
pub struct HeapTree<T> {
  /// Actual storage container
  pub(crate) tree: Vec<T>,
  /// Height of the tree, public (tree with a single element has height 1)
  pub height: usize,
  /// Number of children of each internal node.
  pub branching_factor: usize,
  /// Number of leaves in the tree.
  pub leaf_count: usize,
}

impl<T> HeapTree<T>
where
  T: Default + Clone,
{
  /// Initializes a new binary heap tree with a certain height.
  pub fn new(height: usize) -> Self {
    Self::new_with_branching_factor(height, 2)
  }

  /// Initializes a new heap tree with a certain height and branching factor.
  pub fn new_with_branching_factor(height: usize, branching_factor: usize) -> Self {
    Self::new_with_branching_factor_and_value(height, branching_factor, T::default())
  }
}

impl<T> HeapTree<T>
where
  T: Clone,
{
  /// Initializes a new binary heap tree with a certain height and a default value.
  pub fn new_with(height: usize, default: T) -> Self {
    Self::new_with_branching_factor_and_value(height, 2, default)
  }

  /// Initializes a new heap tree with a certain height, branching factor, and default value.
  pub fn new_with_branching_factor_and_value(
    height: usize,
    branching_factor: usize,
    default: T,
  ) -> Self {
    let leaf_count = level_width(height.saturating_sub(1), branching_factor);
    Self::new_with_leaf_count_and_value(height, branching_factor, leaf_count, default)
  }

  /// Initializes a new heap tree with a certain height, branching factor, leaf count, and default value.
  pub fn new_with_leaf_count_and_value(
    height: usize,
    branching_factor: usize,
    leaf_count: usize,
    default: T,
  ) -> Self {
    assert!(branching_factor >= 2);
    assert!(height > 0);
    assert!(leaf_count > 0);
    debug_assert!(leaf_count <= level_width(height - 1, branching_factor));
    debug_assert!(height == 1 || leaf_count > level_width(height - 2, branching_factor));
    let tree = vec![default; node_count(height, branching_factor, leaf_count)];
    Self { tree, height, branching_factor, leaf_count }
  }
}

impl<T> HeapTree<T> {
  /// Returns the number of nodes at `depth`.
  #[inline]
  pub fn level_width(&self, depth: usize) -> usize {
    debug_assert!(depth < self.height);
    level_width(depth, self.branching_factor).min(self.leaf_count)
  }

  /// Returns the number of leaves in the tree.
  #[inline]
  pub fn leaf_count(&self) -> usize {
    self.leaf_count
  }

  /// Get the index of a node at a certain depth and path
  #[inline]
  pub fn get_index(&self, depth: usize, path: PositionType) -> usize {
    debug_assert!(depth < self.height);
    debug_assert!((path as usize) < self.leaf_count);
    if self.branching_factor == 2 {
      let level_offset = (1 << depth) - 1;
      let mask = level_offset;
      if self.level_width(depth) == level_width(depth, self.branching_factor) {
        return level_offset + (path as usize & mask);
      }
    }

    if self.branching_factor == 2 {
      let level_offset = node_count(depth, self.branching_factor, self.leaf_count);
      let mask = (1 << depth) - 1;
      return level_offset + (path as usize & mask);
    }

    if self.branching_factor.is_power_of_two() {
      let shift = depth * self.branching_factor.trailing_zeros() as usize;
      let mask = (1 << shift) - 1;
      if self.level_width(depth) == level_width(depth, self.branching_factor) {
        let level_offset = mask / (self.branching_factor - 1);
        return level_offset + (path as usize & mask);
      }

      let level_offset = node_count(depth, self.branching_factor, self.leaf_count);
      return level_offset + (path as usize & mask);
    }

    let level_width = self.level_width(depth);
    node_count(depth, self.branching_factor, self.leaf_count) + (path as usize % level_width)
  }

  /// Get a node of a certain path at a certain depth
  /// Reveals depth and path
  #[inline]
  pub fn get_path_at_depth(&self, depth: usize, path: PositionType) -> &T {
    let index = self.get_index(depth, path);
    // UNDONE(git-10): Make sure this doesn't have bounds checking and is safe
    &self.tree[index]
  }

  /// Get a node of a certain path at a certain depth
  /// Reveals depth and path
  #[inline]
  pub fn get_path_at_depth_mut(&mut self, depth: usize, path: PositionType) -> &mut T {
    let index = self.get_index(depth, path);

    // UNDONE(git-10): Make sure this doesn't have bounds checking and is safe
    &mut self.tree[index]
  }

  /// Given a path and a node at certain depth, return the other child of that node's parent.
  pub fn get_sibling(&self, depth: usize, path: PositionType) -> &T {
    debug_assert!(self.branching_factor == 2);
    let new_path = path ^ (1 << (depth - 1));
    self.get_path_at_depth(depth, new_path)
  }

  /// Given a path and a node at certain depth, return a selected child of that node's parent.
  pub fn get_sibling_at(&self, depth: usize, path: PositionType, sibling: usize) -> &T {
    debug_assert!(depth > 0);
    debug_assert!(depth < self.height);
    debug_assert!(sibling < self.branching_factor);

    let new_path = if self.branching_factor.is_power_of_two() {
      let digit_shift = (depth - 1) * self.branching_factor.trailing_zeros() as usize;
      let digit_mask = (self.branching_factor - 1) << digit_shift;
      (path as usize & !digit_mask) | (sibling << digit_shift)
    } else {
      let digit_scale = self.branching_factor.pow((depth - 1) as u32);
      let current_digit = (path as usize / digit_scale) % self.branching_factor;
      path as usize + (sibling * digit_scale) - (current_digit * digit_scale)
    };
    self.get_path_at_depth(depth, new_path as PositionType)
  }
}

#[inline]
fn node_count(height: usize, branching_factor: usize, leaf_count: usize) -> usize {
  if height == 0 {
    0
  } else if leaf_count >= level_width(height - 1, branching_factor) && branching_factor == 2 {
    (1 << height) - 1
  } else if leaf_count >= level_width(height - 1, branching_factor)
    && branching_factor.is_power_of_two()
  {
    let shift = height * branching_factor.trailing_zeros() as usize;
    ((1 << shift) - 1) / (branching_factor - 1)
  } else {
    (0..height).map(|depth| level_width(depth, branching_factor).min(leaf_count)).sum()
  }
}

#[inline]
fn level_width(depth: usize, branching_factor: usize) -> usize {
  if branching_factor == 2 {
    1 << depth
  } else if branching_factor.is_power_of_two() {
    let shift = depth * branching_factor.trailing_zeros() as usize;
    1 << shift
  } else {
    branching_factor.pow(depth as u32)
  }
}

#[cfg(test)]
mod tests {
  use super::HeapTree;
  use crate::prelude::PositionType;

  fn print_depth_pos_index(height: usize, depth: usize, path: PositionType) {
    debug_assert!(depth < height);
    let level_offset = (1 << depth) - 1;
    let mask = level_offset as PositionType;
    let _ret = level_offset + (path & mask) as usize;
  }
  #[test]
  fn print_heap_tree_info() {
    for depth in 0..3 {
      for path in 0..4 {
        print_depth_pos_index(3, depth, path);
      }
    }
  }

  #[test]
  fn binary_tree_keeps_existing_indices() {
    let tree = HeapTree::<usize>::new(3);

    assert_eq!(tree.branching_factor, 2);
    assert_eq!(tree.tree.len(), 7);
    assert_eq!(tree.leaf_count(), 4);

    assert_eq!(tree.get_index(0, 0), 0);
    assert_eq!(tree.get_index(1, 0), 1);
    assert_eq!(tree.get_index(1, 1), 2);
    assert_eq!(tree.get_index(2, 0), 3);
    assert_eq!(tree.get_index(2, 2), 5);
    assert_eq!(tree.get_index(2, 1), 4);
    assert_eq!(tree.get_index(2, 3), 6);
  }

  #[test]
  fn ternary_tree_uses_branching_factor_for_size_and_indices() {
    let tree = HeapTree::<usize>::new_with_branching_factor(3, 3);

    assert_eq!(tree.branching_factor, 3);
    assert_eq!(tree.tree.len(), 13);
    assert_eq!(tree.leaf_count(), 9);

    assert_eq!(tree.get_index(0, 8), 0);
    assert_eq!(tree.get_index(1, 0), 1);
    assert_eq!(tree.get_index(1, 1), 2);
    assert_eq!(tree.get_index(1, 2), 3);
    assert_eq!(tree.get_index(2, 0), 4);
    assert_eq!(tree.get_index(2, 5), 9);
    assert_eq!(tree.get_index(2, 8), 12);
  }

  #[test]
  fn quaternary_tree_can_select_siblings_by_digit() {
    let mut tree = HeapTree::<usize>::new_with_branching_factor(3, 4);
    for (i, item) in tree.tree.iter_mut().enumerate() {
      *item = i;
    }

    assert_eq!(tree.branching_factor, 4);
    assert_eq!(tree.tree.len(), 21);
    assert_eq!(tree.leaf_count(), 16);

    assert_eq!(*tree.get_sibling_at(2, 14, 0), tree.get_index(2, 2));
    assert_eq!(*tree.get_sibling_at(2, 14, 1), tree.get_index(2, 6));
    assert_eq!(*tree.get_sibling_at(2, 14, 2), tree.get_index(2, 10));
    assert_eq!(*tree.get_sibling_at(2, 14, 3), tree.get_index(2, 14));
  }

  #[test]
  fn quaternary_tree_does_not_require_full_bottom_layer() {
    let tree = HeapTree::<usize>::new_with_leaf_count_and_value(3, 4, 8, 0);

    assert_eq!(tree.branching_factor, 4);
    assert_eq!(tree.height, 3);
    assert_eq!(tree.leaf_count(), 8);
    assert_eq!(tree.level_width(0), 1);
    assert_eq!(tree.level_width(1), 4);
    assert_eq!(tree.level_width(2), 8);
    assert_eq!(tree.tree.len(), 13);

    assert_eq!(tree.get_index(0, 7), 0);
    assert_eq!(tree.get_index(1, 7), 4);
    assert_eq!(tree.get_index(2, 7), 12);
  }
}
