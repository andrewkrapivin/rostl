use bytemuck::{Pod, Zeroable};

use rostl_oram::{
  linear_oram::{oblivious_write_index, oblivious_read_index, oblivious_memcpy, oblivious_read_update_index},
  prelude::{PositionType, DUMMY_POS}
};
use rostl_primitives::{
  cmov_body, cxchg_body,
  ooption::OOption,
  traits::{_Cmovbase, Cmov},
};

pub(crate) const SUCCESSOR: usize = 2;
pub(crate) const EXACT_MATCH: usize = 1;
pub(crate) const PREDECESSOR: usize = 0;


#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// Key and value returned by a B+ tree lookup.
pub struct KeyValuePair<T, V>
where
  T: Cmov + Pod,
  V: Cmov + Pod,
{
  /// The matched key.
  pub key: T,
  /// The value associated with the matched key.
  pub value: V,
}

unsafe impl<T, V> Zeroable for KeyValuePair<T, V>
where
  T: Cmov + Pod,
  V: Cmov + Pod,
{}

unsafe impl<T, V> Pod for KeyValuePair<T, V>
where
  T: Cmov + Pod,
  V: Cmov + Pod,
{}

/// Automatically implement the Cmov trait for a generic type with type and const parameters.
/// usage:
/// ```ignore
/// impl_cmov_for_generic_pod_with_const!(impl [T, const N: usize] for Type<T, N> [where T: Cmov + Pod])
/// ```
/// UNDONE: move to asm.rs?
macro_rules! impl_cmov_for_generic_pod_with_const {
  (impl [$($impl_generics:tt)*] for $ty:ty where [$($where_clause:tt)*]) => {
    impl<$($impl_generics)*> Cmov for $ty
    where
      $ty: Pod,
      $($where_clause)*
    {
      fn cmov(&mut self, other: &Self, choice: bool) {
        cmov_body!(self, other, choice);
      }

      fn cxchg(&mut self, other: &mut Self, choice: bool) {
        cxchg_body!(self, other, choice);
      }
    }
  };
  (impl [$($impl_generics:tt)*] for $ty:ty) => {
    impl<$($impl_generics)*> Cmov for $ty
    where
      $ty: Pod,
    {
      fn cmov(&mut self, other: &Self, choice: bool) {
        cmov_body!(self, other, choice);
      }

      fn cxchg(&mut self, other: &mut Self, choice: bool) {
        cxchg_body!(self, other, choice);
      }
    }
  };
}

/// UNDONE: Figure out if Ord needs to be made constant time by using the subtle crate
/// UNDONE: maybe merge into the Array datatype? Kind of reimplemented that, although this is perhaps a little more specialized
/// UNDONE: make it actually return the key not just the index, we want the key so that predecessor/successor searches can be supported!
/// Simple oblivious array, with option to be ordered if you give it a key that can be ordered.
/// Can access by index, or can access by searching for key.
/// For now its static: define size at compile time and when create it give it all the keys.
/// In the future will have dynamic insert operation, which will simply shift all keys after the given key to the right.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ObliviousArray<T, const N: usize>
where
  T: Cmov + Pod,
{
  /// actual data
  data: [T; N]
}

/// UNDONE: figure out a safe implementation for Pod and Zeroable?
unsafe impl<T, const N: usize> Zeroable for ObliviousArray<T, N>
where
  T: Cmov + Pod,
{}

unsafe impl<T, const N: usize> Pod for ObliviousArray<T, N>
where
  T: Cmov + Pod,
{}

impl_cmov_for_generic_pod_with_const!(
  impl [T, V] for KeyValuePair<T, V> where [T: Cmov + Pod, V: Cmov + Pod]
);

impl_cmov_for_generic_pod_with_const!(
  impl [T, const N: usize] for ObliviousArray<T, N> where [T: Cmov + Pod]
);

/// Performs an oblivious read and possibly update at the specified index.
///
/// # Arguments
///
/// * `data` - A mutable slice of data.
/// * `index` - The index to read and update, if out of bounds, the function will not modify `ret` or `data`.
/// * `ret` - A mutable reference to store the read value.
/// * `value` - The (optional) value to write at the specified index. If no write is desired (that is, if this is actually a read), then pass in none
///
/// # Oblivious
/// * Memory access pattern depends only on `data.len()`
#[inline]
pub fn oblivious_read_maybe_update_index<T: Cmov + Pod>(data: &mut [T], index: usize, ret: &mut T, value: &OOption<T>) {
  for (i, item) in data.iter_mut().enumerate() {
    let choice = i == index;
    ret.cmov(item, choice);
    value.cmov_to_other_if_some(item, choice);
  }
}

impl<T,const N: usize> ObliviousArray<T,N>
where 
  T: Cmov + Pod
{
  ///initialize with data. If the key supports comparisons, it should be already sorted from smallest to largest.
  pub fn new(data: [T;N]) -> Self {
    Self { data }
  }
  ///reinitialize with new data. If the key supports comparisons, it should be already sorted from smallest to largest.
  pub fn renew(&mut self, new_data: &[T]) {
    oblivious_memcpy(&mut self.data, new_data, 0);
  }

  ///linear scan the entire array, move the element out when index matches
  pub fn read(&self, index: usize, ret: &mut T) {
    oblivious_read_index(& self.data, index, ret)
  }
  ///linear scan the entire array, write to the index if the index matches
  pub fn write(&mut self, index: usize, value: T) {
    oblivious_write_index(&mut self.data, index, value)
  }
  ///linear scan the entire array, read the element out when index matches, write to the index if the index matches
  pub fn read_update(&mut self, index: usize, value: T, ret: &mut T) {
    oblivious_read_update_index(&mut self.data, index, ret, value);
  }
  ///linear scan the entire array, read the element out when index matches, write to the index if the index matches
  pub fn read_maybe_update(&mut self, index: usize, value: &OOption<T>, ret: &mut T) {
    oblivious_read_maybe_update_index(&mut self.data, index, ret, value);
  }
}

impl<T,const N: usize> ObliviousArray<T,N>
where 
  T: Cmov + Pod + PartialEq
{ 
  /// Returns the index of the first copy of K in data, N if there are none.
  /// Searches a prefix of the array as specified by end index.
  pub fn search_exact_prefix(&self, key: T, end_index: usize) -> usize {
    let mut i = N;
    for j in 0..self.data.len() {
      let equal: bool = self.data[j] == key;
      let keep_checking = j < end_index;
      i.cmov(&j, equal & keep_checking);
    }
    i
  }
  /// Returns the index of the first copy of K in data, N if there are none.
  /// Searches whole array
  pub fn search_exact(&self, key: T) -> usize {
    self.search_exact_prefix(key, N)
  }
}

impl<T,const N: usize> ObliviousArray<T,N>
where 
  T: Cmov + Pod + Ord
{
  /// Obliviously search either the successor of a given key in a prefix of the array
  /// If there is a key in node equal to target key, it is *not* considered as the successor.
  /// Also calculates if there is an exact match.
  /// Pred would always be succ index minus one (can be negative).
  /// This is used for branching, and branch taken would always be equal to the succ index, UNLESS you have exact match set to true and you are searching for the predecessor. In that case you subtract one
  pub fn search_succ_prefix(&self, key: T, index: &mut usize, exact_match: &mut bool, end_index: usize) {
    debug_assert!(self.data[..end_index].windows(2).all(|window| window[0] <= window[1]));
    *index = 0;
    *exact_match = false;
    for i in 0..self.data.len() {
      let more = key >= self.data[i];
      let keep_looking = i < end_index;
      *index += (more & keep_looking) as usize;
      *exact_match |= (self.data[i] == key) & keep_looking;
    }
    debug_assert!(!((*index == 0) & (*exact_match)));
  }

  /// search over the entire array.
  pub fn search_succ(&self, key: T, index: &mut usize, exact_match: &mut bool) {
    self.search_succ_prefix(key, index, exact_match, N);
  }
}

impl<T, const N: usize> ObliviousArray<T,N>
where
  T: Cmov + Pod + std::fmt::Debug
{
  #[cfg(test)]
  pub(crate) fn print_for_debug(&self) {
    for i in 0..self.data.len() {
      print!("{:?}, ", self.data[i]);
    }
    println!();
  }
}






/// A B+ tree node. BM1 is B-1, since there is some issue with const expressions in Rust
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BpTreeNode<T, const BM1: usize, const B: usize>
where 
  T: Cmov + Pod + Ord
{
  keys: ObliviousArray<T, BM1>,
  ptrs: ObliviousArray<PositionType, B>,
  /// refers to the number of keys currently actually in the node
  len: usize
}

/// UNDONE: figure out a safe implementation for Pod and Zeroable?
unsafe impl<T, const BM1: usize, const B: usize> Zeroable for BpTreeNode<T, BM1, B>
where
  T: Cmov + Pod + Ord,
{}

unsafe impl<T, const BM1: usize, const B: usize> Pod for BpTreeNode<T, BM1, B>
where
  T: Cmov + Pod + Ord,
{}

impl_cmov_for_generic_pod_with_const!(
  impl [T, const BM1: usize, const B: usize] for BpTreeNode<T, BM1, B> where [T: Cmov + Pod + Ord]
);

impl<T, const BM1: usize, const B: usize> Default for BpTreeNode<T, BM1, B>
where 
  T: Cmov + Pod + Ord
{
  fn default() -> Self {
    Self {
      keys: ObliviousArray { data: [T::zeroed(); BM1] },
      ptrs: ObliviousArray { data: [DUMMY_POS; B] },
      len: 0
    }
  }
}

impl<T, const BM1: usize, const B: usize> BpTreeNode<T, BM1, B>
where 
  T: Cmov + Pod + Ord
{
  /// UNDONE: make this implementation less bad.
  /// initialize with data. If the key supports comparisons, it should be already sorted from smallest to largest.
  /// len refers to the number of keys. 
  pub fn new(keys: &[T], ptrs: &[PositionType], len: usize) -> Self {
    debug_assert!(keys.windows(2).all(|window| window[0] <= window[1]));
    debug_assert!((keys.len() >= len) & (ptrs.len() >= len + 1));
    
    let mut new_node = Self { 
      keys: ObliviousArray{data: [T::zeroed(); BM1]},
      ptrs: ObliviousArray { data: [DUMMY_POS; B]},
      len: len
    };

    new_node.keys.renew(keys);
    new_node.ptrs.renew(ptrs);

    new_node
  }
  ///reinitialize with new data. If the key supports comparisons, it should be already sorted from smallest to largest.
  pub fn renew(&mut self, keys: &[T;BM1], ptrs: &[PositionType; B], len: usize) {
    debug_assert!(keys.windows(2).all(|window| window[0] <= window[1]));
    self.keys.renew(keys);
    self.ptrs.renew(ptrs);
    self.len = len;
  }
  /// Returns the old pointer for where to go down the tree and sets a new ptr
  /// Additionally returns index of this returned pointer in the node.
  /// search_type: whether searching for SUCCESSOR, PREDECESSOR, or EXACT_MATCH
  /// For our purposes, in a BpTree node, searching for SUCCESSOR and EXACT_MATCH are exactly the same.
  pub fn search_update_index(&mut self, key: T, new_ptr: PositionType, search_type: usize) -> (usize, PositionType) {
    let mut index = 0;
    let mut exact_match = false;
    self.keys.search_succ_prefix(key, &mut index, &mut exact_match, self.len);
    let subtract_index: bool = (search_type == PREDECESSOR) & exact_match;
    index -= subtract_index as usize;
    let mut ret: PositionType = DUMMY_POS;
    self.ptrs.read_update(index, new_ptr, &mut ret);
    (index, ret)
  }

  pub fn search_update_key_index(
    &mut self,
    key: T,
    new_ptr: PositionType,
    search_type: usize,
  ) -> (OOption<T>, usize, PositionType) {
    let mut index = 0;
    let mut exact_match = false;
    self.keys.search_succ_prefix(key, &mut index, &mut exact_match, self.len);

    let subtract_index: bool = (search_type == PREDECESSOR) & exact_match;
    let ptr_index = index - subtract_index as usize;
    let mut ret: PositionType = DUMMY_POS;
    self.ptrs.read_update(ptr_index, new_ptr, &mut ret);

    let decrement = (exact_match & ((search_type == EXACT_MATCH) | (search_type == PREDECESSOR))) as usize
      + (search_type == PREDECESSOR) as usize;
    let search_succeeded =
      (index >= decrement) & (index < self.len + decrement) & (exact_match | (search_type != EXACT_MATCH));
    index -= decrement * search_succeeded as usize;
    index.cmov(&self.len, !search_succeeded);

    let mut ret_key: T = T::zeroed();
    self.keys.read(index, &mut ret_key);
    (OOption::new(ret_key, search_succeeded), ptr_index, ret)
  }
}

/// A B tree node. Each key has an inline value and each gap has a child pointer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BTreeNode<T, V, const BM1: usize, const B: usize>
where
  T: Cmov + Pod + Ord,
  V: Cmov + Pod,
{
  keys: ObliviousArray<T, BM1>,
  values: ObliviousArray<V, BM1>,
  ptrs: ObliviousArray<PositionType, B>,
  /// refers to the number of keys currently actually in the node
  len: usize,
}

/// UNDONE: figure out a safe implementation for Pod and Zeroable?
unsafe impl<T, V, const BM1: usize, const B: usize> Zeroable for BTreeNode<T, V, BM1, B>
where
  T: Cmov + Pod + Ord,
  V: Cmov + Pod,
{}

unsafe impl<T, V, const BM1: usize, const B: usize> Pod for BTreeNode<T, V, BM1, B>
where
  T: Cmov + Pod + Ord,
  V: Cmov + Pod,
{}

impl_cmov_for_generic_pod_with_const!(
  impl [T, V, const BM1: usize, const B: usize] for BTreeNode<T, V, BM1, B> where [T: Cmov + Pod + Ord, V: Cmov + Pod]
);

impl<T, V, const BM1: usize, const B: usize> Default for BTreeNode<T, V, BM1, B>
where
  T: Cmov + Pod + Ord,
  V: Cmov + Pod,
{
  fn default() -> Self {
    Self {
      keys: ObliviousArray { data: [T::zeroed(); BM1] },
      values: ObliviousArray { data: [V::zeroed(); BM1] },
      ptrs: ObliviousArray { data: [DUMMY_POS; B] },
      len: 0
    }
  }
}

impl<T, V, const BM1: usize, const B: usize> BTreeNode<T, V, BM1, B>
where
  T: Cmov + Pod + Ord,
  V: Cmov + Pod,
{
  /// initialize with data. If the key supports comparisons, it should be already sorted from smallest to largest.
  /// len refers to the number of keys.
  pub fn new(keys: &[T], values: &[V], ptrs: &[PositionType], len: usize) -> Self {
    debug_assert!(keys.windows(2).all(|window| window[0] <= window[1]));
    debug_assert!((keys.len() >= len) & (values.len() >= len) & (ptrs.len() >= len + 1));

    let mut new_node = Self {
      keys: ObliviousArray{data: [T::zeroed(); BM1]},
      values: ObliviousArray { data: [V::zeroed(); BM1]},
      ptrs: ObliviousArray { data: [DUMMY_POS; B]},
      len: len
    };

    new_node.keys.renew(keys);
    new_node.values.renew(values);
    new_node.ptrs.renew(ptrs);

    new_node
  }

  ///reinitialize with new data. If the key supports comparisons, it should be already sorted from smallest to largest.
  pub fn renew(&mut self, keys: &[T;BM1], values: &[V; BM1], ptrs: &[PositionType; B], len: usize) {
    debug_assert!(keys.windows(2).all(|window| window[0] <= window[1]));
    self.keys.renew(keys);
    self.values.renew(values);
    self.ptrs.renew(ptrs);
    self.len = len;
  }

  /// Returns the matched key-value pair, child index, and old child pointer for the requested search type.
  pub fn search_update_pair_index(
    &mut self,
    key: T,
    new_ptr: PositionType,
    search_type: usize,
  ) -> (OOption<KeyValuePair<T, V>>, usize, PositionType) {
    let mut index = 0;
    let mut exact_match = false;
    self.keys.search_succ_prefix(key, &mut index, &mut exact_match, self.len);

    let decrement = (exact_match & ((search_type == EXACT_MATCH) | (search_type == PREDECESSOR))) as usize
      + (search_type == PREDECESSOR) as usize;
    let search_succeeded =
      (index >= decrement) & (index < self.len + decrement) & (exact_match | (search_type != EXACT_MATCH));
    let ptr_index = index - ((search_type == PREDECESSOR) & exact_match) as usize;
    let mut key_value_index = index - decrement * search_succeeded as usize;
    key_value_index.cmov(&self.len, !search_succeeded);

    let mut old_ptr: PositionType = DUMMY_POS;
    self.ptrs.read_update(ptr_index, new_ptr, &mut old_ptr);

    let mut ret_key: T = T::zeroed();
    let mut ret_value: V = V::zeroed();
    self.keys.read(key_value_index, &mut ret_key);
    self.values.read(key_value_index, &mut ret_value);
    (OOption::new(KeyValuePair { key: ret_key, value: ret_value }, search_succeeded), ptr_index, old_ptr)
  }
}





/// A value node. For each key, there is an exact value.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BpValueNode<T, V, const B: usize>
where 
  T: Cmov + Pod + Ord,
  V: Cmov + Pod
{
  keys: ObliviousArray<T, B>,
  values: ObliviousArray<V, B>,
  /// refers to the number of keys currently actually in the node
  len: usize
}

/// UNDONE: figure out a safe implementation for Pod and Zeroable?
unsafe impl<T, V, const B: usize> Zeroable for BpValueNode<T, V, B>
where
  T: Cmov + Pod + Ord,
  V: Cmov + Pod,
{}

unsafe impl<T, V, const B: usize> Pod for BpValueNode<T, V, B>
where
  T: Cmov + Pod + Ord,
  V: Cmov + Pod,
{}

impl_cmov_for_generic_pod_with_const!(
  impl [T, V, const B: usize] for BpValueNode<T, V, B> where [T: Cmov + Pod + Ord, V: Cmov + Pod]
);

impl<T, V, const B: usize> Default for BpValueNode<T, V, B>
where 
  T: Cmov + Pod + Ord,
  V: Cmov + Pod
{
  fn default() -> Self {
    Self {
      keys: ObliviousArray { data: [T::zeroed(); B] },
      values: ObliviousArray { data: [V::zeroed(); B] },
      len: 0
    }
  }
}

impl<T, V, const B: usize> BpValueNode<T, V, B>
where 
  T: Cmov + Pod + Ord,
  V: Cmov + Pod
{
  /// UNDONE: make this implementation less bad.
  /// initialize with data. If the key supports comparisons, it should be already sorted from smallest to largest.
  /// len refers to the number of keys. 
  pub fn new(keys: &[T], values: &[V], len: usize) -> Self {
    debug_assert!(keys.windows(2).all(|window| window[0] <= window[1]));
    debug_assert!((keys.len() >= len) & (values.len() >= len));
    
    let mut new_node = Self { 
      keys: ObliviousArray{data: [T::zeroed(); B]},
      values: ObliviousArray { data: [V::zeroed(); B]},
      len: len
    };

    new_node.keys.renew(keys);
    new_node.values.renew(values);

    new_node
  }
  ///reinitialize with new data. If the key supports comparisons, it should be already sorted from smallest to largest.
  pub fn renew(&mut self, keys: &[T;B], values: &[V; B], len: usize) {
    debug_assert!(keys.windows(2).all(|window| window[0] <= window[1]));
    self.keys.renew(keys);
    self.values.renew(values);
    self.len = len;
  }
  /// Returns the old pointer for where to go down the tree and sets a new ptr
  /// May return none, if no exact match found or there is no successor or no predecessor, depending on search type
  /// If returns none then it does not use the new value
  pub fn search_update(&mut self, key: T, new_value: V, search_type: usize) -> OOption<V> {
    let mut index = 0;
    let mut exact_match = false;
    self.keys.search_succ_prefix(key, &mut index, &mut exact_match, self.len);
    // if there is an exact match, it is the key before the index given by successor, so subtract one if looking for predecessor or exact match
    let decrement = (exact_match & ((search_type == EXACT_MATCH) | (search_type == PREDECESSOR))) as usize
      + (search_type == PREDECESSOR) as usize;
    let search_succeeded = (index >= decrement) & (index < self.len + decrement) & (exact_match | (search_type != EXACT_MATCH));
    index -= decrement * search_succeeded as usize;
    index.cmov(&self.len, !search_succeeded);
    let mut ret: V = V::zeroed();
    self.values.read_update(index, new_value, &mut ret);
    OOption::new(ret, search_succeeded)
  }

  /// Returns the old pointer for where to go down the tree and sets a new ptr
  pub fn search(&mut self, key: T, search_type: usize) -> OOption<V> {
    let mut index = 0;
    let mut exact_match = false;
    self.keys.search_succ_prefix(key, &mut index, &mut exact_match, self.len);
    let decrement = (exact_match & ((search_type == EXACT_MATCH) | (search_type == PREDECESSOR))) as usize
      + (search_type == PREDECESSOR) as usize;
    let search_succeeded = (index >= decrement) & (index < self.len + decrement) & (exact_match | (search_type != EXACT_MATCH));
    index -= decrement * search_succeeded as usize;
    index.cmov(&self.len, !search_succeeded);
    let mut ret: V = V::zeroed();
    self.values.read(index, &mut ret);
    OOption::new(ret, search_succeeded)
  }

  /// UNDONE: Somewhat inneficient implementation as it scans over the keys twice.
  /// Returns the matched key-value pair for the requested search type.
  pub fn search_pair(&mut self, key: T, search_type: usize) -> OOption<KeyValuePair<T, V>> {
    let mut index = 0;
    let mut exact_match = false;
    self.keys.search_succ_prefix(key, &mut index, &mut exact_match, self.len);
    let decrement = (exact_match & ((search_type == EXACT_MATCH) | (search_type == PREDECESSOR))) as usize
      + (search_type == PREDECESSOR) as usize;
    let search_succeeded =
      (index >= decrement) & (index < self.len + decrement) & (exact_match | (search_type != EXACT_MATCH));
    index -= decrement * search_succeeded as usize;
    index.cmov(&self.len, !search_succeeded);
    let mut ret_key: T = T::zeroed();
    let mut ret_value: V = V::zeroed();
    self.keys.read(index, &mut ret_key);
    self.values.read(index, &mut ret_value);
    OOption::new(KeyValuePair { key: ret_key, value: ret_value }, search_succeeded)
  }

  /// Searches for key using requested search type
  pub fn search_key(&mut self, key: T, search_type: usize) -> OOption<T> {
    let mut index = 0;
    let mut exact_match = false;
    self.keys.search_succ_prefix(key, &mut index, &mut exact_match, self.len);
    let decrement = (exact_match & ((search_type == EXACT_MATCH) | (search_type == PREDECESSOR))) as usize
      + (search_type == PREDECESSOR) as usize;
    let search_succeeded =
      (index >= decrement) & (index < self.len + decrement) & (exact_match | (search_type != EXACT_MATCH));
    index -= decrement * search_succeeded as usize;
    index.cmov(&self.len, !search_succeeded);
    let mut ret_key: T = T::zeroed();
    self.keys.read(index, &mut ret_key);
    OOption::new(ret_key, search_succeeded)
  }
}


#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn b_tree_node_has_values_and_ptrs() {
    let keys = [10_u64, 20, 30];
    let values = [100_u64, 200, 300];
    let ptrs = [1, 2, 3, 4];
    let mut node = BTreeNode::<u64, u64, 3, 4>::new(&keys, &values, &ptrs, 3);

    let (exact, index, ptr) = node.search_update_pair_index(20, 8, EXACT_MATCH);
    assert!(exact.is_some());
    let exact = exact.unwrap();
    assert_eq!(exact.key, 20);
    assert_eq!(exact.value, 200);
    assert_eq!(index, 2);
    assert_eq!(ptr, 3);

    let (successor, index, ptr) = node.search_update_pair_index(25, 9, SUCCESSOR);
    assert!(successor.is_some());
    let successor = successor.unwrap();
    assert_eq!(successor.key, 30);
    assert_eq!(successor.value, 300);
    assert_eq!(index, 2);
    assert_eq!(ptr, 8);

    let (predecessor, index, ptr) = node.search_update_pair_index(25, 10, PREDECESSOR);
    assert!(predecessor.is_some());
    let predecessor = predecessor.unwrap();
    assert_eq!(predecessor.key, 20);
    assert_eq!(predecessor.value, 200);
    assert_eq!(index, 2);
    assert_eq!(ptr, 9);
  }

  #[test]
  fn value_node_search_pair_supports_successor_and_predecessor() {
    let keys = [10_u64, 20, 30, 40];
    let values = [100_u64, 200, 300, 400];
    let mut node = BpValueNode::<u64, u64, 4>::new(&keys, &values, 4);

    let successor = node.search_pair(25, SUCCESSOR);
    assert!(successor.is_some());
    let successor = successor.unwrap();
    assert_eq!(successor.key, 30);
    assert_eq!(successor.value, 300);

    let predecessor = node.search_pair(25, PREDECESSOR);
    assert!(predecessor.is_some());
    let predecessor = predecessor.unwrap();
    assert_eq!(predecessor.key, 20);
    assert_eq!(predecessor.value, 200);

    let exact_successor = node.search_pair(30, SUCCESSOR);
    assert!(exact_successor.is_some());
    let exact_successor = exact_successor.unwrap();
    assert_eq!(exact_successor.key, 40);
    assert_eq!(exact_successor.value, 400);

    let exact_predecessor = node.search_pair(30, PREDECESSOR);
    assert!(exact_predecessor.is_some());
    let exact_predecessor = exact_predecessor.unwrap();
    assert_eq!(exact_predecessor.key, 20);
    assert_eq!(exact_predecessor.value, 200);

    let below_min_successor = node.search_pair(5, SUCCESSOR);
    assert!(below_min_successor.is_some());
    let below_min_successor = below_min_successor.unwrap();
    assert_eq!(below_min_successor.key, 10);
    assert_eq!(below_min_successor.value, 100);

    let above_max_predecessor = node.search_pair(45, PREDECESSOR);
    assert!(above_max_predecessor.is_some());
    let above_max_predecessor = above_max_predecessor.unwrap();
    assert_eq!(above_max_predecessor.key, 40);
    assert_eq!(above_max_predecessor.value, 400);

    let exact_min = node.search_pair(10, EXACT_MATCH);
    assert!(exact_min.is_some());
    let exact_min = exact_min.unwrap();
    assert_eq!(exact_min.key, 10);
    assert_eq!(exact_min.value, 100);

    let exact_max = node.search_pair(40, EXACT_MATCH);
    assert!(exact_max.is_some());
    let exact_max = exact_max.unwrap();
    assert_eq!(exact_max.key, 40);
    assert_eq!(exact_max.value, 400);

    assert!(!node.search_pair(40, SUCCESSOR).is_some());
    assert!(!node.search_pair(10, PREDECESSOR).is_some());
    assert!(!node.search_pair(25, EXACT_MATCH).is_some());
  }
}
