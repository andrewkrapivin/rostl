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

const SUCCESSOR: usize = 2;
const EXACT_MATCH: usize = 1;
const PREDECESSOR: usize = 0;


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
  impl [T, const N: usize] for ObliviousArray<T, N> where [T: Cmov + Pod]
);

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
      let equal: bool = (self.data[i] == key);
      let keep_checking = (j < end_index);
      i.cmov(&j, equal && keep_checking);
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
    debug_assert!(self.data.windows(2).all(|window| window[0] <= window[1]));
    *index = 0;
    *exact_match = false;
    for i in 0..self.data.len() {
      let more = (key >= self.data[i]);
      let keep_looking = (i < end_index);
      *index += (more && keep_looking) as usize;
      *exact_match |= (self.data[i] == key && keep_looking);
    }
    debug_assert!(!(*index == 0 && *exact_match));
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
    debug_assert!(keys.len() >= len && ptrs.len() >= len + 1);
    
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
    let subtract_index: bool = (search_type == PREDECESSOR) && exact_match;
    index -= subtract_index as usize;
    let mut ret: PositionType = DUMMY_POS;
    self.ptrs.read_update(index, new_ptr, &mut ret);
    (index, ret)
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
    debug_assert!(keys.len() >= len && values.len() >= len);
    
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
    index -= (exact_match && (search_type == EXACT_MATCH || search_type == PREDECESSOR)) as usize;
    index -= (search_type == PREDECESSOR) as usize;
    let search_succeeded = (0 <= index) && (index < self.len) && (exact_match || (search_type != EXACT_MATCH));
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
    index -= (exact_match && (search_type == EXACT_MATCH || search_type == PREDECESSOR)) as usize;
    index -= (search_type == PREDECESSOR) as usize;
    let search_succeeded = (0 <= index) && (index < self.len) && (exact_match || (search_type != EXACT_MATCH));
    index.cmov(&self.len, !search_succeeded);
    let mut ret: V = V::zeroed();
    self.values.read(index, &mut ret);
    OOption::new(ret, search_succeeded)
  }
}


// UNDONE: add tests
