//! Fixed-lane ORAM with `Y` pooled slots in every lane minibucket.
//!
//! A physical bucket is laid out as `[Z][Y]`.  Slots within one minibucket
//! share a tree depth; `Y` is therefore a storage-layout width, not additional
//! logical tree levels.  Each lane executes one Circuit-ORAM `EvictOnceFast`
//! chain on every accessed path while all lanes share the stash.

use rostl_primitives::traits::Cmov;

use crate::{
  lane_oram::{
    read_and_remove_path32, Block32, KeyType, PosType, DUMMY_KEY, DUMMY_POS, EMPTY_BLOCK_DATA,
  },
  wide_heap_tree::WideHeapTree,
};

/// Default number of fixed lanes.
pub const DEFAULT_Z: usize = 3;
/// Default pooled slots per lane minibucket.
pub const DEFAULT_Y: usize = 2;
/// Default stash capacity.
pub const DEFAULT_S: usize = 64;
/// Default branching factor.
pub const DEFAULT_B: usize = 2;

/// A physical bucket containing `Z` fixed, `Y`-wide minibuckets.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FixedBucket<const Z: usize, const Y: usize>(pub [[Block32; Y]; Z]);

impl<const Z: usize, const Y: usize> Default for FixedBucket<Z, Y> {
  fn default() -> Self {
    Self([[Block32::default(); Y]; Z])
  }
}

/// Production cache-line lane ORAM with pooled minibuckets.
#[derive(Debug)]
pub struct LaneORAMFixed<
  const Z: usize = DEFAULT_Z,
  const Y: usize = DEFAULT_Y,
  const S: usize = DEFAULT_S,
  const B: usize = DEFAULT_B,
  const E: usize = 1,
> {
  /// Logical capacity rounded up to a power of `B`.
  pub max_n: usize,
  /// Tree height, including the root.
  pub h: usize,
  /// Wide tree with contiguous `[lane][minibucket slot]` buckets.
  pub tree: WideHeapTree<FixedBucket<Z, Y>>,
  /// Contiguous `[stash | path]` working array.
  pub stash_and_path: Vec<Block32>,
}

impl<const Z: usize, const Y: usize, const S: usize, const B: usize, const E: usize>
  LaneORAMFixed<Z, Y, S, B, E>
{
  /// Creates an empty fixed-lane ORAM.
  pub fn new(max_n: usize) -> Self {
    assert!(max_n > 1 && Z > 0 && Y > 0 && E > 0);
    assert!(B >= 2 && B.is_power_of_two());
    let mut rounded = 1usize;
    let mut h = 1usize;
    while rounded < max_n {
      rounded *= B;
      h += 1;
    }
    assert!(h <= 64, "Circuit target arrays support at most 64 levels");
    Self {
      max_n: rounded,
      h,
      tree: WideHeapTree::new(h, B),
      stash_and_path: vec![Block32::default(); S + h * Z * Y],
    }
  }

  #[inline]
  const fn width() -> usize {
    Z * Y
  }

  #[inline]
  fn path_index(level: usize, lane: usize, slot: usize) -> usize {
    S + level * Self::width() + lane * Y + slot
  }

  #[inline]
  fn legal_depth(&self, pos: PosType, path: PosType) -> i32 {
    let bits_per_digit = B.trailing_zeros();
    let used = ((self.h - 1) as u32) * bits_per_digit;
    let common_bits = (pos ^ path).wrapping_shl(PosType::BITS - used).leading_zeros();
    let mut depth = (common_bits / bits_per_digit) as i32;
    depth.cmov(&-1, pos == DUMMY_POS);
    depth
  }

  fn read_path(&mut self, path: PosType) {
    for level in 0..self.h {
      let node = self.tree.get_index(level, path);
      let bucket = &self.tree.tree[node];
      let mut out = S + level * Self::width();
      for lane in 0..Z {
        for slot in 0..Y {
          #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
          self.stash_and_path[out].store_avx512(bucket.0[lane][slot].load_avx512());
          #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
          {
            self.stash_and_path[out] = bucket.0[lane][slot];
          }
          out += 1;
        }
      }
    }
  }

  fn write_path(&mut self, path: PosType) {
    for level in 0..self.h {
      let node = self.tree.get_index(level, path);
      let bucket = &mut self.tree.tree[node];
      let mut input = S + level * Self::width();
      for lane in 0..Z {
        for slot in 0..Y {
          #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
          bucket.0[lane][slot].store_avx512(self.stash_and_path[input].load_avx512());
          #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
          {
            bucket.0[lane][slot] = self.stash_and_path[input];
          }
          input += 1;
        }
      }
    }
  }

  /// Runs Circuit ORAM Algorithms 2--4 for one fixed lane.
  fn evict_lane_fast(&mut self, path: PosType, lane: usize) {
    let mut deepest = [-1i32; 64];
    let mut deepest_idx = [0i32; 64];
    let mut target = [-1i32; 64];
    let mut has_empty = [false; 64];
    let mut src = -1i32;
    let mut dst = -1i32;

    // Stash plus this lane's root minibucket form Circuit level zero.
    for index in 0..S {
      let depth = self.legal_depth(self.stash_and_path[index].pos, path);
      let better = depth > dst;
      dst.cmov(&depth, better);
      deepest_idx[0].cmov(&(index as i32), better);
    }
    for slot in 0..Y {
      let index = Self::path_index(0, lane, slot);
      let depth = self.legal_depth(self.stash_and_path[index].pos, path);
      let better = depth > dst;
      dst.cmov(&depth, better);
      deepest_idx[0].cmov(&(index as i32), better);
    }
    src.cmov(&0, dst != -1);

    for level in 1..self.h {
      deepest[level].cmov(&src, dst >= level as i32);
      let mut bucket_depth = -1i32;
      for slot in 0..Y {
        let index = Self::path_index(level, lane, slot);
        let block = &self.stash_and_path[index];
        let empty = block.is_empty();
        has_empty[level].cmov(&true, empty);
        let depth = self.legal_depth(block.pos, path);
        let better = (!empty) & (depth > bucket_depth);
        bucket_depth.cmov(&depth, better);
        deepest_idx[level].cmov(&(index as i32), better);
      }
      let better = bucket_depth > dst;
      src.cmov(&(level as i32), better);
      dst.cmov(&bucket_depth, better);
    }

    src = -1;
    dst = -1;
    for level in (1..self.h).rev() {
      let is_src = level as i32 == src;
      target[level].cmov(&dst, is_src);
      src.cmov(&-1, is_src);
      dst.cmov(&-1, is_src);
      let change =
        (((dst == -1) & has_empty[level]) | (target[level] != -1)) & (deepest[level] != -1);
      src.cmov(&deepest[level], change);
      dst.cmov(&(level as i32), change);
    }
    target[0].cmov(&dst, src == 0);

    let mut hold = Block32::default();
    for index in 0..S {
      let take = deepest_idx[0] == index as i32 && target[0] != -1;
      hold.cmov(&self.stash_and_path[index], take);
      self.stash_and_path[index].pos.cmov(&DUMMY_POS, take);
    }
    for slot in 0..Y {
      let index = Self::path_index(0, lane, slot);
      let take = deepest_idx[0] == index as i32 && target[0] != -1;
      hold.cmov(&self.stash_and_path[index], take);
      self.stash_and_path[index].pos.cmov(&DUMMY_POS, take);
    }
    dst = target[0];

    for level in 1..self.h - 1 {
      let has_target = target[level] != -1;
      let place = level as i32 == dst && !has_target;
      for slot in 0..Y {
        let index = Self::path_index(level, lane, slot);
        let take = deepest_idx[level] == index as i32 && has_target;
        let write = self.stash_and_path[index].is_empty() && place;
        hold.cxchg(&mut self.stash_and_path[index], take | write);
      }
      dst.cmov(&target[level], has_target | place);
    }

    let level = self.h - 1;
    let place = level as i32 == dst;
    let mut written = false;
    for slot in 0..Y {
      let index = Self::path_index(level, lane, slot);
      let write = self.stash_and_path[index].is_empty() && place && !written;
      written |= write;
      self.stash_and_path[index].cmov(&hold, write);
    }
    debug_assert!(hold.is_empty() | written);
  }

  /// Returns the number of occupied stash slots. Intended for diagnostics.
  pub fn stash_len(&self) -> usize {
    self.stash_and_path[..S].iter().filter(|block| !block.is_empty()).count()
  }

  /// Updates or inserts a cache-line block and performs one eviction per lane.
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  pub fn update<T, F>(
    &mut self,
    pos: PosType,
    new_pos: PosType,
    key: KeyType,
    update_func: F,
  ) -> (bool, T)
  where
    F: FnOnce(&mut Block32) -> T,
  {
    debug_assert!((pos as usize) < self.max_n && (new_pos as usize) < self.max_n);
    debug_assert!(key != DUMMY_KEY);
    self.read_path(pos);
    let lookup = Block32 { pos: DUMMY_POS, key, data: EMPTY_BLOCK_DATA };
    let mut block = read_and_remove_path32(&lookup, &mut self.stash_and_path);
    let found = !block.is_empty();
    let result = update_func(&mut block);
    block.pos = new_pos;
    block.key = key;

    let mut written = false;
    for candidate in &mut self.stash_and_path[..S] {
      let write = candidate.is_empty() & !written;
      candidate.cmov(&block, write);
      written |= write;
    }
    assert!(written, "lane ORAM stash overflow");
    for _ in 0..E {
      for lane in 0..Z {
        self.evict_lane_fast(pos, lane);
      }
    }
    self.write_path(pos);
    (found, result)
  }
}

#[cfg(test)]
mod tests {
  use super::LaneORAMFixed;

  #[test]
  fn minibucket_layout_has_z_times_y_path_slots() {
    let oram = LaneORAMFixed::<3, 2, 20, 4>::new(16);
    assert_eq!(oram.h, 3);
    assert_eq!(oram.stash_and_path.len(), 20 + 3 * 3 * 2);
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[test]
  fn randomized_updates_preserve_payloads() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    const N: usize = 256;
    let mut oram = LaneORAMFixed::<3, 2, 64, 2>::new(N);
    let mut positions = [0u32; N];
    let mut values = [0u8; N];
    let mut initialized = [false; N];
    let mut rng = StdRng::seed_from_u64(7);
    for position in &mut positions {
      *position = rng.random_range(0..N) as u32;
    }
    for operation in 0..20_000 {
      let key = operation % N;
      let new_pos = rng.random_range(0..N) as u32;
      let new_value = operation as u8;
      let (found, old) = oram.update(positions[key], new_pos, key as u32, |block| {
        let old = block.data[0];
        block.data[0] = new_value;
        old
      });
      assert_eq!(found, initialized[key]);
      if found {
        assert_eq!(old, values[key]);
      }
      initialized[key] = true;
      values[key] = new_value;
      positions[key] = new_pos;
    }
    assert!(oram.stash_len() < 64, "unexpected final stash: {}", oram.stash_len());
  }

  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
  #[test]
  fn two_pass_updates_preserve_payloads() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    const N: usize = 64;
    let mut oram = LaneORAMFixed::<2, 3, 80, 2, 2>::new(N);
    let mut positions = [0u32; N];
    let mut values = [0u8; N];
    let mut initialized = [false; N];
    let mut rng = StdRng::seed_from_u64(11);
    for position in &mut positions {
      *position = rng.random_range(0..N) as u32;
    }
    for operation in 0..10_000 {
      let key = operation % N;
      let new_pos = rng.random_range(0..N) as u32;
      let new_value = operation as u8;
      let (found, old) = oram.update(positions[key], new_pos, key as u32, |block| {
        let old = block.data[0];
        block.data[0] = new_value;
        old
      });
      assert_eq!(found, initialized[key]);
      if found {
        assert_eq!(old, values[key]);
      }
      initialized[key] = true;
      values[key] = new_value;
      positions[key] = new_pos;
    }
  }
}
