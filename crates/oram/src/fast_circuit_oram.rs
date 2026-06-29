//! Specialized building blocks for a faster Circuit ORAM implementation.
//!
//! The first bucket type in this module stores two `(key, pos)` pairs and a
//! compact variable-width counter array in one cache line.

#![allow(clippy::needless_bitwise_bool)]

use bytemuck::{Pod, Zeroable};
use rostl_primitives::{
  cmov_body, cxchg_body, impl_cmov_for_pod,
  traits::{_Cmovbase, Cmov},
};
use static_assertions::const_assert_eq;

use crate::prelude::PositionType;

/// Number of `(key, pos)` pairs in a cache-line counter bucket.
pub const CACHELINE_COUNTER_BUCKET_BLOCKS: usize = 2;
/// Number of `u32` words used by the key/position slots.
pub const CACHELINE_COUNTER_BUCKET_BLOCK_WORDS: usize = 4;
/// Number of counters packed into the bucket.
pub const CACHELINE_COUNTER_BUCKET_COUNTERS: usize = 64;
/// Number of metadata bits used to describe counter widths.
pub const CACHELINE_COUNTER_BUCKET_METADATA_BITS: usize = 2 * 64 + 96;
/// Number of bits available for packed counter payloads.
pub const CACHELINE_COUNTER_BUCKET_COUNTER_BITS: usize = 64 + 96;

const RAW_WORDS: usize = 8;
const METADATA_WORDS: usize = 4;
const COUNTER_WORDS: usize = 3;
const RAW_BITS: usize = RAW_WORDS * u64::BITS as usize;
const BLOCK_BITS: usize = CACHELINE_COUNTER_BUCKET_BLOCK_WORDS * u32::BITS as usize;
const METADATA_BIT_OFFSET: usize = BLOCK_BITS;
const COUNTER_BIT_OFFSET: usize = METADATA_BIT_OFFSET + CACHELINE_COUNTER_BUCKET_METADATA_BITS;
const U32_MASK: u64 = u32::MAX as u64;
const METADATA_WORD_MASKS: [u64; METADATA_WORDS] = [u64::MAX, u64::MAX, u64::MAX, U32_MASK];
const COUNTER_WORD_MASKS: [u64; COUNTER_WORDS] = [u64::MAX, u64::MAX, U32_MASK];

const_assert_eq!(RAW_BITS, 512);
const_assert_eq!(COUNTER_BIT_OFFSET + CACHELINE_COUNTER_BUCKET_COUNTER_BITS, RAW_BITS);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CounterLocation {
  start: usize,
  width: usize,
  delimiter: usize,
  used_bits: usize,
}

/// A 64-byte, cache-line-aligned bucket with two `(key, pos)` pairs and 64 counters.
///
/// Layout:
/// * bits `0..128`: four `u32` block words, interpreted as
///   `[key0, pos0, key1, pos1]`;
/// * bits `128..352`: counter metadata with exactly 64 one bits;
/// * bits `352..512`: packed counter payload bits.
///
/// Counter `i` is assigned the zeros between the previous metadata one and the
/// `i`th metadata one. Zeros after the 64th one are spare counter capacity.
#[allow(non_camel_case_types)]
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Zeroable)]
pub struct Cacheline_Counter_Bucket {
  raw: [u64; RAW_WORDS],
}

unsafe impl Pod for Cacheline_Counter_Bucket {}
impl_cmov_for_pod!(Cacheline_Counter_Bucket);

impl Default for Cacheline_Counter_Bucket {
  fn default() -> Self {
    let mut raw = [0u64; RAW_WORDS];
    // Width zero for every counter: the first 64 metadata bits are delimiters
    // and the remaining 160 metadata zeros are spare capacity.
    raw[METADATA_BIT_OFFSET / u64::BITS as usize] = u64::MAX;
    Self { raw }
  }
}

const_assert_eq!(core::mem::size_of::<Cacheline_Counter_Bucket>(), 64);
const_assert_eq!(core::mem::align_of::<Cacheline_Counter_Bucket>(), 64);

impl Cacheline_Counter_Bucket {
  /// Creates an empty bucket with all counters set to zero.
  #[inline]
  pub fn new() -> Self {
    Self::default()
  }

  /// Returns the four block words as `[key0, pos0, key1, pos1]`.
  #[inline]
  pub const fn block_words(&self) -> [u32; CACHELINE_COUNTER_BUCKET_BLOCK_WORDS] {
    [self.raw[0] as u32, (self.raw[0] >> 32) as u32, self.raw[1] as u32, (self.raw[1] >> 32) as u32]
  }

  /// Replaces the four block words interpreted as `[key0, pos0, key1, pos1]`.
  #[inline]
  pub const fn set_block_words(&mut self, words: [u32; CACHELINE_COUNTER_BUCKET_BLOCK_WORDS]) {
    self.raw[0] = words[0] as u64 | ((words[1] as u64) << 32);
    self.raw[1] = words[2] as u64 | ((words[3] as u64) << 32);
  }

  /// Returns the key in one of the two block slots.
  #[inline]
  pub fn key(&self, slot: usize) -> u32 {
    debug_assert!(slot < CACHELINE_COUNTER_BUCKET_BLOCKS);
    self.block_words()[slot * 2]
  }

  /// Returns the position in one of the two block slots.
  #[inline]
  pub fn pos(&self, slot: usize) -> PositionType {
    debug_assert!(slot < CACHELINE_COUNTER_BUCKET_BLOCKS);
    self.block_words()[slot * 2 + 1]
  }

  /// Sets the key and position in one of the two block slots.
  #[inline]
  pub fn set_key_pos(&mut self, slot: usize, key: u32, pos: PositionType) {
    debug_assert!(slot < CACHELINE_COUNTER_BUCKET_BLOCKS);
    let mut words = self.block_words();
    words[slot * 2] = key;
    words[slot * 2 + 1] = pos;
    self.set_block_words(words);
  }

  /// Returns the low 64 bits of the counter at `index`.
  ///
  /// The storage format can represent counters wider than 64 bits. For callers
  /// that need the complete value, use [`Self::get_counter_bits`].
  ///
  /// # Oblivious
  /// This uses fixed-word metadata scans, popcount-based bit selection, and a
  /// fixed set of shifts over the counter payload.
  #[inline]
  pub fn get_counter(&self, index: usize) -> u64 {
    let location = self.counter_location(index);
    self.get_counter_low_at(location.start, location.width)
  }

  /// Returns counters in `start_index..end_index`.
  ///
  /// The returned array always has 64 entries. Entries outside the requested
  /// range are zeroed so the work and output shape do not depend on the range
  /// length.
  ///
  /// # Oblivious
  /// This performs the same fixed scan for every possible counter index.
  #[inline]
  pub fn get_counters(&self, start_index: usize, end_index: usize) -> [u64; 64] {
    debug_assert!(start_index <= end_index);
    debug_assert!(end_index <= CACHELINE_COUNTER_BUCKET_COUNTERS);

    let mut out = [0u64; CACHELINE_COUNTER_BUCKET_COUNTERS];
    for (i, item) in out.iter_mut().enumerate() {
      let value = self.get_counter(i);
      let in_range = (i >= start_index) & (i < end_index);
      *item = select_u64(0, value, in_range);
    }
    out
  }

  /// Returns the full 160-bit counter value as little-endian 64-bit limbs.
  ///
  /// Bits above the counter width are zero.
  ///
  /// # Oblivious
  /// This uses word shifts and masks over the three counter payload words.
  #[inline]
  pub fn get_counter_bits(&self, index: usize) -> [u64; 3] {
    let location = self.counter_location(index);
    self.get_counter_bits_at(location.start, location.width)
  }

  /// Increments the counter at `index`.
  ///
  /// Returns `false` if the increment would need one more payload bit but all
  /// 160 payload bits are already assigned to counters. In that case, the bucket
  /// is left unchanged.
  ///
  /// # Oblivious
  /// This uses fixed-word popcount/leading-zero selection, word shifts by zero
  /// or one bit, and masked multiword addition. The memory footprint is the
  /// single cache line backing the bucket.
  #[inline]
  pub fn increment_counter(&mut self, index: usize) -> bool {
    let location = self.counter_location(index);
    let end = location.start + location.width;
    let counter = self.get_counter_bits_at(location.start, location.width);
    let counter_mask = counter_width_mask(location.width);
    let grow = (counter[0] == counter_mask[0])
      & (counter[1] == counter_mask[1])
      & (counter[2] == counter_mask[2]);
    let has_free_bit = location.used_bits < CACHELINE_COUNTER_BUCKET_COUNTER_BITS;
    let can_increment = (!grow) | has_free_bit;
    let grow_enabled = grow & can_increment;

    let metadata = insert_zero_bit_4(
      self.metadata_words(),
      location.delimiter,
      grow_enabled,
      METADATA_WORD_MASKS,
    );
    self.set_metadata_words(metadata);

    let counters = insert_zero_bit_3(self.counter_words(), end, grow_enabled, COUNTER_WORD_MASKS);
    let counters = add_power_of_two_3(counters, location.start, can_increment);
    self.set_counter_words(counters);

    can_increment
  }

  #[inline]
  fn counter_location(&self, index: usize) -> CounterLocation {
    debug_assert!(index < CACHELINE_COUNTER_BUCKET_COUNTERS);

    let words = self.metadata_words();
    let mut ones_before = 0usize;
    let mut previous_delimiter = usize::MAX;
    let mut delimiter = 0usize;
    let mut last_delimiter = usize::MAX;

    for (word_index, word) in words.iter().enumerate() {
      let word = *word & METADATA_WORD_MASKS[word_index];
      let ones_in_word = word.count_ones() as usize;
      let current_in_word = (index >= ones_before) & (index < ones_before + ones_in_word);
      let rank_in_word = index.wrapping_sub(ones_before);
      let word_offset = word_index * 64;
      let current_relative = select_nth_one_u64(word, rank_in_word);
      let current_delimiter = word_offset + current_relative;

      let lower_bits = word & low_bits_mask(current_relative);
      let previous_relative = 63usize.wrapping_sub(lower_bits.leading_zeros() as usize);
      let previous_in_word = word_offset.wrapping_add(previous_relative);
      let previous_candidate = select_usize(last_delimiter, previous_in_word, lower_bits != 0);

      previous_delimiter = select_usize(previous_delimiter, previous_candidate, current_in_word);
      delimiter = select_usize(delimiter, current_delimiter, current_in_word);

      let last_relative = 63usize.wrapping_sub(word.leading_zeros() as usize);
      let last_in_word = word_offset.wrapping_add(last_relative);
      last_delimiter = select_usize(last_delimiter, last_in_word, word != 0);
      ones_before += ones_in_word;
    }

    let width = delimiter.wrapping_sub(previous_delimiter).wrapping_sub(1);
    let end = delimiter.wrapping_sub(index);
    let start = end.wrapping_sub(width);
    let used_bits = last_delimiter.wrapping_sub(CACHELINE_COUNTER_BUCKET_COUNTERS - 1);

    CounterLocation { start, width, delimiter, used_bits }
  }

  #[inline]
  const fn get_counter_bits_at(&self, start: usize, width: usize) -> [u64; 3] {
    let mut out = shifted_counter_words(self.counter_words(), start);
    let masks = counter_width_mask(width);
    out[0] &= masks[0];
    out[1] &= masks[1];
    out[2] &= masks[2];
    out
  }

  #[inline]
  const fn get_counter_low_at(&self, start: usize, width: usize) -> u64 {
    shifted_counter_word0(self.counter_words(), start) & low_bits_mask(width)
  }

  #[inline]
  const fn metadata_words(&self) -> [u64; METADATA_WORDS] {
    [self.raw[2], self.raw[3], self.raw[4], self.raw[5] & U32_MASK]
  }

  #[inline]
  const fn set_metadata_words(&mut self, words: [u64; METADATA_WORDS]) {
    self.raw[2] = words[0];
    self.raw[3] = words[1];
    self.raw[4] = words[2];
    self.raw[5] = (self.raw[5] & !U32_MASK) | (words[3] & U32_MASK);
  }

  #[inline]
  const fn counter_words(&self) -> [u64; COUNTER_WORDS] {
    [
      (self.raw[5] >> 32) | (self.raw[6] << 32),
      (self.raw[6] >> 32) | (self.raw[7] << 32),
      self.raw[7] >> 32,
    ]
  }

  #[inline]
  const fn set_counter_words(&mut self, words: [u64; COUNTER_WORDS]) {
    self.raw[5] = (self.raw[5] & U32_MASK) | (words[0] << 32);
    self.raw[6] = (words[0] >> 32) | (words[1] << 32);
    self.raw[7] = (words[1] >> 32) | ((words[2] & U32_MASK) << 32);
  }
}

#[inline]
const fn select_nth_one_u64(word: u64, rank: usize) -> usize {
  let (word, rank, offset) = select_nth_one_step(word, rank, 0, 32);
  let (word, rank, offset) = select_nth_one_step(word, rank, offset, 16);
  let (word, rank, offset) = select_nth_one_step(word, rank, offset, 8);
  let (word, rank, offset) = select_nth_one_step(word, rank, offset, 4);
  let (word, rank, offset) = select_nth_one_step(word, rank, offset, 2);

  let first = word & word.wrapping_neg();
  let without_first = word & word.wrapping_sub(1);
  let second = without_first & without_first.wrapping_neg();
  let selected = select_u64(first, second, rank != 0) | 1;
  offset + 63usize.wrapping_sub(selected.leading_zeros() as usize)
}

#[inline]
const fn select_nth_one_step(
  word: u64,
  rank: usize,
  offset: usize,
  bits: usize,
) -> (u64, usize, usize) {
  let low_mask = (1u64 << bits) - 1;
  let low = word & low_mask;
  let high = word >> bits;
  let low_count = low.count_ones() as usize;
  let take_high = rank >= low_count;

  (
    select_u64(low, high, take_high),
    select_usize(rank, rank.wrapping_sub(low_count), take_high),
    offset + (bits & mask_usize(take_high)),
  )
}

#[inline]
const fn shifted_counter_words(words: [u64; COUNTER_WORDS], start: usize) -> [u64; COUNTER_WORDS] {
  let shift = start & 63;
  let word_index = start >> 6;

  let shifted0 = shr_pair(words[0], words[1], shift);
  let shifted1 = shr_pair(words[1], words[2], shift);
  let shifted2 = shr_pair(words[2], 0, shift);

  [
    select_by_word_index(shifted0, shifted1, shifted2, word_index),
    select_by_word_index(shifted1, shifted2, 0, word_index),
    select_by_word_index(shifted2, 0, 0, word_index),
  ]
}

#[inline]
const fn shifted_counter_word0(words: [u64; COUNTER_WORDS], start: usize) -> u64 {
  let shift = start & 63;
  let word_index = start >> 6;

  let shifted0 = shr_pair(words[0], words[1], shift);
  let shifted1 = shr_pair(words[1], words[2], shift);
  let shifted2 = shr_pair(words[2], 0, shift);

  select_by_word_index(shifted0, shifted1, shifted2, word_index)
}

#[inline]
const fn shr_pair(low: u64, high: u64, shift: usize) -> u64 {
  low.wrapping_shr(shift as u32)
    | (high.wrapping_shl((64usize.wrapping_sub(shift) & 63) as u32) & mask_u64(shift != 0))
}

#[inline]
const fn select_by_word_index(word0: u64, word1: u64, word2: u64, word_index: usize) -> u64 {
  let selected = select_u64(word0, word1, word_index == 1);
  select_u64(selected, word2, word_index == 2)
}

#[inline]
const fn counter_width_mask(width: usize) -> [u64; COUNTER_WORDS] {
  [limb_width_mask(width, 0), limb_width_mask(width, 1), limb_width_mask(width, 2) & U32_MASK]
}

#[inline]
const fn limb_width_mask(width: usize, limb: usize) -> u64 {
  let limb_start = limb * 64;
  let remaining = width.wrapping_sub(limb_start) & mask_usize(width >= limb_start);
  low_bits_mask(remaining)
}

#[inline]
const fn low_bits_mask(bits: usize) -> u64 {
  mask_u64(bits >= 64)
    | (1u64.wrapping_shl((bits & 63) as u32).wrapping_sub(1) & mask_u64(bits < 64))
}

#[inline]
const fn insert_zero_bit_3(
  words: [u64; COUNTER_WORDS],
  bit_index: usize,
  enable: bool,
  valid_masks: [u64; COUNTER_WORDS],
) -> [u64; COUNTER_WORDS] {
  let mask0 = low_mask_for_word(bit_index, 0);
  let mask1 = low_mask_for_word(bit_index, 1);
  let mask2 = low_mask_for_word(bit_index, 2);
  let upper = [words[0] & !mask0, words[1] & !mask1, words[2] & !mask2];
  let shifted = shl1_3(upper, valid_masks);
  let inserted = [
    (words[0] & mask0) | shifted[0],
    (words[1] & mask1) | shifted[1],
    (words[2] & mask2) | shifted[2],
  ];

  [
    select_u64(words[0], inserted[0], enable) & valid_masks[0],
    select_u64(words[1], inserted[1], enable) & valid_masks[1],
    select_u64(words[2], inserted[2], enable) & valid_masks[2],
  ]
}

#[inline]
const fn insert_zero_bit_4(
  words: [u64; METADATA_WORDS],
  bit_index: usize,
  enable: bool,
  valid_masks: [u64; METADATA_WORDS],
) -> [u64; METADATA_WORDS] {
  let mask0 = low_mask_for_word(bit_index, 0);
  let mask1 = low_mask_for_word(bit_index, 1);
  let mask2 = low_mask_for_word(bit_index, 2);
  let mask3 = low_mask_for_word(bit_index, 3);
  let upper = [words[0] & !mask0, words[1] & !mask1, words[2] & !mask2, words[3] & !mask3];
  let shifted = shl1_4(upper, valid_masks);
  let inserted = [
    (words[0] & mask0) | shifted[0],
    (words[1] & mask1) | shifted[1],
    (words[2] & mask2) | shifted[2],
    (words[3] & mask3) | shifted[3],
  ];

  [
    select_u64(words[0], inserted[0], enable) & valid_masks[0],
    select_u64(words[1], inserted[1], enable) & valid_masks[1],
    select_u64(words[2], inserted[2], enable) & valid_masks[2],
    select_u64(words[3], inserted[3], enable) & valid_masks[3],
  ]
}

#[inline]
const fn low_mask_for_word(bit_index: usize, word_index: usize) -> u64 {
  let word_start = word_index * 64;
  let bits_in_word = bit_index.wrapping_sub(word_start) & mask_usize(bit_index >= word_start);
  low_bits_mask(bits_in_word)
}

#[inline]
const fn shl1_3(
  words: [u64; COUNTER_WORDS],
  valid_masks: [u64; COUNTER_WORDS],
) -> [u64; COUNTER_WORDS] {
  [
    (words[0] << 1) & valid_masks[0],
    ((words[1] << 1) | (words[0] >> 63)) & valid_masks[1],
    ((words[2] << 1) | (words[1] >> 63)) & valid_masks[2],
  ]
}

#[inline]
const fn shl1_4(
  words: [u64; METADATA_WORDS],
  valid_masks: [u64; METADATA_WORDS],
) -> [u64; METADATA_WORDS] {
  [
    (words[0] << 1) & valid_masks[0],
    ((words[1] << 1) | (words[0] >> 63)) & valid_masks[1],
    ((words[2] << 1) | (words[1] >> 63)) & valid_masks[2],
    ((words[3] << 1) | (words[2] >> 63)) & valid_masks[3],
  ]
}

#[inline]
const fn add_power_of_two_3(
  words: [u64; COUNTER_WORDS],
  bit_index: usize,
  enable: bool,
) -> [u64; COUNTER_WORDS] {
  let word_index = bit_index >> 6;
  let bit = 1u64 << (bit_index & 63);
  let enable_mask = mask_u64(enable);

  let add0 = bit & enable_mask & mask_u64(word_index == 0);
  let add1 = bit & enable_mask & mask_u64(word_index == 1);
  let add2 = bit & enable_mask & mask_u64(word_index == 2);

  let (sum0, carry0) = words[0].overflowing_add(add0);

  let (partial1, carry1a) = words[1].overflowing_add(add1);
  let (sum1, carry1b) = partial1.overflowing_add(carry0 as u64);
  let carry1 = carry1a | carry1b;

  let (partial2, _) = words[2].overflowing_add(add2);
  let (sum2, _) = partial2.overflowing_add(carry1 as u64);

  [sum0, sum1, sum2 & U32_MASK]
}

#[inline]
const fn mask_u64(choice: bool) -> u64 {
  0u64.wrapping_sub(choice as u64)
}

#[inline]
const fn mask_usize(choice: bool) -> usize {
  0usize.wrapping_sub(choice as usize)
}

#[inline]
const fn select_u64(old: u64, new: u64, choice: bool) -> u64 {
  let mask = mask_u64(choice);
  (old & !mask) | (new & mask)
}

#[inline]
const fn select_usize(old: usize, new: usize, choice: bool) -> usize {
  let mask = mask_usize(choice);
  (old & !mask) | (new & mask)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn cacheline_bucket_layout_is_one_cacheline() {
    assert_eq!(core::mem::size_of::<Cacheline_Counter_Bucket>(), 64);
    assert_eq!(core::mem::align_of::<Cacheline_Counter_Bucket>(), 64);
  }

  #[test]
  fn default_bucket_has_zero_width_zero_counters() {
    let bucket = Cacheline_Counter_Bucket::default();

    assert_eq!(bucket.raw[2], u64::MAX);
    for i in 0..CACHELINE_COUNTER_BUCKET_COUNTERS {
      assert_eq!(bucket.get_counter(i), 0);
      assert_eq!(bucket.get_counter_bits(i), [0, 0, 0]);
    }
  }

  #[test]
  fn block_words_round_trip() {
    let mut bucket = Cacheline_Counter_Bucket::default();

    bucket.set_block_words([7, 11, 13, 17]);
    assert_eq!(bucket.block_words(), [7, 11, 13, 17]);

    bucket.set_key_pos(1, 19, 23);
    assert_eq!(bucket.key(0), 7);
    assert_eq!(bucket.pos(0), 11);
    assert_eq!(bucket.key(1), 19);
    assert_eq!(bucket.pos(1), 23);
  }

  #[test]
  fn increments_grow_and_preserve_neighbor_counters() {
    let mut bucket = Cacheline_Counter_Bucket::default();

    assert!(bucket.increment_counter(7));
    assert!(bucket.increment_counter(3));
    assert!(bucket.increment_counter(7));
    assert!(bucket.increment_counter(7));
    assert!(bucket.increment_counter(63));

    assert_eq!(bucket.get_counter(3), 1);
    assert_eq!(bucket.get_counter(7), 3);
    assert_eq!(bucket.get_counter(63), 1);
    assert_eq!(bucket.get_counter(6), 0);
    assert_eq!(bucket.counter_location(3).width, 1);
    assert_eq!(bucket.counter_location(7).width, 2);
    assert_eq!(bucket.counter_location(63).width, 1);
  }

  #[test]
  fn get_counters_zeroes_values_outside_requested_range() {
    let mut bucket = Cacheline_Counter_Bucket::default();

    for _ in 0..5 {
      assert!(bucket.increment_counter(10));
    }
    assert!(bucket.increment_counter(11));
    assert!(bucket.increment_counter(12));

    let counters = bucket.get_counters(10, 12);
    assert_eq!(counters[9], 0);
    assert_eq!(counters[10], 5);
    assert_eq!(counters[11], 1);
    assert_eq!(counters[12], 0);
  }

  #[test]
  fn increment_reports_when_no_free_growth_bit_remains() {
    let mut bucket = Cacheline_Counter_Bucket::default();

    bucket.set_metadata_words([0, 0, u64::MAX << 32, U32_MASK]);
    bucket.set_counter_words([u64::MAX, u64::MAX, U32_MASK]);

    assert_eq!(bucket.counter_location(0).width, CACHELINE_COUNTER_BUCKET_COUNTER_BITS);
    assert!(!bucket.increment_counter(0));
    assert_eq!(bucket.get_counter_bits(0), [u64::MAX, u64::MAX, u32::MAX as u64]);
  }
}
