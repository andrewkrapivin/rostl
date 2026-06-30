//! Specialized building blocks for a faster Circuit ORAM implementation.
//!
//! The first bucket type in this module stores two `(key, pos)` pairs and a
//! compact variable-width counter array in one cache line.
//!
//! # Cacheline counter specification
//!
//! `Cacheline_Counter_Bucket` is one 64-byte, 64-byte-aligned cache line. It
//! contains two `(key, pos)` slots, metadata for 64 counters, and 160 packed
//! payload bits shared by those counters.
//!
//! Metadata is a unary delimiter stream with exactly 64 one-bits. For counter
//! `i`, the `i`th one-bit is its delimiter; the previous delimiter is the
//! `(i - 1)`th one-bit, or virtual bit `-1` for counter 0. The number of zeros
//! between the two delimiters is the counter width. Width 0 represents logical
//! value 0 and consumes no payload bits.
//!
//! Nonzero counters store `counter - 1` in little-endian order in the payload.
//! Thus values `1..=2` use one bit, values `3..=4` use two bits, and so on.
//! A single counter may use at most 64 payload bits. The zeros after the 64th
//! metadata delimiter are free payload capacity, up to 160 total payload bits.
//!
//! `increment_counter` keeps the representation minimal: it grows a counter
//! only when the stored `counter - 1` value is all ones for the current width.
//! Growth inserts one metadata zero before that counter's delimiter and one
//! payload zero at the end of that counter's payload range. If growth would
//! exceed 64 bits for one counter or 160 assigned payload bits globally, the
//! increment fails and leaves the bucket unchanged.
//!
//! The public operations are written with fixed-size word scans and masked
//! selection. They should not perform secret-dependent memory accesses outside
//! this single cache line.

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
  /// First payload bit assigned to this counter.
  start: usize,
  /// Number of payload bits assigned to this counter.
  width: usize,
  /// Metadata bit index of this counter's ending one-bit delimiter.
  delimiter: usize,
  /// Total payload bits assigned across all counters in this bucket.
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
/// Counter widths are encoded by delimiter gaps in metadata; nonzero payload
/// values encode `counter - 1`. See the module-level specification for the full
/// representation contract.
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

  /// Returns the logical counter value at `index`.
  ///
  /// Width-zero counters return 0. Nonzero-width counters return the packed
  /// payload value plus 1.
  ///
  /// # Oblivious
  /// This uses fixed-word metadata scans, popcount-based bit selection, and a
  /// fixed set of shifts over the counter payload.
  #[inline]
  pub fn get_counter(&self, index: usize) -> u64 {
    let (start, end) = self.get_counter_endpoints(index, index + 1);
    let width = end - start;
    let stored = self.counter_at(start, width);
    select_u64(0, stored.wrapping_add(1), width != 0)
  }

  /// Returns the payload bit range containing counters in `start_index..end_index`.
  ///
  /// When `end_index == start_index + 1`, this returns the start and end bits
  /// for the single counter at `start_index`. Empty or all-zero-width ranges
  /// have equal start/end offsets.
  ///
  /// # Oblivious
  /// This scans the fixed-size metadata bitmap once.
  #[inline]
  pub fn get_counter_endpoints(&self, start_index: usize, end_index: usize) -> (usize, usize) {
    debug_assert!(start_index <= end_index);
    debug_assert!(end_index <= CACHELINE_COUNTER_BUCKET_COUNTERS);

    let words = self.metadata_words();
    let start_target = start_index.wrapping_sub(1);
    let end_target = end_index.wrapping_sub(1);
    let need_start = start_index != 0;
    let need_end = end_index != 0;
    let mut ones_before = 0usize;
    let mut start = 0usize;
    let mut end = 0usize;

    for (word_index, word) in words.iter().enumerate() {
      let word = *word & METADATA_WORD_MASKS[word_index];
      let ones_in_word = word.count_ones() as usize;
      let word_offset = word_index * 64;

      let start_in_word =
        need_start & (start_target >= ones_before) & (start_target < ones_before + ones_in_word);
      let start_rank = start_target.wrapping_sub(ones_before);
      let start_delimiter = word_offset + select_nth_one_u64(word, start_rank);
      start = select_usize(start, start_delimiter.wrapping_sub(start_target), start_in_word);

      let end_in_word =
        need_end & (end_target >= ones_before) & (end_target < ones_before + ones_in_word);
      let end_rank = end_target.wrapping_sub(ones_before);
      let end_delimiter = word_offset + select_nth_one_u64(word, end_rank);
      end = select_usize(end, end_delimiter.wrapping_sub(end_target), end_in_word);

      ones_before += ones_in_word;
    }

    (start, end)
  }

  /// Increments the logical counter at `index`.
  ///
  /// Returns `false` if the increment would need a 65th bit for this counter,
  /// or if it would need one more payload bit but all 160 payload bits are
  /// already assigned to counters. In that case, the bucket is left unchanged.
  /// Otherwise, the counter is incremented and the payload continues to store
  /// `counter - 1`.
  ///
  /// # Oblivious
  /// This uses fixed-word popcount/leading-zero selection, word shifts by zero
  /// or one bit, and masked multiword addition. The memory footprint is the
  /// single cache line backing the bucket.
  #[inline]
  pub fn increment_counter(&mut self, index: usize) -> bool {
    let location = self.counter_location(index);
    let end = location.start + location.width;
    let counter = self.counter_at(location.start, location.width);
    let grow = (location.width == 0) | (counter == low_bits_mask(location.width));
    let has_free_bit = location.used_bits < CACHELINE_COUNTER_BUCKET_COUNTER_BITS;
    let can_grow = has_free_bit & (location.width < u64::BITS as usize);
    let would_overflow_u64 = (location.width == u64::BITS as usize) & (counter >= u64::MAX - 1);
    let can_increment = ((!grow) | can_grow) & !would_overflow_u64;
    let grow_enabled = grow & can_increment;
    let add_enabled = can_increment & (location.width != 0);

    let metadata =
      insert_zero_bit(self.metadata_words(), location.delimiter, grow_enabled, METADATA_WORD_MASKS);
    self.set_metadata_words(metadata);

    let counters = insert_zero_bit(self.counter_words(), end, grow_enabled, COUNTER_WORD_MASKS);
    let counters = add_bit(counters, location.start, add_enabled, COUNTER_WORD_MASKS);
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
  const fn counter_at(&self, start: usize, width: usize) -> u64 {
    extract_u64(self.counter_words(), start, width)
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
const fn extract_u64<const N: usize>(words: [u64; N], start: usize, width: usize) -> u64 {
  let shift = start & 63;
  let word_index = start >> 6;
  let mut out = 0u64;
  let mut i = 0;

  while i < N {
    let high = if i + 1 < N { words[i + 1] } else { 0 };
    let candidate = shr_pair(words[i], high, shift);
    out = select_u64(out, candidate, i == word_index);
    i += 1;
  }

  out & low_bits_mask(width)
}

#[inline]
const fn shr_pair(low: u64, high: u64, shift: usize) -> u64 {
  low.wrapping_shr(shift as u32)
    | (high.wrapping_shl((64usize.wrapping_sub(shift) & 63) as u32) & mask_u64(shift != 0))
}

#[inline]
const fn low_bits_mask(bits: usize) -> u64 {
  mask_u64(bits >= 64)
    | (1u64.wrapping_shl((bits & 63) as u32).wrapping_sub(1) & mask_u64(bits < 64))
}

#[inline]
const fn insert_zero_bit<const N: usize>(
  words: [u64; N],
  bit_index: usize,
  enable: bool,
  valid_masks: [u64; N],
) -> [u64; N] {
  let mut out = [0u64; N];
  let mut carry = 0u64;
  let mut i = 0;

  while i < N {
    let keep_mask = low_mask_for_word(bit_index, i);
    let upper = words[i] & !keep_mask;
    let inserted = (words[i] & keep_mask) | ((upper << 1) | carry);
    out[i] = select_u64(words[i], inserted & valid_masks[i], enable);
    carry = upper >> 63;
    i += 1;
  }

  out
}

#[inline]
const fn low_mask_for_word(bit_index: usize, word_index: usize) -> u64 {
  let word_start = word_index * 64;
  let bits_in_word = bit_index.wrapping_sub(word_start) & mask_usize(bit_index >= word_start);
  low_bits_mask(bits_in_word)
}

#[inline]
const fn add_bit<const N: usize>(
  mut words: [u64; N],
  bit_index: usize,
  enable: bool,
  valid_masks: [u64; N],
) -> [u64; N] {
  let word_index = bit_index >> 6;
  let bit = 1u64 << (bit_index & 63);
  let enable_mask = mask_u64(enable);
  let mut carry = 0u64;
  let mut i = 0;

  while i < N {
    let add = ((bit & mask_u64(i == word_index)) | carry) & enable_mask;
    let (sum, overflow) = words[i].overflowing_add(add);
    words[i] = sum & valid_masks[i];
    carry = overflow as u64;
    i += 1;
  }

  words
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
    assert_eq!(bucket.get_counter(7), 1);
    assert!(bucket.increment_counter(3));
    assert!(bucket.increment_counter(7));
    assert_eq!(bucket.get_counter(7), 2);
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
  fn get_counter_endpoints_returns_payload_range_for_counter_range() {
    let mut bucket = Cacheline_Counter_Bucket::default();

    for _ in 0..5 {
      assert!(bucket.increment_counter(10));
    }
    assert!(bucket.increment_counter(11));
    assert!(bucket.increment_counter(12));

    assert_eq!(bucket.get_counter(10), 5);
    assert_eq!(bucket.get_counter(11), 1);
    assert_eq!(bucket.get_counter(12), 1);
    assert_eq!(bucket.get_counter_endpoints(10, 11), (0, 3));
    assert_eq!(bucket.get_counter_endpoints(11, 12), (3, 4));
    assert_eq!(bucket.get_counter_endpoints(12, 13), (4, 5));
    assert_eq!(bucket.get_counter_endpoints(10, 13), (0, 5));
    assert_eq!(bucket.get_counter_endpoints(0, 64), (0, 5));
  }

  #[test]
  fn increment_reports_when_counter_would_exceed_64_bits() {
    let mut bucket = Cacheline_Counter_Bucket::default();

    bucket.set_metadata_words([0, u64::MAX, 0, 0]);
    bucket.set_counter_words([u64::MAX - 1, 0, 0]);

    assert_eq!(bucket.counter_location(0).width, 64);
    assert_eq!(bucket.get_counter(0), u64::MAX);
    assert!(!bucket.increment_counter(0));
    assert_eq!(bucket.get_counter(0), u64::MAX);
  }
}
