//! Alternate cache-line counter blocks for Circuit ORAM.
//!
//! This representation makes each logical counter block a full 64-byte cache
//! line. A bucket is two such blocks and is 128-byte aligned. This spends two
//! cache lines per ORAM bucket, but makes path read/write as simple as copying
//! two blocks, with no split/merge packing work.
//!
//! # Alternate cacheline counter specification
//!
//! `Counter_Block_Alt` is one 64-byte, 64-byte-aligned block. It owns one
//! `(key, pos)` pair and 64 counters:
//! * bits `0..64`: `[key, pos]`, each `u32`;
//! * bits `64..320`: 256 metadata bits;
//! * bits `320..512`: 192 packed counter payload bits.
//!
//! Metadata is the same unary delimiter stream used by `fast_buckets`: exactly
//! 64 one-bits delimit 64 counters. The number of zeros before each delimiter
//! gives that counter's bit width. Width zero represents value zero. Nonzero
//! counters store `counter - 1` in little-endian payload bits.

#![allow(clippy::needless_bitwise_bool)]

use bytemuck::{Pod, Zeroable};
use rostl_primitives::traits::Cmov;
use static_assertions::const_assert_eq;

use crate::prelude::{PositionType, DUMMY_POS};

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "bmi2"))]
#[cfg(target_arch = "x86")]
use core::arch::x86::_pdep_u64;
#[cfg(target_arch = "x86")]
use core::arch::x86::{
  __m512i, __mmask8, _mm512_load_si512, _mm512_mask_mov_epi64, _mm512_store_si512,
};
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
use core::arch::x86_64::_pdep_u64;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
  __m512i, __mmask8, _mm512_load_si512, _mm512_mask_mov_epi64, _mm512_store_si512,
};

/// Blocks per alternate counter bucket.
pub const ALT_COUNTER_BUCKET_BLOCKS: usize = 2;
/// Counters controlled by one alternate counter block.
pub const ALT_COUNTER_BLOCK_COUNTERS: usize = 64;
/// Metadata bits in one alternate counter block.
pub const ALT_COUNTER_BLOCK_METADATA_BITS: usize = 256;
/// Packed payload bits in one alternate counter block.
pub const ALT_COUNTER_BLOCK_COUNTER_BITS: usize = 192;

const RAW_WORDS: usize = 8;
const METADATA_WORDS: usize = 4;
const COUNTER_WORDS: usize = 3;
const U32_MASK: u64 = u32::MAX as u64;
const METADATA_WORD_MASKS: [u64; METADATA_WORDS] = [u64::MAX; METADATA_WORDS];
const COUNTER_WORD_MASKS: [u64; COUNTER_WORDS] = [u64::MAX; COUNTER_WORDS];

const_assert_eq!(RAW_WORDS * u64::BITS as usize, 512);
const_assert_eq!(
  u64::BITS as usize + ALT_COUNTER_BLOCK_METADATA_BITS + ALT_COUNTER_BLOCK_COUNTER_BITS,
  512
);

/// A 64-byte, cache-line-aligned block controlling 64 counters.
#[allow(non_camel_case_types)]
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Zeroable)]
pub struct Counter_Block_Alt {
  raw: [u64; RAW_WORDS],
}

unsafe impl Pod for Counter_Block_Alt {}

impl Default for Counter_Block_Alt {
  fn default() -> Self {
    let mut raw = [0u64; RAW_WORDS];
    raw[0] = (DUMMY_POS as u64) << 32;
    raw[1] = u64::MAX;
    Self { raw }
  }
}

impl Cmov for Counter_Block_Alt {
  #[inline(always)]
  fn cmov(&mut self, other: &Self, choice: bool) {
    cmov_raw_words(&mut self.raw, &other.raw, choice);
  }

  #[inline(always)]
  fn cxchg(&mut self, other: &mut Self, choice: bool) {
    cxchg_raw_words(&mut self.raw, &mut other.raw, choice);
  }
}

const_assert_eq!(core::mem::size_of::<Counter_Block_Alt>(), 64);
const_assert_eq!(core::mem::align_of::<Counter_Block_Alt>(), 64);

impl Counter_Block_Alt {
  /// Creates an empty block with all counters set to zero.
  #[inline]
  pub fn new() -> Self {
    Self::default()
  }

  /// Returns this block's key.
  #[inline]
  pub const fn key(&self) -> u32 {
    self.raw[0] as u32
  }

  /// Returns this block's position.
  #[inline]
  pub const fn pos(&self) -> PositionType {
    (self.raw[0] >> 32) as u32
  }

  /// Sets this block's key and position.
  #[inline]
  pub fn set_key_pos(&mut self, key: u32, pos: PositionType) {
    self.raw[0] = key as u64 | ((pos as u64) << 32);
  }

  /// Conditionally marks this block empty by setting only the position.
  #[inline(always)]
  pub fn cmov_empty(&mut self, choice: bool) {
    let empty_raw0 = (self.raw[0] & U32_MASK) | ((DUMMY_POS as u64) << 32);
    self.raw[0] = select_u64(self.raw[0], empty_raw0, choice);
  }

  /// Returns whether this block is empty.
  #[inline]
  pub const fn is_empty(&self) -> bool {
    self.pos() == DUMMY_POS
  }

  /// Returns the logical value of counter `index`.
  #[inline(always)]
  pub fn get_counter(&self, index: usize) -> u64 {
    let (start, end) = self.get_counter_endpoints(index, index + 1);
    let width = end - start;
    let stored = extract_counter_u64(self.counter_words(), start, width);
    select_u64(0, stored.wrapping_add(1), width != 0)
  }

  /// Returns the local payload bit range for counters in `start_index..end_index`.
  #[inline(always)]
  pub fn get_counter_endpoints(&self, start_index: usize, end_index: usize) -> (usize, usize) {
    debug_assert!(start_index <= end_index);
    debug_assert!(end_index <= ALT_COUNTER_BLOCK_COUNTERS);

    let words = self.metadata_words();
    let start = payload_end_for_index(words, start_index);
    let end = payload_end_for_index(words, end_index);

    (start, end)
  }

  /// Increments counter `index`, returning whether capacity was available.
  #[inline(always)]
  pub fn increment_counter(&mut self, index: usize) -> bool {
    self.get_and_increment_counter(index).1
  }

  /// Returns counter `index`'s old value and increments it.
  #[inline(always)]
  pub fn get_and_increment_counter(&mut self, index: usize) -> (u64, bool) {
    self.get_and_increment_counter_if(index, true)
  }

  /// Reads or increments `selected_index` after the containing block has been selected.
  #[inline(always)]
  pub fn access_counter_oblivious(&mut self, selected_index: usize, increment: bool) -> u64 {
    debug_assert!(selected_index < ALT_COUNTER_BLOCK_COUNTERS);

    if increment {
      let (old, incremented) = self.get_and_increment_counter(selected_index);
      debug_assert!(incremented);
      old
    } else {
      self.get_counter(selected_index)
    }
  }

  #[inline(always)]
  fn get_and_increment_counter_if(&mut self, index: usize, enable: bool) -> (u64, bool) {
    debug_assert!(index < ALT_COUNTER_BLOCK_COUNTERS);

    let words = self.metadata_words();
    let previous_rank = index.wrapping_sub(1);
    let previous_delimiter =
      select_usize(select_delimiter(words, previous_rank), usize::MAX, index == 0);
    let delimiter = select_delimiter(words, index);
    let width = delimiter.wrapping_sub(previous_delimiter).wrapping_sub(1);
    let end = delimiter.wrapping_sub(index);
    let start = end.wrapping_sub(width);
    let used_bits = last_delimiter(words).wrapping_sub(ALT_COUNTER_BLOCK_COUNTERS - 1);
    let counter = extract_counter_u64(self.counter_words(), start, width);
    let old_value = select_u64(0, counter.wrapping_add(1), width != 0);
    let grow = (width == 0) | (counter == low_bits_mask(width));
    let has_free_bit = used_bits < ALT_COUNTER_BLOCK_COUNTER_BITS;
    let can_grow = has_free_bit & (width < u64::BITS as usize);
    let would_overflow_u64 = (width == u64::BITS as usize) & (counter >= u64::MAX - 1);
    let can_increment = ((!grow) | can_grow) & !would_overflow_u64;
    let grow_enabled = grow & can_increment & enable;
    let add_enabled = can_increment & (width != 0) & enable;

    let metadata =
      insert_zero_bit(self.metadata_words(), delimiter, grow_enabled, METADATA_WORD_MASKS);
    self.set_metadata_words(metadata);

    let counters = insert_zero_bit(self.counter_words(), end, grow_enabled, COUNTER_WORD_MASKS);
    let counters = add_bit(counters, start, add_enabled, COUNTER_WORD_MASKS);
    self.set_counter_words(counters);

    (old_value, can_increment)
  }

  #[inline(always)]
  const fn metadata_words(&self) -> [u64; METADATA_WORDS] {
    [self.raw[1], self.raw[2], self.raw[3], self.raw[4]]
  }

  #[inline(always)]
  const fn set_metadata_words(&mut self, words: [u64; METADATA_WORDS]) {
    self.raw[1] = words[0];
    self.raw[2] = words[1];
    self.raw[3] = words[2];
    self.raw[4] = words[3];
  }

  #[inline(always)]
  const fn counter_words(&self) -> [u64; COUNTER_WORDS] {
    [self.raw[5], self.raw[6], self.raw[7]]
  }

  #[inline(always)]
  const fn set_counter_words(&mut self, words: [u64; COUNTER_WORDS]) {
    self.raw[5] = words[0];
    self.raw[6] = words[1];
    self.raw[7] = words[2];
  }
}

/// A 128-byte, 128-byte-aligned ORAM bucket with two alternate counter blocks.
#[allow(non_camel_case_types)]
#[repr(C, align(128))]
#[derive(Clone, Copy, Debug, Zeroable)]
pub struct Cacheline_Counter_Bucket_Alt {
  /// The two blocks in this ORAM bucket.
  pub blocks: [Counter_Block_Alt; ALT_COUNTER_BUCKET_BLOCKS],
}

unsafe impl Pod for Cacheline_Counter_Bucket_Alt {}

impl Default for Cacheline_Counter_Bucket_Alt {
  fn default() -> Self {
    Self { blocks: [Counter_Block_Alt::default(); ALT_COUNTER_BUCKET_BLOCKS] }
  }
}

const_assert_eq!(core::mem::size_of::<Cacheline_Counter_Bucket_Alt>(), 128);
const_assert_eq!(core::mem::align_of::<Cacheline_Counter_Bucket_Alt>(), 128);

#[inline(always)]
fn select_delimiter(words: [u64; METADATA_WORDS], rank: usize) -> usize {
  let word0 = words[0];
  let count0 = word0.count_ones() as usize;
  let word1 = words[1];
  let rank1 = rank.wrapping_sub(count0);
  let count1 = word1.count_ones() as usize;
  let word2 = words[2];
  let rank2 = rank1.wrapping_sub(count1);
  let count2 = word2.count_ones() as usize;
  let rank3 = rank2.wrapping_sub(count2);

  let in0 = rank < count0;
  let in1 = (!in0) & (rank1 < count1);
  let in2 = (!in0) & (!in1) & (rank2 < count2);

  let candidate0 = select_nth_one_u64(word0, rank);
  let candidate1 = 64 + select_nth_one_u64(word1, rank1);
  let candidate2 = 128 + select_nth_one_u64(word2, rank2);
  let candidate3 = 192 + select_nth_one_u64(words[3], rank3);

  let selected = select_usize(candidate3, candidate2, in2);
  let selected = select_usize(selected, candidate1, in1);
  select_usize(selected, candidate0, in0)
}

#[inline(always)]
fn payload_end_for_index(words: [u64; METADATA_WORDS], index: usize) -> usize {
  let rank = index.wrapping_sub(1);
  let end = select_delimiter(words, rank).wrapping_sub(rank);
  select_usize(end, 0, index == 0)
}

#[inline(always)]
fn last_delimiter(words: [u64; METADATA_WORDS]) -> usize {
  let word0 = words[0];
  let word1 = words[1];
  let word2 = words[2];
  let word3 = words[3];

  let selected = last_one_with_offset(word0, 0);
  let selected = select_usize(selected, last_one_with_offset(word1, 64), word1 != 0);
  let selected = select_usize(selected, last_one_with_offset(word2, 128), word2 != 0);
  select_usize(selected, last_one_with_offset(word3, 192), word3 != 0)
}

#[inline(always)]
fn select_nth_one_u64(word: u64, rank: usize) -> usize {
  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "bmi2"))]
  {
    let selected = unsafe { _pdep_u64(1u64 << (rank & 63), word) };
    return selected.trailing_zeros() as usize;
  }

  #[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "bmi2")))]
  {
    select_nth_one_u64_scalar(word, rank)
  }
}

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "bmi2")))]
#[inline(always)]
const fn select_nth_one_u64_scalar(word: u64, rank: usize) -> usize {
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

#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "bmi2")))]
#[inline(always)]
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

#[inline(always)]
fn extract_counter_u64(words: [u64; COUNTER_WORDS], start: usize, width: usize) -> u64 {
  let shift = start & 63;
  let word_index = start >> 6;
  let out = shr_pair(word_or_zero(words, word_index), word_or_zero(words, word_index + 1), shift);

  out & low_bits_mask(width)
}

#[inline(always)]
fn word_or_zero<const N: usize>(words: [u64; N], index: usize) -> u64 {
  let mut out = 0;
  let mut i = 0;
  while i < N {
    out = select_u64(out, words[i], i == index);
    i += 1;
  }
  out
}

#[inline(always)]
const fn shr_pair(low: u64, high: u64, shift: usize) -> u64 {
  low.wrapping_shr(shift as u32)
    | (high.wrapping_shl((64usize.wrapping_sub(shift) & 63) as u32) & mask_u64(shift != 0))
}

#[inline(always)]
const fn low_bits_mask(bits: usize) -> u64 {
  mask_u64(bits >= 64)
    | (1u64.wrapping_shl((bits & 63) as u32).wrapping_sub(1) & mask_u64(bits < 64))
}

#[inline(always)]
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

#[inline(always)]
const fn low_mask_for_word(bit_index: usize, word_index: usize) -> u64 {
  let word_start = word_index * 64;
  let bits_in_word = bit_index.wrapping_sub(word_start) & mask_usize(bit_index >= word_start);
  low_bits_mask(bits_in_word)
}

#[inline(always)]
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

#[inline(always)]
fn cmov_raw_words(dst: &mut [u64; RAW_WORDS], src: &[u64; RAW_WORDS], choice: bool) {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::arch::is_x86_feature_detected!("avx512f") {
      unsafe {
        cmov_raw_words_avx512(dst, src, choice);
      }
      return;
    }
  }

  let mask = mask_u64(choice);
  let mut i = 0;

  while i < RAW_WORDS {
    dst[i] = (dst[i] & !mask) | (src[i] & mask);
    i += 1;
  }
}

#[inline(always)]
fn cxchg_raw_words(a: &mut [u64; RAW_WORDS], b: &mut [u64; RAW_WORDS], choice: bool) {
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  {
    if std::arch::is_x86_feature_detected!("avx512f") {
      unsafe {
        cxchg_raw_words_avx512(a, b, choice);
      }
      return;
    }
  }

  let mask = mask_u64(choice);
  let mut i = 0;

  while i < RAW_WORDS {
    let delta = (a[i] ^ b[i]) & mask;
    a[i] ^= delta;
    b[i] ^= delta;
    i += 1;
  }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn cmov_raw_words_avx512(dst: &mut [u64; RAW_WORDS], src: &[u64; RAW_WORDS], choice: bool) {
  let mask = avx512_lane_mask(choice);
  let dst_vec = unsafe { _mm512_load_si512(dst.as_ptr() as *const __m512i) };
  let src_vec = unsafe { _mm512_load_si512(src.as_ptr() as *const __m512i) };
  let out = _mm512_mask_mov_epi64(dst_vec, mask, src_vec);
  unsafe {
    _mm512_store_si512(dst.as_mut_ptr() as *mut __m512i, out);
  }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn cxchg_raw_words_avx512(a: &mut [u64; RAW_WORDS], b: &mut [u64; RAW_WORDS], choice: bool) {
  let mask = avx512_lane_mask(choice);
  let a_vec = unsafe { _mm512_load_si512(a.as_ptr() as *const __m512i) };
  let b_vec = unsafe { _mm512_load_si512(b.as_ptr() as *const __m512i) };
  let new_a = _mm512_mask_mov_epi64(a_vec, mask, b_vec);
  let new_b = _mm512_mask_mov_epi64(b_vec, mask, a_vec);
  unsafe {
    _mm512_store_si512(a.as_mut_ptr() as *mut __m512i, new_a);
    _mm512_store_si512(b.as_mut_ptr() as *mut __m512i, new_b);
  }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn avx512_lane_mask(choice: bool) -> __mmask8 {
  0u8.wrapping_sub(choice as u8) as __mmask8
}

#[inline(always)]
const fn mask_u64(choice: bool) -> u64 {
  0u64.wrapping_sub(choice as u64)
}

#[inline(always)]
const fn mask_usize(choice: bool) -> usize {
  0usize.wrapping_sub(choice as usize)
}

#[inline(always)]
const fn select_u64(old: u64, new: u64, choice: bool) -> u64 {
  let mask = mask_u64(choice);
  (old & !mask) | (new & mask)
}

#[inline(always)]
const fn select_usize(old: usize, new: usize, choice: bool) -> usize {
  let mask = mask_usize(choice);
  (old & !mask) | (new & mask)
}

#[inline(always)]
const fn last_one_with_offset(word: u64, offset: usize) -> usize {
  offset.wrapping_add((u64::BITS as usize - 1).wrapping_sub(word.leading_zeros() as usize))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn alt_layouts_are_cacheline_sized() {
    assert_eq!(core::mem::size_of::<Counter_Block_Alt>(), 64);
    assert_eq!(core::mem::align_of::<Counter_Block_Alt>(), 64);
    assert_eq!(core::mem::size_of::<Cacheline_Counter_Bucket_Alt>(), 128);
    assert_eq!(core::mem::align_of::<Cacheline_Counter_Bucket_Alt>(), 128);
  }

  #[test]
  fn alt_block_reads_zero_by_default() {
    let block = Counter_Block_Alt::default();

    for index in 0..ALT_COUNTER_BLOCK_COUNTERS {
      assert_eq!(block.get_counter(index), 0);
    }
  }

  #[test]
  fn alt_block_increments_match_reference() {
    let mut block = Counter_Block_Alt::default();
    let mut reference = [0u64; ALT_COUNTER_BLOCK_COUNTERS];

    for step in 0..512 {
      let index = (step * 37 + 11) & (ALT_COUNTER_BLOCK_COUNTERS - 1);
      let (old, incremented) = block.get_and_increment_counter(index);
      assert!(incremented);
      assert_eq!(old, reference[index]);
      reference[index] += 1;

      let read_index = (step * 19 + 7) & (ALT_COUNTER_BLOCK_COUNTERS - 1);
      assert_eq!(block.get_counter(read_index), reference[read_index]);
    }
  }

  #[test]
  fn alt_block_key_pos_and_empty_round_trip() {
    let mut block = Counter_Block_Alt::default();

    assert!(block.is_empty());
    block.set_key_pos(17, 23);
    assert!(!block.is_empty());
    assert_eq!(block.key(), 17);
    assert_eq!(block.pos(), 23);

    block.cmov_empty(true);
    assert!(block.is_empty());
    assert_eq!(block.key(), 17);
  }
}
