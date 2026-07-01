//! 1.5-cacheline counter blocks for Circuit ORAM.
//!
//! Each logical counter block is 96 bytes. A bucket contains two blocks, so it
//! is exactly 192 bytes, or three cache lines, and is 64-byte aligned.
//!
//! # 1.5-cacheline counter specification
//!
//! `Counter_Block_15` owns one `(key, pos)` pair and 64 counters:
//! * bits `0..64`: `[key, pos]`, each `u32`;
//! * bits `64..234`: 170 metadata bits;
//! * bits `234..766`: 532 counter payload bits;
//! * bits `766..768`: spare.
//!
//! Every counter has `b = 5` base payload bits. Metadata is a unary delimiter
//! stream with exactly 64 one-bits. Each zero before a counter's delimiter gives
//! that counter one extra chunk of `c = 2` payload bits. Thus the 106 available
//! metadata zeros allocate 212 extra payload bits across the 64 counters.
//! Counters store their value directly in the allocated bits and are capped at
//! width 63.

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
  __m512i, __mmask8, _mm512_loadu_si512, _mm512_mask_mov_epi64, _mm512_storeu_si512,
};
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
use core::arch::x86_64::_pdep_u64;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
  __m512i, __mmask8, _mm512_loadu_si512, _mm512_mask_mov_epi64, _mm512_storeu_si512,
};

/// Blocks per 1.5-cacheline counter bucket.
pub const COUNTER_15_BUCKET_BLOCKS: usize = 2;
/// Counters controlled by one 1.5-cacheline counter block.
pub const COUNTER_15_BLOCK_COUNTERS: usize = 64;
/// Base bits automatically assigned to each counter.
pub const COUNTER_15_BASE_BITS: usize = 5;
/// Extra bits assigned by one metadata zero.
pub const COUNTER_15_CHUNK_BITS: usize = 2;
/// Maximum counter width.
pub const COUNTER_15_MAX_COUNTER_BITS: usize = 63;
/// Metadata bits in one 1.5-cacheline counter block.
pub const COUNTER_15_METADATA_BITS: usize = 170;
/// Packed payload bits in one 1.5-cacheline counter block.
pub const COUNTER_15_COUNTER_BITS: usize = 532;
/// Maximum metadata chunks available across one block.
pub const COUNTER_15_MAX_CHUNKS: usize = 106;

const RAW_WORDS: usize = 12;
const METADATA_WORDS: usize = 3;
const COUNTER_WORDS: usize = 9;
const KEY_POS_BITS: usize = 64;
const METADATA_BIT_OFFSET: usize = KEY_POS_BITS;
const COUNTER_BIT_OFFSET: usize = METADATA_BIT_OFFSET + COUNTER_15_METADATA_BITS;
const U32_MASK: u64 = u32::MAX as u64;
const RAW_WORD_MASKS: [u64; RAW_WORDS] = [u64::MAX; RAW_WORDS];
const METADATA_WORD_MASKS: [u64; METADATA_WORDS] = [u64::MAX, u64::MAX, (1u64 << 42) - 1];
const COUNTER_WORD_MASKS: [u64; COUNTER_WORDS] = [
  u64::MAX,
  u64::MAX,
  u64::MAX,
  u64::MAX,
  u64::MAX,
  u64::MAX,
  u64::MAX,
  u64::MAX,
  (1u64 << 20) - 1,
];

const_assert_eq!(RAW_WORDS * u64::BITS as usize, 768);
const_assert_eq!(
  KEY_POS_BITS + COUNTER_15_METADATA_BITS + COUNTER_15_COUNTER_BITS + 2,
  RAW_WORDS * u64::BITS as usize
);
const_assert_eq!(
  COUNTER_15_BLOCK_COUNTERS * COUNTER_15_BASE_BITS + COUNTER_15_MAX_CHUNKS * COUNTER_15_CHUNK_BITS,
  COUNTER_15_COUNTER_BITS
);
const_assert_eq!(COUNTER_15_BLOCK_COUNTERS + COUNTER_15_MAX_CHUNKS, COUNTER_15_METADATA_BITS);

/// A 96-byte counter block controlling 64 counters.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable)]
pub struct Counter_Block_15 {
  raw: [u64; RAW_WORDS],
}

unsafe impl Pod for Counter_Block_15 {}

impl Default for Counter_Block_15 {
  fn default() -> Self {
    let mut raw = [0u64; RAW_WORDS];
    raw[0] = (DUMMY_POS as u64) << 32;
    raw[1] = u64::MAX;
    Self { raw }
  }
}

impl Cmov for Counter_Block_15 {
  #[inline(always)]
  fn cmov(&mut self, other: &Self, choice: bool) {
    cmov_raw_words(&mut self.raw, &other.raw, choice);
  }

  #[inline(always)]
  fn cxchg(&mut self, other: &mut Self, choice: bool) {
    cxchg_raw_words(&mut self.raw, &mut other.raw, choice);
  }
}

const_assert_eq!(core::mem::size_of::<Counter_Block_15>(), 96);
const_assert_eq!(core::mem::align_of::<Counter_Block_15>(), 8);

impl Counter_Block_15 {
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
    let (start, width) = self.counter_start_width(index);
    extract_counter_u64(self.counter_words(), start, width)
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
    debug_assert!(selected_index < COUNTER_15_BLOCK_COUNTERS);

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
    debug_assert!(index < COUNTER_15_BLOCK_COUNTERS);

    let metadata = self.metadata_words();
    let previous_rank = index.wrapping_sub(1);
    let previous_delimiter =
      select_usize(select_delimiter(metadata, previous_rank), usize::MAX, index == 0);
    let delimiter = select_delimiter(metadata, index);
    let chunks = delimiter.wrapping_sub(previous_delimiter).wrapping_sub(1);
    let chunks_before = select_usize(previous_delimiter.wrapping_sub(previous_rank), 0, index == 0);
    let used_chunks = last_delimiter(metadata).wrapping_sub(COUNTER_15_BLOCK_COUNTERS - 1);
    let width = COUNTER_15_BASE_BITS + chunks * COUNTER_15_CHUNK_BITS;
    let start = index * COUNTER_15_BASE_BITS + chunks_before * COUNTER_15_CHUNK_BITS;
    let end = start + width;
    let counter = extract_counter_u64(self.counter_words(), start, width);
    let grow = counter == low_bits_mask(width);
    let can_grow = (used_chunks < COUNTER_15_MAX_CHUNKS)
      & (width + COUNTER_15_CHUNK_BITS <= COUNTER_15_MAX_COUNTER_BITS);
    let can_increment = (!grow) | can_grow;
    let grow_enabled = grow & can_increment & enable;

    let metadata =
      insert_zero_bit(self.metadata_words(), delimiter, grow_enabled, METADATA_WORD_MASKS);
    self.set_metadata_words(metadata);

    let counters = insert_zero_bits(
      self.counter_words(),
      end,
      COUNTER_15_CHUNK_BITS,
      grow_enabled,
      COUNTER_WORD_MASKS,
    );
    let counters = add_bit(counters, start, can_increment & enable, COUNTER_WORD_MASKS);
    self.set_counter_words(counters);

    (counter, can_increment)
  }

  #[inline(always)]
  fn counter_start_width(&self, index: usize) -> (usize, usize) {
    debug_assert!(index < COUNTER_15_BLOCK_COUNTERS);

    let metadata = self.metadata_words();
    let previous_rank = index.wrapping_sub(1);
    let previous_delimiter =
      select_usize(select_delimiter(metadata, previous_rank), usize::MAX, index == 0);
    let delimiter = select_delimiter(metadata, index);
    let chunks = delimiter.wrapping_sub(previous_delimiter).wrapping_sub(1);
    let chunks_before = select_usize(previous_delimiter.wrapping_sub(previous_rank), 0, index == 0);
    (
      index * COUNTER_15_BASE_BITS + chunks_before * COUNTER_15_CHUNK_BITS,
      COUNTER_15_BASE_BITS + chunks * COUNTER_15_CHUNK_BITS,
    )
  }

  #[inline(always)]
  fn metadata_words(&self) -> [u64; METADATA_WORDS] {
    extract_region(self.raw, METADATA_BIT_OFFSET, COUNTER_15_METADATA_BITS, METADATA_WORD_MASKS)
  }

  #[inline(always)]
  fn set_metadata_words(&mut self, words: [u64; METADATA_WORDS]) {
    set_region(&mut self.raw, METADATA_BIT_OFFSET, words, COUNTER_15_METADATA_BITS);
  }

  #[inline(always)]
  fn counter_words(&self) -> [u64; COUNTER_WORDS] {
    extract_region(self.raw, COUNTER_BIT_OFFSET, COUNTER_15_COUNTER_BITS, COUNTER_WORD_MASKS)
  }

  #[inline(always)]
  fn set_counter_words(&mut self, words: [u64; COUNTER_WORDS]) {
    set_region(&mut self.raw, COUNTER_BIT_OFFSET, words, COUNTER_15_COUNTER_BITS);
  }
}

/// A 192-byte, 64-byte-aligned ORAM bucket with two 1.5-cacheline blocks.
#[allow(non_camel_case_types)]
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Zeroable)]
pub struct Cacheline_Counter_Bucket_15 {
  /// The two blocks in this ORAM bucket.
  pub blocks: [Counter_Block_15; COUNTER_15_BUCKET_BLOCKS],
}

unsafe impl Pod for Cacheline_Counter_Bucket_15 {}

impl Default for Cacheline_Counter_Bucket_15 {
  fn default() -> Self {
    Self { blocks: [Counter_Block_15::default(); COUNTER_15_BUCKET_BLOCKS] }
  }
}

const_assert_eq!(core::mem::size_of::<Cacheline_Counter_Bucket_15>(), 192);
const_assert_eq!(core::mem::align_of::<Cacheline_Counter_Bucket_15>(), 64);

#[inline(always)]
fn select_delimiter(words: [u64; METADATA_WORDS], rank: usize) -> usize {
  let word0 = words[0];
  let count0 = word0.count_ones() as usize;
  let word1 = words[1];
  let rank1 = rank.wrapping_sub(count0);
  let count1 = word1.count_ones() as usize;
  let rank2 = rank1.wrapping_sub(count1);

  let in0 = rank < count0;
  let in1 = (!in0) & (rank1 < count1);

  let candidate0 = select_nth_one_u64(word0, rank);
  let candidate1 = 64 + select_nth_one_u64(word1, rank1);
  let candidate2 = 128 + select_nth_one_u64(words[2] & METADATA_WORD_MASKS[2], rank2);

  let selected = select_usize(candidate2, candidate1, in1);
  select_usize(selected, candidate0, in0)
}

#[inline(always)]
fn last_delimiter(words: [u64; METADATA_WORDS]) -> usize {
  let word0 = words[0];
  let word1 = words[1];
  let word2 = words[2] & METADATA_WORD_MASKS[2];

  let selected = last_one_with_offset(word0, 0);
  let selected = select_usize(selected, last_one_with_offset(word1, 64), word1 != 0);
  select_usize(selected, last_one_with_offset(word2, 128), word2 != 0)
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
const fn insert_zero_bits<const N: usize>(
  mut words: [u64; N],
  bit_index: usize,
  count: usize,
  enable: bool,
  valid_masks: [u64; N],
) -> [u64; N] {
  let mut i = 0;
  while i < count {
    words = insert_zero_bit(words, bit_index, enable, valid_masks);
    i += 1;
  }
  words
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
fn extract_region<const N: usize>(
  raw: [u64; RAW_WORDS],
  start: usize,
  len: usize,
  masks: [u64; N],
) -> [u64; N] {
  let mut out = [0u64; N];
  copy_bits(&mut out, 0, raw, start, len, masks);
  out
}

#[inline(always)]
fn set_region<const N: usize>(
  raw: &mut [u64; RAW_WORDS],
  start: usize,
  words: [u64; N],
  len: usize,
) {
  copy_bits(raw, start, words, 0, len, RAW_WORD_MASKS);
}

#[inline(always)]
fn copy_bits<const DST_N: usize, const SRC_N: usize>(
  dst: &mut [u64; DST_N],
  dst_start: usize,
  src: [u64; SRC_N],
  src_start: usize,
  len: usize,
  dst_masks: [u64; DST_N],
) {
  let dst_end = dst_start + len;
  let mut i = 0;

  while i < DST_N {
    let word_start = i * u64::BITS as usize;
    let word_end = word_start + u64::BITS as usize;
    let write_start = max_usize(dst_start, word_start);
    let write_end = min_usize(dst_end, word_end);
    let has_overlap = write_start < write_end;
    let width = write_end.wrapping_sub(write_start) & mask_usize(has_overlap);
    let shift = write_start.wrapping_sub(word_start) & 63;
    let src_bit = src_start.wrapping_add(write_start).wrapping_sub(dst_start);
    let value = extract_bits(src, src_bit, width).wrapping_shl(shift as u32);
    let mask = low_bits_mask(width).wrapping_shl(shift as u32);
    dst[i] = (dst[i] & !mask) | (value & mask);

    dst[i] &= dst_masks[i];
    i += 1;
  }
}

#[inline(always)]
fn extract_bits<const N: usize>(words: [u64; N], start: usize, width: usize) -> u64 {
  let word_index = start >> 6;
  let shift = start & 63;
  let low = word_or_zero(words, word_index);
  let high = word_or_zero(words, word_index + 1);

  shr_pair(low, high, shift) & low_bits_mask(width)
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
  let dst_vec = unsafe { _mm512_loadu_si512(dst.as_ptr() as *const __m512i) };
  let src_vec = unsafe { _mm512_loadu_si512(src.as_ptr() as *const __m512i) };
  let out = _mm512_mask_mov_epi64(dst_vec, mask, src_vec);
  unsafe {
    _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, out);
  }
  cmov_tail_words(dst, src, choice);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn cxchg_raw_words_avx512(a: &mut [u64; RAW_WORDS], b: &mut [u64; RAW_WORDS], choice: bool) {
  let mask = avx512_lane_mask(choice);
  let a_vec = unsafe { _mm512_loadu_si512(a.as_ptr() as *const __m512i) };
  let b_vec = unsafe { _mm512_loadu_si512(b.as_ptr() as *const __m512i) };
  let new_a = _mm512_mask_mov_epi64(a_vec, mask, b_vec);
  let new_b = _mm512_mask_mov_epi64(b_vec, mask, a_vec);
  unsafe {
    _mm512_storeu_si512(a.as_mut_ptr() as *mut __m512i, new_a);
    _mm512_storeu_si512(b.as_mut_ptr() as *mut __m512i, new_b);
  }
  cxchg_tail_words(a, b, choice);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn avx512_lane_mask(choice: bool) -> __mmask8 {
  0u8.wrapping_sub(choice as u8) as __mmask8
}

#[inline(always)]
fn cmov_tail_words(dst: &mut [u64; RAW_WORDS], src: &[u64; RAW_WORDS], choice: bool) {
  let mask = mask_u64(choice);
  let mut i = 8;
  while i < RAW_WORDS {
    dst[i] = (dst[i] & !mask) | (src[i] & mask);
    i += 1;
  }
}

#[inline(always)]
fn cxchg_tail_words(a: &mut [u64; RAW_WORDS], b: &mut [u64; RAW_WORDS], choice: bool) {
  let mask = mask_u64(choice);
  let mut i = 8;
  while i < RAW_WORDS {
    let delta = (a[i] ^ b[i]) & mask;
    a[i] ^= delta;
    b[i] ^= delta;
    i += 1;
  }
}

#[inline(always)]
const fn min_usize(a: usize, b: usize) -> usize {
  select_usize(b, a, a < b)
}

#[inline(always)]
const fn max_usize(a: usize, b: usize) -> usize {
  select_usize(b, a, a > b)
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
  fn layout_math_adds_up() {
    assert_eq!(core::mem::size_of::<Counter_Block_15>(), 96);
    assert_eq!(core::mem::align_of::<Counter_Block_15>(), 8);
    assert_eq!(core::mem::size_of::<Cacheline_Counter_Bucket_15>(), 192);
    assert_eq!(core::mem::align_of::<Cacheline_Counter_Bucket_15>(), 64);
    assert_eq!(KEY_POS_BITS + COUNTER_15_METADATA_BITS + COUNTER_15_COUNTER_BITS, 766);
  }

  #[test]
  fn counters_start_at_zero_and_use_five_base_bits() {
    let mut block = Counter_Block_15::default();

    for index in 0..COUNTER_15_BLOCK_COUNTERS {
      assert_eq!(block.get_counter(index), 0);
    }
    for value in 0..32 {
      assert_eq!(block.get_and_increment_counter(7), (value, true));
    }
    assert_eq!(block.get_counter(7), 32);
  }

  #[test]
  fn counters_match_reference_after_mixed_increment_patterns() {
    let mut block = Counter_Block_15::default();
    let mut reference = [0u64; COUNTER_15_BLOCK_COUNTERS];

    for step in 0..512 {
      let index = (step * 37 + 11) & (COUNTER_15_BLOCK_COUNTERS - 1);
      let (old, incremented) = block.get_and_increment_counter(index);
      assert!(incremented);
      assert_eq!(old, reference[index]);
      reference[index] += 1;

      let read_index = (step * 19 + 7) & (COUNTER_15_BLOCK_COUNTERS - 1);
      assert_eq!(block.get_counter(read_index), reference[read_index]);
    }
  }

  #[test]
  fn counter_width_caps_at_sixty_three_bits() {
    let mut block = Counter_Block_15::default();
    let mut metadata = [0u64; METADATA_WORDS];
    // Counter 0 gets 29 extra chunks: width = 5 + 29 * 2 = 63.
    metadata[0] = 1u64 << 29;
    for bit in 30..COUNTER_15_BLOCK_COUNTERS + 29 {
      metadata[bit / 64] |= 1u64 << (bit & 63);
    }
    block.set_metadata_words(metadata);
    let mut counters = [0u64; COUNTER_WORDS];
    counters[0] = low_bits_mask(63);
    block.set_counter_words(counters);

    assert_eq!(block.get_and_increment_counter(0), (low_bits_mask(63), false));
    assert_eq!(block.get_counter(0), low_bits_mask(63));
  }

  #[test]
  fn key_pos_and_empty_round_trip() {
    let mut block = Counter_Block_15::default();

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
