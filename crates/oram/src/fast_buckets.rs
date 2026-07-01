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
//! A bucket is conceptually two `Counter_Block`s. Each block is also one
//! 64-byte, 64-byte-aligned cache line, owns one `(key, pos)` pair, and controls
//! 32 of the bucket's 64 counters. A block keeps its metadata and counter
//! payload at the same bit offsets used by a bucket; its second 8-byte word is
//! `[metadata_bits_len, counter_bits_len]` instead of the bucket's second
//! `(key, pos)` slot. Merging two blocks copies the first block as a bucket,
//! appends the second block's metadata and counters at those recorded lengths,
//! then overwrites the second 8-byte word with the second block's `(key, pos)`.
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

/// Number of `(key, pos)` pairs in a cache-line counter bucket.
pub const CACHELINE_COUNTER_BUCKET_BLOCKS: usize = 2;
/// Number of `u32` words used by the key/position slots.
pub const CACHELINE_COUNTER_BUCKET_BLOCK_WORDS: usize = 4;
/// Number of counters packed into the bucket.
pub const CACHELINE_COUNTER_BUCKET_COUNTERS: usize = 64;
/// Number of counters controlled by one cache-line counter block.
pub const CACHELINE_COUNTER_BLOCK_COUNTERS: usize = 32;
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
const_assert_eq!(
  CACHELINE_COUNTER_BLOCK_COUNTERS * CACHELINE_COUNTER_BUCKET_BLOCKS,
  CACHELINE_COUNTER_BUCKET_COUNTERS
);

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

impl Cmov for Cacheline_Counter_Bucket {
  #[inline(always)]
  fn cmov(&mut self, other: &Self, choice: bool) {
    cmov_raw_words(&mut self.raw, &other.raw, choice);
  }

  #[inline(always)]
  fn cxchg(&mut self, other: &mut Self, choice: bool) {
    cxchg_raw_words(&mut self.raw, &mut other.raw, choice);
  }
}

impl Default for Cacheline_Counter_Bucket {
  fn default() -> Self {
    let mut raw = [0u64; RAW_WORDS];
    raw[0] = (DUMMY_POS as u64) << 32;
    raw[1] = (DUMMY_POS as u64) << 32;
    // Width zero for every counter: the first 64 metadata bits are delimiters
    // and the remaining 160 metadata zeros are spare capacity.
    raw[METADATA_BIT_OFFSET / u64::BITS as usize] = u64::MAX;
    Self { raw }
  }
}

const_assert_eq!(core::mem::size_of::<Cacheline_Counter_Bucket>(), 64);
const_assert_eq!(core::mem::align_of::<Cacheline_Counter_Bucket>(), 64);

/// A 64-byte, cache-line-aligned half-bucket block.
///
/// Layout:
/// * bits `0..64`: one `(key, pos)` pair, interpreted as `[key, pos]`;
/// * bits `64..128`: `[metadata_bits_len, counter_bits_len]`;
/// * bits `128..352`: local metadata for 32 counters;
/// * bits `352..512`: local packed counter payload bits.
///
/// The metadata stream is the same unary delimiter encoding as the bucket, but
/// local to 32 counters. `metadata_bits_len` is `32 + counter_bits_len`.
#[allow(non_camel_case_types)]
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Zeroable)]
pub struct Counter_Block {
  raw: [u64; RAW_WORDS],
}

unsafe impl Pod for Counter_Block {}

impl Cmov for Counter_Block {
  #[inline(always)]
  fn cmov(&mut self, other: &Self, choice: bool) {
    cmov_raw_words(&mut self.raw, &other.raw, choice);
  }

  #[inline(always)]
  fn cxchg(&mut self, other: &mut Self, choice: bool) {
    cxchg_raw_words(&mut self.raw, &mut other.raw, choice);
  }
}

impl Default for Counter_Block {
  fn default() -> Self {
    let mut raw = [0u64; RAW_WORDS];
    raw[0] = (DUMMY_POS as u64) << 32;
    raw[1] = CACHELINE_COUNTER_BLOCK_COUNTERS as u64;
    raw[METADATA_BIT_OFFSET / u64::BITS as usize] = low_bits_mask(CACHELINE_COUNTER_BLOCK_COUNTERS);
    Self { raw }
  }
}

const_assert_eq!(core::mem::size_of::<Counter_Block>(), 64);
const_assert_eq!(core::mem::align_of::<Counter_Block>(), 64);

impl Counter_Block {
  /// Creates an empty block with all local counters set to zero.
  #[inline]
  pub fn new() -> Self {
    Self::default()
  }

  /// Returns the block's key.
  #[inline]
  pub const fn key(&self) -> u32 {
    self.raw[0] as u32
  }

  /// Returns the block's position.
  #[inline]
  pub const fn pos(&self) -> PositionType {
    (self.raw[0] >> 32) as u32
  }

  /// Sets the block's key and position.
  #[inline]
  pub fn set_key_pos(&mut self, key: u32, pos: PositionType) {
    self.raw[0] = key as u64 | ((pos as u64) << 32);
  }

  /// Conditionally marks this block as empty by setting only the position.
  #[inline(always)]
  pub fn cmov_empty(&mut self, choice: bool) {
    let empty_raw0 = (self.raw[0] & U32_MASK) | ((DUMMY_POS as u64) << 32);
    self.raw[0] = select_u64(self.raw[0], empty_raw0, choice);
  }

  /// Returns whether this block is an empty ORAM slot.
  #[inline]
  pub const fn is_empty(&self) -> bool {
    self.pos() == DUMMY_POS
  }

  /// Returns the logical counter value at a local suffix in `0..32`.
  #[inline(always)]
  pub fn get_counter(&self, index: usize) -> u64 {
    let (start, end) = self.get_counter_endpoints(index, index + 1);
    let width = end - start;
    let stored = extract_counter_u64(self.counter_words(), start, width);
    select_u64(0, stored.wrapping_add(1), width != 0)
  }

  /// Returns the local payload bit range containing counters in `start_index..end_index`.
  #[inline(always)]
  pub fn get_counter_endpoints(&self, start_index: usize, end_index: usize) -> (usize, usize) {
    debug_assert!(start_index <= end_index);
    debug_assert!(end_index <= CACHELINE_COUNTER_BLOCK_COUNTERS);

    let words = self.metadata_words();
    let start = payload_end_for_index(words, start_index);
    let end = payload_end_for_index(words, end_index);

    (start, end)
  }

  /// Increments a local counter and returns whether the packed block had capacity.
  #[inline(always)]
  pub fn increment_counter(&mut self, index: usize) -> bool {
    self.get_and_increment_counter(index).1
  }

  /// Returns a local counter's old value and then increments it.
  #[inline(always)]
  pub fn get_and_increment_counter(&mut self, index: usize) -> (u64, bool) {
    self.get_and_increment_counter_if(index, true)
  }

  /// Reads or increments `selected_index` after the containing block has been selected.
  #[inline(always)]
  pub fn access_counter_oblivious(&mut self, selected_index: usize, increment: bool) -> u64 {
    debug_assert!(selected_index < CACHELINE_COUNTER_BLOCK_COUNTERS);

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
    debug_assert!(index < CACHELINE_COUNTER_BLOCK_COUNTERS);

    let words = self.metadata_words();
    let previous_rank = index.wrapping_sub(1);
    let previous_delimiter =
      select_usize(select_delimiter(words, previous_rank), usize::MAX, index == 0);
    let delimiter = select_delimiter(words, index);
    let width = delimiter.wrapping_sub(previous_delimiter).wrapping_sub(1);
    let end = delimiter.wrapping_sub(index);
    let start = end.wrapping_sub(width);
    let used_bits = self.counter_bits_len();
    let counter = extract_counter_u64(self.counter_words(), start, width);
    let old_value = select_u64(0, counter.wrapping_add(1), width != 0);
    let grow = (width == 0) | (counter == low_bits_mask(width));
    let has_free_bit =
      used_bits < CACHELINE_COUNTER_BUCKET_COUNTER_BITS / CACHELINE_COUNTER_BUCKET_BLOCKS;
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
    self.set_lengths(
      self.metadata_bits_len() + usize::from(grow_enabled),
      self.counter_bits_len() + usize::from(grow_enabled),
    );

    (old_value, can_increment)
  }

  /// Returns the number of valid local metadata bits.
  #[inline]
  pub fn metadata_bits_len(&self) -> usize {
    self.raw[1] as u32 as usize
  }

  /// Returns the number of valid local counter payload bits.
  #[inline]
  pub fn counter_bits_len(&self) -> usize {
    (self.raw[1] >> 32) as u32 as usize
  }

  #[inline(always)]
  fn set_lengths(&mut self, metadata_bits_len: usize, counter_bits_len: usize) {
    debug_assert!(metadata_bits_len <= CACHELINE_COUNTER_BUCKET_METADATA_BITS);
    debug_assert!(counter_bits_len <= CACHELINE_COUNTER_BUCKET_COUNTER_BITS);
    self.raw[1] = metadata_bits_len as u64 | ((counter_bits_len as u64) << 32);
  }

  #[inline(always)]
  const fn metadata_words(&self) -> [u64; METADATA_WORDS] {
    [self.raw[2], self.raw[3], self.raw[4], self.raw[5] & U32_MASK]
  }

  #[inline(always)]
  const fn set_metadata_words(&mut self, words: [u64; METADATA_WORDS]) {
    self.raw[2] = words[0];
    self.raw[3] = words[1];
    self.raw[4] = words[2];
    self.raw[5] = (self.raw[5] & !U32_MASK) | (words[3] & U32_MASK);
  }

  #[inline(always)]
  const fn counter_words(&self) -> [u64; COUNTER_WORDS] {
    [
      (self.raw[5] >> 32) | (self.raw[6] << 32),
      (self.raw[6] >> 32) | (self.raw[7] << 32),
      self.raw[7] >> 32,
    ]
  }

  #[inline(always)]
  const fn set_counter_words(&mut self, words: [u64; COUNTER_WORDS]) {
    self.raw[5] = (self.raw[5] & U32_MASK) | (words[0] << 32);
    self.raw[6] = (words[0] >> 32) | (words[1] << 32);
    self.raw[7] = (words[1] >> 32) | ((words[2] & U32_MASK) << 32);
  }
}

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

  /// Splits this bucket into the two cache-line blocks that own its counter halves.
  #[inline(always)]
  pub fn split_into_blocks(&self) -> [Counter_Block; CACHELINE_COUNTER_BUCKET_BLOCKS] {
    let metadata = self.metadata_words();
    let first_counter_bits_len = select_delimiter(metadata, CACHELINE_COUNTER_BLOCK_COUNTERS - 1)
      - (CACHELINE_COUNTER_BLOCK_COUNTERS - 1);
    let total_counter_bits_len = select_delimiter(metadata, CACHELINE_COUNTER_BUCKET_COUNTERS - 1)
      - (CACHELINE_COUNTER_BUCKET_COUNTERS - 1);
    let second_counter_bits_len = total_counter_bits_len - first_counter_bits_len;
    let first_metadata_bits_len = CACHELINE_COUNTER_BLOCK_COUNTERS + first_counter_bits_len;
    let second_metadata_bits_len = CACHELINE_COUNTER_BLOCK_COUNTERS + second_counter_bits_len;

    let counters = self.counter_words();

    let mut first = Counter_Block { raw: [0u64; RAW_WORDS] };
    first.raw[0] = self.raw[0];
    first.set_lengths(first_metadata_bits_len, first_counter_bits_len);
    first.set_metadata_words(prefix_bits(metadata, first_metadata_bits_len, METADATA_WORD_MASKS));
    first.set_counter_words(prefix_bits(counters, first_counter_bits_len, COUNTER_WORD_MASKS));

    let mut second = Counter_Block { raw: [0u64; RAW_WORDS] };
    second.raw[0] = self.raw[1];
    second.set_lengths(second_metadata_bits_len, second_counter_bits_len);
    second.set_metadata_words(shr_words(metadata, first_metadata_bits_len, METADATA_WORD_MASKS));
    second.set_counter_words(shr_words(counters, first_counter_bits_len, COUNTER_WORD_MASKS));

    [first, second]
  }

  /// Merges two cache-line blocks into one bucket.
  #[inline(always)]
  pub fn merge_blocks(blocks: [Counter_Block; CACHELINE_COUNTER_BUCKET_BLOCKS]) -> Self {
    let first = blocks[0];
    let second = blocks[1];
    let first_metadata_bits_len = first.metadata_bits_len();
    let second_metadata_bits_len = second.metadata_bits_len();
    let first_counter_bits_len = first.counter_bits_len();
    let second_counter_bits_len = second.counter_bits_len();

    debug_assert_eq!(
      first_metadata_bits_len,
      CACHELINE_COUNTER_BLOCK_COUNTERS + first_counter_bits_len
    );
    debug_assert_eq!(
      second_metadata_bits_len,
      CACHELINE_COUNTER_BLOCK_COUNTERS + second_counter_bits_len
    );
    debug_assert!(
      first_metadata_bits_len + second_metadata_bits_len <= CACHELINE_COUNTER_BUCKET_METADATA_BITS
    );
    debug_assert!(
      first_counter_bits_len + second_counter_bits_len <= CACHELINE_COUNTER_BUCKET_COUNTER_BITS
    );

    let mut bucket = Self { raw: first.raw };

    let mut metadata = bucket.metadata_words();
    copy_bits(
      &mut metadata,
      first_metadata_bits_len,
      second.metadata_words(),
      0,
      second_metadata_bits_len,
      METADATA_WORD_MASKS,
    );
    bucket.set_metadata_words(metadata);

    let mut counters = bucket.counter_words();
    copy_bits(
      &mut counters,
      first_counter_bits_len,
      second.counter_words(),
      0,
      second_counter_bits_len,
      COUNTER_WORD_MASKS,
    );
    bucket.set_counter_words(counters);

    bucket.raw[1] = second.raw[0];
    bucket
  }

  /// Returns the logical counter value at `index`.
  ///
  /// Width-zero counters return 0. Nonzero-width counters return the packed
  /// payload value plus 1.
  ///
  /// # Memory access
  /// This reads metadata and payload from the single cache line backing the
  /// bucket. The optimized metadata selector branches within that cache line.
  #[inline(always)]
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
  /// # Memory access
  /// This reads metadata from the single cache line backing the bucket. The
  /// optimized metadata selector branches within that cache line.
  #[inline(always)]
  pub fn get_counter_endpoints(&self, start_index: usize, end_index: usize) -> (usize, usize) {
    debug_assert!(start_index <= end_index);
    debug_assert!(end_index <= CACHELINE_COUNTER_BUCKET_COUNTERS);

    let words = self.metadata_words();
    let start = payload_end_for_index(words, start_index);
    let end = payload_end_for_index(words, end_index);

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
  /// # Memory access
  /// This updates metadata and payload inside the single cache line backing the
  /// bucket. The optimized metadata selector branches within that cache line.
  #[inline(always)]
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

  #[inline(always)]
  fn counter_location(&self, index: usize) -> CounterLocation {
    debug_assert!(index < CACHELINE_COUNTER_BUCKET_COUNTERS);

    let words = self.metadata_words();
    let previous_rank = index.wrapping_sub(1);
    let previous_delimiter =
      select_usize(select_delimiter(words, previous_rank), usize::MAX, index == 0);
    let delimiter = select_delimiter(words, index);

    let width = delimiter.wrapping_sub(previous_delimiter).wrapping_sub(1);
    let end = delimiter.wrapping_sub(index);
    let start = end.wrapping_sub(width);
    let used_bits = last_delimiter(words).wrapping_sub(CACHELINE_COUNTER_BUCKET_COUNTERS - 1);

    CounterLocation { start, width, delimiter, used_bits }
  }

  #[inline(always)]
  fn counter_at(&self, start: usize, width: usize) -> u64 {
    extract_counter_u64(self.counter_words(), start, width)
  }

  #[inline(always)]
  const fn metadata_words(&self) -> [u64; METADATA_WORDS] {
    [self.raw[2], self.raw[3], self.raw[4], self.raw[5] & U32_MASK]
  }

  #[inline(always)]
  const fn set_metadata_words(&mut self, words: [u64; METADATA_WORDS]) {
    self.raw[2] = words[0];
    self.raw[3] = words[1];
    self.raw[4] = words[2];
    self.raw[5] = (self.raw[5] & !U32_MASK) | (words[3] & U32_MASK);
  }

  #[inline(always)]
  const fn counter_words(&self) -> [u64; COUNTER_WORDS] {
    [
      (self.raw[5] >> 32) | (self.raw[6] << 32),
      (self.raw[6] >> 32) | (self.raw[7] << 32),
      self.raw[7] >> 32,
    ]
  }

  #[inline(always)]
  const fn set_counter_words(&mut self, words: [u64; COUNTER_WORDS]) {
    self.raw[5] = (self.raw[5] & U32_MASK) | (words[0] << 32);
    self.raw[6] = (words[0] >> 32) | (words[1] << 32);
    self.raw[7] = (words[1] >> 32) | ((words[2] & U32_MASK) << 32);
  }
}

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
  let candidate3 = 192 + select_nth_one_u64(words[3] & U32_MASK, rank3);

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
  let word3 = words[3] & U32_MASK;

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
fn prefix_bits<const N: usize>(words: [u64; N], len: usize, valid_masks: [u64; N]) -> [u64; N] {
  let mut out = [0u64; N];
  let mut i = 0;

  while i < N {
    let word_start = i * u64::BITS as usize;
    let word_end = word_start + u64::BITS as usize;
    let full_word = len >= word_end;
    let partial_word = (!full_word) & (len > word_start);
    let partial = words[i] & low_bits_mask(len.wrapping_sub(word_start));
    let selected = select_u64(0, partial, partial_word);
    out[i] = select_u64(selected, words[i], full_word) & valid_masks[i];
    i += 1;
  }

  out
}

#[inline(always)]
fn shr_words<const N: usize>(words: [u64; N], shift: usize, valid_masks: [u64; N]) -> [u64; N] {
  let mut out = [0u64; N];
  let word_shift = shift >> 6;
  let bit_shift = shift & 63;
  let mut i = 0;

  while i < N {
    let low = word_or_zero(words, i + word_shift).wrapping_shr(bit_shift as u32);
    let high = word_or_zero(words, i + word_shift + 1)
      .wrapping_shl((64usize.wrapping_sub(bit_shift) & 63) as u32)
      & mask_u64(bit_shift != 0);
    out[i] = (low | high) & valid_masks[i];
    i += 1;
  }

  out
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
fn extract_bits<const N: usize>(words: [u64; N], start: usize, width: usize) -> u64 {
  let word_index = start >> 6;
  let shift = start & 63;
  let low = word_or_zero(words, word_index);
  let high = word_or_zero(words, word_index + 1);

  shr_pair(low, high, shift) & low_bits_mask(width)
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
  fn cacheline_bucket_layout_is_one_cacheline() {
    assert_eq!(core::mem::size_of::<Cacheline_Counter_Bucket>(), 64);
    assert_eq!(core::mem::align_of::<Cacheline_Counter_Bucket>(), 64);
    assert_eq!(core::mem::size_of::<Counter_Block>(), 64);
    assert_eq!(core::mem::align_of::<Counter_Block>(), 64);
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
  fn counter_block_header_round_trips() {
    let mut block = Counter_Block::default();

    assert_eq!(block.metadata_bits_len(), CACHELINE_COUNTER_BLOCK_COUNTERS);
    assert_eq!(block.counter_bits_len(), 0);
    assert_eq!(block.metadata_words()[0], low_bits_mask(CACHELINE_COUNTER_BLOCK_COUNTERS));

    block.set_key_pos(29, 31);
    assert_eq!(block.key(), 29);
    assert_eq!(block.pos(), 31);
  }

  #[test]
  fn default_blocks_merge_to_default_bucket() {
    let bucket =
      Cacheline_Counter_Bucket::merge_blocks([Counter_Block::default(), Counter_Block::default()]);

    assert_eq!(bucket.raw, Cacheline_Counter_Bucket::default().raw);
  }

  #[test]
  fn split_blocks_record_half_lengths_and_keys() {
    let mut bucket = Cacheline_Counter_Bucket::default();
    bucket.set_key_pos(0, 7, 11);
    bucket.set_key_pos(1, 13, 17);

    for _ in 0..5 {
      assert!(bucket.increment_counter(0));
    }
    assert!(bucket.increment_counter(31));
    assert!(bucket.increment_counter(32));
    for _ in 0..3 {
      assert!(bucket.increment_counter(63));
    }

    let blocks = bucket.split_into_blocks();

    assert_eq!(blocks[0].key(), 7);
    assert_eq!(blocks[0].pos(), 11);
    assert_eq!(blocks[1].key(), 13);
    assert_eq!(blocks[1].pos(), 17);
    assert_eq!(blocks[0].counter_bits_len(), 4);
    assert_eq!(blocks[0].metadata_bits_len(), CACHELINE_COUNTER_BLOCK_COUNTERS + 4);
    assert_eq!(blocks[1].counter_bits_len(), 3);
    assert_eq!(blocks[1].metadata_bits_len(), CACHELINE_COUNTER_BLOCK_COUNTERS + 3);
  }

  #[test]
  fn split_then_merge_preserves_bucket() {
    let mut bucket = Cacheline_Counter_Bucket::default();
    bucket.set_block_words([101, 103, 107, 109]);

    for i in 0..CACHELINE_COUNTER_BUCKET_COUNTERS {
      let increments = (i * 13 + 7) & 7;
      for _ in 0..increments {
        assert!(bucket.increment_counter(i));
      }
    }

    let merged = Cacheline_Counter_Bucket::merge_blocks(bucket.split_into_blocks());

    assert_eq!(merged.raw, bucket.raw);
    for i in 0..CACHELINE_COUNTER_BUCKET_COUNTERS {
      assert_eq!(merged.get_counter(i), bucket.get_counter(i));
    }
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

  #[test]
  fn counters_match_reference_after_mixed_increment_patterns() {
    let mut bucket = Cacheline_Counter_Bucket::default();
    let mut reference = [0u64; CACHELINE_COUNTER_BUCKET_COUNTERS];

    for round in 0..4 {
      for step in 0..CACHELINE_COUNTER_BUCKET_COUNTERS {
        let index = (step * 17 + round * 11) & 63;
        assert!(bucket.increment_counter(index));
        reference[index] += 1;

        for (counter_index, expected) in reference.iter().enumerate() {
          assert_eq!(bucket.get_counter(counter_index), *expected);
        }
      }
    }

    let mut expected_start = 0usize;
    for (counter_index, expected) in reference.iter().enumerate() {
      let width = if *expected == 0 {
        0
      } else {
        let stored = *expected - 1;
        usize::max(1, u64::BITS as usize - stored.leading_zeros() as usize)
      };
      assert_eq!(
        bucket.get_counter_endpoints(counter_index, counter_index + 1),
        (expected_start, expected_start + width)
      );
      expected_start += width;
    }
    assert_eq!(
      bucket.get_counter_endpoints(0, CACHELINE_COUNTER_BUCKET_COUNTERS),
      (0, expected_start)
    );
  }
}
