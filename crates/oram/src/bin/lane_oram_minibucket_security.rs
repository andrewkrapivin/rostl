//! Security simulator for Circuit-style fixed lanes with Y-slot minibuckets.
//!
//! Each physical bucket is split into `Z` fixed lanes, each containing `Y`
//! pooled slots. Every lane performs one non-oblivious `EvictOnceSlow` chain:
//! the shared stash and that lane's root minibucket form level zero, and at
//! most one block is carried down the lane at a time. This is a metadata-only
//! reference transition; payload bytes do not affect placement.

use std::{collections::BTreeSet, env, fs, process, str::FromStr, time::Instant};

const EMPTY_KEY: u32 = u32::MAX;
const OS_RANDOM_BATCH_WORDS: usize = 1 << 20;

#[derive(Clone, Copy, Debug)]
struct SimUpdate {
  found: bool,
  insertion_demand: usize,
  stash_after: usize,
  overflowed: bool,
}

#[derive(Clone, Copy, Debug)]
struct Level {
  offset: usize,
  path_mask: usize,
  path_shift: u32,
  subtree_span: usize,
}

#[derive(Clone, Copy, Debug)]
enum Source {
  Stash { key: u32, depth: usize },
  Path { index: usize, level: usize, key: u32, depth: usize },
}

impl Source {
  const fn key(self) -> u32 {
    match self {
      Self::Stash { key, .. } | Self::Path { key, .. } => key,
    }
  }

  const fn depth(self) -> usize {
    match self {
      Self::Stash { depth, .. } | Self::Path { depth, .. } => depth,
    }
  }

  fn is_better_than(self, other: Option<Self>) -> bool {
    other.is_none_or(|other| {
      self.depth() > other.depth() || (self.depth() == other.depth() && self.key() < other.key())
    })
  }
}

/// Dynamic set of stashed keys supporting the Circuit ORAM deepest-key tie
/// rule: maximize legal depth, then minimize logical key.
#[derive(Debug)]
struct StashIndex {
  keys_by_label: Vec<BTreeSet<u32>>,
  range_min: Vec<u32>,
  leaf_count: usize,
  len: usize,
}

impl StashIndex {
  fn new(leaf_count: usize) -> Self {
    debug_assert!(leaf_count.is_power_of_two());
    Self {
      keys_by_label: (0..leaf_count).map(|_| BTreeSet::new()).collect(),
      range_min: vec![EMPTY_KEY; leaf_count * 2],
      leaf_count,
      len: 0,
    }
  }

  const fn len(&self) -> usize {
    self.len
  }

  fn contains(&self, label: u32, key: u32) -> bool {
    self.keys_by_label[label as usize].contains(&key)
  }

  fn insert(&mut self, label: u32, key: u32) {
    assert!(self.keys_by_label[label as usize].insert(key), "duplicate stash key {key}");
    self.len += 1;
    self.refresh_label(label as usize);
  }

  fn remove(&mut self, label: u32, key: u32) {
    assert!(self.keys_by_label[label as usize].remove(&key), "missing stash key {key}");
    self.len -= 1;
    self.refresh_label(label as usize);
  }

  fn refresh_label(&mut self, label: usize) {
    let mut index = self.leaf_count + label;
    self.range_min[index] = self.keys_by_label[label].first().copied().unwrap_or(EMPTY_KEY);
    while index > 1 {
      index /= 2;
      self.range_min[index] = self.range_min[index * 2].min(self.range_min[index * 2 + 1]);
    }
  }

  fn minimum_key(&self, mut start: usize, mut end: usize) -> Option<u32> {
    debug_assert!(start < end && end <= self.leaf_count);
    start += self.leaf_count;
    end += self.leaf_count;
    let mut result = EMPTY_KEY;
    while start < end {
      if start & 1 != 0 {
        result = result.min(self.range_min[start]);
        start += 1;
      }
      if end & 1 != 0 {
        end -= 1;
        result = result.min(self.range_min[end]);
      }
      start /= 2;
      end /= 2;
    }
    (result != EMPTY_KEY).then_some(result)
  }
}

#[derive(Debug)]
struct MinibucketLaneSimulator {
  max_n: usize,
  height: usize,
  z: usize,
  y: usize,
  bucket_slots: usize,
  stash_capacity: usize,
  levels: Vec<Level>,
  tree: Vec<u32>,
  path: Vec<u32>,
  stashes: Vec<StashIndex>,
  random_queues: bool,
  eviction_policy: EvictionPolicy,
  eviction_chains: usize,
  labels: Vec<u32>,
  initialized: Vec<bool>,
}

impl MinibucketLaneSimulator {
  fn new(max_n: usize, z: usize, y: usize, b: usize, stash_capacity: usize) -> Self {
    assert!(max_n > 0 && max_n.is_power_of_two());
    assert!(z > 0 && y > 0 && stash_capacity > 0);
    assert!(b >= 2 && b.is_power_of_two());

    let mut rounded_max_n = 1usize;
    let mut height = 1usize;
    while rounded_max_n < max_n {
      rounded_max_n *= b;
      height += 1;
    }
    assert_eq!(rounded_max_n, max_n, "N must be an exact power of B");

    let bits_per_digit = b.trailing_zeros();
    let mut levels = Vec::with_capacity(height);
    let mut node_count = 0usize;
    let mut level_width = 1usize;
    for depth in 0..height {
      levels.push(Level {
        offset: node_count,
        path_mask: level_width - 1,
        path_shift: ((height - 1 - depth) as u32) * bits_per_digit,
        subtree_span: max_n / level_width,
      });
      node_count += level_width;
      level_width *= b;
    }

    let bucket_slots = z.checked_mul(y).expect("Z*Y overflow");
    Self {
      max_n,
      height,
      z,
      y,
      bucket_slots,
      stash_capacity,
      levels,
      tree: vec![EMPTY_KEY; node_count * bucket_slots],
      path: vec![EMPTY_KEY; height * bucket_slots],
      stashes: std::iter::once_with(|| StashIndex::new(max_n)).collect(),
      random_queues: false,
      eviction_policy: EvictionPolicy::Fixed,
      eviction_chains: 1,
      labels: vec![0; max_n],
      initialized: vec![false; max_n],
    }
  }

  fn with_random_queues(mut self) -> Self {
    self.stashes = (0..self.z).map(|_| StashIndex::new(self.max_n)).collect();
    self.random_queues = true;
    self
  }

  fn with_eviction_strategy(mut self, policy: EvictionPolicy, chains: usize) -> Self {
    assert!(chains > 0);
    self.eviction_policy = policy;
    self.eviction_chains = chains;
    self
  }

  fn stash_len(&self) -> usize {
    self.stashes.iter().map(StashIndex::len).sum()
  }

  const fn max_n(&self) -> usize {
    self.max_n
  }

  fn update(&mut self, old_pos: u32, new_pos: u32, key: u32) -> SimUpdate {
    self.update_in_queue(old_pos, new_pos, key, 0)
  }

  fn update_in_queue(&mut self, old_pos: u32, new_pos: u32, key: u32, queue: usize) -> SimUpdate {
    self.update_with_schedule(old_pos, new_pos, key, queue, true, &[])
  }

  fn update_with_schedule(
    &mut self,
    old_pos: u32,
    new_pos: u32,
    key: u32,
    queue: usize,
    evict_accessed_path: bool,
    background_paths: &[u32],
  ) -> SimUpdate {
    assert!((old_pos as usize) < self.max_n);
    assert!((new_pos as usize) < self.max_n);
    assert!((key as usize) < self.labels.len());
    if self.initialized[key as usize] {
      debug_assert_eq!(self.labels[key as usize], old_pos);
    }

    self.read_path(old_pos);
    let found = self.remove_key(key);
    self.labels[key as usize] = new_pos;
    self.initialized[key as usize] = true;

    let insertion_demand = self.stash_len() + 1;
    if insertion_demand > self.stash_capacity {
      return SimUpdate {
        found,
        insertion_demand,
        stash_after: self.stash_len(),
        overflowed: true,
      };
    }
    let queue = if self.random_queues { queue % self.z } else { 0 };
    self.stashes[queue].insert(new_pos, key);

    if evict_accessed_path {
      self.evict_loaded_path(old_pos);
    }
    self.write_path(old_pos);
    for &path in background_paths {
      assert!((path as usize) < self.max_n);
      self.read_path(path);
      self.evict_loaded_path(path);
      self.write_path(path);
    }

    SimUpdate { found, insertion_demand, stash_after: self.stash_len(), overflowed: false }
  }

  fn evict_loaded_path(&mut self, path: u32) {
    match self.eviction_policy {
      EvictionPolicy::Fixed => {
        for _ in 0..self.eviction_chains {
          for lane in 0..self.z {
            self.evict_lane(path, lane);
          }
        }
      }
      EvictionPolicy::Pooled => {
        for _ in 0..self.eviction_chains {
          self.evict_pooled(path);
        }
      }
    }
  }

  fn node_index(&self, depth: usize, path: u32) -> usize {
    let level = self.levels[depth];
    level.offset + (((path as usize) >> level.path_shift) & level.path_mask)
  }

  fn read_path(&mut self, path: u32) {
    for depth in 0..self.height {
      let node = self.node_index(depth, path);
      let source_start = node * self.bucket_slots;
      let target_start = depth * self.bucket_slots;
      self.path[target_start..target_start + self.bucket_slots]
        .copy_from_slice(&self.tree[source_start..source_start + self.bucket_slots]);
    }
  }

  fn write_path(&mut self, path: u32) {
    for depth in 0..self.height {
      let node = self.node_index(depth, path);
      let source_start = depth * self.bucket_slots;
      let target_start = node * self.bucket_slots;
      for &key in &self.path[source_start..source_start + self.bucket_slots] {
        debug_assert!(key == EMPTY_KEY || self.max_depth(key, path) >= depth);
      }
      self.tree[target_start..target_start + self.bucket_slots]
        .copy_from_slice(&self.path[source_start..source_start + self.bucket_slots]);
    }
  }

  fn remove_key(&mut self, key: u32) -> bool {
    if self.initialized[key as usize] {
      let label = self.labels[key as usize];
      for stash in &mut self.stashes {
        if stash.contains(label, key) {
          stash.remove(label, key);
          return true;
        }
      }
    }

    let mut found = false;
    for candidate in &mut self.path {
      if *candidate == key {
        *candidate = EMPTY_KEY;
        assert!(!found, "duplicate tree key {key}");
        found = true;
      }
    }
    found
  }

  fn path_index(&self, level: usize, lane: usize, slot: usize) -> usize {
    level * self.bucket_slots + lane * self.y + slot
  }

  fn max_depth(&self, key: u32, path: u32) -> usize {
    let label = self.labels[key as usize] as usize;
    let path = path as usize;
    for depth in (0..self.height).rev() {
      let span = self.levels[depth].subtree_span;
      if label / span == path / span {
        return depth;
      }
    }
    0
  }

  fn best_stash_source(&self, path: u32, lane: usize, minimum_depth: usize) -> Option<Source> {
    let stash = if self.random_queues { &self.stashes[lane] } else { &self.stashes[0] };
    for depth in (minimum_depth..self.height).rev() {
      let span = self.levels[depth].subtree_span;
      let start = (path as usize / span) * span;
      if let Some(key) = stash.minimum_key(start, start + span) {
        return Some(Source::Stash { key, depth });
      }
    }
    None
  }

  fn best_source_above(&self, path: u32, lane: usize, destination: usize) -> Option<Source> {
    let mut best = self.best_stash_source(path, lane, destination);
    for level in 0..destination {
      for slot in 0..self.y {
        let index = self.path_index(level, lane, slot);
        let key = self.path[index];
        if key == EMPTY_KEY {
          continue;
        }
        let depth = self.max_depth(key, path);
        if depth < destination {
          continue;
        }
        let source = Source::Path { index, level, key, depth };
        if source.is_better_than(best) {
          best = Some(source);
        }
      }
    }
    best
  }

  /// Algorithm 1 (`EvictOnceSlow`) from Circuit ORAM, restricted to one fixed
  /// lane whose buckets contain Y pooled slots. The shared stash plus this
  /// lane's root minibucket are level zero.
  fn evict_lane(&mut self, path: u32, lane: usize) {
    let mut destination = self.height - 1;
    while destination > 0 {
      let empty_slot = (0..self.y)
        .map(|slot| self.path_index(destination, lane, slot))
        .find(|&index| self.path[index] == EMPTY_KEY);
      let Some(empty_slot) = empty_slot else {
        destination -= 1;
        continue;
      };

      let Some(source) = self.best_source_above(path, lane, destination) else {
        destination -= 1;
        continue;
      };

      let source_level = match source {
        Source::Stash { key, .. } => {
          let label = self.labels[key as usize];
          let stash =
            if self.random_queues { &mut self.stashes[lane] } else { &mut self.stashes[0] };
          stash.remove(label, key);
          self.path[empty_slot] = key;
          0
        }
        Source::Path { index, level, key, .. } => {
          debug_assert_eq!(self.path[index], key);
          self.path[empty_slot] = std::mem::replace(&mut self.path[index], EMPTY_KEY);
          level
        }
      };
      destination = source_level;
    }
  }

  fn best_pooled_source_above(&self, path: u32, destination: usize) -> Option<Source> {
    let mut best = self.best_stash_source(path, 0, destination);
    for level in 0..destination {
      for slot in 0..self.bucket_slots {
        let index = level * self.bucket_slots + slot;
        let key = self.path[index];
        if key == EMPTY_KEY {
          continue;
        }
        let depth = self.max_depth(key, path);
        if depth < destination {
          continue;
        }
        let source = Source::Path { index, level, key, depth };
        if source.is_better_than(best) {
          best = Some(source);
        }
      }
    }
    best
  }

  /// One Circuit eviction chain over the entire pooled Z*Y-slot bucket.
  fn evict_pooled(&mut self, path: u32) {
    let mut destination = self.height - 1;
    while destination > 0 {
      let start = destination * self.bucket_slots;
      let empty_slot =
        (start..start + self.bucket_slots).find(|&index| self.path[index] == EMPTY_KEY);
      let Some(empty_slot) = empty_slot else {
        destination -= 1;
        continue;
      };
      let Some(source) = self.best_pooled_source_above(path, destination) else {
        destination -= 1;
        continue;
      };
      let source_level = match source {
        Source::Stash { key, .. } => {
          let label = self.labels[key as usize];
          self.stashes[0].remove(label, key);
          self.path[empty_slot] = key;
          0
        }
        Source::Path { index, level, key, .. } => {
          debug_assert_eq!(self.path[index], key);
          self.path[empty_slot] = std::mem::replace(&mut self.path[index], EMPTY_KEY);
          level
        }
      };
      destination = source_level;
    }
  }

  #[cfg(test)]
  fn block_count(&self) -> usize {
    self.stash_len() + self.tree.iter().filter(|&&key| key != EMPTY_KEY).count()
  }
}

#[derive(Clone, Copy, Debug)]
enum Workload {
  Cycle,
  Uniform,
}

#[derive(Clone, Copy, Debug)]
enum QueuePolicy {
  Shared,
  Random,
}

#[derive(Clone, Copy, Debug)]
enum EvictionPolicy {
  Fixed,
  Pooled,
}

#[derive(Clone, Copy, Debug)]
enum PathSchedule {
  Same,
  AccessPlusUniform,
  PaperRandom,
  PaperDeterministic,
  AccessPlusOneDeterministic,
  AccessPlusTwoDeterministic,
  AccessPlusRateDeterministic,
  AccessPlusPaperDeterministic,
  AccessPlusRadixDeterministic,
}

impl PathSchedule {
  const fn name(self) -> &'static str {
    match self {
      Self::Same => "same",
      Self::AccessPlusUniform => "access-plus-uniform",
      Self::PaperRandom => "paper-random",
      Self::PaperDeterministic => "paper-deterministic",
      Self::AccessPlusOneDeterministic => "access-plus-one-deterministic",
      Self::AccessPlusTwoDeterministic => "access-plus-two-deterministic",
      Self::AccessPlusRateDeterministic => "access-plus-rate-deterministic",
      Self::AccessPlusPaperDeterministic => "access-plus-paper-deterministic",
      Self::AccessPlusRadixDeterministic => "access-plus-radix-deterministic",
    }
  }
}

impl FromStr for PathSchedule {
  type Err = String;
  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match value {
      "same" => Ok(Self::Same),
      "access-plus-uniform" => Ok(Self::AccessPlusUniform),
      "paper-random" => Ok(Self::PaperRandom),
      "paper-deterministic" => Ok(Self::PaperDeterministic),
      "access-plus-one-deterministic" => Ok(Self::AccessPlusOneDeterministic),
      "access-plus-two-deterministic" => Ok(Self::AccessPlusTwoDeterministic),
      "access-plus-rate-deterministic" => Ok(Self::AccessPlusRateDeterministic),
      "access-plus-paper-deterministic" => Ok(Self::AccessPlusPaperDeterministic),
      "access-plus-radix-deterministic" => Ok(Self::AccessPlusRadixDeterministic),
      _ => Err(format!("unknown path schedule: {value}")),
    }
  }
}

impl EvictionPolicy {
  const fn name(self) -> &'static str {
    match self {
      Self::Fixed => "fixed",
      Self::Pooled => "pooled",
    }
  }
}

impl FromStr for EvictionPolicy {
  type Err = String;
  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match value {
      "fixed" => Ok(Self::Fixed),
      "pooled" => Ok(Self::Pooled),
      _ => Err(format!("unknown eviction policy: {value}")),
    }
  }
}

impl QueuePolicy {
  const fn name(self) -> &'static str {
    match self {
      Self::Shared => "shared",
      Self::Random => "random",
    }
  }
}

impl FromStr for QueuePolicy {
  type Err = String;
  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match value {
      "shared" => Ok(Self::Shared),
      "random" => Ok(Self::Random),
      _ => Err(format!("unknown queue policy: {value}")),
    }
  }
}

impl Workload {
  const fn name(self) -> &'static str {
    match self {
      Self::Cycle => "cycle",
      Self::Uniform => "uniform",
    }
  }

  fn key(self, operation: u64, n: usize, random: &mut OsRandomWords) -> usize {
    match self {
      Self::Cycle => operation as usize & (n - 1),
      Self::Uniform => random.next_usize() & (n - 1),
    }
  }
}

impl FromStr for Workload {
  type Err = String;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match value {
      "cycle" => Ok(Self::Cycle),
      "uniform" => Ok(Self::Uniform),
      _ => Err(format!("unknown workload: {value}")),
    }
  }
}

#[derive(Debug)]
struct Args {
  n: usize,
  b: usize,
  z: usize,
  y: usize,
  warmup: u64,
  operations: u64,
  workload: Workload,
  queue_policy: QueuePolicy,
  eviction_policy: EvictionPolicy,
  eviction_chains: usize,
  path_schedule: PathSchedule,
  deterministic_numerator: usize,
  deterministic_denominator: usize,
  epoch_operations: u64,
  epoch_output: String,
  output: String,
}

impl Args {
  fn parse() -> Result<Self, String> {
    let mut result = Self {
      n: 1 << 12,
      b: 2,
      z: 2,
      y: 2,
      warmup: 1 << 22,
      operations: 1 << 24,
      workload: Workload::Cycle,
      queue_policy: QueuePolicy::Shared,
      eviction_policy: EvictionPolicy::Fixed,
      eviction_chains: 1,
      path_schedule: PathSchedule::Same,
      deterministic_numerator: 1,
      deterministic_denominator: 1,
      epoch_operations: 0,
      epoch_output: String::new(),
      output: "lane_oram_minibucket_security.csv".to_owned(),
    };

    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
      let value = arguments.next().ok_or_else(|| format!("missing value for {argument}"))?;
      match argument.as_str() {
        "--n" => result.n = parse_number(&value)?,
        "--b" => result.b = parse_number(&value)?,
        "--z" => result.z = parse_number(&value)?,
        "--y" => result.y = parse_number(&value)?,
        "--queue-policy" => result.queue_policy = value.parse()?,
        "--eviction-policy" => result.eviction_policy = value.parse()?,
        "--eviction-chains" => result.eviction_chains = parse_number(&value)?,
        "--path-schedule" => result.path_schedule = value.parse()?,
        "--deterministic-numerator" => result.deterministic_numerator = parse_number(&value)?,
        "--deterministic-denominator" => result.deterministic_denominator = parse_number(&value)?,
        "--epoch-operations" => result.epoch_operations = parse_number(&value)?,
        "--epoch-output" => result.epoch_output = value,
        "--warmup" => result.warmup = parse_number(&value)?,
        "--operations" => result.operations = parse_number(&value)?,
        "--workload" => result.workload = value.parse()?,
        "--output" => result.output = value,
        _ => return Err(format!("unknown argument: {argument}")),
      }
    }

    if !result.n.is_power_of_two()
      || result.z == 0
      || result.y == 0
      || result.operations == 0
      || result.eviction_chains == 0
      || result.deterministic_denominator == 0
    {
      return Err(
        "N must be a power of two; Z, Y, operations, eviction chains, and the deterministic denominator must be nonzero".to_owned(),
      );
    }
    if matches!(result.eviction_policy, EvictionPolicy::Pooled)
      && matches!(result.queue_policy, QueuePolicy::Random)
    {
      return Err("pooled eviction requires the shared queue policy".to_owned());
    }
    if result.b < 2 || !result.b.is_power_of_two() {
      return Err("B must be a power of two at least 2".to_owned());
    }
    if matches!(
      result.path_schedule,
      PathSchedule::PaperRandom
        | PathSchedule::PaperDeterministic
        | PathSchedule::AccessPlusPaperDeterministic
    ) && (result.b != 2 || result.n < 2)
    {
      return Err("paper path schedules require B=2 and N>=2".to_owned());
    }
    if matches!(
      result.path_schedule,
      PathSchedule::AccessPlusOneDeterministic
        | PathSchedule::AccessPlusTwoDeterministic
        | PathSchedule::AccessPlusRateDeterministic
        | PathSchedule::AccessPlusRadixDeterministic
    ) && result.n < 2
    {
      return Err("deterministic path schedules require N>=2".to_owned());
    }
    if matches!(result.path_schedule, PathSchedule::AccessPlusRateDeterministic)
      && result.deterministic_numerator == 0
    {
      return Err("the deterministic numerator must be nonzero for a rate schedule".to_owned());
    }
    if result.epoch_operations == 0 && !result.epoch_output.is_empty() {
      return Err("--epoch-output requires nonzero --epoch-operations".to_owned());
    }
    if result.epoch_operations != 0
      && (result.epoch_output.is_empty()
        || result.epoch_operations > result.operations
        || result.operations % result.epoch_operations != 0)
    {
      return Err(
        "epoch tracing requires an output path and an epoch size that exactly divides operations"
          .to_owned(),
      );
    }
    let mut exact_capacity = 1usize;
    while exact_capacity < result.n {
      exact_capacity *= result.b;
    }
    if exact_capacity != result.n {
      return Err("N must be an exact power of B".to_owned());
    }
    Ok(result)
  }
}

#[derive(Debug)]
struct OsRandomWords {
  words: Box<[u32]>,
  next: usize,
}

impl OsRandomWords {
  fn new() -> Self {
    let mut result = Self { words: vec![0; OS_RANDOM_BATCH_WORDS].into_boxed_slice(), next: 0 };
    result.refill();
    result
  }

  fn refill(&mut self) {
    getrandom::fill(bytemuck::cast_slice_mut(&mut self.words))
      .expect("the operating system random source failed");
    self.next = 0;
  }

  fn next_usize(&mut self) -> usize {
    if self.next == self.words.len() {
      self.refill();
    }
    let result = self.words[self.next] as usize;
    self.next += 1;
    result
  }
}

fn parse_number<T>(value: &str) -> Result<T, String>
where
  T: FromStr,
  T::Err: std::fmt::Display,
{
  value.parse().map_err(|error| format!("invalid number {value}: {error}"))
}

fn reverse_path_bits(mut value: usize, bits: u32) -> u32 {
  let mut result = 0u32;
  for _ in 0..bits {
    result = (result << 1) | (value as u32 & 1);
    value >>= 1;
  }
  result
}

fn reverse_path_digits(mut value: usize, n: usize, b: usize) -> u32 {
  let bits_per_digit = b.trailing_zeros();
  let digits = n.trailing_zeros() / bits_per_digit;
  let digit_mask = b - 1;
  let mut result = 0usize;
  for _ in 0..digits {
    result = (result << bits_per_digit) | (value & digit_mask);
    value >>= bits_per_digit;
  }
  result as u32
}

fn paper_deterministic_paths(timestep: u64, n: usize) -> [u32; 2] {
  let even = (timestep as usize).wrapping_mul(2) & (n - 1);
  let bits = n.trailing_zeros();
  [reverse_path_bits(even, bits), reverse_path_bits(even | 1, bits)]
}

fn one_deterministic_path(timestep: u64, n: usize, b: usize) -> u32 {
  reverse_path_digits(timestep as usize & (n - 1), n, b)
}

fn deterministic_path_group(timestep: u64, n: usize, b: usize, count: usize, paths: &mut Vec<u32>) {
  debug_assert!(count.is_power_of_two() && count <= n);
  paths.clear();
  let first = (timestep as usize).wrapping_mul(count) & (n - 1);
  paths.extend((0..count).map(|offset| reverse_path_digits((first + offset) & (n - 1), n, b)));
}

fn radix_deterministic_paths(timestep: u64, n: usize, b: usize, paths: &mut Vec<u32>) {
  deterministic_path_group(timestep, n, b, b, paths);
}

fn rate_deterministic_paths(
  timestep: u64,
  n: usize,
  b: usize,
  numerator: usize,
  denominator: usize,
  paths: &mut Vec<u32>,
) {
  let timestep = timestep as u128;
  let numerator = numerator as u128;
  let denominator = denominator as u128;
  let first = timestep * numerator / denominator;
  let end = (timestep + 1) * numerator / denominator;
  paths.clear();
  paths.extend((first..end).map(|sequence| reverse_path_digits(sequence as usize & (n - 1), n, b)));
}

fn perform_update(
  simulator: &mut MinibucketLaneSimulator,
  positions: &mut [u32],
  key: usize,
  random: &mut OsRandomWords,
  queue_random: &mut OsRandomWords,
  eviction_random: &mut OsRandomWords,
  eviction_counter: &mut u64,
  path_schedule: PathSchedule,
  b: usize,
  deterministic_numerator: usize,
  deterministic_denominator: usize,
  radix_paths: &mut Vec<u32>,
) -> SimUpdate {
  let old_pos = positions[key];
  let new_pos = (random.next_usize() & (simulator.max_n() - 1)) as u32;
  let queue = queue_random.next_usize();
  let result = match path_schedule {
    PathSchedule::Same => simulator.update_in_queue(old_pos, new_pos, key as u32, queue),
    PathSchedule::AccessPlusUniform => {
      let background = (eviction_random.next_usize() & (simulator.max_n() - 1)) as u32;
      simulator.update_with_schedule(old_pos, new_pos, key as u32, queue, true, &[background])
    }
    PathSchedule::PaperRandom => {
      let half = simulator.max_n() / 2;
      let left = (eviction_random.next_usize() & (half - 1)) as u32;
      let right = (half | (eviction_random.next_usize() & (half - 1))) as u32;
      simulator.update_with_schedule(old_pos, new_pos, key as u32, queue, false, &[left, right])
    }
    PathSchedule::PaperDeterministic | PathSchedule::AccessPlusPaperDeterministic => {
      let paths = paper_deterministic_paths(*eviction_counter, simulator.max_n());
      *eviction_counter = eviction_counter.wrapping_add(1);
      let evict_accessed = matches!(path_schedule, PathSchedule::AccessPlusPaperDeterministic);
      simulator.update_with_schedule(old_pos, new_pos, key as u32, queue, evict_accessed, &paths)
    }
    PathSchedule::AccessPlusOneDeterministic => {
      let path = one_deterministic_path(*eviction_counter, simulator.max_n(), b);
      *eviction_counter = eviction_counter.wrapping_add(1);
      simulator.update_with_schedule(old_pos, new_pos, key as u32, queue, true, &[path])
    }
    PathSchedule::AccessPlusTwoDeterministic => {
      deterministic_path_group(*eviction_counter, simulator.max_n(), b, 2, radix_paths);
      *eviction_counter = eviction_counter.wrapping_add(1);
      simulator.update_with_schedule(old_pos, new_pos, key as u32, queue, true, radix_paths)
    }
    PathSchedule::AccessPlusRateDeterministic => {
      rate_deterministic_paths(
        *eviction_counter,
        simulator.max_n(),
        b,
        deterministic_numerator,
        deterministic_denominator,
        radix_paths,
      );
      *eviction_counter = eviction_counter.wrapping_add(1);
      simulator.update_with_schedule(old_pos, new_pos, key as u32, queue, true, radix_paths)
    }
    PathSchedule::AccessPlusRadixDeterministic => {
      radix_deterministic_paths(*eviction_counter, simulator.max_n(), b, radix_paths);
      *eviction_counter = eviction_counter.wrapping_add(1);
      simulator.update_with_schedule(old_pos, new_pos, key as u32, queue, true, radix_paths)
    }
  };
  positions[key] = new_pos;
  result
}

fn append_tail(output: &mut String, args: &Args, metric: &str, histogram: &[u64], maximum: usize) {
  let mut exceed_count = args.operations;
  for threshold in 0..=maximum {
    exceed_count -= histogram[threshold];
    let probability = exceed_count as f64 / args.operations as f64;
    let log2_inverse = if probability == 0.0 { f64::INFINITY } else { -probability.log2() };
    output.push_str(&format!(
      "{},{},{},{},{},{},{},{},{},os-getrandom,random-old-and-new-paths,{},{},{},{:.17},{:.9},{},{},{},{},{}\n",
      args.b,
      args.z,
      args.y,
      args.n,
      args.warmup,
      args.operations,
      args.workload.name(),
      args.z * args.y,
      args.queue_policy.name(),
      metric,
      threshold,
      exceed_count,
      probability,
      log2_inverse,
      args.eviction_policy.name(),
      args.eviction_chains,
      args.path_schedule.name(),
      args.deterministic_numerator,
      args.deterministic_denominator,
    ));
  }
}

#[allow(clippy::too_many_arguments)]
fn append_epoch_tail(
  output: &mut String,
  args: &Args,
  epoch: u64,
  metric: &str,
  histogram: &[u64],
  minimum: usize,
  sum: u128,
  maximum: usize,
) {
  let samples = args.epoch_operations;
  let mut exceed_count = samples;
  for threshold in 0..=maximum {
    exceed_count -= histogram.get(threshold).copied().unwrap_or(0);
    let probability = exceed_count as f64 / samples as f64;
    let log2_inverse = if probability == 0.0 { f64::INFINITY } else { -probability.log2() };
    output.push_str(&format!(
      "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.17},{:.9},{},{:.9},{},{},{},{},os-getrandom,random-old-and-new-paths\n",
      args.b,
      args.z,
      args.y,
      args.n,
      args.operations,
      args.epoch_operations,
      epoch,
      epoch * samples,
      (epoch + 1) * samples,
      args.workload.name(),
      args.z * args.y,
      metric,
      threshold,
      exceed_count,
      probability,
      log2_inverse,
      minimum,
      sum as f64 / samples as f64,
      maximum,
      args.path_schedule.name(),
      args.deterministic_numerator,
      args.deterministic_denominator,
    ));
  }
}

fn main() {
  let args = Args::parse().unwrap_or_else(|error| {
    eprintln!("error: {error}");
    process::exit(2);
  });
  let start = Instant::now();
  let mut simulator = MinibucketLaneSimulator::new(args.n, args.z, args.y, args.b, args.n + 1);
  if matches!(args.queue_policy, QueuePolicy::Random) {
    simulator = simulator.with_random_queues();
  }
  simulator = simulator.with_eviction_strategy(args.eviction_policy, args.eviction_chains);
  let mut positions = Vec::with_capacity(args.n);
  let mut label_random = OsRandomWords::new();
  let mut workload_random = OsRandomWords::new();
  let mut queue_random = OsRandomWords::new();
  let mut eviction_random = OsRandomWords::new();
  let mut eviction_counter = 0u64;
  let rate_capacity = args.deterministic_numerator.div_ceil(args.deterministic_denominator);
  let mut radix_paths = Vec::with_capacity(args.b.max(rate_capacity));

  for key in 0..args.n {
    positions.push((label_random.next_usize() & (args.n - 1)) as u32);
    let result = perform_update(
      &mut simulator,
      &mut positions,
      key,
      &mut label_random,
      &mut queue_random,
      &mut eviction_random,
      &mut eviction_counter,
      args.path_schedule,
      args.b,
      args.deterministic_numerator,
      args.deterministic_denominator,
      &mut radix_paths,
    );
    assert!(!result.found && !result.overflowed);
  }
  for operation in 0..args.warmup {
    let key = args.workload.key(operation, args.n, &mut workload_random);
    let result = perform_update(
      &mut simulator,
      &mut positions,
      key,
      &mut label_random,
      &mut queue_random,
      &mut eviction_random,
      &mut eviction_counter,
      args.path_schedule,
      args.b,
      args.deterministic_numerator,
      args.deterministic_denominator,
      &mut radix_paths,
    );
    assert!(result.found && !result.overflowed);
  }
  let measurement_start_post = simulator.stash_len();

  let mut post_histogram = vec![0u64; args.n + 2];
  let mut demand_histogram = vec![0u64; args.n + 2];
  let mut post_sum = 0u128;
  let mut demand_sum = 0u128;
  let mut post_min = usize::MAX;
  let mut post_max = 0usize;
  let mut demand_min = usize::MAX;
  let mut demand_max = 0usize;
  let mut epoch_csv = String::from(
    "b,z,y,n,operations,epoch_operations,epoch,first_operation,last_operation,workload,bucket_slots,metric,threshold,exceed_count,probability,log2_inverse,minimum,mean,maximum,path_schedule,deterministic_numerator,deterministic_denominator,rng_source,initialization\n",
  );
  let mut epoch_post_histogram = Vec::<u64>::new();
  let mut epoch_demand_histogram = Vec::<u64>::new();
  let mut epoch_post_sum = 0u128;
  let mut epoch_demand_sum = 0u128;
  let mut epoch_post_min = usize::MAX;
  let mut epoch_post_max = 0usize;
  let mut epoch_demand_min = usize::MAX;
  let mut epoch_demand_max = 0usize;
  for operation in 0..args.operations {
    let key = args.workload.key(args.warmup + operation, args.n, &mut workload_random);
    let result = perform_update(
      &mut simulator,
      &mut positions,
      key,
      &mut label_random,
      &mut queue_random,
      &mut eviction_random,
      &mut eviction_counter,
      args.path_schedule,
      args.b,
      args.deterministic_numerator,
      args.deterministic_denominator,
      &mut radix_paths,
    );
    assert!(result.found && !result.overflowed);
    post_histogram[result.stash_after] += 1;
    demand_histogram[result.insertion_demand] += 1;
    post_sum += result.stash_after as u128;
    demand_sum += result.insertion_demand as u128;
    post_min = post_min.min(result.stash_after);
    post_max = post_max.max(result.stash_after);
    demand_min = demand_min.min(result.insertion_demand);
    demand_max = demand_max.max(result.insertion_demand);
    if args.epoch_operations != 0 {
      if epoch_post_histogram.len() <= result.stash_after {
        epoch_post_histogram.resize(result.stash_after + 1, 0);
      }
      if epoch_demand_histogram.len() <= result.insertion_demand {
        epoch_demand_histogram.resize(result.insertion_demand + 1, 0);
      }
      epoch_post_histogram[result.stash_after] += 1;
      epoch_demand_histogram[result.insertion_demand] += 1;
      epoch_post_sum += result.stash_after as u128;
      epoch_demand_sum += result.insertion_demand as u128;
      epoch_post_min = epoch_post_min.min(result.stash_after);
      epoch_post_max = epoch_post_max.max(result.stash_after);
      epoch_demand_min = epoch_demand_min.min(result.insertion_demand);
      epoch_demand_max = epoch_demand_max.max(result.insertion_demand);

      if (operation + 1) % args.epoch_operations == 0 {
        let epoch = operation / args.epoch_operations;
        append_epoch_tail(
          &mut epoch_csv,
          &args,
          epoch,
          "post",
          &epoch_post_histogram,
          epoch_post_min,
          epoch_post_sum,
          epoch_post_max,
        );
        append_epoch_tail(
          &mut epoch_csv,
          &args,
          epoch,
          "insertion",
          &epoch_demand_histogram,
          epoch_demand_min,
          epoch_demand_sum,
          epoch_demand_max,
        );
        epoch_post_histogram.clear();
        epoch_demand_histogram.clear();
        epoch_post_sum = 0;
        epoch_demand_sum = 0;
        epoch_post_min = usize::MAX;
        epoch_post_max = 0;
        epoch_demand_min = usize::MAX;
        epoch_demand_max = 0;
      }
    }
  }

  let mut csv = String::from(
    "b,z,y,n,warmup,operations,workload,bucket_slots,queue_policy,rng_source,initialization,metric,threshold,exceed_count,probability,log2_inverse,eviction_policy,eviction_chains,path_schedule,deterministic_numerator,deterministic_denominator\n",
  );
  append_tail(&mut csv, &args, "post", &post_histogram, post_max);
  append_tail(&mut csv, &args, "insertion", &demand_histogram, demand_max);
  fs::write(&args.output, csv).unwrap_or_else(|error| {
    eprintln!("failed to write {}: {error}", args.output);
    process::exit(1);
  });
  if args.epoch_operations != 0 {
    fs::write(&args.epoch_output, epoch_csv).unwrap_or_else(|error| {
      eprintln!("failed to write {}: {error}", args.epoch_output);
      process::exit(1);
    });
  }

  let elapsed = start.elapsed().as_secs_f64();
  eprintln!(
    "B={} Z={} Y={} bucket_slots={} n={} workload={} queue_policy={} eviction_policy={} eviction_chains={} path_schedule={} deterministic_rate={}/{} rng=os-getrandom warmup={} measurement_start_post={} measured={} epoch_operations={} seconds={:.3} Mop/s={:.3} post[min/mean/max]={}/{:.3}/{} insertion[min/mean/max]={}/{:.3}/{} output={} epoch_output={}",
    args.b,
    args.z,
    args.y,
    args.z * args.y,
    args.n,
    args.workload.name(),
    args.queue_policy.name(),
    args.eviction_policy.name(),
    args.eviction_chains,
    args.path_schedule.name(),
    args.deterministic_numerator,
    args.deterministic_denominator,
    args.warmup,
    measurement_start_post,
    args.operations,
    args.epoch_operations,
    elapsed,
    (args.warmup + args.operations) as f64 / elapsed / 1e6,
    post_min,
    post_sum as f64 / args.operations as f64,
    post_max,
    demand_min,
    demand_sum as f64 / args.operations as f64,
    demand_max,
    args.output,
    args.epoch_output,
  );
}

#[cfg(test)]
mod tests {
  use super::{MinibucketLaneSimulator, EMPTY_KEY};
  use rostl_oram::circuit_oram::{Block, CircuitORAM, S as CIRCUIT_STASH_SIZE, Z as CIRCUIT_Y};

  fn position(operation: usize, key: usize, n: usize) -> u32 {
    operation.wrapping_mul(0x9e37_79b1).wrapping_add(key.wrapping_mul(0x85eb_ca77)).rotate_left(13)
      as u32
      & (n as u32 - 1)
  }

  #[test]
  fn random_like_updates_conserve_every_block() {
    const N: usize = 64;
    let mut simulator = MinibucketLaneSimulator::new(N, 3, 2, 2, N + 1);
    let mut positions = [0u32; N];
    for key in 0..N {
      positions[key] = position(0, key, N);
      let new_pos = position(1, key, N);
      let result = simulator.update(positions[key], new_pos, key as u32);
      assert!(!result.found && !result.overflowed);
      positions[key] = new_pos;
      assert_eq!(simulator.block_count(), key + 1);
    }
    for operation in 0..50_000 {
      let key = operation & (N - 1);
      let new_pos = position(operation + 2, key, N);
      let result = simulator.update(positions[key], new_pos, key as u32);
      assert!(result.found && !result.overflowed);
      positions[key] = new_pos;
      assert_eq!(simulator.block_count(), N);
    }
  }

  #[test]
  fn random_queue_updates_conserve_every_block() {
    let n = 256;
    let mut simulator = MinibucketLaneSimulator::new(n, 3, 1, 2, n + 1).with_random_queues();
    let mut positions = vec![0; n];
    for operation in 0..50_000 {
      let key = operation % n;
      let new_pos = position(operation, key, n);
      let queue = position(operation + 17, key + 3, n) as usize % 3;
      let result = simulator.update_in_queue(positions[key], new_pos, key as u32, queue);
      assert_eq!(result.found, operation >= n);
      assert!(!result.overflowed);
      positions[key] = new_pos;
      assert_eq!(simulator.block_count(), n.min(operation + 1));
    }
  }

  #[test]
  fn pooled_chains_conserve_every_block() {
    let n = 256;
    let mut simulator = MinibucketLaneSimulator::new(n, 3, 2, 2, n + 1)
      .with_eviction_strategy(super::EvictionPolicy::Pooled, 3);
    let mut positions = vec![0; n];
    for operation in 0..20_000 {
      let key = operation % n;
      let new_pos = position(operation, key, n);
      let result = simulator.update(positions[key], new_pos, key as u32);
      assert_eq!(result.found, operation >= n);
      assert!(!result.overflowed);
      positions[key] = new_pos;
      assert_eq!(simulator.block_count(), n.min(operation + 1));
    }
  }

  #[test]
  fn background_path_schedules_conserve_every_block() {
    let n = 256;
    for paper_random in [false, true] {
      let mut simulator = MinibucketLaneSimulator::new(n, 1, 3, 2, n + 1);
      let mut positions = vec![0; n];
      for operation in 0..20_000 {
        let key = operation % n;
        let new_pos = position(operation, key, n);
        let background = if paper_random {
          [
            position(operation + 31, key + 11, n) & (n as u32 / 2 - 1),
            (n as u32 / 2) | (position(operation + 37, key + 13, n) & (n as u32 / 2 - 1)),
          ]
        } else {
          [position(operation + 31, key + 11, n), 0]
        };
        let paths = if paper_random { &background[..] } else { &background[..1] };
        let result = simulator.update_with_schedule(
          positions[key],
          new_pos,
          key as u32,
          0,
          !paper_random,
          paths,
        );
        assert_eq!(result.found, operation >= n);
        assert!(!result.overflowed);
        positions[key] = new_pos;
        assert_eq!(simulator.block_count(), n.min(operation + 1));
      }
    }
  }

  #[test]
  fn paper_deterministic_pairs_partition_the_leaves() {
    let n = 256;
    let mut seen = vec![false; n];
    for timestep in 0..n / 2 {
      let [left, right] = super::paper_deterministic_paths(timestep as u64, n);
      assert!((left as usize) < n / 2);
      assert!((right as usize) >= n / 2);
      assert!(!seen[left as usize]);
      assert!(!seen[right as usize]);
      seen[left as usize] = true;
      seen[right as usize] = true;
    }
    assert!(seen.into_iter().all(|visited| visited));
  }

  #[test]
  fn one_deterministic_path_cycles_all_leaves() {
    let n = 256;
    for b in [2, 4] {
      let mut seen = vec![false; n];
      for timestep in 0..n {
        let path = super::one_deterministic_path(timestep as u64, n, b) as usize;
        if b == 2 {
          assert_eq!(path as u32, super::reverse_path_bits(timestep, n.trailing_zeros()));
        }
        assert!(!seen[path]);
        seen[path] = true;
      }
      assert!(seen.into_iter().all(|visited| visited));
    }
  }

  #[test]
  fn radix_deterministic_groups_partition_the_leaves() {
    let n = 256;
    for b in [2, 4] {
      let mut seen = vec![false; n];
      let mut paths = Vec::with_capacity(b);
      for timestep in 0..n / b {
        super::radix_deterministic_paths(timestep as u64, n, b, &mut paths);
        assert_eq!(paths.len(), b);
        if b == 2 {
          assert_eq!(paths.as_slice(), &super::paper_deterministic_paths(timestep as u64, n));
        }
        for (root_digit, &path) in paths.iter().enumerate() {
          assert_eq!(path as usize / (n / b), root_digit);
          assert!(!seen[path as usize]);
          seen[path as usize] = true;
        }
      }
      assert!(seen.into_iter().all(|visited| visited));
    }
  }

  #[test]
  fn quaternary_two_path_groups_cycle_all_leaves() {
    let n = 256;
    let mut seen = vec![false; n];
    let mut paths = Vec::with_capacity(2);
    for timestep in 0..n / 2 {
      super::deterministic_path_group(timestep as u64, n, 4, 2, &mut paths);
      assert_eq!(paths.len(), 2);
      let first_root = timestep % 2 * 2;
      for (offset, &path) in paths.iter().enumerate() {
        assert_eq!(path as usize / (n / 4), first_root + offset);
        assert!(!seen[path as usize]);
        seen[path as usize] = true;
      }
    }
    assert!(seen.into_iter().all(|visited| visited));
  }

  #[test]
  fn fractional_rates_emit_the_gap_free_leaf_sequence() {
    let n = 256;
    for (numerator, denominator, pattern) in
      [(1, 2, &[0usize, 1][..]), (2, 3, &[0usize, 1, 1][..]), (3, 2, &[1usize, 2][..])]
    {
      let mut sequence = 0usize;
      let mut timestep = 0u64;
      let mut seen = vec![false; n];
      let mut paths = Vec::new();
      while sequence < n {
        super::rate_deterministic_paths(timestep, n, 4, numerator, denominator, &mut paths);
        assert_eq!(paths.len(), pattern[timestep as usize % pattern.len()]);
        for &path in &paths {
          assert_eq!(path, super::one_deterministic_path(sequence as u64, n, 4));
          assert!(!seen[path as usize]);
          seen[path as usize] = true;
          sequence += 1;
        }
        timestep += 1;
      }
      assert!(seen.into_iter().all(|visited| visited));
    }
  }

  #[test]
  fn accessed_plus_one_deterministic_path_conserves_every_block() {
    let n = 256;
    let mut simulator = MinibucketLaneSimulator::new(n, 2, 3, 2, n + 1);
    let mut positions = vec![0; n];
    for operation in 0..20_000 {
      let key = operation % n;
      let new_pos = position(operation, key, n);
      let path = super::one_deterministic_path(operation as u64, n, 2);
      let result =
        simulator.update_with_schedule(positions[key], new_pos, key as u32, 0, true, &[path]);
      assert_eq!(result.found, operation >= n);
      assert!(!result.overflowed);
      positions[key] = new_pos;
      assert_eq!(simulator.block_count(), n.min(operation + 1));
    }
  }

  #[test]
  fn paper_deterministic_schedules_conserve_every_block() {
    let n = 256;
    for evict_accessed in [false, true] {
      let mut simulator = MinibucketLaneSimulator::new(n, 1, 2, 2, n + 1);
      let mut positions = vec![0; n];
      for operation in 0..20_000 {
        let key = operation % n;
        let new_pos = position(operation, key, n);
        let paths = super::paper_deterministic_paths(operation as u64, n);
        let result = simulator.update_with_schedule(
          positions[key],
          new_pos,
          key as u32,
          0,
          evict_accessed,
          &paths,
        );
        assert_eq!(result.found, operation >= n);
        assert!(!result.overflowed);
        positions[key] = new_pos;
        assert_eq!(simulator.block_count(), n.min(operation + 1));
      }
    }
  }

  #[test]
  fn quaternary_grouped_deterministic_schedules_conserve_every_block() {
    let n = 256;
    for count in [2, 4] {
      let mut simulator = MinibucketLaneSimulator::new(n, 3, 1, 4, n + 1);
      let mut positions = vec![0; n];
      let mut paths = Vec::with_capacity(count);
      for operation in 0..20_000 {
        let key = operation % n;
        let new_pos = position(operation, key, n);
        super::deterministic_path_group(operation as u64, n, 4, count, &mut paths);
        let result =
          simulator.update_with_schedule(positions[key], new_pos, key as u32, 0, true, &paths);
        assert_eq!(result.found, operation >= n);
        assert!(!result.overflowed);
        positions[key] = new_pos;
        assert_eq!(simulator.block_count(), n.min(operation + 1));
      }
    }
  }

  #[test]
  fn quaternary_fractional_schedules_conserve_every_block() {
    let n = 256;
    for (numerator, denominator) in [(1, 2), (2, 3), (3, 2)] {
      let mut simulator = MinibucketLaneSimulator::new(n, 3, 2, 4, n + 1);
      let mut positions = vec![0; n];
      let mut paths = Vec::new();
      for operation in 0..20_000 {
        let key = operation % n;
        let new_pos = position(operation, key, n);
        super::rate_deterministic_paths(operation as u64, n, 4, numerator, denominator, &mut paths);
        let result =
          simulator.update_with_schedule(positions[key], new_pos, key as u32, 0, true, &paths);
        assert_eq!(result.found, operation >= n);
        assert!(!result.overflowed);
        positions[key] = new_pos;
        assert_eq!(simulator.block_count(), n.min(operation + 1));
      }
    }
  }

  fn reverse_low_bits(mut value: u32, bits: usize) -> u32 {
    let mut result = 0;
    for _ in 0..bits {
      result = (result << 1) | (value & 1);
      value >>= 1;
    }
    result
  }

  #[test]
  fn one_minibucket_lane_matches_circuit_evict_once() {
    const N: usize = 8;
    const L: usize = 3;
    assert_eq!(CIRCUIT_Y, 2);

    let mut simulator = MinibucketLaneSimulator::new(N, 1, CIRCUIT_Y, 2, CIRCUIT_STASH_SIZE);
    // On path 000 these labels have unique legal depths 3, 2, 1, and 0.
    let labels = [0u32, 1, 3, 7];
    for (key, &label) in labels.iter().enumerate() {
      simulator.labels[key] = label;
      simulator.initialized[key] = true;
    }
    simulator.stashes[0].insert(labels[3], 3);
    let root = simulator.path_index(0, 0, 0);
    let level1 = simulator.path_index(1, 0, 0);
    let level2 = simulator.path_index(2, 0, 0);
    simulator.path[root] = 2;
    simulator.path[level1] = 1;
    simulator.path[level2] = 0;

    let mut circuit = CircuitORAM::<u64>::new(N);
    circuit.stash[0] = Block { pos: reverse_low_bits(labels[3], L), key: 3, value: 0 };
    circuit.stash[CIRCUIT_STASH_SIZE] =
      Block { pos: reverse_low_bits(labels[2], L), key: 2, value: 0 };
    circuit.stash[CIRCUIT_STASH_SIZE + CIRCUIT_Y] =
      Block { pos: reverse_low_bits(labels[1], L), key: 1, value: 0 };
    circuit.stash[CIRCUIT_STASH_SIZE + 2 * CIRCUIT_Y] =
      Block { pos: reverse_low_bits(labels[0], L), key: 0, value: 0 };

    simulator.evict_lane(0, 0);
    circuit.evict_once_fast(0);

    let simulator_stash: Vec<_> = labels
      .iter()
      .enumerate()
      .filter_map(|(key, &label)| {
        simulator.stashes[0].contains(label, key as u32).then_some(key as u32)
      })
      .collect();
    let circuit_stash: Vec<_> = circuit.stash[..CIRCUIT_STASH_SIZE]
      .iter()
      .filter(|block| !block.is_empty())
      .map(|block| block.key as u32)
      .collect();
    assert_eq!(simulator_stash, circuit_stash);

    for level in 0..simulator.height {
      let mut simulated: Vec<_> = (0..CIRCUIT_Y)
        .map(|slot| simulator.path[simulator.path_index(level, 0, slot)])
        .filter(|&key| key != EMPTY_KEY)
        .collect();
      let mut actual: Vec<_> = circuit.stash
        [CIRCUIT_STASH_SIZE + level * CIRCUIT_Y..CIRCUIT_STASH_SIZE + (level + 1) * CIRCUIT_Y]
        .iter()
        .filter(|block| !block.is_empty())
        .map(|block| block.key as u32)
        .collect();
      simulated.sort_unstable();
      actual.sort_unstable();
      assert_eq!(simulated, actual, "level {level}");
    }
  }

  #[test]
  fn each_lane_carries_at_most_one_root_block() {
    let mut simulator = MinibucketLaneSimulator::new(4, 2, 2, 2, 8);
    simulator.labels[0] = 0;
    simulator.labels[1] = 0;
    simulator.labels[2] = 0;
    simulator.labels[3] = 0;
    for key in 0..4 {
      simulator.initialized[key] = true;
    }
    // Two compatible root blocks in lane zero and empty descendants.
    let root0 = simulator.path_index(0, 0, 0);
    let root1 = simulator.path_index(0, 0, 1);
    simulator.path[root0] = 0;
    simulator.path[root1] = 1;
    simulator.evict_lane(0, 0);

    let root_keys = (0..2)
      .map(|slot| simulator.path[simulator.path_index(0, 0, slot)])
      .filter(|&key| key != EMPTY_KEY)
      .count();
    assert_eq!(root_keys, 1, "one Circuit chain must clear exactly one root source");
  }

  #[test]
  fn separate_lanes_can_carry_one_chain_each() {
    let mut simulator = MinibucketLaneSimulator::new(4, 2, 2, 2, 8);
    simulator.labels[0] = 0;
    simulator.labels[1] = 0;
    simulator.initialized[0] = true;
    simulator.initialized[1] = true;
    let lane0 = simulator.path_index(0, 0, 0);
    let lane1 = simulator.path_index(0, 1, 0);
    simulator.path[lane0] = 0;
    simulator.path[lane1] = 1;
    simulator.evict_lane(0, 0);
    simulator.evict_lane(0, 1);
    assert_eq!(simulator.path[lane0], EMPTY_KEY);
    assert_eq!(simulator.path[lane1], EMPTY_KEY);
  }
}
