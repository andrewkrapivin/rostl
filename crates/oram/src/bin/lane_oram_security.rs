//! Reproducible stationary-stash experiment for the Lane ORAM simulator.

use std::{env, fs, process, str::FromStr, time::Instant};

use rostl_oram::lane_oram_sim::LaneOramSimulator;

#[derive(Clone, Copy, Debug)]
enum Workload {
  Cycle,
  Reverse,
  Affine,
  BitReverse,
  Uniform,
  Hot16,
  Same,
}

impl FromStr for Workload {
  type Err = String;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match value {
      "cycle" => Ok(Self::Cycle),
      "reverse" => Ok(Self::Reverse),
      "affine" => Ok(Self::Affine),
      "bit-reverse" => Ok(Self::BitReverse),
      "uniform" => Ok(Self::Uniform),
      "hot16" => Ok(Self::Hot16),
      "same" => Ok(Self::Same),
      _ => Err(format!("unknown workload: {value}")),
    }
  }
}

impl Workload {
  const fn name(self) -> &'static str {
    match self {
      Self::Cycle => "cycle",
      Self::Reverse => "reverse",
      Self::Affine => "affine",
      Self::BitReverse => "bit-reverse",
      Self::Uniform => "uniform",
      Self::Hot16 => "hot16",
      Self::Same => "same",
    }
  }

  fn key(self, operation: u64, n: usize, random: &mut OsRandomWords) -> usize {
    let mask = n - 1;
    match self {
      Self::Cycle => operation as usize & mask,
      Self::Reverse => mask - (operation as usize & mask),
      Self::Affine => (operation as usize).wrapping_mul(0x9e37_79b1) & mask,
      Self::BitReverse => {
        let bits = n.trailing_zeros();
        (operation as usize & mask).reverse_bits() >> (usize::BITS - bits)
      }
      Self::Uniform => random.next_usize() & mask,
      Self::Hot16 => random.next_usize() & mask.min(15),
      Self::Same => 0,
    }
  }
}

#[derive(Debug)]
struct Args {
  n: usize,
  b: usize,
  z: usize,
  warmup: u64,
  operations: u64,
  workload: Workload,
  deterministic_numerator: usize,
  deterministic_denominator: usize,
  output: String,
}

impl Args {
  fn parse() -> Result<Self, String> {
    let mut result = Self {
      n: 1 << 12,
      b: 2,
      z: 3,
      warmup: 1 << 24,
      operations: 1 << 28,
      workload: Workload::Cycle,
      deterministic_numerator: 0,
      deterministic_denominator: 1,
      output: "lane_oram_security.csv".to_owned(),
    };

    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
      let value = arguments.next().ok_or_else(|| format!("missing value for {argument}"))?;
      match argument.as_str() {
        "--n" => result.n = parse_number(&value)?,
        "--b" => result.b = parse_number(&value)?,
        "--z" => result.z = parse_number(&value)?,
        "--warmup" => result.warmup = parse_number(&value)?,
        "--operations" => result.operations = parse_number(&value)?,
        "--workload" => result.workload = value.parse()?,
        "--deterministic-numerator" => result.deterministic_numerator = parse_number(&value)?,
        "--deterministic-denominator" => result.deterministic_denominator = parse_number(&value)?,
        "--output" => result.output = value,
        _ => return Err(format!("unknown argument: {argument}")),
      }
    }

    if !result.n.is_power_of_two() {
      return Err("n must be a power of two".to_owned());
    }
    if !result.b.is_power_of_two() || result.b < 2 {
      return Err("b must be a power of two at least 2".to_owned());
    }
    if result.z == 0 || result.operations == 0 || result.deterministic_denominator == 0 {
      return Err("z, operations, and the deterministic denominator must be nonzero".to_owned());
    }
    let mut exact_capacity = 1usize;
    while exact_capacity < result.n {
      exact_capacity *= result.b;
    }
    if exact_capacity != result.n {
      return Err("n must also be an exact power of b".to_owned());
    }
    Ok(result)
  }
}

const OS_RANDOM_BATCH_WORDS: usize = 1 << 20;

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

fn perform_update(
  simulator: &mut LaneOramSimulator,
  positions: &mut [u32],
  key: usize,
  label_random: &mut OsRandomWords,
  eviction_counter: &mut u64,
  b: usize,
  deterministic_numerator: usize,
  deterministic_denominator: usize,
  background_paths: &mut Vec<u32>,
) -> rostl_oram::lane_oram_sim::SimUpdate {
  let old_pos = positions[key];
  let new_pos = (label_random.next_usize() & (simulator.max_n() - 1)) as u32;
  rate_deterministic_paths(
    *eviction_counter,
    simulator.max_n(),
    b,
    deterministic_numerator,
    deterministic_denominator,
    background_paths,
  );
  *eviction_counter = eviction_counter.wrapping_add(1);
  let result =
    simulator.update_with_background_paths(old_pos, new_pos, key as u32, background_paths);
  positions[key] = new_pos;
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

fn main() {
  let args = Args::parse().unwrap_or_else(|error| {
    eprintln!("error: {error}");
    process::exit(2);
  });
  let start = Instant::now();
  let mut simulator = LaneOramSimulator::new(args.n, args.z, args.b, args.n + 1);
  let mut positions = Vec::with_capacity(args.n);
  let mut label_random = OsRandomWords::new();
  let mut workload_random = OsRandomWords::new();
  let mut eviction_counter = 0u64;
  let mut background_paths =
    Vec::with_capacity(args.deterministic_numerator.div_ceil(args.deterministic_denominator));

  for key in 0..args.n {
    let initial_pos = (label_random.next_usize() & (simulator.max_n() - 1)) as u32;
    positions.push(initial_pos);
    let result = perform_update(
      &mut simulator,
      &mut positions,
      key,
      &mut label_random,
      &mut eviction_counter,
      args.b,
      args.deterministic_numerator,
      args.deterministic_denominator,
      &mut background_paths,
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
      &mut eviction_counter,
      args.b,
      args.deterministic_numerator,
      args.deterministic_denominator,
      &mut background_paths,
    );
    assert!(result.found && !result.overflowed);
  }

  let mut post_histogram = vec![0u64; args.n + 2];
  let mut demand_histogram = vec![0u64; args.n + 2];
  let mut post_sum = 0u128;
  let mut demand_sum = 0u128;
  let mut post_min = usize::MAX;
  let mut post_max = 0usize;
  let mut demand_min = usize::MAX;
  let mut demand_max = 0usize;

  for operation in 0..args.operations {
    let key = args.workload.key(args.warmup + operation, args.n, &mut workload_random);
    let result = perform_update(
      &mut simulator,
      &mut positions,
      key,
      &mut label_random,
      &mut eviction_counter,
      args.b,
      args.deterministic_numerator,
      args.deterministic_denominator,
      &mut background_paths,
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
  }

  let mut csv = String::from(
    "b,z,n,warmup,operations,workload,rng_source,initialization,metric,threshold,exceed_count,probability,log2_inverse,eviction_policy,path_schedule,deterministic_numerator,deterministic_denominator\n",
  );
  append_tail(&mut csv, &args, "post", &post_histogram, post_max);
  append_tail(&mut csv, &args, "insertion", &demand_histogram, demand_max);
  fs::write(&args.output, csv).unwrap_or_else(|error| {
    eprintln!("failed to write {}: {error}", args.output);
    process::exit(1);
  });

  let elapsed = start.elapsed().as_secs_f64();
  eprintln!(
    "B={} Z={} n={} workload={} eviction_policy=legacy-root-lane path_schedule=access-plus-rate-deterministic deterministic_rate={}/{} rng=os-getrandom initialization=random-old-and-new-paths warmup={} measured={} seconds={:.3} Mop/s={:.3} post[min/mean/max]={}/{:.3}/{} insertion[min/mean/max]={}/{:.3}/{} output={}",
    args.b,
    args.z,
    args.n,
    args.workload.name(),
    args.deterministic_numerator,
    args.deterministic_denominator,
    args.warmup,
    args.operations,
    elapsed,
    (args.warmup + args.operations) as f64 / elapsed / 1e6,
    post_min,
    post_sum as f64 / args.operations as f64,
    post_max,
    demand_min,
    demand_sum as f64 / args.operations as f64,
    demand_max,
    args.output,
  );
}

fn append_tail(output: &mut String, args: &Args, metric: &str, histogram: &[u64], maximum: usize) {
  let mut exceed_count = args.operations;
  for threshold in 0..=maximum {
    exceed_count -= histogram[threshold];
    let probability = exceed_count as f64 / args.operations as f64;
    let log2_inverse = if probability == 0.0 { f64::INFINITY } else { -probability.log2() };
    output.push_str(&format!(
      "{},{},{},{},{},{},os-getrandom,random-old-and-new-paths,{},{},{},{:.17},{:.9},legacy-root-lane,access-plus-rate-deterministic,{},{}\n",
      args.b,
      args.z,
      args.n,
      args.warmup,
      args.operations,
      args.workload.name(),
      metric,
      threshold,
      exceed_count,
      probability,
      log2_inverse,
      args.deterministic_numerator,
      args.deterministic_denominator,
    ));
  }
}
