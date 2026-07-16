#![allow(missing_docs)]

use std::{env, hint::black_box, time::Instant};

use rostl_oram::{
  big_kv_optimized_circuit_oram::{BigKvOptimizedCircuitOram, DATA_SIZE as SPLIT_DATA_SIZE},
  optimized_circuit_oram_big_values::{
    OptimizedCircuitORAMBigValuesWithStash, DATA_SIZE as CACHELINE_DATA_SIZE,
  },
};

#[inline(always)]
fn next_random(state: &mut u64) -> u64 {
  *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
  *state
}

fn run_split(log_n: u32, operations: usize) {
  let capacity = 1usize << log_n;
  let mut oram = BigKvOptimizedCircuitOram::new(capacity);
  oram.update(0, 0, 0, |value| *value = [0; SPLIT_DATA_SIZE]);
  oram.eviction_credit = 0;
  let mut current_pos = 0u64;
  let mut random_state = 0xd1b5_4a32_d192_ed03 ^ log_n as u64;
  let position_mask = oram.max_n as u64 - 1;

  let mut update = || {
    oram.eviction_credit = 0;
    let random = next_random(&mut random_state);
    let new_pos = (random >> 32) & position_mask;
    let (_, old) = oram.update(current_pos, new_pos, 0, |value| {
      let old = *value;
      *value = [random as u8; SPLIT_DATA_SIZE];
      old
    });
    oram.eviction_credit = 0;
    current_pos = new_pos;
    black_box(old);
  };

  for _ in 0..operations.min(1 << 16) {
    update();
  }
  let start = Instant::now();
  for _ in 0..operations {
    update();
  }
  let ordinary_ns = start.elapsed().as_secs_f64() * 1e9 / operations as f64;
  drop(update);

  for _ in 0..operations.min(1 << 14) {
    oram.benchmark_deterministic_eviction_batch();
  }
  let start = Instant::now();
  for _ in 0..operations {
    oram.benchmark_deterministic_eviction_batch();
  }
  let batch_ns = start.elapsed().as_secs_f64() * 1e9 / operations as f64;
  println!(
    "split40,2^{log_n},ordinary_ns={ordinary_ns:.3},batch_ns={batch_ns:.3},amortized_gap4_ns={:.3}",
    ordinary_ns + batch_ns / 4.0
  );
}

fn run_cacheline(log_n: u32, operations: usize) {
  let capacity = 1usize << log_n;
  let mut oram = OptimizedCircuitORAMBigValuesWithStash::<40>::new(capacity);
  oram.update(0, 0, 0, |value| *value = [0; CACHELINE_DATA_SIZE]);
  oram.eviction_credit = 0;
  let mut current_pos = 0u32;
  let mut random_state = 0xd1b5_4a32_d192_ed03 ^ log_n as u64;
  let position_mask = oram.max_n as u32 - 1;

  let mut update = || {
    oram.eviction_credit = 0;
    let random = next_random(&mut random_state);
    let new_pos = (random >> 32) as u32 & position_mask;
    let (_, old) = oram.update(current_pos, new_pos, 0, |value| {
      let old = *value;
      *value = [random as u8; CACHELINE_DATA_SIZE];
      old
    });
    oram.eviction_credit = 0;
    current_pos = new_pos;
    black_box(old);
  };

  for _ in 0..operations.min(1 << 16) {
    update();
  }
  let start = Instant::now();
  for _ in 0..operations {
    update();
  }
  let ordinary_ns = start.elapsed().as_secs_f64() * 1e9 / operations as f64;
  drop(update);

  for _ in 0..operations.min(1 << 14) {
    oram.benchmark_deterministic_eviction_batch();
  }
  let start = Instant::now();
  for _ in 0..operations {
    oram.benchmark_deterministic_eviction_batch();
  }
  let batch_ns = start.elapsed().as_secs_f64() * 1e9 / operations as f64;
  println!(
    "cacheline64,2^{log_n},ordinary_ns={ordinary_ns:.3},batch_ns={batch_ns:.3},amortized_gap4_ns={:.3}",
    ordinary_ns + batch_ns / 4.0
  );
}

fn main() {
  let mut args = env::args().skip(1);
  let implementation = args.next().expect("split | cacheline");
  let log_n: u32 = args.next().expect("log_n").parse().unwrap();
  let operations: usize =
    args.next().unwrap_or_else(|| (1usize << 20).to_string()).parse().unwrap();
  match implementation.as_str() {
    "split" => run_split(log_n, operations),
    "cacheline" => run_cacheline(log_n, operations),
    _ => panic!("unknown implementation"),
  }
}
