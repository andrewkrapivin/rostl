#![allow(missing_docs)]
use bytemuck::{Pod, Zeroable};
use criterion::{
  criterion_group, criterion_main, measurement::Measurement, AxisScale, BatchSize, BenchmarkId,
  Criterion, PlotConfiguration, Throughput,
};
use rand::{rng, RngCore};
use rostl_primitives::{
  cmov_body, cxchg_body, impl_cmov_for_pod,
  traits::{_Cmovbase, Cmov},
};

use std::{hint::black_box, mem::size_of, time::Duration};

use rostl_oram::{
  circuit_oram::CircuitORAM, fast_buckets::Cacheline_Counter_Bucket, linear_oram::LinearORAM,
  prelude::PositionType, recursive_oram::RecursivePositionMap,
};
use rostl_oram::{
  fast_circuit_oram::FastCircuitCounterORAM, fast_circuit_oram_15::FastCircuitCounterORAM15,
  fast_circuit_oram_alt::FastCircuitCounterORAMAlt,
};

const BENCH_INTERNAL_NODE_FAN_OUT: usize = 64 / size_of::<PositionType>();
const BENCH_INTERNAL_NODE_MASK: usize = BENCH_INTERNAL_NODE_FAN_OUT - 1;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct BenchInternalNode([PositionType; BENCH_INTERNAL_NODE_FAN_OUT]);
impl_cmov_for_pod!(BenchInternalNode);

impl Default for BenchInternalNode {
  fn default() -> Self {
    Self([PositionType::default(); BENCH_INTERNAL_NODE_FAN_OUT])
  }
}

pub fn benchmark_oram_initialization<T: Measurement + 'static>(c: &mut Criterion<T>) {
  let mut group = c.benchmark_group(format!(
    "ORAM_Initialization/{}",
    std::any::type_name::<T>().split(':').next_back().unwrap()
  ));
  let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
  group.plot_config(plot_config);

  let test_set = &[128, 1 << 10, 1 << 20];

  for &size in test_set {
    group.bench_with_input(BenchmarkId::new("LinearORAM", size), &size, |b, &size| {
      b.iter(|| {
        black_box(LinearORAM::<u64>::new(size));
      });
    });
    group.bench_with_input(BenchmarkId::new("CircuitORAM", size), &size, |b, &size| {
      b.iter(|| {
        black_box(CircuitORAM::<u64>::new(size));
      });
    });
    group.bench_with_input(BenchmarkId::new("RecursivePositionMap", size), &size, |b, &size| {
      b.iter(|| {
        black_box(RecursivePositionMap::new(size));
      });
    });
  }
}

pub fn benchmark_oram_ops<T: Measurement + 'static>(c: &mut Criterion<T>) {
  let mut group = c.benchmark_group(format!(
    "ORAM_Ops/{}",
    std::any::type_name::<T>().split(':').next_back().unwrap()
  ));
  let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
  group.plot_config(plot_config);

  let test_set = &[128, 1 << 10, 1 << 20];

  for &size in test_set {
    group.bench_with_input(BenchmarkId::new("LinearORAM_Read", size), &size, |b, &size| {
      let mut oram = LinearORAM::<u64>::new(size);
      oram.write(0, 0);
      b.iter(|| {
        let mut _ign = black_box(0);
        oram.read(black_box(0), black_box(&mut _ign));
      });
    });
    group.bench_with_input(BenchmarkId::new("CircuitORAM_Read", size), &size, |b, &size| {
      let mut oram = CircuitORAM::<u64>::new(size);
      oram.write_or_insert(0, 0, 0, 0);
      b.iter(|| {
        let mut _ign = black_box(0);
        oram.read(black_box(0), black_box(0), black_box(0), &mut _ign);
      });
    });
    group.bench_with_input(BenchmarkId::new("CircuitORAM_Write", size), &size, |b, &size| {
      let mut oram = CircuitORAM::<u64>::new(size);
      oram.write_or_insert(0, 0, 0, 0);
      b.iter(|| {
        oram.write(black_box(0), black_box(0), black_box(0), black_box(0));
      });
    });
    group.bench_with_input(BenchmarkId::new("RecursivePositionMap", size), &size, |b, &size| {
      let mut oram = RecursivePositionMap::new(size);
      b.iter(|| {
        oram.access_position(black_box(0), black_box(0));
      });
    });
  }

  group.finish();
}

fn two_increment_counter_bucket() -> Cacheline_Counter_Bucket {
  let mut bucket = Cacheline_Counter_Bucket::default();
  for index in 0..64 {
    bucket.increment_counter(index);
    bucket.increment_counter(index);
  }
  bucket
}

pub fn benchmark_fast_circuit_oram_bucket<T: Measurement + 'static>(c: &mut Criterion<T>) {
  const READ_OPS: u64 = 4096;
  const INCREMENT_OPS: u64 = 64;

  let mut group = c.benchmark_group(format!(
    "FastCircuitORAMBucket/{}",
    std::any::type_name::<T>().split(':').next_back().unwrap()
  ));

  let bucket = two_increment_counter_bucket();
  group.throughput(Throughput::Elements(READ_OPS));
  group.bench_function("get_counter", |b| {
    b.iter(|| {
      let mut acc = 0u64;
      for i in 0..READ_OPS {
        acc ^= bucket.get_counter(black_box((i as usize) & 63));
      }
      black_box(acc);
    });
  });

  group.throughput(Throughput::Elements(INCREMENT_OPS));
  group.bench_function("increment_grow_width1_to_width2", |b| {
    b.iter_batched(
      two_increment_counter_bucket,
      |mut bucket| {
        for i in 0..INCREMENT_OPS {
          black_box(bucket.increment_counter(black_box(i as usize)));
        }
        black_box(bucket);
      },
      BatchSize::SmallInput,
    );
  });

  group.bench_function("increment_grow_width0_to_width1", |b| {
    b.iter_batched(
      Cacheline_Counter_Bucket::default,
      |mut bucket| {
        for i in 0..INCREMENT_OPS {
          black_box(bucket.increment_counter(black_box(i as usize)));
        }
        black_box(bucket);
      },
      BatchSize::SmallInput,
    );
  });

  group.throughput(Throughput::Elements(1));
  group.bench_function("split_into_blocks", |b| {
    b.iter(|| {
      black_box(bucket.split_into_blocks());
    });
  });

  let blocks = bucket.split_into_blocks();
  group.bench_function("merge_blocks", |b| {
    b.iter(|| {
      black_box(Cacheline_Counter_Bucket::merge_blocks(black_box(blocks)));
    });
  });

  group.finish();
}

fn setup_circuit_counter_oram(size: usize) -> (CircuitORAM<u64>, Vec<PositionType>) {
  let mut oram = CircuitORAM::<u64>::new(size);
  let mut positions = vec![0; size];

  for (key, pos) in positions.iter_mut().enumerate() {
    *pos = ((key * 17 + 3) % oram.max_n) as PositionType;
    oram.write_or_insert(0, *pos, key, 0);
  }

  (oram, positions)
}

fn setup_internal_node_circuit_oram_for_keys(
  counter_space_size: usize,
  keys: &[usize],
) -> (CircuitORAM<BenchInternalNode>, Vec<PositionType>) {
  let node_count = (counter_space_size / BENCH_INTERNAL_NODE_FAN_OUT).max(1);
  let mut oram = CircuitORAM::<BenchInternalNode>::new(node_count);
  let mut rng = rng();
  let mut positions = vec![0; keys.len()];

  for (key, pos) in keys.iter().zip(positions.iter_mut()) {
    let node_key = key / BENCH_INTERNAL_NODE_FAN_OUT;
    *pos = bench_pos(&mut rng, oram.max_n);
    oram.write_or_insert(0, *pos, node_key, BenchInternalNode::default());
  }

  (oram, positions)
}

#[inline]
fn bench_key(step: usize, size: usize) -> usize {
  (step * 37 + 11) & (size - 1)
}

#[inline]
fn bench_pos(rng: &mut rand::rngs::ThreadRng, size: usize) -> PositionType {
  debug_assert!(size.is_power_of_two());
  (rng.next_u32() & (size as u32 - 1)) as PositionType
}

fn setup_fast_counter_positions(max_blocks: usize) -> Vec<PositionType> {
  let mut rng = rng();
  (0..max_blocks).map(|_| bench_pos(&mut rng, max_blocks)).collect()
}

#[inline]
fn fast_counter_read_key(
  oram: &mut FastCircuitCounterORAM,
  positions: &mut [PositionType],
  rng: &mut rand::rngs::ThreadRng,
  key: usize,
) -> u64 {
  let map_key = oram.position_map_key_for_key(key);
  let pos = positions[map_key];
  let new_pos = bench_pos(rng, oram.max_blocks);
  let value = oram.read_key(pos, new_pos, key);
  positions[map_key] = new_pos;
  value
}

#[inline]
fn fast_counter_read_key_and_incr(
  oram: &mut FastCircuitCounterORAM,
  positions: &mut [PositionType],
  rng: &mut rand::rngs::ThreadRng,
  key: usize,
) -> u64 {
  let map_key = oram.position_map_key_for_key(key);
  let pos = positions[map_key];
  let new_pos = bench_pos(rng, oram.max_blocks);
  let value = oram.read_key_and_incr(pos, new_pos, key);
  positions[map_key] = new_pos;
  value
}

#[inline]
fn fast_counter_alt_read_key(
  oram: &mut FastCircuitCounterORAMAlt,
  positions: &mut [PositionType],
  rng: &mut rand::rngs::ThreadRng,
  key: usize,
) -> u64 {
  let map_key = oram.position_map_key_for_key(key);
  let pos = positions[map_key];
  let new_pos = bench_pos(rng, oram.max_blocks);
  let value = oram.read_key(pos, new_pos, key);
  positions[map_key] = new_pos;
  value
}

#[inline]
fn fast_counter_alt_read_key_and_incr(
  oram: &mut FastCircuitCounterORAMAlt,
  positions: &mut [PositionType],
  rng: &mut rand::rngs::ThreadRng,
  key: usize,
) -> u64 {
  let map_key = oram.position_map_key_for_key(key);
  let pos = positions[map_key];
  let new_pos = bench_pos(rng, oram.max_blocks);
  let value = oram.read_key_and_incr(pos, new_pos, key);
  positions[map_key] = new_pos;
  value
}

#[inline]
fn fast_counter_15_read_key(
  oram: &mut FastCircuitCounterORAM15,
  positions: &mut [PositionType],
  rng: &mut rand::rngs::ThreadRng,
  key: usize,
) -> u64 {
  let map_key = oram.position_map_key_for_key(key);
  let pos = positions[map_key];
  let new_pos = bench_pos(rng, oram.max_blocks);
  let value = oram.read_key(pos, new_pos, key);
  positions[map_key] = new_pos;
  value
}

#[inline]
fn fast_counter_15_read_key_and_incr(
  oram: &mut FastCircuitCounterORAM15,
  positions: &mut [PositionType],
  rng: &mut rand::rngs::ThreadRng,
  key: usize,
) -> u64 {
  let map_key = oram.position_map_key_for_key(key);
  let pos = positions[map_key];
  let new_pos = bench_pos(rng, oram.max_blocks);
  let value = oram.read_key_and_incr(pos, new_pos, key);
  positions[map_key] = new_pos;
  value
}

pub fn benchmark_fast_counter_oram_ops<T: Measurement + 'static>(c: &mut Criterion<T>) {
  const OPS_PER_ITER: usize = 64;

  let mut group = c.benchmark_group(format!(
    "FastCounterORAM_Compare/{}",
    std::any::type_name::<T>().split(':').next_back().unwrap()
  ));
  group.throughput(Throughput::Elements(OPS_PER_ITER as u64));

  for &size in &[1 << 10, 1 << 15] {
    group.bench_function(BenchmarkId::new("FastCounterORAMPacked_Read", size), |b| {
      let mut oram = FastCircuitCounterORAM::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = bench_key(step, size);
          acc ^= fast_counter_read_key(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    group.bench_function(BenchmarkId::new("CircuitORAM_Read", size), |b| {
      let (mut oram, mut positions) = setup_circuit_counter_oram(size);
      let mut rng = rng();
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = bench_key(step, size);
          let pos = positions[key];
          let new_pos = bench_pos(&mut rng, oram.max_n);
          let mut value = 0u64;
          let found = oram.read(pos, new_pos, key, &mut value);
          debug_assert!(found);
          positions[key] = new_pos;
          acc ^= value;
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    group.bench_function(BenchmarkId::new("FastCounterORAM1CLBlock_Read", size), |b| {
      let mut oram = FastCircuitCounterORAMAlt::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = bench_key(step, size);
          acc ^= fast_counter_alt_read_key(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    group.bench_function(BenchmarkId::new("FastCounterORAM1_5CLBlock_Read", size), |b| {
      let mut oram = FastCircuitCounterORAM15::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = bench_key(step, size);
          acc ^= fast_counter_15_read_key(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    group.bench_function(BenchmarkId::new("FastCounterORAMPacked_ReadAndIncr", size), |b| {
      let mut oram = FastCircuitCounterORAM::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = bench_key(step, size);
          acc ^=
            fast_counter_read_key_and_incr(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    group.bench_function(BenchmarkId::new("FastCounterORAM1_5CLBlock_ReadAndIncr", size), |b| {
      let mut oram = FastCircuitCounterORAM15::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = bench_key(step, size);
          acc ^=
            fast_counter_15_read_key_and_incr(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    group.bench_function(BenchmarkId::new("FastCounterORAM1CLBlock_ReadAndIncr", size), |b| {
      let mut oram = FastCircuitCounterORAMAlt::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = bench_key(step, size);
          acc ^=
            fast_counter_alt_read_key_and_incr(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    group.bench_function(BenchmarkId::new("CircuitORAM_ReadAndIncr", size), |b| {
      let (mut oram, mut positions) = setup_circuit_counter_oram(size);
      let mut rng = rng();
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = bench_key(step, size);
          let pos = positions[key];
          let new_pos = bench_pos(&mut rng, oram.max_n);
          let (found, old) = oram.update(pos, new_pos, key, |value| {
            let old = *value;
            *value = value.wrapping_add(1);
            old
          });
          debug_assert!(found);
          positions[key] = new_pos;
          acc ^= old;
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });
  }

  group.finish();
}

pub fn benchmark_fast_counter_oram_large_n<T: Measurement + 'static>(c: &mut Criterion<T>) {
  const KEY_COUNT: usize = 16;
  const OPS_PER_ITER: usize = 16;
  const SIZES: &[(usize, bool)] = &[
    (1 << 20, true),
    (1 << 24, true),
    // The internal-node CircuitORAM at 2^28 still allocates several GiB.
    (1 << 28, false),
  ];

  let mut group = c.benchmark_group(format!(
    "FastCounterORAM_LargeN/{}",
    std::any::type_name::<T>().split(':').next_back().unwrap()
  ));
  group.sample_size(10);
  group.warm_up_time(Duration::from_millis(250));
  group.measurement_time(Duration::from_millis(750));
  group.throughput(Throughput::Elements(OPS_PER_ITER as u64));

  for &(size, include_circuit) in SIZES {
    let keys: [usize; KEY_COUNT] = core::array::from_fn(|i| bench_key(i, size));

    group.bench_function(BenchmarkId::new("FastCounterORAMPacked_Read", size), |b| {
      let mut oram = FastCircuitCounterORAM::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      for key in keys {
        black_box(fast_counter_read_key(&mut oram, &mut positions, &mut rng, black_box(key)));
      }
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = keys[step & (KEY_COUNT - 1)];
          acc ^= fast_counter_read_key(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    if include_circuit {
      group.bench_function(BenchmarkId::new("CircuitORAMInternalNode_Read", size), |b| {
        let (mut oram, mut positions) = setup_internal_node_circuit_oram_for_keys(size, &keys);
        let mut rng = rng();
        let mut step = 0usize;
        b.iter(|| {
          let mut acc = 0u32;
          for _ in 0..OPS_PER_ITER {
            let key_index = step & (KEY_COUNT - 1);
            let key = keys[key_index];
            let node_key = key / BENCH_INTERNAL_NODE_FAN_OUT;
            let offset = key & BENCH_INTERNAL_NODE_MASK;
            let pos = positions[key_index];
            let new_pos = bench_pos(&mut rng, oram.max_n);
            let mut value = BenchInternalNode::default();
            let found = oram.read(pos, new_pos, node_key, &mut value);
            debug_assert!(found);
            positions[key_index] = new_pos;
            acc ^= value.0[offset];
            step = step.wrapping_add(1);
          }
          black_box(acc);
        });
      });
    }

    group.bench_function(BenchmarkId::new("FastCounterORAM1CLBlock_Read", size), |b| {
      let mut oram = FastCircuitCounterORAMAlt::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      for key in keys {
        black_box(fast_counter_alt_read_key(&mut oram, &mut positions, &mut rng, black_box(key)));
      }
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = keys[step & (KEY_COUNT - 1)];
          acc ^= fast_counter_alt_read_key(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    group.bench_function(BenchmarkId::new("FastCounterORAM1_5CLBlock_Read", size), |b| {
      let mut oram = FastCircuitCounterORAM15::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      for key in keys {
        black_box(fast_counter_15_read_key(&mut oram, &mut positions, &mut rng, black_box(key)));
      }
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = keys[step & (KEY_COUNT - 1)];
          acc ^= fast_counter_15_read_key(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    group.bench_function(BenchmarkId::new("FastCounterORAMPacked_ReadAndIncr", size), |b| {
      let mut oram = FastCircuitCounterORAM::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = keys[step & (KEY_COUNT - 1)];
          acc ^=
            fast_counter_read_key_and_incr(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    group.bench_function(BenchmarkId::new("FastCounterORAM1_5CLBlock_ReadAndIncr", size), |b| {
      let mut oram = FastCircuitCounterORAM15::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = keys[step & (KEY_COUNT - 1)];
          acc ^=
            fast_counter_15_read_key_and_incr(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    group.bench_function(BenchmarkId::new("FastCounterORAM1CLBlock_ReadAndIncr", size), |b| {
      let mut oram = FastCircuitCounterORAMAlt::new(size);
      let mut positions = setup_fast_counter_positions(oram.max_blocks);
      let mut rng = rng();
      let mut step = 0usize;
      b.iter(|| {
        let mut acc = 0u64;
        for _ in 0..OPS_PER_ITER {
          let key = keys[step & (KEY_COUNT - 1)];
          acc ^=
            fast_counter_alt_read_key_and_incr(&mut oram, &mut positions, &mut rng, black_box(key));
          step = step.wrapping_add(1);
        }
        black_box(acc);
      });
    });

    if include_circuit {
      group.bench_function(BenchmarkId::new("CircuitORAMInternalNode_ReadAndIncr", size), |b| {
        let (mut oram, mut positions) = setup_internal_node_circuit_oram_for_keys(size, &keys);
        let mut rng = rng();
        let mut step = 0usize;
        b.iter(|| {
          let mut acc = 0u32;
          for _ in 0..OPS_PER_ITER {
            let key_index = step & (KEY_COUNT - 1);
            let key = keys[key_index];
            let node_key = key / BENCH_INTERNAL_NODE_FAN_OUT;
            let offset = key & BENCH_INTERNAL_NODE_MASK;
            let pos = positions[key_index];
            let new_pos = bench_pos(&mut rng, oram.max_n);
            let (found, old) = oram.update(pos, new_pos, node_key, |node| {
              let old = node.0[offset];
              node.0[offset] = old.wrapping_add(1);
              old
            });
            debug_assert!(found);
            positions[key_index] = new_pos;
            acc ^= old;
            step = step.wrapping_add(1);
          }
          black_box(acc);
        });
      });
    }
  }

  group.finish();
}

criterion_group!(name = benches_time;
    config = Criterion::default().warm_up_time(std::time::Duration::from_millis(500)).measurement_time(std::time::Duration::from_secs(3));
    targets = benchmark_oram_initialization, benchmark_oram_ops, benchmark_fast_circuit_oram_bucket, benchmark_fast_counter_oram_ops, benchmark_fast_counter_oram_large_n);
criterion_main!(benches_time);
