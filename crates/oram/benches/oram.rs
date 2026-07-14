#![allow(missing_docs)]
use criterion::{
  criterion_group, criterion_main, measurement::Measurement, AxisScale, BenchmarkId, Criterion,
  PlotConfiguration,
};

use std::hint::black_box;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use bytemuck::{Pod, Zeroable};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use rand::{rngs::StdRng, Rng, SeedableRng};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use rostl_oram::lane_oram::{Block32, LaneORAM};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use rostl_oram::optimized_circuit_oram::{OptimizedCircuitORAM, DATA_SIZE};
use rostl_oram::{
  circuit_oram::CircuitORAM, linear_oram::LinearORAM, recursive_oram::RecursivePositionMap,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use rostl_primitives::{
  cmov_body, cxchg_body, impl_cmov_for_pod,
  traits::{_Cmovbase, Cmov},
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct Value32([u64; 4]);

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl_cmov_for_pod!(Value32);

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

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn benchmark_random_oram_updates<T: Measurement + 'static>(c: &mut Criterion<T>) {
  const UPDATE_COUNT: usize = 1 << 20;

  let mut group = c.benchmark_group(format!(
    "ORAM_Random_Update/{}",
    std::any::type_name::<T>().split(':').next_back().unwrap()
  ));
  let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
  group.plot_config(plot_config);

  for log_n in 10..=24 {
    let size = 1usize << log_n;

    group.bench_with_input(BenchmarkId::new("CircuitORAM_32B", log_n), &size, |b, &size| {
      let mut position_rng = StdRng::seed_from_u64(log_n as u64);
      let mut value_rng = StdRng::seed_from_u64(0x1000 + log_n as u64);
      let mut positions = vec![0u32; UPDATE_COUNT];
      let mut values = vec![Value32::default(); UPDATE_COUNT];
      for index in 0..UPDATE_COUNT {
        positions[index] = position_rng.random_range(0..size) as u32;
        value_rng.fill(&mut values[index].0);
      }

      let mut oram = CircuitORAM::<Value32>::new(size);
      oram.write_or_insert(0, 0, 0, Value32::default());
      let mut current_pos = 0;
      let mut update_index = 0;

      b.iter(|| {
        let new_pos = positions[update_index];
        let replacement = values[update_index];
        let (_, old) =
          oram.update(black_box(current_pos), black_box(new_pos), black_box(0), |value| {
            let old = *value;
            *value = replacement;
            old
          });
        current_pos = new_pos;
        update_index = (update_index + 1) & (UPDATE_COUNT - 1);
        black_box(old);
      });
    });

    group.bench_with_input(
      BenchmarkId::new("OptimizedCircuitORAM_24B_2Lane", log_n),
      &size,
      |b, &size| {
        let mut position_rng = StdRng::seed_from_u64(log_n as u64);
        let mut value_rng = StdRng::seed_from_u64(0x1000 + log_n as u64);
        let mut positions = vec![0u32; UPDATE_COUNT];
        let mut values = vec![[0u8; DATA_SIZE]; UPDATE_COUNT];
        for index in 0..UPDATE_COUNT {
          positions[index] = position_rng.random_range(0..size) as u32;
          value_rng.fill(&mut values[index]);
        }

        let mut oram = OptimizedCircuitORAM::new(size);
        oram.write_or_insert(0, 0, 0, [0; DATA_SIZE]);
        let mut current_pos = 0;
        let mut update_index = 0;

        b.iter(|| {
          let new_pos = positions[update_index];
          let replacement = values[update_index];
          let (_, old) =
            oram.update(black_box(current_pos), black_box(new_pos), black_box(0), |value| {
              let old = *value;
              *value = replacement;
              old
            });
          current_pos = new_pos;
          update_index = (update_index + 1) & (UPDATE_COUNT - 1);
          black_box(old);
        });
      },
    );

    group.bench_with_input(BenchmarkId::new("LaneORAM_56B_Z3", log_n), &size, |b, &size| {
      let mut position_rng = StdRng::seed_from_u64(log_n as u64);
      let mut value_rng = StdRng::seed_from_u64(0x2000 + log_n as u64);
      let mut updates = vec![Block32::default(); UPDATE_COUNT];
      for index in 0..UPDATE_COUNT {
        updates[index].pos = position_rng.random_range(0..size) as u32;
        updates[index].key = 0;
        value_rng.fill(&mut updates[index].data[..]);
      }

      let mut oram = LaneORAM::<3, 20, 2>::new(size);
      oram.update(0, 0, 0, |block| {
        *block = Block32 { pos: 0, key: 0, data: [0; 56] };
      });
      let mut current_pos = 0;
      let mut update_index = 0;

      b.iter(|| {
        let replacement = updates[update_index];
        let new_pos = replacement.pos;
        let (_, old) =
          oram.update(black_box(current_pos), black_box(new_pos), black_box(0), |block| {
            let old = *block;
            *block = replacement;
            old
          });
        current_pos = new_pos;
        update_index = (update_index + 1) & (UPDATE_COUNT - 1);
        black_box(old);
      });
    });
  }

  group.finish();
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub fn benchmark_random_oram_updates<T: Measurement + 'static>(_c: &mut Criterion<T>) {}

criterion_group!(name = benches_time;
    config = Criterion::default().warm_up_time(std::time::Duration::from_millis(500)).measurement_time(std::time::Duration::from_secs(3));
    targets = benchmark_oram_initialization, benchmark_oram_ops, benchmark_random_oram_updates);
criterion_main!(benches_time);
