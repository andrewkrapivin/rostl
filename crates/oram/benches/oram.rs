#![allow(missing_docs)]
use criterion::{
  criterion_group, criterion_main, measurement::Measurement, AxisScale, BenchmarkGroup,
  BenchmarkId, Criterion, PlotConfiguration,
};

use std::hint::black_box;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use bytemuck::{Pod, Zeroable};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use rand::{rngs::StdRng, Rng, SeedableRng};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use rostl_oram::lane_oram::{Block32, LaneORAM};
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use rostl_oram::lane_oram_fixed::LaneORAMFixed;
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
struct Value56([u64; 7]);

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl_cmov_for_pod!(Value56);

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
fn add_lane_benchmark<T: Measurement + 'static, const Z: usize, const B: usize>(
  group: &mut BenchmarkGroup<'_, T>,
  name: &str,
  log_n: usize,
  size: usize,
  value_seed: u64,
) {
  group.bench_with_input(BenchmarkId::new(name, log_n), &size, |bench, &size| {
    let mut position_rng = StdRng::seed_from_u64(log_n as u64);
    let mut value_rng = StdRng::seed_from_u64(value_seed + log_n as u64);
    let mut updates = vec![Block32::default(); 1 << 20];
    for update in &mut updates {
      update.pos = position_rng.random_range(0..size) as u32;
      update.key = 0;
      value_rng.fill(&mut update.data[..]);
    }

    let mut oram = LaneORAM::<Z, 20, B>::new(size);
    oram.update(0, 0, 0, |block| {
      *block = Block32 { pos: 0, key: 0, data: [0; 56] };
    });
    let mut current_pos = 0;
    let mut update_index = 0;

    bench.iter(|| {
      let replacement = updates[update_index];
      let new_pos = replacement.pos;
      let (_, old) =
        oram.update(black_box(current_pos), black_box(new_pos), black_box(0), |block| {
          let old = *block;
          *block = replacement;
          old
        });
      current_pos = new_pos;
      update_index = (update_index + 1) & ((1 << 20) - 1);
      black_box(old);
    });
  });
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn add_fixed_lane_benchmark<
  T: Measurement + 'static,
  const Z: usize,
  const Y: usize,
  const S: usize,
  const B: usize,
  const E: usize,
>(
  group: &mut BenchmarkGroup<'_, T>,
  name: &str,
  log_n: usize,
  size: usize,
  value_seed: u64,
) {
  group.bench_with_input(BenchmarkId::new(name, log_n), &size, |bench, &size| {
    let mut position_rng = StdRng::seed_from_u64(log_n as u64);
    let mut value_rng = StdRng::seed_from_u64(value_seed + log_n as u64);
    let mut updates = vec![Block32::default(); 1 << 20];
    for update in &mut updates {
      update.pos = position_rng.random_range(0..size) as u32;
      update.key = 0;
      value_rng.fill(&mut update.data[..]);
    }
    let mut oram = LaneORAMFixed::<Z, Y, S, B, E>::new(size);
    let mut current_pos = position_rng.random_range(0..size) as u32;
    oram.update(current_pos, current_pos, 0, |block| {
      *block = Block32 { pos: current_pos, key: 0, data: [0; 56] };
    });
    let mut update_index = 0;
    bench.iter(|| {
      let replacement = updates[update_index];
      let new_pos = replacement.pos;
      let (_, old) =
        oram.update(black_box(current_pos), black_box(new_pos), black_box(0), |block| {
          let old = *block;
          *block = replacement;
          old
        });
      current_pos = new_pos;
      update_index = (update_index + 1) & ((1 << 20) - 1);
      black_box(old);
    });
  });
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

    group.bench_with_input(BenchmarkId::new("CircuitORAM_56B", log_n), &size, |b, &size| {
      let mut position_rng = StdRng::seed_from_u64(log_n as u64);
      let mut value_rng = StdRng::seed_from_u64(0x1800 + log_n as u64);
      let mut positions = vec![0u32; UPDATE_COUNT];
      let mut values = vec![Value56::default(); UPDATE_COUNT];
      for index in 0..UPDATE_COUNT {
        positions[index] = position_rng.random_range(0..size) as u32;
        value_rng.fill(&mut values[index].0);
      }

      let mut oram = CircuitORAM::<Value56>::new(size);
      oram.write_or_insert(0, 0, 0, Value56::default());
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

    add_fixed_lane_benchmark::<T, 2, 3, 64, 2, 1>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z2_Y3_S64",
      log_n,
      size,
      0x7100,
    );
    add_fixed_lane_benchmark::<T, 2, 3, 320, 2, 1>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z2_Y3_S320",
      log_n,
      size,
      0x7110,
    );
    add_fixed_lane_benchmark::<T, 3, 2, 64, 2, 1>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z3_Y2_S64",
      log_n,
      size,
      0x7200,
    );
    add_fixed_lane_benchmark::<T, 3, 2, 128, 2, 1>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z3_Y2_S128",
      log_n,
      size,
      0x7210,
    );
    add_fixed_lane_benchmark::<T, 3, 3, 48, 2, 1>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z3_Y3_S48",
      log_n,
      size,
      0x7300,
    );
    add_fixed_lane_benchmark::<T, 3, 3, 64, 2, 1>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z3_Y3_S64",
      log_n,
      size,
      0x7310,
    );
    add_fixed_lane_benchmark::<T, 3, 3, 8, 2, 1>(
      &mut group,
      "LaneORAMFixed_Profile_B2_Z3_Y3_E1_S8",
      log_n,
      size,
      0x73a0,
    );
    add_fixed_lane_benchmark::<T, 3, 3, 16, 2, 1>(
      &mut group,
      "LaneORAMFixed_Profile_B2_Z3_Y3_E1_S16",
      log_n,
      size,
      0x73b0,
    );
    add_fixed_lane_benchmark::<T, 3, 3, 32, 2, 1>(
      &mut group,
      "LaneORAMFixed_Profile_B2_Z3_Y3_E1_S32",
      log_n,
      size,
      0x73c0,
    );
    add_fixed_lane_benchmark::<T, 3, 3, 128, 2, 1>(
      &mut group,
      "LaneORAMFixed_Profile_B2_Z3_Y3_E1_S128",
      log_n,
      size,
      0x73d0,
    );
    add_fixed_lane_benchmark::<T, 3, 1, 20, 2, 1>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z3_Y1_S20_Control",
      log_n,
      size,
      0x7320,
    );
    add_fixed_lane_benchmark::<T, 3, 3, 20, 2, 1>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z3_Y3_S20_Control",
      log_n,
      size,
      0x7330,
    );
    add_fixed_lane_benchmark::<T, 3, 1, 64, 2, 1>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z3_Y1_S64_Control",
      log_n,
      size,
      0x7340,
    );
    add_fixed_lane_benchmark::<T, 2, 3, 80, 2, 2>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z2_Y3_E2_S80",
      log_n,
      size,
      0x7350,
    );
    add_fixed_lane_benchmark::<T, 3, 2, 72, 2, 2>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z3_Y2_E2_S72",
      log_n,
      size,
      0x7360,
    );
    add_fixed_lane_benchmark::<T, 4, 2, 72, 2, 1>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z4_Y2_E1_S72",
      log_n,
      size,
      0x7370,
    );
    add_fixed_lane_benchmark::<T, 2, 5, 104, 2, 1>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z2_Y5_E1_S104",
      log_n,
      size,
      0x7380,
    );
    add_fixed_lane_benchmark::<T, 1, 5, 336, 2, 2>(
      &mut group,
      "LaneORAMFixed_56B_B2_Z1_Y5_E2_S336",
      log_n,
      size,
      0x7390,
    );

    if log_n % 2 == 0 {
      add_fixed_lane_benchmark::<T, 4, 3, 64, 4, 1>(
        &mut group,
        "LaneORAMFixed_56B_B4_Z4_Y3_S64",
        log_n,
        size,
        0x7400,
      );
      add_fixed_lane_benchmark::<T, 4, 3, 96, 4, 1>(
        &mut group,
        "LaneORAMFixed_56B_B4_Z4_Y3_S96",
        log_n,
        size,
        0x7500,
      );
      add_fixed_lane_benchmark::<T, 4, 4, 96, 4, 1>(
        &mut group,
        "LaneORAMFixed_56B_B4_Z4_Y4_S96",
        log_n,
        size,
        0x7510,
      );
      add_fixed_lane_benchmark::<T, 5, 3, 112, 4, 1>(
        &mut group,
        "LaneORAMFixed_56B_B4_Z5_Y3_S112",
        log_n,
        size,
        0x7520,
      );
      add_fixed_lane_benchmark::<T, 1, 10, 176, 4, 3>(
        &mut group,
        "LaneORAMFixed_56B_B4_Z1_Y10_E3_S176",
        log_n,
        size,
        0x7530,
      );
      add_fixed_lane_benchmark::<T, 1, 8, 416, 4, 3>(
        &mut group,
        "LaneORAMFixed_56B_B4_Z1_Y8_E3_S416",
        log_n,
        size,
        0x7540,
      );
      group.bench_with_input(BenchmarkId::new("LaneORAM_56B_B4_Z5", log_n), &size, |b, &size| {
        let mut position_rng = StdRng::seed_from_u64(log_n as u64);
        let mut value_rng = StdRng::seed_from_u64(0x3000 + log_n as u64);
        let mut updates = vec![Block32::default(); UPDATE_COUNT];
        for index in 0..UPDATE_COUNT {
          updates[index].pos = position_rng.random_range(0..size) as u32;
          updates[index].key = 0;
          value_rng.fill(&mut updates[index].data[..]);
        }

        let mut oram = LaneORAM::<5, 20, 4>::new(size);
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

      add_lane_benchmark::<T, 6, 4>(&mut group, "LaneORAM_56B_B4_Z6", log_n, size, 0x4000);
    }

    if log_n % 3 == 0 {
      add_fixed_lane_benchmark::<T, 1, 16, 192, 8, 7>(
        &mut group,
        "LaneORAMFixed_56B_B8_Z1_Y16_E7_S192",
        log_n,
        size,
        0x7550,
      );
      add_lane_benchmark::<T, 9, 8>(&mut group, "LaneORAM_56B_B8_Z9", log_n, size, 0x5000);
      add_lane_benchmark::<T, 10, 8>(&mut group, "LaneORAM_56B_B8_Z10", log_n, size, 0x6000);
    }
  }

  group.finish();
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub fn benchmark_random_oram_updates<T: Measurement + 'static>(_c: &mut Criterion<T>) {}

criterion_group!(name = benches_time;
    config = Criterion::default().warm_up_time(std::time::Duration::from_millis(500)).measurement_time(std::time::Duration::from_secs(3));
    targets = benchmark_oram_initialization, benchmark_oram_ops, benchmark_random_oram_updates);
criterion_main!(benches_time);
