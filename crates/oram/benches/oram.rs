#![allow(missing_docs)]
use criterion::{
  criterion_group, criterion_main, measurement::Measurement, AxisScale, BatchSize, BenchmarkId,
  Criterion, PlotConfiguration, Throughput,
};

use std::hint::black_box;

use rostl_oram::{
  circuit_oram::CircuitORAM, fast_circuit_oram::Cacheline_Counter_Bucket, linear_oram::LinearORAM,
  recursive_oram::RecursivePositionMap,
};

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

  group.finish();
}

criterion_group!(name = benches_time;
    config = Criterion::default().warm_up_time(std::time::Duration::from_millis(500)).measurement_time(std::time::Duration::from_secs(3));
    targets = benchmark_oram_initialization, benchmark_oram_ops, benchmark_fast_circuit_oram_bucket);
criterion_main!(benches_time);
