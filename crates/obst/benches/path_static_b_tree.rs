#![allow(missing_docs)]
use criterion::{
  criterion_group, criterion_main, measurement::Measurement, AxisScale, BatchSize, BenchmarkId,
  Criterion, PlotConfiguration,
};

use std::hint::black_box;

use rostl_obst::path_static_b_tree::BigInlineBpTree;

const EXACT_MATCH: usize = 1;
const PREDECESSOR: usize = 0;
const SUCCESSOR: usize = 2;

type BenchTree = BigInlineBpTree<u64, u64, 15, 16, 7, 8, 8>;

fn input(size: usize) -> (Vec<u64>, Vec<u64>) {
  let keys = (1..=size as u64).rev().collect::<Vec<_>>();
  let values = keys.iter().map(|key| key * 10).collect::<Vec<_>>();
  (keys, values)
}

fn build_tree(size: usize) -> BenchTree {
  let (mut keys, mut values) = input(size);
  BenchTree::new(&mut keys, &mut values)
}

pub fn benchmark_path_static_b_tree_initialization<T: Measurement + 'static>(
  c: &mut Criterion<T>,
) {
  let mut group = c.benchmark_group(format!(
    "PathStaticBTree_Initialization/{}",
    std::any::type_name::<T>().split(':').next_back().unwrap()
  ));
  let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
  group.plot_config(plot_config);

  let test_set = &[128, 1024, 8192, 1 << 20, 1 << 25];

  for &size in test_set {
    group.bench_with_input(BenchmarkId::new("BigInlineBpTree", size), &size, |b, &size| {
      b.iter_batched(
        || input(size),
        |(mut keys, mut values)| {
          black_box(BenchTree::new(black_box(&mut keys), black_box(&mut values)));
        },
        BatchSize::SmallInput,
      );
    });
  }

  group.finish();
}

pub fn benchmark_path_static_b_tree_ops<T: Measurement + 'static>(c: &mut Criterion<T>) {
  let mut group = c.benchmark_group(format!(
    "PathStaticBTree_Ops/{}",
    std::any::type_name::<T>().split(':').next_back().unwrap()
  ));
  let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
  group.plot_config(plot_config);

  let test_set = &[128, 1024, 8192, 1 << 20, 1 << 25];

  for &size in test_set {
    group.bench_with_input(BenchmarkId::new("LookupExact", size), &size, |b, &size| {
      let mut tree = build_tree(size);
      let mut query = 1_u64;

      b.iter(|| {
        black_box(tree.point_query_kv(black_box(query)));
        query += 1;
        if query > size as u64 {
          query = 1;
        }
      });
    });

    group.bench_with_input(BenchmarkId::new("SearchKeyExact", size), &size, |b, &size| {
      let mut tree = build_tree(size);
      let mut query = 1_u64;

      b.iter(|| {
        black_box(tree.query_key(black_box(query), black_box(EXACT_MATCH)));
        query += 1;
        if query > size as u64 {
          query = 1;
        }
      });
    });

    group.bench_with_input(BenchmarkId::new("SearchKeySuccessor", size), &size, |b, &size| {
      let mut tree = build_tree(size);
      let mut query = 1_u64;
      let max_query = size as u64 - 1;

      b.iter(|| {
        black_box(tree.query_key(black_box(query), black_box(SUCCESSOR)));
        query += 1;
        if query > max_query {
          query = 1;
        }
      });
    });

    group.bench_with_input(BenchmarkId::new("SearchKeyPredecessor", size), &size, |b, &size| {
      let mut tree = build_tree(size);
      let mut query = 2_u64;

      b.iter(|| {
        black_box(tree.query_key(black_box(query), black_box(PREDECESSOR)));
        query += 1;
        if query > size as u64 {
          query = 2;
        }
      });
    });
  }

  group.finish();
}

criterion_group!(name = benches_time;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(3));
    targets = benchmark_path_static_b_tree_initialization, benchmark_path_static_b_tree_ops);
criterion_main!(benches_time);
