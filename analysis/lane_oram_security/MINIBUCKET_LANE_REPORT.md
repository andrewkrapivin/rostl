# Lane ORAM minibucket experiment (`Y`)

## Definition

`Z` is the number of fixed lanes and `Y` is the number of pooled slots in each
lane minibucket.  A physical tree bucket therefore contains `Z * Y` slots.  On
an access, each lane performs one Circuit-ORAM-style eviction chain: at most one
block is carried at a time, and a vacated source becomes the next destination.
The stash and that lane's root minibucket are treated as level zero.

The `Y` slots are at the same tree depth.  They must not be modeled as `Y`
successive one-slot levels: that would give the slots different placement
constraints and would not reproduce a Circuit ORAM bucket.

For `Z=1, Y=2`, one lane transition is regression-tested against the repository's
`CircuitORAM::evict_once_fast` on a uniquely ordered state.  This establishes
the local eviction equivalence.  It does **not** make the entire `Z=2, Y=2`
construction identical to standard Circuit ORAM: the lanes remain partitioned,
and both lane chains use the access path, whereas Circuit ORAM pools all bucket
slots and schedules independent/complementary eviction paths.

## Method

- `B=2`, cyclic accesses `0..N-1`, fresh independent path label on every access.
- Initial and replacement labels come directly from batched OS `getrandom`.
- Metadata-only simulation; every operation asserts that the requested block
  exists and periodic tests verify conservation of all `N` blocks.
- Entries below are post-eviction stash occupancy.  Insertion demand is exactly
  one larger for these runs.
- Runs through `log2(N)=10` use `2^20` warm-up and `2^20` samples.  Runs at
  `12..18` use `2^23` warm-up and `2^22` samples.  `log2(N)=20` uses `2^24`
  warm-up and `2^23` samples.  The `Z=2` values at 16 and 18 use an additional
  `2^25`-operation warm-up control.

## Results

| log2(N) | Z=2,Y=2 mean | Z=2,Y=2 max | Z=3,Y=2 mean | Z=3,Y=2 max |
|---:|---:|---:|---:|---:|
| 6  | 1.890 | 17 | 1.143 | 13 |
| 8  | 2.717 | 23 | 1.186 | 15 |
| 10 | 4.757 | 33 | 1.238 | 17 |
| 12 | 11.017 | 52 | 1.302 | 18 |
| 14 | 35.994 | 103 | 1.377 | 20 |
| 16 | 135.821 | 251 | 1.467 | 24 |
| 18 | 533.825 | 719 | 1.578 | 23 |
| 20 | 2084.770 | 2384 | 1.715 | 27 |

At large `N`, the `Z=2,Y=2` mean is about `0.002 N`; it is therefore not a
constant-stash construction under this schedule.  The long-warm-up controls
slightly increased rather than removed the backlog (`N=2^16`: 132.416 to
135.821; `N=2^18`: 519.792 to 533.825), ruling out random initialization as the
explanation.

`Z=3,Y=2` is dramatically better over the tested range.  At `N=2^20`, its
insertion-demand exceedance probabilities were approximately `1.09e-2` at
stash 8, `1.28e-3` at 12, `1.52e-4` at 16, `1.57e-5` at 20, and `1.67e-6` at
24.  The tail worsens slowly with `N`, so these measurements are promising but
are not a proof of an `N`-independent tail or cryptographic security.

## Reproduction

```sh
cargo run --release -p rostl-oram --bin lane_oram_minibucket_security -- \
  --b 2 --z 3 --y 2 --log-n 20 --warmup 16777216 --operations 8388608 \
  --workload cycle --output /tmp/z3-y2-n20.csv
```

The simulator intentionally isolates placement/security behavior.  It is not
an AVX-512 payload-performance implementation.
