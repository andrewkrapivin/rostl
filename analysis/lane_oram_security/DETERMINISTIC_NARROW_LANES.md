# Deterministic sweeps for narrow Lane ORAM configurations

## Schedules and method

Every access uses a fresh position label from OS `getrandom`; initialization
also randomizes both old and replacement labels.  The workload cycles through
the keys, the stash is shared by all fixed lanes, and each lane performs one
Circuit-style chain on every scheduled path.

The normal accessed path is always evicted.  The public background schedules
are:

- **One sweep:** evict `digit_reverse_B(t mod N)`.  This visits every leaf once
  per `N` operations and gives two total path evictions per access.
- **Two-path sweep:** evict `digit_reverse_B(2*t+j)` for `j` in `0..2`.  For
  `B=4`, consecutive operations alternate between root quarters 0/1 and 2/3,
  every leaf is visited once per `N/2` operations, and there are three total
  path evictions per access.
- **Radix sweep:** evict `digit_reverse_B(B*t+j)` for every `j` in `0..B`.
  The paths occupy distinct root children and visit every leaf once per `N/B`
  operations.  This is the old binary complementary pair when `B=2`; for
  `B=4` it adds four paths, for five total path evictions per access.

Unit tests verify exact leaf coverage, distinct root children, equivalence of
the generalized `B=2` sequences to the prior bit-reversal implementation, and
block conservation under quaternary updates.  All 14 simulator tests pass.

Unless marked otherwise, scaling runs use a `2^25`-operation warm-up followed
by `2^20` measured operations.  The `N=2^10` pilots use `2^20` warm-up and
`2^22` measured operations.  Tables report post-eviction stash mean/maximum;
insertion-time demand is exactly one block larger.

## Binary `Y=1`

At `N=2^20`:

| Z | Bucket slots | One sweep mean/max | Two-sweep pair mean/max |
|---:|---:|---:|---:|
| 2 | 2 | 4.824279 / 28 | 0.005726 / 5 |
| 3 | 3 | 0.260403 / 5 | 0.000002861 / 1 |
| 4 | 4 | 0.250098 / 3 | 0 / 0* |
| 5 | 5 | 0.249735 / 1 | 0 / 0* |

`*` The `Z=4,5` two-sweep endpoints hit the 20-minute cap with the full run,
so their replacements use `2^24` warm-up and `2^19` measured operations.

The two configurations closest to their capacity threshold scale as follows:

| N | `Z=2`, two sweeps mean/max | `Z=3`, one sweep mean/max |
|---:|---:|---:|
| 2^10 | 0.001632 / 4 | 0.252720 / 4 |
| 2^14 | 0.003214 / 4 | 0.254880 / 5 |
| 2^18 | 0.004730 / 5 | 0.259093 / 5 |
| 2^20 | 0.005726 / 5 | 0.260403 / 5 |

Thus `Z=2,Y=1` is not convincing with one sweep and still worsens mildly with
two sweeps.  `Z=3,Y=1` with one sweep looks approximately stable over this
range, while `Z=4` and `Z=5` have substantially cleaner observed tails.  At
`N=2^20`, `Z=3` one-sweep has `Pr[S>1]=1.945e-3` and `Pr[S>4]=1.907e-6`;
`Z=4` has `Pr[S>1]=1.907e-5` and `Pr[S>2]=9.537e-7`; `Z=5` never exceeded
one in the measured sample.

## Quaternary `Y=1` and `Y=2`

The natural radix sweep adds one deterministic path in each of the four root
quarters.  The endpoint comparison is:

| Z,Y | Slots | One sweep at 2^10 | One sweep at 2^20 | Four-quarter sweep at 2^10 | Four-quarter sweep at 2^20 |
|---:|---:|---:|---:|---:|---:|
| 2,1 | 2 | 54.214 / 94 | not scaled | 3.485 / 20 | 3130.617 / 3362 |
| 3,1 | 3 | 10.404 / 31 | not scaled | 0.076 / 6 | 17.657 / 49 |
| 4,1 | 4 | 2.663 / 17 | 1730.242 / 1915 | 0.003101 / 3 | 0.045343 / 5 |
| 2,2 | 4 | 1.676 / 14 | 228.951 / 312 | 0.000137 / 2 | 0.000216 / 2 |
| 3,2 | 6 | 0.980 / 8 | 1.139703 / 9 | 0 / 0 | 0 / 0 |
| 4,2 | 8 | 0.952 / 4 | 0.959836 / 5 | 0 / 0 | 0 / 0 |

Intermediate controls distinguish slow growth from a plateau:

| N | `Z=4,Y=1`, four paths | `Z=2,Y=2`, four paths | `Z=3,Y=2`, one path | `Z=4,Y=2`, one path |
|---:|---:|---:|---:|---:|
| 2^10 | 0.003101 / 3 | 0.000137 / 2 | 0.980282 / 8 | 0.951768 / 4 |
| 2^14 | 0.011097 / 4 | 0.000132 / 2 | 1.018257 / 8 | 0.953214 / 5 |
| 2^18 | 0.031568 / 5 | 0.000192 / 2 | 1.087843 / 9 | 0.957003 / 6 |
| 2^20 | 0.045343 / 5 | 0.000216 / 2 | 1.139703 / 9 | 0.959836 / 5 |

No tested `B=4,Y=1` configuration has a clearly `N`-independent mean.  Even
`Z=4` with the expensive four-quarter sweep increases by about 15x.  In
contrast, `Y=2` gives two useful tradeoffs:

- `Z=2,Y=2` plus four paths uses only four tree slots and has maximum two, but
  requires five total path transfers per access.
- `Z=4,Y=2` plus one path uses eight tree slots and two total paths; its mean
  is stable near 0.96, with maximum five.  At `N=2^20`, its tail is
  `Pr[S>2]=2.397e-2`, `Pr[S>3]=1.793e-4`, and `Pr[S>4]=1.049e-5`.

For `Z=2,Y=2` with four paths at `N=2^20`, `Pr[S>0]=2.117e-4` and
`Pr[S>1]=4.768e-6`; no `S>2` event occurred.  `Z=3,4,Y=2` with four paths
ended with an empty stash after every measured operation.

## Quaternary two-path middle point

Adding exactly two deterministic background paths does not rescue `Y=1`:

| Z,Y | Mean/max at 2^10 | Mean/max at 2^20 |
|---:|---:|---:|
| 2,1 | 13.987 / 36 | 13522.309 / 13937 |
| 3,1 | 1.231 / 11 | 692.340 / 816 |
| 4,1 | 0.448 / 7 | 29.349 / 61 |

All three means grow drastically with `N`.  With `Y=2`, the same schedule is
much better:

| N | `Z=2,Y=2` mean/max | `Z=3,Y=2` mean/max | `Z=4,Y=2` mean/max |
|---:|---:|---:|---:|
| 2^10 | 0.398 / 6 | 0.375 / 3 | 0.375 / 1 |
| 2^14 | 0.420 / 6 | 0.376 / 3 | 0.375 / 1 |
| 2^18 | 0.445 / 6 | 0.376 / 3 | 0.375 / 2 |
| 2^20 | 0.458 / 7 | 0.377 / 3 | 0.375 / 2 |

`Z=2` has a visible slow trend, while `Z=3` and `Z=4` are empirically flat.
At `N=2^20`, their exact post-stash tails are:

| Z | Nonzero exceedance probabilities |
|---:|:---|
| 2 | `Pr[S>1]=2.429e-2`, `Pr[S>3]=5.379e-4`, `Pr[S>5]=1.526e-5`, `Pr[S>6]=9.537e-7` |
| 3 | `Pr[S>0]=0.376480`, `Pr[S>1]=2.069e-4`, `Pr[S>2]=1.907e-6` |
| 4 | `Pr[S>0]=0.374822`, `Pr[S>1]=9.537e-7` |

For three total path transfers per access, `Z=3,Y=2` is the apparent sweet
spot: six tree slots, stable scaling, and maximum three.  `Z=4,Y=2` spends two
extra tree slots to reduce the observed maximum to two.  `Z=2,Y=2` saves two
slots but has both a heavier tail and measurable growth with `N`.

These are finite empirical results, not `2^-96` bounds.  In particular, zero
events in `2^20` samples cannot resolve a cryptographic tail, and configurations
with visible `N` dependence should not be extrapolated as constant-stash
constructions.  The simulator measures placement behavior only; the real
payload implementation does not yet implement these extra-path schedules, and
each background path would add a complete path read/write.

Raw binary files are in `lane_deterministic_y1_runs/`; raw quaternary files are
in `lane_deterministic_b4_runs/`.  The generalized schedules are implemented in
`crates/oram/src/bin/lane_oram_minibucket_security.rs`.
