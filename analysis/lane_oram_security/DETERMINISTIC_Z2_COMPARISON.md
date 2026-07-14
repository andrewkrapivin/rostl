# Deterministic Circuit ORAM at bucket capacity two

## Schedules

Paper bucket capacity `Z=2` is represented by repository parameters
`B=2,Z=1,Y=2`. Two schedules were compared:

1. `paper-deterministic`: Algorithm 6 exactly, with eviction paths
   `bitrev(2t mod N)` and `bitrev((2t+1) mod N)` and no full eviction on
   the accessed path.
2. `access-plus-paper-deterministic`: the same two background paths, plus a
   full Circuit eviction on the already-loaded accessed path before it is
   written back.

The second schedule does not add another path read/write: the accessed path
must already be loaded for `ReadAndRm`. It adds eviction target/source
selection and block movement on that path.

Tests verify that each deterministic pair lies in opposite root halves, that
one period covers every leaf exactly once, and that both schedules conserve
every block.

## N-scaling results

Every run used OS randomness for labels, the cyclic worst-case workload,
`2^25` warm-up operations, and `2^20` measured operations. Values below are
post-eviction stash occupancy.

| N | Paper mean | Paper max | Access+paper mean | Access+paper max | Mean reduction |
|---:|---:|---:|---:|---:|---:|
| 2^10 | 0.052625 | 8 | 0.003234 | 5 | 16.3x |
| 2^14 | 0.061175 | 8 | 0.002829 | 4 | 21.6x |
| 2^18 | 0.063293 | 10 | 0.002725 | 4 | 23.2x |
| 2^20 | 0.065992 | 7 | 0.003056 | 4 | 21.6x |

Both schedules are empirically independent of `N` to within a small constant
variation. This reproduces the paper's empirical claim for deterministic
bucket capacity two.

At `N=2^20`, the extra accessed-path eviction changes representative tail
probabilities as follows:

| Event | Paper deterministic | Access + paper deterministic | Reduction |
|---|---:|---:|---:|
| `Pr[S>0]` | 0.044829 | 0.002716 | 16.5x |
| `Pr[S>2]` | 0.004853 | 0.00003624 | 134x |
| `Pr[S>4]` | 639 / 2^20 | 0 / 2^20 | unresolved beyond sample limit |

A zero observation is not a cryptographic bound; these runs directly resolve
only about 20 tail bits.

## Simulator cost

| N | Paper seconds | Access+paper seconds | Added runtime |
|---:|---:|---:|---:|
| 2^10 | 62.481 | 84.718 | 35.6% |
| 2^14 | 129.262 | 171.106 | 32.4% |
| 2^18 | 247.604 | 316.420 | 27.8% |
| 2^20 | 339.566 | 429.814 | 26.6% |

This is metadata-simulator runtime, not a payload benchmark. It isolates the
additional eviction computation but should not be treated as the exact
production ORAM slowdown.

Raw CSVs are in `deterministic_z2_runs/`. The schedules are implemented in
`crates/oram/src/bin/lane_oram_minibucket_security.rs`.
