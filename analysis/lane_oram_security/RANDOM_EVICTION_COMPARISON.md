# Random Circuit eviction versus two width-3 lanes

## Notation and schedules

The Circuit ORAM paper's bucket capacity `Z` corresponds to this repository's
pooled minibucket width `Y`. Thus paper-`Z=3` is represented by
`B=2, Z=1, Y=3`. Repository `Z=2, Y=3` instead means two separate
three-slot lanes, for six physical slots per tree node.

Three schedules were measured:

1. `access-plus-uniform`: evict the accessed old-label path and one
   independently uniform path in a single pooled width-3 tree.
2. `paper-random`: perform no full eviction on the accessed path, then apply
   Algorithm 5 from the paper: one independently random path from each half of
   the binary tree, again in a pooled width-3 tree.
3. `same`: repository `Z=2,Y=3`, one Circuit eviction chain per fixed lane
   on the accessed old-label path.

All new labels, initial labels, queue choices, and random eviction paths use OS
`getrandom`. The workload cycles through keys `0..N-1`. The simulator has a
direct one-lane/one-chain equivalence test against the real Circuit ORAM
eviction and block-conservation tests for both background-path schedules.

## Formal result from the paper

For randomized eviction, the paper proves only for bucket capacity at least
five:

```text
Pr[stash > R] <= 42 * 0.6^R,  paper-Z >= 5.
```

If it applied, this expression would fall below `2^-96` at `R=138`.
It does **not** apply to the width-3 experiments here. The paper says width 3
looked bounded empirically, but Figure 3 and its capacity-scaling experiment
use deterministic-order eviction. Therefore there is no published formal
width-3 random-eviction bound to transfer to these runs.

## Measurements

The table reports post-eviction stash occupancy. Insertion demand, which is the
relevant quantity for a finite stash overflow check, has mean and maximum
exactly one larger.

| N | Schedule | Warm-up | Samples | Mean | Observed max |
|---:|---|---:|---:|---:|---:|
| 2^10 | accessed + uniform | 2^24 | 2^26 | 1.062 | 24 |
| 2^10 | paper random | 2^24 | 2^26 | 0.429 | 26 |
| 2^10 | lane Z=2,Y=3 | 2^24 | 2^26 | 1.839 | 40 |
| 2^18 | accessed + uniform | 2^25 | 2^22 | 4.149 | 46 |
| 2^18 | paper random | 2^25 | 2^22 | 4.392 | 45 |
| 2^18 | lane Z=2,Y=3 | 2^25 | 2^22 | 3.912 | 40 |
| 2^20 | accessed + uniform | 2^25 | 2^20 | 8.443 | 54 |
| 2^20 | paper random | 2^25 | 2^20 | 13.366 | 61 |
| 2^20 | lane Z=2,Y=3 | 2^25 | 2^20 | 5.256 | 49 |

The large increase with `N` remains after the paper's full `2^25`-operation
warm-up. Width-3 randomized eviction is therefore not showing an
`N`-independent stash distribution in these experiments. At `N=2^20`, the
two-lane construction is better than both pooled randomized schedules, though
it too grows with `N`.

## Empirical tail estimates

For insertion demand, a straight line was fitted to
`log2 Pr[demand > S]` over tail windows selected by exceedance count. The
following ranges vary the fit window; they are sensitivity ranges, not
confidence intervals:

| N | Schedule | Empirical S for per-operation 2^-96 |
|---:|---|---:|
| 2^10 | accessed + uniform | about 105 |
| 2^10 | paper random | about 121 |
| 2^10 | lane Z=2,Y=3 | about 144 |
| 2^18 | accessed + uniform | 247--276 |
| 2^18 | paper random | 268--269 |
| 2^18 | lane Z=2,Y=3 | 215--233 |
| 2^20 | accessed + uniform | 272--298 |
| 2^20 | paper random | 303--389 |
| 2^20 | lane Z=2,Y=3 | 307--335 |

At `N=2^20`, cautious rounded engineering values under the assumed
log-linear continuation are 304, 400, and 336 blocks respectively. Only about
20 tail bits were directly observed at that capacity, so extrapolating to 96
bits is substantial and cannot establish a cryptographic bound.

## Interpretation

Paper-random balances service at the root by selecting one path in each half,
but deeper prefixes still experience geometrically distributed gaps and
bursts. Deterministic bit-reversal gives regular service to every prefix,
which is why the paper's deterministic strategy can behave much better at
small bucket widths. The randomized proof needs bucket capacity five to
provide enough slack for concentration across all root subtrees.

`access-plus-uniform` gives the just-accessed branch immediate eviction
service and eventually outperforms paper-random at large `N`, despite lacking
the latter's exact root-half balance. The two-lane design has twice the tree
capacity and two same-path chains, which wins empirically by `N=2^20`, but
fixed lanes cannot pool holes or blocks and still do not inherit Circuit
ORAM's deterministic stash bound.

Raw CSV files are in `random_eviction_runs/`. The schedule implementation is
in `crates/oram/src/bin/lane_oram_minibucket_security.rs`.

## Paper-Z=5 randomized-eviction sanity check

The exact paper-random schedule was also tested at paper bucket capacity five,
represented here by `B=2,Z=1,Y=5`. Every run used the cyclic workload, OS
randomness, a `2^25`-operation warm-up, and `2^20` measured operations.

| N | Mean post-eviction stash | Observed maximum |
|---:|---:|---:|
| 2^10 | 0.013175 | 7 |
| 2^14 | 0.013118 | 8 |
| 2^18 | 0.014719 | 8 |
| 2^20 | 0.014118 | 9 |

The well-sampled portion of the tail is likewise stable. Across these four
capacities, `Pr[S>0]` ranges from 0.00823 to 0.00918, `Pr[S>2]` from 0.00106
to 0.00140, and `Pr[S>4]` from 0.000122 to 0.000211. Differences farther into
the tail are dominated by counts in the tens or single digits. These results
are consistent with an `N`-independent stash distribution and provide a useful
sanity check that the simulator reproduces the regime covered by the paper's
randomized-eviction theorem. The raw files are in
`random_eviction_z5_runs/`.
