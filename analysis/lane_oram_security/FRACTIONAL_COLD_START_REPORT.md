# Fractional deterministic eviction without discarded warm-up

## Question and method

This experiment checks whether the earlier `2^25`-operation warm-up concealed
different behavior near the beginning of a trajectory.  It also produces a
Circuit-ORAM-Figure-3-style insertion-demand tail with a longer directly
measured range.

The simulator still performs the mandatory randomized initialization of all
`N` blocks.  Initial old labels and replacement labels use OS `getrandom`, and
the initialization itself uses the same oblivious eviction schedule.  There is
then **no discarded access warm-up**: all `2^25` subsequent cyclic accesses are
measured immediately.  Thus each run contains 32 complete key cycles at
`N=2^20`.

The six configurations are `B=2` with `(Z,Y)` equal to `(3,1)`, `(2,2)`, or
`(4,1)`, each at deterministic background rate `r=1/2` or `r=2/3`.  The
schedules distribute service evenly as `0,1,0,1,...` and `0,1,1,0,1,1,...`;
the accessed path is evicted on every operation.  Epoch traces contain a full
tail and mean/minimum/maximum for every `2^16` accesses, or 16 epochs per key
cycle.  All replacement labels remain independent OS-random values.

## Aggregate result versus the warmed runs

The cold aggregate uses 32 times as many measured operations as the earlier
endpoint, so its larger observed maxima are expected and should not be compared
directly with the `2^20`-sample maxima.  The means are directly comparable.

| `Z,Y` | `r` | no-warm-up post mean/max | earlier warmed post mean | mean difference |
|:--|---:|--:|--:|--:|
| `3,1` | `1/2` | 3.911 / 33 | 3.979 | -1.7% |
| `3,1` | `2/3` | 0.615 / 12 | 0.617 | -0.4% |
| `2,2` | `1/2` | 0.685 / 12 | 0.683 | +0.2% |
| `2,2` | `2/3` | 0.467 / 8 | 0.467 | -0.1% |
| `4,1` | `1/2` | 0.584 / 10 | 0.580 | +0.6% |
| `4,1` | `2/3` | 0.427 / 7 | 0.427 | less than +0.01% |

All differences are below 1.8%.  The earlier warmed values are therefore not
an artifact of throwing away an unfavorable or favorable beginning.

## Does the beginning behave differently?

The table reports post-eviction stash occupancy.  “Epoch 0” is the first
`2^16` accesses; cycles 1 and 32 each contain `N=2^20` accesses.

| `Z,Y` | `r` | stash immediately before measurement | epoch-0 mean | cycle-1 mean/max | cycle-32 mean/max |
|:--|---:|---:|--:|--:|--:|
| `3,1` | `1/2` | 0 | 4.595 | 3.956 / 33 | 4.096 / 27 |
| `3,1` | `2/3` | 1 | 0.651 | 0.615 / 9 | 0.621 / 10 |
| `2,2` | `1/2` | 0 | 0.689 | 0.680 / 9 | 0.685 / 9 |
| `2,2` | `2/3` | 0 | 0.469 | 0.466 / 7 | 0.467 / 6 |
| `4,1` | `1/2` | 0 | 0.584 | 0.584 / 9 | 0.584 / 7 |
| `4,1` | `2/3` | 1 | 0.426 | 0.427 / 5 | 0.428 / 5 |

`Z=3,Y=1` has a mild first-epoch bump: its first epoch lies at the 85.5th
percentile of epoch means for `r=1/2` and the 91.0th percentile for `r=2/3`.
Those are high but not extreme observations, and the evolution plot shows no
monotone relaxation.  Once averaged over one complete key cycle, cycle 1 and
cycle 32 differ by only 3.5% for `r=1/2` and 1.1% for `r=2/3`.  Every other
configuration differs by less than 0.7%.

The best interpretation is that randomized initialization plus the `N`
scheduled initialization updates already places these systems close to their
long-run regime.  There may be a small sub-cycle initialization effect for the
narrow `Z=3,Y=1` case, but no warm-up-scale transient is visible.  This finding
is specific to these configurations and the randomized initialization; it is
not a general theorem that ORAM warm-up is unnecessary.

## Direct Figure-3 tail measurements

Failure-relevant insertion demand is one block larger than the post-eviction
stash.  Each entry below is the smallest capacity `S` whose observed
per-operation exceedance probability is at most the stated value.

| `Z,Y` | `r` | `S` at `2^-8` | `S` at `2^-16` | `S` at `2^-20` | `S` at `2^-24` | observed demand max |
|:--|---:|---:|---:|---:|---:|---:|
| `3,1` | `1/2` | 16 | 25 | 29 | 33 | 34 |
| `2,2` | `1/2` | 5 | 9 | 11 | 12 | 13 |
| `4,1` | `1/2` | 4 | 7 | 9 | 11 | 11 |
| `3,1` | `2/3` | 5 | 9 | 11 | 12 | 13 |
| `2,2` | `2/3` | 3 | 6 | 8 | 9 | 9 |
| `4,1` | `2/3` | 3 | 5 | 6 | 7 | 8 |

This makes the tradeoff especially clear:

- At `r=1/2`, `Z=2,Y=2` is dramatically better than `Z=3,Y=1`, but
  `Z=4,Y=1` is slightly better still at the same four tree slots.
- Raising the rate to `r=2/3` reduces the directly observed `2^-24` capacity
  from 12 to 9 for `Z=2,Y=2`, and from 11 to 7 for `Z=4,Y=1`.
- `Z=3,Y=1,r=1/2` remains the outlier: its `2^-24` observed capacity is 33.
  Its small absolute mean at `N=2^20` does not compensate for its much heavier
  tail or its previously observed growth with `N`.

These are direct empirical quantiles only through about 25 bits.  They do not
establish a `2^-96` tail or an `N`-independent stash distribution.

## Artifacts

- `fractional_cold_start_figure3.svg`: Figure-3-style insertion-demand tails.
- `fractional_cold_start_evolution.svg`: `2^16`-access epoch means over all 32
  cycles, split by rate.
- `fractional_cold_start_summary.csv`: aggregate, early/late, and observed-tail
  summary.
- `cold_start_runs/cold_*_tail.csv`: aggregate tail counts.
- `cold_start_runs/cold_*_epochs.csv`: every epoch tail and summary.
- `scripts/analyze_fractional_cold_start.py`: dependency-free reproduction of
  the summary and plots.

The epoch instrumentation is implemented in
`crates/oram/src/bin/lane_oram_minibucket_security.rs`.  All 16 simulator tests
pass, including fractional leaf-order coverage and block conservation.
