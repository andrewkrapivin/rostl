# Lane ORAM experiment map and current conclusions

This is the index and comparison sheet for the Lane ORAM experiments run on
this machine.  It separates placement/security simulations from real AVX-512
payload benchmarks, because the former do not measure memory traffic and the
latter do not by themselves establish a safe stash size.

## Bottom line

- Accessed-path-only fixed lanes with `Y=1` are not viable at fixed `Z`: their
  downward service rate decreases with tree height, eventually leaving a stash
  linear in `N`.
- Pooling several slots inside a lane (`Y>1`), repeating eviction chains on an
  already loaded path (`E>1`), or adding regularly scheduled background paths
  can each move the system away from that congestion boundary.
- The best measured same-path production point is currently
  `B=2,Z=3,Y=3,E=1,S=64`: 1.914 us/update at `N=2^20`.  Its `S=64` choice is
  an empirical `2^-96` tail extrapolation, not a proof.
- Deterministic background paths are exceptionally effective.  For example,
  `B=2,Z=2,Y=2` changes from mean/max `2084.770/2384` with only the accessed
  path to `0.251/3` after adding one deterministic path per operation.  The
  real payload implementation does not yet implement these background
  schedules, so their production latency has not been benchmarked.
- Small-tree results are not enough.  Several quaternary `Y=1` settings look
  excellent at `N=2^10` and fail badly at `N=2^20`.

## Notation and measurement rules

| Symbol | Meaning |
|:--|:--|
| `B` | Tree branching factor. `N` must be an exact power of `B`. |
| `Z` | Number of fixed lanes. |
| `Y` | Slots in each lane minibucket at one tree level. |
| `Z*Y` | Physical tree slots per bucket. |
| `E` | Circuit-style eviction chains per lane on each loaded path. |
| `r` | Deterministic background paths per logical access. Access-plus schedules transfer `1+r` full paths on average. |
| `S` | Stash capacity. Failure occurs when transient insertion demand exceeds `S`. |

Unless stated otherwise, security runs use a cyclic `0..N-1` workload, a
shared stash, fixed lanes, and OS `getrandom` for every replacement label and
for randomized initial old and new labels.  Tables report the post-eviction
stash; insertion demand is exactly one block larger.  A reported zero count in
`2^20` samples resolves only about 20 tail bits and is not a `2^-96` bound.
The sizing target used in the benchmark tables is per-operation failure at
most `2^-96`; a union bound then gives failure at most `2^-64` over `2^32`
operations.

The fractional deterministic scheduler emits a gap-free base-`B`
digit-reversal sequence at rational rate `p/q`.  The per-operation counts are:

| `r` | Repeating background-path counts | Mean full paths/access |
|---:|:--|---:|
| `1/2` | `0,1` | 1.5 |
| `2/3` | `0,1,1` | 1.667 |
| `3/2` | `1,2` | 2.5 |
| `1` | `1` | 2 |
| `2` | `2` | 3 |
| `B` | one path in each root child | `1+B` |

## Compact decision table

This table contains the configurations worth retaining as baselines or design
points.  “Flat” means empirically approximately independent of `N` over the
tested range, not theoretically proven.

| Construction | Slots | Path policy | `N=2^20` post mean/max | Scaling verdict | Real `N=2^20` time |
|:--|---:|:--|--:|:--|--:|
| Circuit, paper deterministic `B2,Y2` | 2 | two background, no accessed eviction | 0.066 / 7 | flat | not isolated here |
| Circuit, paper random `B2,Y5` | 5 | two randomized half-tree paths | 0.014 / 9 | flat; paper bound applies | not isolated here |
| Lane `B2,Z3,Y3,E1` | 9 | accessed path only | 1.154 / 13 | clearest same-path plateau | 1.914 us (`S=64`) |
| Lane `B2,Z4,Y2,E1` | 8 | accessed path only | 1.087 / 15 | promising | 2.062 us (`S=72`) |
| Lane `B2,Z2,Y3,E2` | 6 | two chains/lane, same path | 1.086 / 17 | promising | 2.151 us (`S=80`) |
| Lane `B2,Z2,Y2,E1` | 4 | accessed + `r=1` deterministic | 0.251 / 3 | flat | not implemented |
| Lane `B2,Z3,Y1,E1` | 3 | accessed + `r=1` deterministic | 0.260 / 5 | approximately flat | not implemented |
| Lane `B4,Z4,Y2,E1` | 8 | accessed + `r=1` deterministic | 0.960 / 5 | flat | not implemented |
| Lane `B4,Z3,Y2,E1` | 6 | accessed + `r=2` deterministic | 0.377 / 3 | flat | not implemented |
| Lane `B4,Z2,Y2,E1` | 4 | accessed + `r=4` deterministic | 0.000216 / 2 | nearly empty, mild trend | not implemented |
| Lane `B4,Z1,Y10,E3` | 10 | accessed path only | 3.872 / 32 | apparent plateau | 2.693 us (`S=176`) |

The deterministic rows have striking observed maxima, but there are too few
tail samples to infer a cryptographic stash capacity from those maxima.  The
best proved reference remains Circuit ORAM in the parameter regimes covered by
its theorems.

## Fractional deterministic background-rate sweep

The new pilot uses `N=2^10`, `2^20` warm-up operations, and `2^22` measured
operations.  Every setting below was screened.  Values are post-stash
mean/maximum.

### Binary pilot

| `Z,Y` | slots | `r=1/2` | `r=2/3` | `r=3/2` |
|:--|---:|--:|--:|--:|
| `3,1` | 3 | 0.643 / 9 | 0.447 / 7 | 0.125 / 2 |
| `4,1` | 4 | 0.541 / 6 | 0.419 / 4 | 0.125 / 1 |
| `5,1` | 5 | 0.532 / 5 | 0.417 / 4 | 0.125 / 1 |
| `2,2` | 4 | 0.606 / 9 | 0.446 / 8 | 0.125 / 2 |

### Quaternary pilot

| `Z,Y` | slots | `r=1/2` | `r=2/3` | `r=3/2` |
|:--|---:|--:|--:|--:|
| `2,2` | 4 | 12.141 / 40 | 5.632 / 28 | 0.716 / 9 |
| `3,2` | 6 | 2.209 / 15 | 1.545 / 12 | 0.613 / 4 |
| `4,2` | 8 | 1.712 / 10 | 1.374 / 8 | 0.610 / 3 |

The selected `N=2^20` endpoints measure `2^20` operations.  Binary runs use a
`2^25` warm-up except for the capped `Z=5` replacements; quaternary runs and
those replacements use `2^24`.  They are intentionally reported separately
below so a favorable pilot cannot hide tree-height dependence.

<!-- FRACTIONAL_FULL_RESULTS_START -->
### Binary full-size comparison

| `Z,Y` | `r` | `N=2^10` mean/max | `N=2^20` mean/max | Interpretation |
|:--|---:|--:|--:|:--|
| `3,1` | `1/2` | 0.643 / 9 | 3.979 / 26 | clear growth; below threshold |
| `3,1` | `2/3` | 0.447 / 7 | 0.617 / 10 | visible growth; not convincingly flat |
| `3,1` | `3/2` | 0.125 / 2 | 0.125 / 2 | flat at these endpoints |
| `4,1` | `1/2` | 0.541 / 6 | 0.580 / 8 | mild growth |
| `4,1` | `2/3` | 0.419 / 4 | 0.427 / 6 | nearly flat |
| `5,1` | `1/2` | 0.532 / 5 | 0.536 / 5* | flat at these endpoints |
| `5,1` | `2/3` | 0.417 / 4 | 0.417 / 3* | flat at these endpoints |
| `2,2` | `1/2` | 0.606 / 9 | 0.683 / 10 | mild growth |
| `2,2` | `2/3` | 0.446 / 8 | 0.467 / 7 | nearly flat |
| `2,2` | `3/2` | 0.125 / 2 | 0.125 / 1 | flat at these endpoints |

`*` The two `Z=5` full-warm-up runs crossed the 20-minute limit and were
stopped.  Their replacements use `2^24` rather than `2^25` warm-up operations;
all other binary endpoints use `2^25`.  Every endpoint measures `2^20`
operations.

### Quaternary full-size comparison

| `Z,Y` | `r` | `N=2^10` mean/max | `N=2^20` mean/max | Interpretation |
|:--|---:|--:|--:|:--|
| `2,2` | `1/2` | 12.141 / 40 | not scaled | already poor in pilot |
| `2,2` | `2/3` | 5.632 / 28 | not scaled | already poor in pilot |
| `2,2` | `3/2` | 0.716 / 9 | 1.308 / 12 | clear slow growth |
| `3,2` | `1/2` | 2.209 / 15 | 176.932 / 252 | catastrophic growth |
| `3,2` | `2/3` | 1.545 / 12 | 12.793 / 43 | clear growth |
| `3,2` | `3/2` | 0.613 / 4 | 0.619 / 5 | flat at these endpoints |
| `4,2` | `1/2` | 1.712 / 10 | 2.363 / 16 | visible growth |
| `4,2` | `2/3` | 1.374 / 8 | 1.483 / 10 | mild growth; promising, not proved flat |
| `4,2` | `3/2` | 0.610 / 3 | 0.610 / 3 | flat at these endpoints |

The quaternary endpoints use `2^24` warm-up operations and `2^20` measured
operations to remain below the 20-minute limit.  Together with the previous
integer-rate runs, they locate the practical boundary: `B4,Z3,Y2` needs more
than `2/3` of a background path and is clean at `3/2`; `B4,Z4,Y2` is much
closer to critical at `2/3` and was already flat at the previously measured
integer rate `r=1`.  `B4,Z2,Y2` still grows at `r=3/2`, consistent with its
slow trend even at the earlier `r=2` point.

For binary trees, `Z=4,Y=1,r=2/3` and `Z=2,Y=2,r=2/3` are attractive
four-slot candidates at only 1.667 full paths per access, but the former has a
slightly cleaner observed tail.  Spending five one-slot lanes permits reducing
the rate to `r=1/2` with no visible endpoint drift.  These comparisons expose a
continuous three-way tradeoff among bucket storage, lane service parallelism,
and path bandwidth.
<!-- FRACTIONAL_FULL_RESULTS_END -->

## No-discarded-warm-up check

Six follow-up runs measure `2^25` accesses immediately after mandatory
randomized initialization, with no subsequent discarded warm-up.  Each trace
contains 512 epochs of `2^16` accesses.  The aggregate cold means differ from
the earlier `2^25`-warm-up endpoint means by less than 1.8% in every case, and
the first and last complete key cycles differ by at most 3.5%.

| `Z,Y` | `r` | cold post mean/max | first-cycle mean | last-cycle mean | directly observed insertion `S` at `2^-24` |
|:--|---:|--:|--:|--:|---:|
| `3,1` | `1/2` | 3.911 / 33 | 3.956 | 4.096 | 33 |
| `2,2` | `1/2` | 0.685 / 12 | 0.680 | 0.685 | 12 |
| `4,1` | `1/2` | 0.584 / 10 | 0.584 | 0.584 | 11 |
| `3,1` | `2/3` | 0.615 / 12 | 0.615 | 0.621 | 12 |
| `2,2` | `2/3` | 0.467 / 8 | 0.466 | 0.467 | 9 |
| `4,1` | `2/3` | 0.427 / 7 | 0.427 | 0.428 | 7 |

The first `2^16`-access epoch is mildly high for both `Z=3,Y=1` runs, but it
is not an extreme epoch and there is no monotone relaxation.  The randomized
initialization and its `N` scheduled updates already leave these six systems
close to their long-run regime.  Therefore the earlier conclusions are not a
warm-up artifact.  See the [full cold-start report](FRACTIONAL_COLD_START_REPORT.md),
[Figure-3-style tail](fractional_cold_start_figure3.svg), and
[epoch evolution plot](fractional_cold_start_evolution.svg).

## What every experiment family taught us

| Experiment family | Representative result | Verdict |
|:--|:--|:--|
| Legacy accessed-only `Y=1` (`B2/Z3`, `B4/Z5,6`, `B8/Z9,10`) | At `N=4096`, mean stash 1887--3146; larger runs approach a linear fraction of `N`. | Rejected for security; its sub-microsecond benchmarks are not security-valid. |
| Workload checks | Permutation cycles all agree; IID keys are better and a hot set/same key is dramatically better. | Cycling is adverse and useful, though not proved globally worst-case. |
| Random per-lane queues | `B2,Z2,Y1,N=2^16`: shared 3611 versus random queues 5252 mean. | Rejected; partitioning loses the ability to borrow storage/service. |
| Same-path pooling | At equal width and chain count, fixed lanes are slightly better; one pooled chain is severely under-served. | Pooling alone is not a repair. |
| Larger `Y`, fixed lanes | `B2,Z3,Y3` and `B4,Z4,Y4` are nearly flat through `N=2^20`. | Viable empirical same-path family, with wider memory traffic. |
| Extra same-path chains `E` | `B2,Z2,Y3,E2` lowers fitted `S96` from about 309 to 75 without another path transfer. | Strong memory-capacity tradeoff; costs more selection scans. |
| One wide lane | `B2,Y5,E2` grows; `B4,Y10,E3` plateaus; `B4,Y8,E3` congests; `B8,Y16` needs `E=7`. | Sharp service thresholds; none beats `B2,Z3,Y3` for speed. |
| Circuit random eviction | Width 3 grows with `N`; paper width 5 stays near mean 0.014 and matches the proven regime. | Simulator sanity check passed; random service needs slack. |
| Circuit deterministic eviction | Paper width 2 stays near mean 0.06 through `N=2^20`. Adding accessed-path eviction lowers it to about 0.003. | Very effective regular prefix service. |
| Lane + integer deterministic sweeps | Binary `Y=1/2` and quaternary `Y=2` can become flat with modest `r`; quaternary `Y=1` still fails even at `r=2`. | Most promising route to narrow buckets, but adds full path traffic. |
| Lane + fractional deterministic sweeps | See the new rate tables above. | Locates the bandwidth/security threshold more finely than integer sweeps. |
| Per-lane path-bit “mixing” | A lane-specific path interpretation also requires lane-specific block eligibility/queues, defeating the intended shared-source construction. | Abandoned before simulation as a different, less attractive design. |

## Security-sized same-path production benchmarks

These are real AVX-512 updates of a 56-byte Lane payload.  `S96` is a
log-linear engineering extrapolation from only about 4--19 measured tail bits.

| Configuration | slots | estimated `S96` | benchmark `S` | `N=2^20` |
|:--|---:|---:|---:|---:|
| `B2,Z3,Y3,E1` | 9 | 63 | 64 | 1.914 us |
| `B2,Z3,Y2,E1` | 6 | 126 | 128 | 1.968 us |
| `B2,Z4,Y2,E1` | 8 | 71 | 72 | 2.062 us |
| `B2,Z2,Y5,E1` | 10 | not separately reported | 104 | 2.125 us |
| `B2,Z2,Y3,E2` | 6 | 75 | 80 | 2.151 us |
| `B4,Z4,Y4,E1` | 16 | 95 | 96 | 2.309 us |
| `B4,Z5,Y3,E1` | 15 | 106 | 112 | 2.535 us |
| `B4,Z1,Y10,E3` | 10 | about 166 | 176 | 2.693 us |
| `B2,Z2,Y3,E1` | 6 | 309 | 320 | 2.702 us |
| Circuit ORAM, 56-byte payload | 2 | not security-matched | 20 | 1.370 us |

At `B2,Z3,Y3,E1,S=64,N=2^20`, stash-capacity-dependent scans account for
about 23.8% of runtime by regression; path I/O, path checks, eviction work, and
fixed bookkeeping account for the remaining 76.2%.  The same-path Lane point
is 1.40x the equal-payload Circuit timing, but the stash failure guarantees
have not been matched.

## Machine

| Item | Value |
|:--|:--|
| CPU | AMD Ryzen AI 9 365, 8 exposed cores under KVM |
| ISA | AVX-512F, DQ, BW, VL, IFMA, VNNI, VBMI/VBMI2, and related subsets exposed |
| Cache | 512 KiB L1d total, 4 MiB L2 total, 128 MiB reported L3 |
| Memory | 8.7 GiB RAM, 4.0 GiB swap |

The original `B=8,logN=24` payload benchmark is infeasible in this memory: its
tree allocation reached about 8.6 GiB resident plus 1.7 GiB swap before being
stopped.  Reducing Criterion samples cannot reduce that allocation.

## Evidence and detailed reports

- [Original implementation, workload, scaling, and legacy benchmarks](REPORT.md)
- [Why fixed `Y=1` lanes depend on `N`](QUEUEING_ANALYSIS.md)
- [Minibucket definition and first `Y=2` results](MINIBUCKET_LANE_REPORT.md)
- [Same-path tail fits and AVX-512 performance](MINIBUCKET_SECURITY_PERFORMANCE_REPORT.md)
- [Same-path `E`, pooling, and one-wide-lane strategies](LANE_STRATEGY_EXPERIMENTS.md)
- [Random queues and early quaternary results](RANDOM_QUEUE_AND_B4_RESULTS.md)
- [Circuit random eviction and paper-width-5 sanity check](RANDOM_EVICTION_COMPARISON.md)
- [Lane integer deterministic sweeps](LANE_DETERMINISTIC_SWEEPS.md)
- [Circuit paper-deterministic width-2 comparison](DETERMINISTIC_Z2_COMPARISON.md)
- [Binary `Y=1` and quaternary deterministic sweeps](DETERMINISTIC_NARROW_LANES.md)
- [No-discarded-warm-up fractional sweeps](FRACTIONAL_COLD_START_REPORT.md)

Raw fractional-rate CSVs are in `lane_fractional_runs/`.  The schedule and its
coverage/conservation tests are in
`crates/oram/src/bin/lane_oram_minibucket_security.rs`; the production
minibucket implementation is in `crates/oram/src/lane_oram_fixed.rs`.

## Limits on conclusions

No finite run here establishes a `2^-96` failure probability.  Tail fits are
useful for engineering comparisons but remain extrapolations from far shorter
tails, with serially correlated samples.  A production claim still needs a
proof, a defensible dominating process, or rare-event/importance sampling.
Background schedules also need a constant-time real implementation and payload
benchmark before their bandwidth advantage can be compared fairly with
same-path configurations or Circuit ORAM.
