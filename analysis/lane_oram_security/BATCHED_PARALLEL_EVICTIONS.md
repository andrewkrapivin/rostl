# Selected-ORAM and paired-access deterministic evictions

## Correct construction

Let X be the number of constituent ORAMs and M the number of leaves in each
binary ORAM. There are K logical blocks and one shared client stash. Each
block's private position entry contains a leaf and a constituent-ORAM ID.
The load reported here is

    rho = K / (X M).

Because a one-slot binary tree has approximately 2M physical slots, the raw
tree-slot utilization is approximately rho/2.

On every logical lookup:

1. read only the constituent ORAM recorded for the key;
2. remove the key, sample a fresh leaf and constituent ORAM with OS randomness,
   and insert it into the one shared stash;
3. evict the accessed path in a fixed group of A lanes containing the old
   constituent ORAM;
4. at rate E/X, perform a synchronized deterministic round that evicts the
   same bit-reversed path in all X constituent ORAMs.

A=1 is the exact selected-ORAM construction. A=2 uses fixed pairs
(0,1), (2,3), and so on. Only the assigned ORAM is searched; the partner is
touched only for eviction. The average scalar eviction-chain work is A+E.
A synchronized background round occurs E/X times per operation.

The stash is physically shared and has one aggregate overflow limit. The
simulator uses per-ORAM indexes only to accelerate candidate selection for a
block's recorded assignment; these are not separate capped stashes.

## Correction to the earlier experiment

The earlier implementation read one wide path and evicted all X lanes during
every normal lookup. It therefore performed X+E chains per operation. For
X=8,E=2 that was 10 chains, whereas the intended selected-ORAM construction
performs 1+E=3. Those older stash conclusions are invalid for this
construction.

At the original 12.5% load, corrected X=8,A=1,E=2 scales as follows.
Entries are post-eviction stash mean / observed maximum.

| Leaves per ORAM M | 2^10 | 2^13 | 2^15 | 2^17 |
|---:|---:|---:|---:|---:|
| Corrected X=8, A=1, E=2 | 8.353 / 24 | 15.620 / 39 | 31.297 / 78 | 84.652 / 136 |

Thus E=2 is not size-independent even at 12.5% load.

## Methodology

- B=2 and one slot per constituent-ORAM bucket (Y=1).
- X in {1,2,4,8} for A=1 and X in {2,4,8,16} for A=2.
- Loads 10%, 20%, 25%, 40%, and 50%. Non-dyadic block counts are rounded
  to the nearest integer; the load error is below one block.
- The workload cycles through every key.
- Initial leaves, every replacement leaf, and every replacement ORAM ID use
  OS getrandom. Bounded ORAM selection uses rejection sampling.
- M=2^10 pilots used 2^18 warm-up and 2^20 measured operations.
- M=2^13, 2^15, and 2^17 checks used 2^20, 2^21, and 2^22 warm-up,
  respectively, followed by 2^20 measured operations in eight epochs.
- E was searched in quarter steps near the transition.

The metadata implementation internally loads a wide temporary buffer for
convenience, but a selected lookup searches and modifies only its configured
group. Regression tests verify block conservation, that A=1 leaves every
unaccessed ORAM unchanged, and that A=2 evicts exactly the accessed pair.

## A=1: one normal lane eviction

E<=2.75 grows clearly with M. E=3 is stationary within each run but shows a
small depth trend. E=3.25 is the smallest quarter-step setting that was flat
from M=2^10 through M=2^17 at the worst 50% load.

At M=2^17 and E=3.25:

| X | 10% | 20% | 25% | 40% | 50% |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.025 / 7 | 0.026 / 6 | 0.027 / 5 | 0.032 / 7 | 0.030 / 6 |
| 2 | 0.286 / 8 | 0.288 / 7 | 0.298 / 7 | 0.292 / 8 | 0.291 / 7 |
| 4 | 1.120 / 8 | 1.143 / 10 | 1.127 / 9 | 1.155 / 11 | 1.177 / 10 |
| 8 | 2.905 / 14 | 2.899 / 15 | 2.921 / 12 | 2.919 / 13 | 2.913 / 12 |

Occupancy has little effect above the stable-rate transition. Increasing X
mainly combines more constituent stash processes into the shared aggregate.

## A=2: paired normal lane evictions

Pairing reduces the required synchronized background rate. E=2 grows
substantially, E=2.25 grows roughly with tree depth, E=2.5 grows slowly, and
E=2.75 is the smallest tested quarter-step setting that is empirically flat.

At M=2^17 and E=2.75:

| X | 10% | 20% | 25% | 40% | 50% |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.310 / 7 | 0.331 / 9 | 0.336 / 8 | 0.374 / 11 | 0.347 / 10 |
| 4 | 1.182 / 10 | 1.191 / 10 | 1.206 / 9 | 1.244 / 10 | 1.263 / 12 |
| 8 | 3.043 / 16 | 3.078 / 13 | 3.160 / 15 | 3.098 / 15 | 3.196 / 14 |
| 16 | 6.819 / 21 | 6.846 / 22 | 6.819 / 20 | 6.893 / 23 | 7.123 / 22 |

### Lower-E tradeoff at 50% load

Each cell gives M=2^10 mean/max -> M=2^17 mean/max.

| X | Paired E=2 | Paired E=2.5 | Paired E=2.75 |
|---:|---:|---:|---:|
| 2 | 0.836/11 -> 4.768/26 | 0.403/9 -> 0.565/10 | 0.294/8 -> 0.347/10 |
| 4 | 2.416/16 -> 10.260/40 | 1.403/10 -> 1.771/14 | 1.135/9 -> 1.263/12 |
| 8 | 5.678/22 -> 20.508/58 | 3.507/14 -> 4.255/20 | 2.936/13 -> 3.196/14 |
| 16 | 12.318/33 -> 44.839/86 | 7.693/21 -> 9.680/32 | 6.580/19 -> 7.123/22 |

Paired E=2 may be reasonable for a fixed maximum M if the stash is sized for
the observed growth. It uses A+E=4 scalar chains per operation. Paired E=2.5
uses 4.5 chains and has much slower growth. Paired E=2.75 uses 4.75 chains and
is the conservative flat choice.

## Pooled Circuit-ORAM pairs

In this variant each physical pair is one pooled Y=2 Circuit ORAM rather
than two independent Y=1 lanes. Total width X therefore contains X/2
constituent Circuit ORAMs. A lookup searches its assigned pair and performs
one Circuit eviction chain. A synchronized round performs one Circuit chain
in every pair.

Here E counts two-wide Circuit chains per operation. The synchronized-round
rate is

    E / (X/2) = 2E/X.

E=1 grows strongly, E=1.25 grows slowly, and E=1.375 is the smallest tested
eighth-step setting that is empirically flat through M=2^17.

At M=2^17 and E=1.375:

| Total width X | 10% | 20% | 25% | 40% | 50% |
|---:|---:|---:|---:|---:|---:|
| 8 | 3.713 / 20 | 3.726 / 19 | 3.700 / 18 | 3.742 / 17 | 3.776 / 18 |
| 16 | 8.481 / 26 | 8.367 / 25 | 8.586 / 27 | 8.441 / 25 | 8.429 / 27 |

At 50% load, each cell below gives M=2^10 mean/max -> M=2^17 mean/max.

| X | E=1 | E=1.25 | E=1.375 | E=1.5 |
|---:|---:|---:|---:|---:|
| 8 | 8.635/37 -> 85.315/155 | 4.484/17 -> 5.483/25 | 3.615/19 -> 3.776/18 | 3.069/16 -> 3.100/15 |
| 16 | 18.642/48 -> 164.810/272 | 10.044/32 -> 11.789/40 | 8.186/24 -> 8.429/27 | 7.055/20 -> 7.120/23 |

Circuit-pair E=1.375 and independent-pair E=2.75 have the same wide
background-path rate, 2.75/X. They also scan the same number of bucket slots
per level. The smaller numerical E for the Circuit-pair version is a change of
units, not a factor-two compute saving. One pooled chain must inspect and
arbitrate between both slots. The two independent one-slot chains can execute
in parallel or SIMD and can propagate two blocks, whereas one pooled chain
propagates at most one block. The actual memory work is comparable and the
pooled state transition may be more sequential. A real implementation
benchmark is needed to decide which is faster.

## Extended occupancy sweep

Here rho still means K/(XM), where X is total bucket width. Thus rho=100%
means one key per leaf-level slot and approximately 50% utilization of all
tree slots. Values above 100% are possible because internal buckets also
provide storage.

The sweep used the same random initialization, cyclic workload, warm-ups, and
2^20 measured operations as above. Each sequence is M=2^10, 2^13, 2^15,
2^17 and reports post-eviction mean/maximum. The listed conservative edges
are empirical near-stationarity through 2^17, not asymptotic proofs.

### Two independent Y=1 lanes per normal access

| E | rho | X=8 scaling | X=16 scaling | Interpretation |
|---:|---:|---|---|---|
| 2.75 | 75% | 3.571/16 -> 3.836/20 -> 4.269/19 -> 4.453/19 | 7.924/26 -> 8.541/31 -> 9.119/31 -> 9.478/33 | Conservative measured edge |
| 2.75 | 80% | 4.016/18 -> 4.719/20 -> 5.427/24 -> 6.957/30 | 8.709/29 -> 9.858/35 -> 11.214/36 -> 14.558/43 | Small but depth-dependent |
| 2.75 | 85% | 4.780/20 -> 6.942/29 -> 8.996/32 -> 17.261/47 | 10.417/33 -> 13.344/36 -> 18.347/53 -> 31.014/76 | Clear growth |
| 3.0 | 80% | 3.348/17 -> 3.563/19 -> 3.892/20 -> 4.377/21 | 7.335/24 -> 7.726/27 -> 8.216/29 -> 8.340/30 | Conservative measured edge |
| 3.0 | 85% | 3.866/24 -> 4.771/21 -> 5.193/23 -> 8.904/37 | 8.869/30 -> 10.429/34 -> 13.197/41 -> 14.180/39 | Aggressive finite-M edge |
| 3.0 | 90% | 5.206/25 -> 7.409/28 -> 12.457/40 -> 18.379/48 | 11.027/33 -> 16.315/47 -> 22.427/55 -> 46.648/102 | Clear growth |

At the largest tested M, 80% with E=2.75 is still quite usable with a
moderately larger stash, but it should not be described as size-independent.
Increasing E to 3.0 moves the conservative edge from about 75% to about 80%.

### Pooled Y=2 Circuit pairs

| E | rho | X=8 scaling | X=16 scaling | Interpretation |
|---:|---:|---|---|---|
| 1.375 | 120% | 4.611/23 -> 4.884/25 -> 5.653/30 -> 5.391/29 | 9.974/35 -> 10.980/40 -> 11.332/37 -> 13.391/41 | Conservative measured edge |
| 1.375 | 125% | 5.509/27 -> 7.810/35 -> 9.359/36 -> 15.489/52 | 11.848/42 -> 14.790/48 -> 18.840/51 -> 34.851/82 | Clear growth |
| 1.5 | 125% | 4.463/23 -> 4.683/29 -> 4.561/24 -> 5.991/33 | 10.028/40 -> 9.991/40 -> 10.756/38 -> 11.902/40 | Conservative measured edge |
| 1.5 | 130% | 5.885/31 -> 7.693/36 -> 11.677/53 -> 20.489/72 | 11.965/43 -> 17.199/60 -> 24.847/75 -> 36.708/86 | Clear growth |

At M=2^17, reducing rho to 110% lowers both the mean and the directly observed
maximum:

| E | X=8 mean/max | X=16 mean/max |
|---:|---:|---:|
| 1.375 | 4.050 / 21 | 9.311 / 37 |
| 1.5 | 3.125 / 14 | 7.215 / 24 |

Pooling therefore raises the measured density edge substantially: about 120%
versus 75% for the lower-rate pair, and about 125% versus 80% for the higher
rate pair. At these high loads, capacity sharing and reduced fragmentation
dominate the pooled chain's lower propagation rate.

The original console summary divided occupancy by ZM and consequently printed
twice the intended value for Y=2. The block counts, tree construction, result
CSVs, and rho labels above used ZYM and are correct. The simulator's display
has now been fixed to print `leaf_slot_occupancy=K/(ZYM)`.

## Practical recommendation

| Construction | Smallest empirically flat E at rho=50% | Wide-round rate | Nominal state-transition chains/op | Bucket-slot scans/op |
|---|---:|---:|---:|---:|
| One selected Y=1 lane | 3.25 | 3.25/X | 4.25 | 4.25 |
| Two independent Y=1 lanes | 2.75 | 2.75/X | 4.75 | 4.75 |
| One pooled Y=2 Circuit pair | 1.375 | 2.75/X | 2.375 | 4.75 |

The one-lane construction uses the least slot bandwidth at its flat 50% load
threshold. The two-wide independent and pooled constructions have the same
bucket-slot bandwidth. Their nominal chain counts are not directly comparable:
a pooled chain processes two slots and arbitrates jointly, while two
independent chains may run in parallel and can propagate twice as many blocks.
There is no current runtime evidence for a factor-two compute advantage.
Pooling's demonstrated advantage is its much higher sustainable occupancy.

For a finite tree, Circuit-pair E=1.25 and independent-pair E=2.5 are the
strongest measured lower-bandwidth compromises; both use wide-round rate
2.5/X and grow slowly. Circuit-pair E=1 and independent-pair E=2 grow much
more substantially. A final choice must include the cost of scanning the
shared constant-time stash.

## Security limitation

These runs establish empirical stationarity and directly observed maxima, not
a 2^-96 overflow bound. One million measured operations per final
configuration cannot resolve cryptographic tail probabilities. The maxima
must not be treated as security bounds. A security-sized stash still requires
a theoretical tail argument, importance sampling, or a justified extrapolation
validated by substantially longer independent runs.

The follow-up pooled Y=2, E=1.5 experiment fixes total bottom capacity at
2^20 and measures 2^25 operations per configuration. It shows that the earlier
rho=125% stationarity result was too optimistic for cryptographic tails: the
insertion-tail 2^-96 point estimates are 181 blocks for X=8 and 259 for X=16.
At rho=100--110%, conservative settings are still 112--128 blocks. See
`POOLED_E1_5_STASH_SECURITY.md` and `pooled_e1_5_figure3.png`.

Raw results are under analysis/lane_oram_security/selected_partitioned_runs/.
The extended-density results are under
analysis/lane_oram_security/occupancy_extension_runs/.
The simulator is crates/oram/src/bin/lane_oram_minibucket_security.rs.
