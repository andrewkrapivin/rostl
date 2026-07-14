# Deterministic background sweeps for Lane ORAM

## Construction

The experiment uses `B=2,Z=2,Y=3` with a shared stash and one Circuit-style
chain per fixed lane. Three public eviction schedules are compared:

1. `same`: evict both lanes only on the accessed old-label path.
2. `access-plus-one-deterministic`: additionally evict both lanes on
   `bitrev(t mod N)`.
3. `access-plus-paper-deterministic`: additionally evict both lanes on the
   two Algorithm-6 paths `bitrev(2t mod N)` and `bitrev((2t+1) mod N)`.

The one-path sequence visits every leaf once per `N` operations. The paired
sequence visits every leaf once per `N/2` operations and places its two paths
in opposite root halves. Unit tests verify these properties and block
conservation.

## Stash scaling

Each one-sweep scaling run used OS randomness, the cyclic workload, a `2^25`
warm-up, and `2^20` measured operations.

| N | Mean post-stash | Observed maximum |
|---:|---:|---:|
| 2^10 | 0.250 | 2 |
| 2^14 | 0.251 | 3 |
| 2^18 | 0.251 | 2 |
| 2^20 | 0.250 | 2 |

The one-sweep distribution is empirically independent of `N`. At `N=2^20`,
the matched full-warm-up comparison is:

| Schedule | Mean post-stash | Observed maximum |
|---|---:|---:|
| Accessed path only | 5.256 | 49 |
| Access + one deterministic sweep | 0.249822 | 2 |
| Access + two deterministic sweeps | 0 | 0 |

For one sweep at `N=2^20`, `Pr[S>0]=0.249814`, `Pr[S>1]=7.63e-6`,
and no `S>2` event occurred. For two sweeps, every measured operation ended
with an empty stash; insertion demand was always exactly one. These are
empirical observations over `2^20` samples, not cryptographic bounds or
proofs that the stated maxima are invariants.

The deterministic sweeps repair the service irregularity of accessed-path-only
Lane ORAM. They do not remove fixed-lane isolation, and the construction does
not automatically inherit Circuit ORAM's stash proof.

## Reducing the minibucket to `Y=2`

The same experiment was repeated with `B=2,Z=2,Y=2`, so each physical bucket
has only four slots.  The sweep runs again used OS randomness, cyclic accesses,
a `2^25`-operation warm-up, and `2^20` measured operations at every `N`.

| N | One sweep: mean | One sweep: max | Two sweeps: mean | Two sweeps: max |
|---:|---:|---:|---:|---:|
| 2^10 | 0.250645 | 4 | 0 | 0 |
| 2^14 | 0.250876 | 3 | 0 | 0 |
| 2^18 | 0.251799 | 3 | 0 | 0 |
| 2^20 | 0.251351 | 3 | 0 | 0 |

At `N=2^20`, one sweep gave `Pr[S>0]=0.251112`,
`Pr[S>1]=2.346e-4`, `Pr[S>2]=4.768e-6`, and no `S>3` event.  The
two-sweep post-eviction stash was empty in all `2^20` measured operations at
every tested `N`; insertion demand was always exactly one.

The one-sweep mean is almost unchanged from `Y=3`, but its rare tail is worse:
at `N=2^20`, `Pr[S>1]` is about 31 times larger (`2.346e-4` versus
`7.63e-6`) and the observed maximum rises from 2 to 3.

For comparison, accessed-path-only `Z=2,Y=2` previously had mean/max stash
`4.757/33`, `35.994/103`, `533.825/719`, and `2084.770/2384` at these four
tree sizes.  Thus one public bit-reversal sweep changes this tested
configuration from an approximately linear-in-`N` backlog to an empirically
`N`-stable, small stash.  A second sweep makes the measured post-eviction stash
empty.  Neither finite experiment establishes a cryptographic tail bound, and
the added paths still cost bandwidth in a real payload implementation.

## `Y=3` metadata-simulator timing

Three isolated short timing repetitions at `N=2^20` gave:

| Schedule | Mean seconds | Standard deviation | Relative to baseline |
|---|---:|---:|---:|
| Accessed path only | 29.424 | 0.311 | 1.000 |
| Access + one deterministic sweep | 23.522 | 0.155 | 0.799 |
| Access + two deterministic sweeps | 26.812 | 0.121 | 0.911 |

The cleaner states reduce metadata source-search and movement work enough to
offset extra path processing in this simulator. This does not imply the same
speedup in the real payload implementation: each added background path
requires another tree path read/write, whose memory traffic is absent from a
placement-only performance conclusion.

The `Y=3` raw CSVs are in `lane_deterministic_runs/`; the `Y=2` raw CSVs are in
`lane_deterministic_y2_runs/`. The schedules are implemented in
`crates/oram/src/bin/lane_oram_minibucket_security.rs`.
