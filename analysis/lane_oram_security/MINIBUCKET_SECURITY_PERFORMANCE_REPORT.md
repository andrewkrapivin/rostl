# Fixed-minibucket Lane ORAM: scaling, tail fits, and performance

## Security target

The design target used here is a per-operation overflow probability at most
`2^-96`.  By a union bound, at most `2^32` operations then have total overflow
probability at most `2^-64`.

No feasible empirical run can observe a `2^-96` event.  The reported capacities
are log-linear extrapolations fitted to the measured insertion-demand tail
between approximately 4 and 19 security bits at `N=2^20`.  Solid portions of
the Figure-3a-style plot are measured; dashed portions are extrapolated.  The
samples are serially correlated, and the fit residual is not a rigorous
confidence bound.  These numbers are engineering estimates, not 96-bit proofs.

## Stash scaling

The metadata simulator used cyclic accesses, independent OS-random initial and
replacement positions, `2^21` warm-up operations, and `2^20` measured operations
per point.  Each configuration was evaluated at `log2(N)=10,12,...,20`.

| configuration | mean at 2^10 | mean at 2^16 | mean at 2^20 | max at 2^20 |
|:--|--:|--:|--:|--:|
| B=2,Z=2,Y=3 | 1.825 | 3.050 | 5.304 | 47 |
| B=2,Z=3,Y=2 | 1.245 | 1.473 | 1.702 | 20 |
| B=2,Z=3,Y=3 | 1.113 | 1.144 | 1.154 | 13 |
| B=4,Z=4,Y=3 | 3.445 | 4.580 | 9.761 | 50 |
| B=4,Z=5,Y=3 | 3.118 | 3.252 | 3.424 | 23 |
| B=4,Z=4,Y=4 | 3.209 | 3.331 | 3.403 | 20 |

`B=2,Z=3,Y=3`, `B=4,Z=5,Y=3`, and `B=4,Z=4,Y=4` show the clearest
near-flat scaling.  `B=2,Z=2,Y=3` has a modest mean but a much less favorable
tail, demonstrating that mean occupancy is not sufficient for sizing a stash.

## Tail extrapolation at N=2^20

| configuration | slots/bucket | fitted slots/security bit | fit residual | estimated S at 2^-96 | benchmark S |
|:--|--:|--:|--:|--:|--:|
| B=2,Z=3,Y=3 | 9 | 0.635 | 0.181 | 63 | 64 |
| B=4,Z=4,Y=4 | 16 | 0.945 | 0.098 | 95 | 96 |
| B=4,Z=5,Y=3 | 15 | 1.074 | 0.321 | 106 | 112 |
| B=2,Z=3,Y=2 | 6 | 1.313 | 0.219 | 126 | 128 |
| B=4,Z=4,Y=3 | 12 | 2.072 | 1.080 | 211 | not selected |
| B=2,Z=2,Y=3 | 6 | 3.172 | 2.360 | 309 | 320 |

The poor residuals for `Z=2,Y=3` and `B=4,Z=4,Y=3` make their extrapolations
especially uncertain.  A conservative production choice should add margin to
the fitted capacity and should be backed by a proof or importance-sampling
analysis before claiming a 96-bit bound.

## AVX-512 performance

Criterion used ten samples, 0.2 seconds warm-up, and one second measurement.
Times are per update of a 56-byte payload.  Circuit ORAM's baseline has a
32-byte payload and is therefore not an equal-payload comparison.

| configuration | estimated secure S | N=2^16 | N=2^20 |
|:--|--:|--:|--:|
| B=2,Z=3,Y=3 | 64 | 1.493 us | 1.914 us |
| B=2,Z=3,Y=2 | 128 | 1.581 us | 1.968 us |
| B=4,Z=4,Y=4 | 96 | 1.781 us | 2.309 us |
| B=4,Z=5,Y=3 | 112 | 2.128 us | 2.535 us |
| B=2,Z=2,Y=3 | 320 | 2.384 us | 2.702 us |
| Circuit ORAM, 32-byte payload | built-in | 0.810 us | 1.239 us |
| old Lane ORAM Z=3,Y=1 (not secure) | S=20 | 0.317 us | 0.471 us |

The best security/performance point in this experiment is
`B=2,Z=3,Y=3,S=64`.  `B=2,Z=3,Y=2,S=128` is close and uses one-third less tree
memory; it was slightly faster at `N=2^20`.  The old `Y=1` implementation is
faster precisely because it scans a tiny stash and moves fewer cache lines, but
the security experiments show that configuration is not viable.

An added equal-payload control gives Circuit ORAM a 56-byte value as well:

| 56-byte payload | N=2^16 | N=2^20 |
|:--|--:|--:|
| Circuit ORAM, S=20 | 0.873 us | 1.370 us |
| Fixed Lane ORAM, B=2,Z=3,Y=3,S=64 | 1.409 us | 1.914 us |
| Lane / Circuit | 1.61x | 1.40x |

This equalizes payload size but not demonstrated failure probability.  The
repository hard-codes Circuit ORAM's stash to 20, whereas Lane's 64-entry
stash is selected from the empirical `2^-96` extrapolation.  Circuit's S=20
must be analyzed with the same tail methodology before calling this a
security-matched comparison.

Explicit AVX-512 cache-line path loads/stores reduced the `B=2,Z=3,Y=3,S=64`
time at `N=2^20` from 2.094 us to 1.914 us.  Matched controls isolate the width
cost from stash size and eviction-policy cost:

| full-Circuit fixed implementation | Y=1 | Y=3 | Y=3 / Y=1 |
|:--|--:|--:|--:|
| Z=3,S=20,N=2^20 | 0.809 us | 1.606 us | 1.99x |
| Z=3,S=64,N=2^20 | 1.104 us | 1.914 us | 1.73x |

Thus increasing `Y` from one to three costs less than 3x when stash and
algorithm are held fixed.  The larger comparison against the old Lane ORAM
also includes increasing `S` from 20 to 64 and replacing its simple lane-mask
eviction with full Circuit target/source selection.

## Implementation

`LaneORAMFixed<Z,Y,S,B>` stores each physical bucket as nested `[Z][Y]`
cache-line blocks.  All `Y` slots have the same legal depth.  Each lane runs a
fixed-loop adaptation of Circuit ORAM Algorithms 2--4 using conditional moves
and conditional swaps; the shared stash is rescanned for each lane.  Initial
position-map entries must be independently randomized.

The implementation currently makes one Circuit-style chain per lane on the
accessed path.  It does not add Circuit ORAM's two extra scheduled eviction
paths per operation.  Overflow remains terminal (`assert!`) rather than being
reported as a recoverable error.

## Runtime decomposition by stash-capacity regression

Holding `B=2,Z=3,Y=3,E=1,N=2^20` fixed and varying only allocated stash
capacity gives the following Criterion point estimates:

| S | ns/update |
|---:|---:|
| 8 | 1498.1 |
| 16 | 1594.2 |
| 20 | 1594.2 |
| 32 | 1654.9 |
| 48 | 1784.1 |
| 64 | 1891.8 |
| 128 | 2360.5 |

The least-squares fit is `T(S) = 1449.7 + 7.0565 S` ns with `R^2=0.9968`.
At the security-sized `S=64`, this attributes approximately 451.6 ns (23.8%)
to stash-capacity-dependent checking and 1449.7 ns (76.2%) to path I/O, path
checking, Circuit eviction preparation/movement, and fixed update bookkeeping.
The stash is traversed five times per operation: lookup, insertion, and once
for each of three lane evictions.  This is a differential decomposition, not a
cycle-level profiler, so the intercept also contains small constant overheads.
