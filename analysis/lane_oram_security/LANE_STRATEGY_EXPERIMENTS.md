# Same-path Lane ORAM eviction strategies

This experiment tests ways to reduce stash occupancy without giving up Lane
ORAM's locality advantage.  Every strategy reads and writes only the accessed
tree path.  Results use `B=2`, cyclic accesses, independently randomized initial
and replacement labels from OS `getrandom`, and exact Circuit-style source
selection.

## Strategies

- **Fixed, one pass:** one Circuit eviction chain per fixed lane.
- **Fixed, two passes:** repeat every lane chain after the first pass, while the
  path remains in the contiguous cache buffer.  Tree traffic is unchanged.
- **Pooled:** remove lane partitioning and run global Circuit chains over all
  `Z*Y` slots.  Comparisons use the same total number of chains as fixed lanes.
- **Reallocation:** change `Z` and `Y` while retaining one fixed-lane pass.

## N=2^18 screening

| Z | Y | strategy | total chains | bucket width | mean stash | maximum |
|---:|---:|:--|---:|---:|---:|---:|
| 3 | 3 | fixed 1 pass | 3 | 9 | 1.152 | 14 |
| 3 | 3 | fixed 2 passes | 6 | 9 | 1.006 | 13 |
| 3 | 3 | pooled | 3 | 9 | 1.095 | 13 |
| 3 | 2 | fixed 1 pass | 3 | 6 | 1.576 | 20 |
| 3 | 2 | fixed 2 passes | 6 | 6 | 1.061 | 15 |
| 3 | 2 | pooled | 3 | 6 | 1.121 | 17 |
| 2 | 3 | fixed 1 pass | 2 | 6 | 3.924 | 41 |
| 2 | 3 | fixed 2 passes | 4 | 6 | 1.073 | 15 |
| 2 | 3 | pooled | 2 | 6 | 1.560 | 21 |
| 4 | 2 | fixed 1 pass | 4 | 8 | 1.078 | 14 |
| 2 | 4 | fixed 1 pass | 2 | 8 | 1.775 | 26 |
| 2 | 5 | fixed 1 pass | 2 | 10 | 1.529 | 20 |
| 4 | 1 | fixed 1 pass | 4 | 4 | 130.222 | 236 |
| 5 | 1 | fixed 1 pass | 5 | 5 | 5.969 | 43 |

At equal chain count and bucket width, fixed lanes are slightly better than
global pooling.  This supports the original lane-independence intuition.  One
pooled chain on a nine-slot bucket is grossly under-served (mean stash about
28,603), so pooling cannot substitute for service rate.

## Focused N=2^20 tails

Fits use insertion-demand points between approximately 4 and 19 measured
security bits.  `S96` is a log-linear extrapolation, not a proof.

| Z | Y | passes | mean | maximum | fitted blocks/bit | fit residual | estimated S96 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 2 | 1.086 | 17 | 0.771 | 0.083 | 75 |
| 3 | 2 | 2 | 1.065 | 15 | 0.711 | 0.048 | 69 |
| 3 | 3 | 2 | 1.004 | 17 | 0.672 | 0.189 | 65 |
| 4 | 2 | 1 | 1.087 | 15 | 0.728 | 0.148 | 71 |

For comparison, the one-pass estimates were approximately 309 for `Z=2,Y=3`,
126 for `Z=3,Y=2`, and 63 for `Z=3,Y=3`.  A second pass therefore rescues the
six-slot designs but does not improve the already subcritical `Z=3,Y=3` tail.

## Production AVX-512 performance at N=2^20

| configuration | benchmark stash | time/update | tree storage relative to Z3Y3 |
|:--|---:|---:|---:|
| Z=3,Y=3,E=1 | 64 | **1.914 us** | 1.00 |
| Z=4,Y=2,E=1 | 72 | 2.062 us | 0.89 |
| Z=2,Y=5,E=1 | 104 | 2.125 us | 1.11 |
| Z=2,Y=3,E=2 | 80 | 2.151 us | 0.67 |
| Z=3,Y=2,E=2 | 72 | 2.480 us | 0.67 |

The best speed remains `Z=3,Y=3,E=1,S=64`.  `Z=2,Y=3,E=2` is useful when
tree memory matters more than latency: it reduces tree storage by one third for
about 12% more time.  Extra same-path passes preserve tree locality, but the
additional Circuit metadata/source-selection scans outweigh their path-width
savings on this machine.

The production implementation exposes `E` as the fifth const generic:
`LaneORAMFixed<Z,Y,S,B,E>`.  Existing four-parameter uses retain `E=1`.

## One five-slot lane with two evictions

Interpreting a traditional `Z=5` bucket as one pooled lane gives our parameters
`Z=1,Y=5,E=2`.  Its scaling results were:

| log2(N) | mean stash | maximum |
|---:|---:|---:|
| 10 | 1.770 | 22 |
| 12 | 1.823 | 20 |
| 14 | 1.919 | 28 |
| 16 | 2.026 | 35 |
| 18 | 2.217 | 39 |
| 20 | 2.796 | 51 |

The visible growth and irregular heavy tail reject this as a constant-stash
candidate over the tested range.  At `N=2^20`, a short-tail fit estimated
`S96≈328` with a very poor residual of 2.72 blocks.  Benchmarking the rounded
`S=336` configuration took 2.792 us/update, versus 1.914 us for
`Z=3,Y=3,E=1,S=64`.  The one-lane design uses only five tree blocks per bucket,
but loses both security and speed because its large stash must be scanned on
every access and on both eviction chains.

## Quaternary one-lane Y=10 with three evictions

For `B=4,Z=1,Y=10,E=3`, the mean was 3.837 with maximum 32 at `N=2^18`,
and 3.872 with maximum 32 at `N=2^20`.  This is much stronger evidence of an
N-independent plateau than the binary `Z=1,Y=5,E=2` construction.  The
`N=2^20` insertion tail fit had slope 1.706 blocks/bit, residual 0.756, and
estimated `S96≈166`.  With margin `S=176`, the production implementation took
2.693 us/update at `N=2^20`.

Compared with `B=4,Z=4,Y=4,E=1,S=96`, it uses 10 rather than 16 tree blocks
per bucket (37.5% less tree storage), but is about 17% slower (2.693 versus
2.309 us) and has a less convincing tail fit.  It is therefore a reasonable
tree-memory tradeoff, not the speed or security optimum.

## Wider quaternary and octary one-lane configurations

At the common exact size `N=2^18`:

| B | Y | E | mean | maximum | fitted S96 | fit residual | benchmark S | us/update |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 10 | 3 | 3.837 | 32 | 186 | 0.606 | 176* | 2.473 |
| 4 | 8 | 3 | 23.485 | 100 | 411 | 2.068 | 416 | 3.940 |
| 8 | 16 | 7 | 8.200 | 43 | 191 | 0.912 | 192 | 6.043 |
| 8 | 16 | 3 | 24.222 | 130 | 841 | 5.673 | not run | not run |

`*` The B4/Y10 benchmark uses the more stable N=2^20 fit (`S96≈166`) plus
margin; its N=2^18 timing is shown for comparison.

For `B=4`, reducing Y from 10 to 8 crosses a sharp congestion threshold.  For
`B=8`, three chains are grossly insufficient; raising the service rate to
`E=B-1=7` controls the mean but makes full Circuit selection very expensive.
None improves on the binary `Z=3,Y=3,E=1,S=64` speed/security point.
