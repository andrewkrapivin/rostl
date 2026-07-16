# Fixed-gap-5 overnight experiment

## Configuration

The overnight experiment used the `X=16` pooled construction: eight
constituent ORAMs with `Y=2`, `Z=8`, total bottom capacity `2^20`, and one
ordinary accessed-constituent eviction per operation. Every five operations,
the same next deterministic path was evicted in all eight constituents. The
amortized background rate was therefore `E=8/5=1.6`.

For each of `rho=110%` and `rho=120%`, four independent runs used fresh OS
randomness, `2^24` warm-up operations, and `2^30` measured operations. The
four histograms were combined, giving `2^32` measured operations per load.
Because the pre-batch boundary occurs once every five operations, each
combined pre-batch histogram contains 858,993,460 samples.

## Main results

| rho | Pre-batch mean | Pre-batch maximum | Insertion mean | Insertion maximum |
|---:|---:|---:|---:|---:|
| 110% | 8.3817 | 30 | 7.4142 | 30 |
| 120% | 8.8494 | 41 | 7.8802 | 41 |

For the pre-batch metric, the smallest integer threshold `R` whose empirical
tail satisfies `Pr[S>R] <= 2^-b` was:

| Target b | rho=110%: R (observed tail bits) | rho=120%: R (observed tail bits) |
|---:|---:|---:|
| 4 | 11 (4.015) | 13 (4.729) |
| 8 | 15 (9.066) | 18 (8.648) |
| 12 | 18 (12.866) | 23 (12.513) |
| 16 | 21 (16.437) | 28 (16.561) |
| 20 | 24 (20.006) | 32 (20.190) |
| 24 | 27 (24.034; 50 exceedances) | 36 (24.820; 29 exceedances) |
| 28 | 29 (28.093; 3 exceedances) | 40 (29.678; 1 exceedance) |

The 24- and 28-bit rows are sparse observations. With 858,993,460 boundary
samples, the conservative 64-exceedance boundary is 23.68 bits. No direct
empirical statement near `2^-96` is possible from these data.

## Figure-3 analogue

[PNG figure](fixed_gap5_overnight_figure3.png) and
[SVG figure](fixed_gap5_overnight_figure3.svg).

The horizontal axis is `log2(1/Pr[pre-batch stash > R])`; the vertical axis
is stash threshold `R` in blocks. Solid curves contain at least 64 observed
exceedances and dashed curves contain 1--63. No extrapolated segment is
drawn.

## Tail shape

Local fits report blocks of additional stash per probability bit:

| rho | bits 4--8 | 8--12 | 12--16 | 16--20 | 20--24 |
|---:|---:|---:|---:|---:|---:|
| 110% | 0.799 | 0.783 | 0.832 | 0.859 | 0.795 |
| 120% | 1.272 | 1.293 | 1.251 | 1.107 | 0.876 |

At 110%, the tail is close to linear throughout the reliable measured
range. At 120%, the slope clearly decreases after about 16 bits, confirming
the suspected taper. The run does not establish that this taper continues
to 96 bits, so extending the last local slope that far would still be an
engineering extrapolation, not an empirical bound.

## Independent-run agreement

| rho | Pre-batch replicate means | Replicate maxima | Maximum KS distance from combined tail |
|---:|---:|---:|---:|
| 110% | 8.3796--8.3834 | 29--30 | 0.000349 |
| 120% | 8.8447--8.8555 | 36--41 | 0.000836 |

The agreement is excellent. It is substantially stronger than in the short
high-load check and supports treating time samples from this fixed-gap
embedded chain as estimates of the stationary pre-batch distribution. It
does not by itself prove a mixing-time bound or the 96-bit tail.

## Artifacts

- Combined and per-replicate data: `fixed_gap5_overnight_runs/`
- Plot renderer: `../../scripts/analyze_fixed_gap5_overnight.py`
- Run launcher: `../../scripts/run_fixed_gap5_overnight.sh`
- Combiner and diagnostics: `../../scripts/combine_fixed_gap5_overnight.py`
