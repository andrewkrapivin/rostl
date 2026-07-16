# Fixed-gap pre-batch stash tails

## Scope

This experiment uses the X=16 pooled construction: eight constituent ORAMs,
each with Y=2 slots per minibucket. Total bottom capacity is fixed at 2^20
slots, so each constituent has M=2^16 leaves. Loads are
rho=K/(16M)=100%, 105%, 110%, 115%, 120%, and 125%.

A normal operation accesses and evicts only the key's assigned constituent.
Every G operations, a synchronized batch evicts the next bit-reversed
deterministic path in all eight constituents. Thus:

- fixed gap G=5 has amortized background rate E=8/5=1.6;
- fixed gap G=6 has amortized background rate E=8/6=1.333...;
- the earlier fractional schedule E=1.5 lies between them.

The new pre_batch metric is sampled after the ordinary accessed-path
eviction and immediately before the synchronized wide batch. A regression
test verifies that insertion demand is at least pre_batch, which is at least
the post-batch stash.

Each main run uses 2^21 warm-up operations followed by 2^24 measured cyclic
accesses in sixteen 2^20-operation epochs. Random initialization, replacement
leaves, and constituent assignments use OS getrandom.

## Observed pre-batch stash

| Gap | Load | Boundary samples | Mean | Observed maximum |
|---:|---:|---:|---:|---:|
| 5 | 100% | 3,355,443 | 8.340 | 25 |
| 5 | 105% | 3,355,443 | 8.345 | 22 |
| 5 | 110% | 3,355,443 | 8.368 | 23 |
| 5 | 115% | 3,355,444 | 8.443 | 28 |
| 5 | 120% | 3,355,443 | 8.923 | 28 |
| 5 | 125% | 3,355,443 | 10.254 | 37 |
| 6 | 100% | 2,796,202 | 11.565 | 37 |
| 6 | 105% | 2,796,202 | 11.822 | 37 |
| 6 | 110% | 2,796,202 | 12.422 | 40 |
| 6 | 115% | 2,796,203 | 13.793 | 48 |
| 6 | 120% | 2,796,203 | 17.964 | 60 |
| 6 | 125% | 2,796,203 | 33.774 | 97 |

Gap 5 is nearly load-insensitive through 115%, worsens mildly at 120%, and
shows a clearer penalty at 125%. Gap 6 begins degrading earlier and is
particularly poor at 120--125%.

## Measured Figure-3 analogues

- [Fixed gap 5](fixed_gap5_prebatch_figure3.png)
- [Fixed gap 6](fixed_gap6_prebatch_figure3.png)
- [Direct gap-5 versus gap-6 comparison for every load](fixed_gap_prebatch_by_load.png)

The horizontal axis is log2(1/Pr[pre-batch stash > R]); the vertical axis is
the threshold R in blocks. Solid segments have at least 64 directly observed
exceedances. Dashed segments contain only 1--63 exceedances. No extrapolated
line is drawn.

## Tail curvature

Local linear fits report blocks of additional stash per probability bit.
A decreasing value in deeper windows means that R versus log2(1/p) is
flattening and the tail is decaying faster than a single exponential fit
would predict.

| Gap | Load | Bits 4--8 | Bits 8--12 | Bits 12--16 |
|---:|---:|---:|---:|---:|
| 5 | 100% | 0.771 | 0.695 | 0.750 |
| 5 | 120% | 1.335 | 1.213 | 0.846 |
| 5 | 125% | 1.916 | 1.487 | 1.135 |
| 6 | 100% | 1.319 | 1.224 | 1.026 |
| 6 | 120% | 3.090 | 2.210 | 1.295 |
| 6 | 125% | 4.686 | 2.770 | 2.044 |

The taper is clear at high load and visible for gap 6 even at 100%. Gap 5 at
100% is close to linear over the measured range. These data support the
hypothesis that a larger shared stash makes it increasingly likely that more
constituents have an eligible block for the scheduled path. They do not yet
establish the asymptotic form of the tail.

## Single long run versus an ensemble

For each gap, the endpoint loads 100% and 125% were also tested using eight
independent runs. Each independent run used fresh OS randomness, 2^21 warm-up
operations, and 2^20 measured operations. Their histograms were pooled and
compared with the corresponding single 2^24-operation trajectory.

[Long-run versus ensemble tails](fixed_gap_time_vs_ensemble.png)

| Gap | Load | Long mean | Ensemble mean | KS distance | Max tail difference, bits (counts >=64) |
|---:|---:|---:|---:|---:|---:|
| 5 | 100% | 8.340 | 8.326 | 0.003 | 0.427 |
| 5 | 125% | 10.254 | 9.935 | 0.034 | 3.285 |
| 6 | 100% | 11.565 | 11.654 | 0.011 | 0.548 |
| 6 | 125% | 33.774 | 35.639 | 0.058 | 2.307 |

At 100%, agreement is excellent. This is direct empirical support for using a
single long trajectory to estimate the batch-boundary stationary
distribution.

At 125%, the disagreement is larger, but the process itself also varies much
more slowly:

| Gap | Long-run epoch-mean range | Independent-run mean range |
|---:|---:|---:|
| 5 | 9.427--11.713 | 9.428--10.267 |
| 6 | 26.678--39.123 | 28.187--42.846 |

The ranges overlap. There is no evidence here that time averaging converges
to a different distribution from ensemble averaging. Instead, high load has
stronger autocorrelation or slower mixing, so neither a 2^24 run nor short
independent runs determine the deep tail precisely.

## Methodological conclusion

The fixed-gap boundary process is a finite Markov chain after augmenting the
full ORAM state with the cyclic request phase and deterministic-path phase.
Within its recurrent class, the Markov-chain ergodic theorem justifies the
single-long-run time average. Fixed gaps remove the additional 5,5,6 credit
phase from the earlier fractional schedule.

This experiment supports that methodology at low load and is consistent with
it at high load. It does not prove uniqueness of the recurrent class or give
a mixing-time bound. A 2^32 run should therefore retain epoch histograms, and
the high-load result should be checked against at least a few independent
checkpoints.

For security sizing, retain the every-operation insertion tail as well.
pre_batch is the correct embedded-chain statistic, but an earlier operation
between batches can still have the larger capacity demand.

## Artifacts

- Main and independent raw runs: fixed_gap_prebatch_runs/
- Full summary: fixed_gap_prebatch_summary.csv
- Local slopes: fixed_gap_prebatch_local_slopes.csv
- Long-run/ensemble comparison: fixed_gap_time_vs_ensemble.csv
- Analysis and SVG renderer: ../../scripts/analyze_fixed_gap_prebatch.py
