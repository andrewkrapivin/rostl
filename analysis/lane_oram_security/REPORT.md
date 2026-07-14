# Lane ORAM stash-security study

> **Scope correction (2026-07-12):** This report models the current partial
> Rust implementation exactly: one eviction on the accessed old path and no
> independent/background eviction paths. Its primary raw runs used zero as the
> absent-key access path, although a random-path initialization control later
> converged to the same long-run distribution for this one-path transition.
> A previous intended Lane ORAM simulation reportedly keeps the stash below 10
> and likely includes a different eviction schedule. Therefore the results
> below diagnose the current one-path Rust code; they are not yet a security
> estimate for the intended Lane ORAM algorithm.

## Bottom line

The current partial Lane ORAM does not exhibit Circuit ORAM's constant,
exponentially-tailed stash behavior at full load. For every requested `(B,Z)`
configuration, the stash is thousands of blocks at `N=4096` and its fraction
of `N` increases with capacity. A stash of 20 blocks is not close to adequate.

There are two distinct results:

1. **Actual initialization:** a theorem for the current API shows that normal
   sequential insertion almost certainly fills a finite stash before the ORAM
   can reach steady state. At `N=4096,S=20`, survival probability is at most
   `2^-4052` for B2/Z3 and smaller for the other configurations.
2. **Hypothetical stationary behavior with an effectively unbounded stash:**
   after enough cyclic warm-up, mean stash ranges from 1,886 to 3,146 blocks at
   `N=4096`. This is the experiment analogous to Circuit ORAM Figure 3a.

In release builds the real implementation silently drops the updated block at
overflow; only debug builds assert. Therefore this is a correctness issue as
well as a security-parameter issue.

## Methodology

The supplied Circuit ORAM paper uses `N=2^10`, one `2^25`-access warm-up, and
one `2^33`-access measured trajectory under repeated `1..N` requests. It plots
stash threshold `R` against `log2(1/Pr[stash>R])`.

This study uses:

- common exact capacity `N=2^12=4096`, because it is simultaneously an exact
  power of B=2, B=4, and B=8;
- cyclic access through all initialized keys;
- `2^24` warm-up operations;
- `2^27=134,217,728` primary measured operations per configuration;
- four additional seeds, each with `2^25` measured operations;
- an effectively unbounded ordered stash of `N+1` slots;
- both paper-compatible post-writeback occupancy and the transient insertion
  demand that actually determines failure in this implementation.

The primary run has finite raw resolution `2^-27`. Samples are correlated, so
the final few observed tail points must not be read as independent-sample
confidence bounds or extrapolated to 80-bit security.

### Simulator equivalence

The optimized simulator stores one key per slot plus `labels[key]`; it omits
the 56-byte payload. This is control-flow equivalent because payload never
affects placement and every occupied block's position equals its current
label. Ordered stash slots, first-empty insertion, first-nonempty root
admission, bucket-major paths, prefix tests, and fixed-lane moves are retained.

Validation performed:

- exact differential comparison with the AVX-512 implementation for all five
  requested configurations;
- 5,000 operations per configuration at `N=64`;
- every ordered stash slot and every tree path compared every 97 operations;
- a separate small-stash test predicted the exact update where the real debug
  implementation asserted;
- all 27 ORAM crate tests pass with AVX-512 enabled.

## Rigorous initialization failure bound

The current API inserts an absent key with old position zero, so loading
distinct keys reads path zero repeatedly. When a free root lane admits a block
whose first base-B label digit is nonzero, that lane cannot move below the root
on path zero and is never requested again during loading. It is permanently
sealed.

Let `A` be total tree admissions during loading, `Q=N-A` the final stash, and
`p=(B-1)/B`. Then

```text
E[A] <= Z/p = ZB/(B-1)
E[Q] >= N - ZB/(B-1)
Pr[Q < N-m] <= Pr[Binomial(m,p) < Z].
```

A capacity-S stash survives loading only if `A >= N-S`, hence

```text
Pr[survive loading] <= Pr[Binomial(N-S-1,p) < Z].
```

For `N=4096,S=20`:

| Configuration | E[tree admissions] upper bound | Survival upper bound |
|---|---:|---:|
| B=2, Z=3 | 6.00 | 2^-4052 |
| B=4, Z=5 | 6.67 | 2^-8100 |
| B=4, Z=6 | 8.00 | 2^-8089 |
| B=8, Z=9 | 10.29 | 2^-12122 |
| B=8, Z=10 | 11.43 | 2^-12110 |

These are rigorous upper bounds on survival, not empirical estimates.

As a sanity check, the exact simulator left only 5, 6, 7, 10, and 11 blocks in
the tree immediately after loading for B2/Z3, B4/Z5, B4/Z6, B8/Z9, and B8/Z10,
respectively. One later update can reduce stash occupancy by at most Z, so the
initial linear lower bound persists for the first `Theta(N)` cyclic updates.

## Stationary insertion-demand results

The following table is conditional on using a large enough stash to complete
initialization and warm-up. `R(lambda)` is the smallest observed capacity with
per-operation insertion-demand tail at most `2^-lambda` in the primary run.

| Configuration | Mean | Observed range | R(2^-8) | R(2^-16) | R(2^-24) |
|---|---:|---:|---:|---:|---:|
| B=2, Z=3 | 1886.63 | 1535–2342 | 2145 | 2311 | 2341 |
| B=4, Z=5 | 2863.72 | 2605–3228 | 3046 | 3178 | 3227 |
| B=4, Z=6 | 2617.35 | 2291–2993 | 2813 | 2957 | 2992 |
| B=8, Z=9 | 3146.48 | 2901–3404 | 3295 | 3379 | 3402 |
| B=8, Z=10 | 3040.88 | 2779–3304 | 3197 | 3284 | 3303 |

The `R(2^-24)` column is a finite-run empirical quantile, not a proof of
24-bit failure security. For a lifetime of `T` operations, a snapshot tail
`delta_R` only gives the general union bound `Pr[any failure] <= T*delta_R`.

## Scaling with N

Mean post-writeback stash fractions at the largest exact capacities tested:

| Configuration | N | Mean stash | Mean stash/N |
|---|---:|---:|---:|
| B=2, Z=3 | 16,384 | 8,215.53 | 0.501 |
| B=4, Z=5 | 16,384 | 12,130.84 | 0.740 |
| B=4, Z=6 | 16,384 | 11,295.54 | 0.689 |
| B=8, Z=9 | 32,768 | 27,283.91 | 0.833 |
| B=8, Z=10 | 32,768 | 26,636.49 | 0.813 |

The tree has ample nominal capacity—between 6N and 11.43N block slots for
these configurations. The large stash is caused by the fixed-lane dynamic
admission/eviction rule failing to exploit that capacity, not by a static
balls-into-bins overload.

One-lane diagnostics suggest the empirical law

```text
E[tree blocks in one lane]/N ~ c_k / (log_B N)^(k/2), k=log2(B),
```

with fitted constants about 0.62, 0.36, and 0.21 for B=2,4,8. If this law is
asymptotic, fixed Z gives `stash/N -> 1`. This is a conjecture, not a theorem.

## Is cyclic access adversarial?

For any fixed logical schedule independent of the secret labels, the physical
paths read are IID uniform. However, a newly sampled label becomes the old
label at the next access to the same key, so reuse distance still changes Lane
ORAM's state transitions.

At `N=4096`, B2/Z3:

| Workload | Mean post-writeback stash |
|---|---:|
| cycle | 1885.44 |
| reverse cycle | 1886.35 |
| affine permutation cycle | 1885.36 |
| bit-reversal cycle | 1886.36 |
| IID uniform keys | 1659.64 |
| uniform hot set of 16 | 0.30 |
| same key | 0.00 |

All repeated permutations have the same distribution by key-renaming
symmetry. Round-robin is maximally nonrepeating and is clearly adverse compared
with IID and hot-key workloads. It is **not proved globally worst-case**. The
Path/Circuit ORAM duplicate-deletion argument does not directly apply to fixed
lanes, priority root admission, and lane-local hole chains.

## Missing performance benchmarks

Criterion point estimates, 100 samples, AVX-512 enabled:

| logN | B4/Z6 | B8/Z9 | B8/Z10 |
|---:|---:|---:|---:|
| 10 | 176.09 ns | — | — |
| 12 | 219.69 ns | 238.09 ns | 265.01 ns |
| 14 | 227.72 ns | — | — |
| 15 | — | 354.63 ns | 391.26 ns |
| 16 | 343.79 ns | — | — |
| 18 | 415.98 ns | 460.78 ns | 491.63 ns |
| 20 | 492.16 ns | — | — |
| 21 | — | 562.23 ns | 611.12 ns |
| 22 | 581.61 ns | — | — |
| 24 | 630.99 ns | infeasible in RAM | infeasible in RAM |

B=4 is run only at even logN and B=8 only when logN is divisible by 3, so the
logical capacity is an exact power of B. At B=8/logN=24, the tree exceeds this
machine's 8.7 GiB RAM; the run reached 8.6 GiB resident plus 1.7 GiB swap and
was stopped. Sample reduction cannot fix the tree allocation itself.

These timings access one key repeatedly, as in the pre-existing benchmark.
They measure path memory operations but do not represent a full, correctly
loaded finite-stash ORAM.

## Artifacts

- `figure3a_lane_oram.svg`: actual insertion-demand tail, Figure-3a style.
- `stash_scaling.svg`: stash fraction versus capacity.
- `primary_summary.csv`, `replicate_summary.csv`, `scaling_summary.csv`, and
  `workload_summary.csv`.
- `benchmark_summary.csv`: Criterion estimates and confidence intervals.
- `lane_oram_access_schedule.tex` and compiled PDF: proofs, caveats, and
  theoretical analysis.
- `primary_*.csv`, `replicate_*.csv`, `scaling_*.csv`, and `workload_*.csv`:
  raw tail counts.
- `scripts/analyze_lane_oram_security.py`: dependency-free reproduction of
  summaries and SVGs.
