# Legacy root-lane eviction versus Circuit source selection

## Question

Does the original Lane ORAM transition—move the first stashed block into each
free root lane, then propagate holes downward—need only a few more stash slots
than the Circuit-style transition that selects the deepest compatible source?

No. The one-operation service difference is bounded by the number of lanes,
but placement choices change later vacancies. The measured stash distributions
can therefore diverge without an additive bound.

## Matched methodology

Both policies use:

- a binary tree, one-slot lanes (`Y=1`), a shared ordered stash, cyclic keys,
  and a fresh label from OS `getrandom` on every initialization and measured
  access;
- the same gap-free bit-reversed deterministic background-path sequence;
- an accessed-path eviction plus an average of `r` background paths;
- randomized old and new positions during initialization;
- post-eviction stash occupancy. Insertion demand is exactly one larger.

The new `legacy-root-lane` mode reuses the metadata simulator already
regression-tested against the real AVX-512 `LaneORAM`. A new test verifies
block conservation with fractional background paths. The exact-state
equivalence tests and the new conservation test all pass.

The policies differ only in the loaded-path transition:

- **Legacy:** admit the first stash block to each free root lane, calculate the
  original lane masks, and propagate holes downward.
- **Circuit:** for each lane, select the deepest compatible source from the
  stash/path and move it toward the deepest available destination.

## N = 2^10 pilots

Each row uses `2^20` warm-up and `2^22` measured accesses.

| Z | r | Legacy mean / maximum | Circuit mean / maximum |
|---:|---:|---:|---:|
| 3 | 2/3 | 1.562 / 94 | 0.448 / 8 |
| 4 | 1/2 | 0.100 / 40 | 0.542 / 6 |
| 4 | 2/3 | 0.012 / 29 | 0.419 / 6 |
| 5 | 1/2 | 0.003 / 10 | 0.532 / 4 |

The legacy policy can have a lower mean while having a dramatically worse
tail. Most accesses leave an empty stash, but occasional congestion episodes
become large.

## N = 2^20 scaling check

Each row measures `2^20` accesses. The `Z=4` rows use `2^25` warm-up;
the `Z=5` row uses `2^24`, matching the existing Circuit endpoint.

| Z | r | Legacy mean / maximum | Circuit mean / maximum |
|---:|---:|---:|---:|
| 4 | 1/2 | 10662.266 / 13219 | 0.580 / 8 |
| 4 | 2/3 | 0.825 / 116 | 0.427 / 6 |
| 5 | 1/2 | 0.127 / 51 | 0.536 / 5 |

Thus `Z=4,r=1/2` is unstable under legacy eviction. Increasing service to
`r=2/3`, or increasing to five lanes, controls the mean but does not reproduce
the short Circuit tail.

Representative observed post-stash tails:

| Configuration | Threshold R | Legacy Pr[S > R] | Circuit Pr[S > R] |
|:--|---:|---:|---:|
| Z=4, r=2/3 | 2 | 5.257e-2 | 4.196e-4 |
| Z=4, r=2/3 | 4 | 3.801e-2 | 7.629e-6 |
| Z=4, r=2/3 | 6 | 2.985e-2 | 0 observed |
| Z=4, r=2/3 | 32 | 6.008e-3 | 0 observed |
| Z=4, r=2/3 | 96 | 3.891e-4 | 0 observed |
| Z=5, r=1/2 | 2 | 1.211e-2 | 8.236e-3 |
| Z=5, r=1/2 | 4 | 7.151e-3 | 2.861e-6 |
| Z=5, r=1/2 | 32 | 2.251e-4 | 0 observed |

Zeros mean no event in `2^20` samples, not a cryptographic bound.

## Is the extra stash selection worth it?

For a real constant-time implementation with `E=1`, the approximate number
of full allocated-stash passes per logical access is:

- legacy: `2 + Z(1+r)`;
- Circuit selection: `2 + 2Z(1+r)`.

For `Z=4,r=2/3`, that is about 8.67 versus 15.33 passes. The additional
passes are expensive, but here they reduce the observed maximum from 116 to 6
and eliminate the visible heavy congestion episodes. For a small,
security-sized stash, Circuit selection is therefore worth its scan cost.

A legacy implementation might still be competitive if given substantially
more lanes, a higher background rate, or a much larger stash. It cannot be
treated as Circuit selection plus an additive one- or two-entry stash penalty.

These are empirical comparisons, not `2^-96` proofs. The real payload
implementation also does not yet implement background paths; these runs isolate
placement security rather than production latency.
