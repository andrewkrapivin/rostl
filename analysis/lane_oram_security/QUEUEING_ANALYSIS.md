# Queueing analysis of the current Lane ORAM transition

This note analyzes the current block-conserving, fixed-lane transition for
binary trees. It separates exact conservation identities from mean-field and
empirical claims.

## Exact conservation identities

Let `S_t` be post-update stash occupancy. During the next update, let `I_t` be
one when the requested key was in the tree and zero when it was already in the
stash, and let `A_t` be the number of stash blocks admitted into free root
slots. Block conservation gives

\[
S_{t+1}-S_t=I_t-A_t.
\]

Consequently, in stationarity,

\[
\mathbb E[A_t]=\Pr[I_t=1].
\]

This is also why the measured mean transient insertion demand minus mean
post-writeback stash equals the tree-hit/root-admission rate.

For a cut below tree depth `d`, let `U_d` be the number of blocks at depths at
most `d`, `R_d` indicate removal of the requested block from that upper
region, and `C_d` count blocks moved downward across the cut. Since Lane ORAM
never moves blocks upward,

\[
U'_{d}-U_d=A_t-R_d-C_d.
\]

Combining stationary balance with the root identity yields

\[
\mathbb E[C_d]
=\Pr[\text{the requested block is in the tree below depth }d].
\]

Thus a small-stash ORAM must sustain nearly one aggregate downward crossing
per operation across every shallow cut. Nominal tree capacity alone is not
enough.

## Why lower occupancy does not automatically give rate slack

More lanes do help: they split the offered load and create more holes. But an
individual binary lane can cross a level only when an occupied source, a
compatible next label bit, and a reachable vacancy line up in the same lane.

In an independent Bernoulli path-state model with occupied density `rho`, an
expansion of the exact production mask's crossing current is

\[
J(\rho)=\frac{\rho}{2}-\frac{3}{64}\rho^3+O(\rho^4).
\]

The quadratic term cancels because the mask can jump over one incompatible
blocker. If a lane carries load `q`, light-traffic occupancy is approximately
`rho = 2q`, giving

\[
J(2q)=q-\frac{3}{8}q^3+O(q^4).
\]

The relative routing loss at one level is therefore of order `q^2`. Across
`L` marginal prefix-routing levels it accumulates as order `L q^2`, suggesting

\[
q_L=\Theta(L^{-1/2}).
\]

This derivation explains the exponent but is not a proof for the correlated
stationary ORAM process.

## Measured per-lane law and crossover

Saturated one-lane diagnostics for the production transition give

\[
q_L\sqrt L=0.632,0.622,0.622,0.625,0.632,0.633
\]

for `L = 10,12,14,16,18,20`. Hence

\[
q_L\approx\frac{c}{\sqrt L},\qquad c\approx0.627.
\]

The numerical value is close to `sqrt(pi/8)`, but that constant has not been
derived. A zero-drift prefix-credit/random-walk argument plausibly explains
both the square-root law and this constant; proving the mapping while handling
mask and admission correlations remains open.

When lanes are backlogged, they are approximately parallel copies, leading to
the finite-height approximation

\[
\frac{\mathbb E[S]}{N}
\approx
\left(1-\frac{0.627Z}{\sqrt{\log_2N}}\right)_+.
\]

Equivalently,

\[
Z_{\rm crit}(L)\approx\frac{\sqrt L}{0.627}
\approx1.595\sqrt L.
\]

| Z | Predicted crossover height | Observed behavior |
|---:|---:|---|
| 5 | 9.8 | Degradation begins around `N=2^10` |
| 6 | 14.2 | Degradation begins around `N=2^14` |
| 7 | 19.3 | Small through `2^18`, then large at `2^20` |
| 8 | 25.2 | Small but increasingly heavy-tailed through `2^20` |
| 9 | 31.8 | Small through `2^20`; not an asymptotic proof |

At `N=2^20`, the direct-OS-random Z=7 run had post-stash
min/mean/max `1542 / 9416.649 / 16682`. Z=8 had
`0 / 0.873 / 237`, and Z=9 had `0 / 0.049 / 52`.

The fit is unusually accurate across Z=3,5,6,7 and the tested heights, but it
is still an empirical/mean-field security model, not a theorem. In particular,
small mean stash at one height does not establish an N-independent tail.

## Greedy fixed-lane control

A corrected, block-conserving deepest-hole eviction improves the measured
constant from about `0.627` to about `0.82`, but retains the `1/sqrt(L)` law.
It therefore postpones rather than removes the fixed-lane crossover. This
suggests the shallow-hole production rule is costly, while the main scaling
bottleneck is fixed lane identity plus prefix-routing of vacancies.

## Why Circuit ORAM differs

Circuit ORAM pools bucket slots and chooses the deepest compatible candidate
from the stash and path. A vacancy can therefore be matched to a block in any
slot and moved directly to the source level instead of diffusing through one
fixed lane. Its standard construction also performs two eviction paths per
logical access, providing genuine rate slack. Lane ORAM's Z potential chains
are useful, but their realized aggregate current is reduced by same-lane
compatibility and vacancy routing at every tree level.

## Data

- `b2_z5_scaling_summary.csv`
- `b2_z67_scaling_summary.csv`
- `b2_z7_n262144_replicates.csv`
- `b2_z7_n1048576_summary.csv`
- `b2_z89_large_summary.csv`
