# Random queues and `B=4` minibucket results

All measurements use cyclic accesses, independently randomized initial and new
positions from OS `getrandom`, and report post-eviction stash occupancy.

`queue_policy=shared` is the original minibucket experiment: all lanes select
from one pooled stash. `queue_policy=random` assigns each newly accessed block
uniformly to one of `Z` independent lane queues; only that lane may propagate
the block. The random assignment is redrawn on every access.

## Random queues, `B=2`, `N=2^16`

| Z | Y | policy | mean | minimum | maximum |
|---:|---:|:---|---:|---:|---:|
| 2 | 1 | shared | 3610.699 | 3381 | 3827 |
| 2 | 1 | random | 5252.471 | 4980 | 5547 |
| 3 | 1 | random | 1066.55 | 901 | 1271 |
| 2 | 3 | shared | 2.981 | 0 | 29 |

Randomly partitioning the queue does not cure the `Y=1` backlog. It is worse
than sharing the stash because statistical lane imbalance cannot borrow unused
service or storage from another queue. In contrast, pooling three slots inside
each of two lanes is highly effective.

For `B=2,Z=2,Y=3` at `N=2^18` (warm-up `2^23`, samples `2^22`), the stash was
`0 / 3.727 / 36` (minimum/mean/maximum), showing only slow growth over the
`N=2^16` result.

## `B=4`, `N=2^16`

| Z | Y | bucket slots | policy | mean | minimum | maximum |
|---:|---:|---:|:---|---:|---:|---:|
| 2 | 1 | 2 | shared | 16043.158 | 15634 | 16403 |
| 2 | 1 | 2 | random | 18974.153 | 18363 | 19417 |
| 3 | 1 | 3 | random | 11707.487 | 11325 | 12031 |
| 4 | 1 | 4 | shared | 4212.907 | 3938 | 4517 |
| 4 | 1 | 4 | random | 7510.822 | 7226 | 7844 |
| 2 | 2 | 4 | shared | 5562.465 | 5232 | 5916 |
| 2 | 3 | 6 | shared | 1869.260 | 1641 | 2105 |
| 3 | 2 | 6 | shared | 1109.800 | 907 | 1309 |
| 2 | 4 | 8 | shared | 517.218 | 363 | 688 |
| 4 | 2 | 8 | shared | 172.620 | 91 | 275 |
| 3 | 3 | 9 | shared | 73.470 | 15 | 164 |

## `B=4`, `N=2^18`

| Z | Y | bucket slots | mean | minimum | maximum |
|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 8 | 686.125 | 514 | 865 |
| 3 | 3 | 9 | 267.617 | 158 | 409 |
| 4 | 3 | 12 | 5.803 | 0 | 35 |

Thus `B=4` needs considerably more service/storage slack. Among the tested
points, `Z=4,Y=3` is the first clearly good configuration at `N=2^18`.
Comparisons with different `Z*Y` also show that more independent propagation
chains help: at eight total slots, `Z=4,Y=2` is much better than `Z=2,Y=4` at
`N=2^16`; nevertheless, sufficient within-lane pooling remains necessary.

These are empirical finite-run results, not security proofs.
