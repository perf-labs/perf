# Studies

| Slot | Studies |
| ---- | --- |
| Retiring | [retiring/ipc](x86_64/retiring/ipc.ipynb) |
| Bad speculation | [bad_speculation/branch](x86_64/bad_speculation/branch.ipynb), [bad_speculation/if_else](x86_64/bad_speculation/if_else.ipynb) |
| Frontend bound | [frontend_bound/icache](x86_64/frontend_bound/icache.ipynb) |
| Backend bound | [backend_bound/dcache](x86_64/backend_bound/dcache.ipynb), [backend_bound/dsize](x86_64/backend_bound/dsize.ipynb) |
| Machine | [x86_64](x86_64/x86_64.ipynb) |

## Method

1. Measure with `-e 'topdown-*'` and take the biggest slot.
2. Change one thing that relieves exactly that bound.
3. Re-measure and check the slot moved.

```sh
perf benchmark a.out:foo -m latency -e 'topdown-*'
```

`topdown-*` needs a PMU that exports it. Otherwise it fails with
`unknown event 'topdown-retiring'` — check `perf list | grep topdown`
and fall back to raw counters (`cycles`, `instructions`, `cache-misses`,
`branch-misses`).

Pin the cache, TLB, branch, alignment and input controls that are not the
variable. Inputs that the notebook says are random are refreshed by the
harness; for a heap-backed sort, `config.heap.mode=random` gives every measured
call a fresh buffer and `fixed` reuses the target's mutated buffer.
