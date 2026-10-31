# Studies

| Slot | Studies |
| ---- | --- |
| Retiring | [retiring](x86_64/retiring) |
| Bad speculation | [bad_speculation](x86_64/bad_speculation) |
| Frontend bound | [frontend_bound](x86_64/frontend_bound) |
| Backend bound | [backend_bound](x86_64/backend_bound) |

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
