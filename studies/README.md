# Studies

Worked analyses on real hardware: one notebook per level-1 top-down slot. Each
one is a small self-contained experiment — write a C function, compile it,
sweep one axis with `perf.benchmark()`, read the top-down slots, and confirm
with raw counters.

| Slot | Notebook | Question | Result |
| --- | --- | --- | --- |
| Retiring | [retiring/instructions_per_cycle.ipynb](x86_64/retiring/instructions_per_cycle.ipynb) | A dependent chain vs ILP — how much does IPC alone tell you? | IPC 1.08 → 3.68, retiring share 0.28 → 0.72 |
| Bad speculation | [bad_speculation/branch_mispredit.ipynb](x86_64/bad_speculation/branch_mispredit.ipynb) | Does `branch=predictable\|unpredictable` actually produce mispredicts? | **No** — inputs vary, mispredicts do not; negative result |
| Frontend bound | [frontend_bound/icache_tiers.ipynb](x86_64/frontend_bound/icache_tiers.ipynb) | Hot vs cold instruction cache — what does a code miss really cost? | IPC 4.84 → 1.54, fe-bound becomes the dominant slot |
| Backend bound | [backend_bound/cache_level.ipynb](x86_64/backend_bound/cache_level.ipynb) | L1-resident vs DRAM-resident data — what does a miss really cost? | IPC 1.25 → 0.18, be-bound share 0.51 → 0.90 |

Every number in them is measured on the machine that ran them, not copied
anywhere. The notebooks pin themselves, sweep one axis at a time, and print
medians with the raw `cycles`/`instructions`/miss counters next to the top-down
slots so the two can be cross-checked. Each one ends in `check(...)` calls that
assert the claim, and the notebooks record the designs that *failed* — the ones
that looked convincing and measured nothing.

## Top-down analysis method

A core retires at most 4–6 instructions per cycle. The four top-down *slots*
partition every retired instruction by **what stopped it from retiring**:

| Slot | The core stopped because | Usual fix |
| --- | --- | --- |
| `topdown-retiring` | nothing — it issued | the good case; more ILP |
| `topdown-bad-spec` | a branch or load was mispredicted | inline, reorder, make it predictable |
| `topdown-fe-bound` | the front-end could not fetch fast enough | less code, better layout, fewer branches, a smaller instruction footprint |
| `topdown-be-bound` | the core could not get the data | prefetch, layout for locality, shrink the working set |

They sum to ~100% of slots, so the biggest slot is where the cycles are. The
method is always the same:

1. Measure the target and read the four slots (`-e 'topdown-*'`).
2. Take the slot that dominates and name the bound.
3. Change **one** thing that would relieve exactly that bound.
4. Re-measure the same target and check the slot moved.

```sh
perf benchmark a.out:foo -m latency -e 'topdown-*'
```

`topdown-*` is a documented alias, not a hardware guarantee: the pattern always
expands to the four slots above, but a CPU whose PMU does not export them
(most non-Sapphire-Rapids parts) then fails with `unknown event
'topdown-retiring'` rather than silently reporting zeros. A bare `-e '*'`
expands only to what the host's sysfs PMUs really publish. Check
`perf list | grep topdown` before planning a top-down study.
