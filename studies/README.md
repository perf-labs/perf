# Studies

Worked analyses on real hardware: one notebook per level-1 top-down slot. Each
one is a small self-contained experiment — write a C function, compile it,
sweep one axis with `perf.benchmark()`, read the top-down slots, and confirm
with raw counters.

| Slot | Notebook | Question |
| --- | --- | --- |
| Retiring | [retiring/instructions_per_cycle.ipynb](x86_64/retiring/instructions_per_cycle.ipynb) | A dependent chain vs ILP — how much does IPC alone tell you? |
| Bad speculation | [bad_speculation/branch_mispredit.ipynb](x86_64/bad_speculation/branch_mispredit.ipynb) | Predictable vs unpredictable branches — what do mispredicts cost? |
| Frontend bound | [frontend_bound/dispatch_width.ipynb](x86_64/frontend_bound/dispatch_width.ipynb) | Narrow vs wide dispatch — where is the front-end limit? |
| Backend bound | [backend_bound/cache_level.ipynb](x86_64/backend_bound/cache_level.ipynb) | L1-resident vs DRAM-resident data — what does a miss really cost? |

Every number in them is measured on the machine that ran them, not copied
anywhere. The notebooks pin themselves, sweep one axis at a time, and print
medians with the raw `cycles`/`instructions`/miss counters next to the top-down
slots so the two can be cross-checked.

## Top-down analysis in one page

A core retires at most 4–6 instructions per cycle. The four top-down *slots*
partition every retired instruction by **what stopped it from retiring**:

| Slot | The core stopped because | Usual fix |
| --- | --- | --- |
| `topdown-retiring` | nothing — it issued | the good case; more ILP |
| `topdown-bad-spec` | a branch or load was mispredicted | inline, reorder, make it predictable |
| `topdown-fe-bound` | the front-end could not fetch fast enough | less code, better layout, fewer branches |
| `topdown-fe-bound` (large) | an instruction cache or TLB miss | inline the hot path, shrink the working set |

They sum to ~100% of slots, so the biggest slot is where the cycles are. The
method is always the same:

1. Measure the target and read the four slots (`-e 'topdown-*'`).
2. Take the slot that dominates and name the bound.
3. Change **one** thing that would relieve exactly that bound.
4. Re-measure the same target and check the slot moved.

```sh
perf benchmark a.out:foo -m latency -e 'topdown-*'
```

Slot and counter disagreeing is a finding, not noise: high `retiring` with low
`instructions/cycles` means a dependency chain, not a throughput limit; a
`fe-bound` number that barely moves under `--config.itlb=cold` is decoder
density, not a TLB.

Two caveats worth keeping in mind: on hybrid CPUs the slots only exist on some
core types (the tools pin to a capable CPU), and slot percentages say *where*,
never *how much to fix* — the raw counters and `perf compare` decide that.

`topdown-*` is a documented alias, not a hardware guarantee: the pattern always
expands to the four slots above, but a CPU whose PMU does not export them
(most non-Sapphire-Rapids parts) then fails with `unknown event
'topdown-retiring'` rather than silently reporting zeros. A bare `-e '*'`
expands only to what the host's sysfs PMUs really publish. Check
`perf list | grep topdown` before planning a top-down study.
