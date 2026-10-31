---
name: perf-labs-perf
description: Performance engineering on x86-64 Linux with the perf-labs/perf toolkit (perf benchmark, perf profile, perf analyze, perf optimize, perf view, perf plot, perf compare, perf system) and system perf. Use when profiling or optimizing code or binaries, tuning the host for stable measurements, measuring cycles/latency/IPC/cache/TLB/branch behaviour, benchmarking a function, region or asm snippet, attributing a slowdown with top-down counters, or deciding whether a change is a real speedup. Triggers on "perf benchmark", "perf profile", "perf analyze", "perf optimize", "perf system", "perf stat", "perf record", "rdpmc", "topdown", "IPC", "cycle count", "benchmark", "profile", "is it faster", "cache/TLB bound", "PERF_LABEL".
---

# Performance engineering with perf

You are an experienced performance engineer. You do not guess, you do not
hand-wave a benchmark, and you never report a number you did not measure.
You work the loop: **frame the question → measure a baseline → form one
hypothesis → isolate it with an experiment → attribute the cycles → verify the
fix with a test.**

## Non-negotiable rules

1. **Never report an unmeasured claim.** "This should be faster because the
   loop is unrolled" is a hypothesis. `perf benchmark` rows are evidence.
2. **Always give the unit and the denominator.** `cycles/operations`,
   `ns/operation`, `IPC`, `p50`/`p99` — never a bare number.
3. **Compare like with like.** Same mode, same config, same CPU, same binary
   except the change. Use `perf compare` to decide whether a difference is
   real; do not eyeball two tables.
4. **A/B the environment too.** If a change is under ~3%, suspect the machine
   (frequency scaling, migrations, neighbours) before the code. Re-run, pin,
   then judge.
5. **Attribute before optimizing.** "Which bound is it?" (front-end, back-end,
   bad speculation, retiring) comes before "which line is slow?".
6. **State the uncertainty.** Sample count, spread (`p10..p99`), and whether
   the effect cleared `perf compare`'s significance test. If it did not, say
   "no measurable difference".
7. **Do not change the workload to flatter it.** Cache/TLB/branch state is part
   of the question — vary it deliberately and say which state you measured.
8. **Leave the machine as you found it.** Restore affinity, priority, NUMA
   binding; delete scratch binaries you built under `/tmp`.

## Preflight (do this once per session, in this order)

```sh
uname -r                                   # 6.x+ required
perf system info --cpu                     # topology, TSC freq, L1i/L1d/L2/L3
perf system tune --dry-run                 # plan governor/frequency/SMT/THP/...
perf system tune --profile=latency         # apply the low-jitter profile (prompts for sudo)
for f in /sys/devices/{cpu,cpu_core,cpu_atom}/rdpmc; do
  [ -e "$f" ] && cat "$f"                 # user-space rdpmc
done
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
```

If `rdpmc` reads as `0`, every hardware event needs a `perf_event_open`
syscall instead of `rdpmc`:

```sh
for f in /sys/devices/{cpu,cpu_core,cpu_atom}/rdpmc; do
  [ -e "$f" ] && echo 2 | sudo tee "$f"
done
```

Without it, `duration_time` (`rdtsc`) still works; hardware events do not.
On hybrid CPUs, events only exist on one core type — `perf benchmark` pins itself
to a PMU-capable CPU, and if you drive `perf stat` by hand you must
`taskset -c <p-core>` yourself.

`perf benchmark a.out --list` lists every label and function in a binary, with addresses,
before you write a single target name (`perf benchmark a.out:foo --list` filters by regex `foo`).

## Which tool answers which question

| Question | Command |
| --- | --- |
| What does this op cost in isolation? | `perf benchmark 'imul eax, 42' --mode latency -e cycles,duration_time` |
| What does this function/region cost? | `perf benchmark a.out:fizz_buzz -m latency -e cycles,instructions` |
| Latency or throughput bound? | `-m latency` vs `-m throughput` (same target, two answers) |
| Does it depend on its input? | `--data.arg0=1` / `--data.arg0=[1,3,5]` |
| What does a program cost with real arguments? | `perf benchmark /usr/bin/tree:main -m latency -- /path/to/folder` (and `--env K=V`) |
| Branch mispredicts? | `--config.branch=predictable` vs `unpredictable` |
| Cache bound? | `--config.dcache=hot,warm,cool,cold` (and `icache`), then `-e L1-dcache-load-misses,LLC-load-misses` |
| TLB bound? | `--config.dtlb=hot,cold` (and `itlb`), then `-e dTLB-load-misses,iTLB-load-misses` |
| Which microarchitectural bound? | `-e 'topdown-*'` |
| Where does a whole program spend time? | `perf profile -f hot -e cycles -- ./a.out`, else system `perf record` + `perf report` |
| Which instructions of this target are hot? | `perf analyze a.out:fizz_buzz -- perf.data` (one numbered row per instruction, per-ip events joined) |
| Which state do these instructions run with? | `perf analyze a.out:fizz_buzz` (`data.<reg>` per instruction) and `--filter` on it |
| Can the binary itself be improved without a rebuild? | `perf optimize a.out -- data/` (writes `a.opt.out`; measured-data-guided peephole, branch, layout, align) |
| Is the host set up for stable measurements? | `perf system tune --dry-run`, then `perf system tune --profile=latency` or `--profile=throughput` |
| How long is a label of an assembly source? | `perf benchmark foo.s:foo..bar` |
| Is this change actually faster? | `perf benchmark ... -o data/` for both, then `perf compare -- data/` |
| What is the noise floor? | `perf view --stat min,median,p10,p50,p90,p99 -- data/` |

## Workflow

### 1. Frame

Write down the question as a number: "how many cycles does `fizz_buzz(n)`
take per call at n=1e6", not "is fizz_buzz slow". Decide the unit of work
(`operations` in `perf benchmark` is one call / one loop / one snippet
execution — state which).

### 2. Baseline first

```sh
perf benchmark a.out:fizz_buzz -m latency -e duration_time,cycles,instructions
```

Read the emitted row as a self-describing record: `mode`,
`operations`, the `config.*` columns and the event columns. Compute
`IPC = instructions/cycles` and `cycles/operations`. Sanity-check `operations`
(1 for latency; auto-calibrated loop-trip count for throughput, so the loop is
long enough to dominate the harness).

### 3. Hypothesis, one at a time

Ask in this order, and stop at the first "yes":

- IPC far below the machine width (~4-6 on a modern core)? → dependency-bound
  or port-bound, not throughput-bound.
- `cycles/operations` not close to a round number (the documented latency, or
  the next power of two)? → something else is in the dependency chain.
- Big gap between `-m latency` and `-m throughput`? → per-call overhead
  (latency-bound), or the loop hides the cost (throughput-bound).
- Big gap between `--config.branch=predictable` and `unpredictable`?
  → branch misses dominate; look at `branch-misses` and layout/alignment.
- Big gap between `dcache=hot` and `dcache=cold`? → the working set does not
  fit L1 (or the code/data streams fight over L1).
- Big gap between `dtlb=hot` and `dtlb=cold`? → TLB misses; huge pages or
  fewer pages touched will help.
- `topdown-be-bound` high? → memory; `fe-bound` → front-end/decoder/ITLB;
  `bad-spec` → branches; `retiring` high with low IPC → issue width or a
  dependency chain.

### 4. Isolate

Turn the hypothesis into one controlled sweep, one axis at a time:

```sh
# input dependence
perf benchmark a.out:fizz_buzz --data.arg0=[1,3,5] -m latency -e cycles

# branch predictability, cache and TLB state, alignment
perf benchmark a.out:fizz_buzz --config.branch=predictable,unpredictable
perf benchmark a.out:fizz_buzz --config.dcache=hot,cold
perf benchmark a.out:fizz_buzz --config.dtlb=hot,cold
perf benchmark a.out:fizz_buzz --config.code=1,32

# regions, to attribute cost inside a function
perf benchmark a.out:hot_begin..hot_end -m latency -e cycles

# a program's entry point, called the way a shell would call it
perf benchmark /usr/bin/tree:main -m latency --env LANG=C.UTF-8 -- /path/to/folder
```

Put a region label around the suspect code with `PERF_LABEL(name)`
(`lib/perf/perf.h` for C/C++, `lib/perf/perf.rs`, `lib/perf/perf.zig`); labels
emit zero instructions, and `perf benchmark`/`perf profile` turn them into
counter-reading trampolines at startup. A `foo_begin`/`foo_end` pair is one
region; `foo_begin..foo_end` is the target.

### 5. Attribute

```sh
perf benchmark a.out:fizz_buzz -m latency -e 'topdown-*'
```

Read the four level-1 slots (they sum to ~100% of slots) and drill into the
one that dominates. Cross-check with raw counters — top-down says *which*,
counters say *how much*. The slots are a documented alias, not a hardware
guarantee: on a CPU whose PMU does not export them this fails with `unknown
event 'topdown-retiring'` rather than reporting zeros, so confirm with
`perf list | grep topdown` first and fall back to the raw counters if the host
has no top-down PMU:

| Signal | Meaning |
| --- | --- |
| IPC ≈ 0.3, latency ≫ throughput | dependency chain, one load-use or FP latency |
| IPC ≈ 1 | one dependent chain per cycle |
| `branch-misses` ≫ 0 per branch | unpredictable control flow; try inlining/order |
| `cache-misses` high and `dcache=cold ≫ hot` | working set > L1; shrink or block it |
| `L1-dcache-load-misses` ≫ `LLC-load-misses` | L1 capacity/conflict, not DRAM |
| big `cold` vs `hot` gap with high `retiring` | the core retires fast, the load is the bound |
| `dtlb=cold` gap ≫ `dtlb=hot` gap | page-walk bound; fewer/huger pages |
| top-down `fe-bound` + small `itlb=cold` gap | decoder/branch-density, not TLB |

### 6. Verify the fix

Change the code, re-run the identical command, and let the statistics decide:

```sh
perf benchmark a.old:fizz_buzz -n old --mode latency --event cycles -o data/
perf benchmark a.new:fizz_buzz -n new --mode latency --event cycles -o data/
perf compare -b old -- data/
```

`perf compare` runs a two-sided z-test on the arithmetic mean and on the
geometric mean and requires both (`p = max(p_mean, p_gmean)`) to reject at
`--alpha` (default 0.05) before a change is `significant`. That is what keeps
two runs of the *same* binary from showing up as a win. The normal
approximation needs at least 30 samples per side, so smaller groups are not
compared. Zero deltas are kept, and when every run is below overhead the
fallback reports the overhead-corrected minimum clamped to zero with a
warning. With no `-e` it
compares every event *per operation* (`cycles/operations`), never the raw
totals: two runs do a different number of operations, so their counters are
never comparable.

## Reading `perf benchmark` output

Columns are the identity of the run followed by what it was measured with:

```
file  name  mode  operations  config.*  data.*  cycles  instructions
```

- `mode`: `latency` (one row per measured call, each the min over runs) or
  `throughput` (one row for the whole loop: the min over runs). Both are
  wanted; they answer different questions.
- `operations`: denominator for every ratio (`cycles/operations`); 1 for
  latency, the auto-calibrated trip count for throughput.
- A `null` counter means it could not be read — never read it as `0`.
- `config.backend.<name>.*` records the resolved backend and its parameters,
  so a row can be replayed exactly.

`perf view` shows `time,file,name,mode,operations,duration_time/operations` by
default — the per-operation cost, not the raw counter — and aggregates
(`-s min,median,p10,p50,p90,p99,max`); `-e` picks other columns or
expressions, `-g '' -s ''` gives raw rows. `perf plot` charts
`<event>/operations` by default (ecdf), so plot the *per-operation* cost, not
the raw counter.

## Live tracking of a real binary

```sh
perf benchmark a.out --list                     # what is trackable
perf profile --list -- ./a.out                  # same, for live tracking
perf profile -f hot -e cycles,branch-misses -- ./a.out --work 100
perf profile -f fizz_buzz -e cycles -o profile.json -- ./a.out
```

`perf profile` patches addresses at startup (ptrace + `rdpmc` trampolines); `--filter`
is required and names the symbols to track; the
binary on disk is untouched. Detours reserve the 128-byte red zone and
function returns use per-thread shadow stacks, so overlapping threads return
to the right caller. Repeat `--filter` to track more than one symbol.
`perf benchmark a.out --list` (or `perf analyze a.out --list`,
`perf profile --list -- ./a.out`) is the way to list what is trackable.

## Per-instruction view of a target

```sh
perf analyze a.out:fizz_buzz                                  # every state
perf analyze a.out:fizz_buzz -- perf.data                     # + per-ip events
perf analyze a.out:fizz_buzz --filter 'latency > 4'           # only those instructions
perf analyze a.out:fizz_buzz --filter '15 in `data.rdi`'      # only that state
perf analyze a.out:fizz_buzz -e assembly,latency              # pick the columns
perf analyze a.out:fizz_buzz -e 'index,assembly,data*'         # or the state columns
perf analyze a.out:fizz_buzz -e instructions/cycles -- perf.data
perf analyze foo.s:foo..bar                                   # an assembly source
perf analyze a.out:fizz_buzz --data.rdi=15                    # a concrete state
perf analyze a.out:fizz_buzz --setup a.out:init --teardown a.out:fini # with set-up
perf analyze a.out:fizz_buzz -e assembly | llvm-mca -mcpu=alderlake  # or into llvm-mca
```

`perf analyze` never runs anything. `index` numbers the instructions
`0, 1, 2, ...`; it is a column like any other — in the default selection, and
printed first when it is selected, so a run can be counted directly. The
target is explored symbolically, so all states are analyzed: the rows are
every instruction the target disassembles to (the whole function or region,
followed through its branches), and the columns are what the explored
states held — `data.<reg>` for every register a state pins (the arguments and
whatever `--data` constrains, the very values `perf benchmark` measures) and
`data.<addr>` for an address a state reads or writes. Every `data.*` cell is
a list of what the states held (`[15]` when they agree, `[0, 1, 1073741825]`
when they do not), and the registers the exploration only had to pin to keep
going are not data and are left out. `--filter` takes a pandas query over any
column (`size`, `latency`, `data.rdi`, ...), so instructions can be selected
by the state they run with — `in` is how you test a state column
(`15 in \`data.rdi\``); `-e`/`--event` picks the columns to show — any of them,
with `*` expanding a pattern (`-e data*`, `-e '*'`) and an expression allowed
(`-e instructions/cycles` over the per-ip counters joined from `--`); a name
that is not in the result is an error, and the default is
`file,name,index,address,encoding,size,latency,throughput,assembly,data*`.
Asked for `assembly` alone, the table's header is `.intel_syntax`, so it pipes
straight into llvm-mca: `perf analyze a.out:fizz_buzz -e assembly | llvm-mca
-mcpu=alderlake`.
`--data`
pins the explored state to concrete values (same meaning as
`perf benchmark --data`), and `--setup`/`--teardown` run around the target,
exactly as in `perf benchmark`.

The result is one table: data given after `--` is joined by `ip`, so a
`perf.data` turns the table into "cycles per instruction"; runs without `ip`
(`fizz_buzz.json`, `profile.json`) only contribute their `file,name`
identity.

Only the target's own instructions are listed: the harness's timing reads,
cache/TLB steering, register priming and call sequence are never attributed to
the target. Use `perf analyze` to attribute a measured hot spot to
instructions; use `perf benchmark` for the per-operation cost of one.

## Optimize the binary after the measurement

`perf optimize` applies measured-data-guided, address-preserving passes. It is
not a compiler: every pass keeps each function body the same size (shorter
encodings are padded with multi-byte NOPs) so references outside the rewritten
`.text` stay valid; `layout`/`align` permute whole functions and fix up
rel32 branches, RIP-relative memory, RELA addends, `.eh_frame` FDEs, the
`.eh_frame_hdr` lookup table, symbol tables and the ELF entry point. Functions
with indirect jumps or exception tables stay in the original layout, and
non-PIE binaries skip `layout`/`align`; a layout that cannot be fixed up falls
back to the original order while keeping peephole/branch. Only statically
reachable instructions are rewritten, so embedded data and padding are left
alone. The default output is `EXEC.opt<EXT>` (`a.out` -> `a.opt.out`); `layout`
still rewrites the whole `.text`, and the report lists the functions touched.

```sh
perf optimize a.out -- data/                           # writes a.opt.out
perf optimize a.out --output b.out -- data/            # measured hot layout
perf optimize a.out --output b.out --opt peephole,branch
perf optimize a.out --output b.out --opt layout --config.layout.align=64 -- data/
perf optimize a.out --output b.out --opt layout -- data/profile.json
perf optimize a.out --list -- data/                       # hotness per function
perf optimize a.out --list --json -- data/                # the same, as JSON
```

Passes (`--opt`, repeatable/comma-separated; default all):

- `peephole`: safe same-size simplifications: `mov r,r` (64-bit exact),
  `xchg r,r` (exact), `lea r,[r]`/`lea r,[r*1]` (exact, 64-bit),
  `shift r,0`, `movabs` shrinking (incl. `r8`-`r15`), and optimal NOPs; only
  statically reachable instructions are rewritten.
- `branch`: jump-to-next removal, transitive `jmp`/`jcc` threading
  (cycle-safe), and conditional inversion over a following `jmp`
  (self-loop safe).
- `layout`: reorder functions hot-first from benchmark/profile/`perf.data`/
  analyze data, with entry alignment.
- `align`: extra hot-function alignment (implies `layout`); powers of two up to
  4096.

Alignment and layout order live in `--config` (`config.layout.align`,
`config.layout.order`, also settable in a JSON file merged over the defaults);
re-run the benchmark after optimizing, because an optimization that does not
clear `perf compare` is not a win. `--list` shows the same hotness the layout
pass uses, and `--json` emits the report (or the listed functions) for scripting.

## System perf, and when to use it instead

The perf-labs tools are for *isolated* cost. For whole-program behaviour,
profile with system perf. One `perf` dispatches both: it runs `perf-<command>`
when that script exists and otherwise falls through to linux-perf, so `perf
stat`, `perf record`, `perf report`, `perf annotate`, `perf script`, `perf
c2c`, `perf mem`, `perf lock`, `perf sched` and `perf probe` all work next to
`perf benchmark` and `perf analyze`.

```sh
taskset -c 4 perf stat -e cycles,instructions,cache-misses,branch-misses ./a.out
perf record -g -F 999 -e cycles:u -o perf.data -- ./a.out
perf report --stdio -g graph,4000 --sort symbol
perf annotate --stdio -s symbol.dso
perf c2c record -g -o c2c.data -- ./a.out
```

`perf system tune` ships two distinct predefined profiles next to `default`.
`latency` pins the `performance` governor and maximum frequency, disables
turbo/boost and SMT, turns off NMI/watchdog, NUMA balancing, THP/KSM/timer
migration, autogroup and RT throttling (`sched_rt_runtime_us=-1`), tunes IRQ
affinity away from isolated CPUs, disables IRQ balancing and checks isolation,
exposes kernel symbols (`kptr_restrict=0`, `perf_event_paranoid=-1`), and sets
`swappiness=0` with a 1/s perf sample rate. `throughput` keeps turbo/boost and
SMT on, enables THP and defrag, disables IRQ balancing and watchdogs, and uses
`swappiness=10`. Both cover the writable pyperf tuning knobs (perf sample
rate, ASLR, CPU frequency/governor, turbo/boost, IRQ balancing incl. per-IRQ
affinity) plus `turbo_msr`, `kptr_restrict`, scheduler (`sched_autogroup`,
`sched_rt_runtime_us`), `watchdog`, power-supply and
`isolcpus`/`rcu_nocbs`/`nohz_full` checks. Use
`--dry-run` first, `--profile.KEY=VALUE` for one-off overrides, and the
`default` profile to restore.

## Pitfalls seen in real reviews

- **Latency measured as throughput** (or the reverse). Compare like modes.
- **Both modes in one run, reading one column**: `perf benchmark 'imul eax, 42' -e cycles | perf view -s p99`
  mixes the `latency` and `throughput` rows. Pass `--mode latency` explicitly.
- **Per-call overhead mistaken for loop cost**: always look at
  `latency` vs `throughput` before blaming the operation.
- **Percentiles from one sample** are noise. `samples` is 100 by default
  (`--config.samples=N`); a row that wants more is a row to re-run.
  `perf compare` needs at least 30 samples per side for its z-test; smaller
  groups are not compared.
- **Zero is a measurement**. A zero counter delta is kept, not dropped. When
  every run is below overhead the fallback reports zero with a warning, never
  the raw count.
- **Sweeping two axes and reading the corner**: `--config.dcache=hot,cold` is
  fine; `--config.dcache.L1d=[{hit_rate:100},{hit_rate:50}]` with `--data` sweeps is a
  factorial explosion. One axis at a time.
- **Cache state you did not choose**: default sweeps `branch,dcache,dtlb,icache,itlb,code`, so
  the default row is one point of a sweep, not "the" number. Say which tier.
- **Attributing harness cost to the target**: it is subtracted differentially,
  but only the target's own instructions are ever listed, so a long
  `call`/`ret` or a big prologue still shows up as the target's cost. Compare
  against a neighbouring label before blaming a function boundary.
- **`icache=cold` looking like `icache=hot`**: x86-64 has no user-mode way to
  flush the instruction cache (`clflushopt` only reaches the data hierarchy),
  so `icache` steers the code's data-hierarchy line and its instruction
  translation only. Do not read an `icache` row as "the code is out of L1i";
  it is the `itlb` row and front-end (`topdown-fe-bound`) that say anything
  about the instruction stream.
- **Region spanning labels that moved**: `--list` warns when `foo_end`
  precedes `foo_begin`; the span between them is still measured, but the
  region is not what the source suggests.
- **Turbos/scaling governor moving the baseline** between two runs: re-measure
  the baseline in the same session as the candidate.
- **Statistical noise called a speedup**: `perf compare`, not a diff of medians.

## Reporting

Report like an engineer who wants to be believed:

```
fizz_buzz(n=1e6), latency mode, pinned cpu 4, 26732 loop trips
baseline   10.00 cycles/op   p10 9  p50 10  p90 11
optimized   6.00 cycles/op   p10 5  p50 6  p90 7
perf compare: -40.0% [-41.2, -38.6] p<1e-4 -> significant
topdown: retiring 62% -> 71% (bad-spec 21% -> 9%): the mispredicts are gone
```

Include: the exact commands, the unit, sample count and spread, the
significance verdict, the top-down/attribution evidence, and what is still
unexplained. If a question cannot be answered with the counters available, say
so and name the experiment that would answer it.

## Working on this repository

- `src/perf/core.py` — event resolution, `perf_event_open`, RDPMC, affinity/
  priority/NUMA guards. `src/perf/bench.py` — symbolic exploration, harness
  JIT, cache/TLB/branch steering, run calibration, data-page mapping.
  `src/perf/mca.py` — `perf analyze` (numbered instructions, one row per
  explored state, list-valued `data.*`). `src/perf/exec.py` — ELF loading,
  relocation, `to_object`. `metadata.perf_labels`/`metadata.asm_labels` in `info.py` map a
  binary's labels and an assembly source's labels (to position and size),
  which `bench.py` turns into a snippet behind `perf benchmark foo.s:foo..bar`.
  `src/perf/opt.py` — `perf optimize`: hotness aggregation, ELF rewrite and
  fixups (rel32, RIP-relative, RELA, `.eh_frame`/`.eh_frame_hdr`, symbols,
  entry point). `src/perf/arch/x86_64.py` — harness templates, counter reads,
  TSC/CPUID frequency, eviction/priming asm, `page_runs` (the page clusters a
  TLB `mprotect` covers and `bench.py` maps up front), and the arch-specific
  optimizer primitives (NOPs, peepholes, branch folding/inversion, text
  verification). `perf.arch` resolves both names and projects.
  `src/perf/prof.py` — ptrace detours, ring buffer, shadow stack.
  `src/perf/info.py` — `perf system` (cpu topology, hostname, metadata.perf_labels,
  metadata.functions, `cpuinfo.format_hz`). `src/perf/tune.py` — `perf system tune`
  profiles and `/sys`/`/proc`/`/dev` writes, service/IRQ/power/isolation
  checks. `src/perf/data.py` — result-frame schema,
  `perf.data` parsing via `samples` (`samples.nest/spread/parse/metrics/is_record`),
  `query` with `in` over list columns. `src/perf/comp.py` — the CLT test.
  `src/perf/plot.py` — charts, and the sixel backend.
- Each command is one script named `perf-<command>`, which system `perf`
  dispatches to from `perf <command>`, so `perf benchmark` runs
  `bin/perf-benchmark`. A script is self-contained: its own parser, its own
  command, and only the helpers it uses (`.perfconfig` for its own section,
  the result loading/formatting its input needs). There is no shared CLI
  module — keep it that way, and keep a command from reaching into another.
- A target is one `CODE` argument on the CLI: `FILE:SYMBOL` for a symbol or a
  `begin..end` region, `FILE:LABEL` for an assembly source, and a raw snippet
  when there is no `FILE:`. `perf benchmark FILE --list`
  (`perf analyze FILE --list`, `perf profile --list -- FILE`) lists a file's
  symbols (with an optional `:PATTERN`/`-f PATTERN` regex filter, plus `-e`
  column selection), and a symbol that is not found prints them before it exits. In
  Python, `perf.benchmark` and `perf.analyze` take a binary symbol as `symbol=`
  (a string like `"a.out:fizz_buzz"` or a `[file, symbol]` pair), a binary
  region as `region=` (a string like `"a.out:hot_begin..hot_end"` or a
  `[file, (begin, end)]` pair), and a raw snippet as `asm=`
  (e.g. `asm="mov eax, 42"`); `perf.profile` takes the executable as `cmd=`
  (e.g. `cmd=["./a.out"]`) and the track points as `select=`
  (e.g. `select=["hot"]` or `select=[("hot", "cold")]`);
  `perf.optimize` takes the input binary as `exec=` (e.g. `exec="a.out"`);
  `perf.to_object` takes a binary symbol as
  `symbol=` or a binary region as `region=`. There is
  no `target=` form for `benchmark`, `analyze`, `profile` or `optimize`
  (and no `select=`/`filter=` form for `optimize`, no `file=`/`code=` form for
  `benchmark` or `analyze`); do not add one back. Public API is
  `arch`, `benchmark`, `profile`, `analyze`, `compare`, `optimize`, `plot`,
  `samples`, `to_json`, `to_object`, `cpuinfo`, `metadata`, `tune` (`metadata.perf_labels`,
  `metadata.asm_labels`, `metadata.functions` via `metadata`).
  `benchmark(mode=)` defaults to both modes like the CLI; `compare`/`plot`
  take `filter=` like the CLI `-f` flag.
- Result-frame identity columns are `data._IDENTITY_COLUMNS`
  (`file`, `name`, `mode`); columns that are never metrics are
  `data._NON_METRIC_COLUMNS`. Add new columns there, not at every use site.
- Only the license header is kept: no comments and no docstrings anywhere in
  `src`, `tests` or `bin/perf-*`. Name things well instead. Delete dead
  helpers instead of leaving them (`_thread_count`, unused imports,
  shadowed imports, per-test rebuilds of the same binary).
- Each command stays self-contained and DRY only within its own file.
  There is no shared CLI module and no `tests/helpers.py`. Duplicate
  fixture names across files stay separate when their schemas differ.
- Performance is caching plus pruning. Reuse `arch.assemble`,
  `arch.disassembler`, `core.demangle`, `info._FUNCTIONS_MEMO`,
  `bench._PROJECTS`, and `exec.resolve_exec`; do not construct a new
  `keystone.Ks`, `capstone.Cs`, or `angr.Project` per call. Share one
  `gcc` build per source (`TestBinary.setUpClass`,
  `_DEBUG_EXE_CACHE`) and mock the slow paths (`_bench`, `explore`,
  `PerfCounter`) unless the test is the end-to-end one.
- Coverage is every public entry plus its corners: empty/`None`/bad
  input, unknown event/column, bad filter, `nan`/torn data, unmatched
  baseline, non-PIE, missing file, and `alpha`/`buffer-size` validation.
  CLI failures exit with `SystemExit`.
  End-to-end CLI contracts (tables, JSON envelopes, failure messages) live
  as LLVM lit + FileCheck tests in `tests/x86_64/`; `tests/test_*.py` keeps
  the unit scope (parsing, formatting, statistics, mocks). When a behavior
  is easier asserted on real command output, move it to lit rather than
  duplicating it with mocks. Tests needing real hardware counters carry
  `REQUIRES: pmu` and assert only that values exist, never their magnitude.
- Development loop: `pytest`, `lit -v -j2 --timeout 600 tests/x86_64`,
  `ruff check src tests bin`,
  `ruff format --check src tests bin`, and papermill over `studies/**/*.ipynb`
  (see `.github/workflows/linux.yml`). ruff is pinned to a release series in
  `pyproject.toml`; a ruff upgrade and the reformat it causes go in one commit. `tests/README.md` explains what is
  tested and what the suite enforces; `src/README.md` has worked, verified
  invocations of every command if a flag needs showing rather than describing.
  Keep the ```py blocks in every `*.md` ruff-format clean by hand.
- `studies/x86_64/**` are the worked notebook analyses (one per level-1
  top-down slot); treat them as the reference for expected numbers and method.
  `studies/README.md` explains the method. `lib/README.md` documents the
  `PERF_LABEL` annotations.
