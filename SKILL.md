---
name: perf-labs-perf
description: Evidence-driven performance work on x86-64 Linux with the perf-labs/perf toolkit. Use when profiling or optimizing binaries, benchmarking a function/region/snippet, tuning a host for stable measurements, validating a host with known-answer checks, attributing a slowdown with top-down counters, or deciding whether a change is a real speedup. Runs observe -> recall -> hypothesize -> predict -> experiment -> evaluate -> remember.
---

# Performance intelligence

You are a performance engineer running a closed, evidence-driven loop. Your
value is not command recall; it is answering three questions without inventing
either the hypothesis or the number: what do we already know, what would
falsify this, and what did we actually measure?

```text
OBSERVE ──► RECALL ──► HYPOTHESIZE ──► PREDICT
   ▲                                      │
   │                                      ▼
DECIDE ◄── REMEMBER ◄── EVALUATE ◄── EXPERIMENT
```

The loop is a state machine, not a menu. You may go back one step when an
experiment is invalid; you may not skip RECALL or REMEMBER.

| Step        | Question                                                   | Artifact                                                                       |
| ----------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------ |
| OBSERVE     | What is the host, target, and current cost?                | `perf system info/tune/test`, `perf benchmark`, `perf profile`, `perf analyze` |
| RECALL      | What did we already measure on this state?                 | experiment ledger                                                              |
| HYPOTHESIZE | What single mechanism explains the dominant bound?         | top-down slots, raw counters, disassembly                                      |
| PREDICT     | Which metric moves, in which direction, by what threshold? | reasoning record                                                               |
| EXPERIMENT  | What one variable changes?                                 | one benchmark/axis or one code change                                          |
| EVALUATE    | Is the delta real and in the predicted direction?          | `perf compare`                                                                 |
| REMEMBER    | What is the verdict and where is the evidence?             | append-only ledger record                                                      |
| DECIDE      | Accept, reject, or inconclusive; what next?                | reasoning record and next bound                                                |

Evidence hierarchy; never promote a weaker item to a stronger one:

1. a measured artifact (`benchmark`, `profile`, `compare`) and its path;
2. a measured counter or top-down slot in that artifact;
3. a modelled instruction property (`perf analyze` latency/throughput), labelled
   as a model;
4. a hypothesis or expectation, which is not evidence.

## Reasoning invariants

1. **Recall before hypothesis.** Query the ledger before proposing a change.
   A known accepted result is reused; a known rejected result is not re-tested
   without new evidence.
2. **Cite the field.** Every claim names its artifact and field, for example
   `topdown-be-bound=61.2` or `disassembly_snippet[7].latency=5`. The
   `perf compare` record is the verdict artifact; it adds no counter.
3. **No run without a prediction.** State the expected metric, direction, and
   threshold before measuring. "It should be faster" is not a prediction.
4. **No number without unit, denominator, and artifact.** Report
   `cycles/operations`, `ns/operation`, `IPC`, `p50`/`p99` — and the file the
   number came from. A bare `cycles` total is never evidence.
5. **No conclusion without `perf compare`.** Two tables side by side are not a
   result. The shipped CLT z-test decides significance.
6. **No accepted change without a mechanism.** A significant delta plus a
   top-down/per-instruction explanation is a result; a delta alone is a
   coincidence candidate.
7. **Never invent data.** If a value is `null`, it could not be read — it is
   not `0`. Model output is labelled as a model, never as a measurement.
8. **Top-down before line-level.** Establish the dominant level-1 slot and its
   raw-counter evidence before proposing an instruction edit. If the PMU does
   not export top-down events, say so and use raw counters; do not substitute a
   guess for the missing bound.

## 1. Preflight: control the host

```sh
uname -r                                      # 6.x+ expected
perf system info --cpu --output env.json      # topology, TSC frequency, caches
perf system tune --status                     # read-only: values tune would set
perf system tune --status --profile=latency   # a specific profile
perf system tune --dry-run                    # plan; does not write
perf system tune --dry-run --profile=latency
perf system tune --profile=latency            # apply (may re-exec through sudo)
perf system test                              # known-answer checks
for f in /sys/devices/{cpu,cpu_core,cpu_atom}/rdpmc; do
  [ -e "$f" ] && cat "$f"
done
```

| Check              | Meaning                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------- |
| `perf system test` | Host clock, decode width, ROB/BTB/TLB, instruction and memory hierarchy, branch accuracy |
| `rdpmc` contents   | `1`/`2` enable user-space counter reads; `0` means hardware events will fail             |
| `--status`         | Read-only target values; `unavailable`/`unsupported`/`denied`/`failed` explain gaps      |
| `--dry-run`        | Exact `before`/`after` plan without writing or `sudo`                                    |

Run `perf system test` once per host/session. A failing check is an environment
problem, not a target hypothesis: fix it and re-run. Use one tuning profile for
the whole experiment; never change governor, frequency, SMT, THP, affinity, or
kernel between baseline and candidate.

`duration_time` uses `rdtsc` and works without `rdpmc`; every other event needs
`rdpmc`. `.github/workflows/perf.sh` is the authoritative smoke suite for exact
flags; `perf <command> -h` (or `perf-<command> --help`) and `bin/README.md` are
the reference.

## 2. Controlled experiments

### 2.1 Identity

There is no single `PerfState` file. Build an experiment record from the actual
artifacts and keep each artifact next to its command line.

| Field                    | Source                                                               |
| ------------------------ | -------------------------------------------------------------------- |
| target, mode, operations | benchmark envelope (`file`, `name`, `mode`, `operations`)            |
| binary identity          | `file` label (`a.out@b04f8237`); also hash the ELF yourself          |
| inputs and controls      | `config.*` and `data.*` in the benchmark envelope                    |
| environment              | `info.cpu`, `info.hostname`, `perf system info/tune --status` output |
| top-down                 | `perf benchmark -e 'topdown-*'` columns                              |
| disassembly              | `perf analyze --output` records                                      |
| hotness                  | `perf optimize --list --output` records                              |
| verdict                  | `perf compare --output` records                                      |

`null`/missing is unknown, never zero. A result belongs to the exact binary,
host, tune profile, inputs, and target region that produced it; a new build or
host is a new state.

### 2.2 One variable per experiment

| Kind                 | Allowed change                                                                                            | Verdict                                 |
| -------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| Accepted config axis | `--config.dcache={hot,cold}`, `--config.icache={hot,cold}`, `--config.branch={predictable,unpredictable}` | May be `accept`                         |
| Observation          | `dtlb`, `itlb`, alignment, layout, input data, extra cache tiers                                          | Descriptive; `inconclusive` as a change |
| Intervention         | One code or binary change                                                                                 | May be `accept` with a mechanism        |

Do not combine a code change with a config change. Do not sweep two axes and
read the corner. Do not change the workload to flatter the candidate. Baseline
and candidate must be identical in every other respect:

| Control        | How                                                                       |
| -------------- | ------------------------------------------------------------------------- |
| Mode / events  | Same `-m` and `-e` groups on both sides                                   |
| Target         | Same symbol, region, or snippet; same binary for config experiments       |
| Samples / runs | `--config.samples`, `--config.backend.<name>.runs`, `--config.iterations` |
| Backend        | `--backend loop` or `unroll`, same on both sides                          |
| Cache / TLB    | Pin `dtlb`, `itlb`, and the non-varied cache tier to `hot`                |
| Branch         | `--config.branch=predictable` unless branch is the axis                   |
| Placement      | Same `--config.thread.affinity`, `numa`, `priority`                       |
| Host           | Same tune profile, governor, turbo, SMT, kernel, session                  |

`--config.thread.affinity=0` is a valid pin on most hosts; choose a PMU-capable
CPU and keep it. On hybrid CPUs, top-down events may exist on only one core type.

### 2.3 Canonical A/B

Write explicit JSON files; a directory output also works, using the default
`{time}-{command}-{target}-{id}.json` template.

```sh
EXP=".perf/experiments/$(date -u +%Y%m%dT%H%M%S)-fizz"
mkdir -p "$EXP"

PIN=(--config.backend=loop --config.backend.loop.runs=10
     --config.branch=predictable --config.dcache=hot --config.dtlb=hot
     --config.icache=hot --config.itlb=hot
     --config.func.align=16 --config.func.order=as-is
     --config.thread.affinity=0)

perf system info --cpu --output "$EXP/env.json"
perf system tune --status --output "$EXP/status.json"
perf system test --output "$EXP/selftest.json"

perf benchmark a.old:fizz_buzz -n old -m latency -e cycles,instructions \
  "${PIN[@]}" -o "$EXP/old.json"
perf benchmark a.new:fizz_buzz -n new -m latency -e cycles,instructions \
  "${PIN[@]}" -o "$EXP/new.json"
perf compare -b old --output "$EXP/compare.json" -- "$EXP/old.json" "$EXP/new.json"
```

For a config experiment, keep the binary identical and change exactly one
whitelisted axis in `PIN`. For a code experiment, keep `PIN` identical and change
only the binary. When loop-trip count affects the workload, pin
`--config.iterations` as well.

### 2.4 Evaluate with the shipped CLT test

`perf compare` is the only significance test. Do not substitute a t-test,
bootstrap, or a diff of medians. Fix `--alpha` before the run.

| Compare result                                            | Verdict                                           |
| --------------------------------------------------------- | ------------------------------------------------- |
| `significant: true`, desired direction, mechanism moved   | candidate for `accept`                            |
| `significant: false` or `diff: "-"`                       | `inconclusive`; report "no measurable difference" |
| `significant: true`, opposite direction                   | `reject`                                          |
| `p_value` null/NaN, fewer than 30 usable samples per side | `inconclusive`; collect more data                 |
| environment or binary changed                             | invalid; re-baseline and record `inconclusive`    |

The test uses two-sided z-tests on the arithmetic mean and geometric mean. When
both p-values are finite, `p = max(p_mean, p_gmean)` must be below `--alpha`
(default `0.05`). If the log test is undefined, the finite test alone decides.
A comparison with too few samples or no shared mode is an `inconclusive` row,
not a rejection.

`diff` is the signed percentage change reported by the test; negative is faster
for cycles/duration, positive is faster for IPC/throughput. `ci_low` and
`ci_high` are in the same percentage units. Use `significant` for the decision,
not the interval alone. Latency rows are per call, so a latency A/B needs at
least 30 loop trips per side; throughput needs at least 30 samples.

### 2.5 Memory ledger

Keep one authoritative store per project. A JSONL file is enough:

```text
.perf/
  experiments/<id>/       # env, status, old, new, topdown, analyze, compare
  experiments.jsonl       # one verdict record per line
```

```json
{
  "id": "20261112T002500-fizz-H1",
  "timestamp": "2026-11-12T00:25:00Z",
  "target": "a.out:fizz_buzz",
  "mode": "latency",
  "metric": "cycles/operations",
  "binary": "a.out@b04f8237",
  "config": {"dcache": "cold", "branch": "predictable"},
  "data": {"rdi": 1000000},
  "hypothesis_id": "H-1",
  "result_delta": -12.4,
  "p_value": 0.0001,
  "verdict": "accept",
  "command": "perf benchmark ...; perf compare ...",
  "artifacts": ["experiments/20261112T002500-fizz-H1/compare.json"],
  "notes": "mispredict removed"
}
```

`verdict` is `accept`, `reject`, or `inconclusive`. `result_delta` and `p_value`
are `null` for an exploratory or incomparable run. Keep the raw envelopes as
the source of truth; the ledger points at them.

Recall before a new hypothesis:

```sh
jq -c --arg target "$TARGET" \
  'select(.target == $target)' .perf/experiments.jsonl
```

Append after every run; never rewrite an earlier record:

```sh
printf '%s\n' "$RECORD_JSON" >> .perf/experiments.jsonl
```

| Memory finding                                   | Action                                                     |
| ------------------------------------------------ | ---------------------------------------------------------- |
| `accept`, same binary/config/data/host           | Reuse the conclusion; do not re-run                        |
| `reject`, same state and hypothesis              | Do not retry the same command; change the mechanism        |
| `inconclusive`                                   | Not evidence; retry only after fixing the cause in `notes` |
| same bottleneck/host, different binary or inputs | Treat as a prior; revalidate on the current state          |
| no match                                         | Form a new hypothesis; do not invent a past result         |

### 2.6 Reasoning record

Write this before the experiment; fill in evidence after verification:

```text
Recall:         <ledger records for this state, or "none for this state">
Observation:    <artifact field> = <value> (<spread>) under <host>
Interpretation: <bound/mechanism> because <topdown/counter/analyze evidence>
Hypothesis:     H-<n>: If <single change>, then <metric> should <direction>
                by <threshold>, while <control metric> stays within <range>
Falsifier:      <observation that would reject the hypothesis>
Experiment:     <one axis or one code change>, <samples>, <exact command>
Evidence:       perf compare: diff=<...>, CI=[<...>, <...>],
                p=<...>, significant=<...>
Mechanism:      <topdown/analyze confirmation or refutation>
Verdict:        accept | reject | inconclusive
Artifacts:      <env.json, old.json, new.json, compare.json, ...>
Memory:         <experiment id / ledger line>
```

If you cannot fill a line, you are not ready to run the experiment.

### 2.7 Stop rules

- Stop when the goal is met, the hypotheses for the dominant bound are
  exhausted, or the counters needed to discriminate the next hypothesis do not
  exist.
- Budget about six experiments per bottleneck. Two consecutive `inconclusive`
  results mean the state or measurement is incomplete, not that the code cannot
  be optimized.
- Report a PMU capability gap and the experiment that would answer it; do not
  invent an event.
- Restore affinity, priority, NUMA binding, and the pre-tune snapshot
  (`perf system tune --reset`) if you tuned. Remove scratch binaries.

## 3. Attribute with top-down and per-instruction evidence

```sh
perf benchmark a.out:func -m latency -e 'topdown-*' \
  --config.backend=loop --config.backend.loop.runs=10 \
  --config.branch=predictable --config.dcache=hot --config.dtlb=hot \
  --config.icache=hot --config.itlb=hot \
  --config.func.align=16 --config.func.order=as-is \
  --config.thread.affinity=0
```

1. Measure the four level-1 slots; record all four.
2. If the PMU does not export them (`perf list | grep topdown`), the command
   fails with `unknown event`, not zeros. Fall back to raw counters and leave
   top-down fields `null`.
3. Name the dominant slot and the hypothesis it suggests.
4. Measure the raw counters that discriminate it.
5. Use `perf analyze` to find the instructions.
6. Turn the mechanism into one controlled experiment; predict the slot that
   should move.

The window includes the harness around the measured region; treat the split as
whole-loop attribution, not as one instruction in isolation.

| Signal                                          | Meaning                                     |
| ----------------------------------------------- | ------------------------------------------- |
| IPC far below machine width                     | dependency chain or port pressure           |
| `branch-misses` high per branch                 | unpredictable control flow; bad speculation |
| `L1-dcache-load-misses` >> `LLC-load-misses`    | L1 capacity/conflict, not DRAM              |
| `dcache=cold` >> `dcache=hot`                   | working set does not fit L1                 |
| `dtlb=cold` gap >> `dtlb=hot` gap               | page-walk bound; fewer or huge pages        |
| `topdown-fe-bound` high with a small `itlb` gap | decoder/branch density, not TLB             |
| `topdown-be-bound` high                         | memory/back-end bound                       |
| `topdown-retiring` high with low IPC            | issue width or a dependency chain           |

| Dominant slot                           | First observation                              | Experiment                                               | Accept when                                               |
| --------------------------------------- | ---------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| `bad-spec`                              | `branch-misses`; predictable vs unpredictable  | `--config.branch=predictable,unpredictable`              | unpredictable is significantly slower and bad-spec rises  |
| `fe-bound` with an icache gap           | `icache=hot` vs `cold`, iTLB, code size        | `--config.icache=hot,cold`                               | cold is significantly slower and fe-bound rises           |
| `fe-bound` with an alignment/layout gap | code alignment, function order, hotness        | one code change (`perf optimize ... --opt layout,align`) | change clears `perf compare` and fe-bound/icache improves |
| `be-bound` with a dcache gap            | L1d misses vs LLC misses                       | `--config.dcache=hot,cold`                               | cold is significantly slower and L1d misses rise          |
| `be-bound` with a dtlb gap              | `dtlb=hot` vs `cold`, dTLB misses              | observation only; test a code change that reduces pages  | change clears `perf compare` and dTLB misses fall         |
| `be-bound` without a cache/TLB gap      | per-instruction latency, IPC                   | `perf analyze`; one code change                          | change clears `perf compare` and IPC/field moves          |
| `retiring` with low IPC                 | latency vs throughput, per-instruction latency | one code change; `perf analyze`                          | change clears `perf compare` and IPC rises                |
| `retiring` near machine width           | no dominant slot                               | stop; do not optimize noise                              | no experiment is justified                                |
| `null` / unsupported                    | raw counters only                              | do not guess a bottleneck                                | only a measured code change is accepted                   |

The worked notebooks are the reference for method, not a source of numbers:

| Notebook                                                                | Shows                                         |
| ----------------------------------------------------------------------- | --------------------------------------------- |
| [retiring/ipc](studies/x86_64/retiring/ipc.ipynb)                       | IPC and dispatch width from a NOP-width sweep |
| [bad_speculation/branch](studies/x86_64/bad_speculation/branch.ipynb)   | Predictable vs unpredictable branches         |
| [bad_speculation/if_else](studies/x86_64/bad_speculation/if_else.ipynb) | Branch layout for an if/else shape            |
| [frontend_bound/icache](studies/x86_64/frontend_bound/icache.ipynb)     | Code alignment and instruction footprint      |
| [backend_bound/dcache](studies/x86_64/backend_bound/dcache.ipynb)       | Cache tiers, load latency, top-down shares    |
| [backend_bound/dsize](studies/x86_64/backend_bound/dsize.ipynb)         | `qsort` input-size scaling                    |

Re-measure on the current host; do not copy notebook numbers.

## 4. Command map

| Question                                  | Command                                                                               |
| ----------------------------------------- | ------------------------------------------------------------------------------------- |
| What is this machine?                     | `perf system info --cpu`                                                              |
| What does an op cost in isolation?        | `perf benchmark 'imul eax, 42' -m latency -e cycles,duration_time`                    |
| What does a function/region cost?         | `perf benchmark a.out:fizz_buzz -m latency -e cycles,instructions`                    |
| Latency or throughput bound?              | `-m latency` vs `-m throughput` on the same target                                    |
| Does it depend on input?                  | `--data.arg0=1` / `--data.arg0='[1,3,5]'` (observation only)                          |
| Program cost with real arguments?         | `perf benchmark /usr/bin/tree:main -m latency -- /path/to/folder`                     |
| Branch mispredicts?                       | `--config.branch=predictable` vs `unpredictable`                                      |
| Cache bound?                              | `--config.dcache=hot,warm,cool,cold`, then `-e L1-dcache-load-misses,LLC-load-misses` |
| TLB bound?                                | `--config.dtlb=hot,cold`, then `-e dTLB-load-misses,iTLB-load-misses`                 |
| Microarchitectural bound?                 | `-e 'topdown-*'`                                                                      |
| Which instructions?                       | `perf analyze a.out:fizz_buzz -- perf.data`                                           |
| Which state do they run with?             | `perf analyze a.out:fizz_buzz` (`data.*`) and `--filter`                              |
| Can the binary improve without a rebuild? | `perf optimize a.out --list -- bench.json`, then `perf optimize a.out -- bench.json`  |
| Is the host stable?                       | `perf system tune --status --profile=latency`, `--dry-run`, `perf system test`        |
| Is a change faster?                       | `perf compare -b old -- old.json new.json`                                            |
| What is the noise floor?                  | `perf view -s min,median,p10,p50,p90,p99 -- bench.json`                               |
| Re-open a result                          | `perf view -- bench.json` or `perf plot -- bench.json`                                |

List targets before naming one:

```sh
perf benchmark a.out --list
perf benchmark a.out:foo --list        # regex
perf analyze a.out --list
perf profile --list -- ./a.out
```

A target is one `CODE` argument: `FILE:SYMBOL` or `FILE:begin..end` for a binary,
`FILE:LABEL` or `FILE:begin..end` for an assembly source, or a raw snippet when
there is no `FILE:`. Put `PERF_LABEL(name)` around suspect regions
(`lib/perf/perf.h` for C/C++, `lib/perf/perf.rs`, `lib/perf/perf.zig`). Labels
emit no instructions; `foo_begin..foo_end` is one region.

### `perf profile`, `benchmark`, `analyze`, `optimize`

```sh
# live: --filter is required; repeat it for several targets
perf profile -f hot -e cycles,branch-misses -- ./a.out --work 100
perf profile -f hot -f cold -e cycles -o profile.json -- ./a.out

# isolated: latency and throughput answer different questions
perf benchmark a.out:fizz_buzz -m latency -e cycles,instructions
perf benchmark a.out:fizz_buzz -m throughput -e cycles,instructions
perf benchmark a.out:fizz_buzz -e cycles --data.arg0='[1,3,5]'

# analyze: never runs the target; counters are joined by ip
perf analyze a.out:fizz_buzz
perf analyze a.out:fizz_buzz -- perf.data
perf analyze a.out:fizz_buzz --filter 'latency > 4'
perf analyze a.out:fizz_buzz --filter '15 in `data.rdi`'
perf analyze a.out:fizz_buzz -e assembly | llvm-mca -mcpu=alderlake

# optimize: DATA is required; non-.json output is the binary
perf optimize a.out -- bench.json
perf optimize a.out --output b.out --opt peephole,branch -- bench.json
perf optimize a.out --list --output hot.json -- bench.json
```

A benchmark target must be callable on its own and a program target must be
position-independent. A non-PIE executable is refused by name. Benchmark the
surrounding function or a `foo_begin..foo_end` region instead. Program
arguments go after `--`; `--data.rdi/rsi/rdx` are refused there because the
process ABI sets them.

### System perf

The perf-labs tools measure *isolated* cost. For whole-program attribution, use
linux-perf; `perf <command>` dispatches to `perf-<command>` when the script
exists and otherwise falls through to the kernel tool.

```sh
taskset -c 0 perf stat -e cycles,instructions,cache-misses,branch-misses ./a.out
perf record -g -F 999 -e cycles:u -o perf.data -- ./a.out
perf report --stdio --sort symbol
perf annotate --stdio fizz_buzz
perf c2c record -g -o c2c.data -- ./a.out
```

## 5. Reading benchmark output

| Column                        | Meaning                                                   |
| ----------------------------- | --------------------------------------------------------- |
| `file`/`name`/`mode`          | Identity of the run                                       |
| `config.*`                    | The exact alternatives that produced the row              |
| `data.*`                      | Scalar/explored inputs; latency rows carry the slot input |
| `operations`                  | Denominator for every ratio; `1` in a latency row         |
| `cycles`, `instructions`, ... | Measured counters; `null` means unread, not zero          |
| `config.backend.<name>.*`     | Resolved backend and parameters                           |

A latency frame has one row per timed call; a throughput frame has one row per
sample, the minimum of `backend.<name>.runs` whole-loop runs. `perf view`
projects the per-operation value and aggregates it; `perf plot` charts
`<event>/operations` by default. Use `-g '' -s ''` for raw rows.

## 6. Pitfalls

| Pitfall                                  | Correction                                                                                   |
| ---------------------------------------- | -------------------------------------------------------------------------------------------- |
| Latency measured as throughput           | Compare like modes; merging populations mixes two questions                                  |
| Per-call overhead mistaken for loop cost | Inspect latency vs throughput; repeatable snippets use the unroll differential               |
| Percentiles from one sample              | Throughput takes `samples` minima; latency reports one row per trip                          |
| Zero is a measurement                    | Zero deltas are kept; negative timer noise is clamped; raw fallback only for unread counters |
| Sweeping two axes and reading the corner | One axis per experiment                                                                      |
| Cache state you did not choose           | Pin `dcache`/`dtlb`; the default config sweeps                                               |
| Attributing harness cost to the target   | Harness is subtracted differentially; a large target prologue is still target cost           |
| `icache=cold` looking like `icache=hot`  | x86-64 has no user-mode L1i flush; `icache` steers the data-hierarchy line and translation   |
| Region spanning moved labels             | `--list` warns when `foo_end` precedes `foo_begin`                                           |
| Governor drift between runs              | Re-baseline in the same session; record the environment                                      |
| Eyeballing medians                       | Use `perf compare`; a non-significant delta is not a speedup                                 |
| Reusing memory without checking identity | A past number is not a current measurement                                                   |
| Reporting a model as a measurement       | `perf analyze` latency/throughput are models                                                 |
| Target stdout in the JSON stream         | Keep `--config.external.stdout=false` when `--output` must own stdout                        |

## 7. Reporting

```text
fizz_buzz(n=1e6), latency, cpu 0, 26732 loop trips
state:        a.out@b04f8237, performance profile, kernel 6.14
topdown old:  retiring 24%  bad-spec 21%  fe-bound 11%  be-bound 44%
topdown new:  retiring 71%  bad-spec  9%  fe-bound 10%  be-bound 10%
baseline old: 10.00 cycles/op  p10 9  p50 10  p90 11  (n=26732)
candidate new: 6.00 cycles/op  p10 5  p50 6   p90 7   (n=26732)
compare:      diff -40.0% [-41.2, -38.6], p=1e-5, significant at alpha=0.05
mechanism:    the hot mispredict is gone; bad-spec 21% -> 9%
artifacts:    .perf/experiments/.../{old,new,topdown,analyze,compare}.json
verdict:      accept; ledger id H-7
```

Include the exact commands and artifact paths, the unit and denominator, sample
count and spread, the compare verdict, the top-down/analyze evidence, the
environment, and what is still unexplained. If a question cannot be answered
with the available counters, say so and name the experiment that would answer it.

## 8. Repository map

| Path                        | Contents                                                                       |
| --------------------------- | ------------------------------------------------------------------------------ |
| `src/perf/core.py`          | Event resolution, `perf_event_open`, RDPMC, affinity/priority/NUMA guards      |
| `src/perf/bench.py`         | Symbolic exploration, harness JIT, cache/TLB/branch steering, calibration      |
| `src/perf/mca.py`           | `perf analyze`: instructions, list-valued `data.*`, joins                      |
| `src/perf/exec.py`          | ELF loading, relocations, `to_object`, perf map                                |
| `src/perf/opt.py`           | Hotness aggregation, ELF rewrite, fixups                                       |
| `src/perf/arch/x86_64.py`   | Harness templates, counters, TSC/CPUID, eviction/priming, optimizer primitives |
| `src/perf/prof.py`          | ptrace detours, ring buffer, shadow stack                                      |
| `src/perf/tune.py`          | Profiles, sysfs/procfs/dev reads and writes, snapshots                         |
| `src/perf/info.py`          | CPU topology, labels, functions, metadata, formatting                          |
| `src/perf/data.py`          | Frames, `perf.data` samples, queries, output templates                         |
| `src/perf/comp.py`          | CLT z-tests                                                                    |
| `src/perf/plot.py`          | Charts and the sixel backend                                                   |
| `bin/perf-*`                | One parser per command; `perf <command>` dispatches to it                      |
| `tests/README.md`           | Test and coverage commands                                                     |
| `.github/workflows/perf.sh` | End-to-end smoke suite                                                         |

Development loop: `pytest --cov`, `lit -v -j2 --timeout 600 tests/x86_64`,
`ruff check src tests bin`, `ruff format --check src tests bin`, and papermill
over `studies/**/*.ipynb`. Keep Python code blocks in every `*.md` format-clean.
`bin/README.md` is the CLI reference; `src/README.md` is the API and internals.
