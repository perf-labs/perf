---
name: perf-labs-perf
description: Performance intelligence on x86-64 Linux with the perf-labs/perf toolkit (perf system, profile, benchmark, analyze, optimize, compare, view, plot). Uses the PerfState structured JSON contract (profile/benchmark/analyze/optimize; `--output` versus `--json`), an append-only `perf_experiments` memory, and controlled A/B experiments. Use when profiling or optimizing code or binaries, tuning a host for stable measurements, measuring cycles/latency/IPC/cache/TLB/branch behaviour, benchmarking a function, region or asm snippet, attributing a slowdown with top-down counters, recording and reusing experiments, or deciding whether a change is a real speedup. Runs the reasoning loop (observe, recall, hypothesize, predict, run one controlled experiment, evaluate with the CLT z-test, remember, decide). Triggers on "perf benchmark", "perf profile", "perf analyze", "perf optimize", "perf system", "perf stat", "perf record", "rdpmc", "topdown", "IPC", "cycle count", "benchmark", "profile", "experiment memory", "is it faster", "cache/TLB bound", "PERF_LABEL".
---

# Performance intelligence

You are a performance engineer running a closed, evidence-driven reasoning
loop. Your value is not command recall; it is answering three questions without
inventing either the hypothesis or the number: what do we already know, what
would falsify this, and what did we actually measure?

```text
OBSERVE ──► RECALL ──► HYPOTHESIZE ──► PREDICT
   ▲                                      │
   │                                      ▼
DECIDE ◄── REMEMBER ◄── EVALUATE ◄── EXPERIMENT
```

The loop is a state machine, not a menu. You may go back one step when an
experiment is invalid; you may not skip RECALL or REMEMBER.

| Step | Question | Structured artifact |
| --- | --- | --- |
| OBSERVE | What is the host, target and current cost? | `perf system info --json`, `perf system tune --dry-run --json`, `perf profile -o`, `perf benchmark --json`, `perf analyze --json` |
| RECALL | What did we already measure on this state? | `perf_experiments` query |
| HYPOTHESIZE | What single mechanism explains the dominant bound? | top-down slots, raw counters, disassembly |
| PREDICT | Which metric moves, in which direction, by what threshold? | the reasoning record in 3.8 |
| EXPERIMENT | What one variable changes? | one `perf benchmark` run with one `--config` axis or one code change |
| EVALUATE | Is the delta real and in the predicted direction? | `perf compare --json` (CLT z-test) |
| REMEMBER | What is the verdict and where is the evidence? | append-only `perf_experiments` record |
| DECIDE | Accept, reject or inconclusive; what next? | the reasoning record and the next bound |

Use this evidence hierarchy; never promote a weaker item to a stronger one:

1. a measured artifact (`benchmark`, `profile`, `compare`) with its path;
2. a measured counter or top-down slot from that artifact;
3. a modeled instruction property (`perf analyze` latency/throughput), labelled
   as a model;
4. a hypothesis or expectation, which is not evidence at all.

No stage may silently consume a weaker item as if it were a stronger one.

## Reasoning invariants

These rules outrank any command example in this document.

1. **Recall before hypothesis.** Query `perf_experiments` before proposing a
   change. A known accepted result is reused; a known rejected result is not
   re-tested without new evidence.
2. **Reason about the target only from `PerfState`.** Every claim cites a
   field, for example `topdown_l1.be-bound=61.2` or
   `disassembly_snippet[7].latency=5`. The `perf compare` record is the verdict
   artifact: it evaluates a prediction but adds no counter or offset. Never
   invent a counter, event, address, offset or instruction encoding.
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
   not `0`. If a number came from a model (`perf analyze` latency/throughput),
   label it as a model, not a measurement.
8. **Top-down before line-level.** Establish the dominant level-1 slot and its
   raw-counter evidence before proposing an instruction edit. If the PMU does
   not export top-down events, say so and use the raw-counter fallback; do not
   substitute a guess for the missing bound.

## 1. Structured State Contract

`PerfState` is the only observation of the target you may reason from. It is a
projection of the structured artifacts emitted by the four stage commands, not
a file the toolkit emits and not a license to invent fields. The separate
`perf compare` record is the verdict artifact for a prediction; it does not add
counters or offsets to the state.

The state-producing forms are:

| Stage | Structured command | What it contributes |
| --- | --- | --- |
| `profile` | `perf profile -f SYM -e EVENT -o trace.json -- ./a.out` | live per-call/per-sample trace for a running program; optional hot-function context |
| `benchmark` | `perf benchmark TARGET -m latency -e EVENT -o data/ --json run.json` | self-describing run envelope: target, host, config, data, operations, measured events |
| `analyze` | `perf analyze TARGET --json disasm.json -- data/run.json` | one record per instruction: encoding, size, modeled latency/throughput, explored `data.*`, joined counters |
| `optimize` | `perf optimize BIN --list --json hot.json -- data/` (or `-o BIN.opt --json report.json`) | measured hotness per function, or the rewrite report for a produced binary |

`perf system info --json` and `perf system tune --dry-run --json` supply the
environment fingerprint, and `perf compare --json` supplies the verdict; none
of them adds target counters.

`perf analyze` has no `--output`; its structured flag is `--json [FILE]`.
`perf optimize --output` is the output binary, not a JSON file. The complete
`PerfState` is assembled from the envelopes named in the provenance table below.

```sh
perf analyze a.out:fizz_buzz --json .perf/state/disasm.json
```

The envelope is per-instruction. Passing `--output json` to `perf analyze` is
an error, not a state request.

### 1.1 Schema

```json
{
  "state_hash": "sha256:9f2c4e8a1b7d6350e4c8a2f6d1b9037e5c4a8f2d6e1b7c3a9f5d0e4b8c2a6f1d",
  "target": {
    "file": "a.out@b04f8237",
    "name": "fizz_buzz",
    "mode": "latency",
    "operations": 1
  },
  "hardware_config": {
    "hostname": "host",
    "cpu": {
      "vendor": "Example CPU",
      "family": 6,
      "model": 154,
      "stepping": 3,
      "arch": "x86_64",
      "platform": "Linux-6.14-x86_64",
      "cores": 8,
      "smt": false,
      "hybrid": false
    },
    "caches": {"L1i": "32Kb", "L1d": "48Kb", "L2": "1280Kb", "L3": "24576Kb"},
    "freq_hz": 2688000000,
    "affinity": 4,
    "governor": "performance",
    "turbo": false,
    "rdpmc": true,
    "kernel": "6.14"
  },
  "tma_bottleneck": "be-bound",
  "topdown_l1": {
    "retiring": 24.1,
    "bad-spec": 3.8,
    "fe-bound": 10.9,
    "be-bound": 61.2
  },
  "hot_functions": [
    {"name": "fizz_buzz", "address": "0x401000", "size": 64, "hotness": 612.4}
  ],
  "profile_trace": null,
  "disassembly_snippet": [
    {
      "index": 0,
      "address": "0x401000",
      "encoding": "89 f8",
      "size": 2,
      "assembly": "mov eax, edi",
      "latency": 1,
      "throughput": 0.25,
      "data": {"rdi": [15]}
    }
  ],
  "missing": []
}
```

`missing` lists fields that could not be measured. Never fill a missing field
from memory, intuition, another target, or another machine. `hostname` is
display-only: it is excluded from `state_hash` and from the canonical JSON
stored in `perf_experiments`. `profile_trace` is optional context: populate it
only when a profile artifact was recorded, and never use it as the A/B verdict.

### 1.2 Provenance

| Field | Source | Rule |
| --- | --- | --- |
| `hardware_config` | `perf system info --cpu --json FILE`, `perf system tune --dry-run --json FILE`, and `/sys`/`/proc`/`uname` reads | Canonicalize the emitted fields (vendor, family, model, stepping, arch, platform, freq, L1i/L1d/L2/L3) plus derived topology (`cores`, `smt`, `hybrid`), affinity, governor, turbo, rdpmc and kernel. Use `null` when a value cannot be read. |
| `topdown_l1` | `perf benchmark TARGET -m latency -e 'topdown-*' --json FILE` | The four values are percentages of slots and sum to approximately 100. If the PMU does not export the events, set the field to `null` and record the error. |
| `tma_bottleneck` | Derived from `topdown_l1` | The dominant level-1 slot, only when all four values are present and finite. This toolkit stops at L1: do not name an L2 or L3 TMA node. If `retiring` dominates, treat it as "no clear L1 bottleneck" and inspect IPC and dependency evidence. |
| `profile_trace` (optional) | `perf profile -f SYM -e EVENT -o trace.json -- ./a.out` | Live `mode=record` rows with `config.events` and `config.overhead`; use for whole-program context. It is accepted as `DATA` by `perf optimize --list`, but it is not an isolated benchmark and does not replace `benchmark` for an A/B. |
| `hot_functions` | `perf optimize BIN --list --json -- DATA` (`DATA` may be a benchmark, profile or analyze envelope, or `perf.data`) | Measured hotness only. `hotness` is a raw score (cycles/op, duration/op, instructions/op, or summed modeled instruction latency/throughput), not a percentage and not normalized. `perf optimize --list` emits `0.0` when no measured data exists; record that value, not `null`. Addresses are emitted as integers; render them as `0x` without changing the value. |
| `disassembly_snippet` | `perf analyze TARGET --json FILE` | Exact target instructions, addresses and encodings as emitted. The envelope emits integer addresses and space-separated hex byte encodings (`89 f8`); normalize addresses to the exact `0x` rendering without changing the value. Do not renumber, rebase or synthesize an offset. `data.*` values are the explored state lists. |
| `state_hash` | Derived | SHA-256 of the canonical state identity: target (`file`, `name`, `mode`), hardware config without hostname, measured `config`/`data`, binary content hash, and instruction addresses plus encodings. Exclude counters, auto-calibrated `operations`, timestamps and host-specific paths. |
| `target` | The command line | File, symbol or region, mode and operations. Never infer the target from a symbol name. |

### 1.3 State hash and validity

```text
state_hash = "sha256:" + sha256(canonical_json({
  target:       {file, name, mode},
  hardware:     hardware_config without hostname,
  config:       measured config.*,
  data:         measured data.*,
  binary:       sha256 of the ELF,
  disassembly:  [{address, encoding, size}]
}))
```

The hash is an identity, not a result: counters and timestamps are excluded so
two runs of the same state hash are comparable. A new binary, host, tune
profile, input or instruction encoding is a new state.

Validate the state before reasoning:

- `topdown_l1` has all four slots and sums to 100 ± 2; otherwise set it and
  `tma_bottleneck` to `null`. The window includes the harness around the
  measured region, so read the split as attribution for the whole measured
  loop, not for one instruction in isolation.
- On a hybrid CPU the kernel may not schedule all four slots; an empty slot is
  `null`, not `0`. Re-measure the missing slots separately if the hypothesis
  needs the complete split.
- `tma_bottleneck` is the argmax of a complete `topdown_l1`.
- Every `disassembly_snippet` address belongs to the measured target. Every
  `hot_functions` address belongs to the measured binary.
- `hardware_config` has a CPU model, cache sizes, `freq_hz` and affinity.
- `state_hash` matches the canonical projection.

If validation fails, fix the measurement, not the JSON.

### 1.4 `--output` and `--json` are not synonyms

Use `--json [FILE]` when the question is "give me the structured result". Use
`--output` only for the object the command actually writes. The meanings are
command-specific:

| Command | `--output` / `-o` | `--json [FILE]` |
| --- | --- | --- |
| `benchmark` | output **root directory**; writes `<root>/<file>@<hash>/<name>-<id>/<time>.json`, and prints no table | emit the run envelope to stdout, or write it to `FILE` |
| `profile` | JSON **file** for the trace (default `profile.json`; `none` prints a table) | also emit trace records to stdout or `FILE` |
| `analyze` | not supported | write instruction records to stdout or `FILE` |
| `optimize` | output **binary** (`a.opt.out` by default) | write the optimization report, or with `--list` the function records, to stdout or `FILE` |
| `compare` | not supported | write verdict records to stdout or `FILE` |
| `system info` / `tune` / `reset` | not supported | write environment/tuning records to stdout or `FILE` |
| `view` | not supported | derived records to stdout or `FILE` |
| `plot` | image path | also emit the plotted data as structured records |

Rules:

- For a benchmark experiment, `-o data/` is the canonical store. With both
  `-o data/` and `--json run.json`, the tree under `data/` is canonical and
  `run.json` is a portable copy.
- For `profile`, `-o` is a file, not a directory. For `optimize`, `-o` is a
  binary, not data. For `plot`, `-o` is an image, not a result table. Using the
  wrong one is a protocol error, not a typo.
- **Read the artifact, not stdout.** Re-open raw runs with `perf view --json`,
  produce the verdict with `perf compare --json`, or inspect the files with
  `jq`; do not scrape a rendered table.
- Keep the command line next to the artifact. A result whose input, config, or
  binary is unknown is not reproducible and cannot be accepted.
- Hash the binary and record it. A benchmark result belongs to a specific ELF,
  not to a filename that a later build can overwrite.
- `--json` with no `FILE` writes to stdout and must own it. Keep
  `--config.external.stdout=false` for benchmark targets that print.

### 1.5 Canonical experiment run

Pin every control that is not the experimental variable. The same command line
is used on both sides; only the measured target or the one whitelisted axis
changes.

```sh
export EXP=".perf/experiments/$(date -u +%Y%m%dT%H%M%S)-fizz"
mkdir -p "$EXP/data"
PIN="--config.backend.loop.runs=10 --config.branch=predictable --config.dcache=hot \
--config.dtlb=hot --config.icache=hot --config.itlb=hot \
--config.func.align=16 --config.func.order=as-is"

perf system info --cpu --json "$EXP/env.json"
perf system tune --dry-run --json "$EXP/tune.json"

perf benchmark a.old:fizz_buzz -n old -m latency -e cycles,instructions \
  $PIN -o "$EXP/data/" --json "$EXP/old.json"
perf benchmark a.new:fizz_buzz -n new -m latency -e cycles,instructions \
  $PIN -o "$EXP/data/" --json "$EXP/new.json"
perf benchmark a.new:fizz_buzz -m latency -e 'topdown-*' \
  $PIN --json "$EXP/topdown.json"
perf analyze a.new:fizz_buzz --json "$EXP/analyze.json" -- "$EXP/new.json"
perf optimize a.new --list --json -- "$EXP/new.json" > "$EXP/hot.json"
perf compare -b old --json "$EXP/compare.json" -- "$EXP/data/"

# optional live context for the whole process; not an A/B substitute
perf profile -f fizz_buzz -e cycles -o "$EXP/profile.json" -- ./a.new
```

`$PIN` is shell expansion, not a tool feature; if you quote it, write the flags
out. Pin `--config.iterations` as well when loop-trip count affects the
workload. For a config experiment, keep the binary identical and change exactly
one axis in `$PIN` (for example `--config.dcache=hot` to `cold`). For a code
experiment, keep `$PIN` identical and change only the binary.

The benchmark envelope carries the identity (`file`, `name`, `mode`), the
inputs (`config.*`, `data.*`), every measured event, and `operations` (1 for a
latency row, the auto-calibrated trip count for throughput). `perf view`
projects the per-operation value. The compare records carry
`event`, `baseline`, `challenger`, `diff` (signed percentage change), `ci_low`
and `ci_high` (same percentage units), `p_value`, and `significant`. Those
fields — not your recollection — populate memory.

### 1.6 Artifact checklist

A run is complete only when all of these are true:

- [ ] target, mode, events, config, and data are recorded;
- [ ] the binary hash and `hardware_config` are recorded;
- [ ] `operations` (1 for latency, the trip count for throughput) and the
      per-operation denominator are present;
- [ ] the `topdown_l1` values and derived `tma_bottleneck` are recorded, or
      both are `null` with the reason;
- [ ] `null` counters are listed as `null`, never converted to zero;
- [ ] the profile trace, analyze records, and optimize report are recorded when
      those stages were used;
- [ ] the artifact paths are in the experiment directory;
- [ ] if a change was evaluated, `compare.json` has a `significant` verdict.

### 1.7 Hard rules

1. **Cite the field.** Every claim names its evidence, for example
   `topdown_l1.be-bound=61.2` or `disassembly_snippet[7].latency=5`.
2. **`null` is not `0`.** An unread counter, an unsupported event and a failed
   run are all unknown, not zero.
3. **Never invent counters or offsets.** If a counter, event, address or
   instruction encoding is not in `PerfState`, it is unknown. Do not compute an
   offset from a symbol address unless both addresses are in the state.
4. **Do not mix states.** Rows from different hardware configs, binaries, target
   regions, modes or input data are different states.
5. **Derived fields follow their rule.** `tma_bottleneck` is the argmax of a
   complete `topdown_l1`; `state_hash` is the canonical hash. No other
   derivation is allowed.
6. **Do not mix units.** Top-down slots are percentages of slots; counters are
   counts; `duration_time` is nanoseconds; `latency`/`throughput` from
   `perf analyze` are instruction properties. Do not add or multiply across
   units.
7. **Incomplete state blocks acceptance.** If the hypothesis needs a missing
   field, either measure it or mark the experiment `inconclusive`. Never fill
   the gap with a plausible number.
8. **A new measurement is a new state.** Never edit a state after hashing it.
   Re-measure and produce a new hash.
9. **Only whitelisted variables.** An accepted experiment changes exactly one
   of `--config.icache={cold,hot}`, `--config.dcache={cold,hot}`,
   `--config.branch={predictable,unpredictable}` or
   `--config.decode={stall,free}` (reserved; not implemented), or makes one
   code/binary change. Everything else (affinity, sampling, backend, host
   tuning, dtlb, itlb, alignment, input data) is a fixed control or an
   observation; a hypothesis that needs it is `inconclusive` until the
   whitelist is extended.

## 2. Experiment Memory

The memory is a queryable ledger named `perf_experiments`, owned by the agent.
The toolkit does not ship memory; this ledger is part of the protocol. The
contract query surface is the table in 2.7; use SQLite when it is available.
When it is not, a JSONL ledger with the same fields is an acceptable substitute:
query it with the equivalent predicate and say which store you used.

- **SQLite** — `.perf/experiments.db` (or
  `${XDG_STATE_HOME:-$HOME/.local/state}/perf/experiments.db` when the
  workspace is read-only) is the preferred store for indexed recall over many
  experiments.
- **JSON** — one file per experiment, or a `.jsonl` append-only log. Keep the
  raw envelopes as the source of truth and append one verdict record per
  experiment that points to them.

Pick one authoritative store and say which file it is. When SQLite is used, the
`perf_experiments` table is the query surface; the JSON artifacts remain the
durable evidence.

If you only have raw `benchmark`/`profile` JSON files, you have artifacts, not
yet memory. Keep those files unchanged and add a small verdict record that
points to them.

### 2.1 Store layout

```text
.perf/
  experiments/<id>/       # raw artifacts: env.json, old.json, new.json,
                          # compare.json, topdown.json, analyze.json, ...
  experiments.jsonl       # substitute verdict ledger, one JSON object per line
                          # (or experiments/<id>/verdict.json, one per run)
  experiments.db          # perf_experiments table when SQLite is used
  state/                  # PerfState projections and envelopes
```

Keep the ledger in the project. Do not put it in a shared temporary directory
that disappears between sessions.

### 2.2 Schema

The ledger record is storage-neutral. One record per tested hypothesis:

```json
{
  "id": "20261112T002500-fizz-H1",
  "state_hash": "sha256:9f2c...",
  "hypothesis_id": "H-1",
  "tma_bottleneck": "bad-spec",
  "hardware_config": {"cpu": "Example CPU", "caches": {"L1d": "48Kb"}},
  "target": "a.out:fizz_buzz",
  "metric": "cycles/operations",
  "command": "perf benchmark ...; perf compare ...",
  "result_delta": -12.4,
  "p_value": 0.0001,
  "verdict": "accept",
  "timestamp": "2026-11-12T00:25:00Z",
  "notes": "mispredict removed",
  "artifacts": ["experiments/20261112T002500-fizz-H1/compare.json"]
}
```

`verdict` is one of `accept`, `reject`, `inconclusive`. The `artifacts` list may
be omitted when the record lives beside its experiment directory. The same
record can be stored as one JSON file, one line in a JSONL log, or a row in the
`perf_experiments` table (2.7).

`hardware_config` is canonical JSON (sorted keys, no hostname) so equality
means the same measurement environment. `result_delta` is the signed percentage
change of the primary metric from baseline to candidate, as reported by
`perf compare` (`diff`); the means are geometric when the log test is defined,
arithmetic otherwise:

```text
result_delta = (candidate - baseline) / baseline * 100
```

Record `metric` with it. For cycles and duration, a negative delta is faster.
For IPC and throughput, a positive delta is faster. If the baseline is zero or
the metric is missing, store `NULL` and mark the verdict `inconclusive`.

### 2.3 Recall before hypothesis

Run the recall query before writing a hypothesis, not after the benchmark.

If the ledger is JSONL:

```sh
jq -c --arg state "$STATE_HASH" --arg hyp "$HYPOTHESIS_ID" \
  'select(.state_hash == $state and .hypothesis_id == $hyp)' \
  .perf/experiments.jsonl

# Contract recall before a new hypothesis: same bottleneck and hardware.
jq -c --arg b "$TMA_BOTTLENECK" --argjson hw "$HARDWARE_CONFIG_JSON" \
  'select(.tma_bottleneck == $b and .hardware_config == $hw)' \
  .perf/experiments.jsonl
```

If it is one JSON file per record:

```sh
jq -s --arg state "$STATE_HASH" --arg hyp "$HYPOTHESIS_ID" \
  'map(select(.state_hash == $state and .hypothesis_id == $hyp))' \
  .perf/experiments/*.json
```

For a single JSON array, use the same `map(select(...))` without `-s`. For a
prior on the same bottleneck and hardware, filter `tma_bottleneck` and
`hardware_config` in the same expression.

The contract key is `(tma_bottleneck, hardware_config)`. Prefer records with
the same `target` and `metric`; treat a cross-target record as a prior, not as
proof. An exact `state_hash` match is stronger than either. If you use the
`perf_experiments` table, run the query in 2.7.

If the query returns no records, record "no prior experiments for this state"
and continue with a new hypothesis. Absence of memory is not evidence for the
new idea, and a remembered result is not evidence for a changed binary or host.

### 2.4 Reuse policy

| Memory finding | Action |
| --- | --- |
| `accept`, same `state_hash` and hypothesis | Reuse the conclusion. Do not re-run it. Apply or verify the change and move to the next bottleneck. |
| `reject`, same `state_hash` and hypothesis | Do not retry the same command. Change the mechanism or the measurement; record why the new experiment is different. |
| `inconclusive` | Do not count it as evidence. Retry only after fixing the cause in `notes` (samples, PMU, pin, invalid run). |
| same `tma_bottleneck` and `hardware_config`, different `state_hash` | Reuse as a prior, but revalidate on the current state before accepting. |
| no match | Form a new hypothesis. Do not invent a past result. |

An accepted experiment is a **conclusion**, not a permanent measurement. If
the source or binary changed, the old result no longer describes the current
artifact; reuse the hypothesis, re-measure the implementation. The ledger is
append-only: never delete or rewrite records. A superseding result is a new
record.

### 2.5 Append after every run

Append one verdict record per tested hypothesis, after the comparison. A run
without a comparison is still recorded, so the ledger never loses work.

For JSONL, append one object on one line; for one file per experiment, write the
same object to `experiments/<id>/verdict.json`. Never rewrite an earlier record.
The fields are those in 2.2.

If you use the `perf_experiments` table, insert the same record with the
statement in 2.7.

The required append fields are `state_hash`, `hypothesis_id`, `command`,
`result_delta`, `verdict` (`accept` | `reject` | `inconclusive`) and
`timestamp`. Also record `hardware_config`, `target`, `metric` and `p_value`;
`result_delta` and `p_value` are `NULL` for an exploratory or otherwise
incomparable run. The matching keys, `notes` and `artifacts` make the record
useful later.

This applies to every measured stage: a profile or analyze run that supported a
hypothesis, an optimize pass that produced a candidate binary, and every A/B
comparison. The record captures the experiment and its verdict; the artifacts
stay in the experiment directory.

A run record is not a substitute for the artifacts: keep the envelope and
comparison files, and point at them from `notes` or the experiment directory.
If a run is exploratory and has no comparison, still append it with
`verdict = 'inconclusive'` and `result_delta = NULL`, so the work is not lost.
Use a distinct `explore-*` hypothesis id for an exploratory run, so the dedup
key (the SQLite unique index or your own check) cannot suppress the later
verdict record for a tested hypothesis. `state_hash` is the candidate state;
record the baseline state and command in `notes` or the experiment directory.

### 2.6 Verdicts

| Verdict | When |
| --- | --- |
| `accept` | `perf compare` reports `significant = true`, the delta is in the predicted direction, the mechanism field moved as predicted, and the environment is valid. |
| `reject` | The result is significant in the opposite direction, or the predicted mechanism is falsified. |
| `inconclusive` | Not significant, `p_value` is NaN, fewer than 30 samples per side, a counter is `null`, a flag is unsupported, the variable was not isolated, or the run was invalid. |

An invalid run is never a rejection. Record it as `inconclusive` with the cause
in `notes` and re-run it. `accept` requires all of: significant result, desired
direction, attributed mechanism, valid environment, and recorded artifacts. If
any one is missing, the verdict is `inconclusive`.

The accepted-variable whitelist is exactly
`--config.icache={cold,hot}`, `--config.dcache={cold,hot}`,
`--config.branch={predictable,unpredictable}` and
`--config.decode={stall,free}` (reserved; not implemented by this CLI). A
hypothesis outside it is `inconclusive`, not silently substituted; a code/binary
change is an intervention and is verified the same way as a config axis.

### 2.7 `perf_experiments` table

Use this table when SQLite is available; it is the contract's query surface.
The JSON artifacts remain the durable evidence, and the table can be rebuilt
from the verdict records. Create it on first use:

```sql
CREATE TABLE IF NOT EXISTS perf_experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    state_hash      TEXT NOT NULL,
    hypothesis_id   TEXT NOT NULL,
    tma_bottleneck  TEXT,
    hardware_config TEXT NOT NULL,
    target          TEXT NOT NULL,
    metric          TEXT NOT NULL,
    command         TEXT NOT NULL,
    result_delta    REAL,
    p_value         REAL,
    verdict         TEXT NOT NULL
                    CHECK (verdict IN ('accept', 'reject', 'inconclusive')),
    timestamp       TEXT NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    notes           TEXT,
    artifacts       TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS perf_experiments_dedup
    ON perf_experiments (state_hash, hypothesis_id, command);

CREATE INDEX IF NOT EXISTS perf_experiments_lookup
    ON perf_experiments (tma_bottleneck, hardware_config, target);
```

Recall:

```sql
SELECT hypothesis_id, command, metric, result_delta, p_value,
       verdict, timestamp, notes
FROM perf_experiments
WHERE tma_bottleneck IS :tma_bottleneck
  AND hardware_config = :hardware_config
ORDER BY (target = :target) DESC, timestamp DESC;
```

Append:

```sql
INSERT INTO perf_experiments
    (state_hash, hypothesis_id, tma_bottleneck, hardware_config, target,
     metric, command, result_delta, p_value, verdict, timestamp, notes,
     artifacts)
VALUES
    (:state_hash, :hypothesis_id, :tma_bottleneck, :hardware_config, :target,
     :metric, :command, :result_delta, :p_value, :verdict, :timestamp, :notes,
     :artifacts)
ON CONFLICT (state_hash, hypothesis_id, command) DO NOTHING;
```

Use `sqlite3 .perf/experiments.db` (or bind the same parameters from Python).
Bind `hardware_config` to the canonical JSON text of the verdict record's
`hardware_config` object.

## 3. Controlled Experiment Discipline

### 3.1 Preflight: control the host first

```sh
uname -r                                   # 6.x+ required
perf system info --cpu --json env.json     # per-CPU topology, TSC freq, caches
perf system info --cpu -e cpu,core,L1d     # narrow the table when inspecting
perf system tune --dry-run                 # plan governor/frequency/SMT/THP/...
perf system tune --dry-run --json tune.json
perf system tune --dry-run --profile=latency
perf system tune --dry-run --profile=throughput
perf system tune --dry-run --profile=latency \
  --profile.aslr=true --profile.cpu_frequency=max --profile.smt=false
perf system tune --profile=latency         # low-jitter profile (prompts for sudo)
for f in /sys/devices/{cpu,cpu_core,cpu_atom}/rdpmc; do
  [ -e "$f" ] && cat "$f"                 # user-space rdpmc
done
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
```

`.github/workflows/perf.sh` is the authoritative smoke suite for exact flag
spelling and for the `system info` / `system tune` variants. Read it or
`perf <command> --help` before inventing an option.

If `rdpmc` reads `0`, hardware events fail in the perf-labs tools (the kernel
exposes no RDPMC index and there is no syscall fallback on the counting path).
Enable it with `perf system tune --profile=latency`; `duration_time` still
works through `rdtsc`, but hardware events do not. For a one-off
kernel-syscall measurement use system `perf stat`, which falls back to
`perf_event_open`.

On hybrid CPUs, a PMU event may exist only on one core type. `perf benchmark`
pins itself to a PMU-capable CPU; if you drive `perf stat` by hand, pin it
yourself with `taskset -c <p-core>`.

Use one tuning profile consistently for the whole experiment: `latency` for
low jitter, `throughput` for sustained throughput. Do not change the profile
between baseline and candidate. If the governor, frequency, SMT, THP, kernel,
or affinity changed between baseline and candidate, the comparison is invalid:
re-baseline. An effect under about 3% is a machine suspect first: re-run pinned,
in the same session, before believing the code changed.

### 3.2 One variable per command

**One variable per `perf benchmark` command.** One command changes one
experimental variable. A one-axis sweep, such as
`--config.dcache=hot,cold`, is still one variable.

> **Contract whitelist (only accepted flags):**
> `--config.icache={cold,hot}`, `--config.dcache={cold,hot}`,
> `--config.branch={predictable,unpredictable}`,
> `--config.decode={stall,free}`.
> `decode` is reserved and not implemented; a decode hypothesis is
> `inconclusive`. The axes listed in 3.3 are observation and candidate
> extensions, not accepted variables.

Baseline and candidate must be identical in every other respect:

- same mode (`-m`), events (`-e`) and output root (`-o`);
- same target, symbol/region and, for a config experiment, binary content;
- same `--data` (input variation is exploratory and `inconclusive`),
  `--config.samples` (throughput), `--config.backend.<name>.runs` (latency),
  `--config.iterations`, `--backend`;
- same branch, cache, TLB and alignment/layout controls (`--config.branch`,
  `--config.dcache`, `--config.icache`, `--config.dtlb`, `--config.itlb`,
  `--config.func.align`, `--config.func.order`, and `--config.code.align` for
  snippet targets); only a whitelisted flag may vary;
- same CPU affinity, NUMA binding and priority;
- same host tuning state, governor, turbo and SMT.

Do not combine a code change with a config change. Do not sweep two axes and
read the corner. Do not change the workload to flatter the candidate. Do not
compare a pinned run with an unpinned one.

### 3.3 Observation axes and candidate extensions

One observation or candidate extension changes exactly one of these axes; only
the 3.2 whitelist yields `accept`:

```text
--config.branch={predictable,unpredictable}
--config.dcache={hot,warm,cool,cold}
--config.icache={hot,cold}
--config.dtlb={hot,cold}
--config.itlb={hot,cold}
--config.code.align=N (any power of two >= 1; e.g. 1,16,32,64)
--config.func.align=N (any power of two >= 1) | --config.func.order={as-is,random}
--data.*
one code/binary change
perf optimize --opt {peephole,branch,layout,align}
```

Notes that keep this executable:

- `dcache` and `icache` also accept `warm`, `cool` and hit-rate forms, but the
  clean two-point contrast is `cold` versus `hot`.
- `branch` accepts exactly `predictable` and `unpredictable`.
- There is no `--config.decode` knob in this CLI. A decode-bound hypothesis is
  discriminated with `topdown_l1.fe-bound`, `icache`/`itlb` observations,
  instruction encoding and code size; do not invent a decode counter.
- Alignment is an axis in two places: `--config.code.align` aligns the measured
  snippet and is ignored for binary targets; `--config.func.align`/
  `--config.func.order` control binary function layout. `perf optimize --opt
  layout,align` tests the rewritten binary as one code variable.
- `--data.*` is identical on both sides; varying it is exploratory and cannot
  yield `accept` until the whitelist includes it.
- A one-axis sweep such as `--config.dtlb=hot,cold` is an observation; it
  cannot yield `accept` until `dtlb` is whitelisted. Do not combine it with a
  cache or branch sweep.

If a hypothesis needs a flag outside the whitelist, substitute the closest
whitelisted contrast where one exists (for example `dcache=hot` vs `cold` for a
broader cache sweep) and record the substitution; otherwise mark the hypothesis
`inconclusive`. Do not substitute one axis for another silently. Any other
documented `--config` key (stack, seed, backend options, external library
modelling) may be a fixed control; add it here before using it as the one
variable.

### 3.4 Fixed controls and command template

These are never the variable and must not drift between baseline and candidate:

| Control | How |
| --- | --- |
| CPU | `--config.thread.affinity=N`; for system `perf`, `taskset -c N` |
| NUMA / priority | `--config.thread.numa`, `--config.thread.priority` |
| Samples | `--config.samples=N` (throughput only; default 100). For latency, pin `--config.backend.<name>.runs` and `--config.iterations` |
| Iterations | `--config.iterations` |
| Backend | `--backend loop` / `unroll`, same on both sides |
| Input data | `--data.*`, identical on both sides; input variation is exploratory and cannot yield `accept` |
| Events | `-e` groups, same on both sides |
| Mode | `-m latency` or `-m throughput`, never mixed in one comparison |
| Branch | `--config.branch=predictable` unless branch is the whitelisted axis |
| Cache / TLB | `--config.dcache=hot --config.icache=hot --config.dtlb=hot --config.itlb=hot`; only dcache/icache may be the whitelisted axis, dtlb/itlb stay fixed |
| Alignment / layout | `--config.func.align=16` and `--config.func.order=as-is` for binary targets; `--config.code.align=16` for snippets; alignment/layout changes are code/binary changes, not config axes |
| Host tuning | One `perf system tune --profile=latency` or `--profile=throughput` before the session; never change it mid-experiment |

```sh
# Baseline
perf benchmark a.out:fizz_buzz -m latency -e cycles,instructions \
  -n base --config.backend.loop.runs=10 --config.thread.affinity=4 \
  --config.branch=predictable --config.dcache=hot --config.dtlb=hot \
  --config.icache=hot --config.itlb=hot --config.func.align=16 \
  --config.func.order=as-is -o data/

# Candidate: exactly one whitelisted axis differs (dcache)
perf benchmark a.out:fizz_buzz -m latency -e cycles,instructions \
  -n cand --config.backend.loop.runs=10 --config.thread.affinity=4 \
  --config.branch=predictable --config.dcache=cold --config.dtlb=hot \
  --config.icache=hot --config.itlb=hot --config.func.align=16 \
  --config.func.order=as-is -o data/

perf compare -b base -- data/
```

For a code change, keep every flag identical and change only the binary:

```sh
perf benchmark old.out:fizz_buzz -m latency -e cycles,instructions \
  -n old --config.backend.loop.runs=10 --config.thread.affinity=4 \
  --config.branch=predictable --config.dcache=hot --config.dtlb=hot \
  --config.icache=hot --config.itlb=hot --config.func.align=16 \
  --config.func.order=as-is -o data/
perf benchmark new.out:fizz_buzz -m latency -e cycles,instructions \
  -n new --config.backend.loop.runs=10 --config.thread.affinity=4 \
  --config.branch=predictable --config.dcache=hot --config.dtlb=hot \
  --config.icache=hot --config.itlb=hot --config.func.align=16 \
  --config.func.order=as-is -o data/
perf compare -b old -- data/
```

### 3.5 Attribute with top-down and per-instruction evidence

```sh
perf benchmark a.out:func -m latency -e 'topdown-*' \
  --config.backend.loop.runs=10 --config.branch=predictable --config.dcache=hot \
  --config.dtlb=hot --config.icache=hot --config.itlb=hot \
  --config.func.align=16 --config.func.order=as-is
```

Read the four level-1 slots and drill into the dominant one. Do it in this
order:

1. Measure `-e 'topdown-*'` on the current state; record all four slots.
2. Confirm the host exports them (`perf list | grep topdown`). If the PMU does
   not, the command fails with `unknown event` rather than reporting zeros —
   fall back to raw counters and leave `topdown_l1` and `tma_bottleneck`
   `null`.
3. Name the dominant slot and the hypothesis it suggests from the table below.
4. Measure the raw counters that discriminate the hypothesis, for example
   `branch-misses` for bad speculation or L1d/LLC/dTLB misses for the backend.
5. Only then use `perf analyze` to attribute the bound to instructions.
6. Turn the mechanism into one controlled experiment and predict the slot that
   should move.

The top-down window includes the harness around the measured region; treat the
split as attribution for the whole measured loop, not for one instruction in
isolation.

| Signal | Meaning |
| --- | --- |
| IPC far below the machine width | dependency chain or port pressure |
| IPC ≈ 1 | about one dependent chain per cycle |
| `branch-misses` high per branch | unpredictable control flow; bad speculation |
| `L1-dcache-load-misses` >> `LLC-load-misses` | L1 capacity/conflict, not DRAM |
| `dcache=cold` >> `dcache=hot` | working set does not fit L1 |
| `dtlb=cold` gap >> `dtlb=hot` gap | page-walk bound; fewer or huge pages |
| `topdown-fe-bound` high with a small `itlb` gap | decoder/branch density, not TLB |
| `topdown-be-bound` high | memory/back-end bound |
| `topdown-retiring` high with low IPC | issue width or a dependency chain |

Top-down says **which** bound; raw counters say **how much**; `perf analyze`
says **which instructions**. Use them in that order. Do not jump from a
suspicion to a line-level edit without the bound.

| `tma_bottleneck` | First observation | Allowed experiment | Accept when |
| --- | --- | --- | --- |
| `bad-spec` | `branch-misses`; predictable vs unpredictable | `--config.branch=predictable,unpredictable` | unpredictable is significantly slower and bad-spec rises |
| `fe-bound` | `icache=hot` vs `cold`; iTLB; code size | `--config.icache=hot,cold` or one code change | cold is significantly slower and fe-bound rises |
| `fe-bound` with an alignment or layout gap | code alignment, function order, hotness | one code/binary change (`perf optimize --opt layout,align`); config alignment flags are observations, not accepted axes | the change clears `perf compare` and fe-bound/icache improves |
| `fe-bound` without an icache gap | instruction count, branch density, encoding | no decode knob: use `perf analyze` and a code change | change clears `perf compare` and fe-bound falls |
| `be-bound` with a dcache gap | L1d misses vs LLC misses | `--config.dcache=hot,cold` | cold is significantly slower and L1d misses rise |
| `be-bound` with a dtlb gap | `dtlb=hot` vs `cold`; dTLB misses | no accepted axis: a dTLB hypothesis is `inconclusive` until `dtlb` is whitelisted; test a code change that reduces pages | a code change clears `perf compare` and dTLB misses fall |
| `be-bound` without a cache/TLB gap | per-instruction latency, IPC | `perf analyze`; one code change | change clears `perf compare` and IPC/field moves |
| `retiring` with low IPC | latency vs throughput; per-instruction latency | one code change; `perf analyze` | change clears `perf compare` and IPC rises |
| `retiring` near machine width | no dominant slot | stop; do not optimize noise | no experiment is justified |
| `null` / unsupported | raw counters only | do not guess a bottleneck | only a measured code change is accepted |

The worked studies in `studies/x86_64/` are the reference for method and for
the expected shape of each level-1 bound:

| Study | Shows |
| --- | --- |
| `retiring/ipc.ipynb` | IPC and dispatch width from a NOP-width sweep |
| `bad_speculation/branch.ipynb`, `if_else.ipynb` | predictable vs unpredictable branches; misprediction rate and penalty |
| `frontend_bound/icache.ipynb` | code alignment and instruction footprint; top-down shares vs alignment |
| `backend_bound/dcache.ipynb` | `dcache=hot,warm,cool,cold`; tier effect on load latency and top-down shares |
| `backend_bound/dsize.ipynb` | input-size scaling on `qsort` |
| `x86_64.ipynb` | machine characterization: clock, dispatch width, ROB, ALU throughput, branch, TLB, cache, DRAM, and measured vs `llvm-mca` instruction properties |

Use them to check that an experiment isolates the intended bound. Do not copy
their numbers: re-measure on the current `state_hash` and `hardware_config`.
`studies/README.md` states the method compactly: measure with `-e 'topdown-*'`,
change one thing that relieves the biggest slot, re-measure and check that the
slot moved.

### 3.6 Verify with the shipped CLT z-test, and respect pinning

`perf compare` is the only significance test. Do not substitute a t-test,
bootstrap, or a diff of medians. Fix `--alpha` before the run; do not tune it
after seeing the data.

```sh
perf benchmark a.old:func -n old -m latency -e cycles,instructions \
  --config.backend.loop.runs=10 --config.thread.affinity=4 \
  --config.branch=predictable --config.dcache=hot --config.dtlb=hot \
  --config.icache=hot --config.itlb=hot --config.func.align=16 \
  --config.func.order=as-is -o data/
perf benchmark a.new:func -n new -m latency -e cycles,instructions \
  --config.backend.loop.runs=10 --config.thread.affinity=4 \
  --config.branch=predictable --config.dcache=hot --config.dtlb=hot \
  --config.icache=hot --config.itlb=hot --config.func.align=16 \
  --config.func.order=as-is -o data/
perf compare -b old --json compare.json -- data/
```

Benchmark `name` values carry a `-<8hex>` run-id suffix (`func-1a2b3c4d`);
matching strips that suffix only when it is exactly 8 hex chars, so
`func-1a2b3c4d` matches `func` but `func-zzzzzzzz` does not. A piped JSON
table with a `columns`/`data` length mismatch is empty, not a 1-row frame:
re-read the artifact instead of reasoning from the garbage row.

The existing test is a two-sided z-test on the arithmetic mean plus a two-sided
z-test on the geometric mean. When both p-values are finite, both must reject,
with `p = max(p_mean, p_gmean)`, before the change is `significant` at `--alpha`
(default `0.05`). When the geometric test is undefined (for example any sample
is `<= 0`, so zero counters are kept but the log test returns NaN), the verdict
rests on the finite test alone; when no p-value is finite, `perf compare`
emits no comparison row for that event (and, when no event has a finite
p-value, a baseline-only row with `p_value` null), which is `inconclusive`,
not a rejection.
The normal approximation needs at least 30 usable samples per
side; plan for that and collect more when the spread is large. Latency rows are
per call, so a latency A/B needs at least 30 loop trips per side (set
`--config.iterations` or repeat runs); throughput needs at least 30 samples.
With no `-e`, every event is compared per operation (`cycles/operations`) —
two runs do a different number of operations, so raw totals are never compared,
except when the input has no `operations` column (a `perf.data` profile trace),
which falls back to its raw counters.

`diff` is the signed percentage change reported by the test: the geometric-mean
change when the log test is defined, otherwise the arithmetic-mean change
(`(challenger - baseline) / baseline * 100` over those means). The rendered
table prints `-` when the change is not significant; the JSON record still
carries the numeric `diff` plus `p_value` and `significant`, so use
`significant` for the verdict.
`ci_low` and `ci_high` are in the same signed-percentage units; use
`significant`, not the interval alone, for the decision. Use `diff` directly as
`result_delta`; do not recompute a percentage from `diff`. For cycles and
duration a negative delta is faster; for IPC and throughput a positive delta is
faster. Zero deltas are kept; when every run is below overhead, the fallback
reports the overhead-corrected minimum clamped to zero with a warning.

CPU pinning rules:

- `perf benchmark` pins itself to a PMU-capable CPU by default. Set
  `--config.thread.affinity=N` explicitly and keep it for every run in the
  experiment.
- On a hybrid CPU, pin to the core type whose PMU exports the events; top-down
  events may exist on only one type.
- For system `perf stat`, pin yourself: `taskset -c 4 perf stat ...`.
- Keep NUMA binding and priority constant. A pinned run and an unpinned run are
  not comparable.

| Compare result | Verdict |
| --- | --- |
| `significant: true`, desired direction, CI excludes 0 | candidate for `accept` (still needs mechanism) |
| `significant: false` or `diff: "-"` | `inconclusive`; report "no measurable difference" |
| `significant: true`, opposite direction | `reject` |
| `p_value` is `null`/`NaN`, no verdict record, or fewer than 30 samples per side | `inconclusive`; collect more data |
| environment or binary changed | invalid; re-baseline and record `inconclusive` |

Respect the verdict. A 2% delta that did not clear the test is not a win, and a
"win" whose mechanism is unexplained does not become `accept` by repeating the
run until it looks good.

### 3.7 Evidence discipline

- Only values present in `PerfState` and the comparison artifact are evidence.
- `perf analyze` `latency`/`throughput` columns are models. They are useful for
  ranking instructions, not for claiming a measured cycle count.
- `null` is not zero; a missing counter is missing, not "no misses".
- Correlation is not attribution. A change is accepted only after an isolated
  axis and a mechanism.
- Do not copy a number from memory into a current report unless its
  `state_hash`, `hardware_config`, and binary hash all still match. Otherwise
  re-measure it.
- State uncertainty: sample count, spread (`p10..p99`), confidence interval,
  and significance. "No measurable difference" is a complete, valid result.

### 3.8 The reasoning record

Before each experiment, write this in the experiment notes; after verification,
fill in the evidence and verdict:

```text
Recall:         <perf_experiments records for this state, or "none for this state">.
Observation:    <PerfState field> = <value> (<spread>) under <hardware_config>.
Interpretation: <bound/mechanism> because <topdown/counter/analyze evidence>.
Hypothesis:     H-<n>: If <single change>, then <metric> should <direction>
                by <threshold>, while <control metric> stays within <range>.
Falsifier:      <observation that would reject the hypothesis>.
Experiment:     <one whitelisted axis or one code change>, <samples>,
                <exact command>, state_hash=<...>.
Evidence:       perf compare: diff=<signed percentage>, CI=[<...>, <...>],
                p=<...>, significant=<...>.
Mechanism:      <topdown/analyze confirmation or refutation>.
Verdict:        accept | reject | inconclusive.
Artifacts:      <env.json, old.json, new.json, compare.json, ...>.
Memory:         <perf_experiments id / experiment directory>.
```

If you cannot fill a line, you are not ready to run the experiment.

### 3.9 Stop rules

- Stop when the goal is met, the hypotheses for the dominant bound are
  exhausted, or the counters needed to discriminate the next hypothesis do not
  exist.
- Budget about six experiments per bottleneck. Two consecutive `inconclusive`
  results mean the state or measurement is incomplete, not that the code cannot
  be optimized.
- If the next hypothesis needs a counter the PMU does not export, report the
  capability gap and the experiment that would answer it. Do not invent an
  event.
- Leave the machine as you found it: restore affinity, priority, and NUMA
  binding; `perf system reset` if you tuned; remove scratch binaries under
  `/tmp`.

## 4. Stage playbooks

### 4.1 Tool map

This map is for observation and fixed controls. An accepted experiment changes
exactly one variable from the 3.2 whitelist.

| Question | Command |
| --- | --- |
| What is this machine? | `perf system info --cpu` (topology, TSC frequency, cache sizes) |
| What does an op cost in isolation? | `perf benchmark 'imul eax, 42' -m latency -e cycles,duration_time` |
| What does a function/region cost? | `perf benchmark a.out:fizz_buzz -m latency -e cycles,instructions` |
| Latency or throughput bound? | `-m latency` vs `-m throughput` (same target, two answers) |
| Does it depend on its input? | `--data.arg0=1` / `--data.arg0=[1,3,5]` (observation only) |
| What does a program cost with real arguments? | `perf benchmark /usr/bin/tree:main -m latency -- /path/to/folder` |
| Branch mispredicts? | `--config.branch=predictable` vs `unpredictable` |
| Cache bound? | `--config.dcache=hot,warm,cool,cold` (observation; whitelist contrast is hot/cold), then `-e L1-dcache-load-misses,LLC-load-misses` |
| TLB bound? | `--config.dtlb=hot,cold` (observation only; not whitelisted), then `-e dTLB-load-misses,iTLB-load-misses` |
| Which microarchitectural bound? | `-e 'topdown-*'` |
| Where does a whole program spend time? | `perf profile -f hot -e cycles -- ./a.out`, else system `perf record` + `perf report` |
| Which instructions are hot? | `perf analyze a.out:fizz_buzz -- perf.data` |
| Which state do those instructions run with? | `perf analyze a.out:fizz_buzz` (`data.<reg>`) and `--filter` |
| Can the binary be improved without a rebuild? | `perf optimize a.out --list -- data/`, then `perf optimize a.out -- data/` |
| Is the host set up for stable measurements? | `perf system tune --dry-run`, then `perf system tune` or `--profile=latency` |
| Is this change actually faster? | both runs `-o data/`, then `perf compare -b old -- data/` |
| What is the noise floor? | `perf view -s min,median,p10,p50,p90,p99 -- data/` |
| How do I look at the result again? | `perf view -- data/` or `perf plot -- data/` |

List every target before naming one:

```sh
perf benchmark a.out --list
perf benchmark a.out:foo --list            # regex filter
perf analyze a.out --list
perf profile --list -- ./a.out
```

A target is one `CODE` argument: `FILE:SYMBOL` or `FILE:begin..end` for a
binary symbol or region, `FILE:LABEL` or `FILE:begin..end` for an assembly
source, or a raw snippet when there is no `FILE:`. Put `PERF_LABEL(name)`
around suspect regions
(`lib/perf/perf.h` for C/C++, `lib/perf/perf.rs`, `lib/perf/perf.zig`); labels
emit no instructions and become counter-reading trampolines at startup. A
`foo_begin`/`foo_end` pair is one region; `foo_begin..foo_end` is the target.

### 4.2 `perf profile`

Track a live binary without recompiling. `--filter` is required and names the
symbols to track; the binary on disk is untouched. Repeat `--filter` for more
than one symbol.

```sh
perf profile --list -- ./a.out
perf profile -f hot -e cycles,branch-misses -- ./a.out --work 100
perf profile -f fizz_buzz -e cycles -o profile.json -- ./a.out
```

`-o` is the JSON trace file (default `profile.json`; `none` prints a table
instead of saving). `--json FILE` also emits the trace records; use it when
piping. `--buffer-size` is a fixed control across an A/B, not a tuning knob.

Detours reserve the 128-byte red zone; function returns use per-thread shadow
stacks, so overlapping threads return to the right caller. Use `perf profile`
for isolated function/region cost inside a real process; use system `perf
record` for whole-program attribution.

### 4.3 `perf benchmark`

```sh
perf benchmark a.out:fizz_buzz -m latency -e cycles,instructions
perf benchmark a.out:fizz_buzz -m throughput -e cycles,instructions
perf benchmark a.out:fizz_buzz --data.arg0=[1,3,5] -m latency -e cycles  # observation
perf benchmark a.out:fizz_buzz --config.branch=predictable,unpredictable
perf benchmark a.out:fizz_buzz --config.dcache=hot,cold
perf benchmark a.out:fizz_buzz --config.dtlb=hot,cold  # observation only
perf benchmark a.out:hot_begin..hot_end -m latency -e cycles
perf benchmark /usr/bin/tree:main -m latency --env LANG=C.UTF-8 -- /path/to/folder
```

`-o DIR` is the canonical output root; `--json FILE` is the portable envelope.
`-c -o FILE.o` stops after exploration and writes the execution image instead
of measuring.

A target must be callable on its own, and a program target must be
position-independent: the harness runs it once in a throwaway child and refuses
a target that faults, exits, or replaces the process when called on its own
(for example a program entry point such as `_start`). A non-PIE executable can
only be mapped at its link address, which usually collides with the harness, so
the run stops with that reason. Benchmark the surrounding function or a
`foo_begin..foo_end` region instead.

Program arguments go after `--`, literally, with `FILE` as `argv[0]` unless
the first word already equals it. `--data.rdi`, `--data.rsi`, and `--data.rdx`
(and their `arg0`/`arg1`/`arg2` aliases) are refused when `--` is used: the
process ABI already decides those registers.

### 4.4 `perf analyze`

`perf analyze` never runs the target. It disassembles it, explores its states
symbolically, and joins measured data (`perf.data`, profile or benchmark
records) onto the instruction rows; counter columns come from matching `ip`.

```sh
perf analyze a.out:fizz_buzz
perf analyze a.out:fizz_buzz -- perf.data
perf analyze a.out:fizz_buzz --filter 'latency > 4'
perf analyze a.out:fizz_buzz --filter '15 in `data.rdi`'
perf analyze a.out:fizz_buzz -e assembly | llvm-mca -mcpu=alderlake
perf analyze a.out:fizz_buzz --json analyze.json -- perf.data
```

`--json FILE` is the structured sink; `--output` is not supported by
`perf analyze`. The JSON envelope is one record per instruction.

`index` numbers the instructions from 0. The default columns are
`file,name,index,address,encoding,size,latency,throughput,assembly,data*` plus
joined counters; `encoding` is the emitted space-separated hex byte string
(`89 f8`). Every `data.*` cell is a list of the values the explored states held
(`[15]` when they agree); `--filter` takes a pandas query, and `in` tests a
list. Only the target's own instructions are listed: timing reads,
cache/TLB steering, register priming, and the call sequence are not attributed
to it. Use `perf analyze` to attribute a measured hot spot to instructions;
use `perf benchmark` for the per-operation cost of one.

### 4.5 `perf optimize`

`perf optimize` applies measured-data-guided passes. It is not a compiler.
`peephole` and `branch` keep each function body the same size (shorter
encodings are padded with multi-byte NOPs), so references outside the rewritten
`.text` stay valid. `layout`/`align` permute whole functions and fix up rel32
branches, RIP-relative memory, RELA/RELR addends, `.eh_frame` FDEs, the
`.eh_frame_hdr` lookup table, symbol tables, and the ELF entry point. A binary
with indirect jumps, absolute `.text` addresses, text relocations, or exception
tables keeps the original layout; non-PIE binaries skip `layout`/`align`; an
unfixable layout falls back to the original image while keeping the
size-preserving passes. Only statically reachable instructions are rewritten.

```sh
perf optimize a.out -- data/                           # writes a.opt.out
perf optimize a.out --output b.out -- data/            # measured hot layout
perf optimize a.out --output b.out --opt peephole,branch
perf optimize a.out --output b.out --opt layout --config.layout.align=64 -- data/
perf optimize a.out --list --json -- data/              # hotness per function
```

Passes: `peephole` (safe same-size simplifications, optimal NOPs), `branch`
(jump-to-next removal, transitive threading, conditional inversion), `layout`
(hot-first function order from benchmark/profile/`perf.data`/analyze data),
`align` (extra hot-function alignment; implies `layout`). `--output` is the
output binary; `--json` is the report, or with `--list` the hotness records.
Re-run the benchmark and `perf compare` after optimizing: an optimization that
does not clear the test is not a win.

### 4.6 System perf

The perf-labs tools measure *isolated* cost. For whole-program behaviour, use
system perf; `perf <command>` dispatches to `perf-<command>` when that script
exists and otherwise falls through to linux-perf.

```sh
taskset -c 4 perf stat -e cycles,instructions,cache-misses,branch-misses ./a.out
perf record -g -F 999 -e cycles:u -o perf.data -- ./a.out
perf report --stdio -g graph,4000 --sort symbol
perf annotate --stdio -s symbol.dso
perf c2c record -g -o c2c.data -- ./a.out
```

`perf system tune` profiles:

- `performance` — the common subset: performance governor, EPP, max frequency,
  user-space `rdpmc`, ASLR/watchdogs/NUMA/KSM/IRQ balancing off, perf limits
  exposed; leaves turbo/boost, SMT, THP, isolation and scheduler knobs at
  defaults.
- `latency` — low jitter: also turbo/boost off, SMT off, NMI/watchdog, NUMA
  balancing, THP/KSM/timer migration off, `swappiness=0`, 1/s samples, IRQ
  isolation and affinity checks.
- `throughput` — sustained throughput: turbo/boost and SMT on, THP always,
  IRQ balancing off, `swappiness=10`, 1/s samples.

Use `--dry-run` first, `--profile.KEY=VALUE` for one-off overrides, and
`perf system reset` to restore stock settings:

```sh
perf system tune --dry-run --profile=latency --json tune.json
perf system tune --dry-run --profile=throughput --json tune-throughput.json
perf system tune --dry-run --profile=latency \
  --profile.aslr=true --profile.cpu_frequency=max \
  --profile.governor=performance --profile.smt=false
printf '{"aslr": true}\n' > latency-profile.json
perf system tune --dry-run --profile=latency-profile.json
```

### 4.7 Reference invocations

`.github/workflows/perf.sh` is the end-to-end smoke suite: it exercises
`system info/tune`, `benchmark` (snippet, symbol, region, PIE, object file,
assembly source, libc, live binary with arguments), `profile`, `view`, `plot`,
`compare`, `analyze`, `optimize`, `perf stat` and `taskset`. It is the first
place to check exact flag spelling and a working invocation; `bin/README.md`
is the full reference. Its `--config.samples=1 --config.iterations=10` settings
are smoke settings for speed, not a statistical A/B: use at least 30 usable
rows per side before trusting `perf compare`. The studies in `studies/x86_64/`
are the reference for the analysis method, not a source of numbers to copy.

## 5. Reading `perf benchmark` output

Columns are the identity of the run followed by what it was measured with:

```
file  name  mode  config.*  data.*  operations  cycles  instructions
```

- `mode`: `latency` emits one row per timed call (one loop trip), so `perf
  view` can aggregate its distribution; `throughput` emits one row per
  whole-loop measurement. Both answer different questions.
- `operations`: denominator for every ratio (`cycles/operations`); 1 for a
  latency row, the auto-calibrated trip count for throughput.
- A `null` counter means it could not be read — never read it as `0`.
- `config.backend.<name>.*` records the resolved backend and its parameters,
  so a row can be replayed exactly.

`perf view` shows `time,file,name,mode,operations,duration_time/operations`
by default — the per-operation cost, not the raw counter — plus the `data.*`
grouping columns, aggregated by `-s` (default
`min,median,p10,p50,p90,p99,max`). `-e` selects columns or expressions;
`-g '' -s ''` gives raw rows. `perf plot` charts `<event>/operations` by
default, so plot the per-operation cost, not the raw counter.

## 6. Pitfalls seen in real reviews

- **Latency measured as throughput** (or the reverse). Compare like modes;
  merging both and reading one column mixes two populations.
- **Per-call overhead mistaken for loop cost.** Look at `latency` vs
  `throughput` before blaming the operation.
- **Percentiles from one sample.** A throughput run takes
  `--config.samples=N` samples (default 100); a latency run reports one row per
  loop trip. `perf compare` needs at least 30 samples per side.
- **Zero is a measurement.** A zero counter delta is kept, not dropped. When
  every run is below overhead, the fallback reports the overhead-corrected
  minimum clamped to zero with a warning.
- **Sweeping two axes and reading the corner.** One axis at a time; a
  factorial explosion is not attribution.
- **Cache state you did not choose.** The default sweeps
  `branch,dcache,dtlb,icache,itlb,code`; the default row is one sweep point.
- **Attributing harness cost to the target.** Harness timing and steering are
  subtracted differentially, but a long `call`/`ret` or a big prologue still
  shows up as target cost. Compare a neighbouring label before blaming a
  boundary.
- **`icache=cold` looking like `icache=hot`.** x86-64 has no user-mode way to
  flush the instruction cache (`clflushopt` only reaches the data hierarchy),
  so `icache` steers the code's data-hierarchy line and its translation.
  `itlb` and `topdown-fe-bound` say something about the instruction stream.
- **Region spanning labels that moved.** `--list` warns when `foo_end`
  precedes `foo_begin`; the measured span is still measured, but it is not the
  region the source suggests.
- **Governor drift between two runs.** Re-measure the baseline in the same
  session as the candidate; record the environment fingerprint.
- **Eyeballing medians instead of `perf compare`.** A non-significant delta is
  not a speedup.
- **Reusing memory without checking `state_hash` and `hardware_config`.** A
  past number is not a current measurement.
- **Two memories that disagree.** Keep one authoritative ledger (JSON or
  SQLite). A second store is an index that can be rebuilt; never let the index
  outlive the artifacts it points to.
- **Confusing `--output` with `--json`.** Benchmark `-o` is a directory;
  profile `-o` is a JSON file; optimize `-o` is a binary; plot `-o` is an image;
  analyze and compare take `--json` and no `--output`.
- **Reporting a model as a measurement.** `perf analyze` `latency` and
  `throughput` come from instruction tables, not from the counter run. Label
  them; use `perf benchmark` for measured cycles.
- **Treating a default sweep as an A/B.** With no explicit `--config`, the
  benchmark sweeps branch, caches, TLB and alignment. Pin the controls before
  comparing two builds.
- **Target stdout polluting the JSON stream.** Keep
  `--config.external.stdout=false` when using `perf benchmark --json`; the
  envelope must own stdout.

## 7. Reporting

Report like an engineer who wants to be believed:

```
fizz_buzz(n=1e6), latency, cpu 4, 26732 loop trips
state_hash:    sha256:9f2c... (a.out@b04f8237, bad-spec)
topdown old:   retiring 24%  bad-spec 21%  fe-bound 11%  be-bound 44%
topdown new:   retiring 71%  bad-spec  9%  fe-bound 10%  be-bound 10%
baseline old:  10.00 cycles/op  p10 9  p50 10  p90 11  (n=26732)
candidate new:  6.00 cycles/op  p10 5  p50 6   p90 7   (n=26732)
perf compare:  diff -40.0% [-41.2, -38.6] p=1e-5, significant at alpha=0.05
mechanism:     the hot mispredict is gone; bad-spec 21% -> 9%
artifacts:     .perf/experiments/.../{old,new,topdown,analyze,compare}.json
verdict:       accept; perf_experiments id 42 (H-7)
```

Include: the exact commands and artifact paths, the unit and denominator,
sample count and spread, the `perf compare` verdict, the top-down/analyze
evidence, the environment, and what is still unexplained. The compare `diff`
and its interval are signed percentages; state the sign convention. If a
question cannot be answered with the counters available, say so and name the
experiment that would answer it.

## 8. Working on this repository

- `src/perf/core.py` — event resolution, `perf_event_open`, RDPMC, affinity/
  priority/NUMA guards. `src/perf/bench.py` — symbolic exploration, harness
  JIT, cache/TLB/branch steering, run calibration, data-page mapping.
  `src/perf/mca.py` — `perf analyze` (numbered instructions, one row per
  instruction, list-valued `data.*` from the explored states). `src/perf/exec.py`
  — ELF loading, relocation, `to_object`. `perf_labels`/`asm_labels` in `info.py`
  map a binary's labels to addresses and an assembly source's labels to source
  positions and spans, which `bench.py` turns into a snippet behind
  `perf benchmark foo.s:foo..bar`.
  `src/perf/opt.py` — `perf optimize`: hotness aggregation, ELF rewrite and
  fixups (rel32, RIP-relative, RELA/RELR, `.eh_frame`/`.eh_frame_hdr`, symbols,
  entry point). `src/perf/arch/x86_64.py` — harness templates, counter reads,
  TSC/CPUID frequency, eviction/priming asm, `page_runs`, and the
  arch-specific optimizer primitives. `src/perf/arch` resolves architecture
  names (x86-64/x86_64/amd64) and selects the arch module. `src/perf/prof.py` —
  ptrace detours, ring buffer, shadow stack. `src/perf/info.py` — `perf system`
  (cpu topology, hostname, perf_labels, functions, `info.format_hz`).
  `src/perf/tune.py` — `perf system tune` profiles and `/sys`/`/proc`/`/dev`
  writes, service/IRQ/power/isolation checks. `src/perf/data.py` — result-frame
  schema, `perf.data.samples`, `query` with `in` over list columns.
  `src/perf/comp.py` — the CLT test.
  `src/perf/plot.py` — charts, and the sixel backend.
- Each command is one script named `perf-<command>`, which system `perf`
  dispatches to from `perf <command>`, so `perf benchmark` runs
  `bin/perf-benchmark`. A script owns its own parser and its own command and
  imports only the helpers it uses (`.perfconfig` for its own section, the
  result loading/formatting its input needs from `src/perf/`). There is no
  shared CLI (argparse) module — keep it that way, and keep a CLI script from
  importing another CLI script; importing shared helpers from `src/perf/` is
  the normal path (`perf plot` uses `perf.bench`, `perf view` uses
  `perf.plot`).
- A target is one `CODE` argument on the CLI. In Python, `perf.benchmark` and
  `perf.analyze` take a binary symbol as `symbol=` (a string like
  `"a.out:fizz_buzz"` or a `[file, symbol]` pair), a binary region as
  `region=` (a string like `"a.out:hot_begin..hot_end"` or a
  `[file, (begin, end)]` pair), and a raw snippet as `asm=`;
  `perf.profile` takes the executable as `cmd=` and the track points as
  `filter=`; `perf.optimize` takes the input binary as `exec=`;
  `df.to_object(path=...)` on a benchmark frame writes its binary symbol or
  region (the `perf.exec.to_object` helper takes `symbol=`/`region=`). There is
  no `target=` form for `benchmark`, `analyze`, `profile`, or `optimize` (and
  no `filter=` form for `optimize`, no `file=`/`code=` form for `benchmark` or
  `analyze`); do not add one back. Public API is `benchmark`, `profile`,
  `analyze`, `compare`, `optimize`, `plot`, plus `system`
  (`system.info`/`tune`/`reset`); a benchmark frame also has
  `to_json()` and `to_object()`. Binary metadata is `perf.info.metadata` and
  `perf.data` samples are `perf.data.samples`. `benchmark(mode=)` defaults to
  both modes like the CLI; `compare`/`plot` take `filter=` like the CLI `-f`.
- Result-frame identity columns are `data._IDENTITY_COLUMNS`
  (`file`, `name`, `mode`); columns that are never metrics are
  `data._NON_METRIC_COLUMNS`. Add new columns there, not at every use site.
- Development loop: `pytest --cov`, `lit -v -j2 --timeout 600 tests/x86_64`,
  `ruff check src tests bin`, `ruff format --check src tests bin`, and
  papermill over `studies/**/*.ipynb` (see `.github/workflows/linux.yml`).
  ruff is pinned to a release series in `pyproject.toml`; a ruff upgrade and
  the reformat it causes go in one commit. `tests/README.md` explains what is
  tested and what the suite enforces; `src/README.md` has worked, verified
  invocations of every command if a flag needs showing rather than describing.
  Keep the Python code blocks in every `*.md` ruff-format clean by hand.
- `studies/x86_64/**` are the worked notebook analyses (one per level-1
  top-down slot, plus a machine-characterization notebook); treat them as the
  reference for method and expected shape, never as numbers to copy.
  `studies/README.md` explains the method.
  `.github/workflows/perf.sh` is the end-to-end smoke suite and the first place
  to check exact flag spelling; `bin/README.md` is the full CLI reference.
  `lib/README.md` documents the `PERF_LABEL` annotations.
