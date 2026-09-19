# perf-labs/perf

[![Build](https://github.com/perf-labs/perf/actions/workflows/linux.yml/badge.svg)](https://github.com/perf-labs/perf/actions/workflows/linux.yml)

Symbolic benchmarking, live tracking on real hardware without changing the binary.

## Features

- Benchmark `assembly` or `functions`, `regions` in any compiled binary (`C++`, `Rust`, `Zig`, ...).
- CPU cache state (`L1d`/`L2`/`L3`/`DRAM`, `TLBd`/`TLBi`, `L1i`) and branch predictability `control`.
- Hardware counters via `RDPMC` (`cycles`, `instructions`, `cache-misses`, `branch-misses`, ...) and [top-down](https://rcs.uwaterloo.ca/~ali/cs854-f23/papers/topdown.pdf) analysis.
- Live tracking (`perf track`) of any `function`, `region` in any compiled binary.
- Central Limit Theorem null-hypothesis tests (`perf compare`).
- View/plot (`perf view`, `perf plot`) results in the terminal (`sixel`), and interact via IPython (`--interactive`) or Jupyter notebooks.

## Requirements

- x86-64 Linux, kernel 6.x+
- Python 3.11+ (see [pyproject.toml](pyproject.toml))
- `linux-perf` and user-space `rdpmc` access (see [Setup](#setup))

## Install

```sh
pip install git+https://github.com/perf-labs/perf.git
```

For development:

```sh
git clone https://github.com/perf-labs/perf.git && cd perf
python3 -m venv .venv && . .venv/bin/activate
pip install -e .[test]
pytest
ruff check src tests bin
ruff format --check src tests bin
```

## Setup

```sh
echo 2 | sudo tee /sys/devices/cpu_core/rdpmc
```

This enables user-space `RDPMC` hardware counters without syscalls on the hot path.
This is not required for `duration_time` (Time-Stamp Counter).

## Quick start

```sh
perf info cpu
perf info a.out
```

```sh
# bench assembly
perf bench asm 'imul eax, 0'

# bench function
perf bench a.out fizz_buzz
perf bench a.out fizz_buzz --event cycles,instructions
perf bench a.out fizz_buzz --event topdown

# bench region
perf bench a.out hot_begin..hot_end
perf bench a.out 0x401000..0x401020
```

```sh
# track live
perf track -- ./a.out
perf track -e cycles,branch-misses -o track.json -- ./a.out
```

```sh
# view / plot
perf bench a.out fizz_buzz | perf view
perf bench a.out fizz_buzz | perf plot
perf bench a.out fizz_buzz_v1 --output data/
perf bench a.out fizz_buzz_v2 --output data/
perf compare --baseline fizz_buzz_v1 -- data/
perf view -- data/
perf plot -- data/
```

```py
# python
import perf

df = perf.bench(file="a.out", target="fizz_buzz", mode="latency", event=["duration_time"])
df.duration_time.describe()

df = perf.bench(asm="mov eax, 42", mode="latency", event=["cycles"])
df.cycles.plot()
```

## CLI

### `perf info`

`perf info cpu` shows CPU topology, frequency, and cache sizes.
`perf info a.out` returns all binary metadata (labels, functions, regions).

```sh
perf info cpu
perf info a.out
perf info a.out --json
```

The benchmarkable targets of a binary (labels and functions) are
also listed with `perf bench a.out --list` and `perf track --list -- ./a.out`
(see [perf bench](#perf-bench-asm--func--region)).

### `perf bench` (asm, function, region)

Measure an assembly snippet, function, or region (of any compiled binary) with
hardware counters.

```sh
perf bench a.out fizz_buzz --event cycles,instructions
perf bench a.out fizz_buzz --event cycles --event instructions
perf bench a.out hot_begin..hot_end
perf bench a.out fizz_buzz --event topdown
perf bench asm 'imul eax, 0' --mode latency
perf bench a.out fizz_buzz --mode throughput
perf bench a.out --list
perf bench mem basic
```

```sh
# hot - everything served from L1
perf bench asm 'mov rax, [rdi]' --config.cache=hot --event cache-misses,cycles
perf bench a.out fizz_buzz --config.cache=cold

# cold load from a fixed address
perf bench asm 'mov rax, [rdi]' \
  --data.rdi=0x42000000000 --data[0x42000000000]=123 \
  --config.cache=cold --event cache-misses,cycles

# branch predictability
perf bench a.out fizz_buzz --data.arg0=1
perf bench a.out fizz_buzz --data.arg0=3
perf bench a.out fizz_buzz --data.arg0=[1,3,5]
```

| Option | Meaning |
| --- | --- |
| `TARGET` | func/region to benchmark (`a..b` is a region; `FILE` first, `TARGET` second; `asm` benchmarks raw code) |
| `--list` | list labels/functions of the binary instead of measuring |
| `-m MODE`, `--mode` | `latency`, `throughput`, or both (default: both) |
| `-e EVENT`, `--event EVENT` | comma-separated events share one measurement; a repeated `-e` is a separate measurement; `topdown` expands to `topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound` (one group, slots-leader added automatically) (see [Events](#events)) |
| `-n NAME`, `--name NAME` | benchmark name; labels the result `name` column |
| `--backend {loop,unroll}` | `loop` (one snippet/call per iteration) or `unroll` (2N-N differential; factor via `unroll_n`) |
| `--config ...` | config JSON file plus dotted overrides (see [Configuration](#configuration)) |
| `--data ...` | input data as JSON string/file plus dotted overrides (see [Input data](#input-data)) |
| `--setup ASM` / `--teardown ASM` | asm run before/after each iteration |
| `-o OUTPUT`, `--output OUTPUT` | output root (see [Output format](#output-format)); for `-S`/`-c` it names a direct file |
| `--json` | emit the JSON envelope to stdout |
| `-S` | print disassembly of the measured snippet (`... -S \| llvm-mca`) |
| `-c` | emit the relocatable object (`-o bench.o`, then link it yourself) |
| `--debug` | print config, found solutions, synthesized data, full assembly and per-run results to stderr |
| `-i` | interactive IPython session with `df` |

`--config` takes a JSON file plus dotted overrides; `perf.bench(config=...)`
takes the same dict. Keys:

| Key | Meaning / examples |
| --- | --- |
| `cache` | `hot`, `warm`, `cool`, `cold`, or per-tier hit rates, e.g. `{"L1d": 100}`. `TLBd`/`TLBi` tiers are flushed precisely with `mprotect` PTE-protection toggles (a ring-3 alternative to the privileged `invlpg`), keeping measurement noise low without flooding the TLB |
| `branch` | `predictable` / `unpredictable` (plus `unpredictable.exponential`; `unpredictable.uniform` is an alias for `unpredictable`), globally or per `mem`/`regs` entry |
| `thread.affinity` | CPU pinning, e.g. `[1]` or `"0,1"` |
| `thread.priority` | nice value or `lowest`/`low`/`normal`/`high`/`highest` |
| `func.align` / `func.order` | function alignment / `as-is` or `random` order |
| `code.align` | measured-code alignment |
| `stack.size` / `stack.align` | symbolic-execution stack setup |
| `samples` / `runs` / `probe_runs` | samples per series, measurement runs, calibration runs |
| `iterations` / `min_iterations` / `max_iterations` / `target_rel_se` | loop-trip calibration (auto by default) |
| `backend` / `unroll_n` | `loop` or `unroll`; unroll factor |
| `seed` | deterministic input sampling |

Config
```json
{
    "thread": {
        "affinity": null,
        "priority": null
    },
    "stack": {
        "size": 2097152,
        "align": 16
    },
    "code": {
        "align": 16
    },
    "func": {
        "align": 16,
        "order": "as-is"
    },
    "cache": {
        "L1i": {"hit_rate": 100},
        "L1d": {"hit_rate": 100},
        "L2":  {"hit_rate": 100},
        "L3":  {"hit_rate": 100}
    },
    "branch": "unpredictable",
    "samples": 100,
    "runs": 10
}
```

`--data` takes a JSON string/file plus dotted overrides; `perf.bench(data=...)`
takes the same dict with `regs` and `mem` sections. Values may be scalars or
lists (one sample per entry, cycling/predictable per the `branch` config).

```sh
# rdi = 15, or sweep several inputs in one run
perf bench a.out fizz_buzz --data.rdi=15
perf bench a.out fizz_buzz --data.rdi=[1,3,5]

# memory operand at a fixed address
perf bench asm 'mov rax, [rdi]' \
  --data.rdi=0x42000000000 --data[0x42000000000]=123
```

With no explicit `data`, inputs are discovered symbolically: every code path
is explored (IR + SMT solving) and sampled each iteration, so a single run
covers all branches.

Data
```json
{
    "rdi": 15,
    "rsi": "0xFF",
    "0x42000000000": 123,
    "0x42000000001": [1, 2, 3]
}
```

### `perf track`

Track functions and addresses in a live process with single-startup ptrace and native `rdpmc`/`rdtsc` trampolines
(see [Annotations](#annotations-libperf)). Labels, hex addresses and function
entries are patched with detours that relocate the overwritten instructions
(RIP-relative fixups, short-branch expansion) and resume after them, so any
address can be tracked. With no filter, every label in the binary is tracked.
A function filter (e.g. `-f fizz_buzz`) patches only the entry and intercepts
the return address through a shadow stack, so all `ret` sites are covered
with a single patch.

```sh
perf track -- ./a.out --work 100
perf track -f hot -f cold -e cycles,branch-misses -o track.json -- ./a.out
perf track -f fizz_buzz -- ./a.out
perf track -f work_begin..work_end -- ./a.out
perf track -e topdown -- ./a.out
perf track --list -- ./a.out
```

| Option | Meaning |
| --- | --- |
| `-f FILTER`, `--filter FILTER` | only track these targets (repeatable): label/function names, hex addresses, or `begin..end` regions (sides may mix; a bare function name tracks entry/exit and is named after the resolved symbol, e.g. `-f fizz_buzz` → `fizz_buzz(int)`; default: all labels) |
| `-e EVENT` | events via `rdpmc` (default: `cycles`; `duration_time` uses `rdtsc`; `topdown` expands to the four level-1 topdown events) |
| `--list` | list trackable labels/functions of the binary instead of measuring (`track --list -- ./a.out`) |
| `-o OUTPUT` | JSON output path (default: `track.json`; `none` prints a table instead of saving) |
| `--buffer-size N` | ring-buffer size in slots, rounded up to a power of two (default: `65536`) |
| `-i` | interactive IPython session |

### `perf view`

View saved or piped data with data frames.

```sh
perf view -- data/
perf view --stat p50,p99 --event cycles/instructions -- data/
perf bench a.out fizz_buzz | perf view
```

| Option | Meaning |
| --- | --- |
| `-e EVENT`, `--event EVENT` | columns/expressions, e.g. `cycles`, `cycles/instructions`; `<group>.<metric>` reads another group's rows sample-aligned (speedup vs baseline) |
| `-g GROUPBY`, `--group-by GROUPBY` | pandas groupby keys (default: `file,name,mode`; `''` for raw rows) |
| `-f FILTER`, `--filter FILTER` | pandas query filter, e.g. `--filter 'name == "fizz_buzz"'` |
| `-s STAT`, `--stat STAT` | aggregation (default: `min,median,p10,p50,p90,p99,max`; `''` for raw rows) |
| `-c COLUMN`, `--column COLUMN` | always-visible columns (default: `time,file,name,mode,samples`) |
| `-i` | interactive IPython session |

### `perf plot`

Chart saved or piped measurements (terminal via `sixel`, or save to file).

```sh
perf plot -- data/
perf plot --type ecdf --type bar --event cycles --event instructions -- data/
perf bench a.out fizz_buzz | perf plot
```

| Option | Meaning |
| --- | --- |
| `-e EVENT`, `--event EVENT` | metric/expression per chart column; comma overlays, repeated `-e` adds columns (default: one column per measured event as `<event>/operations`) |
| `-x XAXIS`, `--xaxis XAXIS` | x-axis column, e.g. `-x data.rsi` for scaling over a parameter |
| `-t TYPE`, `--type TYPE` | chart types (`ecdf`, `bar`, `boxen`, `hist`, `line`, `point`, `scatter`, ...); comma overlays, repeated `-t` adds charts |
| `--logx` / `--logy` | log scales |
| `-g GROUPBY`, `--group-by GROUPBY` | hue grouping (default: `file,name,mode`) |
| `-f FILTER`, `--filter FILTER` | pandas query filter |
| `--config CONFIG` | plot config JSON (matplotlib style/rcParams) |
| `-o OUTPUT`, `--output OUTPUT` | save to file (`chart.png`, `chart.pdf`, or a directory for one file per chart) |
| `-i` | interactive IPython session |

### `perf compare`

Compare data with a null-hypothesis test built on the Central Limit Theorem:
a two-sided z-test on the arithmetic mean plus a two-sided z-test on the geometric mean.
Both must reject (`p = max(p_mean, p_gmean)`) before a change reports as `significant`
(`p_value < alpha`), which keeps repeated runs of the same binary stable.

```sh
perf compare -- data/ --baseline base
perf compare -- data/ -e cycles -e instructions --alpha 0.01 --json
```

| Option | Meaning |
| --- | --- |
| `-e EVENT` | metrics to compare (default: all numeric metrics, minus run metadata) |
| `-f FILTER`, `--filter FILTER` | pandas query filter |
| `-b BASELINE` | baseline name others are compared against (default: first sorted) |
| `--alpha ALPHA` | significance level for H0 rejection (default: `0.05`) |
| `--json` | JSON records instead of a table |
| `-i` | interactive IPython session |

## Python API

All public entry points are re-exported from `perf`:

```py
import perf

# benchmark: binary/assembly file target / asm snippet (a `..` selects a region,
# and a tuple target is the same region: ("begin", "end"))
df = perf.bench(file="a.out", target="fizz_buzz", mode="latency", event=["duration_time"])
df = perf.bench(asm="mov eax, 42", mode="latency", event=["cycles"])
df = perf.bench(file="a.out", target="hot_begin..hot_end", mode="latency",
                event=["topdown"], data={"regs": {"rdi": 15}})
df = perf.bench(file="a.out", target=("hot_begin", "hot_end"), mode="latency")
df = perf.bench(file="a.s", target="myfunc", mode="latency")

# disassembly / relocatable object of the measured snippet
text = perf.disassemble(file="a.out", target="fizz_buzz")
path = perf.to_object(file="a.out", target="fizz_buzz", path="bench.o")

# the JSON run envelope as a string (also used by `perf bench --json`):
# identical to the CLI's persisted format, so library and CLI stay in sync
env = perf.to_json(df, indent=4)

# binary/cpu metadata, labels, regions, targets
cpu = perf.cpuinfo()
meta = perf.metadata("a.out")
labels = perf.labels("a.out")

# live tracking (labels, functions, hex addresses or (begin, end) regions)
df = perf.track(cmd=["./a.out"], event=["cycles"])
df = perf.track(cmd=["./a.out"], event=["cycles"], filter=["hot"])
df = perf.track(cmd=["./a.out"], event=["cycles"], filter=["fizz_buzz"])
df = perf.track(cmd=["./a.out"], event=["cycles"], filter=[("work_begin", "work_end")])

# compare
cmp = perf.compare(df, events=["cycles"], baseline="base")

perf.plot(df, ["ecdf"], [["cycles"]])
```

## How it works

`bench` and `track` use the same primitives: native counters read without a
syscall, and code patched live in the measured process.

`perf bench` builds a harness around the measured snippet.

```asm
; iteration prologue
push rcx
lfence
rdtsc
mov <base>, rax         ; baseline in a spare register
lfence
pop rcx
; ... measured body ...
; iteration epilogue
push rcx
rdtscp
lfence
sub rax, <base>         ; delta
mov [r10], rax          ; record sample; r10 advances per iteration
add r10, 8
pop rcx
```

Before measuring, `bench` discovers what to feed the snippet:

- Symbolic execution: the target (function, `begin..end` region, or `asm`
  snippet) is lifted to VEX IR and explored with `angr` (`call_state` for
  functions so the ABI prototype is honoured, `blank_state` plus a mapped
  stack otherwise; `--setup` asm is prepended as a jump stub). All general
  purpose registers start as symbolic (`claripy.BVS`), explicit `--data`
  registers are constrained, and every memory read/write is recorded with
  breakpoints. Exploration runs until each `ret` (or the snippet end) is
  found, so one run covers every code path.
- Solver: each found state is solved with the SMT solver. Registers,
  read/write addresses, lengths, and values are evaluated to concrete
  models; a symbolic return value forks the state once per leaf. For
  `branch.*.mem = predictable` addresses the solver also records
  `branch_deps` (which registers/memory each conditional branch consumes),
  so predictable runs can pin exactly those inputs.
- VEX: the executed basic-block addresses (`bbl_addrs` history) select the
  measured code — `-S` prints only executed blocks, and static tables
  (`.rodata`) found by scanning the range are added as extra addresses.
- Data synthesis: models become a per-iteration table (one column per
  register, plus value + cache-tier columns per address). Each iteration
  picks one model (round-robin when `predictable`, random sampling
  otherwise), overlays explicit `--data.rdi` / `--data[addr]` values, and
  writes values plus the sampled tier into the table the harness reads via
  `prime`/`steer` asm. With no `--data`, sampling cycles through all
  paths/values, so branches are covered automatically.
- Cache eviction: every address gets a tier per iteration (`L1d`/`L2`/`L3`/
  `DRAM`, plus `L1i` and `TLBd`/`TLBi`). `hot` keeps everything in `L1`,
  `cold` flushes to `DRAM`; per-tier hit rates or per-address levels split
  addresses across tiers. Emitted asm is `mov`+`mfence` for `L1d`,
  `clflushopt`+`prefetcht1/t2` for `L2`/`L3` (`cldemote` when available),
  `clflushopt` for `DRAM`, and ring-3 `mprotect` PTE-protection toggles for
  `TLB` (a non-privileged `invlpg` alternative), followed by `pause` settle.
- Branch data: `branch` selects predictability globally or per `regs`/`mem`
  entry (`predictable`, `unpredictable`, `unpredictable.exponential`;
  `unpredictable.uniform` is accepted as an alias for `unpredictable`).
  `predictable` replays models/values in order and
  pins branch dependencies; the rest sample an index at random (exponential
  shaped when asked), so the same binary measures both
  best-case and branch-miss-heavy behaviour without changing code.

`perf track` patches each tracked address with a 5-byte direct jump to a
trampoline that claims a slot in a shared per-process ring buffer and records
the counter deltas. A label detour looks like this:

```asm
pushfq                          ; keep flags out of the measured path
push rax
push rcx
push rdx
push r11
mov  r11, <ring_base>
mov  rax, [r11 + 8]             ; head
mov  rcx, rax
and  rcx, <cap_mask>
shl  rcx, <slot-stride-log2>    ; byte offset of the slot
lea  rdx, [rax + 1]             ; reserve slot
mov  [r11 + 8], rdx
lea  r11, [r11 + rcx + 64]      ; slot address (64-byte header at buf start)
mov  dword ptr [r11], <point_id>
mov  qword ptr [r11 + 8], rax   ; sequence number
lfence
mov  ecx, <pmc-index>
rdpmc                           ; raw counter (or rdtsc for duration_time)
shl  rdx, 32
or   rax, rdx
mov  [r11 + 16], rax
pop  r11
pop  rdx
pop  rcx
pop  rax
popfq
; <relocated original instructions>
jmp  <resume address>
```

The overwritten instructions are disassembled, relocated (RIP-relative
operands fixed up, short branches expanded to their rel32 forms), and emitted
between the recording block and the jump back, so the process resumes
exactly where it left off. Function entry patches additionally push the real
return address onto a shadow stack and rewrite the saved slot with the
trampoline's exit stub, so every `ret` is intercepted by one patch. The
1 MiB trampoline cave is mapped within 1 GiB (`2^30` bytes) of the binary; the original
file on disk is untouched.

### Events

Any `perf list` event works (`cycles`, `instructions`, `cache-misses`,
`branch-misses`, `branch-instructions`, raw `rXXXX`, `cpu/.../` PMU paths,
`:u`/`:k`/`:p` modifiers), plus `duration_time` (nanoseconds via `rdtsc`,
the default; needs no PMU). When no `-e` is given to `perf view` /
`perf compare`, an `<event>/operations` ratio is derived for
every measured event alongside the raw metrics (e.g. `cycles/operations`,
`instructions/operations`, `duration_time/operations`). `perf plot` with no
`-e` shows only `<event>/operations` (one chart per measured event, e.g.
`duration_time[ns]/operations`); pass `-e` explicitly to plot raw counters
or custom expressions.

For a top-down breakdown pass `--event topdown` (or `event=["topdown"]` in
`perf.bench`/`perf.track`) instead of listing the four level-1 events:

```sh
# separate measurement groups (repeated -e), plus a topdown group
perf bench a.out fizz_buzz \
  --event cycles,instructions \
  --event topdown
```

`topdown` is an alias for
`topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound`
(one group, slots-leader added automatically). `topdown/operations`
expands to one `/operations` expression per event, so `view`/`plot`
divide each event separately.

A repeated `-e` is a separate measurement (separate counter scheduling), so
group events that must be read together in a single `-e`.

### `.perfconfig`

CLI defaults can live in `~/.perfconfig` (or `$PERF_CONFIG`):

```ini
[default]
mode = latency,throughput

[bench]
mode = latency

[compare]
baseline = base

[plot]
config.style = dark_background
```

Each section names a subcommand (`bench`, `view`, `plot`, `compare`,
`info`, `track`); `[default]` applies to all of them. Keys are long
option names without `--` (e.g. `mode`, `baseline`, `config.style`),
including dotted `--config.*`/`--data.*` overrides. Explicit CLI flags
always win over the file.

### Output

Each run emits a JSON envelope:

```json
{
    "file": "a.out@b04f8237",
    "name": "fizz_buzz-<run-id>",
    "id": "<run-id>",
    "time": "2026-01-01 00:00:00",
    "info": {"cpu": {}, "binary": {}},
    "config": {},
    "mode": "latency",
    "data": {"regs": {}, "mem": {}},
    "output": [{"samples": 0, "operations": 1, "iterations": 128, "cycles": 42}]
}
```

`perf.bench(...)` returns the same `output` rows as a pandas `DataFrame`
with a `file`/`name`/`mode` index and only these columns, in order:
`config.*`, `data.*`, `iterations`, `samples`, `operations` and the
measured event columns. The full run `config`/`data`/`info` stay in
`df.attrs` and in the JSON envelope.

### Annotations (C++, Rust, Zig)

- C++:  `lib/perf/perf.hpp` - `PERF_LABEL(name)`
- Rust: `lib/perf/perf.rs`  - `perf_label!(name)`
- Zig:  `lib/perf/perf.zig` - `perf_label(name)`

Labels are portable section metadata present in every language binding, and
are discoverable by `perf bench a.out --list`, `perf info a.out` and `perf track --list -- ./a.out`:

```cpp
#include <perf.hpp>

int work(int n) {
    auto s = 0;
    PERF_LABEL(loop_begin);
    for (int i = 0; i < n; i++) {
        s += i;
    }
    PERF_LABEL(loop_end);
    return s;
}
```

Labels produce zero instructions and only add section metadata, which
`perf bench`/`perf track` rewrite into counter-reading trampolines at startup
(the binary on disk is untouched). `foo_begin`/`foo_end` pairs become one
region sample each; whole functions (without any label) can be tracked the
same way by name.

## Resources

- [References](https://github.com/perf-labs/perf/wiki/references)
- [Studies](studies)

## License

[MIT](.github/LICENSE)
