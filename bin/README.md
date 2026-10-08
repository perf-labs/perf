# perf(1)

## Name

`perf` — symbolic benchmarking, live profiling, and per-instruction analysis
on x86-64 Linux.

## Synopsis

```sh
perf info [FILE] [-e EVENT] [--json [FILE]] [-i]
perf benchmark [CODE] [-m MODE] [-e EVENT] [-n NAME] [--backend BACKEND]
    [--config ...] [--data ...] [--setup CODE] [--teardown CODE]
    [-o OUTPUT] [--json [FILE]] [-c] [--debug] [-i]
    [--env KEY=VALUE] [-- ARGS...]
perf profile [-- COMMAND ...] [-t TARGET] [-e EVENT] [-o OUTPUT]
    [--buffer-size N] [--json [FILE]] [-i]
perf view [-- DATA ...] [-e EVENT] [-g GROUPBY] [-f FILTER] [-s STAT]
    [--json [FILE]] [-i]
perf plot [-- DATA ...] [-e EVENT] [-x XAXIS] [-y YAXIS] [-t TYPE] [--logx]
    [--logy] [-g GROUPBY] [-f FILTER] [--config CONFIG] [-o OUTPUT]
    [--json [FILE]] [-i]
perf compare [-- DATA ...] [-e EVENT] [-f FILTER] [-b BASELINE] [--alpha ALPHA]
    [--json [FILE]] [-i]
perf analyze [CODE] [-- DATA ...] [-n NAME] [--setup SETUP] [--teardown TEARDOWN]
    [-f FILTER] [--config CONFIG] [--data ...] [-e EVENT] [-g] [--json [FILE]] [-i]
```

## Description

Each command is installed as a `perf-<command>` script (`perf-benchmark`,
`perf-analyze`, `perf-view`, `perf-plot`, `perf-compare`, `perf-info`,
`perf-profile`), which linux-perf dispatches to by name, so this document
always writes them with a space: `perf benchmark a.out:func`.

## Commands

### perf info

#### Synopsis

```sh
perf info [FILE] [-e EVENT] [--json [FILE]] [-i]
```

#### Description

`perf info cpu` shows CPU topology, frequency, and cache sizes. The
`freq` column is a single number in Hz, read from CPUID leaf 0x15 (the exact
TSC frequency) and falling back to a short wall-clock measurement off the TSC
and then to the frequency the OS reports (`--json` prints it, the table
abbreviates it to `2.7Ghz`).
`perf info a.out` returns all binary metadata (labels, functions): one
row per target with `kind`, `begin`, `end` and `name`, plus a `size`
column that only functions have (a label is an address, so it has no
length; `--json` reports it as `null`).

The benchmarkable targets of a binary (labels and functions) are listed by
`perf info a.out`, and a target that is not found prints the same table before
it exits (see [perf benchmark](#perf-benchmark)).

#### Options

| Option | Meaning |
| --- | --- |
| `FILE` | `cpu` for CPU info, or a binary to inspect |
| `-e EVENT`, `--event EVENT` | columns to show, comma-separated or repeated, with `*` expanding a pattern, e.g. `-e cpu,core,L1d`, `-e 'L*'` (default: all) |
| `--json [FILE]` | emit JSON records to stdout, or to FILE when a path is given, instead of a table |
| `-i`, `--interactive` | interactive IPython session with `df` |

#### Examples

```sh
perf info cpu
perf info a.out
perf info a.out --json
perf info cpu -e cpu,core,L1d
perf info a.out -e 'name,begin,end'
```

### perf benchmark

#### Synopsis

```sh
perf benchmark [CODE] [-m MODE] [-e EVENT] [-n NAME]
    [--backend BACKEND] [--config ...] [--data ...] [--setup CODE]
    [--teardown CODE] [-o OUTPUT] [--json [FILE]] [-c] [--debug] [-i]
    [--env KEY=VALUE] [-- ARGS...]
```

#### Description

Measure an assembly snippet, function, region (of any compiled binary), or the
label/region of an assembly source with hardware counters. The target is one
`CODE` argument: `FILE:TARGET`, or a raw snippet with no `FILE:` at all.

`FILE` may be any x86-64 ELF the machine runs: a program, an object or archive
(linked into a temporary copy that keeps the original name, so `q.o` still
reports as `q.o`), or a shared library, in which case `TARGET` is one of its
exported symbols (`perf info libstdc++.so.6` lists them). A symbol that is an
indirect function (`strlen`, `memcpy`, anything glibc picks per CPU) resolves to
the implementation the loader installed, so `perf benchmark libc.so.6:strlen`
measures the one this machine runs. A program has to be position-independent:
the harness maps the target file into the measuring process, and a non-PIE
executable can only be mapped where it is linked, which for the usual `0x400000`
is where the interpreter already is — the run then stops with that reason
rather than overwriting the process.

The harness calls the target directly, so a target has to be callable on its
own. Before measuring anything it is run once in a throwaway child, and the run
is refused with the reason if the child faults, exits, replaces the process or
does not come back — calling a program entry point (`_start`), or a library
routine that runs on state the harness does not own (the harness maps the
target's own file, so `malloc` sees a fresh arena), would otherwise take the
measurement down with it. In that case benchmark the surrounding function, or
restrict the target to a `foo_begin..foo_end` region.

#### Options

| Option | Meaning |
| --- | --- |
| `CODE` | what to benchmark: `FILE:TARGET` for a func or a `begin..end` region of a binary (`a.out:func`, `a.out:hot_begin..hot_end`, `a.out:0x401000..0x401020`), `FILE:LABEL` (or `FILE:begin..end`) for an assembly file (`a.s:label`), or a raw snippet when there is no `FILE:` (`perf benchmark 'mov eax, 42'`); a bare `FILE` is an error, use `perf info FILE` to list its targets |
| `-m MODE`, `--mode MODE` | `latency` and/or `throughput` (comma-separated or a repeated `-m`, default: both); each mode is measured against every config combination |
| `-e EVENT`, `--event EVENT` | comma-separated events share one measurement; a repeated `-e` is a separate measurement; a wildcard/regex expands to every matching event, e.g. `topdown-*` expands to `topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound` (one group, slots-leader added automatically) (see [Events](#events)) |
| `-n NAME`, `--name NAME` | benchmark name; labels the result `name` column and picks the `<name>-<id>` directory under the output root |
| `--backend {loop,unroll}` | `loop` (one snippet/call per iteration) or `unroll` (2N-N differential; factor via `--config.backend.unroll.count`). For a snippet the backend follows the mode: `latency` uses `unroll`, `throughput` uses `loop`, unless you say otherwise |
| `--config ...` | config JSON file plus dotted overrides (see [Config](#config)) |
| `--data ...` | input data as JSON string/file plus dotted overrides (see [Data](#data)) |
| `--setup CODE` | setup run before each iteration: `FILE:TARGET` of the benchmarked file, or an asm snippet for a snippet target |
| `--teardown CODE` | teardown run after each iteration, same forms as `--setup` |
| `--env KEY=VALUE` | environment variable for the benchmarked program, repeated per variable; goes into `envp` next to `argv` (default: an empty environment) |
| `-- ARGS...` | the benchmarked program's own arguments, after everything else (see [Program arguments](#program-arguments)) |
| `-o OUTPUT`, `--output OUTPUT` | output root (see [Output](#output)); for `-c` it names a direct file |
| `--json [FILE]` | emit the JSON envelope to stdout, or to FILE when a path is given; otherwise a result table is printed |
| `-c` | stop after explore: write the execution image object file (use `-o a.o`) |
| `--debug` | print config, found solutions, synthesized data, full assembly and per-run results to stderr |
| `-i`, `--interactive` | interactive IPython session with `df` |

#### Program arguments

Everything after `--` is handed to the benchmarked program, the way a shell
hands `$1`, `$2`, ... to a program:

```sh
perf benchmark /usr/bin/tree:main -m latency -- /path/to/folder
```

runs `main(2, ["/usr/bin/tree", "/path/to/folder"], envp)` once per iteration. The
words after `--` are `argv` as a shell would pass them, so the first one is
`argv[0]`; a bare `--` with nothing after it means `argv[0]` is the `FILE` of
`CODE` and `argc` is 1. `perf.benchmark(argv=[...], env=[...])` takes the same
two lists.

- `argc` goes in `rdi`, `argv` in `rsi` and `envp` in `rdx`, as the System V
  ABI prescribes, so a program's `main` sees them exactly as it would from
  `__libc_start_main`.
- The vectors and the strings they point at are written into the harness's own
  stack mapping (`--config.stack.size`, 2 MiB by default), below the return
  address the harness pushes, with the auxiliary vector (`AT_PHDR`, `AT_PHENT`,
  `AT_PHNUM`, `AT_PAGESZ`, `AT_ENTRY`, `AT_NULL`) filled in from the mapped ELF.
  `getauxval()` therefore answers for the image that is being measured.
- `--env KEY=VALUE` adds to `envp`; with no `--env` the environment is empty
  (`envp[0] == NULL`), the same as `env -i`.
- `--data.rdi`, `--data.rsi` and `--data.rdx` (and their `arg0`/`arg1`/`arg2`
  aliases) are refused while `--` is used: the process ABI already decides what
  those three hold. Everything else in `--data` still applies.
- `--` is only for the program's arguments. Options for `perf benchmark` come
  before it, so a program that takes `--help` or `-v` needs no escaping.
- Words after `--` are taken literally, never glob-expanded or de-quoted by
  `perf`; the shell expands them first, as usual.

```sh
# a program that reads its environment as well as its arguments
perf benchmark a.out:main -m latency --env LANG=C.UTF-8 -- /path/to/folder --help

# the same from Python (argv is literal, including argv[0])
perf.benchmark(
    target="a.out:main",
    mode=["latency"],
    argv=["a.out", "/path/to/folder", "--help"],
    env=["LANG=C.UTF-8"],
)
```

#### Config

`--config` takes a JSON file plus dotted overrides; `perf.benchmark(config=...)`
takes the same dict. Keys:

| Key | Meaning / examples |
| --- | --- |
| `dcache` | data-side cache residency: `hot` (all from `L1d`), `warm` (from `L2`), `cool` (from `L3`), `cold` (from DRAM), a single `{"hit_rate": 50}`, or per-level hit rates, e.g. `{"L1d": 100}`. It is the only cache key with levels, so it is also the only one taking a level name. All four are swept by default |
| `icache` | instruction-side residency of the measured code (`L1i` under the hood): `hot`, `warm`, `cool`, `cold`, `50`, or `{"hit_rate": 50}`. Applies to the pages the measured code itself lives at. Note that x86-64 has no user-mode instruction to flush `L1i` (`clflushopt` only reaches the data hierarchy), so this steers the code's *data-hierarchy* line and its translation; the instruction cache itself is only displaced by code pressure |
| `dtlb` | data-translation residency (`TLBd` under the hood): `hot`, `warm`, `cool`, `cold`, `50`, or `{"hit_rate": 50}`. Translation tiers are flushed precisely with `mprotect` PTE-protection toggles (a ring-3 alternative to the privileged `invlpg`), keeping measurement noise low without flooding the TLB. The pages that need it are coalesced into one `mprotect` range (and mapped as one mapping up front), so a target touching many pages costs a syscall per cluster, not per page |
| `itlb` | instruction-translation residency (`TLBi` under the hood), same forms as `dtlb`; applies to the pages the measured code itself lives at |
| `external.lib` | `false` (default) leaves imports as opaque externs and never touches a shared library; `true` models the target's real library code with angr. This replaces the old `lib` key |
| `external.stdout` | `false` (default) discards everything the target writes to stdout while it is measured; `true` lets it through to your terminal (or to `perf benchmark --json`, which must own stdout). The program's own buffering is discarded too, so nothing it queued reaches the terminal once the run is over |
| `external.stderr` | as `external.stdout`, for stderr. Set either to `true` to watch a target that reports what it is doing |
| `branch` | `predictable` (in order) / `unpredictable` (at random), globally or per `mem`/`regs` entry. Those are the only two values. The global default is the sweep of both; `predictable` is what a run falls back to when nothing sets the global prediction (a `branch` dict with only `mem`/`regs`, or a collapsed sweep) |
| `thread` | threads of a run, as a list of alternatives where each alternative is a list of threads: `[[{"numa": 0, "affinity": 4, "priority": "normal"}]]` is one thread with its memory bound to numa node 0, pinned to cpu 4 at `normal` priority. Single thread is the only supported form for now; `numa` (a node id, binding the memory of the run with `set_mempolicy(MPOL_BIND)`, best effort), `affinity` (a cpu id, a list like `[1, 2]`, or a taskset string like `"0,1"`, defaulting to the first cpu of the PMU that can actually schedule the measured events — so a run is never pinned to a core whose PMU has no such event) and `priority` (a nice value or `lowest`/`low`/`normal`/`high`/`highest`) are the keys |
| `func` | function alignment / layout: `[{"align": 16, "order": "as-is"}]`, `order` is `as-is` or `random` |
| `code` | alignment of the measured code inside the harness: `[{"align": 1}, {"align": 16}]`, `1` is native, anything else a power of two; see [Code alignment](#code-alignment) |
| `stack` | symbolic-execution stack setup: `[{"size": 2097152, "align": 16}]` |
| `samples` | samples per series |
| `iterations` | loop-trip count. A scalar pins it (`100`); a dict is the calibration window `{"min": 100, "max": 1000000}`. Calibration is automatic unless pinned |
| `backend` | `loop` or `unroll`, or a per-backend dict of options (see below) |
| `backend.<name>.probes` | calibration runs for that backend |
| `backend.<name>.runs` | measurement runs for that backend |
| `backend.<name>.target` | target relative standard error the calibration aims for |
| `backend.unroll.count` | unroll factor (the 2N-N differential uses N and 2N copies) |
| `seed` | deterministic input sampling; `0` by default. `null` asks for a fresh seed each run |

A list sweeps the cartesian product with every other config alternative
(`--config.dcache.L1d=[{hit_rate:100},{hit_rate:50}]` runs both hit rates); a
single value runs once. `thread`, `code`, `stack` and `func` keep that list on
the first level, i.e. a list of alternatives whose entries may still have list
leaves of their own, so a shorthand such as `--config.code=32` or
`{"func": "random"}` is normalized into the list form.

Every nested key has a dotted override, and `backend` doubles as a selector.
An override on a `thread`/`code`/`stack`/`func` key pins that container to its
first alternative and sets the key there:

```sh
perf benchmark 'rdtsc' -m latency                      # latency -> unroll backend
perf benchmark 'rdtsc' -m latency,throughput           # still unroll + loop
perf benchmark a.out:func --config.backend=loop   # force the loop backend
perf benchmark 'rdtsc' -m latency --backend loop       # ... for the snippet too
perf benchmark a.out:func --config.backend.unroll.count=4
perf benchmark a.out:func --config.backend.loop.target=0.01
perf benchmark a.out:func --config.backend.unroll.runs=20
perf benchmark a.out:func --config.iterations=100 # pin the loop-trip count
perf benchmark a.out:func --config.iterations.max=10000
perf benchmark a.out:func --config.external.lib=true  # model real library code
perf benchmark /usr/bin/tree:main                           # output faked (default)
perf benchmark /usr/bin/tree:main --config.external.stdout=true  # ... or watch it
perf benchmark a.out:func --config.seed=123
perf benchmark a.out:func --config.thread.affinity=2  # pin to cpu 2
perf benchmark a.out:func --config.thread.numa=1       # bind memory to node 1
perf benchmark a.out:func --config.thread.priority=high
perf benchmark a.out:func --config.func.order=random
perf benchmark a.out:func --config.code.align=32
```

The target is explored symbolically once, and combinations that cannot tell
each other apart are skipped, based on what the exploration itself
recorded: `branch` runs once when no state was found for a
data-dependent branch (i.e. a single explored path), and `dcache`/`dtlb`
run once when no memory read/write was recorded. Every axis the target can
observe is still swept.

The backend follows the mode, not the number of modes asked for: for a snippet
the CLI measures `latency` with `unroll` (where amortising the call away
matters) and `throughput` with `loop`, so `-m latency` alone and
`-m latency,throughput` report the same latency. `perf.benchmark` itself always
defaults to the `loop` backend unless it is given a per-mode map such as
`backend={"latency": "unroll", "throughput": "loop"}`. An explicit `--backend`
(or `--config.backend`) always wins.

```json
{
    "seed": 0,
    "thread": [
        [ { "numa": null, "affinity": null, "priority": "normal" } ]
    ],
    "branch": [ "predictable", "unpredictable" ],
    "dcache": [ "hot", "warm", "cool", "cold" ],
    "icache": [ "hot", "cold" ],
    "dtlb":   [ "hot", "cold" ],
    "itlb":   [ "hot", "cold" ],
    "code": [
        { "align": 1 },
        { "align": 16 }
    ],
    "stack": [
        { "size": [ 2097152 ], "align": 16 }
    ],
    "func": [
        { "align": 16, "order": "as-is" }
    ],
    "samples": 100,
    "iterations": { "min": 100, "max": 1000000 },
    "external": {
        "lib": false,
        "stdout": false,
        "stderr": false
    }
}
```

#### Data

`--data` takes a JSON string/file plus dotted overrides; `perf.benchmark(data=...)`
takes the same dict. Keys are register names or memory addresses, either flat or
grouped into `regs`/`mem` sections. Values may be scalars or lists (one sample
per entry, cycling/predictable per the `branch` config).

```sh
# rdi = 15, or sweep several inputs in one run
perf benchmark a.out:func --data.rdi=15
perf benchmark a.out:func --data.rdi=[1,3,5]

# memory operand at a fixed address
perf benchmark 'mov rax, [rdi]' \
  --data.rdi=0x42000000000 --data[0x42000000000]=123
```

With no explicit `data`, inputs are discovered symbolically: every code path
is explored (IR + SMT solving) and sampled each iteration, so a single run
covers all branches.

```json
{
    "rdi": 15,
    "rsi": "0xFF",
    "0x42000000000": 123,
    "0x42000000001": [1, 2, 3]
}
```

The same data, grouped into sections:

```json
{
    "regs": {"rdi": 15, "rsi": "0xFF"},
    "mem": {"0x42000000000": 123, "0x42000000001": [1, 2, 3]}
}
```

`r8`-`r10`, `rsp`, `rip` and `flags` drive the harness itself (loop counter,
input/output cursors, stack, instruction pointer, condition codes) and are
rejected as data; writing them would corrupt the measurement rather than
describe the target.

#### Examples

```sh
perf benchmark a.out:func --event cycles,instructions
perf benchmark a.out:func --event cycles --event instructions
perf benchmark a.out:hot_begin..hot_end
perf benchmark a.out:func --event 'topdown-*'
perf benchmark 'imul eax, 0' --mode latency
perf benchmark a.out:func --mode throughput
perf benchmark a.out:func --mode latency,throughput
perf benchmark a.out:func --mode latency --mode throughput
perf benchmark foo.s:foo
perf benchmark foo.s:foo..bar
```

```sh
# hot - everything served from L1
perf benchmark 'mov rax, [rdi]' --config.dcache=hot --event cache-misses,cycles
perf benchmark a.out:func --config.dcache=cold

# cold load from a fixed address
perf benchmark 'mov rax, [rdi]' \
  --data.rdi=0x42000000000 --data[0x42000000000]=123 \
  --config.dcache=cold --event cache-misses,cycles

# branch predictibility
perf benchmark a.out:func --data.arg0=1
perf benchmark a.out:func --data.arg0=3
perf benchmark a.out:func --data.arg0=[1,3,5]
```

#### Member functions

`perf benchmark` calls the target directly, so a C++ member function needs its
`this` pointer in the first argument register (`rdi`) and the member fields at
the addresses the code dereferences. Neither is known to the harness, so give
it an object of its own: an address for `this` plus the bytes the fields hold.

```cpp
#include <perf.h>

struct Counter {
    long value;
    long step;

    __attribute__((noinline)) long member(long n) {   // rdi = this, rsi = n
        PERF_LABEL(member_begin);
        long s = 0;
        for (long i = 0; i < n; ++i) {
            s += value;
            value += step;
        }
        PERF_LABEL(member_end);
        return s;
    }
};

Counter global{0, 3};

void setup() { global.value = 1; }
void teardown() { global.value = 0; }

int main() { return (int)global.member(10); }
```

```sh
g++ -O2 -I lib/perf -o a.out a.cpp

# the target is the name exactly as `perf info` prints it
perf info a.out | grep member

# `this` in rdi, `n` in rsi, and the two fields as bytes at `this`
perf benchmark "a.out:Counter::member(long)" -m latency -e cycles,instructions \
  --setup a.out:setup --teardown a.out:teardown \
  --data.rdi=0x42000000000 --data.rsi=1000 \
  --data[0x42000000000:]=1 --data[0x42000000008:]=3

# or only the loop, as a region (same target, narrower span)
perf benchmark a.out:member_begin..member_end -m latency -e cycles \
  --setup a.out:setup --teardown a.out:teardown \
  --data.rdi=0x42000000000 --data.rsi=1000 \
  --data[0x42000000000:]=1 --data[0x42000000008:]=3
```

- `--data[ADDR:]=BYTES` writes the bytes the target reads at `ADDR`. Say where
  the object is with `--data.rdi` and the harness has no contents for it, so the
  call faults; leave the address out and the exploration supplies one (and the
  `data.*` columns of the row say which cells it found).
- `ADDR` must be a page the harness can map, not one the binary already owns
  (`choose a different data address` if it is), and it must be 8-aligned per
  field.
- `--setup`/`--teardown` take a target of the benchmarked file
  (`a.out:setup`, the symbol without its argument list, or the demangled name
  `perf info` prints) or, for a snippet target, an asm snippet (`mov ebx, 1`).
- A function that reads state no data can stand in for — a null pointer, an
  uninitialised device register, a thread's stack — still faults on its own;
  wrap the interesting part in a `PERF_LABEL` region and target that instead.

### perf profile

#### Synopsis

```sh
perf profile [-- COMMAND ...] [-t TARGET] [-e EVENT] [-o OUTPUT]
    [--buffer-size N] [--json [FILE]] [-i]
```

#### Description

Track functions and addresses in a live process with single-startup ptrace and native `rdpmc`/`rdtsc` trampolines
(see [Annotations](../lib/README.md)). Labels, hex addresses and function
entries are patched with detours that relocate the overwritten instructions
(RIP-relative fixups, short-branch expansion) and resume after them, so any
address can be tracked. By default main is tracked; pass `-t` to select
other targets.
A function target (e.g. `-t func`) patches only the entry and intercepts
the return address through a shadow stack, so all `ret` sites are covered
with a single patch.

#### Options

| Option | Meaning |
| --- | --- |
| `COMMAND` | the binary and its arguments to run, after `--` (e.g. `-- ./a.out --work 100`) |
| `-t TARGET`, `--target TARGET` | only track these targets (repeatable): label/function names, hex addresses, or `begin..end` regions (sides may mix; a bare function name tracks entry/exit and is named after the resolved symbol, e.g. `-t func` → `func(int)`; default: main) |
| `-e EVENT`, `--event EVENT` | events via `rdpmc` (default: `cycles`; `duration_time` uses `rdtsc`; a wildcard/regex expands to every matching event, e.g. `topdown-*`) |
| `-o OUTPUT`, `--output OUTPUT` | JSON output path (default: `profile.json`; `none` prints a table instead of saving) |
| `--buffer-size N` | ring-buffer size in slots, rounded up to a power of two (default: `65536`); the ring is a sparse temporary file, only the occupied slots are ever read back and it is removed when the run ends |
| `--json [FILE]` | also emit the trace as JSON records to stdout, or to FILE when a path is given |
| `-i`, `--interactive` | interactive IPython session |

#### Examples

```sh
perf profile -t hot -t cold -e cycles,branch-misses -o profile.json -- ./a.out
perf profile -t func -- ./a.out
perf profile -t work_begin..work_end -- ./a.out
perf profile -t hot -e 'topdown-*' -- ./a.out
```

### perf view

#### Synopsis

```sh
perf view [-- DATA ...] [-e EVENT] [-g GROUPBY] [-f FILTER] [-s STAT]
    [--json [FILE]] [-i]
```

#### Description

View saved or piped data with data frames.  `-e` picks what is shown: the
identity/context columns are carried through, every other name is aggregated
by `--stat` (or evaluated first, so an expression is aggregated like a
measured column), and `-g '' -s ''` prints the raw rows.  With no `-e` the
default is `time,file,name,mode,samples,duration_time/operations`, i.e. the
per-operation cost of the run, and whatever the data does not have is
skipped.

#### Options

| Option | Meaning |
| --- | --- |
| `DATA` | saved runs to read, after `--` (e.g. `-- data/`); omitted when piping JSON on stdin |
| `-e EVENT`, `--event EVENT` | columns/expressions to show, comma-separated or repeated, with `*` expanding a pattern (`-e 'data*'`), e.g. `cycles`, `cycles/instructions`; `<group>.<metric>` reads another group's rows sample-aligned (speedup vs baseline); every name given must be in the data, a missing one is an error (default: `time,file,name,mode,samples,duration_time/operations`) |
| `-g GROUPBY`, `--group-by GROUPBY` | pandas groupby keys (default: `file,name,mode`; `''` for raw rows) |
| `-f FILTER`, `--filter FILTER` | pandas query filter, e.g. `--filter 'name == "func"'` |
| `-s STAT`, `--stat STAT` | aggregation (default: `min,median,p10,p50,p90,p99,max`; `''` for raw rows) |
| `--json [FILE]` | emit JSON records to stdout, or to FILE when a path is given, instead of a table |
| `-i`, `--interactive` | interactive IPython session |

#### Examples

```sh
perf view -- data/
perf view --stat p50,p99 --event cycles/instructions -- data/
perf view --event 'file,name,duration_time/operations' -- data/
perf view --event 'data*' --stat '' -- data/
perf benchmark a.out:func | perf view
```

### perf plot

#### Synopsis

```sh
perf plot [-- DATA ...] [-e EVENT] [-x XAXIS] [-y YAXIS] [-t TYPE] [--logx]
    [--logy] [-g GROUPBY] [-f FILTER] [--config CONFIG] [-o OUTPUT]
    [--json [FILE]] [-i]
```

#### Description

Chart saved or piped measurements (terminal via `sixel`, or save to file).

#### Options

| Option | Meaning |
| --- | --- |
| `DATA` | saved runs to read, after `--` (e.g. `-- data/` or `-- run.txt`); omitted when piping JSON or a result table on stdin |
| `-e EVENT`, `--event EVENT` | metric/expression per chart column, with `*` expanding a pattern over the columns (`-e 'data*'`); comma overlays, repeated `-e` adds columns (default: one column per measured event as `<event>/operations`) |
| `-x XAXIS`, `--xaxis XAXIS` | x-axis column, e.g. `-x data.rsi` for scaling over a parameter (default: `samples/time`) |
| `-y YAXIS`, `--yaxis YAXIS` | y-axis column, overriding the event (default: the event itself) |
| `-t TYPE`, `--type TYPE` | chart types (`ecdf`, `bar`, `boxen`, `hist`, `line`, `point`, `scatter`, ...); comma overlays, repeated `-t` adds charts (default: `ecdf`) |
| `-g GROUPBY`, `--group-by GROUPBY` | hue grouping (default: `file,name,mode`; `''` for a single curve) |
| `-f FILTER`, `--filter FILTER` | pandas query filter |
| `--config CONFIG` | plot config JSON file (matplotlib style/rcParams); `--config.*` dotted overrides win |
| `-o OUTPUT`, `--output OUTPUT` | save to file (`chart.png`, `chart.pdf`, or a directory for one file per chart) |
| `--logx` | log scale on the x-axis |
| `--logy` | log scale on the y-axis |
| `--json [FILE]` | also emit the plotted data as JSON records to stdout, or to FILE when a path is given |
| `-i`, `--interactive` | interactive IPython session |

#### Examples

```sh
perf plot -- data/
perf plot --type ecdf --type bar --event cycles --event instructions -- data/
perf benchmark a.out:func | perf plot
perf plot --logy -- run.txt   # a result table, as printed by `perf benchmark`
```

### perf compare

#### Synopsis

```sh
perf compare [-- DATA ...] [-e EVENT] [-f FILTER] [-b BASELINE]
    [--alpha ALPHA] [--json [FILE]] [-i]
```

#### Description

Compare data with a null-hypothesis test built on the Central Limit Theorem:
a two-sided z-test on the arithmetic mean plus a two-sided z-test on the geometric mean.
Both must reject (`p = max(p_mean, p_gmean)`) before a change reports as `significant`
(`p_value < alpha`), which keeps repeated runs of the same binary stable.

What is compared is a rate, not a total: a run does a different number of
operations every time, so the raw counters of two runs are never comparable.
With no `-e` every measured event is therefore compared per operation
(`cycles/operations`, `duration_time/operations`, ...) and only data that has
no `operations` column (a `perf.data` profile) falls back to its raw counters.

#### Options

| Option | Meaning |
| --- | --- |
| `DATA` | saved runs to read, after `--` (e.g. `-- data/`); omitted when piping JSON on stdin |
| `-e EVENT`, `--event EVENT` | columns/expressions to compare, with `*` expanding a pattern over the columns (`-e '*'`); (default: every measured event per operation, e.g. `cycles/operations`; the raw totals of two runs are always different, so they are only compared when asked for) |
| `-f FILTER`, `--filter FILTER` | pandas query filter |
| `-b BASELINE`, `--baseline BASELINE` | baseline name the others are compared against; must equal a recorded `name` (i.e. `<name>-<id>`) or its basename (default: first sorted) |
| `--alpha ALPHA` | significance level for H0 rejection (default: `0.05`) |
| `--json [FILE]` | emit JSON records to stdout, or to FILE when a path is given, instead of a table |
| `-i`, `--interactive` | interactive IPython session |

#### Examples

```sh
perf compare -- data/ --baseline base-<id>
perf compare -- data/ --event cycles,instructions --alpha 0.01 --json
```

### perf analyze

#### Synopsis

```sh
perf analyze [CODE] [-- DATA ...] [-n NAME] [--setup SETUP] [--teardown TEARDOWN]
    [-f FILTER] [--config CONFIG] [--data ...] [-e EVENT] [-g] [--json [FILE]] [-i]
```

#### Description

Disassemble a snippet, an assembly source label, a function or a region into
one row per instruction and join already measured data onto it, without running
anything.  Every input whose `ip` matches an instruction becomes a column.

The target is explored symbolically, so *all* of its states are analyzed: the
rows are every instruction the target disassembles to (the whole function, or
the whole region, followed through its branches), and the columns describe what
the explored states held.  Symbolic exploration supplies the state columns, not
the row set, so a call the emulator cannot follow (an imported library routine
it has no model for, an access to memory the program never set up) never cuts a
listing short half way through a function.  `data.<reg>` is the value a register the state
pins holds (the argument registers and whatever `--data` constrains, the values
the same exploration hands `perf benchmark` as its models), `data.<addr>` the
value the state has at an address it reads or writes — all recorded
from the exploration itself, so they are the state the measurement models, not
a second opinion.  Every `data.*` cell is a list of the values the explored states held
(`[15]` when they agree, `[0, 1, 1073741825]` when they do not), so a
value is never mistaken for a scalar.  `--filter` selects instructions by
any column, and `in` tests a list: `--filter '15 in `data.rdi`'`.

`index` numbers the instructions from 0. It is a column like any other: it is
in the default selection and comes first when it is selected, and
`-e assembly` leaves it out.

When `assembly` is the only column shown, its header is printed as
`.intel_syntax`, which is exactly what it is, so the table is a valid script
for [llvm-mca](https://llvm.org/docs/CommandGuide/llvm-mca.html) and can be
piped straight into it: `perf analyze a.out:func -e assembly | llvm-mca
-mcpu=alderlake`. The column itself is still called `assembly` everywhere
else (`-e assembly`, `df.assembly`, `--json`, the Python API).

`latency` and `throughput` are the modelled cycles of each instruction from
[OSACA](https://github.com/RRZE-HPC/OSACA) (`latency` is the dependency
latency, `throughput` the port pressure) for its default micro-architecture
model of the ISA (`ICX` for x86-64, `A64FX` for aarch64) - they are `nan` when
the model has no entry for the form.

#### Options

| Option | Meaning |
| --- | --- |
| `CODE` | what to analyze: `FILE:TARGET` for a func or a `begin..end` region of a binary (e.g. `a.out:func`, `a.out:hot_begin..hot_end`, `a.out:0x401000..0x401020`), `FILE:LABEL` for an assembly source (e.g. `a.s:label`, `a.s:foo..bar`), or a raw snippet with no file (e.g. `perf analyze 'mov eax, 42'`) |
| `DATA` | data to join, after `--` (e.g. `-- perf.data`); every `ip` that matches an instruction adds a column to the table (runtime addresses from `perf.data` are translated back to the file for PIE executables, so ASLR does not break the join) |
| `--data ...` | input data for the exploration, same meaning as `perf benchmark --data`: a JSON string/file plus dotted overrides (`--data.rdi=15`, `--data[0x1000]=1`) |
| `-f FILTER`, `--filter FILTER` | pandas query over any column, e.g. `--filter 'size > 4'`; a `data.*` column is a list, so test it with `in` (`--filter '15 in `data.rdi`'`) |
| `-e EVENT`, `--event EVENT` | the columns to show, comma-separated or repeated, so any column of the result can be picked: the instruction's own (`index,assembly,encoding,size,latency,throughput`), the identity (`file,name`), the state (`data.rdi`, `data.0x42000000000` or its `--data` form `data[0x42000000000]`), a joined counter per ip (`cycles`, from `-- perf.data`) or an expression over those (`instructions/cycles`); `*` expands a pattern, so `-e data*` is every state column and `-e '*'` all of them; every name given must be in the result, a missing one is an error, and a column that is not selected is not shown (default: `file,name,index,address,encoding,size,latency,throughput,assembly,data*` plus all joined counters, which prints `index` first) |
| `-n NAME`, `--name NAME` | label of the `name` column (default: the target or the snippet) |
| `--setup SETUP` | setup symbol or asm snippet run before the target during exploration |
| `--teardown TEARDOWN` | teardown symbol or asm snippet run after the target during exploration |
| `--config CONFIG` | bench config JSON file; `--config.*` dotted overrides win (e.g. `--config.stack.size=...`) |
| `-g`, `--debug` | show source code before each instruction (DWARF via angr, like `perf annotate`) |
| `--json [FILE]` | emit the JSON envelope to stdout, or to FILE when a path is given (one record per instruction), instead of the table |
| `-i`, `--interactive` | interactive IPython session |

#### Examples

```sh
perf analyze 'mov eax, 42'
perf analyze foo.s:foo..bar
perf analyze a.out:foo..bar
perf analyze a.out:func
perf analyze a.out:func --filter 'latency > 4'
perf analyze a.out:func --filter '15 in `data.rdi`'
perf analyze a.out:func --data.rdi=15 --data[0x42000000000]=123
perf analyze a.out:func --setup init --teardown fini
perf analyze a.out:func -- perf.data func.json profile.json
perf analyze a.out:func -e assembly,encoding -- llvm_mca.json
perf analyze a.out:func -e 'assembly,data*'
perf analyze a.out:func -e assembly | llvm-mca -mcpu=alderlake
perf analyze a.out:func -e instructions/cycles -- perf.data
perf analyze a.out:func -e '*' --json
```

## Events

Any `perf list` event works (`cycles`, `instructions`, `cache-misses`,
`branch-misses`, `branch-instructions`, raw `rXXXX`, `cpu/.../` PMU paths,
`:u`/`:k`/`:p` modifiers), plus `duration_time` (nanoseconds via `rdtsc`,
the default; needs no PMU). When no `-e` is given to `perf view`, the
default `time,file,name,mode,samples,duration_time/operations` applies (and
`perf compare` derives an `<event>/operations` ratio for every measured event
alongside the raw metrics, e.g. `cycles/operations`,
`instructions/operations`, `duration_time/operations`). `perf plot` with no
`-e` shows only `<event>/operations` (one chart per measured event, e.g.
`duration_time[ns]/operations`); pass `-e` explicitly to plot raw counters
or custom expressions.

In `perf view`, `perf compare`, `perf plot`, `perf info` and `perf analyze` a
`-e` name carries a `*`/`?` pattern that expands to the columns it matches
(`-e 'data*'`, `-e '*'` for every metric), so the same spelling selects
columns wherever the data already has them. In `perf view` and `perf analyze`
every name given must be in the data (or an expression over it): a name that
is not is an error, while the default list skips whatever the data does not
have. In `perf analyze` a column that is not selected is not printed at all,
`index` included.

Any `-e` naming a counter may be a wildcard or a regex instead of an exact
name, and expands to every supported event it matches. For a top-down
breakdown pass `--event 'topdown-*'` (or `event=["topdown-*"]` in `perf.benchmark`/`perf.analyze`)
instead of listing the four level-1 events:

```sh
# separate measurement groups (repeated -e), plus a topdown group
perf benchmark a.out:func \
  --event cycles,instructions \
  --event 'topdown-*' # `topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound`
```

A repeated `-e` is a separate measurement (separate counter scheduling), so
group events that must be read together in a single `-e`.

`topdown-*` is an alias, not a hardware guarantee: the pattern always expands
to the four slots, but a CPU whose PMU does not export them fails with
`unknown event 'topdown-retiring'` instead of silently reporting zeros. A bare
`-e '*'` expands only to the events the host's sysfs PMUs really publish, so
`perf list` is the ground truth for what a machine can measure.

## Output

Each run emits a JSON envelope; every entry of `output` repeats the
backend, the specific backend parameters and the `config`/`data` it was
measured with, so a row is self-describing:

```json
{
    "file": "a.out@b04f8237",
    "name": "func-<run-id>",
    "id": "<run-id>",
    "time": "2026-01-01 00:00:00",
    "info": {"cpu": {}},
    "output": [
        {
            "mode": "latency",
            "config": {
                "backend": {
                    "loop": {
                        "probes": 3,
                        "runs": 10,
                        "target": 0.005
                    }
                },
                "branch": "predictable",
                "dcache": "cold"
            },
            "data": {
                "rdi": 15
            },
            "iterations": 26732,
            "samples": 99,
            "operations": 1,
            "duration_time": 0.7440459244608819
        }
    ]
}
```

## `.perfconfig`

CLI defaults can live in a `.perfconfig` file. The CLI looks for
`.perfconfig` in the current directory first, then in each parent
directory up until (and including) the home directory, and uses the
nearest one it finds:

```ini
[benchmark]
mode = latency,throughput
; a cold TLB costs an extra mprotect per iteration
; config.dtlb = cold
; config.itlb = cold

[plot]
logy = true
config.style = dark_background # inline comments are stripped
```

Each section names a command (`benchmark`, `analyze`, `view`, `plot`,
`compare`, `info`, `profile`); `[default]` applies to all of them, and a
command reads only `[default]` and its own section. Keys are long option
names without `--` (e.g. `mode`, `group-by`, `config.style`), including
dotted `--config.*`/`--data.*` overrides (`--config.*`/`--data.*` only
apply to `benchmark`, `analyze` and, for `--config.*`, `plot`). `#` and `;`
start a comment — a whole line, or after a value when preceded by
whitespace — and a key without a value is ignored, so commented out
settings are never applied. Values are taken literally — there is no
pattern matching, so `baseline` has to be the exact recorded name (or its
basename). An explicit CLI flag always wins over the file, whatever
spelling of the option it uses.
