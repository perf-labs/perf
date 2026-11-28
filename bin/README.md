# perf(1)

## Name

`perf` — symbolic benchmarking, live profiling, and per-instruction analysis on x86-64 Linux.

## Synopsis

```sh
perf system info [-e EVENT] [--output [FILE]] [-i]
perf system tune [--profile NAME|FILE [--profile.KEY=VALUE]] [--status]
    [--reset] [--dry-run] [--output [FILE]] [-i]
perf system test [--cpu N] [--config.KEY=VALUE] [--output [FILE]] [-i]
perf benchmark [CODE] [-m MODE] [-e EVENT] [-n NAME] [--backend BACKEND]
    [--config ...] [--data ...] [--setup CODE] [--teardown CODE]
    [-o|--output [FILE]] [--debug] [-i] [--list]
    [--env KEY=VALUE] [-- ARGS...]
perf profile [-- COMMAND ...] --filter FILTER [-e EVENT] [-o|--output [FILE]]
    [--buffer-size N] [--list]
perf view [-- DATA ...] [-e EVENT] [-g GROUPBY] [-f FILTER] [-s STAT]
    [-o|--output [FILE]] [-i]
perf plot [-- DATA ...] [-e EVENT] [-x XAXIS] [-y YAXIS] [-t TYPE]
    [-g GROUPBY] [-f FILTER] [--config CONFIG] [-o|--output [PATH]]
    [-i]
perf compare [-- DATA ...] [-e EVENT] [-f FILTER] [-b BASELINE] [--alpha ALPHA]
    [-o|--output [FILE]] [-i]
perf analyze [CODE] [-- DATA ...] [-n NAME] [--setup SETUP] [--teardown TEARDOWN]
    [-f FILTER] [--config CONFIG] [--data ...] [-e EVENT]
    [-o|--output [FILE]] [-i] [--list]
perf optimize EXEC [-o|--output [PATH]] [--opt PASS]
    [--config CONFIG] [--config.KEY=VALUE] [--dry-run]
    [--list] -- DATA ...
```

## Description

- Installed as `perf-<command>` (`perf-benchmark`, `perf-analyze`, `perf-view`, `perf-plot`,
  `perf-compare`, `perf-system`, `perf-profile`, `perf-optimize`); linux-perf dispatches
  to them by name.
- This document writes them with a space: `perf benchmark a.out:func`.

## Commands

### perf system

- `perf system info`: CPU topology, frequency, and cache sizes (read-only).
  - `freq` is Hz from CPUID leaf 0x15 (exact TSC frequency), falling back to a wall-clock
    measurement off the TSC, then to the OS value.
  - `--output` writes Hz as JSON; the table abbreviates it (e.g. `2.7Ghz`).
- `perf system tune --status`: checks the tuning against a profile
  (`performance` by default) without writing anything.
  - Prints one row per setting with `setting`, `path`, `status`; it never shows
    `before`/`after`.
  - `status` is the actual value `tune` would set — the same value as the
    `after` column — not a `tuned`/`untuned` verdict, e.g. `off` for ASLR or
    `performance` for the governor.
  - Controls that are missing or cannot take the requested value report
    `unavailable`/`unsupported`; read failures report `denied` or `failed`.
  - Accepts the same `--profile NAME|FILE` and `--profile.KEY=VALUE` overrides
    as `tune`, so it can check a hypothetical configuration.
  - `--profile=default` is accepted here (unlike an applying `tune`) to check
    the stock settings.
  - Read-only: it never writes and never re-executes through `sudo`.
- `perf system tune`: applies the `performance` profile by default.
  - `--profile NAME|FILE` selects a preset or JSON profile; `--profile.KEY=VALUE`
    overrides one setting.
  - Before writing, it saves the pre-tune configuration to `.perftune` (next to
    the nearest `.perfconfig`, or in the current directory). Later tunes keep the
    first snapshot, so `--reset` always returns to the configuration from before
    the tuning session; remove `.perftune` to start a new baseline.
  - Prints one row per setting with `setting`, `path`, `before`, `after`.
  - A JSON profile is merged over the defaults, so it only has to list the settings it
    changes; every `--profile.KEY=VALUE` wins over both preset and file.
  - `--profile=default` is rejected when applying; use `tune --reset`.
- `perf system tune --reset`: restores the configuration saved in `.perftune`
  before the last tuning session.
  - Restores file-backed settings (`governor`, frequency limits, SMT, THP, perf
    knobs, `irqbalance`, `turbo_msr`, IRQ affinity, ...) to their captured values.
  - `--dry-run` shows the planned restore without writing anything.
  - Without a `.perftune` snapshot it reports that `perf system tune` must run
    first.
  - Removes `.perftune` after a successful restore; a failed restore keeps it so
    the command can be retried.
  - Rejects `--profile` and `--profile.KEY=VALUE`; it restores the captured
    configuration rather than a preset.
- `perf system test`: runs the architecture-specific
  `perf-system-test` (e.g. `bin/x86_64/perf-system-test`) with the current
  arguments.
  - Checks the core clock against the cpufreq maximum, the decode width against
    `llvm-mca` (or the shipped OSACA model when `llvm-mca` is absent), and the
    latency/throughput of common instructions and an L1d pointer chase against
    the same model.
  - Measures the reorder buffer, BTB and dTLB capacities, integer/floating-point
    throughput, branch accuracy and misprediction penalty, the L1/L2/L3/DRAM
    latency and bandwidth hierarchy, and memory-level parallelism.
  - Reads only: it runs the same harness as `perf benchmark` and requires
    user-space `rdpmc`; without it, run `perf system tune --profile=latency`.
  - Prints one row per check with `property`, `measured`, `documented`, `unit`,
    `delta`, `status`, `detail`, and `source`; exits zero only when every check
    is `ok`.
  - `--config.iterations=N`, `--config.samples=N`, and `--config.cpu=N` set the
    measurement size; they follow `perf benchmark`'s `--config.KEY=VALUE`
    spelling.
  - `source` names where the documented value came from (`cpufreq`,
    `llvm-mca`, `osaca`, `sysfs`), or `asm`/`counter`/`sanity` for
    measured-only checks.
  - The runner is selected from the `[system]` `test` mapping
    (`{arch}/perf-system-test` by default); override it with
    `--config.test=PATH` or `--config.test.x86_64=PATH`.
- As a normal user, an applying `tune` and `tune --reset` re-execute through
  `sudo`; `--dry-run`, `--status`, and `test` read the current state without
  writing or using `sudo`.

| Profile       | Intended use                                                                                                                                                                                                                                                                                                                                          |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `default`     | Kernel/firmware defaults: turbo/boost on, user-space `rdpmc` restricted, IRQ balance active, isolation checks off                                                                                                                                                                                                                                     |
| `performance` | Default tuning: settings shared by `latency`/`throughput` (performance governor and max CPU frequency, user-space `rdpmc` on, ASLR off, IRQ balance/watchdogs off, NUMA balancing/KSM off, perf limits exposed); leaves turbo/boost, SMT, THP, isolation, and scheduler choices at their defaults                                                     |
| `latency`     | Low jitter: performance governor and max CPU frequency (base clock once turbo is off; the kernel rejects turbo frequencies with turbo disabled), user-space `rdpmc` on, turbo/boost off, SMT off, IRQ balance off plus isolation/affinity checks, NMI watchdog/NUMA balancing/THP/khugepaged/KSM/timer migration off, `swappiness=0`, sample rate 1/s |
| `throughput`  | Sustained throughput: performance governor and max CPU frequency, turbo/boost and SMT on, user-space `rdpmc` on, IRQ balance off, THP always, `swappiness=10`, sample rate 1/s                                                                                                                                                                        |

The three presets are predefined: `performance` is the common subset; `latency` trades
peak clock for low jitter; `throughput` keeps SMT and turbo and uses huge pages.

Settings: `aslr`, `governor`, `energy_performance_preference`, `cpu_frequency`, `turbo`,
`turbo_msr`, `boost`, `rdpmc`, `smt`, `nmi_watchdog`, `watchdog`, `numa_balancing`, `thp`,
`thp_defrag`, `khugepaged_defrag`, `ksm`, `swappiness`, `stat_interval`, `timer_migration`,
`perf_event_max_sample_rate`, `perf_event_mlock_kb`, `perf_event_paranoid`,
`kptr_restrict`, `sched_autogroup_enabled`, `sched_rt_runtime_us`, `irqbalance`,
`irq_affinity`, `power_supply`, `isolcpus`, `rcu_nocbs`, and `nohz_full`.

Value forms:

- Booleans accept `true`/`false`.
- `aslr`: `0`, `1`, `2`; `true` writes `2`, `false` writes `0`.
- `rdpmc`: `0` off, `1` restricted, `2` on; `true` writes `2`, `false` writes `0`.
- `cpu_frequency`: `range` (restore `cpuinfo_min_freq..cpuinfo_max_freq`), `min`, `max`,
  `base`, or explicit kHz/unit (`--profile.cpu_frequency=2400000`, `2.4GHz`, `800MHz`).
- `thp`: `always`, `madvise`, `never`; `thp_defrag` also accepts `defer`, `defer+madvise`.
- Integer keys use kernel ranges (`perf_event_paranoid` `[-1, 4]`, `kptr_restrict`
  `[0, 2]`, `sched_rt_runtime_us` `[-1, 1000000]`).
- `turbo` is Intel `intel_pstate`; `turbo_msr` its `IA32_MISC_ENABLE` fallback via
  `/dev/cpu/*/msr`; `boost` the generic `cpufreq` switch on AMD. Missing controls report
  `unavailable`.
- `rdpmc` writes `/sys/devices/*/rdpmc`: presets set `2`; `reset` restores the
  captured pre-tune level (the stock `default` profile writes `1`).
- `irqbalance` uses `systemctl`, falling back to `service`.
- `irq_affinity` tunes `/proc/irq/default_smp_affinity` and `/proc/irq/N/smp_affinity` to
  exclude isolated CPUs (from `/sys/devices/system/cpu/isolated` or
  `isolcpus`/`nohz_full`).
- `power_supply`, `isolcpus`, `rcu_nocbs`, `nohz_full` are read-only; a mismatch reports
  `unsupported`. `nohz_full` + `intel_pstate` warns (see
  <https://bugzilla.redhat.com/show_bug.cgi?id=1378529>).

This covers the writable pyperf system-tune knobs and adds the CPU/cache/memory controls
above.

Benchmarkable targets are listed by `perf benchmark FILE --list` and
`perf analyze FILE --list` (filter with `FILE:PATTERN`), and by
`perf profile --list -- FILE` (filter with `-f PATTERN`). A target that is not found
prints the same table before it exits (see [perf benchmark](#perf-benchmark)).

#### Options

| Option                      | Meaning                                                                                                                                                                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `info`                      | show CPU info                                                                                                                                                                                                                     |
| `tune`                      | apply `performance` (or the `--profile` selected profile); rows are `setting`, `path`, `before`, `after`                                                                                                                          |
| `--status`                  | show the tuning values for `--profile` (read-only); rows are `setting`, `path`, `status`                                                                                                                                          |
| `--reset`                   | restore the configuration captured in `.perftune` before the last tune                                                                                                                                                            |
| `test`                      | run the architecture-specific known-answer measurement checks                                                                                                                                                                     |
| `--cpu N`                   | CPU to pin `test` to (default: the first online CPU)                                                                                                                                                                              |
| `--config.KEY=VALUE`        | `test` config: `cpu`, `iterations`, `samples`, or `test` runner path                                                                                                                                                              |
| `--profile NAME\|FILE`      | preset (`default`, `performance`, `latency`, `throughput`) or JSON profile (e.g. `--profile=latency.json`); default `performance`; `default` is for `--status` only                                                               |
| `--profile.KEY=VALUE`       | override one setting for `--status` or an applying `tune`, e.g. `--profile.aslr=true`, `--profile.governor=performance`, `--profile.rdpmc=2`, `--profile.swappiness=0`                                                            |
| `--dry-run`                 | show `before`/`after` without writing (and without `sudo`); applies to `tune` and `--reset`                                                                                                                                       |
| `-e EVENT`, `--event EVENT` | columns to show, comma-separated or repeated; `*` expands (`-e cpu,core,L1d`, `-e 'L*'`; default: all)                                                                                                                            |
| `-o`, `--output [FILE]`     | write JSON records to FILE, or to stdout without FILE; repeat for several destinations; a directory (or a path ending in `/`) uses the output template (default `{time}-{command}-{target}-{id}.json`, override in `.perfconfig`) |
| `-i`, `--interactive`       | interactive IPython session with `df`                                                                                                                                                                                             |

#### Examples

```sh
perf system info
perf system info --output
perf system info -e cpu,core,L1d
perf system tune --status
perf system tune --status --profile=latency
perf system tune --status --profile=latency --profile.aslr=true
perf system tune --status --profile=latency --output
perf system tune --dry-run
perf system tune --profile=latency
perf system tune --profile=latency.json
perf system tune --profile=latency --profile.aslr=true --profile.cpu_frequency=max
perf system tune --reset
perf system tune --reset --dry-run
perf system test
perf system test --cpu 0 --config.iterations=50
perf system test --output selftest.json
perf benchmark a.out --list
perf analyze a.out --list -e name,begin,end
perf profile --list -- ./a.out
```

### perf benchmark

Measure an assembly snippet, function, region, or assembly-source label/region with
hardware counters. `CODE` is `FILE:SYMBOL`, `FILE:begin..end`, `FILE:LABEL`, or a raw
snippet with no `FILE:`.

`FILE` may be any x86-64 ELF: a program, an object or archive (linked into a temporary
copy that keeps the original name, so `q.o` still reports as `q.o`), or a shared library
(then `SYMBOL` is an exported symbol; `perf benchmark libstdc++.so.6 --list` lists them).
An indirect function (`strlen`, `memcpy`, anything glibc picks per CPU) resolves to the
implementation the loader installed. Programs must be position-independent: a non-PIE
executable cannot be mapped where it is linked (usually `0x400000`), so the run stops
with that reason instead of overwriting the process.

The harness calls the target directly. It first runs it once in a throwaway child, and
refuses the run if the child faults, exits, replaces the process, or does not return
(e.g. a program entry point such as `_start`, or a library routine that expects state the
harness does not own). In that case, benchmark the surrounding function or restrict the
target to a `foo_begin..foo_end` region. Entry points named `main`, `_main`, `__main`,
`wmain`, or `WinMain` are special-cased for throughput: they use one configuration and
single-call batches because they are not re-entrant. Latency still measures them
normally.

#### Options

| Option                      | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CODE`                      | target: `FILE:SYMBOL` or `FILE:begin..end` (`a.out:func`, `a.out:hot_begin..hot_end`, `a.out:0x401000..0x401020`), `FILE:LABEL` for assembly (`a.s:label`), or a raw snippet (`perf benchmark 'mov eax, 42'`); a bare `FILE` is an error, use `--list`                                                                                                                                                                  |
| `-m MODE`, `--mode MODE`    | `latency` and/or `throughput` (comma-separated or repeated; default: both); each mode is measured against every config combination                                                                                                                                                                                                                                                                                      |
| `-e EVENT`, `--event EVENT` | comma-separated events share one measurement; a repeated `-e` is a separate measurement; wildcards/regex expand, e.g. `topdown-*` (see [Events](#events))                                                                                                                                                                                                                                                               |
| `-n NAME`, `--name NAME`    | benchmark name; labels the `name` column and supplies `{name}` in the output template                                                                                                                                                                                                                                                                                                                                   |
| `--backend {loop,unroll}`   | `loop` (latency: one snippet/call per iteration; throughput: `backend.loop.count` copies per iteration so loop control does not dominate; repeatable snippets below timer resolution are re-measured with the differential) or `unroll` (latency differential: snippets measure N vs 2N copies, binary targets N calls against N zero-cost calls; factor via `--config.backend.unroll.count`); `unroll` is latency-only |
| `--config ...`              | config JSON file plus dotted overrides (see [Config](#config))                                                                                                                                                                                                                                                                                                                                                          |
| `--data ...`                | input data as JSON string/file plus dotted overrides (see [Data](#data))                                                                                                                                                                                                                                                                                                                                                |
| `--setup CODE`              | setup run once before the measurement loop: `FILE:SYMBOL` of the benchmarked file, or an asm snippet for a snippet target                                                                                                                                                                                                                                                                                               |
| `--teardown CODE`           | teardown run once after the measurement loop, same forms as `--setup`                                                                                                                                                                                                                                                                                                                                                   |
| `--list`                    | list targets instead of measuring; `a.out:foo --list` filters by regex `foo`                                                                                                                                                                                                                                                                                                                                            |
| `--env KEY=VALUE`           | environment variable for the benchmarked program, repeated per variable, added to `envp` (default: empty environment)                                                                                                                                                                                                                                                                                                   |
| `-- ARGS...`                | the benchmarked program's arguments (see [Program arguments](#program-arguments))                                                                                                                                                                                                                                                                                                                                       |
| `-o`, `--output [FILE]`     | write the JSON envelope to FILE, or to stdout without FILE; repeat for several destinations; a directory (or a path ending in `/`) uses the output template (default `{time}-{command}-{target}-{id}.json`, override in `.perfconfig`)                                                                                                                                                                                  |
| `--debug`                   | print config, found solutions, synthesized data, full assembly, and per-run results to stderr                                                                                                                                                                                                                                                                                                                           |
| `-i`, `--interactive`       | interactive IPython session with `df`                                                                                                                                                                                                                                                                                                                                                                                   |

#### Program arguments

Everything after `--` is passed to the benchmarked program as `argv`:

- `FILE` is prepended as `argv[0]` unless the first word already equals it; a bare `--`
  means `argv = [FILE]`, `argc = 1`.
- `argc` goes in `rdi`, `argv` in `rsi`, `envp` in `rdx` (System V ABI), so `main` sees
  them as from `__libc_start_main`.
- Vectors and strings are written to the harness stack (`--config.stack.size`, 2 MiB by
  default), below the return address, with `AT_PHDR`, `AT_PHENT`, `AT_PHNUM`, `AT_PAGESZ`,
  `AT_ENTRY`, `AT_NULL` filled from the mapped ELF; `getauxval()` answers for that image.
- `--env KEY=VALUE` adds to `envp`; without it `envp[0] == NULL` (like `env -i`).
- With `--`, `--data.rdi`, `--data.rsi`, `--data.rdx` (and `arg0`/`arg1`/`arg2`) are
  refused because the process ABI sets them. Other `--data` keys still apply.
- `perf benchmark` options come before `--`; words after it are literal and never
  glob-expanded or de-quoted by `perf` (the shell expands them first, as usual).
- Python `perf.benchmark(argv=[...], env=[...])` takes literal lists including `argv[0]`.

```sh
perf benchmark /usr/bin/tree:main -m latency -- /path/to/folder
perf benchmark a.out:main -m latency --env LANG=C.UTF-8 -- /path/to/folder --help
```

```py
perf.benchmark(
    target="a.out:main",
    mode=["latency"],
    argv=["a.out", "/path/to/folder", "--help"],
    env=["LANG=C.UTF-8"],
)
```

#### Config

`--config` takes a JSON file plus dotted overrides; `perf.benchmark(config=...)` takes the
same dict.

| Key                     | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dcache`                | Data-cache residency. Shortcut `"hot"` (`L1d`: 100%), `"warm"` (`L2`: 100%), `"cool"` (`L3`: 100%), `"cold"` (0% everywhere); a number `0..100` (= `{"hit_rate": N}`); `{"hit_rate": 50}` (global rate: `L1d` = N, `L2`/`L3` = 0); or per-level rates `{"L1": 100}` / `{"L1": {"hit_rate": 50}, "L2": ..., "L3": ...}` (`L1`/`L2`/`L3` are the config level names for `L1d`/`L2`/`L3`; the internal tier names `L1d`/`L2`/`L3` are rejected). Only cache key with levels. Default sweep: all four shortcuts.                                                                                   |
| `icache`                | Instruction-side residency of measured code (`L1i`): `"hot"` (100%), `"warm"`/`"cool"`/`"cold"` (accepted, all 0% — non-resident), a number `0..100`, `{"hit_rate": 50}`, or `{"L1": ...}` (`L1` is the only level of `icache`; `L1i` is rejected as an internal tier name). Applies to the measured code's pages. Steers the code's data-hierarchy line and translation; x86-64 has no user-mode `L1i` flush, so the instruction cache is only displaced by code pressure.                                                                                                                    |
| `dtlb`                  | Data-translation residency (`TLBd`): same single-tier forms as `icache` (`"hot"` = 100%, `"warm"`/`"cool"`/`"cold"` = 0%, number, `{"hit_rate": N}`, `{"L1": ...}`). Flushed with `mprotect` PTE toggles (ring-3 alternative to `invlpg`); pages coalesce into one range/mapping, one syscall per cluster.                                                                                                                                                                                                                                                                                     |
| `itlb`                  | Instruction-translation residency (`TLBi`), same single-tier forms as `dtlb`; applies to code pages.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `branch`                | `predictable` (in order) or `unpredictable` (at random). Forms: `"predictable"`/`"unpredictable"` (case-insensitive), `true` (= `predictable`) / `false` (= `unpredictable`), or a dict `{"prediction": ..., "mem": {"<addr>": ...}, "regs": {"<reg>": ...}}` (flat `{"<addr-or-reg>": ...}` entries are also accepted alongside those keys; a dict with only `mem`/`regs` falls back to `"prediction": "unpredictable"`). Default sweep both; a pruned sweep (no data-dependent branches) keeps the first alternative, `predictable`.                                                         |
| `code`                  | Measured-code alignment: `[{"align": <power-of-two>}]`; `1` keeps the native layout. See [Code alignment](#code-alignment).                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `stack`                 | `[{"size": "auto", "align": "auto", "mode": "random"}]`. `size`: `"auto"` (2 MiB, `0x200000`) or an integer `>= 256`. `align`: `"auto"` (16) or a power of two `>= 1`. `mode`: `random` (refresh per iteration), `zero` (refresh zeros), `fixed` (initialize once). Latency binary calls run on this stack and refresh its top window; asm snippets use the process stack, so `mode` only affects their exploration stack. Symbolic exploration seeds the same 1 KiB top window, and stack-insensitive snippets skip the stack fill entirely.                                                  |
| `heap`                  | Harness buffers for missing leading argument registers: `[{"size": "auto", "mode": "random"}]`. `size`: `"auto"` (from explicit count/stride, default `4096`, max `65536`) or an integer `8..65536` (rounded down to a multiple of 8). `mode`: `random` (fresh 64-bit words per call), `zero`, `fixed` (initialize once, never refresh). Scalar shorthand: an integer/string sets `size` (`--config.heap=8192`). Explicit pointer args win; generated addresses are not emitted as `data.*`.                                                                                                   |
| `func`                  | Function alignment/layout: `[{"align": <power-of-two>, "order": "as-is"\|"random"}]`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `thread`                | Thread alternatives, each a list of concurrent threads: `[[{"numa": 0, "affinity": 4, "priority": "normal"}]]`. Reports `threads` (use `cycles/operations/threads`). Keys: `numa` (node id, `MPOL_BIND`, best effort; `null`/omitted leaves placement alone), `affinity` (CPU id, list `[1,2]`, or taskset `"0,1"`; `null`/omitted pins the first allowed CPU of the PMU event groups, NUMA-aware), `priority` (nice integer or `lowest` (19)/`low` (10)/`normal` (0)/`high` (-10)/`highest` (-20)). A bare dict normalizes to `[[{...}]]`; a list of dicts sweeps one thread per alternative. |
| `seed`                  | Deterministic sampling. Default `0`. An integer is masked to 64 bits; a string is hashed (SHA-256, first 8 bytes); `"random"` (case-insensitive) draws a fresh seed per run. Single value only — it must not be wrapped in a list.                                                                                                                                                                                                                                                                                                                                                             |
| `samples`               | Samples per throughput series. Default `100`, integer `>= 1`, single value only (no list). Throughput takes `samples × backend.<name>.runs` whole-loop measurements and reports one minimum per sample. Latency reports one row per loop trip and ignores `samples`. A throughput row's `operations` is the trip count times `backend.loop.count`.                                                                                                                                                                                                                                             |
| `iterations`            | Loop-trip count. Default `{"min": 100, "max": 1000000}` calibration window (automatic unless pinned). A scalar integer `>= 1` pins it; a dict `{"min": M, "max": N}` (`M <= N`, each `>= 1`) sets the window; lists under `min`/`max` sweep, e.g. `{"min": [100, 1000]}`. Single value or dict only at the top level — use `min`/`max` lists to sweep.                                                                                                                                                                                                                                         |
| `backend`               | `"loop"`, `"unroll"`, a list of those to sweep (`--config.backend=loop,unroll`), a per-backend options dict (`{"loop": {"count": 16, ...}}`), or a config-file per-mode map `{"latency": "unroll", "throughput": "loop"}` (keys must be exactly `latency`/`throughput`, values `loop`/`unroll`). Unknown names are rejected.                                                                                                                                                                                                                                                                   |
| `backend.<name>.probes` | Calibration runs used to size the trip count. Integer `>= 1` (default `3`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `backend.<name>.runs`   | Measurement runs. Integer `>= 1` (default `10`); every run is measured. Throughput groups `runs` whole-loop measurements per sample and reports their minimum. Timing events report their minimum, count events their mean; `unpredictable` branch runs draw fresh random inputs and report means.                                                                                                                                                                                                                                                                                             |
| `backend.<name>.target` | Target relative standard error for calibration. Number `> 0` (default `0.01`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `backend.loop.count`    | Copies of the target measured per loop trip in throughput (default `16`, integer `>= 1`). Keeps the `dec`/`jnz` loop control from dominating sub-cycle operations; set `1` to measure one copy per trip. `operations` reports `iterations × count`. Cache/TLB tiers are established once before the loop, so throughput is steady-state; use latency to measure per-access miss cost.                                                                                                                                                                                                          |
| `backend.unroll.count`  | Unroll factor (default `5`, integer `>= 1`); snippets use N and 2N copies, binary targets N calls against N zero-cost calls.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `external.lib`          | `false` (default) leaves imports opaque and never touches shared libraries; `true` models real library code with angr. Replaces old `lib`.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `external.stdout`       | `false` (default) discards target stdout; `true` passes it through to the terminal (or `--output`, which must own stdout). Program buffering is discarded too.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `external.stderr`       | As `external.stdout`, for stderr. Set either to `true` to watch a target reporting its progress.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `source`                | Source annotation backend: `{"backend": "auto"}` (default; uses debug info when available, otherwise decompiles), `"debug"`, or `"decompile"`. Used by `perf analyze -e source`.                                                                                                                                                                                                                                                                                                                                                                                                                   |

Config files are JSON objects with the keys above; unknown keys are rejected and
`data` must be passed separately via `--data`. `seed`/`samples`/`iterations`
(scalar form) accept a single value only — wrapping them in a list is rejected.
`code`/`stack`/`heap`/`func` alternatives in a config file must be concrete dicts
(no list leaves inside one entry; sweep with several entries instead). `thread`
alternatives sweep the cartesian product with every other key. `backend` also
accepts the per-mode map `{"latency": "unroll", "throughput": "loop"}`.
`perf analyze` accepts the same keys.

Sweeps and overrides:

- Combinations a target cannot observe are pruned before measuring: no
  conditional branches drops `branch` sweeps, and no data-memory accesses
  drops `dcache`/`dtlb` sweeps. Explicit `--data` memory keeps them.
- Config keys have dotted overrides. A list sweeps the cartesian product with every other
  alternative; a single value runs once:
  `--config.dcache=cold,hot`, `--config.dcache.L1=[{hit_rate:100},{hit_rate:50}]`.
- Cache/TLB numeric shorthand: `--config.dcache=50` means `{"hit_rate": 50}`; the same
  applies to `icache`, `dtlb`, and `itlb`.
- `--config.dcache=hot --config.dcache.hit_rate=50` replaces level rates with the global
  50%; `.hit_rate` on a single-level key behaves the same.
- `thread`, `code`, `stack`, `heap`, and `func` take first-level alternatives; scalar
  shorthand normalizes (`--config.code=32`, `{"func": "random"}`). Config-file
  alternatives for `code`/`stack`/`heap`/`func` must be concrete (no list leaves).
- A scalar dotted override on one of those containers pins its first alternative and sets
  the key there (`--config.thread.affinity=2`, `--config.stack.size=4096`).
- CLI list leaves on `code`, `stack`, `heap`, `func` compose as a cartesian sweep:
  `--config.stack.size=4096,8192 --config.stack.align=16,64` runs four setups.
- A per-address override applies inside every current alternative:
  - `--config.dcache.L1[0x1000]=cold` keeps all four default data-cache alternatives;
    `--config.icache[0x1000]=hot`, `--config.dtlb[0x1000]=cold`,
    `--config.itlb[0x1000]=cold` work the same way.
  - `--config.branch[0x401000]=unpredictable` keeps both global branch alternatives.
  - A later plain key (`--config.dcache=cold`) pins one value; a comma value on the
    address sweeps that address.
- A register key under a cache/TLB level applies that level to addresses the register
  points at: `{"dcache": {"L1": {"arg0": "cold"}}}`, `{"dtlb": {"arg0": "cold"}}`.
  - It resolves pointer values supplied by `--data` or symbolic exploration.
  - Accesses in the next 4 KiB window are steered; if none, the pointer itself is.
  - An explicit address override wins over the register-derived spec.
- `--config.backend=loop,unroll` sweeps the backend;
  `--config.backend.loop.runs=3,5` sweeps backend options (`probes`, `target`, `count`
  too). A config file may map modes:
  `"backend": {"latency": "unroll", "throughput": "loop"}`.
- `external.lib`, `external.stdout`, and `external.stderr` may be lists and are swept,
  e.g. `--config.external.lib=true,false`.

```sh
perf benchmark a.out:func --config.backend=loop
perf benchmark a.out:func --config.backend.unroll.count=4
perf benchmark a.out:func --config.backend.loop.count=16
perf benchmark a.out:func --config.backend.loop.runs=3,5
perf benchmark a.out:func --config.iterations=100
perf benchmark a.out:func --config.seed=123
perf benchmark a.out:func --config.dcache=50
perf benchmark a.out:func --config.dcache=hot --config.dcache.hit_rate=50
perf benchmark a.out:func '--config.dcache.L1[0x1000]=cold'
perf benchmark a.out:func '--config.branch[0x401000]=unpredictable'
perf benchmark a.out:func --config.stack.size=4096,8192 --config.stack.align=16,64
perf benchmark a.out:func --config.heap.size=8192 --config.heap.mode=random
perf benchmark a.out:func --config.thread.affinity=0 --config.thread.numa=0
perf benchmark a.out:func --config.func.order=random --config.func.align=32
perf benchmark foo.s:foo --config.code.align=32
perf benchmark /usr/bin/tree:main --config.external.stdout=true
```

Backend selection:

- A snippet's backend follows the mode, not the number of modes: `latency` uses `unroll`
  when the steered data cache/TLB tiers stay resident, `throughput` uses `loop`. `-m
  latency` and `-m latency,throughput` therefore report the same latency.
- Explicit non-resident cache/TLB steering (non-resident tier, per-address/register
  override) makes latency default to `loop` (only the first unrolled copy sees the steered
  state); the default `dcache`/`dtlb` sweep chooses per combination (non-resident ->
  `loop`); default `icache`/`itlb` keep `unroll` unless configured.
- Binary targets default to `loop` in both modes unless `--backend` or
  `--config.backend` says otherwise.
- Pin every steered tier to `hot`, or pass `--backend unroll`, to use `unroll` throughout.
  Explicit `--backend`/`--config.backend` always wins.
- `perf.benchmark()` defaults to `loop`; `auto_backend=True` restores the CLI's
  cache-sensitive `unroll` downgrade.
- In `unroll`, snippet labels, including inline `; label:` definitions, are renamed per
  copy (`label__unroll0`, `label__unroll1`, ...).
- The target is explored once; combinations that cannot differ are skipped (`branch` once
  without data-dependent state, `dcache`/`dtlb` once without memory access). Observable
  axes are still swept.

Default config (`src/perf/bench.py::_DEFAULT_BENCH`):

```json
{
  "seed": 0,
  "samples": 100,
  "iterations": {"min": 100, "max": 1000000},
  "dcache": ["hot", "warm", "cool", "cold"],
  "icache": ["hot", "cold"], "dtlb": ["hot", "cold"], "itlb": ["hot", "cold"],
  "branch": ["predictable", "unpredictable"],
  "code":   [{"align": 1}, {"align": 16}],
  "stack" : [{"size": "auto", "align": "auto", "mode": "random"}],
  "heap":   [{"size": "auto", "mode": "random"}],
  "func":   [{"align": 16, "order": "as-is"}],
  "thread": [[{"numa": null, "affinity": null, "priority": "normal"}]],
  "backend": {
    "loop":   {"count": 16, "probes": 3, "runs": 10, "target": 0.01},
    "unroll": {"count": 5, "probes": 3, "runs": 10, "target": 0.01}
  },
  "external": {"lib": false, "stdout": false, "stderr": false},
  "source": {"backend": "auto"}
}
```

Resolved shorthands: `stack.size "auto"` = 2 MiB (`0x200000`),
`stack.align "auto"` = 16, `heap.size "auto"` = from the explicit
count/stride (default `4096`, hard max `65536`).

#### Code alignment

- `config.code.align` sets the entry alignment of the measured snippet inside the harness
  (the `.align` directive before `{code}`).
- `1` keeps the native layout; any other value must be a power of two.
- Binary targets ignore it; use `config.func.align` and `config.func.order` instead.
- Alignment changes what the front end fetches; treat it as an observation or a
  code/binary change, not a free control in an A/B.

#### Data

- `--data` takes a JSON string/file plus dotted overrides; `perf.benchmark(data=...)`
  and `perf.analyze(data=...)` take the same dict. The config file must not
  contain `data`; data gets its own identity hash.
- Keys are register names or memory addresses, flat or grouped as
  `{"regs": {...}, "mem": {...}}`. A flat `{"rdi": 15, "0x1000": 1}` merges to the
  same grouped form. Unknown `--data.*` spellings are rejected — use
  `--data.REG=..`, `--data.ADDR=..`, or `--data[ADDR]=..` / `--data[ADDR:]=..`.
- Register keys are lowercased; `arg0`..`arg5` alias `rdi,rsi,rdx,rcx,r8,r9`
  (`argN` = Nth System V argument register). Values are unsigned 64-bit:
  integers or `"0x.."` strings (`int(v, 0) & 0xFFFFFFFFFFFFFFFF`).
- Scalars are pinned for every iteration; lists cycle one sample per entry
  (`--data.rdi=[1,3,5]`; cycling/predictable per `branch`).
- Memory keys are addresses parseable as `int(addr, 0)` (e.g. `"0x42000000000"`).
  `--data[0xADDR]=VALUE` stores one word per address (scalar `u64`, or a list of
  `u64` cycling per iteration). `--data[0xADDR:]=BYTES` (trailing `:`) writes a
  byte range: `bytes`/`str` (UTF-8 bytes), or scalar/list items each `0..255`
  (wider stores must use `--data[ADDR]=` without the colon).
- With no explicit data, inputs are discovered symbolically: every code path is explored
  (IR + SMT solving) and sampled each iteration, so one run covers all branches.
- Reserved registers are rejected: `r8`-`r10`, `rsp`, `rip`, and `flags` drive the
  harness (loop counter, cursors, stack, instruction pointer, condition codes), so
  any value for them is an error. Other harness registers (`r11`-`r15`, `rbx`,
  `rbp`) accept scalars but reject per-iteration lists (the list would be silently
  truncated to its first value as the harness holds timing baselines there).
- With `--` (program `argv`), `--data.rdi`, `--data.rsi`, `--data.rdx` (and
  `arg0`/`arg1`/`arg2`) are refused because the process ABI sets them. Other
  `--data` keys still apply.

```sh
perf benchmark a.out:func --data.rdi=15
perf benchmark a.out:func --data.rdi='[1,3,5]'
perf benchmark 'mov rax, [rdi]' --data.rdi=0x42000000000 '--data[0x42000000000]=123'
perf benchmark a.out:func --data.rdi=0x42000000000 '--data[0x42000000000:]=[1,2,3]'
```

```json
{
  "rdi": 15,
  "rsi": "0xFF",
  "0x42000000000": 123,
  "0x42000000001": [1, 2, 3]
}
```

The same data can be grouped: `{"regs": {"rdi": 15}, "mem": {"0x42000000000": 123}}`.

#### Member functions

A C++ member function needs its `this` pointer in `rdi` and its fields at the addresses the
code dereferences. Give it an object: an address for `this` plus the bytes of its fields.

```cpp
#include <perf.h>

struct Counter {
    long value, step;
    __attribute__((noinline)) long member(long n) {   // rdi = this, rsi = n
        PERF_LABEL(member_begin);
        long s = 0;
        for (long i = 0; i < n; ++i) { s += value; value += step; }
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

# the target is the name exactly as `--list` prints it
perf benchmark a.out --list | grep member

# `this` in rdi, `n` in rsi, and the two fields as bytes at `this`
perf benchmark "a.out:Counter::member(long)" -m latency -e cycles,instructions \
  --setup a.out:setup --teardown a.out:teardown \
  --data.rdi=0x42000000000 --data.rsi=1000 \
  '--data[0x42000000000:]=1' '--data[0x42000000008:]=3'

# or only the loop, as a region
perf benchmark a.out:member_begin..member_end -m latency -e cycles \
  --setup a.out:setup --teardown a.out:teardown \
  --data.rdi=0x42000000000 --data.rsi=1000 \
  '--data[0x42000000000:]=1' '--data[0x42000000008:]=3'
```

- `--data[ADDR:]=BYTES` writes the bytes the target reads at `ADDR`.
  - Say where the object is with `--data.rdi`; without contents the call faults.
  - Leave the address out and exploration supplies one; the `data.*` columns say which
    cells it found.
- `ADDR` must be a page the harness can map, not one the binary owns (`choose a different
  data address` if it is), and 8-aligned per field.
- `--setup`/`--teardown` take a target of the benchmarked file (`a.out:setup`, the symbol
  without its argument list, or the demangled name `--list` prints) or, for a snippet, an
  asm snippet (`mov ebx, 1`).
- A function reading state no data can stand in for (null pointer, uninitialised device
  register, thread stack) still faults; wrap the interesting part in a `PERF_LABEL` region
  and target that.

#### Examples

```sh
perf benchmark a.out:func --event cycles,instructions
perf benchmark a.out:func --event cycles --event instructions
perf benchmark a.out:hot_begin..hot_end
perf benchmark a.out:func --event 'topdown-*'
perf benchmark 'imul eax, 0' --mode latency
perf benchmark a.out:func --mode throughput
perf benchmark a.out:func --mode latency,throughput
perf benchmark foo.s:foo
perf benchmark foo.s:foo..bar
```

```sh
# hot: everything served from L1
perf benchmark 'mov rax, [rdi]' --config.dcache=hot --event cache-misses,cycles
perf benchmark a.out:func --config.dcache=cold

# cold load from a fixed address
perf benchmark 'mov rax, [rdi]' \
  --data.rdi=0x42000000000 '--data[0x42000000000]=123' \
  --config.dcache=cold --event cache-misses,cycles

# branch predictability
perf benchmark a.out:func --data.arg0=1
perf benchmark a.out:func --data.arg0='[1,3,5]'
```

### perf profile

- Tracks functions and addresses in a live process with single-startup ptrace and native
  `rdpmc`/`rdtsc` trampolines (see [Annotations](../lib/README.md)).
- Labels, hex addresses, and function entries are patched with detours that relocate the
  overwritten instructions (RIP-relative fixups, short-branch expansion) and resume after
  them, so any address can be tracked.
- `--filter` is required and selects symbols. `perf profile --list -- ./a.out` lists
  trackable targets, one row per label/function, with optional `-f PATTERN`.
- A function target (`-f func`) patches only the entry and intercepts the return through a
  shadow stack, covering all `ret` sites with one patch.
- Each hit records its thread id (`tid`, via `fs`) alongside the counters, so view/plot can
  group by `tid` or use `cycles/operations/threads`.
- Detours preserve every general-purpose register; function entry/exit do not preserve
  `rflags`, which a call may clobber anyway.

#### Options

| Option                         | Meaning                                                                                                                                                                                                                                        |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `COMMAND`                      | binary and arguments to run, after `--` (e.g. `-- ./a.out --work 100`)                                                                                                                                                                         |
| `-f FILTER`, `--filter FILTER` | symbols to track (repeatable, required unless `--list`): label/function names, hex addresses, or `begin..end` regions; sides may mix; a bare function name tracks entry/exit and is named after the resolved symbol (`-f func` -> `func(int)`) |
| `-e EVENT`, `--event EVENT`    | events via `rdpmc` (default `cycles`; `duration_time` uses `rdtsc`; wildcard/regex expands, e.g. `topdown-*`)                                                                                                                                  |
| `-o`, `--output [FILE]`        | write the JSON envelope to FILE, or to stdout without FILE; repeat for several destinations; a directory (or a path ending in `/`) uses the output template (default `{time}-{command}-{target}-{id}.json`, override in `.perfconfig`)         |
| `--buffer-size N`              | ring-buffer slots, rounded up to a power of two (default `65536`); the ring is a sparse temporary file, only occupied slots are read back, and it is removed at exit                                                                           |
| `--list`                       | list targets instead of profiling; `-f foo --list` filters by regex `foo`                                                                                                                                                                      |

#### Examples

```sh
perf profile --list -- ./a.out
perf profile -f foo --list -- ./a.out
perf profile -f hot -f cold -e cycles,branch-misses -o profile.json -- ./a.out
perf profile -f func -- ./a.out
perf profile -f work_begin..work_end -- ./a.out
perf profile -f hot -e 'topdown-*' -- ./a.out
```

### perf view

- Views saved or piped data with data frames.
- Default columns:
  `time,file,name,mode,operations,duration_time/operations` (per-operation cost);
  missing ones are skipped, and present `data.*` grouping columns are shown.
- With `-e`, exactly those columns are selected, plus `data.*` grouping columns.
  Every other metric is aggregated by `--stat`; an expression is evaluated first and
  then aggregated like a measured column. `-g '' -s ''` prints raw rows.
- Default grouping is `file,name,mode` plus every present `data.*` column, so runs
  differing only in explored data parameters stay separate.
- Besides perf envelopes and result tables, reads pandas `df.to_json()` table shapes. A
  split/table object that fails to parse is treated as empty, so the command fails cleanly
  instead of plotting a garbage frame.

#### Options

| Option                             | Meaning                                                                                                                                                                                                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATA`                             | saved runs to read, after `--` (e.g. `-- data/`); omitted when piping JSON on stdin                                                                                                                                                                                     |
| `-e EVENT`, `--event EVENT`        | columns/expressions, comma-separated or repeated; `*` expands (`-e 'data*'`), e.g. `cycles`, `cycles/instructions`; `<group>.<metric>` reads another group's mean (speedup vs baseline); every name must exist in the data (default: the default columns plus `data.*`) |
| `-g GROUPBY`, `--group-by GROUPBY` | pandas groupby keys (default `file,name,mode,data.*`; `data.*` expands to every data column; `''` for raw rows)                                                                                                                                                         |
| `-f FILTER`, `--filter FILTER`     | pandas query filter, e.g. `--filter 'name == "func"'`                                                                                                                                                                                                                   |
| `-s STAT`, `--stat STAT`           | aggregation (default `min,median,p10,p50,p90,p99,max`; `''` for raw rows)                                                                                                                                                                                               |
| `-o`, `--output [FILE]`            | write JSON records to FILE, or to stdout without FILE; repeat for several destinations; a directory (or a path ending in `/`) uses the output template (default `{time}-{command}-{target}-{id}.json`, override in `.perfconfig`)                                       |
| `-i`, `--interactive`              | interactive IPython session                                                                                                                                                                                                                                             |

#### Examples

```sh
perf benchmark a.out:func -m latency -e cycles,instructions,duration_time -o bench.json
perf view -- bench.json
perf view --stat p50,p99 --event cycles/instructions -- bench.json
perf view --event 'file,name,duration_time/operations' -- bench.json
perf view --event 'data*' --stat '' -- bench.json
perf benchmark a.out:func | perf view
```

> A directory as `--output` expands the default `{time}-{command}-{target}-{id}.json`
> template (override with `output = ...` in `.perfconfig`, see [Output](#output));
> file paths always work. A directory after `--` reads every envelope inside it.

### perf plot

- Charts saved or piped measurements in the terminal (`sixel`/`ascii`) or saves to a file.
- Default hue groups: `file,name,mode` plus every `data.*` column.
- Reads perf envelopes, result tables, and pandas `df.to_json()` tables.
- Renderer is deduced from terminal capabilities (queried once and cached):
  `sixel` when the terminal reports sixel and both the backend and ImageMagick are
  installed; else `ascii` when the ASCII backend is installed; else `agg`.
- `--config.backend` (or `[plot] config.backend` in `.perfconfig`) selects any matplotlib
  backend, e.g. `window`, `ascii`, `sixel`, or `module://...`.
- `-t cfg` draws a control-flow graph from analyze rows, or from a binary passed after
  `--`. Every CFG plot is laid out and rendered with Graphviz DOT; image files keep
  the DOT-rendered output and `.dot`/`.gv` outputs keep the DOT source. Without
  `--output` the graph is shown through the selected backend (`sixel`, GUI, or the
  ASCII backend—never a separate text renderer). Every block and its complete
  instruction listing is shown. Analyze rows need the `address`, `size`, and
  `assembly` columns (the default `perf analyze` selection); `-e` projections that
  drop any of them are not valid CFG input. Binaries use a static `CFGFast` by
  default; `--config.cfg=emulated` (also `dynamic`, `emu`, `cfgemulated`,
  `cfg-emulated`, `cfg_emulated`) generates a dynamic `CFGEmulated(keep_state=True)`
  graph instead. Analyze rows always use the static row-derived graph; `config.cfg`
  applies to binaries only.
- `-t auto` (or `[plot] type = auto`) picks the chart from the data: analyze rows
  with the required CFG columns and flow graphs use `cfg`, an explicit multi-value
  `xaxis` uses `line`, and benchmark or profile samples use `ecdf`. An explicit `-t`
  overrides it.

#### Options

| Option                             | Meaning                                                                                                                                                                                                        |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATA`                             | saved runs to read, after `--` (e.g. `-- data/` or `-- run.txt`); omitted when piping JSON or a result table on stdin                                                                                          |
| `-e EVENT`, `--event EVENT`        | metric/expression per chart column; `*` expands (`-e 'data*'`); comma overlays, repeated `-e` adds columns (default: one column per measured event as `<event>/operations`)                                    |
| `-x XAXIS`, `--xaxis XAXIS`        | x-axis column, e.g. `-x data.rsi` (default `time`)                                                                                                                                                             |
| `-y YAXIS`, `--yaxis YAXIS`        | y-axis column, overriding the event (default: the event)                                                                                                                                                       |
| `-t TYPE`, `--type TYPE`           | chart types (`auto`, `ecdf`, `bar`, `boxen`, `hist`, `line`, `point`, `scatter`, `cfg`, ...); comma overlays, repeated `-t` adds charts; `auto` deduces the chart (default: `auto`)                                                                                                      |
| `-g GROUPBY`, `--group-by GROUPBY` | hue grouping (default `file,name,mode,data.*`; `data.*` expands to every data column; `''` for one curve)                                                                                                      |
| `-f FILTER`, `--filter FILTER`     | pandas query filter                                                                                                                                                                                            |
| `--config CONFIG`                  | plot config JSON file (matplotlib style/rcParams); `--config.*` overrides win (e.g. `--config.backend=sixel`, `--config.logx=true`); unknown keys are rejected                                                                       |
| `-o`, `--output [PATH]`            | save a chart to PATH (`chart.png`, `chart.pdf`, or a directory for one file per chart); a `.json` PATH writes plotted JSON records instead, and no PATH writes them to stdout; repeat for several destinations |
| `-i`, `--interactive`              | interactive IPython session                                                                                                                                                                                    |

#### Examples

```sh
perf benchmark a.out:func -m latency -e cycles,instructions -o bench.json
perf plot -- bench.json
perf plot --config.backend=window -- bench.json
perf plot --config.backend=ascii -- bench.json
perf plot --type ecdf --type bar --event cycles --event instructions -- bench.json
perf benchmark a.out:func | perf plot
perf benchmark a.out:func -e cycles | perf plot
perf analyze a.out:func | perf plot
perf plot --config.logy=true -- run.txt   # a result table, as printed by `perf benchmark`
perf analyze a.out:func | perf plot -t cfg
perf plot -t auto -- bench.json
perf plot -t cfg -- a.out
perf plot -t cfg -o cfg.dot -- a.out
perf plot -t cfg --config.cfg=emulated -- a.out
```

> For `plot`, a directory gets one image per chart type (`<dir>/ecdf.png`, ...).
> Use a `.json` path (or a bare `--output`) for plotted JSON records instead.

### perf compare

- Compares data with a CLT null-hypothesis test: two-sided z-tests on the arithmetic mean
  and geometric mean. When both p-values are finite, both must reject
  (`p = max(p_mean, p_gmean)`) before a change is `significant` (`p_value < alpha`);
  when the log test is undefined, the finite test alone decides.
- `diff` shows `-` when not significant and a signed percentage when it is (negative
  faster, positive slower); `p_value` and `significant` carry the verdict.
- Comparison is by rate, not total: runs do different operation counts.
  - With no `-e`, every measured event is compared per operation
    (`cycles/operations`, `duration_time/operations`, ...).
  - Data without an `operations` column (a `perf.data` profile) falls back to raw counters.
- Runs are matched on `mode` and every `data.*` value, not on name or binary, so
  differently named builds with the same inputs are compared.
- If the requested baseline is absent from a group, the first variant is used; groups with
  no challenger still print the baseline row. Comparisons that cannot be evaluated (too
  few samples, disjoint modes, failed test) print an `inconclusive` row with `p_value`
  null instead of being dropped.
- Saved data may be a pandas `df.to_json()` table, like view and plot.

#### Options

| Option                               | Meaning                                                                                                                                                                                                                           |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATA`                               | saved runs to read, after `--` (e.g. `-- data/`); omitted when piping JSON on stdin                                                                                                                                               |
| `-e EVENT`, `--event EVENT`          | columns/expressions to compare; `*` expands (`-e '*'`); default: every measured event per operation (raw totals always differ, so they are compared only when asked)                                                              |
| `-f FILTER`, `--filter FILTER`       | pandas query filter                                                                                                                                                                                                               |
| `-b BASELINE`, `--baseline BASELINE` | baseline name; must equal a recorded `name` (`<name>-<id>`) or its basename (default: first sorted)                                                                                                                               |
| `--alpha ALPHA`                      | significance level for H0 rejection (default `0.05`)                                                                                                                                                                              |
| `-o`, `--output [FILE]`              | write JSON records to FILE, or to stdout without FILE; repeat for several destinations; a directory (or a path ending in `/`) uses the output template (default `{time}-{command}-{target}-{id}.json`, override in `.perfconfig`) |
| `-i`, `--interactive`                | interactive IPython session                                                                                                                                                                                                       |

#### Examples

```sh
perf compare --baseline base -- data/
perf compare --event cycles,instructions --alpha 0.01 --output -- data/
```

### perf analyze

- Disassembles a snippet, assembly-source label, function, or region into one row per
  instruction and joins measured data onto it, without running anything.
- Every input whose `ip` matches an instruction becomes a column; pandas `df.to_json()`
  tables and perf envelopes work as `DATA`.
- `perf analyze a.out --list` lists analyzable targets, one row per label/function, with
  optional `a.out:foo` regex filter and `-e` column selection.
- The target is explored symbolically, so all states are analyzed:
  - Rows are every instruction the target disassembles to (whole function or region,
    followed through branches).
  - Exploration supplies state columns, not the row set, so a call the emulator cannot
    follow never cuts a listing short.
  - `data.<reg>` is the value a pinned register holds (argument registers and whatever
    `--data` constrains, from the same exploration `perf benchmark` uses).
  - `data.<addr>` is the value the state has at an address it reads or writes.
  - Every `data.*` cell is a list (`[15]` when states agree, `[0, 1, 1073741825]` when
    not), so a value is never mistaken for a scalar.
- `--filter` selects by any column; `in` tests a list:
  ``--filter '15 in `data.rdi`'``.
- `index` numbers instructions from 0; it is in the default selection and comes first when
  selected. `-e assembly` leaves it out.
- When `assembly` is the only column shown, its header is printed as `.intel_syntax`, so
  the table is a valid [llvm-mca](https://llvm.org/docs/CommandGuide/llvm-mca.html) script:
  `perf analyze a.out:func -e assembly | llvm-mca -mcpu=alderlake`. The column is still
  called `assembly` everywhere else (`-e assembly`, `df.assembly`, `--output`, Python API).
- `latency`/`throughput` are modelled cycles per instruction from
  [OSACA](https://github.com/RRZE-HPC/OSACA): dependency latency and port pressure for its
  default model (`ICX` for x86-64, `A64FX` for aarch64). They are `nan` when the model has
  no entry.
- Source rows use the latest mapped DWARF line, skip a closing brace when a better line is
  available, and each source line is shown once (repeated lines become blanks). The
  decompiler backend applies the same de-duplication.
- With no `--event`, the built-in selection includes `file,name`; the shipped
  `.perfconfig` selects `index,address,encoding,size,latency,throughput,assembly,data.*`
  plus `source` on its own line. An explicit `--event` wins over the file.

#### Options

| Option                         | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CODE`                         | `FILE:SYMBOL` or `FILE:begin..end` for a binary (`a.out:func`, `a.out:hot_begin..hot_end`, `a.out:0x401000..0x401020`), `FILE:LABEL` for assembly (`a.s:label`, `a.s:foo..bar`), or a raw snippet (`perf analyze 'mov eax, 42'`)                                                                                                                                                                                                                                                             |
| `DATA`                         | data to join, after `--` (e.g. `-- perf.data`); every `ip` that matches an instruction adds a column; PIE runtime addresses are translated back to the file, so ASLR does not break the join                                                                                                                                                                                                                                                                                                 |
| `--data ...`                   | input data for exploration, same as `perf benchmark --data`: JSON string/file plus dotted overrides (`--data.rdi=15`, `--data[0x1000]=1`)                                                                                                                                                                                                                                                                                                                                                    |
| `-f FILTER`, `--filter FILTER` | pandas query over any column, e.g. `--filter 'size > 4'`; a `data.*` column is a list, so test it with `in` (``--filter '15 in `data.rdi`'``)                                                                                                                                                                                                                                                                                                                                                |
| `-e EVENT`, `--event EVENT`    | columns to show, comma-separated or repeated: instruction columns (`index,assembly,encoding,size,latency,throughput`), identity (`file,name`), state (`data.rdi`, `data.0x42000000000` or `data[0x42000000000]`), joined counters (`cycles`), or expressions (`instructions/cycles`); `*` expands; every name must exist; unselected columns are not shown (default: `file,name,index,address,encoding,size,latency,throughput,assembly,data*` plus all joined counters, with `index` first). Commas make every name a column, including `source` (`-e assembly,source`); a repeated `-e` puts source lines on their own lines instead (`-e assembly -e source`). `source` holds the DWARF source line when debug info is available and falls back to the angr `Decompiler` pseudocode (`CFGFast(normalize=True)`, binary targets); `--config.source.backend=auto|debug|decompile` selects the backend |
| `-n NAME`, `--name NAME`       | label of the `name` column (default: symbol or snippet)                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `--setup SETUP`                | setup symbol or asm snippet run before the symbol during exploration                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `--teardown TEARDOWN`          | teardown symbol or asm snippet run after the symbol during exploration                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `--config CONFIG`              | bench config JSON file; `--config.*` overrides win (e.g. `--config.stack.size=...`)                                                                                                                                                                                                                                                                                                                                                                                                         |
| `--list`                       | list targets instead of analyzing; `a.out:foo --list` filters by regex `foo`                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `-o`, `--output [FILE]`        | write the JSON envelope to FILE, or to stdout without FILE (one record per target with `--list`, one per instruction otherwise); repeat for several destinations; a directory (or a path ending in `/`) uses the output template (default `{time}-{command}-{target}-{id}.json`, override in `.perfconfig`)                                                                                                                                                                                  |
| `-i`, `--interactive`          | interactive IPython session                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

#### Examples

```sh
perf analyze a.out --list
perf analyze a.out:foo --list
perf analyze 'mov eax, 42'
perf analyze foo.s:foo..bar
perf analyze a.out:foo..bar
perf analyze a.out:func
perf analyze a.out:func -e source
perf analyze a.out:func -e source --config.source.backend=debug
perf analyze a.out:func -e source --config.source.backend=decompile
perf analyze a.out:func -e assembly,source
perf analyze a.out:func -e assembly -e source
perf analyze a.out:func --filter 'latency > 4'
perf analyze a.out:func --filter '15 in `data.rdi`'
perf analyze a.out:func --data.rdi=15 '--data[0x42000000000]=123'
perf analyze a.out:func --setup a.out:start --teardown a.out:stop
perf analyze a.out:func -- perf.data func.json profile.json
perf analyze a.out:func -e 'assembly,data*'
perf analyze a.out:func -e assembly | llvm-mca -mcpu=alderlake
perf analyze a.out:func -e instructions/cycles -- perf.data
perf analyze a.out:func -e '*' --output
```

### perf optimize

Rewrite a binary with measured-data-guided, semantics-preserving passes. `DATA` after
`--` is required and must be non-empty: pass one or more benchmark/profile/analyze JSON
files, a directory of them, `perf.data`, or a pandas `df.to_json()` table. Missing or
unusable data is an error; `perf optimize` never silently degrades to an unmeasured run.

- `peephole` and `branch` keep each function body the same size (shorter encodings are
  padded with optimal multi-byte NOPs), so no address outside the rewritten `.text` moves.
- `layout` permutes whole functions and fixes up every reference: rel32 branches,
  RIP-relative memory, RELA and packed RELR addends, `.eh_frame` FDEs, the
  `.eh_frame_hdr` lookup table, symbol tables, and the ELF entry point.
- Safe fallbacks:
  - Indirect jumps, absolute `.text` addresses, text relocations, or exception tables keep
    the original layout.
  - Non-PIE binaries skip `layout`/`align`.
  - Unfixable/unverifiable layout falls back to the original image without losing the
    size-preserving passes.
  - Jump-table functions and layout hazards are pinned; fall-through functions,
    incomplete `.text` coverage, or an aligned layout that does not fit keep the original
    order.
- Only statically reachable instructions are rewritten, so peephole/branch leave embedded
  data and padding alone; layout rewrites whole functions and fills uncovered `.text`
  gaps. Trailing-NOP stripping and branch-target protection are fail-closed.
- Output defaults to the input name with `.opt` before its extension (`a.out` ->
  `a.opt.out`, `tool` -> `tool.opt`); `--output` overrides it. The report lists every
  touched function.

Config keys, merged over defaults: `opt` (default all passes), `layout.align` (default 16,
power of two 1..4096), and `layout.order` (default `hot-first`, or `as-is`).
`--config FILE` loads a JSON object, `--config.KEY=VALUE` overrides it, and `--opt` wins
over `config.opt`.

Passes (`--opt`, comma-separated or repeated; default all):

- `peephole`: safe same-size simplifications: `mov r,r` (64-bit, exact register),
  `xchg r,r`, `lea r,[r]`/`lea r,[r*1]` (exact register, 64-bit address size),
  `shift r,0` (`shl/sal/shr/sar/rol/ror`), `movabs` shrinking to `mov r32/r64` (incl.
  `r8`-`r15`), and NOP normalization. Only statically reachable instructions.
- `branch`: fold transitive `jmp`-to-`jmp`/`jcc`-to-`jmp` chains (cycle-safe), remove
  jumps to the next instruction, and invert a conditional branch over a following `jmp`
  (self-loop safe).
- `layout`: reorder functions hot-first from measured data (benchmark, profile,
  `perf.data`, and analyze all contribute) with entry alignment.
- `align`: extra hot-function alignment (implies `layout`).

Data after `--` guides layout: benchmark envelopes (per-op cost), profile envelopes
(total time), pandas `df.to_json()` tables, `perf.data` samples (per-symbol counts), and
analyze envelopes (modeled latency) are normalized and summed. `--dry-run` prints the plan
without writing, `--list` lists functions with hotness, and `--output` emits the report (or
listed functions with `--list`). All of these forms require `DATA`.

#### Options

| Option                  | Meaning                                                                                                                                                                           |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EXEC`                  | input binary (`a.out`)                                                                                                                                                            |
| `-o`, `--output [PATH]` | output binary path (default `EXEC.opt<EXT>`, e.g. `a.opt.out`); a `.json` PATH writes the report instead, and no PATH writes it to stdout; repeat for several binaries or reports |
| `--opt PASS`            | passes to run (repeatable/comma-separated; default all)                                                                                                                           |
| `--config FILE`         | config JSON file merged over defaults (`opt`, `layout.align`, `layout.order`)                                                                                                     |
| `--config.KEY=VALUE`    | config override, e.g. `--config.layout.align=64`, `--config.layout.order=as-is`, `--config.opt=peephole,branch`                                                                   |
| `--dry-run`             | print the plan without writing                                                                                                                                                    |
| `--list`                | list functions with hotness instead of optimizing (`DATA` is still required)                                                                                                      |

#### Examples

```sh
perf optimize a.out -- data/                  # writes a.opt.out
perf optimize a.out --output b.out -- data/
perf optimize a.out --output b.out --opt peephole,branch -- data/
perf optimize a.out --output b.out --opt layout --config.layout.align=64 -- data/
perf optimize a.out --output b.out --opt layout --config.layout.order=as-is -- data/
perf optimize a.out --output b.out -- data/bench.json data/profile.json perf.data
perf optimize a.out --output b.out --dry-run -- data/
perf optimize a.out --list --output -- data/
```

## Events

- Hardware: `cycles`, `instructions`, `cache-references`, `cache-misses`,
  `branch-instructions`, `branch-misses`, `bus-cycles`, `stalled-cycles-frontend`,
  `stalled-cycles-backend`, `ref-cycles`.
- Hardware-cache:
  `L1-dcache-{loads,load-misses,stores,store-misses,prefetches,prefetch-misses}`,
  `L1-icache-{loads,load-misses,prefetches,prefetch-misses}`,
  `LLC-{loads,load-misses,stores,store-misses,prefetches,prefetch-misses}`,
  `dTLB-{loads,load-misses,stores,store-misses,prefetches,prefetch-misses}`,
  `iTLB-{loads,load-misses,prefetches,prefetch-misses}`,
  `branch-{loads,load-misses}`,
  `node-{loads,load-misses,stores,store-misses,prefetches,prefetch-misses}`.
- Aliases: `idle-cycles-frontend`, `idle-cycles-backend`, and `duration_time`
  (nanoseconds via `rdtsc`, the default; needs no PMU).
- Raw `rXXXX` events and every sysfs PMU event (`cpu/...`, `cpu_core/...`,
  `cpu_atom/...`, ...) work.
- Counting modifiers: `:u`, `:k`, `:h`, `:G`, `:H`, `:p`/`:pp`/`:ppp`. Sampling-only
  modifiers (callchains, skid, weights) are rejected; they do not apply to the `rdpmc`
  counting path.
- Default `-e` selections:
  - `perf view`: `time,file,name,mode,operations,duration_time/operations` plus `data.*`.
  - `perf compare`: an `<event>/operations` ratio for every measured event
    (`cycles/operations`, `instructions/operations`, `duration_time/operations`, ...).
  - `perf plot`: only `<event>/operations`, one chart per measured event (e.g.
    `duration_time[ns]/operations`); pass `-e` for raw counters or custom expressions.
- In view, compare, plot, system, and analyze, `-e` accepts `*`/`?` patterns (`-e 'data*'`,
  `-e '*'`). In view and analyze every name must exist (or be an expression over the data);
  the default list skips missing columns. In analyze, unselected columns are not printed,
  `index` included.
- `perf analyze --list` takes `-e` for listing columns (`-e 'name,begin,end'`);
  `perf benchmark --list` and `perf profile --list` also accept `-e`.
- Any counter name may be a wildcard/regex and expands to every supported match. For
  top-down use `--event 'topdown-*'` (or `event=["topdown-*"]` in Python):

```sh
# separate measurement groups (repeated -e), plus a topdown group
perf benchmark a.out:func --event cycles,instructions \
  --event 'topdown-*' # the four topdown slots
```

- A repeated `-e` is a separate measurement (separate counter scheduling); group events
  that must be read together in one `-e`.
- `topdown-*` expands to `topdown-retiring`, `topdown-bad-spec`, `topdown-fe-bound`,
  `topdown-be-bound`, with the slots leader added automatically. It always expands to the
  four slots and reports percentages of the slots leader (near 100% when all four are
  exported); a PMU that does not export them fails with
  `unknown event 'topdown-retiring'`, not zeros, and a missing slot is not imputed.
- Bare `-e '*'` expands to supported hardware, hardware-cache, and software events; the
  core events above are built in, so `perf list` is the ground truth for anything more.

## Output

Every command has one output flag: `-o`/`--output [PATH]`
(repeatable). There is no `--json`.

- Without `--output`, commands print their normal table and write no file.
- A bare `--output` writes the structured result to stdout.
- `--output FILE` writes exactly that file.
- A directory (or a path ending in `/`) expands the default
  `{time}-{command}-{target}-{id}.json` template, or the `output` template from
  the nearest `.perfconfig` when set. `{time}` is a filesystem-safe timestamp,
  `{command}` the command (`benchmark`, `profile`, `analyze`, ...), `{target}`
  the binary basename with its content hash (`a.out@b04f8237`) or the
  snippet/result name, `{file}` and `{hash}` the split target, `{name}` the
  result name, and `{id}` the identity hash.
- Repeat `--output` to write several destinations:
  `perf benchmark ... --output a.json --output b.json`.
- `output = ...` in `.perfconfig` sets the same destinations as a default. It is
  additive: the configured files are written and the normal table is still
  printed. An explicit CLI `--output` wins and suppresses the table.
- `plot` writes an image for a non-`.json` path (or a directory), and JSON
  records for a `.json` path or a bare `--output`.
- `optimize` writes a binary for a non-`.json` path, and the JSON report for a
  `.json` path or a bare `--output`; use two `--output` flags to write both.

Each benchmark run emits a JSON envelope. Every entry of `output` repeats the
backend, its specific parameters, and the `config`/`data` it was measured with,
so a row is self-describing:

```json
{
  "file": "a.out@b04f8237", "name": "func-<run-id>", "id": "<run-id>",
  "time": "2026-01-01 00:00:00", "info": {"cpu": {}, "hostname": "host"},
  "output": [{
    "mode": "throughput",
    "config": {"backend": {"loop": {"count": 16, "probes": 3, "runs": 10,
                                   "target": 0.01}},
               "branch": "predictable", "dcache": "cold"},
    "data": {"rdi": 15},
    "operations": 16000,
    "duration_time": 0.7440459244608819
  }]
}
```

## `.perfconfig`

The CLI uses the nearest `.perfconfig` found in the current directory or a parent up to
(and including) the home directory. Only that one file is read; files are not merged.

- Sections name commands: `benchmark`, `analyze`, `optimize`, `view`, `plot`, `compare`,
  `system`, `profile`. A command reads only its own section; there is no `[default]`.
  (`perf system test` is an exception to the wrapper rule below: the `perf system`
  wrapper ignores the file for `test` and forwards `--config.*` to
  `perf-system-test`, which itself reads `[system]` for the runner, the measurement
  sizes, and `output`.)
- Keys are the command's long options without `--` (`mode`, `group-by`, `config.style`),
  including dotted overrides. Either spelling of an option is accepted (`groupby` or
  `group-by`, `buffer-size` or `buffer_size`); an explicit CLI flag wins whatever
  spelling it uses. Positionals (`CODE`, `EXEC`, paths after `--`) cannot come from
  the file.
- Dotted prefixes by command:
  - `--config.*`: `benchmark`, `analyze`, `plot`, `optimize`;
  - `--data.*`: `benchmark`, `analyze`;
  - `--profile.*`: `system tune` (including `--status`).
  - `view`, `compare`, `profile` accept plain options only (no dotted prefix).
- `output` sets the default `--output` destination(s). It accepts the same
  file paths, directories, and `{time}-{command}-{target}-{id}` template as
  the flag. Config output is additive: files are written and the table is still
  printed. An explicit CLI `--output` wins and suppresses the table (config output
  keeps it). A directory (or a path ending in `/`) expands the default
  `{time}-{command}-{target}-{id}.json` template unless `output` overrides it.
  Template fields: `{time}` (filesystem-safe timestamp), `{command}`, `{target}`
  (binary basename plus content hash, e.g. `a.out@b04f8237`, or the result name),
  `{file}`/`{hash}` (the split target), `{name}` (result name, run id stripped),
  `{id}` (identity hash).
- Flag-only booleans (`interactive`, `debug`, `list`, `dry-run`, `status`, `reset`)
  take `true`/`false` in the file (`1`/`true`/`yes`/`on` vs
  `0`/`false`/`no`/`off`; anything else is an error). `on` enables the flag.
  Plot log scales are config booleans, e.g. `config.logx = true` /
  `config.logy = false`.
- `#` and `;` start a comment: a whole line, or after a value when preceded by whitespace.
- A key without a value is ignored, so commented-out settings are never applied.
- Values are literal (no pattern matching), e.g. `baseline` must be the exact recorded
  name or its basename. Comma-separated list options (`mode`, `event`, `opt`, `stat`,
  `type`, `env`) are written the same way as on the CLI.
- A key may be repeated. Its values stay in file order and are injected once per
  occurrence, exactly like repeating the CLI flag: two `event =` lines become two
  `-e` arguments (e.g. one column group and one annotation group), not one merged
  value.
- An explicit CLI flag wins over the file, whatever spelling it uses. Dotted keys merge
  per subkey: CLI `--config.stack.size=...` overrides only that subkey, other
  `config.*` file entries still apply. (`event` from the file is skipped for `--list`
  listings unless `-e` is also given on the CLI.)
- Inside one process the file's entries are injected before the CLI flags (after the
  subcommand for `perf system`), so CLI parsing, validation, and `unknown argument`
  errors behave exactly as if the entries had been typed before the CLI flags.

Per-command keys and defaults:

| Section                      | Keys (long option without `--`)                                                                                                                                                                                                                                                           | Defaults                                                                                                                                                                                                                              |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[benchmark]`                | `mode`, `backend`, `config` (JSON file path), `data` (JSON string/file path), `event`, `env` (`KEY=VALUE`, repeatable), `setup`, `teardown`, `name`, `output`, flags `interactive`, `debug`, `list`, plus `config.*` (every [Config](#config) key) and `data.*` (every [Data](#data) key) | `mode = latency,throughput`; `event = duration_time` (via `rdtsc`); no `output`; full [Config](#config) defaults                                                                                                                      |
| `[analyze]`                  | `name`, `setup`, `teardown`, `filter`, `config` (file), `data`, `event`, `output`, flags `list`, `interactive`, plus `config.*` (every `benchmark` config key) and `data.*`                                                                                                              | built-in `event = file,name,index,address,encoding,size,latency,throughput,assembly,data*` plus all joined counters; the shipped `.perfconfig` uses `event = index,address,encoding,size,latency,throughput,assembly,data.*` plus `source` annotation |
| `[view]`                     | `event`, `group-by`, `filter`, `stat`, `output`, `interactive`                                                                                                                                                                                                                            | `event = time,file,name,mode,operations,duration_time/operations` plus `data.*`; `group-by = file,name,mode,data.*`; `stat = min,median,p10,p50,p90,p99,max` (`''` = raw rows)                                                        |
| `[plot]`                     | `event`, `xaxis`, `yaxis`, `type`, `group-by`, `filter`, `config` (file), `output`, `interactive`, plus `config.*` (see below)                                                                                                                                            | `type = auto`; `group-by = file,name,mode,data.*`; `config.logx = true`, `config.logy = false`; `config.backend = auto`; `config.style = dark_background`; `config.figure.figsize = [25, 20]` in the shipped `.perfconfig` (code defaults `false`, `false`, `(10, 5)`) |
| `[compare]`                  | `event`, `filter`, `baseline`, `alpha`, `output`, `interactive`                                                                                                                                                                                                                           | `event` = every measured event per `operations`; `baseline` = first sorted; `alpha = 0.05`                                                                                                                                            |
| `[profile]`                  | `event`, `filter` (repeatable), `output`, `buffer-size`, `list`, `interactive`                                                                                                                                                                                                            | `event = cycles` (`duration_time` uses `rdtsc`); `buffer-size = 65536` slots (rounded up to a power of two)                                                                                                                           |
| `[optimize]`                 | `opt`, `config` (file), `output`, flags `dry-run`, `list`, plus `config.*` (`config.opt`, `config.layout.align`, `config.layout.order`)                                                                                                                                                   | `opt = peephole,branch,layout,align`; `config.layout.align = 16` (power of two `1..4096`); `config.layout.order = hot-first` (`hot-first`/`as-is`); `--opt` wins over `config.opt`; `align` implies `layout`                          |
| `[system]` (info)            | `event`, `output`, `interactive`                                                                                                                                                                                                                                                          | shows all columns                                                                                                                                                                                                                     |
| `[system]` (tune/`--status`) | `profile`, `profile.*` (one override per tunable setting), `output`, `dry-run`, `status`, `reset`, `interactive`                                                                                                                                                                          | `profile = performance`; `--status` accepts `--profile=default`, applying `tune` rejects it (use `--reset`); `--reset` rejects `profile`/`profile.*`; `--status` rejects `--dry-run`; `--status` and `--reset` are mutually exclusive |
| `[system]` (test)            | `test`, `test.{arch}`, `config.test`, `config.test.{arch}` (runner path, `{arch}` expanded), `cpu`/`config.cpu`, `iterations`/`config.iterations`, `samples`/`config.samples`, `output`                                                                                                   | runner `{arch}/perf-system-test`; `cpu` = first online CPU; `iterations = 20`, `samples = 1`; no `output`                                                                                                                             |

`perf system test` measurement sizes: the shipped `bin/x86_64/perf-system-test`
uses `{"cpu": null, "iterations": 20, "samples": 1}` (`--cpu N` pins the CPU,
integers `>= 1` for the sizes). `perf benchmark`-style `--config.iterations=N`,
`--config.samples=N`, `--config.cpu=N` spellings are accepted (the wrapper forwards
them; `test`/`config.test*` select the runner instead).

`[plot] config.*` reference: `config.backend` selects any matplotlib backend
(`auto` = sixel/ascii by terminal capability, else `agg`; aliases `ascii`,
`sixel`; `window`; `module://...`), `config.style` a matplotlib style,
`config.figure.figsize` e.g. `[25, 20]`, `config.logx`/`config.logy`,
`config.cfg` (`static` or `emulated` plus `dynamic`, `emu`, `cfgemulated`,
`cfg-emulated`, `cfg_emulated` as `emulated`; binaries only, for `-t cfg`), and any other matplotlib
`rcParam` (`config.axes.grid`, ...). Unknown `config.*` keys are rejected
(`unknown plot config: ...`). Plot defaults are
`figure.figsize (10, 5)`, `axes.spines.top/bottom False`, `grid.color gray`,
`grid.alpha 0.5`, `grid.linestyle --`, `axes.grid True`.

Tunable `profile.*` keys for `system tune` (31 settings; `perf system tune --status`
shows the resolved values): `aslr` (`true`/`false` or `0`/`1`/`2`),
`governor`/`energy_performance_preference` (text), `cpu_frequency`
(`range`/`min`/`max`/`base` — `auto`/`default` = `range`, `low` = `min`,
`high` = `max`, `nominal` = `base` — or kHz/unit like `2400000`, `2.4GHz`),
`turbo`/`turbo_msr`/`boost`/`nmi_watchdog`/`watchdog`/`numa_balancing`/
`khugepaged_defrag`/`ksm`/`timer_migration`/`sched_autogroup_enabled`/
`irqbalance`/`irq_affinity` (booleans), `rdpmc` (`0` off / `1` restricted / `2` on;
booleans map `true`→`2`, `false`→`0`), `smt` (boolean), `thp`
(`always`/`madvise`/`never`), `thp_defrag`
(`always`/`madvise`/`defer`/`defer+madvise`/`never`), `swappiness` (`0..200`),
`stat_interval` (`1..3600`), `perf_event_max_sample_rate` (`1..1000000000`),
`perf_event_mlock_kb` (`0..16777216`), `perf_event_paranoid` (`-1..4`),
`kptr_restrict` (`0..2`), `sched_rt_runtime_us` (`-1..1000000`),
`power_supply` (`ac`/`battery`/`any`), and read-only `isolcpus`/`rcu_nocbs`/
`nohz_full` (a mismatch reports `unsupported`). Missing/unsupported controls report
`unavailable`; read failures `denied`/`failed`. Presets: `default` (stock), plus
`performance` (default profile), `latency`, `throughput`; a JSON profile file is
merged over `default` and every `--profile.KEY=VALUE` wins over both.

```ini
[benchmark]
mode = latency,throughput
event = duration_time
; default destination/template; remove output to write no file
output = {time}-{command}-{target}-{id}.json
; a cold TLB costs an extra mprotect per iteration in latency mode
; (throughput establishes the TLB state once before the loop)
; config.dtlb = cold
; config.itlb = cold

[analyze]
event = index,address,encoding,size,latency,throughput,assembly,data.*
event = source                    # auto debug/decompile source lines
config.source.backend = auto      # auto|debug|decompile

[view]
group-by = file,name,mode,data.*
stat = min,median,p10,p50,p90,p99,max

[plot]
group-by = file,name,mode,data.*
type = auto                      # auto|ecdf|cfg|... deduced from the data
config.logx = true               # log scale on the x-axis
config.logy = false              # log scale on the y-axis
config.backend = auto            # auto|sixel|ascii|window|module://..., sixel/ascii by terminal, else agg
config.style = dark_background   # inline comments are stripped
config.figure.figsize = [25, 20]
; config.cfg = emulated         # -t cfg: static CFGFast or dynamic CFGEmulated

[compare]
alpha = 0.05

[profile]
buffer-size = 65536

[system]
profile = latency
profile.aslr = true
profile.turbo = false
test = {arch}/perf-system-test
; config.iterations = 20
; config.samples = 1
; config.cpu = 4

[optimize]
opt = peephole,branch,layout,align
config.layout.align = 64
config.layout.order = hot-first
```
