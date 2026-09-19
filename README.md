# perf-labs/perf

[![Build](https://github.com/perf-labs/perf/actions/workflows/linux.yml/badge.svg)](https://github.com/perf-labs/perf/actions/workflows/linux.yml)

Symbolic benchmarking, live profiling on real hardware without changing the binary.

## Features

- benchmark assemby, functions, regions of any binary with hardware performarance counters and CPU state 'control'.
- profile live binaries without re-compiling.
- analyze machine code together with measured data.
- compare data via central limit theorem and null-hypothesis tests.
- view/plot in the terminal, via interactive mode or jupyter notebooks.

## Documentation

| Doc | What lives there |
| --- | --- |
| [CLI reference](bin/README.md) | Command line options, configuration |
| [Python API](src/README.md) | Python library synopsis |
| [Code Markers](lib/README.md) | [Optional] Zero-instruction code (`C++, Rust, Zig`) markers |
| [Studies](studies/README.md) | Top-down Microarchitecture Analysis Method studies: [retiring](studies/x86_64/retiring), [bad-speculation](studies/x86_64/bad_speculation), [frontend-bound](studies/x86_64/frontend_bound), [backend-bound](studies/x86_64/backend_bound) |
| [Agent skill](SKILL.md) | Performance-engineering loop for humans and agents: profile → benchmark → analyze → optimize |
| [References](https://github.com/perf-labs/perf/wiki/references) | Specs, manuals, publications |

## Requirements

- x86-64 Linux, kernel 6.x+
- Python 3.11+ (see [pyproject.toml](pyproject.toml))
- linux-perf and user-space rdpmc access (see [Setup](#setup))

## Install

```sh
pip install git+https://github.com/perf-labs/perf.git
```

See [Development](src/README.md).

## Setup

```sh
echo 2 | sudo tee /sys/devices/{cpu_core, cpu_atom}/rdpmc
```

This enables user-space `RDPMC` hardware counters without syscalls on the hot path.
This is not required for `duration_time` (Time-Stamp Counter).

## Quick start

```cpp
auto fizz_buzz(int n) -> int; // no headers or annotations are needed
```

```sh
# info
perf info cpu
perf info a.out

# harness
perf benchmark a.out:fizz_buzz -m throughput -e cycles,instructions
perf benchmark a.out:fizz_buzz -e 'topdown-*'
perf benchmark a.out:0x401000..0x401020
perf benchmark a.out:hot_begin..hot_end # see code markers
perf benchmark 'imul eax, 0' -m latency -e cycles
perf benchmark snippet.s:label

# explore
perf benchmark a.out:fizz_buzz --data.rdi=15
perf benchmark a.out:fizz_buzz --config.branch=predictable
perf benchmark a.out:fizz_buzz --config.dcache=cold -e cache-misses,cycles
perf benchmark 'mov rax, [rdi]' --data.rdi=0x42000000000 --data[0x42000000000]=123

# profile
perf profile -f fizz_buzz -e cycles -o profile.json -- ./a.out

# analyze
perf analyze a.out:fizz_buzz --filter 'latency > 4'
perf analyze a.out:fizz_buzz -S | llvm-mca
perf analyze a.out:fizz_buzz -- profile.json

# view/plot
perf benchmark a.out:fizz_buzz | perf view -s p99
perf benchmark a.out:fizz_buzz | perf plot -t ecdf

# compare - Central Limit Theorem null-hypothesis tests
perf benchmark clang.out:fizz_buzz_v1 --output data/
perf benchmark gcc.out:fizz_buzz_v2 --output data/
perf compare --baseline fizz_buzz_v1 -- data/
perf plot -e instructions/cycles -- data/
```

See [CLI reference](bin/README.md).

```py
import perf

df = perf.benchmark(["a.out", "fizz_buzz"])
df.duration_time.describe()

df = perf.benchmark(code="mov eax, 42", mode=["latency"], event=["cycles"])
df.cycles.plot()
```

See [Python API](src/README.md).

## How it works

> benchmark
> - explore symbolically (what inputs reach every path)
> - synthesize per-iteration data (registers + memory values)
> - JIT a harness around the target (latency vs throughput loop)
> - steer cache / TLB / branch state per iteration
> - read counters with rdtsc/rdpmc, subtract overhead, calibrate

> analyze
> - explore symbolically (what inputs reach every path)
> - combine with actual measurments (what measurements says)

> profile
> - patch live process 
> - record hardware performance counters on every hit in ring-buffer

See [How it works](src/README.md#how-it-works).

## License

[MIT](.github/LICENSE)
