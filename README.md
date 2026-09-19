# perf-labs/perf

> profile → benchmark → analyze → optimize ↻

## Features

- Benchmark assembly snippets, functions and regions of any binary with hardware performance counters and CPU state 'control'.
- Pass a program's own arguments, so a `main` is measured the way a shell calls it.
- Profile live binaries without re-compiling.
- Analyze machine code together with measured data.
- Compare data via central limit theorem and null-hypothesis tests.
- View/Plot in the terminal, via interactive mode or jupyter notebooks.

## Documentation

| Name | Description |
| ---- | ---- |
| [CLI reference](bin/README.md) | Command line options, configuration |
| [Python API](src/README.md#api) | Architecture, synopsis |
| [Code Markers](lib/README.md) | Zero-instruction code (C, C++, Rust, Zig) markers |
| [Skills](SKILL.md) | The Human–Agent performance engineering loop |
| [Studies](studies/README.md) | Top-down Microarchitecture Analysis Method studies: [retiring](studies/x86_64/retiring), [bad-speculation](studies/x86_64/bad_speculation), [frontend-bound](studies/x86_64/frontend_bound), [backend-bound](studies/x86_64/backend_bound) |
| [How it works](src/README.md#how-it-works) | Symbolic exploration, data synthesis, JIT harness, CPU 'control' (cache/branch/...), code patching, performance hardware counters |
| [References](src/README.md#references) | Specifcations, publications, manuals |

## Requirements

- x86-64 Linux, kernel 6.x+
- Python 3.11+ (see [pyproject.toml](pyproject.toml))
- linux-perf and user-space rdpmc access (see [Setup](#setup))

## Install

```sh
pip install git+https://github.com/perf-labs/perf.git
```

See [Development](tests/README.md).

## Setup

```sh
echo 2 | sudo tee /sys/devices/{cpu_core,cpu_atom}/rdpmc
```

This enables user-space `RDPMC` hardware counters without syscalls on the hot path.
This is not required for `duration_time` (Time-Stamp Counter).

## Quick start

```sh
# benchmark
perf benchmark 'imul eax, 0' -m latency -e cycles
perf benchmark snippet.s:label
perf benchmark a.out:func -e 'topdown-*'
perf benchmark a.out:0x401000..0x401020
perf benchmark a.out:func -m throughput -e cycles,instructions
perf benchmark a.out:hot_begin..hot_end # see code markers
perf benchmark /usr/bin/tree:main -m latency -- .

# explore
perf benchmark a.out:func --data.rdi=15
perf benchmark a.out:func --config.branch=predictable
perf benchmark a.out:func --config.dcache=cold -e cache-misses,cycles
perf benchmark 'mov rax, [rdi]' --data.rdi=0x42000000000 --data[0x42000000000]=123

# analyze
perf analyze a.out:func --filter 'latency > 4'
perf analyze a.out:func -e 'index,assembly,data*'
perf analyze a.out:func -e assembly | llvm-mca
perf analyze a.out:func --filter '15 in `data.rdi`'
perf analyze a.out:func -- perf.data profile.json

# profile
perf profile -f begin..end -e cycles -o profile.json -- ./a.out

# view/plot
perf benchmark a.out:func | perf view -s p99
perf benchmark a.out:func | perf plot -t ecdf

# compare # Central Limit Theorem -> null-hypothesis test
perf benchmark clang.out:func_v1 --output data/
perf benchmark gcc.out:func_v2 --output data/
perf compare --baseline func_v1 -- data/
perf plot -e instructions/cycles -- data/
```

See [CLI reference](bin/README.md).

```py
import perf

df = perf.benchmark(["a.out", "func"], mode=["latency"])
df.duration_time.describe()

df = perf.benchmark(code="mov eax, 42", mode=["latency"], event=["cycles"])
df.cycles.plot()

df = perf.benchmark(code=["/usr/bin/tree", "main"], mode=["latency"], argv=["."])
perf.to_json(df)
```

See [Python API](src/README.md#api).

## Cite

```bibtex
@software{jusiak2026perf,
  author  = {Kris Jusiak},
  title   = {perf},
  year    = {2026},
  url     = {https://github.com/perf-labs/perf}
}
```
