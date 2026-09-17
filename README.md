# perf {asm,func,region}

[![Build](https://github.com/perf-labs/perf/actions/workflows/linux.yml/badge.svg)](https://github.com/perf-labs/perf/actions/workflows/linux.yml)

Symbolic benchmarking on real hardware without changing the binary.

## Features

- Benchmark `asm`, `functions`, `regions` in any compiled binary (`C++`, `Rust`, `Zig`, ...).
- Symbolic inputs discovery covers every code path (IR exploration + SMT solving).
- Control cache state (`L1i`/`L1d`/`L2d`/`L3d`/`DRAM`/`TLBd`,`TLBi`) and branch predictability per run, globally or per address/register.
- Read hardware counters via `RDPMC` (`cycles`, `instructions`, `cache-misses`, `branch-misses`, `topdown`, ...).
- Latency and throughput modes with `loop`/`unroll` backends.
- View/plot results in the terminal (`sixel`), interact via IPython (`--interactive`) or Jupyter notebooks.

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
```

## Setup

```sh
echo 2 | sudo tee /sys/devices/cpu_core/rdpmc
```

## Quick start

```sh
# asm
perf bench asm 'imul eax, 0' --mode latency

# const char* fizz_buzz(int);
perf bench func fizz_buzz --exec a.out --mode throughput
perf bench func fizz_buzz --exec a.out --mode latency --event cycles,instructions

# region
perf bench region hot_begin..hot_end --exec a.out --mode latency
perf bench region 0x401000..0x401020 --exec a.out --mode latency
perf bench region main..main+0x20 --exec a.out --mode latency
perf bench region main..foo --exec a.out --mode latency
```

```sh
# info
perf info cpu
perf info metadata --exec a.out

# view
perf view -- data/
perf view --stat p50,p99 --event cycles/instructions -- data/
perf bench func fizz_buzz --exec a.out --mode latency | perf view

# plot
perf plot -- data/
perf plot --type ecdf --type bar --event cycles --event instructions -- data/
perf bench func fizz_buzz --exec a.out --mode latency | perf plot
```

```py
# python
import perf

df = perf.bench(exec="a.out", func="fizz_buzz", mode="latency", event=["duration_time"])
df.duration_time.describe()

df = perf.bench(asm="mov eax, 42", mode="latency", event=["cycles"])
df.cycles.plot()
```

### Data

```json
{
    "rdi": 15,
    "rsi": "0xFF",
    "0x42000000000": 123,
    "0x42000000001": [1, 2, 3]
}
```

### Config

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

## Examples

```sh
# cache
perf bench asm 'mov rax, [rdi]'
    --mode latency
    --config.cache=hot
    --event cache-misses,cycles

perf bench asm 'mov rax, [rdi]'
    --mode latency
    --data.rdi=0x42000000000
    --data[0x42000000000]=123
    --config.cache=cold
    --event cache-misses,cycles

# branch
perf bench func fizz_buzz --exec a.out --mode latency --data.arg0=1
perf bench func fizz_buzz --exec a.out --mode latency --data.arg0=3
perf bench func fizz_buzz --exec a.out --mode latency --data.arg0=[1,3,5] # distribution
```

```sh
# topdown
perf bench func ".*" # all exposed functions
  --exec a.out
  --mode latency
  --event cycles,instructions
  --event topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound
```

```sh
# mca
perf bench func fizz_buzz --exec a.out --data.rdi=15 -S | llvm-mca

# exec
perf bench func fizz_buzz --exec a.out --mode latency -c -o bench.o
$CXX bench.o ...
```

```python
# python
hot  = {"L1d":{"hit_rate":100}}
warm = {"L1d":{"hit_rate":0}, "L2":{"hit_rate":100}}
cool = {"L1d":{"hit_rate":0}, "L2":{"hit_rate":0}, "L3":{"hit_rate":100}}
cold = {"L1d":{"hit_rate":0}, "L2":{"hit_rate":0}, "L3":{"hit_rate":0}}

for size in range(1, 2**10):
    for cache in (hot, warm, cool, cold):
        for branch in ("predictable", "unpredictable"):
            perf.bench(
                exec="libstdc++.so",
                func="std::sort",
                mode="latency",
                config={"branch": branch, "cache": cache},
                data={"arg1": size},
                event=["cycles", "instructions"],
            )
```

## How it works

```
Machine Code                    # C++, Rust, Zig, Assembly, ...
  → Symbolic Exploration        # path0, path1, ..., pathN
     → Data Synthesis           # path0: rdi=[0], mem[addr]=[2], ...
        → Benchmarking Harness  # setup, latency/throughput, teardown
           → Native Execution   # 0b010101010101100100010100111
```

## License

[MIT](.github/LICENSE)
