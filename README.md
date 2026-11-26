# perf-labs/perf

> profile → benchmark → analyze → optimize ↻

## Features

- Tune the host with different profiles and validate the results against the documented values.
- Profile binaries without re-compiling with minimal overhead.
- Benchmark assembly snippets, functions and regions of a binary with hardware performance counters and CPU/OS state control.
- Analyze machine code together with measurments.
- Compare results via central limit theorem and null-hypothesis tests.
- Optimize binaries with measured-data-guided passes.
- View/Plot in the terminal, via interactive mode or jupyter notebooks.

## Documentation

| Name                                       | Description                                                                                                                                                                                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [CLI reference](bin/README.md)             | Command line options, configuration                                                                                                                                                                                                        |
| [Python API](src/README.md#api)            | Architecture, synopsis                                                                                                                                                                                                                     |
| [Code Markers](lib/README.md)              | Zero-instruction code (C, C++, Rust, Zig) markers                                                                                                                                                                                          |
| [Examples](.github/workflows/perf.sh)      | End-to-end CLI examples and smoke checks                                                                                                                                                                                                   |
| [Studies](studies/README.md)               | Top-down Microarchitecture Analysis Method studies: [retiring](studies/x86_64/retiring), [bad-speculation](studies/x86_64/bad_speculation), [frontend-bound](studies/x86_64/frontend_bound), [backend-bound](studies/x86_64/backend_bound) |
| [Skills](SKILL.md)                         | The Human–Agent performance engineering loop                                                                                                                                                                                               |
| [How it works](src/README.md#how-it-works) | Symbolic exploration, data synthesis, stack/heap preparation, JIT harness, CPU control (cache/branch/...), code patching, hardware counters                                                                                                |

## Requirements

- [x86-64](https://github.com/perf-labs/perf/blob/main/REFERENCES.md#specifications) (requires user-space [rdpmc](https://intel.github.io/SDM/instruction/RDPMC.html) access)
- [Linux 6.x+](https://kernel.org) (requires [linux-perf](https://perfwiki.github.io))
- [Python 3.11+](https://www.python.org) (see [pyproject.toml](pyproject.toml))

## Install

```sh
pip install git+https://github.com/perf-labs/perf.git
```

See [Development](tests/README.md).

## Quick start

```sh
# info
perf system info

# tune
perf system tune  # --status --reset

# validate
perf system test  # measured vs documented

# profile
perf profile -f hot_begin..hot_end -e cycles -o profile.json -- ./a.out

# benchmark
perf benchmark 'imul eax, 0' -m latency -e cycles
perf benchmark snippet.s:label
perf benchmark a.out:func -m throughput -e cycles,instructions
perf benchmark a.out:func -e L1-dcache-load-misses,LLC-load-misses,dTLB-load-misses
perf benchmark a.out:func -e 'topdown-*'
perf benchmark a.out:0x401000..0x401020
perf benchmark a.out:hot_begin..hot_end # see code markers
perf benchmark 'mov rax, [rdi]' --data.rdi=0x42000000000 --data[0x42000000000]=123
perf benchmark /usr/bin/tree:main -m latency -- /path/to/folder
perf benchmark /usr/lib/llvm-20/lib/libc++.so:"std::__1::to_string(int)"
perf benchmark $(echo 'int fn(int n){ return n; }' | g++ -xc++ -c -o cc.o - && echo cc.o):fn

# compare
perf benchmark clang.out:func_v1 --output data/
perf benchmark gcc.out:func_v2 --output data/
perf compare --baseline func_v1 -- data/

# analyze
perf analyze a.out:func --filter 'latency > 4'
perf analyze a.out:func --decompile
perf analyze a.out:func -e 'index,assembly,data*'
perf analyze a.out:func -e assembly | llvm-mca
perf analyze a.out:func --filter '15 in `data.rdi`'
perf analyze a.out:func -- perf.data profile.json

# optimize
perf optimize a.out -- data/
perf optimize a.out --output b.out -- data/
perf optimize a.out --output b.out --opt peephole,branch -- data/
perf optimize a.out --output b.out --opt layout --config.layout.align=64 -- data/

# view/plot
# type = auto: benchmark -> ecdf, analyze -> cfg, multi-value -x -> line
perf benchmark a.out:func | perf view
perf benchmark a.out:func -e cycles,instructions | perf view -e cycles/instructions
perf benchmark a.out:func | perf plot
perf analyze a.out:func | perf plot
perf plot -e instructions/cycles -- perf.data
perf analyze a.out:func | perf plot -t cfg
perf plot -t cfg -- a.out
perf plot -t cfg --config.cfg=emulated -- a.out
```

See [CLI reference](bin/README.md).

```py
import perf

# info
print(perf.system.info())

# tune
perf.system.tune()

# validate
print(perf.system.test())

# profile
df = perf.profile(cmd=["./a.out"], event=["cycles"], filter=[("hot_begin", "hot_end")])

# benchmark
df = perf.benchmark(target={"asm": "imul eax, 0"}, mode=["latency"], event=["cycles"])
df = perf.benchmark(target="imul eax, 0", mode=["latency"], event=["cycles"])
df = perf.benchmark(target={"file": "snippet.s", "symbol": "label"})
df = perf.benchmark(target="snippet.s:label")
df = perf.benchmark(target={"file": "a.out", "symbol": "func"}, event=["cycles", "instructions"])
df = perf.benchmark(target="a.out:func", event=["L1-dcache-load-misses"])
df = perf.benchmark(target="a.out:func", event=["topdown-*"])
df = perf.benchmark(target={"file": "a.out", "region": ["0x401000", "0x401020"]})
df = perf.benchmark(target="a.out:0x401000..0x401020")
df = perf.benchmark(target={"file": "a.out", "region": ["hot_begin", "hot_end"]})
df = perf.benchmark(target="a.out:hot_begin..hot_end")
df = perf.benchmark(target={"asm": "mov rax, [rdi]"}, data={"rdi": 0x42000000000, "0x42000000000": 123})
df = perf.benchmark(target={"file": "/usr/bin/tree", "symbol": "main"}, mode=["latency"])
df = perf.benchmark(target={"file": "/usr/lib/llvm-20/lib/libc++.so", "symbol": "std::__1::to_string(int)"})

# compare
v1 = perf.benchmark(target={"file": "clang.out", "symbol": "func_v1"})
v2 = perf.benchmark(target={"file": "gcc.out", "symbol": "func_v2"})
print(perf.compare(pd.concat([v1, v2]), baseline="func_v1"))

# analyze
df = perf.analyze(target={"file": "a.out", "symbol": "func"}, filter="latency > 4")
df = perf.analyze(target="a.out:func", decompile=True)
df = perf.analyze(target="a.out:func", event=["index", "assembly", "data*"])
df = perf.analyze(target="a.out:func", event=["assembly"])
df = perf.analyze(target="a.out:func", filter="15 in `data.rdi`")
df = perf.analyze(target="a.out:func", data=[perf.data.parse(["perf.data"])])

# optimize
df = perf.optimize(exec="a.out", data=["data/"])
df = perf.optimize(exec="a.out", data=["data/"])
df = perf.optimize(exec="a.out", data=["data/"], opt=["peephole", "branch"])

# view/plot
print(perf.benchmark(target="a.out:func"))
df = perf.benchmark(target="a.out:func", event=["cycles", "instructions"])
print((df["cycles"] / df["instructions"]).describe())
df = perf.benchmark(target="a.out:func")
perf.plot(df)
perf.plot(perf.analyze(target="a.out:func"), type=["cfg"])
```

See [Python API](src/README.md#api).

## Citation

```bibtex
@software{perflabs2026perf,
  author  = {{perf-labs}},
  title   = {{perf}: Profile, Benchmark, Analyze, Optimize},
  year    = {2026},
  url     = {https://github.com/perf-labs/perf}
}
```

## References

See [REFERENCES.md](REFERENCES.md).
