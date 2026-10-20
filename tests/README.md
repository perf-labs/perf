## Setup

```sh
git clone https://github.com/perf-labs/perf.git && cd perf
python3 -m venv .venv && . .venv/bin/activate
pip install -e .[test]
```

This also installs the `perf-*` commands, so `perf benchmark` works
from any directory.

For hardware counters, enable user-space `rdpmc` once:

```sh
echo 2 | sudo tee /sys/devices/{cpu_core,cpu_atom}/rdpmc
```

Without it only `duration_time` works; the rest skip.

## Run

```sh
pytest                              # unit tests
lit -v -j2 --timeout 600 tests/x86_64   # regression tests (-j2 bounds angr memory)
pytest tests/test_bench.py -q       # one file
pytest -k ProcessStack              # one topic
pytest -x -q                        # stop at the first failure
```

```text
# REQUIRES: x86_64, gcc
# RUN: gcc -O2 -fPIE -pie -o %t %S/benchmark_entry.c
# RUN: perf benchmark %t:main -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 2>&1 | %FileCheck %s
# CHECK-NOT: Traceback
# CHECK: file name mode
# CHECK: iterations samples operations threads duration_time
# CHECK-NEXT: main latency
# CHECK: 10 0 1 1 {{[0-9.]+(ns|us|ms|s)}}
# CHECK-NOT: Traceback
```

## Check

CI runs exactly this, so run it before pushing:

```sh
pytest
lit -v -j2 --timeout 600 tests/x86_64
ruff check src tests bin
ruff format --check src tests bin
```

CI then re-runs every notebook in `studies/**/*.ipynb` with papermill
when hardware counters are available (see `.github/workflows/linux.yml`):

```sh
for notebook in $(find studies -name "*.ipynb" -not -path "*checkpoint*"); do
  papermill "$notebook" "/tmp/perf-notebooks/$(echo "$notebook" | sed 's|studies/||; s|/|-|g')"
done
```
