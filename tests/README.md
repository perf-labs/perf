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
for f in /sys/devices/{cpu,cpu_core,cpu_atom}/rdpmc; do
  [ -e "$f" ] && echo 2 | sudo tee "$f"
done
```

Without it only `duration_time` works; the rest skip.

## Run

```sh
pytest --cov                       # unit/API tests with fakes and controlled inputs + coverage
lit -v -j2 --timeout 600 tests/x86_64   # end-to-end CLI results (-j2 bounds angr memory)
pytest tests/test_bench.py -q --cov  # one file
pytest -k ProcessStack             # one topic
pytest -x -q                       # stop at the first failure
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

`tests/x86_64/*.test` covers the end-to-end CLI for every command; `pytest`
keeps the unit/API scope (parsing, formatting, statistics, mocks and controlled
fixtures). Architectural or compiler-specific behavior is exercised through the
CLI in lit: isolation failures, faked streams, layout shuffling, data-path
joins, and profile selection including hex points. The `optimize` tests exercise each
pass, layout fixups, data-source aggregation, filters, default output,
JSON/list/dry-run output, embedded data and the safe fallbacks (non-PIE,
jump tables, exceptions, unreachable fixups).

## Check

CI runs exactly this, so run it before pushing:

```sh
pytest --cov
lit -v -j2 --timeout 600 tests/x86_64
ruff check src tests bin
ruff format --check src tests bin
```

Coverage must stay at or above 80% (`fail_under = 80` in
`pyproject.toml`). Reports land in `htmlcov/index.html` (or `coverage.json`).

CI then re-runs every notebook in `studies/**/*.ipynb` with papermill
when hardware counters are available (see `.github/workflows/linux.yml`):

```sh
for notebook in $(find studies -name "*.ipynb" -not -path "*checkpoint*"); do
  papermill "$notebook" "/tmp/perf-notebooks/$(echo "$notebook" | sed 's|studies/||; s|/|-|g')"
done
```
