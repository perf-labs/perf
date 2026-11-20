# Tests

## Setup

```sh
git clone https://github.com/perf-labs/perf.git && cd perf
python3 -m venv .venv && . .venv/bin/activate
pip install -e .[test] # installs the `perf-*` commands, so `perf benchmark` works from any directory.
```

## Tune

```
perf system tune # --profile=latency
```

## Run

```sh
pytest --cov                            # unit/API tests with coverage
lit -v -j2 --timeout 600 tests/x86_64   # end-to-end CLI results (-j2 bounds angr memory)
pytest tests/test_bench.py -q --cov     # one file
pytest -k ProcessStack                  # one topic
pytest -x -q                            # stop at the first failure
```

Lit test shape:

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

CI runs exactly this; run it before pushing:

```sh
pytest --cov
lit -v -j2 --timeout 600 tests/x86_64
ruff check src tests bin
ruff format --check src tests bin
```

- Coverage must stay at or above 90% (`fail_under` in `pyproject.toml`).
- Use `# pragma: no cover` only for defensive or platform-specific branches.
- Hardware and end-to-end paths are covered by lit and the notebooks.
- Reports: `htmlcov/index.html` or `.coverage.json`.

When hardware counters are available, CI also re-runs every notebook:

```sh
find studies -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -print0 |
  while IFS= read -r -d '' notebook; do
    papermill "$notebook" "/tmp/perf-notebooks/$(echo "$notebook" | sed 's|studies/||; s|/|-|g')"
  done
```
