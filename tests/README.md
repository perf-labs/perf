# Tests

## Setup

```sh
git clone https://github.com/perf-labs/perf.git && cd perf
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[test]' # installs the perf-* commands and test tools
```

## Run

| Suite      | Command                                 | Notes                                       |
| ---------- | --------------------------------------- | ------------------------------------------- |
| Unit + API | `pytest --cov`                          | Python tests with coverage                  |
| End-to-end | `lit -v -j2 --timeout 600 tests/x86_64` | CLI result tables; `-j2` bounds angr memory |
| One module | `pytest tests/test_bench.py -q --cov`   |                                             |
| One topic  | `pytest -k ProcessStack`                |                                             |
| Stop early | `pytest -x -q`                          |                                             |

Coverage must stay at or above 90% (`fail_under` in [pyproject.toml](../pyproject.toml)).
Generate the reports explicitly:

```sh
coverage report
coverage html      # .htmlcov/index.html
coverage json      # .coverage.json
```

Use `# pragma: no cover` only for defensive or platform-specific branches.
Hardware paths are covered by lit and the notebooks.

## Lit shape

Each `tests/x86_64/*.test` is a lit shell test. A complete, current example is
[tests/x86_64/analyze_mov.test](x86_64/analyze_mov.test):

```text
# REQUIRES: x86_64
# RUN: perf analyze 'mov eax, 42' -e assembly,latency,throughput 2>&1 | %FileCheck %s
# CHECK-NOT: Traceback
# CHECK: assembly latency throughput
# CHECK-NEXT: mov eax, 0x2a 1.00 0.20
# CHECK-NOT: Traceback
```

`FileCheck` must be on `PATH`; CI installs LLVM if needed.

## Before pushing

CI runs (see [.github/workflows/linux.yml](../.github/workflows/linux.yml)):

```sh
pytest --cov -q
coverage report
coverage json --fail-under=90
lit -v -j2 --timeout 600 tests/x86_64
ruff check src tests bin
ruff format --check src tests bin
```

Release also runs the notebook sweep and the end-to-end smoke suite
(see [.github/workflows/release.sh](../.github/workflows/release.sh)):

```sh
find studies -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -print0 | xargs -0 -I{} papermill {} /dev/null
.github/workflows/perf.sh  # authoritative CLI smoke suite
```

When hardware counters are available, CI also reruns every notebook:

```sh
mkdir -p /tmp/perf-notebooks
find studies -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -print0 |
  while IFS= read -r -d '' notebook; do
    name=$(printf '%s' "${notebook#studies/}" | sed 's|/|-|g')
    papermill "$notebook" "/tmp/perf-notebooks/$name"
  done
```
