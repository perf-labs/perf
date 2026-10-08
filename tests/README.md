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
pytest                          # everything, about a minute
pytest tests/test_bench.py -q   # one file
pytest -k ProcessStack          # one topic
pytest -x -q                    # stop at the first failure
```

Tests that cannot run skip with a reason (missing `gcc`, `perf`,
or PMU access), so a green run does not mean the machine had all of it.
If a measuring test fails, re-run it alone before investigating.

## Check

CI runs exactly this, so run it before pushing:

```sh
pytest
ruff check src tests bin
ruff format --check src tests bin
```

`bin/` is included because the CLI scripts are Python too.
