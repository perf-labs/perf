# Development

```sh
git clone https://github.com/perf-labs/perf.git && cd perf
python3 -m venv .venv && . .venv/bin/activate
pip install -e .[test]
pytest
ruff check src tests bin
ruff format --check src tests bin
```
