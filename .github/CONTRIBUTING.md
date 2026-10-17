# How to contribute

Thanks for considering a contribution. Small, focused patches are easiest
to review.

## Workflow

1. Fork the repository and create a branch from `main`.
2. Make your change on that branch.
3. Run the checks below before pushing.
4. Open a pull request against `main` with a clear problem/solution
   description and a linked issue.

## Checks

Run from the repository root:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e .[test]
python -m pytest
lit -v -j2 --timeout 600 tests/x86_64
ruff check src tests bin
ruff format --check src tests bin
```

`bin/` is included because the CLI scripts are Python too.

Tests that cannot run report as unsupported (missing compiler or
FileCheck feature) or skip with a reason (no PMU access). If a measuring
test fails, re-run it alone before investigating (`pytest` test path or
`lit -v tests/x86_64/<name>.test`).

## Patch rules

- Keep `src`, `tests`, and `bin/perf-*` free of comments and docstrings.
  Only the license header stays. Name things well instead.
- Keep each `bin/perf-*` command self-contained. There is no shared CLI
  module and no `tests/helpers.py`.
- Add or update tests with every behavior change, including corner cases
  (empty input, unknown event/column, bad filter, missing file).
- Update docs when flags, config keys, or output columns change
  (`bin/README.md`, `src/README.md`, `tests/README.md`, `SKILL.md`).
- Add a `CHANGELOG.md` entry under `[Unreleased]`.
- One concern per commit. Commit messages use the existing style
  (`:wrench: [area] summary`, `:new: [area] summary`, `:books: [area] summary`).

## Studies

Notebook changes under `studies/` must re-execute cleanly with
`papermill`. Strip large outputs before committing.
See `studies/README.md` for the method.
