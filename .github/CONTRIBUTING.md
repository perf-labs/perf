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

## Rules

- Add or update tests
- Update docs, skills, configs
- Ensure checks, tests, studies are passinf
