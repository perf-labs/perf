## Getting set up

```sh
git clone https://github.com/perf-labs/perf.git && cd perf
python3 -m venv .venv && . .venv/bin/activate
pip install -e .[test]
```

`pip install -e .[test]` also installs the seven `perf-*` scripts, so `perf
benchmark`, `perf view` and the rest work from any directory. Without them the
tests still pass — they load the scripts straight out of `bin/` — but you lose
the commands themselves.

## What you need on the machine

Everything below is soft: a test that cannot run skips itself with a reason, so
a green run does not mean the machine had all of it.

| Tool | Needed for | Missing means |
| --- | --- | --- |
| x86-64 Linux, kernel 6.x | everything | — |
| `gcc` | fixtures built by `test_code.py`, `test_elf.py`, `test_info.py` | those classes skip |
| `g++` | the C++/label fixtures in `test_perf.py`, `test_prof.py` | those classes skip |
| `perf` (linux-perf) | tests that parse a recorded `perf.data` | those tests skip |
| a readable PMU (`/proc/sys/kernel/perf_event_paranoid` ≤ 2, `rdpmc` enabled) | the measured paths | they skip or report a permission error |
| `zstd`/`lz4` fixtures | `test_data.py` compressed records | those tests skip |

To enable user-mode `rdpmc` for the hardware events:

```sh
echo 2 | sudo tee /sys/devices/system/cpu/cpu_core/rdpmc
```

## Running

```sh
pytest                          # everything
pytest tests/test_arch.py       # one file
pytest -k ProcessStack          # one topic
pytest -q tests/test_bench.py::TestBackend::test_asm_loop_single_copy
pytest -x -q                    # stop at the first failure
```

A full run takes about a minute; most of it is `test_prof.py` and
`test_bench.py`, which really do fork, assemble and count. Tests that measure
are also the flaky ones — if one fails, run it again on its own before you go
looking for a cause:

```sh
pytest tests/test_bench.py -q            # re-run; nothing to install
```

## The gates

CI runs exactly this, so run it before you push:

```sh
pytest
ruff check src tests bin
ruff format --check src tests bin
```

`ruff check` is the lint gate (`E`, `F`, `I`, `UP`, `W`; line length 88,
`known-first-party = ["perf"]`). `ruff format --check` is the format gate. Both
read `pyproject.toml`, so there is no separate config to keep in sync, and
`bin/` is included because the seven CLI scripts are Python too, just without a
`.py` suffix.

ruff is pinned to a release series (`ruff>=0.15,<0.16`) because it moves
formatting between releases: unpinned, the same tree can pass locally and fail
in CI, or the reverse. If a ruff upgrade changes the formatting, the upgrade and
the reformat belong in one commit, and `pyproject.toml` with it.

Markdown is part of the format gate: `test_perf.py::TestSourceLayout::
test_python_examples_in_the_docs_are_formatted` runs `ruff format --check` over
every ```` ```py ```` block in every `*.md` in the repo, so a code sample in a
README cannot drift from what the formatter would write. Fix a failure with

```sh
ruff format src/README.md        # only works in preview mode, see below
```

or by hand — the diff `ruff format --check` prints names the file, the line and
the exact replacement.

> `ruff format` on a `.md` file needs preview mode (`ruff format --preview`),
> and preview mode also changes how it formats `.py` files. The repository is
> therefore formatted with the stable style, and the doc test formats the
> extracted blocks with the stable style too, so both agree.

## Conventions the tests enforce

These are not style preferences; they are `test_perf.py` assertions, so a
change that breaks them fails the suite:

- `src/perf/**/*.py` is laid out as imports → constants → public definitions →
  private definitions, in that order (`TestSourceLayout`). Move a new helper
  below the public section rather than next to its caller.
- `src/perf/**` has no docstrings (`test_no_docstrings_in_src`). Say what a
  thing does in `bin/README.md`, `src/README.md` or a comment at the call site.
- The license header is the only comment in the tree
  (`test_only_the_license_header_is_commented`, and every source has to carry
  it: `test_every_source_carries_the_license_header`).
- Blank lines are spelled out, not left to taste
  (`test_blank_lines_are_consistent`): none between two constants, one after
  the imports when a statement follows and two when a definition does, two
  around every top-level definition, one inside a body, none between a header
  and its first statement. `ruff format` agrees, so the two never disagree.
- Everything public is re-exported from `perf`, `perf.arch` and `perf.exec`,
  once each (`test_everything_is_exported_once`). A new public name needs an
  entry in that module's `__all__`.
- `src/perf/exec/elf.py` may not name an x86 symbol or an architecture string
  (`test_elf_leaves_x86_to_the_arch_module`); the only exception is the
  `ElfConst.R_X86_64_*` alias, whose values must stay the arch module's
  (`tests/test_arch.py`). Instruction bytes, encodings, the disassembler and
  the word/stack shape belong in `src/perf/arch/x86_64.py`.
- User-facing failures are messages, not exceptions: the library raises
  `ValueError`/`TypeError` and the CLI in `bin/` turns the first line into the
  text it exits with. Tests assert on that text, so a new error should be
  raised with a sentence that starts with what the user did wrong.

## Adding a test

Put it next to the code it covers, in the file for that area, inside a
`TestCase` class named after the thing (`TestProcessStack`, not `TestElf3`). Name
the method for the behaviour, not the function under test
(`test_argc_and_the_vectors_are_written`, not `test_setup_argv`), and assert on
observable results rather than on internals — the internals move.

If a test needs a binary, compile one into a temporary directory in `setUp`
and skip if the compiler is missing, the way `_build_exe` and
`_binary_with_argv` do in `test_perf.py`. If it needs hardware, call
`_needs_perf()` first and let `SkipTest` do the talking.

Tests that measure real time should pin everything they can: a fixed config
(`_FAST` in `test_perf.py`), a fixed seed (`--config.seed`), and `assertFalse
(df.empty)` rather than a threshold on the measured value. A benchmark number
is never an assertion.
