# Annotations

`PERF_LABEL(name)` marks a point in a program. It emits **zero instructions** —
only a `(address, name)` pair in a `.perf.label` section — so the cost of
annotating is nothing at runtime.

The perf-labs tools read that section and turn every label into a
counter-reading trampoline at startup (`perf benchmark`, `perf profile`); the
binary on disk is never modified. Without a trampoline the labels are simply
metadata.

| Language | File | Macro |
| --- | --- | --- |
| C / C++ | `perf.h` | `PERF_LABEL(name)` |
| Rust | `perf.rs` | `perf_label!(name)` |
| Zig | `perf.zig` | `perf_label(name)` |

## Compile

Headers are used as-is; add the directory to the include path.

```sh
gcc -O2 -I lib/perf -o a.out a.c
g++ -O2 -I lib/perf -o a.out a.cpp
zig cc -O2 -I lib/perf -o a.out a.c
rustc -O -I lib/perf --cfg 'feature="crt-static"' -o a.out a.rs
```

C and C++ share `perf.h`; it picks the right inline-asm dialect for the
compiler (gcc gets `asm goto`, clang gets `asm volatile`).

Rust and Zig do not need the include path at all — `perf.rs` and `perf.zig` are
single files you copy next to your source.

## Run

Write a pair of labels around the code you care about:

```c
#include <perf.h>

int work(int n) {
    int s = 0;
    PERF_LABEL(loop_begin);
    for (int i = 0; i < n; i++) {
        s += i;
    }
    PERF_LABEL(loop_end);
    return s;
}
```

List what is measurable, then measure it:

```sh
perf info a.out                      # every label and function, with addresses
perf benchmark a.out: --list         # the same table, as benchmark targets
perf profile --list -- ./a.out       # the same table, as profile targets
perf benchmark a.out:work -m latency -e cycles,instructions
perf benchmark a.out:loop_begin..loop_end -m latency -e cycles
perf profile -e cycles -- ./a.out     # live counters for the whole process
```

A `foo_begin` / `foo_end` pair is one **region**: target it as
`foo_begin..foo_end`. Whole functions work the same way and need no labels at
all (`perf benchmark a.out:work`). Labels are portable section metadata present
in every language binding, and are discoverable by `perf info a.out`,
`perf benchmark a.out: --list` and `perf profile --list -- ./a.out`.

## Rules

- A label name is used verbatim, so keep it unique and short.
- Labels are addresses: moving code moves them, so use
  `perf benchmark a.out: --list` (never a hard-coded address) as the source of
  truth for names.
- `perf benchmark` calls the target directly, so it must run with just its
  arguments. If the function needs the program's set-up, pass it as
  `--setup a.out:SYMBOL` and run it as `--teardown a.out:SYMBOL`; if it still faults, use a
  `PERF_LABEL` region instead.
- On a hybrid CPU only some core types have PMU counters; the tools pin
  themselves to a capable CPU.
