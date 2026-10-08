# Code Markers

`PERF_LABEL(name)` marks a point in a program. It emits **zero instructions** —
only a `(address, name)` pair in a `.perf.label` section — so the cost of
marking is nothing at runtime.

The perf-labs tools read that section and turn every label into a
counter-reading trampoline at startup (`perf benchmark`, `perf profile`); the
binary on disk is never modified. Without a trampoline the labels are simply
metadata.

| Language | File | Macro |
| --- | --- | --- |
| C / C++ | `perf.h` | `PERF_LABEL(name)` |
| Rust | `perf.rs` | `perf_label!(name)` |
| Zig | `perf.zig` | `perf_label(name)` |

C and C++ share `perf.h`; it picks the right inline-asm dialect for the
compiler (gcc gets `asm goto`, clang gets `asm volatile`). Include it as
`perf/perf.h` with `-I lib` (or copy `lib/perf/perf.h` next to your source and
include it as `perf.h`). Rust and Zig need no include path — `perf.rs` and
`perf.zig` are single files you copy next to your source.

## Examples

The same function in every language. One `loop_begin` / `loop_end` pair around
the loop, so the loop is a **region** that can be measured without measuring
the rest of the function.

```c
#include "perf/perf.h"

int work(int n) {
    int s = 1;
    PERF_LABEL(loop_begin);
    for (int i = 0; i < n; i++) s = s * 3 + i;
    PERF_LABEL(loop_end);
    return s;
}

int main(void) { return work(1000) > 0 ? 0 : 1; }
```

```cpp
#include "perf/perf.h"

int work(int n) {
    int s = 1;
    PERF_LABEL(loop_begin);
    for (int i = 0; i < n; i++) s = s * 3 + i;
    PERF_LABEL(loop_end);
    return s;
}

int main() { return work(1000) > 0 ? 0 : 1; }
```

```rust
include!("perf.rs");

#[inline(never)]
#[no_mangle]
pub extern "C" fn work(n: i32) -> i32 {
    let mut s: i32 = 1;
    perf_label!(loop_begin);
    for i in 0..n {
        s = s * 3 + i;
    }
    perf_label!(loop_end);
    s
}

fn main() {
    println!("{}", work(1000));
}
```

```zig
const perf = @import("perf.zig");

extern "C" fn work(n: i32) i32 {
    var s: i32 = 1;
    perf.perf_label("loop_begin");
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        s = s * 3 + i;
    }
    perf.perf_label("loop_end");
    return s;
}

pub fn main() void {
    _ = work(1000);
}
```

`#[inline(never)]`/`@inline(never)` and `no_mangle` are what keep the compiler
from folding `work` away or renaming it into something the tools cannot find;
they are not part of the marker.

## Build & Run

```sh
gcc -O2 -I lib -o work_c work.c          # C
g++ -O2 -I lib -o work_cpp work.cpp      # C++ (same perf.h)
cp lib/perf/perf.rs . && rustc -O -o work_rust work.rs   # Rust (no -I needed)
cp lib/perf/perf.zig . && zig build-exe work.zig -O ReleaseFast   # Zig (no -I needed)
```

Every language ends up in the same place: the label names are in the binary's
`.perf.label` section, and the tools find them from there.

```sh
perf info work_c                     # every label and function, with addresses
perf info work_c | grep loop         # just this function's markers

perf benchmark work_c:work -m latency -e cycles,instructions
perf benchmark work_c:loop_begin..loop_end -m latency -e cycles
perf benchmark work_c:loop_begin..loop_end --data.rdi=1000 -m latency -e cycles

perf profile -e cycles -- ./work_c                       # live counters
perf profile -f loop_begin -e cycles -o p.json -- ./work_c # just the marker
```

The label names are identical in all four binaries, so the same commands work
on `work_rust` or `work_zig` unchanged. What differs is only what the compiler
made of the loop, which is the point: measure, do not assume.

If a compiler clones a marked block (loop unrolling does), the marker is
emitted once per copy. `perf info` says so, and a region ending at that name
spans to the last copy.

A `foo_begin` / `foo_end` pair is one **region**: target it as
`foo_begin..foo_end`. Whole functions work the same way and need no labels at
all (`perf benchmark work_c:work`).

`perf info FILE` is the one place that lists a file's targets; a target that
does not resolve prints that same table before it exits.
