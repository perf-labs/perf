# Code Markers

`PERF_LABEL(name)` marks a point in a program.

- Emits **zero instructions**.
- Adds only an `(address, name)` pair to a `.perf.label` section.
- The perf-labs tools read that section and turn each label into a
  counter-reading trampoline at startup (`perf benchmark`, `perf profile`).
- The binary on disk is never modified.

| Language | File | Macro |
| --- | --- | --- |
| C / C++ | `perf.h` | `PERF_LABEL(name)` |
| Rust | `perf.rs` | `perf_label!(name)` |
| Zig | `perf.zig` | `perf_label(name)` |

- C/C++: include as `perf/perf.h` with `-I lib` (or copy it next to the source).
- Rust/Zig: copy the single header next to the source; no include path needed.

## Examples

One `loop_begin` / `loop_end` pair makes the loop a measurable **region**
without measuring the rest of the function.

```c
#include "perf/perf.h"

int work(int n) {
    int s = 1;
    PERF_LABEL(loop_begin);
    for (int i = 0; i < n; i++) s = s * 3 + i;
    PERF_LABEL(loop_end);
    return s;
}

int main() {
    return !work(1000);
}
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

int main() {
    (void)work(1000);
    return 0;
}
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

- `#[inline(never)]` / `@inline(never)` and `no_mangle` keep the compiler from
  folding or renaming `work`; they are not part of the marker.

## Build & Run

```sh
gcc -O2 -I lib -o work_c work.c                          # C
g++ -O2 -I lib -o work_cpp work.cpp                      # C++ (same perf.h)
cp lib/perf/perf.rs . && rustc -O -o work_rust work.rs   # Rust
cp lib/perf/perf.zig . && zig build-exe work.zig -O ReleaseFast  # Zig
```

Every language ends up in the same place: the names are in `.perf.label`.

```sh
perf benchmark work_c --list              # every label and function, with addresses
perf benchmark work_c:loop --list         # markers of one function (regex filter)

perf benchmark work_c:work -m latency -e cycles,instructions
perf benchmark work_c:loop_begin..loop_end -m latency -e cycles
perf benchmark work_c:loop_begin..loop_end --data.rdi=1000 -m latency -e cycles

perf profile -f work -e cycles -- ./work_c                  # live counters
perf profile -f loop_begin -e cycles -o p.json -- ./work_c  # just the marker
```

- Label names are identical in all four binaries, so the commands work
  unchanged on `work_rust` and `work_zig`.
- What differs is what the compiler made of the loop; measure, do not assume.
- If a compiler clones a marked block, the marker is emitted once per copy.
  `--list` says so; a region ending at that name spans to the last copy.
- `foo_begin` / `foo_end` is one region: `foo_begin..foo_end`.
- Whole functions need no labels: `perf benchmark work_c:work`.
- `perf benchmark FILE --list`, `perf analyze FILE --list` and
  `perf profile --list -- FILE` list a file's symbols.
