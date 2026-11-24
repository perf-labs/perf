# Code Markers

`PERF_LABEL(name)` marks a point in a program.

| Property       | Value                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------ |
| Instructions   | Zero                                                                                       |
| ELF artifact   | One `(address, name)` pair in `.perf.label`                                                |
| Consumers      | `perf benchmark`, `perf analyze` (region boundaries); `perf profile` (counter trampolines) |
| Binary on disk | Never modified                                                                             |

| Language | File       | Marker               |
| -------- | ---------- | -------------------- |
| C / C++  | `perf.h`   | `PERF_LABEL(name)`   |
| Rust     | `perf.rs`  | `perf_label!(name)`  |
| Zig      | `perf.zig` | `perf_label("name")` |

C/C++ includes `perf/perf.h` with `-I lib`, or copies the header next to the
source. Rust and Zig copy the single header next to the source.

## Example

One `loop_begin` / `loop_end` pair makes the loop a measurable **region**
without measuring the rest of the function.

```c
#include "perf/perf.h"

__attribute__((noinline)) int work(int n) {
    int s = 1;
    PERF_LABEL(loop_begin);
    for (int i = 0; i < n; i++) s = s * 3 + i;
    PERF_LABEL(loop_end);
    return s;
}

int main(void) {
    return work(1000) == 0;
}
```

Save the same code as `work.cpp` for C++.

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

export fn work(n: i32) i32 {
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

Keep `work` visible to the tools: C/C++ uses `__attribute__((noinline))`, Rust
uses `#[inline(never)]` and `#[no_mangle]`, and Zig uses `export`. These are
compiler hints, not part of the marker.

## Build & Run

| Language | Build                                                             |
| -------- | ----------------------------------------------------------------- |
| C        | `gcc -O2 -I lib -o work_c work.c`                                 |
| C++      | `g++ -O2 -I lib -o work_cpp work.cpp` (same `perf.h`)             |
| Rust     | `cp lib/perf/perf.rs . && rustc -O -o work_rust work.rs`          |
| Zig      | `cp lib/perf/perf.zig . && zig build-exe work.zig -O ReleaseFast` |

Every language ends up in the same place: the names are in `.perf.label`.

```sh
perf benchmark work_c --list              # every label and function
perf benchmark work_c:loop --list         # regex filter
perf benchmark work_c:work -m latency -e cycles,instructions
perf benchmark work_c:loop_begin..loop_end -m latency -e cycles

perf profile -f work -e cycles -- ./work_c
perf profile -f loop_begin -e cycles -o p.json -- ./work_c
```

| Rule          | Meaning                                                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------------------- |
| Same names    | All four binaries expose the same label names; commands work unchanged on `work_rust` and `work_zig`              |
| Measure       | What differs is what the compiler made of the loop; measure, do not assume                                        |
| Cloned blocks | A cloned marker is emitted once per copy; `--list` warns, and a region ending at that name spans to the last copy |
| Region form   | `foo_begin` / `foo_end` is one region: `foo_begin..foo_end`                                                       |
| Functions     | Whole functions need no labels: `perf benchmark work_c:work`                                                      |
| Listing       | `perf benchmark FILE --list`, `perf analyze FILE --list`, `perf profile --list -- FILE`                           |
