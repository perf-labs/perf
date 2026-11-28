# Code Markers

`PERF_LABEL(name)` marks a point in a program.

| Property       | Value                                            |
| -------------- | ------------------------------------------------ |
| Instructions   | Zero                                             |
| ELF artifact   | One `(address, name)` pair in `.perf.label`      |
| Consumers      | `perf benchmark`, `perf analyze`; `perf profile` |

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

auto work(int n) -> int {
    auto s = 1;
    PERF_LABEL(loop_begin);
    for (int i = 0; i < n; i++) {
      s = s * 3 + i;
    }
    PERF_LABEL(loop_end);
    return s;
}

int main() {
    return work(1000) == 0;
}
```

Save the same code as `work.cpp` for C++.

```rust
include!("perf.rs");

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

## Build & Run

| Language | Build                                                             |
| -------- | ----------------------------------------------------------------- |
| C        | `gcc -O2 -I lib -o work_c work.c`                                 |
| C++      | `g++ -O2 -I lib -o work_cpp work.cpp` (same `perf.h`)             |
| Rust     | `cp lib/perf/perf.rs . && rustc -O -o work_rust work.rs`          |
| Zig      | `cp lib/perf/perf.zig . && zig build-exe work.zig -O ReleaseFast` |

## Usage

```sh
objdump -s -j .perf.label work_c
```

```sh
perf benchmark work_c --list # every label and function
perf benchmark work_c:loop --list # regex filter
perf benchmark work_c:work -m latency -e cycles,instructions
perf benchmark work_c:loop_begin..loop_end -m latency -e cycles
perf profile -f work -e cycles -- ./work_c
perf profile -f loop_begin -e cycles -o p.json -- ./work_c
```
