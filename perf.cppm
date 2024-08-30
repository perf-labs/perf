// <!--
//
// Copyright (c) 2024-2025 Kris Jusiak <kris@jusiak.net>
//
// | license          | namespace      | guard            | description                        |
// | ---------------- | -------------- | ---------------- | ---------------------------------- |
// | MIT [1]          | `perf::*`      | -                | https://opensource.org/license/mit |
// | Apache2+LLVM [2] | `perf::mca::*` | `PERF_LLVM == 1` | https://llvm.org/LICENSE.txt       |
//
// [1] The MIT License (MIT)
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// See https://opensource.org/license/mit for license information.
//
// [2] Apache License v2.0 with LLVM exceptions
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#if 0
// -->
[Overview](#Overview) / [Features](#Features) / [Requirements](#Requirements) / [Showcase](#Showcase) / [API](#API) / [FAQ](#FAQ) / [User Guide](#User-Guide) / [License](#License)

## `perf`: C++2x Performance library

  > [!WARNING]
  > Under development

  [![Version](https://img.shields.io/github/v/release/qlibs/perf)](https://github.com/qlibs/perf/releases)
  [![Build](https://github.com/qlibs/perf/actions/workflows/linux.yml/badge.svg)](https://github.com/qlibs/perf/actions/workflows/linux.yml)
  [![Try it online](https://img.shields.io/badge/try%20it-online-blue.svg)](https://godbolt.org/z/58hcsz6W9)

  > [`Performance`](https://en.wikipedia.org/wiki/Computer_performance) is not a number!

  <a href="https://kris-jusiak.github.io/talks/cppcon-2025/#/0/1"><img src="https://github.com/user-attachments/assets/66c7515d-9098-4a5a-a092-9eb0c1c910d0"/></a>

### Overview

  > Single [`header`](https://github.com/qlibs/perf/blob/main/perf)/[`module`](https://github.com/qlibs/perf/blob/main/perf.cppm) `performance` library that combines the power of: <br/>
    [`c++2x`](https://en.cppreference.com/w/cpp/23),
    [`linux/perf`](https://perf.wiki.kernel.org),
    [`llvm/mca`](https://llvm.org/docs/CommandGuide/llvm-mca.html),
    [`gnuplot/sixel`](http://www.gnuplot.info),
    ...

### Features

  > <details>
  >   <summary>Profiling, Analyzing, Benchmarking, Testing</summary>
  >
  > | names | description | [API](#API) |
  > | ----------| ----------- | --- |
  > | [`info`](https://github.com/qlibs/perf/blob/main/perf.cppm#L39) | hardware/software info | `compiler`, `cpu`, `memory`, `sys`, `proc`, `bin` |
  > | [`[core]`](https://github.com/qlibs/perf/blob/main/perf.cppm#L39) | low-level utilities | `code`, `compiler`, `cpu`, `memory` |
  > | [`profiler`](https://github.com/qlibs/perf/blob/main/perf.cppm#L39) | timing ([rdtsc](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=rdtsc)/[clock](https://man7.org/linux/man-pages/man3/clock_gettime.3.html)/[chrono](https://en.cppreference.com/w/cpp/header/chrono)) | `tsc`, `process_time`, `thread_time`, `steady_time`, |
  > | [`profiler`](https://github.com/qlibs/perf/blob/main/perf.cppm#L39) | counting ([linux/perf](https://perf.wiki.kernel.org)) | `instructions`, `cycles`, ..., [`top_down`](https://github.com/andikleen/pmu-tools/wiki/toplev-manual) |
  > | [`profiler`](https://github.com/qlibs/perf/blob/main/perf.cppm#L39) | sampling ([linux/perf](https://perf.wiki.kernel.org)) | `instructions`, `cycles`, ..., `mem_loads`, `mem_stores`, [`top_down`](https://github.com/andikleen/pmu-tools/wiki/toplev-manual) |
  > | [`profiler`](https://github.com/qlibs/perf/blob/main/perf.cppm#L39) | tracing ([linux/intel_pt](https://www.intel.com/content/www/us/en/products/docs/processors/core/core-technical-resources.html)) | `instructions`, `cycles`, `tsc` |
  > | [`analyzer`](https://github.com/qlibs/perf/blob/main/perf.cppm#L39) | disassembling ([llvm](https://github.com/llvm/llvm-project)) | `mca::assembly`, `mca::address`, `mca::encoding`, `mca::size`, `mca::uops`, `mca::latency`, `mca::rthroughput`, `mca::may_load`, `mca::may_store`, `mca::has_side_effects`, ..., [`mca::debug::source`](https://man7.org/linux/man-pages/man1/addr2line.1.html) |
  > | [`analyzer`](https://github.com/qlibs/perf/blob/main/perf.cppm#L39) | analyzing ([llvm/mca](https://llvm.org/docs/CommandGuide/llvm-mca.html)) | `mca::timeline`, `mca::resource_pressure`, `mca::bottleneck` |
  > | [`runner`](https://github.com/qlibs/perf/blob/main/perf.cppm#L39) | benchmarking | `bench::baseline`, [`bench::latency`](#FAQ), [`bench::throughput`](#FAQ) |
  > | [`[io]`](https://github.com/qlibs/perf/blob/main/perf.cppm#L39) | logging/plotting ([gnuplot/sixel](http://www.gnuplot.info)) | `log`, `json`, `report`, `annotate`, `plot` ([`hist`](https://en.wikipedia.org/wiki/Histogram), [`box`](https://en.wikipedia.org/wiki/Box_plot), [`bar`](https://en.wikipedia.org/wiki/Bar_chart), [`line`](https://en.wikipedia.org/wiki/Line_chart), [`ecdf`](https://en.wikipedia.org/wiki/Empirical_distribution_function)) |
  >
  > [] - `inline namespace`
  >  </details>

### Requirements

  >  <details>
  >    <summary>Minimal <i>(Core functionality)</i></summary>
  >
  >    - ([`clang-16+`](https://clang.llvm.org) | [`gcc-12+`](https://gcc.gnu.org)) / [`c++2x+`](https://en.cppreference.com/w/cpp/23)
  >
  >  </details>
  >
  >  <details>
  >    <summary>Optimal <i>(All Features)</i></summary>
  >
  >    - [`intel-12th+`](https://www.intel.com/content/www/us/en/products/docs/processors/core/12th-gen-processors.html) with [`PEBS`](https://www.intel.com/content/www/us/en/developer/articles/technical/timed-process-event-based-sampling-tpebs.html), [`IPT`](https://www.intel.com/content/www/us/en/products/docs/processors/core/core-technical-resources.html) support
  >      - [`libipt`](https://github.com/intel/libipt) - `apt-get install libipt-dev`
  >    - ([`clang-19+`](https://clang.llvm.org) | [`gcc-13+`](https://gcc.gnu.org)) / [`c++2x+`](https://en.cppreference.com/w/cpp/23)
  >      - [`llvm-19+`](https://llvm.org) - `apt-get install llvm-dev`
  >    - [`linux-6.x+`](https://www.kernel.org)
  >      - [`perf-event-open`](https://perf.wiki.kernel.org) - `apt-get install linux-tools-common`
  >    - [`terminal`](https://invisible-island.net/xterm) with [`sixel`](https://www.arewesixelyet.com) support
  >      - [`gnuplot`](http://www.gnuplot.info) - `apt-get install gnuplot`
  >
  >  </details>
  >
  >  <details>
  >    <summary>Auxiliary <i>(Enhancements)</i></summary>
  >
  >    - [`ut`](https://github.com/qlibs/ut) - `https://github.com/qlibs/ut`
  >    - [`gh`](https://cli.github.com) - `apt-get install gh`
  >
  >  </details>
  >
  > [`Dockerfile`](https://github.com/qlibs/perf/blob/main/.github/workflows/Dockerfile)

### Showcase

  > <details>
  >   <summary>Usage</summary>
  >
  >   ```cpp
  >   /**
  >    * Info/Core
  >    */
  >   import perf; // #include <perf>
  >
  >   int main() {
  >     auto&& spec = perf::info::spec{{
  >       {"time",          std::chrono::system_clock::now()},
  >       {"sys.triple",    perf::info::sys::triple()},
  >       {"cxx.name",      perf::info::compiler::name()},
  >       {"cxx.version",   perf::info::compiler::version()},
  >       {"cpu.name",      perf::info::cpu::name()},
  >       {"cpu.code_name", perf::info::cpu::code_name()},
  >       {"cpu.version",   perf::info::cpu::version()},
  >       {"mem.cache",     perf::info::memory::dcache())},
  >       // ...
  >     }};
  >
  >     perf::log(spec);
  >   }
  >   ```
  >
  >   ```cpp
  >   /**
  >    * Profiling/Analyzing
  >    */
  >   import perf; // #include <perf>
  >
  >   int main() {
  >     perf::profiler profiler{
  >       perf::stat::tsc, perf::stat::cycles, perf::trace::instructions,
  >     };
  >     perf::analyzer analyzer{
  >       perf::mca::assembly, perf::mca::timeline,
  >     };
  >
  >     auto invoke = [&](auto&& fn, auto&&... ts) {
  >       profiler.start();
  >       perf::compiler::prevent_elision(fn(ts...));
  >       profiler.stop();
  >     };
  >
  >     invoke(fn, std::rand());
  >
  >     perf::log(profiler[]);
  >     perf::log(analyzer[]);
  >
  >     analyzer << profiler[perf::trace::instructions];
  >
  >     perf::verify(profiler[perf::stat::tsc] > 0ns);
  >     perf::verify(analyzer[perf::mca::timeline][0u].cycle_dispatched);
  >     perf::verify(analyzer[perf::mca::assembly][0u].contains("add"));
  >   }
  >   ```
  >
  >   ```cpp
  >   /**
  >    * Benchmarking
  >    */
  >   import perf; // #include <perf>
  >
  >   int main() {
  >     perf::runner bench{perf::bench::latency{}};
  >
  >     bench(fn, perf::data::sequence<int>{{3, 5, 15}});
  >     bench(fn, perf::data::range<int>{.start = 0, .stop = 15});
  >     bench(fn, perf::data::unpredictable<int>);
  >
  >     perf::report(bench[perf::stat::tsc]);
  >     perf::plot::bar(bench[perf::stat::tsc]);
  >     perf::annotate<perf::vsplit>(bench[perf::mca::assembly]);
  >   }
  >   ```
  >
  >   [`API`](#API)
  >
  >  </details>
  >
  > <details>
  >   <summary>Build & Test</summary>
  >
  >   ```sh
  >   # module
  >   clang++ -std=c++23 -O3 -I. --precompile perf.cppm # -DNTEST disables compile-time tests
  >   clang++ -std=c++23 -O3 -fprebuilt-module-path=. perf.pcm <source_file> -lLLVM -lipt
  >
  >   # header
  >   $CXX -std=c++23 -O3 -I. <source_file> -lLLVM -lipt # -DNTEST disables compile-time tests
  >   ```
  >
  >   ```
  >   .github/scripts/tune.sh # See #FAQ for more
  >   ```
  >
  >   [`scripts/tune.sh`](https://github.com/qlibs/perf/blob/main/.github/scripts/tune.sh) <br/>
  >   [`Dockerfile`](https://github.com/qlibs/perf/blob/main/.github/workflows/Dockerfile)
  >
  >  </details>
  >
  > <details>
  >   <summary>Export & Share</summary>
  >
  >   ```asm
  >   ./a.out | .github/scripts/export.sh html | gh gist create --public
  >   ```
  >
  >   [`scripts/export.sh`](https://github.com/qlibs/perf/blob/main/.github/scripts/export.sh) - `html`, `markdown`, `notebook` <br/>
  >   [`gh`](https://gist.github.com) - `apt-get install gh`
  >
  > </details>
  >
  > [`Studies`](https://github.com/qlibs/perf/wiki/studies)

### API

  > <details>
  >   <summary>Configuration</summary>
  >
  >   ```cpp
  >   /**
  >    * GNU # default: deduced based on `__GNUC__`
  >    * - 0 not compatible
  >    * - 1 compatible
  >    */
  >   #define PERF_GNU 0/1
  >
  >   /**
  >    * Linux # default: deduced based on `__linux__`
  >    * - 0 not supported
  >    * - 1 supported
  >    */
  >   #define PERF_LINUX 0/1
  >
  >   /**
  >    * UEFI # default: 0
  >    * - 0 not supported
  >    * - 1 supported
  >    */
  >   #define PERF_UEFI 0/1
  >
  >   /**
  >    * LLVM # default: deduced based on `llvm-dev` headers
  >    * - 0 not supported
  >    * - 1 supported
  >    */
  >   #define PERF_LLVM 0/1
  >
  >   /**
  >    * Intel Processor Trace # default: deduced based on `intel_pt` headers
  >    * - 0 not supported
  >    * - 1 supported
  >    */
  >   #define PERF_INTEL 0/1
  >
  >   /**
  >    * I/O support # default: 1
  >    * - 0 not compiled
  >    * - 1 supported (`log, json, report, annotate, plot`)
  >    */
  >   #define PERF_IO 0/1
  >
  >   /**
  >    * tests # default: not-defined
  >    * - defined:     disables all compile-time, run-time tests
  >    * - not-defined: compile-time tests executed,
  >    *                run-time tests available by `perf::self::test()` API
  >    */
  >   #define NTEST
  >   ```
  >
  >   ```sh
  >   /**
  >    * gnuplot terminal # see `gnuplot -> set terminal` # default: 'sixel'
  >    * - 'sixel'                  # console image # https://www.arewesixelyet.com
  >    * - 'wxt'                    # popup window
  >    * - 'canvas'                 # html
  >    * - 'dumb size 150,25 ansi'  # console with colors
  >    * - 'dumb size 80,25'        # console
  >    */
  >   ENV:PERF_IO_PLOT_TERM
  >
  >   /**
  >    * style # default: dark
  >    * - light
  >    * - dark
  >    */
  >   ENV:PERF_IO_PLOT_STYLE
  >   ```
  >
  > </details>
  >
  > <details>
  >   <summary>Info/Core</summary>
  >
  >   ```cpp
  >   namespace perf::info {
  >     /**
  >      * static_assert(version().major == 1);
  >      * static_assert(version().minor == 0);
  >      * static_assert(version().patch == 0);
  >      */
  >     inline constexpr auto version = [] -> sem_ver;
  >   } // namespace perf::info
  >   ```
  >
  >   ```cpp
  >   namespace perf::info::compiler {
  >     /**
  >      * verify(name() == "clang"s);
  >      */
  >     inline constexpr auto name = [] -> std::string_view;
  >
  >     /**
  >      * static_assert(version().major == 20);
  >      * static_assert(version().minor == 0);
  >      * static_assert(version().patch == 0);
  >      */
  >     inline constexpr auto version = [] -> sem_ver;
  >   } // namespace perf::info::compiler
  >   ```
  >
  >   ```cpp
  >   // perf::info::cpu::name
  >   assert(perf::info::cpu::name() == "12th Gen Intel(R) Core(TM) i7-12650"s);
  >
  >   // perf::info::cpu::code_name
  >   assert(perf::info::cpu::code_name() == "alderlake"s);
  >
  >   // perf::info::cpu::version
  >   assert(perf::info::cpu::version().family == 6);
  >   assert(perf::info::cpu::version().model == 154);
  >   assert(perf::info::cpu::version().stepping == 3);
  >
  >   // perf::info::cpu::dispatch_width
  >   assert(perf::info::cpu::dispatch_width() == 6);
  >
  >   // perf::info::cpu::features
  >   assert(perf::info::cpu::features() == std::vector{"avx", "avx2", "bmi", ...});
  >   ```
  >
  >   ```cpp
  >   // info::memory::icache
  >   assert(perf::info::memory::icache() ==
  >     std::map{{level::L1, {.size = 448KiB, .line_size = 64, .assoc = 8}}}
  >   );
  >
  >   // info::memory::dcache
  >   assert(perf::info::memory::dcache() == std::map{
  >     {level::L1, {.size = 416KiB, .line_size = 64, .assoc = 8}}, ...},
  >     {level::L2, {.size = 9.5MiB, .line_size = 64, .assoc = 12}}, ...},
  >     {level::L3, {.size = 24Mib,  .line_size = 64, .assoc = 12}}, ...},
  >   });
  >   ```
  >
  >   ```cpp
  >   // info::sys::name
  >   assert(perf::info::sys::name() == "linux"s);
  >
  >   // info::sys::triple
  >   assert(perf::info::sys::triple() == "x86_64-pc-linux-gnu"s);
  >
  >   // info::sys::page_size
  >   assert(perf::info::sys::page_size() == 4096b);
  >   ```
  >
  >   ```cpp
  >   // info::proc::name
  >   assert(perf::info::proc::self::name() == "/full/path/example.out"s);
  >
  >   // info::proc::base_address
  >   assert(perf::info::proc::self::base_address());
  >   ```
  >
  >   ```cpp
  >   // info::bin::addr_to_fn_name
  >   static auto fn = [] {};
  >   auto&& fn_name = perf::info::bin::addr_to_fn_name(
  >     perf::info::proc::self::name(),
  >     std::uint64_t(&fn) - perf::info::proc::self::base_address()
  >   );
  >   assert(fn_name.has_value() and *fn_name == "fn"s);
  >
  >   // info::bin::addr_to_name
  >   static auto var = 0;
  >   auto&& var_name = perf::info::bin::addr_to_name(
  >     perf::info::proc::self::name(),
  >     std::uint64_t(&var) - perf::info::proc::self::base_address()
  >   );
  >   assert(var_name.has_value() and *var_name == "var"s);
  >
  >   // info::bin::addr_to_line # requires debug symbols (-g)
  >   label:; auto&& source = perf::info::bin::addr_to_line(
  >     perf::info::proc::self::name(),
  >     std::uint64_t(&&label) - perf::info::proc::self::base_address()
  >   );
  >   assert(source.has_value() and source->contains("label:;"));
  >   ```
  >
  >   ```cpp
  >   // code::align
  >   perf::code::align<std::align_val_t(64u)>(); for (...) { }
  >
  >   // code::label
  >   perf::code::label<"begin">(); // begin:
  >   perf::code::label<"end">();   // end:
  >   assert(perf::code::labels["begin"] != perf::code::labels["end"]);
  >   ```
  >
  >   ```cpp
  >   // compiler::prevent_reorder # std::atomic_signal_fence
  >   perf::compiler::prevent_reorder();
  >
  >   // copmiler::prevent_elision
  >   int i{};
  >   assert(perf::compiler::prevent_elision(i++));
  >
  >   // copiler::is_elided
  >   assert(perf::compiler::is_elided([] { }));
  >   assert(perf::compiler::is_elided([] { int i{}; i++; }));
  >   assert(not perf::compiler::is_elided([] {
  >     int i{};
  >     perf::compiler::prevent_elision(i++);
  >   }));
  >   ```
  >
  >   ```cpp
  >   // cpu::pipeline::flush
  >   perf::cpu::pipeline::flush();
  >   ```
  >
  >   ```cpp
  >   // memory::align/memory::is_aligned
  >   auto aligned_addr = perf::memory::align<perf::memory::direction::up>(
  >     addr, std::align_val_t(64u)
  >   );
  >   assert(perf::memory::is_aligned(aligned_addr, std::align_val_t(64u)));
  >
  >   // memory::synchronize # std::atomic_thread_fence
  >   perf::memory::synchronize();
  >
  >   // memory::prefetch
  >   perf::memory::prefetch<perf::memory::operation::write,
  >                          perf::memory::locality::high>(addr);
  >
  >   // memory::lock # scoped{mlockall, munlockall}
  >   {
  >     perf::memory::lock _;
  >   }
  >
  >   // memory::protect
  >   const std::array add{ // x86-64
  >     0x89, 0xf8,         // mov eax, edi
  >     0x01, 0xf0,         // add eax, esi
  >     0xc3                // ret
  >   };
  >   perf::memory::protect(
  >     std::span(add),
  >     perf::memory::protection::read |
  >     perf::memory::protection::write |
  >     perf::memory::protection::exec)
  >   );
  >   assert(invoke(add, 1, 2) == 3);
  >   assert(invoke(add, 2, 3) == 5);
  >
  >   // memory::pollute # pollutes memory by making allocations
  >   perf::memory::pollute(1024u);
  >
  >   // memory::pre_fault # touches all pages used by data
  >   perf::memory::pre_fault(std::span(...));
  >
  >   // memory::flush # clears all cache lines used by data
  >   perf::memory::cache::flush(std::span(...));
  >   ```
  >
  >   ```cpp
  >   // sys::affinity
  >   perf::thread::affinity::set(perf::thread::self, 2u);
  >   assert(2u == perf::thread::affinity::get(perf;:thread::self));
  >   ```
  >
  > </details>
  >
  > <details>
  >   <summary>Profiling/Analyzing</summary>
  >
  > </details>
  >
  > <details>
  >   <summary>Benchmarking/Testing</summary>
  >
  > </details>
  >
  >  [`Synopsis`](https://github.com/qlibs/perf/blob/main/perf.cppm#L39)

### User Guide

  > <details>
  >   <summary>Setup Guide</summary>
  >
  >   - How to setup `perf` [docker](https://www.docker.com)?
  >     > [`Dockerfile`](https://github.com/qlibs/pref/blob/main/.github/workflows/Dockerfile)
  >
  >     ```sh
  >     docker build -t perf .
  >     ```
  >     ```sh
  >     docker run \
  >       -it \
  >       --privileged \
  >       --network=host \
  >       -e DISPLAY=${DISPLAY} \
  >       -v ${PWD}:${PWD} \
  >       -w ${PWD} \
  >       perf
  >     ```
  >
  >   - How to install `perf` depenencies?
  >     ```sh
  >     apt-get install linux-tools-common # linux-perf (perf::stat/perf::record)
  >     apt-get install llvm-dev           # llvm (perf::mc/perf::mca)
  >     apt-get install libipt-dev         # libipt (perf::trace)
  >     apt-get install gnuplot            # (perf::plot)
  >     ```
  >
  >   - How to setup `linux performance counters`?
  >     > [`scripts/setup.sh`](https://github.com/qlibs/perf/blob/main/.github/scripts/setup.sh)
  >     ```sh
  >     .github/scripts/setup.sh --perf # --rdpmc --max-sample-rate 10000
  >     ```
  >
  >     > [`linux`](https://www.kernel.org)
  >     ```sh
  >     sudo mount -o remount,mode=755 /sys/kernel/debug
  >     sudo mount -o remount,mode=755 /sys/kernel/debug/tracing
  >     sudo chown `whoami` /sys/kernel/debug/tracing/uprobe_events
  >     sudo chmod a+rw /sys/kernel/debug/tracing/uprobe_events
  >     echo 0 | sudo tee /proc/sys/kernel/kptr_restrict
  >     echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
  >     echo 1000 | sudo tee /proc/sys/kernel/perf_event_max_sample_rate
  >     ```
  >     ```sh
  >     echo 2 | sudo tee /sys/devices/cpu_core/rdpmc
  >     ```
  >
  >   - How to reduce `execution variability`?
  >     > [`scripts/tune.sh`](https://github.com/qlibs/pref/blob/main/.github/scripts/tune.sh)
  >     ```sh
  >     .github/scripts/tune.sh
  >     ```
  >
  >     > [`pyperf`](https://github.com/psf/pyperf) - `pip3 install pyperf`
  >     ```sh
  >     sudo pyperf system tune
  >     sudo pyperf system show
  >     sudo pyperf system reset
  >     ```
  >
  >     > [`linux`](https://www.kernel.org)
  >     ```sh
  >     # Set Process CPU Affinity (apt install util-linux)
  >     taskset -c 0 ./a.out
  >
  >     # Set Process Scheduling Priority (apt install coreutils)
  >     nice -n -20 taskset -c 0 ./a.out # -20..19 (most..less favorable to the process)
  >
  >     # Disable CPU Frequency Scaling (apt install cpufrequtils)
  >     sudo cpupower frequency-set --governor performance
  >     # cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  >
  >     # Disable Address Space Randomization
  >     echo 0 > /proc/sys/kernel/randomize_va_space
  >
  >     # Disable Processor Boosting
  >     echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
  >
  >     # Disable Turbo Mode
  >     echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
  >
  >     # Disable Hyperthreading/SMT
  >     echo off | sudo tee /sys/devices/system/cpu/smt/control
  >
  >     # Restrict memory to a single socket
  >     numactl -m 0 -N 0 ./a.out
  >
  >     # Enable Huge Pages
  >     sudo numactl --cpunodebind=1 --membind=1 hugeadm \
  >       --obey-mempolicy --pool-pages-min=1G:64
  >     sudo hugeadm --create-mounts
  >     ```
  >
  >     > [`bootloader`](https://en.wikipedia.org/wiki/GNU_GRUB)
  >     ```sh
  >     # Enable Kernel Mode Task-Isolation (https://lwn.net/Articles/816298)
  >     # cat /sys/devices/system/cpu/isolated
  >     isolcpus=<cpu number>,...,<cpu number>
  >
  >     # Disable P-states and C-states
  >     # cat /sys/devices/system/cpu/intel_pstate/status
  >     idle=pool intel_pstate=disable intel_idle.max_cstate=0 processor.max_cstate=1
  >
  >     # Disable NMI watchdog
  >     # cat /proc/sys/kernel/nmi_watchdog
  >     nmi_watchdog=0
  >     ```
  >
  >     > [`uefi`](https://simple.wikipedia.org/wiki/Unified_Extensible_Firmware_Interface)
  >     ```make
  >     clang++-std=c++20 -I. \
  >       -target x86_64-pc-win32-coff \
  >       -fno-stack-protector \
  >       -fshort-wchar \
  >       -mno-red-zone \
  >       -c uefi.cpp -o uefi.o
  >
  >     lld-link \
  >       -filealign:16 \
  >       -subsystem:efi_application \
  >       -nodefaultlib
  >       -dll \
  >       -entry:efi_main \
  >       -out:BOOTX64.EFI \
  >       uefi.o
  >
  >     mkdir -p efi/boot && cp BOOTX64.EFI /usr/share/ovmf/OVMF.fd efi/boot
  >
  >     qemu-system-x86_64 \
  >       -drive if=pflash,format=raw,file=efi/boot/OVMF.fd \
  >       -drive format=raw,file=fat:rw:. \
  >       -net none
  >     ```
  >
  > </details>
  >
  > <details>
  >   <summary>Usage Guide</summary>
  >
  >   - How to compile `perf` with [`modules`](https://en.cppreference.com/w/cpp/language/modules)?
  >     > [clang](https://clang.llvm.org)
  >     ```sh
  >     clang++ -std=c++23 -O3 -I. --precompile perf.cppm
  >     clang++ -std=c++23 -O3 -fprebuilt-module-path=. perf.pcm <source_file> -lLLVM -lipt
  >     ```
  >
  >     ```cpp
  >     import perf;
  >     ```
  >
  >   - How to change `assembly` syntax?
  >     ```cpp
  >     perf::llvm llvm{
  >       {.syntax = perf::arch::syntax::att} // default: intel
  >     };
  >     ```
  >
  >   - How to `analyze` for a different architecture?
  >     ```cpp
  >     perf::llvm llvm{
  >       .triple = "x86_64-pc-linux-gnu" // see `llvm-llc` for details
  >     };
  >     ```
  >
  >   - Which `terminal` can display images?
  >     > Any terminal with sixel support - https://www.arewesixelyet.com <br/> ([`Visual Studio Code`](https://code.visualstudio.com) images support in terminal - `Terminal -> Enable images option`)
  >
  >   - How to change plotting `terminal`?
  >     ```sh
  >     PERF_IO_PLOT_TERM='sixel'                 # terminal - sixel
  >     PERF_IO_PLOT_TERM='dumb size 80,25'       # terminal asci
  >     PERF_IO_PLOT_TERM='dumb size 150,25 ansi' # terminal ansi
  >     PERF_IO_PLOT_TERM='wxt'                   # popup windows
  >     PERF_IO_PLOT_TERM='canvas'                # html
  >     PERF_IO_PLOT_TERM='png'                   # png
  >     ```
  >
  >     ```sh
  >     gnuplot: set terminal # available options
  >     ```
  >     - http://www.bersch.net/gnuplot-doc/complete-list-of-terminals.html
  >
  >   - How to change plot style?
  >     ```sh
  >     PERF_IO_PLOT_STYLE='dark'  # dark - default
  >     PERF_IO_PLOT_STYLE='light' # light
  >     ```
  >
  >   - How to save plot?
  >     ```cpp
  >     perf::plot::gnuplot plt{{.term = "png"}};
  >     plt.send("set output 'output.png'");
  >     perf::plot::bar(plt, ...);
  >     ```
  >
  >   - How to `export` results?
  >     > [`scripts/export.sh`](https://github.com/qlibs/perf/blob/main/.github/scripts/export.sh)
  >     ```sh
  >     ./a.out 2>&1 | .github/scripts/export.sh markdown > results.md
  >     ./a.out 2>&1 | .github/scripts/export.sh notebook > results.ipynb
  >     ./a.out 2>&1 | .github/scripts/export.sh html > results.html
  >     ```
  >
  >   - How to `share` results?
  >     > [`gh`](https://gist.github.com) - `apt-get install gh`
  >
  >     ```sh
  >     # https://jbt.github.io/markdown-editor
  >     gh gist create --public --web results.md
  >     ```
  >     ```sh
  >     # https://jupyter.org
  >     gh gist create --public --web results.ipynb
  >     ```
  >     ```sh
  >     # https://htmlpreview.github.io
  >     gh gist create --public --web results.html
  >     ```
  >
  >   - How to write custom `profiler`?
  >     ```cpp
  >     struct my_profiler {
  >       constexpr auto start();
  >       constexpr auto stop();
  >       [[nodiscard]] constexpr auto operator[](Ts...) const;
  >     };
  >     ```
  >
  >     ```cpp
  >     static_assert(perf::profiler_like<my_profiler>);
  >     ```
  >
  >     ```cpp
  >     perf::runner bench{
  >       [](auto&& fn, auto&&... ts) {
  >         my_profiler profiler{};
  >         profiler.start();
  >         perf::compiler::prevent_elision(fn(ts...));
  >         profiler.stop();
  >       }
  >     };
  >     ```
  >
  >   - How to integrate with `profiling` tools?
  >     - [`linux-perf`](https://perf.wiki.kernel.org) - `apt get install linux-tools-common`
  >     - [`intel-vtune`](https://www.intel.com/content/www/us/en/docs/vtune-profiler) - `apt get install intel-oneapi-vtune`
  >     - [`amd-uprof`](https://www.amd.com/en/developer/uprof.html) - `https://www.amd.com/en/developer/uprof.html#downloads`
  >     - [`gperftools`](https://github.com/gperftools/gperftools) - `apt get install google-perftools`
  >     - [`llvm-xray`](https://llvm.org/docs/XRay.html) - `apt-get install llvm`
  >     - [`callgrind`](https://valgrind.org/docs/manual/cl-manual.html) - `apt-get install valgrind`
  >     <br />
  >
  >     ```cpp
  >     #include <fcntl.h>
  >     #include <unistd.h>
  >     /*
  >      * https://perf.wiki.kernel.org
  >      * https://perfwiki.github.io
  >      *
  >      * int main() {
  >      *   linux_perf profiler{"/dev/shm/perf"};
  >      *   profiler.start();
  >      *   // ...
  >      *   profiler.stop();
  >      * }
  >      *
  >      * $CXX -std=c++20 -O3 -g perf.cpp
  >      * perf record --control=fifo:/dev/shm/perf --delay=-1 ./a.out
  >      */
  >     class linux_perf {
  >       static constexpr auto enable = "enable\n";
  >       static constexpr auto disable = "disable\n";
  >
  >      public:
  >       constexpr explicit linux_perf(std::string&& control)
  >         : fd_{open(control.c_str(), O_WRONLY)}
  >       { }
  >       constexpr linux_perf(const linux_perf&) = delete;
  >       constexpr linux_perf(linux_perf&& other)
  >         : fd_{std::move(other.fd_)} {
  >         other.fd_ = -1;
  >       }
  >       constexpr ~linux_perf() noexcept {
  >         if (fd_ == -1) return;
  >         close(fd_);
  >       }
  >
  >       constexpr auto start() {
  >         return write(fd_, enable, sizeof(enable));
  >       }
  >
  >       constexpr auto stop() {
  >         return write(fd_, disable, sizeof(disable));
  >       }
  >
  >      private:
  >       int fd_{};
  >     };
  >     ```
  >
  >     ```cpp
  >     #include <ittnotify.h>
  >     /**
  >      * https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html
  >      * https://github.com/intel/ittapi
  >      *
  >      * int main() {
  >      *   intel_vtune profiler{"domain", "task"};
  >      *   profiler.start();
  >      *   // ...
  >      *   profiler.stop();
  >      * }
  >      *
  >      * $CXX -std=c++20 -O3 -g vtune.cpp -littnotify
  >      * vtune -collect performance-snapshot -start-paused -finalization-mode=full -r result -- ./a.out
  >      */
  >     class intel_vtune {
  >      public:
  >       constexpr explicit intel_vtune(std::string&& domain, std::string&& task)
  >         : domain_{__itt_domain_create(domain.c_str())}, task_name_{__itt_string_handle_create(task.c_str())} {
  >         __itt_task_begin(domain_, __itt_null, __itt_null, task_name_);
  >         assert(__builtin_cpu_is("intel"));
  >       }
  >       constexpr intel_vtune(intel_vtune&&) = default;
  >       constexpr intel_vtune(const intel_vtune&) = delete;
  >       constexpr ~intel_vtune() noexcept {
  >         __itt_task_end(domain_);
  >       }
  >
  >       constexpr auto start() {
  >         return __itt_resume();
  >       }
  >
  >       constexpr auto stop() {
  >         return __itt_pause();
  >       }
  >
  >      private:
  >       __itt_domain* domain_{};
  >       __itt_string_handle* task_name_{};
  >     };
  >     ```
  >
  >     ```cpp
  >     #include <AMDProfileController.h>
  >     /**
  >      * https://www.amd.com/en/developer/uprof.html
  >      *
  >      * int main() {
  >      *   amd_uprof profiler{};
  >      *   profiler.start();
  >      *   // ...
  >      *   profiler.stop();
  >      * }
  >      * $CXX -std=c++20 -O3 -g uprof.cpp -lAMDProfileController
  >      * AMDuProfCLI collect --config tbp --start-paused ./a.out
  >      */
  >     class amd_uprof {
  >      public:
  >       constexpr amd_uprof() {
  >         assert(__builtin_cpu_is("amd"));
  >       }
  >       constexpr amd_uprof(amd_uprof&&) = default;
  >       constexpr amd_uprof(const amd_uprof&) = delete;
  >
  >       constexpr auto start() {
  >         return amdProfileResume();
  >       }
  >
  >       constexpr auto stop() {
  >         return amdProfilePause();
  >       }
  >     };
  >     ```
  >
  >     ```cpp
  >     #include <gperftools/profiler.h>
  >     /**
  >      * int main() {
  >      *   gperf profiler{"gperf"};
  >      *   profiler.start();
  >      *   // ...
  >      *   profiler.stop();
  >      *   profiler.flush(); // optional
  >      * }
  >      *
  >      * $CXX -g -O3 perf.cpp -lprofiler
  >      * CPUPROFILE_FREQUENCY=1000 ./a.out
  >      * google-pprof a.out profile.prof
  >      */
  >     class gperf {
  >      public:
  >       constexpr explicit gperf(std::string&& fname)
  >         : fname_{std::move(fname)}
  >       { }
  >       constexpr gperf(gperf&&) = default;
  >       constexpr gperf(const gperf&) = delete;
  >       constexpr ~gperf() noexcept {
  >         flush();
  >       }
  >
  >       constexpr auto start() {
  >         return ProfilerStart(fname_.c_str());
  >       }
  >
  >       constexpr auto stop() {
  >         return ProfilerStop();
  >       }
  >
  >       constexpr void flush() {
  >         return ProfilerFlush();
  >       }
  >
  >      private:
  >       std::string fname_{};
  >     };
  >     ```
  >
  >     ```cpp
  >     #include <xray/xray_interface.h>
  >     #include <xray/xray_log_interface.h>
  >     /**
  >      * https://llvm.org/docs/XRay.html
  >      * https://godbolt.org/z/WhsEYf9cc
  >      *
  >      * int main() {
  >      *   llvm_xray profiler{"xray-fdr"};
  >      *   profiler.start();
  >      *   // ...
  >      *   profiler.stop();
  >      *   profiler.flush(); // optional
  >      * }
  >      *
  >      * [[clang::xray_always_instrument]] void always_profile();
  >      * [[clang::xray_always_instrument, clang::xray_log_args(1)]] void always_profile_and_log_i(int i);
  >      * [[clang::xray_never_instrument]] void never_profile();
  >      *
  >      * void handler([[maybe_unused]] int32_t func_id, XRayEntryType entry) {
  >      *   if (entry == XRayEntryType::ENTRY) {
  >      *     // profiler.start();
  >      *   } else {
  >      *     // profiler.stop();
  >      *   }
  >      * }
  >      *
  >      * int main() {
  >      *   __xray_set_handler(handler);
  >      *   __xray_patch();
  >      * }
  >      *
  >      * clang++ std=c++20 -O3 -g xray.cpp -fxray-instrument -fxray-instruction-threshold=1
  >      * ./a.out
  >      * llvm-xray account xray-log.* --top=10 --sort=sum --sortorder=dsc --instr_map=./a.out
  >      * llvm-xray extract ./a.out --symbolize
  >      */
  >     class llvm_xray {
  >      public:
  >       constexpr explicit llvm_xray(
  >         std::string&& mode = "xray-fdr",
  >         std::string&& cfg = "xray_logfile_base=xray-log.%"
  >       ) {
  >         __xray_log_select_mode(mode.c_str());
  >         __xray_log_init_mode(mode.c_str(), cfg.c_str());
  >       }
  >       constexpr llvm_xray(llvm_xray&&) = default;
  >       constexpr llvm_xray(const llvm_xray&) = delete;
  >       constexpr ~llvm_xray() noexcept {
  >         flush();
  >       }
  >
  >       [[clang::xray_never_instrument]] constexpr auto start() {
  >         return __xray_patch();
  >       }
  >
  >       [[clang::xray_never_instrument]] constexpr auto stop() {
  >         return __xray_unpatch();
  >       }
  >
  >       [[clang::xray_never_instrument]] constexpr void flush() {
  >         __xray_log_finalize();
  >         __xray_log_flushLog();
  >       }
  >     };
  >     ```
  >
  >     ```cpp
  >     #include <valgrind/callgrind.h>
  >     /**
  >      * int main() {
  >      *   callgrind callgrind{"profile"};
  >      *
  >      *   while (true) {
  >      *     profiler.start(); // resets profile
  >      *
  >      *     if (should_trigger()) {
  >      *       trigger();
  >      *       profiler.stop();
  >      *       proflier.flush(); // dumps `example` profile
  >      *     }
  >      *   }
  >      * }
  >      *
  >      * $CXX std=c++20 -O3 -g callgrind.cpp
  >      * valgrind --tool=callgrind --instr-atstart=no \ # 5x-100x overhead
  >      *  --cache-sim=yes --branch-sim=yes --collect-jumps=yes --dump-instr=yes a.out
  >      * kcachegrind callgrind.*
  >      */
  >     class callgrind {
  >      public:
  >       constexpr explicit callgrind(std::string&& profile)
  >         : profile_{std::move(profile)}
  >       { }
  >       constexpr callgrind(callgrind&&) = default;
  >       constexpr callgrind(const callgrind&) = delete;
  >       constexpr ~callgrind() noexcept {
  >         flush();
  >       }
  >
  >       constexpr auto start() {
  >         CALLGRIND_START_INSTRUMENTATION;
  >       }
  >
  >       constexpr auto stop() {
  >         CALLGRIND_STOP_INSTRUMENTATION;
  >       }
  >
  >       constexpr void flush() {
  >         CALLGRIND_DUMP_STATS_AT(profile_.c_str());
  >       }
  >
  >      private:
  >       std::string profile_{};
  >     };
  >     ```
  >
  >   - How to integrate with `unit-testing` frameworks?
  >     ```cpp
  >     import perf;
  >     import ut; // https://github.com/qlibs/ut
  >
  >     int main() {
  >       perf::runner bench{perf::bench::latency{}};
  >       perf::scoped _ {
  >         .on_exit = [&] {
  >           perf::report(bench[perf::stat::cpu_time]);
  >         }
  >       };
  >
  >       "benchmark1"_test = [] {
  >         bench(fn1, ts1...);
  >       };
  >
  >       "benchmark1"_test = [] {
  >         bench(fn2, ts2...);
  >       };
  >     }
  >     ```
  >
  >   - How `perf` tests are working?
  >     > `compile-time` tests are executed upon `include/import` (<i>enabled by default</i>) <br/>
  >     > `run-time/sanity check` tests can be executed at run-time
  >
  >     ```cpp
  >     int main() {
  >       perf::self::test({.verbose = true}); // run-time/sanity check tests
  >     }
  >     ```
  >
  >     > `-DNTEST` can be used to disable tests (<i>not recommended</i>)
  >     ```cpp
  >     $CXX -DNTEST ... # tests will NOT be compiled
  >     ```
  >
  >     > `perf` tests execution model
  >     ```cpp
  >     #ifndef NTEST
  >     "perf"_suite = [] {
  >       "run-time and compile-time"_test = [] constexpr {
  >         expect(3 == accumulate({1, 2, 3}, 0));
  >       };
  >
  >       "run-time"_test = [] mutable {
  >         expect(std::rand() >= 0);
  >       };
  >
  >       "compile-time"_test = [] consteval {
  >         expect(sizeof(int) == sizeof(0));
  >       };
  >     };
  >     #endif
  >     ```
  >
  >   - How to integrate with `jupyter`?
  >     > [`jupyter`](https://jupyter.org) (`apt install jupyter`) can be used for data analysis (`python`)
  >
  >     ```cpp
  >     int main() {
  >       // ... perf
  >       perf::json("perf.json");
  >     }
  >     ```
  >
  >     ```python
  >     # notebook.ipynb
  >     import pandas as pd
  >     df = pd.read_json("perf.json")
  >     print(df.head())
  >     ```
  >
  >     ```sh
  >     jupyter notebook notebook.ipynb
  >     ```
  >
  >   - What is the difference between `latency` and `throughput`?
  >     > `latency` is the time it takes for a single operation to complete (ns) <br/>
  >     > `throughput` is the total number of operations or tasks completed in a given amount of time (op/s) <br/>
  >
  >   - What is `top-down microarchitecture analysis` method?
  >     > https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-0/top-down-microarchitecture-analysis-method.html <br/>
  >     > https://github.com/andikleen/pmu-tools/wiki/toplev-manual <br/>
  >
  >   - What are `performance` compilation flags?
  >     ```sh
  >     -O1                     # optimizations (O1) [0]
  >     -O2                     # optimizations (O1 + O2) [0]
  >     -O3                     # optimizations (O1 + O2 + O3) [0]
  >     -march=native           # architecture specific [1]
  >     -DNDEBUG                # disables asserts, etc.
  >     ```
  >
  >     ```sh
  >     -fno-omit-frame-pointer # keeps the frame pointer in a register
  >     ```
  >
  >     ```sh
  >     -ffast-math             # [unsafe] faster but non-conforming math [2]
  >     -fcf-protection=none    # [unsafe] stops emmitting `endbr64`
  >     ```
  >
  >     > [0] https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html <br/>
  >     > [1] https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html <br/>
  >     > [2] https://gcc.gnu.org/wiki/FloatingPointMath <br/>
  >
  >   - What are `performance` compiler attributes?
  >     > `gnu::target`
  >     ```cpp
  >     [[gnu::target("avx2")]]
  >     [[gnu::target("bmi")]]
  >     ```
  >
  >     > `gnu::optimize`
  >     ```cpp
  >     [[gnu::optimize("O3")]
  >     [[gnu::optimize("ffast-math")]
  >     ```
  >
  >     > https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html
  >
  > </details>
  >
  >  [`Resources`](https://github.com/qlibs/perf/wiki/resources)

### License

  > <details>
  >   <summary>MIT/Apache2+LLVM</summary>
  >
  >   | license      | namespace      | guard            | description                        |
  >   | ------------ | -------------- | ---------------- | ---------------------------------- |
  >   | MIT          | `perf::*`      | -                | https://opensource.org/license/mit |
  >   | Apache2+LLVM | `perf::mca::*` | `PERF_LLVM == 1` | https://llvm.org/LICENSE.txt       |
  >
  > [`LICENSE`](LICENSE)
  >
  > </details>

<!--
#endif

module;
#include <perf>
export module perf;

export namespace perf {
  inline namespace utility {
    using perf::verify;                                 // verifies whether condition is satisfied - aborts if not
    using perf::named;                                  // named type with run-time name
    using perf::scoped;                                 // scoped{.on_entry = ctor, .on_exit = dtor}
  } // namespace utility

  namespace mp {
    using perf::mp::unroll;                             // repeats fn N times
    using perf::mp::for_each;                           // for each element
  } // namespace mp

  namespace data {
    inline namespace random {
      using perf::data::choice;                         // probabilitity choice
      using perf::data::uniform;                        // uniform distrubution
      using perf::data::normal;                         // normal distrubution
      using perf::data::unpredictable;                  // not elided and not predicted
    } // namespace random

    inline namespace fixed {
      using perf::data::sequence;                       // sequence
      using perf::data::range;                          // range
    } // namespace fixed

    using perf::data::sample;                           // data sampler
  } // namespace data

  namespace metric {
    inline namespace dsl {
      using perf::metric::operator+;                    // ex. `instructions + cycles`
      using perf::metric::operator-;                    // ex. `instructions - cycles`
      using perf::metric::operator*;                    // ex. `instructions * cycles`
      using perf::metric::operator/;                    // ex. `instructions / cycles`
    } // namespace dsl

    inline namespace stat {
      using perf::metric::stat::min;                    // min element
      using perf::metric::stat::max;                    // max element
      using perf::metric::stat::mean;                   // mean
      using perf::metric::stat::geomean;                // geometric mean
      using perf::metric::stat::median;                 // median
      using perf::metric::stat::percentile;             // percentile
      using perf::metric::stat::p10;
      using perf::metric::stat::p25;
      using perf::metric::stat::p50;
      using perf::metric::stat::p75;
      using perf::metric::stat::p90;
      using perf::metric::stat::p95;
      using perf::metric::stat::p99;
      using perf::metric::stat::variance;               // variance
      using perf::metric::stat::stddev;                 // standard deviation (variability), degrees_of_freedom = 1
      using perf::metric::stat::sem;                    // standard error (precision)
      using perf::metric::stat::mae;                    // median absolute error
      using perf::metric::stat::mad;                    // median absolute deviation
      using perf::metric::stat::cv;                     // coefficient of variation
      using perf::metric::stat::z_score;                // z-score
      using perf::metric::stat::t_score;                // t-score
    } // namespace stat

    using perf::metric::value;
  } // namespace metric::dsl

  namespace info {
    using perf::info::spec;                             // performance specification

    namespace compiler {
      using perf::info::compiler::name;                 // ex. 'clang'
      using perf::info::compiler::version;              // ex. 20.0.0 { major, minor, patch }
    } // namespace compiler

    namespace cpu {
      #if PERF_GNU == 1
      using perf::info::cpu::name;                      // ex. '12th Gen Intel(R) Core(TM) i7-12650H'
      using perf::info::cpu::version;                   // cpu version {family, model, stepping}, ex. '15.0.3'
      #endif
      #if PERF_LLVM == 1
      using perf::info::cpu::code_name;                 // ex. 'alderlake', 'skylake', ...
      using perf::info::cpu::dispatch_width;            // the number of micro-operations a core can send to its execution units per clock cycle determines the degree of parallel instruction execution
      using perf::info::cpu::features;                  // features supported by cpu ex. '{avx, avx2, bmi2, ...}'
      #endif
    } // namespace cpu

    namespace memory {
      #if PERF_LINUX == 1
      using perf::info::memory::level;                  // enum level{L0, L1, L2, L3, L4}
      using perf::info::memory::icache;                 // instruction cache info
      using perf::info::memory::dcache;                 // data cache info
      using perf::info::memory::peak_rss;               // maximum resident set size
      #endif
    } // namespace memory

    namespace sys {
      using perf::info::sys::name;                      // ex. 'linux'
      using perf::info::sys::page_size;                 // ex. 4096 [kb]
      #if PERF_LLVM == 1
      using perf::info::sys::triple;                    // ex. 'x86_64-pc-linux-gnu'
      #endif
    } // namespace sys

    namespace proc::inline self {
      #if PERF_LINUX == 1
      using perf::info::proc::self::name;               // ex. '/tmp/a.out' # full path
      using perf::info::proc::self::base_address;       // proc starting memory address, ex. `instruction_pointer - base_address()`
      #endif
    } // namespace proc::self

    namespace bin {
      #if PERF_LLVM == 1
      using perf::info::bin::addr_to_line;              // returns `filename:line` from executable/address
      using perf::info::bin::addr_to_fn_name;           // returns `variable name` from excutable/address
      using perf::info::bin::addr_to_name;              // returns `function name` from executable/address
      using perf::info::bin::fn_addrs;
      #endif
    } // namespace bin
  } // namespace info

  inline namespace core {
    namespace code {
      #if PERF_GNU == 1
      using perf::code::align;                          // gnu version of `[[clang::code_align(alignment)]]`
      using perf::code::label;                          // global compile-time code label, ex. `label<"loop.begin">()`
      using perf::code::labels;                         // global compile-time code labels, ex. `auto&& label = labels["loop.bagin"]`
      #endif
    } // namespace code

    namespace compiler {
      #if PERF_GNU == 1
      using perf::compiler::prevent_reorder;            // prevents memory instructions reorder
      using perf::compiler::prevent_elision;            // prevents elision of given value
      using perf::compiler::is_elided;                  // returns true if function is elided by the compiler, false otherwise
      #endif
    } // namespace compiler

    namespace cpu::pipeline {
      #if PERF_GNU == 1 and defined(__x86_64__)
      using perf::cpu::pipeline::flush;                 // flushes cpu pipeline (x86: cpuid)
      #endif
    } // namespace cpu::pipeline

    namespace memory {
      using perf::memory::direction;                    // enum class direction{up, down}
      using perf::memory::align;                        // aligns given pointer to specific alignment and offset
      using perf::memory::is_aligned;                   // returns true if pointer is aligned to specified alignment
      #if PERF_GNU == 1 and defined(__x86_64__)
      using perf::memory::synchronize;                  // hardware memory barrier (x86: fence or lock)
      #endif
      #if __has_builtin(__builtin_prefetch)
      using perf::memory::operation;                    // enum class operation{read, write}
      using perf::memory::locality;                     // enum class locality{none, low, moderate, high}
      using perf::memory::prefetch;                     // prefetches memory from address
      #endif
      #if PERF_LINUX == 1
      using perf::memory::protection;                   // enum class protection{read, write, exec}
      using perf::memory::protect;                      // sets protection on a region of memory
      using perf::memory::pre_fault;                    // touches all pages used by data
      using perf::memory::lock;                         // scoped lock for all currently mapped pages
      #endif
      using perf::memory::pollute;                      // prevents continuous allocation

      namespace cache {
        #if PERF_GNU == 1
        using perf::memory::cache::flush;               // clears all cache lines used by data
        #endif
        #if PERF_UEFI == 1 and PERF_GNU == 1
        using perf::memory::cache::enable;              // enables cache
        using perf::memory::cache::disable;             // disables cache
        using perf::memory::cache::invalidate;          // invalidates cache
        #endif // PERF_UEFI and PERF_GNU
      } // namespace cache
    } // namespace memory
  } // namespace core

  inline namespace backend {
    #if PERF_LLVM == 1
    using perf::arch;                                   // march configuration
    using perf::region;
    using perf::analyzer;                               // analyzer{assembly, source, timeline, ...}

    namespace mca {
      namespace stat {
        using perf::mca::stat::cycles;
        using perf::mca::stat::instructions;
        using perf::mca::stat::uops;
      } // namespace stat

      using perf::mca::assembly;                         // ex. 'mov eax, 1' # perf::arch::syntax::intel
      using perf::mca::address;
      using perf::mca::encoding;
      using perf::mca::size;
      using perf::mca::uops;
      using perf::mca::latency;
      using perf::mca::rthroughput;
      using perf::mca::may_load;
      using perf::mca::may_store;
      using perf::mca::has_side_effects;

      namespace branch {
        using perf::mca::branch::is_conditional;
        using perf::mca::branch::is_unconditional;
        using perf::mca::branch::is_indirect;
        using perf::mca::branch::is_call;
        using perf::mca::branch::is_ret;
      } // namespace branch

      using perf::mca::timeline;                        // produces a detailed report of each instruction’s state transitions through an instruction pipeline
      using perf::mca::resource_pressure;               // reports the average number of resource cycles consumed every iteration by instructions for every processor resource unit available on the target
      using perf::mca::bottleneck;                      // correlates increases in backend pressure to dynamic dispatch stalls

      inline namespace debug {
        using perf::mca::debug::source;                 // instruction to source code
      } // namespace debug
    } // namespace mca
    #endif
  } // namespace backend

  inline namespace prof {
    using perf::profiler;                               // profiler{timer{...}, counter{...}, sampler{...}, ...}

    namespace stat {
      using perf::stat::timer;                          // timer{steady_clock, cpu, thread, ...}
      using perf::stat::steady_time;                    // monotonic time # guranateed to be always increasing
      #if PERF_LINUX == 1
      using perf::stat::cpu_time;                       // user time + sys-time
      using perf::stat::thread_time;                    // cpu time for the current thread
      using perf::stat::real_time;                      // wall clock time
      #endif
      #if __has_builtin(__builtin_ia32_rdtsc)
      using perf::stat::tsc;                             // time-stamp-counter
      #endif

      #if PERF_LINUX == 1
      using perf::stat::config;                         // counter configuration
      using perf::stat::counter;                        // perf stat -e

      inline namespace event {
        using perf::stat::cpu_clock;
        using perf::stat::task_clock;
        using perf::stat::faults;
        using perf::stat::page_faults;
        using perf::stat::major_faults;
        using perf::stat::minor_faults;
        using perf::stat::alignment_faults;
        using perf::stat::emulation_faults;
        using perf::stat::context_switches;
        using perf::stat::cgroup_switches;
        using perf::stat::cpu_migrations;
        using perf::stat::migrations;
        using perf::stat::instructions;
        using perf::stat::cycles;
        using perf::stat::stalled_cycles_backend;
        using perf::stat::idle_cycles_backend;
        using perf::stat::stalled_cycles_frontend;
        using perf::stat::idle_cycles_frontend;
        using perf::stat::branches;
        using perf::stat::branch_misses;
        using perf::stat::branch_instructions;
        using perf::stat::cache_references;
        using perf::stat::cache_misses;
        using perf::stat::llc_misses;
        using perf::stat::l1_misses;
        using perf::stat::l1_dcache_loads;
        using perf::stat::l1_dcache_load_misses;
        using perf::stat::l1_icache_loads;
        using perf::stat::l1_icache_load_misses;
        using perf::stat::dtlb_loads;
        using perf::stat::dtlb_load_misses;
        using perf::stat::itlb_loads;
        using perf::stat::itlb_load_misses;

        inline namespace metric {
          using perf::stat::ipc;                        // instruction per cycle (ipc)
          using perf::stat::cpi;                        // cycles per instruction (cpi, inverse of ipc)
          using perf::stat::branch_miss_rate;           // branch miss rate (branch misses per branch instruction)
          using perf::stat::cache_miss_rate;            // cache miss rate (cache misses per cache reference)
          using perf::stat::llc_miss_rate;              // llc miss rate
          using perf::stat::l1_dcache_miss_rate;        // l1 data cache miss rate
          using perf::stat::l1_icache_miss_rate;        // l1 instruction cache miss rate
          using perf::stat::dtlb_miss_rate;             // dtlb miss rate
          using perf::stat::itlb_miss_rate;             // itlb miss rate
          using perf::stat::frontend_stall_rate;        // stalled cycles rate (frontend)
          using perf::stat::backend_stall_rate;         // stalled cycles rate (backend)
          using perf::stat::memory_stall_ratio;         // memory access rate
          using perf::stat::total_stall_rate;           // overall stall rate
          using perf::stat::cpu_migration_rate;         // cpu migrations per cycles
          using perf::stat::context_switch_rate;        // context switches per cycles
          using perf::stat::page_fault_rate;            // page fault rate
          using perf::stat::major_fault_rate;           // page fault rate (major faults per total faults)
          using perf::stat::minor_fault_rate;           // page fault rate (minor faults per total faults)
        } // namespace metric

        namespace top_down {
          using perf::stat::top_down::retiring;
          using perf::stat::top_down::heavy_operations;
          using perf::stat::top_down::light_operations;

          using perf::stat::top_down::bad_speculation;
          using perf::stat::top_down::branch_mispredict;
          using perf::stat::top_down::machine_clears;

          using perf::stat::top_down::frontend_bound;
          using perf::stat::top_down::fetch_latency;
          using perf::stat::top_down::fetch_bandwidth;

          using perf::stat::top_down::backend_bound;
          using perf::stat::top_down::memory_bound;
          using perf::stat::top_down::core_bound;
        } // namespace top_down
      } // namespace event
      #endif // PERF_LINUX
    } // namespace stat

    namespace record {
      #if PERF_LINUX == 1
      using perf::record::config;                       // sampler configuration
      using perf::record::sampler;                      // perf record -e / perf mem record

      inline namespace event {
        using perf::record::cpu_clock;
        using perf::record::task_clock;
        using perf::record::page_faults;
        using perf::record::faults;
        using perf::record::major_faults;
        using perf::record::minor_faults;
        using perf::record::alignment_faults;
        using perf::record::emulation_faults;
        using perf::record::context_switches;
        using perf::record::cgroup_switches;
        using perf::record::cpu_migrations;
        using perf::record::migrations;
        using perf::record::cycles;
        using perf::record::instructions;
        using perf::record::branch_misses;
        using perf::record::bus_cycles;
        using perf::record::cache_misses;
        using perf::record::cache_references;
        using perf::record::branches;
        using perf::record::branch_instructions;
        using perf::record::stalled_cycles_backend;
        using perf::record::idle_cycles_backend;
        using perf::record::stalled_cycles_frontend;
        using perf::record::idle_cycles_frontend;
        using perf::record::llc_misses;
        using perf::record::l1_misses;
        using perf::record::l1_dcache_loads;
        using perf::record::l1_dcache_load_misses;
        using perf::record::l1_icache_loads;
        using perf::record::l1_icache_load_misses;
        using perf::record::dtlb_loads;
        using perf::record::dtlb_load_misses;
        using perf::record::itlb_loads;
        using perf::record::itlb_load_misses;
        using perf::record::mem_loads;
        using perf::record::mem_stores;
      } // namespace event
      #endif
    } // namespace record

    namespace trace {
      #if PERF_LINUX == 1 and PERF_INTEL == 1
      using perf::trace::config;                        // tracing configuration
      using perf::trace::tracer;                        // perf record -e intel_pt/

      inline namespace event {
        using perf::trace::instructions;
        using perf::trace::cycles;
      } // namespace event
      #endif
    } // namespace trace
  } // namespace prof

  inline namespace engine {
    using perf::name;                                   // name of `fn(args...)`
    using perf::dataset;                                // type-safe/type-erased data storage
    using perf::metadata;                               // type-safe/meta-data storage
    using perf::runner;                                 // executes functions via runners and stores results in metadata
  } // namespace engine

  namespace bench {
    using perf::bench::baseline;
    using perf::bench::latency;                         // latency is the time it takes for a single operation to complete (ns)
    using perf::bench::throughput;                      // throughput is the total number of operations or tasks completed in a given amount of time (op/s)

    namespace policy {
      using perf::bench::policy::seq;
      using perf::bench::policy::unseq;
      using perf::bench::policy::unroll;
      using perf::bench::policy::par;
      using perf::bench::policy::omp;
      using perf::bench::policy::cuda;
    } // namespace policy

    inline namespace stat {
      using perf::bench::operations;
      using perf::bench::samples;
    } // namespace stat
  } // namespace bench
} // namespace perf

#if not defined(PERF_IO) or PERF_IO == 1
export namespace perf {
  inline namespace io {
    using enum perf::misc::split;                       // enum class split{hsplit, vsplit}

    using perf::log;                                    // logs to the output stream
    using perf::report;                                 // reports perf benchmark / per operation
    using perf::annotate;                               // annotates per benchmark / per instruction
    using perf::json;                                   // saves results to the output stream in json format

    namespace plot {
      using perf::plot::kind;                           // enum class kind{point, line, bar, box, err}
      using perf::plot::config;                         // plotting configuration
      using perf::plot::gnuplot;                        // http://www.gnuplot.info

      using perf::plot::hist;                           // histogram
      using perf::plot::bar;                            // bar
      using perf::plot::box;                            // box
      using perf::plot::errorbar;                       // error bar
      using perf::plot::ecdf;                           // empirical distribution function
      using perf::plot::flamegraph;                     // https://github.com/brendangregg/FlameGraph
      using perf::plot::flowgraph;                      // https://en.wikipedia.org/wiki/Control-flow_graph
    } // namespace plot
  } // namespace io
} // namespace perf
#endif // PERF_IO

#ifndef NTEST
export namespace perf {
  inline namespace self {
    using perf::self::test;                             // run-time sanity tests
  } // namespace self
} // namespace perf
#endif // NTEST
