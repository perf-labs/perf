#!/bin/bash
#
# The MIT License (MIT)
#
# Copyright (c) 2026 Kris Jusiak <kris@jusiak.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -euo pipefail

# info
perf system info
perf system info -e cpu,core,L1d

perf system tune --status
perf system tune --profile=latency --profile.aslr=true --profile.cpu_frequency=max --dry-run
perf system tune --profile=throughput --dry-run
perf system tune
perf system test

# benchmark
perf benchmark 'imul eax, 42' -m latency -e duration_time
perf benchmark 'mov eax, 42' -m throughput -e duration_time -o snippet.json
perf benchmark 'mov eax, 42' -m latency,throughput -e duration_time
perf benchmark 'mov eax, 42' -e cycles,instructions
perf benchmark 'mov eax, 42' -e cycles --event instructions
perf benchmark 'mov eax, 42' -e 'topdown-*'
perf benchmark 'imul eax, 42' -m latency -e cycles --backend unroll
perf benchmark 'mov rax, [rdi]' -m throughput -e cycles --backend loop
perf benchmark 'mov eax, ebx' --setup 'mov ebx, 1' -e cycles
perf benchmark 'mov rax, [rdi]' -e cycles --data.rdi=0x42000000000 --data[0x42000000000]=123
perf benchmark 'mov eax, 42' -e cycles --config.dcache=hot,cold
perf benchmark 'mov eax, 42' -e cycles --config.branch=predictable,unpredictable
perf benchmark 'mov eax, 42' -e cycles --config.code.align=1,16
perf benchmark 'mov eax, 42' -e cycles --config.seed=123
perf benchmark 'mov eax, 42' -e cycles --config.thread.affinity=0
perf benchmark 'mov eax, 42' -e cycles --config.stack.size=4096
perf benchmark 'mov rax, [rdi]' -e cycles --config.heap.size=4096
perf benchmark 'mov eax, [rdi]' -e cycles --data.rdi=0x42000000000 --data[0x42000000000]=42 --config.dcache.L1='[{hit_rate:100},{hit_rate:50}]'
perf benchmark 'mov eax, 42' -e cycles --debug

cat > fizz.cpp <<'EOF'
extern "C" [[nodiscard, gnu::noinline]] auto func(int n) -> const char* {
    auto r = "Unknown";
    if (n % 15 == 0) r = "FizzBuzz";
    else if (n % 3 == 0) r = "Fizz";
    else if (n % 5 == 0) r = "Buzz";
    return r;
}

extern "C" [[nodiscard, gnu::noinline]] auto hot(int n) -> int {
    auto s = 0;
    PERF_LABEL(hot_begin);
    for (int i = 0; i < n; ++i) {
        s += i;
    }
    PERF_LABEL(hot_end);
    return s;
}

int main(int argc, const char**) {
    return func(argc + 14)[0] + hot(argc + 99);
}
EOF

g++ -O3 -I lib -include perf/perf.h -c -o fizz.o fizz.cpp
g++ -o fizz fizz.o

perf benchmark fizz.o --list
perf benchmark fizz.o:hot -m latency -e duration_time --data.arg0=100 -o hot.json
perf benchmark fizz.o:hot_begin..hot_end -m latency -e cycles --data.arg0=100
perf benchmark fizz.o:func -m latency -e cycles --data.arg0=15
perf benchmark fizz.o:func -e cycles --data.arg0='[1,3,5]'

cat > work.cpp <<'EOF'
extern "C" [[gnu::noinline]] auto work(int n) -> int {
    auto s = 0;
    for (auto i = 0; i < n; ++i) {
        s += i;
    }
    return s;
}

int main(int argc, const char**) {
    return work(argc);
}
EOF

g++ -O3 -c -o work.o work.cpp
g++ -o work work.o
perf benchmark work.o:work -m latency -e duration_time --data.rdi=100 -o work.json
perf benchmark work.o:work -m throughput -e cycles --data.rdi=100
perf benchmark work.o:work -e cycles --data.rdi=100

cat > asm.s <<'EOF'
.text
.globl asm_bench
.type asm_bench, @function
asm_bench:
  mov eax, 42
asm_end:
  ret
.size asm_bench, .-asm_bench
EOF

perf benchmark asm.s:asm_bench -e cycles
perf benchmark asm.s:asm_bench..asm_end -e cycles

cat > member.cpp <<'EOF'
struct counter {
    [[nodiscard, gnu::noinline]] auto member(long n) -> long {
        PERF_LABEL(member_begin);
        auto s = 0;
        for (long i = 0; i < n; ++i) {
            s += value; value += step;
        }
        PERF_LABEL(member_end);
        return s;
    }

    long value{};
    long step{};
};

counter global{0, 3};
void setup() { global.value = 1; }
void teardown() { global.value = 0; }
int main() { return static_cast<int>(global.member(10)); }
EOF

g++ -O3 -I lib -include perf/perf.h -c -o member.o member.cpp
g++ -O3 -shared -o libwork.so work.cpp

perf benchmark "member.o:counter::member(long)" -m latency -e cycles --setup member.o:setup --teardown member.o:teardown --data.rdi=0x42000000000 --data.rsi=100 --data[0x42000000000:]=1 --data[0x42000000008:]=3
perf benchmark libwork.so:work -e cycles --data.rdi=100

# profile
perf profile -f hot -e cycles -o profile.json -- ./fizz
perf profile -f func -f hot -e cycles -o profile-multi.json -- ./fizz
perf profile -f hot_begin..hot_end -e cycles -o profile-region.json -- ./fizz
perf profile -f hot -e cycles --buffer-size 1024 -o profile-buf.json -- ./fizz
perf profile -f hot -e cycles --output -- ./fizz
perf profile -f work -e cycles -o profile-work.json -- ./work

# view
perf view -e 'duration_time/operations' -s min,median,max -- hot.json
perf view -e 'duration_time/operations' -s min,median,p99 -- snippet.json
perf view -f 'mode == "latency"' -- hot.json
perf view -e 'data*' -s '' -- hot.json
perf view -g '' -s '' -- work.json
perf view --output view.json -- work.json
perf view -- view.json
perf view -- profile.json
perf view -- profile-multi.json
perf view -- profile-region.json
perf view -- profile-buf.json
perf view -- profile-work.json
mkdir -p results
cp hot.json snippet.json work.json results/ 2>/dev/null || true
perf view -- results/
perf benchmark 'mov eax, 42' -m latency -e duration_time | perf view -e 'duration_time/operations' -s min,median,p99

# plot
perf plot --config.backend=ascii -e 'duration_time/operations' -t hist -- hot.json
perf plot --config.backend=ascii -t ecdf,bar -- hot.json
perf plot --config.backend=ascii -t line -t scatter -t point -- hot.json
perf plot --config.backend=ascii -e 'duration_time/operations' -x time -y 'duration_time/operations' --config.logx=true --config.logy=true -- hot.json
perf plot --config.backend=ascii -g file,mode -f 'mode == "latency"' -- hot.json
perf plot -o plot.png --config.backend=agg -e 'duration_time/operations' -- hot.json
perf plot -t boxen -o boxen.png --config.backend=agg -- hot.json
perf plot --output plot.json -- hot.json
perf plot --config.backend=ascii -- profile.json

# compare
perf benchmark 'mov eax, 42' -m latency -e duration_time -n base -o base.json
perf benchmark 'mov eax, 42' -m latency -e duration_time -n cand -o cand.json
perf compare --baseline base -- base.json cand.json
perf compare -e 'duration_time/operations' -f 'mode == "latency"' --alpha 0.01 -- base.json cand.json

# analyze
perf analyze 'mov eax, 42' -e assembly,latency,throughput
perf analyze 'mov eax, 42' -e assembly | llvm-mca
perf analyze fizz.o:hot --data.arg0=100 -e assembly,latency,throughput
perf analyze fizz.o:hot_begin..hot_end --data.arg0=100
perf analyze fizz.o:func -e 'assembly,data*' --data.arg0=15
perf analyze fizz.o:hot --filter 'latency > 1'
perf analyze fizz.o:hot --data.arg0=100 --filter '100 in `data.rdi`'
perf analyze "member.o:counter::member(long)" --data.rdi=0x42000000000 --data.rsi=100
perf analyze asm.s:asm_bench
perf analyze fizz.o:hot -- profile.json
perf analyze fizz.o:hot -- hot.json
perf analyze fizz.o:hot --data.arg0=100 -e assembly,latency,throughput --output analyze.json
perf view -- analyze.json

cat > debug.cpp <<'EOF'
auto func(int x) -> int {
    auto sum = 0;
    for (auto i = 0; i < 4; i++) {
        if (x > 2) {
            sum += x * i;
        } else {
            sum -= i;
        }
    }
    return sum;
}
EOF
g++ -O3 -g -c -o debug.o debug.cpp

perf analyze debug.o:func -e index,assembly,source --config.source.backend=debug
perf analyze debug.o:func -e index,assembly,source --config.source.backend=decompile
perf record -e cycles:u -c 100000 -o perf.data -- ./work
perf analyze work:work -- perf.data

# optimize
perf optimize work --list -- work.json
perf optimize work --dry-run -- work.json
perf optimize work --output work.opt -- work.json
perf optimize work --output work.peephole --opt peephole -- work.json
perf optimize work --output work.branch --opt branch -- work.json
perf optimize work --output work.layout --opt layout -- work.json
perf optimize work --output work.align --opt align --config.layout.align=64 -- work.json
perf optimize work --output work.all --opt peephole,branch,layout,align -- work.json
perf optimize work --output work.hot --opt layout --config.layout.order=hot-first -- work.json profile-work.json
perf optimize work --output work.report -- work.json
perf optimize work --config.opt=peephole,branch --output work.config -- work.json

# run
perf stat -e cycles,instructions -- ./work
taskset -c 1 perf stat -e cycles,instructions -- ./work

./work.opt
./work.peephole
./work.branch
./work.layout
./work.align
./work.all
./work.hot
./work.config
./work.report
