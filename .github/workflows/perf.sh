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

TMP="${TMP:-/tmp}"

cat > $TMP/work.c <<'EOF'
long work(long n) { long s = 0; for (long i = 0; i < n; ++i) s += i; return s; }
int main() { volatile long s = 0; for (int i = 0; i < 100; ++i) s += work(10); return 0; }
EOF
cat > $TMP/fizz.c <<'EOF'
#include "perf/perf.h"
int add42(int x) { return x + 42; }
const char *func(int n) {
    PERF_LABEL(fizz_begin);
    const char *r = "Unknown";
    if (n % 15 == 0) r = "FizzBuzz";
    else if (n % 3 == 0) r = "Fizz";
    else if (n % 5 == 0) r = "Buzz";
    PERF_LABEL(fizz_end);
    return r;
}
int hot(int n) {
    PERF_LABEL(hot_begin);
    int s = 0;
    for (int i = 0; i < n; ++i) s += i;
    PERF_LABEL(hot_end);
    return s;
}
int main() { return func(15)[0] + add42(1) + hot(10); }
EOF
cat > $TMP/member.cpp <<'EOF'
#include "perf/perf.h"
struct Counter {
    long value;
    long step;
    __attribute__((noinline)) long member(long n) {
        PERF_LABEL(member_begin);
        long s = 0;
        for (long i = 0; i < n; ++i) { s += value; value += step; }
        PERF_LABEL(member_end);
        return s;
    }
};
Counter global{0, 3};
void setup() { global.value = 1; }
void teardown() { global.value = 0; }
int main() { return (int)global.member(10); }
EOF
cat > $TMP/setup.c <<'EOF'
long counter = 0;
void myinit() { counter = 42; }
long mywork(long x) { return x + counter; }
int main() { return 0; }
EOF
cat > $TMP/asm.s <<'EOF'
	.text
	.globl asm_bench
	.type asm_bench, @function
asm_bench:
	mov eax, 42
	ret
	.size asm_bench, .-asm_bench
EOF
cat > $TMP/argv.c <<'EOF'
#include <stdio.h>
int main(int argc, char **argv) { printf("argc=%d\n", argc); return 0; }
EOF
cat > $TMP/debug.c <<'EOF'
long myfunc(long x){
  long sum = 0;
  for (int i = 0; i < 4; i++) {
    if (x > 2) sum += x * i;
    else sum -= i;
  }
  return sum;
}
int main() { return (int)myfunc(3); }
EOF
gcc -O2 -fPIE -pie -o $TMP/work $TMP/work.c
gcc -O2 -fPIE -pie -I lib -o $TMP/fizz $TMP/fizz.c
g++ -O2 -fPIE -pie -I lib -o $TMP/member $TMP/member.cpp
gcc -O2 -fPIE -pie -o $TMP/setup $TMP/setup.c
gcc -O2 -fPIE -pie -o $TMP/argv $TMP/argv.c
gcc -O2 -g -fno-inline -o $TMP/debug $TMP/debug.c
gcc -O2 -fPIE -pie -c -o $TMP/func.o $TMP/work.c

perf system info --cpu
perf system info --cpu -e cpu,core,L1d
perf system info --cpu -e arch
perf system info --cpu -e 'L*'
perf system info --cpu --json $TMP/system.json

perf system tune --dry-run
perf system reset --dry-run
perf system tune --profile=latency --dry-run
perf system tune --profile=throughput --dry-run
perf system tune --profile=latency --dry-run --profile.aslr=true
perf system tune --profile=latency --dry-run --profile.cpu_frequency=max
perf system tune --profile=latency --dry-run --profile.governor=performance --profile.smt=false
perf system tune --profile=latency --dry-run --json
perf system tune --profile=throughput --dry-run --json
echo '{"aslr": true}' > $TMP/tune.json
perf system tune --dry-run --profile=$TMP/tune.json
perf system tune --dry-run --profile=$TMP/tune.json --json

perf benchmark $TMP/fizz --list
perf benchmark $TMP/fizz:hot --list
perf benchmark $TMP/fizz --list -e 'name,begin,end'
perf analyze $TMP/fizz --list
perf analyze $TMP/fizz:hot --list
perf profile --list -- $TMP/fizz

perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m throughput -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'imul eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'imul eax, 42' -m latency -e duration_time --backend unroll --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov rax, [rdi]' -m throughput -e duration_time --backend loop --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency,throughput -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'imul eax, ebx' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 --config.backend.unroll.count=4
perf benchmark 'imul eax, ebx' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 --config.backend.unroll.runs=5
perf benchmark 'mov eax, 42' -m throughput -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 --config.backend.loop.runs=5 --config.backend.loop.target=0.01
perf benchmark 'mov eax, ebx' --setup 'mov ebx, 1' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16

perf benchmark 'mov eax, 42' -m latency -e cycles --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e cycles,instructions --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e cycles --event instructions --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e branch-misses,branch-instructions --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e cache-misses,cache-references --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e L1-dcache-load-misses,LLC-load-misses,dTLB-load-misses --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e 'topdown-*' --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e cycles,instructions --event 'topdown-*' --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov rax, [rdi]' -m latency -e duration_time --data.rdi=0x42000000000 --data[0x42000000000]=123 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16

perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable,unpredictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot,warm,cool,cold --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot,cold --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot,cold --config.itlb=hot,cold --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=1,16
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 --config.seed=123
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 --config.thread.affinity=0 --config.thread.priority=normal
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 --config.stack.size=2097152 --config.func.align=16 --config.func.order=as-is
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 --config.func.order=random
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 --config.external.lib=true
perf benchmark 'mov eax, 42' -m latency -e duration_time --data.rdi=0x42000000000 --data[0x42000000000]=123 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=cold --config.dtlb=cold --config.icache=hot --config.itlb=hot --config.code.align=16

perf benchmark 'mov rax, [rdi]' -m latency -e duration_time --data.rdi=0x42000000000 --data[0x42000000000]=123 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov rax, [rdi]' -m latency -e duration_time --data.rdi=0x42000000000 --data[0x42000000000]=[1,2,3] --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable,unpredictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 | perf view

perf benchmark $TMP/fizz:hot -m latency -e duration_time --data.rdi=10 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/fizz:hot_begin..hot_end -m latency -e duration_time --data.rdi=10 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/fizz:func -m latency -e duration_time --data.arg0=15 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/fizz:func -m latency -e duration_time --data.arg0=[1,3,5,15] --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/fizz:add42 -m latency -e cycles --data.rdi=1 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/fizz:func -m latency -e cycles,instructions --data.arg0=15 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/fizz:func -m latency -e 'topdown-*' --data.arg0=15 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/work:work -m latency -e duration_time --data.rdi=10 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/work:work -m throughput -e duration_time --data.rdi=10 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/work:work -m latency -e duration_time --data.rdi=100 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/asm.s:asm_bench -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/func.o:work -m latency -e duration_time --data.rdi=10 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/setup:mywork --setup $TMP/setup:myinit -m latency -e duration_time --data.rdi=1 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark "$TMP/member:Counter::member(long)" -m latency -e duration_time --setup $TMP/member:setup --teardown $TMP/member:teardown --data.rdi=0x42000000000 --data.rsi=100 --data[0x42000000000:]=1 --data[0x42000000008:]=3 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark $TMP/member:member_begin..member_end -m latency -e duration_time --setup $TMP/member:setup --teardown $TMP/member:teardown --data.rdi=0x42000000000 --data.rsi=100 --data[0x42000000000:]=1 --data[0x42000000008:]=3 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark /usr/lib/x86_64-linux-gnu/libc.so.6:strlen -m latency -e duration_time --data.rdi=0x4200000000 --data[0x4200000000:]=hello --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark /usr/lib/x86_64-linux-gnu/libc.so.6:strlen -m latency -e cycles --data.rdi=0x4200000000 --data[0x4200000000:]=aaaaaaaaaaaaaaaa --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark /usr/lib/x86_64-linux-gnu/libc.so.6:qsort -m latency -e cycles,instructions --data.rsi=[1,10,100,1000] --data.rdx=4 --data.rcx=0x4200000000 --data[0x4200000000:]=[139,7,59,6,15,159,192,15,156,194,15,182,192,15,182,210,41,208,195] --config.samples=1 --config.iterations=4 --config.branch=predictable --config.dcache=hot --config.dtlb=cold --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark /usr/bin/ls:main -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16
perf benchmark /usr/bin/ls:main -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 -- $TMP
perf benchmark /usr/bin/ls:main -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 --env LC_ALL=C -- $TMP
perf benchmark $TMP/argv:main -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 -- /hello /world
perf benchmark 'mov eax, [rdi]' -m latency -e duration_time --data.rdi=0x42000000000 --data[0x42000000000]=42 --config.branch=predictable --config.icache=hot --config.itlb=hot --config.samples=1 --config.dcache.L1d='[{hit_rate:100},{hit_rate:50}]' --config.dtlb[0x4200000000]=hot,cold --config.iterations.count=10,20 --config.code.align=1,16
perf benchmark $TMP/work:work -m latency -e duration_time --data.rdi=10 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 --json $TMP/bench.json
perf benchmark $TMP/work:work -m latency -e duration_time --data.rdi=10 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 -c -o $TMP/hot.o

perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 -o $TMP/out-snippet
perf benchmark $TMP/work:work -m latency -e duration_time --data.rdi=10 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 -o $TMP/out-work
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 | perf view
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 | perf plot --config.backend=ascii
perf benchmark $TMP/fizz:hot -m latency -e duration_time --data.rdi=10 --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 -o $TMP/out-fizz
perf benchmark $TMP/fizz:func -m latency -e cycles,instructions --data.arg0=[1,3,5] --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 -o $TMP/out-sweep

perf profile -f hot -e cycles -o $TMP/profile.json -- $TMP/fizz
perf profile -f hot -f func -e cycles -o $TMP/profile-multi.json -- $TMP/fizz
perf profile -f hot_begin..hot_end -e cycles -o $TMP/profile-region.json -- $TMP/fizz
perf profile -f hot -e cycles --buffer-size 1024 -o $TMP/profile-buf.json -- $TMP/fizz
perf profile -f hot -e cycles -o none -- $TMP/fizz
perf profile -f hot -e duration_time -o $TMP/profile-tsc.json -- $TMP/fizz
perf profile -f hot -e branch-misses -o $TMP/profile-br.json -- $TMP/fizz
perf profile -f work -e cycles -o $TMP/profile-work.json -- $TMP/work
perf profile -f hot -e cycles -o $TMP/profile-json.json --json -- $TMP/fizz
perf view -- $TMP/profile.json
perf plot --config.backend=ascii -- $TMP/profile.json

perf view -- $TMP/out-snippet
perf view -e 'duration_time/operations' -s min,median,max -- $TMP/out-snippet
perf view -e 'duration_time/operations' -s min,median,p10,p50,p90,p99,max -- $TMP/out-work
perf view -g '' -s '' -- $TMP/out-snippet
perf view -f 'mode == "latency"' -- $TMP/out-snippet
perf view -f 'mode == "latency"' -g 'file,name,mode' -- $TMP/out-fizz
perf view -e 'duration_time,data*' -s '' -- $TMP/out-fizz
perf view -e 'cycles/instructions' -- $TMP/out-sweep
perf view --json $TMP/view.json -- $TMP/out-snippet
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=10 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 | perf view -s min,median,p99

perf plot --config.backend=ascii -- $TMP/out-snippet
perf plot --config.backend=ascii -e 'duration_time/operations' -- $TMP/out-snippet
perf plot --config.backend=ascii -t ecdf -t bar -- $TMP/out-snippet
perf plot --config.backend=ascii -t hist -- $TMP/out-snippet
perf plot --config.backend=ascii -t line -- $TMP/out-snippet
perf plot --config.backend=ascii -t scatter -- $TMP/out-snippet
perf plot --config.backend=ascii -t point -- $TMP/out-snippet
perf plot --config.backend=ascii -e 'duration_time/operations' -x time -y 'duration_time/operations' --logx --logy -- $TMP/out-snippet
perf plot --config.backend=ascii -g file,mode -- $TMP/out-snippet
perf plot --config.backend=ascii -f 'mode == "latency"' -- $TMP/out-snippet
perf plot --config.backend=ascii -x data.rdi -- $TMP/out-sweep
perf plot -o $TMP/plot.png --config.backend=agg -- $TMP/out-snippet
perf plot -t boxen -o $TMP/boxen.png --config.backend=agg -- $TMP/out-snippet
perf plot --json $TMP/plot.json --config.backend=ascii -- $TMP/out-snippet

rm -rf $TMP/cmp
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=40 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 -n base -o $TMP/cmp
perf benchmark 'mov eax, 42' -m latency -e duration_time --config.samples=1 --config.iterations=40 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 -n cand -o $TMP/cmp
perf compare --baseline base -- $TMP/cmp
perf compare -e 'duration_time/operations' -f 'mode == "latency"' --alpha 0.01 -- $TMP/cmp
perf compare --baseline base --json $TMP/compare.json -- $TMP/cmp
rm -rf $TMP/cmp-work
perf benchmark $TMP/work:work -m latency -e duration_time --data.rdi=100 --config.iterations=40 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 -n base -o $TMP/cmp-work
perf benchmark $TMP/work:work -m latency -e duration_time --data.rdi=100 --config.iterations=40 --config.branch=predictable --config.dcache=hot --config.dtlb=hot --config.icache=hot --config.itlb=hot --config.code.align=16 -n cand -o $TMP/cmp-work
perf compare --baseline base -- $TMP/cmp-work

python3 -c "import json, random; r = random.Random(7); [json.dump({'file': 'a.out@x', 'name': n, 'id': 'x', 'time': 't', 'info': {}, 'output': [{'mode': 'latency', 'operations': 1, 'cycles': r.gauss(m, 5.0)} for _ in range(60)]}, open(f'$TMP/synth.{n}.json', 'w')) for n, m in (('base', 100.0), ('cand', 80.0))]"
perf compare --baseline base -- $TMP/synth.base.json $TMP/synth.cand.json
perf compare -e cycles --alpha 0.01 --json -- $TMP/synth.base.json $TMP/synth.cand.json

perf analyze 'mov eax, 42'
perf analyze 'mov eax, 42' -e assembly
perf analyze 'mov eax, 42' -e 'index,assembly,encoding,size,latency,throughput'
perf analyze 'mov eax, 42' -e 'data*'
perf analyze 'mov eax, 42' --filter 'size > 2'
perf analyze 'mov eax, 42' --json $TMP/analyze-snippet.json
perf analyze 'mov eax, 42' -e assembly | llvm-mca

perf analyze $TMP/fizz:hot --data.rdi=10
perf analyze $TMP/fizz:hot_begin..hot_end --data.rdi=10
perf analyze $TMP/fizz:func --data.arg0=15 -e assembly,latency,throughput
perf analyze $TMP/fizz:hot -e 'index,assembly,encoding,size,latency,throughput'
perf analyze $TMP/fizz:hot -e 'assembly,data*'
perf analyze $TMP/fizz:hot -e '*'
perf analyze $TMP/fizz:func --filter 'latency > 1'
perf analyze $TMP/fizz:func --data.arg0=15 --filter '15 in `data.rdi`'
perf analyze $TMP/fizz:hot --data.rdi=10 --filter '10 in `data.rdi`'
perf analyze "$TMP/member:Counter::member(long)" --data.rdi=0x42000000000 --data.rsi=100
perf analyze $TMP/asm.s:asm_bench
perf analyze $TMP/debug:myfunc -e index,assembly --debug
perf analyze $TMP/fizz:hot --json $TMP/analyze.json
perf analyze $TMP/fizz:hot -- $TMP/profile.json
perf analyze $TMP/fizz:hot -- $TMP/out-work
perf analyze $TMP/fizz:hot -e 'index,assembly' -- $TMP/profile.json
perf record -F 999 -e cycles:u -o $TMP/perf.data -- $TMP/work
perf analyze $TMP/work:work -- $TMP/perf.data
perf analyze $TMP/work:work -e 'index,assembly,cycles' -- $TMP/perf.data
perf analyze $TMP/fizz:hot -e assembly | llvm-mca

perf analyze $TMP/setup:mywork --setup $TMP/setup:myinit --data.rdi=1
perf analyze 'mov eax, ebx' --setup 'mov ebx, 1'
perf analyze $TMP/fizz:hot --data.rdi=10 -n hot10

perf optimize $TMP/fizz --list -- $TMP/out-work
perf optimize $TMP/fizz --list --json -- $TMP/out-work
perf optimize $TMP/fizz --dry-run -- $TMP/out-work
perf optimize $TMP/work --dry-run -- $TMP/out-work
perf optimize $TMP/work --output $TMP/work.peephole --opt peephole -- $TMP/out-work
perf optimize $TMP/work --output $TMP/work.branch --opt branch -- $TMP/out-work
perf optimize $TMP/work --output $TMP/work.layout --opt layout -- $TMP/out-work
perf optimize $TMP/work --output $TMP/work.align --opt align --config.layout.align=64 -- $TMP/out-work
perf optimize $TMP/work --output $TMP/work.all --opt peephole,branch,layout,align --config.layout.align=32 --config.layout.order=as-is -- $TMP/out-work
perf optimize $TMP/work --output $TMP/work.hot --opt layout --config.layout.order=hot-first -- $TMP/out-work $TMP/profile-work.json
perf optimize $TMP/work --json $TMP/optimize.json --output $TMP/work.json -- $TMP/out-work
$TMP/work.peephole
$TMP/work.branch
$TMP/work.layout
$TMP/work.align
$TMP/work.all
$TMP/work.hot
$TMP/work.json

perf stat -e cycles,instructions -- $TMP/work
taskset -c 0 perf stat -e cycles,instructions -- $TMP/work
