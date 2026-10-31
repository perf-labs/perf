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

time perf system info --cpu
time perf system info --cpu -e cpu,core,L1d
time perf system info --cpu --json system.json
time perf system tune --profile=default --dry-run
time perf system tune --profile=latency --dry-run --profile.aslr=true --profile.cpu_frequency=max
time perf system tune --profile=throughput --dry-run

time perf benchmark 'imul eax, 42' -e cycles | perf view
time perf benchmark 'imul eax, 42' -m latency -e duration_time --backend unroll | perf view
time perf benchmark 'imul eax, 42' -m latency -e duration_time --config.samples=10 --json | perf view
time perf benchmark 'mov rax, [rdi]' -m throughput -e duration_time --backend loop | perf view
time perf benchmark 'mov rax, [rdi]' -e duration_time --data.rdi=0x42000000000 --data[0x42000000000]=123 --config.dcache=cold --config.dtlb=cold | perf view
time perf benchmark 'mov rax, [rdi]' -m latency -e duration_time --config.icache=hot --config.itlb=hot --config.dtlb=hot --config.branch=predictable --config.code.align=16 | perf plot --config.backend=ascii
time perf benchmark $(printf 'long work(long n){ long s=0; for(long i=0;i<n;i++) s+=i; return s; } int main(){ volatile long s=0; for(int i=0;i<100;i++) s+=work(10); return 0; }' | g++ -x c++ -o work - && echo work):work --list
time perf benchmark $(printf 'long work(long n){ long s=0; for(long i=0;i<n;i++) s+=i; return s; } int main(){ volatile long s=0; for(int i=0;i<100;i++) s+=work(10); return 0; }' | g++ -x c++ -o work - && echo work):work -m latency -e duration_time --data.rdi=100 --config.samples=10 --config.iterations=10 | perf view
time perf benchmark $(printf 'long work(long n){ long s=0; for(long i=0;i<n;i++) s+=i; return s; } int main(){ volatile long s=0; for(int i=0;i<100;i++) s+=work(10); return 0; }' | g++ -x c++ -o work - && echo work):work -m latency -e duration_time --data.rdi=100 --config.samples=10 --config.iterations=10 -o data
time perf benchmark $(printf 'long work(long n){ long s=0; for(long i=0;i<n;i++) s+=i; return s; } int main(){ volatile long s=0; for(int i=0;i<100;i++) s+=work(10); return 0; }' | g++ -x c++ -o work - && echo work):work -m latency -e duration_time --data.rdi=100 --config.samples=10 -c -o work.o
time perf benchmark /usr/bin/ls:main -m latency --json

time perf profile --list -- ./work
time perf profile -f work --list -- ./work
time perf profile -f work -e duration_time -o profile.json -- ./work
time perf view -- profile.json
time perf plot --config.backend=ascii -- profile.json

time perf view -- data
time perf view -e 'duration_time/operations' -s min,median,max -- data
time perf view -f 'mode == "latency"' -g 'file,name,mode' -- data
time perf view -e 'duration_time,data*' -s '' -- data
time perf view --json view.json -- data

time perf plot --config.backend=ascii -- data
time perf plot -e 'duration_time/operations' -t ecdf -t bar --config.backend=ascii -- data
time perf plot -e 'duration_time/operations' -x time -y 'duration_time/operations' --logx --logy --config.backend=ascii -- data
time perf plot -o plot.png --config.backend=agg -- data
time perf plot --json plot.json --config.backend=ascii -- data

time perf benchmark $(printf 'long work(long n){ long s=0; for(long i=0;i<n;i++) s+=i; return s; } int main(){ volatile long s=0; for(int i=0;i<100;i++) s+=work(10); return 0; }' | g++ -x c++ -o work - && echo work):work -n base -m latency -e duration_time --data.rdi=100 --config.samples=10 --config.iterations=10 -o compare
time perf benchmark $(printf 'long work(long n){ long s=0; for(long i=0;i<n;i++) s+=i; return s; } int main(){ volatile long s=0; for(int i=0;i<100;i++) s+=work(10); return 0; }' | g++ -x c++ -o work - && echo work):work -n cand -m latency -e duration_time --data.rdi=100 --config.samples=10 --config.iterations=10 -o compare
time perf compare --baseline base -- compare
time perf compare -e 'duration_time/operations' -f 'mode == "latency"' --alpha 0.01 --json compare.json -- compare

time perf analyze 'mov eax, 42'
time perf analyze $(printf 'long work(long n){ long s=0; for(long i=0;i<n;i++) s+=i; return s; } int main(){ volatile long s=0; for(int i=0;i<100;i++) s+=work(10); return 0; }' | g++ -x c++ -o work - && echo work):work --list
time perf analyze /usr/bin/ls:main -e assembly | llvm-mca
time perf analyze work:work -e 'index,assembly,encoding,size,latency,throughput'
time perf analyze work:work --data.rdi=100
time perf analyze work:work --filter 'size > 2'
time perf analyze work:work --data.rdi=10 --filter '10 in `data.rdi`'
time perf analyze work:work --json analyze.json
time perf analyze work:work -e assembly | llvm-mca

time perf optimize ./work --list -- data
time perf optimize ./work --output work.opt --dry-run -- data
time perf optimize ./work --output work.opt --opt peephole,branch,layout,align --config.layout.align=64 -- data
time ./work.opt
time perf optimize ./work --output work.opt2 --opt layout --config.layout.order=as-is -- data
time ./work.opt2
