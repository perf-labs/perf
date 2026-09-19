# Python API (`src/perf`)

All public entry points are re-exported from `perf`, so `import perf` is the
whole import. Anything the CLI can do, the API can do with the same `code` /
`config` / `data` / `event` values — see [bin/README.md](../bin/README.md) for
the CLI reference (flags, `--config` / `--data` keys, events, the output
envelope, `.perfconfig`).

## `code`

Every entry point that takes a target takes it as `code`: a `[file, target]`
pair, or a raw asm snippet. A `"file:target"` string is accepted as a
shorthand for the pair form.

```py
code = ["a.out", "fizz_buzz"]              # function of a binary
code = ["a.out", ("hot_begin", "hot_end")] # region of a binary
code = ["foo.s", "foo"]                    # label of an assembly source
code = ["foo.s", ("foo", "bar")]           # region of an assembly source
code = "mov eax, 42"                       # raw asm snippet (no file)
code = "a.out:fizz_buzz"                   # shorthand string for the pair form
```

## Synopsis

```py
import perf

# benchmark: a mode is a list, events/config/data mirror the CLI flags
df = perf.benchmark(code=["a.out", "fizz_buzz"], mode=["latency"],
                    event=["duration_time"])
df = perf.benchmark(code=["a.out", ("hot_begin", "hot_end")], mode=["latency"],
                    event=["topdown-*"], data={"regs": {"rdi": 15}})
df = perf.benchmark(code="mov eax, 42", mode=["latency"], event=["cycles"])
df = perf.benchmark(code=["a.s", "myfunc"], mode=["latency"])
df = perf.benchmark(code=["foo.s", ("foo", "bar")], mode=["latency"])
df = perf.benchmark(code=["a.out", "fizz_buzz"], mode=["latency", "throughput"],
                    config={"dcache": ["hot", "cold"]})
df = perf.benchmark(code="mov rax, [rdi]", mode=["latency"],
                    config={"code": [{"align": 1}, {"align": 32}],
                            "thread": [[{"numa": 0, "affinity": 1,
                                         "priority": "normal"}]]})

# per-instruction analysis (index 0..n, one row per instruction)
df = perf.analyze(code=["a.out", "fizz_buzz"])
df = perf.analyze(code=["foo.s", ("foo", "bar")])
df = perf.analyze(code=["a.out", ("hot_begin", "hot_end")],
                  event=["assembly", "encoding"])
df = perf.analyze(code=["a.out", "fizz_buzz"], filter="latency > 4")
df = perf.analyze(code="mov rax, rdi", results=[measured])
df = perf.analyze(code=["a.out", "fizz_buzz"], data={"regs": {"rdi": 15}})
df = perf.analyze(code=["a.out", "fizz_buzz"], setup="init", teardown="fini")

# labels of an assembly source as (name, position, size) in the file
print(perf.asm_labels("foo.s"))

# disassembly / relocatable object of the measured target
text = perf.disassemble(code=["a.out", "fizz_buzz"])
text = perf.disassemble(code=["a.out", ("hot_begin", "hot_end")])

path = perf.to_object(code=["a.out", "fizz_buzz"], path="bench.o")
path = perf.to_object(code=["a.out", ("hot_begin", "hot_end")], path="region.o")
json = perf.to_json(df, indent=4)

# binary/cpu metadata (labels, functions)
cpu = perf.cpuinfo()
meta = perf.metadata("a.out")

# live profiling (labels, functions, hex addresses or (begin, end) regions)
df = perf.profile(cmd=["./a.out"], event=["cycles"])
df = perf.profile(cmd=["./a.out"], event=["cycles"], filter=["hot"])
df = perf.profile(cmd=["./a.out"], event=["cycles"], filter=["fizz_buzz"])
df = perf.profile(cmd=["./a.out"], event=["cycles"], filter=[("work_begin", "work_end")])

# compare
cmp = perf.compare(df, events=["cycles"], baseline="base")

perf.plot(df, ["ecdf"], [["cycles"]])
```

## How it works

Big picture: see the [main README](../README.md#how-it-works) — `perf benchmark` explores symbolically, synthesizes per-iteration data, JITs a harness, steers cache/TLB/branch state, and reads counters, while `perf profile` patches the live process instead. What follows is the full walkthrough.

### 1. A harness around the target

`perf benchmark` JITs a loop around the measured snippet. `latency`
times one call per iteration, `throughput` times the whole loop.
`r8` is the trip count, `r9`/`r10` are the input/output cursors,
`{t0}`/`{t1}` are the counter reads. The prologue keeps `rsp` 16-byte
aligned so the `call` into the target is ABI-correct
(`src/perf/arch/x86_64.py: _BENCH`):

```asm
; latency: per-iteration sample
mov r8, [rdi]                 ; iterations
.perfloop:
    mov rdi, [r9 + 8*r8]      ; per-iteration slot
    ; {data}  steer: values + cache/TLB tier for this iteration
    ; {data2} prime: regs for this iteration (after t0, see below)
    ; {t0}    counter prologue (below)
    ; {code}  guard + call rax (+ lfence) / raw asm snippet
    ; {t1}    counter epilogue (below)
    dec r8
    jnz .perfloop

; throughput: one sample for the whole loop
    ; {data}  steer once
    ; {data2} explicit register reload
    ; {t0}
.perfloop:
    ; {data_iter} values + regs per iteration
    ; {code}
    dec r8
    jnz .perfloop
    ; {t1}
```

Example — what you ask vs what runs:

```sh
perf benchmark a.out:fizz_buzz -m latency -e cycles
# -> latency loop above, {code} = guard + call fizz_buzz, timed per iteration

perf benchmark a.out:fizz_buzz -m throughput -e cycles
# -> throughput loop above, one sample for all iterations

perf benchmark 'imul eax, 0' -m latency
# -> same loops, {code} = the raw snippet (no call)
```

Only the target's own instructions are ever reported. `perf analyze`
disassembles the target itself, never the harness, and the harness's own
timing reads, cache/TLB steering, register priming and call sequence are
subtracted by the differential rather than attributed to the target. The one
harness instruction deliberately inside the measured window is the per-iteration
register priming (`{data2}` in latency, `{data_iter}` in throughput): it has to
run after `rdtsc`/`rdpmc`, which clobber `rax` and `rdx`, so that the target
sees its inputs. It is identical in the `nop` baseline, so it cancels.

### 2. Counters without syscalls, overhead subtracted

Counter reads (`timing()`) need no syscall — `duration_time` uses
`rdtsc`/`rdtscp`, everything else uses `rdpmc` (needs
`echo 2 | sudo tee /sys/devices/{cpu_core,cpu_atom}/rdpmc` once):

```asm
; t0 (duration_time, single event; baseline kept in r11, r10 = output cursor)
push rcx
lfence
rdtsc
mov r11, rax
lfence
pop rcx
; t1
push rcx
rdtscp
lfence
sub rax, r11               ; delta
mov [r10], rax
add r10, 8
pop rcx
; rdpmc events: same shape with `mov ecx, <pmc-index> + lfence + rdpmc + lfence`
; (the read is ordered before and after the window; `rdpmc` alone is only
;  ordered against other counter reads)
; multi-event: one baseline reg per event, `mov [r10+8*i], rax` + `add r10, 8*N`
```

Overhead is subtracted differentially: `loop` measures a `nop`
(`call_seq_nop_asm` for functions) and subtracts its median; `unroll`
measures N vs 2N copies (`backend.unroll.count`, default `5`) and divides by
N. Trip count auto-calibrates (`iterations.min/max`, default `100/1000000`)
to a target relative standard error (`backend.<name>.target`, default
`0.005`) unless `iterations` is pinned. The resolved backend and the
options it ran with are written back into `backend`, keyed by the backend
name (`{"loop": {...}}`), so the `output` rows carry them.

Example — check what the harness did:

```sh
perf benchmark a.out:fizz_buzz --debug 2>&1 | head -50
# prints config, found solutions, synthesized data, full asm, per-run rows

perf benchmark a.out:fizz_buzz --config.iterations=1000
# pin the trip count instead of auto-calibrating

perf benchmark 'rdtsc' -m latency --backend loop
# force the loop backend (snippets default latency -> unroll)
```

Each run is wrapped in the placement guards of `thread` — `_numa_guard`
(`set_mempolicy(MPOL_BIND, node)`, best effort), `_affinity_guard`
(`sched_setaffinity`) and `_priority_guard` (nice) — which restore whatever
the process had before once the runs are done.

```sh
perf benchmark a.out:fizz_buzz --config.thread.affinity=2  # pin to cpu 2
perf benchmark a.out:fizz_buzz --config.thread.numa=1       # bind memory to node 1
```

### 3. Inputs are discovered, not guessed

Before measuring, `bench` discovers what to feed the snippet:

Symbolic exploration (`src/perf/bench.py: explore`): the target (function, `(begin, end)` region, or `asm`
snippet) is lifted to VEX IR and explored with `angr`. Functions use
`call_state` so the ABI prototype is honored, anything else uses
`blank_state` plus a mapped stack (`stack.size/align`); `--setup` asm is
assembled at `_SETUP_BASE` (`0x1000000`) and prepended as a jump stub to the
target. All general purpose registers start as symbolic (`claripy.BVS`),
explicit `--data` scalar registers are constrained (`== value`; list values
stay symbolic and are sampled later), explicit `--data` pages are mapped,
and every memory read/write is recorded with `mem_read`/`mem_write`
breakpoints. Exploration runs `simgr.explore(find=ret_addrs)` (every `ret`
in the function, else the region/snippet end), so one run covers every code
path. Raw `asm` goes through the same path via `explore_asm`: the snippet
is assembled at `_SETUP_BASE`, unknown registers are pinned to scratch
pages (`_ASM_SCRATCH_BASE 0x4100001000 + idx*0x1000`) so pointer derefs do
not fault, and `find=[base+len(encoding)]`.

Low-level example — this snippet has two paths. Exploration finds both
`ret`s, so one benchmark run covers both branches automatically:

```asm
cmp edi, 0
jle .else
mov eax, 1
ret
.else:
mov eax, 2
ret
```

```py
# conceptually what explore() sets up (src/perf/bench.py:explore)
state.registers.store("rdi", claripy.BVS("rdi", 64))  # every GPR symbolic
state.inspect.b("mem_read", when=BP_AFTER, action=record)   # same for mem_write
simgr.explore(find=[ret1, ret2])                            # both rets
# found[0] constraints: rdi > 0,  bbl_addrs: [entry, .then]
# found[1] constraints: rdi <= 0, bbl_addrs: [entry, .else]
```

Solving (`solve()`): each found state is solved with the SMT solver.
Registers and every recorded `(addr_sym, length_sym, value_sym)` are
`solver.eval`'d to concrete models; a symbolic return value (`If`-tree over
`BVV` leaves) forks the state once per leaf. For
`branch.*.mem = predictable` addresses the solver also records
`branch_deps` (which registers/memory each conditional branch consumes via
`cond_branch_analysis`), so predictable runs can pin exactly those inputs.

Continuing the example:

```py
solver.eval(found[0].regs["rdi"])  # -> 1,  constraint rdi > 0
solver.eval(found[1].regs["rdi"])  # -> 0,  constraint rdi <= 0
models = [{"regs": {"rdi": 1}, "reads": [], "writes": []},
          {"regs": {"rdi": 0}, "reads": [], "writes": []}]
```

With memory, `mov rax, [rdi]` records one read; solving yields a concrete
address/value pair, e.g. `reads: [(0x4100001000, 8, 123)]` where
`0x4100001000` is the pinned scratch page for the symbolic `rdi`.

Try it — constrain the explored state yourself:

```sh
# default: both paths sampled automatically
perf benchmark a.out:fizz_buzz -m latency -e cycles

# pin one input: only that state is explored/measured
perf benchmark a.out:fizz_buzz --data.rdi=15 -m latency -e cycles

# sweep three inputs in one run (one measurement per value set)
perf benchmark a.out:fizz_buzz --data.rdi=[1,3,5] -m latency -e cycles
perf analyze a.out:fizz_buzz --data.rdi=15
```

VEX: the executed basic-block addresses (`bbl_addrs` history, plus the block
each state stopped in) select the measured code — `perf analyze` lists the
instructions of every state, `-S` prints only the executed blocks. Static tables are found
by scanning RIP-relative/displacement operands in the range and keeping
the ones the exploration recorded as *data* (`_static_table_addrs`): a
memory access performed by the very instruction that also exits the block
is a `jmp`/`call` target table, not data, so jump tables stay out of the
cache model and the harness never writes over them.
Data synthesis (`_per_iter_data`): models become a per-iteration table (`_per_iter_data`: one
row per register, plus one value row per address, `K = max(iterations+1,
2*addresses)` columns indexed backwards as `r8 = iterations - it`). Each
iteration picks one model — round-robin (`it % n`) when `predictable`,
`rng.randrange(n)` when `unpredictable` — overlays explicit
`--data.rdi` / `--data[addr]` values, and the harness reads the table via
`prime`/`steer` asm. With no `--data`, sampling cycles through all
paths/values, so branches are covered automatically. When one model and
only scalar data are in play the table is filled in one vectorized pass
instead of per iteration.

```asm
; prime_asm: regs for this iteration (r8 = iterations - it)
push r14
movabs r14, 0x...            ; base of the rdi column
mov rdi, [r14 + r8*8]
pop r14

; _value_write_loop: memory values for this iteration
xor r15, r15
.perfwrite:
    movabs r14, 0x...        ; evict-table: target addresses
    mov r12, [r14 + r15*8]
    movabs r14, 0x...        ; evict-table: value-column pointers
    mov r14, [r14 + r15*8]
    mov r13, [r14 + r8*8]    ; value for iteration r8
    mov [r12], r13
.perfwritedone:
    inc r15
    cmp r15, 0x2
    jne .perfwrite
```

### 4. Cache, TLB and branch state per iteration

Cache eviction (`evict()`): every address gets a tier per iteration (`L1d`/`L2`/`L3`/
`DRAM`, plus `L1i` and `TLBd`/`TLBi`). `hot` keeps everything in `L1`,
`cold` flushes to `DRAM`; per-tier hit rates or per-address levels split
addresses across tiers. `steer_asm` writes the values first, then emits
(`evict()`):

```asm
mov rax, [0x...]             ; L1d: touch + fence (stays resident)
mfence
mov r12, 0x...               ; L2: flush, fence, prefetch into L2
clflushopt [r12]
mfence
prefetcht1 [r12]
mov r12, 0x...               ; L3: flush, fence, prefetch into L3
clflushopt [r12]             ; (cldemote when the CPU has it)
mfence
prefetcht2 [r12]
mov r12, 0x...               ; DRAM: flush only
clflushopt [r12]
mfence
; 16x pause                  ; settle
```

Example — one target, two opposite memory worlds:

```sh
# everything served from L1
perf benchmark 'mov rax, [rdi]' --config.dcache=hot -e cache-misses,cycles

# same code, flushed to DRAM every iteration
perf benchmark 'mov rax, [rdi]' --config.dcache=cold -e cache-misses,cycles

# compare L2 vs L3 residency
perf benchmark a.out:fizz_buzz --config.dcache=warm,cool -e cycles
```

The `TLBd`/`TLBi` tiers are orthogonal to the cache tier of the same
address, so an address can be `L1d` hot and `TLBd` cold at the same time
and gets both the cache op and a ring-3 `mprotect` protection toggle (a
non-privileged `invlpg` alternative, `tlb_inval_asm`):

```asm
mov rdi, 0x...               ; page base
mov rsi, 0x2000              ; 2 adjacent pages
mov rdx, 0x7                 ; PROT_RWX
mov rax, r8                  ; the loop counter
and rax, 1                   ; its parity flips the protection ...
shl rax, 1                   ; ... by 2 for code pages, by 1 for data
sub rdx, rax                 ; PROT_RWX <-> PROT_RX / PROT_RW
mov rax, 10                  ; __NR_mprotect
syscall
```

One syscall per iteration, not a `PROT_NONE`/`PROT_RWX` pair, and the
protection the loop counter selects alternates on its own: the page stays
present and usable, so the target only pays the page walk it was asked to
measure. Pages within four pages of each other are coalesced into one range
and the whole block shares a single push/pop prologue, so a spread-out
access pattern costs a handful of syscalls rather than one per page; the
harness restores `PROT_RWX` on every steered page once the loop is done.

`dtlb` applies to the addresses the target loads/stores and `itlb` to the
pages its code lives at (as does `icache`, via the code's data-hierarchy
line). Both take a global rate (`"hot"` or
`{"hit_rate": 50}`) and, per address or register, a rate of their own
(`{"dtlb": {"0x42000000000": {"hit_rate": 0}}}`) on top of it — the tier
names are the internal ones and are not config keys.

Branch data (`branch`): selects predictability globally or per `regs`/`mem`
entry, and takes exactly two values. `predictable` replays models/values in
order (`vals[it % len(vals)]`) and pins branch dependencies, so the target
sees the sequence it was solved with; `unpredictable` samples an index at
random, so the same binary measures both best-case and branch-miss-heavy
behavior without changing code.

```sh
# best case vs branch-miss-heavy, same binary
perf benchmark a.out:fizz_buzz --config.branch=predictable,unpredictable \
  -e cycles,branch-misses
```

### 5. Live profiling without a harness

`perf profile` (`src/perf/prof.py`) patches each tracked address with a 5-byte direct jump to a
trampoline that claims a slot in a shared per-process ring buffer and records
the counter deltas. A label detour looks like this:

```asm
pushfq                          ; keep flags out of the measured path
push rax
push rcx
push rdx
push r11
mov  r11, <ring_base>
mov  rax, [r11 + 8]             ; head
mov  rcx, rax
and  rcx, <cap_mask>
shl  rcx, <slot-stride-log2>    ; byte offset of the slot
lea  rdx, [rax + 1]             ; reserve slot
mov  [r11 + 8], rdx
lea  r11, [r11 + rcx + 64]      ; slot address (64-byte header at buf start)
mov  dword ptr [r11], <point_id>
mov  qword ptr [r11 + 8], rax   ; sequence number
lfence
mov  ecx, <pmc-index>
rdpmc                           ; raw counter (or rdtsc for duration_time)
shl  rdx, 32
or   rax, rdx
mov  [r11 + 16], rax
pop  r11
pop  rdx
pop  rcx
pop  rax
popfq
; <relocated original instructions>
jmp  <resume address>
```

The overwritten instructions are disassembled, relocated (RIP-relative
operands fixed up, short branches expanded to their rel32 forms), and emitted
between the recording block and the jump back, so the process resumes
exactly where it left off. Function entry patches additionally push the real
return address onto a shadow stack and rewrite the saved slot with the
trampoline's exit stub, so every `ret` is intercepted by one patch. The
1 MiB trampoline cave is mapped within 1 GiB (`2^30` bytes) of the binary; the original
file on disk is untouched.

Example — profile, then attribute:

```sh
perf profile --list -- ./a.out                 # what can be tracked
perf profile -f fizz_buzz -e cycles -- ./a.out # one function, live
perf profile -e 'topdown-*' -- ./a.out         # where is it bound?
perf analyze a.out:fizz_buzz -- profile.json   # join counts onto instructions
```
