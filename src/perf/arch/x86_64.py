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

import functools
import re

import capstone
import keystone

_DATA_TIERS = ("L1d", "L2", "L3")
_CACHE_TIER_TAGS = {tier: tier for tier in ("L1i", *_DATA_TIERS, "DRAM")}

ARG_REGS = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
HARNESS_REGS = {
    "r8",
    "r9",
    "r10",
    "r11",
    "r12",
    "r13",
    "r14",
    "r15",
    "rbx",
    "rbp",
    "rsp",
    "rip",
    "esp",
    "ip",
    "flags",
}
CACHE_TIERS = [*_DATA_TIERS, "DRAM"]
PERF_SYSCALL_NR = 298
FLAG_WRITERS = {
    "adc",
    "add",
    "and",
    "cmp",
    "dec",
    "imul",
    "inc",
    "mul",
    "neg",
    "or",
    "sbb",
    "sar",
    "shl",
    "shr",
    "sub",
    "test",
    "xor",
}
ANGR_ARCH_NAME = "AMD64"
SETUP_BASE = 0x1000000
ASM_SCRATCH_BASE = 0x4100001000
STACK_ADDR = 0x7FFF00000000
STACK_SIZE = 0x100000
PRIME_SCRATCH_REG = "r14"
SETTLE_PAUSES = 16
HARNESS_LOOP_REGS = ["r8", "r9", "r10"]
TSC_READER_ASM = "rdtsc\nshl rdx, 32\nor rax, rdx\nret"
BENCH = {
    # void bench(u64 iterations, const void* inputs, void* outputs)
    "latency": r"""
        push rbx
        push rbp
        push r12
        push r13
        push r14
        push r15
        mov r8, [rdi]
        mov r9, [rdi + 0x8]
        mov r10, [rdi + 0x10]
        {setup}
    .loop:
        mov rdi, [r9 + 8*r8]
        {data}
        {t0}
        {data2}
        {code}
        {t1}
        dec r8
        jnz .loop
        {teardown}
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbp
        pop rbx
        ret
    """,
    # void bench(u64 iterations, const void* inputs, void* outputs)
    "throughput": r"""
        push rbx
        push rbp
        push r12
        push r13
        push r14
        push r15
        mov r8, [rdi]
        mov r9, [rdi + 0x8]
        mov r10, [rdi + 0x10]
        {setup}
        {data}
        {t0}
        {data2}
    .loop:
        {data_iter}
        {code}
        dec r8
        jnz .loop
        {t1}
        {teardown}
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbp
        pop rbx
        ret
    """,
}

_MEMORY_SHORTCUTS = {
    "hot": {"L1d": 100, "L2": 0, "L3": 0},
    "warm": {"L1d": 0, "L2": 100, "L3": 0},
    "cool": {"L1d": 0, "L2": 0, "L3": 100},
    "cold": {"L1d": 0, "L2": 0, "L3": 0},
}
_REG_CODES = {
    "rax": (0, False),
    "rcx": (1, False),
    "rdx": (2, False),
    "rbx": (3, False),
    "rsp": (4, False),
    "rbp": (5, False),
    "rsi": (6, False),
    "rdi": (7, False),
    "r8": (0, True),
    "r9": (1, True),
    "r10": (2, True),
    "r11": (3, True),
    "r12": (4, True),
    "r13": (5, True),
    "r14": (6, True),
    "r15": (7, True),
    "eax": (0, False),
    "ecx": (1, False),
    "edx": (2, False),
    "ebx": (3, False),
    "esp": (4, False),
    "ebp": (5, False),
    "esi": (6, False),
    "edi": (7, False),
}
_BASELINE_REGS = ["rbx", "rbp", "r12", "r13", "r14", "r15", "r11"]
_CALLEE_SAVED_BASELINES = ["rbx", "rbp", "r12", "r13", "r14", "r15"]
_REG_PATTERN = re.compile(
    r"\b(r(?:8|9|1[0-5])|rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|"
    r"eax|ebx|ecx|edx|esi|edi|ebp|esp|"
    r"ax|bx|cx|dx|si|di|bp|sp|"
    r"al|bl|cl|dl|sil|dil|bpl|spl|r8b|r9b|r10b|r11b|r12b|r13b|r14b|r15b|"
    r"ah|bh|ch|dh)\b",
    re.IGNORECASE,
)
_SUBREG_TO_FULL = {
    "eax": "rax",
    "ax": "rax",
    "al": "rax",
    "ah": "rax",
    "ebx": "rbx",
    "bx": "rbx",
    "bl": "rbx",
    "bh": "rbx",
    "ecx": "rcx",
    "cx": "rcx",
    "cl": "rcx",
    "ch": "rcx",
    "edx": "rdx",
    "dx": "rdx",
    "dl": "rdx",
    "dh": "rdx",
    "esi": "rsi",
    "si": "rsi",
    "sil": "rsi",
    "edi": "rdi",
    "di": "rdi",
    "dil": "rdi",
    "ebp": "rbp",
    "bp": "rbp",
    "bpl": "rbp",
    "esp": "rsp",
    "sp": "rsp",
    "spl": "rsp",
    "r8b": "r8",
    "r9b": "r9",
    "r10b": "r10",
    "r11b": "r11",
    "r12b": "r12",
    "r13b": "r13",
    "r14b": "r14",
    "r15b": "r15",
}


def assembler():
    return keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)


def disassembler():
    return capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)


def assemble(asm, addr=0):
    encoding, _ = assembler().asm(asm, addr)
    return bytes(encoding)


def imm(v):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return str(v)
    return f"0x{v & 0xFFFFFFFFFFFFFFFF:x}"


def canonical_data_reg(NAME):
    n = str(NAME).strip().lower()
    if n.startswith("arg"):
        try:
            idx = int(n[3:], 0)
        except ValueError:
            idx = None
        if idx is not None and 0 <= idx < len(ARG_REGS):
            return ARG_REGS[idx]
    return n


def arg_alias(reg):
    try:
        canon = canonical_data_reg(reg)
    except Exception:
        return None
    try:
        idx = ARG_REGS.index(canon)
    except ValueError:
        return None
    return f"arg{idx}"


def data_reg_keys(reg):
    try:
        canon = canonical_data_reg(reg)
    except Exception:
        canon = str(reg).strip().lower()
    keys = {str(reg), canon}
    try:
        alias = arg_alias(canon)
    except Exception:
        alias = None
    if alias:
        keys.add(alias)
        keys.add(alias.lower())

    out = set()
    for k in keys:
        if k is None:
            continue
        s = str(k).strip()
        if s:
            out.add(s)
            out.add(s.lower())
    return out


def normalize_asm(code):
    if not code:
        return code

    def _repl(m):
        prefix, num, suffix = m.group(1), m.group(2), m.group(3)
        try:
            v = int(num, 10)
        except ValueError:
            return m.group(0)
        return f"{prefix}0x{v & 0xFFFFFFFFFFFFFFFF:x}{suffix}"

    return re.sub(r"(,\s*)(-?\d+)\s*(;|$|\n)", _repl, code)


def tsc_reader_bytes():
    return assemble(TSC_READER_ASM)


def data_reload_asm(data):
    if not data or not data.get("regs"):
        return ""
    return "\n".join(
        f"mov {canonical_data_reg(reg)}, {imm(_first_val(val))}"
        for reg, val in data.get("regs", {}).items()
    )


def data_setup_asm(data):
    from ..core import _parse_addr_key

    if not data:
        return ""
    asm = data_reload_asm(data).splitlines() if data.get("regs") else []
    for addr_str, val in data.get("mem", {}).items():
        base_str, is_range = _parse_addr_key(addr_str)
        try:
            base = int(base_str, 0)
        except (TypeError, ValueError):
            continue
        vals = list(val) if (is_range and isinstance(val, (list, tuple))) else [val]
        if is_range:
            for i, v in enumerate(vals):
                asm.extend(_store_asm(base + i, v))
        else:
            asm.extend(_store_asm(base, _first_val(val)))
    return "\n".join(asm)


def call_asm(target):
    return f"mov rax, {imm(target)}\ncall rax\n"


def call_seq_asm(target, fence=False):
    body = "call rax\nlfence\n" if fence else "call rax\n"
    return (
        f"push r8\npush r9\npush r10\nmov rax, {imm(target)}\n{body}"
        "pop r10\npop r9\npop r8"
    )


def call_seq_nop_asm(fence=False):
    body = "mov rax, 0\nlfence\n" if fence else "mov rax, 0\n"
    return f"push r8\npush r9\npush r10\n{body}pop r10\npop r9\npop r8"


def jump_asm(target):
    return f"jmp {imm(target)}"


def setup_jump_asm(setup_asm, target):
    return f"{setup_asm.rstrip(chr(10))}\n" + jump_asm(target)


def map_stack(state, size=None, align=16):
    try:
        size = (
            int(size, 0)
            if isinstance(size, str)
            else int(size)
            if size is not None
            else STACK_SIZE
        )
    except (TypeError, ValueError):
        size = STACK_SIZE
    if size <= 0:
        size = STACK_SIZE
    try:
        align = (
            int(align, 0)
            if isinstance(align, str)
            else int(align)
            if align is not None
            else 16
        )
    except (TypeError, ValueError):
        align = 16
    if align < 1 or (align & (align - 1)):
        raise ValueError(f"stack align must be a power of two >= 1, got {align!r}")
    state.memory.map_region(STACK_ADDR, size, 7)
    rsp = STACK_ADDR + size // 2
    rsp &= ~(align - 1)
    state.regs.rsp = rsp


def set_ip(state, addr):
    state.regs.rip = addr


def data_cache_levels(config):
    cache_cfg = (config or {}).get("cache", {})
    if isinstance(cache_cfg, str):
        levels = _memory_shortcut(cache_cfg)
        if levels is None:
            raise ValueError(
                f"unknown cache shortcut {cache_cfg!r}; "
                "expected one of 'hot', 'warm', 'cool', 'cold'"
            )
        return dict(levels)

    cache = {}
    for _cd in _collect_cache_dicts(config):
        cache.update(_cd)
    if not cache:
        return None
    if any(k in ("mem", "memory") for k in cache):
        raise ValueError(
            "unknown cache key 'mem'; use 'cache' directly "
            "(e.g. {'cache': {'L1d': {'hit_rate': 100}}})"
        )

    levels = {tier: None for tier in _DATA_TIERS}
    for key, value in cache.items():
        tier = _CACHE_TIER_TAGS.get(str(key).strip())
        if tier is None or tier not in _DATA_TIERS:
            continue
        rate = _entry_rate(tier, value)
        if rate is not None:
            levels[tier] = rate

    if all(v is None for v in levels.values()):
        return None
    return levels


def normalize_cache_spec(spec):
    if spec is None:
        return None
    if isinstance(spec, str):
        shortcut = _memory_shortcut(spec)
        if shortcut:
            return dict(shortcut)
        m = re.fullmatch(r"hit_rate\s*[:=]\s*(-?\d+)\s*", spec.strip())
        if m:
            rate = int(m.group(1))
            return {"L1d": rate, "L2": 0, "L3": 0}
        return None
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        try:
            rate = int(spec)
        except (TypeError, ValueError):
            return None
        return {"L1d": rate, "L2": 0, "L3": 0}
    if not isinstance(spec, dict):
        return None
    src = spec.get("hit_rate", None)
    if src is None:
        src = {k: v for k, v in spec.items() if k != "hit_rate"}
    if isinstance(src, (int, float)) and not isinstance(src, bool):
        src = {"L1d": src}
    if not isinstance(src, dict):
        return None
    out = {}
    for key, value in src.items():
        tier = _CACHE_TIER_TAGS.get(str(key).strip())
        if tier is None:
            continue
        try:
            out[tier] = int(value)
        except (TypeError, ValueError):
            continue
    return out or None


def resolve_mem_cache(config):
    out = {}
    if not isinstance(config, dict):
        return out
    for cache in _collect_cache_dicts(config):
        for key, entry in cache.items():
            if key in ("mem", "memory"):
                raise ValueError(
                    "unknown cache key 'mem'; use 'cache' directly "
                    "(e.g. {'cache': {'0x1000': 'hot'}})"
                )
            tier = _CACHE_TIER_TAGS.get(str(key).strip())
            if tier is not None:
                if tier not in _DATA_TIERS and tier != "L1i":
                    continue
                _, addrs = _split_default_addrs(entry)
                for addr, spec in addrs.items():
                    rate = _entry_rate(tier, spec)
                    if rate is not None:
                        _merge_mem_cache(out, addr, {tier: rate})
                for reg, spec in _split_default_regs(entry).items():
                    rate = _entry_rate(tier, spec)
                    if rate is not None:
                        _merge_mem_cache(out, reg, {tier: rate})
                continue

            s = str(key).strip()
            if _is_addr_key(s):
                try:
                    _merge_mem_cache(out, int(s, 0), normalize_cache_spec(entry))
                except (TypeError, ValueError):
                    continue
            elif _is_reg_key(s):
                try:
                    canon = canonical_data_reg(s)
                except Exception:
                    canon = s.lower()
                _merge_mem_cache(out, canon, normalize_cache_spec(entry))
    return out


def addr_tier(levels):
    if isinstance(levels, dict) and _tier_rate(levels, "L1i") is not None:
        return "L1i"
    best, best_rate = "DRAM", 0.0
    for tier in _DATA_TIERS:
        try:
            rate = _tier_rate(levels, tier)
        except AttributeError:
            rate = None
        if rate is None:
            continue
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            continue
        if rate > best_rate:
            best, best_rate = tier, rate
    return best


def sample_tier(levels, rng):
    if not levels:
        return "L1d"
    for tier in CACHE_TIERS[:-1]:
        hit = _tier_rate(levels, tier)
        if hit is None:
            continue
        if rng.random() * 100 < float(hit):
            return tier
    return "DRAM"


@functools.lru_cache(maxsize=1)
def has_cldemote():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("flags"):
                    _, _, flags = line.partition(":")
                    if "cldemote" in flags.split():
                        return True
    except OSError:
        pass
    return False


def cldemote_asm(reg):
    n = str(reg).strip().lower()
    if n not in _REG_CODES:
        raise ValueError(f"unknown register {reg!r} for cldemote")
    code, ext = _REG_CODES[n]
    prefix = "0x41, " if ext else ""
    low = code & 0x7
    if low == 4:
        return f".byte {prefix}0x0F, 0x1C, 0x04, 0x24"
    if low == 5:
        return f".byte {prefix}0x0F, 0x1C, 0x45, 0x00"
    return f".byte {prefix}0x0F, 0x1C, 0x{low:02x}"


def settle_asm(n=SETTLE_PAUSES):
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return ""
    return "\n".join(["pause"] * n)


def evict(
    addresses,
    levels=None,
    reg="r12",
    settle=SETTLE_PAUSES,
    cldemote=None,
    mem_levels=None,
):
    if not addresses:
        return ""
    levels = levels or {"L1d": 100, "L2": 100, "L3": 100}
    mem_levels = mem_levels or {}

    def _rate(tier):
        try:
            return int(_tier_rate(levels, tier) or 0)
        except (TypeError, ValueError):
            return 0

    groups = {"L1d": [], "L1i": [], "L2": [], "L3": [], "DRAM": []}
    rest = []
    for a in addresses:
        spec = mem_levels.get(a)
        if isinstance(spec, dict) and spec:
            groups[addr_tier(spec)].append(a)
        else:
            rest.append(a)

    remaining = len(rest)
    n_l1 = int(remaining * _rate("L1d") / 100.0)
    remaining -= n_l1
    n_l2 = int(remaining * _rate("L2") / 100.0)
    remaining -= n_l2
    n_l3 = int(remaining * _rate("L3") / 100.0)
    remaining -= n_l3
    groups["L1d"].extend(rest[:n_l1])
    groups["L2"].extend(rest[n_l1 : n_l1 + n_l2])
    groups["L3"].extend(rest[n_l1 + n_l2 : n_l1 + n_l2 + n_l3])
    groups["DRAM"].extend(rest[n_l1 + n_l2 + n_l3 :])

    use_cldemote = bool(has_cldemote()) if cldemote is None else bool(cldemote)
    lines = []
    for a in groups["L1d"]:
        lines.append(f"mov rax, [{imm(a)}]")
        lines.append("mfence")
    for a in groups["L1i"]:
        lines.append(f"mov {reg}, {imm(a)}")
        lines.append(f"clflushopt [{reg}]")
        lines.append("mfence")
    for a in groups["L2"]:
        lines.append(f"mov {reg}, {imm(a)}")
        lines.append(f"clflushopt [{reg}]")
        lines.append("mfence")
        lines.append(f"prefetcht1 [{reg}]")
    for a in groups["L3"]:
        if use_cldemote:
            lines.append(f"mov rax, [{imm(a)}]")
            lines.append("mfence")
            lines.append(f"mov {reg}, {imm(a)}")
            lines.append(cldemote_asm(reg))
        else:
            lines.append(f"mov {reg}, {imm(a)}")
            lines.append(f"clflushopt [{reg}]")
            lines.append("mfence")
            lines.append(f"prefetcht2 [{reg}]")
    for a in groups["DRAM"]:
        lines.append(f"mov {reg}, {imm(a)}")
        lines.append(f"clflushopt [{reg}]")
        lines.append("mfence")

    lines.append(settle_asm(settle))
    return "\n".join(lines).rstrip()


def steer_asm(meta, data_ptr, settle=SETTLE_PAUSES, mem_levels=None, write_values=True):
    if not meta or not (meta.get("mem_addrs") or meta.get("l1i_addrs")):
        return ""
    if mem_levels is None:
        mem_levels = meta.get("mem_levels")
    parts = []
    if meta.get("mem_addrs"):
        parts.extend(
            p
            for p in (
                _value_write_loop(meta, data_ptr) if write_values else None,
                evict(
                    meta["mem_addrs"],
                    meta.get("levels"),
                    settle=settle,
                    mem_levels=mem_levels,
                ),
            )
            if p
        )
    l1i_addrs = meta.get("l1i_addrs") or []
    if l1i_addrs:
        l1i_levels = {
            a: {"L1i": spec.get("L1i")}
            for a, spec in (mem_levels or {}).items()
            if isinstance(spec, dict) and a in l1i_addrs and spec.get("L1i") is not None
        }
        p = evict(l1i_addrs, None, settle=settle, mem_levels=l1i_levels)
        if p:
            parts.append(p)
    return "\n".join(parts)


def prime_asm(meta, data_ptr):
    if not meta or not meta.get("reg_col"):
        return ""
    K = meta["K"]
    base = data_ptr or 0
    lines = []
    for NAME, c in meta["reg_col"].items():
        col_base = base + c * K * 8
        lines.append(f"movabs r14, {imm(col_base)}")
        lines.append(f"mov {NAME}, [r14 + r8*8]")
    return "\n".join(lines)


def per_iter_asm(meta, data_ptr):
    """Per-iteration data loading for throughput loops: writes each memory
    address's value from the per-iteration buffer and loads each register's
    value, indexed by the current iteration counter (r8)."""
    parts = []
    if meta and meta.get("mem_addrs"):
        parts.append(_value_write_loop(meta, data_ptr))
    prime = prime_asm(meta, data_ptr) if meta else ""
    if prime:
        parts.append(prime)
    return "\n".join(parts)


def mem_regs(asm_text, addr=0):
    if not asm_text or not str(asm_text).strip():
        return set()
    try:
        encoding, _ = assembler().asm(str(asm_text), addr)
    except Exception:
        return set()
    if not encoding:
        return set()
    md = disassembler()
    try:
        md.detail = True
    except Exception:
        pass
    regs = set()
    try:
        insns = md.disasm(bytes(encoding), addr)
    except Exception:
        return set()
    for insn in insns:
        try:
            operands = insn.operands
        except Exception:
            continue
        for op in operands:
            try:
                is_mem = op.type == capstone.x86.X86_OP_MEM
            except Exception:
                continue
            if not is_mem:
                continue
            for attr in ("base", "index"):
                try:
                    code = getattr(op.mem, attr)
                except Exception:
                    continue
                if code:
                    try:
                        regs.add(_canonical_reg(md.reg_name(code)))
                    except Exception:
                        pass
    return regs


def used_regs(asm_text):
    if not asm_text:
        return set()
    return {_canonical_reg(m.group(0)) for m in _REG_PATTERN.finditer(asm_text)}


def pick_baselines(n, avoid=(), callee_saved=False):
    avoid_full = {_canonical_reg(r) for r in (avoid or ())}
    pool = list(_CALLEE_SAVED_BASELINES) + ([] if callee_saved else ["r11"])
    free = [r for r in pool if r not in avoid_full]
    if len(free) < n:
        raise ValueError(
            f"cannot pick {n} baseline register(s) disjoint from "
            f"{sorted(avoid_full)}; free callee-saved: {free}"
        )
    return free[:n]


def timing_regs(events="duration_time", indices=None, callee_saved=False, avoid=()):
    if isinstance(events, str):
        events = [events]
    regs = pick_baselines(len(events), avoid=avoid, callee_saved=callee_saved)
    if len(events) == 1 and not callee_saved:
        if _canonical_reg("r11") not in {_canonical_reg(r) for r in (avoid or ())}:
            regs = ["r11"]
    return list(regs)


def guard_asm(regs):
    regs = [str(r).strip().lower() for r in (regs or []) if str(r).strip()]
    seen, uniq = set(), []
    for r in regs:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    pre = [f"push {r}" for r in uniq]
    post = [f"pop {r}" for r in reversed(uniq)]
    if len(uniq) % 2:
        pre.append("sub rsp, 8")
        post.insert(0, "add rsp, 8")
    return ("\n".join(pre), "\n".join(post))


def timed_guard(events="duration_time", indices=None, callee_saved=False, avoid=()):
    regs = list(HARNESS_LOOP_REGS)
    for reg in timing_regs(events, indices, callee_saved=callee_saved, avoid=avoid):
        if reg not in regs:
            regs.append(reg)
    return guard_asm(regs)


def timing(events="duration_time", indices=None, callee_saved=False, avoid=()):
    if isinstance(events, str):
        events = [events]

    if isinstance(indices, int) or indices is None:
        indices = [indices] * len(events)

    def _read(i):
        if events[i] == "duration_time":
            return "lfence\nrdtsc\nshl rdx, 32\nor rax, rdx"
        idx = indices[i]
        if idx is None:
            raise ValueError(f"event {events[i]!r} requires an rdpmc counter index")
        return f"mov ecx, {imm(idx)}\nlfence\nrdpmc\nshl rdx, 32\nor rax, rdx"

    if len(events) > len(_BASELINE_REGS):
        raise ValueError(
            f"too many simultaneous events ({len(events)}); "
            f"supported: {len(_BASELINE_REGS)}"
        )
    if callee_saved and len(events) > len(_BASELINE_REGS) - 1:
        raise ValueError(
            "too many simultaneous events for a throughput measurement; the "
            f"{len(events)}th event would need the volatile r11 baseline, "
            "which the timed loop body can clobber"
        )

    regs = timing_regs(events, indices, callee_saved=callee_saved, avoid=avoid)

    _save = "push rcx"
    _restore = "pop rcx"

    if len(events) == 1 and events[0] == "duration_time":
        base = regs[0]
        return (
            f"{_save}\nlfence\nrdtsc\nmov {base}, rax\nlfence\n{_restore}",
            f"{_save}\nrdtscp\nlfence\nsub rax, {base}\n"
            f"mov [r10], rax\nadd r10, {imm(8)}\n{_restore}",
        )

    if len(events) == 1:
        base = regs[0]
        return (
            f"{_save}\n{_read(0)}\nmov {base}, rax\nlfence\n{_restore}",
            f"{_save}\n{_read(0)}\nsub rax, {base}\n"
            f"mov [r10], rax\nadd r10, {imm(8)}\n{_restore}",
        )

    start = [_save]
    for i, reg in enumerate(regs):
        start.append(_read(i))
        start.append(f"mov {reg}, rax")
    start.append("lfence")
    start.append(_restore)
    end = [_save]
    for i, reg in enumerate(regs):
        end.append(_read(i))
        end.append(f"sub rax, {reg}")
        end.append(f"mov [r10 + {imm(8 * i)}], rax" if i else "mov [r10], rax")
    end.append(f"add r10, {imm(8 * len(events))}")
    end.append(_restore)
    return ("\n".join(start), "\n".join(end))


def _first_val(val):
    if isinstance(val, (list, tuple)):
        return val[0] if val else 0
    return val


def _store_asm(addr, v):
    try:
        v = int(v, 0) if isinstance(v, str) else int(v)
    except (TypeError, ValueError):
        v = 0
    head = f"mov r12, {imm(addr)}"
    if 0 <= v < 256:
        return [head, f"mov byte ptr [r12], {imm(v)}"]
    if v <= 0xFFFFFFFF:
        return [head, f"mov dword ptr [r12], {imm(v)}"]
    return [head, f"mov rax, {imm(v)}", "mov qword ptr [r12], rax"]


def _is_addr_key(key):
    try:
        int(str(key), 0)
        return True
    except (TypeError, ValueError):
        return False


def _split_default_addrs(entry):
    if not isinstance(entry, dict):
        return entry, {}
    addrs = {}
    for key, spec in entry.items():
        if key == "default" or not _is_addr_key(key):
            continue
        addrs[int(str(key).strip(), 0)] = spec
    if not addrs:
        has_regs = any(
            k != "default" and not _is_addr_key(k) and _is_reg_key(k) for k in entry
        )
        if not has_regs:
            return entry, {}
        default = entry.get("default", entry)
        return default, {}
    default = entry.get("default", entry)
    return default, addrs


def _is_reg_key(key):
    s = str(key).strip()
    if not s or s == "default":
        return False
    if _is_addr_key(s):
        return False
    if _CACHE_TIER_TAGS.get(s.strip()) is not None:
        return False
    if s.lower() in ("default", "hit_rate"):
        return False
    return True


def _split_default_regs(entry):
    if not isinstance(entry, dict):
        return {}
    return {
        canonical_data_reg(key): spec for key, spec in entry.items() if _is_reg_key(key)
    }


def _entry_rate(tier, spec):
    spec, _ = _split_default_addrs(spec)
    if spec is None:
        return None
    if isinstance(spec, str):
        s = spec.strip()
        if not s:
            return None
        m = re.fullmatch(r"hit_rate\s*[:=]\s*(-?\d+)\s*", s)
        if m:
            return int(m.group(1))
        sc = _memory_shortcut(s)
        return sc.get(tier) if sc is not None else None
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return int(spec)
    if not isinstance(spec, dict):
        return None
    src = spec.get("hit_rate")
    if isinstance(src, (int, float)) and not isinstance(src, bool):
        return int(src)
    flat = src if isinstance(src, dict) else spec
    if not isinstance(flat, dict):
        return None
    parsed = normalize_cache_spec(flat)
    return parsed.get(tier) if parsed else None


def _tier_rate(levels, tier):
    return levels.get(tier) if isinstance(levels, dict) else None


def _memory_shortcut(NAME):
    if not isinstance(NAME, str):
        return None
    return _MEMORY_SHORTCUTS.get(NAME.strip().lower())


def _collect_cache_dicts(config):
    out = []
    if not isinstance(config, dict):
        return out
    cache = config.get("cache")
    if isinstance(cache, dict):
        out.append(cache)
    return out


def _merge_mem_cache(out, addr, levels):
    if not levels:
        return
    cur = out.get(addr)
    if cur is None:
        out[addr] = dict(levels)
        return
    for tier, rate in levels.items():
        cur[tier] = rate


def _value_write_loop(meta, data_ptr):
    K = meta["K"]
    base = data_ptr or 0
    addrs = meta["mem_addrs"]
    n = len(addrs)
    t_addr = base + meta["evict_table_col"] * K * 8
    t_vb = t_addr + n * 8
    return f"""\
    xor r15, r15
.Ld:
    movabs r14, {imm(t_addr)}
    mov r12, [r14 + r15*8]
    movabs r14, {imm(t_vb)}
    mov r14, [r14 + r15*8]
    mov r13, [r14 + r8*8]
    mov [r12], r13
.Ldn:
    inc r15
    cmp r15, {imm(n)}
    jne .Ld
"""


def _canonical_reg(NAME):
    n = str(NAME).strip().lower()
    return _SUBREG_TO_FULL.get(n, n)
