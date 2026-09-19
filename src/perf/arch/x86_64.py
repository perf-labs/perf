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

import ctypes
import functools
import mmap
import re
import struct

import capstone
import keystone

_DATA_TIERS = ("L1d", "L2", "L3")
_TLB_TIERS = ("TLBd", "TLBi")
_CACHE_TIER_TAGS = {tier: tier for tier in (*_TLB_TIERS, "L1i", *_DATA_TIERS, "DRAM")}
_CACHE_TIERS_BY_KEY = {
    "dcache": _DATA_TIERS,
    "icache": ("L1i",),
    "dtlb": ("TLBd",),
    "itlb": ("TLBi",),
}
_CACHE_TIERS_BY_KEY_TIER = {
    tier: tiers for tiers in _CACHE_TIERS_BY_KEY.values() for tier in tiers
}
_ARG_REGS = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
_HARNESS_REGS = {
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
_FLAG_WRITERS = {
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
_ANGR_ARCH_NAME = "AMD64"
_SETUP_BASE = 0x1000000
_ASM_SCRATCH_BASE = 0x4100001000
_STACK_ADDR = 0x7FFF00000000
_STACK_SIZE = 0x100000
_PRIME_SCRATCH_REG = "r14"
_SETTLE_PAUSES = 16
_NR_MPROTECT = 10
_PROT_R = 0x1
_PROT_W = 0x2
_PROT_RW = _PROT_R | _PROT_W
_PROT_X = 0x4
_PROT_RX = _PROT_R | _PROT_X
_PROT_RWX = _PROT_R | _PROT_W | _PROT_X
_PAGE_SIZE = mmap.PAGESIZE
_SUBREG_TO_FULL = {
    "eax": "rax",
    "ax": "rax",
    "al": "rax",
    "ah": "rax",
    "ip": "rip",
    "eip": "rip",
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
_REG_PATTERN = re.compile(
    r"\b(r(?:8|9|1[0-5])|rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|"
    r"eax|ebx|ecx|edx|esi|edi|ebp|esp|"
    r"ax|bx|cx|dx|si|di|bp|sp|"
    r"al|bl|cl|dl|sil|dil|bpl|spl|r8b|r9b|r10b|r11b|r12b|r13b|r14b|r15b|"
    r"ah|bh|ch|dh)\b",
    re.IGNORECASE,
)
_CALLEE_SAVED_BASELINES = ["rbx", "rbp", "r12", "r13", "r14", "r15"]
_BASELINE_REGS = ["rbx", "rbp", "r12", "r13", "r14", "r15", "r11"]
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
_MEMORY_SHORTCUTS = {
    "hot": {"L1d": 100, "L2": 0, "L3": 0},
    "warm": {"L1d": 0, "L2": 100, "L3": 0},
    "cool": {"L1d": 0, "L2": 0, "L3": 100},
    "cold": {"L1d": 0, "L2": 0, "L3": 0},
}
_TIER_SHORTCUTS = {
    "hot": {"L1i": 100, "TLBd": 100, "TLBi": 100},
    "warm": {"L1i": 0, "TLBd": 0, "TLBi": 0},
    "cool": {"L1i": 0, "TLBd": 0, "TLBi": 0},
    "cold": {"L1i": 0, "TLBd": 0, "TLBi": 0},
}
_BENCH = {
    "latency": r"""
        push rbx
        push rbp
        push r12
        push r13
        push r14
        push r15
        sub rsp, 0x8
        mov r8, [rdi]
        mov r9, [rdi + 0x8]
        mov r10, [rdi + 0x10]
        {setup}
    .perfloop:
        mov rdi, [r9 + 8*r8]
        {data}
        {t0}
        {data2}
        {code}
        {t1}
        dec r8
        jnz .perfloop
        {tlb}
        {teardown}
        add rsp, 0x8
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbp
        pop rbx
        ret
    """,
    "throughput": r"""
        push rbx
        push rbp
        push r12
        push r13
        push r14
        push r15
        sub rsp, 0x8
        mov r8, [rdi]
        mov r9, [rdi + 0x8]
        mov r10, [rdi + 0x10]
        {setup}
        {data}
        {t0}
        {data2}
    .perfloop:
        {data_iter}
        {code}
        dec r8
        jnz .perfloop
        {t1}
        {tlb}
        {teardown}
        add rsp, 0x8
        pop r15
        pop r14
        pop r13
        pop r12
        pop rbp
        pop rbx
        ret
    """,
}
_USER_REGS_FIELDS = (
    "r15",
    "r14",
    "r13",
    "r12",
    "rbp",
    "rbx",
    "r11",
    "r10",
    "r9",
    "r8",
    "rax",
    "rcx",
    "rdx",
    "rsi",
    "rdi",
    "orig_rax",
    "rip",
    "cs",
    "eflags",
    "rsp",
    "ss",
    "fs_base",
    "gs_base",
    "ds",
    "es",
    "fs",
    "gs",
)
_TRACK_SLOT_EVENTS_OFF = 16
_TRACK_SLOT_SEQ_OFF = 8
_TRACK_SLOT_ID_OFF = 0
_TRACK_RING_DATA_OFF = 64
_TRACK_RING_HEAD_OFF = 8
_TRACK_RING_HDR_SIZE = 64
_TRACK_RING_MAGIC = 0x5045524654524B00
_JMP_OP = 0xE9
_PATCH_SIZE = 5
_NR_PERF_EVENT_OPEN = 298
_NR_MMAP = 9
_PERF_SYSCALL_NR = _NR_PERF_EVENT_OPEN
_INT3_OP = b"\xcc"
_SYSCALL_OP = b"\x0f\x05"
_TSC_READER_ASM = "rdtsc\nshl rdx, 32\nor rax, rdx\nret"
_HARNESS_LOOP_REGS = ["r8", "r9", "r10"]
_HARNESS_RESERVED_REGS = ("r8", "r9", "r10", "rsp", "rip", "flags")
_TLB_SAVE = ("push rax", "push rdi", "push rsi", "push rdx", "push rcx", "push r11")
_TLB_RESTORE = ("pop r11", "pop rcx", "pop rdx", "pop rsi", "pop rdi", "pop rax")
_TLB_MERGE_GAP = 64
_ARGV_SCRATCH = "r11"
_ARGV_REGS = ("rdi", "rsi", "rdx")
_TLB_RUN_PAGES = 512
_ELF_MACHINE = 62
_R_X86_64_64 = 1
_R_X86_64_COPY = 5
_R_X86_64_GLOB_DAT = 6
_R_X86_64_JUMP_SLOT = 7
_R_X86_64_RELATIVE = 8
_R_X86_64_IRELATIVE = 37
_POINTER_SIZE = 8
_ADDR_MASK = 0xFFFFFFFFFFFFFFFF
_FILL_BYTE = 0xCC
_NOP_BYTE = 0x90
_BRANCH_REACH = 2**31
_STACK_ENTRY_ALIGN = 16
_ABS_MOV_OPCODES = (b"\xa0", b"\xa1", b"\xa2", b"\xa3")
_MOV_IMM32 = b"\x48\xc7\xc0"
_MOVABS = b"\x48\xb8"
_MOVABS_IMM_OFF = 2
_MOVABS_LEN = 10
_NATIVE_RET_TYPE = ctypes.c_long
_NATIVE_ARG_SCRATCH = (
    "xor eax, eax",
    "xor ebx, ebx",
    "xor ecx, ecx",
    "xor edx, edx",
    "xor esi, esi",
    "xor edi, edi",
    "xor r8d, r8d",
    "xor r9d, r9d",
)


def tlb_inval_asm(page, pages=1, prot=_PROT_RW, prologue=True):
    return _mprotect_asm(page, pages, prot, True, prologue)


def tlb_restore_call(page, pages=1, prologue=True):
    return _mprotect_asm(page, pages, _PROT_RWX, False, prologue)


@functools.lru_cache(maxsize=1)
def assembler():
    return keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)


def disassembler():
    return capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)


def att_disassembler():
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.syntax = capstone.CS_OPT_SYNTAX_ATT
    return md


def osaca_isa():
    return "x86"


@functools.lru_cache(maxsize=4096)
def assemble(asm, addr=0):
    encoding, _ = assembler().asm(asm, addr)
    return bytes(encoding)


@functools.lru_cache(maxsize=16384)
def imm(v):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return str(v)
    return f"0x{v & 0xFFFFFFFFFFFFFFFF:x}"


@functools.lru_cache(maxsize=4096)
def canonical_data_reg(NAME):
    n = str(NAME).strip().lower()
    if n.startswith("arg"):
        try:
            idx = int(n[3:], 0)
        except ValueError:
            idx = None
        if idx is not None and 0 <= idx < len(_ARG_REGS):
            return _ARG_REGS[idx]
    return n


def arg_alias(reg):
    try:
        canon = canonical_data_reg(reg)
    except Exception:
        return None
    try:
        idx = _ARG_REGS.index(canon)
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


@functools.lru_cache(maxsize=4096)
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
    return assemble(_TSC_READER_ASM)


def data_reload_asm(data):
    if not data or not data.get("regs"):
        return ""
    return "\n".join(
        f"mov {canonical_data_reg(reg)}, {imm(_first_val(val))}"
        for reg, val in data.get("regs", {}).items()
    )


def data_setup_asm(data):
    if not data:
        return ""
    asm = data_reload_asm(data).splitlines() if data.get("regs") else []
    for addr_str, val in data.get("mem", {}).items():
        s = str(addr_str).strip()
        is_range = s.endswith(":")
        base_str = s[:-1].strip() if is_range else s
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


def argv_asm(frame):
    if not frame:
        return "", "", ""
    save = int(frame["save"])
    sp = int(frame["sp"]) - 8
    enter = "\n".join(
        [
            f"mov {_ARGV_SCRATCH}, {save:#x}",
            f"mov [{_ARGV_SCRATCH}], rsp",
            f"mov {_ARGV_SCRATCH}, {sp:#x}",
            f"mov rsp, {_ARGV_SCRATCH}",
        ]
    )
    leave = "\n".join(
        [f"mov {_ARGV_SCRATCH}, {save:#x}", f"mov rsp, [{_ARGV_SCRATCH}]"]
    )
    load = "\n".join(
        f"mov {reg}, {int(frame[key]):#x}"
        for reg, key in zip(_ARGV_REGS, ("argc", "argv", "envp"))
    )
    return enter, leave, load


def call_seq_asm(target, fence=False):
    body = "call rax\nlfence\n" if fence else "call rax\n"
    return (
        f"push r8\npush r9\npush r10\npush r11\n"
        f"mov rax, {imm(target)}\n{body}"
        "pop r11\npop r10\npop r9\npop r8"
    )


def call_seq_nop_asm(fence=False):
    body = "mov rax, 0\nlfence\n" if fence else "mov rax, 0\n"
    return (
        f"push r8\npush r9\npush r10\npush r11\n{body}pop r11\npop r10\npop r9\npop r8"
    )


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
            else _STACK_SIZE
        )
    except (TypeError, ValueError):
        size = _STACK_SIZE
    if size <= 0:
        size = _STACK_SIZE
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
    state.memory.map_region(_STACK_ADDR, size, 7)
    rsp = _STACK_ADDR + size // 2
    rsp &= ~(align - 1)
    state.regs.rsp = rsp


def set_ip(state, addr):
    state.regs.rip = addr


def validate_cache_key(key, value):
    tiers = _CACHE_TIERS_BY_KEY.get(key)
    if not tiers or not isinstance(value, dict):
        return
    for name in value:
        error = _tier_key_error(key, name, tiers)
        if error is not None:
            raise error


def data_cache_levels(config):
    cache_cfg = (config or {}).get("dcache", {})
    if isinstance(cache_cfg, str):
        levels = _memory_shortcut(cache_cfg)
        if levels is None:
            raise ValueError(
                f"unknown cache shortcut {cache_cfg!r}; "
                "expected one of 'hot', 'warm', 'cool', 'cold'"
            )
        return dict(levels)

    cache = {}
    for _cd in _collect_cache_dicts(config, keys=("dcache",)):
        cache.update(_cd)
    if not cache:
        return None
    if any(k in ("mem", "memory") for k in cache):
        raise ValueError(
            "unknown cache key 'mem'; use 'dcache' directly "
            "(e.g. {'dcache': {'L1d': {'hit_rate': 100}}})"
        )

    levels = {tier: None for tier in _DATA_TIERS}
    for key, value in cache.items():
        error = _tier_key_error("dcache", key, _DATA_TIERS)
        if error is not None:
            raise error
        tier = _tier_tag(key)
        if str(key) == "hit_rate":
            rate = _rate(value)
            if rate is not None:
                levels[_DATA_TIERS[0]] = rate
                for tier_name in _DATA_TIERS[1:]:
                    levels[tier_name] = 0
            continue
        if tier is None:
            continue
        rate = _entry_rate(tier, value)
        if rate is not None:
            levels[tier] = rate

    if all(v is None for v in levels.values()):
        return None
    return levels


def instruction_cache_levels(config):
    return _tier_levels(config, "icache", _CACHE_TIERS_BY_KEY["icache"])


def data_tlb_levels(config):
    return _tier_levels(config, "dtlb", _CACHE_TIERS_BY_KEY["dtlb"])


def instruction_tlb_levels(config):
    return _tier_levels(config, "itlb", _CACHE_TIERS_BY_KEY["itlb"])


def normalize_cache_spec(spec, tiers=_DATA_TIERS):
    tiers = tuple(tiers) or _DATA_TIERS
    primary = tiers[0]
    if spec is None:
        return None
    if isinstance(spec, str):
        shortcut = _shortcut_levels(tiers, spec)
        if shortcut:
            return shortcut
        rate = _rate(spec)
        return _plain_rate(primary, tiers, rate) if rate is not None else None
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        try:
            rate = max(0, min(100, int(spec)))
        except (TypeError, ValueError):
            return None
        return _plain_rate(primary, tiers, rate)
    if not isinstance(spec, dict):
        return None
    src = spec.get("hit_rate", None)
    if src is None:
        src = {k: v for k, v in spec.items() if k != "hit_rate"}
    if isinstance(src, (int, float)) and not isinstance(src, bool):
        src = {primary: src}
    if not isinstance(src, dict):
        return None
    out = {}
    for key, value in src.items():
        tier = _tier_tag(key)
        if tier is None or tier not in tiers:
            continue
        rate = _rate(value)
        if rate is not None:
            out[tier] = rate
    return out or None


def resolve_mem_cache(config):
    out = {}
    if not isinstance(config, dict):
        return out
    for key, tiers in _CACHE_TIERS_BY_KEY.items():
        for cache in _collect_cache_dicts(config, keys=(key,)):
            for name, entry in cache.items():
                if name in ("mem", "memory"):
                    raise ValueError(
                        f"unknown cache key 'mem'; use {key!r} directly "
                        f"(e.g. {{{key!r}: {{'0x1000': 'hot'}}}})"
                    )
                error = _tier_key_error(key, name, tiers)
                if error is not None:
                    raise error
                tag = _tier_tag(name)
                if tag is not None:
                    _, addrs = _split_default_addrs(entry)
                    for addr, spec in addrs.items():
                        rate = _entry_rate(tag, spec)
                        if rate is not None:
                            _merge_mem_cache(out, addr, {tag: rate})
                    for reg, spec in _split_default_regs(entry).items():
                        rate = _entry_rate(tag, spec)
                        if rate is not None:
                            _merge_mem_cache(out, reg, {tag: rate})
                    continue
                if str(name) == "hit_rate":
                    continue
                s = str(name).strip()
                if _is_addr_key(s):
                    try:
                        _merge_mem_cache(
                            out, int(s, 0), normalize_cache_spec(entry, tiers)
                        )
                    except (TypeError, ValueError):
                        continue
                elif _is_reg_key(s):
                    try:
                        canon = canonical_data_reg(s)
                    except Exception:
                        canon = s.lower()
                    _merge_mem_cache(out, canon, normalize_cache_spec(entry, tiers))
    return out


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


@functools.lru_cache(maxsize=4096)
def cldemote_asm(address_reg):
    n = str(address_reg).strip().lower()
    if n not in _REG_CODES:
        raise ValueError(f"unknown register {address_reg!r} for cldemote")
    code, ext = _REG_CODES[n]
    prefix = "0x41, " if ext else ""
    low = code & 0x7
    if low == 4:
        return f".byte {prefix}0x0F, 0x1C, 0x04, 0x24"
    if low == 5:
        return f".byte {prefix}0x0F, 0x1C, 0x45, 0x00"
    return f".byte {prefix}0x0F, 0x1C, 0x{low:02x}"


def settle_asm(n=_SETTLE_PAUSES):
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return ""
    return "\n".join(["pause"] * n)


def page_runs(addresses, merge_gap=_TLB_MERGE_GAP, max_pages=_TLB_RUN_PAGES, avoid=()):
    pages = sorted({int(a) & ~(_PAGE_SIZE - 1) for a in addresses or ()})
    runs = _merge_pages(pages, merge_gap, max_pages)
    if not avoid:
        return runs
    blocked = sorted({int(a) & ~(_PAGE_SIZE - 1) for a in avoid})
    return [part for run in runs for part in _split_run(run, blocked) if part]


def tlb_restore_asm(meta, data_ptr=None, tlb=None):
    if not meta or not (meta.get("mem_addrs") or meta.get("l1i_addrs")):
        return ""
    tlb = tlb if tlb is not None else (meta.get("tlb_levels") or {})
    l1i_set = set(meta.get("l1i_addrs") or [])
    data_addrs = [a for a in meta.get("mem_addrs") or [] if a not in l1i_set]
    l1i_levels = {}
    for a in l1i_set:
        spec = (meta.get("mem_levels") or {}).get(a)
        spec = (
            {k: v for k, v in spec.items() if k == "L1i" or k in _TLB_TIERS}
            if isinstance(spec, dict)
            else {}
        )
        l1i_levels[a] = {"L1i": 100, **spec}
    blocked = list(meta.get("tlb_avoid") or ())
    parts = []
    for addresses, levels, extra, avoid in (
        (data_addrs, meta.get("mem_levels"), {"TLBd": tlb.get("TLBd")}, l1i_set),
        (list(l1i_set), l1i_levels, {"TLBi": tlb.get("TLBi")}, data_addrs),
    ):
        groups = _evict_groups(addresses, lambda _tier: 100, levels, extra)
        for tier in _TLB_TIERS:
            block = _tlb_block(groups[tier], _PROT_RWX, False, (*avoid, *blocked))
            if block:
                parts.append(block)
    return "\n".join(parts)


def evict(
    addresses,
    levels=None,
    reg="r12",
    settle=_SETTLE_PAUSES,
    cldemote=None,
    mem_levels=None,
    tlb_levels=None,
    avoid=(),
):
    if not addresses:
        return ""
    levels = levels or {"L1d": 100, "L2": 100, "L3": 100}
    mem_levels = mem_levels or {}
    tlb_levels = tlb_levels or {}

    def _rate(tier):
        try:
            raw = int(_tier_rate(levels, tier) or 0)
        except (TypeError, ValueError):
            return 0
        return max(0, min(100, raw))

    groups = _evict_groups(addresses, _rate, mem_levels, tlb_levels)

    use_cldemote = bool(has_cldemote()) if cldemote is None else bool(cldemote)
    lines = []
    for a in groups["L1d"]:
        lines.append(f"mov rax, [{imm(a)}]")
        lines.append("mfence")
    for a in groups["L1i"]:
        lines.append(f"mov {reg}, {imm(a)}")
        lines.append(f"clflushopt [{reg}]")
        lines.append("mfence")
    for tier, prot in (("TLBi", _PROT_RX), ("TLBd", _PROT_RW)):
        block = _tlb_block(groups[tier], prot, True, avoid)
        if block:
            lines.append(block)
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


def steer_asm(
    meta, data_ptr, settle=_SETTLE_PAUSES, mem_levels=None, write_values=True, tlb=None
):
    if not meta or not (meta.get("mem_addrs") or meta.get("l1i_addrs")):
        return ""
    if mem_levels is None:
        mem_levels = meta.get("mem_levels")
    tlb = tlb if tlb is not None else (meta.get("tlb_levels") or {})
    l1i_addrs = meta.get("l1i_addrs") or []
    l1i_set = set(l1i_addrs)
    data_addrs = [a for a in meta.get("mem_addrs") or [] if a not in l1i_set]
    blocked = list(meta.get("tlb_avoid") or ())
    parts = []
    if meta.get("mem_addrs"):
        parts.extend(
            p
            for p in (
                _value_write_loop(meta, data_ptr) if write_values else None,
                evict(
                    data_addrs,
                    meta.get("levels"),
                    settle=settle,
                    mem_levels=mem_levels,
                    tlb_levels={"TLBd": (tlb or {}).get("TLBd")},
                    avoid=(*l1i_addrs, *blocked),
                ),
            )
            if p
        )
    if l1i_addrs:
        l1i_levels = {}
        for a in l1i_addrs:
            spec = (mem_levels or {}).get(a)
            spec = (
                {k: v for k, v in spec.items() if k == "L1i" or k in _TLB_TIERS}
                if isinstance(spec, dict)
                else {}
            )
            l1i_levels[a] = {"L1i": 100, **spec}
        p = evict(
            l1i_addrs,
            None,
            settle=settle,
            mem_levels=l1i_levels,
            tlb_levels={"TLBi": (tlb or {}).get("TLBi")},
            avoid=(*data_addrs, *blocked),
        )
        if p:
            parts.append(p)
    return "\n".join(parts)


def prime_asm(meta, data_ptr):
    if not meta or not meta.get("reg_col"):
        return ""
    K = meta["K"]
    base = data_ptr or 0
    lines = ["push r14"]
    for NAME, c in meta["reg_col"].items():
        col_base = base + c * K * 8
        lines.append(f"movabs r14, {imm(col_base)}")
        lines.append(f"mov {NAME}, [r14 + r8*8]")
    lines.append("pop r14")
    return "\n".join(lines)


def per_iter_asm(meta, data_ptr):
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


def timing_regs(events="duration_time", callee_saved=False, avoid=()):
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


def timed_guard(events="duration_time", callee_saved=False, avoid=()):
    regs = list(_HARNESS_LOOP_REGS)
    for reg in timing_regs(events, callee_saved=callee_saved, avoid=avoid):
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
        return f"mov ecx, {imm(idx)}\nlfence\nrdpmc\nlfence\nshl rdx, 32\nor rax, rdx"

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

    regs = timing_regs(events, callee_saved=callee_saved, avoid=avoid)

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


def patch_jmp(patch_addr, tramp_addr):
    rel = int(tramp_addr) - (int(patch_addr) + _PATCH_SIZE)
    if not -(2**31) <= rel < 2**31:
        raise ValueError(
            f"trampoline {tramp_addr:#x} out of rel32 range from {patch_addr:#x}"
        )
    return struct.pack("<Bi", _JMP_OP, rel)


def profile_ring_layout(n_events, slots=65536):
    capacity = 1
    while capacity < max(int(slots), 1):
        capacity <<= 1
    stride = 8 * (2 + max(int(n_events), 0))
    return {
        "magic": _TRACK_RING_MAGIC,
        "header_size": _TRACK_RING_HDR_SIZE,
        "head_off": _TRACK_RING_HEAD_OFF,
        "data_off": _TRACK_RING_DATA_OFF,
        "capacity": capacity,
        "stride": stride,
        "total": _TRACK_RING_DATA_OFF + capacity * stride,
        "mask": capacity - 1,
        "events_off": _TRACK_SLOT_EVENTS_OFF,
    }


def detour_patch_len(code, addr=0, need=_PATCH_SIZE):

    code = bytes(code or b"")
    if len(code) < int(need):
        raise _Undetourable(
            f"only {len(code)} bytes available at {int(addr):#x}, need {int(need)}"
        )
    md = disassembler()
    try:
        md.detail = True
    except Exception:
        pass
    insns = []
    try:
        raw = list(md.disasm(code, int(addr)))
    except Exception as ex:
        raise _Undetourable(f"cannot disassemble at {int(addr):#x}: {ex}") from ex
    total = 0
    for insn in raw:
        if insn.size <= 0:
            raise _Undetourable(f"zero-size insn at {int(addr):#x}")
        insns.append(insn)
        total += int(insn.size)
        if total >= int(need):
            break
    if total < int(need):
        raise _Undetourable(
            f"cannot cover {int(need)} bytes at {int(addr):#x} "
            f"(only {total} disassembled)"
        )
    return total, insns


def relocate_detour_bytes(orig_bytes, orig_addr, new_addr):

    orig_bytes = bytes(orig_bytes or b"")
    if not orig_bytes:
        return b""
    md = disassembler()
    try:
        md.detail = True
    except Exception:
        pass
    try:
        insns = list(md.disasm(orig_bytes, int(orig_addr)))
    except Exception as ex:
        raise _Undetourable(f"cannot disassemble at {int(orig_addr):#x}: {ex}") from ex
    out = bytearray()
    consumed = 0
    for insn in insns:
        if consumed + int(insn.size) > len(orig_bytes):
            break
        raw = bytes(orig_bytes[consumed : consumed + int(insn.size)])
        start_len = len(out)
        new_insn_addr = int(new_addr) + len(out)
        try:
            operands = list(insn.operands or ())
        except Exception:
            operands = []

        try:
            rip_rel = any(
                op.type == capstone.x86.X86_OP_MEM
                and getattr(op.mem, "base", 0) == capstone.x86.X86_REG_RIP
                for op in operands
            )
        except Exception:
            rip_rel = False
        if rip_rel:
            if int(insn.size) < 5:
                raise _Undetourable(f"short RIP-relative insn at {insn.address:#x}")
            try:
                disp = None
                for op in operands:
                    try:
                        if (
                            op.type == capstone.x86.X86_OP_MEM
                            and getattr(op.mem, "base", 0) == capstone.x86.X86_REG_RIP
                        ):
                            disp = int(op.mem.disp)
                            break
                    except Exception:
                        continue
                if disp is None:
                    raise _Undetourable("missing RIP disp")
            except _Undetourable:
                raise
            except Exception as ex:
                raise _Undetourable(f"bad RIP disp at {insn.address:#x}: {ex}") from ex
            old_target = int(insn.address) + int(insn.size) + int(disp)
            new_disp = int(old_target) - (int(new_insn_addr) + int(insn.size))
            buf = bytearray(raw)
            try:
                buf[len(buf) - 4 :] = struct.pack("<i", new_disp)
            except struct.error as ex:
                raise _Undetourable(
                    f"RIP target out of range at {insn.address:#x}"
                ) from ex
            out += bytes(buf)
            consumed += int(insn.size)
            if consumed >= len(orig_bytes):
                break
            continue

        try:
            groups = set(insn.groups or ())
        except Exception:
            groups = set()
        if capstone.x86.X86_GRP_CALL in groups or capstone.x86.X86_GRP_JUMP in groups:
            for op in operands:
                try:
                    direct = op.type == capstone.x86.X86_OP_IMM
                except Exception:
                    continue
                if not direct:
                    continue
                try:
                    tgt = int(op.imm)
                except Exception:
                    raise _Undetourable(f"bad branch target at {insn.address:#x}")
                if int(insn.size) == 2:
                    new_rel8 = int(tgt) - (int(new_insn_addr) + 2)
                    if -128 <= new_rel8 <= 127:
                        buf = bytearray(raw)
                        buf[1] = new_rel8 & 0xFF
                        out += bytes(buf)
                    else:
                        new_rel32_len = 5 if raw[0] == 0xEB else 6
                        new_rel = int(tgt) - (int(new_insn_addr) + new_rel32_len)
                        try:
                            out += _expand_short_branch(raw, new_rel)
                        except struct.error as ex:
                            raise _Undetourable(
                                f"branch at {insn.address:#x} out of range"
                            ) from ex
                elif int(insn.size) in (5, 6):
                    new_rel = int(tgt) - (int(new_insn_addr) + int(insn.size))
                    buf = bytearray(raw)
                    try:
                        buf[len(buf) - 4 :] = struct.pack("<i", new_rel)
                    except struct.error as ex:
                        raise _Undetourable(
                            f"branch at {insn.address:#x} out of range"
                        ) from ex
                    out += bytes(buf)
                else:
                    if int(insn.size) not in (2, 5, 6):
                        m = str(insn.mnemonic or "").strip().lower()
                        if m in ("call", "jmp") or (m.startswith("j") and m != "jmp"):
                            raise _Undetourable(
                                f"unsupported branch size {insn.size} "
                                f"at {insn.address:#x}"
                            )
                break
        if len(out) == start_len:
            out += bytes(raw)
        consumed += int(insn.size)
        if consumed >= len(orig_bytes):
            break
    return bytes(out)


def build_profile_detour(
    point_id,
    events,
    indices,
    buf_base,
    tramp_addr,
    ret_addr,
    layout=None,
    orig_bytes=b"",
    save_flags=True,
):

    events = list(events or [])
    indices = list(indices or [])
    layout = dict(layout or profile_ring_layout(len(events)))
    tramp_addr, ret_addr = int(tramp_addr), int(ret_addr)
    fixed = bytes(orig_bytes or b"")
    prefix = bytes(
        assemble(
            "\n".join(
                _detour_prefix_lines(
                    point_id, events, indices, buf_base, layout, save_flags
                )
            ),
            tramp_addr,
        )
    )
    exec_addr = tramp_addr + len(prefix)
    jmp = _assemble_jmp(exec_addr + len(fixed), ret_addr)
    return bytes(prefix) + bytes(fixed) + bytes(jmp)


def build_profile_detour_raw(
    point_id,
    events,
    indices,
    buf_base,
    tramp_addr,
    patch_addr,
    patch_len,
    live_bytes,
    layout=None,
    save_flags=True,
):

    layout = dict(layout or profile_ring_layout(len(list(events or []))))
    tramp_addr, patch_addr, patch_len = (
        int(tramp_addr),
        int(patch_addr),
        int(patch_len),
    )
    exec_addr = tramp_addr + _prefix_len(
        _detour_prefix_lines(point_id, events, indices, buf_base, layout, save_flags),
        tramp_addr,
    )
    fixed = relocate_detour_bytes(bytes(live_bytes)[:patch_len], patch_addr, exec_addr)
    return build_profile_detour(
        point_id,
        events,
        indices,
        buf_base,
        tramp_addr,
        patch_addr + patch_len,
        layout,
        fixed,
        save_flags=save_flags,
    )


def build_profile_func_entry(
    point_id,
    events,
    indices,
    buf_base,
    tramp_addr,
    ret_addr,
    layout=None,
    orig_bytes=b"",
    shadow_top=0,
    exit_addr=0,
):

    events = list(events or [])
    indices = list(indices or [])
    layout = dict(layout or profile_ring_layout(len(events)))
    tramp_addr, ret_addr = int(tramp_addr), int(ret_addr)
    fixed = bytes(orig_bytes or b"")
    prefix = bytes(
        assemble(
            "\n".join(
                _func_entry_prefix_lines(
                    point_id, events, indices, buf_base, layout, shadow_top, exit_addr
                )
            ),
            tramp_addr,
        )
    )
    exec_addr = tramp_addr + len(prefix)
    jmp = _assemble_jmp(exec_addr + len(fixed), ret_addr)
    return bytes(prefix) + bytes(fixed) + bytes(jmp)


def build_profile_func_entry_raw(
    point_id,
    events,
    indices,
    buf_base,
    tramp_addr,
    patch_addr,
    patch_len,
    live_bytes,
    layout=None,
    shadow_top=0,
    exit_addr=0,
):
    layout = dict(layout or profile_ring_layout(len(list(events or []))))
    tramp_addr, patch_addr, patch_len = (
        int(tramp_addr),
        int(patch_addr),
        int(patch_len),
    )
    exec_addr = tramp_addr + _prefix_len(
        _func_entry_prefix_lines(
            point_id, events, indices, buf_base, layout, shadow_top, exit_addr
        ),
        tramp_addr,
    )
    fixed = relocate_detour_bytes(bytes(live_bytes)[:patch_len], patch_addr, exec_addr)
    return build_profile_func_entry(
        point_id,
        events,
        indices,
        buf_base,
        tramp_addr,
        patch_addr + patch_len,
        layout,
        fixed,
        shadow_top=shadow_top,
        exit_addr=exit_addr,
    )


def build_profile_func_exit(
    point_id, events, indices, buf_base, tramp_addr, layout=None, shadow_top=0
):

    events = list(events or [])
    indices = list(indices or [])
    layout = dict(layout or profile_ring_layout(len(events)))
    tramp_addr = int(tramp_addr)
    lines = _profile_record_lines(point_id, events, indices, buf_base, layout)
    lines += [
        "pop r11",
        "pop rdx",
        "pop rcx",
        "pop rax",
        "push rcx",
        "push r11",
        f"movabs rcx, 0x{int(shadow_top) & 0xFFFFFFFFFFFFFFFF:x}",
        "mov r11, [rcx]",
        "mov r11, [r11-8]",
        "sub qword ptr [rcx], 8",
        "mov [rsp], r11",
        "pop r11",
        "pop rcx",
        "jmp r11",
    ]
    return bytes(assemble("\n".join(lines), tramp_addr))


def is_branch_mnemonic(mnemonic):
    m = str(mnemonic or "").strip().lower()
    return m in ("call", "jmp") or m.startswith("j") or m.startswith("loop")


def is_return_mnemonic(mnemonic):
    m = str(mnemonic or "").strip().lower()
    return m in ("ret", "retf", "retq", "iret", "iretq")


def branch_imm_targets(insn):
    if not is_branch_mnemonic(insn.mnemonic):
        return ()
    out = []
    try:
        operands = insn.operands
    except Exception:
        return ()
    for op in operands:
        try:
            if op.type == capstone.x86.X86_OP_IMM:
                out.append(int(op.imm))
        except Exception:
            continue
    return out


def mem_refs(insn):
    out = []
    try:
        operands = insn.operands
    except Exception:
        return ()
    for op in operands:
        try:
            if op.type != capstone.x86.X86_OP_MEM:
                continue
            base, index, disp = op.mem.base, op.mem.index, int(op.mem.disp)
        except Exception:
            continue
        if base == capstone.x86.X86_REG_RIP:
            out.append(int(insn.address) + int(insn.size) + disp)
        elif not base and not index:
            out.append(disp & 0xFFFFFFFFFFFFFFFF)
    return tuple(out)


def branch_label(addr):
    return f".L{int(addr):x}"


def cond_branch_analysis(insns):
    insn_list = list(insns or ())
    jcc = None
    for insn in insn_list:
        m = str(insn.mnemonic).strip().lower()
        if m.startswith("j") and m != "jmp":
            jcc = insn
    if jcc is None or not insn_list:
        return set(), []

    def _written(insn):
        out = set()
        try:
            ops = insn.operands or ()
        except Exception:
            ops = ()
        for op in ops:
            try:
                is_write = bool(op.access & capstone.CS_AC_WRITE)
                is_read = op.type == capstone.x86.X86_OP_REG
            except Exception:
                continue
            if is_write and is_read:
                try:
                    nm = insn.reg_name(op.reg)
                except Exception:
                    nm = None
                if nm:
                    out.add(_canonical_reg(nm))
        return out

    def _read_regs(insn):
        regs = set()
        mems = []
        try:
            ops = insn.operands or ()
        except Exception:
            ops = ()
        for op in ops:
            try:
                otype = op.type
                acc = op.access
            except Exception:
                otype, acc = None, None
            if otype == capstone.x86.X86_OP_MEM:
                base = index = 0
                scale, disp = 1, 0
                try:
                    base, index = op.mem.base, op.mem.index
                    scale, disp = op.mem.scale, op.mem.disp
                except Exception:
                    pass
                for code in (base, index):
                    if code:
                        try:
                            nm = insn.reg_name(code)
                        except Exception:
                            nm = None
                        if nm:
                            regs.add(_canonical_reg(nm))
                mems.append((base, index, scale, disp))
            elif otype == capstone.x86.X86_OP_REG:
                try:
                    nm = insn.reg_name(op.reg)
                    reads = acc is None or bool(acc & capstone.CS_AC_READ)
                except Exception:
                    nm, reads = None, True
                if nm and reads:
                    regs.add(_canonical_reg(nm))
        return regs, mems

    flag_idx = None
    for i in range(len(insn_list) - 1, -1, -1):
        if insn_list[i] is jcc:
            continue
        if insn_list[i].mnemonic in _FLAG_WRITERS:
            flag_idx = i
            break
    regs = set()
    mems = []
    if flag_idx is None:
        for insn in insn_list:
            r, m = _read_regs(insn)
            regs |= r
            mems.extend(m)
        return regs, mems
    r, m = _read_regs(insn_list[flag_idx])
    regs |= r
    mems.extend(m)
    need = set(r)
    for j in range(flag_idx - 1, -1, -1):
        if not need:
            break
        insn = insn_list[j]
        if not (_written(insn) & need):
            continue
        r2, m2 = _read_regs(insn)
        regs |= r2
        mems.extend(m2)
        need |= r2
    return regs, mems


class Unrelocatable(ValueError):
    pass


def call_native_asm(addr):
    return "; ".join(
        (
            *_NATIVE_ARG_SCRATCH,
            f"mov r11, {int(addr):#x}",
            "sub rsp, 8",
            "call r11",
            "add rsp, 8",
            "ret",
        )
    )


def call_buffered(code_addr):
    fn = ctypes.CFUNCTYPE(_NATIVE_RET_TYPE)(int(code_addr))
    return int(fn())


def pack_pointer(value):
    return struct.pack("<Q", int(value) & _ADDR_MASK)


def unpack_pointer(data, at=0):
    return int.from_bytes(bytes(data)[int(at) : int(at) + _POINTER_SIZE], "little")


def process_stack_top(base, size, align=_STACK_ENTRY_ALIGN):
    align = int(align)
    return ((int(base) + int(size)) & ~(align - 1)) - _STACK_ENTRY_ALIGN


def entry_sp(top, gap=0):
    return ((int(top) - int(gap) + _POINTER_SIZE) & ~(_STACK_ENTRY_ALIGN - 1)) | (
        _POINTER_SIZE
    )


def movabs64_bytes(blob):
    blob = bytes(blob or b"")
    i = blob.find(_MOV_IMM32)
    if i < 0 or i + 7 > len(blob):
        return blob
    imm32 = struct.unpack("<i", blob[i + 3 : i + 7])[0]
    return blob[:i] + _MOVABS + struct.pack("<q", imm32) + blob[i + 7 :]


def clear_movabs(blob, start=0, end=None):
    blob = bytes(blob or b"")
    end = len(blob) if end is None else int(end)
    at = blob.find(_MOVABS, int(start), end)
    if at < 0 or at + _MOVABS_LEN > end:
        return blob, -1
    at += _MOVABS_IMM_OFF
    out = bytearray(blob)
    out[at : at + _POINTER_SIZE] = b"\x00" * _POINTER_SIZE
    return bytes(out), at


def layout_hazard(code, in_image, pie=True, md=None):
    md = md or disassembler()
    try:
        md.detail = True
    except Exception:
        pass
    for insn in md.disasm(bytes(code or b""), 0):
        if insn.bytes[:1] in _ABS_MOV_OPCODES:
            return True
        try:
            groups = set(insn.groups)
        except Exception:
            groups = set()
        if capstone.x86.X86_GRP_CALL in groups or capstone.x86.X86_GRP_JUMP in groups:
            continue
        try:
            operands = insn.operands
        except Exception:
            continue
        for op in operands:
            try:
                is_imm = op.type == capstone.x86.X86_OP_IMM
            except Exception:
                continue
            if not is_imm:
                continue
            try:
                size = int(op.size)
                val = int(op.imm) & _ADDR_MASK
            except Exception:
                continue
            if (size >= _POINTER_SIZE or not pie) and size >= 4 and in_image(val):
                return True
    return False


def relocate_code(code, base, orig_base, resolve, span=None, md=None):
    md = md or disassembler()
    try:
        md.detail = True
    except Exception:
        pass
    base = int(base)
    orig_base = int(orig_base)
    fs, fe = span if span else (0, 0)
    out = bytearray(code or b"")
    for insn in md.disasm(bytes(code or b""), orig_base):
        off = insn.address - orig_base
        try:
            groups = set(insn.groups)
            operands = list(insn.operands)
        except Exception:
            raise Unrelocatable()
        for op in operands:
            try:
                is_mem = op.type == capstone.x86.X86_OP_MEM
                rip_base = op.mem.base == capstone.x86.X86_REG_RIP
            except Exception:
                continue
            if not (is_mem and rip_base):
                continue
            if insn.size < 5 or off + insn.size > len(out):
                raise Unrelocatable()
            disp = struct.unpack(
                "<i", bytes(out[off + insn.size - 4 : off + insn.size])
            )[0]
            try:
                new_tgt = resolve(insn.address + insn.size + disp - orig_base + fs)
            except ValueError as e:
                raise Unrelocatable() from e
            try:
                out[off + insn.size - 4 : off + insn.size] = struct.pack(
                    "<i", new_tgt - (base + off + insn.size)
                )
            except struct.error as e:
                raise Unrelocatable() from e
        if capstone.x86.X86_GRP_CALL in groups or capstone.x86.X86_GRP_JUMP in groups:
            if not operands:
                continue
            try:
                direct = operands[0].type == capstone.x86.X86_OP_IMM
            except Exception:
                continue
            if not direct:
                continue
            try:
                tgt_rt = int(operands[0].imm)
            except Exception:
                raise Unrelocatable()
            tgt_file = tgt_rt - orig_base + fs
            if not fs <= tgt_file < fe:
                try:
                    new_tgt = resolve(tgt_file)
                except ValueError as e:
                    raise Unrelocatable() from e
                new_disp = new_tgt - (base + off + insn.size)
                if insn.size == 2:
                    if not -128 <= new_disp <= 127:
                        raise Unrelocatable()
                    out[off + 1] = new_disp & 0xFF
                elif insn.size in (5, 6):
                    try:
                        out[off + insn.size - 4 : off + insn.size] = struct.pack(
                            "<i", new_disp
                        )
                    except struct.error as e:
                        raise Unrelocatable() from e
                else:
                    raise Unrelocatable()
    return bytes(out)


def _pte_protection(prot):
    return (int(prot) & _PROT_W, not int(prot) & _PROT_X)


@functools.lru_cache(maxsize=4096)
def _mprotect_asm(page, pages, prot, toggle, prologue):
    try:
        page = int(page) & ~(_PAGE_SIZE - 1)
        pages = max(int(pages), 1)
    except (TypeError, ValueError):
        pages = 1
    body = [
        f"mov rdi, {imm(page)}",
        f"mov rsi, {imm(pages * _PAGE_SIZE)}",
        f"mov rdx, 0x{_PROT_RWX:x}",
    ]
    if toggle:
        delta = _PROT_RWX ^ int(prot)
        if (
            not delta
            or delta & (delta - 1)
            or _pte_protection(_PROT_RWX) == _pte_protection(prot)
        ):
            raise ValueError(
                f"protection 0x{int(prot):x} cannot be alternated with "
                f"0x{_PROT_RWX:x}: an mprotect that leaves the pte unchanged is "
                f"a no-op in the kernel, so the translation is never dropped"
            )
        body += [
            "mov rax, r8",
            "and rax, 1",
            f"shl rax, {delta.bit_length() - 1}",
            "sub rdx, rax",
        ]
    body += [f"mov rax, {_NR_MPROTECT}", "syscall"]
    if not prologue:
        return "\n".join(body)
    return "\n".join((*_TLB_SAVE, *body, *_TLB_RESTORE))


def _merge_pages(pages, merge_gap, max_pages):
    runs = []
    for page in pages:
        count = (page - runs[-1][0]) // _PAGE_SIZE + 1 if runs else 1
        if runs and count <= runs[-1][1] + merge_gap and count <= max_pages:
            runs[-1][1] = count
        else:
            runs.append([page, 1])
    return [(page, count) for page, count in runs]


def _split_run(run, blocked):
    start, count = run
    stop = start + count * _PAGE_SIZE
    out, low = [], start
    for page in blocked:
        if page < low or page >= stop:
            continue
        if page > low:
            out.append((low, (page - low) // _PAGE_SIZE))
        low = page + _PAGE_SIZE
    if low < stop:
        out.append((low, (stop - low) // _PAGE_SIZE))
    return out


def _tlb_block(addresses, prot, invalidate, avoid=()):
    runs = page_runs(addresses, avoid=avoid)
    if not runs:
        return ""
    if invalidate:
        body = [tlb_inval_asm(page, count, prot, False) for page, count in runs]
    else:
        body = [tlb_restore_call(page, count, False) for page, count in runs]
    return "\n".join((*_TLB_SAVE, *body, *_TLB_RESTORE))


def _evict_groups(addresses, rate, mem_levels=None, tlb_levels=None):
    mem_levels = mem_levels or {}
    tlb_levels = tlb_levels or {}
    groups = {
        "L1d": [],
        "L1i": [],
        "L2": [],
        "L3": [],
        "TLBd": [],
        "TLBi": [],
        "DRAM": [],
    }
    rest = []
    for a in addresses:
        spec = mem_levels.get(a)
        spec = spec if isinstance(spec, dict) else {}
        for tier in _TLB_TIERS:
            value = spec.get(tier, tlb_levels.get(tier, None))
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                groups[tier].append(a)
        data_spec = {k: v for k, v in spec.items() if k in _DATA_TIERS}
        if data_spec:
            groups[_cache_tier(data_spec)].append(a)
        elif spec.get("L1i") is not None:
            try:
                l1i = int(_tier_rate(spec, "L1i"))
            except (TypeError, ValueError):
                l1i = 0
            if l1i <= 0:
                groups["L1i"].append(a)
        else:
            rest.append(a)
    remaining = len(rest)
    quotas = {"L1d": int(remaining * rate("L1d") / 100.0)}
    remaining -= quotas["L1d"]
    quotas["L2"] = int(remaining * rate("L2") / 100.0)
    remaining -= quotas["L2"]
    quotas["L3"] = int(remaining * rate("L3") / 100.0)
    remaining -= quotas["L3"]
    quotas["DRAM"] = remaining
    order = ("L1d", "L2", "L3", "DRAM")
    idx = 0
    for a in rest:
        for _ in range(len(order)):
            tier = order[idx % len(order)]
            idx += 1
            if quotas[tier] > 0:
                groups[tier].append(a)
                quotas[tier] -= 1
                break
    return groups


def _detour_prefix_lines(point_id, events, indices, buf_base, layout, save_flags=True):
    lines = ["pushfq"] if save_flags else []
    lines += _profile_record_lines(point_id, events, indices, buf_base, layout)
    lines += ["pop r11", "pop rdx", "pop rcx", "pop rax"]
    if save_flags:
        lines += ["popfq"]
    return lines


def _prefix_len(lines, tramp_addr):
    return len(assemble("\n".join(lines), int(tramp_addr)))


def _func_entry_prefix_lines(
    point_id, events, indices, buf_base, layout, shadow_top, exit_addr
):
    lines = _profile_record_lines(point_id, events, indices, buf_base, layout)
    lines += ["pop r11", "pop rdx", "pop rcx", "pop rax"]
    lines += [
        "push rax",
        "push rcx",
        "push r11",
        "mov rax, [rsp+24]",
        f"movabs rcx, 0x{int(shadow_top) & 0xFFFFFFFFFFFFFFFF:x}",
        "mov r11, [rcx]",
        "mov [r11], rax",
        "lea r11, [r11+8]",
        "mov [rcx], r11",
        f"movabs r11, 0x{int(exit_addr) & 0xFFFFFFFFFFFFFFFF:x}",
        "mov [rsp+24], r11",
        "pop r11",
        "pop rcx",
        "pop rax",
    ]
    return lines


def _tier_tag(key):
    try:
        s = str(key).strip()
    except Exception:
        return None
    return _tier_tag_cached(s)


@functools.lru_cache(maxsize=64)
def _tier_tag_cached(s):
    hit = _CACHE_TIER_TAGS.get(s)
    if hit is not None:
        return hit
    low = s.lower()
    for k, v in _CACHE_TIER_TAGS.items():
        if k.lower() == low:
            return v
    return None


def _tier_family(tier):
    if tier in _DATA_TIERS or tier == "DRAM":
        return _DATA_TIERS
    return _CACHE_TIERS_BY_KEY_TIER.get(tier, (tier,))


def _rate(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0, min(100, int(value)))
    if isinstance(value, str):
        m = re.fullmatch(r"hit_rate\s*[:=]\s*(-?\d+)\s*", value.strip())
        if m:
            return max(0, min(100, int(m.group(1))))
    return None


def _shortcut_levels(tiers, name):
    if not isinstance(name, str):
        return None
    if tuple(tiers) == _DATA_TIERS:
        levels = _memory_shortcut(name)
        return dict(levels) if levels else None
    shortcut = _tier_shortcut(name)
    if not shortcut:
        return None
    return {tier: shortcut[tier] for tier in tiers}


def _tier_key_error(key, name, tiers):
    if len(tiers) > 1 and _tier_tag(name) in tiers:
        return None
    tag = _tier_tag(name)
    if tag is None:
        return None
    if len(tiers) == 1:
        wanted = f"'hit_rate' (the only tier of {key})"
    else:
        wanted = ", ".join(tiers)
    return ValueError(
        f"{tag!r} is not a {key} key; use {wanted} "
        f"(e.g. {{'{key}': {{'hit_rate': 50}}}}) or a shortcut like 'hot'/'cold'"
    )


def _tier_levels(config, key, tiers):
    spec = (config or {}).get(key)
    if spec is None:
        return None
    tier = tiers[0]
    if isinstance(spec, str):
        shortcut = _shortcut_levels(tiers, spec)
        if shortcut is None:
            raise ValueError(
                f"unknown cache shortcut {spec!r} for {key}; "
                "expected one of 'hot', 'warm', 'cool', 'cold'"
            )
        return shortcut
    if not isinstance(spec, dict):
        rate = _rate(spec)
        return {tier: rate} if rate is not None else None

    out = {t: None for t in tiers}
    for name, value in spec.items():
        error = _tier_key_error(key, name, tiers)
        if error is not None:
            raise error
        tag = _tier_tag(name)
        if tag is not None:
            rate = _entry_rate(tag, value)
            if rate is not None:
                out[tag] = rate
            continue
        if str(name) == "hit_rate":
            rate = _rate(value)
            if rate is not None:
                out[tier] = rate
    if all(v is None for v in out.values()):
        return None
    return out


@functools.lru_cache(maxsize=64)
def _tier_shortcut(NAME):
    if not isinstance(NAME, str):
        return None
    return _TIER_SHORTCUTS.get(NAME.strip().lower())


def _plain_rate(primary, tiers, rate):
    return {tier: rate if tier == primary else 0 for tier in tiers}


def _cache_tier(levels):
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


@functools.lru_cache(maxsize=4096)
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


@functools.lru_cache(maxsize=4096)
def _is_reg_key(key):
    s = str(key).strip()
    if not s or s == "default":
        return False
    if _is_addr_key(s):
        return False
    if _tier_tag(s) is not None:
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
    spec, _addrs = _split_default_addrs(spec)
    if spec is None:
        return None
    if isinstance(spec, str):
        s = spec.strip()
        if not s:
            return None
        rate = _rate(s)
        if rate is not None:
            return rate
        levels = _shortcut_levels(_tier_family(tier), s)
        return levels.get(tier) if levels is not None else None
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return max(0, min(100, int(spec)))
    if not isinstance(spec, dict):
        return None
    src = spec.get("hit_rate")
    rate = _rate(src)
    if rate is not None:
        return rate
    flat = src if isinstance(src, dict) else spec
    if not isinstance(flat, dict):
        return None
    parsed = normalize_cache_spec(flat, _tier_family(tier))
    return parsed.get(tier) if parsed else None


def _tier_rate(levels, tier):
    return levels.get(tier) if isinstance(levels, dict) else None


@functools.lru_cache(maxsize=4096)
def _memory_shortcut(NAME):
    if not isinstance(NAME, str):
        return None
    return _MEMORY_SHORTCUTS.get(NAME.strip().lower())


def _collect_cache_dicts(config, keys=("dcache", "icache", "dtlb", "itlb")):
    out = []
    if not isinstance(config, dict):
        return out
    for key in keys:
        value = config.get(key)
        if isinstance(value, dict):
            out.append(value)
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
    push r12
    push r13
    push r14
    push r15
    xor r15, r15
.perfwrite:
    movabs r14, {imm(t_addr)}
    mov r12, [r14 + r15*8]
    movabs r14, {imm(t_vb)}
    mov r14, [r14 + r15*8]
    mov r13, [r14 + r8*8]
    mov [r12], r13
.perfwritedone:
    inc r15
    cmp r15, {imm(n)}
    jne .perfwrite
    pop r15
    pop r14
    pop r13
    pop r12
"""


def _canonical_reg(NAME):
    n = str(NAME).strip().lower()
    return _SUBREG_TO_FULL.get(n, n)


def _is_pow2(n):
    try:
        n = int(n)
    except (TypeError, ValueError):
        return False
    return n > 0 and (n & (n - 1)) == 0


def _scale_slot_lines(stride):
    stride = int(stride)
    if stride == 1:
        return []
    if _is_pow2(stride):
        return [f"shl rcx, {stride.bit_length() - 1}"]
    if stride % 8 == 0:
        c = stride // 8
        lea = {3: "lea rcx, [rcx+rcx*2]", 5: "lea rcx, [rcx+rcx*4]"}
        if c in lea:
            return [lea[c], "shl rcx, 3"]
        if c == 9:
            return ["lea rcx, [rcx+rcx*8]", "shl rcx, 3"]
    return [f"imul rcx, rcx, {stride}"]


def _profile_record_lines(point_id, events, indices, buf_base, layout):
    events = list(events or [])
    indices = list(indices or [])
    layout = dict(layout or profile_ring_layout(len(events)))
    cap_mask, stride, data_off, head_off = (
        layout["mask"],
        layout["stride"],
        layout["data_off"],
        layout["head_off"],
    )
    mid = int(point_id) & 0xFFFFFFFF
    lines = [
        "push rax",
        "push rcx",
        "push rdx",
        "push r11",
        f"mov r11, 0x{int(buf_base) & 0xFFFFFFFFFFFFFFFF:x}",
        f"mov rax, [r11 + {int(head_off)}]",
        "mov rcx, rax",
        f"and rcx, 0x{int(cap_mask) & 0xFFFFFFFFFFFFFFFF:x}",
    ]
    lines += _scale_slot_lines(stride)
    lines += [
        "lea rdx, [rax+1]",
        f"mov [r11 + {int(head_off)}], rdx",
        f"lea r11, [r11 + rcx + {int(data_off)}]",
        f"mov dword ptr [r11], 0x{mid:x}",
        "mov qword ptr [r11+8], rax",
    ]
    for i, ev in enumerate(events):
        off = _TRACK_SLOT_EVENTS_OFF + 8 * i
        lines.append("lfence")
        if str(ev).strip() == "duration_time":
            lines.append("rdtsc")
        else:
            idx = indices[i] if i < len(indices) else None
            if idx is None:
                raise ValueError(f"event {ev!r} requires an rdpmc counter index")
            lines.append(f"mov ecx, 0x{int(idx) & 0xFFFFFFFF:x}")
            lines.append("rdpmc")
        lines.append("shl rdx, 32")
        lines.append("or rax, rdx")
        lines.append(f"mov [r11+{off}], rax")
    return lines


def _assemble_jmp(from_addr, to_addr):
    return bytes(
        assemble(f"jmp 0x{int(to_addr) & 0xFFFFFFFFFFFFFFFF:x}", int(from_addr))
    )


class _Undetourable(ValueError):
    pass


def _expand_short_branch(insn_bytes, new_rel):

    if len(insn_bytes) < 1:
        raise _Undetourable("empty branch insn")
    op0 = insn_bytes[0]
    if op0 == 0xEB:
        return b"\xe9" + struct.pack("<i", new_rel)
    if 0x70 <= op0 <= 0x7F:
        return b"\x0f" + bytes([0x80 + (op0 - 0x70)]) + struct.pack("<i", new_rel)
    raise _Undetourable(
        f"short branch opcode 0x{op0:02x} has no rel32 form "
        "(loop/jcxz cannot be relocated far)"
    )


_CACHE_TIERS_BY_KEY_TIER["DRAM"] = _DATA_TIERS
