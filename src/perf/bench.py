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
import ctypes.util
import datetime
import functools
import hashlib
import json
import mmap
import os
import random
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import core
from .arch import arch as get_arch
from .arch import load as load_arch
from .core import (
    _DURATION_TIME,
    _THREAD_HINT,
    _affinity_guard,
    _affinity_spec,
    _current_priority_value,
    _func_prototype,
    _is_mem_addr_key,
    _numa_guard,
    _numa_spec,
    _parse_addr_key,
    _parse_affinity,
    _parse_numa,
    _parse_priority,
    _priority_guard,
    _priority_spec,
    _split_list,
    _text,
    _thread_cfg,
    _thread_entries,
    _thread_store,
    _to_int_or,
    _to_u64,
)
from .data import _IDENTITY_COLUMNS, nest
from .exec import (
    _MAP_FIXED_NOREPLACE,
    Elf,
    resolve_exec,
    write_perf_map,
)
from .info import (
    _CPUINFO_FIELDS,
    asm_labels,
    cpuinfo,
    functions,
    hex_addresses,
    metadata,
)
from .info import is_asm_source as _is_asm_source
from .info import targets as resolve_targets

_ASM_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*|#[^\n]*", re.S)
_ASM_LABEL = re.compile(r"[.\w$@]+")

_SCALAR_KEYS = frozenset(
    {
        "samples",
        "iterations",
        "backend",
        "seed",
    }
)
_DEFAULT_ALIGN = 16
_DEFAULT_STACK_SIZE = 0x200000
_DEFAULT_ITER_MIN = 100
_DEFAULT_ITER_MAX = 1_000_000
_FUNC_ORDERS = ("as-is", "random")
_DEFAULT_BENCH = {
    "seed": 0,
    "thread": [
        [
            {"numa": None, "affinity": None, "priority": "normal"},
        ]
    ],
    "branch": ["predictable", "unpredictable"],
    "dcache": ["hot", "warm", "cool", "cold"],
    "dtlb": ["hot", "cold"],
    "icache": ["hot", "cold"],
    "itlb": ["hot", "cold"],
    "code": [
        {"align": 1},
        {"align": 16},
    ],
    "stack": [
        {"size": _DEFAULT_STACK_SIZE, "align": _DEFAULT_ALIGN},
    ],
    "func": [
        {"align": _DEFAULT_ALIGN, "order": "as-is"},
    ],
    "samples": 100,
    "iterations": {
        "min": _DEFAULT_ITER_MIN,
        "max": _DEFAULT_ITER_MAX,
    },
    "backend": {
        "loop": {
            "probes": 3,
            "runs": 10,
            "target": 0.005,
        },
        "unroll": {
            "count": 5,
            "probes": 3,
            "runs": 10,
            "target": 0.005,
        },
    },
    "external": {
        "lib": False,
    },
}
_BACKENDS = ("loop", "unroll")
_MODES = ("latency", "throughput")
_JIT_PAGES = []
_JIT_PAGES_MAX = 32
_MAPPED_DATA_PAGES = set()
_UNMAPPABLE_DATA_PAGES = set()
_PROJECTS = {}
_ASM_SCRATCH_BASE = get_arch()._ASM_SCRATCH_BASE
_HEXDIGITS = set("0123456789abcdefABCDEF")
_VALID_TOP_KEYS = frozenset(_DEFAULT_BENCH.keys())
_VALID_FUNC_KEYS = frozenset({"align", "order"})
_VALID_CODE_KEYS = frozenset({"align"})
_VALID_STACK_KEYS = frozenset({"size", "align"})
_VALID_ITERATIONS_KEYS = frozenset({"min", "max"})
_VALID_EXTERNAL_KEYS = frozenset({"lib"})
_BACKEND_OPTION_KEYS = frozenset({"probes", "runs", "target", "count"})
_CONTAINER_TOPS = frozenset({"iterations", "external"})
_LIST_TOPS = frozenset({"code", "stack", "func"})
_THREAD_TOP = "thread"
_LIST_TOP_KEYS = {
    "code": _VALID_CODE_KEYS,
    "stack": _VALID_STACK_KEYS,
    "func": _VALID_FUNC_KEYS,
}
_LIST_TOP_HINT = {
    "code": '[{"align": 1}, {"align": 16}]',
    "stack": '[{"size": 2097152, "align": 16}]',
    "func": '[{"align": 16, "order": "as-is"}]',
}
_ENTRY_DEFAULTS = {
    "code": {"align": _DEFAULT_ALIGN},
    "stack": {"size": _DEFAULT_STACK_SIZE, "align": _DEFAULT_ALIGN},
    "func": {"align": _DEFAULT_ALIGN, "order": _FUNC_ORDERS[0]},
}
_CACHE_KEYS = ("dcache", "icache", "dtlb", "itlb")
_PRUNE_KEYS = ("branch", "dcache", "dtlb")
_BRANCH_JUMPKINDS = frozenset({"ijk_boring", "ijk_indirect"})
_BRANCH_CHOICES = (
    "predictable",
    "unpredictable",
)

_BENCH_KEEP_META = frozenset({"iterations", "samples", "operations"})
_COMBINE_KEYS = (*_IDENTITY_COLUMNS, "iterations", "samples", "operations")
_EXPLORE_CACHE = {}
_MODEL_CACHE = {}
_SOLVE_CACHE_MAX = 4
_RUNNABLE_HARNESS = set()
_CONTAINER_RR = {}
_FAULT_MESSAGES = {
    4: "an illegal instruction (SIGILL)",
    6: "an abort (SIGABRT)",
    7: "a bus error (SIGBUS)",
    8: "a divide error (SIGFPE)",
    11: "a segmentation fault (SIGSEGV)",
}


def explore(
    proj,
    start,
    end,
    setup_asm=None,
    funcs=None,
    data=None,
    prototype=None,
    stack_size=None,
    stack_align=16,
    track=None,
    teardown_asm=None,
):
    if funcs is None:
        funcs = dict(functions(proj))
    is_func = start in {s for s, e in funcs.values()}

    if is_func:
        state = proj.factory.call_state(addr=start, prototype=prototype)
    else:
        state = proj.factory.blank_state(addr=start)
        load_arch(proj).map_stack(state, size=stack_size, align=stack_align)

    if setup_asm:
        setup_asm = setup_asm.rstrip("\n")
        arch = load_arch(proj)
        setup_addr = arch._SETUP_BASE
        full_asm = arch.setup_jump_asm(setup_asm, start)
        encoding = arch.assemble(full_asm, setup_addr)
        state.memory.map_region(setup_addr, len(encoding) + 256, 7, init_zero=True)
        state.memory.store(setup_addr, encoding)
        arch.set_ip(state, setup_addr)

    _symbolic_options(state)
    sym_regs = _symbolic_regs(proj, state)

    if data is None:
        data = {"regs": {}, "mem": {}}

    explicit_regs = _normalize_data_regs((data or {}).get("regs") or {})
    for name, sym in sym_regs.items():
        if name not in explicit_regs:
            continue
        v = explicit_regs[name]
        if isinstance(v, (list, tuple)):
            continue
        try:
            state.solver.add(sym == int(v))
        except (TypeError, ValueError):
            pass

    _map_state_mem_pages(state, (data or {}).get("mem"))

    _track_mem_access(state)
    _track_branches(state)
    _track_state(state, track)

    ret_addrs = _ret_addrs(proj, start, end)

    def _run(state):
        simgr = proj.factory.simgr(state)
        simgr.explore(find=list(ret_addrs))
        return simgr

    def _keep(found, name, sym):
        try:
            if not found.solver.symbolic(sym):
                return False
        except Exception:
            return False
        try:
            return any(name in str(c.variables) for c in found.solver.constraints)
        except Exception:
            return False

    simgr = _run(state)
    states = simgr.found + simgr.active + simgr.deadended
    if _needs_reg_pinning(states, simgr, sym_regs, _keep):
        try:
            arch = load_arch(proj)
        except Exception:
            arch = get_arch()
        explicit_canon = set()
        for name in explicit_regs:
            try:
                explicit_canon.add(arch._canonical_reg(name))
            except Exception:
                explicit_canon.add(str(name).strip().lower())
        _pin_unknown_regs(state, sym_regs, explicit_canon, arch)
        simgr = _run(state)
        states = simgr.found + simgr.active + simgr.deadended
    if teardown_asm:
        return _explore_teardown(proj, states, sym_regs, teardown_asm, _keep)
    return _solution_records(states, sym_regs, _keep)


def _teardown_end(proj, teardown_asm, offset=0x10000):
    arch = get_arch()
    base = arch._SETUP_BASE + offset
    encoding = arch.assemble(arch.normalize_asm(f"{teardown_asm};"), base)
    if not encoding:
        return None, 0
    return base, len(encoding)


def _explore_teardown(proj, states, sym_regs, teardown_asm, keep):
    import angr

    arch = get_arch()
    base, size = _teardown_end(proj, teardown_asm)
    if not size:
        return []
    out = []
    for state in states or ():
        try:
            found = state.copy()
        except Exception:
            continue
        try:
            found.memory.map_region(base, size + 256, 7, init_zero=True)
            found.memory.store(
                base, arch.assemble(arch.normalize_asm(f"{teardown_asm};"), base)
            )
            found.history.replay(angr.SimState, project=proj)
            arch.set_ip(found, base)
            simgr = proj.factory.simgr(found)
            simgr.explore(find=[base + size])
        except Exception:
            continue
        out.extend(
            _solution_records(
                simgr.found + simgr.active + simgr.deadended, sym_regs, keep
            )
        )
    return out


def explore_asm(
    code,
    setup_asm=None,
    data=None,
    arch=None,
    stack_size=None,
    stack_align=16,
    track=None,
    teardown_asm=None,
):
    if arch is None:
        arch = get_arch()
    if data is None:
        data = {"regs": {}, "mem": {}}

    full = ""
    if setup_asm:
        full += setup_asm.rstrip("\n") + "\n"
    full += (code or "").rstrip("\n") + "\n"
    base = arch._SETUP_BASE
    try:
        encoding = arch.assemble(full, base)
    except Exception:
        return []
    if not encoding:
        return []

    import angr

    proj = angr.load_shellcode(
        bytes(encoding), arch=arch._ANGR_ARCH_NAME, load_address=base
    )
    state = proj.factory.blank_state(addr=base)
    arch.map_stack(state, size=stack_size, align=stack_align)

    _symbolic_options(state)
    sym_regs = _symbolic_regs(proj, state, prefix="asm_")

    explicit_regs = (data or {}).get("regs") or {}
    explicit_canon = {}
    for k, v in explicit_regs.items():
        try:
            ck = arch._canonical_reg(arch.canonical_data_reg(k))
        except Exception:
            continue
        try:
            explicit_canon[ck] = (
                list(v)
                if isinstance(v, (list, tuple))
                else int(v, 0)
                if isinstance(v, str)
                else int(v)
            )
        except (TypeError, ValueError):
            continue
    for name, sym in sym_regs.items():
        cname = arch._canonical_reg(name)
        if cname in explicit_canon:
            ev = explicit_canon[cname]
            if isinstance(ev, (list, tuple)):
                continue
            try:
                state.solver.add(sym == (ev & 0xFFFFFFFFFFFFFFFF))
            except Exception:
                pass

    _pin_unknown_regs(state, sym_regs, explicit_canon, arch)
    _map_state_mem_pages(state, (data or {}).get("mem"))

    _track_mem_access(state)
    _track_branches(state)
    _track_state(state, track)

    end = base + len(encoding)
    if teardown_asm:
        td_base, td_size = _teardown_end(proj, teardown_asm)
        if td_size:
            try:
                td = arch.assemble(arch.normalize_asm(f"{teardown_asm};"), td_base)
                state.memory.map_region(td_base, td_size + 256, 7, init_zero=True)
                state.memory.store(td_base, td)
                jump = arch.assemble(arch.jump_asm(td_base), base + len(encoding))
                state.memory.store(base + len(encoding), jump)
                end = td_base + td_size
            except Exception:
                pass
    simgr = proj.factory.simgr(state)
    try:
        simgr.explore(find=[end])
    except Exception:
        return []

    try:
        mem_bases = set(arch.mem_regs(full))
    except Exception:
        mem_bases = set()

    def _keep(found, name, sym):
        try:
            if not found.solver.symbolic(sym):
                return False
        except Exception:
            return False
        cname = arch._canonical_reg(name)
        return cname in explicit_canon or cname in mem_bases or name in mem_bases

    return _solution_records(
        simgr.found + simgr.active + simgr.deadended, sym_regs, _keep
    )


def _pinned_branch_addrs(branch_cfg):
    pinned = set()
    for addr, value in ((branch_cfg or {}).get("mem") or {}).items():
        if value == "predictable":
            try:
                pinned.add(int(addr))
            except (TypeError, ValueError):
                continue
    return pinned


def _models_cached(solutions, branch_cfg, arch):
    solutions = solutions if solutions is not None else []
    key = (
        id(solutions),
        frozenset(_pinned_branch_addrs(branch_cfg)),
        id(arch),
    )
    hit = _MODEL_CACHE.get(key)
    if hit is not None and hit[0] is solutions:
        return hit[1]
    models = solve(solutions, branch_cfg=branch_cfg, arch=arch)
    if len(_MODEL_CACHE) >= _SOLVE_CACHE_MAX:
        _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = (solutions, models)
    return models


def solve(solutions, branch_cfg=None, arch=None):
    if not solutions:
        return []

    pinned_addrs = _pinned_branch_addrs(branch_cfg)

    models = []
    for solution in solutions:
        state = solution["state"]

        ret_sym = _return_sym(state)
        ret_values = _return_values(ret_sym) if ret_sym is not None else []
        states = [state]
        if len(ret_values) > 1:
            states = []
            for v in ret_values:
                st = state.copy()
                st.solver.add(ret_sym == v)
                states.append(st)

        for st in states:
            solver = st.solver
            regs = {}
            for name, sym in solution["reg"].items():
                try:
                    regs[name] = int(solver.eval(sym))
                except Exception:
                    continue

            reads, writes = [], []
            for accesses, out in (
                (solution["reads"], reads),
                (solution["writes"], writes),
            ):
                for addr_sym, length_sym, value_sym in accesses:
                    if addr_sym is None or length_sym is None:
                        continue
                    try:
                        addr = int(solver.eval(addr_sym))
                        if not (0 <= addr < (1 << 64)):
                            continue
                    except Exception:
                        continue
                    try:
                        length = int(solver.eval(length_sym))
                    except Exception:
                        continue
                    try:
                        value = (
                            int(solver.eval(value_sym)) if value_sym is not None else 0
                        )
                    except Exception:
                        value = 0
                    out.append((addr, length, value))

            model = {"regs": regs, "reads": reads, "writes": writes}
            if pinned_addrs:
                model["blocks"] = _executed_bbl_addrs([solution])
                model["branch_deps"] = _branch_deps(st, model, pinned_addrs, arch)
            models.append(model)

    return models


def region(target):
    if isinstance(target, (list, tuple)):
        begin, end = (str(part).strip() for part in target)
        if not begin or not end:
            raise ValueError(f"invalid region {target!r}; expected BEGIN..END")
        return (begin, end)
    text = str(target or "").strip()
    if not text:
        return None
    if ".." not in text:
        return text
    begin, _, end = text.partition("..")
    begin, end = begin.strip(), end.strip()
    if not begin or not end:
        raise ValueError(f"invalid target {target!r}; expected NAME or BEGIN..END")
    return (begin, end)


def parse_code(code):
    if code is None:
        return None, None, None
    if isinstance(code, (list, tuple)):
        parts = list(code)
        if not parts:
            return None, None, None
        file = str(parts[0]).strip()
        if len(parts) == 1:
            return (file or None), None, None
        return (file or None), region(parts[1]), None
    text = str(code).strip()
    if not text:
        return None, None, None
    file, sep, rest = text.partition(":")
    if not sep:
        return None, None, text
    return (file.strip() or None), region(rest), None


def benchmark(
    code=None,
    name=None,
    *,
    mode,
    config=None,
    data=None,
    event=None,
    setup=None,
    teardown=None,
    backend=None,
    unroll_n=None,
    debug=False,
):
    file, target, code = parse_code(code)

    if code is None and file is not None:
        snippet = asm_source_code(file, target)
        if snippet:
            code = snippet
            name = name or asm_source_target(target)
            file, target = None, None
        elif _is_asm_source(file):
            raise ValueError(f"cannot resolve {target!r} in {file!r}")

    modes = _modes(mode)

    events = _normalize_groups(event)

    spec, combos = _concrete_configs(config)
    _pin_default_affinity(spec, events)
    _EXPLORE_CACHE.clear()
    _MODEL_CACHE.clear()
    _RUNNABLE_HARNESS.clear()
    if debug:
        _debug_json("config spec", spec)
        _debug_json("config combos", combos)
    data = _merge_data(None, data)
    check_data(data)
    combos = _prune_for_target(file, target, code, combos, data, setup, teardown, debug)
    if debug:
        _debug_json("effective config combos", combos)
    frames = []
    frame_combos = []
    per_mode = _mode_backends(backend)
    for m in modes:
        for combo in combos:
            df = _bench_one(
                file=file,
                target=target,
                code=code,
                name=name,
                mode=m,
                config=dict(combo),
                data=data,
                event=events,
                setup=setup,
                teardown=teardown,
                backend=per_mode.get(m, None) if per_mode else backend,
                unroll_n=unroll_n,
                debug=debug,
            )
            frames.append(df)
            frame_combos.append(combo)
    if not frames:
        raise ValueError("config expands to no combinations")

    tagged = []
    for combo, df in zip(frame_combos, frames):
        resolved = (getattr(df, "attrs", {}) or {}).get("config")
        _tag_config_columns(df, resolved if isinstance(resolved, dict) else combo)
        try:
            filtered = _filter_bench_columns(df)
            try:
                filtered.attrs.update(getattr(df, "attrs", {}) or {})
            except Exception:
                pass
            tagged.append(filtered)
        except Exception:
            tagged.append(df)
    frames = tagged
    if len(frames) == 1:
        try:
            return _order_bench_columns(frames[0])
        except Exception:
            return frames[0]

    df = pd.concat(frames, axis=0)
    first = frames[0]
    for attr in ("info", "file", "data", "code", "state", "distribution"):
        if attr in first.attrs:
            df.attrs[attr] = first.attrs[attr]
    df.attrs["config"] = spec
    try:
        binary = first.attrs.get("info", {}).get("binary")
    except Exception:
        binary = None
    try:
        df.attrs["id"] = _id_hash(data, spec, binary)
    except Exception:
        pass
    try:
        df = _order_bench_columns(df)
    except Exception:
        pass
    return df


def check_data(data, arch=None):
    if arch is None:
        arch = get_arch()
    reserved = getattr(arch, "_HARNESS_RESERVED_REGS", ()) or ()
    for reg in (data or {}).get("regs") or {}:
        try:
            canon = arch.canonical_data_reg(reg)
        except Exception:
            canon = str(reg).strip().lower()
        try:
            canon = arch._canonical_reg(canon)
        except Exception:
            pass
        if canon in reserved:
            raise ValueError(
                f"data register {reg!r} is reserved by the measurement harness "
                f"({', '.join(reserved)}): the loop counter, "
                "its cursors, the stack and the instruction pointer drive the "
                "harness, so writing them would corrupt the measurement"
            )
    return data


def asm_source_target(target):
    if target is None:
        return ""
    if isinstance(target, (list, tuple)):
        parts = [str(part).strip() for part in target]
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2 and all(parts):
            return f"{parts[0]}..{parts[1]}"
        return ""
    return str(target).strip()


def asm_source_code(file, target):
    if not file or not _is_asm_source(file):
        return None
    try:
        code = _asm_region(file, target)
    except Exception:
        return None
    arch = get_arch()
    try:
        if not arch.assemble(arch.normalize_asm(f"{code};"), arch._SETUP_BASE):
            return None
    except Exception:
        return None
    return code


def _asm_region(file, target):
    text = _asm_text(file)
    target = asm_source_target(target)
    if not target:
        raise ValueError("a label or a `begin..end` region is required")
    begin, dotdot, end = target.partition("..")
    begin = begin.strip()
    end = end.strip() if dotdot else None
    if begin == end:
        raise ValueError(
            f"empty region {target!r}: {begin!r} and {end!r} are the same label"
        )
    bodies = {name: (position, size) for name, position, size in asm_labels(file)}
    known = ", ".join(n for n in bodies if not n.startswith(".")) or "none"
    if begin not in bodies:
        raise ValueError(f"cannot resolve {target!r}; labels: {known}")
    position, size = bodies[begin]
    if end is not None:
        if end not in bodies:
            raise ValueError(f"cannot resolve {end!r}; labels: {known}")
        if bodies[end][0] < position:
            raise ValueError(
                f"empty region {target!r}: {end!r} starts before {begin!r}"
            )
        position, size = position, bodies[end][0] - position
    out = _asm_body(text[position : position + size])
    if not out:
        raise ValueError(f"region {target!r} has no instructions")
    return "; ".join(out)


def _asm_text(file):
    with open(file, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _asm_body(text):
    out, returned = [], False
    for raw in str(text or "").splitlines():
        for statement in raw.split(";"):
            line = " ".join(_ASM_COMMENT.sub(" ", statement).split())
            if not line:
                continue
            head, colon, rest = line.partition(":")
            if colon and _ASM_LABEL.fullmatch(head.strip()):
                out.append(f"{head.strip()}:")
                line = rest.strip()
                returned = False
            if not line or line.startswith("."):
                continue
            if returned:
                continue
            returned = line.split(" ", 1)[0].lower().startswith("ret")
            if not returned:
                out.append(line)
    while out and out[-1].endswith(":"):
        out.pop()
    return out


def disassemble(
    code=None,
    config=None,
    data=None,
    setup=None,
    teardown=None,
):
    file, target, code = parse_code(code)
    target = _normalize_target(target)
    if code is None and file is not None:
        code = asm_source_code(file, target)
        if code:
            file = None
        elif _is_asm_source(file):
            raise ValueError(f"cannot resolve {target!r} in {file!r}")
    if code is not None:
        body = get_arch().normalize_asm(f"{code};").replace(";", "\n")
        return ".intel_syntax noprefix\n" + _left_align_asm(body)
    if file is None:
        raise ValueError(
            "a target is required, e.g. disassemble('a.out:fizz_buzz') or "
            "disassemble('mov eax, 42')"
        )

    import angr

    cfg = _merge_config(config)
    if isinstance(cfg, dict) and "data" in cfg:
        raise ValueError(
            "config must not contain 'data'; pass data separately via data={...}"
        )
    check_data(_merge_data(None, data))
    project = angr.Project(
        resolve_exec(file),
        auto_load_libs=_resolve_lib(cfg),
        load_debug_info=False,
    )
    funcs, prototypes = functions(project)
    pat = target
    if not pat:
        raise ValueError("a target is required")

    arch = load_arch(project)
    eff_data = _merge_data(None, data)
    obj = project.loader.main_object
    setup_asm_for_explore = _setup_asm_for(project, obj, arch, setup)
    teardown_asm_for_explore = _setup_asm_for(project, obj, arch, teardown)
    excluded = []
    out = []
    matched = list(resolve_targets(project, pat, funcs))
    if not matched:
        table = _available_table(project)
        print(
            f"cannot resolve {pat!r} in {file!r}; available targets:",
            file=sys.stderr,
        )
        print(table, file=sys.stderr)
        raise ValueError(f"cannot resolve {pat!r} in {file!r}")
    if len(matched) > 1:
        table = _available_table(project)
        print(
            f"ambiguous target {pat!r} in {file!r}; available targets:",
            file=sys.stderr,
        )
        print(table, file=sys.stderr)
        raise ValueError(f"ambiguous target {pat!r} in {file!r}")
    for _label, start, end in matched:
        try:
            sols = explore(
                project,
                start,
                max(end, start + 1),
                setup_asm=setup_asm_for_explore,
                funcs=funcs,
                data=eff_data,
                prototype=_func_prototype(prototypes.get(_label), eff_data),
                teardown_asm=teardown_asm_for_explore,
            )
        except Exception:
            sols = []
        addrs = _executed_blocks(sols)
        for _name, (_first, _last) in funcs.items():
            if _name in _setup_names(setup) or _name in _setup_names(teardown):
                excluded.append((int(_first), int(_last)))
        text = _disasm_executed_asm(project, addrs, arch, excluded) if addrs else ""
        if not text:
            text = _disasm_target(project, start, max(end, start + 1), arch)
        out.append(text)
    return "\n".join(out)


def to_json(df, indent=4):
    return json.dumps(_record_envelope(df), indent=indent, default=str)


def normalize_container(key, value):
    if value is None:
        return None
    if key == _THREAD_TOP:
        if isinstance(value, dict):
            return [[dict(value)]]
        if isinstance(value, (list, tuple)):
            items = list(value)
            if items and all(isinstance(v, dict) for v in items):
                return [[dict(v)] for v in items]
            return [
                [dict(t) if isinstance(t, dict) else t for t in group]
                if isinstance(group, (list, tuple))
                else group
                for group in items
            ]
        return value
    if isinstance(value, (list, tuple)):
        return [dict(v) if isinstance(v, dict) else _list_entry(key, v) for v in value]
    return [_list_entry(key, value)]


def data_param_columns(data, include_mem=True):
    cols = {}

    canon_regs = {}
    for reg, val in ((data or {}).get("regs") or {}).items():
        v = _param_columns_value(val)
        if v is None:
            continue
        if _is_scratch_value(val):
            continue
        canon = _canonical_data_reg_key(reg)
        canon_regs[canon] = v
    for reg, v in canon_regs.items():
        cols[f"data.{reg}"] = v
    if include_mem:
        mem = (data or {}).get("mem") or {}
        for addr, val in mem.items():
            v = _param_columns_value(val)
            if v is None:
                continue

            if f"data.{addr}" not in cols:
                cols[f"data.{addr}"] = v
    return cols


def data_param_value(data, key):

    if not isinstance(key, str) or not key.startswith("data."):
        return None
    rest = key[len("data.") :]
    if (
        rest.startswith("regs.")
        or rest.startswith("mem.")
        or rest.startswith("memory.")
    ):
        return None
    regs = (data or {}).get("regs") or {}
    index = {}
    for k, v in regs.items():
        if v is None:
            continue
        canon = _canonical_data_reg_key(k)
        index[canon] = v
    try:
        canon = _canonical_data_reg_key(rest)
    except Exception:
        canon = rest
    for cand in (canon, rest):
        if cand in index:
            v = _first_scalar(index[cand])
            return v
    mem = (data or {}).get("mem") or {}
    if rest in mem:
        return _first_scalar(mem[rest])
    return None


def config_param_columns(config):
    from .plot import normalize_config

    flat = normalize_config(config)
    return {f"config.{k}": v for k, v in flat.items()}


def explored_data_dict(solutions=None, models=None):
    if models is None:
        try:
            models = _models_cached(solutions or [], None, None)
        except Exception:
            models = []
    models = models or []
    try:
        harness = set(get_arch()._HARNESS_REGS or ())
    except Exception:
        harness = set()
    regs_vals = {}
    mem_vals = {}
    for m in models or []:
        for name, v in (m.get("regs") or {}).items():
            canon = _canonical_data_reg_key(name)
            if canon in harness:
                continue
            try:
                iv = int(v) & 0xFFFFFFFFFFFFFFFF
            except (TypeError, ValueError):
                continue
            lst = regs_vals.setdefault(canon, [])
            if iv not in lst:
                lst.append(iv)
        for a, _, v in (m.get("reads") or []) + (m.get("writes") or []):
            try:
                key = f"0x{int(a):x}"
            except (TypeError, ValueError):
                continue
            try:
                iv = int(v) & 0xFFFFFFFFFFFFFFFF
            except (TypeError, ValueError):
                iv = 0
            lst = mem_vals.setdefault(key, [])
            if iv not in lst:
                lst.append(iv)
    regs = {k: (v[0] if len(v) == 1 else list(v)) for k, v in regs_vals.items()}
    mem = {k: (v[0] if len(v) == 1 else list(v)) for k, v in mem_vals.items()}
    return {"regs": regs, "mem": mem}


def add_param_columns(df, data):
    if df is None or getattr(df, "empty", False):
        return df
    try:
        cols = data_param_columns(data)
    except Exception:
        return df
    new_cols = {c: v for c, v in cols.items() if c not in df.columns}
    if not new_cols:
        try:
            return _order_bench_columns(df)
        except Exception:
            return df
    try:
        const = pd.DataFrame(
            {c: pd.Series([v] * len(df), index=df.index) for c, v in new_cols.items()}
        )
        if len(const.columns):
            _ended = pd.concat([df, const], axis=1)
            if len(_ended) == len(df):
                df = _ended
            else:
                for c in list(const.columns):
                    try:
                        df[c] = const[c]
                    except Exception:
                        continue
    except Exception:
        for col, val in new_cols.items():
            try:
                df[col] = val
            except Exception:
                continue
    try:
        df = _order_bench_columns(df)
    except Exception:
        pass
    return df


def validate_spec(config):
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {config!r}")
    for key in config:
        if key == "data":
            raise ValueError(
                "config must not contain 'data'; pass data separately via data={...}"
            )
        if key not in _VALID_TOP_KEYS:
            raise ValueError(
                f"unknown config key {key!r}; expected one of "
                f"{', '.join(sorted(_VALID_TOP_KEYS))}"
            )
    if config.get("thread") is not None:
        _validate_thread_top(config["thread"])
    for key in _LIST_TOPS:
        value = config.get(key, None)
        if value is None:
            continue
        if isinstance(value, dict):
            _check_list_entry(key, value)
        elif isinstance(value, list):
            _validate_list_top(key, value)
        elif not isinstance(_list_entry(key, value), dict):
            raise ValueError(
                f"unknown {key} config {value!r}; expected a list of alternatives "
                f"like {_LIST_TOP_HINT[key]}"
            )
    for cache_key in _CACHE_KEYS:
        _validate_cache_key(config, cache_key)
    _validate_lib(config)
    iterations = config.get("iterations")
    if isinstance(iterations, dict):
        for key in iterations:
            if key not in _VALID_ITERATIONS_KEYS and key != "count":
                raise ValueError(
                    f"unknown iterations key {key!r}; expected one of "
                    f"{', '.join(sorted(_VALID_ITERATIONS_KEYS))}"
                )
        for sub in ("min", "max", "count"):
            value = iterations.get(sub)
            if value is None:
                continue
            try:
                iv = int(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"iterations.{sub} must be an integer, got {value!r}"
                ) from e
            if iv < 1:
                raise ValueError(f"iterations.{sub} must be >= 1, got {value!r}")
        lo, hi = iterations.get("min"), iterations.get("max")
        if lo is not None and hi is not None and int(lo) > int(hi):
            raise ValueError(
                f"iterations.min ({lo!r}) must be <= iterations.max ({hi!r})"
            )
    _validate_backend(config)
    branch = config.get("branch", None)
    if branch is not None and not isinstance(branch, (list, str, bool, dict)):
        raise ValueError(
            f"branch config must be a list of alternatives, a single value, "
            f"or a per-address dict, got {branch!r}"
        )

    def _require_alternatives(path, value):
        if isinstance(value, list) and not value:
            raise ValueError(
                f"config {path} must be a non-empty list of alternatives, got {value!r}"
            )

    for key, value in config.items():
        if key in _SCALAR_KEYS:
            _validate_scalar_key(key, value)
        elif key in _LIST_TOPS and isinstance(value, list):
            for i, entry in enumerate(value):
                if isinstance(entry, dict):
                    for sub, sv in entry.items():
                        if isinstance(sv, (list, tuple)):
                            raise ValueError(
                                f"config {key}[{i}].{sub} must be a single value, "
                                f"got {sv!r}; sweep via multiple {key} "
                                "alternatives instead"
                            )
        elif key in _CONTAINER_TOPS and isinstance(value, dict):
            for sub, sv in value.items():
                _require_alternatives(f"{key}.{sub}", sv)
        else:
            _require_alternatives(key, value)

    if config.get("branch") is not None:
        alts = (
            config["branch"]
            if isinstance(config["branch"], list)
            else [config["branch"]]
        )
        for v in alts:
            if isinstance(v, str):
                if _branch_value(v) is None:
                    choices = ", ".join(_BRANCH_CHOICES)
                    raise ValueError(
                        f"branch prediction must be one of {choices}, got {v!r}"
                    )
            elif isinstance(v, bool):
                continue
            elif isinstance(v, dict):
                _branch_config({"branch": v})
            else:
                raise ValueError(
                    f"branch config elements must be a string, bool, or dict, got {v!r}"
                )
    return config


def _stable(value):
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return repr(value)


def _explore_cached(key, factory):
    try:
        if key in _EXPLORE_CACHE:
            return _EXPLORE_CACHE[key]
    except TypeError:
        return factory()
    out = factory()
    try:
        _EXPLORE_CACHE[key] = out
    except TypeError:
        pass
    return out


def _warn_exploration(target, ex):
    print(
        f"cannot explore {str(target)[:120]!r}: {ex}; measuring it with "
        "unconstrained inputs",
        file=sys.stderr,
    )


def _explore_asm_cached(code, setup_asm, data, arch, stack_size, stack_align):
    key = (
        "asm",
        str(code),
        str(setup_asm or ""),
        _stable(data),
        int(stack_size or 0),
        int(stack_align or 0),
    )

    def _run():
        try:
            return explore_asm(
                code,
                setup_asm=setup_asm,
                data=data,
                arch=arch,
                stack_size=stack_size,
                stack_align=stack_align,
            )
        except Exception as ex:
            _warn_exploration(code, ex)
            return []

    return _explore_cached(key, _run)


def _explore_target_cached(
    project,
    cache_id,
    target_label,
    start,
    end,
    ang_setup,
    funcs,
    data,
    prototype,
    stack_size,
    stack_align,
):
    key = (
        "file",
        cache_id,
        str(target_label),
        int(start),
        int(end),
        str(ang_setup or ""),
        _stable(data),
        int(stack_size or 0),
        int(stack_align or 0),
    )
    return _explore_cached(
        key,
        lambda: explore(
            project,
            start,
            end,
            setup_asm=ang_setup,
            funcs=funcs,
            data=data,
            prototype=prototype,
            stack_size=stack_size,
            stack_align=stack_align,
        ),
    )


def _project_cache_id(file, config):
    container = config or {}
    orders = [
        _entry("func", alt).get("order")
        for alt in _entry_alternatives("func", container.get("func"))
    ]
    return (
        str(file),
        _resolve_lib(config),
        _entry_signature("stack", container.get("stack")),
        _entry_signature("code", container.get("code")),
        _entry_signature("func", container.get("func")),
        _config_seed_int(config) if _FUNC_ORDERS[1] in orders else None,
    )


def _branches_seen(solutions):
    return len(solutions or ()) > 1 or any(
        sol.get("branches") for sol in solutions or ()
    )


def _memory_seen(solutions):
    return any(sol.get("reads") or sol.get("writes") for sol in solutions or ())


def _target_features(file, target, code, config, data, setup=None, teardown=None):
    stack = _container(config, "stack")
    size, align = stack["size"], stack["align"]
    if code:
        arch = get_arch()
        setup_asm = arch.normalize_asm("\n".join(setup) + "\n") if setup else None
        solutions = _explore_asm_cached(
            arch.normalize_asm(f"{code};"), setup_asm, data, arch, size, align
        )
    else:
        binary = resolve_exec(file) if file else None
        project, obj, funcs, prototypes = _bench_project(
            binary, config, stack, _container(config, "func")
        )
        arch = load_arch(project)
        targets = list(resolve_targets(project, target, funcs))
        if len(targets) != 1:
            raise ValueError(f"cannot resolve {target!r}")
        target_label, start, end = targets[0]
        solutions = _explore_target_cached(
            project,
            _project_cache_id(binary, config),
            target_label,
            start,
            end,
            _setup_asm_for(project, obj, arch, setup),
            funcs,
            data,
            _func_prototype(prototypes.get(target_label), data),
            size,
            align,
        )
    return {
        "branches": _branches_seen(solutions),
        "memory": _memory_seen(solutions) or bool((data or {}).get("mem")),
    }


def _setup_names(setup):
    names = setup if isinstance(setup, (list, tuple)) else [setup]
    return [str(name).strip() for name in (names or ()) if name and str(name).strip()]


def _call_seq_asm(obj, arch, names):
    return "\n".join(arch.call_seq_asm(obj.get_symbol(name)) for name in names)


def _setup_asm_for(project, obj, arch, setup):
    if not setup:
        return None
    names = setup if isinstance(setup, (list, tuple)) else [setup]
    parts = []
    for name in names:
        if not name:
            continue
        try:
            parts.append(arch.call_seq_asm(obj.get_symbol(name)))
        except Exception:
            parts.append(str(name))
    if not parts:
        return None
    for name in names:
        if not name:
            continue
        sym = None
        try:
            sym = project.loader.find_symbol(name)
        except Exception:
            sym = None
        if sym is not None:
            try:
                return arch.call_asm(sym.rebased_addr)
            except Exception:
                break
    return "\n".join(parts)


def _prune_for_target(
    file, target, code, combos, data, setup=None, teardown=None, debug=False
):
    if len(combos) < 2:
        return combos
    if not any(
        len({_stable(combo.get(key)) for combo in combos}) > 1 for key in _PRUNE_KEYS
    ):
        return combos
    try:
        features = _target_features(
            file, target, code, combos[0], data, setup, teardown
        )
    except Exception as ex:
        if debug:
            _debug_log(f"config pruning skipped: {ex}")
        return combos
    pruned = _prune_combos(combos, features)
    if debug and len(pruned) != len(combos):
        _debug_log(
            f"pruned {len(combos) - len(pruned)} of {len(combos)} config "
            f"combinations (branches={features.get('branches')}, "
            f"memory={features.get('memory')})"
        )
    return pruned


def _prune_combos(combos, features):
    prunable = []
    if not features.get("branches"):
        prunable.append("branch")
    if not features.get("memory"):
        prunable.extend(("dcache", "dtlb"))
    if not any(
        len({_stable(combo.get(key)) for combo in combos}) > 1 for key in prunable
    ):
        return combos
    out, seen = [], set()
    for combo in combos:
        signature = _stable({k: v for k, v in combo.items() if k not in prunable})
        if signature in seen:
            continue
        seen.add(signature)
        out.append(combo)
    return out


def _debug_log(msg):
    try:
        sys.stderr.write(f"[perf debug] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _debug_json(title, obj):
    try:
        text = json.dumps(obj, indent=2, default=str)
    except Exception:
        try:
            text = str(obj)
        except Exception:
            text = "<unprintable>"
    _debug_log(f"{title}:\n{text}")


def _solution_summary(solutions):
    out = []
    for i, sol in enumerate(solutions or []):
        sol = sol or {}
        out.append(
            {
                "index": i,
                "regs": sorted((sol.get("reg") or {}).keys()),
                "branches": sol.get("branches", 0),
                "reads": len(sol.get("reads") or []),
                "writes": len(sol.get("writes") or []),
            }
        )
    return out


def _ret_addrs(proj, start, end):
    arch = load_arch(proj)
    addrs = set()
    try:
        for f in proj.kb.functions.values():
            if f.addr == start:
                for b in f.blocks:
                    for insn in b.capstone.insns:
                        if arch.is_return_mnemonic(insn.mnemonic):
                            addrs.add(insn.address)
                break
    except Exception:
        pass
    if not addrs:
        try:
            md = arch.disassembler()
            size = max(int(end) - int(start), 1)
            blob = proj.loader.memory.load(int(start), size)
            for insn in md.disasm(bytes(blob), int(start)):
                if arch.is_return_mnemonic(insn.mnemonic):
                    addrs.add(insn.address)
        except Exception:
            pass
    if not addrs:
        addrs = {end}
    return addrs


def _pin_unknown_regs(state, sym_regs, explicit_canon, arch):
    harness = set(arch._HARNESS_REGS or ())
    idx = 0
    for name in sorted(sym_regs):
        cname = arch._canonical_reg(name)
        if cname in explicit_canon or cname in harness:
            continue
        try:
            if state.solver.symbolic(sym_regs[name]):
                state.solver.add(sym_regs[name] == _ASM_SCRATCH_BASE + idx * 0x1000)
                idx += 1
        except Exception:
            pass
    try:
        scratch_page = _ASM_SCRATCH_BASE & ~0xFFF
        state.memory.map_region(scratch_page, (idx + 1) * 0x1000, 7, init_zero=True)
    except Exception:
        pass
    return idx


def _faulted_on_unmapped(simgr):
    for st in getattr(simgr, "errored", None) or ():
        err = getattr(st, "error", None)
        if "unmapped" in str(err).lower():
            return True
    return False


def _states_touched_memory(states):
    for st in states or ():
        for key in ("reads", "writes"):
            for entry in (st.globals.get(key) or ()) if st.globals else ():
                addr = entry[0]
                try:
                    if not addr.symbolic:
                        return True
                except AttributeError:
                    return True
    return False


def _needs_reg_pinning(states, simgr, sym_regs, keep):
    if not states:
        return _faulted_on_unmapped(simgr)
    if _states_touched_memory(states):
        return False
    return not any(
        keep(st, name, sym) for st in states for name, sym in sym_regs.items()
    )


def _as_records(df):
    index = getattr(df, "index", None)
    names = list(getattr(index, "names", []) or [])
    try:
        if isinstance(df.index, pd.MultiIndex) or any(
            n in names for n in _IDENTITY_COLUMNS
        ):
            return df.reset_index()
    except Exception:
        pass
    return df


def _record_envelope(df):
    rec = _as_records(df)
    attrs = getattr(df, "attrs", {}) or {}
    config = dict(attrs.get("config", {}) or {})
    info = dict(attrs.get("info", {}) or {})
    data = attrs.get("data", None)
    binary = info.get("binary") if isinstance(info, dict) else None
    run_id = _id_hash(data, config, binary)
    try:
        file_label = _text(rec["file"].iloc[0]) if len(rec) else ""
        name = _text(rec["name"].iloc[0]) if len(rec) else ""
    except Exception:
        file_label = name = ""
    if run_id and name:
        name = f"{name}-{run_id}"
    info = {"cpu": info.get("cpu")} if isinstance(info, dict) else info
    drop = {"file", "name", "time"}
    try:
        if len(rec):
            for col in ("file", "name"):
                if col in rec.columns and rec[col].nunique() > 1:
                    drop.discard(col)
    except Exception:
        pass
    results = rec.drop(columns=[c for c in drop if c in rec.columns])
    rows = []
    for row in results.to_dict(orient="records"):
        row = nest(_json_safe(row))
        for key in ("config", "data"):
            if isinstance(row.get(key), dict):
                row[key] = _prune_nulls(row[key])
        rows.append(row)
    return {
        "file": file_label,
        "name": name,
        "id": run_id,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "info": info,
        "output": rows,
    }


def _prune_nulls(node):
    out = {}
    for key, value in node.items():
        if value is None:
            continue
        if isinstance(value, dict):
            value = _prune_nulls(value)
            if not value:
                continue
        out[key] = value
    return out


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _external_cfg(config):
    return (config or {}).get("external") or {}


def _lib_value(spec):
    if spec is None:
        return False
    if isinstance(spec, str):
        return spec.strip().lower() in ("load", "true", "yes", "on", "1")
    return bool(spec)


def _resolve_lib(config):
    return _lib_value(_external_cfg(config).get("lib"))


def _validate_lib(config):
    external = (config or {}).get("external")
    if external is None:
        return
    if not isinstance(external, dict):
        raise ValueError(
            f"unknown external config {external!r}; expected a dict like "
            "{'lib': False}"
        )
    for key in external:
        if key not in _VALID_EXTERNAL_KEYS:
            raise ValueError(
                f"unknown external key {key!r}; expected one of "
                f"{', '.join(sorted(_VALID_EXTERNAL_KEYS))}"
            )
    lib = external.get("lib")
    if lib is not None and not isinstance(lib, (bool, int, str)):
        raise ValueError(
            f"unknown external.lib config {lib!r}; expected a boolean "
            "(True loads shared libraries, False fakes them)"
        )


def _validate_cache_key(config, key):
    value = config.get(key, None)
    if value is not None and not isinstance(value, (str, dict, list, int, float)):
        raise ValueError(
            f"unknown {key} config {value!r}; expected a shortcut like 'hot'/'cold' "
            f"or a dict like {{'{key}': {{'hit_rate': 100}}}} or a list of "
            "alternatives"
        )
    arch = get_arch()
    validate = getattr(arch, "validate_cache_key", None)
    if validate is None:
        return
    for entry in value if isinstance(value, list) else [value]:
        validate(key, entry)


def _validate_backend(config):
    backend = (config or {}).get("backend")
    if backend is None:
        return
    if isinstance(backend, str):
        if backend not in _BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected one of {', '.join(_BACKENDS)}"
            )
        return
    if not isinstance(backend, dict):
        raise ValueError(
            f"unknown backend config {backend!r}; expected 'loop' or 'unroll', or a "
            "dict like {'unroll': {'count': 5}}"
        )
    for name, opts in backend.items():
        if name not in _BACKENDS:
            raise ValueError(
                f"unknown backend {name!r}; expected one of {', '.join(_BACKENDS)}"
            )
        if opts is None:
            continue
        if not isinstance(opts, dict):
            raise ValueError(
                f"unknown backend.{name} config {opts!r}; expected a dict like "
                "{'probes': 3, 'target': 0.005}"
            )
        for key in opts:
            if key not in _BACKEND_OPTION_KEYS:
                raise ValueError(
                    f"unknown backend.{name} key {key!r}; expected one of "
                    f"{', '.join(sorted(_BACKEND_OPTION_KEYS))}"
                )
        for opt in ("probes", "runs"):
            value = opts.get(opt)
            if value is None:
                continue
            try:
                if int(value) < 1:
                    raise ValueError(
                        f"backend.{name}.{opt} must be >= 1, got {value!r}"
                    )
            except (TypeError, ValueError) as e:
                if "must be >=" in str(e):
                    raise
                raise ValueError(
                    f"backend.{name}.{opt} must be an integer, got {value!r}"
                ) from e
        target = opts.get("target")
        if target is not None:
            try:
                if not float(target) > 0:
                    raise ValueError(
                        f"backend.{name}.target must be > 0, got {target!r}"
                    )
            except (TypeError, ValueError) as e:
                if "must be >" in str(e):
                    raise
                raise ValueError(
                    f"backend.{name}.target must be a number, got {target!r}"
                ) from e
        count = opts.get("count")
        if count is not None:
            try:
                if int(count) < 1:
                    raise ValueError(
                        f"backend.{name}.count must be >= 1, got {count!r}"
                    )
            except (TypeError, ValueError) as e:
                if "must be >=" in str(e):
                    raise
                raise ValueError(
                    f"backend.{name}.count must be an integer, got {count!r}"
                ) from e


def _list_entry(key, value):
    if isinstance(value, dict):
        return dict(value)
    if key == "func" and isinstance(value, str):
        return {"order": value}
    if key == "code" and isinstance(value, (int, str)) and not isinstance(value, bool):
        return {"align": value}
    return value


def _check_list_entry(key, entry):
    if not isinstance(entry, dict):
        raise ValueError(
            f"config {key} alternatives must be dicts, got {entry!r}; expected "
            f"a list like {_LIST_TOP_HINT[key]}"
        )
    valid = _LIST_TOP_KEYS[key]
    for name in entry:
        if name not in valid:
            raise ValueError(
                f"unknown {key} key {name!r}; expected one of "
                f"{', '.join(sorted(valid))}"
            )
    return entry


def _validate_list_top(key, value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            f"unknown {key} config {value!r}; expected a non-empty list of "
            f"alternatives like {_LIST_TOP_HINT[key]}"
        )
    for entry in value:
        _check_list_entry(key, entry)
        for name, sub in entry.items():
            if isinstance(sub, (list, tuple)):
                raise ValueError(
                    f"config {key}.{name} must be a single value, got {sub!r}; "
                    f"sweep via multiple {key} alternatives instead"
                )


def _validate_thread_top(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"unknown thread config {value!r}; {_THREAD_HINT}")
    _thread_cfg({"thread": value})


def _param_columns_value(val):
    if isinstance(val, (list, tuple)):
        return list(val) or None
    return _to_int_or(val, None)


def _is_scratch_value(value):
    for item in value if isinstance(value, (list, tuple)) else [value]:
        try:
            addr = int(item)
        except (TypeError, ValueError):
            continue
        if _ASM_SCRATCH_BASE <= addr < _ASM_SCRATCH_BASE + 0x100000:
            return True
    return False


def _config_seed_int(config):
    seed = (config or {}).get("seed", None)
    if seed is None:
        return None
    try:
        return int(seed) & 0xFFFFFFFFFFFFFFFF
    except (TypeError, ValueError):
        digest = hashlib.sha256(str(seed).encode()).digest()
        return int.from_bytes(digest[:8], "little")


def _apply_seed(config):
    seed_int = _config_seed_int(config)
    if seed_int is None:
        return None
    try:
        random.seed(seed_int)
    except Exception:
        pass
    try:
        np.random.seed(seed_int % (2**32))
    except Exception:
        pass
    return seed_int


def _make_rng(config):
    seed_int = _config_seed_int(config)
    if seed_int is None:
        return random.Random()
    return random.Random(seed_int)


def _resolve_align_value(value, where):
    if value is None:
        return _DEFAULT_ALIGN
    try:
        align = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where} align must be an integer, got {value!r}") from e
    if align < 1 or (align & (align - 1)):
        raise ValueError(f"{where} align must be a power of two >= 1, got {value!r}")
    return align


def _resolve_stack_size(value):
    try:
        size = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"stack size must be an integer, got {value!r}") from e
    if size <= 0:
        raise ValueError(f"stack size must be > 0, got {value!r}")
    return size


def _entry_alternatives(key, value):
    if value is None:
        return [None]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(
                f"config {key} must be a non-empty list of alternatives, got {value!r}"
            )
        return list(value)
    return [value]


def _entry(key, value):
    entry = _list_entry(key, value)
    if entry is None:
        return dict(_ENTRY_DEFAULTS[key])
    if not isinstance(entry, dict):
        raise ValueError(
            f"unknown {key} config {value!r}; expected a dict like "
            f"{_LIST_TOP_HINT[key]}"
        )
    return {**_ENTRY_DEFAULTS[key], **entry}


def _alternate(values, key, leaf):
    items = list(values)
    if not items:
        raise ValueError(
            f"config {key}.{leaf} must be a non-empty list, got {values!r}"
        )
    slot = _CONTAINER_RR.get((key, leaf), 0)
    _CONTAINER_RR[(key, leaf)] = slot + 1
    return items[slot % len(items)]


def _container(config, key):
    alternatives = _entry_alternatives(key, (config or {}).get(key))
    if len(alternatives) != 1:
        raise ValueError(
            f"config {key} has {len(alternatives)} alternatives here; expand the "
            f"combinations first (e.g. via _expand_config_spec)"
        )
    entry = _entry(key, alternatives[0])
    for leaf in ("align", "size", "order"):
        if leaf in entry and isinstance(entry[leaf], (list, tuple)):
            entry[leaf] = _alternate(entry[leaf], key, leaf)
    entry["align"] = _resolve_align_value(entry.get("align"), key)
    if "size" in entry:
        entry["size"] = _resolve_stack_size(entry["size"])
    if "order" in entry:
        entry["order"] = _resolve_func_order(entry["order"])
    return entry


def _entry_signature(key, value):
    return tuple(_stable(_entry(key, alt)) for alt in _entry_alternatives(key, value))


def _resolve_func_order(value):
    order = str(value).strip().lower()
    if order not in _FUNC_ORDERS:
        raise ValueError(
            f"unknown func order {order!r}; expected {', '.join(_FUNC_ORDERS)}"
        )
    return order


def _align_directive(align):
    try:
        align = int(align)
    except (TypeError, ValueError):
        return ""
    if align <= 1:
        return ""
    return f".align {align}"


def _with_align(code, align):
    directive = _align_directive(align)
    if not directive or not code:
        return code
    return f"{directive}\n{code}"


def _canonical_data_reg_key(key):
    return get_arch().canonical_data_reg(key)


def _normalize_data_regs(regs):
    out = {}
    for key, val in (regs or {}).items():
        canon = _canonical_data_reg_key(key)
        if isinstance(val, (list, tuple)):
            out[canon] = [_to_u64(x) for x in val]
        else:
            try:
                ival = int(val, 0) if isinstance(val, str) else int(val)
            except (TypeError, ValueError):
                continue
            out[canon] = ival & 0xFFFFFFFFFFFFFFFF
    return out


def _mem_entries(addr_str, val):
    base_str, is_range = _parse_addr_key(addr_str)
    try:
        base = int(base_str, 0)
    except (TypeError, ValueError):
        return
    if is_range:
        if isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                yield f"0x{base + i:x}", _to_u64(item)
        else:
            yield f"0x{base:x}", _to_u64(val)
        return
    if isinstance(val, (list, tuple)):
        yield f"0x{base:x}", [_to_u64(item) for item in val]
    else:
        yield f"0x{base:x}", _to_u64(val)


def _config_data(config):
    data = (config or {}).get("data", None) if isinstance(config, dict) else None
    if data is None:
        return {"regs": {}, "mem": {}}
    if not isinstance(data, dict):
        raise ValueError(f"unknown data config {data!r}; expected a dict")
    regs = {}
    mem = {}
    for key, val in data.items():
        s = str(key).strip()
        if _is_mem_addr_key(s):
            for k, v in _mem_entries(s, val):
                mem[k] = v
        elif isinstance(val, (list, tuple)):
            regs[_canonical_data_reg_key(key)] = [_to_u64(x) for x in val]
        else:
            try:
                ival = int(val, 0) if isinstance(val, str) else int(val)
            except (TypeError, ValueError):
                continue
            regs[_canonical_data_reg_key(key)] = ival & 0xFFFFFFFFFFFFFFFF
    return {"regs": regs, "mem": mem}


class _Args(ctypes.Structure):
    _fields_ = [
        ("iterations", ctypes.c_uint64),
        ("inputs", ctypes.c_void_p),
        ("outputs", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
    ]


def _iter_mem_addrs(data):
    if not data or not data.get("mem"):
        return
    for addr_str in data["mem"]:
        base_str, _ = _parse_addr_key(addr_str)
        try:
            yield int(base_str, 0)
        except (TypeError, ValueError):
            continue


def _min_mmap_addr():
    try:
        with open("/proc/sys/vm/mmap_min_addr") as fh:
            return int(fh.read().strip(), 0)
    except (OSError, ValueError):
        return 0x1000


def _map_scratch_reg_pages(models):
    lo = _ASM_SCRATCH_BASE & ~(mmap.PAGESIZE - 1)
    hi = lo
    for m in models or ():
        for value in (m.get("regs") or {}).values():
            for item in value if isinstance(value, (list, tuple)) else [value]:
                try:
                    addr = int(item)
                except (TypeError, ValueError):
                    continue
                if lo <= addr < lo + 0x100000:
                    page = addr & ~(mmap.PAGESIZE - 1)
                    hi = max(hi, page + mmap.PAGESIZE)
    if hi <= lo:
        return
    _map_data_pages({"mem": {str(a): 0 for a in range(lo, hi, mmap.PAGESIZE)}})


@functools.cache
def _mmap_fixed():
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    mmap_fn = libc.mmap
    mmap_fn.restype = ctypes.c_void_p
    mmap_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    return mmap_fn


def _map_data_pages(data, strict=True):
    addrs = list(_iter_mem_addrs(data))
    if not addrs:
        return

    mmap_fn = _mmap_fixed()

    seen = set()
    for addr in addrs:
        page_addr = addr & ~(mmap.PAGESIZE - 1)
        if page_addr in seen:
            continue
        seen.add(page_addr)
        if page_addr in _MAPPED_DATA_PAGES or page_addr in _UNMAPPABLE_DATA_PAGES:
            continue

        ctypes.set_errno(0)
        mapped = mmap_fn(
            ctypes.c_void_p(page_addr),
            mmap.PAGESIZE,
            mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
            mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_FIXED_NOREPLACE,
            -1,
            0,
        )
        errno = ctypes.get_errno()
        if mapped == ctypes.c_void_p(-1).value:
            if not strict:
                _UNMAPPABLE_DATA_PAGES.add(page_addr)
                continue
            raise OSError(
                errno,
                f"failed to map data page at 0x{page_addr:x}; "
                "address already mapped, choose a different data address",
            )
        if mapped != page_addr:
            if not strict:
                _UNMAPPABLE_DATA_PAGES.add(page_addr)
                continue
            raise OSError(
                errno,
                f"failed to map data page at 0x{page_addr:x}; got 0x{mapped:x} instead",
            )
        _MAPPED_DATA_PAGES.add(page_addr)


def _branch_value(spec):
    if isinstance(spec, bool):
        return "predictable" if spec else "unpredictable"
    s = str(spec).strip().lower()
    return s if s in _BRANCH_CHOICES else None


def _branch_entry(key, spec, mem, regs, where):
    value = _branch_value(spec)
    if value is None:
        raise ValueError(
            f"branch{where} value for {key!r} must be "
            f"one of {', '.join(_BRANCH_CHOICES)}, got {spec!r}"
        )
    try:
        mem[int(str(key).strip(), 0)] = value
    except (TypeError, ValueError):
        regs[_canonical_data_reg_key(key)] = value


def _branch_config(config):
    branch = (config or {}).get("branch")
    if branch is None:
        return None
    if isinstance(branch, str):
        value = _branch_value(branch)
        if value is None:
            raise ValueError(
                f"branch prediction must be one of {', '.join(_BRANCH_CHOICES)}, "
                f"got {branch!r}"
            )
        return {"prediction": value, "mem": {}, "regs": {}}
    if isinstance(branch, bool):
        return {
            "prediction": "predictable" if branch else "unpredictable",
            "mem": {},
            "regs": {},
        }
    if not isinstance(branch, dict):
        raise ValueError(f"branch config must be a string or dict, got {branch!r}")
    mem, regs = {}, {}
    for key, spec in branch.items():
        if key in ("prediction", "mem", "regs"):
            continue
        if key == "default":
            raise ValueError(
                "unknown branch key 'default'; use 'prediction' "
                "(e.g. {'prediction': 'predictable'})"
            )
        _branch_entry(key, spec, mem, regs, "")
    for addr_key, spec in (branch.get("mem") or {}).items():
        _branch_entry(addr_key, spec, mem, regs, ".mem")
    for reg_key, spec in (branch.get("regs") or {}).items():
        value = _branch_value(spec)
        if value is None:
            raise ValueError(
                f"branch.regs value for {reg_key!r} must be "
                f"one of {', '.join(_BRANCH_CHOICES)}, got {spec!r}"
            )
        regs[_canonical_data_reg_key(reg_key)] = value
    prediction = branch.get("prediction", "unpredictable")
    if isinstance(prediction, bool):
        prediction = "predictable" if prediction else "unpredictable"
    else:
        prediction = _branch_value(prediction)
        if prediction is None:
            raise ValueError(
                f"branch prediction must be one of {', '.join(_BRANCH_CHOICES)}, "
                f"got {branch.get('prediction')!r}"
            )
    return {"prediction": prediction, "mem": mem, "regs": regs}


def _branch_global_distribution(branch_cfg, predictable):
    try:
        if isinstance(predictable, str):
            value = _branch_value(predictable)
            if value is not None:
                return value
    except Exception:
        pass
    try:
        pred = (branch_cfg or {}).get("prediction")
        if isinstance(pred, str):
            value = _branch_value(pred)
            if value is not None:
                return value
    except Exception:
        pass
    try:
        if bool(predictable):
            return "predictable"
    except Exception:
        pass
    return "unpredictable"


def _branch_reg_distribution(canon, branch_cfg, predictable):
    try:
        regs = (branch_cfg or {}).get("regs") or {}
        if canon in regs:
            value = _branch_value(regs[canon])
            if value is not None:
                return value
    except Exception:
        pass
    return _branch_global_distribution(branch_cfg, predictable)


def _branch_mem_distribution(addr, branch_cfg, predictable):
    try:
        mem = (branch_cfg or {}).get("mem") or {}
        a = int(addr)
        if a in mem:
            value = _branch_value(mem[a])
            if value is not None:
                return value
    except Exception:
        pass
    return _branch_global_distribution(branch_cfg, predictable)


def _branch_sample_index(n, rng, distribution, it):
    n = int(n)
    if n <= 1:
        return 0
    if distribution == "predictable":
        return int(it) % n
    return rng.randrange(n)


def _cond_branch_analysis(insns, arch=None):
    if arch is None:
        arch = get_arch()
    return arch.cond_branch_analysis(insns)


def _branch_deps(state, model, pinned_addrs, arch=None):
    deps = {}
    try:
        project = state.project
        if arch is None:
            arch = load_arch(project)
    except Exception:
        return deps
    regs = model.get("regs") or {}
    for addr in pinned_addrs:
        try:
            block = project.factory.block(addr)
            insns = list(block.capstone.insns)
        except Exception:
            continue
        if not insns or not _is_branch_mnemonic(insns[-1].mnemonic):
            continue
        cregs, mems = _cond_branch_analysis(insns, arch)
        read_addrs = set()
        for base_id, index_id, scale, disp in mems:
            canon = arch._canonical_reg
            base = None
            index = None
            if base_id:
                try:
                    base = regs.get(canon(insns[0].reg_name(base_id)))
                except Exception:
                    base = None
            if index_id:
                try:
                    index = regs.get(canon(insns[0].reg_name(index_id)))
                except Exception:
                    index = None
            if base is None and base_id:
                continue
            if index is None and index_id:
                continue
            read_addrs.add(
                ((base or 0) + (index or 0) * (scale or 1) + (disp or 0))
                & 0xFFFFFFFFFFFFFFFF
            )
        deps[addr] = {"regs": cregs, "mem": read_addrs}
    return deps


def _model_picks(models, predictable, rng, branch_cfg, iterations):
    if not models:
        return []
    dist = _branch_global_distribution(branch_cfg, predictable)
    n = len(models)
    if dist == "predictable":
        return [it % n for it in range(int(iterations))]
    randrange = rng.randrange
    return [randrange(n) for _ in range(int(iterations))]


def _collect_mem_addrs(models):
    addrs = set()
    for m in models:
        for a, _, _ in m["reads"] + m["writes"]:
            addrs.add(a)
    return addrs


def _recorded_data_addrs(solutions):
    data, control = set(), set()
    for sol in solutions or ():
        data |= set(sol.get("data_addrs") or ())
        control |= set(sol.get("control_addrs") or ())
    return data, control - data


def _static_table_addrs(
    project, start, end, elf_obj=None, max_addrs=64, solutions=None
):
    data, _control = _recorded_data_addrs(solutions)
    found = []
    seen = set()

    def _push(a):
        try:
            a = int(a)
        except (TypeError, ValueError):
            return
        if a not in seen:
            seen.add(a)
            found.append(a)

    try:
        arch = get_arch()
        md = arch.disassembler()
        md.detail = True
        try:
            _blob = project.loader.memory.load(
                int(start), max(int(end) - int(start), 1)
            )
        except Exception:
            _blob = b""
        if _blob:
            for _insn in md.disasm(bytes(_blob), int(start)):
                for _tgt in arch.mem_refs(_insn):
                    try:
                        if elf_obj is not None and getattr(elf_obj, "runtime_base", 0):
                            try:
                                _tgt_rt = elf_obj.runtime_addr(_tgt)
                            except Exception:
                                try:
                                    _cle_base = int(
                                        project.loader.main_object.mapped_base or 0
                                    )
                                    _off = int(elf_obj.runtime_base) - int(_cle_base)
                                    _tgt_rt = int(_tgt) + int(_off)
                                except Exception:
                                    _tgt_rt = int(_tgt)
                        else:
                            _tgt_rt = int(_tgt)
                    except Exception:
                        _tgt_rt = int(_tgt)
                    if int(_tgt) in data:
                        _push(_tgt_rt)
                if len(found) >= max_addrs:
                    break
    except Exception:
        pass

    return found


def _static_extra_addrs(extra_addrs=None):
    out = set()
    for a in extra_addrs or ():
        try:
            out.add(int(a, 0) if isinstance(a, str) else int(a))
        except (TypeError, ValueError):
            continue
    return out


def _apply_code_tier_defaults(
    arch, config, mem_levels, extra_addrs=None, code_addrs=None
):
    data_refs = _static_extra_addrs(extra_addrs)
    code = _static_extra_addrs(code_addrs)
    if not data_refs and not code:
        return mem_levels

    def _defaults(resolver):
        fn = getattr(arch, resolver, None)
        if fn is None:
            return {}
        try:
            levels = fn(config)
        except Exception:
            levels = None
        if not levels:
            return {}
        return {t: v for t, v in levels.items() if v is not None}

    out = dict(mem_levels or {})
    code_resolvers = (
        ("instruction_cache_levels", ("L1i",)),
        ("instruction_tlb_levels", ("TLBi",)),
    )
    data_resolvers = (
        ("data_cache_levels", ("L1d",)),
        ("data_tlb_levels", ("TLBd",)),
    )
    for addrs, resolvers in ((code, code_resolvers), (data_refs, data_resolvers)):
        defaults = {}
        for resolver, tiers in resolvers:
            for tier, rate in _defaults(resolver).items():
                if tier in tiers:
                    defaults.setdefault(tier, rate)
        if not defaults:
            continue
        for addr in addrs:
            spec = out.get(addr)
            spec = dict(spec) if isinstance(spec, dict) else {}
            for tier, rate in defaults.items():
                spec.setdefault(tier, rate)
            out[addr] = spec
    return out


def _mem_level_int_addrs(mem_levels):
    out = set()
    for k in mem_levels or {}:
        if isinstance(k, bool):
            continue
        if isinstance(k, int):
            out.add(k)
            continue
        try:
            s = str(k).strip()

            if not s:
                continue
            out.add(int(s, 0))
        except (TypeError, ValueError):
            continue
    return out


def _sample_values(values, distribution, it, rng):
    vals = list(values or [])
    if not vals:
        return 0
    if len(vals) == 1:
        return vals[0]
    if distribution == "predictable":
        return vals[int(it) % len(vals)]
    return vals[_branch_sample_index(len(vals), rng, distribution, it)]


def _fill_dynamic(
    buf,
    iterations,
    models,
    picks,
    reg_names,
    reg_col,
    val_col,
    mem_addrs,
    explicit_regs,
    explicit_mem,
    pin_regs,
    pin_mem,
    model0_mem,
    branch_cfg,
    predictable,
    rng,
):
    mask = 0xFFFFFFFFFFFFFFFF
    n = int(iterations)
    if n <= 0:
        return
    models = list(models) or [{"regs": {}, "reads": [], "writes": []}]
    picks = list(picks)[:n]
    if len(picks) < n:
        picks += [picks[-1] if picks else 0] * (n - len(picks))

    reg_src = {
        name: [(m["regs"].get(name) or 0) & mask for m in models] for name in reg_names
    }
    model0_regs = (models[0] or {}).get("regs") or {}

    for name in reg_names:
        if name in explicit_regs:
            continue
        src = reg_src[name]
        vals = [src[p] for p in picks]
        if name in pin_regs:
            pinned = model0_regs.get(name)
            if pinned is not None:
                vals = [pinned & mask] * n
        buf[reg_col[name], 1 : n + 1] = vals[::-1]

    mem_src = {}
    for a in mem_addrs:
        if a in explicit_mem or a not in val_col:
            continue
        mem_src[a] = [0] * len(models)
    for index, m in enumerate(models):
        for a, _, value in m["reads"] + m["writes"]:
            target = mem_src.get(a)
            if target is not None:
                target[index] = value & mask
    for a, src in mem_src.items():
        vals = [src[p] for p in picks]
        if a in pin_mem and a in model0_mem:
            vals = [int(model0_mem[a]) & mask] * n
        buf[val_col[a], 1 : n + 1] = vals[::-1]

    for name, value in explicit_regs.items():
        if name not in reg_col:
            continue
        if isinstance(value, (list, tuple)):
            dist = _branch_reg_distribution(name, branch_cfg, predictable)
            vals = [_sample_values(value, dist, it, rng) & mask for it in range(n)]
        else:
            vals = [int(value) & mask] * n
        buf[reg_col[name], 1 : n + 1] = vals[::-1]

    for a, value in explicit_mem.items():
        if a not in val_col:
            continue
        if isinstance(value, (list, tuple)):
            dist = _branch_mem_distribution(a, branch_cfg, predictable)
            vals = [_sample_values(value, dist, it, rng) & mask for it in range(n)]
        else:
            vals = [int(value) & mask] * n
        buf[val_col[a], 1 : n + 1] = vals[::-1]


def _per_iter_data(
    models,
    iterations,
    levels,
    predictable,
    rng,
    arch=None,
    explicit=None,
    branch_cfg=None,
    mem_levels=None,
    extra_addrs=None,
    tlb_levels=None,
):
    if arch is None:
        arch = get_arch()
    harness_regs = arch._HARNESS_REGS

    mem_levels = mem_levels or {}

    explicit = explicit or {}

    def _norm_explicit_list(val):
        return [_to_u64(x) for x in val] or [_to_u64(0)]

    explicit_regs = {}
    for key, val in (explicit.get("regs") or {}).items():
        canon = _canonical_data_reg_key(key)
        if canon in (harness_regs or set()):
            continue
        explicit_regs[canon] = (
            _norm_explicit_list(val)
            if isinstance(val, (list, tuple))
            else _to_u64(_to_int_or(val, 0))
        )
    explicit_mem = {}
    for addr_str, val in (explicit.get("mem") or {}).items():
        s = str(addr_str).strip()
        if s.endswith(":"):
            s = s[:-1].strip()
        try:
            base = int(s, 0)
        except (TypeError, ValueError):
            continue
        explicit_mem[base] = (
            _norm_explicit_list(val)
            if isinstance(val, (list, tuple))
            else _to_u64(_to_int_or(val, 0))
        )

    pin_regs = set()
    pin_mem = set()
    mem_cfg = (branch_cfg or {}).get("mem") or {}
    regs_cfg = (branch_cfg or {}).get("regs") or {}
    if mem_cfg or regs_cfg:
        for addr, value in mem_cfg.items():
            if value != "predictable":
                continue

            for m in models:
                deps = (m.get("branch_deps") or {}).get(addr)
                if not deps:
                    continue
                pin_regs |= set(deps.get("regs") or ())
                pin_mem |= set(deps.get("mem") or ())

        for r, value in regs_cfg.items():
            if value == "predictable":
                pin_regs.add(r)

        for addr, value in mem_cfg.items():
            if value == "predictable":
                try:
                    pin_mem.add(int(addr))
                except (TypeError, ValueError):
                    pass

    model0 = models[0] if models else {}
    model0_mem = {}
    for a, _, value in model0.get("reads", []) + model0.get("writes", []):
        model0_mem[a] = value
    for a in list(pin_mem):
        if a not in model0_mem and a not in explicit_mem:
            pin_mem.discard(a)

    reg_names = []
    seen = set()
    for m in models:
        for name in m["regs"]:
            if name in harness_regs or name in seen:
                continue
            seen.add(name)
            reg_names.append(name)
    for name in explicit_regs:
        if name in harness_regs or name in seen:
            continue
        seen.add(name)
        reg_names.append(name)

    l1i_addrs = sorted(
        a
        for a, spec in (mem_levels or {}).items()
        if isinstance(spec, dict)
        and (spec.get("L1i") is not None or spec.get("TLBi") is not None)
    )
    l1i_set = set(l1i_addrs)
    mem_addrs = sorted(
        (
            _collect_mem_addrs(models)
            | set(explicit_mem)
            | _mem_level_int_addrs(mem_levels)
            | _static_extra_addrs(extra_addrs)
        )
        - l1i_set
    )

    reg_col = {name: i for i, name in enumerate(reg_names)}
    val_col = {}
    col = len(reg_names)
    for a in mem_addrs:
        val_col[a] = col
        col += 1

    ncols = col
    naddr = len(mem_addrs)
    k = iterations + 1
    K = max(k, 2 * naddr) if naddr else k

    total_cols = ncols + 2 * naddr
    buf = np.zeros((total_cols, K), dtype=np.uint64)

    static = (
        len(models) == 1
        and predictable
        and not mem_cfg
        and not regs_cfg
        and not any(isinstance(v, (list, tuple)) for v in list(explicit_regs.values()))
        and not any(isinstance(v, (list, tuple)) for v in explicit_mem.values())
    )
    if static:
        m = models[0]
        regs = m["regs"]
        for name in reg_names:
            if name in explicit_regs:
                value = explicit_regs[name]
            else:
                value = regs.get(name, 0)
            try:
                buf[reg_col[name], 1 : iterations + 1] = int(value) & 0xFFFFFFFFFFFFFFFF
            except (TypeError, ValueError):
                buf[reg_col[name], 1 : iterations + 1] = 0
        for a in mem_addrs:
            if a in explicit_mem:
                value = explicit_mem[a]
            else:
                value = model0_mem.get(a, 0)
            try:
                buf[val_col[a], 1 : iterations + 1] = int(value) & 0xFFFFFFFFFFFFFFFF
            except (TypeError, ValueError):
                buf[val_col[a], 1 : iterations + 1] = 0
    else:
        _fill_dynamic(
            buf,
            iterations,
            models,
            picks=_model_picks(models, predictable, rng, branch_cfg, iterations),
            reg_names=reg_names,
            reg_col=reg_col,
            val_col=val_col,
            mem_addrs=mem_addrs,
            explicit_regs=explicit_regs,
            explicit_mem=explicit_mem,
            pin_regs=pin_regs,
            pin_mem=pin_mem,
            model0_mem=model0_mem,
            branch_cfg=branch_cfg,
            predictable=predictable,
            rng=rng,
        )

    meta = {
        "ncols": ncols,
        "k": k,
        "K": K,
        "reg_col": reg_col,
        "val_col": val_col,
        "mem_addrs": mem_addrs,
        "l1i_addrs": l1i_addrs,
        "reg_names": reg_names,
        "levels": levels,
        "mem_levels": mem_levels or {},
        "tlb_levels": tlb_levels or {},
        "evict_table_col": ncols if naddr else None,
    }
    return buf, meta


def _fill_evict_tables(buf, meta, data_ptr):
    addrs = meta["mem_addrs"]
    if not addrs:
        return
    n = len(addrs)
    K = meta["K"]
    et = meta["evict_table_col"]
    for i, a in enumerate(addrs):
        buf[et, i] = a
        buf[et, n + i] = (data_ptr + meta["val_col"][a] * K * 8) & 0xFFFFFFFFFFFFFFFF


def _arch_asm(name, meta, data_ptr, arch=None, **kw):
    if arch is None:
        arch = get_arch()
    fn = getattr(arch, name)
    return fn(meta, data_ptr, **kw) if kw else fn(meta, data_ptr)


def _iterations_config(config):
    spec = (config or {}).get("iterations")
    pinned = None
    lo = hi = None
    if isinstance(spec, dict):
        pinned = spec.get("count")
        lo = spec.get("min")
        hi = spec.get("max")
    elif spec is not None:
        pinned = spec
    return pinned, lo, hi


def _backend_options(config, name):
    spec = (config or {}).get("backend")
    if not isinstance(spec, dict):
        return {}
    opts = spec.get(name)
    return opts if isinstance(opts, dict) else {}


def _backend_option(config, name, key, default=None):
    opts = _backend_options(config, name)
    if key in opts:
        return opts[key]
    return default


def _probe_plan(config, backend="loop"):
    target = _backend_option(config, backend, "target", 0.005)
    _pinned, lo, hi = _iterations_config(config)
    lo = _DEFAULT_ITER_MIN if lo is None else lo
    hi = _DEFAULT_ITER_MAX if hi is None else hi
    return float(target), int(lo), int(hi)


def _plan_iterations(measured, target_rel_se, min_iterations, max_iterations):
    arr = np.asarray(measured, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return int(min_iterations)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if mean <= 0 or std <= 0 or not np.isfinite(mean):
        return int(min_iterations)
    n = (1.96 * std / (target_rel_se * mean)) ** 2
    return int(np.clip(np.ceil(n), min_iterations, max_iterations))


def _timed_regs_avoid(code, setup, teardown, data, arch):
    avoid = set()
    used = arch.used_regs
    for chunk in (code, setup, teardown):
        if chunk:
            try:
                avoid |= set(used(chunk))
            except Exception:
                pass
    if data:
        for r in data.get("regs") or {}:
            try:
                avoid |= set(used(str(arch.canonical_data_reg(r))))
            except Exception:
                pass
            try:
                avoid.add(str(r).strip().lower())
                avoid.add(str(arch.canonical_data_reg(r)).strip().lower())
            except Exception:
                pass
        try:
            avoid = {arch._canonical_reg(r) for r in avoid}
        except Exception:
            pass
    return avoid


def _write_jit_map(addr, size, name):
    try:
        label = (
            str(name or "loop").strip().replace("\n", "_").replace(" ", "_") or "loop"
        )
        write_perf_map([(int(addr), int(size), f"perf_bench_loop:{label}")])
    except Exception:
        pass


def _build_loop_asm(
    iterations,
    mode,
    code,
    setup,
    teardown,
    models,
    events,
    indices,
    levels,
    predictable,
    rng,
    arch,
    data=None,
    branch_cfg=None,
    mem_levels=None,
    extra_addrs=None,
    tlb_levels=None,
):
    events = _names(events)
    data_script = ""
    data2_script = ""
    data_iter = ""
    per_iter_buf = None
    meta = None
    has_explicit_mem = bool(data and data.get("mem"))
    has_explicit_dist = bool(
        data
        and any(
            isinstance(v, (list, tuple))
            for v in list((data.get("regs") or {}).values())
            + list((data.get("mem") or {}).values())
        )
    )
    _extra_set = _static_extra_addrs(extra_addrs)
    _level_addrs = _mem_level_int_addrs(mem_levels)
    if models or has_explicit_mem or has_explicit_dist or _extra_set or _level_addrs:
        buf, meta = _per_iter_data(
            models,
            iterations,
            levels,
            predictable,
            rng,
            arch,
            explicit=data,
            branch_cfg=branch_cfg,
            mem_levels=mem_levels,
            extra_addrs=_extra_set,
            tlb_levels=tlb_levels,
        )
        per_iter_buf = buf
        if meta["reg_col"] or meta["mem_addrs"] or meta.get("l1i_addrs"):
            _fill_evict_tables(buf, meta, buf.ctypes.data)
            prime = _arch_asm("prime_asm", meta, buf.ctypes.data, arch)
            reload_regs = {}
            for reg, val in ((data or {}).get("regs") or {}).items():
                try:
                    canon = _canonical_data_reg_key(reg)
                except Exception:
                    canon = str(reg).strip().lower()
                if canon not in (meta.get("reg_col") or {}):
                    reload_regs[reg] = val
            reload_data = {"regs": reload_regs} if reload_regs else None
            explicit_reload = arch.data_reload_asm(reload_data) if reload_data else ""
            if mode == "throughput":
                data_script = _arch_asm(
                    "steer_asm",
                    meta,
                    buf.ctypes.data,
                    arch,
                    mem_levels=mem_levels,
                    write_values=False,
                )
                data_iter = _arch_asm("per_iter_asm", meta, buf.ctypes.data, arch)
                data2_script = explicit_reload
            else:
                data_script = _arch_asm(
                    "steer_asm", meta, buf.ctypes.data, arch, mem_levels=mem_levels
                )
                parts = [p for p in (prime, explicit_reload) if p]
                data2_script = "\n".join(parts)
        else:
            if data:
                data2_script = arch.data_reload_asm(data)
    elif data:
        data2_script = arch.data_reload_asm(data)

    avoid = _timed_regs_avoid(code, None, None, data, arch)
    avoid.add(arch._PRIME_SCRATCH_REG)
    if meta is not None:
        for _rn in meta.get("reg_col", ()):
            avoid.add(str(_rn).strip().lower())
            try:
                avoid.add(arch._canonical_reg(_rn))
            except Exception:
                pass
    body = " ".join(c for c in (code, setup, teardown, data2_script, data_iter) if c)
    body_norm = " ".join(body.split()).lower()
    may_call = mode == "throughput" or f" {body_norm} ".find(" call ") >= 0
    t0, t1 = arch.timing(events, indices, callee_saved=may_call, avoid=avoid)
    guard_pre, guard_post = arch.timed_guard(events, callee_saved=may_call, avoid=avoid)
    if guard_pre or guard_post:
        code = "\n".join(p for p in (guard_pre, code, guard_post) if p)
    tlb_restore = _arch_asm("tlb_restore_asm", meta, None, arch)
    full_asm = arch._BENCH[mode].format(
        code=code,
        setup=setup,
        teardown=teardown,
        data=data_script,
        data2=data2_script,
        data_iter=data_iter,
        t0=t0,
        t1=t1,
        tlb=tlb_restore,
    )
    return full_asm, (meta if per_iter_buf is not None else None), per_iter_buf


def _asm_error(code, ex):
    msg = str(ex)
    if "Invalid operand" in msg or "KS_ERR_ASM_INVALIDOPERAND" in msg:
        raise ValueError(
            f"invalid assembly for {code!r}: {ex}. "
            "Note `idiv`/`div` take a single r/m operand (e.g. `idiv ecx` "
            "with `--data.eax=.. --data.edx=.. "
            "--data.ecx=..` and "
            "`--backend loop`), and `imul` needs 2-3 operands "
            "(e.g. `imul eax, eax, 42`)."
        ) from ex
    raise ex


def _check_target_runnable(fn, args, map_name=None, key=None):
    if key is not None and key in _RUNNABLE_HARNESS:
        return
    try:
        trips = max(1, int(getattr(args, "iterations", 1)))
    except (TypeError, ValueError):
        trips = 1
    probe = _Args(trips, args.inputs, args.outputs, args.data)
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        try:
            fn(ctypes.byref(probe))
        except BaseException:
            os._exit(1)
        os._exit(0)

    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        signum = os.WTERMSIG(status)
        what = _FAULT_MESSAGES.get(signum, f"signal {signum}")
        where = f" '{map_name}'" if map_name else "the target"
        raise ValueError(
            f"{where} faulted with {what} when called on its own, so it "
            "cannot be measured in isolation. Benchmark a function that runs "
            "without the program's setup, or restrict the benchmark to a "
            "region (e.g. 'foo_begin..foo_end')."
        )
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
        where = f" '{map_name}'" if map_name else "the target"
        raise ValueError(f"the measurement harness raised while running{where}.")
    if key is not None:
        _RUNNABLE_HARNESS.add(key)


def _collect(
    config,
    iterations,
    runs,
    mode,
    code,
    setup,
    teardown,
    models,
    events,
    indices,
    overhead,
    levels,
    predictable,
    rng,
    arch,
    data=None,
    map_name=None,
    branch_cfg=None,
    mem_levels=None,
    extra_addrs=None,
    tlb_levels=None,
    debug=False,
):
    events = _names(events)
    n_events = len(events)
    try:
        full_asm, _meta, per_iter_buf = _build_loop_asm(
            iterations,
            mode,
            code,
            setup,
            teardown,
            models,
            events,
            indices,
            levels,
            predictable,
            rng,
            arch,
            data=data,
            branch_cfg=branch_cfg,
            mem_levels=mem_levels,
            extra_addrs=extra_addrs,
            tlb_levels=tlb_levels,
        )
    except Exception as ex:
        _asm_error(code, ex)
    try:
        code_bytes = arch.assemble(full_asm)
    except Exception as ex:
        _asm_error(code, ex)

    page = mmap.mmap(
        -1,
        len(code_bytes),
        prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
    )
    page.write(code_bytes)
    _JIT_PAGES.append(page)
    while len(_JIT_PAGES) > _JIT_PAGES_MAX:
        _JIT_PAGES.pop(0)

    try:
        _write_jit_map(
            ctypes.addressof(ctypes.c_char.from_buffer(page)),
            len(code_bytes),
            map_name or ",".join(events),
        )
    except Exception:
        pass

    inputs = np.zeros((iterations + 1, 1), dtype=np.uint64)
    n_samples = 1 if mode == "throughput" else iterations
    outputs = np.zeros((n_samples, n_events), dtype=np.uint64)

    data_ptr = per_iter_buf.ctypes.data if per_iter_buf is not None else None
    args = _Args(iterations, inputs.ctypes.data, outputs.ctypes.data, data_ptr)
    fn = ctypes.CFUNCTYPE(None, ctypes.POINTER(_Args))(
        ctypes.addressof(ctypes.c_char.from_buffer(page))
    )

    oh = np.nan_to_num(
        np.broadcast_to(np.asarray(overhead, dtype=np.float64), (n_events,)),
        nan=0.0,
    )
    if debug:
        _debug_log(
            f"harness asm (iterations={iterations} runs={runs} mode={mode} "
            f"events={events}):\n{_left_align_asm(full_asm)}"
        )
    _check_target_runnable(
        fn, args, map_name, key=(hash(code_bytes), int(iterations), int(runs))
    )
    if mode == "throughput":
        diffs = np.full((runs, n_events), np.nan)
        with (
            _numa_guard(config),
            _affinity_guard(config),
            _priority_guard(config),
        ):
            for i in range(runs):
                fn(ctypes.byref(args))
                sub = np.asarray(outputs, dtype=np.float64)[0] - oh
                diffs[i, sub > 0] = sub[sub > 0]
                if debug:
                    _debug_log(f"run {i}: {dict(zip(events, diffs[i].tolist()))}")
        with np.errstate(invalid="ignore"):
            return diffs

    diffs = np.full((runs, iterations, n_events), np.nan)
    with (
        _numa_guard(config),
        _affinity_guard(config),
        _priority_guard(config),
    ):
        for i in range(runs):
            fn(ctypes.byref(args))
            sub = np.asarray(outputs, dtype=np.float64) - oh
            diffs[i, sub > 0] = sub[sub > 0]
            if debug:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        per_min = np.nanmin(diffs[i], axis=0)
                        per_med = np.nanmedian(diffs[i], axis=0)
                except Exception:
                    per_min = per_med = None
                _debug_log(
                    f"run {i}: min={dict(zip(events, np.asarray(per_min).tolist()))} "
                    f"median={dict(zip(events, np.asarray(per_med).tolist()))}"
                )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(diffs, axis=0)


def _normalize_groups(event):
    default = [[_DURATION_TIME]]
    if event is None:
        return default
    if isinstance(event, str):
        parts = _split_list(event)
        return [parts] if parts else default
    if isinstance(event, (list, tuple)):
        if len(event) == 0:
            return default
        if any(isinstance(g, (list, tuple)) for g in event):
            groups = []
            for group in event:
                if isinstance(group, (list, tuple)):
                    names = _split_list(group)
                    if names:
                        groups.append(names)
                else:
                    parts = _split_list(group)
                    if parts:
                        groups.append(parts)
            return groups or default
        names = _split_list(event)
        return [names] if names else default
    s = str(event).strip()
    return [[s]] if s else default


def _names(events):
    return _split_list(events) or [_DURATION_TIME]


def _pin_default_affinity(config, event):
    entries = _thread_entries((config or {}).get("thread"))
    if not entries or any(e.get("affinity") is not None for e in entries):
        return
    groups = _normalize_groups(event)
    try:
        groups = [core.with_leader(g) for g in groups]
    except Exception:
        pass
    cpu = core.default_affinity(
        [g for g in groups if g], _parse_numa(entries[0].get("numa"))
    )
    if cpu is None:
        return
    for entry in entries:
        entry["affinity"] = cpu


def _records(
    file_name, name, mode, result, group, divisor=1, operations=1, iterations=None
):
    def _base(samples):
        return {
            "file": file_name,
            "name": name,
            "mode": mode,
            "iterations": iterations,
            "samples": samples,
            "operations": operations,
        }

    records = []
    if isinstance(result, pd.DataFrame):
        for samples, (_, row) in enumerate(result.iterrows()):
            rec = _base(samples)
            for ev in group:
                rec[ev] = row[ev] / divisor
            records.append(rec)
    else:
        for samples, value in enumerate(result):
            rec = _base(samples)
            for ev in group:
                rec[ev] = value / divisor
            records.append(rec)
    return records


def _combine_results(records_lists):
    frames = [pd.DataFrame(r) for r in records_lists if r]
    if not frames:
        return pd.DataFrame()

    def _merge_group(group):
        merged = group[0]
        for other in group[1:]:
            merged = merged.merge(other, on=list(_COMBINE_KEYS), how="outer")
        return merged

    groups = {}
    for fr in frames:
        try:
            key = tuple(str(fr[c].iloc[0]) for c in _IDENTITY_COLUMNS)
        except Exception:
            key = (None,) * len(_IDENTITY_COLUMNS)
        groups.setdefault(key, []).append(fr)
    merged = [_merge_group(g) for g in groups.values()]
    df = merged[0] if len(merged) == 1 else pd.concat(merged, ignore_index=True)
    cols = list(_COMBINE_KEYS) + [c for c in df.columns if c not in _COMBINE_KEYS]
    df = df[cols].sort_values([*_IDENTITY_COLUMNS, "samples"])
    return df.set_index(list(_IDENTITY_COLUMNS))


def _mode_backends(backend):
    if not isinstance(backend, dict):
        return {}
    out = {}
    for mode, name in backend.items():
        if name is None:
            continue
        out[str(mode).strip()] = name
    return out


def _resolve_backend(backend, config, default):
    if backend is None:
        backend = (config or {}).get("backend", default)
        if isinstance(backend, dict):
            backend = default
    if backend is None:
        backend = default
    name = str(backend).strip()
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {', '.join(_BACKENDS)}"
        )
    return name


def _resolve_unroll_n(unroll_n, config, backend="unroll"):
    default_n = _default_unroll_n()
    if unroll_n is None:
        unroll_n = _backend_option(config, backend, "count", default_n)
    if unroll_n is None:
        unroll_n = default_n
    try:
        unroll_n = int(unroll_n)
    except (TypeError, ValueError) as e:
        raise ValueError(f"unroll_n must be an integer, got {unroll_n!r}") from e
    if unroll_n < 1:
        raise ValueError(f"unroll_n must be >= 1, got {unroll_n!r}")
    return unroll_n


def _resolve_runs(config, backend):
    runs = _backend_option(config, backend, "runs", 10)
    try:
        runs = int(runs)
    except (TypeError, ValueError) as e:
        raise ValueError(f"backend runs must be an integer, got {runs!r}") from e
    if runs < 1:
        raise ValueError(f"backend runs must be >= 1, got {runs!r}")
    return runs


def _resolve_probe_runs(config, backend, runs):
    probes = _backend_option(config, backend, "probes", 3)
    try:
        probes = int(probes)
    except (TypeError, ValueError) as e:
        raise ValueError(f"backend probes must be an integer, got {probes!r}") from e
    return max(1, min(runs, probes))


def _used_backend_options(config, backend):
    runs = _resolve_runs(config, backend)
    target, _lo, _hi = _probe_plan(config, backend)
    used = {
        "probes": _resolve_probe_runs(config, backend, runs),
        "runs": runs,
        "target": target,
    }
    if backend == "unroll":
        used["count"] = _resolve_unroll_n(None, config, backend)
    return {k: used[k] for k in sorted(used)}


def _backend_resolved(config, backend):
    spec = (config or {}).get("backend")
    return isinstance(spec, dict) and set(spec) == {backend}


def _select_backend(config, backend, unroll_n=None):
    used = _used_backend_options(config, backend)
    if backend == "unroll":
        used["count"] = _resolve_unroll_n(unroll_n, config, backend)
    config["backend"] = {backend: used}
    return used


def _bench(
    config,
    mode,
    code,
    setup,
    teardown,
    solutions,
    overhead=0,
    events=None,
    data=None,
    addr_range=None,
    arch=None,
    map_name=None,
    iterations_tracker=None,
    models_tracker=None,
    extra_addrs=None,
    code_addrs=None,
    debug=False,
):
    if arch is None:
        arch = get_arch()
    samples = config.get("samples", 100)
    _backend_name = config.get("backend")
    _backend_name = _backend_name if isinstance(_backend_name, str) else "loop"
    runs = _resolve_runs(config, _backend_name)

    events = core.with_leader(_names(events))
    group = core.is_group(events)
    try:
        force_typ = core.choose_type(events)
    except Exception:
        force_typ = None
    try:
        _req = _parse_affinity(_affinity_spec(config))
    except Exception:
        _req = None
    _restore_aff = core.pin_pmu(config, force_typ, _req)

    counters = []
    indices = []
    try:
        counters, indices = core.open_counters(events, force_typ, group)
    except Exception:
        _restore_aff()
        raise
    if not counters and "rdpmc" in str(code).lower():
        _rdpmc = core.enable_rdpmc()
        if _rdpmc is not None:
            counters.append(_rdpmc)

    def close():
        for counter in counters:
            try:
                counter.disable()
            except Exception:
                pass
            try:
                counter.close()
            except Exception:
                pass

    try:
        return _bench_measure(
            config,
            mode,
            code,
            setup,
            teardown,
            solutions,
            overhead,
            events,
            data,
            arch,
            map_name,
            iterations_tracker,
            extra_addrs,
            code_addrs,
            debug,
            close,
            _restore_aff,
            _backend_name,
            group,
            force_typ,
            counters,
            samples,
            runs,
            indices,
            addr_range,
            models_tracker,
        )
    finally:
        close()
        _restore_aff()


def _bench_measure(
    config,
    mode,
    code,
    setup,
    teardown,
    solutions,
    overhead,
    events,
    data,
    arch,
    map_name,
    iterations_tracker,
    extra_addrs,
    code_addrs,
    debug,
    close,
    _restore_aff,
    _backend_name,
    group,
    force_typ,
    counters,
    samples,
    runs,
    indices,
    addr_range,
    models_tracker=None,
):
    rng = _make_rng(config)
    seed_int = _config_seed_int(config)
    branch_cfg = _branch_config(config)
    predictable = (
        config.get("branch", "unpredictable") == "predictable"
        if branch_cfg is None
        else branch_cfg["prediction"] == "predictable"
    )
    levels = arch.data_cache_levels(config)
    mem_levels = arch.resolve_mem_cache(config)
    mem_levels = _apply_code_tier_defaults(
        arch, config, mem_levels, extra_addrs, code_addrs
    )
    tlb_levels = {}
    for resolver in ("data_tlb_levels", "instruction_tlb_levels"):
        fn = getattr(arch, resolver, None)
        if fn is None:
            continue
        try:
            resolved = fn(config)
        except Exception:
            resolved = None
        if resolved:
            tlb_levels.update({k: v for k, v in resolved.items() if v is not None})

    models = []
    if debug:
        _debug_json("measurement config", config)
        _debug_log(
            f"measuring {map_name or code[:40]!r} mode={mode} events={list(events)}"
        )
    if solutions:
        models = _models_cached(solutions, branch_cfg, arch)
        if debug:
            try:
                _debug_json("synthesised data (models)", models)
            except Exception:
                pass
        if models:
            try:
                stack_base = int(getattr(arch, "_STACK_ADDR", 0x7FFF00000000))
            except (TypeError, ValueError):
                stack_base = 0x7FFF00000000
            min_addr = _min_mmap_addr()

            def _keep_addr(a):
                try:
                    a = int(a)
                except (TypeError, ValueError):
                    return False
                if a < min_addr:
                    return False
                if a >= stack_base:
                    return False
                if addr_range is not None:
                    try:
                        if addr_range[0] <= a < addr_range[1]:
                            return False
                    except (TypeError, ValueError):
                        pass
                return True

            for m in models:
                m["reads"] = [(a, ln, v) for a, ln, v in m["reads"] if _keep_addr(a)]
                m["writes"] = [(a, ln, v) for a, ln, v in m["writes"] if _keep_addr(a)]
            if models_tracker is not None and models:
                try:
                    models_tracker.append(models)
                except Exception:
                    pass
        if models:
            addrs = {str(a): 0 for m in models for a, _, _ in m["reads"] + m["writes"]}
            steered = _static_extra_addrs(extra_addrs)
            steered |= _mem_level_int_addrs(mem_levels)
            for a in steered:
                addrs.setdefault(str(a), 0)
            _map_data_pages({"mem": addrs}, strict=False)
        _map_scratch_reg_pages(models)

    if data:
        _map_data_pages(data)
        data_setup = arch.data_setup_asm(data)
        setup = data_setup + ("\n" + setup if setup else "\n")

    try:
        _pinned_iterations, _lo, _hi = _iterations_config(config)
        if _pinned_iterations is not None:
            iterations = int(_pinned_iterations)
        else:
            target_rel_se, min_iterations, max_iterations = _probe_plan(
                config, _backend_name
            )
            probe_runs = _resolve_probe_runs(config, _backend_name, runs)
            probe = _collect(
                config,
                min_iterations,
                probe_runs,
                mode,
                code,
                setup,
                teardown,
                models,
                events,
                indices,
                overhead,
                levels,
                predictable,
                rng,
                arch,
                data=data,
                map_name=map_name,
                branch_cfg=branch_cfg,
                mem_levels=mem_levels,
                extra_addrs=extra_addrs,
                tlb_levels=tlb_levels,
                debug=debug,
            )
            probe_series = probe[:, 0] if probe.ndim > 1 else probe
            iterations = _plan_iterations(
                probe_series, target_rel_se, min_iterations, max_iterations
            )
        if iterations_tracker is not None:
            try:
                iterations_tracker.append(int(iterations))
            except Exception:
                pass

        measure_runs = int(samples) if mode == "throughput" else runs
        data_series = _collect(
            config,
            iterations,
            measure_runs,
            mode,
            code,
            setup,
            teardown,
            models,
            events,
            indices,
            overhead,
            levels,
            predictable,
            rng,
            arch,
            data=data,
            map_name=map_name,
            branch_cfg=branch_cfg,
            mem_levels=mem_levels,
            extra_addrs=extra_addrs,
            tlb_levels=tlb_levels,
            debug=debug,
        )
    finally:
        close()
        _restore_aff()

    df = pd.DataFrame(data_series, columns=events)
    if df.empty:
        df = pd.DataFrame(
            {ev: [float("nan")] * int(samples) for ev in events},
        )
    if seed_int is None:
        return df.sample(n=min(int(samples), len(df))), iterations
    return (
        df.sample(n=min(int(samples), len(df)), random_state=seed_int % (2**32)),
        iterations,
    )


def _symbolic_options(state):
    import angr

    state.options.discard(angr.options.LAZY_SOLVES)
    for opt in (
        angr.options.TRACK_CONSTRAINTS,
        angr.options.TRACK_JMP_ACTIONS,
        angr.options.TRACK_ACTION_HISTORY,
        angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
        angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
        angr.options.NO_SYMBOLIC_JUMP_RESOLUTION,
        angr.options.STRICT_PAGE_ACCESS,
        angr.options.CONSERVATIVE_READ_STRATEGY,
        angr.options.SYMBOLIC_WRITE_ADDRESSES,
        angr.options.NO_CROSS_INSN_OPT,
    ):
        state.options.add(opt)


def _symbolic_regs(proj, state, prefix=""):
    import claripy

    sym_regs = {}
    ip_name = proj.arch.register_names.get(proj.arch.ip_offset)
    sp_name = proj.arch.register_names.get(proj.arch.sp_offset)
    for reg in proj.arch.register_list:
        if not reg.general_purpose or reg.name == ip_name or reg.name == sp_name:
            continue
        sym = claripy.BVS(f"{prefix}{reg.name}", reg.size * 8)
        state.registers.store(reg.name, sym)
        sym_regs[reg.name] = sym
    return sym_regs


def _track_mem_access(state):
    import angr

    def _on_read(s):
        if "reads" not in s.globals:
            s.globals["reads"] = []
        s.globals["reads"].append(
            (
                s.inspect.mem_read_address,
                s.inspect.mem_read_length,
                s.inspect.mem_read_expr,
            )
        )
        _record_access(s, s.inspect.mem_read_address)

    def _on_write(s):
        if "writes" not in s.globals:
            s.globals["writes"] = []
        s.globals["writes"].append(
            (
                s.inspect.mem_write_address,
                s.inspect.mem_write_length,
                s.inspect.mem_write_expr,
            )
        )
        _record_access(s, s.inspect.mem_write_address)

    state.inspect.b("mem_read", when=angr.BP_AFTER, action=_on_read)
    state.inspect.b("mem_write", when=angr.BP_AFTER, action=_on_write)


def _record_access(state, addr):
    accesses = state.globals.get("accesses") or []
    accesses.append((addr, state.inspect.instruction))
    state.globals["accesses"] = accesses


def _track_branches(state):
    import angr

    def _on_exit(s):
        exits = s.globals.get("exits") or []
        exits.append((s.inspect.exit_target, s.inspect.instruction))
        s.globals["exits"] = exits
        guard = s.inspect.exit_guard
        if guard is None or getattr(guard, "concrete", False):
            return
        if str(s.inspect.exit_jumpkind).lower() not in _BRANCH_JUMPKINDS:
            return
        s.globals["branches"] = s.globals.get("branches", 0) + 1

    state.inspect.b("exit", when=angr.BP_AFTER, action=_on_exit)


def _track_state(state, track):
    if track is None:
        return
    try:
        track(state)
    except Exception:
        pass


def _map_state_mem_pages(state, mem):
    seen = set()
    for addr_str in mem or {}:
        try:
            s = str(addr_str).strip()
            if s.endswith(":"):
                s = s[:-1].strip()
            a = int(s, 0) & ~0xFFF
        except (TypeError, ValueError):
            continue
        if a in seen:
            continue
        seen.add(a)
        try:
            state.memory.map_region(a, 0x1000, 7, init_zero=True)
        except Exception:
            pass


def _eval_addr(state, value):
    try:
        addr = int(state.solver.eval(value))
    except Exception:
        return None
    return addr if 0 <= addr < (1 << 64) else None


def _recorded_stats(state):
    control_insns = set()
    branches = set()
    for target, insn in state.globals.get("exits") or ():
        try:
            insn = int(insn)
        except (TypeError, ValueError):
            insn = None
        if insn is not None:
            control_insns.add(insn)
        addr = _eval_addr(state, target)
        if addr is not None:
            branches.add(addr)
    data, control = set(), set()
    for addr, insn in state.globals.get("accesses") or ():
        value = _eval_addr(state, addr)
        if value is None:
            continue
        try:
            is_control = int(insn) in control_insns
        except (TypeError, ValueError):
            is_control = False
        (control if is_control else data).add(value)
    return data - branches, control - data, branches


def _solution_records(states, sym_regs, keep=None):
    results = []
    for found in states:
        reg = {n: s for n, s in sym_regs.items() if keep is None or keep(found, n, s)}
        data_addrs, control_addrs, branch_addrs = _recorded_stats(found)
        results.append(
            {
                "state": found,
                "reg": reg,
                "branches": int(found.globals.get("branches", 0) or 0),
                "reads": list(found.globals.get("reads", [])),
                "writes": list(found.globals.get("writes", [])),
                "data_addrs": data_addrs,
                "control_addrs": control_addrs,
                "branch_addrs": branch_addrs,
            }
        )
    return results


def _return_sym(state):
    try:
        ret_reg = state.arch.register_names[state.arch.ret_offset]
        return state.registers.load(ret_reg)
    except Exception:
        return None


def _return_values(sym):
    leaves = []
    pending = [sym]
    while pending:
        node = pending.pop()
        if getattr(node, "op", None) == "If":
            pending.append(node.args[1])
            pending.append(node.args[2])
        elif getattr(node, "op", None) == "BVV":
            leaves.append(node.args[0])
        else:
            return []
    return sorted(set(leaves))


def _normalize_target(target):
    if target is None:
        return None
    if isinstance(target, (list, tuple)):
        parts = [str(p).strip() for p in target]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"invalid target {target!r}; expected 'name' or ('begin', 'end')"
            )
        return f"{parts[0]}..{parts[1]}"
    name = str(target).strip()
    if not name:
        return None
    if ".." in name:
        raise ValueError(
            f"invalid target {target!r}; pass a region as a ('begin', 'end') pair, "
            "e.g. ('hot_begin', 'hot_end')"
        )
    return name


def _modes(mode):
    if isinstance(mode, str) or not isinstance(mode, (list, tuple)):
        raise TypeError(
            "mode must be a list, e.g. ['latency'] or ['latency', 'throughput']"
        )
    modes = []
    for m in mode:
        if m not in _MODES:
            raise ValueError(f"unknown mode {m!r}; expected 'latency' or 'throughput'")
        if m not in modes:
            modes.append(m)
    if not modes:
        raise ValueError("mode must name at least one of 'latency', 'throughput'")
    return modes


def _check_mode(mode):
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}; expected 'latency' or 'throughput'")
    return mode


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _normalize_spec(spec):
    import copy as _copy

    def _norm(value):
        if isinstance(value, dict):
            return {k: _norm(v) for k, v in value.items()}
        if isinstance(value, list):
            if len(value) == 1 and not isinstance(value[0], (list, dict)):
                return _copy.deepcopy(value[0])
            return [_norm(v) for v in value]
        return value

    out = {}
    for key, value in spec.items():
        if key == _THREAD_TOP or key in _LIST_TOPS:
            value = normalize_container(key, value)
        out[key] = _norm(value)
    return out


def _merge_config(override):
    import copy

    base = copy.deepcopy(_DEFAULT_BENCH)
    if not override:
        return validate_spec(_normalize_spec(base))
    merged = _deep_merge(base, dict(override))
    return validate_spec(_normalize_spec(merged))


def _validate_scalar_key(key, value):
    if isinstance(value, (list, tuple)):
        raise ValueError(
            f"config {key} is a single value, got {value!r} (do not wrap it in a list)"
        )
    if key == "samples":
        try:
            iv = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{key} must be an integer, got {value!r}") from e
        if iv < 1:
            raise ValueError(f"{key} must be >= 1, got {value!r}")
    elif key == "seed":
        if value is None:
            return
        try:
            int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"seed must be an integer, got {value!r}") from e
    elif key == "iterations":
        if value is None:
            return
        if isinstance(value, dict):
            return
        try:
            iv = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{key} must be an integer, got {value!r}") from e
        if iv < 1:
            raise ValueError(f"{key} must be >= 1, got {value!r}")
    elif key == "backend":
        _validate_backend({"backend": value})


def _leaf_alternatives(path, entry):
    if not isinstance(entry, dict):
        return [entry]
    for key in list(entry):
        value = entry[key]
        if isinstance(value, (list, tuple)):
            raise ValueError(
                f"config {path}.{key} must be a single value, got {value!r}; "
                "sweep via multiple container alternatives instead"
            )
    return [entry]


def _expand_config_spec(spec):
    import copy as _copy
    import itertools as _it

    def _alternatives(path, value):
        if isinstance(value, list):
            if not value:
                raise ValueError(
                    f"config {path} must be a non-empty list of alternatives, "
                    f"got {value!r}"
                )
            return list(value)
        return [value]

    entries = []
    for key, value in spec.items():
        if key in _CONTAINER_TOPS and isinstance(value, dict):
            for sub, sv in value.items():
                entries.append(([key, sub], _alternatives(f"{key}.{sub}", sv)))
        elif key in _LIST_TOPS and isinstance(value, list):
            entries.append(
                (
                    [key],
                    [
                        entry
                        for i, alt in enumerate(value)
                        for entry in _leaf_alternatives(f"{key}[{i}]", alt)
                    ],
                )
            )
        else:
            entries.append(([key], _alternatives(key, value)))
    paths = [p for p, _ in entries]
    choices = [c for _, c in entries]
    combos = []
    for combo in _it.product(*choices):
        concrete = {}
        for path, value in zip(paths, combo):
            node = concrete
            for part in path[:-1]:
                nxt = node.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    node[part] = nxt
                node = nxt
            node[path[-1]] = _copy.deepcopy(value)
        combos.append(concrete)
    return combos


def _concrete_configs(config):
    spec = _merge_config(config)
    combos = _expand_config_spec(spec)
    for combo in combos:
        validate_spec(combo)
    return spec, combos


def _tag_config_columns(df, config):
    if df is None or getattr(df, "empty", False):
        return df
    try:
        cols = config_param_columns(config)
    except Exception:
        return df
    for c, v in cols.items():
        if c not in df.columns:
            try:
                df[c] = v
            except Exception:
                pass
    return df


def _filter_bench_columns(df):
    if df is None or getattr(df, "empty", False):
        return df
    try:
        cols = list(df.columns)
    except Exception:
        return df
    keep = []
    for c in cols:
        s = str(c)
        if s in _BENCH_KEEP_META:
            keep.append(c)
            continue
        if s.startswith("data."):
            keep.append(c)
            continue
        if s.startswith("config.branch") or s.startswith(
            (
                "config.backend",
                "config.dcache",
                "config.icache",
                "config.dtlb",
                "config.itlb",
            )
        ):
            keep.append(c)
            continue
        if s.startswith("config."):
            continue
        if s in ("time", "address", "size", "pid", "ip"):
            continue
        keep.append(c)
    try:
        ordered = [c for c in ("iterations", "samples", "operations") if c in keep]
        ordered += [c for c in keep if c not in ordered]
        if ordered:
            return df[ordered]
    except Exception:
        pass
    return df


def _default_unroll_n():
    default = _DEFAULT_BENCH.get("backend", {})
    default = (
        default.get("unroll", {}).get("count") if isinstance(default, dict) else None
    )
    return default if default is not None else 5


def _content_hash8(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    except OSError:
        return None


def _stable_hash8(payload):
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def _id_hash(data, config, binary=None):
    has_data = bool(data and (data.get("regs") or data.get("mem")))
    has_config = bool(config)
    has_binary = binary is not None and bool(binary)
    if not has_data and not has_config and not has_binary:
        return None
    payload = {
        "data": data if has_data else {"regs": {}, "mem": {}},
        "config": config or {},
    }
    if has_binary:
        payload["binary"] = binary
    return _stable_hash8(payload)


def _as_normalized_data(data):
    if not data:
        return {"regs": {}, "mem": {}}
    if isinstance(data, dict) and any(k in data for k in ("regs", "mem")):
        regs = dict(data.get("regs") or {})
        mem = dict(data.get("mem") or {})
        return {"regs": regs, "mem": mem}
    return _config_data({"data": data})


def _merge_data(base, override):
    b = _as_normalized_data(base)
    o = _as_normalized_data(override)
    b["regs"].update(o["regs"])
    for k, v in o["mem"].items():
        b["mem"][k] = v
    return b


def _generate_seed():
    try:
        return random.SystemRandom().randint(0, 2**31 - 1)
    except Exception:
        return random.randint(0, 2**31 - 1)


def _first_scalar(val):
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    return _to_int_or(val, None)


def _order_bench_columns(df):
    try:
        cols = list(df.columns)
    except Exception:
        return df
    present = set(cols)
    lead = [c for c in _IDENTITY_COLUMNS if c in present]
    config_cols = sorted(c for c in cols if str(c).startswith("config."))
    data_cols = sorted(c for c in cols if str(c).startswith("data."))
    meta = [c for c in ("iterations", "samples", "operations") if c in present]
    ordered = lead + config_cols + data_cols + meta
    seen = set(ordered)
    ordered += [c for c in cols if c not in seen]
    try:
        return df[ordered]
    except Exception:
        return df


def _available_table(project):
    try:
        df = metadata(project.loader.main_object.binary)
    except Exception:
        return ""
    if df.empty:
        return "(no targets found)"
    return hex_addresses(df).to_string(index=False)


def _unpack_bench(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, None


def _measure_group(
    config,
    mode,
    setup,
    teardown,
    solutions,
    group,
    arch,
    map_name,
    tracker,
    data,
    overhead_code,
    main_code,
    extra=None,
    debug=False,
    models_tracker=None,
):
    extra = extra or {}
    overhead_res, _ = _unpack_bench(
        _bench(
            config=config,
            mode=mode,
            code=overhead_code,
            setup=setup,
            teardown=teardown,
            solutions=solutions,
            events=group,
            data=data,
            arch=arch,
            map_name=map_name,
            iterations_tracker=tracker,
            models_tracker=models_tracker,
            debug=debug,
            **extra,
        )
    )
    overhead = np.asarray(overhead_res.median(axis=0), dtype=float)
    result, iterations = _unpack_bench(
        _bench(
            config=config,
            mode=mode,
            setup=setup,
            code=main_code,
            teardown=teardown,
            solutions=solutions,
            overhead=overhead,
            events=group,
            data=data,
            arch=arch,
            map_name=map_name,
            iterations_tracker=tracker,
            models_tracker=models_tracker,
            debug=debug,
            **extra,
        )
    )
    return result, iterations


def _bench_project(file, config, stack, func):
    import angr

    key = _project_cache_id(file, config)
    cached = _PROJECTS.get(key)
    if cached is not None:
        return cached

    project = angr.Project(
        file,
        auto_load_libs=_resolve_lib(config),
        load_debug_info=False,
    )
    obj = Elf(project.loader)
    obj.map_elf()
    obj.setup_stack(size=stack["size"], align=stack["align"])

    funcs, prototypes = functions(project)

    if func["order"] == _FUNC_ORDERS[1]:
        obj.randomize_layout(
            dict(funcs),
            seed=_config_seed_int(config),
            align=func["align"],
        )

    entry = (project, obj, funcs, prototypes)
    _PROJECTS[key] = entry
    return entry


def _bench_one(**kwargs):
    debug = bool(kwargs.get("debug", False))
    mode = _check_mode(kwargs.get("mode"))
    file = kwargs.get("file")
    src_file = file
    if file:
        file = resolve_exec(file)
        if not Path(file).exists():
            raise ValueError(f"file {file!r} does not exist")
    code = kwargs.get("code")
    target = _normalize_target(kwargs.get("target"))
    name = kwargs.get("name") or target or code
    setup = _setup_names(kwargs.get("setup"))
    teardown = _setup_names(kwargs.get("teardown"))
    event = kwargs.get("event", _DURATION_TIME)
    backend_arg = kwargs.get("backend")
    unroll_arg = kwargs.get("unroll_n")

    config = kwargs.get("config")
    if config is None:
        config = {}
    if isinstance(config, dict) and "data" in config:
        raise ValueError(
            "config must not contain 'data'; pass data separately via data={...}"
        )
    data = _merge_data(None, kwargs.get("data"))

    if config.get("seed") is None:
        config["seed"] = _generate_seed()
    entry, _created = _thread_store(config)
    _node = _parse_numa(_numa_spec(config))
    if entry.get("priority") is None:
        entry["priority"] = _current_priority_value()
    _iterations_tracker = []
    _models_tracker = []
    _iterations_pinned, _iter_lo, _iter_hi = _iterations_config(config)
    _parse_affinity(_affinity_spec(config))
    _parse_priority(_priority_spec(config))
    _thread_cfg(config)
    for key in _LIST_TOPS:
        config[key] = _container(config, key)
    _stack_size, _stack_align = config["stack"]["size"], config["stack"]["align"]
    _resolved_order, _resolved_align = config["func"]["order"], config["func"]["align"]
    _code_align = config["code"]["align"]
    _all_solutions = []
    _apply_seed(config)
    if debug:
        _debug_json("effective config", config)
        _debug_json("input data", data)

    groups = _normalize_groups(event)
    try:
        groups = [core.with_leader(g) for g in groups]
    except Exception:
        pass
    measure_groups = [g for g in groups if g]
    if entry.get("affinity") is None:
        _default_cpu = core.default_affinity(measure_groups, _node)
        if _default_cpu is not None:
            entry["affinity"] = _default_cpu

    results = []
    cpu_info = cpuinfo(list(_CPUINFO_FIELDS)).iloc[0].to_dict()

    if code:
        if not name:
            name = code
        arch = get_arch()
        code = arch.normalize_asm(f"{code};")
        setup = arch.normalize_asm("\n".join(setup) + "\n") if setup else ""
        teardown = arch.normalize_asm("\n".join(teardown) + "\n") if teardown else ""
        backend = _resolve_backend(backend_arg, config, "loop")
        unroll_n = _resolve_unroll_n(unroll_arg, config, backend)
        if backend == "unroll" and mode == "throughput":
            raise ValueError("backend 'unroll' can only be used with mode 'latency'")
        _select_backend(config, backend, unroll_n)
        asm_solutions = _explore_asm_cached(
            code,
            setup or None,
            data,
            arch,
            _stack_size,
            _stack_align,
        )
        _all_solutions = asm_solutions or []
        if debug:
            _debug_log(f"found {len(list(asm_solutions or []))} solution(s) (asm)")
            _debug_json("solutions", _solution_summary(asm_solutions))
            try:
                _debug_json("synthesised data", explored_data_dict(asm_solutions))
            except Exception:
                pass
        binary_id = {"asm": code, "name": name}
        for group in measure_groups:
            if backend == "unroll":
                result, _iterations = _measure_group(
                    config,
                    mode,
                    setup,
                    teardown,
                    asm_solutions,
                    group,
                    arch,
                    name,
                    _iterations_tracker,
                    data,
                    _with_align(f"{code}" * unroll_n, _code_align),
                    _with_align(f"{code}" * (2 * unroll_n), _code_align),
                    debug=debug,
                    models_tracker=_models_tracker,
                )
                operations = 1 if mode == "latency" else _iterations
                recs = _records(
                    None,
                    name,
                    mode,
                    result,
                    group,
                    divisor=unroll_n,
                    operations=operations,
                    iterations=_iterations,
                )
            else:
                result, _iterations = _measure_group(
                    config,
                    mode,
                    setup,
                    teardown,
                    asm_solutions,
                    group,
                    arch,
                    name,
                    _iterations_tracker,
                    data,
                    _with_align("nop;", _code_align),
                    _with_align(f"{code}", _code_align),
                    debug=debug,
                    models_tracker=_models_tracker,
                )
                operations = 1 if mode == "latency" else _iterations
                recs = _records(
                    None,
                    name,
                    mode,
                    result,
                    group,
                    operations=operations,
                    iterations=_iterations,
                )
            results.append(recs)
    else:
        if not target:
            raise ValueError("a target is required to analyze a binary (--file)")

        project, obj, funcs, prototypes = _bench_project(
            file, config, config["stack"], config["func"]
        )
        arch = load_arch(project)

        try:
            obj.write_perf_map()
        except Exception:
            pass

        setup_names = _setup_names(setup)
        setup = _call_seq_asm(obj, arch, setup_names)
        teardown = _call_seq_asm(obj, arch, _setup_names(teardown))

        ang_setup = ""
        if setup_names:
            sym = project.loader.find_symbol(setup_names[0])
            if sym is not None:
                ang_setup = arch.call_asm(sym.rebased_addr)
            else:
                ang_setup = setup

        base = os.path.basename(str(src_file))
        binary_id = {"path": base, "sha": _content_hash8(src_file)}
        file_label = f"{base}@{binary_id['sha']}"
        info_dict = {"cpu": cpu_info, "binary": binary_id}
        targets = list(resolve_targets(project, target, funcs))
        if not targets:
            table = _available_table(project)
            print(
                f"cannot resolve {target!r} in {src_file!r}; available targets:",
                file=sys.stderr,
            )
            print(table, file=sys.stderr)
            raise ValueError(f"cannot resolve {target!r} in {src_file!r}")
        if len(targets) > 1:
            table = _available_table(project)
            print(
                f"ambiguous target {target!r} in {src_file!r}; available targets:",
                file=sys.stderr,
            )
            print(table, file=sys.stderr)
            raise ValueError(f"ambiguous target {target!r} in {src_file!r}")
        for target_label, start, end in targets:
            rec_name = name
            if int(end) <= int(start):
                raise ValueError(f"empty region {target_label!r}: end <= start")
            proto = _func_prototype(prototypes.get(target_label), data)
            try:
                solutions = _explore_target_cached(
                    project,
                    _project_cache_id(file, config),
                    target_label,
                    start,
                    end,
                    ang_setup,
                    funcs,
                    data,
                    proto,
                    _stack_size,
                    _stack_align,
                )
            except Exception as ex:
                _warn_exploration(target_label, ex)
                solutions = []

            _all_solutions = solutions or []
            if debug:
                _debug_log(
                    f"found {len(list(solutions or []))} solution(s) "
                    f"for target {target_label}"
                )
                _debug_json("solutions", _solution_summary(solutions))
                try:
                    _debug_json("synthesised data", explored_data_dict(solutions))
                except Exception:
                    pass
            angr_base = project.loader.main_object.mapped_base
            addr_offset = obj.runtime_base - angr_base
            bin_segs = [
                (s.vaddr, s.vaddr + s.memsize)
                for s in project.loader.main_object.segments
            ]
            addr_range = (
                (min(s[0] for s in bin_segs), max(s[1] for s in bin_segs))
                if bin_segs
                else None
            )

            symbol = None
            try:
                symbol = obj.get_symbol(target_label)
            except ValueError:
                pass
            target = symbol if symbol is not None else start + addr_offset
            code_asm = arch.call_seq_asm(target, fence=(mode == "latency"))
            if mode == "throughput":
                restore = arch.data_reload_asm(data)
                if restore:
                    code_asm = f"{restore}\n{code_asm}"
            code_asm = _with_align(code_asm, _resolved_align)

            backend = _resolve_backend(backend_arg, config, "loop")
            unroll_n = _resolve_unroll_n(unroll_arg, config, backend)
            if backend == "unroll" and mode == "throughput":
                raise ValueError(
                    "backend 'unroll' can only be used with mode 'latency'"
                )
            _select_backend(config, backend, unroll_n)
            try:
                _static_addrs = _static_table_addrs(
                    project, start, end, obj, solutions=solutions
                )
            except Exception:
                _static_addrs = []
            try:
                _code_size = max(int(end) - int(start), 1)
                _code_page = mmap.PAGESIZE
                _code_addrs = list(
                    range(
                        int(target) & ~(_code_page - 1),
                        (int(target) + _code_size + _code_page - 1) & ~(_code_page - 1),
                        _code_page,
                    )
                )
            except (TypeError, ValueError):
                _code_addrs = []
            extra = {
                "addr_range": addr_range,
                "extra_addrs": list(_static_addrs or []),
                "code_addrs": _code_addrs,
            }
            _patch_addr = None
            _saved_byte = None
            if ".." in str(target_label):
                try:
                    _patch_addr = int(end) + int(addr_offset)
                    _saved_byte = ctypes.c_ubyte.from_address(_patch_addr).value
                    ctypes.c_ubyte.from_address(_patch_addr).value = 0xC3
                except Exception:
                    _patch_addr = None
                    _saved_byte = None
            try:
                for group in measure_groups:
                    if backend == "unroll":
                        code_n = "\n".join([code_asm] * unroll_n)
                        code_2n = "\n".join([code_asm] * (2 * unroll_n))
                        result, _iterations = _measure_group(
                            config,
                            mode,
                            setup,
                            teardown,
                            solutions,
                            group,
                            arch,
                            target_label,
                            _iterations_tracker,
                            data,
                            code_n,
                            code_2n,
                            extra=extra,
                            debug=debug,
                            models_tracker=_models_tracker,
                        )
                        operations = 1 if mode == "latency" else _iterations
                        recs = _records(
                            file_label,
                            rec_name,
                            mode,
                            result,
                            group,
                            divisor=unroll_n,
                            operations=operations,
                            iterations=_iterations,
                        )
                        results.append(recs)
                        continue
                    result, _iterations = _measure_group(
                        config,
                        mode,
                        setup,
                        teardown,
                        solutions,
                        group,
                        arch,
                        target_label,
                        _iterations_tracker,
                        data,
                        _with_align(
                            arch.call_seq_nop_asm(fence=(mode == "latency")),
                            _resolved_align,
                        ),
                        code_asm,
                        extra=extra,
                        debug=debug,
                    )
                    operations = 1 if mode == "latency" else _iterations
                    recs = _records(
                        file_label,
                        rec_name,
                        mode,
                        result,
                        group,
                        operations=operations,
                        iterations=_iterations,
                    )
                    results.append(recs)
            finally:
                if _patch_addr is not None:
                    try:
                        ctypes.c_ubyte.from_address(_patch_addr).value = _saved_byte
                    except Exception:
                        pass
                    _patch_addr = None

    df = _combine_results(results)
    assert not df.empty, kwargs

    if "duration_time" in df.columns:
        try:
            _freq = float(cpu_info.get("freq"))
        except (TypeError, ValueError):
            _freq = 0.0
        if _freq and _freq > 0:
            df["duration_time"] = df["duration_time"] * 1e9 / _freq

    for ev in {e for group in measure_groups for e in group}:
        if ev not in df.columns:
            df[ev] = np.nan

    if not isinstance(config.get("iterations"), dict):
        config["iterations"] = {}
    if _iterations_pinned is not None:
        try:
            config["iterations"]["count"] = int(_iterations_pinned)
        except (TypeError, ValueError):
            pass
    elif _iterations_tracker:
        try:
            config["iterations"]["count"] = int(max(_iterations_tracker))
        except Exception:
            pass
    else:
        _p, lo, _hi = _iterations_config(config)
        try:
            config["iterations"]["count"] = int(
                lo if lo is not None else _DEFAULT_ITER_MIN
            )
        except (TypeError, ValueError):
            pass
    if "backend" not in locals():
        config["backend"] = config.get("backend")
    elif not _backend_resolved(config, backend):
        _select_backend(config, backend)

    try:
        col_data = {
            "regs": dict((data or {}).get("regs") or {}),
            "mem": dict((data or {}).get("mem") or {}),
        }
        try:
            explored = explored_data_dict(
                _all_solutions, _models_tracker[-1] if _models_tracker else None
            )
            try:
                explicit_canon = set()
                for _k in col_data["regs"]:
                    try:
                        explicit_canon.add(_canonical_data_reg_key(_k))
                    except Exception:
                        explicit_canon.add(str(_k).strip().lower())
            except Exception:
                explicit_canon = set(col_data["regs"])
            for _reg, _val in (explored.get("regs") or {}).items():
                try:
                    _canon = _canonical_data_reg_key(_reg)
                except Exception:
                    _canon = str(_reg).strip().lower()
                if _canon in explicit_canon or _reg in col_data["regs"]:
                    continue
                col_data["regs"][_canon] = _val
                explicit_canon.add(_canon)
            for _addr, _val in (explored.get("mem") or {}).items():
                col_data["mem"].setdefault(_addr, _val)
        except Exception:
            pass
        df = add_param_columns(df, col_data)
    except Exception:
        try:
            df = add_param_columns(df, data)
        except Exception:
            pass

    try:
        flat = [e for g in (measure_groups or []) for e in (g or [])]
    except Exception:
        flat = None
    try:
        df = _filter_bench_columns(df)
    except Exception:
        pass
    if code:
        df.attrs["config"] = config
        df.attrs["info"] = {"cpu": cpu_info, "binary": binary_id}
        df.attrs["file"] = None
    else:
        df.attrs["config"] = config
        df.attrs["info"] = info_dict
        df.attrs["file"] = file_label
    df.attrs["id"] = _id_hash(data, config, binary_id)
    df.attrs["data"] = data
    if debug:
        try:
            _debug_log(f"results ({len(df)} rows):\n{df.to_string()}")
        except Exception:
            pass
    return df


def _is_branch_mnemonic(mnemonic):
    return get_arch().is_branch_mnemonic(mnemonic)


def _left_align_asm(text):
    out = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            out.append("")
        else:
            out.append(line.lstrip())
    if not out:
        return ""
    return "\n".join(out) + "\n"


def _executed_bbl_addrs(solutions):
    addrs = []
    seen = set()
    for sol in solutions or []:
        try:
            hist = (
                sol.get("state").history.bbl_addrs
                if isinstance(sol, dict)
                else sol.history.bbl_addrs
            )
        except Exception:
            continue
        try:
            lst = list(hist)
        except Exception:
            continue
        for a in lst:
            try:
                ai = int(a)
            except (TypeError, ValueError):
                continue
            if ai not in seen:
                seen.add(ai)
                addrs.append(ai)
    return addrs


def _executed_blocks(solutions):
    addrs = _executed_bbl_addrs(solutions)
    seen = set(addrs)
    for sol in solutions or []:
        try:
            state = sol.get("state") if isinstance(sol, dict) else sol
            rip = int(state.solver.eval(state.regs.rip))
        except Exception:
            continue
        if rip not in seen:
            seen.add(rip)
            addrs.append(rip)
    return addrs


def _branch_targets(insns):
    return get_arch().branch_targets(insns)


def _render_insn(insn, labels):
    op_str = insn.op_str or ""
    if _is_branch_mnemonic(insn.mnemonic) and op_str:
        op_str = _sub_intel_labels(op_str, labels)
    return f"{insn.mnemonic} {op_str}".strip()


def _disasm_executed_asm(project, addrs, arch=None, excluded=()):
    if arch is None:
        arch = load_arch(project)
    setup_base = arch._SETUP_BASE
    spans = [
        (int(first), int(last)) for first, last in (excluded or ()) if first < last
    ]
    filtered = []
    for a in addrs or []:
        if setup_base <= a < setup_base + 0x100000:
            continue
        if any(first <= a < last for first, last in spans):
            continue
        try:
            block = project.factory.block(a)
            insns = list(block.capstone.insns)
        except Exception:
            continue
        if not insns:
            continue
        filtered.append((a, insns))
    if not filtered:
        return ""
    try:
        filtered = sorted(filtered, key=lambda p: int(p[0]))
    except Exception:
        pass
    labels = _branch_targets([ins for _, insns in filtered for ins in insns])
    lines = [".intel_syntax noprefix"]
    emitted_labels = set()
    emitted_insns = set()
    for bbl_addr, insns in filtered:
        if bbl_addr in labels and bbl_addr not in emitted_labels:
            lines.append(f"{labels[bbl_addr]}:")
            emitted_labels.add(bbl_addr)
        for insn in insns:
            try:
                _addr = int(insn.address)
            except Exception:
                _addr = insn.address
            if _addr in emitted_insns:
                continue
            emitted_insns.add(_addr)
            if (
                insn.address != bbl_addr
                and insn.address in labels
                and insn.address not in emitted_labels
            ):
                lines.append(f"{labels[insn.address]}:")
                emitted_labels.add(insn.address)
            lines.append(_render_insn(insn, labels))

    for addr, lab in labels.items():
        if addr not in emitted_labels:
            lines.append(f"{lab}:")
            emitted_labels.add(addr)
    return "\n".join(lines) + "\n"


def _disasm_target(project, start, end, arch=None):
    if arch is None:
        arch = load_arch(project)
    try:
        blob = project.loader.memory.load(start, end - start)
    except Exception:
        return ""
    lines = [".intel_syntax noprefix"]
    try:
        insns = list(arch.disassembler().disasm(bytes(blob), start))
    except Exception:
        insns = []
    labels = _branch_targets(insns)
    internal = {a for a in labels if start <= a < end}
    try:
        for insn in insns:
            if insn.address in internal:
                lines.append(f"{labels[insn.address]}:")
            lines.append(_render_insn(insn, labels))
    except Exception:
        pass
    return "\n".join(lines) + "\n"


def _sub_intel_labels(op_str, labels):
    s = op_str or ""
    n = len(s)
    out = []
    i = 0
    while i < n:
        k = s.find("0x", i)
        if k < 0:
            out.append(s[i:])
            break
        j = k + 2
        while j < n and s[j] in _HEXDIGITS:
            j += 1
        out.append(s[i:k])
        tok = s[k:j]
        try:
            out.append(str(labels.get(int(tok, 16), tok)))
        except ValueError:
            out.append(tok)
        i = j
    return "".join(out)
