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

import bisect
import copy
import mmap
import os
import sys
import threading
from collections import Counter, deque

import pandas as pd

from .arch import arch as get_arch
from .cfg import (
    _DEFAULT_STACK_SIZE,
    _FUNC_ORDERS,
    _PRUNE_KEYS,
    _canonical_data_reg_key,
    _config_seed_int,
    _container,
    _entry,
    _entry_alternatives,
    _entry_signature,
    _fake_stderr,
    _fake_stdout,
    _first_scalar,
    _iter_mem_addrs,
    _merge_config,
    _merge_data,
    _normalize_data_regs,
    _normalize_target,
    _output_fakes,
    _parse_addr_key,
    _resolve_lib,
    _resolve_target,
    check_data,
)
from .core import (
    _ASM_SCRATCH_BASE,
    _content_hash8,
    _debug_log,
    _func_prototype,
    _locked,
    _stable,
    _to_int_or,
    _to_u64,
    eval_ints,
    min_ints,
    serialized,
)
from .data import (
    _IDENTITY_COLUMNS,
    _NON_METRIC_COLUMNS,
    _match,
    query,
    quote_columns,
)
from .exec import Elf, _exec_stamp, resolve_exec
from .info import (
    _unresolved,
    functions,
    targets as resolve_targets,
)

_IDENTITY = tuple(c for c in _IDENTITY_COLUMNS if c != "mode")
_RECORD_COLUMNS = ("event", "period")
_ADDRESS_COLUMNS = ("ip", "address", "addr")
_INSN_COLUMNS = ("index", "address", "encoding", "size")
_OSACA_COLUMNS = ("latency", "throughput")
_TEXT_COLUMNS = ("assembly",)
_LEAD_COLUMN = "index"
_SKIP_COLUMNS = ("data.", "config.")
_DATA_PREFIX = "data."
_WILDCARDS = ("*", "?")
_PAGE_MASK = 0xFFF
_DEFAULT_DEBUG_CONTEXT = 3
_DECOMPILE_KEY = "__perf_decompile__"
_DEFAULT_EVENTS = [
    *_IDENTITY,
    *_INSN_COLUMNS,
    *_OSACA_COLUMNS,
    *_TEXT_COLUMNS,
    "data*",
]
_OSACA_LOCK = threading.RLock()
_serialized = serialized(_OSACA_LOCK)

_PROJECTS = {}
_PROJECTS_MAX = 16
_HEXDIGITS = set("0123456789abcdefABCDEF")
_BRANCH_JUMPKINDS = frozenset({"ijk_boring", "ijk_indirect"})
_EXPLORE_CACHE = {}
_EXPLORE_CACHE_MAX = 32
_MODEL_CACHE = {}
_SOLVE_CACHE_MAX = 4
_STATIC_TABLE_CACHE = {}
_STATIC_TABLE_CACHE_MAX = 256
_NONDET_BASE = _ASM_SCRATCH_BASE
_NONDET_SIZE = 0x10000
_UNKNOWN_MAX = 0xFF
_ADDRESS_WINDOW = (_NONDET_BASE, _NONDET_BASE + _NONDET_SIZE)
_ADDRESS_WINDOW_CLASS = None
_EXPLORE_STEPS = 4096
_EXPLORE_STATES = 512
_EXPLORE_BRANCHES = 12


def analyze(
    target=None,
    name=None,
    *,
    event=None,
    args=None,
    config=None,
    data=None,
    setup=None,
    teardown=None,
    filter=None,
    debug=False,
    decompile=False,
):
    debug_context = None
    if isinstance(config, dict) and "debug_context" in config:
        config = dict(config)
        debug_context = _debug_context(config.pop("debug_context"))
    file, resolved_target, snippet, name, raw = _resolve_target(target, name)
    if data is not None:
        data = _merge_data(None, data)
        check_data(data)
    columns = _split_columns(event)
    strict = bool(columns)
    if not columns:
        columns = [*_DEFAULT_EVENTS, "*"]
    records = {}
    debug_map = {}
    if snippet is not None:
        code, insns = _asm_instructions(
            snippet, setup, records, data, teardown, config=config
        )
        file_label, label = None, name or code
        debug_map = {}
    else:
        label, file_label, insns, debug_map = _binary_instructions(
            file,
            resolved_target,
            config,
            setup,
            records,
            data,
            teardown,
            debug=debug,
            decompile=decompile,
        )
        label = name or label
        code = raw
    if not insns:
        raise ValueError(f"no instructions found for {label!r}")

    keep = {int(insn.address) for insn in insns}
    records = {
        int(address): row
        for address, row in (records or {}).items()
        if int(address) in keep
    }
    decompile_info = None
    if isinstance(debug_map, dict):
        decompile_info = debug_map.pop(_DECOMPILE_KEY, None)
    df = _instruction_frame(insns, records)
    df = _merge_args(df, args, file_label, label)
    df = _apply_filter(df, filter)
    debug_rows = _debug_rows(df["address"].tolist(), debug_map) if debug else None
    decompile_rows = (
        _decompile_rows(df["address"].tolist(), decompile_info)
        if decompile_info is not None
        else None
    )
    df = _select(df, columns, strict=strict)
    df.attrs["file"] = file_label
    df.attrs["name"] = label
    df.attrs["code"] = code
    df.attrs["config"] = _merge_config(config)
    if decompile_rows is not None:
        df.attrs["debug"] = decompile_rows
    elif debug_rows is not None:
        df.attrs["debug"] = debug_rows
    if decompile_info is not None:
        text = decompile_info.get("text") or ""
        df.attrs["decompile"] = text
        df.attrs["decompiled"] = text
        df.attrs["decompile_source"] = decompile_info.get("source")
    if debug_context is not None:
        df.attrs["debug_context"] = debug_context
    return df


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
    stack_mode="random",
    stack_seed=None,
    teardown_asm=None,
    errors=False,
):
    if funcs is None:
        funcs, _ = functions(proj)
    is_func = start in {s for s, e in funcs.values()}

    if is_func:
        state = proj.factory.call_state(addr=start, prototype=prototype)
    else:
        state = proj.factory.blank_state(addr=start)
        get_arch(proj).map_stack(
            state,
            size=stack_size,
            align=stack_align,
            mode=stack_mode,
            seed=stack_seed,
        )

    if setup_asm:
        setup_asm = setup_asm.rstrip("\n")
        arch = get_arch(proj)
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
    try:
        arch = get_arch(proj)
    except Exception:
        arch = get_arch()
    explicit_by_reg = {}
    for name, value in explicit_regs.items():
        try:
            canon = arch._canonical_reg(name)
        except Exception:
            canon = str(name).strip().lower()
        explicit_by_reg[canon] = value
    for name, sym in sym_regs.items():
        try:
            cname = arch._canonical_reg(name)
        except Exception:
            cname = name
        if cname not in explicit_by_reg:
            continue
        v = explicit_by_reg[cname]
        if isinstance(v, (list, tuple)):
            continue
        try:
            value = int(v) & 0xFFFFFFFFFFFFFFFF
            state.solver.add(sym == value)
            try:
                import claripy

                try:
                    bits = sym.size()
                except Exception:
                    bits = 64
                state.registers.store(name, claripy.BVV(value, bits))
            except Exception:
                pass
        except (TypeError, ValueError):
            pass

    _map_state_mem_pages(state, (data or {}).get("mem"))

    _nondet_options(state)

    ret_addrs = _ret_addrs(proj, start, end, funcs)
    world = _target_ranges(proj, _setup_ranges(setup_asm))

    def _run(state):
        return _explore_states(proj, state, ret_addrs, world)

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

    pins = set()
    simgr = _run(state)
    states = _reached(simgr, errors)
    if _needs_reg_pinning(states, simgr, sym_regs, _keep):
        try:
            arch = get_arch(proj)
        except Exception:
            arch = get_arch()
        explicit_canon = set()
        for name in explicit_regs:
            try:
                explicit_canon.add(arch._canonical_reg(name))
            except Exception:
                explicit_canon.add(str(name).strip().lower())
        pins = _pin_unknown_regs(state, sym_regs, explicit_canon, arch, concrete=True)
        simgr = _run(state)
        states = _reached(simgr, errors)
    records = _solution_records(states, sym_regs, _keep, pins)
    if teardown_asm:
        return _explore_teardown(proj, states, sym_regs, teardown_asm, _keep, pins)
    return records


def explore_asm(
    code,
    setup_asm=None,
    data=None,
    arch=None,
    stack_size=None,
    stack_align=16,
    stack_mode="random",
    stack_seed=None,
    teardown_asm=None,
    errors=False,
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

    mode = stack_mode
    skip_stack_init = False
    if not teardown_asm and not _data_touches_stack(data, arch, stack_size):
        try:
            skip_stack_init = not arch.uses_stack(encoding, base)
        except Exception:  # pragma: no cover
            skip_stack_init = False
    if skip_stack_init and str(stack_mode).strip().lower() in ("random", "fixed"):
        mode = "zero"

    import angr

    proj = angr.load_shellcode(
        bytes(encoding), arch=arch._ANGR_ARCH_NAME, load_address=base
    )
    state = proj.factory.blank_state(addr=base)
    arch.map_stack(
        state,
        size=stack_size,
        align=stack_align,
        mode=mode,
        seed=stack_seed,
        init_zero=not skip_stack_init,
    )
    try:
        import claripy

        state.memory.store(
            arch.stack_pointer(state),
            claripy.BVV(base - 0x1000, 64),
            size=8,
            endness=state.arch.memory_endness,
        )
    except Exception:
        pass

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

    pins = _pin_unknown_regs(state, sym_regs, explicit_canon, arch)
    _map_state_mem_pages(state, (data or {}).get("mem"))

    end = base + len(encoding)
    td_base, td_size = _teardown_end(teardown_asm) if teardown_asm else (None, 0)
    jump_to = base - 0x1000
    if td_size:
        try:
            td = arch.assemble(arch.normalize_asm(f"{teardown_asm};"), td_base)
            try:
                last = list(arch.disassembler().disasm(td, td_base))[-1]
                has_ret = arch.is_return_mnemonic(last.mnemonic)
            except Exception:
                has_ret = False
            if not has_ret:
                td += arch.return_stub_bytes()
            td_size = len(td)
            state.memory.map_region(td_base, td_size + 256, 7, init_zero=True)
            state.memory.store(td_base, td)
            jump_to = int(td_base)
            end = jump_to + int(td_size)
            try:
                import claripy

                state.memory.store(
                    arch.stack_pointer(state),
                    claripy.BVV(end, 64),
                    size=8,
                    endness=state.arch.memory_endness,
                )
            except Exception:
                pass
        except Exception:  # pragma: no cover
            jump_to = base - 0x1000
            td_size = 0
    try:
        store_addr = end if not td_size else base + len(encoding)
        state.memory.store(
            store_addr,
            arch.assemble(arch.jump_asm(jump_to), store_addr),
        )
    except Exception:
        pass
    world = [(base, base + len(encoding) + 0x1000)]
    if td_size:
        world.append((int(td_base), int(td_base) + int(td_size)))
    _nondet_options(state)

    simgr = _explore_states(proj, state, {end}, world)

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

    return _solution_records(_reached(simgr, errors), sym_regs, _keep, pins)


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
            inputs = solution["reg"]
            regs = {}
            for name, value in zip(list(inputs), min_ints(st, list(inputs.values()))):
                if value is not None:
                    regs[name] = value

            reads, writes = [], []
            read_index, write_index = {}, {}
            for accesses, out, index, last in (
                (solution["reads"], reads, read_index, False),
                (solution["writes"], writes, write_index, True),
            ):
                if not accesses:
                    continue
                flat = [expr for entry in accesses for expr in entry]
                resolved = eval_ints(st, flat)
                for i in range(len(accesses)):
                    entry = resolved[3 * i : 3 * i + 3]
                    if len(entry) != 3:
                        continue
                    addr, length, value = entry
                    if addr is None or length is None:
                        continue
                    if not (0 <= addr < (1 << 64)):
                        continue
                    _collapse_access(
                        out,
                        index,
                        addr,
                        length,
                        0 if value is None else value,
                        last=last,
                    )

            model = {"regs": regs, "reads": reads, "writes": writes}
            if pinned_addrs:
                model["blocks"] = _executed_bbl_addrs([solution])
                model["branch_deps"] = _branch_deps(st, model, pinned_addrs, arch)
            models.append(model)

    return models


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


def _debug_context(value=None):
    if value is None:
        return _DEFAULT_DEBUG_CONTEXT
    try:
        context = int(value)
    except (TypeError, ValueError) as ex:
        raise ValueError(
            f"debug_context must be a non-negative integer, got {value!r}"
        ) from ex
    if context < 0:
        raise ValueError(f"debug_context must be a non-negative integer, got {value!r}")
    return context


def _split_columns(event):
    if event is None:
        return []
    values = [event] if isinstance(event, str) else list(event)
    out = []
    for value in values:
        for name in str(value or "").split(","):
            name = name.strip()
            if name and name not in out:
                out.append(name)
    return out


def _record_regs(records, solution, addrs, harness):
    state = solution.get("state")
    inputs = {
        name: sym
        for name, sym in (solution.get("inputs") or {}).items()
        if name not in harness
    }
    if not inputs:
        return
    values = min_ints(state, list(inputs.values()))
    for (name, _sym), value in zip(inputs.items(), values):
        if value is None:
            continue
        column = f"{_DATA_PREFIX}{name}"
        for insn in addrs:
            records.setdefault(insn, {}).setdefault(column, set()).add(value)


def _record_memory(records, solution, addrs, image):
    state = solution.get("state")
    data_addrs = set(solution.get("data_addrs") or ())
    accesses = [
        entry for entry in (solution.get("accesses") or ()) if entry[1] in addrs
    ]
    if not accesses:
        return
    resolved = eval_ints(
        state, [expr for entry in accesses for expr in (entry[0], entry[3])]
    )
    for i, (_addr_sym, insn, kind, _value_sym) in enumerate(accesses):
        addr, value = resolved[2 * i], resolved[2 * i + 1]
        if addr is None or addr not in data_addrs:
            continue
        if kind == "mem-stores":
            if value is None:
                continue
            image[addr] = value
        elif value is None:
            value = image.get(addr)
        if value is None:
            continue
        row = records.setdefault(insn, {})
        column = f"{_DATA_PREFIX}0x{addr:x}"
        row.setdefault(column, set()).add(value)


def _data_records(arch, solutions, executed, data=None):
    try:
        harness = set(arch._HARNESS_REGS or ())
    except Exception:
        harness = set()
    seed = _memory_image(data)
    records = {}
    for solution, addrs in zip(solutions or (), executed or ()):
        if not addrs:
            continue
        _record_regs(records, solution, addrs, harness)
        _record_memory(records, solution, addrs, dict(seed))
    return records


def _memory_image(data):
    out = {}
    for key, value in ((data or {}).get("mem") or {}).items():
        addr = _to_int_or(_parse_addr_key(key)[0], None)
        value = _to_u64(_first_scalar(value))
        if addr is not None and value is not None:
            out[addr] = value
    return out


def _data_columns(records, addresses):
    records = records or {}
    return {
        column: [
            _data_value(records.get(address, {}).get(column)) for address in addresses
        ]
        for column in sorted({c for row in records.values() for c in row})
    }


def _data_value(values):
    if not isinstance(values, (set, frozenset)):
        return values
    if not values:
        return None
    return sorted(values)


def _file_label(file):
    base = os.path.basename(str(file))
    sha = _content_hash8(file)
    return f"{base}@{sha}" if sha else base


def _branch_labels(arch, insns):
    labels = {}
    for insn in insns:
        for target in arch.branch_imm_targets(insn):
            labels.setdefault(int(target), arch.branch_label(target))
    return labels


def _instructions(arch, blob, base):
    decoder = arch.disassembler()
    try:
        decoder.detail = True
    except Exception:
        pass
    limit = base + len(blob)
    found = {}
    queue = deque([base])
    while queue:
        start = queue.popleft()
        for insn in decoder.disasm(blob[start - base :], start):
            if insn.address in found:
                break
            found[insn.address] = insn
            for target in arch.branch_imm_targets(insn):
                if base <= target < limit and target not in found:
                    queue.append(target)
    return [found[address] for address in sorted(found)]


def _block_addrs(arch, blob, base, start):
    offset = int(start) - base
    if offset < 0 or offset >= len(blob):
        return []
    try:
        full = list(arch.disassembler().disasm(bytes(blob), int(base)))
    except Exception:
        full = []
    if full:
        try:
            idx = next(
                i for i, insn in enumerate(full) if int(insn.address) == int(start)
            )
        except StopIteration:
            idx = None
        if idx is not None:
            targets = set()
            for insn in full:
                try:
                    targets.update(
                        int(target) for target in arch.branch_imm_targets(insn)
                    )
                except Exception:
                    continue
            lo = idx
            while lo > 0:
                try:
                    current = int(full[lo].address)
                except Exception:
                    break
                if current in targets:
                    break
                prev = full[lo - 1]
                try:
                    pm = str(prev.mnemonic or "").strip().lower()
                except Exception:
                    break
                if arch.is_branch_mnemonic(pm) or arch.is_return_mnemonic(pm):
                    break
                lo -= 1
            out = []
            for insn in full[lo:]:
                out.append(int(insn.address))
                try:
                    m = str(insn.mnemonic or "").strip().lower()
                except Exception:
                    break
                if arch.is_branch_mnemonic(m) or arch.is_return_mnemonic(m):
                    break
            return out
    out = []
    for insn in arch.disassembler().disasm(bytes(blob[offset:]), int(start)):
        out.append(int(insn.address))
        if arch.is_branch_mnemonic(insn.mnemonic) or arch.is_return_mnemonic(
            insn.mnemonic
        ):
            break
    return out


def _state_rip(state):
    try:
        return int(state.solver.eval(state.regs.rip))
    except Exception:
        return None


def _executed_by_solution(project, solutions, start, end, ranges=None):
    out = []
    for solution in solutions or ():
        addrs = list(_executed_bbl_addrs([solution]))
        try:
            rip = _state_rip(solution["state"])
        except Exception:
            rip = None
        probe = set()
        if rip is not None:
            addrs.append(rip)
            probe.add(rip)
        if ranges:
            out.append(
                _block_insn_addrs(project, addrs, None, None, ranges, probe=probe)
            )
        else:
            out.append(_block_insn_addrs(project, addrs, start, end, probe=probe))
    return out


def _block_insn_addrs(project, addrs, start, end, ranges=None, probe=None):
    bounds = tuple(
        (int(item[1]), int(item[2])) if len(item) == 3 else (int(item[0]), int(item[1]))
        for item in (ranges or ())
    )
    probe = {int(item) for item in (probe or ())}
    out = set()

    def _keep(at):
        try:
            at = int(at)
        except (TypeError, ValueError):
            return None
        if bounds:
            if not any(lo <= at < hi for lo, hi in bounds):
                return None
        else:
            if start is not None and at < start:
                return None
            if end is not None and at >= end:
                return None
        return at

    def _blob_block(addr):
        lower = upper = None
        if bounds:
            for lo, hi in bounds:
                if lo <= addr < hi and (lower is None or lo > lower):
                    lower, upper = lo, hi
        elif start is not None:
            lower = int(start)
            upper = int(end) if end is not None else None
        if (
            lower is None
            or upper is None
            or upper <= lower
            or not (lower <= addr < upper)
        ):
            return None
        try:
            blob = project.loader.memory.load(int(lower), int(upper - lower))
            return _block_addrs(get_arch(project), blob, int(lower), addr)
        except Exception:
            return None

    for addr in addrs:
        addr = int(addr)
        if addr in probe or (start is not None and end is not None):
            block_addrs = _blob_block(addr)
            if block_addrs:
                for at in block_addrs:
                    kept = _keep(at)
                    if kept is not None:
                        out.add(kept)
                continue
        try:
            insns = list(project.factory.block(addr).capstone.insns)
        except Exception:
            continue
        for insn in insns:
            kept = _keep(insn.address)
            if kept is not None:
                out.add(kept)
    return out


def _asm_instructions(
    asm, setup=None, records=None, data=None, teardown=None, config=None
):
    arch = get_arch()
    code = arch.normalize_asm(f"{asm};")
    base = arch._SETUP_BASE
    setup_text = (
        "\n".join(str(x) for x in setup) if isinstance(setup, (list, tuple)) else setup
    )
    teardown_text = (
        "\n".join(str(x) for x in teardown)
        if isinstance(teardown, (list, tuple))
        else teardown
    )
    setup_asm = arch.normalize_asm(f"{setup_text};") if setup_text else None
    full = f"{setup_asm}\n{code}" if setup_asm else code
    try:
        blob = arch.assemble(full, base)
        setup_len = len(arch.assemble(setup_asm, base)) if setup_asm else 0
    except Exception as ex:
        raise ValueError(f"cannot assemble {asm!r}: {ex}") from ex
    if not blob:
        raise ValueError(f"cannot assemble {asm!r}: no instructions")
    code_base = base + setup_len
    insns = _instructions(arch, blob[setup_len:], code_base)
    _, data_records = _explored_asm_addrs(
        arch,
        code,
        blob,
        base,
        setup_asm,
        data,
        teardown_text,
        config=config,
    )
    if records is not None:
        records.update(data_records)
    return code, insns


def _warn_records(target, ex):
    print(
        f"cannot explore {str(target)[:120]!r}: {ex}; showing instructions "
        "without explored state",
        file=sys.stderr,
    )


def _explored_asm_addrs(
    arch, code, blob, base, setup, data=None, teardown=None, config=None
):
    try:
        stack_size, stack_align = _stack_size(config), _stack_align(config)
        solutions = explore_asm(
            code,
            setup,
            data,
            arch,
            stack_size=stack_size,
            stack_align=stack_align,
            stack_mode=_stack_mode(config),
            stack_seed=_config_seed_int(config),
            teardown_asm=teardown,
            errors=True,
        )
    except Exception as ex:
        _warn_records(code, ex)
        return set(), {}
    executed = []
    keep = set()
    for solution in solutions or ():
        addrs = list(_executed_bbl_addrs([solution]))
        rip = _state_rip(solution["state"])
        if rip is not None:
            addrs.append(rip)
        found = set()
        for addr in addrs:
            found.update(_block_addrs(arch, blob, base, addr))
        executed.append(found)
        keep |= found
    return keep, _data_records(arch, solutions, executed, data)


def _binary_instructions(
    file,
    target,
    config,
    setup,
    records=None,
    data=None,
    teardown=None,
    debug=False,
    decompile=False,
):
    import angr

    spec = _merge_config(config)
    path = resolve_exec(file)
    if not os.path.exists(path):
        raise ValueError(f"file {path!r} does not exist")
    project = angr.Project(
        path,
        auto_load_libs=_resolve_lib(spec),
        load_debug_info=bool(debug or decompile),
    )
    pattern = _normalize_target(target)
    if not pattern:
        raise ValueError(
            "a symbol or region is required, e.g. `perf analyze a.out:fizz_buzz`"
        )
    funcs, prototypes = functions(project)
    matched = list(resolve_targets(project, pattern, funcs))
    if not matched:
        raise _unresolved(project, file, pattern)
    if len(matched) > 1:
        raise _unresolved(project, file, pattern, ambiguous=True)
    label, start, end = matched[0]
    start, end = int(start), int(end)
    if end <= start:
        raise ValueError(f"empty region {label!r}: end <= start")
    try:
        project.loader.memory.load(start, end - start)
    except Exception as ex:
        raise ValueError(f"cannot read {label!r} at 0x{start:x}: {ex}") from ex
    arch = get_arch(project)
    ranges = _followed_ranges(project, label, start, end, funcs)
    insns = _range_instructions(project, arch, ranges)
    _, data_records = _explored_binary_addrs(
        project,
        arch,
        spec,
        label,
        start,
        end,
        funcs,
        prototypes,
        setup,
        data,
        teardown,
        ranges=ranges,
    )
    if records is not None:
        records.update(data_records)
    debug_map = _debug_entries(project) if debug else {}
    if decompile:
        try:
            info = _decompile_info(project, label, start, insns)
        except Exception as ex:
            raise ValueError(f"cannot decompile {label!r}: {ex}") from ex
        debug_map[_DECOMPILE_KEY] = info
    return (
        label,
        _file_label(file),
        insns,
        debug_map,
    )


def _explored_binary_addrs(
    project,
    arch,
    spec,
    label,
    start,
    end,
    funcs,
    prototypes,
    setup,
    data=None,
    teardown=None,
    ranges=None,
):
    keep = set()
    records = {}
    keep_ranges = ranges or _followed_ranges(project, label, start, end, funcs)
    for name, first, last in _explored_ranges(label, start, end, funcs):
        found, data_records = _explored_range(
            project,
            arch,
            spec,
            name,
            first,
            last,
            funcs,
            prototypes,
            setup,
            data,
            teardown,
            ranges=keep_ranges,
        )
        keep |= found
        _merge_records(records, data_records)
    return keep, records


def _merge_records(into, other):
    for insn, row in (other or {}).items():
        target = into.setdefault(insn, {})
        for column, values in row.items():
            target.setdefault(column, set()).update(values)
    return into


def _explored_ranges(label, start, end, funcs):
    ranges = [(label, start, end)]
    seen = {(int(start), int(end))}
    for name, (first, last) in (funcs or {}).items():
        first, last = int(first), int(last)
        bounds = (first, last)
        if start < first < end and bounds != (start, end) and bounds not in seen:
            seen.add(bounds)
            ranges.append((name, first, last))
    return ranges


def _followed_ranges(project, label, start, end, funcs):
    ranges = _explored_ranges(label, start, end, funcs)
    seen = {(int(start), int(end))}
    starts = {}
    for name, (first, last) in (funcs or {}).items():
        first, last = int(first), int(last)
        starts.setdefault(first, (name, first, last))
    pending = list(ranges)
    while pending:
        _name, first, last = pending.pop()
        for target in _tail_jump_targets(project, first, last):
            item = starts.get(int(target))
            if item is None:
                continue
            bounds = (item[1], item[2])
            if bounds in seen:
                continue
            seen.add(bounds)
            ranges.append(item)
            pending.append(item)
    return ranges


def _range_instructions(project, arch, ranges):
    insns = []
    seen = set()
    for _name, first, last in ranges:
        try:
            blob = project.loader.memory.load(int(first), int(last) - int(first))
        except Exception:
            continue
        for insn in _instructions(arch, blob, int(first)):
            address = int(insn.address)
            if address in seen:
                continue
            seen.add(address)
            insns.append(insn)
    return insns


def _explored_range(
    project,
    arch,
    spec,
    label,
    start,
    end,
    funcs,
    prototypes,
    setup,
    data=None,
    teardown=None,
    ranges=None,
):
    prototype = _func_prototype(prototypes.get(label), data)
    try:
        solutions = explore(
            project,
            start,
            end,
            setup_asm=_setup_asm_for(project, project.loader.main_object, arch, setup),
            funcs=funcs,
            data=data,
            prototype=prototype,
            stack_size=_stack_size(spec),
            stack_align=_stack_align(spec),
            stack_mode=_stack_mode(spec),
            stack_seed=_config_seed_int(spec),
            teardown_asm=teardown,
            errors=True,
        )
    except Exception:
        try:
            if isinstance(setup, (list, tuple)):
                fallback_setup = "\n".join(str(x) for x in setup if str(x).strip())
                fallback_setup = fallback_setup or None
            else:
                fallback_setup = setup
            solutions = explore(
                project,
                start,
                end,
                setup_asm=fallback_setup,
                funcs=funcs,
                data=data,
                prototype=prototype,
                stack_size=_stack_size(spec),
                stack_align=_stack_align(spec),
                stack_mode=_stack_mode(spec),
                stack_seed=_config_seed_int(spec),
                teardown_asm=teardown,
                errors=True,
            )
        except Exception as ex:
            _warn_records(label, ex)
            return set(), {}
    executed = _executed_by_solution(project, solutions, start, end, ranges)
    keep = set()
    for addrs in executed:
        keep |= addrs
    return keep, _data_records(arch, solutions, executed, data)


def _stack_entry(spec):
    stack = (spec or {}).get("stack") or {}
    if isinstance(stack, (list, tuple)):
        stack = next((entry for entry in stack if isinstance(entry, dict)), {})
    return stack if isinstance(stack, dict) else {}


def _stack_size(spec):
    value = _stack_entry(spec).get("size")
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return _DEFAULT_STACK_SIZE
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stack_align(spec):
    value = _stack_entry(spec).get("align")
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return 16
    try:
        return int(value)
    except (TypeError, ValueError):
        return 16


def _stack_mode(spec):
    value = _stack_entry(spec).get("mode", "random")
    text = str(value).strip().lower() if isinstance(value, str) else "random"
    return text if text in ("random", "zero", "fixed") else "random"


def _att_lines(insns):
    decoder = get_arch().att_disassembler()
    out = []
    for insn in insns:
        line = None
        try:
            for att in decoder.disasm(bytes(insn.bytes), int(insn.address)):
                line = att.mnemonic + (f" {att.op_str}" if att.op_str else "")
                break
        except Exception:
            line = None
        out.append(line)
    return out


def _osaca_semantics(parser, model):
    import inspect

    from osaca.semantics.arch_semantics import ArchSemantics

    try:
        params = list(inspect.signature(ArchSemantics.__init__).parameters)
        needs_parser = "parser" in params or params.index("machine_model") > 1
    except (TypeError, ValueError):
        needs_parser = False
    return ArchSemantics(parser, model) if needs_parser else ArchSemantics(model)


def _form_value(form, key):
    try:
        value = getattr(form, key)
    except (AttributeError, KeyError, TypeError):
        return None
    return value


@_serialized
def _osaca_cycles(lines):
    if not lines:
        return []
    from osaca.osaca import DEFAULT_ARCHS
    from osaca.parser import get_parser
    from osaca.semantics.hw_model import MachineModel
    from osaca.semantics.isa_semantics import INSTR_FLAGS

    isa = get_arch().osaca_isa()
    parser = get_parser(isa)
    semantics = _osaca_semantics(parser, MachineModel(arch=DEFAULT_ARCHS[isa]))
    forms = []
    for line in lines:
        try:
            forms.append(parser.parse_line(line or ""))
        except Exception:
            forms.append(None)
    known = [form for form in forms if form is not None]
    if known:
        normalize = getattr(semantics, "normalize_instruction_forms", None)
        if normalize is not None:
            normalize(known)
        semantics.add_semantics(known)
    out = []
    for form in forms:
        flags = _form_value(form, "flags") or ()
        latency = _form_value(form, "latency")
        throughput = _form_value(form, "throughput")
        unknown = (
            form is None
            or INSTR_FLAGS.TP_UNKWN in flags
            or INSTR_FLAGS.LT_UNKWN in flags
            or latency is None
            or throughput is None
        )
        try:
            out.append(
                (float("nan"), float("nan"))
                if unknown
                else (float(latency), float(throughput))
            )
        except (TypeError, ValueError):
            out.append((float("nan"), float("nan")))
    return out


def _instruction_frame(insns, records=None):
    labels = _branch_labels(get_arch(), insns)
    cycles = _osaca_cycles(_att_lines(insns))
    df = pd.DataFrame(
        [
            {
                "index": index,
                "address": int(insn.address),
                "encoding": " ".join(f"{byte:02x}" for byte in bytes(insn.bytes)),
                "size": int(insn.size),
                "latency": latency,
                "throughput": throughput,
                "assembly": _render_insn(insn, labels),
            }
            for (index, insn), (latency, throughput) in zip(enumerate(insns), cycles)
        ],
        columns=[*_INSN_COLUMNS, *_OSACA_COLUMNS, *_TEXT_COLUMNS],
    )
    df["index"] = df["index"].astype("int64")
    data = _data_columns(records, df["address"].tolist())
    if data:
        df = pd.concat([df, pd.DataFrame(data, index=df.index)], axis=1)
    return df


def _decompile_function(cfg, label, start):
    funcs = getattr(getattr(cfg, "kb", None), "functions", None)
    if funcs is None:
        return None
    try:
        func = funcs.get_by_addr(int(start))
    except Exception:
        func = None
    if func is not None:
        return func
    for name in (label, str(label or "").split("(")[0].strip()):
        if not name:
            continue
        try:
            func = funcs.get(name)
        except Exception:
            func = None
        if func is not None:
            return func
    try:
        return funcs.floor_func(int(start))
    except Exception:
        return None


def _decompile_info(project, label, start, insns):
    cfg = project.analyses.CFGFast(normalize=True)
    func = _decompile_function(cfg, label, start)
    if func is None:
        raise ValueError(f"no function found at 0x{int(start):x}")
    dec = project.analyses.Decompiler(func, cfg=cfg)
    codegen = getattr(dec, "codegen", None)
    text = str(getattr(codegen, "text", "") or "").rstrip()
    if not text.strip():
        raise ValueError("decompiler produced no code")
    lines = text.splitlines()
    line_starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            line_starts.append(index + 1)

    def _line_at(pos):
        try:
            pos = int(pos)
        except (TypeError, ValueError):
            return None
        if pos < 0:
            return None
        pos = min(pos, max(0, len(text) - 1))
        return bisect.bisect_right(line_starts, pos)

    table = getattr(codegen, "map_addr_to_pos", None)
    mapping = {}
    try:
        items = list(table.items())
    except Exception:
        items = []
    for item in items:
        try:
            addr, element = item
            pos = getattr(element, "posmap_pos", element)
        except Exception:
            continue
        line = _line_at(pos)
        if line is None:
            continue
        try:
            mapping[int(addr)] = line
        except (TypeError, ValueError):
            continue
    source = f"<decompiled {label}>"
    by_addr = {}
    for insn in insns or ():
        try:
            addr = int(insn.address)
        except (TypeError, ValueError):
            continue
        line = mapping.get(addr)
        if line is None and table is not None:
            try:
                line = _line_at(table.get_nearest_pos(addr))
            except Exception:
                line = None
        if line is None or not 0 < line <= len(lines):
            continue
        by_addr[addr] = (source, line, lines[line - 1].rstrip())
    return {"text": text, "source": source, "by_addr": by_addr}


def _decompile_rows(addresses, info):
    by_addr = (info or {}).get("by_addr") or {}
    out = []
    for addr in addresses or ():
        try:
            out.append(by_addr.get(int(addr)))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _debug_entries(project):
    try:
        table = project.loader.main_object.addr_to_line
    except Exception:
        return {}
    try:
        items = list(table.items())
    except Exception:
        return {}
    out = {}
    for addr, entries in items:
        try:
            at = int(addr)
        except Exception:
            continue
        try:
            picked = sorted(entries)[0] if entries else None
        except Exception:
            continue
        if picked is None:
            continue
        try:
            out[at] = (str(picked[0]), int(picked[1]))
        except Exception:
            continue
    return out


def _source_text(path, number, cache):
    try:
        num = int(number)
    except Exception:
        return ""
    if num <= 0:
        return ""
    try:
        lines = cache.get(path)
        if lines is None:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                lines = fh.read().splitlines()
            cache[path] = lines
    except Exception:
        return ""
    try:
        return lines[num - 1].rstrip()
    except Exception:
        return ""


def _debug_rows(addresses, table):
    cache = {}
    out = []
    for addr in addresses or ():
        try:
            at = int(addr)
        except Exception:
            out.append(None)
            continue
        entry = (table or {}).get(at)
        if entry is None:
            out.append(None)
            continue
        try:
            src, num = entry
            out.append((str(src), int(num), _source_text(src, num, cache)))
        except Exception:
            out.append(None)
    return out


def _target_lowmap(addresses):
    out = {}
    for value in addresses or ():
        try:
            at = int(value)
        except (TypeError, ValueError):
            continue
        out.setdefault(at & _PAGE_MASK, []).append(at)
    return out


def _infer_biases(ips, addresses):
    lowmap = _target_lowmap(addresses)
    if not lowmap or not ips:
        return []
    counts = Counter()
    for value in ips:
        try:
            ip = int(value, 0) if isinstance(value, str) else int(value)
        except (TypeError, ValueError):
            continue
        for at in lowmap.get(ip & _PAGE_MASK, ()):
            counts[ip - at] += 1
    if not counts:
        return []
    top = max(counts.values())
    threshold = max(1, int(top * 0.05))
    biases = [(bias, n) for bias, n in counts.items() if n >= threshold]
    biases.sort(key=lambda item: item[1], reverse=True)
    return biases[:32]


def _translate_frame_ips(frame, addresses):
    column = next((c for c in _ADDRESS_COLUMNS if c in frame.columns), None)
    if column is None or column != "ip":
        return frame
    try:
        targets = [int(a) for a in (addresses or ())]
    except (TypeError, ValueError):
        return frame
    if not targets:
        return frame
    raw = [_to_int_or(v, None) for v in frame[column].tolist()]
    lowmap = _target_lowmap(targets)
    keep_dsos = None
    if "dso" in frame.columns:
        dsos = [d if isinstance(d, str) and d else "" for d in frame["dso"].tolist()]
        dso_hits = Counter()
        for value, dso in zip(raw, dsos):
            if value is None:
                continue
            if lowmap.get(int(value) & _PAGE_MASK, ()):
                dso_hits[dso] += 1
        if dso_hits:
            top = max(dso_hits.values())
            keep_dsos = {d for d, n in dso_hits.items() if n >= top}
    else:
        dsos = ["" for _ in raw]
    ips = [
        v
        for v, d in zip(raw, dsos)
        if v is not None and (keep_dsos is None or d in keep_dsos)
    ]
    biases = _infer_biases(ips, targets)
    if not biases:
        return frame
    by_count = dict(biases)
    translated = []
    skipped = 0
    for value, dso in zip(raw, dsos):
        if value is None:
            translated.append(None)
            continue
        if keep_dsos is not None and dso not in keep_dsos:
            translated.append(None)
            skipped += 1
            continue
        candidates = lowmap.get(int(value) & _PAGE_MASK, ())
        best, best_n = None, -1
        for at in candidates:
            n = by_count.get(int(value) - at)
            if n is not None and n > best_n:
                best, best_n = at, n
        if best is not None:
            translated.append(best)
        else:
            translated.append(None)
            skipped += 1
    if skipped:
        print(
            f"cannot map {skipped} sample(s) onto the target; "
            "they are left out of the joined counters",
            file=sys.stderr,
        )
    out = frame.copy()
    out[column] = translated
    return out


def _per_ip_frame(frame):
    try:
        columns = set(frame.columns)
    except Exception:
        return False
    if "mode" in columns:
        try:
            if any(str(value) == "record" for value in frame["mode"].tolist()):
                return False
        except Exception:
            pass
    if {"new_address", "moved"} & columns:
        return False
    if "ip" in columns:
        return True
    if not set(_ADDRESS_COLUMNS) & columns:
        return False
    if {"event", "period"} <= columns:
        return True
    if {"iteration", "name"} <= columns:
        return False
    return True


def _per_ip_table(frame, addresses=None):
    if frame is None or getattr(frame, "empty", False):
        return None
    if not _per_ip_frame(frame):
        return None
    if addresses is not None:
        try:
            frame = _translate_frame_ips(frame, addresses)
        except Exception:
            pass
    column = next((c for c in _ADDRESS_COLUMNS if c in frame.columns), None)
    if column is None:
        return None
    df = frame.copy()
    df[column] = [_to_int_or(v, None) for v in df[column].tolist()]
    df = df[df[column].notna()]
    if df.empty:
        return None
    if all(c in df.columns for c in _RECORD_COLUMNS):
        table = df.pivot_table(
            index=column,
            columns="event",
            values="period",
            aggfunc=lambda values: values.sum(min_count=1),
        )
        table.columns.name = None
        return table
    keep = [
        c
        for c in df.columns
        if c != column
        and c not in _NON_METRIC_COLUMNS
        and not str(c).startswith(_SKIP_COLUMNS)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df.groupby(column)[keep].sum(min_count=1) if keep else None


def _identities(frame, file_label, name):
    fallback = {"file": file_label, "name": name}
    if _per_ip_frame(frame) and any(c in frame.columns for c in _ADDRESS_COLUMNS):
        return [fallback]
    keys = [c for c in _IDENTITY if c in frame.columns]
    if not keys:
        return [fallback]
    out = []
    for values in frame[keys].drop_duplicates().to_dict(orient="records"):
        identity = dict(fallback)
        for key, value in values.items():
            if not pd.isna(value):
                identity[key] = value
        out.append(identity)
    return out


def _args_frames(args):
    frames = args if isinstance(args, (list, tuple)) else [args]
    out = []
    for f in frames:
        if f is None:
            continue
        if not isinstance(f, pd.DataFrame):
            raise ValueError(
                f"args must be a DataFrame, got {type(f).__name__!r}; "
                "pass perf.profile/benchmark output"
            )
        if getattr(f, "empty", False):
            continue
        out.append(f)
    return out


def _identity_matches(identity, name):
    want = str(name or "")
    got = str(identity.get("name", "") or "")
    if not want or not got:
        return False
    if got == want:
        return True
    head, sep, tail = got.rpartition("-")
    if sep and len(tail) == 8:
        try:
            int(tail, 16)
        except ValueError:
            return False
        return head == want
    return False


def _args_identity(args, file_label, name):
    fallback = {"file": file_label, "name": name}
    for frame in _args_frames(args):
        if _per_ip_frame(frame):
            continue
        if not any(c in frame.columns for c in _IDENTITY):
            continue
        identities = _identities(frame, file_label, name)
        if not identities:
            continue
        for identity in identities:
            if _identity_matches(identity, name):
                return identity
        return identities[0]
    return fallback


def _args_table(args, addresses=None):
    tables = [
        t
        for t in (_per_ip_table(f, addresses) for f in _args_frames(args))
        if t is not None
    ]
    if not tables:
        return None
    if len(tables) == 1:
        return tables[0]
    return pd.concat(tables).groupby(level=0).sum(min_count=1)


def _merge_args(df, args, file_label, name):
    identity = _args_identity(args, file_label, name)
    for key in _IDENTITY:
        df[key] = identity.get(key)
    try:
        addresses = df["address"].tolist()
    except (KeyError, TypeError, ValueError):
        addresses = []
    table = _args_table(args, addresses)
    if table is not None:
        index = pd.Index(df["address"].tolist())
        skipped = [c for c in table.columns if c in df.columns]
        if skipped:
            print(
                f"measured column(s) {', '.join(skipped)} already present; "
                "keeping the analyzed values",
                file=sys.stderr,
            )
        added = {}
        for column in table.columns:
            if column in df.columns:
                continue
            added[column] = table[column].reindex(index).to_numpy()
        if added:
            df = pd.concat([df, pd.DataFrame(added, index=df.index)], axis=1)
    return df


def _apply_filter(df, filter):
    if not filter:
        return df
    try:
        return query(df, filter)
    except Exception as ex:
        raise ValueError(f"invalid filter {filter!r}: {ex}") from ex


def _select(df, columns, strict=True):
    wanted = _expand(df, columns)
    matched = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    wild_missing = [c for c in missing if any(w in c for w in _WILDCARDS)]
    concrete = [c for c in missing if c not in wild_missing]
    if concrete:
        _derive(df, concrete, strict=strict)
        matched = [c for c in wanted if c in df.columns]
        missing = [c for c in wanted if c not in df.columns]
        wild_missing = [c for c in missing if any(w in c for w in _WILDCARDS)]
    if wild_missing and strict and not matched:
        available = ", ".join(str(c) for c in df.columns)
        raise ValueError(
            f"unknown columns: {', '.join(wild_missing)}; available: {available}"
        )
    missing = [c for c in missing if c not in wild_missing]
    if missing:
        _derive(df, missing, strict=strict)
        missing = [c for c in wanted if c not in df.columns]
    if missing and strict:
        available = ", ".join(str(c) for c in df.columns)
        raise ValueError(
            f"unknown columns: {', '.join(missing)}; available: {available}"
        )
    wanted = [c for c in wanted if c in df.columns]
    if strict:
        return df[wanted]
    keep = [_LEAD_COLUMN] if _LEAD_COLUMN in wanted else []
    keep += [c for c in wanted if c not in keep]
    return df[keep]


def _expand(df, columns):
    out = []
    seen = set()
    for column in columns:
        text = _bracket_to_dot(column)
        if any(wildcard in text for wildcard in _WILDCARDS):
            matched = False
            for found in df.columns:
                if found not in seen and _match(text, str(found)):
                    seen.add(found)
                    out.append(found)
                    matched = True
            if not matched and text not in seen:
                try:
                    seen.add(text)
                except TypeError:
                    pass
                out.append(text)
            continue
        if text not in seen:
            try:
                seen.add(text)
            except TypeError:
                pass
            out.append(text)
    return out


def _bracket_to_dot(column):
    text = str(column)
    if text.startswith("data[") and text.endswith("]"):
        return f"{_DATA_PREFIX}{text[len('data[') : -1]}"
    return text


def _derive(df, names, strict=False):
    for name in names:
        try:
            df[name] = df.eval(quote_columns(df, name), engine="python")
        except Exception as ex:
            if strict and any(op in str(name) for op in ("+", "-", "*", "/", "%", "(")):
                raise ValueError(f"invalid expression {name!r}: {ex}") from ex
            continue
    return df


def _teardown_end(teardown_asm, offset=0x10000):
    arch = get_arch()
    base = arch._SETUP_BASE + offset
    encoding = arch.assemble(arch.normalize_asm(f"{teardown_asm};"), base)
    if not encoding:
        return None, 0
    return base, len(encoding)


def _explore_teardown(proj, states, sym_regs, teardown_asm, keep, pins=()):
    arch = get_arch()
    base, size = _teardown_end(teardown_asm)
    if not size:
        return []
    out = []
    for state in states or ():
        try:
            found = state.copy()
        except Exception:
            continue
        try:
            td = arch.assemble(arch.normalize_asm(f"{teardown_asm};"), base)
            try:
                last = list(arch.disassembler().disasm(td, base))[-1]
                has_ret = arch.is_return_mnemonic(last.mnemonic)
            except Exception:
                has_ret = False
            if not has_ret:
                td += arch.return_stub_bytes()
            size = len(td)
            found.memory.map_region(base, size + 256, 7, init_zero=True)
            found.memory.store(base, td)
            try:
                import claripy

                found.memory.store(
                    arch.stack_pointer(found),
                    claripy.BVV(base + size, 64),
                    size=8,
                    endness=found.arch.memory_endness,
                )
            except Exception:
                pass
            arch.set_ip(found, base)
            simgr = proj.factory.simgr(found)
            simgr.explore(find=[base + size])
        except Exception:
            continue
        out.extend(
            _solution_records(
                simgr.found + simgr.active + simgr.deadended, sym_regs, keep, pins
            )
        )
    return out


def _reached(simgr, errors=False):
    states = simgr.found + simgr.active + simgr.deadended
    if not errors:
        return states
    return states + [
        getattr(error, "state", error)
        for error in simgr.errored
        if error is not None and not _unmapped_fault(error)
    ]


def _unmapped_fault(error):
    text = str(getattr(error, "error", error) or "").lower()
    return "unmapped" in text or "non-readable" in text or "non-writable" in text


def _explore_states(proj, state, find, world=()):
    simgr = proj.factory.simgr(state)
    wanted = {int(addr) for addr in find or ()}
    ranges = tuple(world or ())
    simgr.stashes.setdefault("found", [])
    simgr.stashes.setdefault("pruned", [])
    for _ in range(_EXPLORE_STEPS):
        simgr.move(from_stash="active", to_stash="pruned", filter_func=_over_budget)
        simgr.move(
            from_stash="active",
            to_stash="found",
            filter_func=lambda s: _terminal(s, wanted, ranges),
        )
        if not simgr.active:
            break
        if len(simgr.found) + len(simgr.active) > _EXPLORE_STATES:
            break
        try:
            simgr.step()
        except Exception as ex:  # pragma: no cover
            _debug_log(f"exploration step failed: {ex!r}")
            break
    return simgr


def _terminal(state, wanted, ranges):
    addr = _state_addr(state)
    if addr is None:
        return True
    if addr in wanted:
        return True
    return not any(lo <= addr < hi for lo, hi in ranges)


def _nondet_options(state):
    _install_address_window(state)
    _map_address_window(state)
    _track_mem_access(state)
    _track_branches(state)


def _install_address_window(state):
    memory = getattr(state, "memory", None)
    if memory is None or not hasattr(memory, "read_strategies"):
        return
    from angr import concretization_strategies

    strategy = _address_window_class()()
    single = concretization_strategies.SimConcretizationStrategyAny()
    reads = list(getattr(memory, "read_strategies", None) or ())
    writes = list(getattr(memory, "write_strategies", None) or ())
    memory.read_strategies = [strategy, single, *reads]
    memory.write_strategies = [strategy, *writes]


def _map_address_window(state):
    lo, hi = _ADDRESS_WINDOW
    try:
        state.memory.map_region(lo, hi - lo, 7)
    except Exception:
        pass


def _setup_ranges(setup_asm):
    if not setup_asm:
        return ()
    try:
        arch = get_arch()
        encoding = arch.assemble(arch.setup_jump_asm(setup_asm.rstrip("\n"), 0), 0)
    except Exception:
        return ()
    return ((arch._SETUP_BASE, len(encoding) + 256),)


def _target_ranges(proj, extra=()):
    ranges = []
    obj = getattr(getattr(proj, "loader", None), "main_object", None)
    for lo, hi in ((getattr(obj, "min_addr", None), getattr(obj, "max_addr", None)),):
        if lo is None or hi is None:
            continue
        try:
            if int(hi) > int(lo):
                ranges.append((int(lo), int(hi)))
        except (TypeError, ValueError):
            continue
    for lo, size in extra or ():
        try:
            if int(size) > 0:
                ranges.append((int(lo), int(lo) + int(size)))
        except (TypeError, ValueError):
            continue
    return ranges


def _over_budget(state):
    try:
        return int(state.globals.get("branches", 0) or 0) > _EXPLORE_BRANCHES
    except (AttributeError, TypeError, ValueError):
        return False


def _state_addr(state):
    try:
        return int(state.addr)
    except Exception:
        return None


def _collapse_access(accesses, index, addr, length, value, last=False):
    try:
        length = int(length)
    except (TypeError, ValueError):
        return
    if length <= 0:
        return
    value = int(value) & ((1 << (8 * min(length, 8))) - 1)
    slot = index.get(addr)
    if slot is None:
        index[addr] = len(accesses)
        accesses.append((addr, length, value))
        return
    if last or length > accesses[slot][1]:
        accesses[slot] = (addr, length, value)


def _pinned_branch_addrs(branch_cfg):
    pinned = set()
    for addr, value in ((branch_cfg or {}).get("mem") or {}).items():
        if value == "predictable":
            try:
                pinned.add(int(addr))
            except (TypeError, ValueError):
                continue
    return pinned


@_locked
def _models_cached(solutions, branch_cfg, arch):
    solutions = solutions if solutions is not None else []
    try:
        solutions_key = _stable(solutions)
    except Exception:
        solutions_key = repr(solutions)
    try:
        arch_key = str(getattr(arch, "_ANGR_ARCH_NAME", "") or type(arch).__name__)
    except Exception:
        arch_key = repr(arch)
    key = (
        solutions_key,
        frozenset(_pinned_branch_addrs(branch_cfg)),
        arch_key,
    )
    hit = _MODEL_CACHE.get(key)
    if hit is not None:
        return copy.deepcopy(hit)
    models = solve(solutions, branch_cfg=branch_cfg, arch=arch)
    if len(_MODEL_CACHE) >= _SOLVE_CACHE_MAX:
        _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = copy.deepcopy(models)
    return models


@_locked
def _explore_cached(key, factory):
    try:
        if key in _EXPLORE_CACHE:
            return _EXPLORE_CACHE[key]
    except TypeError:
        return factory()
    out = factory()
    try:
        if len(_EXPLORE_CACHE) >= _EXPLORE_CACHE_MAX:
            _EXPLORE_CACHE.clear()
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


def _data_touches_stack(data, arch, stack_size=None):
    try:
        lo = int(getattr(arch, "_STACK_ADDR", 0x7FFF00000000))
        size = int(stack_size or getattr(arch, "_STACK_SIZE", 0x100000))
    except (TypeError, ValueError):  # pragma: no cover
        return True
    hi = lo + max(size, 0)

    def _inside(value):
        try:
            value = int(value, 0) if isinstance(value, str) else int(value)
        except (TypeError, ValueError):
            return False
        return lo <= value < hi

    for value in ((data or {}).get("regs") or {}).values():
        items = value if isinstance(value, (list, tuple)) else [value]
        if any(_inside(item) for item in items):
            return True
    for key in (data or {}).get("mem") or {}:
        text = str(key).strip()
        if text.endswith(":"):
            text = text[:-1].strip()
        if _inside(text):
            return True
    return False


def _explore_asm_seed_key(
    code, setup_asm, arch, stack_seed, data=None, stack_size=None
):
    if stack_seed is None:
        return None
    if _data_touches_stack(data, arch, stack_size):
        return int(stack_seed)
    try:
        full = ""
        if setup_asm:
            full += setup_asm.rstrip("\n") + "\n"
        full += str(code or "").rstrip("\n") + "\n"
        encoding = arch.assemble(full, arch._SETUP_BASE)
        if encoding and not arch.uses_stack(encoding, arch._SETUP_BASE):
            return None
    except Exception:  # pragma: no cover
        pass
    return int(stack_seed)


def _explore_asm_cached(
    code,
    setup_asm,
    data,
    arch,
    stack_size,
    stack_align,
    stack_mode="random",
    stack_seed=None,
):
    key = (
        "asm",
        str(code),
        str(setup_asm or ""),
        _stable(data),
        int(stack_size or 0),
        int(stack_align or 0),
        str(stack_mode or "random"),
        _explore_asm_seed_key(
            code, setup_asm, arch, stack_seed, data=data, stack_size=stack_size
        ),
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
                stack_mode=stack_mode,
                stack_seed=stack_seed,
            )
        except Exception as ex:  # pragma: no cover
            _warn_exploration(code, ex)
            return []

    return _explore_cached(key, _run)


def _explore_target_seed_key(start, funcs, stack_seed):
    if stack_seed is None:
        return None
    try:
        if int(start) in {int(s) for s, _e in (funcs or {}).values()}:
            return None
    except (TypeError, ValueError):
        pass
    return int(stack_seed)


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
    stack_mode="random",
    stack_seed=None,
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
        str(stack_mode or "random"),
        _explore_target_seed_key(start, funcs, stack_seed),
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
            stack_mode=stack_mode,
            stack_seed=stack_seed,
        ),
    )


def _project_cache_id(file, config):
    container = config or {}
    orders = [
        str(_entry("func", alt).get("order") or "").strip().lower()
        for alt in _entry_alternatives("func", container.get("func"))
    ]
    stack_modes = [
        str(_entry("stack", alt).get("mode") or "").strip().lower()
        for alt in _entry_alternatives("stack", container.get("stack"))
    ]
    seeded = _FUNC_ORDERS[1] in orders or any(mode == "fixed" for mode in stack_modes)
    try:
        stamp = _exec_stamp(str(file))
    except (OSError, TypeError, ValueError):
        stamp = None
    outputs = (_fake_stdout(config), _fake_stderr(config))
    return (
        str(file),
        stamp,
        _resolve_lib(config),
        _entry_signature("stack", container.get("stack")),
        _entry_signature("func", container.get("func")),
        _config_seed_int(config) if seeded else None,
        outputs,
    )


def _branches_seen(solutions):
    return len(solutions or ()) > 1 or any(
        sol.get("branches") for sol in solutions or ()
    )


def _concrete_addr_value(addr):
    try:
        if isinstance(addr, bool):
            return None
        if isinstance(addr, int):
            return int(addr)
        symbolic = getattr(addr, "symbolic", None)
        if symbolic is True:
            return None
        if symbolic is False:
            try:
                args = getattr(addr, "args", None)
                if args:
                    first = args[0]
                    if isinstance(first, int) and not isinstance(first, bool):
                        return int(first)
            except (TypeError, ValueError):
                pass
            if symbolic is False and getattr(addr, "args", None) is not None:
                return None
        if isinstance(addr, str):
            return int(addr.strip(), 0)
        return int(addr)
    except (TypeError, ValueError, AttributeError):
        return None


def _memory_seen(solutions):
    try:
        stack_base = int(getattr(get_arch(), "_STACK_ADDR", 0x7FFF00000000))
    except (AttributeError, TypeError, ValueError):
        stack_base = 0x7FFF00000000
    for sol in solutions or ():
        accesses = list(sol.get("reads") or []) + list(sol.get("writes") or [])
        for entry in accesses:
            try:
                addr = entry[0] if isinstance(entry, (list, tuple)) else entry
            except (TypeError, ValueError):
                return True
            value = _concrete_addr_value(addr)
            if value is None:
                return True
            try:
                if int(value) >= int(stack_base):
                    continue
            except (TypeError, ValueError):
                return True
            return True
    return False


def _target_features(file, target, code, config, data, setup=None):
    stack = _container(config, "stack")
    size, align, mode = (
        stack["size"],
        stack["align"],
        stack["mode"],
    )
    seed = _config_seed_int(config)
    if code:
        arch = get_arch()
        if isinstance(setup, str):
            setup_text = setup
        else:
            setup_text = "\n".join(setup) if setup else ""
        setup_asm = arch.normalize_asm(setup_text + "\n") if setup_text else None
        solutions = _explore_asm_cached(
            arch.normalize_asm(f"{code};"),
            setup_asm,
            data,
            arch,
            size,
            align,
            mode,
            seed,
        )
    else:
        binary = resolve_exec(file) if file else None
        project, obj, funcs, prototypes = _bench_project(
            binary, config, stack, _container(config, "func")
        )
        arch = get_arch(project)
        target = _normalize_target(target)
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
            mode,
            seed,
        )
    return {
        "branches": _branches_seen(solutions),
        "memory": _memory_seen(solutions) or bool((data or {}).get("mem")),
    }


def _setup_names(setup):
    names = setup if isinstance(setup, (list, tuple)) else [setup]
    return [str(name).strip() for name in (names or ()) if name and str(name).strip()]


def _angr_symbol_addr(project, obj, name):
    candidates = [str(name).strip()]
    try:
        raw = obj.raw_symbol(name)
    except Exception:
        raw = None
    if raw and raw not in candidates:
        candidates.append(raw)
    for candidate in candidates:
        try:
            sym = project.loader.find_symbol(candidate)
        except Exception:
            sym = None
        if sym is not None:
            try:
                return int(sym.rebased_addr)
            except Exception:
                continue
    return None


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
        addr = _angr_symbol_addr(project, obj, name)
        if addr is not None:
            parts.append(arch.call_seq_asm(addr))
        else:
            parts.append(str(name))
    return "\n".join(parts) if parts else None


def _dedupe_combos(combos, ignore):
    if len(combos) < 2 or not ignore:
        return combos
    if not any(
        len({_stable(combo.get(key)) for combo in combos}) > 1 for key in ignore
    ):
        return combos
    out, seen = [], set()
    for combo in combos:
        signature = _stable({k: v for k, v in combo.items() if k not in ignore})
        if signature in seen:
            continue
        seen.add(signature)
        out.append(combo)
    return out


def _prune_for_target(file, target, code, combos, data, setup=None, debug=False):
    if len(combos) < 2:
        return combos
    if file and not code:
        before = len(combos)
        combos = _dedupe_combos(combos, ("code",))
        if debug and len(combos) != before:
            _debug_log(
                f"pruned {before - len(combos)} of {before} config combinations "
                "(code.align is unused for binary targets)"
            )
        if len(combos) < 2:
            return combos
    elif code and not file:
        before = len(combos)
        combos = _dedupe_combos(combos, ("func",))
        if debug and len(combos) != before:
            _debug_log(
                f"pruned {before - len(combos)} of {before} config combinations "
                "(func layout is unused for asm snippets)"
            )
        if len(combos) < 2:
            return combos
    if not any(
        len({_stable(combo.get(key)) for combo in combos}) > 1 for key in _PRUNE_KEYS
    ):
        return combos
    try:
        features = _target_features(file, target, code, combos[0], data, setup)
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


def _tail_jump_targets(proj, start, end):
    arch = get_arch(proj)
    try:
        size = max(int(end) - int(start), 1)
        blob = proj.loader.memory.load(int(start), size)
    except Exception:
        return ()
    targets = []
    for insn in arch.disassembler().disasm(bytes(blob), int(start)):
        if not arch.is_unconditional_jump(insn.mnemonic):
            continue
        for target in arch.branch_imm_targets(insn):
            target = int(target)
            if not (int(start) <= target < int(end)):
                targets.append(target)
    return tuple(targets)


def _ret_addrs(proj, start, end, funcs=None):
    arch = get_arch(proj)
    starts = {}
    for _name, bounds in (funcs or {}).items():
        try:
            first, last = int(bounds[0]), int(bounds[1])
        except (TypeError, ValueError, IndexError):
            continue
        starts.setdefault(first, (first, last))
    return _ret_addrs_at(proj, arch, int(start), int(end), starts, set())


def _ret_addrs_at(proj, arch, start, end, starts, seen):
    if (start, end) in seen:
        return set()
    seen.add((start, end))
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
    for target in _tail_jump_targets(proj, start, end):
        bounds = starts.get(int(target))
        if bounds is None or bounds == (start, end):
            continue
        addrs |= _ret_addrs_at(proj, arch, bounds[0], bounds[1], starts, seen)
    if not addrs:
        addrs = {end}
    return addrs


def _pin_unknown_regs(state, sym_regs, explicit_canon, arch, concrete=False):
    harness = set(arch._HARNESS_REGS or ())
    pinned = set()
    for name in sorted(sym_regs):
        cname = arch._canonical_reg(name)
        if cname in explicit_canon or cname in harness:
            continue
        try:
            if state.solver.symbolic(sym_regs[name]):
                value = _ASM_SCRATCH_BASE + len(pinned) * 0x1000
                state.solver.add(sym_regs[name] == value)
                if concrete:
                    try:
                        import claripy

                        try:
                            bits = sym_regs[name].size()
                        except Exception:
                            bits = 64
                        state.registers.store(name, claripy.BVV(value, bits))
                    except Exception:
                        pass
                pinned.add(name)
        except Exception:
            pass
    try:
        scratch_page = _ASM_SCRATCH_BASE & ~0xFFF
        state.memory.map_region(
            scratch_page, (len(pinned) + 1) * 0x1000, 7, init_zero=True
        )
    except Exception:
        pass
    return pinned


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


def _solution_code_pages(solutions, data):
    try:
        mem_pages = {int(addr) & ~(mmap.PAGESIZE - 1) for addr in _iter_mem_addrs(data)}
    except Exception:
        return []
    if not mem_pages:
        return []
    pages = set()
    for solution in solutions or ():
        state = solution.get("state") if isinstance(solution, dict) else None
        if state is None:
            continue
        try:
            pages.add(int(state.addr) & ~(mmap.PAGESIZE - 1))
        except Exception:
            pass
        try:
            for block in state.history.bbl_addrs:
                pages.add(int(block) & ~(mmap.PAGESIZE - 1))
        except Exception:
            pass
    return sorted(pages & mem_pages)


def _merge_code_pages(data, solutions):
    pages = _solution_code_pages(solutions, data)
    if not pages:
        return data
    merged = dict(data or {})
    merged["code"] = sorted(set(merged.get("code") or ()) | set(pages))
    return merged


def _cond_branch_analysis(insns, arch=None):
    if arch is None:
        arch = get_arch()
    return arch.cond_branch_analysis(insns)


def _branch_deps(state, model, pinned_addrs, arch=None):
    deps = {}
    try:
        project = state.project
        if arch is None:
            arch = get_arch(project)
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


def _recorded_data_addrs(solutions):
    data, control = set(), set()
    for sol in solutions or ():
        data |= set(sol.get("data_addrs") or ())
        control |= set(sol.get("control_addrs") or ())
    return data, control - data


@_locked
def _static_table_addrs(
    project, start, end, elf_obj=None, max_addrs=64, solutions=None
):
    data, _control = _recorded_data_addrs(solutions)
    key = _static_table_key(project, start, end, elf_obj, max_addrs, data)
    if key is not None and key in _STATIC_TABLE_CACHE:
        return list(_STATIC_TABLE_CACHE[key])
    found = _scan_table_addrs(project, start, end, elf_obj, max_addrs, data)
    if key is not None:
        if len(_STATIC_TABLE_CACHE) >= _STATIC_TABLE_CACHE_MAX:
            _STATIC_TABLE_CACHE.clear()
        _STATIC_TABLE_CACHE[key] = list(found)
    return found


def _static_table_key(project, start, end, elf_obj, max_addrs, data):
    try:
        binary = str(project.loader.main_object.binary)
        stamp = _exec_stamp(binary)
        base = int(getattr(elf_obj, "runtime_base", 0) or 0)
        addrs = frozenset(int(a) for a in data)
        return (
            binary,
            stamp,
            base,
            int(start),
            int(end),
            int(max_addrs),
            addrs,
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _scan_table_addrs(project, start, end, elf_obj, max_addrs, data):
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


def _reserved_addresses(elf_obj):
    try:
        return elf_obj.relocated_addresses()
    except Exception:
        return set()


def _symbolic_options(state):
    import angr

    state.options.discard(angr.options.LAZY_SOLVES)
    state.options.discard(angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY)
    for opt in (
        angr.options.TRACK_CONSTRAINTS,
        angr.options.SIMPLIFY_CONSTRAINTS,
        angr.options.STRICT_PAGE_ACCESS,
        angr.options.TRACK_JMP_ACTIONS,
        angr.options.TRACK_ACTION_HISTORY,
        angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
        angr.options.SYMBOL_FILL_UNCONSTRAINED_MEMORY,
        angr.options.NO_SYMBOLIC_JUMP_RESOLUTION,
        angr.options.SYMBOLIC_WRITE_ADDRESSES,
        angr.options.NO_CROSS_INSN_OPT,
    ):
        state.options.add(opt)


@_locked
def _address_window_class():
    global _ADDRESS_WINDOW_CLASS
    if _ADDRESS_WINDOW_CLASS is None:
        from angr import concretization_strategies

        class _AddressWindow(concretization_strategies.SimConcretizationStrategy):
            def _concretize(self, memory, addr, extra_constraints=None, **_kwargs):
                import claripy

                try:
                    bits = addr.size()
                except AttributeError:
                    return None
                if bits < 33:
                    return None
                lo, hi = _ADDRESS_WINDOW
                window = (claripy.UGE(addr, lo), claripy.ULT(addr, hi))
                if extra_constraints:
                    window = (*tuple(extra_constraints), *window)
                try:
                    value = memory.state.solver.eval(addr, extra_constraints=window)
                except Exception:
                    return None
                return [int(value)]

        _ADDRESS_WINDOW_CLASS = _AddressWindow
    return _ADDRESS_WINDOW_CLASS


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
        _record(s, "reads", "mem-loads", "mem_read")

    def _on_write(s):
        _record(s, "writes", "mem-stores", "mem_write")

    def _record(s, key, kind, event):
        addr = getattr(s.inspect, f"{event}_address")
        length = getattr(s.inspect, f"{event}_length")
        value = getattr(s.inspect, f"{event}_expr")
        s.globals[key] = [*tuple(s.globals.get(key) or ()), (addr, length, value)]
        _record_access(s, addr, kind, value)
        _bound_unknown(s, addr, value)

    state.inspect.b("mem_read", when=angr.BP_AFTER, action=_on_read)
    state.inspect.b("mem_write", when=angr.BP_AFTER, action=_on_write)


def _record_access(state, addr, kind, value):
    accesses = tuple(state.globals.get("accesses") or ())
    state.globals["accesses"] = (
        *accesses,
        (addr, state.inspect.instruction, kind, value),
    )


def _bounded_cells(state):
    cells = state.globals.get("cells")
    if cells is None:
        cells = set()
        state.globals["cells"] = cells
    return cells


def _bound_unknown(state, addr, expr):
    import claripy

    lo, hi = _ADDRESS_WINDOW
    if getattr(expr, "size", None) is None or expr.size() < 2:
        return
    if not state.solver.symbolic(expr):
        return
    try:
        values = {value for value in eval_ints(state, [addr]) if value is not None}
        if not any(lo <= value < hi for value in values):
            return
        cells = _bounded_cells(state)
        if not values - cells:
            return
        cells |= values
        bits = expr.size()
        terms = [claripy.ULE(expr, claripy.BVV(_UNKNOWN_MAX, bits))]
        if bits >= 33:
            terms.append(
                claripy.And(
                    claripy.UGE(expr, claripy.BVV(lo, bits)),
                    claripy.ULT(expr, claripy.BVV(hi, bits)),
                )
            )
        state.solver.add(claripy.Or(*terms))
    except Exception:
        return


def _track_branches(state):
    import angr

    def _on_exit(s):
        exits = tuple(s.globals.get("exits") or ())
        s.globals["exits"] = (*exits, (s.inspect.exit_target, s.inspect.instruction))
        guard = s.inspect.exit_guard
        if guard is None or getattr(guard, "concrete", False):
            return
        if str(s.inspect.exit_jumpkind).lower() not in _BRANCH_JUMPKINDS:
            return
        s.globals["branches"] = s.globals.get("branches", 0) + 1

    state.inspect.b("exit", when=angr.BP_AFTER, action=_on_exit)


def _map_state_mem_pages(state, mem):
    seen = set()
    for addr_str, value in (mem or {}).items():
        try:
            s = str(addr_str).strip()
            if s.endswith(":"):
                s = s[:-1].strip()
            addr = int(s, 0)
            page = addr & ~0xFFF
        except (TypeError, ValueError):
            continue
        if page not in seen:
            seen.add(page)
            try:
                state.memory.map_region(page, 0x1000, 7, init_zero=True)
            except Exception:
                pass
        if isinstance(value, (list, tuple)) or value is None:
            continue
        try:
            import claripy

            v = _to_u64(value)
            if v < 256:
                width = 8
            elif v <= 0xFFFFFFFF:
                width = 32
            else:
                width = 64
            state.memory.store(
                addr,
                claripy.BVV(v, width),
                endness=state.arch.memory_endness,
            )
        except Exception:
            pass


def _eval_addrs(state, values):
    out = []
    for addr in eval_ints(state, values):
        out.append(addr if addr is not None and 0 <= addr < (1 << 64) else None)
    return out


def _recorded_stats(state):
    control_insns = set()
    branches = set()
    exits = state.globals.get("exits") or ()
    for insn in (insn for _target, insn in exits):
        try:
            insn = int(insn)
        except (TypeError, ValueError):
            continue
        control_insns.add(insn)
    for addr in _eval_addrs(state, [target for target, _insn in exits]):
        if addr is not None:
            branches.add(addr)
    accesses = state.globals.get("accesses") or ()
    resolved = _eval_addrs(state, [_addr for _addr, _i, _k, _v in accesses])
    data, control = set(), set()
    for (_addr, insn, _kind, _value), value in zip(accesses, resolved):
        if value is None:
            continue
        try:
            is_control = int(insn) in control_insns
        except (TypeError, ValueError):
            is_control = False
        (control if is_control else data).add(value)
    return data - branches, control - data, branches


def _solution_records(states, sym_regs, keep=None, pins=()):
    results = []
    pins = set(pins or ())
    for found in states:
        reg = {n: s for n, s in sym_regs.items() if keep is None or keep(found, n, s)}
        data_addrs, control_addrs, branch_addrs = _recorded_stats(found)
        results.append(
            {
                "state": found,
                "reg": reg,
                "inputs": {n: s for n, s in reg.items() if n not in pins},
                "branches": int(found.globals.get("branches", 0) or 0),
                "reads": list(found.globals.get("reads", [])),
                "writes": list(found.globals.get("writes", [])),
                "accesses": list(found.globals.get("accesses", [])),
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


@_locked
def _bench_project(file, config, stack, func):
    import angr

    key = (
        *_project_cache_id(file, config),
        _stable(stack),
        _stable(func),
    )
    cached = _PROJECTS.get(key)
    if cached is not None:
        return cached
    resolved = resolve_exec(file)
    project = angr.Project(
        resolved,
        auto_load_libs=_resolve_lib(config),
        load_debug_info=False,
    )
    obj = Elf(project.loader)
    obj.requested = file
    obj.map_elf()
    obj.setup_stack(
        size=stack["size"],
        align=stack["align"],
        mode=stack.get("mode", "random"),
        seed=_config_seed_int(config),
    )

    funcs, prototypes = functions(project)

    if func["order"] == _FUNC_ORDERS[1]:
        obj.randomize_layout(
            dict(funcs),
            seed=_config_seed_int(config),
            align=func["align"],
        )

    try:
        obj.run_initializers(output=_output_fakes(config))
    except Exception as ex:
        _debug_log(f"ELF initializers failed for {file!r}: {ex!r}")

    entry = (project, obj, funcs, prototypes)
    if len(_PROJECTS) >= _PROJECTS_MAX:
        _PROJECTS.pop(next(iter(_PROJECTS)))
    _PROJECTS[key] = entry
    return entry


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


def _render_insn(insn, labels):
    op_str = insn.op_str or ""
    if _is_branch_mnemonic(insn.mnemonic) and op_str:
        op_str = _sub_intel_labels(op_str, labels)
    return f"{insn.mnemonic} {op_str}".strip()


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
