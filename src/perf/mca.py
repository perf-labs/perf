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
import os
import re
import sys
import threading
from collections import Counter, deque

import pandas as pd

from .arch import arch as get_arch
from .bench import (
    _content_hash8,
    _executed_bbl_addrs,
    _first_scalar,
    _func_prototype,
    _merge_config,
    _merge_data,
    _normalize_target,
    _render_insn,
    _resolve_lib,
    _resolve_target,
    _setup_asm_for,
    _tail_jump_targets,
    _unresolved,
    check_data,
    explore,
    explore_asm,
)
from .core import _parse_addr_key, _to_int_or, _to_u64, eval_ints, min_ints
from .data import _IDENTITY_COLUMNS, _NON_METRIC_COLUMNS, query, quote_columns
from .exec import resolve_exec
from .info import functions
from .info import targets as resolve_targets

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
_DEFAULT_EVENTS = [
    *_IDENTITY,
    *_INSN_COLUMNS,
    *_OSACA_COLUMNS,
    *_TEXT_COLUMNS,
    "data*",
]
_OSACA_LOCK = threading.RLock()


def _serialized(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _OSACA_LOCK:
            return fn(*args, **kwargs)

    return wrapper


def analyze(
    symbol=None,
    region=None,
    asm=None,
    name=None,
    *,
    event=None,
    results=None,
    config=None,
    data=None,
    setup=None,
    teardown=None,
    filter=None,
    debug=False,
):
    debug_context = None
    if isinstance(config, dict) and "debug_context" in config:
        config = dict(config)
        debug_context = _debug_context(config.pop("debug_context"))
    file, target, snippet, name, raw = _resolve_target(symbol, region, asm, name)
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
            file, target, config, setup, records, data, teardown, debug=debug
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
    df = _instruction_frame(insns, records)
    df = _merge_results(df, results, file_label, label)
    df = _apply_filter(df, filter)
    debug_rows = _debug_rows(df["address"].tolist(), debug_map) if debug else None
    df = _select(df, columns, strict=strict)
    df.attrs["file"] = file_label
    df.attrs["name"] = label
    df.attrs["code"] = code
    df.attrs["config"] = _merge_config(config)
    if debug_rows is not None:
        df.attrs["debug"] = debug_rows
    if debug_context is not None:
        df.attrs["debug_context"] = debug_context
    return df


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
    file, target, config, setup, records=None, data=None, teardown=None, debug=False
):
    import angr

    spec = _merge_config(config)
    path = resolve_exec(file)
    if not os.path.exists(path):
        raise ValueError(f"file {path!r} does not exist")
    project = angr.Project(
        path, auto_load_libs=_resolve_lib(spec), load_debug_info=bool(debug)
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
    return (
        label,
        _file_label(file),
        insns,
        _debug_entries(project) if debug else {},
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
    try:
        return int(_stack_entry(spec).get("size"))
    except (TypeError, ValueError):
        return None


def _stack_align(spec):
    try:
        return int(_stack_entry(spec).get("align"))
    except (TypeError, ValueError):
        return 16


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


def _results_frames(results):
    frames = results if isinstance(results, (list, tuple)) else [results]
    out = []
    for f in frames:
        if f is None:
            continue
        if not isinstance(f, pd.DataFrame):
            raise ValueError(
                f"results must be a DataFrame, got {type(f).__name__!r}; "
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


def _result_identity(results, file_label, name):
    fallback = {"file": file_label, "name": name}
    for frame in _results_frames(results):
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


def _result_table(results, addresses=None):
    tables = [
        t
        for t in (_per_ip_table(f, addresses) for f in _results_frames(results))
        if t is not None
    ]
    if not tables:
        return None
    if len(tables) == 1:
        return tables[0]
    return pd.concat(tables).groupby(level=0).sum(min_count=1)


def _merge_results(df, results, file_label, name):
    identity = _result_identity(results, file_label, name)
    for key in _IDENTITY:
        df[key] = identity.get(key)
    try:
        addresses = df["address"].tolist()
    except (KeyError, TypeError, ValueError):
        addresses = []
    table = _result_table(results, addresses)
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
    # Unmatched wildcards are ignored when other columns matched (e.g.
    # `data*` with no data columns); only derive concrete names.
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


@functools.lru_cache(maxsize=1024)
def _compiled_match(pattern):
    return re.compile(
        "".join(
            ".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern
        )
    )


def _match(pattern, text):
    return _compiled_match(pattern).fullmatch(text) is not None


def _derive(df, names, strict=False):
    for name in names:
        try:
            df[name] = df.eval(quote_columns(df, name), engine="python")
        except Exception as ex:
            if strict and any(op in str(name) for op in ("+", "-", "*", "/", "%", "(")):
                raise ValueError(f"invalid expression {name!r}: {ex}") from ex
            continue
    return df
