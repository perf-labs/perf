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

import os
import re

import pandas as pd

from .arch import arch as get_arch
from .arch import load as load_arch
from .bench import (
    _content_hash8,
    _executed_bbl_addrs,
    _func_prototype,
    _merge_config,
    _normalize_target,
    _render_insn,
    _resolve_lib,
    _setup_asm_for,
    asm_source_code,
    explore,
    explore_asm,
    parse_code,
)
from .bench import (
    asm_source_target as asm_target_text,
)
from .data import _IDENTITY_COLUMNS, _NON_METRIC_COLUMNS, query
from .exec import resolve_exec
from .info import functions, is_asm_source
from .info import targets as resolve_targets

_IDENTITY = tuple(c for c in _IDENTITY_COLUMNS if c != "mode")
_RECORD_COLUMNS = ("event", "period")
_ADDRESS_COLUMNS = ("ip", "address", "addr")
_INSN_COLUMNS = ("index", "address", "encoding", "size")
_OSACA_COLUMNS = ("latency", "throughput")
_TEXT_COLUMNS = ("assembly",)
_LEAD_COLUMNS = (*_IDENTITY, "index", "address")
_SKIP_COLUMNS = ("data.", "config.")
_DATA_PREFIX = "data."
_SYMBOL = re.compile(r"<BV\d+\s+(.*)>")
_SYMBOL_VAR = re.compile(r"([\w.$]+)_\d+_\d+\b")
_U64 = 0xFFFFFFFFFFFFFFFF


def _split_events(event):
    if event is None:
        return []
    groups = [event] if isinstance(event, str) else list(event)
    out = []
    for group in groups:
        for name in str(group or "").split(","):
            name = name.strip()
            if name and name not in out:
                out.append(name)
    return out


def _address(value):
    try:
        return int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        return None


def _used_regs(arch, insns):
    try:
        harness = set(arch._HARNESS_REGS or ())
    except Exception:
        harness = set()
    out = {}
    for insn in insns:
        try:
            addr = int(insn.address)
        except (TypeError, ValueError):
            continue
        regs = [reg for reg in arch.reg_refs(insn) if reg not in harness]
        if regs:
            out[addr] = tuple(regs)
    return out


def _data_tracker(arch, base, end, records, used=None):
    if records is None or not used:
        return None

    def _value(state, reg):
        try:
            expr = getattr(state.regs, reg)
            values = state.solver.eval_upto(expr, 2)
        except Exception:
            return None
        if len(values) != 1:
            return _symbol(state, expr)
        try:
            return int(values[0]) & _U64
        except (TypeError, ValueError):
            return None

    def _on_insn(state):
        try:
            insn = int(state.inspect.instruction)
        except Exception:
            return
        regs = used.get(insn)
        if not regs:
            return
        row = records.setdefault(insn, {})
        for reg in regs:
            column = f"data.{reg}"
            value = _value(state, reg)
            if value is None:
                continue
            if column not in row:
                row[column] = {value}
            else:
                row[column].add(value)

    def _track(state):
        import angr

        state.inspect.b("instruction", when=angr.BP_BEFORE, action=_on_insn)

    return _track


def _symbol(state, expr):
    try:
        if not state.solver.symbolic(expr):
            return None
    except Exception:
        return None
    return _symbol_text(str(expr))


def _symbol_text(text):
    match = _SYMBOL.fullmatch(text)
    if match:
        text = match.group(1)
    return _SYMBOL_VAR.sub(r"\1", text)


def _data_columns(records, addresses):
    out = {}
    columns = sorted({c for row in (records or {}).values() for c in row})
    for column in columns:
        out[column] = [
            _data_value((records or {}).get(address, {}).get(column))
            for address in addresses
        ]
    return out


def _data_value(values):
    if not values:
        return None
    values = sorted(values, key=repr)
    return values[0] if len(values) == 1 else values


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
    queue = [base]
    while queue:
        start = queue.pop(0)
        for insn in decoder.disasm(blob[start - base :], start):
            found[insn.address] = insn
            for target in arch.branch_imm_targets(insn):
                if base <= target < limit and target not in found:
                    queue.append(target)
    return [found[address] for address in sorted(found)]


def _block_addrs(arch, blob, base, start):
    offset = int(start) - base
    if offset < 0 or offset >= len(blob):
        return []
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


def _executed_insn_addrs(project, solutions, start, end):
    addrs = list(_executed_bbl_addrs(solutions))
    for solution in solutions or ():
        try:
            rip = _state_rip(solution["state"])
        except Exception:
            continue
        if rip is not None:
            addrs.append(rip)
    out = set()
    for addr in addrs:
        try:
            insns = list(project.factory.block(int(addr)).capstone.insns)
        except Exception:
            continue
        for insn in insns:
            try:
                at = int(insn.address)
            except (TypeError, ValueError):
                continue
            if start is not None and at < start:
                continue
            if end is not None and at >= end:
                continue
            out.add(at)
    return out


def _asm_instructions(asm, setup=None, records=None, data=None, teardown=None):
    arch = get_arch()
    code = arch.normalize_asm(f"{asm};")
    base = arch._SETUP_BASE
    try:
        blob = arch.assemble(code, base)
    except Exception as ex:
        raise ValueError(f"cannot assemble {asm!r}: {ex}") from ex
    if not blob:
        raise ValueError(f"cannot assemble {asm!r}: no instructions")
    insns = _instructions(arch, blob, base)
    keep = _explored_asm_addrs(
        arch, code, blob, base, setup, insns, records, data, teardown
    )
    if keep:
        insns = [i for i in insns if int(i.address) in keep]
    return code, insns


def _explored_asm_addrs(
    arch, code, blob, base, setup, insns, records=None, data=None, teardown=None
):
    track = _data_tracker(
        arch, base, base + len(blob), records, _used_regs(arch, insns)
    )
    try:
        solutions = explore_asm(
            code, setup, data, arch, track=track, teardown_asm=teardown
        )
    except Exception:
        return set()
    keep = set()
    for solution in solutions or ():
        addrs = list(_executed_bbl_addrs([solution]))
        rip = _state_rip(solution["state"])
        if rip is not None:
            addrs.append(rip)
        for addr in addrs:
            keep.update(_block_addrs(arch, blob, base, addr))
    return keep


def _binary_instructions(
    file, target, config, setup, records=None, data=None, teardown=None
):
    import angr

    spec = _merge_config(config)
    path = resolve_exec(file)
    if not os.path.exists(path):
        raise ValueError(f"file {path!r} does not exist")
    project = angr.Project(
        path, auto_load_libs=_resolve_lib(spec), load_debug_info=False
    )
    pattern = _normalize_target(target)
    if not pattern:
        raise ValueError("a target is required, e.g. `perf analyze a.out:fizz_buzz`")
    funcs, prototypes = functions(project)
    matched = list(resolve_targets(project, pattern, funcs))
    if not matched:
        raise ValueError(f"cannot resolve {pattern!r} in {file!r}")
    if len(matched) > 1:
        raise ValueError(f"ambiguous target {pattern!r} in {file!r}")
    label, start, end = matched[0]
    start, end = int(start), int(end)
    if end <= start:
        raise ValueError(f"empty region {label!r}: end <= start")
    try:
        blob = project.loader.memory.load(start, end - start)
    except Exception as ex:
        raise ValueError(f"cannot read {label!r} at 0x{start:x}: {ex}") from ex
    arch = load_arch(project)
    insns = _instructions(arch, blob, start)
    keep = _explored_binary_addrs(
        project,
        arch,
        spec,
        label,
        start,
        end,
        funcs,
        prototypes,
        setup,
        insns,
        records,
        data,
        teardown,
    )
    if keep:
        insns = [i for i in insns if int(i.address) in keep]
    return label, _file_label(path), insns


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
    insns,
    records=None,
    data=None,
    teardown=None,
):
    keep = set()
    used = _used_regs(arch, insns)
    for name, first, last in _explored_ranges(label, start, end, funcs):
        keep |= _explored_range(
            project,
            arch,
            spec,
            name,
            first,
            last,
            funcs,
            prototypes,
            setup,
            used,
            records,
            data,
            teardown,
        )
    return keep


def _explored_ranges(label, start, end, funcs):
    ranges = [(label, start, end)]
    for name, (first, last) in (funcs or {}).items():
        first, last = int(first), int(last)
        if start < first < end and (first, last) != (start, end):
            ranges.append((name, first, last))
    return ranges


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
    used,
    records,
    data=None,
    teardown=None,
):
    track = _data_tracker(arch, start, end, records, used)
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
            track=track,
            teardown_asm=teardown,
        )
    except Exception:
        try:
            solutions = explore(
                project,
                start,
                end,
                setup_asm=setup if isinstance(setup, str) else None,
                funcs=funcs,
                data=data,
                prototype=prototype,
            )
        except Exception:
            return set()
    return _executed_insn_addrs(project, solutions, start, end)


def _stack_size(spec):
    stack = (spec or {}).get("stack") or {}
    try:
        return int(stack["size"])
    except (KeyError, TypeError, ValueError):
        return None


def _stack_align(spec):
    stack = (spec or {}).get("stack") or {}
    try:
        return int(stack["align"])
    except (KeyError, TypeError, ValueError):
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
    for column, values in _data_columns(records, df["address"].tolist()).items():
        df[column] = values
    return df


def _per_ip_table(frame):
    if frame is None or getattr(frame, "empty", False):
        return None
    column = next((c for c in _ADDRESS_COLUMNS if c in frame.columns), None)
    if column is None:
        return None
    df = frame.copy()
    df[column] = [_address(v) for v in df[column].tolist()]
    df = df[df[column].notna()]
    if df.empty:
        return None
    if all(c in df.columns for c in _RECORD_COLUMNS):
        table = df.pivot_table(
            index=column, columns="event", values="period", aggfunc="sum"
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
    return df.groupby(column)[keep].sum() if keep else None


def _identities(frame, file_label, name):
    fallback = {"file": file_label, "name": name}
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
    return [f for f in frames if f is not None and not getattr(f, "empty", False)]


def _result_identity(results, file_label, name):
    for frame in _results_frames(results):
        identities = _identities(frame, file_label, name)
        if identities:
            return identities[0]
    return {"file": file_label, "name": name}


def _result_table(results):
    tables = [
        t for t in (_per_ip_table(f) for f in _results_frames(results)) if t is not None
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
    table = _result_table(results)
    if table is not None:
        addresses = pd.Index(df["address"].tolist())
        for column in table.columns:
            if column in df.columns:
                continue
            df[column] = table[column].reindex(addresses).to_numpy()
    return df


def _apply_filter(df, filter):
    if not filter:
        return df
    try:
        return query(df, filter)
    except Exception as ex:
        raise ValueError(f"invalid filter {filter!r}: {ex}") from ex


def _select(df, events):
    if not events:
        return df
    missing = [e for e in events if e not in df.columns]
    if missing:
        available = ", ".join(str(c) for c in df.columns)
        raise ValueError(
            f"unknown columns: {', '.join(missing)}; available: {available}"
        )
    keep = list(_LEAD_COLUMNS) + [e for e in events if e not in _LEAD_COLUMNS]
    keep += [c for c in df.columns if c not in keep and str(c).startswith(_DATA_PREFIX)]
    return df[keep]


def _state_key(value):
    if isinstance(value, (list, tuple)):
        return tuple(_state_key(v) for v in value)
    if value is None:
        return None
    try:
        if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _changing_state(df):
    keep = []
    for column in df.columns:
        if not str(column).startswith(_DATA_PREFIX):
            keep.append(column)
            continue
        seen = [
            key
            for key in (_state_key(v) for v in df[column].tolist())
            if key is not None
        ]
        if len(seen) > 1 and all(key == seen[0] for key in seen):
            continue
        keep.append(column)
    return df[keep] if len(keep) != len(df.columns) else df


def _order_columns(df, events=()):
    if events:
        return df
    lead = [c for c in _LEAD_COLUMNS if c in df.columns]
    insns = [c for c in _INSN_COLUMNS if c in df.columns and c not in lead]
    osaca = [c for c in _OSACA_COLUMNS if c in df.columns and c not in lead + insns]
    text = [
        c for c in _TEXT_COLUMNS if c in df.columns and c not in lead + insns + osaca
    ]
    keep = lead + insns + osaca + text
    rest = [c for c in df.columns if c not in keep]
    metrics = [c for c in rest if not str(c).startswith(_SKIP_COLUMNS)]
    return df[keep + metrics + [c for c in rest if c not in metrics]]


def analyze(
    code=None,
    name=None,
    *,
    event=None,
    results=None,
    config=None,
    data=None,
    setup=None,
    teardown=None,
    filter=None,
):
    file, target, asm = parse_code(code)
    if file is None and asm is None:
        raise TypeError(
            "pass 'code' as `FILE:TARGET` (e.g. 'a.out:fizz_buzz'), a "
            "[file, target] pair, or a raw asm snippet (e.g. 'mov eax, 42')"
        )
    events = _split_events(event)
    records = {}
    if asm is None and file is not None:
        asm = asm_source_code(file, target)
        if asm:
            name = name or asm_target_text(target)
            file = None
        elif is_asm_source(file):
            raise ValueError(f"cannot resolve {target!r} in {file!r}")
    if asm is not None:
        code, insns = _asm_instructions(asm, setup, records, data, teardown)
        file_label, label = None, name or code
    else:
        label, file_label, insns = _binary_instructions(
            file, target, config, setup, records, data, teardown
        )
        label = name or label
    if not insns:
        raise ValueError(f"no instructions found for {label!r}")

    df = _instruction_frame(insns, records)
    df = _changing_state(df)
    df = _merge_results(df, results, file_label, label)
    df = _apply_filter(df, filter)
    df = _order_columns(_select(df, events), events)
    df.attrs["file"] = file_label
    df.attrs["name"] = label
    df.attrs["code"] = code
    df.attrs["config"] = _merge_config(config)
    return df
