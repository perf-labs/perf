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
from collections import Counter, deque

import pandas as pd

from .arch import arch as get_arch
from .arch import load as load_arch
from .bench import (
    _content_hash8,
    _executed_bbl_addrs,
    _first_scalar,
    _func_prototype,
    _merge_config,
    _normalize_target,
    _render_insn,
    _resolve_lib,
    _resolve_target,
    _setup_asm_for,
    _unresolved,
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
DEFAULT_EVENTS = [
    *_IDENTITY,
    *_INSN_COLUMNS,
    *_OSACA_COLUMNS,
    *_TEXT_COLUMNS,
    "data*",
]


def analyze(
    target=None,
    asm=None,
    name=None,
    *,
    column=None,
    results=None,
    config=None,
    data=None,
    setup=None,
    teardown=None,
    filter=None,
    debug=False,
):
    file, target, snippet, name, raw = _resolve_target(target, asm, name)
    columns = _split_columns(column)
    strict = bool(columns)
    if not columns:
        columns = [*DEFAULT_EVENTS, "*"]
    records = {}
    debug_map = {}
    if snippet is not None:
        code, insns = _asm_instructions(snippet, setup, records, data, teardown)
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
    return df


def _split_columns(column):
    if column is None:
        return []
    values = [column] if isinstance(column, str) else list(column)
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
    for i, (addr_sym, insn, kind, value_sym) in enumerate(accesses):
        addr, value = resolved[2 * i], resolved[2 * i + 1]
        if addr is None or addr not in data_addrs:
            continue
        row = records.setdefault(insn, {})
        if kind == "mem-stores":
            if value is None:
                continue
            image[addr] = value
        else:
            value = image.get(addr, value)
        if value is None:
            continue
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


def _executed_by_solution(project, solutions, start, end):
    out = []
    for solution in solutions or ():
        addrs = list(_executed_bbl_addrs([solution]))
        try:
            rip = _state_rip(solution["state"])
        except Exception:
            rip = None
        if rip is not None:
            addrs.append(rip)
        out.append(_block_insn_addrs(project, addrs, start, end))
    return out


def _block_insn_addrs(project, addrs, start, end):
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
    _, data_records = _explored_asm_addrs(arch, code, blob, base, setup, data, teardown)
    if records is not None:
        records.update(data_records)
    return code, insns


def _explored_asm_addrs(arch, code, blob, base, setup, data=None, teardown=None):
    try:
        solutions = explore_asm(
            code, setup, data, arch, teardown_asm=teardown, errors=True
        )
    except Exception:
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
        raise ValueError("a target is required, e.g. `perf analyze a.out:fizz_buzz`")
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
        blob = project.loader.memory.load(start, end - start)
    except Exception as ex:
        raise ValueError(f"cannot read {label!r} at 0x{start:x}: {ex}") from ex
    arch = load_arch(project)
    insns = _instructions(arch, blob, start)
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
):
    keep = set()
    records = {}
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
    data=None,
    teardown=None,
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
            solutions = explore(
                project,
                start,
                end,
                setup_asm=setup if isinstance(setup, str) else None,
                funcs=funcs,
                data=data,
                prototype=prototype,
                errors=True,
            )
        except Exception:
            return set(), {}
    executed = _executed_by_solution(project, solutions, start, end)
    keep = set()
    for addrs in executed:
        keep |= addrs
    return keep, _data_records(arch, solutions, executed, data)


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
    threshold = max(3, int(top * 0.05))
    biases = [(bias, n) for bias, n in counts.items() if n >= threshold]
    biases.sort(key=lambda item: item[1], reverse=True)
    return biases[:32]


def _translate_frame_ips(frame, addresses):
    column = next((c for c in _ADDRESS_COLUMNS if c in frame.columns), None)
    if column is None or column != "ip" or "dso" not in frame.columns:
        return frame
    try:
        targets = [int(a) for a in (addresses or ())]
    except (TypeError, ValueError):
        return frame
    if not targets:
        return frame
    raw = [_to_int_or(v, None) for v in frame[column].tolist()]
    ips = [v for v in raw if v is not None]
    biases = _infer_biases(ips, targets)
    if not biases:
        return frame
    by_count = dict(biases)
    lowmap = _target_lowmap(targets)
    translated = []
    for value in raw:
        if value is None:
            translated.append(None)
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
            translated.append(int(value))
    out = frame.copy()
    out[column] = translated
    return out


def _per_ip_table(frame, addresses=None):
    if frame is None or getattr(frame, "empty", False):
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
    if any(c in frame.columns for c in _ADDRESS_COLUMNS):
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
    return [f for f in frames if f is not None and not getattr(f, "empty", False)]


def _result_identity(results, file_label, name):
    fallback = {"file": file_label, "name": name}
    for frame in _results_frames(results):
        if any(c in frame.columns for c in _ADDRESS_COLUMNS):
            continue
        if not any(c in frame.columns for c in _IDENTITY):
            continue
        identities = _identities(frame, file_label, name)
        if identities:
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
        for column in table.columns:
            if column in df.columns:
                continue
            df[column] = table[column].reindex(index).to_numpy()
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
    _alias_memory(df, wanted)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        _derive(df, missing)
        missing = [c for c in wanted if c not in df.columns]
    if missing and strict:
        available = ", ".join(str(c) for c in df.columns)
        raise ValueError(
            f"unknown columns: {', '.join(missing)}; available: {available}"
        )
    wanted = [c for c in wanted if c in df.columns]
    keep = [_LEAD_COLUMN] if _LEAD_COLUMN in wanted else []
    keep += [c for c in wanted if c not in keep]
    return df[keep]


def _alias_memory(df, columns):
    for column in columns:
        text = str(column)
        if column in df.columns or not text.startswith("data["):
            continue
        if not text.endswith("]"):
            continue
        name = f"{_DATA_PREFIX}{text[len('data[') : -1]}"
        if name in df.columns:
            df[column] = df[name]


def _expand(df, columns):
    out = []
    seen = set()
    for column in columns:
        text = str(column)
        if any(wildcard in text for wildcard in _WILDCARDS):
            for found in df.columns:
                if found not in seen and _match(text, str(found)):
                    seen.add(found)
                    out.append(found)
            continue
        if column not in seen:
            try:
                seen.add(column)
            except TypeError:
                pass
            out.append(column)
    return out


@functools.lru_cache(maxsize=1024)
def _compiled_match(pattern):
    return re.compile(
        "".join(
            ".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern
        )
    )


def _match(pattern, text):
    return _compiled_match(pattern).fullmatch(text) is not None


def _derive(df, names):
    for name in names:
        try:
            df[name] = df.eval(quote_columns(df, name), engine="python")
        except Exception:
            continue
    return df
