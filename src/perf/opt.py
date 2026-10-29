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


import copy
import json
import os
import re
import shutil
import struct
import sys

import pandas as pd
from elftools.elf.elffile import ELFFile

from .arch import arch as get_arch
from .exec import resolve_exec

PASSES = ("peephole", "branch", "layout", "align")
_DEFAULT_OPTIMIZE = {
    "opt": list(PASSES),
    "layout": {"align": 16, "order": "hot-first"},
}
_VALID_OPTIMIZE_KEYS = frozenset(_DEFAULT_OPTIMIZE)
_VALID_LAYOUT_KEYS = frozenset({"align", "order"})
_ORDERS = ("hot-first", "as-is")


def discover_functions(path):
    text = _text_section(path)
    lo, hi = text["addr"], text["addr"] + text["size"]
    symtab_found = _symtab_functions(path)
    found = dict(symtab_found)
    if not symtab_found or not _has_symtab(path):
        discovered = _angr_functions(path)
        if discovered:
            found = dict(discovered)
            for name, bounds in symtab_found.items():
                found.setdefault(name, bounds)

    cands = []
    for name, (start, end) in found.items():
        try:
            s, e = int(start), int(end)
        except Exception:
            continue
        if s < lo or s >= hi:
            continue
        if e < s or e > hi:
            e = s
        cands.append((str(name), s, e))

    cands.sort(key=lambda item: (item[1], item[2], item[0]))
    out = {}
    for idx, (name, s, e) in enumerate(cands):
        if e <= s:
            nxt = cands[idx + 1][1] if idx + 1 < len(cands) else hi
            e = min(max(nxt, s + 1), hi)
        if e <= s or s < lo or e > hi:
            continue
        out.setdefault(name, (s, e))

    seen = {}
    deduped = {}
    for name, bounds in sorted(out.items(), key=lambda kv: (kv[1], kv[0])):
        if bounds in seen:
            continue
        seen[bounds] = name
        deduped[name] = bounds
    disjoint = {}
    covered_end = lo
    for name, (start, end) in sorted(
        deduped.items(), key=lambda kv: (kv[1][0], -(kv[1][1] - kv[1][0]), kv[0])
    ):
        if start < covered_end:
            continue
        disjoint[name] = (start, end)
        covered_end = end
    return disjoint


def collect_hotness(paths):
    if paths is None:
        return {}, {}
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    raw_sources = []
    for entry in list(paths):
        if isinstance(entry, pd.DataFrame):
            scores = _hotness_from_dataframe(entry)
            if scores:
                raw_sources.append((scores, "<dataframe>"))
            continue
        path = str(entry)
        if os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for name in sorted(files):
                    full = os.path.join(root, name)
                    if name.endswith(".json") or name == "profile.json":
                        try:
                            payload = _load_json_envelope(full)
                        except Exception:
                            continue
                        scores = _hotness_from_envelope(payload)
                        if scores:
                            raw_sources.append((scores, full))
                    else:
                        try:
                            from .data import is_record as _is_rec

                            if _is_rec(full):
                                scores = _hotness_from_perf_data(full)
                                if scores:
                                    raw_sources.append((scores, full))
                        except Exception:
                            continue
            continue
        if not os.path.isfile(path):
            print(
                f"warning: data path {path!r} does not exist; skipping", file=sys.stderr
            )
            continue
        if path.endswith(".json"):
            try:
                payload = _load_json_envelope(path)
            except Exception as ex:
                print(
                    f"warning: cannot parse {path!r}: {ex}; skipping", file=sys.stderr
                )
                continue
            scores = _hotness_from_envelope(payload)
            if scores:
                raw_sources.append((scores, path))
            continue
        try:
            from .data import is_record as _is_rec2

            if _is_rec2(path):
                scores = _hotness_from_perf_data(path)
                if scores:
                    raw_sources.append((scores, path))
                continue
        except Exception:
            pass

        try:
            payload = _load_json_envelope(path)
            scores = _hotness_from_envelope(payload)
            if scores:
                raw_sources.append((scores, path))
        except Exception:
            print(f"warning: unsupported data file {path!r}; skipping", file=sys.stderr)
            continue

    combined = {}
    details = {}
    for scores, origin in raw_sources:
        try:
            peak = max(float(v) for v in scores.values() if v == v)
        except ValueError:
            continue
        if not peak or peak <= 0:
            continue
        for name, value in scores.items():
            try:
                norm = float(value) / peak
            except (TypeError, ValueError):
                continue
            if norm != norm:
                continue
            combined[name] = combined.get(name, 0.0) + norm
            details.setdefault(name, []).append((origin, float(value)))
    return combined, details


def _parse_opt(value):
    if value is None:
        return list(PASSES)
    if isinstance(value, str):
        selected = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        selected = []
        for item in value:
            selected.extend([p.strip() for p in str(item).split(",") if p.strip()])
    else:
        raise ValueError(
            f"opt must be a comma-separated string or a list, got {value!r}"
        )
    unknown = [p for p in selected if p not in PASSES]
    if unknown:
        raise ValueError(
            f"unknown opt pass(es) {', '.join(unknown)}; "
            f"expected one of {', '.join(PASSES)}"
        )
    return selected


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def merge_config(config=None):
    base = copy.deepcopy(_DEFAULT_OPTIMIZE)
    if config is None:
        merged = base
    elif isinstance(config, dict):
        merged = _deep_merge(base, config)
    else:
        raise ValueError(f"config must be a dict, got {config!r}")

    layout = merged.get("layout")
    if not isinstance(layout, dict):
        raise ValueError(f"layout config must be a dict, got {layout!r}")
    for key in ("align", "order"):
        if key in merged:
            layout[key] = merged.pop(key)
    for key in merged:
        if key not in _VALID_OPTIMIZE_KEYS:
            raise ValueError(
                f"unknown config key {key!r}; expected one of "
                f"{', '.join(sorted(_VALID_OPTIMIZE_KEYS))}"
            )
    for key in layout:
        if key not in _VALID_LAYOUT_KEYS:
            raise ValueError(
                f"unknown layout config key {key!r}; expected one of "
                f"{', '.join(sorted(_VALID_LAYOUT_KEYS))}"
            )

    opt = _parse_opt(merged.get("opt"))
    align = layout.get("align", 16)
    try:
        align = int(align)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"align must be an integer, got {align!r}") from ex
    if align < 1 or align > 4096 or align & (align - 1):
        raise ValueError(
            f"align must be a power of two between 1 and 4096, got {align!r}"
        )
    order = layout.get("order", "hot-first")
    if order not in _ORDERS:
        raise ValueError(f"order must be 'hot-first' or 'as-is', got {order!r}")
    return {"opt": opt, "layout": {"align": align, "order": order}}


def plan(config=None, opt=None):
    merged = merge_config(config)
    if opt is not None:
        merged = {**merged, "opt": _parse_opt(opt)}
    selected = list(merged["opt"])
    if "align" in selected and "layout" not in selected:
        selected = [*selected, "layout"]
    ordered = [p for p in PASSES if p in selected]
    return ordered, merged["layout"]["align"], merged["layout"]["order"]


def _same_file(left, right):
    try:
        if os.path.realpath(os.path.abspath(left)) == os.path.realpath(
            os.path.abspath(right)
        ):
            return True
    except Exception:
        pass
    try:
        return os.path.samefile(left, right)
    except (OSError, ValueError):
        return False


def optimize(
    exec=None,
    output=None,
    data=None,
    opt=None,
    config=None,
    dry_run=False,
    select=None,
):
    selected, align, order = plan(config, opt)

    if exec is None:
        raise ValueError("an input binary is required, e.g. 'a.out'")
    if isinstance(exec, (list, tuple)):
        raise ValueError(
            "exec must be a binary path; pass a function filter as select="
        )
    file = str(exec).strip()
    if not file:
        raise ValueError("an input binary is required, e.g. 'a.out'")
    if not os.path.isfile(file):
        raise ValueError(f"file {file!r} does not exist")
    filt = str(select).strip() if select is not None else None
    if not filt:
        filt = None
    real = resolve_exec(file)
    if output is None:
        raise ValueError("an output path is required (--output b.out)")
    out_path = str(output)
    if not dry_run and (_same_file(real, out_path) or _same_file(file, out_path)):
        raise ValueError("input and output must be different files")
    pie = _is_pie_elf(real)
    text_sec = _text_section(real)
    text_lo = text_sec["addr"]
    with open(real, "rb") as fh:
        original_image = fh.read()
    text_bytes = original_image[
        text_sec["offset"] : text_sec["offset"] + text_sec["size"]
    ]
    all_funcs = discover_functions(real)
    if not all_funcs:
        raise ValueError(f"no functions found in {file!r}")
    funcs = all_funcs
    if filt:
        try:
            pattern = re.compile(filt)
        except re.error:
            pattern = None
        if pattern is not None:
            funcs = {n: b for n, b in all_funcs.items() if pattern.search(n)}
        else:
            funcs = {n: b for n, b in all_funcs.items() if n == filt}
        if not funcs:
            raise ValueError(f"no functions match {filt!r} in {file!r}")

    data_paths = []
    if data is not None:
        data_paths = list(data) if isinstance(data, (list, tuple)) else [data]
    scores, _details = collect_hotness(data_paths)

    blobs = {}
    for name, (start, end) in all_funcs.items():
        lo = start - text_lo
        blobs[name] = bytes(text_bytes[lo : lo + (end - start)])
    peephole_stats = {
        "mov_self": 0,
        "movabs": 0,
        "nop_norm": 0,
        "lea_self": 0,
        "xchg_self": 0,
        "shift_zero": 0,
        "push_pop": 0,
    }
    branch_stats = {
        "jmp_next": 0,
        "jcc_next": 0,
        "jmp_fold": 0,
        "jcc_fold": 0,
        "jcc_invert": 0,
    }
    if "peephole" in selected:
        for name, (start, _end) in funcs.items():
            new_blob, stats = _peephole_function(blobs[name], start)
            blobs[name] = new_blob
            for key, value in stats.items():
                peephole_stats[key] = peephole_stats.get(key, 0) + value
    if "branch" in selected:
        protected = _branch_targets(all_funcs, blobs)
        for name, (start, _end) in funcs.items():
            new_blob, stats = _branch_function(blobs[name], start, protected=protected)
            blobs[name] = new_blob
            for key, value in stats.items():
                branch_stats[key] = branch_stats.get(key, 0) + value

    layout_applied = False
    layout_skipped = ""
    new_layout = None
    if "layout" in selected or "align" in selected:
        if not pie:
            layout_skipped = "non-PIE binaries keep their layout (safe fallback)"
        elif _has_exception_tables(real):
            layout_skipped = "exception tables keep the original layout (safe fallback)"
        elif _has_fallthrough(all_funcs, blobs):
            layout_skipped = (
                "functions fall through to the next function; "
                "layout keeps the original order (safe fallback)"
            )
        elif (
            _text_coverage(
                all_funcs,
                text_lo,
                text_lo + text_sec["size"],
                text=text_bytes,
            )
            < 1.0
        ):
            layout_skipped = (
                "function discovery does not cover all of .text; "
                "layout keeps the original order (safe fallback)"
            )
        else:
            movable = {}
            pinned = {}
            for name, (start, end) in all_funcs.items():
                if _has_jump_table(blobs[name], start):
                    pinned[name] = (start, end)
                else:
                    movable[name] = (start, end)
            if order == "hot-first":
                ordered = sorted(
                    movable,
                    key=lambda n: (
                        -float(_match_hotness(n, scores)),
                        movable[n][0],
                    ),
                )
            else:
                ordered = sorted(movable, key=lambda n: movable[n][0])
            if "align" in selected:
                hot_align = int(align)
            else:
                hot_align = 16

            branch_targets = _branch_targets(all_funcs, blobs)
            stripped = {}
            for name in ordered:
                blob, _pad = _strip_trailing_nops(blobs[name])
                if blob:
                    start, _end = all_funcs[name]
                    keep_from = start + len(blob)
                    stop = start + len(blobs[name])
                    if any(keep_from <= int(t) < stop for t in branch_targets):
                        blob = blobs[name]
                stripped[name] = blob if blob else blobs[name]

            cursor = text_lo

            if pinned:
                layout_skipped = (
                    f"{len(pinned)} function(s) use indirect jumps; "
                    "layout keeps the original order (safe fallback)"
                )
            else:
                trial = {}
                for name in ordered:
                    hot = float(_match_hotness(name, scores)) > 0
                    if "align" in selected:
                        need = hot_align if hot else 16
                    else:
                        need = 16 if hot else 1
                    cursor = (cursor + need - 1) & ~(need - 1)
                    trial[name] = cursor
                    cursor += len(stripped[name])
                if cursor - text_lo > text_sec["size"]:
                    layout_skipped = (
                        "aligned layout does not fit .text; "
                        "keeping the original order (safe fallback)"
                    )
                else:
                    new_layout = trial
                    for name, blob in stripped.items():
                        blobs[name] = blob
                    layout_applied = True

    new_text = bytearray(text_bytes)
    old_to_new = {}
    if layout_applied and new_layout is not None:
        new_text = bytearray(len(text_bytes))
        for i in range(len(new_text)):
            new_text[i] = 0xCC
        for name in sorted(new_layout, key=lambda n: new_layout[n]):
            start = new_layout[name]
            off = start - text_lo
            blob = blobs[name]
            new_text[off : off + len(blob)] = blob
            old_lo, old_hi = all_funcs[name]
            old_to_new[(old_lo, old_hi)] = (start, start + len(blob))
    else:
        for name, (start, end) in all_funcs.items():
            foff = start - text_lo
            new_text[foff : foff + (end - start)] = blobs[name]
            old_to_new[(start, end)] = (start, start + len(blobs[name]))

    fixup_stats = {
        "rel32": 0,
        "rip": 0,
        "rela": 0,
        "eh_frame": 0,
        "eh_frame_hdr": 0,
    }
    if layout_applied:
        new_image = _fixup_layout_image(
            real,
            original_image,
            text_sec,
            text_lo,
            all_funcs,
            blobs,
            old_to_new,
            fixup_stats,
        )
        final_image = bytes(new_image)
    else:
        final_image = (
            original_image[: text_sec["offset"]]
            + bytes(new_text)
            + original_image[text_sec["offset"] + text_sec["size"] :]
        )

    _verify_text(
        bytes(new_text), text_lo, all_funcs, old_to_new if layout_applied else None
    )

    out_rows = []
    for name, (start, end) in sorted(funcs.items(), key=lambda kv: kv[1]):
        new_start = old_to_new.get((start, end), (start, end))[0]
        out_rows.append(
            {
                "name": name,
                "address": start,
                "new_address": new_start,
                "size": end - start,
                "new_size": len(blobs[name]),
                "hotness": float(_match_hotness(name, scores)),
                "moved": bool(layout_applied) and new_start != start,
            }
        )
    result = pd.DataFrame(
        out_rows,
        columns=[
            "name",
            "address",
            "new_address",
            "size",
            "new_size",
            "hotness",
            "moved",
        ],
    )
    result.attrs["opt"] = list(selected)
    result.attrs["peephole"] = dict(peephole_stats)
    result.attrs["branch"] = dict(branch_stats)
    result.attrs["fixups"] = dict(fixup_stats)
    result.attrs["layout"] = bool(layout_applied)
    result.attrs["layout_skipped"] = str(layout_skipped)
    result.attrs["file"] = str(file)
    result.attrs["output"] = out_path
    if dry_run:
        return result

    out_parent = os.path.dirname(os.path.abspath(out_path))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(final_image)
    try:
        shutil.copymode(real, out_path)
    except OSError:
        pass
    try:
        mode = os.stat(out_path).st_mode
        os.chmod(out_path, mode | 0o111)
    except OSError:
        pass
    return result


def report_json(df):
    try:
        records = df.to_dict(orient="records")
    except Exception:
        records = []
    attrs = dict(getattr(df, "attrs", {}) or {})
    return {
        "file": attrs.get("file", ""),
        "output": attrs.get("output", ""),
        "opt": attrs.get("opt", []),
        "peephole": attrs.get("peephole", {}),
        "branch": attrs.get("branch", {}),
        "fixups": attrs.get("fixups", {}),
        "layout": bool(attrs.get("layout", False)),
        "layout_skipped": attrs.get("layout_skipped", ""),
        "functions": records,
    }


def describe(plan_result):
    lines = []
    try:
        selected = plan_result.attrs.get("opt", [])
        lines.append(f"opt: {', '.join(selected)}")
    except Exception:
        pass
    try:
        records = plan_result.to_dict(orient="records")
    except Exception:
        records = []
    try:
        for row in records:
            moved = "moved" if row.get("moved") else "kept"
            lines.append(
                f"{row.get('name')} 0x{int(row.get('address')):x}->"
                f"0x{int(row.get('new_address')):x} "
                f"size {int(row.get('size'))} {moved} "
                f"hotness {float(row.get('hotness')):.3f}"
            )
    except Exception:
        pass
    return "\n".join(lines)


def _is_pie_elf(path):
    try:
        with open(path, "rb") as fh:
            return ELFFile(fh).header["e_type"] == "ET_DYN"
    except Exception:
        return False


def _has_exception_tables(path):
    try:
        with open(path, "rb") as fh:
            elffile = ELFFile(fh)
            sec = elffile.get_section_by_name(".gcc_except_table")
            return sec is not None and int(sec.header["sh_size"]) > 0
    except Exception:
        return False


def _text_section(path):
    with open(path, "rb") as fh:
        elffile = ELFFile(fh)
        sec = elffile.get_section_by_name(".text")
        if sec is None:
            raise ValueError(f"no .text section in {path!r}")
        return {
            "addr": int(sec.header["sh_addr"]),
            "offset": int(sec.header["sh_offset"]),
            "size": int(sec.header["sh_size"]),
        }


def _symtab_functions(path):

    found = {}
    with open(path, "rb") as fh:
        elffile = ELFFile(fh)
        for sec_name in (".symtab", ".dynsym"):
            sec = elffile.get_section_by_name(sec_name)
            if sec is None:
                continue
            for sym in sec.iter_symbols():
                try:
                    typ = sym.entry["st_info"]["type"]
                    shndx = sym.entry["st_shndx"]
                    size = int(sym.entry["st_size"] or 0)
                    value = int(sym.entry["st_value"])
                    name = sym.name
                except Exception:
                    continue
                if str(typ) not in ("STT_FUNC", "STT_GNU_IFUNC", "STT_LOOS"):
                    continue
                if shndx in ("SHN_UNDEF", 0, "SHN_ABS", "SHN_COMMON"):
                    continue
                if not name:
                    continue
                if name in found:
                    continue
                found[name] = (value, value + max(int(size), 0))
    return found


def _has_symtab(path):
    try:
        with open(path, "rb") as fh:
            return ELFFile(fh).get_section_by_name(".symtab") is not None
    except Exception:
        return False


def _angr_functions(path):
    try:
        import angr

        from .info import functions as _info_functions
    except Exception:
        return {}
    try:
        project = angr.Project(
            resolve_exec(path), auto_load_libs=False, load_debug_info=False
        )
        funcs, _ = _info_functions(project)
    except Exception:
        return {}
    try:
        pic = bool(project.loader.main_object.pic)
        base = int(project.loader.main_object.mapped_base or 0) if pic else 0
    except Exception:
        base = 0
    out = {}
    for name, (start, end) in (funcs or {}).items():
        try:
            out[str(name)] = (int(start) - base, int(end) - base)
        except Exception:
            continue
    return out


def _has_jump_table(func_bytes, func_addr):
    return get_arch().has_jump_table(func_bytes, func_addr)


def _peephole_function(blob, base):
    return get_arch().peephole_function(blob, base)


def _branch_targets(funcs, blobs):
    try:
        arch = get_arch()
        md = arch.detailed_disassembler()
    except Exception:
        return set()
    targets = set()
    for name, (start, _end) in funcs.items():
        try:
            blob = bytes(blobs.get(name, b""))
            if not blob:
                continue
            for insn in md.disasm(blob, int(start)):
                target = arch.immediate_target(insn)
                if target is not None:
                    targets.add(int(target))
        except Exception:
            continue
    return targets


def _branch_function(blob, base, protected=None):
    return get_arch().branch_function(blob, base, protected=protected)


def _strip_trailing_nops(blob):
    return get_arch().strip_trailing_nops(blob)


def _load_json_envelope(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or "output" not in payload:
        raise ValueError(f"not a perf envelope: {path!r}")
    return payload


def _strip_run_id(name):
    head, _, tail = str(name).rpartition("-")
    if head and len(tail) == 8:
        try:
            int(tail, 16)
            return head
        except ValueError:
            pass
    return str(name)


def _hotness_from_envelope(payload):
    output = payload.get("output") or []
    top_name = str(payload.get("name") or "")
    scores = {}
    if not output:
        return scores

    def _row_score(row):
        ops = 1.0
        try:
            ops = float(row.get("operations", 1) or 1)
        except (TypeError, ValueError):
            ops = 1.0
        for candidate in ("cycles", "duration_time", "instructions"):
            try:
                value = float(row.get(candidate))
            except (TypeError, ValueError):
                continue
            if value == value:
                return value / ops if ops else value
        return None

    def _add(scores, name, value):
        key = _strip_run_id(name.split("(")[0].strip() or name) or name
        scores[key] = scores.get(key, 0.0) + value
        if ".." in key:
            for part in key.split(".."):
                part = _strip_run_id(part.strip())
                if part:
                    scores[part] = scores.get(part, 0.0) + value
        return scores

    if any(isinstance(r, dict) and "iteration" in r for r in output):
        grouped = {}
        for row in output:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            key = name.split("(")[0].strip() or name
            metric = _row_score(row)
            grouped[key] = grouped.get(key, 0.0) + (
                metric if metric is not None else 1.0
            )
        expanded = {}
        for name, value in grouped.items():
            _add(expanded, name, value)
        return expanded

    if any(isinstance(r, dict) and "address" in r for r in output):
        total = 0.0
        for row in output:
            if not isinstance(row, dict):
                continue
            for key in ("latency", "throughput"):
                try:
                    value = float(row.get(key))
                except (TypeError, ValueError):
                    continue
                if value == value:
                    total += value
                    break
        if total > 0:
            scores[_strip_run_id(top_name) or top_name] = total
        return scores

    named = [str(r.get("name") or "").strip() for r in output if isinstance(r, dict)]
    if any(named):
        grouped = {}
        for row in output:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            key = name.split("(")[0].strip() or _strip_run_id(top_name) or "unknown"
            metric = _row_score(row)
            grouped[key] = grouped.get(key, 0.0) + (
                metric if metric is not None else 1.0
            )
        for name, value in grouped.items():
            _add(scores, name, value)
        return scores

    total = 0.0
    for row in output:
        if not isinstance(row, dict):
            continue
        metric = _row_score(row)
        if metric is not None:
            total += metric
    score = total
    if score <= 0:
        score = float(len(output))
    key = _strip_run_id(top_name) or top_name or "unknown"
    return {key: score}


def _hotness_from_perf_data(path):
    from .data import is_record
    from .data import samples as _samples

    try:
        if not is_record(path):
            return {}
    except Exception:
        return {}
    try:
        df = _samples(str(path))
    except Exception as ex:
        print(
            f"warning: cannot read perf.data {path!r}: {ex}",
            file=sys.stderr,
            flush=True,
        )
        return {}
    if df is None or getattr(df, "empty", False):
        return {}
    scores = {}
    col = "sym" if "sym" in df.columns else None
    if col is None:
        return {}
    try:
        grouped = df.groupby(col)["period"].sum()
    except Exception:
        try:
            grouped = df[col].value_counts()
        except Exception:
            return {}
    for name, value in grouped.items():
        try:
            key = _strip_run_id(str(name).split("(")[0].strip() or str(name))
            scores[key] = scores.get(key, 0.0) + float(value)
        except (TypeError, ValueError):
            continue
    return scores


def _hotness_from_dataframe(df):
    try:
        records = df.to_dict(orient="records")
    except Exception:
        return {}
    payload = {
        "name": str(getattr(df, "attrs", {}).get("name", "") or ""),
        "file": str(getattr(df, "attrs", {}).get("file", "") or ""),
        "output": records,
    }
    try:
        return _hotness_from_envelope(payload)
    except Exception:
        return {}


def _match_hotness(func_name, scores):
    if not scores:
        return 0.0
    try:
        from .core import demangle as _demangle
    except Exception:
        _demangle = None
    candidates = [str(func_name)]
    if _demangle is not None:
        try:
            demangled = str(_demangle(str(func_name)))
            candidates.append(demangled)
            candidates.append(demangled.split("(")[0].strip())
        except Exception:
            pass
    short = str(func_name).split("(")[0].strip()
    candidates.append(short)
    for cand in list(candidates):
        base = str(cand).split("(")[0].strip()
        if base:
            candidates.append(base)
            if "::" in base:
                candidates.append(base.split("::")[-1])
    for cand in candidates:
        if cand in scores:
            return float(scores[cand])
        base = _strip_run_id(cand)
        if base in scores:
            return float(scores[base])
    return 0.0


def _has_fallthrough(all_funcs, blobs):
    try:
        arch = get_arch()
    except Exception:
        return True
    try:
        md = arch.detailed_disassembler()
    except Exception:
        return True
    ordered = sorted(all_funcs.items(), key=lambda kv: kv[1][0])
    for idx, (name, (_start, _end)) in enumerate(ordered):
        blob = blobs.get(name)
        if not blob:
            continue
        try:
            stripped, _ = arch.strip_trailing_nops(bytes(blob))
            probe = bytes(stripped) if stripped else bytes(blob)
        except Exception:
            probe = bytes(blob)
        try:
            insns = list(md.disasm(probe, int(all_funcs[name][0])))
        except Exception:
            return True
        if not insns:
            return True
        last = insns[-1]
        try:
            mnemonic = str(last.mnemonic or "").strip().lower()
        except Exception:
            return True
        if mnemonic in ("ret", "retf", "retq", "jmp", "hlt"):
            continue
        try:
            raw = bytes(last.bytes)
        except Exception:
            raw = b""
        if raw == b"\xcc" * len(raw) and raw:
            continue
        if idx + 1 >= len(ordered):
            continue
        return True
    return False


def _text_coverage(all_funcs, lo, hi, text=None):
    if hi <= lo:
        return 1.0
    spans = []
    for start, end in all_funcs.values():
        try:
            s, e = max(int(start), lo), min(int(end), hi)
        except (TypeError, ValueError):
            continue
        if e > s:
            spans.append((s, e))
    spans.sort()
    missing = []
    cursor = lo
    for start, end in spans:
        if end <= cursor:
            continue
        if start > cursor:
            missing.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < hi:
        missing.append((cursor, hi))
    uncovered = 0
    for start, end in missing:
        size = end - start
        if text is not None:
            try:
                blob = bytes(text[start - lo : end - lo])
                stripped, _ = _strip_trailing_nops(blob)
                if not stripped:
                    continue
                size = len(stripped)
            except Exception:
                pass
        uncovered += size
    return 1.0 - (uncovered / float(hi - lo))


def _parse_eh_frame(data):
    entries = []
    off = 0
    size = len(data)
    while off + 8 <= size:
        try:
            (length,) = struct.unpack_from("<I", data, off)
        except struct.error:
            break
        if length in (0, 0xFFFFFFFF):
            break
        end = off + 4 + length
        if end > size:
            break
        try:
            (cid,) = struct.unpack_from("<I", data, off + 4)
        except struct.error:
            break
        if cid == 0:
            entries.append({"kind": "cie", "offset": off, "end": end})
        else:
            cie_off = off + 4 - int(cid)
            entries.append(
                {"kind": "fde", "offset": off, "end": end, "cie_off": cie_off}
            )
        off = end
    return entries


def _read_uleb(data, pos):
    value = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            break
    return value, pos


def _read_sleb(data, pos):
    value = 0
    shift = 0
    byte = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            if byte & 0x40:
                value -= 1 << shift
            break
    return value, pos


def _encoded_size(enc):
    fmt = int(enc) & 0x0F
    if fmt == 0x00:
        return 8
    if fmt == 0x02:
        return 2
    if fmt == 0x03:
        return 4
    if fmt == 0x04:
        return 8
    if fmt == 0x0A:
        return 2
    if fmt == 0x0B:
        return 4
    if fmt == 0x0C:
        return 8
    return None


def _read_encoded(data, pos, enc, section_addr=0, base=0, size=None):
    if enc is None or int(enc) == 0xFF or int(enc) & 0x80:
        return None
    fmt = int(enc) & 0x0F
    start = pos
    try:
        if fmt == 0x00:
            (raw,) = struct.unpack_from("<Q", data, pos)
            pos += 8
            value = int(raw)
        elif fmt == 0x02:
            (raw,) = struct.unpack_from("<H", data, pos)
            pos += 2
            value = int(raw)
        elif fmt == 0x03:
            (raw,) = struct.unpack_from("<I", data, pos)
            pos += 4
            value = int(raw)
        elif fmt == 0x04:
            (raw,) = struct.unpack_from("<Q", data, pos)
            pos += 8
            value = int(raw)
        elif fmt == 0x09:
            value, pos = _read_sleb(data, pos)
        elif fmt == 0x0A:
            (raw,) = struct.unpack_from("<h", data, pos)
            pos += 2
            value = int(raw)
        elif fmt == 0x0B:
            (raw,) = struct.unpack_from("<i", data, pos)
            pos += 4
            value = int(raw)
        elif fmt == 0x0C:
            (raw,) = struct.unpack_from("<q", data, pos)
            pos += 8
            value = int(raw)
        else:
            return None
    except struct.error:
        return None
    app = int(enc) & 0x70
    if app == 0x10:
        value += int(section_addr) + start
    elif app == 0x30:
        value += int(base)
    value &= 0xFFFFFFFFFFFFFFFF
    if size is not None and pos - start != int(size):
        return None
    return value, pos


def _write_encoded(out, pos, enc, value, section_addr=0, base=0):
    if enc is None or int(enc) == 0xFF or int(enc) & 0x80:
        return False
    fmt = int(enc) & 0x0F
    app = int(enc) & 0x70
    value = int(value)
    if app == 0x10:
        value -= int(section_addr) + int(pos)
    elif app == 0x30:
        value -= int(base)
    try:
        if fmt == 0x00:
            struct.pack_into("<Q", out, pos, value & 0xFFFFFFFFFFFFFFFF)
            return True
        if fmt == 0x02:
            if not 0 <= value <= 0xFFFF:
                return False
            struct.pack_into("<H", out, pos, value)
            return True
        if fmt == 0x03:
            if not 0 <= value <= 0xFFFFFFFF:
                return False
            struct.pack_into("<I", out, pos, value)
            return True
        if fmt == 0x04:
            struct.pack_into("<Q", out, pos, value & 0xFFFFFFFFFFFFFFFF)
            return True
        if fmt == 0x0A:
            if not -(2**15) <= value < 2**15:
                return False
            struct.pack_into("<h", out, pos, value)
            return True
        if fmt == 0x0B:
            if not -(2**31) <= value < 2**31:
                return False
            struct.pack_into("<i", out, pos, value)
            return True
        if fmt == 0x0C:
            if not -(2**63) <= value < 2**63:
                return False
            struct.pack_into("<q", out, pos, value)
            return True
    except struct.error:
        return False
    return False


def _eh_frame_pointer_encoding(data, cie_off):
    try:
        pos = int(cie_off) + 4
        (_length,) = struct.unpack_from("<I", data, int(cie_off))
        (cid,) = struct.unpack_from("<I", data, pos)
        if cid != 0:
            return None
        pos += 4
        version = data[pos]
        pos += 1
        aug_end = data.index(b"\x00", pos)
        aug = data[pos:aug_end].decode("ascii", "replace")
        pos = aug_end + 1
        _code, pos = _read_uleb(data, pos)
        _data, pos = _read_sleb(data, pos)
        if version == 1:
            pos += 1
        else:
            _ret, pos = _read_uleb(data, pos)
        if not aug.startswith("z"):
            return None
        aug_size, pos = _read_uleb(data, pos)
        aug_end = pos + aug_size
        enc = None
        idx = 1
        while idx < len(aug):
            ch = aug[idx]
            idx += 1
            if ch == "L":
                pos += 1
            elif ch == "R":
                if pos >= aug_end:
                    return None
                enc = data[pos]
                pos += 1
            elif ch == "P":
                if pos >= aug_end:
                    return None
                penc = data[pos]
                pos += 1
                size = _encoded_size(penc)
                if size is None:
                    return None
                pos += size
            elif ch == "S":
                continue
            else:
                break
        return enc
    except Exception:
        return None


def _remap_address(addr, old_to_new):
    for (old_lo, old_hi), (new_lo, new_hi) in old_to_new.items():
        if old_lo <= int(addr) < old_hi:
            return new_lo + (int(addr) - old_lo), new_hi
    return None, None


def _patch_eh_frame(data, old_to_new, section_addr=0):
    out = bytearray(data)
    rows = []
    updated = 0
    for entry in _parse_eh_frame(bytes(out)):
        if entry["kind"] != "fde":
            continue
        off = entry["offset"]
        body = off + 8
        enc = _eh_frame_pointer_encoding(bytes(out), entry.get("cie_off", -1))
        if enc is None:
            raise ValueError("cannot parse .eh_frame pointer encoding")
        size = _encoded_size(enc)
        if size is None:
            raise ValueError("cannot update variable-length .eh_frame pointers")
        decoded = _read_encoded(
            bytes(out), body, enc, section_addr=section_addr, size=size
        )
        if decoded is None:
            raise ValueError("cannot decode .eh_frame FDE")
        pc, _ = decoded
        mapped, limit = _remap_address(pc, old_to_new)
        fmt = int(enc) & 0x0F
        range_decoded = _read_encoded(bytes(out), body + size, fmt)
        if range_decoded is None:
            raise ValueError("cannot decode .eh_frame FDE range")
        span, _ = range_decoded
        if mapped is None:
            continue
        if limit is not None and span > limit - mapped:
            span = max(1, limit - mapped)
        rows.append((body, enc, size, mapped, span))
    for body, enc, size, mapped, span in rows:
        if not _write_encoded(out, body, enc, mapped, section_addr=section_addr):
            raise ValueError("cannot encode .eh_frame FDE address")
        fmt = int(enc) & 0x0F
        if not _write_encoded(out, body + size, fmt, span):
            raise ValueError("cannot encode .eh_frame FDE range")
        updated += 1
    return bytes(out), updated


def _patch_eh_frame_hdr(data, old_to_new, section_addr=0):
    out = bytearray(data)
    if len(out) < 4 or int(out[0]) != 1:
        return bytes(out), 0
    ptr_enc, count_enc, table_enc = int(out[1]), int(out[2]), int(out[3])
    pos = 4
    decoded = _read_encoded(
        bytes(out), pos, ptr_enc, section_addr=section_addr, base=section_addr
    )
    if decoded is None:
        raise ValueError("cannot decode .eh_frame_hdr pointer")
    _, pos = decoded
    count_decoded = _read_encoded(bytes(out), pos, count_enc)
    if count_decoded is None:
        raise ValueError("cannot decode .eh_frame_hdr count")
    count, pos = count_decoded
    if table_enc == 0xFF or count == 0:
        return bytes(out), 0
    if (table_enc & 0x0F) in (0x01, 0x09):
        raise ValueError("cannot update variable-length .eh_frame_hdr entries")
    table = pos
    entries = []
    size = _encoded_size(table_enc)
    if size is None:
        raise ValueError("cannot update .eh_frame_hdr table")
    for idx in range(int(count)):
        at = table + idx * size * 2
        decoded = _read_encoded(
            bytes(out), at, table_enc, section_addr=section_addr, base=section_addr
        )
        if decoded is None:
            raise ValueError("cannot decode .eh_frame_hdr entry")
        pc, after = decoded
        target = _read_encoded(
            bytes(out), after, table_enc, section_addr=section_addr, base=section_addr
        )
        if target is None:
            raise ValueError("cannot decode .eh_frame_hdr entry")
        fde, _ = target
        mapped, _limit = _remap_address(pc, old_to_new)
        entries.append((mapped if mapped is not None else pc, fde))
    entries.sort(key=lambda item: item[0])
    updated = 0
    for idx, (pc, fde) in enumerate(entries):
        at = table + idx * size * 2
        if not _write_encoded(
            out, at, table_enc, pc, section_addr=section_addr, base=section_addr
        ):
            raise ValueError("cannot encode .eh_frame_hdr address")
        if not _write_encoded(
            out,
            at + size,
            table_enc,
            fde,
            section_addr=section_addr,
            base=section_addr,
        ):
            raise ValueError("cannot encode .eh_frame_hdr FDE pointer")
        updated += 1
    return bytes(out), updated


def _remap_factory(old_to_new):
    def _remap(old_addr):
        for (old_lo, old_hi), (new_lo, _new_hi) in old_to_new.items():
            if old_lo <= int(old_addr) < old_hi:
                return new_lo + (int(old_addr) - old_lo)
        return None

    return _remap


def _fixup_layout_image(
    real, original_image, text_sec, text_lo, funcs, blobs, old_to_new, fixup_stats
):

    _remap = _remap_factory(old_to_new)
    arch = get_arch()
    with open(real, "rb") as fh:
        elffile = ELFFile(fh)
        exec_secs = []
        for sec in elffile.iter_sections():
            try:
                flags = int(sec.header["sh_flags"])
                if not flags & 0x4:
                    continue
                exec_secs.append(
                    (
                        sec.name,
                        int(sec.header["sh_addr"]),
                        int(sec.header["sh_offset"]),
                        int(sec.header["sh_size"]),
                    )
                )
            except Exception:
                continue
        try:
            e_entry = int(elffile.header["e_entry"])
        except Exception:
            e_entry = None
    new_image = bytearray(original_image)

    new_text = bytearray(
        original_image[text_sec["offset"] : text_sec["offset"] + text_sec["size"]]
    )
    for i in range(len(new_text)):
        new_text[i] = 0xCC
    range_to_blob = {
        (start, end): blobs.get(name, b"") for name, (start, end) in funcs.items()
    }
    for (old_lo, old_hi), (new_lo, _new_hi) in old_to_new.items():
        blob = range_to_blob.get((old_lo, old_hi), b"")
        off = new_lo - text_lo
        new_text[off : off + len(blob)] = blob
    new_image[text_sec["offset"] : text_sec["offset"] + text_sec["size"]] = bytes(
        new_text
    )

    def _patch_rel(blob_arr, insn_addr, size, new_disp):
        if arch.patch_branch(blob_arr, insn_addr, size, new_disp):
            return True
        if size in (2, 3):
            raise ValueError(
                "short branch out of range after layout; "
                "refusing to expand (safe fallback)"
            )
        if size in (5, 6):
            raise ValueError("branch target out of range after layout")
        raise ValueError(f"cannot patch branch size {size} after layout")

    md = arch.detailed_disassembler()
    for sec_name, sec_addr, sec_off, sec_size in exec_secs:
        if sec_name == ".text":
            for (old_lo, old_hi), (new_lo, _new_hi) in old_to_new.items():
                blob = range_to_blob.get((old_lo, old_hi), b"")
                for insn in md.disasm(bytes(blob), old_lo):
                    size = int(insn.size)
                    new_addr = new_lo + (int(insn.address) - old_lo)
                    new_off = (new_addr - text_lo) + text_sec["offset"]
                    old_target = arch.immediate_target(insn)
                    if old_target is not None:
                        mapped = _remap(old_target)
                        new_target = mapped if mapped is not None else old_target
                        new_disp = new_target - (new_addr + size)
                        if _patch_rel(new_image, new_off, size, int(new_disp)):
                            fixup_stats["rel32"] += 1
                    rel = arch.rip_relative(insn)
                    if rel is None:
                        continue
                    disp_off, disp_size, cur_disp = rel
                    if disp_size != 4:
                        continue
                    old_target = (
                        int(insn.address) + size + int(cur_disp)
                    ) & 0xFFFFFFFFFFFFFFFF
                    mapped = _remap(old_target)
                    new_target = mapped if mapped is not None else old_target
                    new_disp = new_target - (new_addr + size)
                    if not -(2**31) <= new_disp < 2**31:
                        raise ValueError(
                            "RIP-relative target out of range after layout"
                        )
                    try:
                        struct.pack_into(
                            "<i",
                            new_image,
                            new_off + int(disp_off),
                            int(new_disp),
                        )
                        fixup_stats["rip"] += 1
                    except struct.error:
                        continue
        else:
            blob = bytes(new_image[sec_off : sec_off + sec_size])
            patched = bytearray(blob)
            changed = False
            for insn in md.disasm(bytes(blob), sec_addr):
                size = int(insn.size)
                ioff = int(insn.address) - sec_addr
                old_target = arch.immediate_target(insn)
                if old_target is not None:
                    mapped = _remap(old_target)
                    if mapped is not None:
                        new_disp = mapped - (int(insn.address) + size)
                        if size in (2, 3) and not -128 <= new_disp <= 127:
                            raise ValueError("short branch out of range after layout")
                        if size in (5, 6) and not -(2**31) <= new_disp < 2**31:
                            raise ValueError("branch target out of range after layout")
                        if size not in (2, 3, 5, 6):
                            raise ValueError(
                                f"cannot patch branch size {size} after layout"
                            )
                        if arch.patch_branch(patched, ioff, size, int(new_disp)):
                            fixup_stats["rel32"] += 1
                            changed = True
                rel = arch.rip_relative(insn)
                if rel is None:
                    continue
                disp_off, disp_size, cur_disp = rel
                if disp_size != 4:
                    continue
                old_target = (
                    int(insn.address) + size + int(cur_disp)
                ) & 0xFFFFFFFFFFFFFFFF
                mapped = _remap(old_target)
                if mapped is None:
                    continue
                new_disp = mapped - (int(insn.address) + size)
                if not -(2**31) <= new_disp < 2**31:
                    raise ValueError("RIP-relative target out of range after layout")
                try:
                    struct.pack_into("<i", patched, ioff + int(disp_off), int(new_disp))
                    fixup_stats["rip"] += 1
                    changed = True
                except struct.error:
                    continue
            if changed:
                new_image[sec_off : sec_off + sec_size] = bytes(patched)

    if e_entry is not None:
        mapped_entry = _remap(e_entry)
        if mapped_entry is not None and mapped_entry != e_entry:
            try:
                struct.pack_into("<Q", new_image, 24, int(mapped_entry))
            except struct.error:
                pass

    with open(real, "rb") as fh:
        elffile3 = ELFFile(fh)
        rela_secs = []
        for sec in elffile3.iter_sections():
            try:
                if sec.header["sh_type"] not in ("SHT_RELA", "SHT_REL"):
                    continue
                rela_secs.append(
                    (
                        sec.name,
                        int(sec.header["sh_offset"]),
                        int(sec.header["sh_entsize"] or 24),
                        int(sec.header["sh_size"]),
                    )
                )
            except Exception:
                continue
    for _rn, sec_off, entsize, sec_size in rela_secs:
        if entsize != 24:
            continue
        for at in range(sec_off, sec_off + sec_size, entsize):
            try:
                _r_off, _r_info, r_add = struct.unpack_from("<QQq", original_image, at)
            except struct.error:
                continue
            try:
                add = int(r_add) & 0xFFFFFFFFFFFFFFFF
            except Exception:
                continue
            mapped = _remap(add)
            if mapped is None:
                continue
            try:
                struct.pack_into("<q", new_image, at + 16, int(mapped))
                fixup_stats["rela"] += 1
            except struct.error:
                continue

    with open(real, "rb") as fh:
        elffile4 = ELFFile(fh)
        frame_secs = []
        hdr_secs = []
        for sec in elffile4.iter_sections():
            try:
                if sec.name == ".eh_frame":
                    frame_secs.append(
                        (
                            int(sec.header["sh_addr"]),
                            int(sec.header["sh_offset"]),
                            int(sec.header["sh_size"]),
                        )
                    )
                elif sec.name == ".eh_frame_hdr":
                    hdr_secs.append(
                        (
                            int(sec.header["sh_addr"]),
                            int(sec.header["sh_offset"]),
                            int(sec.header["sh_size"]),
                        )
                    )
            except Exception:
                continue
    for sec_addr, sec_off, sec_size in frame_secs:
        orig_frame = bytes(original_image[sec_off : sec_off + sec_size])
        new_frame, updated = _patch_eh_frame(orig_frame, old_to_new, sec_addr)
        new_image[sec_off : sec_off + sec_size] = new_frame
        fixup_stats["eh_frame"] += int(updated)
    for sec_addr, sec_off, sec_size in hdr_secs:
        orig_hdr = bytes(original_image[sec_off : sec_off + sec_size])
        new_hdr, updated = _patch_eh_frame_hdr(orig_hdr, old_to_new, sec_addr)
        new_image[sec_off : sec_off + sec_size] = new_hdr
        fixup_stats["eh_frame_hdr"] += int(updated)

    with open(real, "rb") as fh:
        elffile5 = ELFFile(fh)
        sym_secs = []
        for sec in elffile5.iter_sections():
            try:
                is_sym = sec.header["sh_type"] in ("SHT_SYMTAB", "SHT_DYNSYM")
                if not is_sym and sec.name not in (".symtab", ".dynsym"):
                    continue
                sym_secs.append(
                    (
                        sec.name,
                        int(sec.header["sh_offset"]),
                        int(sec.header["sh_entsize"] or 24),
                        int(sec.header["sh_size"]),
                    )
                )
            except Exception:
                continue
    start_map = {o_lo: n_lo for (o_lo, _o_hi), (n_lo, _n_hi) in old_to_new.items()}
    for _sn, sec_off, entsize, sec_size in sym_secs:
        if entsize != 24:
            continue
        for idx in range(sec_size // entsize):
            at = sec_off + idx * entsize
            try:
                _a, _b, _c, _d, st_value, _e = struct.unpack_from(
                    "<IBBHQQ", original_image, at
                )
            except struct.error:
                continue
            try:
                value = int(st_value)
            except Exception:
                continue
            if value in start_map:
                try:
                    struct.pack_into("<Q", new_image, at + 8, int(start_map[value]))
                except struct.error:
                    continue
    return new_image


def _verify_text(new_text, base, funcs, old_to_new=None):
    return get_arch().verify_text(new_text, base, funcs, old_to_new)
