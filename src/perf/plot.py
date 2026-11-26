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
import math
import os
import re
import shutil
import sys
import threading
import time
from fnmatch import fnmatchcase
from importlib import import_module

import pandas as pd

from .arch import arch as get_arch
from .core import _split_groups, _text, serialized

_CHART = {
    "ecdf": lambda df, e, **k: _dist(df, e, sns.ecdfplot, complementary=False, **k),
    "point": lambda df, e, **k: _cat(df, e, sns.pointplot, **k),
    "scatter": lambda df, e, **k: _xy(df, e, sns.scatterplot, **k),
    "line": lambda df, e, **k: _xy(df, e, sns.lineplot, **k),
    "hist": lambda df, e, **k: _dist(df, e, sns.histplot, element="step", **k),
    "bar": lambda df, e, **k: _cat(df, e, sns.barplot, **k),
    "box": lambda df, e, **k: _cat(df, e, sns.boxplot, **k),
    "boxen": lambda df, e, **k: _cat(df, e, sns.boxenplot, **k),
    "rug": lambda df, e, **k: _dist(df, e, sns.rugplot, **k),
    "kde": lambda df, e, **k: _dist(df, e, sns.kdeplot, **k),
    "violin": lambda df, e, **k: _cat(df, e, sns.violinplot, **k),
    "strip": lambda df, e, **k: _cat(df, e, sns.stripplot, **k),
    "swarm": lambda df, e, **k: _cat(df, e, sns.swarmplot, **k),
    "cfg": lambda df, e, **k: _draw_flow(plt.gca(), _flow_for(df)),
}
_PLOT = {
    "figure.figsize": (10, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "gray",
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "axes.grid": True,
}
_GROUPBY = ("file", "name", "mode", "data.*")
_ERRORBAR_CHARTS = {"line", "bar", "point"}
_ERRORBAR_CAPS = 0.1
_LINE_ERR_CAPS = 3
_BAR_DOT_SIZE = 20
_TERMINAL_CELL_W = 7.0
_TERMINAL_CELL_H = 14.0
_SCREEN_MARGIN = 0.92
_CHART_CELL_IN = (5.0, 4.0)
_MAX_FIGURE_IN = (16.0, 12.0)
_IMAGE_EXTS = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
_BEHAVIOR = ("logx", "logy")
_FLOW_NODE_STYLES = {
    "entry": {"face": "#d6f5df", "edge": "#15803d"},
    "branch": {"face": "#ffe6c7", "edge": "#d97706"},
    "return": {"face": "#ffd6d6", "edge": "#c0392b"},
    "exit": {"face": "#e9d5ff", "edge": "#7e22ce"},
    "block": {"face": "#dbeafe", "edge": "#2563eb"},
}
_FLOW_EDGE_STYLES = {
    "fallthrough": {"color": "#6b7280", "style": "-", "width": 0.8},
    "jump": {"color": "#2563eb", "style": "-", "width": 0.9},
    "conditional": {"color": "#d97706", "style": "--", "width": 1.0},
    "unconditional": {"color": "#dc2626", "style": "-", "width": 1.1},
    "call": {"color": "#7c3aed", "style": ":", "width": 1.0},
    "return": {"color": "#b91c1c", "style": ":", "width": 1.0},
    "syscall": {"color": "#0891b2", "style": ":", "width": 1.0},
}
_FLOW_EDGE_PRIORITY = {
    "fallthrough": 0,
    "syscall": 1,
    "jump": 2,
    "conditional": 3,
    "unconditional": 4,
    "call": 5,
    "return": 6,
}
_FLOW_MODES = frozenset(
    {
        "static",
        "emulated",
        "dynamic",
        "emu",
        "cfgemulated",
        "cfg-emulated",
        "cfg_emulated",
    }
)
_BACKEND_ALIASES = {
    "ascii": "module://mpl_ascii",
    "sixel": "module://matplotlib_backend_sixel",
}
_BACKEND_AUTO = ("", "auto", "default")
_DA1_RE = re.compile(rb"\x1b\[\?([0-9;]*)c")
_UNSET = object()
_PLOT_LOCK = threading.RLock()
_serialized = serialized(_PLOT_LOCK)


class PlotConfig:
    def __init__(self):
        self._params = {}

    def __call__(self, *args, **kwargs):
        return plot(*args, **kwargs)

    @property
    def config(self):
        with _PLOT_LOCK:
            return dict(self._params)

    @config.setter
    def config(self, value):
        with _PLOT_LOCK:
            self._params = _plot_cfg(value)


def normalize_config(config):
    return _flatten_config("", config, {})


def flow_graph(source, mode="static"):
    if isinstance(source, dict) and "nodes" in source:
        return source
    if isinstance(source, pd.DataFrame):
        return _flow_from_frame(source)
    return _flow_from_binary(source, _flow_mode({"cfg": mode}))


def ensure_group(df, groupby=None):
    df = df.copy()
    df, keys = _resolve_group_keys(df, groupby)
    if not keys:
        return df, None
    if len(keys) == 1:
        col = keys[0]
        try:
            df[col] = _group_series(df[col])
        except Exception:
            pass
        return df, col
    label = "/".join(keys)
    if label not in df.columns:
        try:
            ngroups = 1
            for key in keys:
                try:
                    ngroups *= max(1, int(df[key].nunique(dropna=False)))
                except Exception:
                    ngroups = 0
                    break
                if ngroups > 100000:
                    break
            if ngroups and ngroups <= 10000 and len(df) > ngroups * 5:
                try:
                    grouped = df.groupby(keys, dropna=False, observed=True)
                    ordered_keys = list(grouped.groups.keys())
                    if ordered_keys and not isinstance(ordered_keys[0], tuple):
                        ordered_keys = [(k,) for k in ordered_keys]
                    ordered_labels = []
                    for gk in ordered_keys:
                        parts = []
                        for value in gk:
                            try:
                                parts.append(_text(value))
                            except Exception:
                                parts.append("")
                        parts = [p for p in parts if p]
                        ordered_labels.append("/".join(parts))
                    codes = grouped.ngroup()
                    df[label] = (
                        pd.Series(codes.to_numpy(), index=df.index)
                        .map(dict(enumerate(ordered_labels)))
                        .fillna("")
                        .astype(str)
                    )
                    return df, label
                except Exception:
                    pass
        except Exception:
            pass
        try:
            cleaned = [_group_series(df[k]) for k in keys]
            first = cleaned[0].astype(str)
            rest = [c.astype(str) for c in cleaned[1:]]
            df[label] = first.str.cat(rest, sep="/")
            try:
                df[label] = df[label].str.strip("/")
                try:
                    if bool(df[label].str.contains("//").any()):
                        df[label] = (
                            df[label].str.replace(r"/+", "/", regex=True).str.strip("/")
                        )
                except Exception:
                    pass
            except Exception:  # pragma: no cover
                try:
                    df[label] = (
                        df[label].str.replace(r"/+", "/", regex=True).str.strip("/")
                    )
                except Exception:
                    pass
            try:
                empty = df[label].str.strip().str.strip("/")
                mask = empty == ""
                if bool(mask.any()):
                    df.loc[mask, label] = ""
            except Exception:
                pass
        except Exception:
            try:
                parts = [_group_series(df[k]).to_numpy(dtype=str) for k in keys]
                df[label] = [
                    "/".join(s for s in row if s and s not in ("nan", "None"))
                    for row in zip(*parts)
                ]
            except Exception:
                try:
                    df[label] = df[keys].astype(str).agg("/".join, axis=1)
                except Exception:
                    df[label] = ""
    return df, label


def _auto_types(df, xaxis=None):
    attrs = getattr(df, "attrs", {}) or {}
    if attrs.get("flow_graph") or attrs.get("cfg"):
        return ["cfg"]
    try:
        columns = {str(column) for column in df.columns}
    except Exception:
        columns = set()
    if "assembly" in columns or (
        "index" in columns and ("address" in columns or "encoding" in columns)
    ):
        return ["cfg"]
    if xaxis and str(xaxis) in columns:
        try:
            if int(df[str(xaxis)].nunique(dropna=True)) > 1:
                return ["line"]
        except Exception:
            pass
    return ["ecdf"]


def _expand_types(type_groups, df, xaxis=None):
    out = []
    for group in type_groups:
        inferred = None
        names = []
        for name in group:
            if str(name).strip().lower() == "auto":
                if inferred is None:
                    inferred = _auto_types(df, xaxis)
                names.extend(inferred)
            else:
                names.append(name)
        if names:
            out.append(names)
    return out


@_serialized
def plot(
    df,
    type=None,
    event=None,
    xaxis=None,
    yaxis=None,
    groupby=None,
    config=None,
    output=None,
    filter=None,
):
    if filter:
        from .data import query as _query

        df = _query(df, filter)
    if type is None:
        type = "auto"
    requested_backend = _config_backend(config)
    backend = _ensure_backend(requested_backend)
    auto_backend = requested_backend is None or str(
        requested_backend
    ).strip().lower() in (_BACKEND_AUTO)
    if auto_backend and output is not None:
        backend = _ensure_backend("Agg")
    text_fallback = False
    if auto_backend and output is None and _is_agg_backend(backend):
        if _module_available("mpl_ascii"):
            backend = _ensure_backend("ascii")
        else:
            text_fallback = True
    ascii_backend = _is_ascii_backend(backend)
    logx, logy = _plot_options(config)
    if ascii_backend:
        logx = logy = False
    type_groups = _expand_types(_split_groups(type, ["ecdf"]), df, xaxis)
    for group in type_groups:
        for name in group:
            if name not in _CHART:
                raise KeyError(
                    f"unknown chart {name!r}; available: "
                    f"{', '.join(sorted(_CHART))}, auto"
                )
    saved = []
    cfg_groups = [group for group in type_groups if "cfg" in group]
    if cfg_groups:
        saved = _plot_flow_groups(
            df, cfg_groups, output, config, ascii_backend or text_fallback
        )
        type_groups = [group for group in type_groups if "cfg" not in group]
        if not type_groups:
            return saved
    if event is None and "duration_time" not in df.columns:
        try:
            from .comp import metric_columns as _metrics

            fallback = _metrics(df)
            if fallback:
                event = [[c] for c in fallback]
        except Exception:
            pass
    event_groups = _split_groups(event, ["duration_time"])
    missing = [e for row in event_groups for e in row if e not in df.columns]
    if missing:
        raise KeyError(f"event(s) not found in data: {', '.join(missing)}")
    df, sample_keys = _resolve_group_keys(df, groupby)
    df = _sample_rows(df, sample_keys, _sample_cap(type_groups))
    if xaxis is not None:
        _xaxis(df, xaxis)
        try:
            df = _materialize_axis(df, xaxis)
            if isinstance(xaxis, str) and xaxis in df.columns:
                df = df.sort_values(xaxis, kind="stable").reset_index(drop=True)
        except Exception:
            pass
    if yaxis is not None:
        _yaxis(df, yaxis)
        try:
            df = _materialize_axis(df, yaxis)
        except Exception:
            pass
    df, label = ensure_group(df, groupby)
    duration_label = _duration_axis_label(df)
    if duration_label is not None:
        try:
            df = df.copy()
            if "duration_time" in df.columns:
                df["duration_time"] = _duration_ns_values(df)
            if "duration_time/operations" in df.columns:
                df["duration_time/operations"] = _duration_per_op_ns_values(df)
        except Exception:
            duration_label = None
    hue_order, palette = _shared_palette(df, label)
    paths = (
        _resolve_plot_paths(output, type_groups)
        if output is not None
        else [None] * len(type_groups)
    )
    for type_group, output_path in zip(type_groups, paths):
        ncharts = len(event_groups)
        rows, cols, cell_w, cell_h = _grid_for(ncharts, output_path)
        fig, axes = plt.subplots(rows, cols, squeeze=False)
        fig.set_size_inches(cols * cell_w, rows * cell_h)
        flat = [ax for row in axes for ax in row]
        for ax in flat[ncharts:]:
            ax.set_visible(False)
        for ax, event_group in zip(flat, event_groups):
            plt.sca(ax)
            ys = [_yaxis(df, yaxis, ev) for ev in event_group]
            for y, ev in zip(ys, event_group):
                for name in type_group:
                    call_kw = dict(_extra_for_event(df, label, hue_order, palette, ev))
                    if name in _ERRORBAR_CHARTS:
                        call_kw.setdefault("errorbar", _nonnegative_sd)
                        if name in ("bar", "point"):
                            call_kw.setdefault("capsize", _ERRORBAR_CAPS)
                        elif name == "line":
                            call_kw.setdefault("err_style", "bars")
                            if "marker" not in call_kw and "markers" not in call_kw:
                                call_kw["marker"] = "o"
                            err_kws = dict(call_kw.get("err_kws") or {})
                            err_kws.setdefault("capsize", _LINE_ERR_CAPS)
                            call_kw["err_kws"] = err_kws
                    if xaxis is not None and "x" not in call_kw:
                        call_kw["x"] = xaxis
                    _CHART[name](df, y, group=label, **call_kw)
                    if name == "bar":
                        _bar_mean_dots(ax)
            _style_axes(ax)
            _apply_duration_label(ax, ys, duration_label)
            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            if isinstance(xaxis, str) and any(
                name in ("line", "scatter") for name in type_group
            ):
                try:
                    ax.set_xlabel(xaxis)
                except Exception:
                    pass
        entries = _legend_entries(flat) if ascii_backend else None
        if ascii_backend:
            _ascii_legend_frames()
            _ascii_prepare(fig)
        nleg = _single_legend(
            fig,
            flat,
            hue_order,
            prefer_below=rows > 1,
            entries=entries,
        )
        if not nleg:
            fig.tight_layout()
        res = _show_or_save(fig, output_path)
        if res is not None:
            saved.append(res)
    return saved


def _flow_mode(config):
    value = None
    if isinstance(config, dict):
        value = config.get("cfg")
        if value is None:
            value = config.get("cfg.mode")
        if isinstance(value, dict):
            value = value.get("mode") or value.get("backend")
    elif config is not None:
        value = getattr(config, "cfg", None)
    text = str(value or "static").strip().lower()
    return "emulated" if text in _FLOW_MODES and text != "static" else "static"


def _empty_flow(name=""):
    return {"nodes": [], "edges": [], "entry": None, "name": name}


def _asm_parts(asm):
    parts = str(asm or "").strip().split(None, 1)
    if not parts:
        return "", ""
    return parts[0].strip().lower(), (parts[1].strip() if len(parts) > 1 else "")


def _is_branch(mnemonic):
    return bool(get_arch().is_branch(mnemonic))


def _is_unconditional_jump(mnemonic):
    return bool(get_arch().is_unconditional_jump(mnemonic))


def _is_terminator(mnemonic):
    return bool(get_arch().is_terminator(mnemonic))


def _is_return(mnemonic):
    return bool(get_arch().is_return_mnemonic(mnemonic))


def _branch_target(mnemonic, operand):
    if not _is_branch(mnemonic):
        return None
    return get_arch().branch_target(mnemonic, operand)


def _node_kind(address, last_mnemonic, has_branch, entry):
    if address == entry:
        return "entry"
    if _is_return(last_mnemonic):
        return "return"
    if _is_terminator(last_mnemonic):
        return "exit"
    if has_branch:
        return "branch"
    return "block"


def _edge_kind(mnemonic, fallthrough=False):
    if fallthrough:
        return "fallthrough"
    if _is_unconditional_jump(mnemonic):
        return "unconditional"
    if _is_branch(mnemonic):
        return "conditional"
    return "jump"


def _add_edge(edges, source, target, kind):
    current = edges.get((source, target))
    if current is None or _FLOW_EDGE_PRIORITY.get(kind, 0) >= _FLOW_EDGE_PRIORITY.get(
        current, 0
    ):
        edges[(source, target)] = kind


def _short_line(text, limit=42):
    text = str(text or "").strip()
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _flow_from_frame(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return _empty_flow()
    addr_col = _first_col(df, "address", "ip", "addr")
    if addr_col is None:
        return _empty_flow()
    asm_col = _first_col(df, "assembly", "disassembly", "text")
    size_col = _first_col(df, "size")
    records = {}
    for _, row in df.iterrows():
        try:
            addr = (
                int(row[addr_col], 0)
                if isinstance(row[addr_col], str)
                else int(row[addr_col])
            )
        except (TypeError, ValueError):
            continue
        if addr in records:
            continue
        try:
            size = int(row[size_col]) if size_col is not None else 0
        except (TypeError, ValueError):
            size = 0
        records[addr] = {
            "address": addr,
            "size": max(0, size),
            "asm": str(row[asm_col]) if asm_col is not None else "",
        }
    items = sorted(records.values(), key=lambda record: record["address"])
    if not items:
        return _empty_flow()
    addresses = {record["address"] for record in items}
    index = {record["address"]: i for i, record in enumerate(items)}
    leaders = {items[0]["address"]}
    for i, record in enumerate(items):
        if i > 0:
            previous = items[i - 1]
            if previous["size"] and (
                record["address"] != previous["address"] + previous["size"]
            ):
                leaders.add(record["address"])
        mnemonic, operand = _asm_parts(record["asm"])
        target = _branch_target(mnemonic, operand)
        if target in addresses:
            record["target"] = target
            leaders.add(target)
        if _is_branch(mnemonic) or _is_terminator(mnemonic):
            if i + 1 < len(items):
                leaders.add(items[i + 1]["address"])

    blocks = []
    current = []
    for record in items:
        if current and (
            record["address"] in leaders
            or _is_branch(_asm_parts(current[-1]["asm"])[0])
            or _is_terminator(_asm_parts(current[-1]["asm"])[0])
        ):
            blocks.append(current)
            current = []
        current.append(record)
    if current:
        blocks.append(current)

    entry = items[0]["address"]
    block_of = {}
    nodes = []
    for block in blocks:
        start = block[0]["address"]
        for record in block:
            block_of[record["address"]] = start
        lines = [f"0x{start:x}"]
        lines.extend(_short_line(record["asm"]) for record in block[:4])
        if len(block) > 4:
            lines.append(f"... +{len(block) - 4}")
        mnemonics = [_asm_parts(record["asm"])[0] for record in block]
        nodes.append(
            {
                "id": start,
                "address": start,
                "size": sum(max(1, record["size"]) for record in block),
                "kind": _node_kind(
                    start,
                    mnemonics[-1],
                    any(_is_branch(name) for name in mnemonics),
                    entry,
                ),
                "label": "\n".join(line for line in lines if line),
                "lines": lines,
            }
        )

    edges = {}

    def _fallthrough(record):
        i = index.get(record["address"])
        if i is None or i + 1 >= len(items):
            return None
        nxt = items[i + 1]
        if record["size"] and nxt["address"] != record["address"] + record["size"]:
            return None
        return block_of.get(nxt["address"])

    for block in blocks:
        start = block[0]["address"]
        last = block[-1]
        mnemonic, operand = _asm_parts(last["asm"])
        target = last.get("target", _branch_target(mnemonic, operand))
        target_block = block_of.get(target) if target is not None else None
        if target_block is not None:
            _add_edge(
                edges,
                start,
                target_block,
                _edge_kind(mnemonic),
            )
        if _is_unconditional_jump(mnemonic):
            continue
        if _is_branch(mnemonic) or not _is_terminator(mnemonic):
            fall = _fallthrough(last)
            if fall is not None and fall != start:
                _add_edge(edges, start, fall, "fallthrough")

    return {
        "nodes": nodes,
        "edges": [
            {"source": source, "target": target, "kind": kind}
            for (source, target), kind in sorted(edges.items())
        ],
        "entry": entry,
        "name": "",
    }


def _flow_from_binary(path, mode="static"):
    import angr

    project = angr.Project(str(path), auto_load_libs=False)
    text = str(mode).strip().lower()
    if text in _FLOW_MODES and text != "static":
        cfg = project.analyses.CFGEmulated(keep_state=True)
    else:
        cfg = project.analyses.CFGFast(normalize=True)
    graph = cfg.graph
    try:
        main = project.loader.main_object
        low = int(getattr(main, "min_addr", 0) or 0)
        high = int(getattr(main, "max_addr", 0) or 0)
    except Exception:
        low = high = 0
    entry = int(getattr(project, "entry", 0) or 0)
    selected = {}
    for node in graph.nodes():
        try:
            addr = int(node.addr)
            size = int(getattr(node, "size", 0) or 0)
        except (TypeError, ValueError):
            continue
        if high > low and not low <= addr < high:
            continue
        lines = [f"0x{addr:x}"]
        name = str(getattr(node, "name", "") or "")
        if name:
            lines.append(_short_line(name))
        block = getattr(node, "block", None)
        try:
            insns = list(getattr(getattr(block, "capstone", None), "insns", ()) or ())
        except Exception:
            insns = []
        for insn in insns[:3]:
            asm_text = f"{insn.mnemonic} {insn.op_str}".strip()
            lines.append(_short_line(asm_text))
        last = str(insns[-1].mnemonic) if insns else ""
        selected[addr] = {
            "id": addr,
            "address": addr,
            "size": max(0, size),
            "kind": _node_kind(
                addr,
                last,
                any(_is_branch(str(insn.mnemonic)) for insn in insns),
                entry,
            ),
            "label": "\n".join(line for line in lines if line),
            "lines": lines,
            "last": last,
        }
    if not selected:
        for node in graph.nodes():
            try:
                addr = int(node.addr)
            except (TypeError, ValueError):
                continue
            selected[addr] = {
                "id": addr,
                "address": addr,
                "size": int(getattr(node, "size", 0) or 0),
                "kind": _node_kind(addr, "", False, entry),
                "label": f"0x{addr:x}",
                "lines": [f"0x{addr:x}"],
                "last": "",
            }
    ids = sorted(selected)[:300]
    keep = set(ids)
    edges = {}
    try:
        graph_edges = list(graph.edges(data=True))
    except Exception:
        try:
            graph_edges = [(a, b, {}) for a, b in graph.edges()]
        except Exception:
            graph_edges = []
    for source, target, data in graph_edges:
        try:
            source_addr, target_addr = int(source.addr), int(target.addr)
        except (AttributeError, TypeError, ValueError):
            continue
        if source_addr in keep and target_addr in keep:
            jumpkind = str((data or {}).get("jumpkind", "") or "")
            if jumpkind == "Ijk_Call":
                kind = "call"
            elif jumpkind == "Ijk_Ret":
                kind = "return"
            elif jumpkind == "Ijk_FakeRet":
                kind = "fallthrough"
            elif jumpkind.startswith("Ijk_Sys"):
                kind = "syscall"
            else:
                kind = _edge_kind(selected[source_addr].get("last", ""))
            _add_edge(edges, source_addr, target_addr, kind)
    if entry not in keep:
        entry = ids[0] if ids else None
    return {
        "nodes": [selected[addr] for addr in ids],
        "edges": [
            {"source": source, "target": target, "kind": kind}
            for (source, target), kind in sorted(edges.items())
        ],
        "entry": entry,
        "name": os.path.basename(str(path)),
    }


def _flow_for(df, config=None):
    attrs = getattr(df, "attrs", {}) or {}
    graph = attrs.get("flow_graph") or attrs.get("cfg")
    if isinstance(graph, dict) and graph.get("nodes"):
        return graph
    return _flow_from_frame(df)


def _flow_levels(graph):
    nodes = list(graph.get("nodes") or [])
    ids = [node.get("id") for node in nodes]
    idset = set(ids)
    succ = {node_id: [] for node_id in ids}
    for edge in graph.get("edges") or []:
        source = edge.get("source")
        target = edge.get("target")
        if source in idset and target in idset:
            succ[source].append(target)
    levels = {}
    entry = graph.get("entry")
    roots = [entry] if entry in idset else []
    roots += [node_id for node_id in ids if node_id not in roots]
    base = 0
    for root in roots:
        if root in levels:
            continue
        levels[root] = base
        queue = [root]
        while queue:
            current = queue.pop(0)
            for target in succ.get(current, ()):
                if target not in levels:
                    levels[target] = levels[current] + 1
                    queue.append(target)
        if levels:
            base = max(levels.values()) + 1
    return levels


def _flow_node_text(node):
    lines = [str(line) for line in node.get("lines") or ()]
    if not lines:
        lines = str(node.get("label") or "").splitlines() or [""]
    address = node.get("address")
    if address is None:
        return (lines[0], lines[1:]) if lines else ("", [])
    try:
        header = f"0x{int(address):x}"
    except (TypeError, ValueError):
        header = str(address)
    body = lines
    if body and body[0].strip() == header:
        body = body[1:]
    return header, body


def _flow_node_size(header, body):
    longest = max([len(header), *(len(line) for line in body)] or [8])
    width = max(1.4, min(4.2, longest * 0.07 + 0.45))
    body_height = max(0.2, len(body) * 0.17 + 0.1)
    return width, 0.24 + body_height


def _flow_order(nodes, levels, edges):
    grouped = {}
    for node in nodes:
        grouped.setdefault(levels.get(node.get("id"), 0), []).append(node)
    for group in grouped.values():
        group.sort(key=lambda node: node.get("address", node.get("id", 0)))
    preds = {node.get("id"): [] for node in nodes}
    succs = {node.get("id"): [] for node in nodes}
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source in preds and target in succs:
            succs[source].append(target)
            preds[target].append(source)

    def _average(neighbors, order, fallback):
        positions = [order[n] for n in neighbors if n in order]
        return sum(positions) / len(positions) if positions else fallback

    keys = sorted(grouped)
    for _ in range(3):
        for index in range(1, len(keys)):
            level = keys[index]
            previous = keys[index - 1]
            order = {node.get("id"): i for i, node in enumerate(grouped[previous])}
            current = {node.get("id"): i for i, node in enumerate(grouped[level])}
            grouped[level].sort(
                key=lambda node: (
                    _average(
                        preds.get(node.get("id"), ()),
                        order,
                        current.get(node.get("id"), 0),
                    ),
                    node.get("address", node.get("id", 0)),
                )
            )
        for index in range(len(keys) - 2, -1, -1):
            level = keys[index]
            following = keys[index + 1]
            order = {node.get("id"): i for i, node in enumerate(grouped[following])}
            current = {node.get("id"): i for i, node in enumerate(grouped[level])}
            grouped[level].sort(
                key=lambda node: (
                    _average(
                        succs.get(node.get("id"), ()),
                        order,
                        current.get(node.get("id"), 0),
                    ),
                    node.get("address", node.get("id", 0)),
                )
            )
    return grouped


def _flow_edge_path(source, target, positions, kind, bounds):
    sx, sy, sw, sh = positions[source][:4]
    tx, ty, tw, th = positions[target][:4]
    from matplotlib.path import Path as MplPath

    if source == target:
        start = (sx + sw / 2.0, sy + sh * 0.18)
        end = (sx + sw / 2.0, sy - sh * 0.18)
        loop = sx + sw / 2.0 + 0.55
        return MplPath(
            [start, (loop, sy + sh * 0.4), (loop, sy - sh * 0.4), end],
            [
                MplPath.MOVETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
            ],
        )
    if ty < sy:
        start_x = sx
        if kind == "conditional":
            start_x += sw * 0.18
        elif kind == "fallthrough":
            start_x -= sw * 0.18
        start = (start_x, sy - sh / 2.0)
        end = (tx, ty + th / 2.0)
        middle = (start[1] + end[1]) / 2.0
        return MplPath(
            [start, (start_x, middle), (tx, middle), end],
            [
                MplPath.MOVETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
            ],
        )
    low, high = bounds
    side = high + 0.55 if tx >= (low + high) / 2.0 else low - 0.55
    start = (sx + (sw / 2.0 if side > sx else -sw / 2.0), sy)
    end = (tx + (tw / 2.0 if side > tx else -tw / 2.0), ty)
    return MplPath(
        [start, (side, start[1]), (side, end[1]), end],
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
        ],
    )


def _draw_flow(ax, graph):
    graph = graph or _empty_flow()
    nodes = list(graph.get("nodes") or [])
    if not nodes:
        ax.text(
            0.5,
            0.5,
            "no control flow data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return graph
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    levels = _flow_levels(graph)
    edges = list(graph.get("edges") or [])
    grouped = _flow_order(nodes, levels, edges)
    sizes = {node.get("id"): _flow_node_size(*_flow_node_text(node)) for node in nodes}
    positions = {}
    top = 0.0
    gap_x, gap_y = 0.45, 0.9
    for level in sorted(grouped):
        group = grouped[level]
        row_height = max(sizes[node.get("id")][1] for node in group)
        row_width = sum(sizes[node.get("id")][0] for node in group)
        row_width += gap_x * max(0, len(group) - 1)
        cursor = -row_width / 2.0
        for node in group:
            node_id = node.get("id")
            width, height = sizes[node_id]
            x = cursor + width / 2.0
            y = top - row_height / 2.0
            positions[node_id] = (x, y, width, height, node)
            cursor += width + gap_x
        top -= row_height + gap_y
    min_x = min(x - width / 2.0 for x, _y, width, _h, _n in positions.values())
    max_x = max(x + width / 2.0 for x, _y, width, _h, _n in positions.values())
    min_y = min(y - height / 2.0 for _x, y, _w, height, _n in positions.values())
    max_y = max(y + height / 2.0 for _x, y, _w, height, _n in positions.values())
    outgoing = {}
    for edge in edges:
        outgoing.setdefault(edge.get("source"), []).append(edge.get("kind"))
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source not in positions or target not in positions:
            continue
        kind = str(edge.get("kind") or "jump")
        style = _FLOW_EDGE_STYLES.get(kind, _FLOW_EDGE_STYLES["jump"])
        try:
            path = _flow_edge_path(source, target, positions, kind, (min_x, max_x))
            ax.add_patch(
                FancyArrowPatch(
                    path=path,
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=style["width"],
                    linestyle=style["style"],
                    color=style["color"],
                    shrinkA=0,
                    shrinkB=0,
                    zorder=1,
                )
            )
        except Exception:
            continue
        label = None
        if kind == "conditional":
            label = "T"
        elif kind == "fallthrough" and "conditional" in outgoing.get(source, ()):
            label = "F"
        if label:
            sx, sy, sw, sh = positions[source][:4]
            ax.text(
                sx + (sw * 0.18 if label == "T" else -sw * 0.18),
                sy - sh / 2.0 - 0.1,
                label,
                ha="center",
                va="top",
                fontsize=6.5,
                color=style["color"],
                zorder=4,
            )
    for node_id, (x, y, width, height, node) in positions.items():
        header, body = _flow_node_text(node)
        style = _FLOW_NODE_STYLES.get(
            str(node.get("kind") or "block"), _FLOW_NODE_STYLES["block"]
        )
        header_height = 0.24
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2.0, y - height / 2.0),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                linewidth=1.0,
                edgecolor=style["edge"],
                facecolor=style["face"],
                zorder=2,
            )
        )
        ax.add_patch(
            Rectangle(
                (x - width / 2.0, y + height / 2.0 - header_height),
                width,
                header_height,
                facecolor=style["edge"],
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x,
            y + height / 2.0 - header_height / 2.0,
            header,
            ha="center",
            va="center",
            fontsize=7,
            family="monospace",
            fontweight="bold",
            color="white",
            zorder=4,
        )
        if body:
            ax.text(
                x,
                y - header_height / 2.0,
                "\n".join(body),
                ha="center",
                va="center",
                fontsize=7,
                family="monospace",
                linespacing=0.95,
                color="#111827",
                zorder=4,
            )
    padding = 0.4
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_axis_off()
    try:
        span_x = max_x - min_x
        span_y = max_y - min_y
        width = max(6.0, min(32.0, span_x * 1.35 + 1.4))
        height = max(4.0, min(40.0, span_y * 0.95 + 1.4))
        ax.figure.set_size_inches(width, height)
    except Exception:
        pass
    return graph


def _flow_ascii(graph):
    nodes = {node.get("id"): node for node in graph.get("nodes") or []}
    if not nodes:
        return "no control flow data"
    levels = _flow_levels(graph)
    grouped = _flow_order(list(nodes.values()), levels, list(graph.get("edges") or []))
    edges = list(graph.get("edges") or [])
    labels = {
        "conditional": "T",
        "fallthrough": "F",
        "unconditional": "J",
        "call": "C",
        "return": "R",
        "syscall": "S",
        "jump": "->",
    }
    out = [f"flow graph: {len(nodes)} blocks, {len(edges)} edges"]
    for level in sorted(grouped):
        for node in grouped[level]:
            node_id = node.get("id")
            header, body = _flow_node_text(node)
            kind = str(node.get("kind") or "block")
            out.append("")
            out.append(f"+-- {header} [{kind}]")
            for line in body:
                out.append(f"|   {line}")
            for edge in sorted(
                (edge for edge in edges if edge.get("source") == node_id),
                key=lambda edge: (
                    levels.get(edge.get("target"), 0),
                    edge.get("target") or 0,
                ),
            ):
                target = nodes.get(edge.get("target"))
                if target is None:
                    continue
                target_header, _ = _flow_node_text(target)
                edge_kind = str(edge.get("kind") or "jump")
                marker = labels.get(edge_kind, "->")
                out.append(
                    f"|   {marker}-> {target_header} [{target.get('kind') or 'block'}]"
                )
    return "\n".join(out)


def _plot_flow_groups(df, type_groups, output, config, ascii_backend):
    from pathlib import Path as _Path

    paths = (
        _resolve_plot_paths(output, type_groups)
        if output is not None
        else [None] * len(type_groups)
    )
    if ascii_backend and output is not None and len(type_groups) == 1:
        base = _Path(str(output))
        if base.suffix.lower() in (".txt", ".dot"):
            paths = [base]
    graph = _flow_for(df, config)
    saved = []
    for output_path in paths:
        if ascii_backend:
            text = _flow_ascii(graph)
            if output_path is None:
                print(text, flush=True)
            else:
                path = _Path(str(output_path))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{text}\n", encoding="utf-8")
                saved.append(str(path))
            continue
        fig, ax = plt.subplots()
        _, _, width, height = _grid_for(1, output_path)
        fig.set_size_inches(width, height)
        _draw_flow(ax, graph)
        res = _show_or_save(fig, output_path)
        if res is not None:
            saved.append(res)
    return saved


def _sample_cap(type_groups):
    flat_types = {t for grp in type_groups for t in (grp or [])}
    if "swarm" in flat_types:
        return 1000
    if "strip" in flat_types:
        return 2000
    if flat_types & {"violin", "box", "boxen"}:
        return 5000
    return 10000


def _sample_rows(df, keys, cap):
    if len(df) <= cap:
        return df
    try:
        if keys:
            try:
                ngroups = max(1, int(df.groupby(keys, dropna=False).ngroups))
            except Exception:
                ngroups = 1
            if ngroups > cap:
                return df.sample(n=cap, random_state=0).reset_index(drop=True)
            quota = max(1, cap // ngroups)
            parts = []
            for _, part in df.groupby(keys, dropna=False):
                if len(part) > quota:
                    part = part.sample(n=quota, random_state=0)
                parts.append(part)
            out = pd.concat(parts, ignore_index=True)
            if len(out) > cap:
                out = out.sample(n=cap, random_state=0).reset_index(drop=True)
            return out
        return df.sample(n=cap, random_state=0).reset_index(drop=True)
    except Exception:
        try:
            return df.sample(n=cap, random_state=0).reset_index(drop=True)
        except Exception:
            return df


class _LazyModule:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        return getattr(import_module(self._name), attr)


def _nonnegative_sd(values):
    import numpy as _np

    try:
        arr = _np.asarray(values, dtype=float)
    except Exception:
        return (0.0, 0.0)
    try:
        arr = arr[_np.isfinite(arr)]
    except Exception:
        return (0.0, 0.0)
    if arr.size == 0:
        return (0.0, 0.0)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if sd != sd:
        sd = 0.0
    return (max(0.0, mean - sd), mean + sd)


def _group_series(s):
    try:
        try:
            has_na = bool(s.isna().any())
        except Exception:
            has_na = False
        if not has_na:
            try:
                uniq = pd.unique(s)
                needs_blank = False
                for value in uniq:
                    try:
                        text = str(value)
                    except Exception:
                        continue
                    if text.strip().lower() in ("", "nan", "none", "nat"):
                        needs_blank = True
                        break
                if not needs_blank:
                    return s.astype(str)
            except Exception:
                pass
        text = s.astype(str)
        mask = text.str.strip().str.lower().isin(("", "nan", "none", "nat"))
        try:
            na = s.isna()
            mask = mask | na
        except Exception:
            pass
        return text.mask(mask, "")
    except Exception:
        return s


def _plot_cfg(config=None):
    flat, style = _split_config(config)
    params = dict(_PLOT)
    params.update(flat)
    if style:
        plt.style.use(style)
    _set_rcparams(params)
    if style is not None:
        params["style"] = style
    return params


def _plot_options(config=None):
    flat, style = _split_config(config)
    if style:
        plt.style.use(style)
    _set_rcparams(flat)
    return _config_flag(config, "logx"), _config_flag(config, "logy")


def _split_config(config):
    if config is None:
        flat = {}
    elif isinstance(config, dict):
        flat = normalize_config(config)
    else:
        flat = normalize_config(vars(config))
    style = flat.pop("style", None)
    flat.pop("backend", None)
    flat.pop("cfg", None)
    flat.pop("cfg.mode", None)
    flat.pop("cfg.backend", None)
    for key in _BEHAVIOR:
        flat.pop(key, None)
    return flat, style


def _config_backend(config):
    if isinstance(config, dict):
        return config.get("backend")
    return getattr(config, "backend", None)


def _config_flag(config, name):
    if isinstance(config, dict):
        return bool(config.get(name, False))
    return bool(getattr(config, name, False))


def _set_rcparams(params):
    for key, value in params.items():
        if key == "figure.figsize" and isinstance(value, (list, tuple)):
            value = tuple(value)
            params[key] = value
        plt.rcParams[key] = value
    return params


def _flatten_config(prefix, node, out):
    for key, value in dict(node or {}).items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_config(full, value, out)
        elif _has_dict(value):
            for alt in _dict_entries(value):
                sub = {}
                _flatten_config("", alt, sub)
                for name, val in sub.items():
                    _add_flat(out, f"{full}.{name}", val)
        else:
            out[full] = value
    return out


def _has_dict(value):
    if isinstance(value, dict):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_dict(v) for v in value)
    return False


def _add_flat(out, column, value):
    if column not in out:
        out[column] = [value]
    elif isinstance(out[column], list):
        out[column].append(value)
    else:
        out[column] = [out[column], value]


def _dict_entries(values):
    out = []
    if isinstance(values, dict):
        return [values]
    for value in values if isinstance(values, (list, tuple)) else ():
        out.extend(_dict_entries(value))
    return out


def _flatten_list(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _first_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _data_reg_name(col):
    if not isinstance(col, str) or not col.startswith("data."):
        return None
    rest = col[len("data.") :]
    if rest.startswith(("regs.", "mem.", "memory.")):
        return None
    try:
        int(str(rest).strip(), 0)
    except (TypeError, ValueError):
        return rest
    return None


def _canonical_data_col(col):
    rest = _data_reg_name(col)
    if rest is None:
        return col
    return f"data.{get_arch().canonical_data_reg(rest)}"


def _alias_data_columns(df):
    out = {}
    arch = get_arch()
    for col in list(getattr(df, "columns", [])):
        rest = _data_reg_name(col)
        if rest is None:
            continue
        try:
            keys = set(arch.data_reg_keys(rest))
        except Exception:
            keys = set()
        canon = arch.canonical_data_reg(rest)
        keys.add(f"data.{canon}")
        try:
            alias = arch.arg_alias(canon)
        except Exception:
            alias = None
        if alias:
            keys.add(f"data.{alias}")
            keys.add(f"data.{str(alias).lower()}")
        for k in keys:
            if k and k not in df.columns and k not in out:
                try:
                    out[k] = df[col]
                except Exception:
                    pass
    return out


def _resolve_x_series(df, x, axis="x"):
    if x is None:
        return None
    if isinstance(x, str) and x in df.columns:
        return df[x]
    if isinstance(x, str):
        try:
            canon_x = _canonical_data_col(x)
        except Exception:
            canon_x = x
        if canon_x != x and canon_x in df.columns:
            s = df[canon_x]
            try:
                s = s.rename(x)
            except Exception:
                pass
            return s
        if x.startswith("data."):
            try:
                for col in list(df.columns):
                    try:
                        if _canonical_data_col(col) == canon_x:
                            s = df[col]
                            try:
                                s = s.rename(x)
                            except Exception:
                                pass
                            return s
                    except Exception:
                        continue
            except Exception:
                pass
            try:
                aliases = _alias_data_columns(df)
                if x in aliases:
                    return aliases[x]
                if canon_x in aliases:
                    return aliases[canon_x]
            except Exception:
                pass
        try:
            attrs = getattr(df, "attrs", {}) or {}
            if x == "data" or x.startswith("data."):
                from .cfg import (
                    data_param_columns as _data_cols,
                    data_param_value as _data_val,
                )

                try:
                    cols = _data_cols(attrs.get("data"))
                except Exception:
                    cols = {}
                if x in cols:
                    return pd.Series(
                        [_flatten_list(cols[x])] * len(df), index=df.index, name=x
                    )
                try:
                    v = _data_val(attrs.get("data"), x)
                except Exception:
                    v = None
                if v is not None:
                    return pd.Series([v] * len(df), index=df.index, name=x)
            if x.startswith("config."):
                from .cfg import config_param_columns as _cfg_cols

                try:
                    cols = _cfg_cols(attrs.get("config"))
                except Exception:
                    cols = {}
                if x in cols:
                    return pd.Series(
                        [_flatten_list(cols[x])] * len(df), index=df.index, name=x
                    )
        except Exception:
            pass
        try:
            vals = df.eval(x, engine="python")
            try:
                vals = pd.Series(vals, index=df.index, name=x)
            except Exception:
                pass
            return vals
        except Exception:
            pass
    raise KeyError(f"{axis} column not found in data: {x!r}")


def _xaxis(df, x=None):
    if x is not None:
        try:
            series = _resolve_x_series(df, x)
        except KeyError:
            raise
        if series is not None:
            return series
    col = _first_col(df, "time")
    if col is not None:
        return df[col]
    if df.index.name:
        return df.index
    return range(len(df))


def _yaxis(df, y=None, event=None):
    if y is None:
        return event
    _resolve_x_series(df, y, "y")
    return y


def _materialize_axis(df, name):
    if not isinstance(name, str) or name in df.columns:
        return df
    series = _resolve_x_series(df, name)
    if series is None:
        return df
    df = df.copy()
    df[name] = pd.Series(list(series), index=df.index, name=name).values
    return df


def _resolve_group_keys(df, groupby=None):
    if groupby is None:
        groupby = list(_GROUPBY)
    if isinstance(groupby, str):
        groupby = [k.strip() for k in groupby.split(",") if k.strip()]
    index_names = list(df.index.names or [])
    available = list(df.columns)
    available += [n for n in index_names if n is not None and n not in df.columns]
    expanded = []
    for key in groupby:
        if isinstance(key, str) and any(w in key for w in ("*", "?")):
            for name in available:
                if name not in expanded and fnmatchcase(str(name), key):
                    expanded.append(name)
            continue
        if key not in expanded:
            expanded.append(key)
    groupby = expanded
    index_set = set(index_names)
    if any(k in index_set and k not in df.columns for k in groupby):
        df = df.reset_index()
    keys = [c for c in groupby if c in df.columns]
    return df, keys


def _default_group(df, group):
    if group is _UNSET:
        _, group = ensure_group(df)
        if group is None:
            group = "name" if "name" in df.columns else None
    return group


def _hue_kw(group, palette, hue_order, **kw):
    if group is not None:
        kw.setdefault("hue", group)
        if palette is not None:
            kw.setdefault("palette", palette)
        if hue_order is not None:
            kw.setdefault("hue_order", hue_order)
    return kw


def _dist(df, event, fn, group=_UNSET, palette=None, hue_order=None, **kw):
    kw.pop("x", None)
    group = _default_group(df, group)
    return fn(data=df, x=event, **_hue_kw(group, palette, hue_order, **kw))


def _cat(df, event, fn, group=_UNSET, palette=None, hue_order=None, **kw):
    kw.pop("x", None)
    group = _default_group(df, group)
    if group is None:
        return fn(data=df, y=event, **kw)
    kw.setdefault("legend", True)
    if hue_order is not None:
        kw.setdefault("order", hue_order)
    out = fn(data=df, x=group, y=event, **_hue_kw(group, palette, hue_order, **kw))
    try:
        ax = plt.gca()
        try:
            handles, labels = ax.get_legend_handles_labels()
        except Exception:
            handles, labels = [], []
        if ax.get_legend() is not None or (handles and labels):
            try:
                ax.set_xlabel("")
            except Exception:
                pass
            try:
                ax.tick_params(labelbottom=False)
            except Exception:
                pass
            try:
                ax.set_xticks([])
            except Exception:
                pass
    except Exception:
        pass
    return out


def _xy(df, event, fn, group=_UNSET, palette=None, hue_order=None, x=None, **kw):
    group = _default_group(df, group)
    return fn(
        data=df, x=_xaxis(df, x), y=event, **_hue_kw(group, palette, hue_order, **kw)
    )


def _shared_palette(df, label):
    if label is None or label not in df.columns:
        return None, None
    hue_order = list(pd.unique(df[label].astype(str)))
    if not hue_order:
        return None, None
    n = len(hue_order)
    if n <= 10:
        colors = sns.color_palette("colorblind", n_colors=n)
    else:
        colors = sns.color_palette("husl", n_colors=n)
    return hue_order, dict(zip(hue_order, colors))


def _available_inches(output_path):
    if output_path is not None:
        return None
    if not sys.stdout.isatty():
        return None
    try:
        dpi = float(plt.rcParams.get("figure.dpi", 100.0))
    except Exception:
        dpi = 100.0
    try:
        cols, lines = os.get_terminal_size()
    except Exception:
        return None
    if cols <= 0 or lines <= 0:
        return None
    return (
        cols * _TERMINAL_CELL_W / dpi * _SCREEN_MARGIN,
        lines * _TERMINAL_CELL_H / dpi * _SCREEN_MARGIN,
    )


def _figsize():
    try:
        fs = plt.rcParams.get("figure.figsize")
        if not fs or len(fs) < 2:
            fs = _PLOT.get("figure.figsize", (10, 5))
        w, h = float(fs[0]), float(fs[1])
    except Exception:
        w, h = _PLOT.get("figure.figsize", (10, 5))
    if not w > 0 or not h > 0:
        w, h = _PLOT.get("figure.figsize", (10, 5))
    return w, h


def _grid_rows_cols(n, avail_w, avail_h, cell):
    cw, ch = cell
    best = (math.inf, n, 1)
    for ncols in range(1, n + 1):
        nrows = math.ceil(n / ncols)
        need = max(ncols * cw / avail_w, nrows * ch / avail_h)
        if need < best[0]:
            best = (need, nrows, ncols)
    need, nrows, ncols = best
    if need > 1:
        scale = need
    else:
        scale = 1.0
    return nrows, ncols, cw / scale, ch / scale


def _grid_for(n, output_path):
    if n == 1:
        cell = _figsize()
    else:
        cell = _CHART_CELL_IN
    avail = _available_inches(output_path)
    if avail is not None:
        return _grid_rows_cols(n, *avail, cell)
    if n == 1:
        return 1, 1, *cell
    return _grid_rows_cols(n, *_MAX_FIGURE_IN, cell)


def _visible_order(df, label, hue_order, event):
    if label is None or label not in df.columns:
        return None
    if hue_order is None:
        return None
    if event not in df.columns:
        return []
    try:
        present = set(df.loc[df[event].notna(), label].astype(str).tolist())
    except Exception:
        return list(hue_order)
    return [h for h in hue_order if str(h) in present]


def _extra_for_event(df, label, hue_order, palette, event):
    ho = _visible_order(df, label, hue_order, event)
    if ho is not None and len(ho) == 0:
        ho = hue_order
    if palette is not None and ho is not None:
        try:
            present = set(ho)
            pal = {k: v for k, v in palette.items() if k in present}
        except Exception:
            pal = palette
    else:
        pal = palette
    extra = {}
    if ho is not None:
        extra["hue_order"] = ho
    if pal is not None:
        extra["palette"] = pal
    return extra


def _numeric_column(df, name):
    try:
        import pandas as _pd

        if name not in df.columns:
            return None
        return _pd.to_numeric(df[name], errors="coerce")
    except Exception:
        return None


def _duration_axis_label(df):
    try:
        cols = list(df.columns)
    except Exception:
        return None
    try:
        per_thread = False
        if "threads" in df.columns:
            vals = _numeric_column(df, "threads")
            if vals is not None:
                vals = vals.dropna()
                per_thread = bool(len(vals) and bool((vals != 1).any()))
    except Exception:
        per_thread = False
    suffix = "/threads" if per_thread else ""
    if "duration_time" not in cols:
        if "duration_time/operations" in cols:
            return f"duration_time[ns]/operations{suffix}"
        return None
    if "operations" not in cols:
        return f"duration_time[ns]{suffix}"
    try:
        vals = _numeric_column(df, "operations")
        if vals is not None:
            vals = vals.dropna()
            if len(vals) and bool((vals != 1).any()):
                return f"duration_time[ns]/operations{suffix}"
    except Exception:
        pass
    return f"duration_time[ns]{suffix}"


def _duration_ns_values(df):
    vals = _numeric_column(df, "duration_time")
    if vals is None:
        return vals
    try:
        ops = _numeric_column(df, "operations")
    except Exception:
        ops = None
    try:
        threads = _numeric_column(df, "threads")
    except Exception:
        threads = None
    try:
        vals = vals.copy()
        if ops is not None and bool(ops.notna().any()):
            mask = ops.notna() & (ops != 0)
            vals[mask] = vals[mask] / ops[mask]
            vals[~mask] = float("nan")
        if threads is not None and bool(threads.notna().any()):
            tmask = threads.notna() & (threads != 0)
            vals[tmask] = vals[tmask] / threads[tmask]
            vals[~tmask] = float("nan")
    except Exception:
        pass
    return vals


def _duration_per_op_ns_values(df):
    import pandas as _pd

    vals = _numeric_column(df, "duration_time/operations")
    if vals is None:
        return _pd.Series(dtype=float)
    try:
        threads = _numeric_column(df, "threads")
    except Exception:
        return vals
    if threads is None:
        return vals
    try:
        vals = vals.copy()
        if bool(threads.notna().any()):
            tmask = threads.notna() & (threads != 0)
            vals[tmask] = vals[tmask] / threads[tmask]
            vals[~tmask] = float("nan")
    except Exception:
        pass
    return vals


def _apply_duration_label(ax, event_group, duration_label):
    if not duration_label:
        return
    try:
        events = set(event_group or [])
    except Exception:
        return
    if not (
        {"duration_time", "duration_time/operations", "duration_time/threads"} & events
        or any(str(e).startswith("duration_time/") for e in events)
    ):
        return
    try:
        if str(ax.get_xlabel()).startswith(
            ("duration_time", "duration_time/operations")
        ):
            ax.set_xlabel(duration_label)
    except Exception:
        pass
    try:
        if str(ax.get_ylabel()).startswith(
            ("duration_time", "duration_time/operations")
        ):
            ax.set_ylabel(duration_label)
    except Exception:
        pass


def _bar_mean_dots(ax):
    try:
        patches = list(getattr(ax, "patches", []) or [])
    except Exception:
        return
    for patch in patches:
        try:
            w = patch.get_width()
            h = patch.get_height()
            if not w > 0 or not h > 0:
                continue
            x = patch.get_x() + w / 2.0
            y = h
        except Exception:
            continue
        try:
            ax.scatter(
                [x],
                [y],
                s=_BAR_DOT_SIZE,
                c="white",
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
        except Exception:
            pass


def _style_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)
    try:
        sns.despine(ax=ax, top=True, right=True)
    except Exception:
        pass


def _legend_right(labels):
    if not labels:
        return 1.0
    try:
        maxlen = max(len(str(lab)) for lab in labels)
    except Exception:
        maxlen = 20
    need = 0.05 + maxlen * 0.0075 + 0.03
    need = min(max(need, 0.12), 0.45)
    return 1.0 - need


def _ascii_canvas(fig):
    try:
        return "mpl_ascii" in type(fig.canvas).__module__
    except Exception:
        return False


def _ascii_legend_frames():
    try:
        from matplotlib.legend import Legend
        from mpl_ascii.parsing import figure as _figure
    except Exception:
        return
    if getattr(_figure, "_perf_legend_frames", False):
        return
    original = _figure.parse_artist

    def parse_artist(artist, insert_order):
        shapes, texts = original(artist, insert_order)
        if isinstance(artist, Legend):
            try:
                if not artist.get_frame_on():
                    shapes = [
                        shape
                        for shape in shapes
                        if not (len(shape.points) >= 4 and len(shape.lines) >= 4)
                    ]
            except Exception:
                pass
        return shapes, texts

    _figure.parse_artist = parse_artist
    _figure._perf_legend_frames = True


def _ascii_legend_height(fig, labels):
    try:
        height_in = float(fig.get_size_inches()[1])
    except Exception:
        height_in = 4.8
    if height_in <= 0:
        height_in = 4.8
    return (len(labels) + 2) * (1.8 / 12.0) / height_in


def _layout_renderer(fig):
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:
        renderer = None
    if renderer is not None:
        return renderer
    canvas = fig.canvas
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        FigureCanvasAgg(fig)
        fig.canvas.draw()
        return fig.canvas.get_renderer()
    except Exception:
        return None
    finally:
        try:
            fig.set_canvas(canvas)
        except Exception:
            pass


def _layout_with_legend(fig, leg, right):
    try:
        fig.tight_layout(rect=[0, 0, right, 1])
    except Exception:
        pass
    try:
        renderer = _layout_renderer(fig)
        leg_bb = leg.get_window_extent(renderer)
        fig_bb = fig.get_window_extent(renderer)
        overflow = max(
            float(leg_bb.x1 - fig_bb.x1),
            float(fig_bb.y0 - leg_bb.y0),
            float(leg_bb.y1 - fig_bb.y1),
        )
        if _ascii_canvas(fig):
            try:
                height_in = float(fig.get_size_inches()[1])
                rows = len(leg.get_texts()) + 2
                ascii_height = rows * (1.8 / 12.0) / height_in
                overflow = max(overflow, (ascii_height - 1.0) * float(fig_bb.height))
            except Exception:
                pass
    except Exception:
        overflow = 0.0
    return right, overflow


def _legend_kwargs(ncol=1):
    return dict(
        title=None,
        fontsize="medium",
        ncol=ncol,
        frameon=False,
        borderaxespad=0.0,
        borderpad=0.4,
        handlelength=1.6,
        handletextpad=0.4,
    )


def _legend_below_ncol(fig, labels):
    n = len(labels)
    try:
        maxlen = max(len(str(lab)) for lab in labels)
    except Exception:
        maxlen = 20
    try:
        fig_w_px = float(fig.get_size_inches()[0] * fig.dpi)
    except Exception:
        fig_w_px = 640.0
    col_w = maxlen * 6.5 + 50
    if n <= 2:
        ncol = n
    elif maxlen > 45:
        ncol = 1 if n <= 4 else 2
    elif maxlen > 28:
        ncol = min(n, 3)
    else:
        ncol = min(n, 6)
    ncol = max(1, ncol)
    while ncol > 1:
        total = ncol * col_w + (ncol - 1) * 20
        if total <= fig_w_px * 0.96:
            break
        ncol -= 1
    return ncol


def _move_legend_below(fig, handles, labels):
    try:
        for old in list(fig.legends):
            try:
                old.remove()
            except Exception:
                pass
    except Exception:
        pass
    ncol = _legend_below_ncol(fig, labels)
    leg = None
    bbox = None
    try:
        fig_w_px = float(fig.get_size_inches()[0] * fig.dpi)
    except Exception:
        fig_w_px = 640.0
    while True:
        leg = fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            **_legend_kwargs(ncol=ncol),
        )
        try:
            renderer = _layout_renderer(fig)
            bbox = leg.get_window_extent(renderer)
        except Exception:
            bbox = None
        if bbox is None or ncol <= 1 or bbox.width <= fig_w_px * 0.96:
            break
        try:
            leg.remove()
        except Exception:
            pass
        ncol -= 1
    bottom = 0.12
    ascii_canvas = _ascii_canvas(fig)
    if ascii_canvas:
        bottom = max(bottom, _ascii_legend_height(fig, labels) + 0.02)
    elif bbox is not None:
        try:
            fig_h_px = float(fig.get_size_inches()[1] * fig.dpi)
        except Exception:
            fig_h_px = 480.0
        if fig_h_px > 0:
            bottom = max(bottom, float(bbox.height) / fig_h_px + 0.02)
    bottom = min(max(bottom, 0.0), 0.9)

    def _apply(value):
        try:
            fig.tight_layout(rect=[0, value, 1, 1])
        except Exception:
            pass
        try:
            fig.subplots_adjust(bottom=value)
        except Exception:
            pass
        _layout_renderer(fig)

    _apply(bottom)
    try:
        renderer = _layout_renderer(fig)
        leg_bb = leg.get_window_extent(renderer)
        fig_h_px = float(fig.get_size_inches()[1] * fig.dpi)
        legend_top = float(leg_bb.y1)
        if ascii_canvas:
            legend_top = max(legend_top, _ascii_legend_height(fig, labels) * fig_h_px)
        need = bottom
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            tight = ax.get_tightbbox(renderer)
            if tight is None:
                continue
            if legend_top > tight.y0 and leg_bb.x1 > tight.x0 and leg_bb.x0 < tight.x1:
                need = max(need, bottom + (legend_top - tight.y0) / fig_h_px + 0.01)
        need = min(need, 0.9)
        if need > bottom:
            bottom = need
            _apply(bottom)
    except Exception:
        pass
    return leg


def _legend_entries(axes):
    seen = {}
    for ax in axes:
        if not ax.get_visible():
            continue
        try:
            h, lab = ax.get_legend_handles_labels()
        except Exception:
            h, lab = [], []
        if h and lab:
            for hh, ll in zip(h, lab):
                if ll not in seen:
                    seen[ll] = hh
        try:
            leg = ax.get_legend()
        except Exception:
            leg = None
        if leg is not None:
            try:
                texts = [t.get_text() for t in leg.get_texts()]
            except Exception:
                texts = []
            try:
                lh = list(
                    getattr(leg, "legend_handles", getattr(leg, "legendHandles", []))
                    or []
                )
            except Exception:
                lh = []
            if texts and lh and len(texts) == len(lh):
                for hh, ll in zip(lh, texts):
                    if ll not in seen:
                        seen[ll] = hh
    return seen


def _single_legend(fig, axes, hue_order=None, prefer_below=False, entries=None):
    seen = dict(entries) if entries is not None else _legend_entries(axes)
    for ax in axes:
        try:
            leg = ax.get_legend()
        except Exception:
            leg = None
        if leg is not None:
            leg.remove()
    if not seen:
        return 0
    if hue_order:
        ordered = [(lab, seen[lab]) for lab in hue_order if lab in seen]
        ordered += [(lab, h) for lab, h in seen.items() if lab not in (hue_order or [])]
    else:
        ordered = list(seen.items())
    if not ordered:
        return 0
    labels, handles = zip(*ordered)
    if prefer_below:
        _move_legend_below(fig, handles, labels)
        return len(labels)
    right = _legend_right(labels)
    leg = fig.legend(
        handles,
        labels,
        title=None,
        loc="upper left",
        bbox_to_anchor=(right, 1.0),
        fontsize="medium",
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        borderpad=0.4,
        handlelength=1.6,
        handletextpad=0.4,
    )
    _, overflow = _layout_with_legend(fig, leg, right)
    if overflow > 1:
        _move_legend_below(fig, handles, labels)
    return len(labels)


def _resolve_plot_paths(output, type_groups):
    from pathlib import Path as _Path

    names = ["+".join(g) for g in type_groups]
    base = _Path(str(output))
    if base.suffix.lower() in _IMAGE_EXTS:
        if len(names) == 1:
            return [base]
        return [base.with_name(f"{base.stem}_{n}{base.suffix}") for n in names]
    return [base / f"{n}.png" for n in names]


def _save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    try:
        plt.close(fig)
    except Exception:
        pass
    return str(path)


def _is_ascii_backend(backend):
    return "ascii" in str(backend or "").lower()


def _is_agg_backend(backend):
    text = str(backend or "").strip().lower()
    return text in ("agg", "module://matplotlib.backends.backend_agg") or text.endswith(
        ".backend_agg"
    )


def _ascii_canvas_guard():
    try:
        from mpl_ascii.rendering.canvas import AsciiCanvas
    except Exception:
        return
    if getattr(AsciiCanvas, "_perf_bounds_guard", False):
        return

    def add(self, point, glyph):
        try:
            row, column = point.get_row_column_coords(self.height, self.width)
        except Exception:
            return self
        if 0 <= row < self.height and 0 <= column < self.width:
            self.array[row][column] = glyph
        return self

    AsciiCanvas.add = add
    AsciiCanvas._perf_bounds_guard = True


def _ascii_prepare(fig):
    _ascii_canvas_guard()
    import numpy as _np
    from matplotlib.collections import LineCollection as _LineCollection

    for ax in fig.axes:
        try:
            leg = ax.get_legend()
        except Exception:
            leg = None
        if leg is not None:
            try:
                leg.remove()
            except Exception:
                pass
        try:
            xlo, _ = ax.get_xlim()
            ylo, _ = ax.get_ylim()
        except Exception:
            xlo = ylo = 0.0
        for line in list(ax.lines):
            try:
                x = _np.asarray(line.get_xdata(), dtype=float)
                y = _np.asarray(line.get_ydata(), dtype=float)
            except Exception:
                continue
            if _np.isnan(x).any() or _np.isnan(y).any():
                try:
                    line.remove()
                except Exception:
                    pass
                continue
            if not _np.isfinite(x).all() or not _np.isfinite(y).all():
                x = _np.where(_np.isfinite(x), x, xlo)
                y = _np.where(_np.isfinite(y), y, ylo)
                try:
                    line.set_data(x, y)
                except Exception:
                    pass
        for coll in list(ax.collections):
            if isinstance(coll, _LineCollection):
                segments = []
                try:
                    raw = coll.get_segments()
                except Exception:
                    raw = []
                for seg in raw:
                    arr = _np.asarray(seg, dtype=float)
                    if len(arr) < 2 or not _np.isfinite(arr).all():
                        continue
                    if len(arr) == 2:
                        segments.append(arr)
                    else:
                        segments.extend(arr[i : i + 2] for i in range(len(arr) - 1))
                try:
                    if segments:
                        coll.set_segments(segments)
                    else:
                        coll.remove()
                except Exception:
                    pass
            elif hasattr(coll, "get_offsets"):
                try:
                    offsets = _np.asarray(coll.get_offsets(), dtype=float)
                    good = _np.isfinite(offsets).all(axis=1)
                    if good.any():
                        coll.set_offsets(offsets[good])
                        try:
                            if _np.asarray(coll.get_facecolor()).size == 0:
                                edge = _np.asarray(coll.get_edgecolor())
                                if edge.size:
                                    coll.set_facecolor(edge[:1])
                                else:
                                    coll.remove()
                        except Exception:
                            pass
                    else:
                        coll.remove()
                except Exception:
                    pass
        for patch in list(ax.patches):
            try:
                values = (
                    patch.get_x(),
                    patch.get_y(),
                    patch.get_width(),
                    patch.get_height(),
                )
            except Exception:
                continue
            if (
                not all(_np.isfinite(v) for v in values)
                or not values[2]
                or not values[3]
            ):
                try:
                    patch.remove()
                except Exception:
                    pass
    for leg in list(fig.legends):
        try:
            leg.remove()
        except Exception:
            pass


def _show_or_save(fig, output_path):
    if output_path is None:
        plt.show()
        return None
    return _save_fig(fig, output_path)


def _backend_name(mpl, backend):
    text = str(backend).strip()
    if text.lower() in ("window", "gui"):
        return (
            mpl.rcParamsDefault.get("backend") or mpl.rcParams.get("backend") or "agg"
        )
    return _BACKEND_ALIASES.get(text.lower(), text)


def _stdout_isatty():
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _module_available(name):
    try:
        import_module(name)
        return True
    except Exception:
        return False


def _da1_supports_sixel(data):
    for match in _DA1_RE.finditer(bytes(data or b"")):
        params = match.group(1).split(b";")
        if b"4" in params:
            return True
    return False


def _query_sixel_support(timeout=0.2):
    try:
        import select as _select
        import termios

        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except Exception:
        return False
    attrs = None
    try:
        attrs = termios.tcgetattr(fd)
        raw = list(attrs)
        raw[3] &= ~(termios.ICANON | termios.ECHO)
        raw[6][termios.VMIN] = 0
        raw[6][termios.VTIME] = 1
        termios.tcsetattr(fd, termios.TCSANOW, raw)
        os.write(fd, b"\x1b[c")
        deadline = time.monotonic() + max(0.01, float(timeout))
        data = b""
        while time.monotonic() < deadline:
            wait = max(0.0, deadline - time.monotonic())
            ready, _, _ = _select.select([fd], [], [], wait)
            if not ready:
                break
            try:
                chunk = os.read(fd, 1024)
            except BlockingIOError:
                continue
            if not chunk:
                break
            data += chunk
            if _DA1_RE.search(data):
                break
        return _da1_supports_sixel(data)
    except Exception:
        return False
    finally:
        if attrs is not None:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, attrs)
            except Exception:
                pass
        try:
            os.close(fd)
        except Exception:
            pass


def _terminal_identity():
    return (
        str(os.environ.get("TERM", "") or "").strip().lower(),
        str(os.environ.get("TERM_PROGRAM", "") or "").strip().lower(),
        str(os.environ.get("LC_TERMINAL", "") or "").strip().lower(),
        _stdout_isatty(),
    )


@functools.cache
def _terminal_capabilities(term, _program, _lc_terminal, tty):
    usable = bool(tty) or term not in ("", "dumb", "unknown")
    sixel = bool(usable and tty and _query_sixel_support())
    return {"sixel": sixel, "ascii": usable}


def _terminal_supports_sixel():
    return bool(_terminal_capabilities(*_terminal_identity()).get("sixel"))


def _terminal_supports_ascii():
    return bool(_terminal_capabilities(*_terminal_identity()).get("ascii"))


def _auto_backend(_mpl, current=None):
    if os.environ.get("MPLBACKEND"):
        return _mpl.get_backend()
    try:
        current = _mpl.get_backend() if current is None else current
    except Exception:
        current = ""
    if "inline" in str(current or "").lower():
        return current
    candidates = []
    if (
        _terminal_supports_sixel()
        and _module_available("matplotlib_backend_sixel")
        and shutil.which("convert")
    ):
        candidates.append("module://matplotlib_backend_sixel")
    if _terminal_supports_ascii() and _module_available("mpl_ascii"):
        candidates.append("module://mpl_ascii")
    candidates.append("Agg")
    for name in candidates:
        try:
            _mpl.use(name, force=True)
            return _mpl.get_backend()
        except Exception:
            continue
    try:
        return _mpl.get_backend()
    except Exception:
        return None


def _ensure_backend(backend=None):
    try:
        import matplotlib as _mpl
    except ImportError:
        return None
    text = "" if backend is None else str(backend).strip()
    if text.lower() not in _BACKEND_AUTO:
        try:
            _mpl.use(_backend_name(_mpl, text), force=True)
        except Exception as ex:
            raise ValueError(f"cannot use matplotlib backend {backend!r}: {ex}") from ex
        return _mpl.get_backend()
    return _auto_backend(_mpl)


sns = _LazyModule("seaborn")
plt = _LazyModule("matplotlib.pyplot")
