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

from .arch import arch
from .core import _split_groups, _text

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
_BACKEND_ALIASES = {
    "ascii": "module://mpl_ascii",
    "sixel": "module://matplotlib_backend_sixel",
}
_BACKEND_AUTO = ("", "auto", "default")
_DA1_RE = re.compile(rb"\x1b\[\?([0-9;]*)c")
_UNSET = object()
_PLOT_LOCK = threading.RLock()


def _serialized(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _PLOT_LOCK:
            return fn(*args, **kwargs)

    return wrapper


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
            except Exception:
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
    requested_backend = _config_backend(config)
    backend = _ensure_backend(requested_backend)
    auto_backend = requested_backend is None or str(
        requested_backend
    ).strip().lower() in (_BACKEND_AUTO)
    if auto_backend and output is not None:
        backend = _ensure_backend("Agg")
    ascii_backend = _is_ascii_backend(backend)
    logx, logy = _plot_options(config)
    if ascii_backend:
        logx = logy = False
    type_groups = _split_groups(type, ["ecdf"])
    if event is None and "duration_time" not in df.columns:
        try:
            from .comp import metric_columns as _metrics

            fallback = _metrics(df)
            if fallback:
                event = [[c] for c in fallback]
        except Exception:
            pass
    event_groups = _split_groups(event, ["duration_time"])
    for group in type_groups:
        for name in group:
            if name not in _CHART:
                raise KeyError(
                    f"unknown chart {name!r}; available: {', '.join(sorted(_CHART))}"
                )
    missing = [e for row in event_groups for e in row if e not in df.columns]
    if missing:
        raise KeyError(f"event(s) not found in data: {', '.join(missing)}")
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
    try:
        flat_types = {t for grp in type_groups for t in (grp or [])}
        if "swarm" in flat_types:
            _cap = 1000
        elif "strip" in flat_types:
            _cap = 2000
        elif flat_types & {"violin", "box", "boxen"}:
            _cap = 5000
        else:
            _cap = 10000
        if len(df) > _cap:
            if label is not None and label in df.columns:
                try:
                    ngroups = max(1, int(df[label].nunique()))
                except Exception:
                    ngroups = 1
                try:
                    quota = max(1, _cap // ngroups)
                    parts = []
                    for _, part in df.groupby(label, dropna=False):
                        if len(part) > quota:
                            part = part.sample(n=quota, random_state=0)
                        parts.append(part)
                    df = pd.concat(parts, ignore_index=True)
                    if len(df) > _cap:
                        df = df.sample(n=_cap, random_state=0).reset_index(drop=True)
                except Exception:
                    try:
                        df = df.sample(n=_cap, random_state=0).reset_index(drop=True)
                    except Exception:
                        pass
            else:
                try:
                    df = df.sample(n=_cap, random_state=0).reset_index(drop=True)
                except Exception:
                    pass
    except Exception:
        pass
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
    saved = []
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
    return f"data.{arch.canonical_data_reg(rest)}"


def _alias_data_columns(df):
    out = {}
    for col in list(getattr(df, "columns", []) or []):
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
                from .bench import data_param_columns as _data_cols
                from .bench import data_param_value as _data_val

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
                from .bench import config_param_columns as _cfg_cols

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


def _dist(df, event, fn, group=_UNSET, palette=None, hue_order=None, x=None, **kw):
    kw.pop("x", None)
    group = _default_group(df, group)
    return fn(data=df, x=event, **_hue_kw(group, palette, hue_order, **kw))


def _cat(df, event, fn, group=_UNSET, palette=None, hue_order=None, x=None, **kw):
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
            pal = {k: v for k, v in palette.items() if k in set(ho)}
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
        try:
            fig.canvas.draw()
        except Exception:
            pass
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


def _ascii_prepare(fig):
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
def _terminal_capabilities(term, program, lc_terminal, tty):
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
