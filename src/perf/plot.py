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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CHART = {
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

_ERRORBAR_CHARTS = {"line", "bar", "point"}
_ERRORBAR_CAPS = 0.1
_LINE_ERR_CAPS = 3
_BAR_DOT_SIZE = 20
_DURATION_NS = 1e9


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


_DEFAULT_GROUPBY = ("file", "name", "mode")
_IMAGE_EXTS = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
_DEFAULT_RCPARAMS = {
    "figure.figsize": (10, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "gray",
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "axes.grid": True,
}
_UNSET = object()


class PlotConfig:
    def __init__(self):
        self._params = {}

    def __call__(self, *args, **kwargs):
        return plot(*args, **kwargs)

    @property
    def config(self):
        return dict(self._params)

    @config.setter
    def config(self, value):
        self._params = _plot_cfg(value)


def normalize_config(config):
    return _flatten_config("", config, {})


def _group_cell(v):
    try:
        import pandas as _pd

        if _pd.isna(v):
            return ""
    except Exception:
        pass
    try:
        s = str(v)
    except Exception:
        return ""
    return "" if s.strip().lower() in ("", "nan", "none", "nat") else s


def ensure_group(df, groupby=None):
    df = df.copy()
    df, keys = _resolve_group_keys(df, groupby)
    if not keys:
        return df, None
    if len(keys) == 1:
        col = keys[0]
        try:
            df[col] = [_group_cell(v) for v in df[col].tolist()]
        except Exception:
            pass
        return df, col
    label = "/".join(keys)
    if label not in df.columns:
        try:
            df[label] = df[keys].apply(
                lambda row: "/".join(s for s in (_group_cell(v) for v in row) if s),
                axis=1,
            )
        except Exception:
            df[label] = df[keys].astype(str).agg("/".join, axis=1)
    return df, label


def plot(
    df,
    types=None,
    event=None,
    groupby=None,
    x=None,
    output=None,
    logx=False,
    logy=False,
    **kwargs,
):
    kwargs.pop("output", None)
    type_groups = _split_groups(types, ["ecdf"])
    event_groups = _split_groups(event, ["duration_time"])
    for group in type_groups:
        for name in group:
            if name not in CHART:
                raise KeyError(
                    f"unknown chart {name!r}; available: {', '.join(sorted(CHART))}"
                )
    missing = [e for row in event_groups for e in row if e not in df.columns]
    if missing:
        raise KeyError(f"event(s) not found in data: {', '.join(missing)}")
    if x is not None:
        _xaxis(df, x)
        try:
            if isinstance(x, str) and x not in df.columns:
                try:
                    series = _resolve_x_series(df, x)
                except Exception:
                    series = None
                if series is not None:
                    try:
                        df = df.copy()
                        df[x] = pd.Series(list(series), index=df.index, name=x).values
                    except Exception:
                        pass
            if isinstance(x, str) and x in df.columns:
                df = df.sort_values(x, kind="stable").reset_index(drop=True)
        except Exception:
            pass
    df, label = ensure_group(df, groupby)
    duration_label = _duration_axis_label(df)
    if duration_label is not None:
        try:
            df = df.copy()
            df["duration_time"] = _duration_ns_values(df)
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
        ncols = len(event_groups)
        fig, axes = plt.subplots(1, ncols, squeeze=False)
        try:
            base_w, base_h = fig.get_size_inches()
        except Exception:
            base_w, base_h = (10, 5)
        fig.set_size_inches(max(base_w, 5 * ncols), base_h)
        flat = [ax for row in axes for ax in row]
        for ax, event_group in zip(flat, event_groups):
            plt.sca(ax)
            for ev in event_group:
                for name in type_group:
                    call_kw = dict(
                        _extra_for_event(df, label, hue_order, palette, ev, kwargs)
                    )
                    call_kw.update(kwargs)
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
                    if x is not None and "x" not in call_kw:
                        call_kw["x"] = x
                    CHART[name](df, ev, group=label, **call_kw)
                    if name == "bar":
                        _bar_mean_dots(ax)
            _style_axes(ax)
            _apply_duration_label(ax, event_group, duration_label)
            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            if isinstance(x, str):
                try:
                    ax.set_xlabel(x)
                except Exception:
                    pass
        nleg = _single_legend(fig, flat, label, hue_order)
        if not nleg:
            fig.tight_layout()
        res = _show_or_save(fig, output_path)
        if res is not None:
            saved.append(res)
    return saved


def _plot_cfg(config=None):
    flat = normalize_config(config)
    style = flat.pop("style", None)
    params = dict(_DEFAULT_RCPARAMS)
    params.update(flat)
    if style:
        plt.style.use(style)
    for key, value in params.items():
        if key == "figure.figsize" and isinstance(value, (list, tuple)):
            value = tuple(value)
            params[key] = value
        plt.rcParams[key] = value
    if style is not None:
        params["style"] = style
    return params


def _flatten_config(prefix, node, out):
    for key, value in dict(node or {}).items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_config(full, value, out)
        else:
            out[full] = value
    return out


def _first_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _canonical_data_col(col):
    if not isinstance(col, str) or not col.startswith("data."):
        return col
    rest = col[len("data.") :]
    if rest.startswith(("regs.", "mem.", "memory.")):
        return col
    try:
        from .bench import _canonical_data_reg_key as _canon

        canon = _canon(rest)
    except Exception:
        canon = str(rest).strip().lower()
    try:
        int(str(rest).strip(), 0)
        return col
    except (TypeError, ValueError):
        pass
    return f"data.{canon}"


def _alias_data_columns(df):
    try:
        from .arch import arch as _get_arch
    except Exception:
        _get_arch = None
    out = {}
    for col in list(getattr(df, "columns", []) or []):
        if not isinstance(col, str) or not col.startswith("data."):
            continue
        rest = col[len("data.") :]
        if rest.startswith(("regs.", "mem.", "memory.")):
            continue
        try:
            int(str(rest).strip(), 0)
            continue
        except (TypeError, ValueError):
            pass
        try:
            if _get_arch is not None:
                arch = _get_arch()
                keys_fn = getattr(arch, "data_reg_keys", None)
                keys = set(keys_fn(rest)) if keys_fn else set()
            else:
                keys = set()
        except Exception:
            keys = set()
        try:
            canon_col = _canonical_data_col(col)
        except Exception:
            canon_col = col
        keys.add(canon_col)
        try:
            from .bench import _canonical_data_reg_key as _canon

            canon = _canon(rest)
            if _get_arch is not None:
                alias_fn = getattr(_get_arch(), "arg_alias", None)
                if alias_fn is not None:
                    try:
                        alias = alias_fn(canon)
                    except Exception:
                        alias = None
                    if alias:
                        keys.add(f"data.{alias}")
                        keys.add(f"data.{str(alias).lower()}")
        except Exception:
            pass
        for k in keys:
            if k and k not in df.columns and k not in out:
                try:
                    out[k] = df[col]
                except Exception:
                    pass
    return out


def _resolve_x_series(df, x):
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
                    return pd.Series([cols[x]] * len(df), index=df.index, name=x)
                try:
                    v = _data_val(attrs.get("data"), x)
                except Exception:
                    v = None
                if v is not None:
                    return pd.Series([v] * len(df), index=df.index, name=x)
            if x.startswith("config."):
                from .bench import config_param_columns as _cfg_cols

                cols = _cfg_cols(attrs.get("config"))
                if x in cols:
                    return pd.Series([cols[x]] * len(df), index=df.index, name=x)
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
    raise KeyError(f"x column not found in data: {x!r}")


def _xaxis(df, x=None):
    if x is not None:
        try:
            series = _resolve_x_series(df, x)
        except KeyError:
            raise
        if series is not None:
            return series
    col = _first_col(df, "samples", "time")
    if col is not None:
        return df[col]
    if df.index.name:
        return df.index
    return range(len(df))


def _resolve_group_keys(df, groupby=None):
    if groupby is None:
        groupby = [
            c
            for c in _DEFAULT_GROUPBY
            if c in df.columns or c in (df.index.names or [])
        ]
    if isinstance(groupby, str):
        groupby = [k.strip() for k in groupby.split(",") if k.strip()]
    keys = [c for c in groupby if c in df.columns]
    if not keys and groupby:
        if df.index.name in groupby or any(
            k in (df.index.names or []) for k in groupby
        ):
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


def _split_groups(value, default):
    from .core import _split_groups as _core_split_groups

    return _core_split_groups(value, default)


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


def _extra_for_event(df, label, hue_order, palette, event, kwargs=None):
    kwargs = kwargs or {}
    ho = _visible_order(df, label, hue_order, event)
    if "hue_order" in kwargs:
        ho = kwargs["hue_order"]
    elif ho is not None and len(ho) == 0:
        ho = hue_order
    if "palette" in kwargs:
        pal = kwargs["palette"]
    elif palette is not None and ho is not None:
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


def _duration_axis_label(df):
    try:
        cols = list(df.columns)
    except Exception:
        return None
    if "duration_time" not in cols:
        return None
    try:
        ops = df["operations"]
    except Exception:
        return "duration_time[ns]"
    try:
        import pandas as _pd

        vals = _pd.to_numeric(ops, errors="coerce")
        vals = vals.dropna()
        if len(vals) and bool((vals != 1).any()):
            return "duration_time[ns]/operations"
    except Exception:
        pass
    return "duration_time[ns]"


def _duration_ns_values(df):
    import pandas as _pd

    vals = _pd.to_numeric(df["duration_time"], errors="coerce") * _DURATION_NS
    try:
        ops = _pd.to_numeric(df["operations"], errors="coerce")
    except Exception:
        return vals
    try:
        mask = ops.notna() & (ops != 0)
        vals = vals.copy()
        vals[mask] = vals[mask] / ops[mask]
    except Exception:
        pass
    return vals


def _apply_duration_label(ax, event_group, duration_label):
    if not duration_label:
        return
    try:
        if "duration_time" not in (event_group or []):
            return
    except Exception:
        return
    try:
        if ax.get_xlabel() == "duration_time":
            ax.set_xlabel(duration_label)
    except Exception:
        pass
    try:
        if ax.get_ylabel() == "duration_time":
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


def _layout_with_legend(fig, leg, right):
    try:
        fig.tight_layout(rect=[0, 0, right, 1])
    except Exception:
        pass
    try:
        for _ in range(3):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            leg_bb = leg.get_window_extent(renderer)
            fig_bb = fig.get_window_extent(renderer)
            overflow = float(leg_bb.x1 - fig_bb.x1)
            if overflow > 1:
                extra = overflow / float(fig_bb.width) + 0.01
                new_right = max(0.55, right - extra)
            else:
                slack = float(fig_bb.x1 - leg_bb.x1)
                if slack > 20:
                    gain = (slack - 10) / float(fig_bb.width)
                    new_right = min(0.9, right + gain)
                else:
                    break
            if abs(new_right - right) < 0.005:
                right = new_right
                break
            right = new_right
            try:
                leg.set_bbox_to_anchor((right, 1.0))
            except Exception:
                pass
            try:
                fig.tight_layout(rect=[0, 0, right, 1])
            except Exception:
                pass
        try:
            fig.subplots_adjust(right=right)
        except Exception:
            pass
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            leg_bb = leg.get_window_extent(renderer)
            fig_bb = fig.get_window_extent(renderer)
            overflow = float(leg_bb.x1 - fig_bb.x1)
            if overflow > 1:
                right = max(0.55, right - overflow / float(fig_bb.width) - 0.01)
                try:
                    leg.set_bbox_to_anchor((right, 1.0))
                except Exception:
                    pass
                try:
                    fig.subplots_adjust(right=right)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overflow = float(
            leg.get_window_extent(renderer).x1 - fig.get_window_extent(renderer).x1
        )
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


def _move_legend_below(fig, handles, labels):
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
        ncol = min(n, 4)
    ncol = max(1, ncol)
    while ncol > 1:
        total = ncol * col_w + (ncol - 1) * 20
        if total <= fig_w_px * 0.96:
            break
        ncol -= 1
    rows_needed = (n + ncol - 1) // ncol
    bottom = min(0.04 + 0.06 * rows_needed, 0.35)
    try:
        for old in list(fig.legends):
            try:
                old.remove()
            except Exception:
                pass
    except Exception:
        pass
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, bottom + 0.02),
        **_legend_kwargs(ncol=ncol),
    )
    try:
        fig.tight_layout(rect=[0, bottom, 1, 1])
    except Exception:
        pass
    try:
        fig.subplots_adjust(bottom=bottom)
    except Exception:
        pass
    return leg


def _single_legend(fig, axes, label, hue_order=None):
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


def _show_or_save(fig, output_path):
    if output_path is None:
        plt.show()
        return None
    return _save_fig(fig, output_path)
