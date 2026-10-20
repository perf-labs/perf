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

import numpy as np
import pandas as pd

from .core import _split_list, _text
from .data import _NON_METRIC_COLUMNS, quote_columns

_DEFAULT_ON = ["name", "mode"]
_SOURCE_COL = "__source__"
_DEFAULT_ALPHA = 0.05
_DATA_PREFIX = "data."
_MISSING_SENTINEL = "\x00__perf_missing__\x00"


def norm_cdf(z):
    try:
        z = float(z)
    except (TypeError, ValueError):
        return float("nan")
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@functools.cache
def z_critical(alpha=_DEFAULT_ALPHA):
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return _ndtri(1.0 - alpha / 2.0)


def ztest(a, b, alpha=_DEFAULT_ALPHA):
    sa, sb = _summarize(a), _summarize(b)
    r = _ztest_from_stats(sa["mean"], sa["se"], sb["mean"], sb["se"])
    try:
        diff_pct = r["diff"] / sa["mean"] * 100.0 if sa["mean"] else float("nan")
    except (TypeError, ZeroDivisionError):
        diff_pct = float("nan")
    try:
        significant = bool(r["p_value"] < float(alpha))
    except (TypeError, ValueError):
        significant = False
    ci_low, ci_high = _diff_ci(r["diff"], r["se_diff"], alpha)
    return {
        "n_a": sa["n"],
        "n_b": sb["n"],
        "mean_a": sa["mean"],
        "mean_b": sb["mean"],
        "std_a": sa["std"],
        "std_b": sb["std"],
        "se_a": sa["se"],
        "se_b": sb["se"],
        "diff": r["diff"],
        "diff_pct": diff_pct,
        "se_diff": r["se_diff"],
        "z": r["z"],
        "p_value": r["p_value"],
        "significant": significant,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def gmean_ztest(a, b, alpha=_DEFAULT_ALPHA):
    xa, xb = _clean(a), _clean(b)
    if xa.size == 0 or xb.size == 0 or np.any(xa <= 0) or np.any(xb <= 0):
        return {
            "n_a": int(xa.size),
            "n_b": int(xb.size),
            "gmean_a": float("nan"),
            "gmean_b": float("nan"),
            "diff_pct": float("nan"),
            "z": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    la, lb = np.log(xa), np.log(xb)
    sa, sb = _summarize(la), _summarize(lb)
    r = _ztest_from_stats(sa["mean"], sa["se"], sb["mean"], sb["se"])
    try:
        gmean_a = float(math.exp(sa["mean"]))
        gmean_b = float(math.exp(sb["mean"]))
        diff_pct = (gmean_b / gmean_a - 1.0) * 100.0 if gmean_a else float("nan")
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        gmean_a, gmean_b, diff_pct = float("nan"), float("nan"), float("nan")
    try:
        significant = bool(r["p_value"] < float(alpha))
    except (TypeError, ValueError):
        significant = False
    ci_low, ci_high = _pct_ci_from_log(r["diff"], r["se_diff"], alpha)
    return {
        "n_a": sa["n"],
        "n_b": sb["n"],
        "gmean_a": gmean_a,
        "gmean_b": gmean_b,
        "diff_pct": diff_pct,
        "z": r["z"],
        "p_value": r["p_value"],
        "significant": significant,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def clt_test(a, b, alpha=_DEFAULT_ALPHA):
    m = ztest(a, b, alpha=alpha)
    g = gmean_ztest(a, b, alpha=alpha)
    p_mean, p_geo = m["p_value"], g["p_value"]
    try:
        if math.isfinite(p_mean) and math.isfinite(p_geo):
            p = max(float(p_mean), float(p_geo))
        elif math.isfinite(p_mean):
            p = float(p_mean)
        elif math.isfinite(p_geo):
            p = float(p_geo)
        else:
            p = float("nan")
    except (TypeError, ValueError):
        p = m["p_value"]
    try:
        diff = g["diff_pct"] if math.isfinite(g["diff_pct"]) else m["diff_pct"]
    except (TypeError, ValueError):
        diff = m["diff_pct"]
    try:
        if math.isfinite(g.get("ci_low", float("nan"))) and math.isfinite(
            g.get("ci_high", float("nan"))
        ):
            ci_low, ci_high = g["ci_low"], g["ci_high"]
        else:
            lo, hi = _diff_ci(m["diff"], m["se_diff"], alpha)
            mean_a = m.get("mean_a", float("nan"))
            try:
                if mean_a and math.isfinite(lo) and math.isfinite(hi):
                    lo, hi = lo / mean_a * 100.0, hi / mean_a * 100.0
                    ci_low, ci_high = (lo, hi) if lo <= hi else (hi, lo)
                else:
                    ci_low, ci_high = float("nan"), float("nan")
            except (TypeError, ValueError, ZeroDivisionError):
                ci_low, ci_high = float("nan"), float("nan")
    except (TypeError, ValueError):
        ci_low, ci_high = float("nan"), float("nan")
    try:
        significant = bool(p < float(alpha))
    except (TypeError, ValueError):
        significant = False
    out = dict(m)
    out["diff_pct"] = diff
    out["diff"] = diff
    out["p_value"] = p
    out["p_mean"] = p_mean
    out["p_gmean"] = p_geo
    out["significant"] = significant
    out["ci_low"] = ci_low
    out["ci_high"] = ci_high
    return out


def _detect_variant(df, source_col=_SOURCE_COL):
    try:
        n_names = df["name"].nunique(dropna=False) if "name" in df.columns else 0
    except Exception:
        n_names = 0
    try:
        n_files = df["file"].nunique(dropna=False) if "file" in df.columns else 0
    except Exception:
        n_files = 0
    try:
        n_src = df[source_col].nunique(dropna=False) if source_col in df.columns else 1
    except Exception:
        n_src = 1
    if n_names >= 2:
        try:
            key = df["name"].astype(str)
            run = key.copy()
            if "file" in df.columns:
                run = run + "\x00" + df["file"].astype(str)
            if source_col in df.columns:
                run = run + "\x00" + df[source_col].astype(str)
            nested = run.groupby(key, dropna=False).nunique(dropna=False) <= 1
            if bool(nested.all()):
                return "name"
        except Exception:
            pass
    if n_files >= 2:
        return "file"
    if n_src >= 2:
        return source_col
    return "name"


def metric_columns(df):
    columns = []
    for column in df.columns:
        if str(column).startswith(("data.", "config.")):
            continue
        if column in _NON_METRIC_COLUMNS:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[column]):
                columns.append(column)
        except Exception:
            continue
    return columns


def derived_metrics(df):
    try:
        numeric = {c: pd.api.types.is_numeric_dtype(df[c]) for c in df.columns}
    except Exception:
        return []
    derived = []
    if numeric.get("operations"):
        for column in metric_columns(df):
            if numeric.get(column):
                derived.append(f"{column}/operations")
    if numeric.get("threads"):
        for column in metric_columns(df):
            if numeric.get(column):
                derived.append(f"{column}/threads")
    if numeric.get("operations") and numeric.get("threads"):
        for column in metric_columns(df):
            if numeric.get(column):
                derived.append(f"{column}/operations/threads")
    if numeric.get("cycles") and numeric.get("instructions"):
        derived.append("instructions/cycles")
    return derived


def _per_operation(df):
    try:
        derived = derived_metrics(df)
    except Exception:
        return None
    try:
        threads = pd.to_numeric(df.get("threads"), errors="coerce")
        per_thread = bool(len(threads.dropna()) and (threads.dropna() != 1).any())
    except Exception:
        per_thread = False
    if per_thread:
        threaded = [d for d in derived if str(d).endswith("/operations/threads")]
        if threaded:
            return threaded
        threaded = [d for d in derived if str(d).endswith("/threads")]
        if threaded:
            return threaded
    return [d for d in derived if str(d).endswith("/operations")] or None


def _eval_events(df, events):
    try:
        df = df.copy()
    except Exception:
        pass
    if events is None:
        wanted = []
    else:
        try:
            wanted = list(events)
        except TypeError:
            wanted = [events]
    if not wanted:
        try:
            return df, metric_columns(df) + derived_metrics(df)
        except Exception:
            return df, metric_columns(df)
    out = []
    for name in wanted:
        if name in df.columns:
            out.append(name)
            continue
        try:
            expr = quote_columns(df, name)
        except Exception:
            continue
        try:
            df[name] = df.eval(expr, engine="python")
        except Exception:
            continue
        out.append(name)
    return df, out


def compare(
    df,
    event=None,
    baseline=None,
    alpha=_DEFAULT_ALPHA,
    on=None,
    variant=None,
    filter=None,
):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("compare expects a pandas DataFrame")
    if filter:
        from .data import query as _query

        df = _query(df, filter)
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    rec = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
    if variant is None:
        variant = _detect_variant(rec)
    keys = [c for c in _resolve_on(rec, on) if c != variant]
    if variant not in rec.columns:
        raise ValueError(f"variant column {variant!r} not found in data")
    if event is None:
        requested = _per_operation(rec)
        if not requested and "sym" in rec.columns:
            try:
                requested = [
                    c for c in metric_columns(rec) if c not in ("pid", "ip")
                ] or None
            except Exception:
                requested = None
    else:
        try:
            requested = _split_list(event, expand=False)
        except Exception:
            requested = [event] if isinstance(event, str) else list(event or [])
    rec, events = _eval_events(rec, requested)
    events = [
        e
        for e in (events or [])
        if e not in ("iterations", "operations", "threads", "tid")
    ]
    if not events:
        raise ValueError("no metric columns found in data")

    def _label(v):
        try:
            return _text(v) or str(v)
        except Exception:
            return str(v)

    all_variants = {_label(v): True for v in rec[variant].tolist()}
    base_label = _resolve_baseline(all_variants, baseline)
    if "mode" in keys and "mode" in rec.columns:
        modes = list(pd.unique(rec["mode"].dropna()))
        nulls = rec["mode"].isna()
        if modes and bool(nulls.any()):
            rec = rec.copy()
            wildcard = rec[nulls]
            rec = pd.concat(
                [rec] + [wildcard.assign(mode=m) for m in modes], ignore_index=True
            )
    group_keys = list(keys)
    if "name" in group_keys and "name" in rec.columns and variant != "name":
        try:
            rec = rec.copy()
            rec["_group_name"] = [
                _strip_run_id(_text(v) or str(v)) for v in rec["name"].tolist()
            ]
            group_keys = ["_group_name" if k == "name" else k for k in group_keys]
        except Exception:
            group_keys = list(keys)
    groups = (
        [rec]
        if not group_keys
        else [frame for _, frame in rec.groupby(group_keys, dropna=False)]
    )
    try:
        strata_cols = _data_columns(rec, exclude=list(group_keys) + [variant])
    except Exception:
        strata_cols = []
    try:
        strata_cols = sorted(set(strata_cols))
    except Exception:
        pass
    rows = []
    for sub in groups:
        if group_keys:
            try:
                first = sub.iloc[0]
                match = {}
                for k in group_keys:
                    if k == "_group_name":
                        match["name"] = _text(first[k])
                    else:
                        match[k] = _text(first[k])
            except Exception:
                match = {}
        else:
            match = {}
        variants = {_label(v): frame for v, frame in sub.groupby(variant, dropna=False)}
        if base_label not in variants:
            continue
        if len(variants) < 2:
            continue
        base = variants[base_label]
        for label in sorted(variants):
            if label == base_label:
                continue
            frame = variants[label]
            if "mode" in rec.columns and "mode" not in keys:
                try:
                    base_modes = set(base["mode"].dropna().astype(str).tolist())
                    frame_modes = set(frame["mode"].dropna().astype(str).tolist())
                except Exception:
                    base_modes = set()
                    frame_modes = set()
                shared = base_modes & frame_modes
                if not shared and (base_modes or frame_modes):
                    continue
                pairs = []
                for m in sorted(shared) if shared else [None]:
                    try:
                        if m is None:
                            pairs.append((m, base, frame))
                        else:
                            pairs.append(
                                (
                                    m,
                                    base[base["mode"].astype(str) == m],
                                    frame[frame["mode"].astype(str) == m],
                                )
                            )
                    except Exception:
                        continue
            else:
                pairs = [(None, base, frame)]
            try:
                sliced = []
                for m, base_m, frame_m in pairs:
                    try:
                        s = (
                            _stratified_groups(base_m, frame_m, strata_cols)
                            if strata_cols
                            else []
                        )
                    except Exception:
                        s = []
                    sliced.append((m, base_m, frame_m, s))
            except Exception:
                sliced = [(m, base_m, frame_m, []) for m, base_m, frame_m in pairs]
            for m, base_m, frame_m, strata in sliced:
                mode_out = match.get("mode", m if m is not None else "")
                for event in events:
                    if event not in base_m.columns or event not in frame_m.columns:
                        continue
                    try:
                        b_vals = base_m[event]
                        f_vals = frame_m[event]
                    except Exception:
                        continue
                    if not _can_compare(b_vals, f_vals):
                        continue
                    try:
                        r = clt_test(b_vals, f_vals, alpha=alpha)
                    except Exception:
                        continue
                    try:
                        if not math.isfinite(float(r["p_value"])):
                            continue
                    except (TypeError, ValueError):
                        continue
                    rows.append(
                        {
                            **match,
                            "event": event,
                            "baseline": base_label,
                            "challenger": label,
                            "mode": mode_out,
                            "ci_low": r["ci_low"],
                            "ci_high": r["ci_high"],
                            "p_value": r["p_value"],
                            "alpha": alpha,
                            "significant": bool(r["significant"]),
                            "diff": r["diff"],
                        }
                    )
                    for data_vals, b_sub, f_sub in strata:
                        try:
                            bb = b_sub[event]
                            ff = f_sub[event]
                        except Exception:
                            continue
                        if not _can_compare(bb, ff):
                            continue
                        try:
                            rd = clt_test(bb, ff, alpha=alpha)
                        except Exception:
                            continue
                        try:
                            if not math.isfinite(float(rd["p_value"])):
                                continue
                        except (TypeError, ValueError):
                            continue
                        rows.append(
                            {
                                **match,
                                **dict(data_vals),
                                "event": event,
                                "baseline": base_label,
                                "challenger": label,
                                "mode": mode_out,
                                "ci_low": rd["ci_low"],
                                "ci_high": rd["ci_high"],
                                "p_value": rd["p_value"],
                                "alpha": alpha,
                                "significant": bool(rd["significant"]),
                                "diff": rd["diff"],
                            }
                        )
    if not rows:
        raise ValueError(
            f"no comparable groups: need at least 2 {variant} variants sharing "
            f"the same group (baseline {base_label!r} never co-occurs "
            "with another variant)"
        )
    try:
        effective_strata = []
        for c in strata_cols:
            try:
                seen = set()
                for r in rows:
                    try:
                        v = r.get(c)
                    except Exception:
                        continue
                    if v is None:
                        continue
                    try:
                        if pd.isna(v):
                            continue
                    except Exception:
                        pass
                    try:
                        seen.add(_hashable_data_value(v))
                    except Exception:
                        try:
                            seen.add(str(v))
                        except Exception:
                            continue
                    if len(seen) > 1:
                        break
                if len(seen) > 1:
                    effective_strata.append(c)
            except Exception:
                continue
    except Exception:
        effective_strata = list(strata_cols)
    try:
        reserved = {
            "baseline",
            "challenger",
            "mode",
            "event",
            "diff",
            "ci_low",
            "ci_high",
            "p_value",
            "alpha",
            "significant",
            *effective_strata,
        }
        cand = []
        for r in rows:
            try:
                keys_r = list(r.keys())
            except Exception:
                continue
            for k in keys_r:
                if k in reserved or k in cand:
                    continue
                cand.append(k)
        order = ["file", "name"]
        leading = [c for c in order if c in cand]
        leading += [c for c in cand if c not in order]
        effective_leading = []
        for c in leading:
            try:
                for r in rows:
                    try:
                        v = r.get(c)
                    except Exception:
                        continue
                    if v is None:
                        continue
                    try:
                        if pd.isna(v):
                            continue
                    except Exception:
                        pass
                    try:
                        if str(v).strip() == "":
                            continue
                    except Exception:
                        pass
                    effective_leading.append(c)
                    break
            except Exception:
                continue
    except Exception:
        effective_leading = []
    cols = [
        *effective_leading,
        "baseline",
        "challenger",
        "mode",
        "event",
        *effective_strata,
        "ci_low",
        "ci_high",
        "p_value",
        "alpha",
        "significant",
        "diff",
    ]
    out = pd.DataFrame(rows, columns=cols)
    out.attrs["alpha"] = alpha
    out.attrs["on"] = keys
    out.attrs["variant"] = variant
    return out


@functools.cache
def _ndtri(p):
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p!r}")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def _clean(values):
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    try:
        arr = arr[np.isfinite(arr)]
    except Exception:
        return np.asarray([], dtype=float)
    return arr


def _summarize(values):
    arr = _clean(values)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "se": float("nan")}
    mean = float(arr.mean())
    if n < 2:
        return {"n": n, "mean": mean, "std": float("nan"), "se": float("nan")}
    std = float(arr.std(ddof=1))
    if std != std:
        std = 0.0
    se = float(std / math.sqrt(n))
    return {"n": n, "mean": mean, "std": std, "se": se}


def _ztest_from_stats(mean_a, se_a, mean_b, se_b):
    try:
        diff = float(mean_b - mean_a)
    except (TypeError, ValueError):
        diff = float("nan")
    try:
        se_diff = math.sqrt(se_a**2 + se_b**2)
    except (TypeError, ValueError):
        se_diff = float("nan")
    if se_diff != se_diff:
        se_diff = float("nan")
    if se_diff == 0.0:
        if diff == 0.0 or (diff != diff):
            return {"diff": diff, "se_diff": 0.0, "z": 0.0, "p_value": 1.0}
        return {
            "diff": diff,
            "se_diff": 0.0,
            "z": math.inf if diff > 0 else -math.inf,
            "p_value": 0.0,
        }
    if not math.isfinite(se_diff) or diff != diff:
        return {
            "diff": diff,
            "se_diff": se_diff,
            "z": float("nan"),
            "p_value": float("nan"),
        }
    z = diff / se_diff
    p = 2.0 * (1.0 - norm_cdf(abs(z)))
    return {
        "diff": diff,
        "se_diff": se_diff,
        "z": z,
        "p_value": min(max(p, 0.0), 1.0),
    }


def _diff_ci(diff, se_diff, alpha=_DEFAULT_ALPHA):
    try:
        d = float(diff)
        se = float(se_diff)
        z = z_critical(alpha)
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if d != d or se != se or not math.isfinite(se) or se < 0.0:
        return float("nan"), float("nan")
    if se == 0.0:
        return d, d
    try:
        return d - z * se, d + z * se
    except (TypeError, ValueError, OverflowError):
        return float("nan"), float("nan")


def _pct_ci_from_log(diff_log, se_log, alpha=_DEFAULT_ALPHA):
    try:
        d = float(diff_log)
        se = float(se_log)
        z = z_critical(alpha)
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if d != d or se != se or not math.isfinite(se) or se < 0.0:
        return float("nan"), float("nan")
    try:
        if se == 0.0:
            pct = (math.exp(d) - 1.0) * 100.0
            return pct, pct
        lo = (math.exp(d - z * se) - 1.0) * 100.0
        hi = (math.exp(d + z * se) - 1.0) * 100.0
    except (TypeError, ValueError, OverflowError):
        return float("nan"), float("nan")
    if lo != lo or hi != hi:
        return float("nan"), float("nan")
    return (lo, hi) if lo <= hi else (hi, lo)


def _resolve_on(df, on=None):
    if on is None:
        on = list(_DEFAULT_ON)
    if isinstance(on, str):
        on = [k.strip() for k in on.split(",") if k.strip()]
    rec = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
    return [c for c in on if c in rec.columns]


def _resolve_baseline(variants, baseline=None):
    labels = sorted(str(v) for v in variants)
    if not labels:
        raise ValueError("no variants found in data")
    if baseline is None:
        return labels[0]
    want = str(baseline)
    if want in variants:
        return want
    try:
        from pathlib import Path as _Path

        cands = [v for v in variants if _Path(str(v)).name == _Path(want).name]
    except Exception:
        cands = []
    if len(cands) == 1:
        return cands[0]
    stripped = [v for v in variants if _strip_run_id(str(v)) == want]
    if len(stripped) == 1:
        return stripped[0]
    raise ValueError(f"baseline {baseline!r} not found; available: {', '.join(labels)}")


def _strip_run_id(name):
    head, _, tail = str(name).rpartition("-")
    if head and len(tail) == 8:
        try:
            int(tail, 16)
        except ValueError:
            return str(name)
        return head
    return str(name)


def _can_compare(a, b):
    try:
        return _clean(a).size > 0 and _clean(b).size > 0
    except Exception:
        return False


def _data_columns(df, exclude=()):
    try:
        cols = [c for c in df.columns if str(c).startswith(_DATA_PREFIX)]
    except Exception:
        return []
    skip = set(exclude or ())
    return [c for c in cols if c not in skip]


def _hashable_data_value(value):
    if isinstance(value, (list, tuple)):
        try:
            return tuple(_hashable_data_value(v) for v in value)
        except Exception:
            return str(value)
    if isinstance(value, dict):
        try:
            return tuple(
                sorted((str(k), _hashable_data_value(v)) for k, v in value.items())
            )
        except Exception:
            return str(value)
    if isinstance(value, np.ndarray):
        try:
            return tuple(_hashable_data_value(v) for v in value.tolist())
        except Exception:
            return str(value)
    try:
        if value is None:
            return _MISSING_SENTINEL
    except Exception:
        pass
    try:
        if pd.isna(value):
            return _MISSING_SENTINEL
    except Exception:
        pass
    try:
        hash(value)
        return value
    except TypeError:
        return str(value)
    except Exception:
        return str(value)


def _stratified_groups(base_df, frame_df, data_cols, min_n=2):
    if not data_cols:
        return []
    try:
        cols = [c for c in data_cols if c in base_df.columns and c in frame_df.columns]
    except Exception:
        return []
    if not cols:
        return []
    try:
        base_keys = {}
        frame_keys = {}
        for col in cols:
            try:
                base_keys[col] = base_df[col].map(_hashable_data_value)
            except Exception:
                try:
                    base_keys[col] = base_df[col].astype(str)
                except Exception:
                    return []
            try:
                frame_keys[col] = frame_df[col].map(_hashable_data_value)
            except Exception:
                try:
                    frame_keys[col] = frame_df[col].astype(str)
                except Exception:
                    return []
    except Exception:
        return []
    try:
        base_tuples = list(zip(*[base_keys[c].tolist() for c in cols]))
        frame_tuples = list(zip(*[frame_keys[c].tolist() for c in cols]))
    except Exception:
        return []
    if not base_tuples or not frame_tuples:
        return []
    try:
        base_set = set(base_tuples)
        frame_set = set(frame_tuples)
    except Exception:
        return []
    if len(base_set) <= 1 and len(frame_set) <= 1:
        return []
    common = base_set & frame_set
    if not common:
        return []
    try:
        base_index = {}
        for i, key in enumerate(base_tuples):
            base_index.setdefault(key, []).append(i)
        frame_index = {}
        for i, key in enumerate(frame_tuples):
            frame_index.setdefault(key, []).append(i)
    except Exception:
        return []
    try:
        base_rows = base_df.reset_index(drop=True)
        frame_rows = frame_df.reset_index(drop=True)
    except Exception:
        return []
    out = []
    try:
        ordered = sorted(common)
    except Exception:
        ordered = sorted(common, key=repr)
    for key in ordered:
        try:
            b_ilocs = base_index.get(key, [])
            f_ilocs = frame_index.get(key, [])
            if len(b_ilocs) < min_n or len(f_ilocs) < min_n:
                continue
            b_sub = base_rows.iloc[b_ilocs]
            f_sub = frame_rows.iloc[f_ilocs]
            try:
                first = b_sub.iloc[0]
                data_vals = {c: first[c] for c in cols}
            except Exception:
                try:
                    first = f_sub.iloc[0]
                    data_vals = {c: first[c] for c in cols}
                except Exception:
                    continue
            out.append((data_vals, b_sub, f_sub))
        except Exception:
            continue
    return out
