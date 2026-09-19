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

from .core import _text

DEFAULT_ON = ["name", "mode"]
VARIANT = "name"
DEFAULT_ALPHA = 0.05
NON_METRIC_COLUMNS = frozenset(
    {
        "file",
        "name",
        "mode",
        "samples",
        "iterations",
        "operations",
        "time",
        "address",
        "size",
        "pid",
        "ip",
    }
)


def norm_cdf(z):
    try:
        z = float(z)
    except (TypeError, ValueError):
        return float("nan")
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


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


@functools.cache
def z_critical(alpha=DEFAULT_ALPHA):
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return _ndtri(1.0 - alpha / 2.0)


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
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if std != std:
        std = 0.0
    se = float(std / math.sqrt(n)) if n > 0 else float("nan")
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


def _diff_ci(diff, se_diff, alpha=DEFAULT_ALPHA):
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


def _pct_ci_from_log(diff_log, se_log, alpha=DEFAULT_ALPHA):
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


def ztest(a, b, alpha=DEFAULT_ALPHA):
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


def gmean_ztest(a, b, alpha=DEFAULT_ALPHA):
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


def clt_test(a, b, alpha=DEFAULT_ALPHA):
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


def _result_label(significant, diff_pct):
    try:
        significant = bool(significant)
    except Exception:
        significant = False
    if not significant:
        return "insignificant"
    try:
        d = float(diff_pct)
    except (TypeError, ValueError):
        return "insignificant"
    if d != d or d in (float("inf"), float("-inf")):
        try:
            import math as _math

            if not _math.isfinite(d):
                return "insignificant"
        except Exception:
            return "insignificant"
    if d < 0:
        return f"faster ({d:+.2f}%)"
    if d > 0:
        return f"slower ({d:+.2f}%)"
    return "insignificant"


def _resolve_on(df, on=None):
    if on is None:
        on = list(DEFAULT_ON)
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
    raise ValueError(f"baseline {baseline!r} not found; available: {', '.join(labels)}")


def _finite_values(values):
    return _clean(values)


def _can_compare(a, b):
    try:
        return _finite_values(a).size > 0 and _finite_values(b).size > 0
    except Exception:
        return False


def compare(
    df, events=None, baseline=None, alpha=DEFAULT_ALPHA, on=None, variant=VARIANT
):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("compare expects a pandas DataFrame")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    rec = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
    keys = [c for c in _resolve_on(rec, on) if c != variant]
    if variant not in rec.columns:
        raise ValueError(f"variant column {variant!r} not found in data")
    if events is None:
        events = [
            c
            for c in rec.columns
            if not str(c).startswith(("data.", "config."))
            and c not in NON_METRIC_COLUMNS
            and pd.api.types.is_numeric_dtype(rec[c])
        ]
    if isinstance(events, str):
        events = [events]

    events = [
        e
        for e in (events or [])
        if e in rec.columns and e not in ("iterations", "samples", "operations")
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
    groups = (
        [rec] if not keys else [frame for _, frame in rec.groupby(keys, dropna=False)]
    )
    rows = []
    for sub in groups:
        if keys:
            try:
                first = sub.iloc[0]
                match = {k: _text(first[k]) for k in keys}
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
            for event in events:
                if event not in base.columns or event not in frame.columns:
                    continue
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
                    modes_to_do = sorted(shared) if shared else [None]
                    for m in modes_to_do:
                        try:
                            if m is None:
                                b_vals = base[event]
                                f_vals = frame[event]
                            else:
                                b_vals = base[base["mode"].astype(str) == m][event]
                                f_vals = frame[frame["mode"].astype(str) == m][event]
                        except Exception:
                            continue
                        if not _can_compare(b_vals, f_vals):
                            continue
                        r = clt_test(b_vals, f_vals, alpha=alpha)
                        try:
                            if not math.isfinite(float(r["p_value"])):
                                continue
                        except (TypeError, ValueError):
                            continue
                        row = {
                            **match,
                            "event": event,
                            "baseline": base_label,
                            "challenger": label,
                            "diff": r["diff"],
                            "ci_low": r["ci_low"],
                            "ci_high": r["ci_high"],
                            "p_value": r["p_value"],
                            "alpha": alpha,
                            "significant": bool(r["significant"]),
                            "result": _result_label(r["significant"], r["diff"]),
                        }
                        row.setdefault(
                            "mode", match.get("mode", "" if m is None else m)
                        )
                        if m is not None and "mode" not in match:
                            row["mode"] = m
                        rows.append(row)
                    continue
                if not _can_compare(base[event], frame[event]):
                    continue
                r = clt_test(base[event], frame[event], alpha=alpha)
                try:
                    if not math.isfinite(float(r["p_value"])):
                        continue
                except (TypeError, ValueError):
                    continue
                row = {
                    **match,
                    "event": event,
                    "baseline": base_label,
                    "challenger": label,
                    "diff": r["diff"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "p_value": r["p_value"],
                    "alpha": alpha,
                    "significant": bool(r["significant"]),
                    "result": _result_label(r["significant"], r["diff"]),
                }
                row.setdefault("mode", match.get("mode", ""))
                rows.append(row)
    if not rows:
        raise ValueError(
            f"no comparable groups: need at least 2 {variant} variants sharing "
            f"the same group (baseline {base_label!r} never co-occurs "
            "with another variant)"
        )
    cols = [
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
        "result",
    ]
    out = pd.DataFrame(rows, columns=cols)
    out.attrs["alpha"] = alpha
    out.attrs["on"] = keys
    out.attrs["variant"] = variant
    return out
