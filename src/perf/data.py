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
import subprocess
from pathlib import Path

import pandas as pd

_PERFILE1 = b"PERFILE1"
_PERFILE2 = b"PERFILE2"
_MEMBERSHIP_RE = re.compile(
    r"(?P<value>[-+]?(?:0x[0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))"
    r"(?![\w.])"
    r"\s+in\s+(?P<column>`[^`]+`|[A-Za-z_][A-Za-z_0-9.]*)"
)
_STRING_RE = re.compile(r"""('[^']*'|"[^"]*")""")
_FIELDS = "comm,pid,time,period,event,ip,sym,dso"
_DATA_BUCKETS = ("regs", "mem", "memory")
_COLUMN_PREFIXES = ("config", "data")
_IDENTITY_COLUMNS = ("file", "name", "mode")
_NON_METRIC_COLUMNS = frozenset(
    {
        *_IDENTITY_COLUMNS,
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


def nest(record, prefixes=_COLUMN_PREFIXES):
    rows = dict(record or {})
    claimed = {p for p in prefixes if p in rows and not isinstance(rows[p], dict)}
    out = {}
    for key, value in rows.items():
        head, _, rest = str(key).partition(".")
        if head in claimed or head not in prefixes or not rest:
            out[key] = value
            continue
        parts = [p for p in rest.split(".") if p]
        if not parts:
            out[key] = value
            continue
        if not isinstance(out.get(head), dict):
            out[head] = {}
        _nest(out[head], parts, value)
    return out


def spread(record, prefixes=_COLUMN_PREFIXES):
    out = {}
    for key, value in dict(record or {}).items():
        name = str(key)
        if name not in prefixes or not isinstance(value, dict):
            out[key] = value
            continue
        if name == "data":
            plain = {k: v for k, v in value.items() if k not in _DATA_BUCKETS}
            for bucket in _DATA_BUCKETS:
                plain.update({k: v for k, v in (value.get(bucket) or {}).items()})
            _spread(out, "data", plain)
            continue
        _spread(out, name, value)
    return out


def is_record(path):
    if isinstance(path, str):
        path = Path(path)
    try:
        with path.open("rb") as f:
            return f.read(8) in (_PERFILE1, _PERFILE2)
    except OSError:
        return False


@functools.cache
def system_perf():
    for directory in os.get_exec_path():
        candidate = os.path.join(directory, "perf")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    for candidate in ("/usr/bin/perf", "/usr/sbin/perf"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def samples(path, bin=None, fields=_FIELDS):
    bin = bin or system_perf()
    if not bin:
        raise RuntimeError(
            "system perf not found; install linux-tools to read perf.data files"
        )
    proc = subprocess.run(
        [bin, "script", "-i", str(path), "-F", fields],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(
            f"`perf script -i {path}` failed: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return _parse_lines(proc.stdout)


def metrics(df, events=None):
    if "event" not in df.columns or "period" not in df.columns:
        return df
    if events is None:
        events = list(pd.unique(df["event"]))
    else:
        have = set(pd.unique(df["event"]))
        events = [e for e in events if e in have]
    if not events:
        return df
    keys = [c for c in df.columns if c not in ("event", "period")]
    if not keys:
        return df
    wide = df.pivot_table(index=keys, columns="event", values="period", aggfunc="max")
    return wide.reset_index()[keys + [c for c in wide.columns if c in events]]


def quote_columns(df, expr):
    expr = str(expr)
    try:
        names = [
            str(c)
            for c in df.columns
            if isinstance(c, str) and c and not c.isidentifier() and c in expr
        ]
    except Exception:
        return expr
    names.sort(key=len, reverse=True)
    for name in names:
        if f"`{name}`" not in expr:
            expr = expr.replace(name, f"`{name}`")
    return expr


def query(df, expression):
    expr = quote_columns(df, expression)
    expr, local = _memberships(df, expr)
    return df.query(expr, engine="python", local_dict=local)


def parse(paths, bin=None):
    frames = []
    for path in paths:
        df = samples(path, bin)
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "file", Path(path).name)
        frames.append(df)
    if frames:
        return metrics(pd.concat(frames, ignore_index=True))
    return pd.DataFrame(columns=["file"] + _FIELDS.split(","))


def _scalar(text):
    if len(text) > 1 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    try:
        return int(text, 0)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _members(series, value):
    want = _scalar(value)
    if isinstance(want, (str, int, float, bool)) or want is None:
        try:
            has_containers = bool(
                series.map(
                    lambda v: isinstance(v, (list, tuple, set, frozenset, dict))
                ).any()
            )
        except Exception:
            has_containers = True
        if not has_containers:
            try:
                return series.isin([want])
            except Exception:
                pass
    try:
        return series.map(
            lambda v: (
                want in v
                if isinstance(v, (list, tuple, set, frozenset, dict))
                else v == want
            )
        )
    except Exception:
        return series == want


def _memberships(df, expr):
    local = {}
    colmap = {str(c): c for c in df.columns}

    def _sub(match):
        name = match.group("column").strip("`")
        column = colmap.get(name)
        if column is None:
            return match.group(0)
        key = f"_perf_in_{len(local)}"
        local[key] = _members(df[column], match.group("value"))
        return f"@{key}"

    out = []
    for text in _STRING_RE.split(str(expr)):
        if not text or text[0] not in "'\"":
            text = _MEMBERSHIP_RE.sub(_sub, text)
        out.append(text)
    return "".join(out), local


def _nest(out, parts, value):
    node = out
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _spread(out, prefix, node):
    for key, value in (node or {}).items():
        if isinstance(value, dict):
            _spread(out, f"{prefix}.{key}", value)
        else:
            out[f"{prefix}.{key}"] = value


@functools.lru_cache(maxsize=4096)
def _norm_event(name):
    parts = name.split("/")
    base = parts[-2] if len(parts) >= 3 else name
    return base.split(":", 1)[0]


def _parse_lines(text):
    rows = []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            pid = int(parts[1])
            period = int(parts[3])
            ip = int(parts[5], 16)
        except ValueError:
            continue
        time = parts[2][:-1] if parts[2].endswith(":") else parts[2]
        event = _norm_event(parts[4][:-1] if parts[4].endswith(":") else parts[4])
        tail = parts[6:]
        if tail and tail[-1].startswith("(") and tail[-1].endswith(")"):
            dso = tail[-1][1:-1]
            sym = " ".join(tail[:-1])
        else:
            dso = ""
            sym = " ".join(tail)
        rows.append(
            {
                "comm": parts[0],
                "pid": pid,
                "time": time,
                "period": period,
                "event": event,
                "ip": ip,
                "sym": sym or "[unknown]",
                "dso": dso,
            }
        )
    return pd.DataFrame(rows, columns=list(_FIELDS.split(",")))


samples.nest = nest
samples.spread = spread
samples.parse = parse
samples.metrics = metrics
samples.is_record = is_record
