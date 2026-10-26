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
        "iteration",
        "iterations",
        "operations",
        "threads",
        "tid",
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


def envelope_columns(row):
    out = {}
    for key, value in dict(row or {}).items():
        if isinstance(value, dict):
            _flatten_into(key, value, out, buckets=key == "data")
        else:
            out[key] = value
    return out


def envelope_frame(payload):
    output = payload.get("output") or []
    shared = payload.get("shared") or {}
    flat_shared = envelope_columns(shared) if shared else {}
    if not output:
        if not flat_shared:
            return pd.DataFrame()
        try:
            return pd.DataFrame([flat_shared])
        except Exception:
            return pd.DataFrame()
    try:
        flat_rows = [envelope_columns(row) for row in output]
    except Exception:
        flat_rows = []
        for row in output:
            try:
                flat_rows.append(envelope_columns(row))
            except Exception:
                continue
    try:
        df = pd.DataFrame(flat_rows)
    except Exception:
        columns: dict = {}
        rows = 0
        for row in flat_rows:
            rows += 1
            seen = set()
            for name, value in (row or {}).items():
                col = columns.get(name)
                if col is None:
                    if rows > 1:
                        columns[name] = [None] * (rows - 1)
                        col = columns[name]
                    else:
                        col = columns[name] = []
                col.append(value)
                seen.add(name)
            if len(seen) != len(columns):
                for name, col in columns.items():
                    if name not in seen:
                        col.append(None)
        try:
            df = pd.DataFrame(columns)
        except Exception:
            return pd.DataFrame()
    if flat_shared:
        try:
            n = len(df)
        except Exception:
            n = 0
        missing = {k: v for k, v in flat_shared.items() if k not in df.columns}
        if missing:
            # Add every shared column at once; assigning them one by one
            # fragments the frame and is slow for large config/data blocks.
            try:
                if n:
                    extra = pd.DataFrame(missing, index=df.index)
                else:
                    extra = pd.DataFrame({k: pd.Series(dtype=object) for k in missing})
                df = pd.concat([df, extra], axis=1)
            except Exception:
                columns = {}
                for k, v in missing.items():
                    try:
                        if n == 0:
                            columns[k] = pd.Series(dtype=object)
                        elif isinstance(v, pd.Series):
                            columns[k] = v.reindex(df.index)
                        elif isinstance(v, dict):
                            columns[k] = pd.Series(v).reindex(df.index)
                        elif isinstance(v, (list, tuple)):
                            if len(v) != n:
                                continue
                            columns[k] = pd.Series(list(v), index=df.index)
                        else:
                            columns[k] = pd.Series([v] * n, index=df.index)
                    except Exception:
                        continue
                if columns:
                    try:
                        df = pd.concat(
                            [df, pd.DataFrame(columns, index=df.index)], axis=1
                        )
                    except Exception:
                        pass
    return df


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
        [
            bin,
            "script",
            "-i",
            str(path),
            "-F",
            ",".join(fields) if isinstance(fields, (list, tuple)) else fields,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(
            f"`perf script -i {path}` failed: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return _parse_lines(proc.stdout, fields)


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
    wide = df.pivot_table(index=keys, columns="event", values="period", aggfunc="sum")
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
    if not names:
        return expr
    names.sort(key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.])"
        r"(" + "|".join(re.escape(name) for name in names) + r")"
        r"(?![A-Za-z0-9_.])"
    )

    def _quote_segment(text):
        return pattern.sub(lambda match: f"`{match.group(1)}`", text)

    out = []
    for part in re.split(r"(`[^`]*`|'[^']*'|\"[^\"]*\")", expr):
        if len(part) >= 2 and part[0] == part[-1] and part[0] in "`'\"":
            out.append(part)
        else:
            out.append(_quote_segment(part))
    return "".join(out)


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


def _flatten_into(prefix, node, out, buckets=False):
    for key, value in (node or {}).items():
        if buckets and key in _DATA_BUCKETS:
            if isinstance(value, dict):
                _flatten_into(prefix, value, out)
            continue
        if isinstance(value, dict):
            _flatten_into(f"{prefix}.{key}", value, out)
        else:
            out[f"{prefix}.{key}"] = value


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
    head, sep, tail = base.rpartition(":")
    if sep and tail and len(tail) <= 3 and all(c in "ukhHGpSDIZW" for c in tail):
        return head
    return base


_SAMPLE_RE = re.compile(
    r"^\s*(?P<comm>.*?)\s+(?P<pid>-?\d+)\s+"
    r"(?P<time>\S+?):?\s+(?P<period>\d+)\s+"
    r"(?P<event>\S+?):?\s+(?P<ip>[0-9a-fA-F]+)\s*(?P<tail>.*)$"
)
_FIELD_PATTERNS = {
    "comm": r"(?P<comm>.+?)",
    "pid": r"(?P<pid>-?\d+)",
    "tid": r"(?P<tid>-?\d+)",
    "cpu": r"(?P<cpu>\d+)",
    "time": r"(?P<time>\S+?)",
    "period": r"(?P<period>\d+)",
    "event": r"(?P<event>\S+?)",
    "ip": r"(?P<ip>[0-9a-fA-F]+)",
    "sym": r"(?P<sym>.+?)",
    "dso": r"(?P<dso>\([^)]*\)|\S+)",
}


def _fields_regex(fields):
    parts = []
    for name in fields:
        pattern = _FIELD_PATTERNS.get(name)
        if pattern is None:
            pattern = rf"(?P<{name}>\S+)"
        parts.append(pattern)
    tail = r"\s*$"
    last = fields[-1] if fields else ""
    if last == "sym":
        parts[-1] = r"(?P<sym>.*)"
    elif last == "dso":
        parts[-1] = r"(?P<dso>.*)"
    elif last == "comm":
        parts[-1] = r"(?P<comm>.+)"
    elif last in ("event", "time"):
        parts[-1] = rf"(?P<{last}>\S+)"
    else:
        tail = r"\s*(?:.*)$"
    return re.compile(r"^\s*" + r"\s+".join(parts) + tail)


def _parse_lines(text, fields=None):
    if fields is None:
        fields = _FIELDS
    if isinstance(fields, (list, tuple)):
        field_names = [str(part).strip() for part in fields if str(part).strip()]
    else:
        field_names = [part.strip() for part in str(fields).split(",") if part.strip()]
    if field_names == _FIELDS.split(","):
        rows = []
        for line in text.splitlines():
            if not line or line[0] == "#":
                continue
            match = _SAMPLE_RE.match(line)
            if match is None:
                continue
            try:
                pid = int(match.group("pid"))
                period = int(match.group("period"))
                ip = int(match.group("ip"), 16)
            except ValueError:
                continue
            time = match.group("time")
            event = _norm_event(match.group("event"))
            tail = match.group("tail").split()
            if tail and tail[-1].startswith("(") and tail[-1].endswith(")"):
                dso = tail[-1][1:-1]
                sym = " ".join(tail[:-1])
            else:
                dso = ""
                sym = " ".join(tail)
            rows.append(
                {
                    "comm": match.group("comm"),
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
    pattern = _fields_regex(field_names)
    rows = []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        match = pattern.match(line)
        if match is None:
            continue
        row = {}
        for name, value in match.groupdict().items():
            value = value.strip()
            if name in ("time", "event") and value.endswith(":"):
                value = value[:-1]
            if name == "event":
                value = _norm_event(value)
            elif name in ("pid", "tid", "cpu", "period"):
                try:
                    value = int(value, 0)
                except (TypeError, ValueError):
                    pass
            elif name == "ip":
                try:
                    value = int(value, 16)
                except (TypeError, ValueError):
                    pass
            elif name == "dso" and value.startswith("(") and value.endswith(")"):
                value = value[1:-1]
            row[name] = value
        rows.append(row)
    return pd.DataFrame(rows, columns=field_names)


samples.nest = nest
samples.spread = spread
samples.parse = parse
samples.metrics = metrics
samples.is_record = is_record
