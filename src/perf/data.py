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

import argparse
import configparser
import functools
import io
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import orjson
import pandas as pd

from .core import _is_duration_event, _split_list, one_line

_PERFILE1 = b"PERFILE1"
_PERFILE2 = b"PERFILE2"
_MEMBERSHIP_RE = re.compile(
    r"(?P<value>[-+]?(?:0x[0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))"
    r"(?![\w.])"
    r"\s+in\s+(?P<column>`[^`]+`|[A-Za-z_][A-Za-z_0-9.]*)"
)
_STRING_RE = re.compile(r"""('[^']*'|"[^"]*")""")
_FIELDS = "comm,pid,time,period,event,ip,sym,dso"
_DATA_BUCKETS = ("regs", "mem")
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
        "index",
        "pid",
        "ip",
    }
)
_OUTPUT_STDOUT = frozenset({"-", "true", "1", "yes", "on", "stdout"})
_OUTPUT_DISABLED = frozenset({"false", "0", "no", "off", "none"})
_PERFCONFIG = ".perfconfig"
_PERFCONFIG_COMMENTS = ("#", ";")
_WILDCARDS = ("*", "?")
_TABLE_EXTS = frozenset({".txt", ".tsv", ".csv"})
_SOURCE_COL = "__source__"
_BASELINE_TMP = "__perf_base"
_BASELINE_RE = re.compile(
    r"""(['"])(?P<key>.+?)\1\s*\.\s*(?P<metric>[A-Za-z_][A-Za-z0-9_\-/]*)"""
)
_PLAIN_COLUMNS = (
    "stat",
    "file",
    "name",
    "mode",
    "time",
    "iterations",
    "operations",
    "threads",
    "tid",
    "index",
    "address",
    "size",
)
_DEFAULT_GROUPBY = list(_IDENTITY_COLUMNS) + ["data.*"]
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


def _nested_key(prefix, node, buckets):
    try:
        return (
            prefix,
            bool(buckets),
            orjson.dumps(
                node,
                option=orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS,
            ),
        )
    except Exception:  # pragma: no cover
        return None


def envelope_columns(row, _cache=None):
    out = {}
    for key, value in dict(row or {}).items():
        if isinstance(value, dict):
            buckets = key == "data"
            cache_key = _nested_key(key, value, buckets) if _cache is not None else None
            flat = _cache.get(cache_key) if cache_key is not None else None
            if flat is None:
                flat = {}
                _flatten_into(key, value, flat, buckets=buckets)
                if cache_key is not None and len(_cache) < 4096:
                    _cache[cache_key] = flat
            out.update(flat)
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
        nested_cache = {}
        flat_rows = [envelope_columns(row, nested_cache) for row in output]
    except Exception:  # pragma: no cover
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
            try:
                if n:
                    extra = pd.DataFrame(missing, index=df.index)
                else:
                    extra = pd.DataFrame({k: pd.Series(dtype=object) for k in missing})
                df = pd.concat([df, extra], axis=1)
            except Exception:  # pragma: no cover
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


_JSON_NODE_LIMIT = 2000


def _json_size(node, limit=_JSON_NODE_LIMIT):
    size = 0
    stack = [node]
    while stack:
        if size >= limit:
            return limit
        current = stack.pop()
        size += 1
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return size


def _json_tables(node, out, budget):
    if budget[0] <= 0:
        return
    budget[0] -= 1
    if isinstance(node, dict):
        for value in node.values():
            _json_tables(value, out, budget)
    elif isinstance(node, list):
        if node and all(isinstance(item, dict) for item in node):
            out.append(node)
            return
        for value in node:
            _json_tables(value, out, budget)


def _json_split_frame(payload):
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    columns = payload.get("columns")
    if isinstance(columns, list):
        index = payload.get("index")
        try:
            return pd.DataFrame(
                data, columns=columns, index=index if isinstance(index, list) else None
            )
        except Exception:
            return pd.DataFrame()
    schema = payload.get("schema")
    if isinstance(schema, dict):
        fields = schema.get("fields")
        if isinstance(fields, list):
            names = [
                field.get("name")
                for field in fields
                if isinstance(field, dict) and field.get("name") is not None
            ]
            if names:
                if not data:
                    return pd.DataFrame(columns=names)
                if all(isinstance(row, dict) for row in data):
                    try:
                        return pd.DataFrame(data)
                    except Exception:
                        return pd.DataFrame()
                if all(isinstance(row, (list, tuple)) for row in data):
                    if all(len(row) == len(names) for row in data):
                        try:
                            return pd.DataFrame(data, columns=names)
                        except Exception:
                            return pd.DataFrame()
                    return pd.DataFrame()
                return pd.DataFrame()
    return None


def _json_dict_frame(payload):
    values = list(payload.values())
    if not values:
        return pd.DataFrame()
    if all(isinstance(value, dict) for value in values):
        try:
            first = values[0]
            if any(len(value) != len(first) for value in values[1:]):
                return None
            if len(first) <= _JSON_NODE_LIMIT:
                keys = frozenset(first)
                if any(frozenset(value) != keys for value in values[1:]):
                    return None
            return pd.DataFrame(payload)
        except Exception:
            return None
    if all(isinstance(value, (list, tuple)) for value in values):
        try:
            return pd.DataFrame(payload)
        except Exception:
            return None
    if all(
        not isinstance(value, (dict, list, tuple, set, frozenset)) for value in values
    ):
        try:
            return pd.DataFrame([payload])
        except Exception:
            return None
    return None


def json_frame(payload):
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, dict):
        frame = _json_split_frame(payload)
        if frame is not None:
            return frame
        frame = _json_dict_frame(payload)
        if frame is not None:
            return frame
        values = list(payload.values())
        if values and all(isinstance(v, (list, tuple)) for v in values):
            lengths = {len(v) for v in values}
            if len(lengths) > 1:
                return pd.DataFrame()
        found = []
        _json_tables(payload, found, [_JSON_NODE_LIMIT])
        for table in found:
            try:
                frame = pd.DataFrame(table)
            except Exception:
                continue
            if not frame.empty:
                return frame
        if _json_size(payload) < _JSON_NODE_LIMIT:
            try:
                frame = pd.json_normalize(payload)
            except Exception:
                frame = None
            if frame is not None and not frame.empty:
                return frame
        try:
            return pd.DataFrame([payload])
        except Exception:
            return pd.DataFrame()
    if isinstance(payload, (list, tuple)):
        if not payload:
            return pd.DataFrame()
        try:
            return pd.DataFrame(payload)
        except Exception:
            return pd.DataFrame()
    try:
        return pd.DataFrame(payload)
    except Exception:
        return pd.DataFrame()


def loads_json(text):
    return orjson.loads(text)


def dumps_json_bytes(value, indent=None, sort_keys=False):
    option = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
    if indent:
        option |= orjson.OPT_INDENT_2
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS
    return orjson.dumps(value, option=option, default=str)


def dumps_json(value, indent=None, sort_keys=False):
    return dumps_json_bytes(value, indent=indent, sort_keys=sort_keys).decode()


def is_record(path):
    if isinstance(path, str):
        path = Path(path)
    try:
        with path.open("rb") as f:
            return f.read(8) in (_PERFILE1, _PERFILE2)
    except (OSError, ValueError, TypeError, AttributeError):
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
        df["file"] = Path(path).name
        try:
            cols = ["file"] + [c for c in df.columns if c != "file"]
            df = df[cols]
        except Exception:
            pass
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


@functools.lru_cache(maxsize=4096)
def _norm_event(name):
    parts = name.split("/")
    base = parts[-2] if len(parts) >= 3 else name
    head, sep, tail = base.rpartition(":")
    if sep and tail and len(tail) <= 3 and all(c in "ukhHGpSDIZW" for c in tail):
        return head
    return base


def _fields_regex(fields):
    fields = list(dict.fromkeys(str(name) for name in fields))
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
    field_names = list(dict.fromkeys(field_names))
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


_TABLE_EMPTY = frozenset({"", "nan", "none", "nat", "null"})
_DURATION_SCALE = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
_DURATION_RE = re.compile(r"^([-+]?(\d+(?:\.\d*)?|\.\d+))(ns|us|ms|s)$")
_DURATION_SCORE_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:ns|us|ms|s)")
_LAYOUT_SAMPLE = 512
_KIND_SAMPLE = 256


def is_duration_col(column):
    return _is_duration_event(str(column).split("/", 1)[0])


def table_number(value, duration=False):
    if isinstance(value, str) and value.strip().lower() in _TABLE_EMPTY:
        return float("nan")
    if duration:
        match = _DURATION_RE.match(str(value).strip())
        if match:
            return float(match.group(1)) * _DURATION_SCALE[match.group(3)]
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def token_spans(text):
    spans, start = [], None
    for i, char in enumerate(text):
        if char.isspace():
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(text)))
    return spans


def table_to_df(text):
    lines = str(text).splitlines()
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        return pd.DataFrame()
    header = lines[0]
    if "," in header or "\t" in header:
        return _read_delimited(lines, header)
    return _read_fixed(lines, header)


def _read_delimited(lines, header):
    sep = "\t" if "\t" in header else ","
    raw = io.StringIO("\n".join(lines))
    lines.clear()
    try:
        df = pd.read_csv(raw, sep=sep, engine="c")
    except Exception:
        raw = io.StringIO(raw.getvalue())
        try:
            df = pd.read_csv(raw, sep=sep, engine="python")
        except Exception as ex:
            raise ValueError(f"unrecognized perf result table ({ex})")
    return _convert_columns(df)


def _read_fixed(lines, header):
    tokens = token_spans(header)
    if not tokens:
        return pd.DataFrame()
    right = _right_spans(tokens)
    left = _left_spans(tokens)
    right_cols = [header[start:end].strip() for start, end in right]
    left_cols = [header[start:end].strip() for start, end in left]
    if any(not column for column in right_cols):
        right = None
    if any(not column for column in left_cols):
        left = None
    if right is None and left is None:
        raise ValueError("unrecognized perf result table")
    data = lines[1:]
    if not data:
        raise ValueError("unrecognized perf result table")
    sample = None
    if left is None:
        spans, columns = right, right_cols
    elif right is None:
        spans, columns = left, left_cols
    else:
        sample_lines = _sample_rows(data, _LAYOUT_SAMPLE)
        right_rows = [_spans_row(line, right) for line in sample_lines]
        left_rows = [_spans_row(line, left) for line in sample_lines]
        right_score = _layout_score(right_rows)
        left_score = _layout_score(left_rows)
        if left_score > right_score:
            spans, columns, sample = left, left_cols, left_rows
        else:
            spans, columns, sample = right, right_cols, right_rows
    if sample is None:
        sample = [
            _spans_row(line, spans) for line in _sample_rows(data, _LAYOUT_SAMPLE)
        ]
    durations = [is_duration_col(column) for column in columns]
    kinds = [
        _classify((row[index] for row in sample), durations[index])
        for index in range(len(columns))
    ]
    dedupe = [
        kind == "text" and _repeated(row[index] for row in sample)
        for index, kind in enumerate(kinds)
    ]
    cells = _parse_fixed(data, spans, kinds, durations, dedupe)
    del data
    lines.clear()
    return pd.DataFrame(dict(zip(columns, cells)))


def _right_spans(tokens):
    ends = [end for _, end in tokens]
    starts = [0] + [ends[i - 1] + 1 for i in range(1, len(ends))]
    spans = list(zip(starts, ends))
    spans[-1] = (spans[-1][0], None)
    return spans


def _left_spans(tokens):
    starts = [0] + [start for start, _ in tokens[1:]]
    return list(zip(starts, [*starts[1:], None]))


def _sample_rows(rows, limit):
    if len(rows) <= limit:
        return rows
    step = max(1, len(rows) // limit)
    sample = rows[::step][:limit]
    if sample and sample[-1] != rows[-1]:
        sample.append(rows[-1])
    return sample


def _spans_row(line, spans):
    length = len(line)
    return [
        (line[start:end] if end is not None else line[start:]).strip()
        if start < length
        else ""
        for start, end in spans
    ]


def _layout_score(rows):
    score = 0
    for row in rows:
        for value in row:
            if not value:
                continue
            try:
                float(value)
                score += 1
                continue
            except ValueError:
                pass
            if _DURATION_SCORE_RE.fullmatch(value):
                score += 1
    return score


def _repeated(values):
    seen = [value for value in values if value]
    return len(seen) > 1 and len(set(seen)) <= len(seen) // 2


def _parse_fixed(data, spans, kinds, durations, dedupe):
    cells = [[] for _ in kinds]
    append = [cell.append for cell in cells]
    cache = [{} if dedupe[index] else None for index in range(len(kinds))]
    nan = float("nan")
    for line in data:
        length = len(line)
        for index, (start, end) in enumerate(spans):
            if start >= length:
                value = ""
            elif end is None:
                value = line[start:].strip()
            else:
                value = line[start:end].strip()
            kind = kinds[index]
            if value:
                if kind == "num":
                    try:
                        append[index](float(value))
                        continue
                    except ValueError:
                        pass
                elif kind == "dur":
                    match = _DURATION_RE.match(value)
                    if match:
                        append[index](
                            float(match.group(1)) * _DURATION_SCALE[match.group(3)]
                        )
                        continue
                    try:
                        append[index](float(value))
                        continue
                    except ValueError:
                        pass
                elif kind == "mixed":
                    append[index](table_number(value, durations[index]))
                    continue
                if value.lower() in _TABLE_EMPTY:
                    append[index](nan)
                else:
                    bucket = cache[index]
                    if bucket is not None:
                        value = bucket.setdefault(value, value)
                    append[index](value)
            else:
                append[index](nan)
    return cells


def _convert_columns(df):
    for column in list(df.columns):
        series = df[column]
        duration = is_duration_col(column)
        kind = _classify(series, duration)
        df[column] = _convert_column(series, duration, kind)
    return df


def _classify(series, duration):
    kind = None
    for value in _sample_values(series):
        if _is_empty(value):
            continue
        if isinstance(value, str):
            value = value.strip()
        try:
            float(value)
            current = "num"
        except (TypeError, ValueError):
            if duration and _DURATION_RE.match(str(value).strip()):
                current = "dur"
            else:
                current = "text"
        if kind is None:
            kind = current
        elif kind != current:
            if {kind, current} <= {"num", "dur"}:
                kind = "dur"
            else:
                return "mixed"
    return kind or "blank"


def _sample_values(series):
    try:
        length = len(series)
    except TypeError:
        return list(series)
    if length <= _KIND_SAMPLE:
        return series.tolist()
    step = max(1, length // _KIND_SAMPLE)
    return series.iloc[::step].iloc[:_KIND_SAMPLE].tolist()


def _is_empty(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _TABLE_EMPTY
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _convert_column(series, duration, kind):
    blank = _blank_mask(series)
    trimmed = _trimmed(series)
    if kind == "blank":
        return pd.Series(float("nan"), index=series.index, dtype=float)
    if kind == "text":
        values = series.astype(object)
        if bool(blank.any()):
            values = values.mask(blank, float("nan"))
        return values
    if kind == "num":
        numbers = pd.to_numeric(trimmed, errors="coerce")
        if bool((numbers.notna() | blank).all()):
            return numbers.astype(float)
    elif kind == "dur":
        try:
            parts = trimmed.str.extract(_DURATION_RE)
            numbers = pd.to_numeric(parts[0], errors="coerce")
            values = numbers * parts[2].map(_DURATION_SCALE)
        except (AttributeError, KeyError, TypeError, ValueError):
            values = None
        if values is not None and bool((values.notna() | blank).all()):
            return values.astype(float)
        return _convert_values(series, duration=True)
    if duration:
        return _convert_values(series, duration=True)
    return _convert_mixed(series, trimmed, blank)


def _convert_values(series, duration):
    return pd.Series(
        [table_number(value, duration) for value in series.tolist()],
        index=series.index,
    )


def _convert_mixed(series, trimmed, blank):
    numbers = pd.to_numeric(trimmed, errors="coerce")
    values = numbers.astype(float).astype(object)
    bad = (~numbers.notna()) & (~blank)
    if bool(bad.any()):
        values[bad] = series.astype(object)[bad]
    if bool(blank.any()):
        values[blank] = float("nan")
    return values


def _blank_mask(series):
    try:
        text = series.str.strip()
        mask = text.str.lower().isin(_TABLE_EMPTY)
    except (AttributeError, TypeError):
        mask = series.map(_is_empty)
    try:
        return mask | series.isna()
    except TypeError:
        return mask


def _trimmed(series):
    try:
        return series.str.strip()
    except (AttributeError, TypeError):
        return series


def output_targets(value):
    if value is None or value is False:
        return []
    if value is True:
        return [True]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(output_targets(item))
        return out
    text = str(value).strip()
    low = text.casefold()
    if not text or low in _OUTPUT_STDOUT:
        return [True]
    if low in _OUTPUT_DISABLED:
        return []
    return [text]


def has_output(value):
    return bool(output_targets(value))


def show_table(value, from_config=False):
    targets = output_targets(value)
    if not targets:
        return True
    return bool(from_config) and not any(target is True for target in targets)


def output_template(value):
    targets = output_targets(value)
    if len(targets) != 1 or targets[0] is True:
        return None
    text = str(targets[0]).strip()
    if not text or text.endswith(("/", os.sep)) or text in (".", ".."):
        return None
    path = Path(text)
    try:
        if path.is_dir():
            return None
    except OSError:
        pass
    if path.is_absolute() or path.parent != Path("."):
        return None
    return text


def _output_component(value, default):
    text = "" if value is None else str(value).strip()
    if not text or text.casefold() in ("nan", "none", "nat"):
        text = ""
    text = text.replace("/", "_").replace(os.sep, "_").replace("\x00", "").strip()
    return text or default


def metadata(payload=None, **extra):
    data = dict(payload) if isinstance(payload, dict) else {}
    data.update({key: value for key, value in extra.items() if value is not None})
    file_label = "" if data.get("file") is None else str(data.get("file"))
    base_file, _, file_hash = file_label.partition("@")
    if base_file:
        base_file = Path(base_file).name
    name = "" if data.get("name") is None else str(data.get("name"))
    run_id = "" if data.get("id") is None else str(data.get("id"))
    if run_id and name.endswith(f"-{run_id}"):
        name = name[: -(len(run_id) + 1)]
    stamp = "" if data.get("time") is None else str(data.get("time")).strip()
    if stamp:
        stamp = stamp.replace(" ", "_").replace(":", "").replace("-", "")
        stamp = stamp.replace("T", "_")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    command = _output_component(data.get("command"), "perf")
    file_value = _output_component(base_file, "")
    hash_value = _output_component(file_hash, "")
    name_value = _output_component(name or data.get("command"), "track")
    target = (
        f"{file_value}@{hash_value}"
        if file_value and hash_value
        else file_value or name_value or command
    )
    return {
        "file": file_value,
        "hash": hash_value,
        "name": name_value,
        "id": _output_component(run_id, ""),
        "time": _output_component(stamp, datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
        "command": command,
        "target": _output_component(target, "track"),
    }


def metadata_from_frame(df, command=None):
    attrs = dict(getattr(df, "attrs", {}) or {})
    record = {}
    try:
        if len(df):
            record = df.iloc[0].to_dict()
    except Exception:
        pass
    return metadata(
        {
            "file": attrs.get("file") or record.get("file"),
            "name": attrs.get("name") or record.get("name"),
            "id": attrs.get("id") or record.get("id"),
            "time": attrs.get("time"),
            "command": attrs.get("command") or command,
        }
    )


def resolve_path(destination, values=None, template=None):
    values = values or metadata()
    text = str(destination)
    directory = text.endswith(("/", os.sep)) or text in (".", "..")
    if not directory:
        try:
            directory = Path(text).is_dir()
        except OSError:
            pass
    if directory:
        if template is None:
            raise SystemExit(
                f"--output {text!r} is a directory; pass a file path or set "
                "`output = <name-template>` in .perfconfig"
            )
        text = str(Path(text) / str(template))
    if "{" in text and "}" in text:
        text = text.format_map(values)
    return Path(text)


def emit_output(
    value,
    payload,
    *,
    values=None,
    template=None,
    indent=2,
    stdout=None,
):
    targets = output_targets(value)
    if not targets:
        return []
    values = values or metadata(payload)
    blob = dumps_json_bytes(payload, indent=indent)
    text = None
    saved = []
    printed = False
    for target in targets:
        if target is True:
            if not printed:
                if text is None:
                    text = blob.decode()
                print(text, end="", file=stdout, flush=True)
                printed = True
            continue
        path = resolve_path(target, values, template)
        try:
            if str(path.parent) not in ("", "."):
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        except OSError as ex:
            raise SystemExit(f"cannot write --output {str(path)!r}: {ex}") from ex
        saved.append(str(path))
    return saved


def split_after_dashes(argv):
    if "--" in argv:
        sep = argv.index("--")
        return list(argv[:sep]), list(argv[sep + 1 :])
    return list(argv), []


def _blank(value):
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def to_float(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_duration(value):
    ns = to_float(value)
    if ns is None:
        try:
            float(value)
            return ""
        except (TypeError, ValueError):
            return "" if _blank(value) else str(value)
    if ns < 0:
        return f"{ns:.2f}"
    if ns < 1e3:
        return f"{ns:.2f}ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f}us"
    if ns < 1e9:
        return f"{ns / 1e6:.2f}ms"
    return f"{ns / 1e9:.2f}s"


def format_number(value):
    number = to_float(value)
    if number is None:
        try:
            float(value)
            return ""
        except (TypeError, ValueError):
            return "" if _blank(value) else str(value)
    return f"{number:.2f}"


def _format_duration_values(series):
    try:
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    except Exception:
        return [format_duration(v) for v in series]
    out = []
    append = out.append
    raw = list(series) if not isinstance(series, list) else series
    for x, original in zip(vals, raw):
        if x != x:
            append(format_duration(original))
        elif x < 0:
            append(f"{x:.2f}")
        elif x < 1e3:
            append(f"{x:.2f}ns")
        elif x < 1e6:
            append(f"{x / 1e3:.2f}us")
        elif x < 1e9:
            append(f"{x / 1e6:.2f}ms")
        else:
            append(f"{x / 1e9:.2f}s")
    return out


def _format_number_values(series):
    try:
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    except Exception:
        return [format_number(v) for v in series]
    out = []
    append = out.append
    raw = list(series) if not isinstance(series, list) else series
    for x, original in zip(vals, raw):
        if x != x:
            append(format_number(original))
        else:
            append(f"{x:.2f}")
    return out


def _is_bool_series(s):
    try:
        if pd.api.types.is_bool_dtype(s):
            return True
    except Exception:
        pass
    try:
        if getattr(s, "dtype", None) is not None and str(
            getattr(s, "dtype")
        ).startswith("bool"):
            return True
    except Exception:
        pass
    return False


def format_table(df, plain_columns=_PLAIN_COLUMNS):
    out = df.copy()
    for column in out.columns:
        if column in plain_columns:
            try:
                if _is_bool_series(out[column]):
                    out[column] = (
                        out[column].map({True: "True", False: "False"}).fillna("")
                    )
                    continue
            except Exception:
                pass
            continue
        if is_duration_col(column):
            try:
                out[column] = _format_duration_values(out[column])
            except Exception:
                out[column] = [format_duration(v) for v in out[column]]
            continue
        try:
            if _is_bool_series(out[column]):
                out[column] = out[column].map({True: "True", False: "False"}).fillna("")
                continue
        except Exception:
            pass
        try:
            if pd.api.types.is_numeric_dtype(out[column]):
                try:
                    out[column] = _format_number_values(out[column])
                except Exception:
                    out[column] = [format_number(v) for v in out[column]]
        except Exception:
            pass
    try:
        out = out.where(pd.notna(out), "")
    except Exception:
        pass
    return out


def _cell(value):
    if value is None or (
        not isinstance(value, (list, tuple, dict, set)) and pd.isna(value)
    ):
        return ""
    return one_line(value)


def is_numeric(df, column):
    try:
        if pd.api.types.is_bool_dtype(df[column]):
            return False
        return bool(pd.api.types.is_numeric_dtype(df[column]))
    except Exception:
        return False


def format_trace_table(df):
    if df is None or not len(df.columns):
        return ""
    columns = list(df.columns)
    cells = {c: [_cell(v) for v in df[c].tolist()] for c in columns}
    numeric = {c: is_numeric(df, c) for c in columns}
    widths = {c: max([len(str(c))] + [len(s) for s in cells[c]]) for c in columns}
    lines = [
        "  ".join(
            str(c).rjust(widths[c]) if numeric[c] else str(c).ljust(widths[c])
            for c in columns
        ).rstrip()
    ]
    for i in range(len(df)):
        lines.append(
            "  ".join(
                cells[c][i].rjust(widths[c])
                if numeric[c]
                else cells[c][i].ljust(widths[c])
                for c in columns
            ).rstrip()
        )
    return "\n".join(lines)


def _json_records(df):
    try:
        records = df.to_dict(orient="records")
    except Exception:
        records = []
    try:
        for row in records:
            for key, value in list(row.items()):
                try:
                    if pd.isna(value):
                        row[key] = None
                except (TypeError, ValueError):
                    continue
    except Exception:
        pass
    return records


def _cell_text(value):
    kind = type(value)
    if kind is str:
        return value
    if value is None:
        return ""
    if kind is float:
        return "" if value != value else str(value)
    if kind is int or kind is bool:
        return str(value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _write_frame(frame, chunk_rows=1000):
    try:
        columns = list(frame.columns)
        if not columns:
            print(frame.to_string(index=False), flush=True)
            return
        headers = [str(c) for c in columns]
        widths = []
        for column, header in zip(columns, headers):
            width = len(header)
            for value in frame[column].tolist():
                size = len(_cell_text(value))
                if size > width:
                    width = size
            try:
                if pd.api.types.is_numeric_dtype(frame[column]):
                    width += 1
            except Exception:
                pass
            widths.append(width)
        total = len(frame)
    except Exception:
        print(frame.to_string(index=False), flush=True)
        return
    chunk = max(1, int(chunk_rows or 1000))
    sys.stdout.write(" ".join(h.rjust(w) for h, w in zip(headers, widths)) + "\n")
    for start in range(0, total, chunk):
        part = frame.iloc[start : start + chunk]
        values = [part[column].tolist() for column in columns]
        lines = [
            " ".join(
                _cell_text(column[i]).rjust(width)
                for column, width in zip(values, widths)
            )
            for i in range(len(part))
        ]
        if lines:
            sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def print_table(df, max_rows=None, chunk_rows=1000, formatter=None):
    try:
        total = len(df)
    except Exception:
        total = 0
    truncated = 0
    if max_rows is not None:
        try:
            if total > max_rows:
                df = df.head(max_rows)
                truncated = total - max_rows
        except Exception:
            pass
    try:
        frame = (formatter or format_table)(df)
    except Exception:
        frame = df
    try:
        _write_frame(frame, chunk_rows)
    except Exception:
        try:
            print(frame.to_string(index=False), flush=True)
        except Exception:
            print(str(frame), flush=True)
    if truncated:
        print(f"... ({truncated} more rows truncated)", flush=True)


def _split_seq(text):
    return [y for y in re.split(r"[,\s]+", text.strip()) if y.strip()]


def _unquote(text):
    s = str(text).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _parse_scalar(text):
    s = _unquote(text)
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s, 0)
    except ValueError:
        pass
    try:
        value = float(s)
        if math.isfinite(value):
            return value
    except ValueError:
        pass
    return s


def _parse_atom(text):
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return parse_value(text)
    if text.startswith("[") and text.endswith("]"):
        return [_parse_atom(y) for y in _split_seq(text[1:-1])]
    if "," in text:
        return [_parse_atom(y) for y in text.split(",") if y.strip()]
    return _parse_scalar(text)


def parse_value(text):
    v = text.strip()
    if v.startswith("module://"):
        return _parse_scalar(v)
    if v.startswith("{") and v.endswith("}"):
        body = v[1:-1].strip()
        if not body:
            return {}
        out = {}
        for item in body.split(","):
            if ":" not in item:
                raise ValueError(f"invalid dict item: {item!r}")
            key, value = item.split(":", 1)
            out[_unquote(key)] = parse_value(value.strip())
        return out
    if v.startswith("[") and v.endswith("]"):
        return [_parse_atom(x) for x in _split_seq(v[1:-1])]
    if "," in v:
        return [x.strip() for x in v.split(",") if x.strip()]
    if ":" in v and not v.startswith("["):
        key, value = v.split(":", 1)
        return {key.strip(): parse_value(value.strip())}
    return _parse_scalar(v)


def _match(pattern, text):
    return (
        re.fullmatch(
            "".join(
                ".*" if c == "*" else "." if c == "?" else re.escape(c)
                for c in str(pattern)
            ),
            str(text),
        )
        is not None
    )


def split_spec(value):
    return _split_list(value, expand=False)


def expand_names(df, names, pool=None):
    names = [str(n) for n in names or []]
    if not any(any(w in n for w in _WILDCARDS) for n in names):
        return list(dict.fromkeys(names))
    columns = (
        [c for c in df.columns if c != _SOURCE_COL] if pool is None else list(pool)
    )
    out = []
    for name in names:
        if any(w in name for w in _WILDCARDS):
            found = [c for c in columns if c not in out and _match(name, c)]
            if found:
                out.extend(found)
                continue
        if name not in out:
            out.append(name)
    return out


def data_columns(df):
    return [c for c in df.columns if str(c).startswith("data.")]


def metric_columns(df):
    columns = []
    for column in df.columns:
        if (
            str(column).startswith(("data.", "config."))
            or column in _NON_METRIC_COLUMNS
        ):
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
    base = [c for c in metric_columns(df) if numeric.get(c)]
    derived = []
    if numeric.get("operations"):
        derived.extend(f"{column}/operations" for column in base)
    if numeric.get("threads"):
        derived.extend(f"{column}/threads" for column in base)
    if numeric.get("operations") and numeric.get("threads"):
        derived.extend(f"{column}/operations/threads" for column in base)
    if numeric.get("cycles") and numeric.get("instructions"):
        derived.append("instructions/cycles")
    return derived


def apply_filter(df, filter):
    if not filter:
        return df
    try:
        return query(df, filter)
    except Exception as ex:
        raise SystemExit(f"invalid --filter {filter!r}: {ex}")


def _tag_source(df, path):
    try:
        df[_SOURCE_COL] = Path(path).name if path is not None else "<stdin>"
    except Exception:
        pass
    return df


def merge_envelope_data(payload, df):
    from .bench import _order_bench_columns

    for key in (*_IDENTITY_COLUMNS, "time"):
        if key in payload and key not in df.columns:
            df[key] = payload[key]
    for key in ("data", "config", "info", "id"):
        if payload.get(key) is not None and key not in df.attrs:
            df.attrs[key] = payload[key]
    return _order_bench_columns(df)


def restore_attrs(frames, df):
    for frame in frames:
        for key in ("data", "config", "info", "id"):
            value = (getattr(frame, "attrs", {}) or {}).get(key)
            if value is not None and key not in df.attrs:
                try:
                    df.attrs[key] = value
                except Exception:
                    pass


def _find_baseline_rows(df, key):
    for column in ("file", "name"):
        if column in df.columns:
            try:
                hit = df[column].astype(str) == key
            except Exception:
                continue
            if bool(hit.any()):
                return df[hit]
    return None


def _baseline_values(df, key, metric):
    base = _find_baseline_rows(df, key)
    if base is None:
        raise ValueError(f"baseline group {key!r} not found in 'file'/'name'")
    if metric not in df.columns:
        raise ValueError(f"baseline metric {metric!r} not found in data")
    try:
        return float(base[metric].mean())
    except (TypeError, ValueError) as ex:
        raise ValueError(f"baseline metric {metric!r} is not numeric") from ex


def _inject_baselines(df, expression, counter):
    temps = []

    def _sub(match):
        metric = match.group("metric")
        values = _baseline_values(df, match.group("key"), metric)
        safe = (
            "".join(c if c.isalnum() or c == "_" else "_" for c in metric) or "metric"
        )
        name = f"{_BASELINE_TMP}_{safe}_{counter[0]}"
        counter[0] += 1
        while name in df.columns:
            name = f"{name}_"
        df[name] = values
        temps.append(name)
        return name

    try:
        return _BASELINE_RE.sub(_sub, expression), temps
    except ValueError:
        for name in temps:
            if name in df.columns:
                del df[name]
        raise


def eval_events(df, events):
    if any(n in (df.index.names or []) for n in _IDENTITY_COLUMNS):
        df = df.reset_index()
    if events is None:
        events = []
    else:
        events = expand_names(
            df, split_spec(events), pool=metric_columns(df) + data_columns(df)
        )
    events = _split_list(events) or (metric_columns(df) + derived_metrics(df))
    if not events:
        raise SystemExit("no metric columns found in data")
    out, created, counter = [], [], [0]
    copied = False
    for event in events:
        if event in df.columns:
            out.append(event)
            continue
        if not copied:
            df = df.copy()
            copied = True
        try:
            rewritten, temps = _inject_baselines(df, event, counter)
            created.extend(temps)
            rewritten = quote_columns(df, rewritten)
        except ValueError as ex:
            print(
                f"Warning: failed to evaluate expression {event!r}: {ex}",
                file=sys.stderr,
            )
            continue
        try:
            df[event] = df.eval(rewritten, engine="python")
            out.append(event)
        except Exception as ex:
            print(
                f"Warning: failed to evaluate expression {event!r}: {ex}",
                file=sys.stderr,
            )
    for name in created:
        if name in df.columns:
            del df[name]
    if not out:
        raise SystemExit("no metric columns found in data")
    return df, out


def generic_frame(payload):
    return json_frame(payload)


def _has_numeric(df):
    try:
        for column in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[column]):
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _read_table_text(text, path, strict=True):
    if str(text).lstrip()[:1] in ("{", "["):
        if not strict:
            return None
        raise SystemExit(
            f"invalid envelope JSON in {path!r}: missing 'output' "
            "(expected a perf envelope or a result table)"
        )
    try:
        df = table_to_df(text)
    except ValueError as ex:
        if not strict:
            return None
        raise SystemExit(f"cannot read {path!r} as a perf result table: {ex}")
    if df is None or df.empty:
        if not strict:
            return None
        raise SystemExit(f"no data rows in {path!r}")
    return _tag_source(df, path)


def _read_envelope_json(path, strict=True, numeric=False):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as ex:
        if not strict:
            return None
        raise SystemExit(f"cannot read {path!r}: {ex}")
    try:
        payload = loads_json(text)
    except ValueError:
        return _read_table_text(text, path, strict)
    if isinstance(payload, dict) and isinstance(payload.get("output"), list):
        return _tag_source(merge_envelope_data(payload, envelope_frame(payload)), path)
    if str(text).lstrip()[:1] in ("{", "["):
        try:
            frame = json_frame(payload)
        except Exception:
            frame = None
        if (
            frame is not None
            and not frame.empty
            and (not numeric or _has_numeric(frame))
        ):
            return _tag_source(frame, path)
    return _read_table_text(text, path, strict)


def _read_result_paths(paths):
    data = []
    for p in map(Path, paths):
        if p.is_file():
            data.append((p, True))
            continue
        if not p.is_dir():
            continue
        for f in p.rglob("*"):
            try:
                if not f.is_file():
                    continue
            except OSError:
                continue
            name = f.name
            if (
                name.endswith(".json")
                or name.startswith("perf.data")
                or name.endswith(".data")
                or f.suffix.lower() in _TABLE_EXTS
            ):
                data.append((f, False))
    return data


def load(args, stdin=True, numeric=False, parse_fn=None):
    parse_fn = parse_fn or parse
    if stdin and not sys.stdin.isatty():
        try:
            raw = sys.stdin.read()
        except Exception:
            raw = ""
        if raw and raw.strip():
            try:
                payload = loads_json(raw)
            except ValueError as ex:
                try:
                    return table_to_df(raw)
                except Exception:
                    raise SystemExit(
                        f"failed to parse piped data "
                        f"(expected a JSON envelope or a perf result table): {ex}"
                    )
            if isinstance(payload, dict) and isinstance(payload.get("output"), list):
                return _tag_source(
                    merge_envelope_data(payload, envelope_frame(payload)), None
                )
            frame = json_frame(payload)
            if frame is None or frame.empty:
                raise SystemExit(
                    "piped JSON must be an envelope with 'output' or a table of records"
                )
            return _tag_source(frame, None)
    if not args.data:
        raise SystemExit("no data paths; use '-- data/' or pipe JSON via stdin")
    files = _read_result_paths(args.data)
    if not files:
        raise SystemExit(f"no data files found in {args.data!r}")
    records, envelopes = [], []
    for path, explicit in files:
        try:
            is_record_file = is_record(path)
        except Exception:
            is_record_file = False
        (records if is_record_file else envelopes).append((path, explicit))
    frames = []
    if envelopes:
        loaded = []
        for path, explicit in envelopes:
            frame = _read_envelope_json(path, strict=explicit, numeric=numeric)
            if frame is None:
                print(
                    f"skipping {path}: not a perf envelope or result table",
                    file=sys.stderr,
                )
                continue
            loaded.append(frame)
        if not loaded and not records:
            raise SystemExit(f"no readable data in {args.data!r}")
        if loaded:
            merged = pd.concat(loaded, ignore_index=True)
            restore_attrs(loaded, merged)
            frames.append(merged)
    if records:
        frames.append(parse_fn([p for p, _ in records]))
    if not frames:
        raise SystemExit(f"no readable data in {args.data!r}")
    out = pd.concat(frames, ignore_index=True)
    restore_attrs(frames, out)
    return out


def resolve_groupby(df, groupby, default_groupby=_DEFAULT_GROUPBY):
    from .bench import _as_records

    rec = _as_records(df)
    return [
        c
        for c in expand_names(
            rec,
            _split_list(groupby if groupby is not None else ",".join(default_groupby)),
            pool=rec.columns,
        )
        if c in rec.columns
    ]


def _list_patterns(target):
    pats = []
    stack = [target] if target is not None else []
    while stack:
        entry = stack.pop(0)
        if entry is None:
            continue
        if isinstance(entry, (list, tuple)):
            stack[:0] = list(entry)
            continue
        text = str(entry).strip()
        if text:
            pats.append(text)
    return pats


def _filter_list_df(df, pats):
    if not pats or df is None or "name" not in df.columns:
        return df
    names = df["name"].astype(str).tolist()
    keep = []
    for name in names:
        hit = False
        for pat in pats:
            try:
                if re.search(pat, name):
                    hit = True
                    break
            except re.error as ex:
                raise SystemExit(f"invalid --list pattern {pat!r}: {ex}")
        keep.append(hit)
    try:
        return df[keep].reset_index(drop=True)
    except Exception:
        return df


def _list_split_spec(value):
    return _split_list(value, expand=False)


def _list_expand_names(df, names):
    return expand_names(df, names)


def _resolve_list_columns(df, events, strict=True):
    spec = _list_expand_names(df, _list_split_spec(events))
    unknown = [c for c in spec if c not in df.columns]
    if unknown:
        if strict:
            raise SystemExit(
                f"unknown columns: {', '.join(unknown)}; "
                f"available: {', '.join(str(c) for c in df.columns)}"
            )
        spec = [c for c in spec if c in df.columns]
    return df[spec] if spec else df


def _list_cell(value):
    if value is None or (
        not isinstance(value, (list, tuple, dict, set)) and pd.isna(value)
    ):
        return ""
    return one_line(value)


def _list_is_numeric(df, column):
    try:
        if pd.api.types.is_bool_dtype(df[column]):
            return False
        return bool(pd.api.types.is_numeric_dtype(df[column]))
    except Exception:
        return False


def _format_list_table(df):
    if df is None or not len(df.columns):
        return ""
    columns = list(df.columns)
    cells = {c: [_list_cell(v) for v in df[c].tolist()] for c in columns}
    numeric = {c: _list_is_numeric(df, c) for c in columns}
    widths = {c: max([len(str(c))] + [len(s) for s in cells[c]]) for c in columns}
    lines = [
        "  ".join(
            str(c).rjust(widths[c]) if numeric[c] else str(c).ljust(widths[c])
            for c in columns
        ).rstrip()
    ]
    for i in range(len(df)):
        lines.append(
            "  ".join(
                cells[c][i].rjust(widths[c])
                if numeric[c]
                else cells[c][i].ljust(widths[c])
                for c in columns
            ).rstrip()
        )
    return "\n".join(lines)


def _looks_like_output(value):
    text = str(value or "").strip()
    if not text:
        return True
    if text.endswith((".json", ".o", ".out")):
        return True
    if text.startswith((".", "~")) or "/" in text or os.sep in text:
        return True
    if "{" in text and "}" in text:
        return True
    try:
        return Path(text).exists()
    except OSError:
        return False


def _fix_output_code_ambiguity(args):
    code = getattr(args, "code", None)
    values = getattr(args, "output", None)
    if code or not isinstance(values, list) or len(values) != 1:
        return args
    value = values[0]
    if not isinstance(value, str) or _looks_like_output(value):
        return args
    args.code = value
    args.output = [True]
    return args


def _is_asm_source_file(path):
    return str(path or "").lower().endswith((".s", ".asm"))


def target_text(target):
    if isinstance(target, (list, tuple)):
        return "..".join(str(p) for p in target)
    return "" if target is None else str(target)


def bench_code(raw):
    from .bench import parse_code

    try:
        return parse_code(raw)
    except ValueError as ex:
        raise SystemExit(str(ex))


def _perfconfig_path():
    try:
        current = Path.cwd().resolve()
        home = Path.home().resolve()
    except OSError:
        return None
    while True:
        candidate = current / _PERFCONFIG
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            pass
        if current == home or current.parent == current:
            return None
        current = current.parent


def perfconfig_entries(section, path=None):
    path = Path(path) if path is not None else _perfconfig_path()
    if path is None or not path.is_file():
        return {}
    parser = configparser.ConfigParser(
        interpolation=None,
        strict=False,
        comment_prefixes=_PERFCONFIG_COMMENTS,
        inline_comment_prefixes=_PERFCONFIG_COMMENTS,
    )
    try:
        with open(path, encoding="utf-8") as fh:
            parser.read_file(fh, source=str(path))
    except (configparser.Error, UnicodeDecodeError, OSError):
        return {}
    entries = {}
    if not parser.has_section(section):
        return entries
    for key, value in parser.items(section):
        if key.strip() and value.strip():
            entries[key] = value.strip()
    return entries


def _known(parser, key, dotted_keys=()):
    if dotted_keys and key.startswith(dotted_keys):
        return True
    return any(
        action.dest == key or key in {o.lstrip("-") for o in action.option_strings}
        for action in parser._actions
    )


def _aliases(parser, key, dotted_keys=()):
    if dotted_keys and key.startswith(dotted_keys):
        return {key}
    aliases = {key}
    for action in parser._actions:
        if action.dest == key:
            aliases |= {o.lstrip("-") for o in action.option_strings}
    return aliases


def _option_name(parser, key):
    for action in parser._actions:
        if action.dest == key or key in {o.lstrip("-") for o in action.option_strings}:
            for option in action.option_strings:
                if option.startswith("--"):
                    return option.lstrip("-")
            if action.option_strings:
                return action.option_strings[0].lstrip("-")
    return key


def _flag_only(parser, key):
    return any(
        (
            action.dest == key
            or _option_name(parser, key)
            in {o.lstrip("-") for o in action.option_strings}
        )
        and isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
        for action in parser._actions
    )


def _given(argv):
    given = set()
    for arg in argv:
        if arg == "--":
            break
        if arg.startswith("--"):
            given.add(arg[2:].split("=", 1)[0])
        elif arg.startswith("-") and len(arg) > 1:
            given.add(arg[:2].lstrip("-"))
    return given


def _truthy(value):
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return None


def apply_perfconfig(
    parser,
    argv,
    command,
    config_path=None,
    *,
    section=None,
    options=None,
    prepend=False,
    skip=None,
    dotted_keys=(),
):
    options = options or parser
    options.set_defaults(_output_from_config=False)
    entries = perfconfig_entries(section or command, config_path)
    if not entries:
        return list(argv)
    argv = list(argv)
    given = _given(argv)
    pending = []
    for key, value in entries.items():
        if skip is not None and skip(key, value, command, given):
            continue
        if key == "event" and "list" in given and not ({"e", "event"} & given):
            continue
        if key == "output":
            template = output_template(value)
            if template is not None:
                options.set_defaults(_output_template=template)
            if not _known(options, key, dotted_keys) or (
                _aliases(options, key, dotted_keys) & given
            ):
                continue
            options.set_defaults(_output_from_config=True)
            pending.append(f"--output={value}")
            continue
        if not _known(options, key, dotted_keys) or (
            _aliases(options, key, dotted_keys) & given
        ):
            continue
        if _flag_only(options, key):
            on = _truthy(value)
            if on is None:
                parser.error(
                    f"invalid [{command}] {key} = {value!r}; expected a boolean "
                    "(true/false)"
                )
            if on:
                pending.append(f"--{_option_name(options, key)}")
            continue
        pending.append(f"--{_option_name(options, key)}={value}")
    if not pending:
        return list(argv)
    if prepend:
        return [argv[0], *pending, *argv[1:]] if argv else pending
    return [*pending, *argv]
