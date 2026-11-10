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
import io
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

from .core import _DURATION_TIME

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
    text = str(column)
    return text == _DURATION_TIME or text.startswith(f"{_DURATION_TIME}/")


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
