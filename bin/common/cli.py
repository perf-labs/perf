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
import io
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import pandas as pd

import perf
from perf.arch import x86_64 as _x86_arch
from perf.bench import (
    _BACKEND_OPTION_KEYS,
    _BACKENDS,
    _BRANCH_CHOICES,
    _DEFAULT_BENCH,
    _LIST_TOPS,
    _VALID_TOP_KEYS,
    _as_records,
    _branch_value,
    _canonical_data_reg_key,
    _deep_merge,
    _mem_entries,
    _merge_data,
    _normalize_spec,
    _order_bench_columns,
    normalize_container,
    parse_code,
    validate_spec,
)
from perf.core import _DURATION_TIME, _is_mem_addr_key, one_line
from perf.core import (
    _split_groups as _core_split_groups,
)
from perf.core import (
    _split_list as _core_split_list,
)
from perf.data import _IDENTITY_COLUMNS, _NON_METRIC_COLUMNS, query, quote_columns

try:
    __import__("matplotlib_backend_sixel")
    matplotlib.use("module://matplotlib_backend_sixel")
except ImportError:
    matplotlib.use("Agg")

pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

perf.plot.config = None

DEFAULT_GROUPBY = list(_IDENTITY_COLUMNS)
DEFAULT_COLUMNS = ["time", "file", "name", "mode", "samples"]
DEFAULT_STATS = ["min", "median", "p10", "p50", "p90", "p99", "max"]
SOURCE_COL = "__source__"
PERFCONFIG_NAME = ".perfconfig"
PERFCONFIG_COMMANDS = (
    "benchmark",
    "analyze",
    "view",
    "plot",
    "compare",
    "info",
    "profile",
)
PERFCONFIG_TARGET_COMMANDS = ("analyze", "benchmark", "plot")
PERFCONFIG_COMMENTS = ("#", ";")

_DURATION_RE = re.compile(r"^([-+]?(\d+(?:\.\d*)?|\.\d+))(ns|us|ms|s)$")
_DURATION_SCALE = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
_TABLE_EMPTY = frozenset({"", "nan", "none", "nat", "null"})
_TABLE_EXTS = frozenset({".txt", ".tsv", ".csv"})
_BASELINE_TMP = "__perf_base"
_BASELINE_RE = re.compile(
    r"""(['"])(?P<key>.+?)\1\s*\.\s*(?P<metric>[A-Za-z_][A-Za-z0-9_\-]*)"""
)
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


def embed(ns):
    def help(_line=None):
        print("""
    df
    df.cycles
    df.groupby(["name", "mode"]).cycles.describe()
    df.ecdf()
    %notebook session.ipynb""")

    from IPython.terminal.embed import InteractiveShellEmbed

    ipshell = InteractiveShellEmbed(user_ns=ns, display_banner=False)
    ipshell.editing_mode = "vi"
    ipshell.register_magic_function(help, magic_kind="line", magic_name="help")
    ipshell()


def is_duration_col(c):
    return str(c) == _DURATION_TIME or str(c).startswith(f"{_DURATION_TIME}/")


def to_float(v):
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def is_finite_number(v):
    return to_float(v) is not None


def format_duration(v):
    ns = to_float(v)
    if ns is None:
        try:
            float(v)
            return ""
        except (TypeError, ValueError):
            return "" if _blank(v) else str(v)
    if ns < 0:
        return f"{ns:.2f}"
    if ns < 1e3:
        return f"{ns:.2f}ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f}us"
    if ns < 1e9:
        return f"{ns / 1e6:.2f}ms"
    return f"{ns / 1e9:.2f}s"


def format_number(v):
    f = to_float(v)
    if f is None:
        try:
            float(v)
            return ""
        except (TypeError, ValueError):
            return "" if _blank(v) else str(v)
    return f"{f:.2f}"


def _blank(v):
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def format_table(df):
    out = df.copy()
    for c in out.columns:
        if c in (
            "stat",
            "file",
            "name",
            "mode",
            "time",
            "samples",
            "iterations",
            "operations",
            "index",
            "address",
            "size",
        ):
            continue
        if is_duration_col(c):
            out[c] = [format_duration(v) for v in out[c]]
        else:
            try:
                if pd.api.types.is_numeric_dtype(out[c]):
                    out[c] = [format_number(v) for v in out[c]]
            except Exception:
                pass
    try:
        out = out.where(pd.notna(out), "")
    except Exception:
        pass
    return out


def _is_trace_numeric(df, col):
    try:
        if pd.api.types.is_bool_dtype(df[col]):
            return False
    except Exception:
        pass
    try:
        return bool(pd.api.types.is_numeric_dtype(df[col]))
    except Exception:
        return False


def _trace_cell(v):
    if v is None or (not isinstance(v, (list, tuple, dict, set)) and pd.isna(v)):
        return ""
    return one_line(v)


def format_trace_table(df):
    if df is None or not len(df.columns):
        return ""
    cols = list(df.columns)
    strs = {c: [_trace_cell(v) for v in df[c].tolist()] for c in cols}
    numeric = {c: _is_trace_numeric(df, c) for c in cols}
    widths = {c: max([len(str(c))] + [len(s) for s in strs[c]]) for c in cols}
    lines = [
        "  ".join(
            str(c).rjust(widths[c]) if numeric[c] else str(c).ljust(widths[c])
            for c in cols
        ).rstrip()
    ]
    for i in range(len(df)):
        lines.append(
            "  ".join(
                strs[c][i].rjust(widths[c])
                if numeric[c]
                else strs[c][i].ljust(widths[c])
                for c in cols
            ).rstrip()
        )
    return "\n".join(lines)


def metric_columns(df):
    cols = []
    for c in df.columns:
        if str(c).startswith(("data.", "config.")) or c in _NON_METRIC_COLUMNS:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        except Exception:
            continue
    return cols


def data_columns(df, exclude=()):
    try:
        skip = set(exclude or ())
    except Exception:
        skip = set()
    try:
        return [c for c in df.columns if str(c).startswith("data.") and c not in skip]
    except Exception:
        return []


def groups(value):
    return _core_split_groups(value, [_DURATION_TIME])


def split_list(value):
    return _core_split_list(value)


def flat_events(value):
    flat = _core_split_list(value)
    return flat or None


def want_json(args):
    return bool(getattr(args, "json", False))


def single_output(args):
    out = getattr(args, "output", None)
    if isinstance(out, (list, tuple)):
        if len(out) > 1:
            raise SystemExit("only a single -o/--output is supported")
        out = out[0] if out else None
    return out


def resolve_output(args, stage=None):
    out = single_output(args)
    if stage == "object" and out == "-":
        raise SystemExit("-c (object) cannot write to stdout; give -o a.o")
    if out is None or out == "-":
        return None, stage == "asm"
    return out, False


def target_text(target):
    if isinstance(target, (list, tuple)):
        return "..".join(str(p) for p in target)
    return "" if target is None else str(target)


def bench_code(raw):
    try:
        return parse_code(raw)
    except ValueError as ex:
        raise SystemExit(str(ex))


def bench_snippet(raw, file=None, what="setup"):
    if raw is None:
        return None
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    names = []
    for value in values:
        other, target, asm = bench_code(value)
        if other is not None:
            if file is None:
                raise SystemExit(
                    f"{what} {value!r} names a file, but the target is a snippet"
                )
            if other != file:
                raise SystemExit(f"{what} {value!r} must be a target of {file!r}")
        name = target if target is not None else asm
        if name is not None:
            names.append(name)
    return names or None


def label(value):
    text = "" if value is None else str(value)
    if not text.strip() or text.strip().lower() in ("nan", "none", "nat"):
        return ""
    return text


def out_dirname(name, file_label):
    outer = label(file_label) or label(name)
    return Path(outer) / label(name)


def write_envelope(df, file_output, to_stdout):
    text = perf.to_json(df)
    if to_stdout:
        print(text, flush=True)
    if file_output is not None:
        try:
            payload = json.loads(text)
        except Exception:
            payload = {}
        rec = _as_records(df)
        try:
            file_label = (
                str(rec["file"].iloc[0]) if len(rec) else payload.get("file", "")
            )
        except Exception:
            file_label = payload.get("file", "")
        now = datetime.now()
        dname = out_dirname(payload.get("name", ""), file_label)
        path = (
            Path(str(file_output)) / dname / f"{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


def merge_envelope_data(payload, df):
    for key in (*_IDENTITY_COLUMNS, "time"):
        if key in payload and key not in df.columns:
            df[key] = payload[key]
    for key in ("data", "config", "info", "id"):
        if payload.get(key) is not None and key not in df.attrs:
            df.attrs[key] = payload[key]
    return _order_bench_columns(df)


def envelope_frame(payload):
    return pd.DataFrame(
        [perf.spread(row) for row in (payload.get("output") or [])],
    )


def restore_attrs(frames, df):
    try:
        for _fr in frames:
            try:
                _a = getattr(_fr, "attrs", {}) or {}
            except Exception:
                _a = {}
            for _k in ("data", "config", "info", "id"):
                if _k in _a and _k not in getattr(df, "attrs", {}):
                    try:
                        df.attrs[_k] = _a[_k]
                    except Exception:
                        pass
    except Exception:
        pass


def _table_number(text, duration=False):
    if isinstance(text, str) and text.strip().lower() in _TABLE_EMPTY:
        return float("nan")
    if duration:
        m = _DURATION_RE.match(str(text).strip())
        if m:
            return float(m.group(1)) * _DURATION_SCALE[m.group(3)]
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def _token_spans(text):
    spans = []
    start = None
    for i, c in enumerate(text):
        if c.isspace():
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
        sep = "\t" if "\t" in header else ","
        try:
            return pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, engine="python")
        except Exception as ex:
            raise ValueError(f"unrecognized perf result table ({ex})")
    tokens = _token_spans(header)
    if not tokens:
        return pd.DataFrame()
    ends = [e for _, e in tokens]
    starts = [0] + [ends[i - 1] + 1 for i in range(1, len(ends))]
    columns = [header[s:e].strip() for s, e in zip(starts, ends)]
    if any(not c for c in columns):
        raise ValueError("unrecognized perf result table")
    rows = [
        [ln[s:e].strip() if s < len(ln) else "" for s, e in zip(starts, ends)]
        for ln in lines[1:]
    ]
    if not rows:
        raise ValueError("perf result table has no rows")
    data = {c: [r[i] for r in rows] for i, c in enumerate(columns)}
    df = pd.DataFrame(data, columns=columns)
    for col in df.columns:
        is_duration = is_duration_col(col)
        df[col] = [_table_number(v, is_duration) for v in df[col].tolist()]
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass
    return df


def _read_result_paths(paths):
    def _is_data(name):
        return name.startswith("perf.data") or name.endswith(".data")

    def _is_table(name):
        return Path(name).suffix.lower() in _TABLE_EXTS

    data = []
    for p in map(Path, paths):
        if p.is_file():
            data.append((p, True))
        else:
            for f in p.rglob("*"):
                try:
                    if not f.is_file():
                        continue
                except OSError:
                    continue
                name = f.name
                if name.endswith(".json") or _is_data(name) or _is_table(name):
                    data.append((f, False))
    return data


def load(args, stdin=True):
    def _tag_source(df, path):
        try:
            df[SOURCE_COL] = Path(path).name if path is not None else "<stdin>"
        except Exception:
            pass
        return df

    def _read_envelope_json(path, strict=True):
        with open(path) as fh:
            text = fh.read()
        try:
            payload = json.loads(text)
        except ValueError:
            return _read_table_text(text, path, strict)
        if not isinstance(payload, dict) or "output" not in payload:
            return _read_table_text(text, path, strict)
        df = envelope_frame(payload)
        return _tag_source(merge_envelope_data(payload, df), path)

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
        if df is None or getattr(df, "empty", True):
            if not strict:
                return None
            raise SystemExit(f"no data rows in {path!r}")
        return _tag_source(df, path)

    if stdin and not sys.stdin.isatty():
        try:
            raw = sys.stdin.read()
        except Exception:
            raw = ""
        if raw and raw.strip():
            try:
                payload = json.loads(raw)
            except ValueError as ex:
                try:
                    return table_to_df(raw)
                except Exception:
                    raise SystemExit(
                        f"failed to parse piped data "
                        f"(expected a JSON envelope or a perf result table): {ex}"
                    )
            if not isinstance(payload, dict) or "output" not in payload:
                raise SystemExit("piped JSON must be an envelope with 'output'")
            df = envelope_frame(payload)
            return _tag_source(merge_envelope_data(payload, df), None)
    if not args.data:
        raise SystemExit("no data paths; use '-- data/' or pipe JSON via stdin")
    files = _read_result_paths(args.data)
    if not files:
        raise SystemExit(f"no data files found in {args.data!r}")
    records, bench_json = [], []
    for f, explicit in files:
        try:
            is_rec = perf.is_record(f)
        except Exception:
            is_rec = False
        (records if is_rec else bench_json).append((f, explicit))
    frames = []
    if bench_json:
        _env_frames = []
        for f, explicit in bench_json:
            frame = _read_envelope_json(f, strict=explicit)
            if frame is None:
                print(
                    f"skipping {f}: not a perf envelope or result table",
                    file=sys.stderr,
                )
                continue
            _env_frames.append(frame)
        if not _env_frames:
            raise SystemExit(f"no readable data in {args.data!r}")
        _merged = pd.concat(_env_frames, ignore_index=True)
        restore_attrs(_env_frames, _merged)
        frames.append(_merged)
    if records:
        frames.append(perf.parse([f for f, _ in records]))
    if not frames:
        raise SystemExit(f"no readable data in {args.data!r}")
    _out = pd.concat(frames, ignore_index=True)
    restore_attrs(frames, _out)
    return _out


def load_paths(paths):
    return [load(SimpleNamespace(data=[path]), stdin=False) for path in paths]


def present_columns(df, spec, default):
    rec = _as_records(df)
    return [
        c
        for c in _core_split_list(default if spec is None else spec)
        if c in rec.columns
    ]


def resolve_groupby(df, groupby):
    return present_columns(df, groupby, ",".join(DEFAULT_GROUPBY))


def resolve_columns(df, columns):
    return present_columns(df, columns, ",".join(DEFAULT_COLUMNS))


def _find_baseline_rows(df, key):
    for col in ("file", "name"):
        if col in df.columns:
            try:
                hit = df[col].astype(str) == key
            except Exception:
                continue
            if bool(hit.any()):
                return df[hit], col
    return None, None


def _baseline_values(df, key, metric):
    base, _ = _find_baseline_rows(df, key)
    if base is None:
        raise ValueError(f"baseline group {key!r} not found in 'file'/'name'")
    if metric not in df.columns:
        raise ValueError(f"baseline metric {metric!r} not found in data")
    if "samples" in df.columns and "samples" in base.columns:
        mapping = base.groupby("samples")[metric].mean()
        return df["samples"].map(mapping)
    return float(base[metric].mean())


def _inject_baselines(df, expr, counter):
    temps = []

    def _sub(m):
        key, metric = m.group("key"), m.group("metric")
        values = _baseline_values(df, key, metric)
        safe = (
            "".join(c if c.isalnum() or c == "_" else "_" for c in metric) or "metric"
        )
        tmp = f"{_BASELINE_TMP}_{safe}_{counter[0]}"
        counter[0] += 1
        while tmp in df.columns:
            tmp = f"{tmp}_"
        df[tmp] = values
        temps.append(tmp)
        return tmp

    try:
        rewritten = _BASELINE_RE.sub(_sub, expr)
    except ValueError:
        for t in temps:
            if t in df.columns:
                del df[t]
        raise
    return rewritten, temps


def derived_metrics(df):
    try:
        is_num = {c: pd.api.types.is_numeric_dtype(df[c]) for c in df.columns}
    except Exception:
        return []
    derived = []
    if is_num.get("operations"):
        for c in metric_columns(df):
            if is_num.get(c):
                derived.append(f"{c}/operations")
    if is_num.get("cycles") and is_num.get("instructions"):
        derived.append("instructions/cycles")
    return derived


def eval_events(df, events):
    df = df.copy()
    if any(n in (df.index.names or []) for n in _IDENTITY_COLUMNS):
        df = df.reset_index()
    events = flat_events(events) or (metric_columns(df) + derived_metrics(df))
    if not events:
        raise SystemExit("no metric columns found in data")
    out = []
    counter = [0]
    created = []
    for e in events:
        if e in df.columns:
            out.append(e)
            continue
        try:
            rewritten, temps = _inject_baselines(df, e, counter)
            created.extend(temps)
            rewritten = quote_columns(df, rewritten)
        except ValueError as ex:
            print(
                f"Warning: failed to evaluate expression {e!r}: {ex}", file=sys.stderr
            )
            continue
        try:
            df[e] = df.eval(rewritten, engine="python")
            out.append(e)
        except Exception as ex:
            print(
                f"Warning: failed to evaluate expression {e!r}: {ex}", file=sys.stderr
            )
    for t in created:
        if t in df.columns:
            del df[t]
    if not out:
        raise SystemExit("no metric columns found in data")
    return df, out


def apply_filter(df, filter):
    if filter:
        try:
            return query(df, filter)
        except Exception as ex:
            raise SystemExit(f"invalid --filter {filter!r}: {ex}")
    return df


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
        return float(s)
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
    if v.startswith("{") and v.endswith("}"):
        body = v[1:-1].strip()
        if not body:
            return {}
        d = {}
        for item in body.split(","):
            if ":" not in item:
                raise ValueError(f"invalid dict item: {item!r}")
            k, val = item.split(":", 1)
            d[_unquote(k)] = parse_value(val.strip())
        return d
    if v.startswith("[") and v.endswith("]"):
        return [_parse_atom(x) for x in _split_seq(v[1:-1])]
    if "," in v:
        return [x.strip() for x in v.split(",") if x.strip()]
    if ":" in v and not v.startswith("[") and "," not in v:
        k, val = v.split(":", 1)
        return {k.strip(): parse_value(val.strip())}
    return _parse_scalar(v)


def discover_perfconfig(name=PERFCONFIG_NAME):
    try:
        cur = Path.cwd().resolve()
    except OSError:
        return None
    try:
        home = Path.home().resolve()
    except Exception:
        home = None
    while True:
        cand = cur / name
        try:
            if cand.is_file():
                return cand
        except OSError:
            pass
        if home is not None and cur == home:
            break
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return None


def load_perfconfig(path=None):
    if path is not None:
        p = Path(os.path.expanduser(path))
    else:
        found = discover_perfconfig()
        if found is None:
            return {}
        p = found
    if not p.is_file():
        return {}
    cp = configparser.ConfigParser(
        interpolation=None,
        strict=False,
        comment_prefixes=PERFCONFIG_COMMENTS,
        inline_comment_prefixes=PERFCONFIG_COMMENTS,
    )
    try:
        cp.read(p, encoding="utf-8")
    except (configparser.Error, UnicodeDecodeError, OSError):
        return {}
    return {
        sec: {
            key: value.strip()
            for key, value in cp.items(sec)
            if key.strip() and value.strip()
        }
        for sec in cp.sections()
    }


def _present_options(argv):
    present = set()
    for a in argv:
        if a == "--":
            break
        if a.startswith("--"):
            present.add(a[2:].split("=", 1)[0])
        elif a.startswith("-") and len(a) > 1:
            present.add(a[:2].lstrip("-"))
    return present


def _parser_actions(parser):
    yield from getattr(parser, "_actions", ())


def _option_aliases(parser, key):
    if key.startswith(("config.", "data.")):
        return {key}
    aliases = {key}
    for action in _parser_actions(parser):
        if action.dest == key:
            aliases |= {o.lstrip("-") for o in action.option_strings}
    return aliases


def _key_applies(parser, command, key):
    if key.startswith(("config.", "data.")):
        return command in PERFCONFIG_TARGET_COMMANDS
    return any(
        action.dest == key or key in {o.lstrip("-") for o in action.option_strings}
        for action in _parser_actions(parser)
    )


def _flag_only(parser, key):
    return any(
        action.dest == key
        and isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
        for action in _parser_actions(parser)
    )


def _flag_value(value):
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return None


def apply_perfconfig(parser, command, argv, config_path=None):
    argv = list(argv)
    cfg = load_perfconfig(config_path)
    if not cfg:
        return argv
    entries = {}
    order = []
    for sec in ("default", command):
        if sec not in cfg:
            continue
        for key, value in cfg[sec].items():
            if key not in entries:
                order.append(key)
            entries[key] = value
    if not entries:
        return argv
    present = _present_options(argv)
    pending = []
    for key in order:
        if key in present:
            continue
        if any(a in present for a in _option_aliases(parser, key)):
            continue
        if not _key_applies(parser, command, key):
            continue
        if _flag_only(parser, key):
            on = _flag_value(entries[key])
            if on is None:
                where = config_path
                if where is None:
                    try:
                        where = str(discover_perfconfig() or PERFCONFIG_NAME)
                    except Exception:
                        where = PERFCONFIG_NAME
                parser.error(
                    f"invalid [{command}] {key} = {entries[key]!r} in {where}; "
                    "expected a boolean (true/false)"
                )
            if on:
                pending.append(f"--{key}")
            continue
        pending.append(f"--{key}={entries[key]}")
    if not pending:
        return argv
    return pending + argv


def apply_perfconfig_argv(parser, command, argv, config_path=None):
    if command not in PERFCONFIG_COMMANDS:
        return list(argv)
    return apply_perfconfig(parser, command, argv, config_path)


def split_after_dashes(argv):
    if "--" in argv:
        sep = argv.index("--")
        return argv[:sep], argv[sep + 1 :]
    return list(argv), []


def parse_target_args(parser, argv):
    args, unknown = parser.parse_known_args(argv)

    config = json.loads(json.dumps(_DEFAULT_BENCH))

    def _parse_cli_value(v):
        try:
            return parse_value(v)
        except ValueError as ex:
            parser.error(str(ex))

    def _store_mem_range(base_str, val, rest_hint=""):
        entries = list(_mem_entries(base_str, val))
        if not entries:
            parser.error(f"unknown data key: --data{rest_hint or base_str}")
            return
        for k, v in entries:
            data["mem"][k] = v

    def _apply_data_override(rest, val):
        bracket = rest.find("[")
        if bracket != -1 and rest.endswith("]"):
            ns = rest[:bracket]
            inner = rest[bracket + 1 : -1]
            if ns == "":
                if _is_mem_addr_key(inner):
                    _store_mem_range(inner, val, rest)
                else:
                    data["regs"][_canonical_data_reg_key(inner)] = val
                return
            parser.error(
                f"unknown data key: --data{rest} "
                "(use --data.rdi=.., --data.0x1000=.., --data[0x1000]=..)"
            )
            return
        if "." in rest:
            parser.error(
                f"unknown data key: --data.{rest} "
                "(use --data.rdi=.., --data.0x1000=.., --data[0x1000]=..)"
            )
            return
        if _is_mem_addr_key(rest):
            _store_mem_range(rest, val, rest)
            return
        data["regs"][_canonical_data_reg_key(rest)] = val

    def _preset_config(loaded):
        if "data" in loaded:
            parser.error(
                "config file must not contain 'data'; use --data instead (e.g. "
                "--data.rdi=15 or --data '{\"rdi\": 15}')"
            )
        return _normalize_spec(_deep_merge(config, loaded))

    def _expand_intermediate(top, cur):
        if top in ("dcache", "icache", "dtlb", "itlb") and isinstance(cur, str):
            table = (
                _x86_arch._MEMORY_SHORTCUTS
                if top == "dcache"
                else _x86_arch._TIER_SHORTCUTS
            )
            sc = table.get(cur.strip().lower())
            if sc is None:
                parser.error(
                    f"unknown cache shortcut {cur!r}; expected one of "
                    "'hot', 'warm', 'cool', 'cold'"
                )
            if top == "dcache":
                return dict(sc)
            tier = {
                "icache": "L1i",
                "dtlb": "TLBd",
                "itlb": "TLBi",
            }[top]
            return {tier: sc[tier]}
        if top == "branch" and isinstance(cur, (str, bool)):
            if isinstance(cur, bool):
                pred = "predictable" if cur else "unpredictable"
            else:
                pred = _branch_value(cur)
                if pred is None:
                    parser.error(
                        "branch prediction must be one of "
                        f"{', '.join(_BRANCH_CHOICES)}, got {cur!r}"
                    )
            return {"prediction": pred}
        return {}

    def _container_top(top):
        return top in _LIST_TOPS or top == "thread"

    def _pin_container(key):
        cur = config.get(key)
        if not isinstance(cur, list):
            return cur if isinstance(cur, dict) else {}
        first = cur[0] if cur else None
        while isinstance(first, list) and first:
            first = first[0]
        entry = dict(first) if isinstance(first, dict) else {}
        config[key] = [[entry]] if key == "thread" else [entry]
        return entry

    def _check_top(top):
        if top not in _VALID_TOP_KEYS:
            parser.error(
                f"unknown argument: --config.{top} "
                f"(expected one of {', '.join(sorted(_VALID_TOP_KEYS))})"
            )

    def _check_nested(top, leaf, parts=()):
        if top == "thread" and leaf not in ("affinity", "priority", "numa"):
            parser.error(
                f"unknown argument: --config.thread.{leaf} "
                "(expected 'affinity', 'numa' or 'priority')"
            )
        if top == "func" and leaf not in ("align", "order"):
            parser.error(
                f"unknown argument: --config.func.{leaf} (expected 'align' or 'order')"
            )
        if top == "code" and leaf not in ("align",):
            parser.error(f"unknown argument: --config.code.{leaf} (expected 'align')")
        if top == "stack" and leaf not in ("size", "align"):
            parser.error(
                f"unknown argument: --config.stack.{leaf} (expected 'size' or 'align')"
            )
        if top == "iterations" and leaf not in ("min", "max", "count"):
            parser.error(
                f"unknown argument: --config.iterations.{leaf} "
                "(expected 'min', 'max' or 'count')"
            )
        if top == "external" and leaf not in ("lib",):
            parser.error(f"unknown argument: --config.external.{leaf} (expected 'lib')")
        if top == "backend":
            name = leaf
            if name not in _BACKENDS:
                parser.error(
                    f"unknown argument: --config.backend.{name} (expected "
                    f"{' or '.join(_BACKENDS)}, or one of "
                    f"{', '.join(sorted(_BACKEND_OPTION_KEYS))})"
                )
            if len(parts) >= 3 and parts[2] not in _BACKEND_OPTION_KEYS:
                parser.error(
                    f"unknown argument: --config.backend.{name}.{parts[2]} "
                    f"(expected one of {', '.join(sorted(_BACKEND_OPTION_KEYS))})"
                )
            if len(parts) > 3:
                parser.error(f"unknown argument: --config.{'.'.join(parts)}")

    def _apply_config_override(rest, value):
        bracket = rest.find("[")
        if bracket != -1 and rest.endswith("]"):
            key = rest[:bracket]
            addr = rest[bracket + 1 : -1]
            parts = [p for p in key.split(".") if p]
            if not parts:
                parser.error(f"unknown argument: --config.{rest}")
            _check_top(parts[0])
            if len(parts) >= 2:
                _check_nested(parts[0], parts[1], parts)
            top = parts[0]
            leaf = parts[-1]
            if top not in ("branch", "dcache", "icache", "dtlb", "itlb"):
                parser.error(
                    f"per-address overrides --config.{rest} are only supported "
                    "for branch and the cache/tlb keys "
                    "(e.g. --config.dcache.L1d[0x1000]=hot)"
                )
            d = config
            for p in parts[:-1]:
                nxt = d.get(p)
                if not isinstance(nxt, dict):
                    if p == parts[0] and nxt is not None:
                        nxt = _expand_intermediate(p, nxt)
                    else:
                        nxt = {} if nxt is None else _expand_intermediate(top, nxt)
                        if not isinstance(nxt, dict):
                            nxt = {}
                    d[p] = nxt
                d = nxt
            cur = d.get(leaf)
            base = dict(cur) if isinstance(cur, dict) else {}
            base[addr] = value
            d[leaf] = base
            return
        d = config
        parts = [p for p in rest.split(".") if p]
        if not parts:
            parser.error(f"unknown argument: --config.{rest}")
        _check_top(parts[0])
        if len(parts) >= 2:
            _check_nested(parts[0], parts[1], parts)
        if _container_top(parts[0]):
            if len(parts) == 1:
                config[parts[0]] = normalize_container(parts[0], value)
                return
            d = _pin_container(parts[0])
            for p in parts[1:-1]:
                nxt = d.get(p)
                if not isinstance(nxt, dict):
                    nxt = {}
                    d[p] = nxt
                d = nxt
            if isinstance(value, list) and parts[0] in ("code", "stack", "func"):
                base = dict(d)
                leaf = parts[-1]
                config[parts[0]] = [{**base, leaf: v} for v in value] or [{**base}]
                return
            d[parts[-1]] = value
            return
        for p in parts[:-1]:
            nxt = d.get(p)
            if not isinstance(nxt, dict):
                nxt = {} if nxt is None else _expand_intermediate(parts[0], nxt)
                d[p] = nxt
            d = nxt
        d[parts[-1]] = value

    data = {"regs": {}, "mem": {}}
    if getattr(args, "config", None):
        with open(args.config) as f:
            config = _preset_config(json.load(f))

    for arg in unknown:
        if "=" not in arg:
            parser.error(f"unknown argument: {arg}")
        if arg.startswith("--data["):
            key, value = arg[len("--data") :].split("=", 1)
            if not key:
                parser.error(f"unknown argument: {arg}")
            _apply_data_override(key, _parse_cli_value(value))
            continue
        if arg.startswith("--data."):
            key, value = arg[len("--data.") :].split("=", 1)
            if not key:
                parser.error(f"unknown argument: {arg}")
            _apply_data_override(key, _parse_cli_value(value))
            continue
        if arg.startswith("--config."):
            key, value = arg[9:].split("=", 1)
            if not key:
                parser.error(f"unknown argument: {arg}")
            _apply_config_override(key, _parse_cli_value(value))
        else:
            parser.error(f"unknown argument: {arg}")

    flag_data = None
    if getattr(args, "data", None):
        raw_data = args.data
        if os.path.isfile(raw_data):
            with open(raw_data) as f:
                flag_data = json.load(f)
        else:
            try:
                flag_data = json.loads(raw_data)
            except json.JSONDecodeError as ex:
                parser.error(f"invalid --data JSON: {ex}")
    try:
        validate_spec(config)
    except ValueError as ex:
        parser.error(str(ex))
    args.config = config
    has_data = bool((data.get("regs") or data.get("mem")) or flag_data)
    args.data = _merge_data(data, flag_data) if has_data else None
    return args
