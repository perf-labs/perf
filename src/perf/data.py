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

import subprocess
from pathlib import Path

import pandas as pd

from . import info as _info

_PERFILE1 = b"PERFILE1"
_PERFILE2 = b"PERFILE2"
_FIELDS = "comm,pid,time,period,event,ip,sym,dso"


def is_record(path):
    if isinstance(path, str):
        path = Path(path)
    try:
        with path.open("rb") as f:
            return f.read(8) in (_PERFILE1, _PERFILE2)
    except OSError:
        return False


def samples(path, bin=None, fields=_FIELDS):
    bin = bin or _info.bin()
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
    events = [e for e in events if df["event"].eq(e).any()]
    if not events:
        return df
    keys = [c for c in df.columns if c not in ("event", "period")]
    if not keys:
        return df
    wide = df.pivot_table(index=keys, columns="event", values="period", aggfunc="max")
    return wide.reset_index()[keys + [c for c in wide.columns if c in events]]


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
