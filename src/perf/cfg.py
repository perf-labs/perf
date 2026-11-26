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

import contextlib
import copy
import ctypes
import hashlib
import itertools
import mmap
import numbers
import os
import random
import re
import warnings

import numpy as np
import pandas as pd

from .arch import arch as get_arch
from .core import (
    _ASM_SCRATCH_BASE,
    _DURATION_TIME,
    _ENTRY_DATA_BASE,
    _FALLBACK_DATA_BASE,
    _FALLBACK_DATA_STRIDE,
    _FALLBACK_GROWN_PAGES,
    _FALLBACK_MAX_DOUBLINGS,
    _FALLBACK_START_PAGES,
    _FALLBACK_STEER_MAX,
    _MPOL_BIND,
    _MPOL_DEFAULT,
    _OUTPUT_LOCK,
    _PLACEMENT_LOCK,
    _STATE_LOCK,
    _UNMAPPABLE_DATA_PAGES,
    _debug_log,
    _fallback_budget_bytes,
    _get_mempolicy,
    _locked,
    _map_data_pages,
    _numa_nodes,
    _parse_range_list,
    _set_mempolicy,
    _silence_fd,
    _split_list,
    _stable,
    _to_int_or,
    _to_u64,
)
from .data import _IDENTITY_COLUMNS
from .exec import (
    _MIN_STACK_SIZE,
    host_symbol,
)
from .info import (
    _asm_read,
    asm_labels,
    is_asm_source as _is_asm_source,
)

_PRIORITY_LEVELS = {
    "lowest": 19,
    "low": 10,
    "normal": 0,
    "high": -10,
    "highest": -20,
}
_THREAD_KEYS = frozenset({"affinity", "priority", "numa"})
_THREAD_HINT = (
    "expected a list of thread alternatives, each a list of threads, e.g. "
    "[[{'affinity': 1, 'numa': 0, 'priority': 'normal'}]]"
)
_VALID_EXTERNAL_KEYS = frozenset({"lib", "stdout", "stderr"})
_VALID_OUTPUT_KEYS = ("stdout", "stderr")
_SCALAR_KEYS = frozenset(
    {
        "samples",
        "iterations",
        "seed",
    }
)
_DEFAULT_ALIGN = 16
_DEFAULT_STACK_SIZE = 0x200000
_DEFAULT_ITER_MIN = 100
_DEFAULT_ITER_MAX = 1_000_000
_FUNC_ORDERS = ("as-is", "random")
_DEFAULT_BENCH = {
    "seed": 0,
    "thread": [
        [
            {"numa": None, "affinity": None, "priority": "normal"},
        ]
    ],
    "branch": ["predictable", "unpredictable"],
    "dcache": ["hot", "warm", "cool", "cold"],
    "dtlb": ["hot", "cold"],
    "icache": ["hot", "cold"],
    "itlb": ["hot", "cold"],
    "code": [
        {"align": 1},
        {"align": 16},
    ],
    "stack": [
        {"size": "auto", "align": "auto", "mode": "random"},
    ],
    "heap": [
        {"size": "auto", "mode": "random"},
    ],
    "func": [
        {"align": _DEFAULT_ALIGN, "order": "as-is"},
    ],
    "samples": 100,
    "iterations": {
        "min": _DEFAULT_ITER_MIN,
        "max": _DEFAULT_ITER_MAX,
    },
    "backend": {
        "loop": {
            "count": 16,
            "probes": 3,
            "runs": 10,
            "target": 0.01,
        },
        "unroll": {
            "count": 5,
            "probes": 3,
            "runs": 10,
            "target": 0.01,
        },
    },
    "external": {
        "lib": False,
        "stdout": False,
        "stderr": False,
    },
}
_BACKENDS = ("loop", "unroll")
_MODES = ("latency", "throughput")
_ENTRY_LABELS = frozenset({"main", "_main", "__main", "wmain", "WinMain"})
_VALID_TOP_KEYS = frozenset(_DEFAULT_BENCH.keys())
_VALID_FUNC_KEYS = frozenset({"align", "order"})
_VALID_CODE_KEYS = frozenset({"align"})
_VALID_STACK_KEYS = frozenset({"size", "align", "mode"})
_VALID_HEAP_KEYS = frozenset({"size", "mode"})
_VALID_ITERATIONS_KEYS = frozenset({"min", "max"})
_BACKEND_OPTION_KEYS = frozenset({"probes", "runs", "target", "count"})
_CONTAINER_TOPS = frozenset({"iterations", "external"})
_LIST_TOPS = frozenset({"code", "stack", "heap", "func"})
_THREAD_TOP = "thread"
_LIST_TOP_KEYS = {
    "code": _VALID_CODE_KEYS,
    "stack": _VALID_STACK_KEYS,
    "heap": _VALID_HEAP_KEYS,
    "func": _VALID_FUNC_KEYS,
}
_LIST_TOP_HINT = {
    "code": '[{"align": 1}, {"align": 16}]',
    "stack": '[{"size": "auto", "align": "auto", "mode": "random"}]',
    "heap": '[{"size": "auto", "mode": "random"}]',
    "func": '[{"align": 16, "order": "as-is"}]',
}
_ENTRY_DEFAULTS = {
    "code": {"align": _DEFAULT_ALIGN},
    "stack": {"size": "auto", "align": "auto", "mode": "random"},
    "heap": {"size": "auto", "mode": "random"},
    "func": {"align": _DEFAULT_ALIGN, "order": _FUNC_ORDERS[0]},
}
_CACHE_KEYS = ("dcache", "icache", "dtlb", "itlb")
_PRUNE_KEYS = ("branch", "dcache", "dtlb")
_BRANCH_CHOICES = (
    "predictable",
    "unpredictable",
)
_BENCH_KEEP_META = frozenset({"operations", "threads"})
_COMBINE_META = ("operations", "threads")
_CONFIG_COLUMNS_CACHE = {}
_DATA_COLUMNS_CACHE = {}
_COLUMNS_CACHE_MAX = 256
_CONTAINER_RR = {}
_STACK_REFRESH_WINDOW = 1024
_STACK_REFRESH_BYTES_MAX = 4 << 20
_REG_LEVEL_WINDOW = 0x1000
_HEAP_RANDOM_LIMIT = 0x1000
_HEAP_RANDOM_LIMIT_MAX = 0x10000
_HEAP_SEED_SALT = 0x9E3779B97F4A7C15
_ASM_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*|#[^\n]*", re.S)
_ASM_LABEL = re.compile(r"[.\w$@]+")


def normalize_container(key, value):
    if value is None:
        return None
    if key == _THREAD_TOP:
        if isinstance(value, dict):
            return [[dict(value)]]
        if isinstance(value, (list, tuple)):
            items = list(value)
            if items and all(isinstance(v, dict) for v in items):
                return [[dict(v)] for v in items]
            return [
                [dict(t) if isinstance(t, dict) else t for t in group]
                if isinstance(group, (list, tuple))
                else group
                for group in items
            ]
        return value
    if isinstance(value, (list, tuple)):
        return [dict(v) if isinstance(v, dict) else _list_entry(key, v) for v in value]
    return [_list_entry(key, value)]


def data_param_columns(data, include_mem=True):
    try:
        key = (_stable(data), bool(include_mem))
    except Exception:
        key = None
    if key is not None:
        with _STATE_LOCK:
            hit = _DATA_COLUMNS_CACHE.get(key)
        if hit is not None:
            return _copy_columns(hit)
    cols = {}

    canon_regs = {}
    for reg, val in ((data or {}).get("regs") or {}).items():
        v = _param_columns_value(val)
        if v is None:
            continue
        if _is_scratch_value(val):
            continue
        canon = _canonical_data_reg_key(reg)
        canon_regs[canon] = v
    for reg, v in canon_regs.items():
        cols[f"data.{reg}"] = v
    if include_mem:
        mem = (data or {}).get("mem") or {}
        for addr, val in mem.items():
            v = _param_columns_value(val)
            if v is None:
                continue
            if _heap_covers(data, addr):
                continue

            if f"data.{addr}" not in cols:
                cols[f"data.{addr}"] = v
    if key is not None:
        with _STATE_LOCK:
            if len(_DATA_COLUMNS_CACHE) >= _COLUMNS_CACHE_MAX:
                _DATA_COLUMNS_CACHE.clear()
            _DATA_COLUMNS_CACHE[key] = _copy_columns(cols)
    return cols


def data_param_value(data, key):
    if not isinstance(key, str) or not key.startswith("data."):
        return None
    rest = key[len("data.") :]
    if (
        rest.startswith("regs.")
        or rest.startswith("mem.")
        or rest.startswith("memory.")
    ):
        return None
    regs = (data or {}).get("regs") or {}
    index = {}
    for k, v in regs.items():
        if v is None:
            continue
        canon = _canonical_data_reg_key(k)
        index[canon] = v
    try:
        canon = _canonical_data_reg_key(rest)
    except Exception:
        canon = rest
    for cand in (canon, rest):
        if cand in index:
            v = _first_scalar(index[cand])
            return v
    mem = (data or {}).get("mem") or {}
    if rest in mem:
        return _first_scalar(mem[rest])
    return None


def config_param_columns(config):
    from .plot import normalize_config

    try:
        key = _stable(config)
    except Exception:
        key = None
    if key is not None:
        with _STATE_LOCK:
            hit = _CONFIG_COLUMNS_CACHE.get(key)
        if hit is not None:
            return _copy_columns(hit)
    flat = normalize_config(config)
    cols = {f"config.{k}": v for k, v in flat.items()}
    if key is not None:
        with _STATE_LOCK:
            if len(_CONFIG_COLUMNS_CACHE) >= _COLUMNS_CACHE_MAX:
                _CONFIG_COLUMNS_CACHE.clear()
            _CONFIG_COLUMNS_CACHE[key] = _copy_columns(cols)
    return cols


def validate_spec(config):
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {config!r}")
    for key in config:
        if key == "data":
            raise ValueError(
                "config must not contain 'data'; pass data separately via data={...}"
            )
        if key not in _VALID_TOP_KEYS:
            raise ValueError(
                f"unknown config key {key!r}; expected one of "
                f"{', '.join(sorted(_VALID_TOP_KEYS))}"
            )
    if config.get("thread") is not None:
        _validate_thread_top(config["thread"])
    for key in _LIST_TOPS:
        value = config.get(key, None)
        if value is None:
            continue
        if isinstance(value, list):
            _validate_list_top(key, value)
        else:
            _validate_list_one(key, value)
    for cache_key in _CACHE_KEYS:
        _validate_cache_key(config, cache_key)
    _validate_lib(config)
    iterations = config.get("iterations")
    if isinstance(iterations, dict):
        for key in iterations:
            if key not in _VALID_ITERATIONS_KEYS:
                raise ValueError(
                    f"unknown iterations key {key!r}; expected one of "
                    f"{', '.join(sorted(_VALID_ITERATIONS_KEYS))}"
                )
        for sub in ("min", "max"):
            value = iterations.get(sub)
            if value is None:
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            if isinstance(value, (list, tuple)) and not values:
                raise ValueError(
                    f"iterations.{sub} must be a non-empty list of alternatives, "
                    f"got {value!r}"
                )
            for item in values:
                try:
                    iv = int(item)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"iterations.{sub} must be an integer, got {value!r}"
                    ) from e
                if iv < 1:
                    raise ValueError(f"iterations.{sub} must be >= 1, got {value!r}")
        lo, hi = iterations.get("min"), iterations.get("max")
        if (
            lo is not None
            and hi is not None
            and not isinstance(lo, (list, tuple))
            and not isinstance(hi, (list, tuple))
            and int(lo) > int(hi)
        ):
            raise ValueError(
                f"iterations.min ({lo!r}) must be <= iterations.max ({hi!r})"
            )
    _validate_backend(config)
    branch = config.get("branch", None)
    if branch is not None and not isinstance(branch, (list, str, bool, dict)):
        raise ValueError(
            f"branch config must be a list of alternatives, a single value, "
            f"or a per-address dict, got {branch!r}"
        )

    def _require_alternatives(path, value):
        if isinstance(value, list) and not value:
            raise ValueError(
                f"config {path} must be a non-empty list of alternatives, got {value!r}"
            )

    for key, value in config.items():
        if key in _SCALAR_KEYS:
            _validate_scalar_key(key, value)
        elif key in _LIST_TOPS and isinstance(value, list):
            for i, entry in enumerate(value):
                if isinstance(entry, dict):
                    for sub, sv in entry.items():
                        if isinstance(sv, (list, tuple)):
                            raise ValueError(
                                f"config {key}[{i}].{sub} must be a single value, "
                                f"got {sv!r}; sweep via multiple {key} "
                                "alternatives instead"
                            )
        elif key in _CONTAINER_TOPS and isinstance(value, dict):
            for sub, sv in value.items():
                _require_alternatives(f"{key}.{sub}", sv)
        else:
            _require_alternatives(key, value)

    if config.get("branch") is not None:
        alts = (
            config["branch"]
            if isinstance(config["branch"], list)
            else [config["branch"]]
        )
        for v in alts:
            if isinstance(v, str):
                if _branch_value(v) is None:
                    choices = ", ".join(_BRANCH_CHOICES)
                    raise ValueError(
                        f"branch prediction must be one of {choices}, got {v!r}"
                    )
            elif isinstance(v, bool):
                continue
            elif isinstance(v, dict):
                for combo in _expand_config_spec({"branch": v}):
                    _branch_config(combo)
            else:
                raise ValueError(
                    f"branch config elements must be a string, bool, or dict, got {v!r}"
                )
    return config


def region(target):
    if isinstance(target, (list, tuple)):
        begin, end = (str(part).strip() for part in target)
        if not begin or not end:
            raise ValueError(f"invalid region {target!r}; expected BEGIN..END")
        return (begin, end)
    text = str(target or "").strip()
    if not text:
        return None
    if ".." not in text:
        return text
    begin, _, end = text.partition("..")
    begin, end = begin.strip(), end.strip()
    if not begin or not end:
        raise ValueError(
            f"invalid symbol or region {target!r}; expected NAME or BEGIN..END"
        )
    return (begin, end)


def parse_code(code):
    if code is None:
        return None, None, None
    if isinstance(code, (list, tuple)):
        parts = list(code)
        if not parts:
            return None, None, None
        file = str(parts[0]).strip()
        if len(parts) == 1:
            return (file or None), None, None
        return (file or None), region(parts[1]), None
    text = str(code).strip()
    if not text:
        return None, None, None
    file, sep, rest = text.partition(":")
    if not sep:
        return (text, None, None) if _is_file(text) else (None, None, text)
    prefix = file.strip()
    if not prefix:
        return None, region(rest), None
    if rest[:1].isspace() and not prefix.startswith("~") and not os.path.exists(prefix):
        if _ASM_LABEL.fullmatch(prefix) and not prefix.lower().endswith(
            (".out", ".so", ".o", ".a", ".exe", ".elf", ".bin", ".dylib", ".dll")
        ):
            return None, None, text
        if prefix.startswith(".") and os.sep not in prefix:
            return None, None, text
        if "." not in prefix and os.sep not in prefix:
            return None, None, text
    if _is_file(prefix):
        return prefix, region(rest), None
    if "\n" in text or "," in prefix or "[" in text:
        return None, None, text
    if any(ch.isspace() for ch in prefix):
        return None, None, text
    return prefix, region(rest), None


def asm_source_target(target):
    if target is None:
        return ""
    if isinstance(target, (list, tuple)):
        parts = [str(part).strip() for part in target]
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2 and all(parts):
            return f"{parts[0]}..{parts[1]}"
        return ""
    return str(target).strip()


def asm_source_code(file, target):
    if not file or not _is_asm_source(file):
        return None
    try:
        code = _asm_region(file, target)
    except Exception:
        return None
    arch = get_arch()
    try:
        if not arch.assemble(arch.normalize_asm(f"{code};"), arch._SETUP_BASE):
            return None
    except Exception:
        return None
    return code


def check_data(data, arch=None):
    if arch is None:
        arch = get_arch()
    reserved = getattr(arch, "_HARNESS_RESERVED_REGS", ()) or ()
    harness = getattr(arch, "_HARNESS_REGS", ()) or ()
    for reg, value in ((data or {}).get("regs") or {}).items():
        try:
            canon = arch.canonical_data_reg(reg)
        except Exception:
            canon = str(reg).strip().lower()
        try:
            canon = arch._canonical_reg(canon)
        except Exception:
            pass
        if canon in reserved:
            raise ValueError(
                f"data register {reg!r} is reserved by the measurement harness "
                f"({', '.join(reserved)}): the loop counter, "
                "its cursors, the stack and the instruction pointer drive the "
                "harness, so writing them would corrupt the measurement"
            )
        if canon in (harness or set()) and isinstance(value, (list, tuple)):
            raise ValueError(
                f"data register {reg!r} is used by the measurement harness "
                "to hold timing baselines, so a per-iteration list would be "
                "silently truncated to its first value; pass a scalar or pick "
                "a different register"
            )
    return data


def _parse_addr_key(key):
    s = str(key).strip()
    is_range = s.endswith(":")
    return (s[:-1].strip() if is_range else s), is_range


def _is_mem_addr_key(key):
    try:
        int(_parse_addr_key(key)[0], 0)
        return True
    except (TypeError, ValueError):
        return False


def _check_thread(entry):
    if not isinstance(entry, dict):
        raise ValueError(f"unknown thread config {entry!r}; {_THREAD_HINT}")
    for key in entry:
        if key not in _THREAD_KEYS:
            raise ValueError(
                f"unknown thread key {key!r}; expected one of "
                f"{', '.join(sorted(_THREAD_KEYS))}"
            )
    return entry


def _thread_entries(thread):
    if thread is None:
        return []
    if isinstance(thread, dict):
        return [_check_thread(thread)]
    if not isinstance(thread, (list, tuple)):
        raise ValueError(f"unknown thread config {thread!r}; {_THREAD_HINT}")
    entries = []
    for group in thread:
        if isinstance(group, dict):
            entries.append(_check_thread(group))
        elif isinstance(group, (list, tuple)):
            if not group:
                raise ValueError(
                    f"config thread alternatives must not be empty: {thread!r}"
                )
            for entry in group:
                entries.append(_check_thread(entry))
        else:
            raise ValueError(f"unknown thread config {group!r}; {_THREAD_HINT}")
    return entries


def _thread_store(config):
    thread = (config or {}).get("thread", None)
    if thread is None:
        entry = {}
        config["thread"] = [entry]
        return entry, True
    entries = _thread_entries(thread)
    if not entries:
        entry = {}
        config["thread"] = [entry]
        return entry, True
    return entries[0], False


def _thread_drop(config, entry):
    thread = config.get("thread", None)
    if isinstance(thread, dict):
        if thread is entry:
            config["thread"] = None
        return
    if not isinstance(thread, list):
        return
    if any(item is entry for item in thread):
        thread = [item for item in thread if item is not entry]
    else:
        thread = [
            [item for item in group if item is not entry]
            if isinstance(group, list)
            else group
            for group in thread
        ]
        thread = [group for group in thread if group != []]
    config["thread"] = thread or None


def _thread_cfg(config):
    thread = (config or {}).get("thread", None)
    if thread is None:
        return {}
    entries = _thread_entries(thread)
    if not entries:
        return {}
    return entries[0]


def _thread_group(config):
    thread = (config or {}).get("thread", None)
    if thread is None:
        return [{}]
    entries = _thread_entries(thread)
    return entries or [{}]


def _affinity_spec(config):
    return _thread_cfg(config).get("affinity", None)


def _priority_spec(config):
    return _thread_cfg(config).get("priority", None)


def _numa_spec(config):
    return _thread_cfg(config).get("numa", None)


def _parse_affinity(spec):
    if spec is None:
        return None
    if isinstance(spec, bool):
        raise ValueError(
            f"unknown affinity {spec!r}; expected a cpu id or a list like [0, 1]"
        )
    if isinstance(spec, str):
        s = spec.strip()
        if s.lower() in ("", "none", "off", "false", "default"):
            return None
        cpus = _parse_range_list(
            s,
            err=(
                f"unknown affinity {spec!r}; expected a cpu id, "
                "a list like [0, 1], or a taskset string like '0,1'"
            ),
        )
        return sorted(cpus) if cpus else None
    if isinstance(spec, numbers.Integral):
        return [int(spec)]
    if isinstance(spec, (list, tuple, set)):
        try:
            cpus = sorted({int(c, 0) if isinstance(c, str) else int(c) for c in spec})
        except (TypeError, ValueError):
            raise ValueError(
                f"unknown affinity {spec!r}; expected a cpu id or a list like [0, 1]"
            )
        if not cpus:
            return None
        return cpus
    raise ValueError(
        f"unknown affinity {spec!r}; expected a cpu id or a list like [0, 1]"
    )


@contextlib.contextmanager
def _affinity_guard(config):
    cpus = _parse_affinity(_affinity_spec(config))
    if cpus is None:
        yield
        return
    with _PLACEMENT_LOCK:
        try:
            old = sorted(os.sched_getaffinity(0))
        except OSError:
            old = None
        if old is not None:
            try:
                os.sched_setaffinity(0, set(cpus))
            except OSError as ex:
                warnings.warn(f"failed to set affinity to {cpus}: {ex}")
                old = None
    if old is None:
        yield
        return
    try:
        yield
    finally:
        with _PLACEMENT_LOCK:
            try:
                os.sched_setaffinity(0, set(old))
            except OSError:
                pass


def _parse_priority(spec):
    if spec is None:
        return None
    if isinstance(spec, bool):
        raise ValueError(f"priority must be an integer nice value, got {spec!r}")
    if isinstance(spec, numbers.Integral):
        return int(spec)
    if isinstance(spec, float):
        try:
            return int(spec)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"priority must be an integer nice value, got {spec!r}"
            ) from e
    if isinstance(spec, str):
        s = spec.strip()
        if s.lower() in ("", "none", "default"):
            return None
        level = _PRIORITY_LEVELS.get(s.lower())
        if level is not None:
            return level
        try:
            return int(s, 0)
        except ValueError:
            raise ValueError(
                f"unknown priority {spec!r}; expected an integer nice value like "
                "1 or -5, or a level like 'lowest', 'low', 'normal', 'high', 'highest'"
            )
    raise ValueError(
        f"unknown priority {spec!r}; expected an integer nice value like "
        "1 or -5, or a level like 'lowest', 'low', 'normal', 'high', 'highest'"
    )


@contextlib.contextmanager
def _priority_guard(config):
    nice = _parse_priority(_priority_spec(config))
    if nice is None:
        yield
        return
    with _PLACEMENT_LOCK:
        old_nice = None
        try:
            old_nice = os.getpriority(os.PRIO_PROCESS, 0)
        except OSError:
            old_nice = None
        try:
            os.setpriority(os.PRIO_PROCESS, 0, int(nice))
        except OSError as ex:
            try:
                warnings.warn(f"failed to set nice to {nice}: {ex}")
            except Exception:
                pass
    try:
        yield
    finally:
        with _PLACEMENT_LOCK:
            if old_nice is not None:
                try:
                    os.setpriority(os.PRIO_PROCESS, 0, int(old_nice))
                except OSError:
                    pass


def _parse_numa(spec):
    if spec is None:
        return None
    if isinstance(spec, str):
        s = spec.strip()
        if s.lower() in ("", "none", "off", "false", "default"):
            return None
        try:
            spec = int(s, 0)
        except ValueError as e:
            raise ValueError(
                f"unknown numa {spec!r}; expected a node id like 0 or 1"
            ) from e
    if isinstance(spec, numbers.Integral) and not isinstance(spec, bool):
        node = int(spec)
        if node < 0:
            raise ValueError(f"numa node must be >= 0, got {spec!r}")
        return node
    raise ValueError(f"unknown numa {spec!r}; expected a node id like 0 or 1")


@contextlib.contextmanager
def _numa_guard(config):
    node = _parse_numa(_numa_spec(config))
    if node is None:
        yield
        return
    nodes = _numa_nodes()
    if nodes and node not in nodes:
        warnings.warn(
            f"numa node {node} is not online (have {', '.join(str(n) for n in nodes)})"
        )
        yield
        return
    if len(nodes) == 1:
        yield
        return
    with _PLACEMENT_LOCK:
        try:
            old_mode = _get_mempolicy()
        except OSError:
            old_mode = _MPOL_DEFAULT
        try:
            if _set_mempolicy(_MPOL_BIND, node) != 0:
                raise OSError(ctypes.get_errno(), "set_mempolicy failed")
        except OSError as ex:
            warnings.warn(f"failed to bind memory to numa node {node}: {ex}")
            old_mode = None
    if old_mode is None:
        yield
        return
    try:
        yield
    finally:
        with _PLACEMENT_LOCK:
            if _set_mempolicy(old_mode) != 0:
                warnings.warn(
                    f"failed to restore the numa memory policy to mode {old_mode}; "
                    "the process stays bound to node "
                    f"{node} for the rest of this run"
                )


def _output_fakes(config):
    out = []
    try:
        if _fake_stdout(config):
            out.append(1)
        if _fake_stderr(config):
            out.append(2)
    except Exception:
        return out
    return out


@contextlib.contextmanager
def _fake_output(config):
    fakes = _output_fakes(config)
    lock = _OUTPUT_LOCK if fakes else contextlib.nullcontext()
    with lock:
        restores = [r for r in (_silence_fd(fd) for fd in fakes) if r]
        try:
            yield
        finally:
            for restore in reversed(restores):
                restore()


def _external_cfg(config):
    return (config or {}).get("external") or {}


def _lib_value(spec):
    if spec is None:
        return False
    if isinstance(spec, str):
        return spec.strip().lower() in ("load", "true", "yes", "on", "1")
    return bool(spec)


def _resolve_lib(config):
    return _lib_value(_external_cfg(config).get("lib"))


def _fake_value(spec):
    if spec is None:
        return True
    if isinstance(spec, str):
        return spec.strip().lower() not in ("real", "true", "yes", "on", "1")
    return not bool(spec)


def _fake_stdout(config):
    return _fake_value(_external_cfg(config).get("stdout"))


def _fake_stderr(config):
    return _fake_value(_external_cfg(config).get("stderr"))


def _validate_lib(config):
    external = (config or {}).get("external")
    if external is None:
        return
    if not isinstance(external, dict):
        raise ValueError(
            f"unknown external config {external!r}; expected a dict like "
            "{'lib': False, 'stdout': False, 'stderr': False}"
        )
    for key in external:
        if key not in _VALID_EXTERNAL_KEYS:
            raise ValueError(
                f"unknown external key {key!r}; expected one of "
                f"{', '.join(sorted(_VALID_EXTERNAL_KEYS))}"
            )
    for key in ("lib",) + _VALID_OUTPUT_KEYS:
        spec = external.get(key)
        if spec is None:
            continue
        for value in _nonempty_values(f"external.{key}", spec):
            if isinstance(value, (bool, int, str)):
                continue
            if key == "lib":
                raise ValueError(
                    f"unknown external.lib config {value!r}; expected a boolean "
                    "(True loads shared libraries, False fakes them)"
                )
            raise ValueError(
                f"unknown external.{key} config {value!r}; expected a boolean "
                f"(False fakes {key}, True lets the target write to it)"
            )


def _nonempty_values(path, value):
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError(
            f"config {path} must be a non-empty list of alternatives, got {value!r}"
        )
    return values


def _canonical_data_reg_key(key):
    return get_arch().canonical_data_reg(key)


def _normalize_data_regs(regs):
    out = {}
    for key, val in (regs or {}).items():
        canon = _canonical_data_reg_key(key)
        if isinstance(val, (list, tuple)):
            out[canon] = [_to_u64(x) for x in val]
        else:
            try:
                ival = int(val, 0) if isinstance(val, str) else int(val)
            except (TypeError, ValueError):
                continue
            out[canon] = ival & 0xFFFFFFFFFFFFFFFF
    return out


def _iter_mem_addrs(data):
    if not data or not data.get("mem"):
        return
    for addr_str in data["mem"]:
        base_str, _ = _parse_addr_key(addr_str)
        try:
            yield int(base_str, 0)
        except (TypeError, ValueError):
            continue


def _first_scalar(val):
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    return _to_int_or(val, None)


def _copy_columns(cols):
    return copy.deepcopy(cols)


def _validate_cache_key(config, key):
    value = config.get(key, None)
    if value is None:
        return
    if isinstance(value, bool) or (
        isinstance(value, list) and any(isinstance(v, bool) for v in value)
    ):
        raise ValueError(
            f"unknown {key} config {value!r}; expected a shortcut like 'hot'/'cold' "
            f"or a dict like {{'{key}': {{'hit_rate': 100}}}} or a list of "
            "alternatives"
        )
    if value is not None and not isinstance(value, (str, dict, list, int, float)):
        raise ValueError(
            f"unknown {key} config {value!r}; expected a shortcut like 'hot'/'cold' "
            f"or a dict like {{'{key}': {{'hit_rate': 100}}}} or a list of "
            "alternatives"
        )
    arch = get_arch()
    validate = getattr(arch, "validate_cache_key", None)
    if validate is None:
        return
    for entry in value if isinstance(value, list) else [value]:
        validate(key, entry)
    resolvers = {
        "dcache": getattr(arch, "data_cache_levels", None),
        "icache": getattr(arch, "instruction_cache_levels", None),
        "dtlb": getattr(arch, "data_tlb_levels", None),
        "itlb": getattr(arch, "instruction_tlb_levels", None),
    }
    resolve = resolvers.get(key)
    if resolve is None:
        return
    for entry in value if isinstance(value, list) else [value]:
        if isinstance(entry, str):
            resolve({key: entry})


def _validate_int_value(path, value, minimum=1):
    for item in _nonempty_values(path, value):
        try:
            number = int(item)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path} must be an integer, got {item!r}") from e
        if number < minimum:
            raise ValueError(f"{path} must be >= {minimum}, got {item!r}")


def _validate_positive_number(path, value):
    for item in _nonempty_values(path, value):
        try:
            number = float(item)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path} must be a number, got {item!r}") from e
        if not number > 0:
            raise ValueError(f"{path} must be > 0, got {item!r}")


def _validate_backend(config):
    backend = (config or {}).get("backend")
    if backend is None:
        return
    if isinstance(backend, (list, tuple)):
        for entry in _nonempty_values("backend", backend):
            _validate_backend({"backend": entry})
        return
    if isinstance(backend, str):
        if backend not in _BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected one of {', '.join(_BACKENDS)}"
            )
        return
    if not isinstance(backend, dict):
        raise ValueError(
            f"unknown backend config {backend!r}; expected 'loop' or 'unroll', or a "
            "dict like {'unroll': {'count': 5}}"
        )
    if any(key in _MODES for key in backend):
        if set(backend) - set(_MODES):
            raise ValueError(
                "config backend map must contain only mode names "
                f"({', '.join(_MODES)}), got {sorted(backend)}"
            )
        for mode, name in backend.items():
            if name not in _BACKENDS:
                raise ValueError(
                    f"unknown backend {name!r} for mode {mode!r}; "
                    f"expected one of {', '.join(_BACKENDS)}"
                )
        return
    for name, opts in backend.items():
        if name not in _BACKENDS:
            raise ValueError(
                f"unknown backend {name!r}; expected one of {', '.join(_BACKENDS)}"
            )
        if opts is None:
            continue
        if not isinstance(opts, dict):
            raise ValueError(
                f"unknown backend.{name} config {opts!r}; expected a dict like "
                "{'probes': 3, 'target': 0.01}"
            )
        for key in opts:
            if key not in _BACKEND_OPTION_KEYS:
                raise ValueError(
                    f"unknown backend.{name} key {key!r}; expected one of "
                    f"{', '.join(sorted(_BACKEND_OPTION_KEYS))}"
                )
        for opt in ("probes", "runs", "count"):
            if opts.get(opt) is not None:
                _validate_int_value(f"backend.{name}.{opt}", opts[opt])
        if opts.get("target") is not None:
            _validate_positive_number(f"backend.{name}.target", opts["target"])


def _list_entry(key, value):
    if isinstance(value, dict):
        return dict(value)
    if key == "func" and isinstance(value, str):
        return {"order": value}
    if key == "code" and isinstance(value, (int, str)) and not isinstance(value, bool):
        return {"align": value}
    return value


def _check_list_entry(key, entry):
    if not isinstance(entry, dict):
        raise ValueError(
            f"config {key} alternatives must be dicts, got {entry!r}; expected "
            f"a list like {_LIST_TOP_HINT[key]}"
        )
    valid = _LIST_TOP_KEYS[key]
    for name in entry:
        if name not in valid:
            raise ValueError(
                f"unknown {key} key {name!r}; expected one of "
                f"{', '.join(sorted(valid))}"
            )
    return entry


def _validate_list_entry_values(key, entry):
    for name, value in entry.items():
        if isinstance(value, (list, tuple)):
            raise ValueError(
                f"config {key}.{name} must be a single value, got {value!r}; "
                f"sweep via multiple {key} alternatives instead"
            )
        if name == "mode":
            _resolve_mode(value, f"{key}.mode")
        elif name == "size":
            if key == "heap":
                _resolve_heap_size(value)
            else:
                _resolve_stack_size(value)
        elif name == "align":
            if key == "stack":
                _resolve_stack_align(value)
            else:
                _resolve_align_value(value, key)
        elif name == "order":
            _resolve_func_order(value)


def _validate_list_one(key, value):
    entry = _list_entry(key, value)
    if not isinstance(entry, dict):
        raise ValueError(
            f"unknown {key} config {value!r}; expected a list of alternatives "
            f"like {_LIST_TOP_HINT[key]}"
        )
    _check_list_entry(key, entry)
    _validate_list_entry_values(key, entry)


def _validate_list_top(key, value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            f"unknown {key} config {value!r}; expected a non-empty list of "
            f"alternatives like {_LIST_TOP_HINT[key]}"
        )
    for entry in value:
        _check_list_entry(key, entry)
        _validate_list_entry_values(key, entry)


def _validate_thread_top(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"unknown thread config {value!r}; {_THREAD_HINT}")
    for alt in value:
        if isinstance(alt, dict):
            _check_thread(alt)
            _validate_thread_values(alt)
            continue
        if not isinstance(alt, (list, tuple)) or not alt:
            raise ValueError(f"unknown thread config {value!r}; {_THREAD_HINT}")
        for entry in alt:
            _check_thread(entry)
            _validate_thread_values(entry)


def _validate_thread_values(entry):
    _parse_affinity(entry.get("affinity"))
    _parse_priority(entry.get("priority"))
    _parse_numa(entry.get("numa"))


def _param_columns_value(val):
    if isinstance(val, (list, tuple)):
        return list(val) or None
    return _to_int_or(val, None)


def _is_scratch_value(value):
    for item in value if isinstance(value, (list, tuple)) else [value]:
        try:
            addr = int(item, 0) if isinstance(item, str) else int(item)
        except (TypeError, ValueError):
            continue
        if _ASM_SCRATCH_BASE <= addr < _ASM_SCRATCH_BASE + 0x100000:
            return True
    return False


def _is_random_seed(value):
    return isinstance(value, str) and value.strip().casefold() == "random"


def _generate_seed():
    try:
        return random.SystemRandom().randint(0, 2**31 - 1)
    except Exception:
        return random.randint(0, 2**31 - 1)


def _config_seed_int(config):
    seed = (config or {}).get("seed", None)
    if seed is None:
        return None
    if _is_random_seed(seed):
        return _generate_seed()
    try:
        return int(seed) & 0xFFFFFFFFFFFFFFFF
    except (TypeError, ValueError):
        digest = hashlib.sha256(str(seed).encode()).digest()
        return int.from_bytes(digest[:8], "little")


@_locked
def _apply_seed(config):
    seed_int = _config_seed_int(config)
    if seed_int is None:
        return None
    return seed_int


def _make_rng(config):
    seed_int = _config_seed_int(config)
    if seed_int is None:
        return random.Random()
    return random.Random(seed_int)


def _resolve_align_value(value, where):
    if value is None:
        return _DEFAULT_ALIGN
    try:
        align = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where} align must be an integer, got {value!r}") from e
    if align < 1 or (align & (align - 1)):
        raise ValueError(f"{where} align must be a power of two >= 1, got {value!r}")
    return align


def _resolve_mode(value, where):
    if value is None:
        return "random"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("random", "zero", "fixed"):
            return text
    raise ValueError(f"{where} mode must be 'random', 'zero' or 'fixed', got {value!r}")


def _resolve_stack_size(value):
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return _DEFAULT_STACK_SIZE
    try:
        size = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"stack size must be an integer, got {value!r}") from e
    if size < _MIN_STACK_SIZE:
        raise ValueError(
            f"stack size must be >= {_MIN_STACK_SIZE} bytes, got {value!r}"
        )
    return size


def _resolve_stack_align(value):
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return _DEFAULT_ALIGN
    return _resolve_align_value(value, "stack")


def _resolve_heap_size(value):
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return "auto"
    try:
        size = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"heap size must be 'auto' or an integer, got {value!r}"
        ) from e
    if size < 8 or size > _HEAP_RANDOM_LIMIT_MAX:
        raise ValueError(
            f"heap size must be 'auto' or between 8 and "
            f"{_HEAP_RANDOM_LIMIT_MAX}, got {value!r}"
        )
    return size & ~7


def _entry_alternatives(key, value):
    if value is None:
        return [None]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(
                f"config {key} must be a non-empty list of alternatives, got {value!r}"
            )
        return list(value)
    return [value]


def _entry(key, value):
    entry = _list_entry(key, value)
    if entry is None:
        return dict(_ENTRY_DEFAULTS[key])
    if not isinstance(entry, dict):
        raise ValueError(
            f"unknown {key} config {value!r}; expected a dict like "
            f"{_LIST_TOP_HINT[key]}"
        )
    return {**_ENTRY_DEFAULTS[key], **entry}


@_locked
def _alternate(values, key, leaf):
    items = list(values)
    if not items:
        raise ValueError(
            f"config {key}.{leaf} must be a non-empty list, got {values!r}"
        )
    slot = _CONTAINER_RR.get((key, leaf), 0)
    _CONTAINER_RR[(key, leaf)] = slot + 1
    return items[slot % len(items)]


def _container(config, key):
    alternatives = _entry_alternatives(key, (config or {}).get(key))
    if len(alternatives) != 1:
        raise ValueError(
            f"config {key} has {len(alternatives)} alternatives here; expand the "
            f"combinations first (e.g. via _expand_config_spec)"
        )
    entry = _entry(key, alternatives[0])
    for leaf in ("align", "size", "order"):
        if leaf in entry and isinstance(entry[leaf], (list, tuple)):
            entry[leaf] = _alternate(entry[leaf], key, leaf)
    if key == "stack":
        entry["size"] = _resolve_stack_size(entry.get("size"))
        entry["align"] = _resolve_stack_align(entry.get("align"))
        entry["mode"] = _resolve_mode(entry.get("mode"), "stack.mode")
    elif key == "heap":
        entry["size"] = _resolve_heap_size(entry.get("size"))
        entry["mode"] = _resolve_mode(entry.get("mode"), "heap.mode")
    else:
        entry["align"] = _resolve_align_value(entry.get("align"), key)
        if "order" in entry:
            entry["order"] = _resolve_func_order(entry["order"])
    return entry


def _entry_signature(key, value):
    return tuple(_stable(_entry(key, alt)) for alt in _entry_alternatives(key, value))


def _resolve_func_order(value):
    order = str(value).strip().lower()
    if order not in _FUNC_ORDERS:
        raise ValueError(
            f"unknown func order {order!r}; expected {', '.join(_FUNC_ORDERS)}"
        )
    return order


def _mem_entries(addr_str, val):
    base_str, is_range = _parse_addr_key(addr_str)
    try:
        base = int(base_str, 0)
    except (TypeError, ValueError):
        return
    if is_range:
        if isinstance(val, (bytes, bytearray)):
            for i, byte in enumerate(bytes(val)):
                yield f"0x{base + i:x}", int(byte) & 0xFF
            return
        if isinstance(val, str):
            for i, byte in enumerate(val.encode()):
                yield f"0x{base + i:x}", int(byte) & 0xFF
            return
        if isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                value = _to_u64(item)
                if not 0 <= value <= 0xFF:
                    raise ValueError(
                        f"byte range value {item!r} at 0x{base + i:x} "
                        "must be 0..255; use --data[0xADDR]=VALUE for "
                        "wider stores"
                    )
                yield f"0x{base + i:x}", value
        else:
            value = _to_u64(val)
            if not 0 <= value <= 0xFF:
                raise ValueError(
                    f"byte range value {val!r} at 0x{base:x} "
                    "must be 0..255; use --data[0xADDR]=VALUE for "
                    "wider stores"
                )
            yield f"0x{base:x}", value
        return
    if isinstance(val, (list, tuple)):
        yield f"0x{base:x}", [_to_u64(item) for item in val]
    else:
        yield f"0x{base:x}", _to_u64(val)


def _host_target(config, file, target_label, *target_names):
    try:
        func_cfg = (config or {}).get("func") or {}
        if (
            func_cfg.get("order", _FUNC_ORDERS[0]) != _FUNC_ORDERS[0]
            or int(func_cfg.get("align", _DEFAULT_ALIGN)) != _DEFAULT_ALIGN
        ):
            return None
        for name in dict.fromkeys((*target_names, target_label)):
            if not name:
                continue
            address = host_symbol(file, name)
            if address is not None:
                return address
        return None
    except Exception:
        return None


def _config_data(config):
    data = (config or {}).get("data", None) if isinstance(config, dict) else None
    if data is None:
        return {"regs": {}, "mem": {}}
    if not isinstance(data, dict):
        raise ValueError(f"unknown data config {data!r}; expected a dict")
    regs = {}
    mem = {}
    for key, val in data.items():
        s = str(key).strip()
        if _is_mem_addr_key(s):
            for k, v in _mem_entries(s, val):
                mem[k] = v
        elif isinstance(val, (list, tuple)):
            regs[_canonical_data_reg_key(key)] = [_to_u64(x) for x in val]
        else:
            try:
                ival = int(val, 0) if isinstance(val, str) else int(val)
            except (TypeError, ValueError):
                continue
            regs[_canonical_data_reg_key(key)] = ival & 0xFFFFFFFFFFFFFFFF
    return {"regs": regs, "mem": mem}


def _is_entry_label(label):
    name = str(label or "").split("..")[0]
    return name.split("@")[0].strip() in _ENTRY_LABELS


def _entry_data(label, file, data, argv=None, env=None):
    regs = data.get("regs") or {}
    if not _is_entry_label(label):
        return data
    base = _ENTRY_DATA_BASE
    words = [str(w) for w in argv] if argv else [os.path.basename(str(file))]
    environ = [str(w) for w in (env or ())]
    texts = [w.encode() + b"\0" for w in words + environ]
    vector_size = 8 * (len(words) + len(environ) + 2)
    at = base + max(0x100, vector_size)
    slots = []
    for text in texts:
        slots.append(at)
        at += (len(text) + 7) & ~7
    argv_at = base
    envp_at = base + 8 * (len(words) + 1)
    mem = dict(data.get("mem") or {})
    for i, slot in enumerate(slots[: len(words)]):
        mem[f"0x{argv_at + 8 * i:x}"] = slot
    mem[f"0x{argv_at + 8 * len(words):x}"] = 0
    for i, slot in enumerate(slots[len(words) :]):
        mem[f"0x{envp_at + 8 * i:x}"] = slot
    mem[f"0x{envp_at + 8 * len(environ):x}"] = 0
    for slot, text in zip(slots, texts):
        packed = text + b"\0" * (-len(text) % 8)
        for i in range(0, len(packed), 8):
            mem[f"0x{slot + i:x}"] = int.from_bytes(packed[i : i + 8], "little")
    synthetic_regs = get_arch().argv_regs(len(words), argv_at, envp_at)
    merged = {
        "regs": {**synthetic_regs, **regs},
        "mem": mem,
    }
    heap = dict(data.get("heap") or {})
    out = _code_data(merged, data.get("code") or ())
    if heap:
        out["heap"] = heap
    return out


def _explicit_bound_values(data):
    bounds = []
    for val in ((data or {}).get("regs") or {}).values():
        items = list(val) if isinstance(val, (list, tuple)) else [val]
        for item in items:
            try:
                iv = int(item, 0) if isinstance(item, str) else int(item)
            except (TypeError, ValueError):
                continue
            iv &= 0xFFFFFFFFFFFFFFFF
            if 0 < iv < (1 << 32):
                bounds.append(iv)
    return bounds


def _is_harness_fault(ex):
    return "faulted with" in str(ex)


def _missing_arg_regs(data, arch=None):
    if arch is None:
        arch = get_arch()
    try:
        arg_regs = list(getattr(arch, "_ARG_REGS", ()) or ())
    except Exception:
        return []
    try:
        harness = set(getattr(arch, "_HARNESS_REGS", ()) or ())
    except Exception:
        harness = set()
    explicit = set()
    for key in (data or {}).get("regs") or {}:
        try:
            explicit.add(_canonical_data_reg_key(key))
        except Exception:  # pragma: no cover
            try:
                explicit.add(str(key).strip().lower())
            except Exception:
                continue
    missing = []
    for reg in arg_regs:
        try:
            canon = _canonical_data_reg_key(reg)
        except Exception:
            canon = str(reg).strip().lower()
        if canon in harness or canon in explicit:
            continue
        missing.append(canon)
    return missing


def _explicit_bound_arg_regs(data, arch=None):
    if arch is None:
        arch = get_arch()
    try:
        arg_regs = {
            _canonical_data_reg_key(reg)
            for reg in (getattr(arch, "_ARG_REGS", ()) or ())
        }
    except Exception:
        arg_regs = set()
    bounds = set()
    for key, val in ((data or {}).get("regs") or {}).items():
        try:
            canon = _canonical_data_reg_key(key)
        except Exception:
            canon = str(key).strip().lower()
        if canon not in arg_regs:
            continue
        items = list(val) if isinstance(val, (list, tuple)) else [val]
        for item in items:
            try:
                iv = int(item, 0) if isinstance(item, str) else int(item)
            except (TypeError, ValueError):
                continue
            iv &= 0xFFFFFFFFFFFFFFFF
            if 0 < iv < (1 << 32):
                bounds.add(canon)
                break
    return bounds


def _sequence_probe_regs(data):
    probes = {}
    for key, val in ((data or {}).get("regs") or {}).items():
        if not isinstance(val, (list, tuple)):
            continue
        bounds = []
        for item in val:
            try:
                iv = int(item, 0) if isinstance(item, str) else int(item)
            except (TypeError, ValueError):
                continue
            iv &= 0xFFFFFFFFFFFFFFFF
            if 0 < iv < (1 << 32):
                bounds.append(iv)
        if bounds:
            probes[key] = max(bounds)
    return probes


def _heap_settings(config):
    raw = (config or {}).get("heap")
    if raw is None:
        entry = dict(_ENTRY_DEFAULTS["heap"])
    elif isinstance(raw, dict):
        entry = _entry("heap", raw)
    elif isinstance(raw, (list, tuple)) and len(raw) == 1:
        entry = _entry("heap", raw[0])
    else:
        raise ValueError(
            f"heap config must be a single entry like {_LIST_TOP_HINT['heap']}, "
            f"got {raw!r}"
        )
    size = _resolve_heap_size(entry.get("size"))
    mode = _resolve_mode(entry.get("mode"), "heap.mode")
    return size, mode


def _heap_base(value):
    try:
        text, _ = _parse_addr_key(value)
        return int(text, 0)
    except (TypeError, ValueError):
        return None


def _heap_covers(data, addr):
    heap = (data or {}).get("heap") or {}
    if not heap:
        return False
    try:
        if isinstance(addr, str):
            text = addr.strip()
            if text.endswith(":"):
                text = text[:-1].strip()
            addr = int(text, 0)
        else:
            addr = int(addr)
    except (TypeError, ValueError):
        return False
    for key, size in heap.items():
        base = _heap_base(key)
        try:
            size = int(size)
        except (TypeError, ValueError):
            continue
        if base is not None and base <= addr < base + size:
            return True
    return False


def _heap_size_hint(data, arch=None):
    if arch is None:
        arch = get_arch()
    try:
        arg_regs = [
            _canonical_data_reg_key(reg)
            for reg in (getattr(arch, "_ARG_REGS", ()) or ())
        ]
    except Exception:  # pragma: no cover
        return None
    count_reg = arg_regs[1] if len(arg_regs) > 1 else None
    stride_reg = arg_regs[2] if len(arg_regs) > 2 else None
    count = None
    stride = None
    for key, val in ((data or {}).get("regs") or {}).items():
        try:
            canon = _canonical_data_reg_key(key)
        except Exception:  # pragma: no cover
            continue
        if canon not in arg_regs or canon not in (count_reg, stride_reg):
            continue
        items = list(val) if isinstance(val, (list, tuple)) else [val]
        values = []
        for item in items:
            try:
                iv = int(item, 0) if isinstance(item, str) else int(item)
            except (TypeError, ValueError):
                continue
            iv &= 0xFFFFFFFFFFFFFFFF
            if 0 < iv < (1 << 32):
                values.append(iv)
        if not values:
            continue
        maximum = max(values)
        if canon == stride_reg:
            stride = maximum
        elif count is None or maximum > count:
            count = maximum
    if count is None:
        return None
    return int(count) * int(stride or 8)


def _heap_arg_regs(data, arch=None):
    if arch is None:
        arch = get_arch()
    missing = _missing_arg_regs(data, arch)
    if not missing:
        return []
    bounds = _explicit_bound_arg_regs(data, arch)
    if not bounds:
        return []
    try:
        order = {
            _canonical_data_reg_key(reg): i
            for i, reg in enumerate(getattr(arch, "_ARG_REGS", ()) or ())
        }
    except Exception:  # pragma: no cover
        order = {}
    out = []
    for reg in missing:
        try:
            idx = order.get(reg, len(order))
        except Exception:  # pragma: no cover
            idx = len(order)
        if any(idx < order.get(bound, -1) for bound in bounds):
            out.append(reg)
    return out


def _heap_rng(config, index=0):
    seed = _config_seed_int(config)
    if seed is None:
        return random.Random()
    mixed = (
        int(seed) ^ _HEAP_SEED_SALT ^ (int(index) * 0x9E3779B1)
    ) & 0xFFFFFFFFFFFFFFFF
    return random.Random(mixed)


def _apply_heap_data(data, config, debug=False):
    size, mode = _heap_settings(config)
    arch = get_arch()
    needed = _heap_size_hint(data, arch)
    limit = _HEAP_RANDOM_LIMIT if size == "auto" else int(size)
    if needed is not None:
        needed = (int(needed) + 7) & ~7
        if size == "auto":
            if needed > _HEAP_RANDOM_LIMIT_MAX:
                if debug:
                    _debug_log(
                        f"heap input needs 0x{needed:x} bytes, above the "
                        f"0x{_HEAP_RANDOM_LIMIT_MAX:x} limit; using the fallback"
                    )
                return data
            limit = needed
        elif needed > limit:
            if debug:
                _debug_log(
                    f"heap input needs 0x{needed:x} bytes, above the "
                    f"configured 0x{limit:x}; using the fallback"
                )
            return data
    targets = _heap_arg_regs(data, arch)
    if not targets:
        return data
    try:
        page_size = int(mmap.PAGESIZE)
    except (TypeError, ValueError):  # pragma: no cover
        page_size = 4096
    try:
        order = {
            _canonical_data_reg_key(reg): i
            for i, reg in enumerate(getattr(arch, "_ARG_REGS", ()) or ())
        }
    except Exception:  # pragma: no cover
        order = {}
    regs = dict((data or {}).get("regs") or {})
    mem = dict((data or {}).get("mem") or {})
    code = list((data or {}).get("code") or ())
    heap = dict((data or {}).get("heap") or {})
    for reg in targets:
        try:
            idx = order.get(reg, len(order))
        except Exception:  # pragma: no cover
            idx = len(order)
        base = _FALLBACK_DATA_BASE + idx * _FALLBACK_DATA_STRIDE
        base &= ~(page_size - 1)
        rng = _heap_rng(config, idx) if mode != "zero" else None
        for offset in range(0, limit, 8):
            key = f"0x{base + offset:x}"
            if key not in mem:
                mem[key] = 0 if rng is None else rng.getrandbits(64)
        regs[reg] = base
        heap[f"0x{base:x}"] = int(limit)
        if debug:
            _debug_log(
                f"heap buffer for {reg}: base=0x{base:x} size=0x{limit:x} mode={mode}"
            )
    out = _code_data({"regs": regs, "mem": mem}, code)
    if heap:
        out["heap"] = heap
    return out


def _needs_pointer_fallback(data, arch=None):
    if arch is None:
        arch = get_arch()
    missing = _missing_arg_regs(data, arch)
    if not missing:
        return False
    bounds = _explicit_bound_arg_regs(data, arch)
    if not bounds:
        return False
    try:
        order = {
            _canonical_data_reg_key(reg): i
            for i, reg in enumerate(getattr(arch, "_ARG_REGS", ()) or ())
        }
    except Exception:
        order = {}
    return any(
        order.get(reg, len(order)) < order.get(bound, -1)
        for reg in missing
        for bound in bounds
    )


def _augment_data_with_fallback(
    data, arch=None, pages=_FALLBACK_START_PAGES, debug=False
):
    if arch is None:
        arch = get_arch()
    missing = _missing_arg_regs(data, arch)
    if not missing:
        return data
    if not _explicit_bound_values(data):
        return data
    try:
        pages = int(pages)
    except (TypeError, ValueError):
        pages = _FALLBACK_START_PAGES
    if pages < 1:
        pages = _FALLBACK_START_PAGES
    try:
        page_size = int(mmap.PAGESIZE)
    except Exception:
        page_size = 4096
    size = int(pages) * page_size
    try:
        arg_regs = list(getattr(arch, "_ARG_REGS", ()) or ())
        order = {str(r).strip().lower(): i for i, r in enumerate(arg_regs)}
    except Exception:
        order = {}
    missing = sorted(set(missing), key=lambda r: order.get(str(r).strip().lower(), 999))
    regs = dict((data or {}).get("regs") or {})
    mem = dict((data or {}).get("mem") or {})
    code = list((data or {}).get("code") or ())
    for reg in missing:
        try:
            idx = order.get(str(reg).strip().lower(), len(order))
        except Exception:
            idx = len(order)
        base = _FALLBACK_DATA_BASE + idx * _FALLBACK_DATA_STRIDE
        base &= ~(page_size - 1)
        pages_list = list(range(base, base + size, page_size))
        try:
            _map_data_pages({"mem": {f"0x{p:x}": 0 for p in pages_list}}, strict=False)
        except Exception as ex:  # pragma: no cover
            if debug:
                _debug_log(f"fallback mapping for {reg} failed: {ex!r}")
            continue
        regs[reg] = base
        if size <= _FALLBACK_STEER_MAX:
            for p in pages_list:
                mem.setdefault(f"0x{p:x}", 0)
        else:
            mem.setdefault(f"0x{base:x}", 0)
        if debug:
            _debug_log(
                f"fallback buffer for {reg}: base=0x{base:x} size=0x{size:x} "
                f"pages={len(pages_list)}"
            )
    return _code_data({"regs": regs, "mem": mem}, code)


def _grow_fallback_data(data, arch, probe, debug=False):
    missing = _missing_arg_regs(data, arch)
    if not missing or not _explicit_bound_values(data):
        return data
    try:
        page_size = int(mmap.PAGESIZE)
    except Exception:
        page_size = 4096
    try:
        _cache_key = (
            _stable(data),
            tuple(sorted(missing)),
            str(getattr(arch, "_ANGR_ARCH_NAME", "") or type(arch).__name__),
        )
    except Exception:
        _cache_key = None
    pages = _FALLBACK_START_PAGES
    if _cache_key is not None:
        try:
            with _STATE_LOCK:
                cached = int(_FALLBACK_GROWN_PAGES.get(_cache_key, 0) or 0)
        except (TypeError, ValueError):
            cached = 0
        if cached > pages:
            pages = cached
            if debug:
                _debug_log(f"fallback starting from cached {pages} page(s) per buffer")
    last = None
    for _ in range(_FALLBACK_MAX_DOUBLINGS + 1):
        try:
            augmented = _augment_data_with_fallback(data, arch, pages, debug)
        except Exception as ex:  # pragma: no cover
            if debug:
                _debug_log(f"fallback mapping for {pages} pages failed: {ex!r}")
            raise
        if augmented is data:
            return data
        try:
            probe(augmented)
            if _cache_key is not None:
                try:
                    with _STATE_LOCK:
                        prev = int(_FALLBACK_GROWN_PAGES.get(_cache_key, 0) or 0)
                        if int(pages) > prev:
                            _FALLBACK_GROWN_PAGES[_cache_key] = int(pages)
                except (TypeError, ValueError):
                    pass
            return augmented
        except ValueError as ex:  # pragma: no cover
            if not _is_harness_fault(ex):
                raise
            last = ex
            if debug:
                _debug_log(
                    f"fallback probe with {pages} page(s) per buffer faulted, "
                    "growing as needed"
                )
            try:
                allowed = _fallback_budget_bytes()
            except Exception:
                allowed = None
            if allowed is not None:
                try:
                    need = int(len(missing)) * int(pages * 2) * int(page_size)
                except (TypeError, ValueError):
                    need = 0
                if need > allowed:
                    if debug:
                        _debug_log(
                            "fallback growth would exceed memory budget, "
                            f"need=0x{need:x} allowed=0x{allowed:x}"
                        )
                    break
            pages *= 2
            continue
    if last is not None:
        raise last
    return data


def _branch_value(spec):
    if isinstance(spec, bool):
        return "predictable" if spec else "unpredictable"
    s = str(spec).strip().lower()
    return s if s in _BRANCH_CHOICES else None


def _branch_entry(key, spec, mem, regs, where):
    value = _branch_value(spec)
    if value is None:
        raise ValueError(
            f"branch{where} value for {key!r} must be "
            f"one of {', '.join(_BRANCH_CHOICES)}, got {spec!r}"
        )
    try:
        mem[int(str(key).strip(), 0)] = value
    except (TypeError, ValueError):
        regs[_canonical_data_reg_key(key)] = value


def _branch_config(config):
    branch = (config or {}).get("branch")
    if branch is None:
        return None
    if isinstance(branch, str):
        value = _branch_value(branch)
        if value is None:
            raise ValueError(
                f"branch prediction must be one of {', '.join(_BRANCH_CHOICES)}, "
                f"got {branch!r}"
            )
        return {"prediction": value, "mem": {}, "regs": {}}
    if isinstance(branch, bool):
        return {
            "prediction": "predictable" if branch else "unpredictable",
            "mem": {},
            "regs": {},
        }
    if not isinstance(branch, dict):
        raise ValueError(f"branch config must be a string or dict, got {branch!r}")
    mem, regs = {}, {}
    for key, spec in branch.items():
        if key in ("prediction", "mem", "regs"):
            continue
        if key == "default":
            raise ValueError(
                "unknown branch key 'default'; use 'prediction' "
                "(e.g. {'prediction': 'predictable'})"
            )
        _branch_entry(key, spec, mem, regs, "")
    for addr_key, spec in (branch.get("mem") or {}).items():
        _branch_entry(addr_key, spec, mem, regs, ".mem")
    for reg_key, spec in (branch.get("regs") or {}).items():
        value = _branch_value(spec)
        if value is None:
            raise ValueError(
                f"branch.regs value for {reg_key!r} must be "
                f"one of {', '.join(_BRANCH_CHOICES)}, got {spec!r}"
            )
        regs[_canonical_data_reg_key(reg_key)] = value
    prediction = branch.get("prediction", "unpredictable")
    if isinstance(prediction, bool):
        prediction = "predictable" if prediction else "unpredictable"
    else:
        prediction = _branch_value(prediction)
        if prediction is None:
            raise ValueError(
                f"branch prediction must be one of {', '.join(_BRANCH_CHOICES)}, "
                f"got {branch.get('prediction')!r}"
            )
    return {"prediction": prediction, "mem": mem, "regs": regs}


def _branch_global_distribution(branch_cfg, predictable):
    try:
        if isinstance(predictable, str):
            value = _branch_value(predictable)
            if value is not None:
                return value
    except Exception:
        pass
    try:
        pred = (branch_cfg or {}).get("prediction")
        if isinstance(pred, str):
            value = _branch_value(pred)
            if value is not None:
                return value
    except Exception:
        pass
    try:
        if bool(predictable):
            return "predictable"
    except Exception:
        pass
    return "unpredictable"


def _branch_reg_distribution(canon, branch_cfg, predictable):
    try:
        regs = (branch_cfg or {}).get("regs") or {}
        if canon in regs:
            value = _branch_value(regs[canon])
            if value is not None:
                return value
    except Exception:
        pass
    return _branch_global_distribution(branch_cfg, predictable)


def _branch_mem_distribution(addr, branch_cfg, predictable):
    try:
        mem = (branch_cfg or {}).get("mem") or {}
        a = int(addr)
        if a in mem:
            value = _branch_value(mem[a])
            if value is not None:
                return value
    except Exception:
        pass
    return _branch_global_distribution(branch_cfg, predictable)


def _branch_sample_index(n, rng, distribution, it):
    n = int(n)
    if n <= 1:
        return 0
    if distribution == "predictable":
        return int(it) % n
    return rng.randrange(n)


def _model_picks(models, predictable, rng, branch_cfg, iterations):
    if not models:
        return []
    dist = _branch_global_distribution(branch_cfg, predictable)
    n = len(models)
    try:
        total = int(iterations)
    except (TypeError, ValueError):
        total = 0
    if total <= 0:
        return []
    if dist == "predictable":
        try:
            return (np.arange(total, dtype=np.int64) % int(n)).tolist()
        except (TypeError, ValueError):
            pass
        return [it % n for it in range(total)]
    try:
        picks = np.frombuffer(rng.randbytes(8 * int(total)), dtype=np.uint64)
        return (picks % np.uint64(n)).astype(np.int64).tolist()
    except (AttributeError, TypeError, ValueError, OverflowError):
        randrange = rng.randrange
        return [randrange(n) for _ in range(total)]


def _collect_mem_addrs(models):
    addrs = set()
    for m in models:
        for a, _, _ in m["reads"] + m["writes"]:
            addrs.add(a)
    return addrs


def _static_extra_addrs(extra_addrs=None):
    out = set()
    for a in extra_addrs or ():
        try:
            out.add(int(a, 0) if isinstance(a, str) else int(a))
        except (TypeError, ValueError):
            continue
    return out


def _apply_code_tier_defaults(
    arch, config, mem_levels, extra_addrs=None, code_addrs=None
):
    data_refs = _static_extra_addrs(extra_addrs)
    code = _static_extra_addrs(code_addrs)
    if not data_refs and not code:
        return mem_levels

    def _defaults(resolver):
        fn = getattr(arch, resolver, None)
        if fn is None:
            return {}
        try:
            levels = fn(config)
        except Exception:
            levels = None
        if not levels:
            return {}
        return {t: v for t, v in levels.items() if v is not None}

    out = dict(mem_levels or {})
    code_resolvers = (
        ("instruction_cache_levels", ("L1i",)),
        ("instruction_tlb_levels", ("TLBi",)),
    )
    data_resolvers = (
        ("data_cache_levels", ("L1d",)),
        ("data_tlb_levels", ("TLBd",)),
    )
    for addrs, resolvers in ((code, code_resolvers), (data_refs, data_resolvers)):
        defaults = {}
        for resolver, tiers in resolvers:
            for tier, rate in _defaults(resolver).items():
                if tier in tiers:
                    defaults.setdefault(tier, rate)
        if not defaults:
            continue
        for addr in addrs:
            spec = out.get(addr)
            spec = dict(spec) if isinstance(spec, dict) else {}
            for tier, rate in defaults.items():
                spec.setdefault(tier, rate)
            out[addr] = spec
    return out


def _mem_level_int_addrs(mem_levels):
    out = set()
    for k in mem_levels or {}:
        if isinstance(k, bool):
            continue
        if isinstance(k, int):
            out.add(k)
            continue
        try:
            s = str(k).strip()

            if not s:
                continue
            out.add(int(s, 0))
        except (TypeError, ValueError):
            continue
    return out


def _reg_level_specs(mem_levels):
    out = {}
    for key, spec in (mem_levels or {}).items():
        if isinstance(key, (bool, int)) or not isinstance(spec, dict):
            continue
        text = str(key).strip()
        if not text or _is_mem_addr_key(text):
            continue
        try:
            canon = _canonical_data_reg_key(text)
        except Exception:
            canon = text.lower()
        out.setdefault(canon, []).append(spec)
    return out


def _reg_level_values(regs):
    out = {}
    for canon, value in _normalize_data_regs(regs).items():
        items = value if isinstance(value, list) else [value]
        addrs = {int(item) & 0xFFFFFFFFFFFFFFFF for item in items}
        addrs.discard(0)
        if addrs:
            out[canon] = addrs
    return out


def _resolve_reg_levels(mem_levels, models, regs=None, mem=None):
    specs = _reg_level_specs(mem_levels)
    if not specs:
        return mem_levels
    pointers = _reg_level_values(regs)
    for model in models or ():
        for canon, addrs in _reg_level_values(model.get("regs") or {}).items():
            pointers.setdefault(canon, set()).update(addrs)
    if not pointers:
        return mem_levels
    accessed = set(_collect_mem_addrs(models or ()))
    for key in mem or {}:
        try:
            base, _ = _parse_addr_key(key)
            accessed.add(int(base, 0))
        except (TypeError, ValueError):
            continue
    out = {}
    for addr, spec in (mem_levels or {}).items():
        if isinstance(addr, bool) or not isinstance(spec, dict):
            continue
        try:
            target = addr if isinstance(addr, int) else int(str(addr).strip(), 0)
        except (TypeError, ValueError):
            continue
        out[target] = dict(spec)
    for canon, tier_specs in specs.items():
        for pointer in pointers.get(canon, ()):
            addrs = {
                addr
                for addr in accessed
                if pointer <= addr < pointer + _REG_LEVEL_WINDOW
            }
            for addr in addrs or (pointer,):
                target = out.setdefault(addr, {})
                for spec in tier_specs:
                    for tier, rate in spec.items():
                        target.setdefault(tier, rate)
    return out


def _code_level_addrs(mem_levels):
    out = set()
    for addr, spec in (mem_levels or {}).items():
        if not isinstance(spec, dict):
            continue
        if spec.get("L1i") is None and spec.get("TLBi") is None:
            continue
        if isinstance(addr, bool):
            continue
        try:
            value = int(addr, 0) if isinstance(addr, str) else int(addr)
        except (TypeError, ValueError):
            continue
        out.add(value)
    return out


def _sample_values(values, distribution, it, rng):
    vals = list(values or [])
    if not vals:
        return 0
    if len(vals) == 1:
        return vals[0]
    if distribution == "predictable":
        return vals[int(it) % len(vals)]
    return vals[_branch_sample_index(len(vals), rng, distribution, it)]


def _addr_in_ranges(addr, ranges):
    for lo, hi in ranges or ():
        try:
            if int(lo) <= int(addr) < int(hi):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _random_u64(rng, n):
    n = int(n)
    try:
        return np.frombuffer(rng.randbytes(n * 8), dtype=np.uint64).copy()
    except Exception:
        return np.asarray([rng.getrandbits(64) for _ in range(n)], dtype=np.uint64)


def _fill_dynamic(
    buf,
    iterations,
    models,
    picks,
    reg_names,
    reg_col,
    val_col,
    mem_addrs,
    explicit_regs,
    explicit_mem,
    pin_regs,
    pin_mem,
    model0_mem,
    branch_cfg,
    predictable,
    rng,
    heap_ranges=None,
    heap_mode=None,
):
    mask = 0xFFFFFFFFFFFFFFFF
    n = int(iterations)
    if n <= 0:
        return
    models = list(models) or [{"regs": {}, "reads": [], "writes": []}]
    try:
        picks = list(picks)[:n]
    except (TypeError, ValueError):
        picks = []
    if len(picks) < n:
        picks += [picks[-1] if picks else 0] * (n - len(picks))
    try:
        picks_arr = np.asarray(picks, dtype=np.int64)
        if picks_arr.shape != (n,):
            picks_arr = None
    except (TypeError, ValueError):
        picks_arr = None

    reg_src = {
        name: [(m["regs"].get(name) or 0) & mask for m in models] for name in reg_names
    }
    model0_regs = (models[0] or {}).get("regs") or {}

    for name in reg_names:
        if name in explicit_regs:
            continue
        src = reg_src[name]
        if name in pin_regs:
            pinned = model0_regs.get(name)
            if pinned is not None:
                try:
                    vals = np.full((n,), int(pinned) & mask, dtype=np.uint64)
                except (TypeError, ValueError):
                    vals = [pinned & mask] * n
                buf[reg_col[name], 1 : n + 1] = vals[::-1]
                continue
        if picks_arr is not None:
            try:
                src_arr = np.asarray(src, dtype=np.uint64)
                vals = src_arr[picks_arr]
                buf[reg_col[name], 1 : n + 1] = vals[::-1]
                continue
            except (TypeError, ValueError, IndexError):
                pass
        vals = [src[p] for p in picks]
        buf[reg_col[name], 1 : n + 1] = vals[::-1]

    mem_src = {}
    for a in mem_addrs:
        if a in explicit_mem or a not in val_col:
            continue
        mem_src[a] = [0] * len(models)
    for index, m in enumerate(models):
        for a, _, value in m["reads"] + m["writes"]:
            target = mem_src.get(a)
            if target is not None:
                target[index] = value & mask
    for a, src in mem_src.items():
        if a in pin_mem and a in model0_mem:
            try:
                vals = np.full((n,), int(model0_mem[a]) & mask, dtype=np.uint64)
            except (TypeError, ValueError):
                vals = [int(model0_mem[a]) & mask] * n
            buf[val_col[a], 1 : n + 1] = vals[::-1]
            continue
        if picks_arr is not None:
            try:
                src_arr = np.asarray(src, dtype=np.uint64)
                vals = src_arr[picks_arr]
                buf[val_col[a], 1 : n + 1] = vals[::-1]
                continue
            except (TypeError, ValueError, IndexError):
                pass
        vals = [src[p] for p in picks]
        buf[val_col[a], 1 : n + 1] = vals[::-1]

    for name, value in explicit_regs.items():
        if name not in reg_col:
            continue
        if isinstance(value, (list, tuple)):
            dist = _branch_reg_distribution(name, branch_cfg, predictable)
            if dist == "predictable":
                try:
                    arr = np.asarray(
                        [int(v) & mask for v in list(value)], dtype=np.uint64
                    )
                    if len(arr) == 0:
                        raise ValueError
                    vals = arr[np.arange(n, dtype=np.int64) % len(arr)]
                    buf[reg_col[name], 1 : n + 1] = vals[::-1]
                    continue
                except (TypeError, ValueError, IndexError):
                    pass
            vals = [_sample_values(value, dist, it, rng) & mask for it in range(n)]
        else:
            vals = [int(value) & mask] * n
        buf[reg_col[name], 1 : n + 1] = vals[::-1]

    for a, value in explicit_mem.items():
        if a not in val_col:
            continue
        if heap_mode in ("random", "zero") and _addr_in_ranges(a, heap_ranges):
            vals = (
                np.zeros(n, dtype=np.uint64)
                if heap_mode == "zero"
                else _random_u64(rng, n)
            )
            buf[val_col[a], 1 : n + 1] = vals[::-1]
            continue
        if isinstance(value, (list, tuple)):
            dist = _branch_mem_distribution(a, branch_cfg, predictable)
            if dist == "predictable":
                try:
                    arr = np.asarray(
                        [int(v) & mask for v in list(value)], dtype=np.uint64
                    )
                    if len(arr) == 0:
                        raise ValueError
                    vals = arr[np.arange(n, dtype=np.int64) % len(arr)]
                    buf[val_col[a], 1 : n + 1] = vals[::-1]
                    continue
                except (TypeError, ValueError, IndexError):
                    pass
            vals = [_sample_values(value, dist, it, rng) & mask for it in range(n)]
        else:
            vals = [int(value) & mask] * n
        buf[val_col[a], 1 : n + 1] = vals[::-1]


def _per_iter_data(
    models,
    iterations,
    levels,
    predictable,
    rng,
    arch=None,
    explicit=None,
    branch_cfg=None,
    mem_levels=None,
    extra_addrs=None,
    tlb_levels=None,
    heap_mode=None,
):
    if arch is None:
        arch = get_arch()
    harness_regs = arch._HARNESS_REGS

    mem_levels = mem_levels or {}

    explicit = explicit or {}

    def _norm_explicit_list(val):
        return [_to_u64(x) for x in val] or [_to_u64(0)]

    explicit_regs = {}
    for key, val in (explicit.get("regs") or {}).items():
        canon = _canonical_data_reg_key(key)
        if canon in (harness_regs or set()):
            if isinstance(val, (list, tuple)):
                raise ValueError(
                    f"data register {key!r} is used by the measurement harness "
                    "to hold timing baselines, so a per-iteration list would be "
                    "silently truncated to its first value; pass a scalar or pick "
                    "a different register"
                )
            continue
        explicit_regs[canon] = (
            _norm_explicit_list(val)
            if isinstance(val, (list, tuple))
            else _to_u64(_to_int_or(val, 0))
        )
    explicit_mem = {}
    for addr_str, val in (explicit.get("mem") or {}).items():
        s = str(addr_str).strip()
        if s.endswith(":"):
            s = s[:-1].strip()
        try:
            base = int(s, 0)
        except (TypeError, ValueError):
            continue
        explicit_mem[base] = (
            _norm_explicit_list(val)
            if isinstance(val, (list, tuple))
            else _to_u64(_to_int_or(val, 0))
        )
    heap_ranges = []
    for base_str, size in (explicit.get("heap") or {}).items():
        base = _heap_base(base_str)
        try:
            size = int(size)
        except (TypeError, ValueError):
            continue
        if base is not None and size > 0:
            heap_ranges.append((base, base + size))

    mem_levels = _resolve_reg_levels(mem_levels, models, explicit_regs, explicit_mem)

    pin_regs = set()
    pin_mem = set()
    mem_cfg = (branch_cfg or {}).get("mem") or {}
    regs_cfg = (branch_cfg or {}).get("regs") or {}
    if mem_cfg or regs_cfg:
        for addr, value in mem_cfg.items():
            if value != "predictable":
                continue

            for m in models:
                deps = (m.get("branch_deps") or {}).get(addr)
                if not deps:
                    continue
                pin_regs |= set(deps.get("regs") or ())
                pin_mem |= set(deps.get("mem") or ())

        for r, value in regs_cfg.items():
            if value == "predictable":
                pin_regs.add(r)

        for addr, value in mem_cfg.items():
            if value == "predictable":
                try:
                    pin_mem.add(int(addr))
                except (TypeError, ValueError):
                    pass

    model0 = models[0] if models else {}
    model0_mem = {}
    for a, _, value in model0.get("writes", []) + model0.get("reads", []):
        model0_mem[a] = value
    for a in list(pin_mem):
        if a not in model0_mem and a not in explicit_mem:
            pin_mem.discard(a)

    reg_names = []
    seen = set()
    for m in models:
        for name in m["regs"]:
            if name in harness_regs or name in seen:
                continue
            seen.add(name)
            reg_names.append(name)
    for name in explicit_regs:
        if name in harness_regs or name in seen:
            continue
        seen.add(name)
        reg_names.append(name)

    l1i_addrs = sorted(_code_level_addrs(mem_levels))
    l1i_set = set(l1i_addrs)
    model_mem = _collect_mem_addrs(models)
    mem_addrs = sorted(
        (
            model_mem
            | set(explicit_mem)
            | _mem_level_int_addrs(mem_levels)
            | _static_extra_addrs(extra_addrs)
        )
        - l1i_set
    )
    write_addrs = [
        a
        for a in mem_addrs
        if (a in model_mem or a in explicit_mem)
        and not (heap_mode == "fixed" and _addr_in_ranges(a, heap_ranges))
    ]

    reg_col = {name: i for i, name in enumerate(reg_names)}
    val_col = {}
    col = len(reg_names)
    for a in mem_addrs:
        val_col[a] = col
        col += 1

    ncols = col
    naddr = len(mem_addrs)
    nwrite = len(write_addrs)
    k = iterations + 1
    K = max(k, 2 * naddr, 2 * nwrite) if (naddr or nwrite) else k

    total_cols = ncols + 2 * naddr + 2 * nwrite
    buf = np.zeros((total_cols, K), dtype=np.uint64)

    static = (
        len(models) == 1
        and predictable
        and heap_mode not in ("random", "zero")
        and not mem_cfg
        and not regs_cfg
        and not any(isinstance(v, (list, tuple)) for v in list(explicit_regs.values()))
        and not any(isinstance(v, (list, tuple)) for v in explicit_mem.values())
    )
    if static:
        m = models[0]
        regs = m["regs"]
        for name in reg_names:
            if name in explicit_regs:
                value = explicit_regs[name]
            else:
                value = regs.get(name, 0)
            try:
                buf[reg_col[name], 1 : iterations + 1] = int(value) & 0xFFFFFFFFFFFFFFFF
            except (TypeError, ValueError):
                buf[reg_col[name], 1 : iterations + 1] = 0
        for a in write_addrs:
            if a in explicit_mem:
                value = explicit_mem[a]
            else:
                value = model0_mem.get(a, 0)
            try:
                buf[val_col[a], 1 : iterations + 1] = int(value) & 0xFFFFFFFFFFFFFFFF
            except (TypeError, ValueError):
                buf[val_col[a], 1 : iterations + 1] = 0
    else:
        _fill_dynamic(
            buf,
            iterations,
            models,
            picks=_model_picks(models, predictable, rng, branch_cfg, iterations),
            reg_names=reg_names,
            reg_col=reg_col,
            val_col=val_col,
            mem_addrs=write_addrs,
            explicit_regs=explicit_regs,
            explicit_mem=explicit_mem,
            pin_regs=pin_regs,
            pin_mem=pin_mem,
            model0_mem=model0_mem,
            branch_cfg=branch_cfg,
            predictable=predictable,
            rng=rng,
            heap_ranges=heap_ranges,
            heap_mode=heap_mode,
        )

    code_pages = set()
    for page in explicit.get("code") or ():
        try:
            code_pages.add(int(page) & ~(mmap.PAGESIZE - 1))
        except (TypeError, ValueError):
            continue
    target_writes = any(m.get("writes") for m in models)
    heap_dynamic = bool(
        heap_mode in ("random", "zero")
        and heap_ranges
        and any(lo <= addr < hi for addr in write_addrs for lo, hi in heap_ranges)
    )
    values_static = bool(
        predictable
        and not heap_dynamic
        and not mem_cfg
        and not regs_cfg
        and len(models) == 1
        and not target_writes
        and not any(isinstance(v, (list, tuple)) for v in list(explicit_regs.values()))
        and not any(isinstance(v, (list, tuple)) for v in explicit_mem.values())
    )
    with _STATE_LOCK:
        tlb_avoid = sorted(set(_UNMAPPABLE_DATA_PAGES) | code_pages)
    meta = {
        "ncols": ncols,
        "k": k,
        "K": K,
        "reg_col": reg_col,
        "val_col": val_col,
        "mem_addrs": mem_addrs,
        "write_addrs": write_addrs,
        "values_static": values_static,
        "target_writes": target_writes,
        "l1i_addrs": l1i_addrs,
        "reg_names": reg_names,
        "levels": levels,
        "mem_levels": mem_levels or {},
        "tlb_levels": tlb_levels or {},
        "tlb_avoid": tlb_avoid,
        "evict_table_col": ncols if naddr else None,
        "write_table_col": (ncols + 2 * naddr) if nwrite else None,
        "explicit_regs": explicit_regs,
        "explicit_mem": explicit_mem,
        "pin_regs": pin_regs,
        "pin_mem": pin_mem,
        "model0_mem": model0_mem,
        "heap_ranges": heap_ranges,
    }
    return buf, meta


def _fill_evict_tables(buf, meta, data_ptr):
    addrs = meta["mem_addrs"]
    if not addrs:
        return
    n = len(addrs)
    K = meta["K"]
    et = meta["evict_table_col"]
    for i, a in enumerate(addrs):
        buf[et, i] = a
        buf[et, n + i] = (data_ptr + meta["val_col"][a] * K * 8) & 0xFFFFFFFFFFFFFFFF
    write_addrs = meta.get("write_addrs")
    wt = meta.get("write_table_col")
    if wt is None or write_addrs is None:
        return
    nw = len(write_addrs)
    for i, a in enumerate(write_addrs):
        buf[wt, i] = a
        buf[wt, nw + i] = (data_ptr + meta["val_col"][a] * K * 8) & 0xFFFFFFFFFFFFFFFF


def _refresh_per_iter_data(
    buf,
    meta,
    iterations,
    models,
    levels,
    predictable,
    rng,
    arch,
    data=None,
    branch_cfg=None,
    mem_levels=None,
    extra_addrs=None,
    tlb_levels=None,
    heap_mode=None,
):
    if buf is None or meta is None or predictable:
        return
    if not meta.get("reg_names") and not meta.get("write_addrs"):
        return
    try:
        _fill_dynamic(
            buf,
            iterations,
            models,
            picks=_model_picks(models, predictable, rng, branch_cfg, iterations),
            reg_names=meta.get("reg_names") or [],
            reg_col=meta.get("reg_col") or {},
            val_col=meta.get("val_col") or {},
            mem_addrs=meta.get("write_addrs") or [],
            explicit_regs=meta.get("explicit_regs") or {},
            explicit_mem=meta.get("explicit_mem") or {},
            pin_regs=meta.get("pin_regs") or set(),
            pin_mem=meta.get("pin_mem") or set(),
            model0_mem=meta.get("model0_mem") or {},
            branch_cfg=branch_cfg,
            predictable=predictable,
            rng=rng,
            heap_ranges=meta.get("heap_ranges") or [],
            heap_mode=heap_mode,
        )
        _fill_evict_tables(buf, meta, buf.ctypes.data)
        return
    except Exception:
        pass
    try:
        fresh, _ = _per_iter_data(
            models,
            iterations,
            levels,
            predictable,
            rng,
            arch,
            explicit=data,
            branch_cfg=branch_cfg,
            mem_levels=mem_levels,
            extra_addrs=_static_extra_addrs(extra_addrs),
            tlb_levels=tlb_levels,
            heap_mode=heap_mode,
        )
    except Exception:
        return
    if fresh.shape != buf.shape:
        return
    buf[...] = fresh
    _fill_evict_tables(buf, meta, buf.ctypes.data)


def _names(events):
    return _split_list(events) or [_DURATION_TIME]


def _cache_rate(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _data_spec_tier(spec):
    best, best_rate = "DRAM", 0
    for tier in ("L1d", "L2", "L3"):
        rate = _cache_rate(spec.get(tier, 0))
        if rate > best_rate:
            best, best_rate = tier, rate
    return best


def _cache_state_sensitive(config, arch=None, keys=None):
    if arch is None:
        arch = get_arch()
    if not isinstance(config, dict) or not config:
        return False
    if keys is not None:
        keys = set(keys)
    try:
        combos = _expand_config_spec(_normalize_spec(config))
    except Exception:
        combos = [config]
    resolvers = {
        "dcache": ("data_cache_levels", "L1d", False),
        "icache": ("instruction_cache_levels", "L1i", True),
        "dtlb": ("data_tlb_levels", "TLBd", True),
        "itlb": ("instruction_tlb_levels", "TLBi", True),
    }
    for combo in combos:
        if not isinstance(combo, dict):
            continue
        for key, (resolver, tier, resident) in resolvers.items():
            if key not in combo:
                continue
            if keys is not None and key not in keys:
                continue
            fn = getattr(arch, resolver, None)
            try:
                levels = fn(combo) if fn is not None else None
            except Exception:
                levels = None
            if not isinstance(levels, dict):
                continue
            rate = _cache_rate(levels.get(tier))
            if resident and rate <= 0:
                return True
            if not resident and rate < 100:
                return True
        try:
            mem_levels = arch.resolve_mem_cache(combo)
        except Exception:
            mem_levels = {}
        for spec in (mem_levels or {}).values():
            if not isinstance(spec, dict):
                continue
            data = {t: spec[t] for t in ("L1d", "L2", "L3") if t in spec}
            if (
                (keys is None or "dcache" in keys)
                and data
                and _data_spec_tier(data) != "L1d"
            ):
                return True
            if (
                (keys is None or "icache" in keys)
                and spec.get("L1i") is not None
                and _cache_rate(spec["L1i"]) <= 0
            ):
                return True
            for tier, key in (("TLBd", "dtlb"), ("TLBi", "itlb")):
                if keys is not None and key not in keys:
                    continue
                if spec.get(tier) is not None and _cache_rate(spec[tier]) <= 0:
                    return True
    return False


def _normalize_target(target):
    if target is None:
        return None
    if isinstance(target, (list, tuple)):
        parts = [str(p).strip() for p in target]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"invalid target {target!r}; expected 'name' or ('begin', 'end')"
            )
        return f"{parts[0]}..{parts[1]}"
    name = str(target).strip()
    if not name:
        return None
    if ".." in name:
        raise ValueError(
            f"invalid target {target!r}; pass a region as a ('begin', 'end') pair, "
            "e.g. ('hot_begin', 'hot_end')"
        )
    return name


def _backend_mode_map(value):
    if not isinstance(value, dict) or not value:
        return None
    if not set(value) <= set(_MODES):
        return None
    if not all(name in _BACKENDS for name in value.values()):
        return None
    return dict(value)


def _modes(mode):
    if isinstance(mode, str) or not isinstance(mode, (list, tuple)):
        raise TypeError(
            "mode must be a list, e.g. ['latency'] or ['latency', 'throughput']"
        )
    modes = []
    for m in mode:
        if m not in _MODES:
            raise ValueError(f"unknown mode {m!r}; expected 'latency' or 'throughput'")
        if m not in modes:
            modes.append(m)
    if not modes:
        raise ValueError("mode must name at least one of 'latency', 'throughput'")
    return modes


def _check_mode(mode):
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}; expected 'latency' or 'throughput'")
    return mode


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _normalize_spec(spec):
    def _norm(value):
        if isinstance(value, dict):
            return {k: _norm(v) for k, v in value.items()}
        if isinstance(value, list):
            if len(value) == 1 and not isinstance(value[0], (list, dict)):
                return copy.deepcopy(value[0])
            return [_norm(v) for v in value]
        return value

    out = {}
    for key, value in spec.items():
        if key == _THREAD_TOP or key in _LIST_TOPS:
            value = normalize_container(key, value)
        out[key] = _norm(value)
    return out


def _merge_config(override):
    base = copy.deepcopy(_DEFAULT_BENCH)
    if override is None:
        return validate_spec(_normalize_spec(base))
    if not isinstance(override, dict):
        raise ValueError(f"config must be a dict, got {override!r}")
    if not override:
        return validate_spec(_normalize_spec(base))
    merged = _deep_merge(base, dict(override))
    if _backend_mode_map(override.get("backend")):
        merged["backend"] = copy.deepcopy(override["backend"])
    return validate_spec(_normalize_spec(merged))


def _validate_scalar_key(key, value):
    if isinstance(value, (list, tuple)):
        raise ValueError(
            f"config {key} is a single value, got {value!r} (do not wrap it in a list)"
        )
    if key == "samples":
        try:
            iv = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{key} must be an integer, got {value!r}") from e
        if iv < 1:
            raise ValueError(f"{key} must be >= 1, got {value!r}")
    elif key == "seed":
        if value is None:
            return
        if isinstance(value, str):
            return
        try:
            int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"seed must be an integer or string, got {value!r}") from e
    elif key == "iterations":
        if value is None:
            return
        if isinstance(value, dict):
            return
        try:
            iv = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{key} must be an integer, got {value!r}") from e
        if iv < 1:
            raise ValueError(f"{key} must be >= 1, got {value!r}")


def _leaf_alternatives(path, entry):
    if not isinstance(entry, dict):
        return [entry]
    for key in list(entry):
        value = entry[key]
        if isinstance(value, (list, tuple)):
            raise ValueError(
                f"config {path}.{key} must be a single value, got {value!r}; "
                "sweep via multiple container alternatives instead"
            )
    return [entry]


def _expand_config_spec(spec):
    def _alternatives(path, value):
        if isinstance(value, (list, tuple)):
            values = list(value)
            if not values:
                raise ValueError(
                    f"config {path} must be a non-empty list of alternatives, "
                    f"got {value!r}"
                )
            return values
        return [value]

    def _expand_value(path, value):
        if isinstance(value, dict):
            keys = list(value)
            choices = [_expand_value(f"{path}.{key}", value[key]) for key in keys]
            return [
                {key: copy.deepcopy(item) for key, item in zip(keys, combo)}
                for combo in itertools.product(*choices)
            ]
        if isinstance(value, (list, tuple)):
            return [
                item
                for alternative in _alternatives(path, value)
                for item in _expand_value(path, alternative)
            ]
        return [value]

    entries = []

    def _collect(prefix, node):
        for key, value in node.items():
            path = [*prefix, key]
            label = ".".join(str(part) for part in path)
            if not prefix and key == _THREAD_TOP:
                entries.append((path, _alternatives(label, value)))
            elif not prefix and key in _LIST_TOPS and isinstance(value, list):
                entries.append(
                    (
                        path,
                        [
                            entry
                            for i, alt in enumerate(value)
                            for entry in _leaf_alternatives(f"{key}[{i}]", alt)
                        ],
                    )
                )
            else:
                entries.append(([key], _expand_value(label, value)))

    _collect([], spec)
    paths = [p for p, _ in entries]
    choices = [c for _, c in entries]
    combos = []
    for combo in itertools.product(*choices):
        concrete = {}
        for path, value in zip(paths, combo):
            node = concrete
            for part in path[:-1]:
                nxt = node.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    node[part] = nxt
                node = nxt
            node[path[-1]] = copy.deepcopy(value)
        combos.append(concrete)
    return combos


def _concrete_configs(config):
    spec = _merge_config(config)
    combos = _expand_config_spec(spec)
    for combo in combos:
        validate_spec(combo)
    return spec, combos


def _tag_config_columns(df, config):
    if df is None or getattr(df, "empty", False):
        return df
    try:
        cols = config_param_columns(config)
    except Exception:
        return df
    try:
        n = len(df)
    except Exception:
        n = 0
    missing = {}
    for c, v in cols.items():
        if c in df.columns:
            continue
        if isinstance(v, (list, tuple)):
            if len(v) == 1:
                v = v[0]
            if isinstance(v, (list, tuple)):
                missing[c] = [copy.deepcopy(v) for _ in range(n)]
                continue
        missing[c] = v
    if not missing:
        return df
    try:
        extra = pd.DataFrame(missing, index=df.index)
        out = pd.concat([df, extra], axis=1)
        try:
            out.attrs.update(getattr(df, "attrs", {}) or {})
        except Exception:
            pass
        return out
    except Exception:  # pragma: no cover
        for c, v in missing.items():
            try:
                df[c] = v
            except Exception:
                continue
    return df


def _filter_bench_columns(df):
    if df is None or getattr(df, "empty", False):
        return df
    try:
        cols = list(df.columns)
    except Exception:
        return df
    keep = []
    for c in cols:
        s = str(c)
        if s in _BENCH_KEEP_META:
            keep.append(c)
            continue
        if s.startswith("data."):
            keep.append(c)
            continue
        if s.startswith("config."):
            keep.append(c)
            continue
        if s in ("time", "address", "size", "pid", "ip"):
            continue
        keep.append(c)
    try:
        ordered = [c for c in ("operations", "threads") if c in keep]
        ordered += [c for c in keep if c not in ordered]
        if ordered:
            if ordered == cols:
                return df
            return df[ordered]
    except Exception:
        pass
    return df


def _default_unroll_n():
    default = _DEFAULT_BENCH.get("backend", {})
    default = (
        default.get("unroll", {}).get("count") if isinstance(default, dict) else None
    )
    return default if default is not None else 5


def _code_data(parsed, code):
    if code:
        parsed["code"] = sorted(code)
    return parsed


def _as_normalized_data(data):
    pages = set()
    if isinstance(data, dict):
        for page in data.get("code") or ():
            try:
                pages.add(int(page) & ~(mmap.PAGESIZE - 1))
            except (TypeError, ValueError):
                continue
    code = sorted(pages)
    if not data:
        return _code_data({"regs": {}, "mem": {}}, code)
    if isinstance(data, dict):
        regs = {}
        mem = {}
        heap = {}
        for base, size in (data.get("heap") or {}).items():
            base = _heap_base(base)
            if base is None:
                continue
            try:
                heap[f"0x{base:x}"] = int(size)
            except (TypeError, ValueError):
                continue
        flat = {
            k: v for k, v in data.items() if k not in ("regs", "mem", "code", "heap")
        }
        if flat:
            parsed = _config_data({"data": flat})
            regs.update(parsed["regs"])
            mem.update(parsed["mem"])
        for bucket in ("regs",):
            regs.update(_normalize_data_regs(data.get(bucket) or {}))
        for key, value in (data.get("mem") or {}).items():
            for addr, val in _mem_entries(str(key), value):
                mem[addr] = val
        out = _code_data({"regs": regs, "mem": mem}, code)
        if heap:
            out["heap"] = heap
        return out
    return _code_data(_config_data({"data": data}), code)


def _merge_data(base, override):
    b = _as_normalized_data(base)
    o = _as_normalized_data(override)
    b["regs"].update(o["regs"])
    for k, v in o["mem"].items():
        b["mem"][k] = v
    heap = dict(b.get("heap") or {})
    heap.update(o.get("heap") or {})
    if heap:
        b["heap"] = heap
    _code_data(b, set(b.get("code") or ()) | set(o.get("code") or ()))
    if not b.get("code"):
        b.pop("code", None)
    return b


def _merge_fallback_data(base, grown_list):
    merged = {
        "regs": dict((base or {}).get("regs") or {}),
        "mem": dict((base or {}).get("mem") or {}),
    }
    heap = dict((base or {}).get("heap") or {})
    code = set((base or {}).get("code") or ())
    for grown in grown_list or ():
        try:
            regs = (grown or {}).get("regs") or {}
            mem = (grown or {}).get("mem") or {}
            pages = (grown or {}).get("code") or ()
            grown_heap = (grown or {}).get("heap") or {}
        except Exception:
            continue
        for k, v in regs.items():
            if k not in merged["regs"]:
                merged["regs"][k] = v
        for k, v in mem.items():
            merged["mem"].setdefault(k, v)
        heap.update(grown_heap)
        code.update(pages)
    if heap:
        merged["heap"] = heap
    _code_data(merged, code)
    return merged


def _order_bench_columns(df, events=None):
    try:
        cols = list(df.columns)
    except Exception:
        return df
    present = set(cols)
    lead = [c for c in _IDENTITY_COLUMNS if c in present]
    config_cols = sorted(c for c in cols if str(c).startswith("config."))
    data_cols = sorted(c for c in cols if str(c).startswith("data."))
    meta = [c for c in ("operations", "threads") if c in present]
    ordered = lead + config_cols + data_cols + meta
    seen = set(ordered)
    if events:
        try:
            for e in events:
                if e in present and e not in seen:
                    ordered.append(e)
                    seen.add(e)
        except Exception:
            pass
    ordered += [c for c in cols if c not in seen]
    if ordered == cols:
        return df
    try:
        return df[ordered]
    except Exception:
        return df


def _is_file(text):
    if any(c.isspace() for c in text) or "," in text:
        return False
    if os.path.exists(text):
        return True
    return os.sep in text or text.startswith((".", "~"))


def _asm_region(file, target):
    text = _asm_read(file)
    target = asm_source_target(target)
    if not target:
        raise ValueError("a label or a `begin..end` region is required")
    begin, dotdot, end = target.partition("..")
    begin = begin.strip()
    end = end.strip() if dotdot else None
    if begin == end:
        raise ValueError(
            f"empty region {target!r}: {begin!r} and {end!r} are the same label"
        )
    bodies = {name: (position, size) for name, position, size in asm_labels(file)}
    known = ", ".join(n for n in bodies if not n.startswith(".")) or "none"
    if begin not in bodies:
        raise ValueError(f"cannot resolve {target!r}; labels: {known}")
    position, size = bodies[begin]
    if end is not None:
        if end not in bodies:
            raise ValueError(f"cannot resolve {end!r}; labels: {known}")
        if bodies[end][0] < position:
            raise ValueError(
                f"empty region {target!r}: {end!r} starts before {begin!r}"
            )
        size = bodies[end][0] - position
    out = _asm_body(text[position : position + size])
    if not out:
        raise ValueError(f"region {target!r} has no instructions")
    return "; ".join(out)


def _asm_body(text):
    out, returned = [], False
    arch = get_arch()
    for raw in str(text or "").splitlines():
        for statement in raw.split(";"):
            line = " ".join(_ASM_COMMENT.sub(" ", statement).split())
            if not line:
                continue
            head, colon, rest = line.partition(":")
            if colon and _ASM_LABEL.fullmatch(head.strip()):
                out.append(f"{head.strip()}:")
                line = rest.strip()
                returned = False
            if not line or line.startswith("."):
                continue
            if returned:
                continue
            mnemonic = line.split(" ", 1)[0].lower()
            try:
                returned = arch.is_return_mnemonic(mnemonic)
            except Exception:
                returned = False
            if not returned:
                out.append(line)
    while out and out[-1].endswith(":"):
        out.pop()
    return out


_TARGET_KEYS = frozenset({"file", "symbol", "region", "asm"})


def _target_parts(target):
    if not isinstance(target, dict):
        return parse_code(target)

    unknown = sorted(str(key) for key in target if key not in _TARGET_KEYS)
    if unknown:
        raise ValueError(
            f"unknown target key(s) {', '.join(unknown)}; expected "
            "file, symbol, region, or asm"
        )
    file = target.get("file")
    symbol = target.get("symbol")
    reg = target.get("region")
    asm = target.get("asm")
    given = [
        label
        for label, value in (
            ("symbol", symbol),
            ("region", reg),
            ("asm", asm),
        )
        if value is not None
    ]
    if len(given) != 1:
        raise ValueError(
            "target must contain exactly one of 'asm', 'symbol', or 'region', "
            f"got {', '.join(given) or 'none'}"
        )
    if asm is not None:
        if file is not None:
            raise ValueError("target with 'asm' must not contain 'file'")
        return None, None, str(asm)
    if file is None or not str(file).strip():
        raise ValueError(f"target {target!r} needs a non-empty 'file'")
    file = str(file).strip()
    if symbol is not None:
        if isinstance(symbol, (list, tuple)):
            raise ValueError(
                f"target symbol {symbol!r} must be a single symbol; use "
                "'region' or BEGIN..END for a range"
            )
        resolved = region(symbol)
        if isinstance(resolved, tuple):
            raise ValueError(
                f"target symbol {symbol!r} is a region; use 'region' instead"
            )
        return file, resolved, None
    reg = region(reg)
    if not isinstance(reg, tuple):
        raise ValueError(
            f"target region {target.get('region')!r} must be BEGIN..END; e.g. "
            "{'file': 'a.out', 'region': ['hot_begin', 'hot_end']}"
        )
    return file, reg, None


def _resolve_target(target=None, name=None):
    if target is None:
        raise TypeError(
            "pass target= as a dict such as {'asm': 'mov eax, 42'}, "
            "{'file': 'a.out', 'symbol': 'main'}, or "
            "{'file': 'a.out', 'region': ['hot_begin', 'hot_end']}; "
            "short strings also work, e.g. target='rdtsc' or target='a.out:main'"
        )
    raw = target
    file, resolved, code = _target_parts(target)
    if code is not None and file is None:
        return None, None, code, name, raw
    if file is None:
        raise ValueError(
            f"invalid target {target!r}; pass an asm snippet, FILE:SYMBOL, or "
            "FILE:BEGIN..END"
        )
    snippet = asm_source_code(file, resolved)
    if snippet:
        code = snippet
        name = name or asm_source_target(resolved)
        return None, None, code, name, raw
    if _is_asm_source(file):
        raise ValueError(f"cannot resolve {resolved!r} in {file!r}")
    return file, resolved, code, name, raw
