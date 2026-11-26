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
import ctypes.util
import datetime
import functools
import hashlib
import mmap
import os
import random
import re
import select
import shlex
import sys
import threading
import time
import warnings
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd

from . import core
from .arch import arch as get_arch
from .cfg import (
    _BACKENDS,
    _COMBINE_META,
    _DEFAULT_BENCH,
    _DEFAULT_ITER_MAX,
    _DEFAULT_ITER_MIN,
    _LIST_TOPS,
    _MODES,
    _STACK_REFRESH_BYTES_MAX,
    _STACK_REFRESH_WINDOW,
    _affinity_guard,
    _affinity_spec,
    _apply_code_tier_defaults,
    _apply_heap_data,
    _apply_seed,
    _backend_mode_map,
    _branch_config,
    _cache_state_sensitive,
    _canonical_data_reg_key,
    _check_mode,
    _code_level_addrs,
    _collect_mem_addrs,
    _concrete_configs,
    _config_seed_int,
    _container,
    _default_unroll_n,
    _entry_data,
    _explicit_bound_values,
    _fake_output,
    _fill_evict_tables,
    _filter_bench_columns,
    _generate_seed,
    _grow_fallback_data,
    _heap_covers,
    _heap_settings,
    _host_target,
    _is_harness_fault,
    _is_random_seed,
    _make_rng,
    _mem_level_int_addrs,
    _merge_data,
    _merge_fallback_data,
    _missing_arg_regs,
    _modes,
    _names,
    _needs_pointer_fallback,
    _normalize_target,
    _numa_guard,
    _order_bench_columns,
    _parse_affinity,
    _parse_numa,
    _parse_priority,
    _per_iter_data,
    _priority_guard,
    _refresh_per_iter_data,
    _resolve_reg_levels,
    _resolve_target,
    _sequence_probe_regs,
    _static_extra_addrs,
    _tag_config_columns,
    _thread_store,
    check_data,
    data_param_columns,
    validate_spec,
)
from .core import (
    _DURATION_TIME,
    _FORK_LOCK,
    _INF,
    _MEMORY_CHECK_EVERY,
    _MODEL_PROBE_TIMEOUT,
    _OUTPUT_LOCK,
    _POKE_LOCK,
    _RUN_MEMORY_FRACTION,
    _SIMULATED_RANGES,
    _STATE_LOCK,
    _Args,
    _available_memory,
    _check_growth,
    _check_target_runnable,
    _content_hash8,
    _current_priority_value,
    _debug_log,
    _drop_unmappable_models,
    _func_prototype,
    _ignore_runtime_warnings,
    _jit_page,
    _jit_release,
    _json_canonical,
    _kill,
    _locked,
    _map_data_pages,
    _map_scratch_reg_pages,
    _measure_threads_count,
    _measure_threads_enter,
    _measure_threads_exit,
    _measure_threads_forget,
    _memory_record,
    _memory_reset,
    _min_mmap_addr,
    _poke_addr_safe,
    _read_raw_counters,
    _reset_fatal_handlers,
    _run_memory,
    _set_simulated_ranges,
    _split_list,
    _stable,
    _text,
    _waitpid,
)
from .data import _IDENTITY_COLUMNS, dumps_json
from .exec import (
    resolve_exec,
    retire_exit_hooks,
    write_perf_map,
)
from .info import (
    _CPUINFO_FIELDS,
    _unresolved,
    cpuinfo,
    targets as resolve_targets,
)
from .mca import (
    _SOLVE_CACHE_MAX,
    _angr_symbol_addr,
    _bench_project,
    _call_seq_asm,
    _explore_asm_cached,
    _explore_target_cached,
    _left_align_asm,
    _merge_code_pages,
    _models_cached,
    _project_cache_id,
    _prune_for_target,
    _reserved_addresses,
    _setup_names,
    _solution_summary,
    _static_table_addrs,
    _warn_exploration,
    explored_data_dict,
)

_MEASURE_BARRIER_TIMEOUT = 30
_MEASURE_JOIN_TIMEOUT = 3600
_MEASURE_ACTIVE = {}
_MEASURE_ACTIVE_LOCK = threading.RLock()
_RUNTABLE_MODELS = {}
_LABEL_DEF = re.compile(r"(?m)(?:^|;)\s*([A-Za-z_.$][\w.$]*)\s*:")


def benchmark(
    target=None,
    name=None,
    *,
    mode=None,
    config=None,
    data=None,
    event=None,
    setup=None,
    teardown=None,
    argv=None,
    env=None,
    backend=None,
    auto_backend=False,
    debug=False,
):
    if config is None:
        config = {"seed": 0}
    elif isinstance(config, dict):
        seed = config.get("seed")
        if seed is None:
            config = {**config, "seed": 0}
        elif _is_random_seed(seed):
            config = {**config, "seed": _generate_seed()}
    file, resolved_target, code, name, _raw = _resolve_target(target, name)

    modes = _modes(mode if mode is not None else ["latency", "throughput"])

    if backend is None and isinstance(config, dict):
        backend = _backend_mode_map(config.get("backend"))

    events = _normalize_groups(event)

    spec, combos = _concrete_configs(config)
    _pin_default_affinity(spec, events)
    simulated_ranges = _set_simulated_ranges(None)
    _memory_reset()
    if debug:
        _debug_json("config spec", spec)
        _debug_json("config combos", combos)
    data = _merge_data(None, data)
    check_data(data)
    argv = _argv_words(file, argv)
    env = _env_words(env)
    _check_argv_regs(data, argv)
    combos = _prune_for_target(file, resolved_target, code, combos, data, setup, debug)
    if debug:
        _debug_json("effective config combos", combos)
    frames = []
    frame_combos = []
    per_mode = _mode_backends(backend)
    auto_keys = None
    if auto_backend and auto_backend is not True:
        auto_keys = frozenset(auto_backend)
    for m in modes:
        for combo in combos:
            combo_backend = per_mode.get(m, None) if per_mode else backend
            if (
                auto_backend
                and combo_backend == "unroll"
                and _cache_state_sensitive(combo, keys=auto_keys)
            ):
                combo_backend = "loop"
            df = _bench_one(
                file=file,
                target=resolved_target,
                code=code,
                name=name,
                mode=m,
                config=copy.deepcopy(combo),
                data=data,
                event=events,
                setup=setup,
                teardown=teardown,
                argv=argv,
                env=env,
                backend=combo_backend,
                debug=debug,
                simulated_ranges=simulated_ranges,
            )
            frames.append(df)
            frame_combos.append(combo)
    if not frames:
        raise ValueError("config expands to no combinations")

    tagged = []
    for combo, df in zip(frame_combos, frames):
        resolved = (getattr(df, "attrs", {}) or {}).get("config")
        df = _tag_config_columns(df, resolved if isinstance(resolved, dict) else combo)
        try:
            filtered = _filter_bench_columns(df)
            try:
                filtered.attrs.update(getattr(df, "attrs", {}) or {})
            except Exception:
                pass
            tagged.append(filtered)
        except Exception:
            tagged.append(df)
    frames = tagged
    if len(frames) == 1:
        try:
            df = _order_bench_columns(frames[0])
        except Exception:
            df = frames[0]
        return _benchmark_frame(df)

    df = pd.concat(frames, axis=0)
    first = frames[0]
    _attrs = ("info", "file", "data", "code", "state", "distribution", "argv", "env")
    for attr in _attrs:
        if attr in first.attrs:
            df.attrs[attr] = first.attrs[attr]
    df.attrs["config"] = spec
    try:
        binary = first.attrs.get("info", {}).get("binary")
    except Exception:
        binary = None
    try:
        df.attrs["id"] = _id_hash(data, spec, binary)
    except Exception:
        pass
    try:
        df = _order_bench_columns(df)
    except Exception:
        pass
    return _benchmark_frame(df)


def to_json(df, indent=4):
    env = _record_envelope(df)
    return dumps_json(env, indent=indent)


def add_param_columns(df, data):
    if df is None or getattr(df, "empty", False):
        return df
    try:
        cols = data_param_columns(data)
    except Exception:
        return df
    new_cols = {c: v for c, v in cols.items() if c not in df.columns}
    if not new_cols:
        try:
            return _order_bench_columns(df)
        except Exception:
            return df
    try:
        const = pd.DataFrame(
            {
                c: pd.Series([copy.deepcopy(v) for _ in range(len(df))], index=df.index)
                for c, v in new_cols.items()
            }
        )
        if len(const.columns):
            _ended = pd.concat([df, const], axis=1)
            if len(_ended) == len(df):
                df = _ended
            else:
                for c in list(const.columns):
                    try:
                        df[c] = const[c]
                    except Exception:
                        continue
    except Exception:  # pragma: no cover
        for col, val in new_cols.items():
            try:
                df[col] = val
            except Exception:
                continue
    try:
        df = _order_bench_columns(df)
    except Exception:
        pass
    return df


def _bench_to_json(self, *args, **kwargs):
    if not args and not (set(kwargs) - {"indent"}):
        return to_json(self, **kwargs)
    return pd.DataFrame.to_json(self, *args, **kwargs)


def _benchmark_frame(df):
    df.to_json = MethodType(_bench_to_json, df)
    return df


def _needs_slots(models, data):
    try:
        if models is not None and len(models) > 1:
            return True
    except Exception:
        pass
    try:
        regs = (data or {}).get("regs") or {}
        for value in regs.values():
            if isinstance(value, (list, tuple)):
                return True
        mem = (data or {}).get("mem") or {}
        for value in mem.values():
            if isinstance(value, (list, tuple)):
                return True
    except Exception:
        pass
    return False


def _explicit_sequence_keys(data):
    try:
        columns = data_param_columns(data)
    except Exception:
        return []
    return [key for key, value in columns.items() if isinstance(value, (list, tuple))]


def _slot_columns(buf, meta, models, data, indices=None):
    try:
        import numpy as _np
    except Exception:
        return None
    if buf is None or not isinstance(meta, dict):
        return None
    try:
        n = int(meta.get("k", 0) or 0) - 1
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    try:
        reg_col = dict(meta.get("reg_col") or {})
        val_col = dict(meta.get("val_col") or {})
    except Exception:
        return None
    if not reg_col and not val_col:
        return None
    l1i_set = set(meta.get("l1i_addrs") or ())
    wanted_mem = set()
    try:
        wanted_mem |= {
            addr
            for addr in _collect_mem_addrs(models or [])
            if not _heap_covers(data, addr)
        }
    except Exception:
        pass
    try:
        for key in (data or {}).get("mem") or {}:
            s = str(key).strip()
            if s.endswith(":"):
                s = s[:-1].strip()
            addr = int(s, 0)
            if not _heap_covers(data, addr):
                wanted_mem.add(addr)
    except (TypeError, ValueError):
        pass
    reg_names = [name for name in (meta.get("reg_names") or ()) if name in reg_col]
    mem_addrs = [
        a
        for a in (meta.get("mem_addrs") or ())
        if a in val_col and a in wanted_mem and a not in l1i_set
    ]
    if not reg_names and not mem_addrs:
        return None
    series = {}
    try:
        idx = None
        if indices is not None:
            try:
                idx = _np.asarray(list(indices), dtype=_np.int64)
                idx = idx[(idx >= 0) & (idx < n)]
                if idx.size == 0:
                    return None
            except Exception:
                idx = None
        mask = _np.uint64(0xFFFFFFFFFFFFFFFF)
        for name in reg_names:
            col = int(reg_col[name])
            vals = _np.asarray(buf[col, 1 : n + 1], dtype=_np.uint64)[::-1]
            if idx is not None:
                try:
                    vals = vals[idx]
                except Exception:
                    pass
            try:
                canon = _canonical_data_reg_key(name)
            except Exception:
                canon = str(name).strip().lower()
            series[f"data.{canon}"] = (vals & mask).tolist()
        for addr in mem_addrs:
            col = int(val_col[addr])
            vals = _np.asarray(buf[col, 1 : n + 1], dtype=_np.uint64)[::-1]
            if idx is not None:
                try:
                    vals = vals[idx]
                except Exception:
                    pass
            series[f"data.0x{int(addr):x}"] = (vals & mask).tolist()
    except (TypeError, ValueError, IndexError):
        return None
    return series or None


def _argv_words(file, argv):
    if argv is None:
        return None
    if isinstance(argv, str):
        argv = shlex.split(argv)
    words = [str(a) for a in argv]
    if words and not words[0]:
        words[0] = str(file) if file else "a.out"
    return words or [str(file) if file else "a.out"]


def _env_words(env):
    if env is None:
        return None
    if isinstance(env, str):
        env = [env]
    words = [str(e) for e in env]
    if any("=" not in w for w in words):
        raise ValueError(f"env entries must be KEY=VALUE, got {env!r}")
    return words or None


def _check_argv_regs(data, argv):
    if not argv:
        return
    taken = dict(zip(get_arch()._ARGV_REGS, ("argc", "argv", "envp")))
    for reg in (data or {}).get("regs") or {}:
        try:
            canon = get_arch()._canonical_reg(_canonical_data_reg_key(reg))
        except Exception:
            canon = str(reg).strip().lower()
        if canon in taken:
            raise ValueError(
                f"argv already sets {canon} to {taken[canon]}; drop --data.{canon} "
                f"or drop the argv arguments"
            )


def _debug_json(title, obj):
    try:
        text = dumps_json(obj, indent=2)
    except Exception:  # pragma: no cover
        try:
            text = str(obj)
        except Exception:
            text = "<unprintable>"
    _debug_log(f"{title}:\n{text}")


def _as_records(df):
    index = getattr(df, "index", None)
    names = list(getattr(index, "names", []) or [])
    try:
        if isinstance(df.index, pd.MultiIndex) or any(
            n in names for n in _IDENTITY_COLUMNS
        ):
            return df.reset_index()
    except Exception:
        pass
    return df


def _json_key(value):
    if isinstance(value, dict):
        return tuple(sorted((str(k), _json_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_json_key(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_json_key(v) for v in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _column_uniques(results, columns):
    try:
        return results.nunique(dropna=False)
    except TypeError:
        counts = {}
        for column in columns:
            try:
                counts[column] = int(results[column].nunique(dropna=False))
            except TypeError:
                counts[column] = len({_json_key(v) for v in results[column].tolist()})
            except Exception:
                counts[column] = None
        return pd.Series(counts)
    except Exception:
        return None


def _record_envelope(df):
    rec = _as_records(df)
    attrs = getattr(df, "attrs", {}) or {}
    config = dict(attrs.get("config", {}) or {})
    info = dict(attrs.get("info", {}) or {})
    data = attrs.get("data", None)
    binary = info.get("binary") if isinstance(info, dict) else None
    run_id = _id_hash(data, config, binary)
    try:
        file_label = _text(rec["file"].iloc[0]) if len(rec) else ""
        name = _text(rec["name"].iloc[0]) if len(rec) else ""
    except Exception:
        file_label = name = ""
    file_label = file_label or _text(attrs.get("file")) or ""
    name = name or _text(attrs.get("name")) or ""
    if run_id and name:
        name = f"{name}-{run_id}"
    if isinstance(info, dict):
        try:
            from .info import hostname as _hostname_fn
        except Exception:
            _hostname_fn = None
        host = info.get("hostname")
        if host is None and _hostname_fn is not None:
            try:
                host = _hostname_fn()
            except Exception:
                host = None
        info = {"cpu": info.get("cpu"), "hostname": host}
    drop = {"file", "name", "time"}
    try:
        if len(rec):
            for col in ("file", "name"):
                if col in rec.columns and rec[col].nunique() > 1:
                    drop.discard(col)
    except Exception:
        pass
    results = rec.drop(columns=[c for c in drop if c in rec.columns])
    try:
        columns = list(results.columns)
    except Exception:
        columns = []
    flat_all = []
    nested_all = {}
    has_config = "config" in columns
    has_data = "data" in columns
    for column in columns:
        head, _, rest = str(column).partition(".")
        if head == "config" and rest and not has_config:
            parts = tuple(p for p in rest.split(".") if p)
            if parts:
                nested_all.setdefault(head, []).append((column, parts))
                continue
        elif head == "data" and rest and not has_data:
            parts = tuple(p for p in rest.split(".") if p)
            if parts:
                nested_all.setdefault(head, []).append((column, parts))
                continue
        flat_all.append(column)
    try:
        nrows = len(results)
    except Exception:
        nrows = 0
    shared = {}
    flat_cols = list(flat_all)
    nested_heads = dict(nested_all)
    vary_columns = list(columns)
    if nrows > 32 and columns:
        uniq = _column_uniques(results, columns)
        if uniq is not None:
            try:
                const = set(uniq[uniq == 1].index.tolist())
            except Exception:
                const = set()
            if const:
                flat_cols = []
                nested_heads = {}
                vary_columns = []
                for column in flat_all:
                    if column in const:
                        try:
                            v = results[column].iloc[0]
                        except Exception:  # pragma: no cover
                            vary_columns.append(column)
                            flat_cols.append(column)
                            continue
                        try:
                            v = _json_scalar(v)
                        except Exception:
                            pass
                        if isinstance(v, (dict, list, tuple, set)):
                            flat_cols.append(column)
                            vary_columns.append(column)
                            continue
                        shared[column] = v
                    else:
                        flat_cols.append(column)
                        vary_columns.append(column)
                for head, spec in nested_all.items():
                    kept = []
                    for column, parts in spec:
                        if column in const:
                            try:
                                v = results[column].iloc[0]
                            except Exception:  # pragma: no cover
                                kept.append((column, parts))
                                vary_columns.append(column)
                                continue
                            try:
                                v = _json_scalar(v)
                            except Exception:
                                pass
                            if v is None:
                                continue
                            if isinstance(v, (dict, list, tuple, set)):
                                kept.append((column, parts))
                                vary_columns.append(column)
                                continue
                            node = shared.setdefault(head, {})
                            for part in parts[:-1]:
                                nxt = node.get(part)
                                if not isinstance(nxt, dict):
                                    nxt = {}
                                    node[part] = nxt
                                node = nxt
                            node[parts[-1]] = v
                        else:
                            kept.append((column, parts))
                            vary_columns.append(column)
                    if kept:
                        nested_heads[head] = kept
    if vary_columns:
        try:
            vary_results = results[vary_columns]
        except Exception:
            vary_results = results
    else:
        try:
            vary_results = results.iloc[0:0]
        except Exception:
            vary_results = results
    try:
        col_lists = vary_results.to_dict(orient="list")
    except Exception:  # pragma: no cover
        col_lists = {}
        for col in list(vary_results.columns):
            try:
                col_lists[col] = vary_results[col].tolist()
            except Exception:
                col_lists[col] = [None] * nrows
    for col in list(vary_results.columns):
        try:
            dtype = vary_results[col].dtype
        except Exception:
            dtype = None
        if dtype is not None:
            try:
                import pandas.api.types as _pdt

                if _pdt.is_integer_dtype(dtype) or _pdt.is_bool_dtype(dtype):
                    continue
                if _pdt.is_float_dtype(dtype):
                    try:
                        import numpy as _npf

                        arr = vary_results[col].to_numpy(dtype=float, copy=False)
                        if bool(_npf.isfinite(arr).all()):
                            continue
                        finite = _npf.isfinite(arr)
                        vals = arr.tolist()
                        for i, ok in enumerate(finite):
                            if not ok:
                                vals[i] = None
                        col_lists[col] = vals
                        continue
                    except Exception:
                        pass
                try:
                    if not bool(vary_results[col].isna().any()):
                        continue
                except Exception:
                    pass
            except Exception:
                pass
        try:
            vals = col_lists.get(col, [])
            col_lists[col] = [_json_scalar(v) for v in vals]
        except Exception:
            pass
    rows = []
    if nrows and (flat_cols or nested_heads):
        try:
            n = len(next(iter(col_lists.values()))) if col_lists else 0
        except Exception:
            n = nrows
        flat_lists = [(column, col_lists.get(column)) for column in flat_cols]
        nested_specs = [
            (
                head,
                [(col_lists.get(column), parts) for column, parts in spec],
            )
            for head, spec in nested_heads.items()
        ]
        for i in range(n):
            out = {}
            for column, values in flat_lists:
                try:
                    out[column] = values[i]
                except Exception:
                    continue
            for head, spec in nested_specs:
                built = {}
                for values, parts in spec:
                    try:
                        value = values[i]
                    except Exception:
                        continue
                    if value is None:
                        continue
                    if len(parts) == 1:
                        built[parts[0]] = value
                        continue
                    node = built
                    for part in parts[:-1]:
                        nxt = node.get(part)
                        if not isinstance(nxt, dict):
                            nxt = {}
                            node[part] = nxt
                        node = nxt
                    node[parts[-1]] = value
                if built:
                    out[head] = built
            rows.append(out)
    if not vary_columns and nrows:
        try:
            rows = [{} for _ in range(int(nrows))]
        except Exception:
            pass
    env = {
        "file": file_label,
        "name": name,
        "id": run_id,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "info": info,
        "output": rows,
    }
    if shared:
        env["shared"] = shared
    return env


def _json_int(value):
    if -(1 << 63) <= value < (1 << 64):
        return value
    return hex(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, int) and not isinstance(value, bool):
        return _json_int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json_scalar(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return _json_int(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value != value or value == _INF or value == -_INF:
            return None
        return value
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, (dict, list, tuple)):
        return _json_safe(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _collect(
    config,
    iterations,
    runs,
    mode,
    code,
    setup,
    teardown,
    models,
    events,
    indices,
    overhead,
    levels,
    predictable,
    rng,
    arch,
    data=None,
    map_name=None,
    branch_cfg=None,
    mem_levels=None,
    extra_addrs=None,
    tlb_levels=None,
    argv_frame=None,
    debug=False,
    slots_out=None,
    overhead_iterations=None,
    raw_counters=None,
    raw_out=None,
    stack_info=None,
    sample_runs=None,
    force_guard=False,
):
    events = _names(events)
    n_events = len(events)
    try:
        _heap_size, heap_mode = _heap_settings(config)
    except Exception:
        heap_mode = None
    sequence_keys = _explicit_sequence_keys(data)
    track_sequence = bool(sequence_keys) and not predictable and mode != "throughput"
    stack_info = _prepare_stack_info(stack_info, iterations, mode, rng, arch)
    try:
        full_asm, _meta, per_iter_buf = _build_loop_asm(
            iterations,
            mode,
            code,
            setup,
            teardown,
            models,
            events,
            indices,
            levels,
            predictable,
            rng,
            arch,
            data=data,
            branch_cfg=branch_cfg,
            mem_levels=mem_levels,
            extra_addrs=extra_addrs,
            tlb_levels=tlb_levels,
            argv_frame=argv_frame,
            heap_mode=heap_mode,
            stack_info=stack_info,
            force_guard=force_guard,
        )
    except Exception as ex:
        _asm_error(code, ex)
    try:
        code_bytes = arch.assemble(full_asm)
    except Exception as ex:
        _asm_error(code, ex)

    page = _jit_page(code_bytes, in_use=True)
    try:
        try:
            _write_jit_map(
                ctypes.addressof(ctypes.c_char.from_buffer(page)),
                len(code_bytes),
                map_name or ",".join(events),
            )
        except Exception:
            pass

        inputs = np.zeros((iterations + 1, 1), dtype=np.uint64)
        n_samples = 1 if mode == "throughput" else iterations
        outputs = np.zeros((n_samples, n_events), dtype=np.uint64)

        data_ptr = per_iter_buf.ctypes.data if per_iter_buf is not None else None
        args = _Args(iterations, inputs.ctypes.data, outputs.ctypes.data, data_ptr)
        fn = ctypes.CFUNCTYPE(None, ctypes.POINTER(_Args))(
            ctypes.addressof(ctypes.c_char.from_buffer(page))
        )

        oh = np.nan_to_num(
            np.broadcast_to(np.asarray(overhead, dtype=np.float64), (n_events,)),
            nan=0.0,
        )
        try:
            if (
                mode == "throughput"
                and overhead_iterations
                and iterations
                and int(overhead_iterations) != int(iterations)
            ):
                oh = oh * (float(iterations) / float(overhead_iterations))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        if debug:
            _debug_log(
                f"harness asm (iterations={iterations} runs={runs} mode={mode} "
                f"events={events}):\n{_left_align_asm(full_asm)}"
            )
        stack_key = None
        if stack_info:
            try:
                stack_key = (
                    stack_info.get("base"),
                    stack_info.get("size"),
                    str(stack_info.get("mode") or ""),
                    stack_info.get("src"),
                    _stable(stack_info.get("frame")),
                )
            except Exception:  # pragma: no cover
                stack_key = str(stack_info.get("base"))
        _check_target_runnable(
            fn,
            args,
            map_name,
            key=(
                mode,
                _stable(code),
                _stable(setup),
                _stable(teardown),
                _stable(data),
                _stable(models),
                _stable(levels),
                _stable(mem_levels),
                _stable(tlb_levels),
                _stable(branch_cfg),
                _stable(argv_frame),
                stack_key,
                str(heap_mode or ""),
                str(map_name or ""),
            ),
            config=config,
        )
        budget = int(_available_memory() * _RUN_MEMORY_FRACTION)
        single_thread = _measure_threads_count() <= 1 and not debug
        output_scope = (
            _fake_output(config) if single_thread else contextlib.nullcontext()
        )
        if mode == "throughput":
            diffs = np.full((runs, n_events), np.nan)
            raw_all = np.full((runs, n_events), np.nan)
            with (
                _numa_guard(config),
                _affinity_guard(config),
                _priority_guard(config),
                output_scope,
            ):
                for i in range(runs):
                    _refresh_per_iter_data(
                        per_iter_buf,
                        _meta,
                        iterations,
                        models,
                        levels,
                        predictable,
                        rng,
                        arch,
                        data=data,
                        branch_cfg=branch_cfg,
                        mem_levels=mem_levels,
                        extra_addrs=extra_addrs,
                        tlb_levels=tlb_levels,
                        heap_mode=heap_mode,
                    )
                    _run_measured(
                        fn,
                        args,
                        map_name,
                        i,
                        config,
                        budget=budget,
                        raw_counters=raw_counters,
                        raw_out=raw_out,
                        managed_output=single_thread,
                    )
                    raw = np.asarray(outputs, dtype=np.float64)[0]
                    raw_all[i] = raw
                    diffs[i] = np.maximum(raw - oh, 0.0)
                    if debug:
                        _debug_log(f"run {i}: {dict(zip(events, diffs[i].tolist()))}")
            with np.errstate(invalid="ignore"), _ignore_runtime_warnings():
                result = (
                    np.nanmin(diffs, axis=0) if runs else np.full((n_events,), np.nan)
                )
            if sample_runs is not None:
                _throughput_fallback_by_sample(
                    diffs, raw_all, oh, events, sample_runs, debug
                )
            else:
                all_nan = [
                    e
                    for e in range(n_events)
                    if runs and bool(np.all(np.isnan(diffs[:, e])))
                ]
                _fallback_all_nan_to_raw(diffs, raw_all, result, oh, events, debug)
                for e in all_nan:
                    try:
                        if np.isfinite(result[e]):
                            diffs[:, e] = result[e]
                    except (TypeError, ValueError):
                        continue
            with np.errstate(invalid="ignore"):
                return diffs

        diffs = np.full((runs, iterations, n_events), np.nan)
        raw_min = np.full((n_events,), np.inf, dtype=np.float64)
        run_sequences = []
        with (
            _numa_guard(config),
            _affinity_guard(config),
            _priority_guard(config),
            output_scope,
        ):
            for i in range(runs):
                _refresh_per_iter_data(
                    per_iter_buf,
                    _meta,
                    iterations,
                    models,
                    levels,
                    predictable,
                    rng,
                    arch,
                    data=data,
                    branch_cfg=branch_cfg,
                    mem_levels=mem_levels,
                    extra_addrs=extra_addrs,
                    tlb_levels=tlb_levels,
                    heap_mode=heap_mode,
                )
                if track_sequence:
                    try:
                        run_slots = _slot_columns(per_iter_buf, _meta, models, data)
                        cols = [run_slots.get(key) for key in sequence_keys]
                        if all(
                            col is not None and len(col) == iterations for col in cols
                        ):
                            run_sequences.append(list(zip(*cols)))
                        else:
                            run_sequences.append(None)
                    except Exception:
                        run_sequences.append(None)
                _run_measured(
                    fn,
                    args,
                    map_name,
                    i,
                    config,
                    budget=budget,
                    raw_counters=raw_counters,
                    raw_out=raw_out,
                    managed_output=single_thread,
                )
                raw = np.asarray(outputs, dtype=np.float64)
                np.minimum(raw_min, raw.min(axis=0), out=raw_min)
                diffs[i] = np.maximum(raw - oh, 0.0)
                if debug:
                    try:
                        with _ignore_runtime_warnings():
                            per_min = np.nanmin(diffs[i], axis=0)
                            per_med = np.nanmedian(diffs[i], axis=0)
                    except Exception:
                        per_min = per_med = None
                    _debug_log(
                        f"run {i}: "
                        f"min={dict(zip(events, np.asarray(per_min).tolist()))} "
                        f"median={dict(zip(events, np.asarray(per_med).tolist()))}"
                    )
        if (
            track_sequence
            and run_sequences
            and all(sequence is not None for sequence in run_sequences)
        ):
            groups = {}
            for run, sequence in enumerate(run_sequences):
                for index, key in enumerate(sequence):
                    groups.setdefault(tuple(key), []).append(diffs[run, index])
            keys = list(groups)
            with np.errstate(invalid="ignore"), _ignore_runtime_warnings():
                result = np.vstack(
                    [np.nanmean(np.asarray(groups[key]), axis=0) for key in keys]
                )
            if slots_out is not None:
                slots_out.append(
                    {
                        name: [key[index] for key in keys]
                        for index, name in enumerate(sequence_keys)
                    }
                )
            return result
        randomized = (
            isinstance(branch_cfg, dict)
            and branch_cfg.get("prediction") == "unpredictable"
        )
        if slots_out is not None and mode != "throughput" and not track_sequence:
            try:
                if _needs_slots(models, data):
                    _slots = _slot_columns(per_iter_buf, _meta, models, data)
                else:
                    _slots = None
            except Exception:
                _slots = None
            if _slots:
                try:
                    slots_out.append(_slots)
                except Exception:
                    pass
        result = _aggregate_over_runs(diffs, events, minimize=not randomized)
        _fallback_all_nan_to_raw_min(diffs, raw_min, result, oh, events, debug)
        return result

    finally:
        _jit_release(page)


def _is_timing_event(event):
    name = str(core._event_base(event) or event).strip().lower()
    if "cycle" in name or name.endswith(("clock", "time")):
        return True
    try:
        typ, config, _ = core.resolve(event)
    except Exception:
        return False
    return (typ == core._PERF_TYPE_HARDWARE and config in (0, 6, 7, 8, 9)) or (
        typ == core._PERF_TYPE_SOFTWARE and config in (0, 1, 9)
    )


def _aggregate_over_runs(values, events, axis=0, minimize=True):
    arr = np.asarray(values, dtype=float)
    events = list(events or ())
    with _ignore_runtime_warnings(), np.errstate(invalid="ignore"):
        if not minimize:
            return np.nanmean(arr, axis=axis)
        result = np.nanmin(arr, axis=axis)
        counts = [i for i, event in enumerate(events) if not _is_timing_event(event)]
        if not counts:
            return result
        means = np.nanmean(np.take(arr, counts, axis=-1), axis=axis)
    result = np.array(result, dtype=float, copy=True)
    if arr.ndim == 2:
        result[counts] = means
    else:
        result[:, counts] = means
    return result


def _throughput_sample_rows(values, samples, runs, events, minimize=True):
    events = list(events or ())
    n_events = len(events)
    try:
        n_samples = max(1, int(samples))
    except (TypeError, ValueError):
        n_samples = 1
    try:
        sample_runs = max(1, int(runs))
    except (TypeError, ValueError):
        sample_runs = 1
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.full((n_samples, n_events), np.nan, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, n_events)
    rows = int(arr.shape[0])
    if rows == n_samples * sample_runs:
        grouped = arr.reshape(n_samples, sample_runs, arr.shape[1])
        out = _aggregate_over_runs(grouped, events, axis=1, minimize=minimize)
        return np.asarray(out, dtype=float).reshape(n_samples, arr.shape[1])
    if rows == n_samples:
        return arr
    if sample_runs > 1 and rows % sample_runs == 0:
        groups = rows // sample_runs
        grouped = arr.reshape(groups, sample_runs, arr.shape[1])
        out = _aggregate_over_runs(grouped, events, axis=1, minimize=minimize)
        return np.asarray(out, dtype=float).reshape(groups, arr.shape[1])
    out = _aggregate_over_runs(arr, events, minimize=minimize)
    return np.asarray(out, dtype=float).reshape(1, arr.shape[1])


def _calibrated_overhead(result, mode=None):
    arr = np.asarray(result, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    with np.errstate(invalid="ignore"), _ignore_runtime_warnings():
        if mode == "throughput":
            return np.asarray(np.nanmin(arr, axis=0), dtype=float)
        return np.asarray(np.nanmedian(arr, axis=0), dtype=float)


def _fallback_all_nan_to_raw(diffs, raw_all, result, overhead, events, debug=False):
    raw_arr = np.asarray(raw_all)
    try:
        with _ignore_runtime_warnings():
            if raw_arr.ndim == 2:
                raw_min = np.nanmin(raw_arr, axis=0)
            elif raw_arr.ndim == 3:
                raw_min = np.nanmin(raw_arr, axis=(0, 1))
            else:
                return
    except Exception:
        return
    return _fallback_all_nan_to_raw_min(diffs, raw_min, result, overhead, events, debug)


def _fallback_all_nan_to_raw_min(diffs, raw_min, result, overhead, events, debug=False):
    diffs_arr = np.asarray(diffs)
    try:
        raw_min_arr = np.asarray(raw_min, dtype=float).reshape(-1)
    except Exception:
        return
    result_arr = np.asarray(result)
    names = list(events or [])
    n_events = len(names)
    if n_events == 0:
        try:
            n_events = int(result_arr.shape[-1])
        except Exception:
            return
    try:
        oh = np.asarray(overhead, dtype=float).reshape(-1)
    except Exception:
        oh = np.full((n_events,), 0.0)
    for e in range(n_events):
        try:
            oh_e = float(oh[e]) if e < oh.size else 0.0
        except Exception:
            oh_e = 0.0
        if oh_e != oh_e:
            oh_e = 0.0
        if e >= raw_min_arr.size:
            continue
        raw_min = raw_min_arr[e]
        if raw_min is None or not bool(np.isfinite(raw_min)):
            continue
        if diffs_arr.ndim == 2:
            if e >= diffs_arr.shape[1]:
                continue
            if e >= result_arr.shape[0]:
                continue
            if not bool(np.isnan(result_arr[e])):
                continue
            corrected = float(float(raw_min) - float(oh_e))
            if corrected != corrected:
                continue
            if corrected < 0.0:
                corrected = 0.0
            result_arr[e] = float(corrected)
            col = diffs_arr[:, e]
            if bool(np.all(np.isnan(col))) and len(col):
                diffs_arr[0, e] = float(corrected)
        elif diffs_arr.ndim == 3:
            if e >= diffs_arr.shape[2]:
                continue
            if result_arr.ndim == 2:
                if e >= result_arr.shape[1]:
                    continue
                if not bool(np.all(np.isnan(result_arr[:, e]))):
                    continue
            elif result_arr.ndim == 1:
                if result_arr.size != n_events:
                    continue
                if not bool(np.isnan(result_arr[e])):
                    continue
            else:
                continue
            corrected = float(float(raw_min) - float(oh_e))
            if corrected != corrected:
                continue
            if corrected < 0.0:
                corrected = 0.0
            if result_arr.ndim == 2:
                result_arr[:, e] = float(corrected)
            else:
                result_arr[e] = float(corrected)
            col = diffs_arr[:, :, e]
            if bool(np.all(np.isnan(col))):
                diffs_arr[0, 0, e] = float(corrected)
        else:
            continue
        name = names[e] if e < len(names) else e
        msg = (
            f"all {diffs_arr.shape[0]} run(s) for event {name!r} "
            f"were < overhead ({oh.tolist()}); "
            f"returning overhead-corrected minimum {float(corrected)} "
            f"(raw {float(raw_min)} minus overhead {float(oh_e)})"
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        if debug:
            _debug_log(msg)


def _throughput_fallback_by_sample(
    diffs, raw_all, overhead, events, sample_runs, debug=False
):
    diffs_arr = np.asarray(diffs)
    raw_arr = np.asarray(raw_all)
    if diffs_arr.ndim != 2 or raw_arr.ndim != 2:
        return
    try:
        sample_runs = max(1, int(sample_runs))
    except (TypeError, ValueError):
        return
    rows = int(diffs_arr.shape[0])
    if rows == 0 or rows % sample_runs:
        return
    n_events = min(diffs_arr.shape[1], raw_arr.shape[1])
    names = list(events or ())
    try:
        oh = np.asarray(overhead, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        oh = np.full((n_events,), 0.0)
    corrected = {}
    for start in range(0, rows, sample_runs):
        block = diffs_arr[start : start + sample_runs]
        raw_block = raw_arr[start : start + sample_runs]
        for e in range(n_events):
            if not bool(np.all(np.isnan(block[:, e]))):
                continue
            with _ignore_runtime_warnings():
                raw_min = float(np.nanmin(raw_block[:, e]))
            if not np.isfinite(raw_min):
                continue
            try:
                oh_e = float(oh[e]) if e < oh.size else 0.0
            except (TypeError, ValueError):
                oh_e = 0.0
            if oh_e != oh_e:
                oh_e = 0.0
            value = float(raw_min) - float(oh_e)
            if value != value:
                continue
            if value < 0.0:
                value = 0.0
            block[:, e] = value
            corrected.setdefault(e, []).append(value)
    for e, values in corrected.items():
        name = names[e] if e < len(names) else e
        msg = (
            f"{len(values)} throughput sample(s) for event {name!r} "
            f"were < overhead ({oh.tolist()}); returning overhead-corrected "
            f"minima (e.g. {float(values[-1])})"
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=4)
        if debug:
            _debug_log(msg)


def _normalize_groups(event):
    default = [[_DURATION_TIME]]
    if event is None:
        return default
    if isinstance(event, str):
        parts = _split_list(event)
        return [parts] if parts else default
    if isinstance(event, (list, tuple)):
        if len(event) == 0:
            return default
        if any(isinstance(g, (list, tuple)) for g in event):
            groups = []
            for group in event:
                if isinstance(group, (list, tuple)):
                    names = _split_list(group)
                    if names:
                        groups.append(names)
                else:
                    parts = _split_list(group)
                    if parts:
                        groups.append(parts)
            return groups or default
        names = _split_list(event)
        return [names] if names else default
    s = str(event).strip()
    return [[s]] if s else default


def _default_thread_cpus(entries, groups, taken=None):
    pending = []
    reserved = set(taken or ())
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("affinity") is None:
            pending.append(entry)
            continue
        try:
            reserved.update(_parse_affinity(entry["affinity"]) or ())
        except Exception:
            reserved.add(str(entry["affinity"]))
    if not pending:
        return
    groups = _normalize_groups(groups)
    try:
        groups = [core.with_leader(g) for g in (groups or [])]
    except Exception:
        pass
    groups = [g for g in (groups or []) if g]
    for entry in pending:
        try:
            numa = _parse_numa(entry.get("numa"))
        except Exception:
            numa = None
        try:
            allowed = core.allowed_cpus(groups, numa)
        except Exception:
            continue
        if not allowed:
            continue
        free = [cpu for cpu in allowed if cpu not in reserved]
        cpu = free[0] if free else allowed[0]
        entry["affinity"] = cpu
        reserved.add(cpu)


def _thread_alternatives(config):
    thread = (config or {}).get("thread", None)
    if thread is None:
        return []
    if isinstance(thread, dict):
        return [[thread]]
    if not isinstance(thread, (list, tuple)):
        return []
    out = []
    for item in thread:
        if isinstance(item, dict):
            out.append([item])
            continue
        if isinstance(item, (list, tuple)):
            out.append([entry for entry in item if isinstance(entry, dict)])
    return [group for group in out if group]


def _pin_default_affinity(config, event):
    if not isinstance(config, dict):
        return
    for group in _thread_alternatives(config):
        for entry in group:
            _parse_numa(entry.get("numa"))
    groups = _normalize_groups(event)
    try:
        groups = [core.with_leader(g) for g in groups]
    except Exception:
        pass
    groups = [g for g in groups if g]
    for group in _thread_alternatives(config):
        if all(isinstance(entry, dict) for entry in group):
            _default_thread_cpus(group, groups)


def _merge_slot_lists(slot_lists):
    combined_slots = {}
    combined_lists = []
    for slots in slot_lists or ():
        if not slots:
            continue
        for part in slots:
            if isinstance(part, dict):
                for column, values in part.items():
                    try:
                        if isinstance(values, (list, tuple)):
                            combined_slots.setdefault(column, []).extend(list(values))
                        else:
                            combined_slots.setdefault(column, []).append(values)
                    except Exception:
                        pass
            elif isinstance(part, list):
                combined_lists.extend(part)
    if combined_slots:
        return [combined_slots]
    if combined_lists:
        return [combined_lists]
    return []


def _slots_as_columns(slots):
    if isinstance(slots, dict):
        try:
            cols = {
                c: list(v)
                for c, v in slots.items()
                if isinstance(v, (list, tuple)) and v
            }
        except Exception:
            return None
        return cols or None
    if not slots:
        return None
    try:
        if not isinstance(slots[0], dict):
            return None
        keys = []
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            for column in slot:
                if column not in keys:
                    keys.append(column)
        cols = {
            column: [
                slot.get(column) if isinstance(slot, dict) else None for slot in slots
            ]
            for column in keys
        }
    except Exception:
        return None
    return cols or None


def _records(
    file_name,
    name,
    mode,
    result,
    group,
    slots=None,
    divisor=1,
    operations=1,
    threads=None,
):
    base = {
        "file": file_name,
        "name": name,
        "mode": mode,
        "operations": operations,
    }
    try:
        if threads is not None and int(threads) > 1:
            base["threads"] = int(threads)
    except (TypeError, ValueError):
        pass
    row_operations = None
    if isinstance(result, pd.DataFrame):
        try:
            candidate = result.attrs.get("_perf_operations")
            if candidate is not None:
                row_operations = [int(value) for value in candidate]
        except (TypeError, ValueError, AttributeError):
            row_operations = None
    try:
        div = float(divisor) if divisor not in (1, 1.0) else 1.0
    except Exception:
        div = 1.0
    slot_columns = _slots_as_columns(slots)
    if isinstance(result, pd.DataFrame):
        names = [name for name in group if name in result.columns]
        try:
            vals = result[names].to_numpy(dtype=float, copy=False)
        except Exception:
            vals = result.iloc[:, :0].to_numpy(dtype=float, copy=False)
        group = names
        if div != 1.0:
            try:
                vals = vals / div
                if isinstance(group, (list, tuple)) and vals.ndim == 2:
                    if vals.shape[1] == len(group):
                        for i, name in enumerate(group):
                            if _is_topdown_event(name):
                                vals[:, i] *= div
            except Exception:
                pass
        n = int(vals.shape[0])
        if row_operations is not None and len(row_operations) != n:
            row_operations = None
        ng = len(group)
        if slot_columns is not None:
            try:
                nslots = len(next(iter(slot_columns.values())))
            except Exception:
                nslots = 0
            if nslots != n:
                slot_columns = None
        records = []
        append = records.append
        for i in range(n):
            rec = dict(base)
            if row_operations is not None:
                rec["operations"] = row_operations[i]
            row = vals[i]
            for j in range(ng):
                try:
                    rec[group[j]] = float(row[j])
                except Exception:  # pragma: no cover
                    try:
                        rec[group[j]] = row[j]
                    except Exception:
                        pass
            if slot_columns is not None:
                try:
                    for column, values in slot_columns.items():
                        if column not in rec:
                            rec[column] = values[i]
                except Exception:
                    pass
            append(rec)
        return records
    try:
        arr = (
            result.to_numpy(dtype=float, copy=False)
            if hasattr(result, "to_numpy")
            else None
        )
    except Exception:
        arr = None
    if arr is not None:
        try:
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if div != 1.0:
                arr = arr / div
                if isinstance(group, (list, tuple)) and arr.ndim == 2:
                    if arr.shape[1] == len(group):
                        for i, name in enumerate(group):
                            if _is_topdown_event(name):
                                arr[:, i] *= div
            n = int(arr.shape[0])
            ng = len(group)
            if slot_columns is not None:
                try:
                    nslots = len(next(iter(slot_columns.values())))
                except Exception:
                    nslots = 0
                if nslots != n:
                    slot_columns = None
            records = []
            append = records.append
            for i in range(n):
                rec = dict(base)
                row = arr[i]
                for j in range(ng):
                    if j >= len(row):
                        continue
                    try:
                        rec[group[j]] = float(row[j])
                    except Exception:
                        pass
                if slot_columns is not None:
                    try:
                        for column, values in slot_columns.items():
                            if column not in rec:
                                rec[column] = values[i]
                    except Exception:
                        pass
                append(rec)
            return records
        except Exception:
            pass
    records = []
    append = records.append
    try:
        items = list(result)
    except Exception:
        return records
    if slot_columns is not None:
        try:
            n = len(items)
        except Exception:
            n = 0
        try:
            nslots = len(next(iter(slot_columns.values())))
        except Exception:
            nslots = 0
        if nslots != n:
            slot_columns = None
    for index, value in enumerate(items):
        rec = dict(base)
        try:
            import numpy as _np

            is_row = isinstance(value, (list, tuple)) or (
                isinstance(value, _np.ndarray) and value.ndim > 0
            )
        except Exception:
            is_row = isinstance(value, (list, tuple))
        if is_row:
            try:
                row = list(value)
            except Exception:
                row = None
            if row is not None:
                if len(row) == len(group):
                    for ev, item in zip(group, row):
                        try:
                            if div != 1.0 and not _is_topdown_event(ev):
                                rec[ev] = float(item) / div
                            else:
                                rec[ev] = float(item)
                        except Exception:
                            try:
                                rec[ev] = item
                            except Exception:
                                pass
                    if slot_columns is not None:
                        try:
                            for column, values in slot_columns.items():
                                if column not in rec:
                                    rec[column] = values[index]
                        except Exception:
                            pass
                    append(rec)
                    continue
                if 0 < len(row) < len(group):
                    for ev, item in zip(group[: len(row)], row):
                        try:
                            if div != 1.0 and not _is_topdown_event(ev):
                                rec[ev] = float(item) / div
                            else:
                                rec[ev] = float(item)
                        except Exception:
                            try:
                                rec[ev] = item
                            except Exception:
                                pass
                    if slot_columns is not None:
                        try:
                            for column, values in slot_columns.items():
                                if column not in rec:
                                    rec[column] = values[index]
                        except Exception:
                            pass
                    append(rec)
                    continue
            if len(group) == 1:
                try:
                    if div != 1.0 and not _is_topdown_event(group[0]):
                        v = float(value) / div
                    else:
                        v = float(value)
                except Exception:
                    try:
                        v = value
                    except Exception:
                        continue
                rec[group[0]] = v
                if slot_columns is not None:
                    try:
                        for column, values in slot_columns.items():
                            if column not in rec:
                                rec[column] = values[index]
                    except Exception:
                        pass
                append(rec)
                continue
            continue
        try:
            if div != 1.0 and not _is_topdown_event(group[0]):
                v = float(value) / div
            else:
                v = float(value)
        except Exception:
            try:
                v = value
            except Exception:
                continue
        rec[group[0]] = v
        if slot_columns is not None:
            try:
                for column, values in slot_columns.items():
                    if column not in rec:
                        rec[column] = values[index]
            except Exception:
                pass
        append(rec)
    return records


def _combine_results(records_lists):
    frames = [pd.DataFrame(r) for r in records_lists if r]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        try:
            return frames[0].reset_index(drop=True).set_index(list(_IDENTITY_COLUMNS))
        except Exception:  # pragma: no cover
            pass

    def _merge_group(group):
        merged = group[0].reset_index(drop=True)
        seen = set(merged.columns)
        parts = [merged]
        for other in group[1:]:
            other = other.reset_index(drop=True)
            if len(other) != len(merged):
                try:
                    n = max(len(merged), len(other))
                    if len(merged) < n:
                        merged = merged.reindex(range(n))
                        parts[0] = merged
                    if len(other) < n:
                        other = other.reindex(range(n))
                except Exception:
                    pass
            drop = [c for c in _COMBINE_META if c in other.columns and c in seen]
            new = [c for c in other.columns if c not in seen and c not in drop]
            if new:
                parts.append(other[new])
                seen.update(new)
        return pd.concat(parts, axis=1) if len(parts) > 1 else merged

    groups = {}
    for fr in frames:
        try:
            key = tuple(str(fr[c].iat[0]) for c in _IDENTITY_COLUMNS)
        except Exception:
            key = (None,) * len(_IDENTITY_COLUMNS)
        groups.setdefault(key, []).append(fr)
    merged = [_merge_group(g) for g in groups.values()]
    if len(merged) == 1:
        df = merged[0]
    else:
        try:
            df = pd.concat(merged, ignore_index=True)
        except Exception:
            df = merged[0]
    return df.set_index(list(_IDENTITY_COLUMNS))


def _topdown_percentages(rows, events, topdown):
    positions = {}
    for index, event in enumerate(events):
        positions.setdefault(core._event_base(event) or str(event), index)
    top = [positions.get(core._event_base(event) or str(event)) for event in topdown]
    slots = next(
        (i for i, event in enumerate(events) if core._event_base(event) == "slots"),
        None,
    )
    all_top = [i for i, event in enumerate(events) if _is_topdown_event(event)]
    out = []
    for row in () if rows is None else rows:
        try:
            if row is None or len(row) == 0:
                continue
        except TypeError:
            continue
        if slots is not None:
            total = row[slots] if slots < len(row) else None
        else:
            total = sum(
                (row[i] or 0) for i in all_top if i < len(row) and row[i] is not None
            )
        try:
            total = float(total)
        except (TypeError, ValueError):
            continue
        if total <= 0:
            continue
        values = []
        for i in top:
            try:
                value = float(row[i]) if i is not None and i < len(row) else 0.0
            except (TypeError, ValueError):
                value = 0.0
            values.append(100.0 * value / total)
        out.append(values)
    return out or None


def _bench(
    config,
    mode,
    code,
    setup,
    teardown,
    solutions,
    overhead=0,
    overhead_iterations=None,
    events=None,
    data=None,
    addr_range=None,
    arch=None,
    map_name=None,
    iterations_tracker=None,
    models_tracker=None,
    fallback_tracker=None,
    extra_addrs=None,
    code_addrs=None,
    argv_frame=None,
    probe_asm=None,
    debug=False,
    slots_out=None,
    overhead_code=None,
    simulated_ranges=None,
    stack_info=None,
    force_guard=False,
):
    if arch is None:
        arch = get_arch()
    samples = config.get("samples", 100)
    _backend_name = config.get("backend")
    if isinstance(_backend_name, dict):
        _backend_name = next(iter(_backend_name), "loop")
    _backend_name = _backend_name if isinstance(_backend_name, str) else "loop"
    runs = _resolve_runs(config, _backend_name)

    events = core.with_leader(_names(events))
    requested_events = list(events)
    topdown = _topdown_group(events)
    counter_events = list(events)
    if topdown:
        counter_events = [
            event for event in counter_events if not core._is_duration_event(event)
        ]
    if topdown and not any(
        core._event_base(event) == "slots" for event in counter_events
    ):
        counter_events.extend(
            event for event in core._TOPDOWN_EVENTS if event not in counter_events
        )
    measure_events = ["duration_time"] if topdown else list(counter_events)
    group = core.is_group(counter_events)
    try:
        force_typ = core.choose_type(counter_events)
    except Exception:
        force_typ = None
    try:
        _req = _parse_affinity(_affinity_spec(config))
    except Exception:
        _req = None
    _restore_aff = core.pin_pmu(config, force_typ, _req)

    counters = []
    indices = []
    try:
        counters, indices = core.open_counters(
            counter_events, force_typ, group, require_indices=not topdown
        )
        if topdown:
            events = measure_events
            indices = [None] * len(events)
            overhead = 0
            overhead_iterations = None
    except Exception:  # pragma: no cover
        _restore_aff()
        raise
    if not counters and arch.uses_rdpmc(code):
        _rdpmc = core.enable_rdpmc()
        if _rdpmc is not None:
            counters.append(_rdpmc)

    def close():
        for counter in counters:
            try:
                counter.disable()
            except Exception:
                pass
            try:
                counter.close()
            except Exception:
                pass

    try:
        return _bench_measure(
            config,
            mode,
            code,
            setup,
            teardown,
            solutions,
            overhead,
            events,
            data,
            arch,
            map_name,
            iterations_tracker,
            extra_addrs,
            code_addrs,
            debug,
            close,
            _restore_aff,
            _backend_name,
            counters,
            samples,
            runs,
            indices,
            addr_range,
            models_tracker,
            argv_frame,
            probe_asm,
            fallback_tracker,
            slots_out,
            overhead_iterations=overhead_iterations,
            overhead_code=overhead_code,
            simulated_ranges=simulated_ranges,
            topdown=topdown,
            counter_events=counter_events,
            requested_events=requested_events,
            stack_info=stack_info,
            force_guard=force_guard,
        )
    finally:
        close()
        _restore_aff()


def _bench_measure(
    config,
    mode,
    code,
    setup,
    teardown,
    solutions,
    overhead,
    events,
    data,
    arch,
    map_name,
    iterations_tracker,
    extra_addrs,
    code_addrs,
    debug,
    close,
    _restore_aff,
    _backend_name,
    counters,
    samples,
    runs,
    indices,
    addr_range,
    models_tracker=None,
    argv_frame=None,
    probe_asm=None,
    fallback_tracker=None,
    slots_out=None,
    overhead_iterations=None,
    overhead_code=None,
    simulated_ranges=None,
    topdown=None,
    counter_events=None,
    requested_events=None,
    stack_info=None,
    force_guard=False,
):
    if simulated_ranges is None:
        with _STATE_LOCK:
            simulated_ranges = tuple(_SIMULATED_RANGES)
    else:
        simulated_ranges = tuple(simulated_ranges)
    rng = _make_rng(config)
    try:
        _heap_size, heap_mode = _heap_settings(config)
    except Exception:
        heap_mode = None
    branch_cfg = _branch_config(config)
    predictable = (
        config.get("branch", "unpredictable") == "predictable"
        if branch_cfg is None
        else branch_cfg["prediction"] == "predictable"
    )
    levels = arch.data_cache_levels(config)
    mem_levels = arch.resolve_mem_cache(config)
    mem_levels = _apply_code_tier_defaults(
        arch, config, mem_levels, extra_addrs, code_addrs
    )
    tlb_levels = {}
    for resolver in ("data_tlb_levels", "instruction_tlb_levels"):
        fn = getattr(arch, resolver, None)
        if fn is None:
            continue
        try:
            resolved = fn(config)
        except Exception:
            resolved = None
        if resolved:
            tlb_levels.update({k: v for k, v in resolved.items() if v is not None})

    models = []
    if data:
        _map_data_pages(data)
    if debug:
        _debug_json("measurement config", config)
        _debug_log(
            f"measuring {map_name or code[:40]!r} mode={mode} events={list(events)}"
        )
    if solutions:
        models = _models_cached(solutions, branch_cfg, arch)
        if debug:
            try:
                _debug_json("synthesised data (models)", models)
            except Exception:
                pass
        if models:
            try:
                stack_base = int(getattr(arch, "_STACK_ADDR", 0x7FFF00000000))
            except (TypeError, ValueError):
                stack_base = 0x7FFF00000000
            min_addr = _min_mmap_addr()

            def _keep_addr(a):
                try:
                    a = int(a)
                except (TypeError, ValueError):
                    return False
                if a < min_addr:
                    return False
                if a >= stack_base:
                    return False
                for lo, hi in simulated_ranges:
                    if lo <= a < hi:
                        return False
                if addr_range is not None:
                    try:
                        if addr_range[0] <= a < addr_range[1]:
                            return False
                    except (TypeError, ValueError):
                        pass
                return True

            for m in models:
                m["reads"] = [(a, ln, v) for a, ln, v in m["reads"] if _keep_addr(a)]
                m["writes"] = [(a, ln, v) for a, ln, v in m["writes"] if _keep_addr(a)]
            models = _runtable_models(models, arch, probe_asm, setup, teardown, config)
            if models_tracker is not None and models:
                try:
                    models_tracker.append(models)
                except Exception:
                    pass
        addrs = {}
        if models:
            addrs = {str(a): 0 for m in models for a, _, _ in m["reads"] + m["writes"]}
        steered = _static_extra_addrs(extra_addrs)
        resolved_levels = _resolve_reg_levels(
            mem_levels,
            models,
            (data or {}).get("regs") or {},
            (data or {}).get("mem") or {},
        )
        for level in (mem_levels, resolved_levels):
            steered |= _mem_level_int_addrs(level) - _code_level_addrs(level)
        for a in steered:
            addrs.setdefault(str(a), 0)
        if addrs:
            _map_data_pages({"mem": addrs}, strict=False)
        if models:
            _drop_unmappable_models(models)
        _map_scratch_reg_pages(models)

    if data and _missing_arg_regs(data, arch) and _explicit_bound_values(data):
        orig_setup = setup
        probe_regs = _sequence_probe_regs(data)

        def _probe(augmented, probe_models=None):
            if probe_models is None:
                probe_models = models
            probe_data = (
                _merge_data(augmented, {"regs": probe_regs})
                if probe_regs
                else augmented
            )
            probe_setup = arch.data_setup_asm(probe_data)
            if probe_setup:
                probe_setup = probe_setup + ("\n" + orig_setup if orig_setup else "\n")
            else:
                probe_setup = orig_setup
            full_asm, _meta, per_iter_buf = _build_loop_asm(
                1,
                mode,
                code,
                probe_setup,
                teardown,
                probe_models,
                ["duration_time"],
                [None],
                levels,
                predictable,
                random.Random(0),
                arch,
                data=probe_data,
                branch_cfg=branch_cfg,
                mem_levels=mem_levels,
                extra_addrs=extra_addrs,
                tlb_levels=tlb_levels,
                argv_frame=argv_frame,
                heap_mode=heap_mode,
            )
            code_bytes = arch.assemble(full_asm)
            page = _jit_page(code_bytes, in_use=True)
            try:
                inputs = np.zeros((2, 1), dtype=np.uint64)
                outputs = np.zeros((1, 1), dtype=np.uint64)
                data_ptr = (
                    per_iter_buf.ctypes.data if per_iter_buf is not None else None
                )
                args = _Args(1, inputs.ctypes.data, outputs.ctypes.data, data_ptr)
                fn = ctypes.CFUNCTYPE(None, ctypes.POINTER(_Args))(
                    ctypes.addressof(ctypes.c_char.from_buffer(page))
                )
                _check_target_runnable(fn, args, map_name, key=None, config=config)
            finally:
                _jit_release(page)

        _needs_growth = False
        if _needs_pointer_fallback(data, arch):
            _needs_growth = True
            if debug:
                _debug_log("explicit data bounds: supplying fallback arguments")
        else:
            try:
                _probe(data, models)
            except ValueError as ex:  # pragma: no cover
                if not _is_harness_fault(ex):
                    raise
                _needs_growth = True
                if debug:
                    _debug_log(f"fallback probe with current data faulted: {ex}")

        if _needs_growth:
            try:
                grown = _grow_fallback_data(data, arch, _probe, debug)
            except Exception as ex:  # pragma: no cover
                if debug:
                    _debug_log(f"fallback growth failed: {ex!r}")
                raise
            if grown is not data:
                data = grown
                if fallback_tracker is not None:
                    try:
                        fallback_tracker.append(grown)
                    except Exception:
                        pass
                if debug:
                    try:
                        _debug_json("fallback data grown as needed", data)
                    except Exception:
                        pass

    if data:
        _map_data_pages(data)
        data_setup = arch.data_setup_asm(data)
        setup = data_setup + ("\n" + setup if setup else "\n")

    try:
        _pinned_iterations, _lo, _hi = _iterations_config(config)
        target_rel_se = None
        if _pinned_iterations is not None:
            iterations = int(_pinned_iterations)
        else:
            target_rel_se, min_iterations, max_iterations = _probe_plan(
                config, _backend_name
            )
            probe_runs = _resolve_probe_runs(config, _backend_name, runs)
            if mode == "throughput":
                probe_runs = max(probe_runs, min(4, runs))
            indices = core.refresh_indices(counters, events, indices)
            probe = _collect(
                config,
                min_iterations,
                probe_runs,
                mode,
                code,
                setup,
                teardown,
                models,
                events,
                indices,
                overhead,
                levels,
                predictable,
                rng,
                arch,
                data=data,
                map_name=map_name,
                branch_cfg=branch_cfg,
                mem_levels=mem_levels,
                extra_addrs=extra_addrs,
                tlb_levels=tlb_levels,
                argv_frame=argv_frame,
                debug=debug,
                overhead_iterations=overhead_iterations,
                stack_info=stack_info,
                force_guard=force_guard,
            )
            if probe.ndim > 1:
                timing_index = next(
                    (
                        index
                        for index, event in enumerate(events)
                        if _is_timing_event(event)
                    ),
                    0,
                )
                probe_series = probe[:, timing_index]
            else:
                probe_series = probe
            iterations = _plan_iterations(
                probe_series, target_rel_se, min_iterations, max_iterations
            )
        if iterations_tracker is not None:
            try:
                iterations_tracker.append(int(iterations))
            except Exception:
                pass

        measure_runs = _measure_runs(mode, samples, runs)
        if (
            overhead_code
            and mode == "throughput"
            and int(iterations) > 0
            and (
                overhead_iterations is None
                or int(overhead_iterations) != int(iterations)
            )
        ):
            indices = core.refresh_indices(counters, events, indices)
            overhead_series = _collect(
                config,
                iterations,
                measure_runs,
                mode,
                overhead_code,
                setup,
                teardown,
                models,
                events,
                indices,
                0,
                levels,
                predictable,
                rng,
                arch,
                data=data,
                map_name=map_name,
                branch_cfg=branch_cfg,
                mem_levels=mem_levels,
                extra_addrs=extra_addrs,
                tlb_levels=tlb_levels,
                argv_frame=argv_frame,
                debug=debug,
                sample_runs=runs if mode == "throughput" else None,
                force_guard=force_guard,
            )
            arr = np.asarray(overhead_series, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            with _ignore_runtime_warnings():
                calibrated = np.nanmin(arr, axis=0)
            overhead = np.asarray(calibrated, dtype=float).reshape(len(events))
            overhead_iterations = iterations
            if debug:
                _debug_log(
                    f"calibrated overhead at iterations={iterations}: "
                    f"{dict(zip(events, overhead.tolist()))}"
                )
        indices = core.refresh_indices(counters, events, indices)
        topdown_raw = []
        data_series = _collect(
            config,
            iterations,
            measure_runs,
            mode,
            code,
            setup,
            teardown,
            models,
            events,
            indices,
            overhead,
            levels,
            predictable,
            rng,
            arch,
            data=data,
            map_name=map_name,
            branch_cfg=branch_cfg,
            mem_levels=mem_levels,
            extra_addrs=extra_addrs,
            tlb_levels=tlb_levels,
            argv_frame=argv_frame,
            debug=debug,
            slots_out=slots_out,
            overhead_iterations=overhead_iterations,
            raw_counters=counters if topdown else None,
            raw_out=topdown_raw if topdown else None,
            stack_info=stack_info,
            sample_runs=runs if mode == "throughput" else None,
            force_guard=force_guard,
        )
    finally:
        close()
        _restore_aff()

    randomized = (
        isinstance(branch_cfg, dict) and branch_cfg.get("prediction") == "unpredictable"
    )
    if topdown:
        percentages = _topdown_percentages(
            topdown_raw, counter_events or events, topdown
        )
        if percentages is not None:
            if mode == "throughput":
                percentages = _throughput_sample_rows(
                    percentages,
                    samples,
                    runs,
                    topdown,
                    minimize=not randomized,
                )
            frame = pd.DataFrame(percentages, columns=topdown)
            if any(core._is_duration_event(e) for e in requested_events or ()):
                values = []
                if data_series is not None:
                    try:
                        series = data_series
                        if mode == "throughput":
                            series = _throughput_sample_rows(
                                series,
                                samples,
                                runs,
                                ["duration_time"],
                                minimize=not randomized,
                            )
                        values = list(np.asarray(series).reshape(-1))
                    except Exception:
                        values = []
                if len(values) == len(frame):
                    frame["duration_time"] = values
            return frame, iterations

    if mode != "throughput":
        df = pd.DataFrame(data_series, columns=events)
        if df.empty:
            df = pd.DataFrame(
                {ev: [float("nan")] * int(samples) for ev in events},
            )
        return df, iterations
    mins = _throughput_sample_rows(
        data_series, samples, runs, events, minimize=not randomized
    )
    df = pd.DataFrame(mins, columns=list(events))
    return df, iterations


def _stable_hash8(payload):
    blob = dumps_json(_json_canonical(payload), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


def _id_hash(data, config, binary=None):
    has_data = bool(data and (data.get("regs") or data.get("mem")))
    has_config = bool(config)
    has_binary = binary is not None and bool(binary)
    if not has_data and not has_config and not has_binary:
        return None
    payload = {
        "data": data if has_data else {"regs": {}, "mem": {}},
        "config": config or {},
    }
    if has_binary:
        payload["binary"] = binary
    return _stable_hash8(payload)


def _unpack_bench(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, None


def _frame_operations(frame, mode, iterations):
    if mode != "throughput" or iterations is None:
        return []
    try:
        count = int(len(frame))
        value = int(iterations)
    except (TypeError, ValueError):
        return []
    return [value] * count


def _scale_row_operations(result, scale):
    if scale == 1 or not isinstance(result, pd.DataFrame):
        return result
    try:
        operations = result.attrs.get("_perf_operations")
    except AttributeError:
        return result
    if operations is None:
        return result
    try:
        result.attrs["_perf_operations"] = [
            int(value) * int(scale) for value in operations
        ]
    except (TypeError, ValueError):
        pass
    return result


def _pin_iterations(config, iterations):
    if config is None or iterations is None:
        return None
    try:
        count = int(iterations)
    except (TypeError, ValueError):
        return None
    if count < 1:
        return None
    config["iterations"] = count
    return count


def _result_low_signal(result, group, threshold=2.0):
    if not isinstance(result, pd.DataFrame):
        return False
    names = [
        name
        for name in (group or ())
        if name in result.columns and _is_timing_event(name)
    ]
    if not names:
        return False
    if any(_is_topdown_event(name) for name in names):
        return False
    try:
        values = result[names].to_numpy(dtype=float, copy=False)
    except Exception:
        return False
    with np.errstate(invalid="ignore"), _ignore_runtime_warnings():
        finite = values[np.isfinite(values)]
        if not finite.size:
            return False
        return float(np.nanmedian(finite)) <= float(threshold)


def _differential_safe(arch, code):
    fn = getattr(arch, "differential_safe", None)
    if fn is None:
        return False
    try:
        return bool(fn(code))
    except Exception:
        return False


def _measure_group(
    config,
    mode,
    setup,
    teardown,
    solutions,
    group,
    arch,
    map_name,
    tracker,
    data,
    overhead_code,
    main_code,
    extra=None,
    debug=False,
    models_tracker=None,
    fallback_tracker=None,
    slots_out=None,
    simulated_ranges=None,
):
    try:
        from .cfg import _thread_group as _tgroup

        _entries = _tgroup(config)
    except Exception:
        _entries = [{}]
    if len(_entries) <= 1:
        return _measure_group_single(
            config,
            mode,
            setup,
            teardown,
            solutions,
            group,
            arch,
            map_name,
            tracker,
            data,
            overhead_code,
            main_code,
            extra=extra,
            debug=debug,
            models_tracker=models_tracker,
            fallback_tracker=fallback_tracker,
            slots_out=slots_out,
            simulated_ranges=simulated_ranges,
        )
    import copy as _copy
    import threading as _threading

    import pandas as _pd

    n = len(_entries)
    try:
        _priorities = {
            str((e or {}).get("priority", "normal")).strip().lower() for e in _entries
        }
    except Exception:
        _priorities = set()
    if len(_priorities) > 1:
        out_frames = []
        out_operations = []
        entry_slot_lists = []
        out_it = None
        for entry in _entries:
            cfg = _copy.deepcopy(config)
            cfg["thread"] = [dict(entry)]
            entry_slots = []
            res, it = _measure_group_single(
                cfg,
                mode,
                setup,
                teardown,
                solutions,
                group,
                arch,
                map_name,
                tracker,
                _copy.deepcopy(data),
                overhead_code,
                main_code,
                extra=_copy.deepcopy(extra),
                debug=debug,
                models_tracker=models_tracker,
                fallback_tracker=fallback_tracker,
                slots_out=entry_slots,
                simulated_ranges=simulated_ranges,
            )
            entry_slot_lists.append(entry_slots)
            frame = res if isinstance(res, _pd.DataFrame) else _pd.DataFrame(res)
            out_frames.append(frame)
            out_operations.extend(_frame_operations(frame, mode, it))
            try:
                out_it = it if out_it is None else max(int(out_it), int(it))
            except Exception:
                out_it = it if out_it is None else out_it
        combined = (
            _pd.concat(out_frames, ignore_index=True) if out_frames else _pd.DataFrame()
        )
        if out_operations and len(out_operations) == len(combined):
            combined.attrs["_perf_operations"] = out_operations
        if slots_out is not None:
            slots_out.extend(_merge_slot_lists(entry_slot_lists))
        try:
            if tracker is not None and out_it is not None:
                tracker.append(int(out_it))
        except (TypeError, ValueError):
            pass
        return combined, out_it
    try:
        _barrier = _threading.Barrier(n)
    except Exception:
        _barrier = None
    _results: list = [None] * n
    _iters: list = [None] * n
    _errors: list = [None] * n
    _local_trackers: list = [[] for _ in range(n)]
    _local_models: list = [[] for _ in range(n)]
    _local_fallback: list = [[] for _ in range(n)]
    _local_slots: list = [[] for _ in range(n)]

    def _run(idx, entry):
        try:
            cfg = _copy.deepcopy(config)
            cfg["thread"] = [dict(entry)]
            if _barrier is not None:
                try:
                    _barrier.wait(timeout=_MEASURE_BARRIER_TIMEOUT)
                except _threading.BrokenBarrierError:
                    _errors[idx] = RuntimeError(
                        "measurement barrier was broken before all workers arrived"
                    )
                    return
                except Exception as ex:
                    _errors[idx] = ex
                    return
            res, it = _measure_group_single(
                cfg,
                mode,
                setup,
                teardown,
                solutions,
                group,
                arch,
                map_name,
                _local_trackers[idx],
                _copy.deepcopy(data),
                overhead_code,
                main_code,
                extra=_copy.deepcopy(extra),
                debug=debug,
                models_tracker=_local_models[idx],
                fallback_tracker=_local_fallback[idx],
                slots_out=_local_slots[idx],
                simulated_ranges=simulated_ranges,
            )
            _results[idx] = res
            _iters[idx] = it
        except BaseException as ex:
            _errors[idx] = ex
            if _barrier is not None:
                try:
                    _barrier.abort()
                except Exception:
                    pass

    _threads = [
        _threading.Thread(target=_run, args=(i, e), daemon=True)
        for i, e in enumerate(_entries)
    ]
    try:
        for t in _threads:
            t.start()
    except BaseException:  # pragma: no cover
        if _barrier is not None:
            try:
                _barrier.abort()
            except Exception:
                pass
        for t in _threads:
            if t.is_alive():
                t.join(timeout=1)
        _measure_threads_forget(_threads)
        for ex in _errors:
            if ex is not None:
                raise ex
        raise
    deadline = time.monotonic() + _MEASURE_JOIN_TIMEOUT
    for t in _threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
    if any(t.is_alive() for t in _threads):
        if _barrier is not None:
            try:
                _barrier.abort()
            except Exception:
                pass
        _measure_threads_forget(_threads)
        error = next((ex for ex in _errors if ex is not None), None)
        timeout = TimeoutError(
            f"measurement threads did not finish within "
            f"{_MEASURE_JOIN_TIMEOUT}s; a thread may be blocked on a "
            "process-wide resource"
        )
        if error is not None:
            raise timeout from error
        raise timeout
    _measure_threads_forget(_threads)
    for ex in _errors:
        if ex is not None:
            raise ex
    frames = []
    row_operations = []
    for idx, r in enumerate(_results):
        if r is None:
            continue
        frame = r if isinstance(r, _pd.DataFrame) else _pd.DataFrame(r)
        frames.append(frame)
        row_operations.extend(_frame_operations(frame, mode, _iters[idx]))
    combined = _pd.concat(frames, ignore_index=True) if frames else _pd.DataFrame()
    if row_operations and len(row_operations) == len(combined):
        combined.attrs["_perf_operations"] = row_operations
    try:
        it = int(max(int(x) for x in _iters if x is not None))
    except Exception:
        it = next((x for x in _iters if x is not None), None)
    try:
        if tracker is not None and it is not None:
            tracker.append(int(it))
    except Exception:
        pass
    try:
        if models_tracker is not None:
            for lst in _local_models:
                models_tracker.extend(lst)
    except Exception:
        pass
    try:
        if fallback_tracker is not None:
            for lst in _local_fallback:
                fallback_tracker.extend(lst)
    except Exception:
        pass
    try:
        if slots_out is not None:
            slots_out.extend(_merge_slot_lists(_local_slots))
    except Exception:
        pass
    return combined, it


def _measure_group_single(
    config,
    mode,
    setup,
    teardown,
    solutions,
    group,
    arch,
    map_name,
    tracker,
    data,
    overhead_code,
    main_code,
    extra=None,
    debug=False,
    models_tracker=None,
    fallback_tracker=None,
    slots_out=None,
    simulated_ranges=None,
):
    extra = extra or {}
    overhead_extra = {key: value for key, value in extra.items() if key != "stack_info"}
    _measure_threads_enter()
    try:
        overhead_tracker = []
        overhead_res, overhead_iterations = _unpack_bench(
            _bench(
                config=config,
                mode=mode,
                code=overhead_code,
                setup=setup,
                teardown=teardown,
                solutions=solutions,
                events=group,
                data=data,
                arch=arch,
                map_name=map_name,
                iterations_tracker=overhead_tracker,
                models_tracker=models_tracker,
                fallback_tracker=fallback_tracker,
                debug=debug,
                simulated_ranges=simulated_ranges,
                **overhead_extra,
            )
        )
        overhead = _calibrated_overhead(overhead_res, mode=mode)
        result, iterations = _unpack_bench(
            _bench(
                config=config,
                mode=mode,
                setup=setup,
                code=main_code,
                teardown=teardown,
                solutions=solutions,
                overhead=overhead,
                overhead_iterations=overhead_iterations,
                overhead_code=overhead_code,
                events=group,
                data=data,
                arch=arch,
                map_name=map_name,
                iterations_tracker=tracker,
                models_tracker=models_tracker,
                fallback_tracker=fallback_tracker,
                debug=debug,
                slots_out=slots_out,
                simulated_ranges=simulated_ranges,
                **extra,
            )
        )
    finally:
        _measure_threads_exit()
    return result, iterations


def _bench_one(**kwargs):
    debug = bool(kwargs.get("debug", False))
    mode = _check_mode(kwargs.get("mode"))
    file = kwargs.get("file")
    src_file = file
    if file:
        file = resolve_exec(file)
        if not Path(file).exists():
            raise ValueError(f"file {file!r} does not exist")
    code = kwargs.get("code")
    target = _normalize_target(kwargs.get("target"))
    target_name = target
    name = kwargs.get("name") or target or code
    setup = _setup_names(kwargs.get("setup"))
    teardown = _setup_names(kwargs.get("teardown"))
    argv = _argv_words(file, kwargs.get("argv"))
    env = _env_words(kwargs.get("env"))
    event = kwargs.get("event", _DURATION_TIME)
    backend_arg = kwargs.get("backend")
    simulated_ranges = kwargs.get("simulated_ranges")

    config = kwargs.get("config")
    if config is None:
        config = {}
    if isinstance(config, dict):
        config = copy.deepcopy(config)
        if "data" in config:
            raise ValueError(
                "config must not contain 'data'; pass data separately via data={...}"
            )
        validate_spec(config)
    data = _merge_data(None, kwargs.get("data"))

    if config.get("seed") is None:
        config["seed"] = 0
    elif _is_random_seed(config.get("seed")):
        config["seed"] = _generate_seed()
    from .cfg import _thread_group as _tgroup

    entries = _tgroup(config)
    holder, _created = _thread_store(config)
    if _created:
        entries = [holder]
    _thread_count = max(1, len(entries))
    for ent in entries:
        if ent.get("priority") is None:
            ent["priority"] = _current_priority_value()
    _iterations_tracker = []
    _models_tracker = []
    _fallback_tracker = []
    _iterations_pinned = _iterations_config(config)[0]
    for ent in entries:
        _parse_affinity(ent.get("affinity"))
        _parse_priority(ent.get("priority"))
        try:
            _parse_numa(ent.get("numa"))
        except Exception:
            pass
    for key in _LIST_TOPS:
        config[key] = _container(config, key)
    _stack_size, _stack_align = config["stack"]["size"], config["stack"]["align"]
    _stack_mode = config["stack"]["mode"]
    _resolved_align = config["func"]["align"]
    _code_align = config["code"]["align"]
    _all_solutions = []
    _apply_seed(config)
    if debug:
        _debug_json("effective config", config)
        _debug_json("input data", data)
        _debug_log(
            f"stack: size=0x{_stack_size:x} align=0x{_stack_align:x} mode={_stack_mode}"
        )

    groups = _normalize_groups(event)
    requested_groups = [list(g) for g in groups]
    try:
        groups = [core.with_leader(g) for g in groups]
    except Exception:
        pass
    measure_groups = [g for g in groups if g]
    _default_thread_cpus(entries, measure_groups)
    _group_iterations = None

    def _pin_group_iterations(iterations):
        nonlocal _group_iterations
        if _group_iterations is None:
            _group_iterations = _pin_iterations(config, iterations)
        elif _group_iterations is not None:
            _pin_iterations(config, _group_iterations)
        return _group_iterations

    results = []
    cpu_info = _cpu_info_dict()

    def _emit_records(file_label, rec_name, result, group, slots_cap, **kw):
        try:
            slots = slots_cap[-1] if slots_cap else None
        except Exception:
            slots = None
        return _records(file_label, rec_name, mode, result, group, slots, **kw)

    if code:
        if argv:
            raise ValueError(
                "argv needs a binary symbol: `perf benchmark a.out:main -- x y`, "
                "not an asm snippet"
            )
        if not name:
            name = code
        arch = get_arch()
        code = _loop_asm(arch.normalize_asm(f"{code};"))
        setup = arch.normalize_asm("\n".join(setup) + "\n") if setup else ""
        teardown = arch.normalize_asm("\n".join(teardown) + "\n") if teardown else ""
        _asm_extra = {
            "force_guard": _needs_guard(arch, code, setup, teardown)
            or _contains_call("\n".join((code, setup, teardown)))
        }
        backend = _resolve_backend(backend_arg, config, "loop")
        unroll_n = _resolve_unroll_n(None, config, "unroll")
        if backend == "unroll" and mode == "throughput":
            raise ValueError("backend 'unroll' can only be used with mode 'latency'")
        _select_backend(config, backend)
        data = _apply_heap_data(data, config, debug)
        asm_solutions = _explore_asm_cached(
            code,
            setup or None,
            data,
            arch,
            _stack_size,
            _stack_align,
            _stack_mode,
            _config_seed_int(config),
        )
        _all_solutions = asm_solutions or []
        data = _merge_code_pages(data, asm_solutions)
        if debug:
            _debug_log(f"found {len(list(asm_solutions or []))} solution(s) (asm)")
            _debug_json("solutions", _solution_summary(asm_solutions))
            try:
                _debug_json("synthesised data", explored_data_dict(asm_solutions))
            except Exception:
                pass
        binary_id = {"asm": code, "name": name}
        for group in measure_groups:
            _pin_group_iterations(None)
            if backend == "unroll":
                _slots: list = []
                result, _iterations = _measure_group(
                    config,
                    mode,
                    setup,
                    teardown,
                    asm_solutions,
                    group,
                    arch,
                    name,
                    _iterations_tracker,
                    data,
                    _with_align(_unroll_code(code, unroll_n), _code_align),
                    _with_align(_unroll_code(code, 2 * unroll_n), _code_align),
                    extra=_asm_extra,
                    debug=debug,
                    models_tracker=_models_tracker,
                    fallback_tracker=_fallback_tracker,
                    slots_out=_slots,
                    simulated_ranges=simulated_ranges,
                )
                _pin_group_iterations(_iterations)
                operations = 1 if mode == "latency" else _iterations
                recs = _emit_records(
                    None,
                    name,
                    result,
                    group,
                    _slots,
                    divisor=unroll_n,
                    operations=operations,
                    threads=_thread_count,
                )
            else:
                _slots = []
                _record_divisor = 1
                _throughput_scale = (
                    _resolve_loop_count(config) if mode == "throughput" else 1
                )
                _main_code = _unroll_code(code, _throughput_scale)
                result, _iterations = _measure_group(
                    config,
                    mode,
                    setup,
                    teardown,
                    asm_solutions,
                    group,
                    arch,
                    name,
                    _iterations_tracker,
                    data,
                    _align_only(_code_align),
                    _with_align(_main_code, _code_align),
                    extra=_asm_extra,
                    debug=debug,
                    models_tracker=_models_tracker,
                    fallback_tracker=_fallback_tracker,
                    slots_out=_slots,
                    simulated_ranges=simulated_ranges,
                )
                _scale_row_operations(result, _throughput_scale)
                if (
                    mode == "latency"
                    and _result_low_signal(result, group)
                    and _differential_safe(arch, code)
                ):
                    if debug:
                        _debug_log(
                            f"{name!r}: loop latency is below timer resolution; "
                            f"using the {unroll_n}-copy differential"
                        )
                    result, _iterations = _measure_group(
                        config,
                        mode,
                        setup,
                        teardown,
                        asm_solutions,
                        group,
                        arch,
                        name,
                        _iterations_tracker,
                        data,
                        _with_align(_unroll_code(code, unroll_n), _code_align),
                        _with_align(_unroll_code(code, 2 * unroll_n), _code_align),
                        extra=_asm_extra,
                        debug=debug,
                        models_tracker=_models_tracker,
                        fallback_tracker=_fallback_tracker,
                        slots_out=_slots,
                        simulated_ranges=simulated_ranges,
                    )
                    _record_divisor = unroll_n
                _pin_group_iterations(_iterations)
                operations = (
                    1
                    if mode != "throughput"
                    else (
                        None
                        if _iterations is None
                        else int(_iterations) * _throughput_scale
                    )
                )
                recs = _emit_records(
                    None,
                    name,
                    result,
                    group,
                    _slots,
                    divisor=_record_divisor,
                    operations=operations,
                    threads=_thread_count,
                )
            results.append(recs)
    else:
        if not target:
            raise ValueError(
                "a symbol or region is required to analyze a binary (--file)"
            )

        project, obj, funcs, prototypes = _bench_project(
            file, config, config["stack"], config["func"]
        )
        arch = get_arch(project)

        try:
            obj.write_perf_map()
        except Exception:
            pass

        argv_frame = None
        if argv:
            try:
                argv_frame = obj.setup_argv(argv, env or [])
            except (ValueError, OSError) as ex:
                raise ValueError(str(ex)) from ex
            if debug:
                _debug_log(
                    f"argc={argv_frame['argc']} argv={argv} env={env or []} "
                    f"sp={argv_frame['sp']:#x}"
                )

        setup_names = _setup_names(setup)
        setup = _call_seq_asm(obj, arch, setup_names)
        teardown = _call_seq_asm(obj, arch, _setup_names(teardown))

        ang_setup = ""
        if setup_names:
            addrs = [_angr_symbol_addr(project, obj, name) for name in setup_names]
            if all(addr is not None for addr in addrs):
                ang_setup = "\n".join(arch.call_seq_asm(addr) for addr in addrs)
            else:
                ang_setup = setup

        base = os.path.basename(str(src_file))
        sha = _content_hash8(src_file)
        binary_id = {"path": base, "sha": sha}
        file_label = f"{base}@{sha}" if sha else base
        try:
            from .info import hostname as _hostname_fn

            _host = _hostname_fn()
        except Exception:
            _host = None
        info_dict = {"cpu": cpu_info, "binary": binary_id, "hostname": _host}
        simulated_ranges = _set_simulated_ranges(
            project.loader, project.loader.main_object
        )
        targets = list(resolve_targets(project, target, funcs))
        if not targets:
            raise _unresolved(project, src_file, target)
        if len(targets) > 1:
            raise _unresolved(project, src_file, target, ambiguous=True)
        for target_label, start, end in targets:
            rec_name = name
            if int(end) <= int(start):
                raise ValueError(f"empty region {target_label!r}: end <= start")
            data = _entry_data(target_label, src_file, data, argv, env)
            data = _apply_heap_data(data, config, debug)
            proto = _func_prototype(prototypes.get(target_label), data)
            try:
                solutions = _explore_target_cached(
                    project,
                    _project_cache_id(file, config),
                    target_label,
                    start,
                    end,
                    ang_setup,
                    funcs,
                    data,
                    proto,
                    _stack_size,
                    _stack_align,
                    _stack_mode,
                    _config_seed_int(config),
                )
            except Exception as ex:  # pragma: no cover
                _warn_exploration(target_label, ex)
                solutions = []

            _all_solutions = solutions or []
            data = _merge_code_pages(data, solutions)
            if debug:
                _debug_log(
                    f"found {len(list(solutions or []))} solution(s) "
                    f"for target {target_label}"
                )
                _debug_json("solutions", _solution_summary(solutions))
                try:
                    _debug_json("synthesised data", explored_data_dict(solutions))
                except Exception:
                    pass
            angr_base = project.loader.main_object.mapped_base
            addr_offset = obj.runtime_base - angr_base
            bin_segs = [
                (s.vaddr, s.vaddr + s.memsize)
                for s in project.loader.main_object.segments
            ]
            addr_range = (
                (min(s[0] for s in bin_segs), max(s[1] for s in bin_segs))
                if bin_segs
                else None
            )

            symbol = None
            try:
                symbol = obj.get_symbol(target_label)
            except ValueError:
                pass
            target = symbol if symbol is not None else start + addr_offset
            host_names = []
            for label, (name_start, name_end) in (funcs or {}).items():
                try:
                    if int(name_start) == int(start) and int(name_end) == int(end):
                        host_names.append(label)
                except (TypeError, ValueError):
                    continue
            host_target = _host_target(
                config, file, target_label, target_name, *host_names
            )
            if host_target is not None:
                target = host_target
                if debug:
                    _debug_log(
                        f"running the host-loaded copy of {target_label} "
                        f"at 0x{target:x}"
                    )
            probe_asm = arch.call_seq_asm(target)
            code_asm = arch.call_seq_asm(target, fence=(mode == "latency"))
            code_asm = _with_align(code_asm, _resolved_align)

            backend = _resolve_backend(backend_arg, config, "loop")
            unroll_n = _resolve_unroll_n(None, config, backend)
            if backend == "unroll" and mode == "throughput":
                raise ValueError(
                    "backend 'unroll' can only be used with mode 'latency'"
                )
            _select_backend(config, backend)
            try:
                _static_addrs = _static_table_addrs(
                    project, start, end, obj, solutions=solutions
                )
                _reserved = _reserved_addresses(obj)
                _static_addrs = [a for a in _static_addrs if a not in _reserved]
            except Exception:
                _static_addrs = []
            try:
                _code_size = max(int(end) - int(start), 1)
                _code_page = mmap.PAGESIZE
                _code_addrs = list(
                    range(
                        int(target) & ~(_code_page - 1),
                        (int(target) + _code_size + _code_page - 1) & ~(_code_page - 1),
                        _code_page,
                    )
                )
            except (TypeError, ValueError):
                _code_addrs = []
            stack_info = None
            if mode == "latency":
                try:
                    stack_info = {
                        "base": int(obj.stack_base),
                        "top": (
                            int(argv_frame["sp"])
                            if argv_frame
                            else int(obj.stack_top) - 8
                        ),
                        "size": int(obj.stack_top) - int(obj.stack_base),
                        "mode": _stack_mode,
                        "frame": (
                            None
                            if argv_frame
                            else {
                                "save": int(obj.stack_base),
                                "sp": int(obj.stack_top) - 8,
                            }
                        ),
                    }
                except (AttributeError, TypeError, ValueError):
                    stack_info = None
            extra = {
                "addr_range": addr_range,
                "extra_addrs": list(_static_addrs or []),
                "code_addrs": _code_addrs,
                "argv_frame": argv_frame,
                "probe_asm": probe_asm,
                "stack_info": stack_info,
                "force_guard": True,
            }
            _patch_addr = None
            _saved_byte = None
            _poke_locked = False
            if ".." in str(target_label):
                try:
                    _candidate = int(end) + int(addr_offset)
                    if _poke_addr_safe(obj, _candidate):
                        _POKE_LOCK.acquire()
                        _poke_locked = True
                        _patch_addr = _candidate
                        _saved_byte = ctypes.c_ubyte.from_address(_patch_addr).value
                        ctypes.c_ubyte.from_address(_patch_addr).value = 0xC3
                except Exception:  # pragma: no cover
                    if _poke_locked:
                        try:
                            _POKE_LOCK.release()
                        except Exception:
                            pass
                        _poke_locked = False
                    _patch_addr = None
                    _saved_byte = None
            try:
                with _call_stub_code(arch, fence=(mode == "latency")) as _stub_call:
                    _stub_call = _with_align(_stub_call, _resolved_align)
                    for group in measure_groups:
                        _pin_group_iterations(None)
                        if backend == "unroll":
                            code_n = "\n".join([code_asm] * unroll_n)
                            stub_n = "\n".join([_stub_call] * unroll_n)
                            _slots = []
                            result, _iterations = _measure_group(
                                config,
                                mode,
                                setup,
                                teardown,
                                solutions,
                                group,
                                arch,
                                target_label,
                                _iterations_tracker,
                                data,
                                stub_n,
                                code_n,
                                extra=extra,
                                debug=debug,
                                models_tracker=_models_tracker,
                                fallback_tracker=_fallback_tracker,
                                slots_out=_slots,
                                simulated_ranges=simulated_ranges,
                            )
                            _pin_group_iterations(_iterations)
                            operations = 1 if mode == "latency" else _iterations
                            recs = _emit_records(
                                file_label,
                                rec_name,
                                result,
                                group,
                                _slots,
                                divisor=unroll_n,
                                operations=operations,
                                threads=_thread_count,
                            )
                            results.append(recs)
                            continue
                        _slots = []
                        _throughput_scale = (
                            _resolve_loop_count(config) if mode == "throughput" else 1
                        )
                        _code = (
                            "\n".join([code_asm] * _throughput_scale)
                            if _throughput_scale > 1
                            else code_asm
                        )
                        _stub = (
                            "\n".join([_stub_call] * _throughput_scale)
                            if _throughput_scale > 1
                            else _stub_call
                        )
                        result, _iterations = _measure_group(
                            config,
                            mode,
                            setup,
                            teardown,
                            solutions,
                            group,
                            arch,
                            target_label,
                            _iterations_tracker,
                            data,
                            _stub,
                            _code,
                            extra=extra,
                            debug=debug,
                            models_tracker=_models_tracker,
                            fallback_tracker=_fallback_tracker,
                            slots_out=_slots,
                            simulated_ranges=simulated_ranges,
                        )
                        _scale_row_operations(result, _throughput_scale)
                        _pin_group_iterations(_iterations)
                        operations = (
                            1
                            if mode != "throughput"
                            else (
                                None
                                if _iterations is None
                                else int(_iterations) * _throughput_scale
                            )
                        )
                        recs = _emit_records(
                            file_label,
                            rec_name,
                            result,
                            group,
                            _slots,
                            operations=operations,
                            threads=_thread_count,
                        )
                        results.append(recs)
            finally:
                if _patch_addr is not None:
                    try:
                        ctypes.c_ubyte.from_address(_patch_addr).value = _saved_byte
                    except Exception:
                        pass
                    _patch_addr = None
                if _poke_locked:
                    try:
                        _POKE_LOCK.release()
                    except Exception:
                        pass

    df = _combine_results(results)
    if df.empty:
        target_label = name or code or kwargs.get("target")
        raise ValueError(f"no measurements collected for {target_label!r}")

    if df.columns.size:
        try:
            _freq = float(cpu_info.get("freq"))
        except (TypeError, ValueError):
            _freq = 0.0
        if _freq and _freq > 0:
            for _column in list(df.columns):
                try:
                    _is_duration = core._is_duration_event(str(_column))
                except Exception:
                    _is_duration = False
                if _is_duration:
                    df[_column] = df[_column] * 1e9 / _freq

    flat_events = []
    for group in requested_groups:
        for event in group or ():
            if event not in flat_events:
                flat_events.append(event)
    for ev in flat_events:
        if ev not in df.columns:
            df[ev] = np.nan

    _resolved_iterations = _iterations_pinned
    if _resolved_iterations is None and _iterations_tracker:
        try:
            _resolved_iterations = max(_iterations_tracker)
        except Exception:
            pass
    if _resolved_iterations is None:
        _pinned_now, lo, _hi = _iterations_config(config)
        _resolved_iterations = _pinned_now if _pinned_now is not None else lo
    if _resolved_iterations is None:
        _resolved_iterations = _DEFAULT_ITER_MIN
    try:
        config["iterations"] = int(_resolved_iterations)
    except (TypeError, ValueError):
        pass
    if not _backend_resolved(config, backend):
        _select_backend(config, backend)

    if _fallback_tracker:
        try:
            data = _merge_fallback_data(data, _fallback_tracker)
        except Exception:
            pass

    try:
        col_data = {
            "regs": dict((data or {}).get("regs") or {}),
            "mem": dict((data or {}).get("mem") or {}),
            "heap": dict((data or {}).get("heap") or {}),
        }
        try:
            explored = explored_data_dict(
                _all_solutions, _models_tracker[-1] if _models_tracker else None
            )
            try:
                explicit_canon = set()
                for _k in col_data["regs"]:
                    try:
                        explicit_canon.add(_canonical_data_reg_key(_k))
                    except Exception:
                        explicit_canon.add(str(_k).strip().lower())
            except Exception:
                explicit_canon = set(col_data["regs"])
            for _reg, _val in (explored.get("regs") or {}).items():
                try:
                    _canon = _canonical_data_reg_key(_reg)
                except Exception:
                    _canon = str(_reg).strip().lower()
                if _canon in explicit_canon or _reg in col_data["regs"]:
                    continue
                col_data["regs"][_canon] = _val
                explicit_canon.add(_canon)
            for _addr, _val in (explored.get("mem") or {}).items():
                col_data["mem"].setdefault(_addr, _val)
        except Exception:
            pass
        df = add_param_columns(df, col_data)
    except Exception:  # pragma: no cover
        try:
            df = add_param_columns(df, data)
        except Exception:
            pass

    try:
        df = _filter_bench_columns(df)
    except Exception:
        pass
    if code:
        df.attrs["config"] = config
        try:
            from .info import hostname as _hostname_fn

            _host = _hostname_fn()
        except Exception:
            _host = None
        df.attrs["info"] = {"cpu": cpu_info, "binary": binary_id, "hostname": _host}
        df.attrs["file"] = None
    else:
        df.attrs["config"] = config
        df.attrs["info"] = info_dict
        df.attrs["file"] = file_label
    df.attrs["id"] = _id_hash(data, config, binary_id)
    df.attrs["data"] = data
    if argv:
        df.attrs["argv"] = list(argv)
    if env:
        df.attrs["env"] = list(env)
    if debug:
        try:
            _debug_log(f"results ({len(df)} rows):\n{df.to_string()}")
        except Exception:
            pass
    return df


@functools.lru_cache(maxsize=1)
def _cpu_info_dict():
    return dict(cpuinfo(list(_CPUINFO_FIELDS)).iloc[0].to_dict())


def _needs_guard(arch, *chunks):
    reserved = {
        arch._canonical_reg(reg)
        for reg in getattr(arch, "_HARNESS_LOOP_REGS", ("r8", "r9", "r10"))
    }
    used = set()
    for chunk in chunks:
        if not chunk:
            continue
        try:
            used |= set(arch.used_regs(chunk))
        except Exception:
            continue
    return bool(used & reserved)


def _align_directive(align):
    try:
        return get_arch().align_directive(align)
    except Exception:
        return ""


def _with_align(code, align):
    directive = _align_directive(align)
    if not directive or not code:
        return code
    return f"{directive}\n{code}"


def _align_only(align):
    directive = _align_directive(align)
    return f"{directive}\n" if directive else ""


def _unroll_code(code, count):
    text = str(code or "")
    count = max(int(count), 1)
    if count == 1:
        return text
    labels = set(_LABEL_DEF.findall(text))
    if not labels:
        if text.endswith((";", "\n")):
            return text * count
        return "\n".join([text] * count)
    copies = []
    for index in range(count):
        copy = text
        for label in sorted(labels, key=len, reverse=True):
            replacement = f"{label}__unroll{index}"
            copy = re.sub(
                rf"(?<![A-Za-z0-9_.$]){re.escape(label)}(?![A-Za-z0-9_.$])",
                replacement,
                copy,
            )
        copies.append(copy)
    return "\n".join(copies)


@functools.lru_cache(maxsize=256)
def _takes_data_ptr(fn):
    import inspect

    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):  # pragma: no cover
        return None
    return (
        len(params) >= 2
        and params[1].name in ("data_ptr", "_data_ptr")
        and params[1].kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


def _arch_asm(name, meta, data_ptr=None, arch=None, **kw):
    if arch is None:
        arch = get_arch()
    fn = getattr(arch, name)
    takes_data_ptr = _takes_data_ptr(fn)
    if takes_data_ptr is None:
        return fn(meta, data_ptr, **kw) if kw else fn(meta, data_ptr)
    if takes_data_ptr:
        return fn(meta, data_ptr, **kw) if kw else fn(meta, data_ptr)
    return fn(meta, **kw)


def _iterations_config(config):
    spec = (config or {}).get("iterations")
    pinned = None
    lo = hi = None
    if isinstance(spec, dict):
        lo = spec.get("min")
        hi = spec.get("max")
    elif spec is not None:
        pinned = spec
    return pinned, lo, hi


def _backend_options(config, name):
    spec = (config or {}).get("backend")
    if not isinstance(spec, dict):
        return {}
    opts = spec.get(name)
    return opts if isinstance(opts, dict) else {}


def _backend_option(config, name, key, default=None):
    opts = _backend_options(config, name)
    value = opts.get(key)
    if value is not None:
        return value
    return default


def _probe_plan(config, backend="loop"):
    target = _backend_option(config, backend, "target", 0.01)
    _pinned, lo, hi = _iterations_config(config)
    lo = _DEFAULT_ITER_MIN if lo is None else lo
    hi = _DEFAULT_ITER_MAX if hi is None else hi
    return float(target), int(lo), int(hi)


def _plan_iterations(measured, target_rel_se, min_iterations, max_iterations):
    arr = np.asarray(measured, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return int(min_iterations)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if mean <= 0 or median <= 0 or std <= 0 or not np.isfinite(mean):
        return int(min_iterations)
    n = (std / (target_rel_se * mean)) ** 2
    return int(np.clip(np.ceil(n), min_iterations, max_iterations))


def _timed_regs_avoid(code, setup, teardown, data, arch):
    avoid = set()
    used = arch.used_regs
    for chunk in (code, setup, teardown):
        if chunk:
            try:
                avoid |= set(used(chunk))
            except Exception:
                pass
    if data:
        for r in data.get("regs") or {}:
            try:
                avoid |= set(used(str(arch.canonical_data_reg(r))))
            except Exception:
                pass
            try:
                avoid.add(str(r).strip().lower())
                avoid.add(str(arch.canonical_data_reg(r)).strip().lower())
            except Exception:
                pass
        try:
            avoid = {arch._canonical_reg(r) for r in avoid}
        except Exception:
            pass
    return avoid


def _write_jit_map(addr, size, name):
    try:
        label = (
            str(name or "loop").strip().replace("\n", "_").replace(" ", "_") or "loop"
        )
        write_perf_map([(int(addr), int(size), f"perf_bench_loop:{label}")])
    except Exception:
        pass


def _join_asm(*parts):
    return "\n".join(str(p).strip("\n") for p in parts if str(p or "").strip())


def _build_loop_asm(
    iterations,
    mode,
    code,
    setup,
    teardown,
    models,
    events,
    indices,
    levels,
    predictable,
    rng,
    arch,
    data=None,
    branch_cfg=None,
    mem_levels=None,
    extra_addrs=None,
    tlb_levels=None,
    argv_frame=None,
    heap_mode=None,
    stack_info=None,
    force_guard=False,
):
    events = _names(events)
    data_script = ""
    data2_script = ""
    data_iter = ""
    throughput_restore = ""
    per_iter_buf = None
    meta = None
    has_explicit_mem = bool(data and data.get("mem"))
    has_explicit_dist = bool(
        data
        and any(
            isinstance(v, (list, tuple))
            for v in list((data.get("regs") or {}).values())
            + list((data.get("mem") or {}).values())
        )
    )
    _extra_set = _static_extra_addrs(extra_addrs)
    if models or has_explicit_mem or has_explicit_dist or _extra_set or mem_levels:
        buf, meta = _per_iter_data(
            models,
            iterations,
            levels,
            predictable,
            rng,
            arch,
            explicit=data,
            branch_cfg=branch_cfg,
            mem_levels=mem_levels,
            extra_addrs=_extra_set,
            tlb_levels=tlb_levels,
            heap_mode=heap_mode,
        )
        per_iter_buf = buf
        if meta["reg_col"] or meta["mem_addrs"] or meta.get("l1i_addrs"):
            _fill_evict_tables(buf, meta, buf.ctypes.data)
            prime = _arch_asm("prime_asm", meta, buf.ctypes.data, arch)
            reload_regs = {}
            for reg, val in ((data or {}).get("regs") or {}).items():
                try:
                    canon = _canonical_data_reg_key(reg)
                except Exception:
                    canon = str(reg).strip().lower()
                if canon not in (meta.get("reg_col") or {}):
                    reload_regs[reg] = val
            reload_data = {"regs": reload_regs} if reload_regs else None
            explicit_reload = arch.data_reload_asm(reload_data) if reload_data else ""
            if mode == "throughput":
                _hoist_values = bool(
                    meta.get("values_static") and not meta.get("target_writes")
                )
                data_script = _arch_asm(
                    "steer_asm",
                    meta,
                    buf.ctypes.data,
                    arch,
                    mem_levels=meta.get("mem_levels"),
                    write_values=_hoist_values,
                    tlb_toggle=False,
                )
                if _hoist_values:
                    data_iter = _join_asm(prime, explicit_reload)
                else:
                    data_iter = _join_asm(
                        _arch_asm("per_iter_asm", meta, buf.ctypes.data, arch),
                        explicit_reload,
                    )
                data2_script = explicit_reload
                throughput_restore = explicit_reload
            else:
                steer = _arch_asm(
                    "steer_asm",
                    meta,
                    buf.ctypes.data,
                    arch,
                    mem_levels=meta.get("mem_levels"),
                )
                data_script = arch.preserve_steer_asm(steer) if steer else ""
                data2_script = _join_asm(prime, explicit_reload)
        else:
            if data:
                data2_script = arch.data_reload_asm(data)
                if mode == "throughput":
                    throughput_restore = data2_script
                    data_iter = _join_asm(data_iter, throughput_restore)
    elif data:
        data2_script = arch.data_reload_asm(data)
        if mode == "throughput":
            throughput_restore = data2_script
            data_iter = _join_asm(data_iter, throughput_restore)

    argv_enter, argv_leave, argv_load = arch.argv_asm(argv_frame)
    if argv_load:
        data2_script = _join_asm(data2_script, argv_load)
        data_iter = _join_asm(data_iter, argv_load)
    if argv_frame:
        setup = _join_asm(setup, argv_enter)
        teardown = _join_asm(argv_leave, teardown)
    stack_refresh = ""
    if stack_info:
        try:
            _stack_mode = str(stack_info.get("mode") or "").strip().lower()
        except Exception:
            _stack_mode = ""
        if _stack_mode in ("random", "zero"):
            stack_refresh = arch.stack_refresh_asm(
                stack_info.get("base"),
                stack_info.get("size"),
                src=stack_info.get("src"),
                iterations=iterations,
                zero=_stack_mode == "zero",
            )
        _frame = stack_info.get("frame")
        if _frame and not argv_frame:
            _enter, _leave, _ = arch.argv_asm(_frame)
            setup = _join_asm(setup, _enter)
            teardown = _join_asm(_leave, teardown)
    tlb_restore = _arch_asm("tlb_restore_asm", meta, arch=arch)
    avoid = _timed_regs_avoid(code, None, None, data, arch)
    avoid.add(arch._PRIME_SCRATCH_REG)
    if meta is not None:
        for _rn in meta.get("reg_col", ()):
            avoid.add(str(_rn).strip().lower())
            try:
                avoid.add(arch._canonical_reg(_rn))
            except Exception:
                pass
    body = " ".join(c for c in (code, setup, teardown, data2_script, data_iter) if c)
    may_call = _contains_call(body)
    harness_regs = {
        arch._canonical_reg(reg)
        for reg in getattr(arch, "_HARNESS_LOOP_REGS", ("r8", "r9", "r10"))
    }
    needs_guard = bool(force_guard or may_call or (avoid & harness_regs))
    t0, t1 = arch.timing(events, indices, callee_saved=needs_guard, avoid=avoid)
    if needs_guard:
        guard_pre, guard_post = arch.timed_guard(events, callee_saved=True, avoid=avoid)
    else:
        guard_pre = guard_post = ""
    if guard_pre or guard_post:
        code = "\n".join(p for p in (guard_pre, code, guard_post) if p)
    full_asm = arch._BENCH[mode].format(
        code=code,
        setup=setup,
        teardown=teardown,
        data=data_script,
        data2=data2_script,
        data_iter=data_iter,
        stack_refresh=stack_refresh,
        t0=t0,
        t1=t1,
        tlb=tlb_restore,
    )
    return full_asm, (meta if per_iter_buf is not None else None), per_iter_buf


def _asm_error(code, ex):
    msg = str(ex)
    try:
        hint = get_arch().asm_error_hint(code, msg)
    except Exception:
        hint = None
    if hint is not None:
        raise ValueError(hint) from ex
    raise ex


def _run_measured(
    fn,
    args,
    map_name=None,
    index=None,
    config=None,
    budget=None,
    raw_counters=None,
    raw_out=None,
    managed_output=False,
):
    check_memory = index is None or int(index) % _MEMORY_CHECK_EVERY == 0
    before = _run_memory() if check_memory else 0
    if raw_counters:
        for counter in raw_counters:
            counter.enable()
    if managed_output:
        fn(ctypes.byref(args))
    else:
        with _fake_output(config):
            fn(ctypes.byref(args))
    if raw_out is not None:
        raw_out.append(_read_raw_counters(raw_counters))
    retire_exit_hooks()
    if not check_memory:
        return
    calls = max(1, int(args.iterations))
    after = _run_memory()
    _memory_record(calls, (after - before) / calls if after > before else 0.0)
    _check_growth(calls, map_name, index, used=after, budget=budget)


@contextlib.contextmanager
def _call_stub_code(arch, fence=False, copies=1):
    try:
        copies = max(1, int(copies))
    except (TypeError, ValueError):
        copies = 1
    page = _jit_page(arch.return_stub_bytes(), in_use=True)
    try:
        target = ctypes.addressof(ctypes.c_char.from_buffer(page))
        call = arch.call_seq_asm(target, fence=fence)
        yield "\n".join([call] * copies)
    finally:
        _jit_release(page)


def _store_model_bytes(addr, length, value):
    try:
        length = int(length)
    except (TypeError, ValueError):
        return
    if length <= 0 or length > 8:
        return
    try:
        payload = (int(value) & ((1 << (8 * length)) - 1)).to_bytes(length, "little")
    except (TypeError, ValueError, OverflowError):
        return
    try:
        ctypes.memmove(ctypes.c_void_p(int(addr)), payload, length)
    except (OSError, TypeError, ValueError):
        return


def _model_world(model):
    world = {
        "regs": dict(model.get("regs") or {}),
        "reads": list(model.get("reads") or []),
        "writes": list(model.get("writes") or []),
    }
    for key in ("blocks", "branch_deps"):
        if key in model:
            world[key] = model[key]
    addrs = {str(a): 0 for a, _, _ in world["reads"] + world["writes"]}
    if addrs:
        _map_data_pages({"mem": addrs}, strict=False)
    _map_scratch_reg_pages([world])
    _drop_unmappable_models([world])
    return world


def _model_reg_asm(arch, regs):
    try:
        return arch.model_reg_asm(regs)
    except Exception:
        return []


def _model_probe_asm(arch, model, body):
    regs = model.get("regs")
    try:
        return arch.model_probe_asm(regs, body)
    except Exception:
        return None


def _model_returns(arch, probe_asm, setup_asm, teardown_asm, model, config=None):
    body = "\n".join(part for part in (setup_asm, probe_asm, teardown_asm) if part)
    encoding = _model_probe_asm(arch, model, body)
    if not encoding:
        return True
    page = _jit_page(encoding, in_use=True)
    try:
        fn = ctypes.CFUNCTYPE(None)(ctypes.addressof(ctypes.c_char.from_buffer(page)))
        writes = list(model.get("reads") or []) + list(model.get("writes") or [])
        return _run_once(fn, writes, config)
    finally:
        _jit_release(page)


def _run_once(fn, writes, config=None):
    with _OUTPUT_LOCK, _FORK_LOCK:
        return _run_once_locked(fn, writes, config)


def _run_once_locked(fn, writes, config=None):
    sys.stdout.flush()
    sys.stderr.flush()
    with _fake_output(config):
        read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
        try:
            pid = os.fork()
        except Exception:  # pragma: no cover
            os.close(read_fd)
            os.close(write_fd)
            raise
        if pid == 0:
            _reset_fatal_handlers()
            try:
                os.close(read_fd)
            except OSError:
                os._exit(1)
            for addr, length, value in writes:
                _store_model_bytes(addr, length, value)
            try:
                fn()
                os.write(write_fd, b"1")
            except BaseException:
                pass
            os._exit(0)
        os.close(write_fd)
        try:
            ready, _w, _x = select.select([read_fd], [], [], _MODEL_PROBE_TIMEOUT)
            returned = bool(ready) and os.read(read_fd, 1) == b"1"
        except OSError:
            returned = False
        finally:
            os.close(read_fd)
            if not returned:
                _kill(pid)
        try:
            _waitpid(pid)
        except OSError:
            pass
    return returned


def _model_probe_key(models, probe_asm, setup_asm, teardown_asm):
    worlds = tuple(
        (
            tuple(sorted((str(k), repr(v)) for k, v in (m.get("regs") or {}).items())),
            tuple(tuple(t) for t in (m.get("reads") or ())),
            tuple(tuple(t) for t in (m.get("writes") or ())),
        )
        for m in models
    )
    return (worlds, str(probe_asm), str(setup_asm or ""), str(teardown_asm or ""))


@_locked
def _runtable_models(
    models, arch, probe_asm, setup_asm="", teardown_asm="", config=None
):
    if not probe_asm or not models:
        return models
    key = _model_probe_key(models, probe_asm, setup_asm, teardown_asm)
    cached = _RUNTABLE_MODELS.get(key)
    if cached is not None:
        return copy.deepcopy(cached)
    keep = [
        world
        for world in (_model_world(m) for m in models)
        if _model_returns(arch, probe_asm, setup_asm, teardown_asm, world, config)
    ]
    if not keep:
        for m in models:
            world = _model_world({"regs": m.get("regs") or {}})
            if _model_returns(arch, probe_asm, setup_asm, teardown_asm, world, config):
                keep = [world]
                break
    if not keep:
        keep = [_model_world(m) for m in models]
    if len(_RUNTABLE_MODELS) >= _SOLVE_CACHE_MAX:
        _RUNTABLE_MODELS.clear()
    _RUNTABLE_MODELS[key] = copy.deepcopy(keep)
    return keep


def _prepare_stack_info(info, iterations, mode, rng, arch):
    if not info or mode != "latency":
        return None
    try:
        stack_mode = str(info.get("mode") or "").strip().lower()
    except Exception:
        return None
    out = dict(info)
    if stack_mode not in ("random", "zero", "fixed"):
        return None
    frame = out.get("frame")
    top = out.get("sp")
    if top is None and isinstance(frame, dict):
        top = frame.get("sp")
    if top is None:
        top = out.get("top")
    base = out.get("base")
    size = out.get("size")
    try:
        sp = int(top) - 8
        stack_base = int(base)
        stack_size = int(size)
        window = min(stack_size, _STACK_REFRESH_WINDOW)
        if iterations > 0:
            window = min(window, max(8, _STACK_REFRESH_BYTES_MAX // int(iterations)))
        window_base = max(stack_base + 16, sp - window)
        window = sp - window_base
    except (TypeError, ValueError):
        return None
    if window <= 0:
        return None
    out["base"] = window_base
    out["size"] = window & ~7
    out["iterations"] = int(iterations)
    if stack_mode == "random":
        try:
            payload = rng.randbytes(int(iterations) * int(out["size"]))
            buffer = (
                np.frombuffer(payload, dtype=np.uint8)
                .reshape(int(iterations), int(out["size"]))
                .copy()
            )
        except Exception:
            buffer = np.asarray(
                [
                    [rng.getrandbits(8) for _ in range(int(out["size"]))]
                    for _ in range(int(iterations))
                ],
                dtype=np.uint8,
            )
        out["_buffer"] = buffer
        out["src"] = int(buffer.ctypes.data)
    return out


def _mode_backends(backend):
    if not isinstance(backend, dict):
        return {}
    out = {}
    for mode, name in backend.items():
        if name is None:
            continue
        mode = str(mode).strip()
        if mode not in _MODES:
            raise ValueError(
                f"unknown mode {mode!r} in backend map; "
                f"expected one of {', '.join(_MODES)}"
            )
        if name not in _BACKENDS:
            raise ValueError(
                f"unknown backend {name!r} for mode {mode!r}; "
                f"expected one of {', '.join(_BACKENDS)}"
            )
        out[mode] = name
    return out


def _contains_call(code):
    try:
        return get_arch().contains_call(code)
    except Exception:
        return False


def _loop_asm(code):
    try:
        return get_arch().loop_asm(code)
    except Exception:
        return str(code or "")


def _resolve_backend(backend, config, default):
    if backend is None:
        backend = (config or {}).get("backend", default)
        if isinstance(backend, dict):
            backend = default
    if backend is None:
        backend = default
    name = str(backend).strip()
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {', '.join(_BACKENDS)}"
        )
    return name


def _resolve_unroll_n(unroll_n, config, backend="unroll"):
    default_n = _default_unroll_n()
    return _resolve_backend_count(unroll_n, config, backend, default_n)


def _resolve_backend_count(value, config, backend, default):
    if value is None:
        value = _backend_option(config, backend, "count", default)
    if value is None:
        value = default
    try:
        value = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{backend} count must be an integer, got {value!r}") from e
    if value < 1:
        raise ValueError(f"{backend} count must be >= 1, got {value!r}")
    return value


def _resolve_loop_count(config):
    default = _DEFAULT_BENCH.get("backend", {}).get("loop", {}).get("count", 16)
    return _resolve_backend_count(None, config, "loop", default)


def _resolve_runs(config, backend):
    runs = _backend_option(config, backend, "runs", 10)
    try:
        runs = int(runs)
    except (TypeError, ValueError) as e:
        raise ValueError(f"backend runs must be an integer, got {runs!r}") from e
    if runs < 1:
        raise ValueError(f"backend runs must be >= 1, got {runs!r}")
    return runs


def _resolve_probe_runs(config, backend, runs):
    probes = _backend_option(config, backend, "probes", 3)
    try:
        probes = int(probes)
    except (TypeError, ValueError) as e:
        raise ValueError(f"backend probes must be an integer, got {probes!r}") from e
    return max(1, min(runs, probes))


def _measure_runs(mode, samples, runs):
    try:
        samples = max(1, int(samples))
    except (TypeError, ValueError):
        samples = 1
    try:
        runs = max(1, int(runs))
    except (TypeError, ValueError):
        runs = 1
    if mode == "throughput":
        return samples * runs
    return runs


def _used_backend_options(config, backend):
    runs = _resolve_runs(config, backend)
    target, _lo, _hi = _probe_plan(config, backend)
    used = {
        "probes": _resolve_probe_runs(config, backend, runs),
        "runs": runs,
        "target": target,
    }
    if backend == "unroll":
        used["count"] = _resolve_unroll_n(None, config, backend)
    elif backend == "loop":
        used["count"] = _resolve_loop_count(config)
    return {k: used[k] for k in sorted(used)}


def _backend_resolved(config, backend):
    spec = (config or {}).get("backend")
    return isinstance(spec, dict) and set(spec) == {backend}


def _select_backend(config, backend):
    used = _used_backend_options(config, backend)
    config["backend"] = {backend: used}
    return used


def _is_topdown_event(event):
    return isinstance(event, str) and core._event_base(event).startswith("topdown-")


def _topdown_group(events):
    names = [
        e for e in events or () if isinstance(e, str) and not core._is_duration_event(e)
    ]
    topdown = [e for e in names if _is_topdown_event(e)]
    if not topdown:
        return []
    others = [e for e in names if e not in topdown]
    if any(core._event_base(e) != "slots" for e in others):
        return []
    return topdown
