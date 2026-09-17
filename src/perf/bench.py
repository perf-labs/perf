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
import ctypes
import ctypes.util
import hashlib
import json
import os
import random
import sys
import warnings
from pathlib import Path

import angr
import capstone
import claripy
import numpy as np
import pandas as pd

from . import core
from .arch import arch as get_arch
from .arch import load as load_arch
from .core import (
    _affinity_guard,
    _affinity_spec,
    _current_affinity_list,
    _current_priority_value,
    _func_prototype,
    _is_mem_addr_key,
    _parse_addr_key,
    _parse_affinity,
    _parse_priority,
    _priority_guard,
    _priority_spec,
    _split_list,
    _thread_cfg,
    _to_int_or,
    _to_u64,
)
from .exec import Elf, resolve_exec
from .info import CPUINFO_FIELDS, cpuinfo, functions
from .info import targets as resolve_targets

_DEFAULT_BENCH = {
    "seed": None,
    "thread": {
        "affinity": None,
        "priority": None,
    },
    "branch": "unpredictable",
    "cache": {
        "L1d": {"hit_rate": 100},
        "L1i": {"hit_rate": 100},
        "L2": {"hit_rate": 100},
        "L3": {"hit_rate": 100},
    },
    "func": {
        "align": 16,
        "order": "as-is",
    },
    "code": {
        "align": 16,
    },
    "stack": {
        "size": 0x200000,
        "align": 16,
    },
    "samples": 100,
    "runs": 10,
    "probe_runs": 3,
    "iterations": None,
    "min_iterations": 128,
    "max_iterations": 5_000_000,
    "target_rel_se": 0.005,
    "backend": None,
    "unroll_n": 5,
}
_BACKENDS = ("loop", "unroll")
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_MAP_PRIVATE = 0x02
_MAP_ANONYMOUS = 0x20
_MAP_FIXED = 0x10
_PROT_READ = 0x1
_PROT_WRITE = 0x2
_PROT_EXEC = 0x4
_JIT_PAGES = []
_ASM_SCRATCH_BASE = get_arch().ASM_SCRATCH_BASE
_HEXDIGITS = set("0123456789abcdefABCDEF")
_VALID_TOP_KEYS = frozenset(_DEFAULT_BENCH.keys())
_VALID_THREAD_KEYS = frozenset({"affinity", "priority"})
_VALID_FUNC_KEYS = frozenset({"align", "order"})
_VALID_CODE_KEYS = frozenset({"align"})
_VALID_STACK_KEYS = frozenset({"size", "align"})


def explore(
    proj,
    start,
    end,
    setup_asm=None,
    funcs=None,
    data=None,
    prototype=None,
    stack_size=None,
    stack_align=16,
):
    if funcs is None:
        funcs = dict(functions(proj))
    is_func = start in {s for s, e in funcs.values()}

    if is_func:
        state = proj.factory.call_state(addr=start, prototype=prototype)
    else:
        state = proj.factory.blank_state(addr=start)
        load_arch(proj).map_stack(state, size=stack_size, align=stack_align)

    if setup_asm:
        setup_asm = setup_asm.rstrip("\n")
        arch = load_arch(proj)
        setup_addr = arch.SETUP_BASE
        full_asm = arch.setup_jump_asm(setup_asm, start)
        encoding = arch.assemble(full_asm, setup_addr)
        state.memory.map_region(setup_addr, len(encoding) + 256, 7, init_zero=True)
        state.memory.store(setup_addr, encoding)
        arch.set_ip(state, setup_addr)

    _symbolic_options(state)
    sym_regs = _symbolic_regs(proj, state)

    if data is None:
        data = {"regs": {}, "mem": {}}

    explicit_regs = _normalize_data_regs((data or {}).get("regs") or {})
    for name, sym in sym_regs.items():
        if name not in explicit_regs:
            continue
        v = explicit_regs[name]
        if isinstance(v, (list, tuple)):
            continue
        try:
            state.solver.add(sym == int(v))
        except (TypeError, ValueError):
            pass

    _map_state_mem_pages(state, (data or {}).get("mem"))

    _track_mem_access(state)

    ret_addrs = set()
    for f in proj.kb.functions.values():
        if f.addr == start:
            for b in f.blocks:
                for insn in b.capstone.insns:
                    if insn.mnemonic == "ret":
                        ret_addrs.add(insn.address)
            break
    if not ret_addrs:
        ret_addrs = {end}

    simgr = proj.factory.simgr(state)
    simgr.explore(find=list(ret_addrs))

    def _keep(found, name, sym):
        try:
            if not found.solver.symbolic(sym):
                return False
        except Exception:
            return False
        try:
            return any(name in str(c.variables) for c in found.solver.constraints)
        except Exception:
            return False

    return _solution_records(
        simgr.found + simgr.active + simgr.deadended, sym_regs, _keep
    )


def explore_asm(
    code, setup_asm=None, data=None, arch=None, stack_size=None, stack_align=16
):
    if arch is None:
        arch = get_arch()
    if data is None:
        data = {"regs": {}, "mem": {}}

    full = ""
    if setup_asm:
        full += setup_asm.rstrip("\n") + "\n"
    full += (code or "").rstrip("\n") + "\n"
    base = arch.SETUP_BASE
    try:
        encoding = arch.assemble(full, base)
    except Exception:
        return []
    if not encoding:
        return []

    proj = angr.load_shellcode(
        bytes(encoding), arch=arch.ANGR_ARCH_NAME, load_address=base
    )
    state = proj.factory.blank_state(addr=base)
    arch.map_stack(state, size=stack_size, align=stack_align)

    _symbolic_options(state)
    sym_regs = _symbolic_regs(proj, state, prefix="asm_")

    explicit_regs = (data or {}).get("regs") or {}
    canon = getattr(arch, "_canonical_reg", None)
    alias_fn = getattr(arch, "canonical_data_reg", None)
    explicit_canon = {}
    for k, v in explicit_regs.items():
        try:
            ak = alias_fn(k) if alias_fn else str(k).strip().lower()
            ck = canon(ak) if canon else ak
        except Exception:
            continue
        try:
            explicit_canon[ck] = (
                list(v)
                if isinstance(v, (list, tuple))
                else int(v, 0)
                if isinstance(v, str)
                else int(v)
            )
        except (TypeError, ValueError):
            continue
    for name, sym in sym_regs.items():
        cname = canon(name) if canon else name
        if cname in explicit_canon:
            ev = explicit_canon[cname]
            if isinstance(ev, (list, tuple)):
                continue
            try:
                state.solver.add(sym == (ev & 0xFFFFFFFFFFFFFFFF))
            except Exception:
                pass

    harness = set(arch.HARNESS_REGS or ())
    idx = 0
    for name in sorted(sym_regs):
        cname = canon(name) if canon else name
        if cname in explicit_canon or cname in harness:
            continue
        try:
            if state.solver.symbolic(sym_regs[name]):
                state.solver.add(sym_regs[name] == _ASM_SCRATCH_BASE + idx * 0x1000)
                idx += 1
        except Exception:
            pass

    try:
        scratch_page = _ASM_SCRATCH_BASE & ~0xFFF
        state.memory.map_region(scratch_page, (idx + 1) * 0x1000, 7, init_zero=True)
    except Exception:
        pass
    _map_state_mem_pages(state, (data or {}).get("mem"))

    _track_mem_access(state)

    end = base + len(encoding)
    simgr = proj.factory.simgr(state)
    try:
        simgr.explore(find=[end])
    except Exception:
        return []

    try:
        mem_bases = set(arch.mem_regs(full))
    except Exception:
        mem_bases = set()

    def _keep(found, name, sym):
        try:
            if not found.solver.symbolic(sym):
                return False
        except Exception:
            return False
        cname = canon(name) if canon else name
        return cname in explicit_canon or cname in mem_bases or name in mem_bases

    return _solution_records(
        simgr.found + simgr.active + simgr.deadended, sym_regs, _keep
    )


def solve(solutions, branch_cfg=None, arch=None):
    if not solutions:
        return []

    pinned_addrs = set()
    if branch_cfg:
        for addr, value in (branch_cfg.get("mem") or {}).items():
            if value == "predictable":
                pinned_addrs.add(int(addr))

    models = []
    for solution in solutions:
        state = solution["state"]

        ret_sym = _return_sym(state)
        ret_values = _return_values(ret_sym) if ret_sym is not None else []
        states = [state]
        if len(ret_values) > 1:
            states = []
            for v in ret_values:
                st = state.copy()
                st.solver.add(ret_sym == v)
                states.append(st)

        for st in states:
            solver = st.solver
            regs = {}
            for name, sym in solution["reg"].items():
                try:
                    regs[name] = int(solver.eval(sym))
                except Exception:
                    continue

            reads, writes = [], []
            for accesses, out in (
                (solution["reads"], reads),
                (solution["writes"], writes),
            ):
                for addr_sym, length_sym, value_sym in accesses:
                    if addr_sym is None:
                        continue
                    try:
                        addr = int(solver.eval(addr_sym))
                        if not (0 <= addr < (1 << 64)):
                            continue
                    except Exception:
                        continue
                    length = int(solver.eval(length_sym))
                    try:
                        value = (
                            int(solver.eval(value_sym)) if value_sym is not None else 0
                        )
                    except Exception:
                        value = 0
                    out.append((addr, length, value))

            model = {"regs": regs, "reads": reads, "writes": writes}
            if pinned_addrs:
                model["blocks"] = _executed_bbl_addrs([solution])
                model["branch_deps"] = _branch_deps(st, model, pinned_addrs, arch)
            models.append(model)

    return models


def bench(
    exec=None,
    func=None,
    region=None,
    asm=None,
    name=None,
    *,
    label=None,
    mode=None,
    config=None,
    data=None,
    event=None,
    setup=None,
    teardown=None,
    backend=None,
    unroll_n=None,
):
    mode = _require_mode(mode)

    if func is not None and region is not None:
        raise TypeError("cannot pass both 'func' and 'region'")
    if asm is not None and (func is not None or region is not None):
        raise TypeError("cannot pass both 'asm' and 'func'/'region'")

    if func is not None:
        _name = func
    elif region is not None:
        if isinstance(region, str):
            begin, _, end = region.partition("..")
            region = [begin.strip(), end.strip()]
        _name = f"{region[0]}..{region[1]}"
    else:
        _name = name

    return _bench_one(
        exec=exec,
        code=asm,
        name=_name,
        label=label or _name,
        mode=mode,
        config=config,
        data=data,
        event=event,
        setup=setup,
        teardown=teardown,
        backend=backend,
        unroll_n=unroll_n,
    )


def disassm(
    exec_path=None,
    func=None,
    region=None,
    asm=None,
    name=None,
    config=None,
    data=None,
    setup=None,
    teardown=None,
):
    if asm is not None:
        return ".intel_syntax noprefix\n" + _normalize_asm_code(f"{asm};").replace(
            ";", "\n"
        )
    if exec_path is None:
        raise ValueError("a binary (--exec) or asm snippet is required")
    project = angr.Project(
        resolve_exec(exec_path), auto_load_libs=False, load_debug_info=False
    )
    funcs, prototypes = functions(project)
    pat = func or region or name
    if region is not None and not isinstance(region, str):
        pat = f"{region[0]}..{region[1]}"
    if not pat:
        raise ValueError("a function name is required")

    arch = load_arch(project)
    cfg = _merge_config(config)
    if isinstance(cfg, dict) and "data" in cfg:
        raise ValueError(
            "config must not contain 'data'; pass data separately via data={...}"
        )
    eff_data = _merge_data(None, data)
    setup_asm_for_explore = None
    if setup:
        setup_names = setup if isinstance(setup, (list, tuple)) else [setup]
        parts = []
        for s in setup_names:
            if not s:
                continue
            try:
                sym = project.loader.find_symbol(s)
            except Exception:
                sym = None
            if sym is not None:
                try:
                    parts.append(arch.call_asm(sym.rebased_addr))
                    continue
                except Exception:
                    pass
            parts.append(str(s))
        if parts:
            setup_asm_for_explore = "\n".join(parts)
    out = []
    matched = list(resolve_targets(project, pat, funcs))
    if not matched:
        kind = "region" if ".." in str(pat) else "func"
        table = _available_table(project, funcs, kind)
        print(
            f"no target matches {pat!r} in {exec_path!r}; available {kind}s:",
            file=sys.stderr,
        )
        print(table, file=sys.stderr)
        raise ValueError(f"no target matches {pat!r} in {exec_path!r}")
    for _label, start, end in matched:
        try:
            sols = explore(
                project,
                start,
                max(end, start + 1),
                setup_asm=setup_asm_for_explore,
                funcs=funcs,
                data=eff_data,
                prototype=_func_prototype(prototypes.get(_label), eff_data),
            )
        except Exception:
            sols = []
        addrs = _executed_bbl_addrs(sols)
        text = _disasm_executed_asm(project, addrs, arch) if addrs else ""
        if not text:
            text = _disasm_target(project, _label, start, max(end, start + 1), arch)
        out.append(text)
    return "\n".join(out)


def validate(config):
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
    thread = config.get("thread", None)
    if thread is not None:
        if not isinstance(thread, dict):
            raise ValueError(
                f"unknown thread config {thread!r}; expected a dict like "
                "{'affinity': [1], 'priority': 1}"
            )
        for key in thread:
            if key not in _VALID_THREAD_KEYS:
                raise ValueError(
                    f"unknown thread key {key!r}; expected one of "
                    f"{', '.join(sorted(_VALID_THREAD_KEYS))}"
                )
    func = config.get("func", None)
    if func is not None:
        if isinstance(func, dict):
            for key in func:
                if key not in _VALID_FUNC_KEYS:
                    if key == "alignment":
                        raise ValueError(
                            "unknown func key 'alignment'; use short 'align' "
                            "(e.g. {'align': 16})"
                        )
                    raise ValueError(
                        f"unknown func key {key!r}; expected one of "
                        f"{', '.join(sorted(_VALID_FUNC_KEYS))}"
                    )
        elif not isinstance(func, str):
            raise ValueError(
                f"unknown func config {func!r}; expected a dict like "
                "{'order': 'random'}"
            )
    code = config.get("code", None)
    if code is not None:
        if isinstance(code, dict):
            for key in code:
                if key not in _VALID_CODE_KEYS:
                    if key == "alignment":
                        raise ValueError(
                            "unknown code key 'alignment'; use short 'align' "
                            "(e.g. {'align': 16})"
                        )
                    raise ValueError(
                        f"unknown code key {key!r}; expected one of "
                        f"{', '.join(sorted(_VALID_CODE_KEYS))}"
                    )
        elif not isinstance(code, (int, str)):
            raise ValueError(
                f"unknown code config {code!r}; expected a dict like {'align': 16}"
            )
    stack = config.get("stack", None)
    if stack is not None:
        if not isinstance(stack, dict):
            raise ValueError(
                f"unknown stack config {stack!r}; expected a dict like "
                "{'size': 2097152, 'align': 16}"
            )
        for key in stack:
            if key not in _VALID_STACK_KEYS:
                if key == "alignment":
                    raise ValueError(
                        "unknown stack key 'alignment'; use short 'align' "
                        "(e.g. {'align': 16})"
                    )
                raise ValueError(
                    f"unknown stack key {key!r}; expected one of "
                    f"{', '.join(sorted(_VALID_STACK_KEYS))}"
                )
    cache = config.get("cache", None)
    if cache is not None and not isinstance(cache, (str, dict)):
        raise ValueError(
            f"unknown cache config {cache!r}; expected a shortcut like 'hot'/'cold' "
            "or a dict like {'L1d': {'hit_rate': 100}}"
        )
    branch = config.get("branch", None)
    if branch is not None and not isinstance(branch, (str, bool, dict)):
        raise ValueError(f"branch config must be a string or dict, got {branch!r}")
    if branch is not None:
        _branch_config(config)
    for int_key in ("samples", "runs", "probe_runs", "unroll_n"):
        if config.get(int_key) is not None:
            try:
                iv = int(config[int_key])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{int_key} must be an integer, got {config[int_key]!r}"
                ) from e
            if iv < 1:
                raise ValueError(f"{int_key} must be >= 1, got {config[int_key]!r}")
    for int_key in ("min_iterations", "max_iterations", "iterations"):
        if config.get(int_key) is not None:
            try:
                iv = int(config[int_key])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{int_key} must be an integer, got {config[int_key]!r}"
                ) from e
            if iv < 1:
                raise ValueError(f"{int_key} must be >= 1, got {config[int_key]!r}")
    if config.get("target_rel_se") is not None:
        try:
            tv = float(config["target_rel_se"])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"target_rel_se must be a number, got {config['target_rel_se']!r}"
            ) from e
        if not tv > 0:
            raise ValueError(
                f"target_rel_se must be > 0, got {config['target_rel_se']!r}"
            )
    if config.get("backend") is not None and str(config["backend"]) not in (
        "loop",
        "unroll",
    ):
        raise ValueError(
            f"unknown backend {config['backend']!r}; expected one of loop, unroll"
        )
    lo = config.get("min_iterations")
    hi = config.get("max_iterations")
    if lo is not None and hi is not None:
        try:
            if int(lo) > int(hi):
                raise ValueError(
                    f"min_iterations ({lo!r}) must be <= max_iterations ({hi!r})"
                )
        except (TypeError, ValueError) as e:
            if "must be <=" in str(e):
                raise
    return config


def _param_columns_value(val):
    if isinstance(val, (list, tuple)):
        if not val:
            return None
        return list(val)
    try:
        return int(val, 0) if isinstance(val, str) else int(val)
    except (TypeError, ValueError):
        return None


def data_param_columns(data, include_mem=True):
    cols = {}

    canon_regs = {}
    for reg, val in ((data or {}).get("regs") or {}).items():
        v = _param_columns_value(val)
        if v is None:
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

            if f"data.{addr}" not in cols:
                cols[f"data.{addr}"] = v
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

    flat = normalize_config(config)
    return {f"config.{k}": v for k, v in flat.items()}


def explored_data_dict(solutions):
    try:
        models = solve(solutions or [])
    except Exception:
        return {"regs": {}, "mem": {}}
    try:
        harness = set(get_arch().HARNESS_REGS or ())
    except Exception:
        harness = set()
    regs_vals = {}
    mem_vals = {}
    for m in models or []:
        for name, v in (m.get("regs") or {}).items():
            canon = _canonical_data_reg_key(name)
            if canon in harness:
                continue
            try:
                iv = int(v) & 0xFFFFFFFFFFFFFFFF
            except (TypeError, ValueError):
                continue
            lst = regs_vals.setdefault(canon, [])
            if iv not in lst:
                lst.append(iv)
        for a, _, v in (m.get("reads") or []) + (m.get("writes") or []):
            try:
                key = f"0x{int(a):x}"
            except (TypeError, ValueError):
                continue
            try:
                iv = int(v) & 0xFFFFFFFFFFFFFFFF
            except (TypeError, ValueError):
                iv = 0
            lst = mem_vals.setdefault(key, [])
            if iv not in lst:
                lst.append(iv)
    regs = {k: (v[0] if len(v) == 1 else list(v)) for k, v in regs_vals.items()}
    mem = {k: (v[0] if len(v) == 1 else list(v)) for k, v in mem_vals.items()}
    return {"regs": regs, "mem": mem}


def add_param_columns(df, data, config=None):
    if df is None or getattr(df, "empty", False):
        return df
    try:
        cols = data_param_columns(data)
    except Exception:
        return df
    new_cols = {c: v for c, v in cols.items() if c not in df.columns}
    if not new_cols:
        try:
            return _order_data_before_events(df)
        except Exception:
            return df
    try:
        const = pd.DataFrame(
            {c: pd.Series([v] * len(df), index=df.index) for c, v in new_cols.items()}
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
    except Exception:
        for col, val in new_cols.items():
            try:
                df[col] = val
            except Exception:
                continue
    try:
        df = _order_data_before_events(df)
    except Exception:
        pass
    return df


def _config_seed_int(config):
    seed = (config or {}).get("seed", None)
    if seed is None:
        return None
    try:
        return int(seed) & 0xFFFFFFFFFFFFFFFF
    except (TypeError, ValueError):
        digest = hashlib.sha256(str(seed).encode()).digest()
        return int.from_bytes(digest[:8], "little")


def _apply_seed(config):
    seed_int = _config_seed_int(config)
    if seed_int is None:
        return None
    try:
        random.seed(seed_int)
    except Exception:
        pass
    try:
        np.random.seed(seed_int % (2**32))
    except Exception:
        pass
    return seed_int


def _make_rng(config):
    seed_int = _config_seed_int(config)
    if seed_int is None:
        return random.Random()
    return random.Random(seed_int)


def _resolve_align_value(value, where):
    if value is None:
        return 16
    try:
        align = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where} align must be an integer, got {value!r}") from e
    if align < 1 or (align & (align - 1)):
        raise ValueError(f"{where} align must be a power of two >= 1, got {value!r}")
    return align


def _resolve_function_order(config):
    func = (config or {}).get("func", None)
    order = None
    if func is None:
        return "as-is"
    if isinstance(func, str):
        order = func
    elif isinstance(func, dict):
        if "alignment" in func:
            raise ValueError(
                "unknown func key 'alignment'; use short 'align' (e.g. {'align': 16})"
            )
        order = func.get("order", "as-is")
    else:
        raise ValueError(
            f"unknown func config {func!r}; expected a dict like {{'order': 'random'}}"
        )
    order = str(order).strip().lower()
    if order == "as-is":
        return "as-is"
    if order == "random":
        return "random"
    raise ValueError(f"unknown func order {order!r}; expected 'as-is' or 'random'")


def _resolve_function_alignment(config):
    func = (config or {}).get("func", None)
    align = 16
    if isinstance(func, dict):
        if "alignment" in func:
            raise ValueError(
                "unknown func key 'alignment'; use short 'align' (e.g. {'align': 16})"
            )
        align = func.get("align", 16)
    elif func is not None and not isinstance(func, str):
        raise ValueError(
            f"unknown func config {func!r}; expected a dict like {'align': 16}"
        )
    return _resolve_align_value(align, "func")


def _resolve_code_align(config):
    code = (config or {}).get("code", None)
    align = 16
    if code is None:
        return 16
    if isinstance(code, dict):
        if "alignment" in code:
            raise ValueError(
                "unknown code key 'alignment'; use short 'align' (e.g. {'align': 16})"
            )
        align = code.get("align", 16)
    else:
        align = code
    return _resolve_align_value(align, "code")


def _resolve_stack_config(config):
    stack = (config or {}).get("stack", None)
    size = 0x200000
    align = 16
    if stack is None:
        return size, align
    if not isinstance(stack, dict):
        raise ValueError(
            f"unknown stack config {stack!r}; expected a dict like "
            "{'size': 2097152, 'align': 16}"
        )
    if "size" in stack:
        size = stack["size"]
    if "align" in stack:
        align = stack["align"]
    if "alignment" in stack:
        raise ValueError(
            "unknown stack key 'alignment'; use short 'align' (e.g. {'align': 16})"
        )
    try:
        size = int(size, 0) if isinstance(size, str) else int(size)
    except (TypeError, ValueError) as e:
        raise ValueError(f"stack size must be an integer, got {size!r}") from e
    if size <= 0:
        raise ValueError(f"stack size must be > 0, got {size!r}")
    return size, _resolve_align_value(align, "stack")


def _align_directive(align):
    try:
        align = int(align)
    except (TypeError, ValueError):
        return ""
    if align <= 1:
        return ""
    return f".align {align}"


def _with_align(code, align):
    directive = _align_directive(align)
    if not directive or not code:
        return code
    return f"{directive}\n{code}"


def _canonical_data_reg_key(key):
    try:
        arch = get_arch()
        fn = getattr(arch, "canonical_data_reg", None)
        if fn is not None:
            return fn(key)
    except Exception:
        pass
    return str(key).strip().lower()


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


def _mem_entries(addr_str, val):
    base_str, is_range = _parse_addr_key(addr_str)
    try:
        base = int(base_str, 0)
    except (TypeError, ValueError):
        return
    if is_range:
        if isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                yield f"0x{base + i:x}", _to_u64(item)
        else:
            yield f"0x{base:x}", _to_u64(val)
        return
    if isinstance(val, (list, tuple)):
        yield f"0x{base:x}", [_to_u64(item) for item in val]
    else:
        yield f"0x{base:x}", _to_u64(val)


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


class _Args(ctypes.Structure):
    _fields_ = [
        ("iterations", ctypes.c_uint64),
        ("inputs", ctypes.c_void_p),
        ("outputs", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
    ]


def _iter_mem_addrs(data):
    if not data or not data.get("mem"):
        return
    for addr_str in data["mem"]:
        base_str, _ = _parse_addr_key(addr_str)
        try:
            yield int(base_str, 0)
        except (TypeError, ValueError):
            continue


def _map_data_pages(data):
    addrs = list(_iter_mem_addrs(data))
    if not addrs:
        return

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    mmap_fn = libc.mmap
    mmap_fn.restype = ctypes.c_void_p
    mmap_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]

    seen = set()
    for addr in addrs:
        page_addr = addr & ~(_PAGE_SIZE - 1)
        if page_addr in seen:
            continue
        seen.add(page_addr)

        mapped = mmap_fn(
            ctypes.c_void_p(page_addr),
            _PAGE_SIZE,
            _PROT_READ | _PROT_WRITE | _PROT_EXEC,
            _MAP_PRIVATE | _MAP_ANONYMOUS | _MAP_FIXED,
            -1,
            0,
        )
        if mapped == ctypes.c_void_p(-1).value:
            raise OSError(
                ctypes.get_errno(),
                f"failed to map data page at 0x{page_addr:x}",
            )


def _state_dict(models):
    out = []
    for m in models or []:
        reads = []
        for a, ln, v in m.get("reads") or []:
            reads.append([int(a), int(ln), int(v)])
        writes = []
        for a, ln, v in m.get("writes") or []:
            writes.append([int(a), int(ln), int(v)])
        out.append(
            {
                "regs": dict(m.get("regs") or {}),
                "reads": reads,
                "writes": writes,
            }
        )
    return out


def _distribution_dict(meta, buf, arch=None):
    if meta is None or buf is None:
        return None
    if arch is None:
        arch = get_arch()
    cache_tiers = list(arch.CACHE_TIERS)
    end = int(meta["k"])
    dist = {"iterations": end - 1}
    levels = meta.get("levels") or {}
    filtered = {t: lv for t, lv in levels.items() if lv is not None}
    if filtered:
        dist["levels"] = filtered

    regs = {}
    for name in meta.get("reg_names") or []:
        row = meta["reg_col"][name]
        vals, counts = np.unique(buf[row, 1:end], return_counts=True)
        regs[name] = [{"value": int(v), "count": int(c)} for v, c in zip(vals, counts)]
    if regs:
        dist["registers"] = regs

    mem = {}
    tier_totals = {t: 0 for t in cache_tiers}
    for a in meta.get("mem_addrs") or []:
        entry = {}
        vrow = meta["val_col"][a]
        vals, counts = np.unique(buf[vrow, 1:end], return_counts=True)
        entry["values"] = [
            {"value": int(v), "count": int(c)} for v, c in zip(vals, counts)
        ]
        if a in meta["tier_col"]:
            trow = meta["tier_col"][a]
            tiers, counts = np.unique(buf[trow, 1:end], return_counts=True)
            cache_counts = {}
            for t, c in zip(tiers, counts):
                idx = int(t)
                tname = cache_tiers[idx] if 0 <= idx < len(cache_tiers) else str(idx)
                cache_counts[tname] = int(c)
                tier_totals[tname] += int(c)
            entry["cache"] = cache_counts
        mem[f"0x{int(a):x}"] = entry
    if mem:
        dist["memory"] = mem

    used = {t: c for t, c in tier_totals.items() if c}
    if used:
        dist["cache"] = used
    if not regs and not mem:
        return None
    return dist


_BRANCH_CHOICES = (
    "predictable",
    "unpredictable",
    "uniform",
    "random",
    "shuffle",
    "normal",
    "exponential",
)


def _branch_value(spec):
    if isinstance(spec, bool):
        return "predictable" if spec else "unpredictable"
    s = str(spec).strip().lower()
    if s in _BRANCH_CHOICES:
        return s
    return None


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
        prediction = str(prediction).strip().lower()
        if _branch_value(prediction) is None:
            raise ValueError(
                f"branch prediction must be one of {', '.join(_BRANCH_CHOICES)}, "
                f"got {prediction!r}"
            )
    return {"prediction": prediction, "mem": mem, "regs": regs}


def _branch_global_distribution(branch_cfg, predictable):
    try:
        if isinstance(predictable, str):
            s = predictable.strip().lower()
            if s in _BRANCH_CHOICES:
                return s
    except Exception:
        pass
    try:
        pred = (branch_cfg or {}).get("prediction")
        if isinstance(pred, str) and pred.strip().lower() in _BRANCH_CHOICES:
            return pred.strip().lower()
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
        if canon in regs and regs[canon] in _BRANCH_CHOICES:
            return regs[canon]
    except Exception:
        pass
    return _branch_global_distribution(branch_cfg, predictable)


def _branch_mem_distribution(addr, branch_cfg, predictable):
    try:
        mem = (branch_cfg or {}).get("mem") or {}
        a = int(addr)
        if a in mem and mem[a] in _BRANCH_CHOICES:
            return mem[a]
    except Exception:
        pass
    return _branch_global_distribution(branch_cfg, predictable)


def _branch_sample_index(n, rng, distribution):
    try:
        n = int(n)
    except Exception:
        return 0
    if n <= 1:
        return 0
    dist = str(distribution or "unpredictable").strip().lower()
    if dist == "predictable":
        return 0
    if dist == "normal":
        try:
            mu = (n - 1) / 2.0
            sigma = max(n / 4.0, 0.5)
            v = rng.gauss(mu, sigma)
        except Exception:
            return rng.randrange(n) if hasattr(rng, "randrange") else 0
        try:
            idx = int(round(v))
        except Exception:
            idx = 0
        return max(0, min(n - 1, idx))
    if dist == "exponential":
        try:
            v = rng.expovariate(1.0)
        except Exception:
            return rng.randrange(n) if hasattr(rng, "randrange") else 0
        try:
            idx = int(v * n / 2.0)
        except Exception:
            idx = 0
        return max(0, min(n - 1, idx))
    try:
        if hasattr(rng, "randrange"):
            return rng.randrange(n)
    except Exception:
        pass
    try:
        return int(rng.choice(list(range(n))))
    except Exception:
        return 0


def _branch_sample_choice(values, rng, distribution, it=None):
    vals = list(values or [])
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    dist = str(distribution or "unpredictable").strip().lower()
    if dist == "predictable":
        try:
            return vals[int(it or 0) % len(vals)]
        except Exception:
            return vals[0]
    idx = _branch_sample_index(len(vals), rng, dist)
    try:
        return vals[idx]
    except Exception:
        return vals[0]


def _cond_branch_analysis(insns, arch=None):
    if arch is None:
        arch = get_arch()
    canon = getattr(arch, "_canonical_reg", lambda r: str(r).strip().lower())
    insn_list = list(insns or ())
    jcc = None
    for insn in insn_list:
        m = str(insn.mnemonic).strip().lower()
        if m.startswith("j") and m != "jmp":
            jcc = insn
    if jcc is None or not insn_list:
        return set(), []

    def _written(insn):
        out = set()
        try:
            ops = insn.operands or ()
        except Exception:
            ops = ()
        for op in ops:
            try:
                is_write = bool(op.access & capstone.CS_AC_WRITE)
                is_read = op.type == capstone.x86.X86_OP_REG
            except Exception:
                continue
            if is_write and is_read:
                try:
                    nm = insn.reg_name(op.reg)
                except Exception:
                    nm = None
                if nm:
                    out.add(canon(nm))
        return out

    def _read_regs(insn):
        regs = set()
        mems = []
        try:
            ops = insn.operands or ()
        except Exception:
            ops = ()
        for op in ops:
            try:
                otype = op.type
                acc = op.access
            except Exception:
                otype, acc = None, None
            if otype == capstone.x86.X86_OP_MEM:
                base = index = 0
                scale, disp = 1, 0
                try:
                    base, index = op.mem.base, op.mem.index
                    scale, disp = op.mem.scale, op.mem.disp
                except Exception:
                    pass
                for code in (base, index):
                    if code:
                        try:
                            nm = insn.reg_name(code)
                        except Exception:
                            nm = None
                        if nm:
                            regs.add(canon(nm))
                mems.append((base, index, scale, disp))
            elif otype == capstone.x86.X86_OP_REG:
                try:
                    nm = insn.reg_name(op.reg)
                    reads = acc is None or bool(acc & capstone.CS_AC_READ)
                except Exception:
                    nm, reads = None, True
                if nm and reads:
                    regs.add(canon(nm))
        return regs, mems

    flag_writers = getattr(arch, "FLAG_WRITERS", None)
    flag_idx = None
    if flag_writers is not None:
        for i in range(len(insn_list) - 1, -1, -1):
            if insn_list[i] is jcc:
                continue
            if insn_list[i].mnemonic in flag_writers:
                flag_idx = i
                break

    regs = set()
    mems = []
    if flag_idx is None:
        for insn in insn_list:
            r, m = _read_regs(insn)
            regs |= r
            mems.extend(m)
        return regs, mems

    r, m = _read_regs(insn_list[flag_idx])
    regs |= r
    mems.extend(m)
    need = set(r)
    for j in range(flag_idx - 1, -1, -1):
        if not need:
            break
        insn = insn_list[j]
        if not (_written(insn) & need):
            continue
        r2, m2 = _read_regs(insn)
        regs |= r2
        mems.extend(m2)
        need |= r2
    return regs, mems


def _branch_deps(state, model, pinned_addrs, arch=None):
    deps = {}
    try:
        project = state.project
        if arch is None:
            arch = load_arch(project)
    except Exception:
        return deps
    regs = model.get("regs") or {}
    for addr in pinned_addrs:
        try:
            block = project.factory.block(addr)
            insns = list(block.capstone.insns)
        except Exception:
            continue
        if not insns or not _is_branch_mnemonic(insns[-1].mnemonic):
            continue
        cregs, mems = _cond_branch_analysis(insns, arch)
        read_addrs = set()
        for base_id, index_id, scale, disp in mems:
            canon = getattr(arch, "_canonical_reg", lambda r: str(r).strip().lower())
            base = None
            index = None
            if base_id:
                try:
                    base = regs.get(canon(insns[0].reg_name(base_id)))
                except Exception:
                    base = None
            if index_id:
                try:
                    index = regs.get(canon(insns[0].reg_name(index_id)))
                except Exception:
                    index = None
            if base is None and base_id:
                continue
            if index is None and index_id:
                continue
            read_addrs.add(
                ((base or 0) + (index or 0) * (scale or 1) + (disp or 0))
                & 0xFFFFFFFFFFFFFFFF
            )
        deps[addr] = {"regs": cregs, "mem": read_addrs}
    return deps


def _models_for_iteration(models, predictable, rng, branch_cfg=None, it=0):
    if not models:
        return [{"regs": {}, "reads": [], "writes": []}]
    dist = _branch_global_distribution(branch_cfg, predictable)
    if (
        dist == "predictable"
        and not (branch_cfg or {}).get("mem")
        and not (branch_cfg or {}).get("regs")
    ):
        try:
            return [models[int(it or 0) % len(models)]]
        except Exception:
            return [models[0]]
    if dist == "predictable":
        try:
            return [models[int(it or 0) % len(models)]]
        except Exception:
            pass
    try:
        idx = _branch_sample_index(len(models), rng, dist)
        return [models[idx]]
    except Exception:
        pass
    return [rng.choice(models)]


def _collect_mem_addrs(models):
    addrs = set()
    for m in models:
        for a, _, _ in m["reads"] + m["writes"]:
            addrs.add(a)
    return addrs


def _static_table_addrs(project, start, end, elf_obj=None, max_addrs=64):
    found = []
    seen = set()

    def _push(a):
        try:
            a = int(a)
        except (TypeError, ValueError):
            return
        if a not in seen:
            seen.add(a)
            found.append(a)

    ro_ranges = []
    try:
        if elf_obj is not None and getattr(elf_obj, "runtime_base", 0):
            try:
                for _v, _msz, _rt, _fl in elf_obj.segment_runtime_ranges():
                    _fls = str(_fl or "")
                    if "R" in _fls and "W" not in _fls and "X" not in _fls:
                        ro_ranges.append((_rt, _rt + int(_msz)))
            except Exception:
                pass
    except Exception:
        pass
    if not ro_ranges:
        try:
            from elftools.elf.elffile import ELFFile as _ELFFile

            _bin = project.loader.main_object.binary
            with open(_bin, "rb") as _fh:
                _elf = _ELFFile(_fh)
                for _sec in _elf.iter_sections():
                    try:
                        _nm = _sec.name
                        _fl = int(_sec.header["sh_flags"])
                    except Exception:
                        continue

                    if _nm in (".rodata", ".rodata.cst8", ".rodata.cst16") or (
                        (_fl & 0x2) and not (_fl & 0x1) and not (_fl & 0x4)
                    ):
                        try:
                            _va = int(_sec.header["sh_addr"])
                            _sz = int(_sec.header["sh_size"])
                        except Exception:
                            continue
                        if _sz <= 0:
                            continue
                        try:
                            if elf_obj is not None and getattr(
                                elf_obj, "runtime_base", 0
                            ):
                                _rt = elf_obj.runtime_addr(_va)
                            else:
                                _rt = _va
                        except Exception:
                            _rt = _va
                        ro_ranges.append((_rt, _rt + _sz))
        except Exception:
            pass

    def _in_ro(a):
        if not ro_ranges:
            return True
        return any(lo <= int(a) < hi for lo, hi in ro_ranges)

    try:
        import capstone as _cs

        _md = _cs.Cs(_cs.CS_ARCH_X86, _cs.CS_MODE_64)
        _md.detail = True
        try:
            _blob = project.loader.memory.load(
                int(start), max(int(end) - int(start), 1)
            )
        except Exception:
            _blob = b""
        if _blob:
            for _insn in _md.disasm(bytes(_blob), int(start)):
                try:
                    _ops = _insn.operands
                except Exception:
                    continue
                for _op in _ops:
                    try:
                        _is_mem = _op.type == _cs.x86.X86_OP_MEM
                    except Exception:
                        continue
                    if not _is_mem:
                        continue
                    try:
                        _base = _op.mem.base
                        _idx = _op.mem.index
                        _disp = int(_op.mem.disp)
                    except Exception:
                        continue
                    try:
                        _rip = _cs.x86.X86_REG_RIP
                    except Exception:
                        _rip = None
                    _tgt = None
                    if _rip is not None and _base == _rip:
                        _tgt = int(_insn.address) + int(_insn.size) + _disp
                    elif _base == 0 and _idx == 0:
                        _tgt = _disp & 0xFFFFFFFFFFFFFFFF
                    if _tgt is None:
                        continue

                    try:
                        if elf_obj is not None and getattr(elf_obj, "runtime_base", 0):
                            try:
                                _tgt_rt = elf_obj.runtime_addr(_tgt)
                            except Exception:
                                try:
                                    _cle_base = int(
                                        project.loader.main_object.mapped_base or 0
                                    )
                                    _off = int(elf_obj.runtime_base) - int(_cle_base)
                                    _tgt_rt = int(_tgt) + int(_off)
                                except Exception:
                                    _tgt_rt = int(_tgt)
                        else:
                            _tgt_rt = int(_tgt)
                    except Exception:
                        _tgt_rt = int(_tgt)
                    if _in_ro(_tgt_rt) or _in_ro(_tgt):
                        _push(_tgt_rt)
                    if len(found) >= max_addrs:
                        break
                if len(found) >= max_addrs:
                    break
    except Exception:
        pass

    if not found and ro_ranges:
        for _lo, _hi in ro_ranges:
            _a = int(_lo)
            while _a < int(_hi) and len(found) < max_addrs:
                _push(_a)
                _a += 64
            if len(found) >= max_addrs:
                break
    return found


def _static_extra_addrs(extra_addrs=None):
    out = set()
    for a in extra_addrs or ():
        try:
            out.add(int(a, 0) if isinstance(a, str) else int(a))
        except (TypeError, ValueError):
            continue
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
):
    if arch is None:
        arch = get_arch()
    harness_regs = arch.HARNESS_REGS
    cache_tiers = arch.CACHE_TIERS

    mem_levels = mem_levels or {}

    def _sample_tier(a):
        spec = mem_levels.get(a)
        if spec:
            return arch.sample_tier(spec, rng)
        return arch.sample_tier(levels, rng)

    explicit = explicit or {}

    def _norm_explicit_list(val):
        return [_to_u64(x) for x in val] or [_to_u64(0)]

    explicit_regs = {}
    for key, val in (explicit.get("regs") or {}).items():
        canon = _canonical_data_reg_key(key)
        if canon in (harness_regs or set()):
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
    model0_regs = model0.get("regs") or {}
    model0_mem = {}
    for a, _, value in model0.get("reads", []) + model0.get("writes", []):
        model0_mem.setdefault(a, value)
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

    mem_addrs = sorted(
        _collect_mem_addrs(models)
        | set(explicit_mem)
        | _mem_level_int_addrs(mem_levels)
        | _static_extra_addrs(extra_addrs)
    )

    reg_col = {name: i for i, name in enumerate(reg_names)}
    val_col = {}
    tier_col = {}
    col = len(reg_names)
    for a in mem_addrs:
        val_col[a] = col
        tier_col[a] = col + 1
        col += 2

    ncols = col
    naddr = len(mem_addrs)
    k = iterations + 1
    K = max(k, 3 * naddr) if naddr else k

    total_cols = ncols + 3 * naddr
    buf = np.zeros((total_cols, K), dtype=np.uint64)

    def _pinned_reg(name):
        return model0_regs.get(name) if name in pin_regs else None

    def _pinned_mem(a, it):
        if a in pin_mem:
            if a in model0_mem:
                return model0_mem.get(a)
            ev = explicit_mem.get(a)
            if isinstance(ev, (list, tuple)):
                return ev[it % len(ev)] if ev else 0
            if ev is not None:
                return ev
        return None

    def _explicit_reg_value(name, it):
        v = explicit_regs.get(name)
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            dist = _branch_reg_distribution(name, branch_cfg, predictable)
            return _branch_sample_choice(list(v), rng, dist, it) if v else 0
        return v

    def _explicit_mem_value(a, it):
        v = explicit_mem.get(a)
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            dist = _branch_mem_distribution(a, branch_cfg, predictable)
            return _branch_sample_choice(list(v), rng, dist, it) if v else 0
        return v

    for it in range(iterations):
        r8 = iterations - it
        chosen = _models_for_iteration(models, predictable, rng, branch_cfg, it)
        for m in chosen:
            for name in reg_names:
                if name in explicit_regs:
                    continue
                if name in m["regs"]:
                    buf[reg_col[name], r8] = m["regs"][name] & 0xFFFFFFFFFFFFFFFF
            for a, _, value in m["reads"] + m["writes"]:
                if a in val_col and a not in explicit_mem:
                    buf[val_col[a], r8] = value & 0xFFFFFFFFFFFFFFFF
                    tier = _sample_tier(a)
                    buf[tier_col[a], r8] = cache_tiers.index(tier)
        for name in reg_names:
            if name in explicit_regs:
                continue
            pinned = _pinned_reg(name)
            if pinned is not None:
                buf[reg_col[name], r8] = pinned & 0xFFFFFFFFFFFFFFFF
        for a in pin_mem:
            if a in val_col and a not in explicit_mem:
                pinned = _pinned_mem(a, it)
                if pinned is not None:
                    buf[val_col[a], r8] = pinned & 0xFFFFFFFFFFFFFFFF
        for name in explicit_regs:
            v = _explicit_reg_value(name, it)
            if v is not None and name in reg_col:
                buf[reg_col[name], r8] = v & 0xFFFFFFFFFFFFFFFF
        for a in explicit_mem:
            v = _explicit_mem_value(a, it)
            if v is not None and a in val_col:
                buf[val_col[a], r8] = v & 0xFFFFFFFFFFFFFFFF
                tier = _sample_tier(a)
                buf[tier_col[a], r8] = cache_tiers.index(tier)

    l1i_addrs = sorted(
        a
        for a, spec in (mem_levels or {}).items()
        if isinstance(spec, dict) and spec.get("L1i") is not None
    )

    meta = {
        "ncols": ncols,
        "k": k,
        "K": K,
        "reg_col": reg_col,
        "val_col": val_col,
        "tier_col": tier_col,
        "mem_addrs": mem_addrs,
        "l1i_addrs": l1i_addrs,
        "reg_names": reg_names,
        "levels": levels,
        "mem_levels": mem_levels or {},
        "evict_table_col": ncols if naddr else None,
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
        buf[et, 2 * n + i] = (
            data_ptr + meta["tier_col"][a] * K * 8
        ) & 0xFFFFFFFFFFFFFFFF


def _arch_asm(name, meta, data_ptr, arch=None, **kw):
    if arch is None:
        arch = get_arch()
    fn = getattr(arch, name, None)
    if fn is None:
        raise NotImplementedError(
            f"architecture {getattr(arch, 'name', arch)!r} does not implement {name}()"
        )
    return fn(meta, data_ptr, **kw) if kw else fn(meta, data_ptr)


def _steer_asm(meta, data_ptr, arch=None, mem_levels=None, write_values=True):
    return _arch_asm(
        "steer_asm",
        meta,
        data_ptr,
        arch,
        mem_levels=mem_levels,
        write_values=write_values,
    )


def _prime_asm(meta, data_ptr, arch=None):
    return _arch_asm("prime_asm", meta, data_ptr, arch)


def _per_iter_asm(meta, data_ptr, arch=None):
    return _arch_asm("per_iter_asm", meta, data_ptr, arch)


def _probe_plan(config):
    target_rel_se = float(config.get("target_rel_se", 0.005))
    min_iterations = int(config.get("min_iterations", 128))
    max_iterations = int(config.get("max_iterations", 5_000_000))
    return target_rel_se, min_iterations, max_iterations


def _plan_iterations(measured, target_rel_se, min_iterations, max_iterations):
    arr = np.asarray(measured, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return int(min_iterations)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if mean <= 0 or std <= 0 or not np.isfinite(mean):
        return int(min_iterations)
    n = (1.96 * std / (target_rel_se * mean)) ** 2
    return int(np.clip(np.ceil(n), min_iterations, max_iterations))


def _timed_regs_avoid(code, setup, teardown, data, arch):
    avoid = set()
    used = getattr(arch, "used_regs", None)
    if used is None:
        return avoid
    for chunk in (code, setup, teardown):
        if chunk:
            try:
                avoid |= set(used(chunk))
            except Exception:
                pass
    if data:
        alias_fn = getattr(arch, "canonical_data_reg", None)
        for r in data.get("regs") or {}:
            try:
                ar = alias_fn(r) if alias_fn else str(r)
                avoid |= set(used(str(ar)))
            except Exception:
                pass
            try:
                avoid.add(str(r).strip().lower())
                if alias_fn:
                    avoid.add(str(alias_fn(r)).strip().lower())
            except Exception:
                pass
        canon = getattr(arch, "_canonical_reg", None)
        if canon is not None:
            try:
                avoid = {canon(r) for r in avoid}
            except Exception:
                pass
    return avoid


def _normalize_asm_code(code):
    return get_arch().normalize_asm(code)


def _write_jit_map(addr, size, name):
    try:
        from .exec import write_perf_map
    except Exception:
        return
    try:
        label = (
            str(name or "loop").strip().replace("\n", "_").replace(" ", "_") or "loop"
        )
        write_perf_map([(int(addr), int(size), f"perf_bench_loop:{label}")])
    except Exception:
        pass


def _build_loop_asm(
    config,
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
):
    events = _names(events)
    data_script = ""
    data2_script = ""
    data_iter = ""
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
    _level_addrs = _mem_level_int_addrs(mem_levels)
    if models or has_explicit_mem or has_explicit_dist or _extra_set or _level_addrs:
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
        )
        per_iter_buf = buf
        if meta["reg_col"] or meta["mem_addrs"] or meta.get("l1i_addrs"):
            _fill_evict_tables(buf, meta, buf.ctypes.data)
            prime = _prime_asm(meta, buf.ctypes.data, arch)
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
                data_script = _steer_asm(
                    meta,
                    buf.ctypes.data,
                    arch,
                    mem_levels,
                    write_values=False,
                )
                data_iter = _per_iter_asm(meta, buf.ctypes.data, arch)
                data2_script = explicit_reload
            else:
                data_script = _steer_asm(meta, buf.ctypes.data, arch, mem_levels)
                parts = [p for p in (prime, explicit_reload) if p]
                data2_script = "\n".join(parts)
        else:
            if data:
                data2_script = arch.data_reload_asm(data)
    elif data:
        data2_script = arch.data_reload_asm(data)

    avoid = _timed_regs_avoid(code, None, None, data, arch)
    avoid.add(arch.PRIME_SCRATCH_REG)
    if meta is not None:
        for _rn in meta.get("reg_col", ()):
            avoid.add(str(_rn).strip().lower())
            try:
                avoid.add(arch._canonical_reg(_rn))
            except Exception:
                pass
    body = " ".join(c for c in (code, setup, teardown, data2_script, data_iter) if c)
    body_norm = " ".join(body.split()).lower()
    may_call = mode == "throughput" or f" {body_norm} ".find(" call ") >= 0
    t0, t1 = arch.timing(events, indices, callee_saved=may_call, avoid=avoid)
    guard_pre, guard_post = arch.timed_guard(
        events, indices, callee_saved=may_call, avoid=avoid
    )
    if guard_pre or guard_post:
        code = "\n".join(p for p in (guard_pre, code, guard_post) if p)
    full_asm = arch.BENCH[mode].format(
        code=code,
        setup=setup,
        teardown=teardown,
        data=data_script,
        data2=data2_script,
        data_iter=data_iter,
        t0=t0,
        t1=t1,
    )
    return full_asm, (meta if per_iter_buf is not None else None), per_iter_buf


def _asm_error(code, ex):
    msg = str(ex)
    if "Invalid operand" in msg or "KS_ERR_ASM_INVALIDOPERAND" in msg:
        raise ValueError(
            f"invalid assembly for {code!r}: {ex}. "
            "Note `idiv`/`div` take a single r/m operand (e.g. `idiv ecx` "
            "with `--data.eax=.. --data.edx=.. "
            "--data.ecx=..` and "
            "`--backend loop`), and `imul` needs 2-3 operands "
            "(e.g. `imul eax, eax, 42`)."
        ) from ex
    raise ex


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
    report=None,
    branch_cfg=None,
    mem_levels=None,
    extra_addrs=None,
):
    import mmap as _mmap

    events = _names(events)
    n_events = len(events)
    try:
        full_asm, _meta, per_iter_buf = _build_loop_asm(
            config,
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
        )
    except Exception as ex:
        _asm_error(code, ex)
    if report is not None:
        lines = _left_align_asm(full_asm).splitlines()
        label = ",".join(events)
        if map_name:
            header = f"# event: {label} target: {map_name}"
        else:
            header = f"# event: {label}"
        if report.get("code"):
            existing = report["code"]
            if isinstance(existing, (list, tuple)):
                existing = list(existing)
            else:
                existing = str(existing).splitlines()
            report["code"] = existing + ["", header] + lines
        else:
            report["code"] = [header] + lines
        distribution = _distribution_dict(_meta, per_iter_buf, arch)
        if distribution is not None:
            report["distribution"] = distribution
    try:
        code_bytes = arch.assemble(full_asm)
    except Exception as ex:
        _asm_error(code, ex)

    page = _mmap.mmap(
        -1,
        len(code_bytes),
        prot=_mmap.PROT_READ | _mmap.PROT_WRITE | _mmap.PROT_EXEC,
        flags=_mmap.MAP_PRIVATE | _mmap.MAP_ANONYMOUS,
    )
    page.write(code_bytes)
    _JIT_PAGES.append(page)

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
    if mode == "throughput":
        diffs = np.full((runs, n_events), np.nan)
        with _affinity_guard(config), _priority_guard(config):
            for i in range(runs):
                fn(ctypes.byref(args))
                sub = np.asarray(outputs, dtype=np.float64)[0] - oh
                diffs[i, sub > 0] = sub[sub > 0]
        with np.errstate(invalid="ignore"):
            return diffs

    diffs = np.full((runs, iterations, n_events), np.nan)
    with _affinity_guard(config), _priority_guard(config):
        for i in range(runs):
            fn(ctypes.byref(args))
            sub = np.asarray(outputs, dtype=np.float64) - oh
            diffs[i, sub > 0] = sub[sub > 0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(diffs, axis=0)


def _normalize_groups(event):
    if event is None:
        return [["duration_time"]]
    if isinstance(event, str):
        parts = _split_list(event)
        return [parts] if parts else [["duration_time"]]
    if isinstance(event, (list, tuple)):
        if len(event) == 0:
            return [["duration_time"]]
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
            return groups or [["duration_time"]]
        names = _split_list(event)
        return [names] if names else [["duration_time"]]
    s = str(event).strip()
    return [[s]] if s else [["duration_time"]]


def _names(events):
    return _split_list(events) or ["duration_time"]


def _records(
    file_name, name, mode, result, group, divisor=1, operations=1, iterations=None
):
    def _base(samples):
        return {
            "file": file_name,
            "name": name,
            "mode": mode,
            "iterations": iterations,
            "samples": samples,
            "operations": operations,
        }

    records = []
    if isinstance(result, pd.DataFrame):
        for samples, (_, row) in enumerate(result.iterrows()):
            rec = _base(samples)
            for ev in group:
                rec[ev] = row[ev] / divisor
            records.append(rec)
    else:
        for samples, value in enumerate(result):
            rec = _base(samples)
            for ev in group:
                rec[ev] = value / divisor
            records.append(rec)
    return records


def _combine_results(records_lists):
    frames = [pd.DataFrame(r) for r in records_lists if r]
    if not frames:
        return pd.DataFrame()

    def _merge_group(group):
        merged = group[0]
        for other in group[1:]:
            merged = merged.merge(
                other,
                on=["file", "name", "mode", "iterations", "samples", "operations"],
                how="outer",
            )
        return merged

    groups = {}
    for fr in frames:
        try:
            key = (
                str(fr["file"].iloc[0]),
                str(fr["name"].iloc[0]),
                str(fr["mode"].iloc[0]),
            )
        except Exception:
            key = (None, None, None)
        groups.setdefault(key, []).append(fr)
    merged = [_merge_group(g) for g in groups.values()]
    if len(merged) == 1:
        df = merged[0]
    else:
        df = pd.concat(merged, ignore_index=True)
    cols = ["file", "name", "mode", "iterations", "samples", "operations"] + [
        c
        for c in df.columns
        if c not in ("file", "name", "mode", "iterations", "samples", "operations")
    ]
    df = df[cols].sort_values(["file", "name", "mode", "samples"])
    return df.set_index(["file", "name", "mode"])


def _resolve_backend(backend, config, default):
    if backend is None:
        backend = (config or {}).get("backend", default)
    if backend is None:
        backend = default
    name = str(backend).strip()
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {', '.join(_BACKENDS)}"
        )
    return name


def _resolve_unroll_n(unroll_n, config):
    default_n = _DEFAULT_BENCH.get("unroll_n", 5)
    if unroll_n is None:
        unroll_n = (config or {}).get("unroll_n", default_n)
    if unroll_n is None:
        unroll_n = default_n
    try:
        unroll_n = int(unroll_n)
    except (TypeError, ValueError) as e:
        raise ValueError(f"unroll_n must be an integer, got {unroll_n!r}") from e
    if unroll_n < 1:
        raise ValueError(f"unroll_n must be >= 1, got {unroll_n!r}")
    return unroll_n


def _bench(
    config,
    mode,
    code,
    setup,
    teardown,
    solutions,
    overhead=0,
    events=None,
    data=None,
    addr_offset=0,
    addr_range=None,
    arch=None,
    map_name=None,
    iterations_tracker=None,
    report=None,
    extra_addrs=None,
):
    if arch is None:
        arch = get_arch()
    samples = config.get("samples", 100)
    runs = int(config.get("runs", 10))

    events = core.with_leader(_names(events))
    group = core.is_group(events)
    try:
        force_typ = core.choose_type(events)
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
        counters, indices = core.open_counters(events, force_typ, group)
    except Exception:
        _restore_aff()
        raise

    def close():
        for counter in counters:
            try:
                counter.disable()
            finally:
                counter.close()

    rng = _make_rng(config)
    seed_int = _config_seed_int(config)
    branch_cfg = _branch_config(config)
    predictable = (
        config.get("branch", "unpredictable") == "predictable"
        if branch_cfg is None
        else branch_cfg["prediction"] == "predictable"
    )
    levels = arch.data_cache_levels(config)
    mem_levels = arch.resolve_mem_cache(config)

    models = []
    if solutions:
        models = solve(solutions, branch_cfg=branch_cfg, arch=arch)
        if models and addr_offset:
            for m in models:
                m["reads"] = [
                    (a + addr_offset, ln, v)
                    for a, ln, v in m["reads"]
                    if addr_range is None or addr_range[0] <= a < addr_range[1]
                ]
                m["writes"] = [
                    (a + addr_offset, ln, v)
                    for a, ln, v in m["writes"]
                    if addr_range is None or addr_range[0] <= a < addr_range[1]
                ]
        if models:
            for m in models:
                for a, _, _ in m["reads"] + m["writes"]:
                    _map_data_pages({"mem": {str(a): 0}})

    if data:
        _map_data_pages(data)
        data_setup = arch.data_setup_asm(data)
        setup = data_setup + ("\n" + setup if setup else "\n")

    try:
        pinned = config.get("iterations")
        if pinned is not None:
            iterations = int(pinned)
        else:
            target_rel_se, min_iterations, max_iterations = _probe_plan(config)
            probe_runs = max(1, min(runs, int(config.get("probe_runs", 3))))
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
                report=None,
                branch_cfg=branch_cfg,
                mem_levels=mem_levels,
                extra_addrs=extra_addrs,
            )
            probe_series = probe[:, 0] if probe.ndim > 1 else probe
            iterations = _plan_iterations(
                probe_series, target_rel_se, min_iterations, max_iterations
            )
        if iterations_tracker is not None:
            try:
                iterations_tracker.append(int(iterations))
            except Exception:
                pass

        data_series = _collect(
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
            data=data,
            map_name=map_name,
            report=report,
            branch_cfg=branch_cfg,
            mem_levels=mem_levels,
            extra_addrs=extra_addrs,
        )
    except Exception:
        close()
        _restore_aff()
        raise

    close()
    _restore_aff()

    if report is not None:
        report["state"] = _state_dict(models)

    df = pd.DataFrame(data_series, columns=events).dropna(how="all")
    if df.empty:
        df = pd.DataFrame(
            {ev: [0.0] * int(samples) for ev in events},
        )
    else:
        df = df.fillna(0.0)
    if seed_int is None:
        return df.sample(n=min(int(samples), len(df))), iterations
    return (
        df.sample(n=min(int(samples), len(df)), random_state=seed_int % (2**32)),
        iterations,
    )


def _symbolic_options(state):
    state.options.discard(angr.options.LAZY_SOLVES)
    for opt in (
        angr.options.TRACK_CONSTRAINTS,
        angr.options.TRACK_JMP_ACTIONS,
        angr.options.TRACK_ACTION_HISTORY,
        angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY,
        angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS,
        angr.options.NO_SYMBOLIC_JUMP_RESOLUTION,
        angr.options.STRICT_PAGE_ACCESS,
        angr.options.CONSERVATIVE_READ_STRATEGY,
        angr.options.SYMBOLIC_WRITE_ADDRESSES,
    ):
        state.options.add(opt)


def _symbolic_regs(proj, state, prefix=""):
    sym_regs = {}
    ip_name = proj.arch.register_names.get(proj.arch.ip_offset)
    sp_name = proj.arch.register_names.get(proj.arch.sp_offset)
    for reg in proj.arch.register_list:
        if not reg.general_purpose or reg.name == ip_name or reg.name == sp_name:
            continue
        sym = claripy.BVS(f"{prefix}{reg.name}", reg.size * 8)
        state.registers.store(reg.name, sym)
        sym_regs[reg.name] = sym
    return sym_regs


def _track_mem_access(state):
    def _on_read(s):
        if "reads" not in s.globals:
            s.globals["reads"] = []
        s.globals["reads"].append(
            (
                s.inspect.mem_read_address,
                s.inspect.mem_read_length,
                s.inspect.mem_read_expr,
            )
        )

    def _on_write(s):
        if "writes" not in s.globals:
            s.globals["writes"] = []
        s.globals["writes"].append(
            (
                s.inspect.mem_write_address,
                s.inspect.mem_write_length,
                s.inspect.mem_write_expr,
            )
        )

    state.inspect.b("mem_read", when=angr.BP_AFTER, action=_on_read)
    state.inspect.b("mem_write", when=angr.BP_AFTER, action=_on_write)


def _map_state_mem_pages(state, mem):
    seen = set()
    for addr_str in mem or {}:
        try:
            s = str(addr_str).strip()
            if s.endswith(":"):
                s = s[:-1].strip()
            a = int(s, 0) & ~0xFFF
        except (TypeError, ValueError):
            continue
        if a in seen:
            continue
        seen.add(a)
        try:
            state.memory.map_region(a, 0x1000, 7, init_zero=True)
        except Exception:
            pass


def _solution_records(states, sym_regs, keep=None):
    results = []
    for found in states:
        reg = {n: s for n, s in sym_regs.items() if keep is None or keep(found, n, s)}
        results.append(
            {
                "state": found,
                "reg": reg,
                "reads": list(found.globals.get("reads", [])),
                "writes": list(found.globals.get("writes", [])),
            }
        )
    return results


def _return_sym(state):
    try:
        ret_reg = state.arch.register_names[state.arch.ret_offset]
        return state.registers.load(ret_reg)
    except Exception:
        return None


def _return_values(sym):
    leaves = []
    pending = [sym]
    while pending:
        node = pending.pop()
        if getattr(node, "op", None) == "If":
            pending.append(node.args[1])
            pending.append(node.args[2])
        elif getattr(node, "op", None) == "BVV":
            leaves.append(node.args[0])
        else:
            return []
    return sorted(set(leaves))


def _require_mode(mode):
    if mode is None:
        raise ValueError("a mode is required (--mode latency|throughput)")
    if isinstance(mode, (list, tuple, set)):
        modes = list(mode)
    elif isinstance(mode, str):
        modes = [m for m in mode.replace(",", " ").split() if m]
    else:
        modes = [mode]
    for m in modes:
        if m not in ("latency", "throughput"):
            raise ValueError(f"unknown mode {m!r}; expected 'latency' or 'throughput'")
    if len(modes) != 1:
        raise ValueError(
            "exactly one mode is required (--mode latency or --mode throughput)"
        )
    return modes[0]


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


def _merge_config(override):
    import copy

    base = copy.deepcopy(_DEFAULT_BENCH)
    if not override:
        return base
    merged = _deep_merge(base, dict(override))
    return validate(merged)


def _content_hash8(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    except OSError:
        return None


def _stable_hash8(payload):
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
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


def _as_normalized_data(data):
    if not data:
        return {"regs": {}, "mem": {}}
    if isinstance(data, dict) and any(k in data for k in ("regs", "mem")):
        regs = dict(data.get("regs") or {})
        mem = dict(data.get("mem") or {})
        return {"regs": regs, "mem": mem}
    return _config_data({"data": data})


def _merge_data(base, override):
    b = _as_normalized_data(base)
    o = _as_normalized_data(override)
    b["regs"].update(o["regs"])
    for k, v in o["mem"].items():
        b["mem"][k] = v
    return b


def _generate_seed():
    try:
        return random.SystemRandom().randint(0, 2**31 - 1)
    except Exception:
        return random.randint(0, 2**31 - 1)


def _first_scalar(val):
    if isinstance(val, (list, tuple)):
        if not val:
            return None
        val = val[0]
    try:
        return int(val, 0) if isinstance(val, str) else int(val)
    except (TypeError, ValueError):
        return None


def _order_data_before_events(df, events=None):
    try:
        cols = list(df.columns)
    except Exception:
        return df
    data_cols = [c for c in cols if str(c).startswith("data.")]
    if not data_cols:
        return df
    rest = [c for c in cols if c not in data_cols]
    if events:
        try:
            evset = set(events)
        except Exception:
            evset = set()
        head = [c for c in rest if c not in evset]
        tail = [c for c in rest if c in evset]
        ordered = head + data_cols + tail
    else:
        meta_order = [
            "file",
            "name",
            "mode",
            "iterations",
            "samples",
            "operations",
            "time",
            "address",
            "size",
        ]
        lead = [c for c in meta_order if c in rest]
        tail = [c for c in rest if c not in lead]
        ordered = lead + data_cols + tail
    ordered += [c for c in cols if c not in ordered]
    try:
        return df[ordered]
    except Exception:
        return df


def _available_table(project, funcs, kind):
    try:
        from .info import metadata as _info_fn
    except Exception:
        return ""
    try:
        df = _info_fn(project.loader.main_object.binary)
    except Exception:
        return ""
    if kind == "func":
        df = df[df["kind"] == "func"]
    else:
        df = df[df["kind"].isin(["label", "region"])]
    if df.empty:
        return "(no targets found)"
    df = df.copy()
    try:
        df["start"] = df["start"].apply(lambda x: f"0x{int(x):x}")
        df["end"] = df["end"].apply(lambda x: f"0x{int(x):x}")
    except Exception:
        pass
    try:
        return df.to_string(index=False)
    except Exception:
        return str(df)


def _unpack_bench(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, None


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
    report,
    overhead_code,
    main_code,
    divisor=1,
    extra=None,
):
    extra = extra or {}
    overhead_res, _ = _unpack_bench(
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
            iterations_tracker=tracker,
            **extra,
        )
    )
    overhead = np.asarray(overhead_res.median(axis=0), dtype=float)
    result, iterations = _unpack_bench(
        _bench(
            config=config,
            mode=mode,
            setup=setup,
            code=main_code,
            teardown=teardown,
            solutions=solutions,
            overhead=overhead,
            events=group,
            data=data,
            arch=arch,
            map_name=map_name,
            iterations_tracker=tracker,
            report=report,
            **extra,
        )
    )
    return result, iterations


def _bench_one(**kwargs):
    mode = _require_mode(kwargs.get("mode"))
    file = kwargs.get("exec")
    src_file = file
    if file:
        file = resolve_exec(file)
        if not Path(file).exists():
            raise ValueError(f"exec file {file!r} does not exist")
    code = kwargs.get("code")
    name = kwargs.get("name")
    label = kwargs.get("label") or name
    setup = kwargs.get("setup")
    teardown = kwargs.get("teardown")
    event = kwargs.get("event", "duration_time")
    backend_arg = kwargs.get("backend")
    unroll_arg = kwargs.get("unroll_n")

    config = _merge_config(kwargs.get("config"))
    if isinstance(config, dict) and "data" in config:
        raise ValueError(
            "config must not contain 'data'; pass data separately via data={...}"
        )
    data = _merge_data(None, kwargs.get("data"))

    if config.get("seed") is None:
        config["seed"] = _generate_seed()
    thread = config.get("thread")
    if not isinstance(thread, dict):
        thread = {}
    if thread.get("affinity") is None:
        _aff = _current_affinity_list()
        thread["affinity"] = _aff if _aff is not None else []
    if thread.get("priority") is None:
        thread["priority"] = _current_priority_value()
    config["thread"] = thread
    _iterations_tracker = []
    _iterations_pinned = config.get("iterations")
    _parse_affinity(_affinity_spec(config))
    _parse_priority(_priority_spec(config))
    _thread_cfg(config)
    _resolved_order = _resolve_function_order(config)
    _resolved_align = _resolve_function_alignment(config)
    config["func"] = {"align": _resolved_align, "order": _resolved_order}
    _code_align = _resolve_code_align(config)
    config["code"] = {"align": _code_align}
    _stack_size, _stack_align = _resolve_stack_config(config)
    config["stack"] = {"size": _stack_size, "align": _stack_align}
    _all_solutions = []
    _apply_seed(config)
    report = {}

    groups = _normalize_groups(event)
    try:
        groups = [core.with_leader(g) for g in groups]
    except Exception:
        pass
    measure_groups = [g for g in groups if g]

    results = []
    cpu_info = cpuinfo(list(CPUINFO_FIELDS)).iloc[0].to_dict()

    if code:
        if not name:
            name = code
        code = _normalize_asm_code(f"{code};")
        setup = _normalize_asm_code("\n".join(setup) + "\n") if setup else ""
        teardown = _normalize_asm_code("\n".join(teardown) + "\n") if teardown else ""
        arch = get_arch()
        backend = _resolve_backend(
            backend_arg, config, "unroll" if mode == "latency" else "loop"
        )
        unroll_n = _resolve_unroll_n(unroll_arg, config)
        if backend == "unroll" and mode == "throughput":
            raise ValueError("backend 'unroll' can only be used with mode 'latency'")
        config["backend"] = backend
        config["unroll_n"] = unroll_n
        try:
            asm_solutions = explore_asm(
                code,
                setup_asm=setup or None,
                data=data,
                arch=arch,
                stack_size=_stack_size,
                stack_align=_stack_align,
            )
        except Exception:
            asm_solutions = []
        _all_solutions.extend(list(asm_solutions or []))
        binary_id = {"asm": code, "name": name}
        for group in measure_groups:
            if backend == "unroll":
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
                    report,
                    _with_align(f"{code}" * unroll_n, _code_align),
                    _with_align(f"{code}" * (2 * unroll_n), _code_align),
                )
                operations = 1 if mode == "latency" else _iterations
                recs = _records(
                    None,
                    label,
                    mode,
                    result,
                    group,
                    divisor=unroll_n,
                    operations=operations,
                    iterations=_iterations,
                )
            else:
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
                    report,
                    _with_align("nop;", _code_align),
                    _with_align(f"{code}", _code_align),
                )
                operations = 1 if mode == "latency" else _iterations
                recs = _records(
                    None,
                    label,
                    mode,
                    result,
                    group,
                    operations=operations,
                    iterations=_iterations,
                )
            results.append(recs)
    else:
        if not name:
            raise ValueError("a name is required to analyze a binary (--exec)")

        project = angr.Project(
            file,
            auto_load_libs=False,
            load_debug_info=False,
        )
        arch = load_arch(project)

        obj = Elf(project.loader)
        obj.map_elf()
        obj.setup_stack(size=_stack_size, align=_stack_align)

        funcs, prototypes = functions(project)

        if _resolve_function_order(config) == "random":
            obj.randomize_layout(
                dict(funcs),
                seed=_config_seed_int(config),
                align=_resolve_function_alignment(config),
            )

        try:
            obj.write_perf_map()
        except Exception:
            pass

        setup_name = setup
        if setup:
            setup = arch.call_seq_asm(obj.get_symbol(setup))
        else:
            setup = ""
        if teardown:
            teardown = arch.call_seq_asm(obj.get_symbol(teardown))
        else:
            teardown = ""

        ang_setup = ""
        if setup_name:
            sym = project.loader.find_symbol(setup_name)
            if sym is not None:
                ang_setup = arch.call_asm(sym.rebased_addr)
            else:
                ang_setup = setup

        base = os.path.basename(str(src_file))
        binary_id = {"path": base, "sha": _content_hash8(src_file)}
        file_label = f"{base}@{binary_id['sha']}"
        info_dict = {"cpu": cpu_info, "binary": binary_id}
        targets = list(resolve_targets(project, name, funcs))
        if not targets:
            kind = "region" if ".." in str(name) else "func"
            table = _available_table(project, funcs, kind)
            print(
                f"no target matches {name!r} in {src_file!r}; available {kind}s:",
                file=sys.stderr,
            )
            print(table, file=sys.stderr)
            raise ValueError(f"no target matches {name!r} in {src_file!r}")
        multi_target = len(targets) > 1
        for target_label, start, end in targets:
            rec_name = target_label if multi_target else label
            proto = _func_prototype(prototypes.get(target_label), data)
            solutions = explore(
                project,
                start,
                end,
                setup_asm=ang_setup,
                funcs=funcs,
                data=data,
                prototype=proto,
                stack_size=_stack_size,
                stack_align=_stack_align,
            )

            _all_solutions.extend(list(solutions or []))
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
            code_asm = arch.call_seq_asm(target, fence=(mode == "latency"))
            if mode == "throughput":
                restore = arch.data_reload_asm(data)
                if restore:
                    code_asm = f"{restore}\n{code_asm}"
            code_asm = _with_align(code_asm, _resolved_align)

            backend = _resolve_backend(backend_arg, config, "loop")
            unroll_n = _resolve_unroll_n(unroll_arg, config)
            if backend == "unroll" and mode == "throughput":
                raise ValueError(
                    "backend 'unroll' can only be used with mode 'latency'"
                )
            config["backend"] = backend
            config["unroll_n"] = unroll_n
            try:
                _static_addrs = _static_table_addrs(project, start, end, obj)
            except Exception:
                _static_addrs = []
            extra = {
                "addr_offset": addr_offset,
                "addr_range": addr_range,
                "extra_addrs": list(_static_addrs or []),
            }
            for group in measure_groups:
                if backend == "unroll":
                    code_n = "\n".join([code_asm] * unroll_n)
                    code_2n = "\n".join([code_asm] * (2 * unroll_n))
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
                        report,
                        code_n,
                        code_2n,
                        extra=extra,
                    )
                    operations = 1 if mode == "latency" else _iterations
                    recs = _records(
                        file_label,
                        rec_name,
                        mode,
                        result,
                        group,
                        divisor=unroll_n,
                        operations=operations,
                        iterations=_iterations,
                    )
                    results.append(recs)
                    continue
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
                    report,
                    _with_align(
                        arch.call_seq_nop_asm(fence=(mode == "latency")),
                        _resolved_align,
                    ),
                    code_asm,
                    extra=extra,
                )
                operations = 1 if mode == "latency" else _iterations
                recs = _records(
                    file_label,
                    rec_name,
                    mode,
                    result,
                    group,
                    operations=operations,
                    iterations=_iterations,
                )
                results.append(recs)

    df = _combine_results(results)
    assert not df.empty, kwargs

    if "duration_time" in df.columns:
        try:
            _hz = float(cpu_info.get("hz"))
        except (TypeError, ValueError):
            _hz = 0.0
        if _hz and _hz > 0:
            df["duration_time"] = df["duration_time"] / _hz

    for ev in {e for group in measure_groups for e in group}:
        if ev not in df.columns:
            df[ev] = np.nan

    if _iterations_pinned is not None:
        try:
            config["iterations"] = int(_iterations_pinned)
        except (TypeError, ValueError):
            pass
    elif _iterations_tracker:
        try:
            config["iterations"] = int(max(_iterations_tracker))
        except Exception:
            pass
    else:
        try:
            config["iterations"] = int(config.get("min_iterations", 128))
        except (TypeError, ValueError):
            pass
    if config.get("backend") is None:
        config["backend"] = backend if "backend" in locals() else "loop"
    if config.get("unroll_n") is None:
        config["unroll_n"] = _DEFAULT_BENCH.get("unroll_n", 5)

    try:
        col_data = {
            "regs": dict((data or {}).get("regs") or {}),
            "mem": dict((data or {}).get("mem") or {}),
        }
        try:
            explored = explored_data_dict(_all_solutions)
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
        df = add_param_columns(df, col_data, config)
    except Exception:
        try:
            df = add_param_columns(df, data, config)
        except Exception:
            pass

    if code:
        df.attrs["config"] = config
        df.attrs["info"] = {"cpu": cpu_info, "binary": binary_id}
        df.attrs["file"] = None
    else:
        df.attrs["config"] = config
        df.attrs["info"] = info_dict
        df.attrs["file"] = file_label
    df.attrs["id"] = _id_hash(data, config, binary_id)
    df.attrs["data"] = data
    if report.get("code") is not None:
        df.attrs["code"] = report["code"]
    if report.get("state") is not None:
        df.attrs["state"] = report["state"]
    if report.get("distribution") is not None:
        df.attrs["distribution"] = report["distribution"]
    return df


def _branch_label(addr):
    return f".L{int(addr):x}"


def _is_branch_mnemonic(mnemonic):
    m = str(mnemonic or "").strip().lower()
    return m in ("call", "jmp") or m.startswith("j") or m.startswith("loop")


def _left_align_asm(text):
    out = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            out.append("")
        else:
            out.append(line.lstrip())
    if not out:
        return ""
    return "\n".join(out) + "\n"


def _hex_values(op_str):
    out = set()
    s = op_str or ""
    n = len(s)
    i = 0
    while i < n:
        i = s.find("0x", i)
        if i < 0:
            break
        j = i + 2
        while j < n and s[j] in _HEXDIGITS:
            j += 1
        if j > i + 2:
            try:
                out.add(int(s[i:j], 16))
            except ValueError:
                pass
        i = j
    return out


def _executed_bbl_addrs(solutions):
    addrs = []
    seen = set()
    for sol in solutions or []:
        try:
            hist = (
                sol.get("state").history.bbl_addrs
                if isinstance(sol, dict)
                else sol.history.bbl_addrs
            )
        except Exception:
            continue
        try:
            lst = list(hist)
        except Exception:
            continue
        for a in lst:
            try:
                ai = int(a)
            except (TypeError, ValueError):
                continue
            if ai not in seen:
                seen.add(ai)
                addrs.append(ai)
    return addrs


def _branch_targets(insns):
    targets = set()
    for insn in insns:
        if _is_branch_mnemonic(insn.mnemonic):
            targets |= _hex_values(insn.op_str or "")
    return {a: _branch_label(a) for a in targets}


def _render_insn(insn, labels):
    op_str = insn.op_str or ""
    if _is_branch_mnemonic(insn.mnemonic) and op_str:
        op_str = _sub_intel_labels(op_str, labels)
    return f"{insn.mnemonic} {op_str}".strip()


def _disasm_executed_asm(project, addrs, arch=None):
    if arch is None:
        arch = load_arch(project)
    setup_base = arch.SETUP_BASE
    filtered = []
    for a in addrs or []:
        if setup_base <= a < setup_base + 0x100000:
            continue
        try:
            block = project.factory.block(a)
            insns = list(block.capstone.insns)
        except Exception:
            continue
        if not insns:
            continue
        filtered.append((a, insns))
    if not filtered:
        return ""
    try:
        filtered = sorted(filtered, key=lambda p: int(p[0]))
    except Exception:
        pass
    labels = _branch_targets([ins for _, insns in filtered for ins in insns])
    lines = [".intel_syntax noprefix"]
    emitted_labels = set()
    emitted_insns = set()
    for bbl_addr, insns in filtered:
        if bbl_addr in labels and bbl_addr not in emitted_labels:
            lines.append(f"{labels[bbl_addr]}:")
            emitted_labels.add(bbl_addr)
        for insn in insns:
            try:
                _addr = int(insn.address)
            except Exception:
                _addr = insn.address
            if _addr in emitted_insns:
                continue
            emitted_insns.add(_addr)
            if (
                insn.address != bbl_addr
                and insn.address in labels
                and insn.address not in emitted_labels
            ):
                lines.append(f"{labels[insn.address]}:")
                emitted_labels.add(insn.address)
            lines.append(_render_insn(insn, labels))

    for addr, lab in labels.items():
        if addr not in emitted_labels:
            lines.append(f"{lab}:")
            emitted_labels.add(addr)
    return "\n".join(lines) + "\n"


def _disasm_target(project, label, start, end, arch=None):
    if arch is None:
        arch = load_arch(project)
    try:
        blob = project.loader.memory.load(start, end - start)
    except Exception:
        return ""
    lines = [".intel_syntax noprefix"]
    try:
        insns = list(arch.disassembler().disasm(bytes(blob), start))
    except Exception:
        insns = []
    labels = _branch_targets(insns)
    internal = {a for a in labels if start <= a < end}
    try:
        for insn in insns:
            if insn.address in internal:
                lines.append(f"{labels[insn.address]}:")
            lines.append(_render_insn(insn, labels))
    except Exception:
        pass
    return "\n".join(lines) + "\n"


def _sub_intel_labels(op_str, labels):
    s = op_str or ""
    n = len(s)
    out = []
    i = 0
    while i < n:
        k = s.find("0x", i)
        if k < 0:
            out.append(s[i:])
            break
        j = k + 2
        while j < n and s[j] in _HEXDIGITS:
            j += 1
        out.append(s[i:k])
        tok = s[k:j]
        try:
            out.append(str(labels.get(int(tok, 16), tok)))
        except ValueError:
            out.append(tok)
        i = j
    return "".join(out)
