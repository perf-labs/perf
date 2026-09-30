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
import ctypes
import fcntl
import functools
import logging
import mmap
import numbers
import os
import warnings

from .arch import arch as _get_arch

_DURATION_TIME = "duration_time"
_PERF_TYPE_HARDWARE = 0
_PERF_TYPE_SOFTWARE = 1
_HW_EVENTS = {
    "cycles": 0x0,
    "cpu-cycles": 0x0,
    "instructions": 0x1,
    "cache-references": 0x2,
    "cache-misses": 0x3,
    "branch-instructions": 0x4,
    "branches": 0x4,
    "branch-misses": 0x5,
    "bus-cycles": 0x6,
    "stalled-cycles-frontend": 0x7,
    "stalled-cycles-backend": 0x8,
    "ref-cycles": 0x9,
    "ref-cpu-cycles": 0x9,
}
_SW_EVENTS = {
    "cpu-clock": 0x0,
    "task-clock": 0x1,
    "page-faults": 0x2,
    "faults": 0x2,
    "context-switches": 0x3,
    "cs": 0x3,
    "cpu-migrations": 0x4,
    "migrations": 0x4,
    "minor-faults": 0x5,
    "major-faults": 0x6,
    "alignment-faults": 0x7,
    "emulation-faults": 0x8,
    _DURATION_TIME: 0x9,
}
_TOPDOWN_EVENTS = (
    "topdown-retiring",
    "topdown-bad-spec",
    "topdown-fe-bound",
    "topdown-be-bound",
)
_DISABLED = 1 << 0
_EXCLUDE_USER = 1 << 4
_EXCLUDE_KERNEL = 1 << 5
_EXCLUDE_HV = 1 << 6
_EXCLUDE_GUEST = 1 << 16
_EXCLUDE_HOST = 1 << 17
_PRECISE_IP = 0x3 << 18
_DEFAULT_FLAGS = _DISABLED | _EXCLUDE_KERNEL | _EXCLUDE_HV
_SYSFS_EVENT_SOURCES = "/sys/bus/event_source/devices"
_LEADER_FOR_PREFIX = {"topdown-": "slots"}
_RETRY_ERRNOS = (22, 19, 95)
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
_SYSFS_NODES = "/sys/devices/system/node"
_NR_SET_MEMPOLICY = 238
_NR_GET_MEMPOLICY = 239
_MPOL_DEFAULT = 0
_MPOL_BIND = 2
_PATTERN_GLOB = "*?["
_PATTERN_REGEX = "()|+^${}\\"
_PATTERN_META = _PATTERN_GLOB + _PATTERN_REGEX


class PerfCounter:
    SYS_perf_event_open = _get_arch()._PERF_SYSCALL_NR
    libc = ctypes.CDLL("libc.so.6", use_errno=True)

    PERF_EVENT_IOC_ENABLE = ord("$") << 8
    PERF_EVENT_IOC_DISABLE = (ord("$") << 8) | 1
    PERF_EVENT_IOC_RESET = (ord("$") << 8) | 3

    class perf_event_attr(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_uint32),
            ("size", ctypes.c_uint32),
            ("config", ctypes.c_uint64),
            ("sample_period", ctypes.c_uint64),
            ("sample_type", ctypes.c_uint64),
            ("read_format", ctypes.c_uint64),
            ("flags", ctypes.c_uint64),
            ("__reserved_2", ctypes.c_uint64),
            ("__reserved_3", ctypes.c_uint64),
        ]

    class perf_event_mmap_page(ctypes.Structure):
        _fields_ = [
            ("version", ctypes.c_uint32),
            ("compat_version", ctypes.c_uint32),
            ("lock", ctypes.c_uint32),
            ("index", ctypes.c_uint32),
            ("offset", ctypes.c_int64),
            ("time_enabled", ctypes.c_uint64),
            ("time_running", ctypes.c_uint64),
            ("capabilities", ctypes.c_uint64),
            ("pmc_width", ctypes.c_uint16),
            ("time_shift", ctypes.c_uint16),
            ("time_mult", ctypes.c_uint32),
            ("time_offset", ctypes.c_uint64),
        ]

    def perf_event_open(self, attr, pid=0, cpu=-1, group_fd=-1, flags=0):
        fd = self.libc.syscall(
            self.SYS_perf_event_open,
            ctypes.byref(attr),
            pid,
            cpu,
            group_fd,
            flags,
        )

        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))

        return fd

    def __init__(self, config, type=None, flags=None, pid=0, cpu=-1, group_fd=-1):
        attr = self.perf_event_attr()

        attr.type = _PERF_TYPE_HARDWARE if type is None else type
        attr.size = ctypes.sizeof(attr)
        attr.config = config

        attr.flags = _DEFAULT_FLAGS if flags is None else flags

        self.fd = self.perf_event_open(attr, pid=pid, cpu=cpu, group_fd=group_fd)

        try:
            self.meta_mmap = mmap.mmap(
                self.fd,
                mmap.PAGESIZE,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ,
            )
        except BaseException:
            os.close(self.fd)
            raise

        self.meta = self.perf_event_mmap_page.from_buffer_copy(
            self.meta_mmap[: ctypes.sizeof(self.perf_event_mmap_page)]
        )

    def refresh_meta(self):
        self.meta = self.perf_event_mmap_page.from_buffer_copy(
            self.meta_mmap[: ctypes.sizeof(self.perf_event_mmap_page)]
        )

    def enable(self):
        fcntl.ioctl(self.fd, self.PERF_EVENT_IOC_RESET, 0)
        fcntl.ioctl(self.fd, self.PERF_EVENT_IOC_ENABLE, 0)

    def disable(self):
        fcntl.ioctl(self.fd, self.PERF_EVENT_IOC_DISABLE, 0)

    @property
    def rdpmc_index(self):
        self.refresh_meta()

        if self.meta.index == 0:
            raise RuntimeError(
                "Kernel did not expose an RDPMC index "
                "(rdpmc disabled or event not scheduled)"
            )

        return self.meta.index - 1

    def close(self):
        self.meta_mmap.close()
        os.close(self.fd)


def one_line(value):
    return str(value).replace("\r", " ").replace("\n", " ")


def concrete_int(expr):
    if getattr(expr, "op", None) != "BVV":
        return None
    try:
        return int(expr.args[0])
    except (IndexError, TypeError, ValueError):
        return None


def eval_int(state, expr):
    if expr is None:
        return None
    value = concrete_int(expr)
    if value is not None:
        return value
    try:
        return int(state.solver.eval(expr))
    except Exception:
        return None


def eval_ints(state, exprs):
    out = [None] * len(exprs)
    pending = {}
    for index, expr in enumerate(exprs):
        if expr is None:
            continue
        value = concrete_int(expr)
        if value is not None:
            out[index] = value
            continue
        try:
            pending.setdefault(expr, []).append(index)
        except TypeError:
            out[index] = eval_int(state, expr)
    for expr, indexes in pending.items():
        value = eval_int(state, expr)
        for index in indexes:
            out[index] = value
    return out


@functools.lru_cache(maxsize=4096)
def demangle(name):
    if not name:
        return name
    s = one_line(name)
    if "_Z" not in s:
        return s
    try:
        import cxxfilt as _cxxfilt

        out = _cxxfilt.demangle(s)
    except Exception:
        return s
    return one_line(out) if out and out != s else s


def resolve(event):
    return _candidates(event)[0]


def resolve_all(event):
    return list(_candidates(event))


def resolve_on(event, typ):
    for t, config, flags in _candidates(event):
        if t == typ:
            return (t, config, flags)
    raise ValueError(f"event {event!r} not exported by PMU type {typ}")


def with_leader(events):
    names = list(events) if events is not None else []
    if not names:
        return names
    leaders = set()
    for e in names:
        if not isinstance(e, str) or _is_duration_event(e):
            continue
        leader = _leader_for(_event_base(e))
        if leader:
            leaders.add(leader)
    present = {_norm(_event_base(e)) for e in names if isinstance(e, str)}
    leaders = {ld for ld in leaders if _norm(ld) not in present}
    if len(leaders) != 1:
        return names
    leader = next(iter(leaders))
    devs = {_explicit_pmu(e) for e in names if _explicit_pmu(e)}
    if len(devs) > 1:
        return names
    if devs:
        dev = next(iter(devs))
        return names if not _pmu_has(dev, leader) else [f"{dev}/{leader}/"] + names
    target = next(
        (
            _pmu_for_type(t)
            for e in names
            if isinstance(e, str) and _leader_for(_event_base(e))
            for t, _c, _f in resolve_all(e)
            if _pmu_for_type(t) and _pmu_has(_pmu_for_type(t), leader)
        ),
        None,
    )
    if target is None:
        return names

    for e in names:
        if not isinstance(e, str) or not _leader_for(_event_base(e)):
            continue
        for t, _c, _f in resolve_all(e):
            dev = _pmu_for_type(t)
            if dev and dev != target and not _pmu_has(dev, leader):
                return names
    return [f"{target}/{leader}/"] + names


def choose_type(events):
    sets = []
    for e in events or []:
        if not isinstance(e, str) or _is_duration_event(e):
            continue
        try:
            typs = [t for t, _c, _f in _candidates(e)]
        except Exception:
            return None
        if not typs:
            return None
        sets.append(set(typs))
    if not sets:
        return None
    common = set.intersection(*sets)
    if not common:
        return None
    if len(common) == 1:
        return next(iter(common))
    needed = {
        _leader_for(_event_base(e))
        for e in events
        if isinstance(e, str) and _leader_for(_event_base(e))
    } - {None}
    if needed:
        free = [
            t
            for t in common
            if _pmu_for_type(t)
            and not any(_pmu_has(_pmu_for_type(t), ld) for ld in needed)
        ]
        if free:
            return sorted(free)[0]
    return sorted(common)[0]


def open_event(event, pid=0, group_fd=-1):
    last = None
    for typ, config, flags in _candidates(event):
        try:
            return _make_counter(config, typ, flags, pid=pid, group_fd=group_fd)
        except OSError as ex:
            last = ex
            if getattr(ex, "errno", None) not in _RETRY_ERRNOS:
                raise
    if last is not None:
        raise last
    typ, config, flags = resolve(event)
    return _make_counter(config, typ, flags, pid=pid, group_fd=group_fd)


def open_event_on(event, typ, pid=0, group_fd=-1):
    t, config, flags = resolve_on(event, typ)
    return _make_counter(config, t, flags, pid=pid, group_fd=group_fd)


def enable_rdpmc():
    try:
        c = open_event("cycles")
    except Exception:
        return None
    try:
        c.enable()
    except Exception:
        pass
    return c


def assert_rdpmc_unique(events, indices):
    real = [i for i in indices if i is not None]
    if len(real) != len(set(real)):
        raise RuntimeError(
            f"events {list(events)!r} alias to the same RDPMC index {real}; "
            "this PMU's grouped events cannot be read via RDPMC. Use the "
            "standalone PMU instead (e.g. cpu_atom/topdown-.../ on hybrid) "
            "or `perf stat`"
        )


@functools.cache
def cpus_for_type(typ):
    dev = _pmu_for_type(typ)
    if dev is None:
        return None
    for cand in (
        os.path.join(_SYSFS_EVENT_SOURCES, dev, "cpus"),
        os.path.join("/sys/devices", dev, "cpus"),
    ):
        try:
            with open(cand) as fh:
                txt = fh.read().strip()
        except OSError:
            continue
        if txt and _parse_cpu_list(txt):
            return _parse_cpu_list(txt)
    return None


def pmu_cpus(typ):
    cpus = None
    try:
        cpus = cpus_for_type(typ)
    except Exception:
        cpus = None
    if cpus or typ != _PERF_TYPE_HARDWARE:
        return cpus
    for dev, dev_type, _ev, _fmt in _sysfs_pmus():
        if dev == "cpu_core":
            try:
                return cpus_for_type(dev_type)
            except Exception:
                return None
    return None


def event_cpus(groups):
    if not groups:
        return None
    if any(isinstance(g, str) for g in groups):
        groups = [list(groups)]
    sets = []
    for events in groups:
        try:
            typ = choose_type(events)
        except Exception:
            typ = None
        if typ is None:
            continue
        cpus = pmu_cpus(typ)
        if cpus:
            sets.append(set(cpus))
    if not sets:
        return None
    return sorted(set.intersection(*sets))


def is_group(events):
    if len(events) < 2:
        return False
    first = _event_base(events[0])
    if not first or first.strip() == _DURATION_TIME:
        return False
    return any(
        isinstance(e, str)
        and not _is_duration_event(e)
        and _leader_for(_event_base(e)) == first
        for e in events[1:]
    )


def pin_pmu(config, typ, requested=None):
    if typ is None:
        return lambda: None
    cpus = pmu_cpus(typ)
    if not cpus:
        return lambda: None
    req = requested
    if req is None:
        try:
            cur = sorted(os.sched_getaffinity(0))
        except OSError:
            cur = None
        avail = [c for c in cpus if cur is None or c in cur]
    else:
        avail = [c for c in req if c in cpus]
    if not avail:
        raise RuntimeError(
            f"events require PMU CPUs {cpus} (PMU type {typ}) but affinity "
            f"is {req}; pin to a matching CPU or use the matching PMU's events"
        )
    try:
        old = sorted(os.sched_getaffinity(0))
    except OSError:
        old = None
    old_cfg, touched, created = None, False, False
    holder = None
    if old is not None:
        os.sched_setaffinity(0, set(avail))
    if isinstance(config, dict):
        try:
            holder, created = _thread_store(config)
        except ValueError:
            holder = None
        if holder is not None:
            old_cfg, touched = holder.get("affinity", None), True
            holder["affinity"] = list(avail)

    def _restore():
        if old is not None:
            try:
                os.sched_setaffinity(0, set(old))
            except OSError:
                pass
        if not touched or holder is None or not isinstance(config, dict):
            return
        try:
            if created and old_cfg is None:
                if set(holder.keys()) == {"affinity"}:
                    _thread_drop(config, holder)
                else:
                    del holder["affinity"]
            else:
                holder["affinity"] = old_cfg
        except Exception:
            pass

    return _restore


def open_counters(events, force_typ=None, group=False, pid=0, require_indices=True):
    import time

    counters, opened = [], {}
    leader_fd = -1
    try:
        for pos, ev in enumerate(events):
            if isinstance(ev, str) and _is_duration_event(ev):
                continue
            if pos == 0 and group:
                c = (
                    open_event_on(ev, force_typ, pid=pid)
                    if force_typ is not None
                    else open_event(ev, pid=pid)
                )
                counters.append(c)
                opened[pos] = c
                try:
                    force_typ = resolve(ev)[0]
                except Exception:
                    pass
                leader_fd = c.fd
                continue
            if force_typ is not None:
                try:
                    c = open_event_on(ev, force_typ, pid=pid, group_fd=leader_fd)
                except ValueError:
                    c = open_event(ev, pid=pid, group_fd=leader_fd)
            else:
                c = open_event(ev, pid=pid, group_fd=leader_fd)
            counters.append(c)
            opened[pos] = c
        for c in counters:
            try:
                c.enable()
            except Exception:
                pass
        indices = []
        for pos, ev in enumerate(events):
            if isinstance(ev, str) and _is_duration_event(ev):
                indices.append(None)
                continue
            if not require_indices:
                indices.append(None)
                continue
            c = opened[pos]
            for _ in range(100):
                try:
                    indices.append(c.rdpmc_index)
                    break
                except RuntimeError:
                    time.sleep(0.001)
            else:
                raise RuntimeError(
                    f"kernel did not expose an RDPMC index for event {ev!r}; "
                    "rdpmc may be disabled or the event may not be a hardware counter"
                )
        assert_rdpmc_unique(events, indices)
    except Exception:
        for c in counters:
            try:
                try:
                    c.disable()
                finally:
                    c.close()
            except Exception:
                pass
        raise
    return counters, indices


def track_target_cpus(force_typ):
    if force_typ is not None:
        cpus = None
        try:
            cpus = pmu_cpus(force_typ)
        except Exception:
            cpus = None
        if cpus:
            return list(cpus)
    return None


def perf_attr_bytes(typ, config, flags):
    attr = PerfCounter.perf_event_attr()
    ctypes.memset(ctypes.byref(attr), 0, ctypes.sizeof(attr))
    attr.type = int(typ)
    attr.size = int(ctypes.sizeof(attr))
    attr.config = int(config) & 0xFFFFFFFFFFFFFFFF
    attr.flags = int(flags) & 0xFFFFFFFFFFFFFFFF
    return bytes(ctypes.string_at(ctypes.byref(attr), ctypes.sizeof(attr)))


def remote_open_self_counters(pid, ev_list, force_typ, group, anchor):
    from .arch import x86_64 as _arch
    from .prof import remote_mmap, remote_syscall, signed_rax, write_mem

    plan = []
    _ft = force_typ
    for pos, ev in enumerate(ev_list):
        if isinstance(ev, str) and _is_duration_event(ev):
            continue
        if pos == 0 and group:
            if _ft is not None:
                try:
                    cands = [resolve_on(ev, _ft)]
                except ValueError:
                    cands = list(_candidates(ev))
            else:
                cands = list(_candidates(ev))
            plan.append((pos, ev, cands))
            try:
                _ft = resolve(ev)[0]
            except Exception:
                pass
            continue
        if _ft is not None:
            try:
                cands = [resolve_on(ev, _ft)]
            except ValueError:
                cands = list(_candidates(ev))
        else:
            cands = list(_candidates(ev))
        plan.append((pos, ev, cands))
    scratch = remote_mmap(pid, 4096, anchor=anchor)
    child_fds = []
    for pos, ev, cands in plan:
        if group and child_fds:
            group_fd = child_fds[0]
        else:
            group_fd = (1 << 64) - 1
        last_err = None
        opened = None
        for typ, config, flags in cands:
            flags = int(flags) & ~1
            blob = perf_attr_bytes(typ, config, flags)
            write_mem(pid, scratch, blob)
            rax = remote_syscall(
                pid,
                _arch._NR_PERF_EVENT_OPEN,
                rdi=scratch,
                rsi=0,
                rdx=(1 << 64) - 1,
                r10=group_fd,
                r8=0,
                anchor=anchor,
            )
            signed = signed_rax(rax)
            if signed >= 0:
                opened = signed
                break
            last_err = signed
            if (-signed) not in (22, 19, 95):
                break
        if opened is None:
            raise OSError(
                -last_err if last_err else 22,
                f"remote perf_event_open failed for {ev!r} in child {pid}",
            )
        child_fds.append(opened)
        try:
            remote_mmap(pid, 4096, fd=opened, shared=True, anchor=anchor)
        except OSError as ex:
            raise OSError(
                getattr(ex, "errno", 22) or 22,
                f"remote mmap failed for {ev!r} (child fd {opened})",
            ) from ex
    return child_fds


def open_task_counters(ev_list, force_typ, group, pid, anchor=None):
    proxy_counters, proxy_indices = open_counters(ev_list, force_typ, group, pid=0)
    try:
        assert_rdpmc_unique(ev_list, proxy_indices)
    finally:
        for c in proxy_counters:
            try:
                try:
                    c.disable()
                finally:
                    c.close()
            except Exception:
                pass
    remote_open_self_counters(pid, ev_list, force_typ, group, anchor)
    return [], proxy_indices


def default_affinity(groups, numa=None):
    cpus = event_cpus(groups)
    if cpus is not None:
        allowed = set(cpus)
        cur = _current_affinity_list()
        if cur:
            allowed &= set(cur)
        if numa is not None:
            node = _numa_cpus(numa)
            if node:
                allowed &= set(node)
        return sorted(allowed)[0] if allowed else None
    if numa is not None:
        cpu = _numa_first_cpu(numa)
        if cpu is not None:
            return cpu
    return first_cpu()


def first_cpu():
    cpus = _current_affinity_list()
    return cpus[0] if cpus else 0


@functools.cache
def supported_events():
    names = []
    seen = set()
    for source in (sorted(_HW_EVENTS), sorted(_SW_EVENTS)):
        for name in source:
            for key in (name, _norm(name)):
                if not key or key in seen:
                    continue
                if not _resolves(key):
                    continue
                seen.add(key)
                names.append(key)
    return tuple(names)


@functools.cache
def event_catalog():
    names = list(_TOPDOWN_EVENTS)
    seen = set(names)
    for name in supported_events():
        if name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


def is_event_pattern(event):
    if not isinstance(event, str):
        return False
    name, _ = _split_modifiers(event.strip())
    base, _ = _split_pattern_suffix(name)
    return any(c in base for c in _PATTERN_META)


def expand_event_alias(event):
    if not isinstance(event, str):
        return [event]
    s = event.strip()
    if not s:
        return [s]
    name, mods = _split_modifiers(s)
    base, suffix = _split_pattern_suffix(name)
    if not any(c in base for c in _PATTERN_META):
        return [s]
    matched = _match_events(base)
    if not matched:
        return [s]
    if mods:
        matched = [f"{e}:{mods}" for e in matched]
    return [f"{e}{suffix}" for e in matched]


def expand_event_aliases(events):
    if isinstance(events, str):
        events = [events]
    out = []
    for e in events or []:
        out.extend(expand_event_alias(e))
    return out


class _SizelessImportFilter(logging.Filter):
    _MESSAGES = (
        "Symbol imported without a known size",
        "has an invalid tls_data_size",
        "has a negative tls_data_start",
    )

    def filter(self, record):
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not any(m in message for m in self._MESSAGES)


def _read_text(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _func_prototype(proto, data=None):
    from angr.calling_conventions import SimTypeFunction, SimTypeNum

    if proto is not None:
        return proto
    n = 0
    if data:
        arg_regs = list(_get_arch()._ARG_REGS or [])
        regs = (data or {}).get("regs") or {}
        canon_regs = set()
        for k in regs:
            try:
                canon_regs.add(_get_arch().canonical_data_reg(k))
            except Exception:
                try:
                    canon_regs.add(str(k).strip().lower())
                except Exception:
                    continue
        for i, r in enumerate(arg_regs):
            try:
                rc = str(r).strip().lower()
            except Exception:
                rc = r
            if r in regs or rc in canon_regs:
                n = i + 1
    return SimTypeFunction([SimTypeNum(64, False)] * n, SimTypeNum(64, False))


@functools.lru_cache(maxsize=4096)
def _split_modifiers(name):
    if ":" not in name:
        return name.strip(), ""
    head, _, rest = name.partition(":")
    return head.strip(), rest.replace(":", "")


@functools.lru_cache(maxsize=4096)
def _modifier_flags(mods):
    flags = _DEFAULT_FLAGS
    precise = 0
    for m in mods:
        if m == "u":
            flags &= ~_EXCLUDE_USER
            flags |= _EXCLUDE_KERNEL | _EXCLUDE_HV
        elif m == "k":
            flags &= ~_EXCLUDE_KERNEL
            flags |= _EXCLUDE_USER | _EXCLUDE_HV
        elif m == "h":
            flags &= ~_EXCLUDE_HV
        elif m == "G":
            flags |= _EXCLUDE_HOST
        elif m == "H":
            flags |= _EXCLUDE_GUEST
        elif m == "p":
            precise += 1
        else:
            raise ValueError(f"unsupported event modifier {m!r} in {mods!r}")
    if precise:
        flags = (flags & ~_PRECISE_IP) | (min(precise, 3) << 18)
    return flags


@functools.lru_cache(maxsize=4096)
def _parse_spec(spec):
    attrs = {}
    for part in spec.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        attrs[key.strip()] = value.strip()
    return attrs


@functools.lru_cache(maxsize=4096)
def _parse_bitspec(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        word = "config"
        if ":" in part:
            word, rng = part.split(":", 1)
            word = word.strip() or "config"
        else:
            rng = part
        rng = rng.split("%", 1)[0].strip()
        if not rng:
            continue
        try:
            if "-" in rng:
                lo_s, hi_s = rng.split("-", 1)
                lo, hi = (int(lo_s, 0) if lo_s else 0), int(hi_s, 0)
            else:
                lo = hi = int(rng, 0)
        except ValueError:
            continue
        out.append((word, lo, hi))
    return out


def _encode(attrs, formats):
    words = {"config": 0, "config1": 0, "config2": 0}
    for key, value in attrs.items():
        bits = formats.get(key)
        if not bits:
            continue
        try:
            value = int(value, 0)
        except (TypeError, ValueError):
            continue
        for word, lo, hi in _parse_bitspec(bits):
            width = hi - lo + 1
            words[word] |= (value & ((1 << width) - 1)) << lo
    return words["config"]


def _sysfs_formats(path):
    formats = {}
    if not os.path.isdir(path):
        return formats
    for name in os.listdir(path):
        spec = _read_text(os.path.join(path, name))
        if spec:
            formats[name] = spec
    return formats


@functools.lru_cache(maxsize=1)
@functools.cache
def _sysfs_pmus():
    def _pref(dev):
        if dev == "cpu":
            return 0
        if dev == "cpu_core":
            return 1
        if dev.startswith("cpu"):
            return 2
        return 3

    pmus = []
    if os.path.isdir(_SYSFS_EVENT_SOURCES):
        for dev in os.listdir(_SYSFS_EVENT_SOURCES):
            devdir = os.path.join(_SYSFS_EVENT_SOURCES, dev)
            typ = _read_text(os.path.join(devdir, "type"))
            evdir = os.path.join(devdir, "events")
            if typ is None or not os.path.isdir(evdir):
                continue
            try:
                typ = int(typ, 0)
            except ValueError:
                continue
            formats = _sysfs_formats(os.path.join(devdir, "format"))
            events = {}
            for name in os.listdir(evdir):
                spec = _read_text(os.path.join(evdir, name))
                if not spec:
                    continue
                try:
                    events[name] = _encode(_parse_spec(spec), formats)
                except Exception:
                    continue
            pmus.append((dev, typ, events, formats))
    pmus.sort(key=lambda p: (_pref(p[0]), p[0]))
    return pmus


@functools.lru_cache(maxsize=4096)
def _norm(name):
    return str(name).replace("_", "-").strip()


@functools.lru_cache(maxsize=4096)
def _event_base(event):
    if isinstance(event, int):
        return ""
    name, _ = _split_modifiers(str(event))
    name = name.strip()
    if "/" in name:
        _, _, rest = name.partition("/")
        rest = rest.rstrip("/").strip()
        if "=" in rest:
            return ""
        return rest.strip()
    return name.strip()


@functools.lru_cache(maxsize=4096)
def _explicit_pmu(event):
    if not isinstance(event, str):
        return None
    name, _ = _split_modifiers(event)
    if "/" not in name:
        return None
    dev, _, _ = name.partition("/")
    return dev.strip() or None


@functools.cache
def _pmu_for_type(typ):
    for dev, t, _events, _formats in _sysfs_pmus():
        if t == typ:
            return dev
    return None


@functools.cache
def _pmu_has(dev, event_name):
    for d, _typ, events, _formats in _sysfs_pmus():
        if d != dev and d.replace("_", "-") != dev:
            continue
        if event_name in events or _norm(event_name) in events:
            return True
    return False


@functools.lru_cache(maxsize=4096)
def _leader_for(base):
    base = _norm(base)
    for prefix, leader in _LEADER_FOR_PREFIX.items():
        if base.startswith(prefix):
            return leader
    return None


@functools.lru_cache(maxsize=4096)
def _candidates(event):
    if isinstance(event, int):
        return [(_PERF_TYPE_HARDWARE, event, _DEFAULT_FLAGS)]
    name, mods = _split_modifiers(event)
    if not name:
        raise ValueError(f"invalid event spec {event!r}")
    flags = _modifier_flags(mods)
    if name.strip() == _DURATION_TIME:
        return [(_PERF_TYPE_SOFTWARE, _SW_EVENTS[_DURATION_TIME], flags)]
    key = _norm(name)
    if key in _HW_EVENTS:
        return [(_PERF_TYPE_HARDWARE, _HW_EVENTS[key], flags)]
    if key in _SW_EVENTS:
        return [(_PERF_TYPE_SOFTWARE, _SW_EVENTS[key], flags)]
    if key.startswith("r") and len(key) > 1:
        try:
            return [(_PERF_TYPE_HARDWARE, int(key[1:], 16), flags)]
        except ValueError:
            pass
    if "/" in name:
        dev, _, rest = name.partition("/")
        rest = rest.rstrip("/").strip()
        for d, typ, events, formats in _sysfs_pmus():
            if d != dev and d.replace("_", "-") != dev:
                continue
            if "=" in rest:
                return [(typ, _encode(_parse_spec(rest), formats), flags)]
            if rest in events:
                return [(typ, events[rest], flags)]
        raise ValueError(f"unknown event {event!r}: PMU {dev!r} exports no {rest!r}")
    out = []
    for _, typ, events, _ in _sysfs_pmus():
        if key in events:
            out.append((typ, events[key], flags))
        elif name in events:
            out.append((typ, events[name], flags))
    if out:
        return out
    hint = (
        f"no supported event matches {event!r} (a wildcard or regex expands to "
        "the events it matches, e.g. 'topdown-*')"
        if is_event_pattern(event)
        else "try 'cycles', 'instructions', 'cache-misses', 'branch-instructions', "
        "'duration_time' or run `perf list` for more"
    )
    raise ValueError(f"unknown event {event!r}; {hint}")


def _make_counter(config, typ, flags, pid=0, group_fd=-1):
    return PerfCounter(config, type=typ, flags=flags, pid=pid, group_fd=group_fd)


@functools.lru_cache(maxsize=4096)
def _parse_range_list(s, err=None):
    cpus = set()
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            try:
                lo, hi = int(lo_s, 0), int(hi_s, 0)
            except ValueError:
                if err is not None:
                    raise ValueError(err)
                continue
            cpus.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            try:
                cpus.add(int(part, 0))
            except ValueError:
                if err is not None:
                    raise ValueError(err)
                continue
    return cpus


@functools.lru_cache(maxsize=4096)
def _parse_cpu_list(s):
    return sorted(_parse_range_list(s))


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


def _text(v):
    try:
        import pandas as _pd

        if _pd.isna(v):
            return ""
    except Exception:
        pass
    try:
        s = str(v)
    except Exception:
        return ""
    return "" if s.strip().lower() in ("", "nan", "none", "nat") else s


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
    if len(entries) > 1:
        raise ValueError(
            f"multiple threads per run are not supported yet, got "
            f"{thread!r}; {_THREAD_HINT}"
        )
    return entries[0]


def _affinity_spec(config):
    return _thread_cfg(config).get("affinity", None)


def _priority_spec(config):
    return _thread_cfg(config).get("priority", None)


def _numa_spec(config):
    return _thread_cfg(config).get("numa", None)


def _parse_affinity(spec):
    if spec is None:
        return None
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
    try:
        old = sorted(os.sched_getaffinity(0))
    except OSError:
        yield
        return
    try:
        os.sched_setaffinity(0, set(cpus))
    except OSError as ex:
        warnings.warn(f"failed to set affinity to {cpus}: {ex}")
        yield
        return
    try:
        yield
    finally:
        try:
            os.sched_setaffinity(0, set(old))
        except OSError:
            pass


def _parse_priority(spec):
    if spec is None:
        return None
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
    old_nice = None
    try:
        old_nice = os.getpriority(os.PRIO_PROCESS, 0)
    except OSError:
        old_nice = None
    try:
        try:
            os.setpriority(os.PRIO_PROCESS, 0, int(nice))
        except OSError as ex:
            warnings.warn(f"failed to set nice to {nice}: {ex}")
    except ValueError:
        raise
    except OSError as ex:
        warnings.warn(f"failed to set priority {nice}: {ex}")
    try:
        yield
    finally:
        if old_nice is not None:
            try:
                os.setpriority(os.PRIO_PROCESS, 0, int(old_nice))
            except OSError:
                pass


def _current_affinity_list():
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        return None


def _current_priority_value():
    try:
        return int(os.getpriority(os.PRIO_PROCESS, 0))
    except Exception:
        pass
    return 0


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


def _numa_nodes():
    try:
        return sorted(
            int(name[4:])
            for name in os.listdir(_SYSFS_NODES)
            if name.startswith("node") and name[4:].isdigit()
        )
    except OSError:
        return []


def _numa_cpus(node):
    try:
        with open(f"{_SYSFS_NODES}/node{int(node)}/cpulist") as f:
            cpus = _parse_range_list(f.read())
    except (OSError, TypeError, ValueError):
        return []
    return sorted(cpus)


def _numa_first_cpu(node):
    return (_numa_cpus(node) or [None])[0]


def _node_mask(node):
    node = int(node)
    size = 8 * ((node // 64) + 1)
    mask = (ctypes.c_ubyte * size)()
    mask[node // 8] |= 1 << (node % 8)
    return mask, size * 8 + 1


@functools.lru_cache(maxsize=1)
def _libc_syscall():
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        return libc.syscall
    except Exception:
        return None


def _set_mempolicy(mode, node=None):
    syscall = _libc_syscall()
    if syscall is None:
        raise OSError("set_mempolicy is unavailable")
    if node is None:
        return syscall(_NR_SET_MEMPOLICY, mode, None, 0)
    mask, maxnode = _node_mask(node)
    return syscall(_NR_SET_MEMPOLICY, mode, ctypes.byref(mask), maxnode)


def _get_mempolicy():
    syscall = _libc_syscall()
    if syscall is None:
        raise OSError("get_mempolicy is unavailable")
    mode = ctypes.c_int(_MPOL_DEFAULT)
    if syscall(_NR_GET_MEMPOLICY, ctypes.byref(mode), None, 0) != 0:
        raise OSError(ctypes.get_errno(), "get_mempolicy failed")
    return int(mode.value)


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
    try:
        old_mode = _get_mempolicy()
    except OSError:
        old_mode = _MPOL_DEFAULT
    try:
        if _set_mempolicy(_MPOL_BIND, node) != 0:
            raise OSError(ctypes.get_errno(), "set_mempolicy failed")
    except OSError as ex:
        warnings.warn(f"failed to bind memory to numa node {node}: {ex}")
        yield
        return
    try:
        yield
    finally:
        if _set_mempolicy(old_mode) != 0:
            warnings.warn(
                f"failed to restore the numa memory policy to mode {old_mode}; "
                "the process stays bound to node "
                f"{node} for the rest of this run"
            )


def _resolves(name):
    try:
        _candidates(name)
        return True
    except Exception:
        return False


def _split_pattern_suffix(name):
    if "/" not in name:
        return name, ""
    head, _, tail = name.partition("/")
    if "/" in head or "=" in head or not head:
        return name, ""
    return head, f"/{tail}"


def _match_events(pattern):
    import fnmatch
    import re

    regex = None
    if any(c in pattern for c in _PATTERN_REGEX) and not any(
        c in pattern for c in _PATTERN_GLOB
    ):
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = None
    pool = supported_events() if _is_bare_pattern(pattern) else event_catalog()
    out = []
    for name in pool:
        if regex is not None and regex.fullmatch(name):
            out.append(name)
            continue
        try:
            if fnmatch.fnmatchcase(name, pattern):
                out.append(name)
        except Exception:
            continue
    return out


def _is_bare_pattern(pattern):
    return bool(pattern) and not any(c.isalnum() for c in pattern)


def _split_list(values, expand=True):
    if values is None:
        return []
    if isinstance(values, str):
        if not values.strip():
            return []
        values = [values]
    elif isinstance(values, (list, tuple, set)):
        if len(values) == 0:
            return []
    out = []
    try:
        items = list(values)
    except TypeError:
        s = str(values).strip()
        if not s:
            return []
        return expand_event_aliases([s]) if expand else [s]
    for v in items:
        if v is None:
            continue
        if isinstance(v, str):
            out.extend([x.strip() for x in v.split(",") if x.strip()])
        else:
            try:
                subs = list(v)
            except TypeError:
                s = str(v).strip()
                if s:
                    out.append(s)
                continue
            for x in subs:
                out.extend([y.strip() for y in str(x).split(",") if y.strip()])
    return expand_event_aliases(out) if expand else out


def _split_groups(value, default, expand=True):
    def _names(names):
        return expand_event_aliases(names) if expand else list(names)

    if value is None:
        return [list(default)]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        parts = _names(parts)
        return [parts] if parts else [list(default)]
    if isinstance(value, (list, tuple)):
        groups = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (list, tuple)):
                names = []
                for sub in item:
                    if sub is None:
                        continue
                    if isinstance(sub, str):
                        names.extend([e.strip() for e in sub.split(",") if e.strip()])
                    elif str(sub).strip():
                        names.append(str(sub).strip())
                names = _names(names)
                if names:
                    groups.append(names)
            elif isinstance(item, str):
                parts = [p.strip() for p in item.split(",") if p.strip()]
                parts = _names(parts)
                if parts:
                    groups.append(parts)
            elif str(item).strip():
                groups.extend(_names([str(item).strip()]))
        return groups or [list(default)]
    s = str(value).strip()
    expanded = _names([s]) if s else []
    return [expanded] if expanded else [list(default)]


def _to_int_or(v, default):
    try:
        return int(v, 0) if isinstance(v, str) else int(v)
    except (TypeError, ValueError):
        return default


def _to_u64(v):
    return _to_int_or(v, 0) & 0xFFFFFFFFFFFFFFFF


def _is_duration_event(event):
    try:
        return str(event).strip() == _DURATION_TIME
    except Exception:
        return False


for _cle_logger in ("cle.loader", "cle.backends.tls.tls_object"):
    logging.getLogger(_cle_logger).addFilter(_SizelessImportFilter())
