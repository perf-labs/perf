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

import bisect
import contextlib
import ctypes
import errno
import fcntl
import functools
import hashlib
import logging
import mmap
import os
import select
import signal
import sys
import threading
import warnings

from .arch import arch as _get_arch, arch as get_arch

_DURATION_TIME = "duration_time"
_PLACEMENT_LOCK = threading.RLock()
_PERF_TYPE_HARDWARE = 0
_PERF_TYPE_SOFTWARE = 1
_PERF_TYPE_HW_CACHE = 3
_PERF_TYPE_RAW = 4
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
    "idle-cycles-frontend": 0x7,
    "stalled-cycles-backend": 0x8,
    "idle-cycles-backend": 0x8,
    "ref-cycles": 0x9,
    "ref-cpu-cycles": 0x9,
}
_HW_CACHE_EVENTS = {
    f"{cache}-{stem}{suffix}": cid | (op << 8) | (result << 16)
    for cache, cid, ops in (
        ("L1-dcache", 0x0, ("load", "store", "prefetch")),
        ("L1-icache", 0x1, ("load", "prefetch")),
        ("LLC", 0x2, ("load", "store", "prefetch")),
        ("dTLB", 0x3, ("load", "store", "prefetch")),
        ("iTLB", 0x4, ("load", "prefetch")),
        ("branch", 0x5, ("load",)),
        ("node", 0x6, ("load", "store", "prefetch")),
    )
    for op, singular, plural in (
        (0, "load", "loads"),
        (1, "store", "stores"),
        (2, "prefetch", "prefetches"),
    )
    if singular in ops
    for suffix, stem, result in (
        ("", plural, 0x0),
        ("-misses", singular, 0x1),
    )
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
_EXCLUDE_GUEST = 1 << 20
_EXCLUDE_HOST = 1 << 19
_PRECISE_IP = 0x3 << 15
_DEFAULT_FLAGS = _DISABLED | _EXCLUDE_KERNEL | _EXCLUDE_HV
_SYSFS_EVENT_SOURCES = "/sys/bus/event_source/devices"
_LEADER_FOR_PREFIX = {"topdown-": "slots"}
_RETRY_ERRNOS = (2, 19, 22, 95)
_SYSFS_NODES = "/sys/devices/system/node"
_NR_SET_MEMPOLICY = 238
_NR_GET_MEMPOLICY = 239
_MPOL_DEFAULT = 0
_MPOL_BIND = 2
_PATTERN_GLOB = "*?["
_PATTERN_REGEX = "()|+^${}\\"
_PATTERN_META = _PATTERN_GLOB + _PATTERN_REGEX
_MIN_VALUE_CAP = None
_MEASURE_ACTIVE = {}
_MEASURE_ACTIVE_LOCK = threading.RLock()
_OUTPUT_LOCK = threading.RLock()
_FORK_LOCK = threading.RLock()
_MAP_FIXED_NOREPLACE = 0x100000
_STATE_LOCK = threading.RLock()
_JIT_LOCK = threading.RLock()
_WARNINGS_LOCK = threading.RLock()
_POKE_LOCK = threading.RLock()
_JIT_PAGES = []
_JIT_PAGES_MAX = 32
_JIT_PAGES_IN_USE = set()
_MADV_NOHUGEPAGE = 15
_MAPPED_DATA_PAGES = set()
_UNMAPPABLE_DATA_PAGES = set()
_SIMULATED_RANGES = []
_ENTRY_DATA_BASE = 0x4200000000
_RUN_MEMORY_FRACTION = 0.5
_MEMORY_CHECK_EVERY = 4
_MEMORY_STATE = threading.local()
_ASM_SCRATCH_BASE = get_arch()._ASM_SCRATCH_BASE
_RUNNABLE_HARNESS = set()
_RUNNABLE_HARNESS_MAX = 4096
_PROBE_TIMEOUT = 10.0
_MODEL_PROBE_TIMEOUT = 0.5
_FATAL_SIGNALS = tuple(
    s
    for s in (
        signal.SIGSEGV,
        signal.SIGBUS,
        signal.SIGILL,
        signal.SIGFPE,
        signal.SIGABRT,
        signal.SIGTRAP,
        getattr(signal, "SIGSYS", None),
    )
    if s is not None
)
_FAULT_MESSAGES = {
    4: "an illegal instruction (SIGILL)",
    5: "a trap (SIGTRAP)",
    6: "an abort (SIGABRT)",
    7: "a bus error (SIGBUS)",
    8: "a divide error (SIGFPE)",
    11: "a segmentation fault (SIGSEGV)",
    31: "a bad system call (SIGSYS)",
}
_MMAP_ARGS = (
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
)
_MADVISE_ARGS = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
_NULL = "/dev/null"
_FALLBACK_DATA_BASE = 0x5200000000
_FALLBACK_DATA_STRIDE = 0x40000000
_FALLBACK_STEER_MAX = 1024 * 1024
_FALLBACK_START_PAGES = 1
_FALLBACK_MAX_DOUBLINGS = 20
_FALLBACK_GROWN_PAGES = {}
_INF = float("inf")


def serialized(lock):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with lock:
                return fn(*args, **kwargs)

        return wrapper

    return decorate


_locked = serialized(_STATE_LOCK)


class PerfCounter:
    SYS_PERF_EVENT_OPEN = _get_arch()._PERF_SYSCALL_NR
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
            ("config1", ctypes.c_uint64),
            ("config2", ctypes.c_uint64),
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

    def __init__(self, config, type=None, flags=None, pid=0, cpu=-1, group_fd=-1):
        self.fd = -1
        self.meta_mmap = None
        attr = self.perf_event_attr()

        attr.type = _PERF_TYPE_HARDWARE if type is None else type
        attr.size = ctypes.sizeof(attr)
        try:
            c0, c1, c2 = config
        except (TypeError, ValueError):
            c0, c1, c2 = config, 0, 0
        attr.config = int(c0) & 0xFFFFFFFFFFFFFFFF
        attr.config1 = int(c1) & 0xFFFFFFFFFFFFFFFF
        attr.config2 = int(c2) & 0xFFFFFFFFFFFFFFFF

        attr.flags = _DEFAULT_FLAGS if flags is None else flags

        self.fd = self.perf_event_open(attr, pid=pid, cpu=cpu, group_fd=group_fd)

        try:
            self.meta_mmap = mmap.mmap(
                self.fd,
                mmap.PAGESIZE,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ,
            )
        except BaseException:  # pragma: no cover
            try:
                fd = self.fd
            except Exception:
                fd = -1
            try:
                self.fd = -1
            except Exception:
                pass
            try:
                self.meta_mmap = None
            except Exception:
                pass
            try:
                if isinstance(fd, int) and fd >= 0:
                    os.close(fd)
            except Exception:
                pass
            raise

        self.meta = self.perf_event_mmap_page.from_buffer_copy(
            self.meta_mmap[: ctypes.sizeof(self.perf_event_mmap_page)]
        )

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        try:
            self.close()
        except Exception:
            pass
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def perf_event_open(self, attr, pid=0, cpu=-1, group_fd=-1, flags=0):
        fd = self.libc.syscall(
            self.SYS_PERF_EVENT_OPEN,
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
                "(event not scheduled or user-space rdpmc is disabled; run "
                "`perf system tune` or write 1 or 2 to /sys/devices/*/rdpmc)"
            )

        return self.meta.index - 1

    def close(self):
        mm = getattr(self, "meta_mmap", None)
        if mm is not None:
            try:
                self.meta_mmap = None
            except Exception:
                pass
            try:
                mm.close()
            except Exception:
                pass
        fd = getattr(self, "fd", -1)
        if isinstance(fd, int) and fd >= 0:
            try:
                self.fd = -1
            except Exception:
                pass
            try:
                os.close(fd)
            except OSError:
                pass


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


def min_int(state, expr, cap=_MIN_VALUE_CAP):
    if expr is None:
        return None
    value = concrete_int(expr)
    if value is not None:
        return value
    bits = _bv_bits(expr)
    if bits is None:
        return eval_int(state, expr)
    maximum = (1 << bits) - 1
    top = maximum if cap is None else min(int(cap), maximum)
    if top < 1:
        return eval_int(state, expr)
    try:
        hi = _any_in_range(state, expr, bits, 1, top)
        if hi is None:
            return eval_int(state, expr)
        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if _any_in_range(state, expr, bits, lo, mid) is not None:
                hi = mid
            else:
                lo = mid + 1
        return hi
    except Exception:
        return eval_int(state, expr)


def min_ints(state, exprs, cap=_MIN_VALUE_CAP):
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
            out[index] = min_int(state, expr, cap)
    for expr, indexes in pending.items():
        value = min_int(state, expr, cap)
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
        cpus = _parse_cpu_list(txt) if txt else []
        if cpus:
            return cpus
    return None


def pmu_cpus(typ):
    cpus = None
    try:
        cpus = cpus_for_type(typ)
    except Exception:
        cpus = None
    if cpus or typ not in (_PERF_TYPE_HARDWARE, _PERF_TYPE_HW_CACHE):
        return cpus
    for want in ("cpu_core", "cpu"):
        for dev, dev_type, _ev, _fmt in _sysfs_pmus():
            if dev == want:
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
    from .cfg import _thread_drop, _thread_store

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
    with _PLACEMENT_LOCK:
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
        with _PLACEMENT_LOCK:
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

    events = [] if events is None else list(events)
    counters, opened = [], {}
    leader_fd = -1
    leader_pos = None
    if group:
        leader_pos = next(
            (
                pos
                for pos, ev in enumerate(events)
                if not (isinstance(ev, str) and _is_duration_event(ev))
            ),
            None,
        )
    try:
        for pos, ev in enumerate(events):
            if isinstance(ev, str) and _is_duration_event(ev):
                continue
            if pos == leader_pos and group:
                c = (
                    open_event_on(ev, force_typ, pid=pid)
                    if force_typ is not None
                    else open_event(ev, pid=pid)
                )
                counters.append(c)
                opened[pos] = c
                if force_typ is None:
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
            c.enable()
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
                    "the event may not be scheduled or user-space rdpmc is "
                    "disabled (run `perf system tune` or write 1 or 2 to "
                    "/sys/devices/*/rdpmc)"
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


def refresh_indices(counters, events, indices=None):
    import time

    events = list(events or [])
    opened = [] if counters is None else list(counters)
    out = []
    pos = 0
    for i, ev in enumerate(events):
        if isinstance(ev, str) and _is_duration_event(ev):
            out.append(None)
            continue
        if pos >= len(opened):
            stale = None
            if indices is not None:
                try:
                    stale = indices[i]
                except Exception:
                    stale = None
            hint = f"; last known index was {stale!r}" if stale is not None else ""
            raise RuntimeError(
                f"no open perf counter for event {ev!r}{hint}; the counter "
                "set no longer matches the measured events"
            )
        c = opened[pos]
        pos += 1
        for _ in range(100):
            try:
                out.append(c.rdpmc_index)
                break
            except RuntimeError:
                time.sleep(0.001)
        else:
            raise RuntimeError(
                f"kernel did not expose an RDPMC index for event {ev!r}; "
                "the event may not be scheduled or user-space rdpmc is "
                "disabled (run `perf system tune` or write 1 or 2 to "
                "/sys/devices/*/rdpmc)"
            )
    assert_rdpmc_unique(events, out)
    return out


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
    try:
        c0, c1, c2 = config
    except (TypeError, ValueError):
        c0, c1, c2 = config, 0, 0
    attr.config = int(c0) & 0xFFFFFFFFFFFFFFFF
    attr.config1 = int(c1) & 0xFFFFFFFFFFFFFFFF
    attr.config2 = int(c2) & 0xFFFFFFFFFFFFFFFF
    attr.flags = int(flags) & 0xFFFFFFFFFFFFFFFF
    return bytes(ctypes.string_at(ctypes.byref(attr), ctypes.sizeof(attr)))


def remote_open_self_counters(pid, ev_list, force_typ, group, anchor):
    from .prof import remote_mmap, remote_syscall, signed_rax, write_mem

    _arch = _get_arch()
    ev_list = [] if ev_list is None else list(ev_list)

    plan = []
    _ft = force_typ
    leader_pos = None
    if group:
        leader_pos = next(
            (
                pos
                for pos, ev in enumerate(ev_list)
                if not (isinstance(ev, str) and _is_duration_event(ev))
            ),
            None,
        )
    for pos, ev in enumerate(ev_list):
        if isinstance(ev, str) and _is_duration_event(ev):
            continue
        if pos == leader_pos and group:
            if _ft is not None:
                try:
                    cands = [resolve_on(ev, _ft)]
                except ValueError:
                    cands = list(_candidates(ev))
            else:
                cands = list(_candidates(ev))
            plan.append((pos, ev, cands))
            if _ft is None:
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
    for _pos, ev, cands in plan:
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
            if (-signed) not in _RETRY_ERRNOS:
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
        remote_open_self_counters(pid, ev_list, force_typ, group, anchor)
        proxy_indices = refresh_indices(proxy_counters, ev_list, proxy_indices)
    finally:
        for c in proxy_counters:
            try:
                try:
                    c.disable()
                finally:
                    c.close()
            except Exception:
                pass
    return [], proxy_indices


def allowed_cpus(groups, numa=None):
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
        ordered = sorted(allowed)
        return ordered if ordered else None
    if numa is not None:
        cpu = _numa_first_cpu(numa)
        if cpu is not None:
            return [cpu]
    cpus = _current_affinity_list()
    return list(cpus) if cpus else [0]


@functools.cache
def supported_events():
    names = []
    seen = set()
    for source in (
        sorted(_HW_EVENTS),
        sorted(_HW_CACHE_EVENTS),
        sorted(_SW_EVENTS),
    ):
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
        return []
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


def _bv_bits(expr):
    try:
        bits = int(expr.size())
    except (AttributeError, TypeError, ValueError):
        return None
    return bits if 0 < bits <= 64 else None


def _any_in_range(state, expr, bits, lo, hi):
    try:
        import claripy
    except ImportError:
        return None
    window = (
        claripy.UGE(expr, claripy.BVV(lo, bits)),
        claripy.ULE(expr, claripy.BVV(hi, bits)),
    )
    try:
        return int(state.solver.eval(expr, extra_constraints=window))
    except Exception:
        return None


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
        flags = (flags & ~_PRECISE_IP) | (min(precise, 3) << 15)
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
        if not bits and key not in words:
            continue
        try:
            value = int(value, 0)
        except (TypeError, ValueError):
            try:
                value = int(value, 10)
            except (TypeError, ValueError):
                raise ValueError(
                    f"invalid value {value!r} for event field {key!r}; "
                    "expected an integer like event=0x3c"
                ) from None
        if not bits:
            if key in words:
                words[key] |= value & 0xFFFFFFFFFFFFFFFF
            continue
        for word, lo, hi in _parse_bitspec(bits):
            if word not in words:
                continue
            width = hi - lo + 1
            words[word] |= (value & ((1 << width) - 1)) << lo
    if words["config1"] or words["config2"]:
        return (words["config"], words["config1"], words["config2"])
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
            for name in sorted(os.listdir(evdir)):
                if "." in name:
                    continue
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


def _pmu_for_type(typ):
    for dev, t, _events, _formats in _sysfs_pmus():
        if t == typ:
            return dev
    return None


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


def _sysfs_event_config(events, name):
    keys = (str(name), _norm(name))
    for key in keys:
        if key in events:
            return True, events[key]
    for key, config in events.items():
        if _norm(key) in keys:
            return True, config
    return False, None


@functools.lru_cache(maxsize=4096)
def _candidates(event):
    if isinstance(event, bool):
        raise ValueError(f"invalid event spec {event!r}")
    if isinstance(event, int):
        if not 0 <= event <= 9:
            raise ValueError(
                f"unknown event {event!r}; hardware events are 0..9 "
                "(e.g. 0 is cycles, 1 is instructions); try 'cycles', "
                "'instructions', 'cache-misses', 'branch-instructions', "
                "'duration_time' or run `perf list` for more"
            )
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
    if key in _HW_CACHE_EVENTS:
        return [(_PERF_TYPE_HW_CACHE, _HW_CACHE_EVENTS[key], flags)]
    if key in _SW_EVENTS:
        return [(_PERF_TYPE_SOFTWARE, _SW_EVENTS[key], flags)]
    lower = key.lower()
    if lower.startswith("r") and len(lower) > 1:
        try:
            return [(_PERF_TYPE_RAW, int(lower[1:], 16), flags)]
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
            found, config = _sysfs_event_config(events, rest)
            if found:
                return [(typ, config, flags)]
            lower_rest = _norm(rest).lower()
            if lower_rest.startswith("r") and len(lower_rest) > 1:
                try:
                    return [(typ, int(lower_rest[1:], 16), flags)]
                except ValueError:
                    pass
            rest_key = _norm(rest)
            if rest_key in _HW_EVENTS:
                return [(_PERF_TYPE_HARDWARE, _HW_EVENTS[rest_key], flags)]
            if rest_key in _HW_CACHE_EVENTS:
                return [(_PERF_TYPE_HW_CACHE, _HW_CACHE_EVENTS[rest_key], flags)]
            if rest_key in _SW_EVENTS:
                return [(_PERF_TYPE_SOFTWARE, _SW_EVENTS[rest_key], flags)]
        raise ValueError(f"unknown event {event!r}: PMU {dev!r} exports no {rest!r}")
    out = []
    for _, typ, events, _ in _sysfs_pmus():
        found, config = _sysfs_event_config(events, name)
        if found:
            out.append((typ, config, flags))
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
def _parse_cpu_list_cached(s):
    return tuple(sorted(_parse_range_list(s)))


def _parse_cpu_list(s):
    return list(_parse_cpu_list_cached(s))


@functools.lru_cache(maxsize=4096)
def _parse_cpu_bitmap_cached(s):
    cpus = set()
    words = [word.strip() for word in str(s).split(",") if word.strip()]
    for index, word in enumerate(reversed(words)):
        try:
            value = int(word, 16)
        except (TypeError, ValueError):
            continue
        bit = 0
        while value:
            if value & 1:
                cpus.add(index * 32 + bit)
            value >>= 1
            bit += 1
    return tuple(sorted(cpus))


def _parse_cpu_bitmap(s):
    return list(_parse_cpu_bitmap_cached(s))


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
    return mask, node + 1


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


@functools.lru_cache(maxsize=512)
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
        if isinstance(values, int) and not isinstance(values, bool):
            return expand_event_aliases([values]) if expand else [values]
        s = str(values).strip()
        if not s:
            return []
        return expand_event_aliases([s]) if expand else [s]
    for v in items:
        if v is None:
            continue
        if isinstance(v, str):
            out.extend([x.strip() for x in v.split(",") if x.strip()])
        elif isinstance(v, int) and not isinstance(v, bool):
            out.append(v)
        else:
            try:
                subs = list(v)
            except TypeError:  # pragma: no cover
                s = str(v).strip()
                if s:
                    out.append(s)
                continue
            for x in subs:
                if x is None:
                    continue
                if isinstance(x, str):
                    out.extend([y.strip() for y in x.split(",") if y.strip()])
                elif isinstance(x, int) and not isinstance(x, bool):
                    out.append(x)
                elif str(x).strip():
                    out.append(str(x).strip())
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
                    elif isinstance(sub, int) and not isinstance(sub, bool):
                        names.append(sub)
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
            elif isinstance(item, int) and not isinstance(item, bool):
                groups.append(_names([item]))
            elif str(item).strip():
                groups.append(_names([str(item).strip()]))
        return groups or [list(default)]
    if isinstance(value, int) and not isinstance(value, bool):
        return [_names([value])]
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
        if not isinstance(event, str):
            return False
        base, _ = _split_modifiers(event)
        return base.strip() == _DURATION_TIME
    except Exception:
        return False


for _cle_logger in ("cle.loader", "cle.backends.tls.tls_object"):
    logging.getLogger(_cle_logger).addFilter(_SizelessImportFilter())


@contextlib.contextmanager
def _ignore_runtime_warnings():
    with _WARNINGS_LOCK:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            yield


def _memory_snapshot():
    state = getattr(_MEMORY_STATE, "value", None) or {}
    try:
        return (float(state.get("per_call", 0.0)), int(state.get("calls", 0)))
    except (TypeError, ValueError):
        return (0.0, 0)


def _memory_reset():
    _MEMORY_STATE.value = {"calls": 0, "per_call": 0.0}
    return _MEMORY_STATE.value


def _memory_record(calls, per_call):
    state = getattr(_MEMORY_STATE, "value", None)
    if state is None:
        state = _memory_reset()
    try:
        state["calls"] = int(state.get("calls", 0)) + int(calls)
        state["per_call"] = max(
            float(state.get("per_call", 0.0) or 0.0), float(per_call or 0.0)
        )
    except (TypeError, ValueError):
        pass
    return state


def _measure_threads_enter():
    tid = threading.get_ident()
    with _MEASURE_ACTIVE_LOCK:
        _MEASURE_ACTIVE[tid] = _MEASURE_ACTIVE.get(tid, 0) + 1


def _measure_threads_exit():
    tid = threading.get_ident()
    with _MEASURE_ACTIVE_LOCK:
        depth = _MEASURE_ACTIVE.get(tid, 0)
        if depth <= 1:
            _MEASURE_ACTIVE.pop(tid, None)
        else:
            _MEASURE_ACTIVE[tid] = depth - 1


def _measure_threads_count():
    with _MEASURE_ACTIVE_LOCK:
        return len(_MEASURE_ACTIVE)


def _measure_threads_forget(threads):
    idents = [getattr(thread, "ident", None) for thread in threads or ()]
    with _MEASURE_ACTIVE_LOCK:
        for ident in idents:
            if ident is not None:
                _MEASURE_ACTIVE.pop(ident, None)


def _projected_memory_growth(iterations):
    per_call, calls_seen = _memory_snapshot()
    try:
        threads = max(1, _measure_threads_count())
    except Exception:
        threads = 1
    try:
        calls = max(0, int(iterations))
    except (TypeError, ValueError):
        calls = 0
    return per_call * calls * threads, per_call, threads, calls_seen


def _sink_fd():
    return os.open(_NULL, os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))


def _json_canonical(value):
    if isinstance(value, dict):
        return {str(key): _json_canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_canonical(item) for item in value), key=repr)
    return value


def _stable(value):
    from .data import dumps_json

    try:
        return dumps_json(value, sort_keys=True)
    except Exception:
        try:
            return dumps_json(_json_canonical(value), sort_keys=True)
        except Exception:
            return repr(value)


@functools.lru_cache(maxsize=256)
def _hashed_content(path, _mtime_ns, _size):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _content_hash8(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    try:
        return _hashed_content(str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _debug_log(msg):
    try:
        sys.stderr.write(f"[perf debug] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _poke_addr_safe(obj, patch_addr):
    try:
        addr = int(patch_addr)
    except (TypeError, ValueError):
        return False
    try:
        ranges = obj.segment_runtime_ranges()
    except Exception:
        return False
    try:
        for _v, msz, rt, _fl in ranges:
            if int(rt) <= addr < int(rt) + int(msz):
                return True
    except (TypeError, ValueError):
        return False
    return False


class _Args(ctypes.Structure):
    _fields_ = [
        ("iterations", ctypes.c_uint64),
        ("inputs", ctypes.c_void_p),
        ("outputs", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
    ]


def _min_mmap_addr():
    try:
        with open("/proc/sys/vm/mmap_min_addr") as fh:
            return int(fh.read().strip(), 0)
    except (OSError, ValueError):
        return 0x1000


@_locked
def _set_simulated_ranges(loader, keep=None):
    global _SIMULATED_RANGES
    ranges = []
    for obj in getattr(loader, "all_objects", None) or ():
        if keep is not None and obj is keep:
            continue
        try:
            lo = int(obj.min_addr)
            hi = int(obj.max_addr)
        except Exception:
            continue
        if hi > lo:
            ranges.append((lo, hi))
    _SIMULATED_RANGES = sorted(ranges)
    return _SIMULATED_RANGES


def _drop_unmappable_models(models):
    for m in models:
        m["reads"] = [t for t in m["reads"] if _page_mappable(t[0])]
        m["writes"] = [t for t in m["writes"] if _page_mappable(t[0])]
    return models


def _page_mappable(addr):
    try:
        parsed = int(addr, 0) if isinstance(addr, str) else int(addr)
    except (TypeError, ValueError):
        return False
    with _STATE_LOCK:
        return (parsed & ~(mmap.PAGESIZE - 1)) not in _UNMAPPABLE_DATA_PAGES


def _map_scratch_reg_pages(models):
    lo = _ASM_SCRATCH_BASE & ~(mmap.PAGESIZE - 1)
    hi = lo
    for m in models or ():
        for value in (m.get("regs") or {}).values():
            for item in value if isinstance(value, (list, tuple)) else [value]:
                try:
                    addr = int(item, 0) if isinstance(item, str) else int(item)
                except (TypeError, ValueError):
                    continue
                if lo <= addr < lo + 0x100000:
                    page = addr & ~(mmap.PAGESIZE - 1)
                    hi = max(hi, page + mmap.PAGESIZE)
    if hi <= lo:
        return
    _map_data_pages({"mem": {str(a): 0 for a in range(lo, hi, mmap.PAGESIZE)}})


@functools.cache
def _libc(name, restype, argtypes):
    fn = getattr(ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True), name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


def _discard_buffered_output():
    try:
        _libc("fflush", ctypes.c_int, (ctypes.c_void_p,))(None)
    except Exception:
        pass


def _silence_fd(fd):
    try:
        saved = os.dup(fd)
    except OSError:
        return None
    try:
        sink = _sink_fd()
    except OSError:  # pragma: no cover
        os.close(saved)
        return None
    try:
        os.dup2(sink, fd)
    except OSError:  # pragma: no cover
        os.close(saved)
        os.close(sink)
        return None
    os.close(sink)

    def _restore():
        _discard_buffered_output()
        try:
            os.dup2(saved, fd)
        except OSError:
            pass
        try:
            os.close(saved)
        except OSError:
            pass

    return _restore


def _mmap_fixed():
    return _libc("mmap", ctypes.c_void_p, _MMAP_ARGS)


def _madvise(addr, size, advice):
    _libc("madvise", ctypes.c_int, _MADVISE_ARGS)(
        ctypes.c_void_p(addr), ctypes.c_size_t(size), advice
    )


def _map_range(addr, size):
    ctypes.set_errno(0)
    mapped = _mmap_fixed()(
        ctypes.c_void_p(addr),
        ctypes.c_size_t(size),
        mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
        mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_FIXED_NOREPLACE,
        -1,
        0,
    )
    errno = ctypes.get_errno()
    if mapped == ctypes.c_void_p(-1).value or mapped != addr:
        return False, errno, mapped
    return True, 0, addr


@_locked
def _map_data_pages(data, strict=True):
    from .cfg import _iter_mem_addrs

    page_size = mmap.PAGESIZE
    pages = sorted({int(a) & ~(page_size - 1) for a in _iter_mem_addrs(data)})
    if not pages:
        return
    pending = [
        p
        for p in pages
        if p not in _MAPPED_DATA_PAGES and p not in _UNMAPPABLE_DATA_PAGES
    ]
    for page, count in get_arch().page_runs(pending):
        stop = page + count * page_size
        if count > 1 and _map_range(page, count * page_size)[0]:
            _madvise(page, count * page_size, _MADV_NOHUGEPAGE)
            _MAPPED_DATA_PAGES.update(range(page, stop, page_size))
            continue
        lo = bisect.bisect_left(pending, page)
        hi = bisect.bisect_left(pending, stop)
        for addr in pending[lo:hi]:
            if addr in _MAPPED_DATA_PAGES or addr in _UNMAPPABLE_DATA_PAGES:
                continue
            ok, errno, mapped = _map_range(addr, page_size)
            if ok:
                _MAPPED_DATA_PAGES.add(addr)
                continue
            if not strict:
                _UNMAPPABLE_DATA_PAGES.add(addr)
                continue
            if mapped == ctypes.c_void_p(-1).value:
                raise OSError(
                    errno,
                    f"failed to map data page at 0x{addr:x}; "
                    "address already mapped, choose a different data address",
                )
            raise OSError(
                errno,
                f"failed to map data page at 0x{addr:x}; got 0x{mapped:x} instead",
            )
        _UNMAPPABLE_DATA_PAGES.update(
            addr
            for addr in pending[lo:hi]
            if addr not in _MAPPED_DATA_PAGES and addr not in _UNMAPPABLE_DATA_PAGES
        )


def _fallback_budget_bytes():
    try:
        budget = int(_available_memory() * _RUN_MEMORY_FRACTION)
    except Exception:
        budget = 0
    if not budget:
        return None
    try:
        used = int(_run_memory())
    except Exception:
        used = 0
    return max(0, budget - used)


def _reset_fatal_handlers():
    seen = set()
    for signum in list(_FATAL_SIGNALS) + list(range(1, 32)):
        if signum in seen:
            continue
        seen.add(signum)
        if signum in (signal.SIGKILL, signal.SIGSTOP):
            continue
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (OSError, ValueError, RuntimeError):
            continue


def _available_memory():
    try:
        with open("/proc/meminfo", "rb") as f:
            for line in f:
                if line.startswith(b"MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _run_memory(pid="self"):
    try:
        with open(f"/proc/{pid}/statm", "rb") as f:
            return int(f.read().split()[0]) * mmap.PAGESIZE
    except (OSError, IndexError, ValueError):
        pass
    try:
        with open(f"/proc/{pid}/status", "rb") as f:
            for line in f:
                if line.startswith(b"VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _check_growth(iterations, map_name=None, index=None, used=None, budget=None):
    if budget is None:
        budget = int(_available_memory() * _RUN_MEMORY_FRACTION)
    if not budget:
        return
    if used is None:
        used = _run_memory()
    projected, per_call, threads, calls_seen = _projected_memory_growth(iterations)
    left = budget - used
    if used + projected <= budget:
        return
    per_call = per_call * threads
    fits = int(left // per_call) if per_call > 0 and left > 0 else 0
    what = f" {map_name!r}" if map_name else " the target"
    cost = (
        f"it holds about {per_call / 1024:.0f} KiB per call over the "
        f"{calls_seen} calls made so far"
        if per_call > 0
        else "it does not give memory back"
    )
    raise MemoryError(
        f"the measurement harness is at {used / 2**30:.1f} GiB of address space "
        f"and perf measures in this process: {cost}, so run {index} of "
        f"{int(iterations)} iterations would need {projected / 2**30:.1f} GiB "
        f"of the {budget / 2**30:.1f} GiB available"
        + (f", which leaves room for about {fits} more calls" if fits else "")
        + f". Benchmark{what} with fewer iterations (--config.iterations) or "
        "samples, or a target that does not allocate per call."
    )


def _is_runnable(key):
    with _STATE_LOCK:
        return key in _RUNNABLE_HARNESS


def _remember_runnable(key):
    with _STATE_LOCK:
        if len(_RUNNABLE_HARNESS) >= _RUNNABLE_HARNESS_MAX:
            _RUNNABLE_HARNESS.clear()
        _RUNNABLE_HARNESS.add(key)


def _check_target_runnable(fn, args, map_name=None, key=None, config=None):
    with _STATE_LOCK:
        if key is not None and _is_runnable(key):
            return
        with _OUTPUT_LOCK, _FORK_LOCK:
            if key is not None and _is_runnable(key):
                return
            result = _check_target_runnable_locked(fn, args, map_name, config)
        if key is not None:
            _remember_runnable(key)
    return result


def _check_target_runnable_locked(fn, args, map_name=None, config=None):
    from .cfg import _fake_output

    trips = 1
    probe = _Args(trips, args.inputs, args.outputs, args.data)
    sys.stdout.flush()
    sys.stderr.flush()
    where = f" '{map_name}'" if map_name else "the target"
    with _fake_output(config):
        read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
        try:
            pid = os.fork()
        except Exception:
            os.close(read_fd)
            os.close(write_fd)
            raise
        if pid == 0:
            _reset_fatal_handlers()
            try:
                os.close(read_fd)
            except OSError:
                os._exit(1)
            try:
                fn(ctypes.byref(probe))
            except BaseException:
                os._exit(1)
            try:
                os.write(write_fd, b"1")
            except OSError:
                pass
            os._exit(0)

        os.close(write_fd)
        try:
            ready, _w, _x = select.select([read_fd], [], [], _PROBE_TIMEOUT)
            returned = bool(ready) and os.read(read_fd, 1) == b"1"
        except OSError:
            returned = False
        finally:
            os.close(read_fd)
            if not returned:
                _kill(pid)
        status = _waitpid(pid)[1]
    if os.WIFSIGNALED(status) and os.WTERMSIG(status) != signal.SIGKILL:
        signum = os.WTERMSIG(status)
        what = _FAULT_MESSAGES.get(signum, f"signal {signum}")
        raise ValueError(
            f"{where} faulted with {what} when called on its own, so it "
            "cannot be measured in isolation. Benchmark a symbol that runs "
            "without the program's setup, or restrict the benchmark to a "
            "region (e.g. 'foo_begin..foo_end')."
        )
    if not returned:
        raise ValueError(
            f"{where} did not return from {trips} call(s) on its own within "
            f"{_PROBE_TIMEOUT:g}s: it never comes back, exits or replaces the "
            "process, or runs on state the harness does not own. Benchmark a "
            "symbol that returns to its caller, or restrict the benchmark to "
            "a region (e.g. 'foo_begin..foo_end')."
        )
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) != 0:
        raise ValueError(f"the measurement harness raised while running{where}.")


def _kill(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _waitpid(pid):
    while True:
        try:
            return os.waitpid(pid, 0)
        except OSError as ex:  # pragma: no cover
            if ex.errno == errno.EINTR:
                continue
            raise


def _evict_jit_pages_locked():
    while len(_JIT_PAGES) > _JIT_PAGES_MAX:
        old = next(
            (page for page in _JIT_PAGES if id(page) not in _JIT_PAGES_IN_USE),
            None,
        )
        if old is None:
            break
        try:
            _JIT_PAGES.remove(old)
        except ValueError:
            continue
        try:
            old.close()
        except Exception:
            pass


def _jit_page(code_bytes, in_use=False):
    page = mmap.mmap(
        -1,
        len(code_bytes),
        prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
    )
    page.write(bytes(code_bytes))
    with _JIT_LOCK:
        _JIT_PAGES.append(page)
        if in_use:
            _JIT_PAGES_IN_USE.add(id(page))
        _evict_jit_pages_locked()
    return page


def _jit_release(page):
    with _JIT_LOCK:
        _JIT_PAGES_IN_USE.discard(id(page))
        _evict_jit_pages_locked()


def _read_raw_counters(counters):
    for counter in counters or ():
        try:
            counter.disable()
        except Exception:
            pass
    values = []
    for counter in counters or ():
        try:
            values.append(int.from_bytes(os.read(counter.fd, 8), "little"))
        except Exception:
            values.append(None)
    return values
