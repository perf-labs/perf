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
import mmap
import numbers
import os
import warnings

import claripy

from .arch import arch as _get_arch

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
    "duration_time": 0x9,
}
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


class PerfCounter:
    SYS_perf_event_open = _get_arch().PERF_SYSCALL_NR
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

        self.meta_mmap = mmap.mmap(
            self.fd,
            mmap.PAGESIZE,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ,
        )

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


def demangle(name):
    if not name:
        return name
    s = str(name)
    if "_Z" not in s:
        return s.replace("\r", " ").replace("\n", " ")
    try:
        import cxxfilt as _cxxfilt

        out = _cxxfilt.demangle(s)
        if out and out != s:
            return str(out).replace("\r", " ").replace("\n", " ")
    except Exception:
        pass
    return s.replace("\r", " ").replace("\n", " ")


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


def assert_rdpmc_unique(events, indices):
    real = [i for i in indices if i is not None]
    if len(real) != len(set(real)):
        raise RuntimeError(
            f"events {list(events)!r} alias to the same RDPMC index {real}; "
            "this PMU's grouped events cannot be read via RDPMC. Use the "
            "standalone PMU instead (e.g. cpu_atom/topdown-.../ on hybrid) "
            "or `perf stat`"
        )


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


def is_group(events):
    if len(events) < 2:
        return False
    first = _event_base(events[0])
    if not first or first.strip() == "duration_time":
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
    try:
        pmu_cpus = cpus_for_type(typ)
    except Exception:
        pmu_cpus = None
    if not pmu_cpus:
        return lambda: None
    req = requested
    if req is None:
        try:
            cur = sorted(os.sched_getaffinity(0))
        except OSError:
            cur = None
        avail = [c for c in pmu_cpus if cur is None or c in cur]
    else:
        avail = [c for c in req if c in pmu_cpus]
    if not avail:
        raise RuntimeError(
            f"events require PMU CPUs {pmu_cpus} (PMU type {typ}) but affinity "
            f"is {req}; pin to a matching CPU or use the matching PMU's events"
        )
    try:
        old = sorted(os.sched_getaffinity(0))
    except OSError:
        old = None
    thread = config.get("thread") if isinstance(config, dict) else None
    old_cfg, touched, created = None, False, False
    if old is not None:
        os.sched_setaffinity(0, set(avail))
    if isinstance(thread, dict):
        old_cfg, touched = thread.get("affinity", None), True
        thread["affinity"] = list(avail)
    elif isinstance(config, dict) and config.get("thread") is None:
        config["thread"] = {"affinity": list(avail)}
        touched, created = True, True

    def _restore():
        if old is not None:
            try:
                os.sched_setaffinity(0, set(old))
            except OSError:
                pass
        if (
            touched
            and isinstance(config, dict)
            and isinstance(config.get("thread"), dict)
        ):
            try:
                if created and old_cfg is None:
                    if set(config["thread"].keys()) == {"affinity"}:
                        config["thread"] = None
                    else:
                        del config["thread"]["affinity"]
                else:
                    config["thread"]["affinity"] = old_cfg
            except Exception:
                pass

    return _restore


def open_counters(events, force_typ=None, group=False):
    import time

    counters, opened = [], {}
    leader_fd = -1
    try:
        for pos, ev in enumerate(events):
            if isinstance(ev, str) and _is_duration_event(ev):
                continue
            if pos == 0 and group:
                c = (
                    open_event_on(ev, force_typ)
                    if force_typ is not None
                    else open_event(ev)
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
                    c = open_event_on(ev, force_typ, group_fd=leader_fd)
                except ValueError:
                    c = open_event(ev, group_fd=leader_fd)
            else:
                c = open_event(ev, group_fd=leader_fd)
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
        try:
            arg_regs = list(getattr(_get_arch(), "ARG_REGS", []) or [])
        except Exception:
            arg_regs = []
        regs = (data or {}).get("regs") or {}
        canon_regs = set()
        for k in regs:
            try:
                fn = getattr(_get_arch(), "canonical_data_reg", None)
                if fn is not None:
                    canon_regs.add(fn(k))
                else:
                    canon_regs.add(str(k).strip().lower())
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


def _split_modifiers(name):
    if ":" not in name:
        return name.strip(), ""
    head, _, rest = name.partition(":")
    return head.strip(), rest.replace(":", "")


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


def _parse_spec(spec):
    attrs = {}
    for part in spec.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        attrs[key.strip()] = value.strip()
    return attrs


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


def _norm(name):
    return str(name).replace("_", "-").strip()


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


def _leader_for(base):
    base = _norm(base)
    for prefix, leader in _LEADER_FOR_PREFIX.items():
        if base.startswith(prefix):
            return leader
    return None


def _candidates(event):
    if isinstance(event, int):
        return [(_PERF_TYPE_HARDWARE, event, _DEFAULT_FLAGS)]
    name, mods = _split_modifiers(event)
    if not name:
        raise ValueError(f"invalid event spec {event!r}")
    flags = _modifier_flags(mods)
    if name.strip() == "duration_time":
        return [(_PERF_TYPE_SOFTWARE, _SW_EVENTS["duration_time"], flags)]
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
    raise ValueError(
        f"unknown event {event!r}; try 'cycles', 'instructions', 'cache-misses', "
        f"'branch-instructions', 'duration_time' or run `perf list` for more"
    )


def _make_counter(config, typ, flags, pid=0, group_fd=-1):
    return PerfCounter(config, type=typ, flags=flags, pid=pid, group_fd=group_fd)


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


def _thread_cfg(config):
    thread = (config or {}).get("thread", None)
    if thread is None:
        return {}
    if not isinstance(thread, dict):
        raise ValueError(
            f"unknown thread config {thread!r}; expected a dict like "
            "{'affinity': [1], 'priority': 1}"
        )
    return thread


def _affinity_spec(config):
    return _thread_cfg(config).get("affinity", None)


def _priority_spec(config):
    return _thread_cfg(config).get("priority", None)


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
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        warnings.warn("affinity requested but os.sched_setaffinity is unavailable")
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
    if hasattr(os, "getpriority") and hasattr(os, "setpriority"):
        try:
            old_nice = os.getpriority(os.PRIO_PROCESS, 0)
        except OSError:
            old_nice = None
    try:
        if hasattr(os, "setpriority"):
            try:
                os.setpriority(os.PRIO_PROCESS, 0, int(nice))
            except OSError as ex:
                warnings.warn(f"failed to set nice to {nice}: {ex}")
        else:
            try:
                cur = os.nice(0)
                os.nice(int(nice) - cur)
            except OSError as ex:
                warnings.warn(f"failed to set nice to {nice}: {ex}")
    except ValueError:
        raise
    except OSError as ex:
        warnings.warn(f"failed to set priority {nice}: {ex}")
    try:
        yield
    finally:
        if old_nice is not None and hasattr(os, "setpriority"):
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
        if hasattr(os, "getpriority"):
            return int(os.getpriority(os.PRIO_PROCESS, 0))
    except Exception:
        pass
    return 0


def _split_list(values):
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
        return [s] if s else []
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
    return out


def _split_groups(value, default):
    if value is None:
        return [list(default)]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
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
                if names:
                    groups.append(names)
            elif isinstance(item, str):
                parts = [p.strip() for p in item.split(",") if p.strip()]
                if parts:
                    groups.append(parts)
            elif str(item).strip():
                groups.append([str(item).strip()])
        return groups or [list(default)]
    s = str(value).strip()
    return [[s]] if s else [list(default)]


def _to_int_or(v, default):
    try:
        return int(v, 0) if isinstance(v, str) else int(v)
    except (TypeError, ValueError):
        return default


def _to_u64(v):
    return _to_int_or(v, 0) & 0xFFFFFFFFFFFFFFFF


def _is_duration_event(event):
    try:
        return str(event).strip() == "duration_time"
    except Exception:
        return False


def _vex_bv_size(bv):
    try:
        return int(bv.size())
    except Exception:
        try:
            return int(len(bv))
        except Exception:
            return 64


def _vex_concrete_bvv(bv):
    try:
        if getattr(bv, "op", None) == "BVV":
            return int(bv.args[0])
    except Exception:
        pass
    return None


def _vex_to_bv(val, size):
    if isinstance(val, int):
        size = int(size)
        if size < 256:
            val &= (1 << size) - 1
        return claripy.BVV(int(val), size)
    return val


def _vex_resize(bv, w):
    try:
        cur = _vex_bv_size(bv)
    except Exception:
        return bv
    if cur == w:
        return bv
    if cur > w:
        return claripy.Extract(w - 1, 0, bv)
    try:
        return bv.zero_extend(w - cur)
    except Exception:
        return claripy.ZeroExt(w - cur, bv)


def _vex_concat_lsb_first(bits):
    if not bits:
        return claripy.BVV(0, 1)
    if len(bits) == 1:
        return bits[0]
    return claripy.Concat(*reversed(bits))


def _vex_pext_concrete_mask(src, mask_val):
    w = _vex_bv_size(src)
    mask_val &= (1 << w) - 1 if w < 1024 else mask_val
    positions = [j for j in range(w) if (mask_val >> j) & 1]
    bits = []
    for k in range(w):
        if k < len(positions):
            bits.append(claripy.Extract(positions[k], positions[k], src))
        else:
            bits.append(claripy.BVV(0, 1))
    return _vex_concat_lsb_first(bits)


def _vex_pdep_concrete_mask(src, mask_val):
    w = _vex_bv_size(src)
    mask_val &= (1 << w) - 1 if w < 1024 else mask_val
    positions = [j for j in range(w) if (mask_val >> j) & 1]
    pos_to_k = {p: k for k, p in enumerate(positions)}
    bits = []
    for j in range(w):
        if j in pos_to_k:
            bits.append(claripy.Extract(pos_to_k[j], pos_to_k[j], src))
        else:
            bits.append(claripy.BVV(0, 1))
    return _vex_concat_lsb_first(bits)


def _vex_popcounts_low(mask_bits):
    pops = [claripy.BVV(0, 8)]
    for b in mask_bits:
        try:
            ext = b.zero_extend(7)
        except Exception:
            ext = claripy.ZeroExt(7, b)
        pops.append(pops[-1] + ext)
    return pops


def _vex_pext_symbolic_mask(src, mask):
    w = _vex_bv_size(src)
    try:
        src_bits = [claripy.Extract(i, i, src) for i in range(w)]
        mask_bits = [claripy.Extract(i, i, mask) for i in range(w)]
    except Exception:
        return claripy.BVV(0, w)
    pops = _vex_popcounts_low(mask_bits)
    one = claripy.BVV(1, 1)
    dst_bits = []
    for k in range(w):
        k8 = claripy.BVV(k, 8)
        acc = claripy.BVV(0, 1)
        for j in range(w):
            try:
                cond = claripy.And(mask_bits[j] == one, pops[j] == k8)
            except Exception:
                continue
            try:
                acc = claripy.If(cond, src_bits[j], acc)
            except Exception:
                continue
        dst_bits.append(acc)
    return _vex_concat_lsb_first(dst_bits)


def _vex_pdep_symbolic_mask(src, mask):
    w = _vex_bv_size(src)
    try:
        src_bits = [claripy.Extract(i, i, src) for i in range(w)]
        mask_bits = [claripy.Extract(i, i, mask) for i in range(w)]
    except Exception:
        return claripy.BVV(0, w)
    pops = _vex_popcounts_low(mask_bits)
    one = claripy.BVV(1, 1)
    zero = claripy.BVV(0, 1)
    dst_bits = []
    for j in range(w):
        sel = claripy.BVV(0, 1)
        for k in range(w):
            try:
                sel = claripy.If(pops[j] == claripy.BVV(k, 8), src_bits[k], sel)
            except Exception:
                continue
        try:
            dst_bits.append(claripy.If(mask_bits[j] == one, sel, zero))
        except Exception:
            dst_bits.append(zero)
    return _vex_concat_lsb_first(dst_bits)


def _vex_pext_impl(state, src, mask):
    del state
    try:
        w_src = _vex_bv_size(src)
    except Exception:
        w_src = 64
    try:
        w_mask = _vex_bv_size(mask)
    except Exception:
        w_mask = w_src
    w = max(int(w_src), int(w_mask))
    try:
        src = _vex_to_bv(src, w_src if not isinstance(src, int) else w)
        mask = _vex_to_bv(mask, w_mask if not isinstance(mask, int) else w)
        src = _vex_resize(src, w)
        mask = _vex_resize(mask, w)
    except Exception:
        pass
    try:
        mv = _vex_concrete_bvv(mask)
    except Exception:
        mv = None
    if mv is not None:
        try:
            return _vex_pext_concrete_mask(src, mv)
        except Exception:
            pass

    try:
        sv = _vex_concrete_bvv(src)
        if sv is not None and mv is not None:
            out = 0
            bit = 0
            for j in range(w):
                if (mv >> j) & 1:
                    if (sv >> j) & 1:
                        out |= 1 << bit
                    bit += 1
            return claripy.BVV(out, w)
    except Exception:
        pass
    return _vex_pext_symbolic_mask(src, mask)


def _vex_pdep_impl(state, src, mask):
    del state
    try:
        w_src = _vex_bv_size(src)
    except Exception:
        w_src = 64
    try:
        w_mask = _vex_bv_size(mask)
    except Exception:
        w_mask = w_src
    w = max(int(w_src), int(w_mask))
    try:
        src = _vex_to_bv(src, w_src if not isinstance(src, int) else w)
        mask = _vex_to_bv(mask, w_mask if not isinstance(mask, int) else w)
        src = _vex_resize(src, w)
        mask = _vex_resize(mask, w)
    except Exception:
        pass
    try:
        mv = _vex_concrete_bvv(mask)
    except Exception:
        mv = None
    if mv is not None:
        try:
            return _vex_pdep_concrete_mask(src, mv)
        except Exception:
            pass
    try:
        sv = _vex_concrete_bvv(src)
        if sv is not None and mv is not None:
            out = 0
            bit = 0
            for j in range(w):
                if (mv >> j) & 1:
                    if (sv >> bit) & 1:
                        out |= 1 << j
                    bit += 1
            return claripy.BVV(out, w)
    except Exception:
        pass
    return _vex_pdep_symbolic_mask(src, mask)


try:
    from angr.engines.vex.claripy import ccall as _vex_ccall_module

    if not hasattr(_vex_ccall_module, "amd64g_calculate_pext"):
        _vex_ccall_module.amd64g_calculate_pext = _vex_pext_impl
    if not hasattr(_vex_ccall_module, "amd64g_calculate_pdep"):
        _vex_ccall_module.amd64g_calculate_pdep = _vex_pdep_impl
except Exception:
    pass
