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
import functools
import json
import mmap
import os
import signal
import struct
import tempfile
from pathlib import Path

from elftools.elf.elffile import ELFFile

from . import info as _info
from .arch import x86_64 as arch
from .core import (
    _split_list,
    choose_type,
    is_group,
    open_task_counters,
    track_target_cpus,
    with_leader,
)
from .core import (
    demangle as _demangle_sym,
)

PTRACE_TRACEME = 0
PTRACE_PEEKTEXT = 2
PTRACE_POKETEXT = 4
PTRACE_CONT = 7
PTRACE_GETREGS = 12
PTRACE_SETREGS = 13
PTRACE_DETACH = 17
PTRACE_SETOPTIONS = 0x4200
PTRACE_O_TRACEEXEC = 0x10
PROT_READ = 0x1
PROT_WRITE = 0x2
PROT_EXEC = 0x4
MAP_SHARED = 0x01
MAP_PRIVATE = 0x02
MAP_ANONYMOUS = 0x20
MAP_FIXED = 0x10
SIGSTOP = int(signal.SIGSTOP)
_MASK64 = 0xFFFFFFFFFFFFFFFF
_CALIBRATION_TRIALS = 64
_TRACK_KINDS = ("label", "func")
_TRACK_KIND_LABEL = "label"
_TRACK_KIND_FUNC = "func"
DEFAULT_OUTPUT = "track.json"
DEFAULT_BUFFER_SIZE = 1 << 16


class UserRegs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in arch.USER_REGS_FIELDS]


@functools.cache
def _libc():
    return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def ptrace(req, pid, addr=0, data=0):
    libc = _libc()
    if isinstance(data, (bytes, bytearray)):
        buf = ctypes.create_string_buffer(bytes(data))
        res = libc.ptrace(
            ctypes.c_ulong(req),
            ctypes.c_int(pid),
            ctypes.c_void_p(addr),
            ctypes.c_void_p(ctypes.addressof(buf)),
        )
        if res == -1:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        return bytes(buf.raw)
    res = libc.ptrace(
        ctypes.c_ulong(req),
        ctypes.c_int(pid),
        ctypes.c_void_p(addr),
        ctypes.c_void_p(data),
    )
    if res == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return res


def peek(pid, addr):
    libc = _libc()
    libc.ptrace.restype = ctypes.c_longlong
    ctypes.set_errno(0)
    val = libc.ptrace(
        ctypes.c_ulong(PTRACE_PEEKTEXT),
        ctypes.c_int(pid),
        ctypes.c_void_p(addr),
        ctypes.c_void_p(0),
    )
    if val == -1 and ctypes.get_errno() != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return val & 0xFFFFFFFFFFFFFFFF


def poke(pid, addr, word):
    return ptrace(PTRACE_POKETEXT, pid, addr, word & 0xFFFFFFFFFFFFFFFF)


def read_mem(pid, addr, size):
    out = bytearray()
    a = int(addr) - (int(addr) % 8)
    lead = int(addr) - a
    while len(out) < lead + int(size):
        out += struct.pack("<Q", peek(pid, a + len(out)))
    return bytes(out[lead : lead + int(size)])


def write_mem(pid, addr, data):
    data = bytes(data)
    a = int(addr) - (int(addr) % 8)
    lead = int(addr) - a
    cur = bytearray()
    total = lead + len(data)
    while len(cur) < total + ((-total) % 8):
        cur += struct.pack("<Q", peek(pid, a + len(cur)))
    cur[lead : lead + len(data)] = data
    for i in range(0, len(cur), 8):
        poke(pid, a + i, struct.unpack_from("<Q", cur, i)[0])


def getregs(pid):
    regs = UserRegs()
    ptrace(PTRACE_GETREGS, pid, 0, ctypes.addressof(regs))
    return regs


def setregs(pid, regs):
    ptrace(PTRACE_SETREGS, pid, 0, ctypes.addressof(regs))


def waitpid(pid):
    _, status = os.waitpid(pid, 0)
    return status


def signed_rax(rax):
    rax &= 0xFFFFFFFFFFFFFFFF
    return rax - (1 << 64) if rax >= (1 << 63) else rax


def remote_syscall(pid, nr, rdi=0, rsi=0, rdx=0, r10=0, r8=0, r9=0, anchor=None):
    from .arch import x86_64 as _arch

    regs = getregs(pid)
    saved_regs = UserRegs()
    ctypes.memmove(ctypes.byref(saved_regs), ctypes.byref(regs), ctypes.sizeof(regs))
    scratch = int(anchor) if anchor is not None else int(regs.rip)
    saved_code = read_mem(pid, scratch, 8)
    write_mem(pid, scratch, _arch.SYSCALL_OP + _arch.INT3_OP + b"\x90" * 5)
    regs.rax = int(nr)
    regs.rdi = int(rdi) & 0xFFFFFFFFFFFFFFFF
    regs.rsi = int(rsi) & 0xFFFFFFFFFFFFFFFF
    regs.rdx = int(rdx) & 0xFFFFFFFFFFFFFFFF
    regs.r10 = int(r10) & 0xFFFFFFFFFFFFFFFF
    regs.r8 = int(r8) & 0xFFFFFFFFFFFFFFFF
    regs.r9 = int(r9) & 0xFFFFFFFFFFFFFFFF
    regs.rip = scratch
    setregs(pid, regs)
    ptrace(PTRACE_CONT, pid, 0, 0)
    waitpid(pid)
    out = getregs(pid)
    rax = int(out.rax)
    write_mem(pid, scratch, saved_code)
    setregs(pid, saved_regs)
    return rax


def remote_mmap(pid, size, fd=-1, shared=False, hint=0, anchor=None):
    from .arch import x86_64 as _arch

    size = int((int(size) + 0xFFF) & ~0xFFF)
    regs = getregs(pid)
    saved_regs = UserRegs()
    ctypes.memmove(ctypes.byref(saved_regs), ctypes.byref(regs), ctypes.sizeof(regs))
    scratch = int(anchor) if anchor is not None else int(regs.rip)
    saved_code = read_mem(pid, scratch, 8)
    write_mem(pid, scratch, _arch.SYSCALL_OP + _arch.INT3_OP + b"\x90" * 5)
    regs.rax = _arch.NR_MMAP
    regs.rdi = int(hint)
    regs.rsi = size
    regs.rdx = PROT_READ | PROT_WRITE | PROT_EXEC
    regs.r10 = (MAP_SHARED if shared else MAP_PRIVATE) | MAP_ANONYMOUS
    if shared and int(fd) >= 0:
        regs.r10 = MAP_SHARED
        regs.r8 = int(fd)
        regs.r9 = 0
    else:
        regs.r8 = 0xFFFFFFFFFFFFFFFF
        regs.r9 = 0
    regs.rip = scratch
    setregs(pid, regs)
    ptrace(PTRACE_CONT, pid, 0, 0)
    waitpid(pid)
    out = getregs(pid)
    addr = int(out.rax)
    write_mem(pid, scratch, saved_code)
    setregs(pid, saved_regs)
    if addr <= 0 or addr > 0xFFFFFFFFFFFF:
        raise OSError(f"remote mmap failed: {addr:#x}")
    return addr


def _find_trackable(exec_path, with_functions=True):
    import angr

    from .exec import resolve_exec
    from .info import functions as _functions
    from .info import labels as _labels

    project = angr.Project(
        resolve_exec(exec_path), auto_load_libs=False, load_debug_info=False
    )
    labels = [{"name": name, "addr": int(addr)} for name, addr in _labels(project)]
    funcs = {}
    if with_functions:
        try:
            found, _ = _functions(project)
        except Exception:
            found = {}
        raw = {}
        for name, bounds in (found or {}).items():
            try:
                start, end = int(bounds[0]), int(bounds[1])
            except (TypeError, ValueError, IndexError):
                continue
            if end <= start:
                continue
            raw[str(name)] = (start, end)
        seen_ranges = set()
        for name in sorted(raw, key=lambda n: ("_Z" in n, n)):
            bounds = raw[name]
            key = (int(bounds[0]), int(bounds[1]))
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            try:
                disp = _demangle_sym(name)
            except Exception:
                disp = str(name)
            funcs[str(disp)] = (int(bounds[0]), int(bounds[1]))
    return {"labels": labels, "functions": funcs}


def _parse_hex_addr(expr):
    try:
        s = str(expr).strip()
    except Exception:
        return None
    if not s:
        return None
    try:
        v = int(s, 0)
    except (TypeError, ValueError):
        return None

    if (
        s.lower().startswith("0x")
        or s.lower().startswith("0o")
        or s.lower().startswith("0b")
    ):
        return v

    try:
        if s.isdigit():
            return v
    except Exception:
        pass
    return None


def _match_function(name, funcs):
    if not name or not funcs:
        return None
    s = str(name).strip()
    if not s:
        return None
    if s in funcs:
        return s
    short = s.split("(")[0].strip()
    if short and short in funcs and short != s:
        return short
    try:
        dem_s = _demangle_sym(s)
    except Exception:
        dem_s = s
    try:
        dem_short = _demangle_sym(short) if short else short
    except Exception:
        dem_short = short
    if dem_s in funcs:
        return dem_s
    if dem_short and dem_short in funcs and dem_short != dem_s:
        return dem_short
    try:
        for fname in funcs:
            try:
                if _demangle_sym(fname) == s or _demangle_sym(fname) == short:
                    return fname
            except Exception:
                continue
            try:
                if _demangle_sym(fname) == dem_s or _demangle_sym(fname) == dem_short:
                    return fname
            except Exception:
                continue
            try:
                if str(fname).split("(")[0].strip() == short:
                    return fname
            except Exception:
                continue
            try:
                if str(fname).split("(")[0].strip() == dem_short:
                    return fname
            except Exception:
                continue
    except Exception:
        pass

    try:
        cands = [f for f in funcs if str(f).split("(")[0].strip() == short]
        if not cands and dem_short and dem_short != short:
            cands = [f for f in funcs if str(f).split("(")[0].strip() == dem_short]
        uniq = {tuple(funcs[c]) for c in cands}
        if len(uniq) == 1 and cands:
            return cands[0]
    except Exception:
        pass
    return None


def _resolve_single_side(side, labels, for_begin=True):
    s = str(side).strip() if side is not None else ""
    if not s:
        return None
    for lb in labels or []:
        if str(lb.get("name", "")) == s:
            return {"name": s, "addr": int(lb["addr"]), "kind": "label"}
    hexed = _parse_hex_addr(s)
    if hexed is not None:
        return {"name": s, "addr": int(hexed), "kind": "label"}
    return None


def _default_track_points(real):
    try:
        min_size = int(arch.PATCH_SIZE)
    except Exception:
        min_size = 5
    trackable = _find_trackable(real, with_functions=True)
    points = [
        {
            "name": str(lb.get("name", "")),
            "addr": int(lb["addr"]),
            "kind": "label",
        }
        for lb in (trackable.get("labels") or [])
    ]
    for sym, bounds in (trackable.get("functions") or {}).items():
        try:
            start, end = int(bounds[0]), int(bounds[1])
        except (TypeError, ValueError, IndexError):
            continue
        if end <= start:
            continue
        try:
            sym = str(_demangle_sym(str(sym)))
        except Exception:
            sym = str(sym)
        base = sym.split("(")[0].strip()
        if base.startswith("sub_") or sym.startswith("Unresolvable"):
            continue
        if end - start < min_size:
            continue
        points.append(
            {"name": sym, "addr": int(start), "kind": "func_entry", "func": sym}
        )
        points.append({"name": sym, "addr": int(end), "kind": "func_exit", "func": sym})
    seen, uniq = set(), []
    for pt in points:
        try:
            key = ("addr", int(pt.get("addr", 0)))
        except (TypeError, ValueError):
            key = ("name", str(pt.get("name", "")))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(pt)
    patchable = [p for p in uniq if p.get("kind") != "func_exit"]
    try:
        patchable.sort(key=lambda p: int(p.get("addr", 0)))
    except Exception:
        pass
    kept_addrs = []
    keep = set()
    for p in patchable:
        try:
            a = int(p.get("addr", 0))
        except (TypeError, ValueError):
            continue
        if any(abs(a - k) < min_size for k in kept_addrs):
            continue
        kept_addrs.append(a)
        try:
            keep.add(("patch", int(p.get("addr", 0))))
        except (TypeError, ValueError):
            pass
    out = []
    for pt in uniq:
        if pt.get("kind") == "func_exit":
            out.append(pt)
            continue
        try:
            key = ("patch", int(pt.get("addr", 0)))
        except (TypeError, ValueError):
            out.append(pt)
            continue
        if key in keep:
            out.append(pt)
    return out


def _resolve_track_points(exec_path, patterns):
    from .exec import resolve_exec

    real = resolve_exec(exec_path)
    if patterns is None or (isinstance(patterns, (list, tuple)) and not patterns):
        return _default_track_points(real)
    if isinstance(patterns, str):
        patterns = [patterns]
    elif (
        isinstance(patterns, tuple)
        and len(patterns) == 2
        and all(isinstance(x, str) for x in patterns)
    ):
        patterns = [patterns]
    pats = []
    for p in patterns or []:
        if p is None:
            continue
        if isinstance(p, (list, tuple)):
            parts = [str(x).strip() if x is not None else "" for x in p]
            if len(parts) != 2 or not all(parts):
                raise ValueError(
                    f"invalid filter {p!r}; expected 'name' or ('begin', 'end')"
                )
            pats.append((parts[0], parts[1]))
            continue
        s = str(p).strip()
        if not s:
            continue
        if ".." in s:
            raise ValueError(
                f"invalid filter {s!r}; use ('begin', 'end') instead of 'begin..end'"
            )
        pats.append(s)
    if not pats:
        return _default_track_points(real)
    labels = _find_trackable(real, with_functions=False).get("labels") or []
    funcs = None
    points = []
    for pat in pats:
        if isinstance(pat, (list, tuple)):
            b, e = str(pat[0]).strip(), str(pat[1]).strip()
            rb = _resolve_single_side(b, labels, for_begin=True)
            re_ = _resolve_single_side(e, labels, for_begin=False)
            if rb is None or re_ is None:
                continue
            points.append(rb)
            points.append(re_)
            continue
        hit_label = None
        for lb in labels:
            if str(lb.get("name", "")) == pat:
                hit_label = lb
                break
        if hit_label is not None:
            points.append(
                {"name": pat, "addr": int(hit_label["addr"]), "kind": "label"}
            )
            continue
        hexed = _parse_hex_addr(pat)
        if hexed is not None:
            points.append({"name": pat, "addr": int(hexed), "kind": "label"})
            continue
        if funcs is None:
            funcs = _find_trackable(real, with_functions=True).get("functions") or {}
        fmatch = _match_function(pat, funcs)
        if fmatch is not None:
            start, end = funcs[fmatch]
            try:
                sym = str(_demangle_sym(str(fmatch)))
            except Exception:
                sym = str(fmatch)
            points.append(
                {
                    "name": sym,
                    "addr": int(start),
                    "kind": "func_entry",
                    "func": sym,
                }
            )
            points.append(
                {
                    "name": sym,
                    "addr": int(end),
                    "kind": "func_exit",
                    "func": sym,
                }
            )
    seen, uniq = set(), []
    for pt in points:
        kind = pt.get("kind")
        if kind == "func_exit":
            key = ("func_exit", str(pt.get("func", "")), str(pt.get("name", "")))
        else:
            try:
                key = ("addr", int(pt.get("addr", 0)))
            except (TypeError, ValueError):
                key = ("name", str(pt.get("name", "")))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(pt)
    return uniq


def load_bias(pid, exec_path):
    with open(exec_path, "rb") as fh:
        elffile = ELFFile(fh)
        min_vaddr, is_pie = None, elffile.header["e_type"] == "ET_DYN"
        for seg in elffile.iter_segments():
            if seg["p_type"] != "PT_LOAD":
                continue
            v = int(seg["p_vaddr"])
            min_vaddr = v if min_vaddr is None else min(min_vaddr, v)
    if min_vaddr is None:
        raise ValueError("no PT_LOAD segments found")
    if not is_pie:
        return 0
    want = os.path.realpath(exec_path)
    with open(f"/proc/{int(pid)}/maps") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                path = os.path.realpath(parts[-1])
            except Exception:
                continue
            if path != want or not parts[1].startswith("r-x"):
                continue
            start = int(parts[0].split("-")[0], 16)
            off = int(parts[2], 16)
            return start - (min_vaddr + off)
    raise RuntimeError(f"executable mapping of {exec_path!r} not found in pid {pid}")


def _elf_image_end(exec_path):
    with open(exec_path, "rb") as fh:
        elffile = ELFFile(fh)
        end = 0
        for seg in elffile.iter_segments():
            if seg["p_type"] != "PT_LOAD":
                continue
            end = max(end, int(seg["p_vaddr"]) + int(seg["p_memsz"]))
    return end


def _find_cave(pid, exec_path, bias, entry, size=1 << 20):
    end = _elf_image_end(exec_path)
    base = (int(bias) + int(end) + 0xFFF) & ~0xFFF
    for step in (0, 0x100000, 0x200000, 0x400000, 0x800000, 0x1000000):
        cave = remote_mmap(pid, size, hint=base + step, anchor=entry)
        if abs(cave - base) < 2**30:
            return cave
    raise SystemExit("could not place trampoline cave within 2G of the executable")


class RingReader:
    def __init__(self, path_or_bytes, events):
        self.events = list(events or [])
        if isinstance(path_or_bytes, (bytes, bytearray)):
            self.blob = bytes(path_or_bytes)
        else:
            with open(path_or_bytes, "rb") as fh:
                self.blob = fh.read()

    def records(self):
        if len(self.blob) < arch.TRACK_RING_HDR_SIZE:
            return []
        magic, head, cap, stride, _n_ev = struct.unpack_from("<QQQQQ", self.blob, 0)
        if magic != arch.TRACK_RING_MAGIC or cap == 0 or stride == 0:
            return []
        n = min(int(head), int(cap))
        start = int(head) - n
        out = []
        for seq in range(start, int(head)):
            off = arch.TRACK_RING_HDR_SIZE + (seq % int(cap)) * int(stride)
            slot = self.blob[off : off + int(stride)]
            if len(slot) < int(stride):
                break
            (mid32,) = struct.unpack_from("<I", slot, arch.TRACK_SLOT_ID_OFF)
            (s64,) = struct.unpack_from("<Q", slot, arch.TRACK_SLOT_SEQ_OFF)
            base = arch.TRACK_SLOT_EVENTS_OFF
            vals = [
                struct.unpack_from("<Q", slot, base + 8 * i)[0]
                for i in range(len(self.events))
                if base + 8 * (i + 1) <= len(slot)
            ]
            rec = {"point_id": mid32, "seq": s64}
            rec.update({e: v for e, v in zip(self.events, vals)})
            out.append(rec)
        return out


def _init_ring_file(path, n_events, slots):
    n_events = list(n_events or [])
    lay = arch.track_ring_layout(len(n_events), slots)
    with open(path, "wb") as fh:
        fh.truncate(lay["total"])
    with open(path, "r+b") as fh:
        _write_zeros = b"\0" * (1 << 20)
        for off in range(0, lay["total"], len(_write_zeros)):
            fh.seek(off)
            fh.write(_write_zeros[: min(len(_write_zeros), lay["total"] - off)])
        fh.flush()
        mm = mmap.mmap(fh.fileno(), lay["total"], access=mmap.ACCESS_WRITE)
        struct.pack_into(
            "<QQQQQ",
            mm,
            0,
            arch.TRACK_RING_MAGIC,
            0,
            lay["capacity"],
            lay["stride"],
            len(list(n_events or [])),
        )
        mm.flush()
        mm.close()
    return lay


def track(
    cmd,
    event=None,
    filter=None,
    output=DEFAULT_OUTPUT,
    buffer_size=DEFAULT_BUFFER_SIZE,
):
    if isinstance(cmd, str):
        import shlex

        cmd = shlex.split(cmd)
    cmd = [str(c) for c in cmd]
    if not cmd:
        raise ValueError("no command to track")
    from .exec import resolve_exec

    exec_path = resolve_exec(cmd[0])
    ev_list = _split_list(event) or ["cycles"]
    ev_list = with_leader(ev_list)
    try:
        _force_typ = choose_type(ev_list)
    except Exception:
        _force_typ = None
    _is_group = is_group(ev_list)
    selected = _resolve_track_points(exec_path, filter)
    if not selected:
        if filter:
            raise SystemExit("no track points left after applying --filter")
        raise SystemExit(
            f"no labels or functions found in {exec_path!r}; "
            "annotate code with PERF_LABEL(name) from lib/perf/ and rebuild, "
            "or track via -f (e.g. -f fizz_buzz)"
        )
    lay = arch.track_ring_layout(len(ev_list), buffer_size)

    ring_tmp = tempfile.NamedTemporaryFile(
        prefix="perf-track-", suffix=".ring", delete=False
    )
    ring_path = ring_tmp.name
    ring_tmp.close()
    lay = _init_ring_file(ring_path, ev_list, buffer_size)
    ring_fd = os.open(ring_path, os.O_RDWR)
    try:
        os.set_inheritable(ring_fd, True)
    except Exception:
        pass

    pid = os.fork()
    if pid == 0:
        try:
            os.set_inheritable(ring_fd, True)
            ptrace(PTRACE_TRACEME, 0, 0, 0)
            os.kill(os.getpid(), SIGSTOP)
            os.execv(exec_path, cmd)
        except Exception:
            os._exit(127)

    failed = False
    old_sigterm = None
    old_sighup = None
    try:
        try:
            old_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _interrupt_handler)
        except (OSError, ValueError):
            old_sigterm = None
        try:
            old_sighup = signal.getsignal(signal.SIGHUP)
            signal.signal(signal.SIGHUP, _interrupt_handler)
        except (OSError, ValueError):
            old_sighup = None
        waitpid(pid)
        ptrace(PTRACE_SETOPTIONS, pid, 0, PTRACE_O_TRACEEXEC)
        ptrace(PTRACE_CONT, pid, 0, 0)
        waitpid(pid)
        entry = int(getregs(pid).rip)
        saved_entry = read_mem(pid, entry, 8)
        write_mem(pid, entry, arch.INT3_OP)
        ptrace(PTRACE_CONT, pid, 0, 0)
        waitpid(pid)
        frozen = getregs(pid)
        frozen.rip = entry
        setregs(pid, frozen)
        bias = load_bias(pid, exec_path)
        pic, angr_base = _info._pic_base(exec_path)

        def _runtime(addr):
            link = int(addr) - angr_base if pic else int(addr)
            return link + bias

        _old_affinity = None
        _target_cpus = track_target_cpus(_force_typ)
        _child_cpu = None
        if _target_cpus:
            try:
                _old_affinity = sorted(os.sched_getaffinity(0))
            except OSError:
                _old_affinity = None
            try:
                _parent_cpu = _target_cpus[0]
                _child_cpu = (
                    _target_cpus[1] if len(_target_cpus) > 1 else _target_cpus[0]
                )
                try:
                    os.sched_setaffinity(0, {_parent_cpu})
                except OSError:
                    pass
                try:
                    os.sched_setaffinity(pid, {_child_cpu})
                except OSError:
                    pass
            except Exception:
                pass

        counters, indices = [], []
        overhead = _zero_overhead(ev_list)
        try:
            counters, indices = open_task_counters(
                ev_list, _force_typ, _is_group, pid, anchor=entry
            )

            try:
                overhead = measure_track_overhead(
                    ev_list,
                    indices,
                    _force_typ,
                    _is_group,
                    layout=lay,
                    cpu=_child_cpu,
                )
            except Exception:
                overhead = _zero_overhead(ev_list)
            for c in counters:
                try:
                    c.enable()
                except Exception:
                    pass
            remote = remote_mmap(
                pid, lay["total"], fd=ring_fd, shared=True, anchor=entry
            )
            cave = _find_cave(pid, exec_path, bias, entry)

            shadow_top = 0
            if any(str(pt.get("kind", "label")) == "func_entry" for pt in selected):
                _slots = 65536
                _size = ((_slots * 8 + 8 + 0xFFF) & ~0xFFF) or 0x1000
                shadow_base = remote_mmap(pid, _size, anchor=entry)
                shadow_top = int(shadow_base)
                write_mem(pid, shadow_top, struct.pack("<Q", shadow_base + 8))

            tramp_addrs, blobs = [None] * len(selected), [None] * len(selected)
            cursor = cave
            func_exit_addr = {}
            for mid, pt in enumerate(selected):
                if str(pt.get("kind", "label")) != "func_exit":
                    continue
                blob = arch.build_track_func_exit(
                    mid, ev_list, indices, remote, cursor, lay, shadow_top=shadow_top
                )
                blobs[mid] = (cursor, blob)
                tramp_addrs[mid] = cursor
                try:
                    func_exit_addr[str(pt.get("func", ""))] = cursor
                except Exception:
                    pass
                cursor += (len(blob) + 15) & ~15
            patch_len_by_mid = {}
            for mid, pt in enumerate(selected):
                kind = str(pt.get("kind", "label"))
                if kind == "func_exit":
                    continue
                rt = _runtime(pt["addr"])
                try:
                    live = read_mem(pid, rt, 16)
                except OSError as ex:
                    raise SystemExit(
                        f"cannot read patch address {rt:#x} "
                        f"for {pt.get('name', '?')!r}: {ex}"
                    ) from ex
                try:
                    patch_len, _ins = arch.detour_patch_len(live, rt)
                except Exception as ex:
                    raise SystemExit(
                        f"cannot detour {pt.get('name', '?')!r} at {rt:#x}: {ex}"
                    ) from ex
                if kind == "func_entry":
                    exit_addr = func_exit_addr.get(str(pt.get("func", "")), 0)
                    if not exit_addr:
                        raise SystemExit(
                            f"no exit trampoline for function {pt.get('func', '?')!r}"
                        )
                    if abs(cursor - rt) >= 2**31 - 256:
                        raise SystemExit(
                            f"function {pt.get('func', '?')!r} too far "
                            "from trampoline cave (>2G)"
                        )
                    try:
                        blob = arch.build_track_func_entry_raw(
                            mid,
                            ev_list,
                            indices,
                            remote,
                            cursor,
                            rt,
                            patch_len,
                            live,
                            lay,
                            shadow_top=shadow_top,
                            exit_addr=exit_addr,
                        )
                    except Exception as ex:
                        raise SystemExit(
                            f"cannot detour function {pt.get('func', '?')!r} "
                            f"entry at {rt:#x}: {ex}"
                        ) from ex
                else:
                    if abs(cursor - rt) >= 2**31 - 256:
                        raise SystemExit(
                            f"patch address {rt:#x} too far from trampoline cave (>2G)"
                        )
                    try:
                        blob = arch.build_track_detour_raw(
                            mid,
                            ev_list,
                            indices,
                            remote,
                            cursor,
                            rt,
                            patch_len,
                            live,
                            lay,
                        )
                    except Exception as ex:
                        raise SystemExit(
                            f"cannot detour {pt.get('name', '?')!r} at {rt:#x}: {ex}"
                        ) from ex
                blobs[mid] = (cursor, blob)
                tramp_addrs[mid] = cursor
                cursor += (len(blob) + 15) & ~15
                patch_len_by_mid[mid] = int(patch_len)

            _ranges = []
            for mid, pt in enumerate(selected):
                kind = str(pt.get("kind", "label"))
                if kind == "func_exit":
                    continue
                rt = _runtime(pt["addr"])
                plen = int(patch_len_by_mid.get(mid, arch.PATCH_SIZE))
                _ranges.append((rt, rt + plen, pt.get("name", "?")))
            _ranges.sort()
            for (a0, a1, n0), (b0, b1, n1) in zip(_ranges, _ranges[1:]):
                if b0 < a1:
                    raise SystemExit(
                        f"track points {n0!r} and {n1!r} overlap "
                        f"({a0:#x}..{a1:#x} vs {b0:#x}..{b1:#x}); "
                        "pick non-overlapping addresses"
                    )
            for addr, blob in [b for b in blobs if b is not None]:
                write_mem(pid, addr, blob)
            for mid, pt in enumerate(selected):
                kind = str(pt.get("kind", "label"))
                if kind == "func_exit":
                    continue
                rt = _runtime(pt["addr"])
                taddr = tramp_addrs[mid]
                write_mem(pid, rt, arch.patch_jmp(rt, taddr))
                plen = int(patch_len_by_mid.get(mid, arch.PATCH_SIZE))
                if plen > arch.PATCH_SIZE:
                    write_mem(
                        pid,
                        rt + arch.PATCH_SIZE,
                        b"\x90" * (plen - arch.PATCH_SIZE),
                    )
            write_mem(pid, entry, saved_entry)
            setregs(pid, frozen)

            ptrace(PTRACE_DETACH, pid, 0, 0)
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        finally:
            for c in counters:
                try:
                    try:
                        c.disable()
                    finally:
                        c.close()
                except Exception:
                    pass
            if _old_affinity is not None:
                try:
                    os.sched_setaffinity(0, set(_old_affinity))
                except OSError:
                    pass
        df = _dump_results(
            ring_path,
            ev_list,
            exec_path,
            cmd,
            selected,
            output,
            overhead,
        )
        return df
    except KeyboardInterrupt:
        failed = True

        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            os.waitpid(pid, 0)
        except Exception:
            pass
        try:
            _dump_results(
                ring_path,
                ev_list,
                exec_path,
                cmd,
                selected,
                output,
                overhead,
            )
        except Exception:
            pass
        raise
    except BaseException:
        failed = True
        raise
    finally:
        for _sig, _old in (
            (signal.SIGTERM, old_sigterm),
            (signal.SIGHUP, old_sighup),
        ):
            if _sig is None or _old is None:
                continue
            try:
                signal.signal(_sig, _old)
            except (OSError, ValueError):
                pass
        if failed:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
            try:
                os.waitpid(pid, 0)
            except Exception:
                pass
        try:
            os.close(ring_fd)
        except Exception:
            pass


def _dump_results(ring_path, ev_list, exec_path, cmd, selected, output, overhead=None):
    records = RingReader(ring_path, ev_list).records()
    id2m = {i: m for i, m in enumerate(selected)}
    hits = [
        {
            "name": id2m.get(r["point_id"], {}).get("name", r["point_id"]),
            "kind": id2m.get(r["point_id"], {}).get("kind", "label"),
            "seq": int(r["seq"]),
            **{e: int(r.get(e, 0)) for e in ev_list},
        }
        for r in records
    ]
    rows = _pair_deltas(hits, ev_list, overhead)
    df = _rows_to_df(rows, ev_list)
    if "duration_time" in df.columns:
        try:
            _hz = float(_info.cpuinfo(["hz"]).iloc[0]["hz"])
        except Exception:
            _hz = 0.0
        if _hz and _hz > 0:
            df["duration_time"] = df["duration_time"] * 1e9 / _hz
    payload = _payload(
        exec_path,
        cmd,
        ev_list,
        df.to_dict(orient="records"),
        overhead=overhead,
    )
    try:
        df["file"] = payload["file"]
    except Exception:
        pass
    try:
        df["mode"] = payload.get("mode", "record")
    except Exception:
        pass
    try:
        df.attrs["file"] = payload["file"]
        df.attrs["id"] = payload["id"]
        df.attrs["info"] = payload["info"]
        df.attrs["config"] = payload["config"]
    except Exception:
        pass
    if output is not None:
        out_path = Path(str(output))
        if out_path.is_dir():
            out_path = out_path / DEFAULT_OUTPUT
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=4) + "\n")
    return df


def _subtract_overhead(raw, overhead):
    try:
        oh = int(overhead or 0)
    except (TypeError, ValueError):
        oh = 0
    if oh <= 0:
        return int(raw) & _MASK64
    raw = int(raw) & _MASK64
    return raw - oh if raw >= oh else 0


def _zero_overhead(ev_list):
    return {kind: {e: 0 for e in ev_list} for kind in _TRACK_KINDS}


def _measure_pairs(fn, ev_list, ring, stride, trials):
    base = arch.TRACK_SLOT_EVENTS_OFF
    deltas = {e: [] for e in ev_list}
    for _ in range(trials):
        try:
            struct.pack_into("<Q", ring, arch.TRACK_RING_HEAD_OFF, 0)
        except Exception:
            continue
        fn()
        try:
            head = struct.unpack_from("<Q", ring, arch.TRACK_RING_HEAD_OFF)[0]
        except Exception:
            head = 0
        if head != 2:
            continue
        for i, e in enumerate(ev_list):
            off0 = arch.TRACK_RING_DATA_OFF + base + 8 * i
            off1 = off0 + stride
            try:
                v0 = struct.unpack_from("<Q", ring, off0)[0]
                v1 = struct.unpack_from("<Q", ring, off1)[0]
            except Exception:
                continue
            deltas[e].append((v1 - v0) & _MASK64)
    out = {}
    for e, vals in deltas.items():
        if not vals:
            out[e] = 0
            continue
        vals.sort()
        out[e] = int(vals[len(vals) // 2])
    return out


def _build_label_calibration(ev_list, indices, ring_addr, exec_addr, layout):
    stub_len = arch.PATCH_SIZE
    b1 = arch.build_track_detour(
        1, ev_list, indices, ring_addr, exec_addr, exec_addr, layout, b"", True
    )
    stub_addr = exec_addr + len(
        arch.build_track_detour(
            0, ev_list, indices, ring_addr, exec_addr, exec_addr, layout, b"", True
        )
    )
    addr1 = (stub_addr + stub_len + 15) & ~15
    addr_ret = addr1 + len(b1)
    b0 = arch.build_track_detour(
        0, ev_list, indices, ring_addr, exec_addr, stub_addr, layout, b"", True
    )
    b1 = arch.build_track_detour(
        1, ev_list, indices, ring_addr, addr1, addr_ret, layout, b"", True
    )
    stub = arch.patch_jmp(stub_addr, addr1)
    pad = addr1 - (stub_addr + stub_len)
    return b0 + stub + b"\x90" * pad + b1 + b"\xc3"


def _build_func_calibration(ev_list, indices, ring_addr, exec_addr, layout, shadow_top):
    stub_len = arch.PATCH_SIZE
    nops = b"\x90" * stub_len
    entry_len = len(
        arch.build_track_func_entry_raw(
            0,
            ev_list,
            indices,
            ring_addr,
            exec_addr,
            exec_addr,
            stub_len,
            nops,
            layout,
            shadow_top,
            0,
        )
    )
    site_addr = exec_addr + entry_len
    resume = site_addr + stub_len
    exit_addr = (resume + 1 + 15) & ~15
    entry = arch.build_track_func_entry_raw(
        0,
        ev_list,
        indices,
        ring_addr,
        exec_addr,
        site_addr,
        stub_len,
        nops,
        layout,
        shadow_top,
        exit_addr,
    )
    exit_ = arch.build_track_func_exit(
        1, ev_list, indices, ring_addr, exit_addr, layout, shadow_top=shadow_top
    )
    pad = exit_addr - (resume + 1)
    return entry + nops + b"\xc3" + b"\x90" * pad + exit_


def _calibrate_overhead(ev_list, force_typ, group, trials=None, layout=None):
    import ctypes as _ct

    from .core import open_counters as _open_counters

    ev_list = list(ev_list or [])
    if not ev_list:
        return {}
    trials = _CALIBRATION_TRIALS if trials is None else max(1, int(trials))
    if layout is None:
        lay = arch.track_ring_layout(len(ev_list), 2)
    else:
        lay = dict(layout)
    stride = lay["stride"]
    total = arch.TRACK_RING_DATA_OFF + 2 * int(stride)
    ring = _ct.create_string_buffer(total)
    ring_addr = _ct.addressof(ring)
    struct.pack_into(
        "<QQQQQ",
        ring,
        0,
        arch.TRACK_RING_MAGIC,
        0,
        lay["capacity"],
        lay["stride"],
        len(ev_list),
    )
    exec_size = 16384
    mm = mmap.mmap(
        -1,
        exec_size,
        prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
    )
    try:
        exec_addr = _ct.addressof(_ct.c_char.from_buffer(mm))
    except Exception:
        mm.close()
        raise RuntimeError("cannot address executable calibration page")
    shpage = None
    shadow_top = 0
    if _TRACK_KIND_FUNC in _TRACK_KINDS:
        shpage = mmap.mmap(-1, 0x10000, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        try:
            shadow_top = _ct.addressof(_ct.c_char.from_buffer(shpage))
            shadow_top = ((shadow_top + 0xFFF) & ~0xFFF) or shadow_top
            struct.pack_into("<Q", shpage, 0, shadow_top + 8)
        except Exception:
            shadow_top = 0
    counters, fresh_indices = _open_counters(ev_list, force_typ, group, pid=0)
    try:
        out = {}
        for kind in _TRACK_KINDS:
            if kind == _TRACK_KIND_FUNC and not shadow_top:
                out[kind] = {e: 0 for e in ev_list}
                continue
            if kind == _TRACK_KIND_LABEL:
                blob = _build_label_calibration(
                    ev_list, fresh_indices, ring_addr, exec_addr, lay
                )
            else:
                blob = _build_func_calibration(
                    ev_list, fresh_indices, ring_addr, exec_addr, lay, shadow_top
                )
            if len(blob) > exec_size:
                raise RuntimeError("calibration blob does not fit the exec page")
            mm.seek(0)
            mm.write(blob)
            try:
                mm.flush()
            except Exception:
                pass
            fn = _ct.CFUNCTYPE(None)(exec_addr)
            out[kind] = _measure_pairs(fn, ev_list, ring, stride, trials)
    finally:
        for c in counters:
            try:
                try:
                    c.disable()
                finally:
                    c.close()
            except Exception:
                pass
        try:
            mm.close()
        except Exception:
            pass
        if shpage is not None:
            try:
                shpage.close()
            except Exception:
                pass
    return out


def measure_track_overhead(
    ev_list,
    indices=None,
    force_typ=None,
    group=False,
    trials=None,
    layout=None,
    cpu=None,
):
    ev_list = list(ev_list or [])
    if not ev_list:
        return {}
    if indices is None:
        indices = [None] * len(ev_list)
    try:
        rfd, wfd = os.pipe()
    except OSError:
        return _zero_overhead(ev_list)
    pid = os.fork()
    if pid == 0:
        try:
            try:
                os.close(rfd)
            except Exception:
                pass
            if cpu is not None:
                try:
                    os.sched_setaffinity(0, {int(cpu)})
                except (OSError, ValueError, TypeError):
                    pass
            oh = _calibrate_overhead(
                ev_list,
                force_typ,
                bool(group),
                trials,
                layout=layout,
            )
            payload = json.dumps(
                {
                    k: {e: int(oh.get(k, {}).get(e, 0)) for e in ev_list}
                    for k in _TRACK_KINDS
                }
            ).encode()
            try:
                os.write(wfd, payload)
            except Exception:
                pass
        except BaseException:
            pass
        finally:
            try:
                os.close(wfd)
            except Exception:
                pass
        os._exit(0)
    try:
        os.close(wfd)
    except Exception:
        pass
    blob = b""
    try:
        while True:
            try:
                chunk = os.read(rfd, 65536)
            except OSError:
                break
            if not chunk:
                break
            blob += chunk
    finally:
        try:
            os.close(rfd)
        except Exception:
            pass
        try:
            _, _ = os.waitpid(pid, 0)
        except Exception:
            pass
    try:
        parsed = json.loads(blob.decode() or "{}")
        return {
            k: {e: int(parsed.get(k, {}).get(e, 0)) for e in ev_list}
            for k in _TRACK_KINDS
        }
    except Exception:
        return _zero_overhead(ev_list)


def _pair_deltas(hits, ev_list, overhead=None):
    ev_list = list(ev_list or [])
    overhead = overhead or {}
    ordered = sorted(hits or [], key=lambda h: int(h.get("seq", 0)))

    def _is_func(h):
        return h.get("kind") in ("func_entry", "func_exit")

    def _kind_of(h):
        return "func" if _is_func(h) else "label"

    def _is_suffix(h):
        n = h.get("name")
        return isinstance(n, str) and (n.endswith("_begin") or n.endswith("_end"))

    func_hits = [h for h in ordered if _is_func(h)]
    suffix_hits = [h for h in ordered if not _is_func(h) and _is_suffix(h)]
    plain_hits = [h for h in ordered if not _is_func(h) and not _is_suffix(h)]

    def _delta(b, h, e):
        raw = (int(h.get(e, 0)) - int(b.get(e, 0))) & _MASK64
        return _subtract_overhead(raw, (overhead.get(_kind_of(b)) or {}).get(e, 0))

    def _row(rname, sample, b, h):
        return {
            "name": rname,
            "samples": sample,
            "operations": 1,
            **{e: _delta(b, h, e) for e in ev_list},
        }

    rows = []

    if not func_hits and not suffix_hits and not plain_hits:
        return rows

    if plain_hits:
        for i in range(0, len(plain_hits) - 1, 2):
            b, h = plain_hits[i], plain_hits[i + 1]
            rows.append(_row(f"{b.get('name')}..{h.get('name')}", i // 2, b, h))

    if func_hits:
        pending, counts = {}, {}
        for h in func_hits:
            sym = h.get("name")
            if not isinstance(sym, str):
                continue
            if h.get("kind") == "func_entry":
                pending.setdefault(sym, []).append(h)
                continue
            stack = pending.get(sym)
            if not stack:
                continue
            b = stack.pop()
            idx = counts.get(sym, 0)
            counts[sym] = idx + 1
            rows.append(_row(sym, idx, b, h))

    if suffix_hits:
        pending, counts = {}, {}
        for h in suffix_hits:
            n = h.get("name")
            if n.endswith("_begin"):
                pending[n[: -len("_begin")]] = h
            elif n.endswith("_end"):
                base = n[: -len("_end")]
                b = pending.pop(base, None)
                if b is None:
                    continue
                rname = f"{base}_begin..{base}_end"
                idx = counts.get(rname, 0)
                counts[rname] = idx + 1
                rows.append(_row(rname, idx, b, h))

    return rows


def _rows_to_df(rows, ev_list):
    import pandas as pd

    return pd.DataFrame(
        rows, columns=["name", "samples", "operations", *list(ev_list or [])]
    )


def _payload(exec_path, cmd, event, output, overhead=None):
    from datetime import datetime

    try:
        from .bench import _content_hash8, _id_hash
    except Exception:
        _content_hash8, _id_hash = None, None
    try:
        base = os.path.basename(str(exec_path or cmd[0]))
    except Exception:
        base = str(cmd[0]) if cmd else "track"
    sha = None
    if _content_hash8 is not None:
        try:
            sha = _content_hash8(str(exec_path))
        except Exception:
            sha = None
    file_label = f"{base}@{sha}" if sha else base
    binary_id = {"path": base, "sha": sha} if sha else {"path": base}
    config = {"events": ",".join(list(event or []))}
    try:
        if overhead is not None:
            config["overhead"] = {
                k: {e: int((overhead.get(k) or {}).get(e, 0)) for e in (event or [])}
                for k in (overhead or {})
            }
    except Exception:
        pass
    run_id = None
    if _id_hash is not None:
        try:
            run_id = _id_hash(None, config, binary_id)
        except Exception:
            run_id = None
    names = sorted({r.get("name") for r in (output or []) if r.get("name")})
    if len(names) == 1:
        top_name = f"{names[0]}-{run_id}" if run_id else names[0]
    else:
        top_name = f"track-{run_id}" if run_id else "track"
    try:
        cpu = _info.cpuinfo(list(_info.CPUINFO_FIELDS)).iloc[0].to_dict()
    except Exception:
        cpu = {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "file": file_label,
        "name": top_name,
        "id": run_id,
        "time": now,
        "mode": "record",
        "info": {"cpu": cpu},
        "config": config,
        "output": list(output or []),
    }


def _interrupt_handler(signum, _frame):
    raise KeyboardInterrupt(f"received signal {signum}")
