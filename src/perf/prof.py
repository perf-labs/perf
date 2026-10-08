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
import sys
import tempfile
from pathlib import Path

from elftools.elf.elffile import ELFFile

from . import info as _info
from .arch import x86_64 as arch
from .core import (
    _split_list,
    choose_type,
    is_group,
    open_counters,
    open_task_counters,
    track_target_cpus,
    with_leader,
)
from .core import (
    demangle as _demangle_sym,
)
from .exec import is_shared_library, resolve_exec
from .info import functions, labels

_PTRACE_TRACEME = 0
_PTRACE_PEEKTEXT = 2
_PTRACE_POKETEXT = 4
_PTRACE_CONT = 7
_PTRACE_GETREGS = 12
_PTRACE_SETREGS = 13
_PTRACE_DETACH = 17
_PTRACE_SETOPTIONS = 0x4200
_PTRACE_O_TRACEEXEC = 0x10
_SIGSTOP = int(signal.SIGSTOP)
_MASK64 = 0xFFFFFFFFFFFFFFFF
_CALIBRATION_TRIALS = 64
_TRACK_KINDS = ("label", "func")
_TRACK_KIND_LABEL = "label"
_TRACK_KIND_FUNC = "func"
_DEFAULT_OUTPUT = "profile.json"
_DEFAULT_BUFFER_SIZE = 1 << 16
_DEFAULT_TARGET = "main"


class UserRegs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in arch._USER_REGS_FIELDS]


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
        ctypes.c_ulong(_PTRACE_PEEKTEXT),
        ctypes.c_int(pid),
        ctypes.c_void_p(addr),
        ctypes.c_void_p(0),
    )
    if val == -1 and ctypes.get_errno() != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return val & 0xFFFFFFFFFFFFFFFF


def poke(pid, addr, word):
    return ptrace(_PTRACE_POKETEXT, pid, addr, word & 0xFFFFFFFFFFFFFFFF)


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
    ptrace(_PTRACE_GETREGS, pid, 0, ctypes.addressof(regs))
    return regs


def setregs(pid, regs):
    ptrace(_PTRACE_SETREGS, pid, 0, ctypes.addressof(regs))


def waitpid(pid):
    _, status = os.waitpid(pid, 0)
    return status


def signed_rax(rax):
    rax &= 0xFFFFFFFFFFFFFFFF
    return rax - (1 << 64) if rax >= (1 << 63) else rax


def remote_syscall(pid, nr, rdi=0, rsi=0, rdx=0, r10=0, r8=0, r9=0, anchor=None):
    regs = getregs(pid)
    saved_regs = UserRegs()
    ctypes.memmove(ctypes.byref(saved_regs), ctypes.byref(regs), ctypes.sizeof(regs))
    scratch = int(anchor) if anchor is not None else int(regs.rip)
    saved_code = read_mem(pid, scratch, 8)
    write_mem(pid, scratch, arch._SYSCALL_OP + arch._INT3_OP + b"\x90" * 5)
    regs.rax = int(nr)
    regs.rdi = int(rdi) & 0xFFFFFFFFFFFFFFFF
    regs.rsi = int(rsi) & 0xFFFFFFFFFFFFFFFF
    regs.rdx = int(rdx) & 0xFFFFFFFFFFFFFFFF
    regs.r10 = int(r10) & 0xFFFFFFFFFFFFFFFF
    regs.r8 = int(r8) & 0xFFFFFFFFFFFFFFFF
    regs.r9 = int(r9) & 0xFFFFFFFFFFFFFFFF
    regs.rip = scratch
    setregs(pid, regs)
    ptrace(_PTRACE_CONT, pid, 0, 0)
    waitpid(pid)
    out = getregs(pid)
    rax = int(out.rax)
    write_mem(pid, scratch, saved_code)
    setregs(pid, saved_regs)
    return rax


def remote_mmap(pid, size, fd=-1, shared=False, hint=0, anchor=None):
    size = int((int(size) + 0xFFF) & ~0xFFF)
    regs = getregs(pid)
    saved_regs = UserRegs()
    ctypes.memmove(ctypes.byref(saved_regs), ctypes.byref(regs), ctypes.sizeof(regs))
    scratch = int(anchor) if anchor is not None else int(regs.rip)
    saved_code = read_mem(pid, scratch, 8)
    write_mem(pid, scratch, arch._SYSCALL_OP + arch._INT3_OP + b"\x90" * 5)
    regs.rax = arch._NR_MMAP
    regs.rdi = int(hint)
    regs.rsi = size
    regs.rdx = mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC
    regs.r10 = (mmap.MAP_SHARED if shared else mmap.MAP_PRIVATE) | mmap.MAP_ANONYMOUS
    if shared and int(fd) >= 0:
        regs.r10 = mmap.MAP_SHARED
        regs.r8 = int(fd)
        regs.r9 = 0
    else:
        regs.r8 = 0xFFFFFFFFFFFFFFFF
        regs.r9 = 0
    regs.rip = scratch
    setregs(pid, regs)
    ptrace(_PTRACE_CONT, pid, 0, 0)
    waitpid(pid)
    out = getregs(pid)
    addr = int(out.rax)
    write_mem(pid, scratch, saved_code)
    setregs(pid, saved_regs)
    if addr <= 0 or addr > 0xFFFFFFFFFFFF:
        raise OSError(f"remote mmap failed: {addr:#x}")
    return addr


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


class RingReader:
    def __init__(self, path_or_bytes, events):
        self.events = list(events or [])
        if isinstance(path_or_bytes, (bytes, bytearray)):
            self.blob = bytes(path_or_bytes)
            self.path = None
        else:
            self.blob = None
            self.path = str(path_or_bytes)

    def _read(self, offset, size):
        if self.blob is not None:
            return self.blob[offset : offset + size]
        with open(self.path, "rb") as fh:
            fh.seek(offset)
            return fh.read(size)

    def records(self):
        header = self._read(0, arch._TRACK_RING_HDR_SIZE)
        if len(header) < arch._TRACK_RING_HDR_SIZE:
            return []
        magic, ring_head, cap, stride, _n_ev = struct.unpack_from("<QQQQQ", header, 0)
        if magic != arch._TRACK_RING_MAGIC or cap == 0 or stride == 0:
            return []
        cap, stride = int(cap), int(stride)
        count = min(int(ring_head), cap)
        if count <= 0:
            return []
        first = (int(ring_head) - count) % cap
        data = self._read(
            arch._TRACK_RING_HDR_SIZE + first * stride, (cap - first) * stride
        )
        if first + count > cap:
            wrap = (count - cap + first) * stride
            data += self._read(arch._TRACK_RING_HDR_SIZE, wrap)
        base = arch._TRACK_SLOT_EVENTS_OFF
        n_ev = len(self.events)
        out = []
        for seq in range(count):
            off = seq * stride
            if off + stride > len(data):
                break
            (mid32,) = struct.unpack_from("<I", data, off + arch._TRACK_SLOT_ID_OFF)
            (s64,) = struct.unpack_from("<Q", data, off + arch._TRACK_SLOT_SEQ_OFF)
            rec = {"point_id": mid32, "seq": s64}
            for i in range(n_ev):
                at = off + base + 8 * i
                if at + 8 > len(data):
                    break
                (value,) = struct.unpack_from("<Q", data, at)
                rec[self.events[i]] = value
            out.append(rec)
        return out


def profile(
    cmd,
    event=None,
    target=None,
    output=_DEFAULT_OUTPUT,
    buffer_size=_DEFAULT_BUFFER_SIZE,
    filter=None,
):
    if filter is not None:
        if target is not None:
            raise TypeError("pass target=, not both target= and filter=")
        target = filter
    if isinstance(cmd, str):
        import shlex

        cmd = shlex.split(cmd)
    cmd = [str(c) for c in cmd]
    if not cmd:
        raise ValueError("no command to track")
    exec_path = resolve_exec(cmd[0])
    if is_shared_library(exec_path):
        raise ValueError(
            f"cannot profile {cmd[0]!r}: it is a shared library; "
            "profile an executable instead (e.g. '-- /usr/bin/ls')"
        )
    ev_list = _split_list(event) or ["cycles"]
    ev_list = with_leader(ev_list)
    try:
        _force_typ = choose_type(ev_list)
    except Exception:
        _force_typ = None
    _is_group = is_group(ev_list)
    selected = _resolve_track_points(exec_path, target)
    if not selected:
        if target:
            raise SystemExit("no track points left after applying --target")
        raise SystemExit(
            f"no {_DEFAULT_TARGET!r} found in {exec_path!r}; "
            'pass target= (e.g. target=["hot"] or target=[("hot", "cold")]) '
            "or -t on the CLI (e.g. -t fizz_buzz)"
        )
    ring_fd, ring_path = tempfile.mkstemp(prefix="perf-track-", suffix=".ring")
    try:
        os.set_inheritable(ring_fd, True)
    except Exception:
        pass
    lay = _init_ring_file(ring_path, ev_list, buffer_size)

    pid = os.fork()
    if pid == 0:
        try:
            os.set_inheritable(ring_fd, True)
            ptrace(_PTRACE_TRACEME, 0, 0, 0)
            os.kill(os.getpid(), _SIGSTOP)
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
        ptrace(_PTRACE_SETOPTIONS, pid, 0, _PTRACE_O_TRACEEXEC)
        ptrace(_PTRACE_CONT, pid, 0, 0)
        waitpid(pid)
        entry = int(getregs(pid).rip)
        saved_entry = read_mem(pid, entry, 8)
        write_mem(pid, entry, arch._INT3_OP)
        ptrace(_PTRACE_CONT, pid, 0, 0)
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
            except Exception as ex:
                _warn_calibration(ex)
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
                blob = arch.build_profile_func_exit(
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
                        blob = arch.build_profile_func_entry_raw(
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
                        blob = arch.build_profile_detour_raw(
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
                plen = int(patch_len_by_mid.get(mid, arch._PATCH_SIZE))
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
                plen = int(patch_len_by_mid.get(mid, arch._PATCH_SIZE))
                if plen > arch._PATCH_SIZE:
                    write_mem(
                        pid,
                        rt + arch._PATCH_SIZE,
                        b"\x90" * (plen - arch._PATCH_SIZE),
                    )
            write_mem(pid, entry, saved_entry)
            setregs(pid, frozen)

            ptrace(_PTRACE_DETACH, pid, 0, 0)
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
                drop_last=True,
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
        try:
            os.unlink(ring_path)
        except OSError:
            pass


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
    if cpu is None:
        try:
            cpu = sorted(os.sched_getaffinity(0))[0]
        except OSError:
            cpu = None
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
    except Exception as ex:
        _warn_calibration(ex)
        return _zero_overhead(ev_list)
    if not parsed:
        _warn_calibration("the calibration harness returned no data")
        return _zero_overhead(ev_list)
    return {
        k: {e: int(parsed.get(k, {}).get(e, 0)) for e in ev_list} for k in _TRACK_KINDS
    }


@functools.cache
def _libc():
    return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def _find_trackable(exec_path, with_functions=True):
    if with_functions:
        import angr

        project = angr.Project(
            resolve_exec(exec_path), auto_load_libs=False, load_debug_info=False
        )
    else:
        project = _info.elf_project(exec_path)
    lbls = [{"name": name, "addr": int(addr)} for name, addr in labels(project)]
    funcs = {}
    if with_functions:
        try:
            found, _ = functions(project)
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
    return {"labels": lbls, "functions": funcs}


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


def _resolve_single_side(side, lbls, for_begin=True):
    s = str(side).strip() if side is not None else ""
    if not s:
        return None
    for lb in lbls or []:
        if str(lb.get("name", "")) == s:
            return {"name": s, "addr": int(lb["addr"]), "kind": "label"}
    hexed = _parse_hex_addr(s)
    if hexed is not None:
        return {"name": s, "addr": int(hexed), "kind": "label"}
    return None


def _resolve_track_points(exec_path, patterns):
    real = resolve_exec(exec_path)
    if patterns is None or (isinstance(patterns, (list, tuple)) and not patterns):
        patterns = [_DEFAULT_TARGET]
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
                    f"invalid target {p!r}; expected 'name' or ('begin', 'end')"
                )
            pats.append((parts[0], parts[1]))
            continue
        s = str(p).strip()
        if not s:
            continue
        if ".." in s:
            raise ValueError(
                f"invalid target {s!r}; use ('begin', 'end') instead of 'begin..end'"
            )
        pats.append(s)
    if not pats:
        return []
    lbls = _find_trackable(real, with_functions=False).get("labels") or []
    funcs = None
    points = []
    for pat in pats:
        if isinstance(pat, (list, tuple)):
            b, e = str(pat[0]).strip(), str(pat[1]).strip()
            rb = _resolve_single_side(b, lbls, for_begin=True)
            re_ = _resolve_single_side(e, lbls, for_begin=False)
            if rb is None or re_ is None:
                continue
            points.append(rb)
            points.append(re_)
            continue
        hit_label = None
        for lb in lbls:
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


def _init_ring_file(path, n_events, slots):
    lay = arch.profile_ring_layout(len(list(n_events or [])), slots)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.ftruncate(fd, lay["total"])
        mm = mmap.mmap(fd, lay["total"], access=mmap.ACCESS_WRITE)
        try:
            struct.pack_into(
                "<QQQQQ",
                mm,
                0,
                arch._TRACK_RING_MAGIC,
                0,
                lay["capacity"],
                lay["stride"],
                len(list(n_events or [])),
            )
            mm.flush()
        finally:
            mm.close()
    finally:
        os.close(fd)
    return lay


def _dump_results(
    ring_path, ev_list, exec_path, cmd, selected, output, overhead=None, drop_last=False
):
    records = RingReader(ring_path, ev_list).records()
    if drop_last and records:
        records = records[:-1]
    id2m = {i: m for i, m in enumerate(selected)}
    hits = [
        {
            "name": id2m.get(r["point_id"], {}).get("name", r["point_id"]),
            "kind": id2m.get(r["point_id"], {}).get("kind", "label"),
            "seq": int(r["seq"]),
            "address": _point_address(id2m.get(r["point_id"], {})),
            **{e: int(r.get(e, 0)) for e in ev_list},
        }
        for r in records
    ]
    rows = _pair_deltas(hits, ev_list, overhead)
    df = _rows_to_df(rows, ev_list)
    if "duration_time" in df.columns:
        try:
            _freq = float(_info.cpuinfo(["freq"]).iloc[0]["freq"])
        except Exception:
            _freq = 0.0
        if _freq and _freq > 0:
            df["duration_time"] = df["duration_time"] * 1e9 / _freq
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
            out_path = out_path / _DEFAULT_OUTPUT
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


def _warn_calibration(ex):
    print(
        f"cannot calibrate the tracking overhead: {ex}; recording overhead is "
        "reported as 0 and stays in every sample",
        file=sys.stderr,
    )


def _zero_overhead(ev_list):
    return {kind: {e: 0 for e in ev_list} for kind in _TRACK_KINDS}


def _measure_pairs(fn, ev_list, ring, stride, trials):
    base = arch._TRACK_SLOT_EVENTS_OFF
    try:
        cap = int(struct.unpack_from("<Q", ring, 16)[0])
        mask = cap - 1 if cap > 0 and not (cap & (cap - 1)) else None
    except Exception:
        mask = None
    try:
        struct.pack_into("<Q", ring, arch._TRACK_RING_HEAD_OFF, 0)
    except Exception:
        pass
    deltas = {e: [] for e in ev_list}
    for trial in range(max(int(trials), 1)):
        fn()
        try:
            head = struct.unpack_from("<Q", ring, arch._TRACK_RING_HEAD_OFF)[0]
        except Exception:
            head = 0
        if head != 2 * (trial + 1):
            continue
        if mask is None:
            first, second = 2 * trial, 2 * trial + 1
        else:
            first, second = (head - 2) & mask, (head - 1) & mask
        for i, e in enumerate(ev_list):
            off0 = arch._TRACK_RING_DATA_OFF + first * stride + base + 8 * i
            off1 = arch._TRACK_RING_DATA_OFF + second * stride + base + 8 * i
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
    stub_len = arch._PATCH_SIZE
    stub_addr, addr1, addr_ret = exec_addr, exec_addr, exec_addr
    b0, b1 = b"", b""
    for _ in range(4):
        nb0 = arch.build_profile_detour(
            0, ev_list, indices, ring_addr, exec_addr, stub_addr, layout, b"", True
        )
        nb1 = arch.build_profile_detour(
            1, ev_list, indices, ring_addr, addr1, addr_ret, layout, b"", True
        )
        nstub = exec_addr + len(nb0)
        naddr1 = (nstub + stub_len + 15) & ~15
        nret = naddr1 + len(nb1)
        b0, b1 = nb0, nb1
        if (nstub, naddr1, nret) == (stub_addr, addr1, addr_ret):
            break
        stub_addr, addr1, addr_ret = nstub, naddr1, nret
    if exec_addr + len(b0) != stub_addr or addr1 + len(b1) != addr_ret:
        raise RuntimeError("calibration layout did not converge")
    stub = arch.patch_jmp(stub_addr, addr1)
    pad = addr1 - (stub_addr + stub_len)
    if pad < 0:
        raise RuntimeError("calibration layout did not converge")
    return b0 + stub + b"\x90" * pad + b1 + b"\xc3"


def _build_func_calibration(ev_list, indices, ring_addr, exec_addr, layout, shadow_top):
    stub_len = arch._PATCH_SIZE
    nops = b"\x90" * stub_len
    site_addr, resume, exit_addr = exec_addr, exec_addr, exec_addr
    entry, exit_ = b"", b""
    for _ in range(4):
        nentry = arch.build_profile_func_entry_raw(
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
        nexit = arch.build_profile_func_exit(
            1, ev_list, indices, ring_addr, exit_addr, layout, shadow_top=shadow_top
        )
        nsite = exec_addr + len(nentry)
        nresume = nsite + stub_len
        nexit_addr = (nresume + 1 + 15) & ~15
        entry, exit_ = nentry, nexit
        if (nsite, nresume, nexit_addr) == (site_addr, resume, exit_addr):
            break
        site_addr, resume, exit_addr = nsite, nresume, nexit_addr
    if exec_addr + len(entry) != site_addr:
        raise RuntimeError("calibration layout did not converge")
    pad = exit_addr - (resume + 1)
    if pad < 0:
        raise RuntimeError("calibration layout did not converge")
    return entry + nops + b"\xc3" + b"\x90" * pad + exit_


def _calibrate_overhead(ev_list, force_typ, group, trials=None, layout=None):
    ev_list = list(ev_list or [])
    if not ev_list:
        return {}
    trials = _CALIBRATION_TRIALS if trials is None else max(1, int(trials))
    if layout is None:
        lay = arch.profile_ring_layout(len(ev_list), 2)
    else:
        lay = dict(layout)
    stride = lay["stride"]
    total = arch._TRACK_RING_DATA_OFF + 2 * max(int(trials), 1) * int(stride)
    ring = ctypes.create_string_buffer(total)
    ring_addr = ctypes.addressof(ring)
    struct.pack_into(
        "<QQQQQ",
        ring,
        0,
        arch._TRACK_RING_MAGIC,
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
        exec_addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    except Exception:
        mm.close()
        raise RuntimeError("cannot address executable calibration page")
    shpage = None
    shadow_top = 0
    if _TRACK_KIND_FUNC in _TRACK_KINDS:
        shpage = mmap.mmap(-1, 0x10000, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        try:
            shadow_top = ctypes.addressof(ctypes.c_char.from_buffer(shpage))
            shadow_top = ((shadow_top + 0xFFF) & ~0xFFF) or shadow_top
            struct.pack_into("<Q", shpage, 0, shadow_top + 8)
        except Exception:
            shadow_top = 0
    counters, fresh_indices = open_counters(ev_list, force_typ, group, pid=0)
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
            fn = ctypes.CFUNCTYPE(None)(exec_addr)
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
        row = {
            "name": rname,
            "samples": sample,
            "operations": 1,
            **{e: _delta(b, h, e) for e in ev_list},
        }
        if b.get("address") is not None:
            row["address"] = b.get("address")
        return row

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


def _point_address(point):
    try:
        return int((point or {}).get("addr"))
    except (TypeError, ValueError):
        return None


def _rows_to_df(rows, ev_list):
    import pandas as pd

    columns = ["name", "samples", "operations", *list(ev_list or [])]
    if any(isinstance(r, dict) and r.get("address") is not None for r in rows or []):
        columns.append("address")
    return pd.DataFrame(rows, columns=columns)


def _payload(exec_path, cmd, event, output, overhead=None):
    from datetime import datetime

    try:
        from .bench import _content_hash8, _id_hash
    except Exception:
        _content_hash8, _id_hash = None, None
    asked = str(cmd[0]) if cmd else exec_path
    try:
        base = os.path.basename(str(asked or exec_path))
    except Exception:
        base = str(cmd[0]) if cmd else "track"
    sha = None
    if _content_hash8 is not None:
        try:
            sha = _content_hash8(str(asked))
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
        cpu = _info.cpuinfo(list(_info._CPUINFO_FIELDS)).iloc[0].to_dict()
    except Exception:
        cpu = {}
    try:
        host = _info.hostname()
    except Exception:
        host = None
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "file": file_label,
        "name": top_name,
        "id": run_id,
        "time": now,
        "mode": "record",
        "info": {"cpu": cpu, "hostname": host},
        "config": config,
        "output": list(output or []),
    }


def _interrupt_handler(signum, _frame):
    raise KeyboardInterrupt(f"received signal {signum}")
