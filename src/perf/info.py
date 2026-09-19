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
import functools
import glob
import mmap
import os
import platform
import re
import struct
import time
from collections import defaultdict

import cpuinfo as _py_cpuinfo
import pandas as pd
from elftools.elf.elffile import ELFFile

from .arch import arch as _arch_fn
from .core import _parse_cpu_list, _read_text, demangle
from .exec import resolve_exec

CPUINFO_FIELDS = (
    "cpu",
    "core",
    "numa",
    "arch",
    "platform",
    "vendor",
    "model",
    "family",
    "stepping",
    "freq",
    "hz",
    "L1i",
    "L1d",
    "L2",
    "L3",
)


def bin(skip=None):
    skip = {os.path.realpath(p) for p in (skip or ())}
    for d in os.get_exec_path():
        candidate = os.path.join(d, "perf")
        if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
            continue
        if os.path.realpath(candidate) in skip:
            continue
        return candidate
    for candidate in ("/usr/bin/perf", "/usr/sbin/perf"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def labels(project, label=".perf.label"):
    sections_map = project.loader.main_object.sections_map
    entries = []
    if label in sections_map:
        region = sections_map[label]
        obj = project.loader.main_object
        pic = bool(getattr(obj, "pic", False))
        base = int(obj.mapped_base or 0) if pic else 0
        with open(obj.binary, "rb") as fh:
            blob = fh.read()[region.offset : region.offset + region.filesize]
        i, size = 0, len(blob)
        while i < size:
            if i + 8 > size:
                raise ValueError("truncated label address in .perf.label section")
            addr = struct.unpack_from("<Q", blob, i)[0]
            i += struct.calcsize("<Q")
            try:
                end = blob.index(b"\0", i)
            except ValueError as e:
                raise ValueError(
                    "missing null terminator in .perf.label section"
                ) from e
            name = blob[i:end].decode("ascii", "replace")
            i = end + 1
            entries.append((name, base + addr))
    return entries


def _demangled_names(raw):
    return {raw, demangle(raw)} if "_Z" in str(raw) else {raw}


def functions(project, fast=None):
    symtab = list(_symtab_functions(project))
    try:
        fsize = os.path.getsize(project.loader.main_object.binary)
    except Exception:
        fsize = 0
    use_fast = bool(fast) or (
        fast is None and (fsize > 1024 * 1024 or len(symtab) > 2000)
    )
    if use_fast and symtab:
        return _symtab_only_functions(project)
    key = None
    if fast is None:
        try:
            binary = project.loader.main_object.binary
            st = os.stat(binary)
            key = (binary, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        except Exception:
            key = None
    if key is not None:
        found, prototypes = _functions_cached(key, tuple(symtab))
        return found, dict(prototypes)
    cfg = project.analyses.CFGFast()
    return _analyze_functions(symtab, cfg)


@functools.cache
def _functions_cached(key, symtab):
    import angr

    project = angr.Project(key[0], auto_load_libs=False, load_debug_info=False)
    cfg = project.analyses.CFGFast()
    return _analyze_functions(symtab, cfg)


def _analyze_functions(symtab, cfg):
    by_range = {}
    for addr, size, sym_name in symtab:
        by_range.setdefault((addr, addr + size), sym_name)
    name_to_func = {}
    for func in cfg.kb.functions.values():
        try:
            end = max(b.addr + b.size for b in func.blocks)
        except Exception:
            continue
        by_range.setdefault((func.addr, end), func.name)
        name_to_func.setdefault(func.name, func)
    found = {}
    prototypes = {}
    for (start, end), raw in by_range.items():
        names = _demangled_names(raw)
        for n in names:
            found[n] = (start, end)
        f = name_to_func.get(raw)
        if f is not None and getattr(f, "prototype", None):
            for n in names:
                prototypes[n] = f.prototype
    return found, prototypes


def regions(entries):
    pending = defaultdict(list)
    found = {}
    for full, addr in entries:
        parts = full.rsplit("_", 1)
        if len(parts) != 2:
            continue
        name, kind = parts
        if kind not in ("begin", "end"):
            continue
        opposite = "end" if kind == "begin" else "begin"
        match_idx = None
        for i in range(len(pending[name]) - 1, -1, -1):
            if pending[name][i][1] == opposite:
                match_idx = i
                break
        if match_idx is not None:
            other_addr, _ = pending[name].pop(match_idx)
            if kind == "begin":
                found[name] = (addr, other_addr)
            else:
                found[name] = (other_addr, addr)
        else:
            pending[name].append((addr, kind))
    return found


def _safe_labels(project):
    try:
        return list(labels(project))
    except Exception:
        return []


def targets(project, name, funcs=None):
    if ".." in name:
        begin, _, end = name.partition("..")
        begin, end = begin.strip(), end.strip()
        start = _parse_region_addr(begin)
        stop = _parse_region_addr(end)
        if start is not None and stop is not None:
            try:
                _base = int(project.loader.main_object.mapped_base or 0)
            except Exception:
                _base = 0
            if _base and start < _base:
                start += _base
            if _base and stop < _base:
                stop += _base
            yield (name, start, stop)
            return
        label_addrs = dict(_safe_labels(project))
        start = start if start is not None else _resolve_label_only(begin, label_addrs)
        stop = stop if stop is not None else _resolve_label_only(end, label_addrs)
        if start is not None and stop is not None:
            try:
                _base2 = int(project.loader.main_object.mapped_base or 0)
            except Exception:
                _base2 = 0
            if _base2 and start < _base2:
                start += _base2
            if _base2 and stop < _base2:
                stop += _base2
            yield (name, start, stop)
            return
        if funcs is None:
            try:
                funcs, _ = functions(project) if project is not None else ({}, {})
            except Exception:
                funcs = {}
        entries = regions(_safe_labels(project)) | (funcs or {})
        start = (
            start
            if start is not None
            else _resolve_region_endpoint(begin, label_addrs, entries, True)
        )
        stop = (
            stop
            if stop is not None
            else _resolve_region_endpoint(end, label_addrs, entries, False)
        )
        if start is None or stop is None:
            return
        try:
            _base3 = int(project.loader.main_object.mapped_base or 0)
        except Exception:
            _base3 = 0
        if _base3 and start < _base3:
            start += _base3
        if _base3 and stop < _base3:
            stop += _base3
        yield (name, start, stop)
        return
    if funcs is None:
        try:
            funcs, _ = functions(project)
        except Exception:
            funcs = {}
    entries = regions(_safe_labels(project)) | (funcs or {})
    if name in entries:
        start, end = entries[name]
        yield (demangle(name), start, end)
        return
    short = str(name).split("(")[0].strip()
    if short:
        if short in entries and short != name:
            start, end = entries[short]
            yield (demangle(short), start, end)
            return
        cands = [
            (label, se)
            for label, se in entries.items()
            if str(label).split("(")[0].strip() == short
        ]
        uniq = {tuple(se) for _, se in cands}
        if len(uniq) == 1:
            start, end = next(iter(uniq))
            yield (demangle(cands[0][0]), start, end)
            return
    return


def metadata(file, kind=None):
    if kind is not None and kind not in ("func", "region"):
        raise ValueError(
            f"unknown target kind {kind!r}; expected 'func', 'region', or None"
        )
    import angr

    project = angr.Project(
        resolve_exec(file),
        auto_load_libs=False,
        load_debug_info=False,
    )
    lbls = labels(project)
    funcs, _ = functions(project)
    regs = regions(lbls)

    def _clean(v):
        try:
            return str(v).replace("\r", " ").replace("\n", " ")
        except Exception:
            return v

    records = []
    for name, addr in lbls:
        records.append(
            {
                "kind": "label",
                "name": _clean(name),
                "start": addr,
                "end": addr,
                "size": 1,
            }
        )
    seen_funcs = set()
    for name, (start, end) in funcs.items():
        dn = _clean(demangle(name))
        if (start, end) in seen_funcs:
            continue
        seen_funcs.add((start, end))
        records.append(
            {
                "kind": "func",
                "name": dn,
                "start": start,
                "end": end,
                "size": end - start,
            }
        )
    for name, (start, end) in regs.items():
        records.append(
            {
                "kind": "region",
                "name": _clean(name),
                "start": start,
                "end": end,
                "size": end - start,
            }
        )
    df = pd.DataFrame(records, columns=["kind", "name", "start", "end", "size"])
    if kind == "func":
        df = df[df["kind"] == "func"]
    elif kind == "region":
        df = df[df["kind"].isin(["label", "region"])]
    return df.reset_index(drop=True)


def _cpuinfo_cache_key():
    try:
        return (
            tuple(_online_cpus()),
            id(_online_cpus),
            id(_sysfs_cache_sizes),
            id(_cpu_core_id),
            id(_cpu_numa_node),
        )
    except Exception:
        return None


def _cpuinfo_frame(rows, fields):
    cols = list(fields) if fields is not None else list(CPUINFO_FIELDS)
    return pd.DataFrame(rows).reindex(columns=cols)


def cpuinfo(fields=None):
    return _cpuinfo_frame(_cpuinfo_rows(), fields)


def _cpuinfo_rows():
    key = _cpuinfo_cache_key()
    if key is not None:
        return _cpuinfo_rows_cached(key)
    return _compute_cpuinfo_rows()


@functools.cache
def _cpuinfo_rows_cached(key):
    return _compute_cpuinfo_rows()


def _compute_cpuinfo_rows():
    chip = _chip_info()
    cpus = _online_cpus()
    if not cpus:
        raise RuntimeError("no online CPUs found under /sys/devices/system/cpu")
    rows = []
    for cpu in cpus:
        sizes = _sysfs_cache_sizes(cpu)
        if not sizes:
            raise RuntimeError(
                f"no cache info for cpu{cpu} under "
                f"/sys/devices/system/cpu/cpu{cpu}/cache; "
                "sysfs may be unavailable in this environment"
            )
        rows.append(
            {
                "cpu": int(cpu),
                "core": _cpu_core_id(cpu),
                "numa": _cpu_numa_node(cpu),
                **chip,
                "L1i": sizes.get("L1i"),
                "L1d": sizes.get("L1d"),
                "L2": sizes.get("L2"),
                "L3": sizes.get("L3"),
            }
        )
    if not rows:
        raise RuntimeError("no CPU info rows collected")
    return rows


def _extract_hz(raw):
    for key in ("hz_actual", "hz_advertised"):
        try:
            v = raw.get(key)
            if isinstance(v, (list, tuple)):
                v = v[0] if v else 0
            v = int(v or 0)
            if v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _tsc_calibrate(delay=0.05):
    try:
        code = _arch_fn().tsc_reader_bytes()
        page = mmap.mmap(
            -1,
            len(code),
            prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        )
        page.write(code)
        fn = ctypes.CFUNCTYPE(ctypes.c_uint64)(
            ctypes.addressof(ctypes.c_char.from_buffer(page))
        )
        try:
            t0 = fn()
            p0 = time.perf_counter()
            time.sleep(delay)
            p1 = time.perf_counter()
            t1 = fn()
        finally:
            page.close()
        dt = p1 - p0
        if dt <= 0:
            return None
        hz = int((t1 - t0) / dt)
        return hz if hz > 0 else None
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _cpu_hz():
    return _tsc_calibrate() or _extract_hz(_cpuinfo())


@functools.cache
def _format_hz(hz):
    try:
        hz = int(hz)
    except (TypeError, ValueError):
        return None
    if hz <= 0:
        return None
    if hz >= 1_000_000_000:
        return f"{hz / 1_000_000_000:.1f}Ghz"
    if hz >= 1_000_000:
        return f"{hz / 1_000_000:.0f}Mhz"
    if hz >= 1_000:
        return f"{hz / 1_000:.0f}Khz"
    return f"{hz}Hz"


@functools.cache
def _format_size(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        m = re.fullmatch(r"([\d.]+)\s*([KMGT]?i?(?:[Bb])?)", s)
        if m:
            num, unit = m.group(1), (m.group(2) or "").lower()
            try:
                f = float(num)
                num_s = f"{f:g}"
            except (TypeError, ValueError):
                num_s = num
            if not unit:
                return num_s or None
            first = unit[0] if unit else "b"
            mapping = {"k": "Kb", "m": "Mb", "g": "Gb", "t": "Tb", "b": "b"}
            return f"{num_s}{mapping.get(first, unit)}"
        try:
            v = int(s, 0)
        except (TypeError, ValueError):
            try:
                v = int(float(s))
            except (TypeError, ValueError):
                return s or None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    if n >= 1024**3:
        return f"{n / 1024**3:g}Gb"
    if n >= 1024**2:
        return f"{n / 1024**2:g}Mb"
    if n >= 1024:
        return f"{n / 1024:g}Kb"
    return f"{n}b"


@functools.lru_cache(maxsize=1)
def _cpuinfo():
    try:
        return dict(_py_cpuinfo.get_cpu_info() or {})
    except Exception:
        return {}


def _online_cpus():
    txt = _read_text("/sys/devices/system/cpu/online")
    cpus = _parse_cpu_list(txt) if txt else []
    if cpus:
        return cpus
    try:
        found = []
        for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*"):
            try:
                found.append(int(os.path.basename(path)[3:]))
            except ValueError:
                continue
        if found:
            return sorted(found)
    except Exception:
        pass
    try:
        n = int((_py_cpuinfo.get_cpu_info() or {}).get("count") or 0)
        if n > 0:
            return list(range(n))
    except Exception:
        pass
    return [0]


def _cpu_core_id(cpu):
    txt = _read_text(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id")
    if txt:
        try:
            return int(txt.split()[0], 0)
        except (TypeError, ValueError):
            pass
    txt = _read_text(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_siblings_list")
    if txt:
        try:
            siblings = _parse_cpu_list(txt)
            if siblings:
                return int(siblings[0])
        except (TypeError, ValueError):
            pass
    return int(cpu)


def _cpu_numa_node(cpu):
    try:
        for node in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
            try:
                nid = int(os.path.basename(node)[4:])
            except ValueError:
                continue
            for cand in (
                os.path.join(node, "cpulist"),
                os.path.join(node, "cpumap"),
            ):
                txt = _read_text(cand)
                if txt and cpu in _parse_cpu_list(txt):
                    return nid
            if os.path.exists(os.path.join(node, f"cpu{cpu}")):
                return nid
    except Exception:
        pass
    return 0


def _sysfs_cache_sizes(cpu):
    out = {}
    try:
        indices = sorted(
            glob.glob(f"/sys/devices/system/cpu/cpu{cpu}/cache/index[0-9]*")
        )
    except Exception:
        indices = []
    for idx in indices:
        level = _read_text(os.path.join(idx, "level"))
        typ = _read_text(os.path.join(idx, "type"))
        size = _read_text(os.path.join(idx, "size"))
        if not level or not size:
            continue
        try:
            lv = int(str(level).strip().split()[0], 0)
        except (TypeError, ValueError):
            continue
        t = str(typ or "Unified").strip().lower()
        if lv == 1 and t.startswith("data"):
            out["L1d"] = _format_size(size)
        elif lv == 1 and t.startswith("instruction"):
            out["L1i"] = _format_size(size)
        elif lv == 2:
            out["L2"] = _format_size(size)
        elif lv == 3:
            out["L3"] = _format_size(size)
    return out


def _chip_info():
    info = _cpuinfo()
    hz = _cpu_hz()
    return {
        "arch": info.get("arch_string_raw"),
        "platform": platform.platform(),
        "vendor": info.get("brand_raw"),
        "model": info.get("model"),
        "family": info.get("family"),
        "stepping": info.get("stepping"),
        "freq": _format_hz(hz),
        "hz": hz,
    }


def _symtab_functions(project):
    try:
        main_obj = project.loader.main_object
        base = int(main_obj.mapped_base or 0) if getattr(main_obj, "pic", False) else 0
        with open(main_obj.binary, "rb") as fh:
            elf = ELFFile(fh)
            for sec_name in (".symtab", ".dynsym"):
                sec = elf.get_section_by_name(sec_name)
                if sec is None:
                    continue
                for sym in sec.iter_symbols():
                    try:
                        typ = sym.entry["st_info"]["type"]
                        shndx = sym.entry["st_shndx"]
                        size = int(sym.entry["st_size"] or 0)
                    except Exception:
                        continue
                    if typ != "STT_FUNC":
                        continue
                    if shndx in ("SHN_UNDEF", "SHN_ABS", "SHN_COMMON"):
                        continue
                    if not sym.name or size <= 0:
                        continue
                    yield (base + int(sym.entry["st_value"]), size, sym.name)
                break
    except Exception:
        return


def _symtab_only_functions(project):
    by_range = {}
    for addr, size, sym_name in _symtab_functions(project):
        by_range.setdefault((addr, addr + size), sym_name)
    found = {}
    for (start, end), raw in by_range.items():
        for n in _demangled_names(raw):
            found[n] = (start, end)
    return found, {}


@functools.cache
def _parse_region_addr(expr):
    try:
        return int(str(expr).strip(), 0)
    except (TypeError, ValueError):
        return None


def _resolve_region_base(base, label_addrs, entries, is_begin):
    addr = _parse_region_addr(base)
    if addr is not None:
        return addr
    if base in label_addrs:
        return label_addrs[base]
    if base in entries:
        start, end = entries[base]
        return start if is_begin else end
    short = str(base).split("(")[0].strip()
    if short:
        if short in label_addrs and short != base:
            return label_addrs[short]
        if short in entries and short != base:
            start, end = entries[short]
            return start if is_begin else end
        cands = {
            tuple(se)
            for label, se in entries.items()
            if str(label).split("(")[0].strip() == short
        }
        if len(cands) == 1:
            start, end = next(iter(cands))
            return start if is_begin else end
    return None


def _resolve_region_endpoint(expr, label_addrs, entries, is_begin):
    s = str(expr).strip()
    if not s:
        return None
    addr = _parse_region_addr(s)
    if addr is not None:
        return addr
    return _resolve_region_base(s, label_addrs, entries, is_begin)


def _resolve_label_only(expr, label_addrs):
    s = str(expr).strip()
    if not s:
        return None
    addr = _parse_region_addr(s)
    if addr is not None:
        return addr
    return label_addrs.get(s)


def _project(exec_path):
    import angr

    return angr.Project(
        resolve_exec(exec_path), auto_load_libs=False, load_debug_info=False
    )


def _pic_base(exec_path):
    try:
        obj = _project(exec_path).loader.main_object
    except Exception:
        return False, 0
    pic = bool(getattr(obj, "pic", False))
    return pic, int(obj.mapped_base or 0) if pic else 0
