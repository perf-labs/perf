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
import sys
import time

import pandas as pd
from elftools.elf.elffile import ELFFile

from .arch import arch as _arch_fn
from .core import _parse_cpu_list, _read_text, demangle, one_line
from .exec import resolve_exec

_PROC_CPUINFO = "/proc/cpuinfo"
_PROC_CPUINFO_KEYS = {
    "vendor_id": ("vendor_id", str),
    "model name": ("brand_raw", str),
    "cpu family": ("family", int),
    "model": ("model", int),
    "stepping": ("stepping", int),
    "cpu cores": ("count", int),
}
_CPUINFO_FIELDS = (
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
    "L1i",
    "L1d",
    "L2",
    "L3",
)
_METADATA_COLUMNS = ("kind", "begin", "end", "size", "name")
_FUNCTIONS_MEMO = {}
_FUNCTIONS_MEMO_MAX = 8
_METADATA_ADDRESS_COLUMNS = ("begin", "end")
_ASM_SUFFIXES = (".s", ".asm")
_SYNTHETIC_PREFIX = "Unresolvable"
_FUNCTION_TYPES = ("STT_FUNC", "STT_GNU_IFUNC", "STT_LOOS")
_ASM_LABEL = re.compile(r"^([.\w$@]+):", re.M)
_ANGR_MAIN_BASE = 0x400000
_X86_MACHINES = ("x86_64", "amd64")


class ElfProject:
    def __init__(self, path):
        self.loader = _Loader(_MainObject(path))


def elf_project(exec_path):
    if platform.machine().lower() not in _X86_MACHINES:
        raise ValueError(f"no elf fallback for {platform.machine()!r}")
    return ElfProject(resolve_exec(exec_path))


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
    entries.sort(key=lambda entry: entry[1])
    return entries


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
        found, prototypes = _functions_cached(key, symtab)
        return found, dict(prototypes)
    cfg = project.analyses.CFGFast()
    return _analyze_functions(symtab, cfg)


def targets(project, name, funcs=None):
    if ".." in name:
        begin, _, end = name.partition("..")
        begin, end = begin.strip(), end.strip()
        start = _parse_region_addr(begin)
        stop = _parse_region_addr(end)
        if start is None or stop is None:
            label_addrs = dict(_safe_labels(project))
            start = _region_endpoint(begin, label_addrs, {}, True)
            stop = _region_endpoint(end, label_addrs, {}, False)
        if start is None or stop is None:
            if funcs is None:
                try:
                    funcs = functions(project)[0]
                except Exception:
                    funcs = {}
            start = _region_endpoint(begin, label_addrs, funcs, True)
            stop = _region_endpoint(end, label_addrs, funcs, False)
        if start is not None and stop is not None:
            yield _span(project, name, start, stop)
        return
    if funcs is None:
        try:
            funcs, _ = functions(project)
        except Exception:
            funcs = {}
    entries = funcs or {}
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
        try:
            addr = int(short, 0)
        except (TypeError, ValueError):
            addr = None
        if addr is not None:
            for label, (start, end) in entries.items():
                try:
                    if int(start) <= addr < int(end):
                        yield (demangle(label), start, end)
                        return
                except (TypeError, ValueError):
                    continue
            yield (short, addr, addr + 1)
    return


def metadata(file):
    project = _project(file)
    lbls = labels(project)
    funcs, _ = functions(project)

    for base, start, stop in _inverted_pairs(lbls):
        print(
            f"warning: '{base}_end' (0x{stop:x}) precedes '{base}_begin' "
            f"(0x{start:x}); the compiler moved the labels, so "
            f"'{base}_begin..{base}_end' measures the span between them",
            file=sys.stderr,
        )

    for name, addrs in _cloned_labels(lbls):
        places = ", ".join(f"0x{a:x}" for a in addrs)
        print(
            f"warning: '{name}' is marked {len(addrs)} times ({places}); the "
            f"compiler cloned the block, so a region ending at '{name}' spans "
            f"to its last copy",
            file=sys.stderr,
        )

    records = [
        {
            "kind": "label",
            "begin": addr,
            "end": addr,
            "size": pd.NA,
            "name": one_line(name),
        }
        for name, addr in lbls
    ]
    seen_funcs = set()
    for name, (start, end) in funcs.items():
        if (start, end) in seen_funcs:
            continue
        seen_funcs.add((start, end))
        records.append(
            {
                "kind": "func",
                "begin": start,
                "end": end,
                "size": end - start,
                "name": one_line(demangle(name)),
            }
        )
    df = pd.DataFrame(records, columns=list(_METADATA_COLUMNS))
    df["size"] = df["size"].astype("Int64")
    return df


def hex_addresses(df, columns=_METADATA_ADDRESS_COLUMNS):
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[col] = out[col].apply(lambda v: f"0x{int(v):x}")
    return out


def is_asm_source(path):
    return str(path).lower().endswith(_ASM_SUFFIXES)


def asm_labels(file):
    text = _asm_read(file)
    found = [(m.group(1), m.start(), m.end()) for m in _ASM_LABEL.finditer(text)]
    out = []
    for index, (name, _start, position) in enumerate(found):
        stop = found[index + 1][1] if index + 1 < len(found) else len(text)
        out.append((name, position, stop - position))
    return out


def cpuinfo(fields=None):
    cols = list(fields) if fields is not None else list(_CPUINFO_FIELDS)
    return pd.DataFrame(_cpuinfo_rows()).reindex(columns=cols)


@functools.lru_cache(maxsize=4096)
def format_hz(hz):
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


class _Section:
    __slots__ = ("offset", "filesize")

    def __init__(self, offset, filesize):
        self.offset = offset
        self.filesize = filesize


class _MainObject:
    def __init__(self, path):
        with open(path, "rb") as fh:
            elf = ELFFile(fh)
            self.binary = path
            self.pic = elf.header["e_type"] == "ET_DYN"
            self.mapped_base = _ANGR_MAIN_BASE if self.pic else 0
            self.sections_map = {
                sec.name: _Section(int(sec["sh_offset"]), int(sec["sh_size"]))
                for sec in elf.iter_sections()
            }


class _Loader:
    def __init__(self, main_object):
        self.main_object = main_object


def _asm_read(path):
    src = str(path)
    if not os.path.isfile(src):
        raise ValueError(f"file {src!r} does not exist")
    try:
        with open(src, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as ex:
        raise ValueError(f"file {src!r} cannot be read: {ex}") from ex


def _demangled_names(raw):
    return {raw, demangle(raw)} if "_Z" in str(raw) else {raw}


def _functions_cached(key, symtab):
    hit = _FUNCTIONS_MEMO.get(key)
    if hit is not None:
        return hit
    import angr

    project = angr.Project(key[0], auto_load_libs=False, load_debug_info=False)
    cfg = project.analyses.CFGFast()
    found = _analyze_functions(tuple(symtab), cfg)
    if len(_FUNCTIONS_MEMO) >= _FUNCTIONS_MEMO_MAX:
        _FUNCTIONS_MEMO.clear()
    _FUNCTIONS_MEMO[key] = found
    return found


def _analyze_functions(symtab, cfg):
    by_range = {}
    for addr, size, sym_name in symtab:
        by_range.setdefault((addr, addr + size), sym_name)
    name_to_func = {}
    for func in cfg.kb.functions.values():
        name = func.name
        if not name or name.startswith(_SYNTHETIC_PREFIX):
            continue
        try:
            end = max(b.addr + b.size for b in func.blocks)
        except Exception:
            continue
        by_range.setdefault((func.addr, end), name)
        name_to_func.setdefault(name, func)
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


def _safe_labels(project):
    try:
        return list(labels(project))
    except Exception:
        return []


def _span(project, name, start, stop):
    begin, end = min(start, stop), max(start, stop)
    try:
        base = int(project.loader.main_object.mapped_base or 0)
    except Exception:
        base = 0
    return (
        name,
        begin + base if base and begin < base else begin,
        end + base if base and end < base else end,
    )


def _inverted_pairs(entries):
    addrs = dict(entries)
    out = []
    for name, addr in entries:
        if not name.endswith("_begin"):
            continue
        base = name[: -len("_begin")]
        other = f"{base}_end"
        if other in addrs and addrs[other] < addr:
            out.append((base, addr, addrs[other]))
    return out


def _cloned_labels(entries):
    seen = {}
    for name, addr in entries:
        seen.setdefault(name, []).append(addr)
    return [(name, addrs) for name, addrs in seen.items() if len(addrs) > 1]


def _project(exec_path):
    import angr

    return angr.Project(
        resolve_exec(exec_path), auto_load_libs=False, load_debug_info=False
    )


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
def _cpu_freq():
    return _tsc_calibrate()


@functools.lru_cache(maxsize=4096)
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
    out = _proc_cpuinfo()
    if out.get("brand_raw"):
        out["arch_string_raw"] = platform.machine()
        return out
    import cpuinfo

    try:
        return dict(cpuinfo.get_cpu_info() or {})
    except Exception:
        return {}


@functools.lru_cache(maxsize=1)
def _proc_cpuinfo():
    out = {}
    try:
        with open(_PROC_CPUINFO) as fh:
            for line in fh:
                key, sep, value = line.partition(":")
                if not sep:
                    continue
                entry = _PROC_CPUINFO_KEYS.get(key.strip().lower())
                if entry is None or entry[0] in out:
                    continue
                name, cast = entry
                try:
                    out[name] = cast(value.strip())
                except ValueError:
                    continue
    except OSError:
        return {}
    return out


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
        n = int(_proc_cpuinfo().get("count") or 0)
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
    return {
        "arch": info.get("arch_string_raw"),
        "platform": platform.platform(),
        "vendor": info.get("brand_raw"),
        "model": info.get("model"),
        "family": info.get("family"),
        "stepping": info.get("stepping"),
        "freq": _cpu_freq(),
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
                    if not _is_function_type(typ):
                        continue
                    if shndx in ("SHN_UNDEF", "SHN_ABS", "SHN_COMMON"):
                        continue
                    if not sym.name or size <= 0:
                        continue
                    yield (base + int(sym.entry["st_value"]), size, sym.name)
                break
    except Exception:
        return


def _is_function_type(typ):
    return str(typ) in _FUNCTION_TYPES


def _symtab_only_functions(project):
    by_range = {}
    for addr, size, sym_name in _symtab_functions(project):
        by_range.setdefault((addr, addr + size), sym_name)
    found = {}
    for (start, end), raw in by_range.items():
        for n in _demangled_names(raw):
            found[n] = (start, end)
    return found, {}


@functools.lru_cache(maxsize=4096)
def _parse_region_addr(expr):
    try:
        return int(str(expr).strip(), 0)
    except (TypeError, ValueError):
        return None


def _region_endpoint(expr, label_addrs, entries, is_begin):
    base = str(expr).strip()
    if not base:
        return None
    addr = _parse_region_addr(base)
    if addr is not None:
        return addr
    if base in label_addrs:
        return label_addrs[base]
    if base in entries:
        return entries[base][0 if is_begin else 1]
    short = base.split("(")[0].strip()
    if short and short != base:
        if short in label_addrs:
            return label_addrs[short]
        if short in entries:
            return entries[short][0 if is_begin else 1]
        cands = {
            tuple(se)
            for label, se in entries.items()
            if str(label).split("(")[0].strip() == short
        }
        if len(cands) == 1:
            return next(iter(cands))[0 if is_begin else 1]
    return None


def _pic_base(exec_path):
    try:
        obj = elf_project(exec_path).loader.main_object
    except Exception:
        return False, 0
    pic = bool(getattr(obj, "pic", False))
    return pic, int(obj.mapped_base or 0) if pic else 0
