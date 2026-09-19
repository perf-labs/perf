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

import atexit
import contextlib
import ctypes
import ctypes.util
import functools
import io
import mmap
import os
import random
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile

from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection

from .arch import arch as get_arch
from .core import demangle

_ARCH = get_arch()
_SHN_UNDEF = 0
_AT_NULL = 0
_AT_PHDR = 3
_AT_PHENT = 4
_AT_PHNUM = 5
_AT_PAGESZ = 6
_AT_ENTRY = 9
_ARGV_GAP = 0x10000
_MAP_FIXED = 0x10
_MAP_FIXED_NOREPLACE = 0x100000
_PERF_MAP = "/tmp/perf-{pid}.map"
_OBJ_LINK_DEPS = []
_EXEC_LINKS = {}
_SHARED_LIBS = {}
_EXIT_HOOKS = {}
_REGISTERED_EXITS = []
_ET_REL = 1
_SHT_NULL, _SHT_PROGBITS, _SHT_SYMTAB, _SHT_STRTAB, _SHT_RELA = 0, 1, 2, 3, 4
_SHF_WRITE, _SHF_ALLOC, _SHF_EXECINSTR = 0x1, 0x2, 0x4
_STB_LOCAL, _STB_GLOBAL = 0, 1
_STT_NOTYPE, _STT_OBJECT, _STT_FUNC, _STT_SECTION, _STT_IFUNC = 0, 1, 2, 3, 10


class ElfConst:
    R_X86_64_64 = _ARCH._R_X86_64_64
    R_X86_64_COPY = _ARCH._R_X86_64_COPY
    R_X86_64_GLOB_DAT = _ARCH._R_X86_64_GLOB_DAT
    R_X86_64_JUMP_SLOT = _ARCH._R_X86_64_JUMP_SLOT
    R_X86_64_RELATIVE = _ARCH._R_X86_64_RELATIVE
    R_X86_64_IRELATIVE = _ARCH._R_X86_64_IRELATIVE
    ASLR_DISABLED = 0x40000
    _PAGE_SIZE = mmap.PAGESIZE


class Elf:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    mmap = libc.mmap
    mmap.restype = ctypes.c_void_p
    mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]

    def __init__(self, loader):
        self.loader = loader
        obj = getattr(loader, "main_object", None)
        self.arch = get_arch(getattr(getattr(obj, "arch", None), "name", None))
        self.runtime_base = 0
        self.mapped_regions = []
        self.applied_relocs = []
        self.moved_functions = {}
        self.layout_names = {}
        self.ifuncs = {}
        self.initializers_done = False
        self.ran_initializers = []

    @staticmethod
    def _disable_aslr():
        try:
            Elf.libc.personality(ElfConst.ASLR_DISABLED)
        except Exception:
            pass

    def map_elf(self):
        self._disable_aslr()
        obj = self.loader.main_object

        with open(obj.binary, "rb") as f:
            data = f.read()

        stream = io.BytesIO(data)
        elf_hdr = ELFFile(stream).header
        is_pie = elf_hdr["e_type"] == "ET_DYN"

        min_addr = float("inf")
        max_addr = 0
        for seg in obj.segments:
            is_loadable = True
            try:
                if hasattr(seg, "type"):
                    is_loadable = seg.type == "PT_LOAD"
                elif hasattr(seg, "header"):
                    is_loadable = seg.header.p_type == "PT_LOAD"
            except Exception:
                pass

            if not is_loadable:
                continue

            vaddr = seg.vaddr
            memsz = seg.memsize
            min_addr = min(min_addr, vaddr)
            max_addr = max(max_addr, vaddr + memsz)

        if min_addr == float("inf"):
            raise ValueError("No PT_LOAD segments found")

        start = align_down(min_addr)
        end = align_up(max_addr)
        total_size = end - start

        map_size = total_size
        if is_pie:
            base = obj.mapped_base
            assert base != 0, "mapped_base must be non-zero"
            map_start = base + start
        else:
            map_start = start
            taken = _live_overlap(map_start, map_size)
            if taken is not None:
                lo, hi = taken
                asked = getattr(self, "requested", None)
                where = (
                    f"{obj.binary!r} (resolved from {str(asked)!r})"
                    if asked and str(asked) != str(obj.binary)
                    else f"{obj.binary!r}"
                )
                raise ValueError(
                    f"cannot map {where} at 0x{map_start:x}-"
                    f"0x{map_start + map_size:x}: this process is already "
                    f"using 0x{lo:x}-0x{hi:x}. A non-PIE executable can only be "
                    "mapped at the address it is linked at, so benchmark a "
                    "position-independent build of it (gcc -fPIE -pie, or "
                    "clang -fPIE -pie), or a function of a program that is "
                    "built that way. Distro compilers are shipped non-PIE, so "
                    "this is expected for /usr/bin/gcc and other system tools; "
                    "perf info still lists their targets, and a compiler built "
                    "with -fPIE -pie can be benchmarked"
                )

        def _mmap_at(hint, at_hint):
            flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
            if at_hint:
                flags |= _MAP_FIXED_NOREPLACE
            return self.mmap(
                hint,
                map_size,
                mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                flags,
                -1,
                0,
            )

        addr = None
        if is_pie:
            addr = _mmap_at(map_start, at_hint=True)
            if addr == ctypes.c_void_p(-1).value:
                addr = _mmap_at(0, at_hint=False)
        else:
            addr = self.mmap(
                map_start,
                map_size,
                mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_FIXED,
                -1,
                0,
            )
        if addr == ctypes.c_void_p(-1).value or addr is None:
            raise OSError(ctypes.get_errno(), "mmap failed")
        self.mapped_regions.append((addr, map_size))

        for seg in obj.segments:
            is_loadable = True
            try:
                if hasattr(seg, "type"):
                    is_loadable = seg.type == "PT_LOAD"
            except Exception:
                pass

            if not is_loadable:
                continue

            dest = addr + seg.vaddr - start

            if seg.filesize > 0:
                file_offset = getattr(seg, "offset", getattr(seg, "header", None))
                if file_offset is None or not hasattr(file_offset, "p_offset"):
                    file_off = seg.vaddr - min_addr
                else:
                    file_off = (
                        file_offset.p_offset if hasattr(file_offset, "p_offset") else 0
                    )

                chunk = data[file_off : file_off + seg.filesize]
                ctypes.memmove(dest, chunk, len(chunk))

            if seg.memsize > seg.filesize:
                bss_start = dest + seg.filesize
                bss_size = seg.memsize - seg.filesize
                ctypes.memset(bss_start, 0, bss_size)

        self.runtime_base = addr
        self.is_pie = is_pie
        self.start = start

        self.load_dependencies(obj.binary)
        self.apply_relocations(obj.binary)
        self.resolve_ifuncs()

        return self

    def load_dependencies(self, binary_path):
        return load_shared_libraries(shared_libraries(binary_path))

    def initializers(self):
        try:
            return initializers(self.loader.main_object.binary)
        except (OSError, ValueError):
            return []

    def run_initializers(self, output=None):
        if self.initializers_done:
            return self.ran_initializers
        self.initializers_done = True
        try:
            binary = self.loader.main_object.binary
        except Exception:
            binary = None
        if binary and is_shared_library(binary):
            return []
        entries = self.initializers()
        if not entries:
            return []
        addresses = []
        for faddr in entries:
            try:
                addresses.append(self.runtime_addr(faddr))
            except ValueError:
                continue
        if not addresses:
            return []
        ok = True
        for addr in addresses:
            try:
                with _output_sink(output):
                    call_native(addr)
            except Exception as e:
                print(
                    f"Warning: initializer {addr:#x} failed: {e}",
                    file=sys.stderr,
                )
                ok = False
        if ok:
            self.ran_initializers = addresses
        return addresses if ok else []

    def apply_relocations(self, binary_path):
        with open(binary_path, "rb") as f:
            data = f.read()

        stream = io.BytesIO(data)
        elffile = ELFFile(stream)
        base = self.runtime_base
        is_pie = self.is_pie

        def resolve_symbol(sym):
            if sym is None:
                return 0
            val = sym.entry["st_value"]
            return base + val if is_pie else val

        self.applied_relocs = []
        for section in elffile.iter_sections():
            if not isinstance(section, RelocationSection):
                continue
            symtab = elffile.get_section(section["sh_link"])
            for reloc in section.iter_relocations():
                sym = symtab.get_symbol(reloc["r_info_sym"])
                S = resolve_symbol(sym) if sym else 0
                A = reloc["r_addend"] if reloc.is_RELA() else 0
                P = base + reloc["r_offset"] if is_pie else reloc["r_offset"]
                rel_type = reloc["r_info_type"]

                try:
                    shndx = sym.entry["st_shndx"] if sym is not None else "SHN_UNDEF"
                except Exception:
                    shndx = "SHN_UNDEF"
                try:
                    sval = int(sym.entry["st_value"]) if sym is not None else None
                except Exception:
                    sval = None
                rec = {
                    "offset": int(reloc["r_offset"]),
                    "type": int(rel_type),
                    "sym": sym.name if sym is not None else None,
                    "value": sval,
                    "undef": shndx in ("SHN_UNDEF", 0),
                    "addend": int(A),
                }

                if rel_type == ElfConst.R_X86_64_RELATIVE:
                    value = base + A if is_pie else A
                elif rel_type in (
                    ElfConst.R_X86_64_GLOB_DAT,
                    ElfConst.R_X86_64_JUMP_SLOT,
                    ElfConst.R_X86_64_64,
                ):
                    if rec.get("undef"):
                        _ext = _resolve_external(sym.name if sym is not None else None)
                        if _ext:
                            value = _ext + A
                        else:
                            value = A
                    else:
                        value = S + A
                elif rel_type == ElfConst.R_X86_64_COPY:
                    self.applied_relocs.append(rec)
                    self._apply_copy(P, sym)
                    continue
                elif rel_type == ElfConst.R_X86_64_IRELATIVE:
                    resolver = base + A if is_pie else A
                    value = self._resolve_ifunc_at(A, resolver)
                else:
                    self.applied_relocs.append(rec)
                    continue

                ctypes.c_uint64.from_address(P).value = value
                self.applied_relocs.append(rec)

    def _resolve_ifunc_at(self, offset, resolver):
        addr = _resolve_external(self.ifunc_names().get(int(offset)))
        return addr or resolver

    def ifunc_names(self):
        cached = getattr(self, "_ifunc_names", None)
        if cached is None:
            cached = {}
            for name, value, _size in _ifunc_symbols(self.loader.main_object.binary):
                cached.setdefault(int(value), name)
            self._ifunc_names = cached
        return cached

    def _apply_copy(self, place, sym):
        src = _resolve_external(sym.name if sym is not None else None)
        size = self.arch._POINTER_SIZE
        try:
            size = int(sym.entry["st_size"]) if sym is not None else size
        except Exception:
            pass
        if src is None or size <= 0:
            print(
                "Warning: no definition for the copied object "
                f"{sym.name if sym is not None else '?'}",
                file=sys.stderr,
            )
            ctypes.c_uint64.from_address(place).value = 0
            return
        try:
            ctypes.memmove(place, src, size)
        except (OSError, ValueError, ctypes.ArgumentError):
            print(
                f"Warning: cannot copy {sym.name if sym is not None else '?'} "
                f"from 0x{src:x}",
                file=sys.stderr,
            )
            ctypes.c_uint64.from_address(place).value = 0

    def get_symbol(self, name):
        resolved = (getattr(self, "ifuncs", {}) or {}).get(name)
        if resolved:
            return resolved
        values, _ = _symtab_values(self.loader.main_object.binary)
        raw = self.raw_symbol(name)
        st_value = values.get(raw) if raw else None
        if st_value is None:
            raise ValueError(f"Symbol {name} not found")

        for fs, (fe, nb) in (getattr(self, "moved_functions", {}) or {}).items():
            if fs <= st_value < fe:
                return nb + (st_value - fs)

        if self.is_pie:
            return self.runtime_base + st_value
        else:
            return st_value

    def raw_symbol(self, name):
        values, demangled = _symtab_values(self.loader.main_object.binary)
        if name in values:
            return name
        if name in demangled:
            return demangled[name]
        short = _short_symbol(name)
        for raw, _value in values.items():
            if short and _short_symbol(demangle(raw)) == short:
                return raw
        return None

    def setup_stack(self, size=0x200000, align=16):
        try:
            size = int(size, 0) if isinstance(size, str) else int(size)
        except (TypeError, ValueError) as e:
            raise ValueError(f"stack size must be an integer, got {size!r}") from e
        if size <= 0:
            raise ValueError(f"stack size must be > 0, got {size!r}")
        try:
            align = int(align, 0) if isinstance(align, str) else int(align)
        except (TypeError, ValueError) as e:
            raise ValueError(f"stack align must be an integer, got {align!r}") from e
        if align < 1 or (align & (align - 1)):
            raise ValueError(f"stack align must be a power of two >= 1, got {align!r}")
        stack = self.mmap(
            0,
            size,
            mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            -1,
            0,
        )
        if stack == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "Stack alloc failed")
        self.mapped_regions.append((stack, size))
        self.stack_base = int(stack)
        self.stack_top = self.arch.process_stack_top(stack, size, align)
        return self.stack_top

    def _offset_to_vaddr(self, offset):
        for seg in self.loader.main_object.segments:
            try:
                p_offset = int(seg.offset)
                p_vaddr = int(seg.vaddr)
                size = int(seg.filesize)
            except Exception:
                continue
            if p_offset <= int(offset) < p_offset + size:
                return self._file_vaddr(p_vaddr + (int(offset) - p_offset))
        return self._file_vaddr(int(offset))

    def auxv(self):
        try:
            with open(self.loader.main_object.binary, "rb") as f:
                header = ELFFile(f).header
            phnum = int(header["e_phnum"])
            phent = int(header["e_phentsize"])
        except Exception:
            return []
        out = []
        if phnum and phent:
            phdr = self.runtime_addr(self._offset_to_vaddr(int(header["e_phoff"])))
            out += [(_AT_PHDR, phdr), (_AT_PHENT, phent), (_AT_PHNUM, phnum)]
        out.append((_AT_PAGESZ, mmap.PAGESIZE))
        try:
            entry = self.runtime_addr(self._offset_to_vaddr(int(header["e_entry"])))
        except ValueError:
            entry = 0
        if entry:
            out.append((_AT_ENTRY, entry))
        out.append((_AT_NULL, 0))
        return out

    def setup_argv(self, argv=(), env=(), gap=_ARGV_GAP):
        if not getattr(self, "stack_top", 0):
            raise ValueError("setup_stack() must run before setup_argv()")
        words = [str(w) for w in argv]
        environ = [str(w) for w in env]
        if not words:
            raise ValueError("argv needs at least argv[0] (the program name)")
        strings = [w.encode() + b"\0" for w in words + environ]
        auxv = self.auxv()
        arch = self.arch
        total = sum(len(s) for s in strings)
        vec = arch._POINTER_SIZE * (3 + len(words) + len(environ) + 2 * len(auxv))
        need = vec + total + arch._POINTER_SIZE
        sp = arch.entry_sp(self.stack_top, max(int(gap), need))
        str_base = sp + vec
        save = str_base + total
        base = int(getattr(self, "stack_base", 0))
        if base and sp < base:
            raise ValueError(
                f"argv leaves {base - sp} bytes of stack below the stack pointer "
                f"but the stack is {int(self.stack_top) - base} bytes in total; "
                "raise --config.stack.size"
            )
        addrs, at = [], str_base
        for text in strings:
            addrs.append(at)
            at += len(text)
        argv_at = addrs[: len(words)]
        env_at = addrs[len(words) :]
        blob = bytearray(arch.pack_pointer(len(words)))
        for slot in argv_at:
            blob += arch.pack_pointer(slot)
        blob += arch.pack_pointer(0)
        for slot in env_at:
            blob += arch.pack_pointer(slot)
        blob += arch.pack_pointer(0)
        for key, value in auxv:
            blob += arch.pack_pointer(key) + arch.pack_pointer(value)
        self.write_memory(sp, bytes(blob))
        self.write_memory(str_base, b"".join(strings))
        self.write_memory(save, arch.pack_pointer(self.stack_top))
        return {
            "argc": len(words),
            "argv": sp + arch._POINTER_SIZE,
            "envp": sp + arch._POINTER_SIZE * (len(words) + 2),
            "auxv": sp + arch._POINTER_SIZE * (len(words) + len(environ) + 3),
            "sp": sp,
            "save": save,
            "strings": str_base,
            "words": words,
            "environ": environ,
        }

    def write_memory(self, addr, data):
        if isinstance(data, str):
            data = data.encode()
        ctypes.memmove(addr, data, len(data))

    def read_memory(self, addr, size):
        buf = ctypes.create_string_buffer(size)
        ctypes.memmove(buf, addr, size)
        return buf.raw

    def _file_vaddr(self, cle_vaddr):
        try:
            base = int(self.loader.main_object.mapped_base or 0)
        except Exception:
            base = 0
        try:
            pie = bool(getattr(self, "is_pie", False))
        except Exception:
            pie = False
        if pie and base:
            return int(cle_vaddr) - base
        return int(cle_vaddr)

    def segment_runtime_ranges(self):
        if not getattr(self, "runtime_base", 0):
            raise RuntimeError("map_elf() must run before querying runtime ranges")
        start = getattr(self, "start", 0)
        base = self.runtime_base
        out = []
        for seg in self.loader.main_object.segments:
            is_loadable = True
            try:
                if hasattr(seg, "type"):
                    is_loadable = seg.type == "PT_LOAD"
            except Exception:
                pass
            if not is_loadable:
                continue
            cle_vaddr = int(seg.vaddr)
            vaddr = self._file_vaddr(cle_vaddr)
            memsz = int(seg.memsize)
            flags = ""
            try:
                raw_flags = getattr(seg, "flags", None)
                if raw_flags is None and hasattr(seg, "header"):
                    raw_flags = seg.header.p_flags
                if isinstance(raw_flags, int):
                    flags = "".join(
                        (
                            "R" if raw_flags & 0x4 else "",
                            "W" if raw_flags & 0x2 else "",
                            "X" if raw_flags & 0x1 else "",
                        )
                    )
                elif raw_flags is not None:
                    flags = str(raw_flags)
            except Exception:
                flags = ""
            out.append((vaddr, memsz, base + cle_vaddr - start, flags))
        return out

    def runtime_addr(self, file_vaddr):
        for v, msz, rt, _fl in self.segment_runtime_ranges():
            if v <= int(file_vaddr) < v + msz:
                return rt + (int(file_vaddr) - v)
        raise ValueError(f"address {int(file_vaddr):#x} outside mapped segments")

    def _text_range(self):
        try:
            with open(self.loader.main_object.binary, "rb") as f:
                sec = ELFFile(f).get_section_by_name(".text")
                if sec is None:
                    return None
                start = int(sec.header["sh_addr"])
                return (start, start + int(sec.header["sh_size"]))
        except Exception:
            return None

    def _relocate_blob(self, code, fs, new_base, reloc_map, md, orig_base):
        fe = fs + len(code)
        arch = self.arch
        size = arch._POINTER_SIZE

        def target_runtime(file_v):
            for s, (e, nb) in reloc_map.items():
                if s <= file_v < e:
                    return nb + (file_v - s)
            return self.runtime_addr(file_v)

        try:
            out = bytearray(
                arch.relocate_code(
                    code,
                    new_base,
                    orig_base,
                    target_runtime,
                    span=(fs, fe),
                    md=md,
                )
            )
        except arch.Unrelocatable as e:
            raise _Unrelocatable() from e

        for rec in getattr(self, "applied_relocs", []) or []:
            p = int(rec["offset"])
            if not (fs <= p < fe) or p + size > fe:
                continue
            rtype = int(rec["type"])
            try:
                if rtype == ElfConst.R_X86_64_RELATIVE:
                    new_val = target_runtime(int(rec["addend"]))
                elif rtype in (
                    ElfConst.R_X86_64_64,
                    ElfConst.R_X86_64_GLOB_DAT,
                    ElfConst.R_X86_64_JUMP_SLOT,
                ):
                    if rec.get("undef") or rec.get("value") is None:
                        continue
                    new_val = target_runtime(int(rec["value"])) + int(rec["addend"])
                else:
                    continue
            except ValueError as e:
                raise _Unrelocatable() from e
            out[p - fs : p - fs + size] = arch.pack_pointer(new_val)
        return bytes(out)

    def randomize_layout(self, funcs, seed=None, align=16, **kwargs):
        if "alignment" in kwargs:
            raise ValueError(
                "unknown layout key 'alignment'; use short 'align' "
                "(e.g. randomize_layout(funcs, align=16))"
            )
        if kwargs:
            raise TypeError(
                f"randomize_layout() got unexpected keyword(s) {sorted(kwargs)}"
            )
        try:
            align = int(align, 0) if isinstance(align, str) else int(align)
        except (TypeError, ValueError) as e:
            raise ValueError(f"func align must be an integer, got {align!r}") from e
        if align < 1 or (align & (align - 1)):
            raise ValueError(f"func align must be a power of two >= 1, got {align!r}")
        if not getattr(self, "runtime_base", 0):
            raise RuntimeError("map_elf() must run before randomize_layout()")
        try:
            cle_base = int(self.loader.main_object.mapped_base or 0)
        except Exception:
            cle_base = 0
        pie = bool(getattr(self, "is_pie", False))

        def to_file(a):
            a = int(a)
            return a - cle_base if (pie and cle_base) else a

        names_of = {}
        for name, bounds in (funcs or {}).items():
            try:
                fs, fe = to_file(bounds[0]), to_file(bounds[1])
            except (TypeError, ValueError, IndexError):
                continue
            if fe <= fs:
                continue
            names_of.setdefault((fs, fe), []).append(str(name))
        items = [(fs, fe, names_of[(fs, fe)]) for fs, fe in names_of]
        text = self._text_range()
        if text is not None:
            t0, t1 = text
            items = [it for it in items if t0 <= it[0] and it[1] <= t1]
        else:
            xr = [
                (v, v + msz)
                for v, msz, _rt, fl in self.segment_runtime_ranges()
                if "X" in str(fl)
            ]
            items = [
                it for it in items if any(a <= it[0] and it[1] <= b for a, b in xr)
            ]
        items = [it for it in items if not any("plt" in n.lower() for n in it[2])]
        if not items:
            self.moved_functions = {}
            self.layout_names = {}
            return {}

        image_ranges = [
            (v, v + msz) for v, msz, _rt, _fl in self.segment_runtime_ranges()
        ]

        def in_image(v):
            return any(a <= v < b for a, b in image_ranges)

        arch = self.arch
        md = arch.disassembler()
        md.detail = True
        safe = []
        for fs, fe, names in items:
            try:
                orig_rt = self.runtime_addr(fs)
                code = bytes(self.read_memory(orig_rt, fe - fs))
            except (ValueError, OSError):
                continue
            if arch.layout_hazard(code, in_image, pie, md=md):
                continue
            safe.append((fs, fe, names, code))
        if not safe:
            self.moved_functions = {}
            self.layout_names = {}
            return {}

        rng = random.Random(seed)
        rng.shuffle(safe)
        reloc_map = {}
        cursor = 0
        plan = []
        for fs, fe, names, code in safe:
            cursor += rng.randrange(0, 16) * 16
            cursor = (cursor + align - 1) & ~(align - 1)
            plan.append((fs, fe, names, code, cursor))
            cursor += len(code)
        img_end = max(
            [self.runtime_base]
            + [rt + msz for _v, msz, rt, _fl in self.segment_runtime_ranges()]
        )
        size = max(cursor, 1)
        region = ctypes.c_void_p(-1).value
        hint = (img_end + 0x1000000) & ~(mmap.PAGESIZE - 1)
        for _ in range(64):
            cand = self.mmap(
                hint,
                size,
                mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                mmap.MAP_PRIVATE
                | mmap.MAP_ANONYMOUS
                | _MAP_FIXED
                | _MAP_FIXED_NOREPLACE,
                -1,
                0,
            )
            if (
                cand != ctypes.c_void_p(-1).value
                and cand == hint
                and abs(cand - self.runtime_base) < arch._BRANCH_REACH
            ):
                region = cand
                break
            hint += 0x4000000
        if region == ctypes.c_void_p(-1).value:
            region = self.mmap(
                0,
                size,
                mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                -1,
                0,
            )
        if region == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "layout mapping failed")
        self.mapped_regions.append((region, max(cursor, 1)))
        ctypes.memset(region, arch._FILL_BYTE, max(cursor, 1))
        for fs, fe, _names, _code, off in plan:
            reloc_map[fs] = (fe, region + off)
        moved, names_out = {}, {}
        for fs, fe, names, code, off in plan:
            new_base = region + off
            try:
                fixed = self._relocate_blob(
                    code, fs, new_base, reloc_map, md, self.runtime_addr(fs)
                )
            except _Unrelocatable:
                del reloc_map[fs]
                continue
            self.write_memory(new_base, fixed)
            moved[fs] = (fe, new_base)
            for n in names:
                names_out[n] = new_base
        self.moved_functions = moved
        self.layout_names = names_out
        return dict(names_out)

    def resolve_ifuncs(self, limit=8192):
        found = {}
        handle = _dlopen_quiet(self.loader.main_object.binary)
        for name, _value, _size in _ifunc_symbols(self.loader.main_object.binary):
            if len(found) >= limit:
                break
            addr = _resolve_external(name) or _dlsym(handle, name)
            if addr:
                found[name] = addr
        self.ifuncs = found
        return found

    def relocated_addresses(self):
        out = set()
        for rec in getattr(self, "applied_relocs", []) or []:
            try:
                out.add(self.runtime_addr(int(rec["offset"])))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def perf_map_entries(self, include_symbols=True, prefix=""):
        entries = []
        try:
            binary = os.path.basename(str(self.loader.main_object.binary))
        except Exception:
            binary = "exec"
        for i, (vaddr, memsz, runtime, _flags) in enumerate(
            self.segment_runtime_ranges()
        ):
            if memsz > 0:
                entries.append((runtime, memsz, f"{prefix}perf_exec:{binary}:seg{i}"))
        if include_symbols:
            try:
                with open(self.loader.main_object.binary, "rb") as f:
                    elffile = ELFFile(f)
                    for sec_name in (".symtab", ".dynsym"):
                        sec = elffile.get_section_by_name(sec_name)
                        if sec is None:
                            continue
                        for sym in sec.iter_symbols():
                            try:
                                st_type = sym.entry["st_info"]["type"]
                                shndx = sym.entry["st_shndx"]
                                size = int(sym.entry["st_size"] or 0)
                            except Exception:
                                continue
                            if st_type not in ("STT_FUNC", "STT_OBJECT", "STT_NOTYPE"):
                                continue
                            if shndx in ("SHN_UNDEF", "SHN_ABS", "SHN_COMMON"):
                                continue
                            if not sym.name or size <= 0:
                                continue
                            try:
                                addr = self.get_symbol(sym.name)
                            except (ValueError, OSError):
                                continue
                            entries.append((addr, size, f"{prefix}{sym.name}"))
            except (OSError, ValueError):
                pass
        for fs, (fe, nb) in (getattr(self, "moved_functions", {}) or {}).items():
            for name, addr in (getattr(self, "layout_names", {}) or {}).items():
                if addr == nb:
                    entries.append((nb, fe - fs, f"{prefix}{name}"))
        return list(dict.fromkeys(entries))

    def write_perf_map(self, pid=None, path=None, append=True, include_symbols=True):
        return write_perf_map(
            self.perf_map_entries(include_symbols=include_symbols),
            pid=pid,
            path=path,
            append=append,
        )

    def save_object(
        self,
        path,
        harnesses=None,
        harness_code=None,
        harness_symbol="perf_bench",
        harness_targets=None,
    ):
        if not getattr(self, "runtime_base", 0):
            raise RuntimeError("map_elf() must run before save_object()")
        blobs = {}
        if harnesses:
            for sym, code in dict(harnesses).items():
                if code is None:
                    continue
                blobs[str(sym)] = bytes(code)
        if harness_code is not None:
            blobs[str(harness_symbol)] = bytes(harness_code)
        targets = {}
        if harness_targets:
            for hs, ts in dict(harness_targets).items():
                if str(hs) in blobs:
                    targets[str(hs)] = str(ts)
        return _write_execution_object(
            self.loader.main_object.binary,
            self._dump_runtime_sections(),
            blobs,
            path,
            harness_targets=targets or None,
        )

    def _dump_runtime_sections(self):
        sections = []
        for vaddr, memsz, runtime, flags in self.segment_runtime_ranges():
            if memsz <= 0:
                continue
            try:
                blob = self.read_memory(runtime, memsz)
            except (OSError, ValueError, ctypes.ArgumentError):
                continue
            sections.append(
                {
                    "name": f".perf.exec{len(sections)}",
                    "vaddr": vaddr,
                    "bytes": bytes(blob),
                    "flags": flags,
                }
            )
        return sections


def perf_map_path(pid=None):
    if pid is None:
        pid = os.getpid()
    return _PERF_MAP.format(pid=int(pid))


def write_perf_map(entries, pid=None, path=None, append=True):
    if path is None:
        path = perf_map_path(pid)
    mode = "a" if append else "w"
    with open(path, mode) as fh:
        for addr, size, name in entries or ():
            try:
                a = int(addr, 0) if isinstance(addr, str) else int(addr)
                s = int(size, 0) if isinstance(size, str) else int(size)
            except (TypeError, ValueError):
                continue
            sym = str(name).strip().replace("\n", "_") or "jit"
            fh.write(f"{a:x} {s:x} {sym}\n")
    return path


def is_relocatable(path):
    return _elf_type(path) == "ET_REL"


def is_archive(path):
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == b"!<arch>\n"
    except OSError:
        return False


def needs_link(path):
    try:
        return is_relocatable(path) or is_archive(path)
    except Exception:
        return False


def link_object(path):
    path = str(path)
    if not needs_link(path):
        return path
    compilers = _candidate_compilers()
    if not compilers:
        raise RuntimeError(
            f"{path!r} is a relocatable .o but no compiler is available to link it "
            "(tried $CXX, $CC, g++, gcc)"
        )
    d = tempfile.mkdtemp(prefix="perf-obj-")
    _OBJ_LINK_DEPS.append(d)
    driver = os.path.join(d, "perf_driver.c")
    with open(driver, "w") as fh:
        fh.write("int main(void) { return 0; }\n")
    exe = os.path.join(d, _link_name(path))
    pie_flags = ["-pie", "-fPIE"]
    nopie_flags = ["-fno-pie", "-no-pie"]
    common = ["-O2", "-Wl,--allow-multiple-definition", "-Wl,--undefined=main"]
    last_err = ""
    for compiler in compilers:
        for flags, pie in ((pie_flags, "pie"), (nopie_flags, "no-pie")):
            cmd = compiler + flags + common + [driver]
            if is_archive(path):
                cmd += ["-Wl,--whole-archive", path, "-Wl,--no-whole-archive"]
            else:
                cmd += [path]
            cmd += ["-o", exe]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(exe):
                return exe
            last_err = (
                f"{' '.join(compiler)} [{pie}]: {r.stderr.strip() or 'link failed'}"
            )
    tried = ", ".join(" ".join(c) for c in compilers)
    raise RuntimeError(f"failed to link object {path!r} with {tried}: {last_err}")


def resolve_exec(path):
    path = str(path)
    stamp = _exec_stamp(path)
    hit = _EXEC_LINKS.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    resolved = link_object(path) if needs_link(path) else path
    _EXEC_LINKS[path] = (stamp, resolved)
    return resolved


def align_down(addr, align=ElfConst._PAGE_SIZE):
    return addr & ~(align - 1)


def align_up(addr, align=ElfConst._PAGE_SIZE):
    return (addr + align - 1) & ~(align - 1)


def to_object(
    code,
    path=None,
):
    import angr

    from .arch import load as _load_arch
    from .bench import _normalize_target, parse_code
    from .info import functions as _info_functions
    from .info import targets as _resolve_targets

    file, target, _asm = parse_code(code)
    if file is None:
        raise ValueError("a binary target is required, e.g. 'a.out:fizz_buzz'")
    if not os.path.exists(file):
        raise ValueError(f"file {file!r} does not exist")
    target = _normalize_target(target)
    project = angr.Project(
        resolve_exec(file), auto_load_libs=False, load_debug_info=False
    )
    elf = Elf(project.loader)
    elf.map_elf()
    funcs, _ = _info_functions(project)
    pat = target
    if not pat:
        raise ValueError("a target is required, e.g. 'a.out:fizz_buzz'")

    arch = _load_arch(project)
    harnesses, htargets = {}, {}
    for label, start, _end in _resolve_targets(project, pat, funcs):
        try:
            symbol = elf.get_symbol(label)
        except ValueError:
            continue
        code_asm = arch.call_seq_asm(symbol)
        try:
            hb = bytes(arch.assemble(code_asm))
        except Exception:
            continue
        sym = _sym_id(label)
        hsym = f"perf_bench_{sym}"
        harnesses[hsym] = hb
        if ".." not in str(label):
            raw = elf.raw_symbol(label)
            if raw is not None:
                htargets[hsym] = raw
    return elf.save_object(
        path, harnesses=harnesses or None, harness_targets=htargets or None
    )


def shared_libraries(binary_path):
    names = []
    try:
        with open(binary_path, "rb") as f:
            elf = ELFFile(f)
            dynamic = elf.get_section_by_name(".dynamic")
            if dynamic is None:
                return names
            for tag in dynamic.iter_tags():
                if tag.entry.d_tag != "DT_NEEDED":
                    continue
                name = getattr(tag, "needed", None) or tag.needed
                if name and name not in names:
                    names.append(name)
    except (OSError, ValueError):
        pass
    return names


def load_shared_libraries(names):
    loaded = {}
    for name in names or ():
        handle = _SHARED_LIBS.get(name)
        if handle is None:
            handle = _dlopen(name)
            _SHARED_LIBS[name] = handle
        loaded[name] = handle
    return loaded


def exit_hooks():
    if not _EXIT_HOOKS:

        def _record(func, arg=0):
            entry = (int(func or 0), int(arg or 0))
            if entry not in _REGISTERED_EXITS:
                _REGISTERED_EXITS.append(entry)
            return 0

        def _record_cxa(func, arg=0, _dso_handle=0):
            return _record(func, arg)

        _EXIT_HOOKS["atexit"] = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(_record)
        _EXIT_HOOKS["__cxa_atexit"] = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )(_record_cxa)
    return _EXIT_HOOKS


def retire_exit_hooks():
    n = len(_REGISTERED_EXITS)
    del _REGISTERED_EXITS[:]
    return n


def initializers(binary_path):
    out = []
    with open(binary_path, "rb") as f:
        data = f.read()
        elf = ELFFile(io.BytesIO(data))
        dynamic = elf.get_section_by_name(".dynamic")
        init = 0
        init_array = None
        init_array_size = 0
        if dynamic is not None:
            for tag in dynamic.iter_tags():
                if tag.entry.d_tag == "DT_INIT":
                    init = int(tag.entry.d_val)
                elif tag.entry.d_tag == "DT_INIT_ARRAY":
                    init_array = int(tag.entry.d_val)
                elif tag.entry.d_tag == "DT_INIT_ARRAYSZ":
                    init_array_size = int(tag.entry.d_val)
        if init:
            out.append(init)
        if init_array is not None and init_array_size > 0:
            arch = get_arch()
            section = elf.get_section_by_name(".init_array")
            size = arch._POINTER_SIZE
            if section is not None and int(section["sh_addr"]) == init_array:
                size = int(section["sh_entsize"] or size) or size
            count = min(init_array_size // size, 64)
            offset = _vaddr_to_offset(elf, init_array)
            if offset is not None:
                for i in range(count):
                    start = offset + i * size
                    if start + size > len(data):
                        break
                    out.append(arch.unpack_pointer(data, start))
        return [a for a in out if a]


def call_native(addr):
    code, buffer = _call_stub(int(addr))
    try:
        return get_arch().call_buffered(
            ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        )
    finally:
        buffer.close()


def is_shared_library(path):
    try:
        with open(path, "rb") as fh:
            elf = ELFFile(fh)
            if elf.header["e_type"] != "ET_DYN":
                return False
            dynamic = elf.get_section_by_name(".dynamic")
            if dynamic is None:
                return False
            for tag in dynamic.iter_tags():
                if tag.entry.d_tag == "DT_SONAME":
                    return True
    except Exception:
        return False
    return False


def _exec_stamp(path):
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


@contextlib.contextmanager
def _output_sink(fds=None):
    restores = []
    for fd in dict.fromkeys(int(f) for f in (fds or ()) if int(f) in (1, 2)):
        saved = sink = None
        try:
            saved = os.dup(fd)
            sink = os.open("/dev/null", os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))
            os.dup2(sink, fd)
        except OSError:
            for handle in (saved, sink):
                if handle is not None:
                    try:
                        os.close(handle)
                    except OSError:
                        pass
            continue
        os.close(sink)
        restores.append((fd, saved))
    try:
        yield
    finally:
        if restores:
            _discard_output()
        for fd, saved in reversed(restores):
            try:
                os.dup2(saved, fd)
            except OSError:
                pass
            try:
                os.close(saved)
            except OSError:
                pass


def _discard_output():
    try:
        Elf.libc.fflush(ctypes.c_void_p(None))
    except Exception:
        pass


@functools.lru_cache(maxsize=32)
def _ifunc_symbols(binary):
    out = []
    try:
        with open(binary, "rb") as f:
            elf = ELFFile(f)
            for sec_name in (".symtab", ".dynsym"):
                sec = elf.get_section_by_name(sec_name)
                if sec is None:
                    continue
                for sym in sec.iter_symbols():
                    try:
                        typ = str(sym.entry["st_info"]["type"])
                        shndx = sym.entry["st_shndx"]
                        size = int(sym.entry["st_size"] or 0)
                    except Exception:
                        continue
                    if typ not in ("STT_GNU_IFUNC", "STT_LOOS"):
                        continue
                    if shndx in ("SHN_UNDEF", "SHN_ABS", "SHN_COMMON"):
                        continue
                    if not sym.name or size <= 0:
                        continue
                    out.append((sym.name, int(sym.entry["st_value"]), size))
                break
    except Exception:
        return []
    return out


@functools.lru_cache(maxsize=4096)
def _short_symbol(name):
    text = str(name or "").strip()
    head = text.split("(")[0].strip()
    return head or None


@functools.lru_cache(maxsize=32)
def _symtab_values(binary):
    values = {}
    demangled = {}
    try:
        with open(binary, "rb") as f:
            elf = ELFFile(f)
            symtab = elf.get_section_by_name(".symtab") or elf.get_section_by_name(
                ".dynsym"
            )
            if symtab is None:
                return values, demangled
            for sym in symtab.iter_symbols():
                if not sym.name:
                    continue
                values.setdefault(sym.name, sym.entry["st_value"])
                demangled.setdefault(demangle(sym.name), sym.entry["st_value"])
    except (OSError, ValueError):
        pass
    return values, demangled


@functools.lru_cache(maxsize=4096)
def _resolve_external(name):
    if not name:
        return None
    hook = exit_hooks().get(name)
    if hook is not None:
        return ctypes.cast(hook, ctypes.c_void_p).value
    for _handle in (*_shared_libs().values(), _main_lib(), _libc_lib()):
        if _handle is None:
            continue
        addr = _dlsym(_handle, name)
        if addr:
            return addr
    return None


@functools.lru_cache(maxsize=1)
def _dlsym_entry():
    try:
        dlsym = Elf.libc.dlsym
        dlsym.restype = ctypes.c_void_p
        dlsym.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
        return dlsym
    except (AttributeError, TypeError):
        return None


def _dlsym(handle, name):
    dlsym = _dlsym_entry()
    if dlsym is None or handle is None:
        return None
    try:
        return dlsym(ctypes.c_void_p(handle._handle), str(name).encode())
    except Exception:
        return None


def _live_ranges():
    out = []
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                parts = line.split(" ", 1)[0].split("-", 1)
                if len(parts) != 2:
                    continue
                try:
                    out.append((int(parts[0], 16), int(parts[1], 16)))
                except ValueError:
                    continue
    except OSError:
        return []
    return sorted(out)


def _live_overlap(lo, size):
    hi = int(lo) + int(size)
    for start, stop in _live_ranges():
        if start < hi and int(lo) < stop:
            return start, stop
    return None


def _dlopen_quiet(name):
    if name in _SHARED_LIBS:
        return _SHARED_LIBS[name]
    try:
        handle = ctypes.CDLL(name, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    except OSError:
        try:
            handle = ctypes.CDLL(name, mode=getattr(os, "RTLD_LAZY", 1))
        except OSError:
            return None
    _SHARED_LIBS[name] = handle
    return handle


def _dlopen(name):
    flags = getattr(os, "RTLD_NOW", 2) | getattr(os, "RTLD_GLOBAL", 0)
    try:
        return ctypes.CDLL(name, mode=flags)
    except OSError:
        pass
    try:
        return ctypes.CDLL(name, mode=getattr(os, "RTLD_LAZY", 1))
    except OSError as e:
        print(f"Warning: cannot load {name}: {e}", file=sys.stderr)
        return None


@functools.cache
def _shared_libs():
    return dict(_SHARED_LIBS)


def _vaddr_to_offset(elf, vaddr):
    for seg in elf.iter_segments():
        if seg.header.p_type != "PT_LOAD":
            continue
        if seg.header.p_vaddr <= vaddr < seg.header.p_vaddr + seg.header.p_filesz:
            return int(seg.header.p_offset + (vaddr - seg.header.p_vaddr))
    return None


def _call_stub(addr):
    code = bytes(get_arch().assemble(get_arch().call_native_asm(int(addr))))
    buffer = mmap.mmap(
        -1,
        len(code),
        prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
    )
    buffer.write(code)
    return code, buffer


@functools.cache
def _main_lib():
    try:
        return ctypes.CDLL(None)
    except Exception:
        return None


@functools.cache
def _libc_lib():
    try:
        return ctypes.CDLL(ctypes.util.find_library("c"))
    except Exception:
        return None


def _link_name(path):
    base = os.path.basename(str(path)).strip()
    return base if base not in ("", ".", "..") else "perf-linked"


def _candidate_compilers():
    out = []
    for c in (os.environ.get("CXX"), os.environ.get("CC"), "g++", "gcc"):
        if not c:
            continue
        try:
            args = shlex.split(c)
        except ValueError:
            args = [c]
        if args and shutil.which(args[0]) and args not in out:
            out.append(args)
    return out


class _Unrelocatable(Exception):
    pass


def _elf_type(path):
    try:
        with open(path, "rb") as fh:
            return ELFFile(fh).header["e_type"]
    except Exception:
        return None


def _cleanup_obj_links():
    while _OBJ_LINK_DEPS:
        shutil.rmtree(_OBJ_LINK_DEPS.pop(), True)


def _write_execution_object(binary_path, dumped, harnesses, path, harness_targets=None):
    arch = get_arch()
    if harness_targets:
        harnesses = dict(harnesses or {})
        for hsym in harness_targets:
            if hsym in harnesses:
                harnesses[hsym] = arch.movabs64_bytes(harnesses[hsym])
    with open(binary_path, "rb") as f:
        image = f.read()
    elffile = ELFFile(io.BytesIO(image))

    sections = []
    for d in dumped:
        blob = bytearray(d["bytes"])
        flags = _SHF_ALLOC
        fstr = str(d.get("flags") or "")
        if "W" in fstr:
            flags |= _SHF_WRITE
        if "X" in fstr:
            flags |= _SHF_EXECINSTR
        sections.append(
            {
                "name": d["name"],
                "vaddr": int(d["vaddr"]),
                "blob": blob,
                "flags": flags,
                "kind": "image",
            }
        )
    harness_items = list((harnesses or {}).items())
    harness_blob = bytearray()
    harness_offsets = {}
    if harness_items:
        for sym, code in harness_items:
            off = align_up(len(harness_blob), 16)
            harness_blob.extend(bytes([arch._NOP_BYTE]) * (off - len(harness_blob)))
            harness_offsets[sym] = off
            harness_blob.extend(bytes(code))
        sections.append(
            {
                "name": ".perf.harness",
                "vaddr": None,
                "blob": harness_blob,
                "flags": _SHF_ALLOC | _SHF_EXECINSTR,
                "kind": "harness",
            }
        )

    def find_section(vaddr):
        for i, s in enumerate(sections):
            if s["kind"] != "image":
                continue
            if s["vaddr"] <= vaddr < s["vaddr"] + len(s["blob"]):
                return i
        return None

    syms = [
        {
            "name": "",
            "bind": _STB_LOCAL,
            "type": _STT_NOTYPE,
            "shndx": 0,
            "value": 0,
            "size": 0,
        }
    ]
    sec_sym_idx = {}
    for i in range(len(sections)):
        sec_sym_idx[i] = len(syms)
        syms.append(
            {
                "name": "",
                "bind": _STB_LOCAL,
                "type": _STT_SECTION,
                "shndx": 1 + i,
                "value": 0,
                "size": 0,
            }
        )
    defined_by_name = {}
    order = []
    try:
        _keep_global = set((harness_targets or {}).values())
    except Exception:
        _keep_global = set()
    for sec_name in (".symtab", ".dynsym"):
        sec = elffile.get_section_by_name(sec_name)
        if sec is None:
            continue
        for sym in sec.iter_symbols():
            try:
                name = sym.name
                info = sym.entry["st_info"]
                try:
                    bind = info["bind"]
                    st_type = info["type"]
                except Exception:
                    bind = int(info) >> 4
                    st_type = int(info) & 0xF
                shndx = sym.entry["st_shndx"]
                value = int(sym.entry["st_value"])
                size = int(sym.entry["st_size"] or 0)
            except Exception:
                continue
            if not name or name in defined_by_name:
                continue
            bind_l = str(bind).upper() if isinstance(bind, str) else bind
            type_l = str(st_type).upper() if isinstance(st_type, str) else st_type
            if isinstance(bind_l, str):
                bind = {"STB_LOCAL": 0, "STB_GLOBAL": 1, "STB_WEAK": 2}.get(bind_l, 1)
            if isinstance(type_l, str):
                st_type = {
                    "STT_NOTYPE": 0,
                    "STT_OBJECT": 1,
                    "STT_FUNC": 2,
                    "STT_SECTION": 3,
                    "STT_IFUNC": 10,
                }.get(type_l, 0)
            if st_type == _STT_SECTION:
                continue
            if shndx in ("SHN_UNDEF", _SHN_UNDEF, "SHN_ABS", "SHN_COMMON"):
                continue
            if st_type not in (_STT_NOTYPE, _STT_OBJECT, _STT_FUNC, _STT_IFUNC):
                continue
            sec_idx = find_section(value)
            if sec_idx is None:
                continue
            if st_type == _STT_IFUNC:
                st_type = _STT_FUNC
            try:
                _bind_out = int(bind) if name in _keep_global else _STB_LOCAL
            except Exception:
                _bind_out = _STB_LOCAL
            defined_by_name[name] = len(syms)
            order.append(name)
            syms.append(
                {
                    "name": name,
                    "bind": _bind_out,
                    "type": int(st_type),
                    "shndx": 1 + sec_idx,
                    "value": value - sections[sec_idx]["vaddr"],
                    "size": size,
                }
            )

    undef_idx = {}

    def ensure_undef(name, bind=_STB_GLOBAL, st_type=_STT_NOTYPE):
        if name in defined_by_name:
            return defined_by_name[name]
        if name in undef_idx:
            return undef_idx[name]
        undef_idx[name] = len(syms)
        syms.append(
            {
                "name": name,
                "bind": int(bind),
                "type": int(st_type),
                "shndx": 0,
                "value": 0,
                "size": 0,
            }
        )
        return undef_idx[name]

    width = arch._POINTER_SIZE
    relocs_by_sec = {i: [] for i in range(len(sections))}
    for relsec in elffile.iter_sections():
        if not isinstance(relsec, RelocationSection):
            continue
        try:
            symtab = elffile.get_section(relsec["sh_link"])
        except Exception:
            continue
        for reloc in relsec.iter_relocations():
            try:
                r_offset = int(reloc["r_offset"])
                r_type = int(reloc["r_info_type"])
                r_sym = int(reloc["r_info_sym"])
                addend = int(reloc["r_addend"]) if reloc.is_RELA() else 0
            except Exception:
                continue
            sec_idx = find_section(r_offset)
            if sec_idx is None:
                continue
            off = r_offset - sections[sec_idx]["vaddr"]
            blob = sections[sec_idx]["blob"]
            if off < 0 or off + width > len(blob):
                continue
            if r_type in (
                ElfConst.R_X86_64_64,
                ElfConst.R_X86_64_GLOB_DAT,
                ElfConst.R_X86_64_JUMP_SLOT,
            ):
                sym = None
                try:
                    sym = symtab.get_symbol(r_sym)
                except Exception:
                    sym = None
                if sym is None:
                    blob[off : off + width] = arch.pack_pointer(addend)
                    continue
                name = sym.name
                try:
                    info = sym.entry["st_info"]
                    try:
                        sbind = info["bind"]
                        stype = info["type"]
                    except Exception:
                        sbind = int(info) >> 4
                        stype = int(info) & 0xF
                except Exception:
                    sbind, stype = "STB_GLOBAL", "STT_NOTYPE"
                if isinstance(sbind, str):
                    sbind = {"STB_LOCAL": 0, "STB_GLOBAL": 1, "STB_WEAK": 2}.get(
                        sbind.upper(), 1
                    )
                if isinstance(stype, str):
                    stype = {
                        "STT_NOTYPE": 0,
                        "STT_OBJECT": 1,
                        "STT_FUNC": 2,
                        "STT_IFUNC": 10,
                    }.get(stype.upper(), 0)
                if name in defined_by_name:
                    idx = defined_by_name[name]
                else:
                    try:
                        sval = int(sym.entry["st_value"])
                        ssec = find_section(sval)
                    except Exception:
                        ssec = None
                    if ssec is not None and sym.entry["st_shndx"] not in (
                        "SHN_UNDEF",
                        _SHN_UNDEF,
                    ):
                        if name not in defined_by_name:
                            defined_by_name[name] = len(syms)
                            try:
                                _b2 = int(sbind) if name in _keep_global else _STB_LOCAL
                            except Exception:
                                _b2 = _STB_LOCAL
                            syms.append(
                                {
                                    "name": name,
                                    "bind": _b2,
                                    "type": _STT_FUNC
                                    if stype == _STT_IFUNC
                                    else int(stype),
                                    "shndx": 1 + ssec,
                                    "value": sval - sections[ssec]["vaddr"],
                                    "size": int(sym.entry["st_size"] or 0),
                                }
                            )
                        idx = defined_by_name[name]
                    else:
                        idx = ensure_undef(name, sbind, stype)
                blob[off : off + width] = arch.pack_pointer(addend)
                relocs_by_sec[sec_idx].append((off, ElfConst.R_X86_64_64, idx, addend))
            elif r_type == ElfConst.R_X86_64_RELATIVE:
                tgt = find_section(addend)
                if tgt is None:
                    continue
                blob[off : off + width] = arch.pack_pointer(
                    addend - sections[tgt]["vaddr"]
                )
                relocs_by_sec[sec_idx].append(
                    (
                        off,
                        ElfConst.R_X86_64_64,
                        sec_sym_idx[tgt],
                        addend - sections[tgt]["vaddr"],
                    )
                )
            else:
                continue

    for i, relocs in relocs_by_sec.items():
        if relocs and sections[i]["kind"] == "image":
            sections[i]["flags"] |= _SHF_WRITE
    sections.append(
        {
            "name": ".note.GNU-stack",
            "vaddr": None,
            "blob": bytearray(),
            "flags": 0,
            "kind": "note",
        }
    )

    harness_sec = None
    for i, s in enumerate(sections):
        if s["kind"] == "harness":
            harness_sec = i
            break
    if harness_sec is not None:
        for sym, off in harness_offsets.items():
            if sym in defined_by_name or sym in undef_idx:
                continue
            defined_by_name[sym] = len(syms)
            syms.append(
                {
                    "name": sym,
                    "bind": _STB_GLOBAL,
                    "type": _STT_FUNC,
                    "shndx": 1 + harness_sec,
                    "value": off,
                    "size": len(bytes((harnesses or {})[sym])),
                }
            )
        hblob = sections[harness_sec]["blob"]
        for hsym, tsym in (harness_targets or {}).items():
            if hsym not in harness_offsets or tsym not in defined_by_name:
                continue
            start = harness_offsets[hsym]
            end = start + len(bytes((harnesses or {})[hsym]))
            hblob, at = arch.clear_movabs(hblob, start, end)
            if at < 0:
                continue
            sections[harness_sec]["blob"] = bytearray(hblob)
            relocs_by_sec[harness_sec].append(
                (at, ElfConst.R_X86_64_64, defined_by_name[tsym], 0)
            )
            sections[harness_sec]["flags"] |= _SHF_WRITE

    strtab = bytearray(b"\x00")
    for s in syms[1:]:
        s["str_off"] = len(strtab) if s["name"] else 0
        if s["name"]:
            strtab.extend(s["name"].encode() + b"\x00")

    names = [""] + [s["name"] for s in sections] + [".symtab", ".strtab", ".shstrtab"]
    rela_names = {}
    for i, s in enumerate(sections):
        if relocs_by_sec.get(i):
            rela_names[i] = f".rela{s['name']}"
            names.append(rela_names[i])
    shstrtab = bytearray()
    name_off = {}
    for n in names:
        if n not in name_off:
            name_off[n] = len(shstrtab)
            shstrtab.extend(n.encode() + b"\x00")

    n_sec_syms = len(sections)
    head = syms[: 1 + n_sec_syms]
    tail = syms[1 + n_sec_syms :]
    locals_first = [s for s in tail if int(s["bind"]) == _STB_LOCAL]
    globals_last = [s for s in tail if int(s["bind"]) != _STB_LOCAL]
    new_tail = locals_first + globals_last
    old_index_of = {}
    for old, s in enumerate(tail, start=1 + n_sec_syms):
        old_index_of.setdefault(id(s), old)
    new_index_of_old = {}
    for new_i, s in enumerate(new_tail, start=1 + n_sec_syms):
        new_index_of_old[old_index_of[id(s)]] = new_i
    syms = head + new_tail
    first_global = 1 + n_sec_syms + len(locals_first)
    for d in list(defined_by_name):
        defined_by_name[d] = new_index_of_old.get(
            defined_by_name[d], defined_by_name[d]
        )
    for d in list(undef_idx):
        undef_idx[d] = new_index_of_old.get(undef_idx[d], undef_idx[d])
    for sec_i, relocs in relocs_by_sec.items():
        relocs_by_sec[sec_i] = [
            (o, t, new_index_of_old.get(si, si), a) for o, t, si, a in relocs
        ]

    shnum = 1 + len(sections) + 3 + len(rela_names)
    symtab_idx = 1 + len(sections)
    strtab_idx = symtab_idx + 1
    shstr_idx = strtab_idx + 1

    off = 64
    for s in sections:
        off = align_up(off, 16)
        s["file_off"] = off
        off += len(s["blob"])
    off = align_up(off, 8)
    symtab_off = off
    off += len(syms) * 24
    strtab_off = off
    off += len(strtab)
    shstrtab_off = off
    off += len(shstrtab)
    rela_off = {}
    for i in sorted(rela_names):
        off = align_up(off, 8)
        rela_off[i] = off
        off += len(relocs_by_sec[i]) * 24
    shoff = align_up(off, 8)

    with open(path, "wb") as fh:
        ident = b"\x7fELF" + bytes([2, 1, 1, 0, 0]) + b"\x00" * 7
        fh.write(
            struct.pack(
                "<16sHHIQQQIHHHHHH",
                ident,
                _ET_REL,
                arch._ELF_MACHINE,
                1,
                0,
                0,
                shoff,
                0,
                64,
                0,
                0,
                64,
                shnum,
                shstr_idx,
            )
        )
        for s in sections:
            fh.seek(s["file_off"])
            fh.write(bytes(s["blob"]))
        fh.seek(symtab_off)
        for s in syms:
            st_info = ((int(s["bind"]) & 0xF) << 4) | (int(s["type"]) & 0xF)
            fh.write(
                struct.pack(
                    "<IBBHQQ",
                    s.get("str_off", 0),
                    st_info,
                    0,
                    int(s["shndx"]),
                    int(s["value"]) & arch._ADDR_MASK,
                    int(s["size"]) & arch._ADDR_MASK,
                )
            )
        fh.seek(strtab_off)
        fh.write(bytes(strtab))
        fh.seek(shstrtab_off)
        fh.write(bytes(shstrtab))
        for i in sorted(rela_names):
            fh.seek(rela_off[i])
            for r_off, r_type, r_sym, r_add in relocs_by_sec[i]:
                fh.write(
                    struct.pack(
                        "<QQq", r_off, (int(r_sym) << 32) | int(r_type), int(r_add)
                    )
                )
        fh.seek(shoff)
        fh.write(struct.pack("<IIQQQQIIQQ", 0, _SHT_NULL, 0, 0, 0, 0, 0, 0, 0, 0))
        for s in sections:
            fh.write(
                struct.pack(
                    "<IIQQQQIIQQ",
                    name_off[s["name"]],
                    _SHT_PROGBITS,
                    s["flags"],
                    0,
                    s["file_off"],
                    len(s["blob"]),
                    0,
                    0,
                    16,
                    0,
                )
            )
        fh.write(
            struct.pack(
                "<IIQQQQIIQQ",
                name_off[".symtab"],
                _SHT_SYMTAB,
                0,
                0,
                symtab_off,
                len(syms) * 24,
                symtab_idx + 1,
                first_global,
                8,
                24,
            )
        )
        fh.write(
            struct.pack(
                "<IIQQQQIIQQ",
                name_off[".strtab"],
                _SHT_STRTAB,
                0,
                0,
                strtab_off,
                len(strtab),
                0,
                0,
                1,
                0,
            )
        )
        fh.write(
            struct.pack(
                "<IIQQQQIIQQ",
                name_off[".shstrtab"],
                _SHT_STRTAB,
                0,
                0,
                shstrtab_off,
                len(shstrtab),
                0,
                0,
                1,
                0,
            )
        )
        for i in sorted(rela_names):
            fh.write(
                struct.pack(
                    "<IIQQQQIIQQ",
                    name_off[rela_names[i]],
                    _SHT_RELA,
                    0,
                    0,
                    rela_off[i],
                    len(relocs_by_sec[i]) * 24,
                    symtab_idx,
                    1 + i,
                    8,
                    24,
                )
            )
    return path


def _sym_id(label):
    base = "".join(c if c.isalnum() or c == "_" else "_" for c in str(label))
    base = base.strip("_") or "target"
    if base[:1].isdigit():
        base = "f_" + base
    return base


atexit.register(_cleanup_obj_links)
