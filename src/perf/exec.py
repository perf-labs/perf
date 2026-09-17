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
import ctypes
import io
import os
import random
import shlex
import shutil
import struct
import subprocess
import tempfile

import capstone
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection

from .core import demangle

_PERF_MAP_TEMPLATE = "/tmp/perf-{pid}.map"
_ET_REL, _EM_X86_64 = 1, 62
_SHT_NULL, _SHT_PROGBITS, _SHT_SYMTAB, _SHT_STRTAB, _SHT_RELA = 0, 1, 2, 3, 4
_SHF_WRITE, _SHF_ALLOC, _SHF_EXECINSTR = 0x1, 0x2, 0x4
_STB_LOCAL, _STB_GLOBAL = 0, 1
_STT_NOTYPE, _STT_OBJECT, _STT_FUNC, _STT_SECTION, _STT_IFUNC = 0, 1, 2, 3, 10
_SHN_UNDEF = 0
_MAP_FIXED_NOREPLACE = 0x100000
_OBJ_LINK_DEPS = []


class ElfConst:
    R_X86_64_64 = 1
    R_X86_64_COPY = 5
    R_X86_64_GLOB_DAT = 6
    R_X86_64_JUMP_SLOT = 7
    R_X86_64_RELATIVE = 8
    ASLR_DISABLED = 0x40000
    PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


class Elf:
    PROT_READ = 0x1
    PROT_WRITE = 0x2
    PROT_EXEC = 0x4
    MAP_PRIVATE = 0x02
    MAP_ANONYMOUS = 0x20
    MAP_FIXED = 0x10

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
        self.runtime_base = 0
        self.mapped_regions = []
        self.applied_relocs = []
        self.moved_functions = {}
        self.layout_names = {}

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

        if is_pie:
            base = obj.mapped_base
            assert base != 0, "mapped_base must be non-zero"
            map_start = base + start
        else:
            map_start = start

        map_size = total_size

        def _mmap_at(hint, fixed):
            flags = self.MAP_PRIVATE | self.MAP_ANONYMOUS
            if fixed:
                flags |= self.MAP_FIXED | _MAP_FIXED_NOREPLACE
            return self.mmap(
                hint,
                map_size,
                self.PROT_READ | self.PROT_WRITE | self.PROT_EXEC,
                flags,
                -1,
                0,
            )

        addr = None
        if is_pie:
            addr = _mmap_at(map_start, fixed=True)
            if addr == ctypes.c_void_p(-1).value:
                addr = self.mmap(
                    0,
                    map_size,
                    self.PROT_READ | self.PROT_WRITE | self.PROT_EXEC,
                    self.MAP_PRIVATE | self.MAP_ANONYMOUS,
                    -1,
                    0,
                )
        else:
            addr = self.mmap(
                map_start,
                map_size,
                self.PROT_READ | self.PROT_WRITE | self.PROT_EXEC,
                self.MAP_PRIVATE | self.MAP_ANONYMOUS | self.MAP_FIXED,
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

        self.apply_relocations(obj.binary)

        return self

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
                    value = S + A
                elif rel_type == ElfConst.R_X86_64_COPY:
                    if sym is None or sym.entry["st_shndx"] == "SHN_UNDEF":
                        print(
                            "Warning: COPY for undefined symbol "
                            f"{sym.name if sym else '?'}"
                        )
                        ctypes.c_uint64.from_address(P).value = 0
                        self.applied_relocs.append(rec)
                        continue
                    src = resolve_symbol(sym)
                    size = sym.entry["st_size"]
                    ctypes.memmove(P, src, size)
                    self.applied_relocs.append(rec)
                    continue
                else:
                    self.applied_relocs.append(rec)
                    continue

                ctypes.c_uint64.from_address(P).value = value
                self.applied_relocs.append(rec)

    def get_symbol(self, name):
        raw = self.raw_symbol(name)
        if raw is None:
            raise ValueError(f"Symbol {name} not found")
        with open(self.loader.main_object.binary, "rb") as f:
            elf = ELFFile(f)
            symtab = elf.get_section_by_name(".symtab") or elf.get_section_by_name(
                ".dynsym"
            )
            symbols = symtab.get_symbol_by_name(raw)
            if not symbols:
                symbols = self._symbols_by_demangled(name)
            if not symbols:
                raise ValueError(f"Symbol {name} not found")
            st_value = symbols[0].entry["st_value"]

        for fs, (fe, nb) in (getattr(self, "moved_functions", {}) or {}).items():
            if fs <= st_value < fe:
                return nb + (st_value - fs)

        if self.is_pie:
            return self.runtime_base + st_value
        else:
            return st_value

    def raw_symbol(self, name):
        try:
            with open(self.loader.main_object.binary, "rb") as f:
                elf = ELFFile(f)
                symtab = elf.get_section_by_name(".symtab") or elf.get_section_by_name(
                    ".dynsym"
                )
                if symtab is None:
                    return None
                candidates = symtab.get_symbol_by_name(name)
                if candidates:
                    return name
                for sym in symtab.iter_symbols():
                    if demangle(sym.name) == name:
                        return sym.name
        except Exception:
            pass
        return None

    def _symbols_by_demangled(self, name):
        try:
            with open(self.loader.main_object.binary, "rb") as f:
                elf = ELFFile(f)
                symtab = elf.get_section_by_name(".symtab") or elf.get_section_by_name(
                    ".dynsym"
                )
                if symtab is None:
                    return []
                return [
                    sym for sym in symtab.iter_symbols() if demangle(sym.name) == name
                ]
        except Exception:
            return []

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
            self.PROT_READ | self.PROT_WRITE,
            self.MAP_PRIVATE | self.MAP_ANONYMOUS,
            -1,
            0,
        )
        if stack == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "Stack alloc failed")
        self.mapped_regions.append((stack, size))
        top = (int(stack) + int(size)) & ~(int(align) - 1)
        self.stack_top = top - 16
        return self.stack_top

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

    def _layout_hazard(self, code, md, in_image, pie):
        for insn in md.disasm(bytes(code), 0):
            if insn.bytes[:1] in (b"\xa0", b"\xa1", b"\xa2", b"\xa3"):
                return True
            try:
                groups = set(insn.groups)
            except Exception:
                groups = set()
            if (
                capstone.x86.X86_GRP_CALL in groups
                or capstone.x86.X86_GRP_JUMP in groups
            ):
                continue
            try:
                operands = insn.operands
            except Exception:
                continue
            for op in operands:
                try:
                    is_imm = op.type == capstone.x86.X86_OP_IMM
                except Exception:
                    continue
                if not is_imm:
                    continue
                try:
                    size = int(op.size)
                    val = int(op.imm) & 0xFFFFFFFFFFFFFFFF
                except Exception:
                    continue
                if (size >= 8 or not pie) and size >= 4 and in_image(val):
                    return True
        return False

    def _relocate_blob(self, code, fs, new_base, reloc_map, md, orig_base):
        fe = fs + len(code)
        out = bytearray(code)

        def target_runtime(file_v):
            for s, (e, nb) in reloc_map.items():
                if s <= file_v < e:
                    return nb + (file_v - s)
            return self.runtime_addr(file_v)

        for insn in md.disasm(bytes(code), orig_base):
            off = insn.address - orig_base
            try:
                groups = set(insn.groups)
                operands = list(insn.operands)
            except Exception:
                raise _Unrelocatable()
            for op in operands:
                try:
                    is_mem = op.type == capstone.x86.X86_OP_MEM
                    rip_base = op.mem.base == capstone.x86.X86_REG_RIP
                except Exception:
                    continue
                if not (is_mem and rip_base):
                    continue
                if insn.size < 5 or off + insn.size > len(out):
                    raise _Unrelocatable()
                disp = struct.unpack(
                    "<i", bytes(out[off + insn.size - 4 : off + insn.size])
                )[0]
                ref_file = insn.address + insn.size + disp - orig_base + fs
                try:
                    new_disp = target_runtime(ref_file) - (new_base + off + insn.size)
                except ValueError as e:
                    raise _Unrelocatable() from e
                try:
                    out[off + insn.size - 4 : off + insn.size] = struct.pack(
                        "<i", new_disp
                    )
                except struct.error as e:
                    raise _Unrelocatable() from e
            if (
                capstone.x86.X86_GRP_CALL in groups
                or capstone.x86.X86_GRP_JUMP in groups
            ):
                if not operands:
                    continue
                try:
                    direct = operands[0].type == capstone.x86.X86_OP_IMM
                except Exception:
                    continue
                if not direct:
                    continue
                try:
                    tgt_rt = int(operands[0].imm)
                except Exception:
                    raise _Unrelocatable()
                tgt_file = tgt_rt - orig_base + fs
                if fs <= tgt_file < fe:
                    continue
                try:
                    new_tgt = target_runtime(tgt_file)
                except ValueError as e:
                    raise _Unrelocatable() from e
                new_disp = new_tgt - (new_base + off + insn.size)
                if insn.size == 2:
                    if not -128 <= new_disp <= 127:
                        raise _Unrelocatable()
                    out[off + 1] = new_disp & 0xFF
                elif insn.size in (5, 6):
                    try:
                        out[off + insn.size - 4 : off + insn.size] = struct.pack(
                            "<i", new_disp
                        )
                    except struct.error as e:
                        raise _Unrelocatable() from e
                else:
                    raise _Unrelocatable()

        for rec in getattr(self, "applied_relocs", []) or []:
            p = int(rec["offset"])
            if not (fs <= p < fe) or p + 8 > fe:
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
            out[p - fs : p - fs + 8] = struct.pack("<Q", new_val & 0xFFFFFFFFFFFFFFFF)
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

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        safe = []
        for fs, fe, names in items:
            try:
                orig_rt = self.runtime_addr(fs)
                code = bytes(self.read_memory(orig_rt, fe - fs))
            except (ValueError, OSError):
                continue
            if self._layout_hazard(code, md, in_image, pie):
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
        hint = (img_end + 0x1000000) & ~0xFFF
        for _ in range(64):
            cand = self.mmap(
                hint,
                size,
                self.PROT_READ | self.PROT_WRITE | self.PROT_EXEC,
                self.MAP_PRIVATE
                | self.MAP_ANONYMOUS
                | self.MAP_FIXED
                | _MAP_FIXED_NOREPLACE,
                -1,
                0,
            )
            if (
                cand != ctypes.c_void_p(-1).value
                and cand == hint
                and abs(cand - self.runtime_base) < 2**31
            ):
                region = cand
                break
            hint += 0x4000000
        if region == ctypes.c_void_p(-1).value:
            region = self.mmap(
                0,
                size,
                self.PROT_READ | self.PROT_WRITE | self.PROT_EXEC,
                self.MAP_PRIVATE | self.MAP_ANONYMOUS,
                -1,
                0,
            )
        if region == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "layout mapping failed")
        self.mapped_regions.append((region, max(cursor, 1)))
        ctypes.memset(region, 0xCC, max(cursor, 1))
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
    return _PERF_MAP_TEMPLATE.format(pid=int(pid))


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
    exe = os.path.join(d, "linked")
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
    return link_object(path) if needs_link(path) else path


def align_down(addr, align=ElfConst.PAGE_SIZE):
    return addr & ~(align - 1)


def align_up(addr, align=ElfConst.PAGE_SIZE):
    return (addr + align - 1) & ~(align - 1)


def obj(
    exec_path,
    func=None,
    region=None,
    name=None,
    path=None,
    config=None,
    setup=None,
    teardown=None,
):
    import angr

    from .arch import load as _load_arch
    from .info import functions as _info_functions
    from .info import targets as _resolve_targets

    project = angr.Project(
        resolve_exec(exec_path), auto_load_libs=False, load_debug_info=False
    )
    obj = Elf(project.loader)
    obj.map_elf()
    funcs, _ = _info_functions(project)
    pat = func or region or name
    if region is not None and not isinstance(region, str):
        pat = f"{region[0]}..{region[1]}"
    if not pat:
        raise ValueError("a function name is required")

    arch = _load_arch(project)
    harnesses, htargets = {}, {}
    for label, start, _end in _resolve_targets(project, pat, funcs):
        try:
            symbol = obj.get_symbol(label)
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
            raw = obj.raw_symbol(label)
            if raw is not None:
                htargets[hsym] = raw
    return obj.save_object(
        path, harnesses=harnesses or None, harness_targets=htargets or None
    )


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


atexit.register(_cleanup_obj_links)


def _as_movabs64(blob):
    i = bytes(blob).find(b"\x48\xc7\xc0")
    if i < 0 or i + 7 > len(blob):
        return bytes(blob)
    imm32 = struct.unpack("<i", bytes(blob)[i + 3 : i + 7])[0]
    return (
        bytes(blob)[:i] + b"\x48\xb8" + struct.pack("<q", imm32) + bytes(blob)[i + 7 :]
    )


def _write_execution_object(binary_path, dumped, harnesses, path, harness_targets=None):
    if harness_targets:
        harnesses = dict(harnesses or {})
        for hsym in harness_targets:
            if hsym in harnesses:
                harnesses[hsym] = _as_movabs64(harnesses[hsym])
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
            harness_blob.extend(b"\x90" * (off - len(harness_blob)))
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
            if off < 0 or off + 8 > len(blob):
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
                    blob[off : off + 8] = struct.pack("<Q", addend & 0xFFFFFFFFFFFFFFFF)
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
                blob[off : off + 8] = struct.pack("<Q", addend & 0xFFFFFFFFFFFFFFFF)
                relocs_by_sec[sec_idx].append((off, ElfConst.R_X86_64_64, idx, addend))
            elif r_type == ElfConst.R_X86_64_RELATIVE:
                tgt = find_section(addend)
                if tgt is None:
                    continue
                blob[off : off + 8] = struct.pack(
                    "<Q", (addend - sections[tgt]["vaddr"]) & 0xFFFFFFFFFFFFFFFF
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
            at = bytes(hblob).find(b"\x48\xb8", start, end)
            if at < 0 or at + 10 > end:
                continue
            hblob[at + 2 : at + 10] = b"\x00" * 8
            relocs_by_sec[harness_sec].append(
                (at + 2, ElfConst.R_X86_64_64, defined_by_name[tsym], 0)
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
                _EM_X86_64,
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
                    int(s["value"]) & 0xFFFFFFFFFFFFFFFF,
                    int(s["size"]) & 0xFFFFFFFFFFFFFFFF,
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
