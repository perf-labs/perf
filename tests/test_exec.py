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

import inspect
import mmap
import os
import shutil
import subprocess
import tempfile
import unittest

from perf.exec import (
    ElfConst,
    align_down,
    align_up,
    perf_map_path,
    resolve_exec,
    write_perf_map,
)


class TestElfConst(unittest.TestCase):
    def test_relocation_types(self):
        self.assertEqual(ElfConst.R_X86_64_64, 1)
        self.assertEqual(ElfConst.R_X86_64_COPY, 5)
        self.assertEqual(ElfConst.R_X86_64_GLOB_DAT, 6)
        self.assertEqual(ElfConst.R_X86_64_JUMP_SLOT, 7)
        self.assertEqual(ElfConst.R_X86_64_RELATIVE, 8)

    def test_aslr_disabled(self):
        self.assertEqual(ElfConst.ASLR_DISABLED, 0x40000)

    def test_page_size_is_power_of_two(self):
        page = ElfConst._PAGE_SIZE
        self.assertEqual(page & (page - 1), 0)


class TestAlign(unittest.TestCase):
    def test_align_down(self):
        self.assertEqual(align_down(0x123456), 0x123000)
        self.assertEqual(align_down(0x1000), 0x1000)

    def test_align_up(self):
        self.assertEqual(align_up(0x123456), 0x124000)
        self.assertEqual(align_up(0x1000), 0x1000)

    def test_custom_alignment(self):
        self.assertEqual(align_up(13, 16), 16)
        self.assertEqual(align_down(13, 16), 0)


class TestPerfMap(unittest.TestCase):
    def test_path_template(self):
        self.assertEqual(perf_map_path(pid=1234), "/tmp/perf-1234.map")
        self.assertTrue(perf_map_path().startswith("/tmp/perf-"))

    def test_write_and_append(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "perf.map")
        out = write_perf_map(
            [(0x1000, 0x20, "a"), (0x2000, 0x30, "b")], path=path, append=False
        )
        self.assertEqual(out, path)
        with open(path) as f:
            self.assertEqual(f.read(), "1000 20 a\n2000 30 b\n")
        write_perf_map([(0x3000, 0x10, "c")], path=path, append=True)
        with open(path) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[2].endswith(" c"))


class TestSaveObject(unittest.TestCase):
    @staticmethod
    def _build(d):
        if shutil.which("gcc") is None:
            return None
        src = os.path.join(d, "t.c")
        exe = os.path.join(d, "t")
        with open(src, "w") as f:
            f.write("long myfunc(long x){return x*3+1;}\nint main(){return 0;}\n")
        r = subprocess.run(["gcc", "-O2", "-o", exe, src], capture_output=True)
        if r.returncode != 0 or not os.path.exists(exe):
            return None
        return exe

    def test_save_object_is_relocatable_with_symbols(self):
        import angr
        from elftools.elf.elffile import ELFFile

        from perf.exec import Elf

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        exe = self._build(d)
        if exe is None:
            self.skipTest("gcc unavailable")
        proj = angr.Project(exe, auto_load_libs=False, load_debug_info=False)
        obj = Elf(proj.loader).map_elf()
        out = os.path.join(d, "t.o")
        obj.save_object(out, harnesses={"perf_bench_myfunc": b"\x90\xc3"})
        with open(out, "rb") as f:
            elf = ELFFile(f)
            self.assertEqual(elf.header["e_type"], "ET_REL")
            self.assertEqual(elf.header["e_machine"], "EM_X86_64")
            sections = {s.name: s for s in elf.iter_sections()}
            self.assertIn(".symtab", sections)
            self.assertIn(".perf.harness", sections)
            self.assertTrue(any(n.startswith(".perf.exec") for n in sections))
            names = [s.name for s in sections[".symtab"].iter_symbols()]
            self.assertIn("myfunc", names)
            self.assertIn("perf_bench_myfunc", names)

    def test_save_object_links_harness_to_symbol(self):
        import angr
        from elftools.elf.elffile import ELFFile
        from elftools.elf.relocation import RelocationSection

        from perf.arch import x86_64 as arch
        from perf.exec import Elf

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        exe = self._build(d)
        if exe is None:
            self.skipTest("gcc unavailable")
        proj = angr.Project(exe, auto_load_libs=False, load_debug_info=False)
        obj = Elf(proj.loader).map_elf()
        harness = bytes(arch.assemble(arch.call_seq_asm(obj.get_symbol("myfunc"))))
        out = os.path.join(d, "t.o")
        obj.save_object(
            out,
            harnesses={"perf_bench_myfunc": harness},
            harness_targets={"perf_bench_myfunc": "myfunc"},
        )
        with open(out, "rb") as f:
            elf = ELFFile(f)
            rela = elf.get_section_by_name(".rela.perf.harness")
            self.assertIsNotNone(rela)
            self.assertTrue(isinstance(rela, RelocationSection))
            symtab = elf.get_section_by_name(".symtab")
            relocs = list(rela.iter_relocations())
            self.assertEqual(len(relocs), 1)
            self.assertEqual(relocs[0]["r_info_type"], 1)
            sym = symtab.get_symbol(relocs[0]["r_info_sym"])
            self.assertEqual(sym.name, "myfunc")


class TestLayoutShuffle(unittest.TestCase):
    @staticmethod
    def _build(d):
        if shutil.which("gcc") is None:
            return None
        src = os.path.join(d, "fb.c")
        exe = os.path.join(d, "fb")
        with open(src, "w") as f:
            f.write(
                'const char* buzz(int n){return n%5==0?"Buzz":"No";}\n'
                "const char* fizz_buzz(int n){\n"
                '  if(n%15==0) return "FizzBuzz";\n'
                '  if(n%3==0) return "Fizz";\n'
                "  return buzz(n);\n"
                "}\n"
                "int main(){return 0;}\n"
            )
        r = subprocess.run(["gcc", "-O2", "-o", exe, src], capture_output=True)
        if r.returncode != 0 or not os.path.exists(exe):
            return None
        return exe

    def _mapped(self, exe):
        import angr

        from perf.exec import Elf
        from perf.info import functions

        proj = angr.Project(exe, auto_load_libs=False, load_debug_info=False)
        funcs, _ = functions(proj)
        obj = Elf(proj.loader).map_elf()
        return obj, dict(funcs)

    def test_shuffle_moves_and_runs(self):
        import ctypes

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        exe = self._build(d)
        if exe is None:
            self.skipTest("gcc unavailable")
        obj, funcs = self._mapped(exe)
        orig = obj.get_symbol("fizz_buzz")
        moved = obj.randomize_layout(funcs, seed=1)
        self.assertIn("fizz_buzz", moved)
        self.assertIn("buzz", moved)
        new = obj.get_symbol("fizz_buzz")
        self.assertNotEqual(new, orig)
        fn = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_int)(new)
        self.assertEqual(
            [fn(n) for n in (15, 3, 5, 7)], [b"FizzBuzz", b"Fizz", b"Buzz", b"No"]
        )

    def test_shuffle_deterministic_per_seed(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        exe = self._build(d)
        if exe is None:
            self.skipTest("gcc unavailable")

        def relative(exe, seed):
            obj, funcs = self._mapped(exe)
            moved = obj.randomize_layout(funcs, seed=seed)
            base = min(moved.values())
            return {n: a - base for n, a in moved.items()}

        self.assertEqual(relative(exe, 11), relative(exe, 11))
        self.assertNotEqual(relative(exe, 11), relative(exe, 12))

    def test_bad_alignment_rejected(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        exe = self._build(d)
        if exe is None:
            self.skipTest("gcc unavailable")
        obj, funcs = self._mapped(exe)
        with self.assertRaises(ValueError):
            obj.randomize_layout(funcs, align=24)
        with self.assertRaises(ValueError):
            obj.randomize_layout(funcs, align="many")


class TestLinkObject(unittest.TestCase):
    @staticmethod
    def _compile_object(d, compiler, code):
        if shutil.which(compiler) is None:
            return None
        src = os.path.join(d, "link.c")
        obj = os.path.join(d, "link.o")
        with open(src, "w") as f:
            f.write(code)
        r = subprocess.run([compiler, "-O2", "-c", src, "-o", obj], capture_output=True)
        if r.returncode != 0 or not os.path.exists(obj):
            return None
        return obj

    def test_links_an_object_once_per_file(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        obj = self._compile_object(d, "gcc", "long triple(long x){return x*3;}\n")
        if obj is None:
            self.skipTest("gcc unavailable")
        from perf.exec import resolve_exec

        self.assertEqual(resolve_exec(obj), resolve_exec(obj))

    def test_the_link_keeps_the_object_name(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        obj = self._compile_object(d, "gcc", "long triple(long x){return x*3;}\n")
        if obj is None:
            self.skipTest("gcc unavailable")
        self.assertEqual(os.path.basename(resolve_exec(obj)), os.path.basename(obj))

    def test_the_link_keeps_the_archive_name(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        obj = self._compile_object(d, "gcc", "long triple(long x){return x*3;}\n")
        if obj is None:
            self.skipTest("gcc unavailable")
        archive = os.path.join(d, "bundle.a")
        r = subprocess.run(["ar", "rcs", archive, obj], capture_output=True)
        if r.returncode != 0 or not os.path.exists(archive):
            self.skipTest("ar unavailable")
        self.assertEqual(
            os.path.basename(resolve_exec(archive)), os.path.basename(archive)
        )

    def test_the_perf_map_names_the_original_object(self):
        import angr

        from perf.exec import Elf

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        obj = self._compile_object(d, "gcc", "long triple(long x){return x*3;}\n")
        if obj is None:
            self.skipTest("gcc unavailable")
        proj = angr.Project(
            resolve_exec(obj), auto_load_libs=False, load_debug_info=False
        )
        elf = Elf(proj.loader).map_elf()
        mapped = [name for _a, _s, name in elf.perf_map_entries()]
        self.assertTrue(any(name.startswith("perf_exec:") for name in mapped), mapped)
        self.assertTrue(
            all(not name.endswith(":linked") for name in mapped),
            mapped,
        )

    def test_links_plain_c_object(self):
        import angr

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        obj = self._compile_object(d, "gcc", "long triple(long x){return x*3;}\n")
        if obj is None:
            self.skipTest("gcc unavailable")
        exe = resolve_exec(obj)
        proj = angr.Project(exe, auto_load_libs=False, load_debug_info=False)
        self.assertGreater(len(proj.loader.main_object.segments), 0)
        self.assertIn("triple", [s.name for s in proj.loader.main_object.symbols])

    def test_links_cxx_object(self):
        import angr

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        obj = self._compile_object(
            d,
            "g++",
            '#include <string>\nstd::string greet(){return "hi";}\n',
        )
        if obj is None:
            self.skipTest("g++ unavailable")
        exe = resolve_exec(obj)
        proj = angr.Project(exe, auto_load_libs=False, load_debug_info=False)
        self.assertGreater(len(proj.loader.main_object.segments), 0)
        self.assertTrue(any("greet" in s.name for s in proj.loader.main_object.symbols))

    def test_links_archive(self):
        import angr

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        obj = self._compile_object(d, "gcc", "long triple(long x){return x*3;}\n")
        if obj is None:
            self.skipTest("gcc unavailable")
        archive = os.path.join(d, "link.a")
        r = subprocess.run(["ar", "rcs", archive, obj], capture_output=True)
        if r.returncode != 0 or not os.path.exists(archive):
            self.skipTest("ar unavailable")
        exe = resolve_exec(archive)
        proj = angr.Project(exe, auto_load_libs=False, load_debug_info=False)
        self.assertGreater(len(proj.loader.main_object.segments), 0)


class TestToObjectEdges(unittest.TestCase):
    @staticmethod
    def _build(d):
        if shutil.which("gcc") is None:
            return None
        src = os.path.join(d, "o.c")
        exe = os.path.join(d, "o")
        with open(src, "w") as f:
            f.write(
                "long add42(long x){return x+42;}\n"
                "long mul3(long x){return x*3;}\n"
                "int main(){return (int)add42(1);}\n"
            )
        r = subprocess.run(["gcc", "-O2", "-o", exe, src], capture_output=True)
        if r.returncode != 0 or not os.path.exists(exe):
            return None
        return exe

    @staticmethod
    def _names(path):
        from elftools.elf.elffile import ELFFile

        with open(path, "rb") as f:
            sec = ELFFile(f).get_section_by_name(".symtab")
            return [s.name for s in sec.iter_symbols()]

    def test_missing_file_raises(self):
        from perf.exec import to_object

        with self.assertRaises(ValueError):
            to_object("/nonexistent/a.out:func")

    def test_needs_a_binary_target(self):
        from perf.exec import to_object

        with self.assertRaises(ValueError):
            to_object("mov eax, 42")

    def test_bare_file_writes_every_function(self):
        from perf.exec import to_object

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        exe = self._build(d)
        if exe is None:
            self.skipTest("gcc unavailable")
        out = os.path.join(d, "all.o")
        to_object(exe, path=out)
        names = self._names(out)
        self.assertIn("perf_bench_add42", names)
        self.assertIn("perf_bench_mul3", names)
        self.assertIn("perf_bench_main", names)

    def test_region_string_writes_a_harness(self):
        from perf.exec import to_object

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        exe = self._build(d)
        if exe is None:
            self.skipTest("gcc unavailable")
        out = os.path.join(d, "region.o")
        to_object(f"{exe}:add42..mul3", path=out)
        self.assertIn("perf_bench_add42__mul3", self._names(out))

    def test_region_pair_writes_a_harness(self):
        from perf.exec import to_object

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        exe = self._build(d)
        if exe is None:
            self.skipTest("gcc unavailable")
        out = os.path.join(d, "region.o")
        to_object([exe, ("add42", "mul3")], path=out)
        self.assertIn("perf_bench_add42__mul3", self._names(out))

    def test_address_target_writes_a_harness(self):
        import angr

        from perf.exec import to_object
        from perf.info import functions

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        exe = self._build(d)
        if exe is None:
            self.skipTest("gcc unavailable")
        proj = angr.Project(exe, auto_load_libs=False, load_debug_info=False)
        funcs, _ = functions(proj)
        _start, end = funcs["add42"]
        out = os.path.join(d, "addr.o")
        to_object(f"{exe}:{end:#x}", path=out)
        self.assertTrue(
            any(n.startswith("perf_bench_") for n in self._names(out)),
            self._names(out),
        )


class TestAsmSource(unittest.TestCase):
    def test_resolve_exec_leaves_assembly_alone(self):
        from perf.exec import resolve_exec

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        src = os.path.join(d, "g.s")
        with open(src, "w") as f:
            f.write(".globl gfunc\n gfunc:\n  ret\n")
        self.assertEqual(resolve_exec(src), src)


class TestExecNoImportSideEffects(unittest.TestCase):
    def test_aslr_disabled_lazily(self):
        from perf.exec import Elf, ElfConst

        self.assertTrue(hasattr(Elf, "_disable_aslr"))
        src = inspect.getsource(Elf)
        self.assertIn("_disable_aslr", src)
        self.assertEqual(ElfConst.ASLR_DISABLED, 0x40000)


class TestRawSymbolNamespace(unittest.TestCase):
    def _elf(self, values, demangled):
        from unittest.mock import Mock, patch

        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.loader = Mock()
        obj.loader.main_object.binary = "mocked"
        patcher = patch(
            "perf.exec._symtab_values", return_value=(dict(values), dict(demangled))
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return obj

    def test_bare_name_matches_namespaced_symbol(self):
        elf = self._elf({"_ZN2ns3fooEv": 0x1000}, {"ns::foo()": 0x1000})
        self.assertEqual(elf.raw_symbol("foo"), "_ZN2ns3fooEv")

    def test_qualified_name_does_not_suffix_match(self):
        elf = self._elf({"_ZN2ns3fooEv": 0x1000}, {"ns::foo()": 0x1000})
        self.assertIsNone(elf.raw_symbol("other::foo"))

    def test_ambiguous_suffix_returns_none(self):
        elf = self._elf(
            {"_ZN1a3fooEv": 0x1000, "_ZN1b3fooEv": 0x2000},
            {"a::foo()": 0x1000, "b::foo()": 0x2000},
        )
        self.assertIsNone(elf.raw_symbol("foo"))

    def test_same_address_suffix_returns_a_candidate(self):
        elf = self._elf(
            {"_ZN1a3fooEv": 0x1000, "_ZN1b3fooEv": 0x1000},
            {"a::foo()": 0x1000, "b::foo()": 0x1000},
        )
        self.assertIn(elf.raw_symbol("foo"), ("_ZN1a3fooEv", "_ZN1b3fooEv"))

    def test_demangled_signature_resolves_to_raw(self):
        elf = self._elf({"_ZN2ns3fooEi": 0x1140}, {"ns::foo(int)": 0x1140})
        self.assertEqual(elf.raw_symbol("ns::foo(int)"), "_ZN2ns3fooEi")

    def test_ambiguous_short_exact_returns_none(self):
        elf = self._elf(
            {"_ZN2ns3fooEi": 0x1000, "_ZN2ns3fooEd": 0x2000},
            {"ns::foo(int)": 0x1000, "ns::foo(double)": 0x2000},
        )
        self.assertIsNone(elf.raw_symbol("ns::foo"))

    def test_missing_symbol_returns_none(self):
        elf = self._elf({"_ZN2ns3fooEv": 0x1000}, {"ns::foo()": 0x1000})
        self.assertIsNone(elf.raw_symbol("bar"))


class TestResolveExecHardening(unittest.TestCase):
    def test_asm_uppercase_passes_through(self):
        from perf.exec import resolve_exec

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        src = os.path.join(d, "g.S")
        with open(src, "w") as f:
            f.write(".globl gfunc\n gfunc:\n  ret\n")
        self.assertEqual(resolve_exec(src), src)

    def test_linker_script_resolves_to_target(self):
        from perf.exec import resolve_exec

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        real = os.path.join(d, "libreal.so")
        with open(real, "wb") as f:
            f.write(b"\x7fELFfake")
        script = os.path.join(d, "libc.so")
        with open(script, "w") as f:
            f.write("GROUP ( " + real + " )")
        self.assertEqual(resolve_exec(script), real)

    def test_linker_script_without_target_raises(self):
        from perf.exec import resolve_exec

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        script = os.path.join(d, "libc.so")
        with open(script, "w") as f:
            f.write("GROUP ( /nonexistent-perf-test-lib.so )")
        with self.assertRaises(ValueError) as ctx:
            resolve_exec(script)
        self.assertIn("linker script", str(ctx.exception))

    def test_non_elf_raises(self):
        from perf.exec import resolve_exec

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "note.txt")
        with open(path, "w") as f:
            f.write("hello")
        with self.assertRaises(ValueError) as ctx:
            resolve_exec(path)
        self.assertIn("not an ELF", str(ctx.exception))

    def test_missing_file_passes_through(self):
        from perf.exec import resolve_exec

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        missing = os.path.join(d, "nope")
        self.assertEqual(resolve_exec(missing), missing)

    def test_needs_link_failure_still_resolves_elf(self):
        from unittest.mock import patch

        from perf.exec import resolve_exec

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
        src = os.path.join(d, "t.c")
        exe = os.path.join(d, "t")
        with open(src, "w") as f:
            f.write("int main(){return 0;}\n")
        r = subprocess.run(["gcc", "-O2", "-o", exe, src], capture_output=True)
        if r.returncode != 0:
            self.skipTest("gcc build failed")
        with patch("perf.exec.needs_link", side_effect=RuntimeError("boom")):
            self.assertEqual(resolve_exec(exe), exe)

    def test_shared_library_without_soname_is_recognised(self):
        from perf.exec import is_shared_library

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
        src = os.path.join(d, "s.c")
        out = os.path.join(d, "libnosoname.so")
        with open(src, "w") as f:
            f.write("long f(long x){return x+1;}\n")
        r = subprocess.run(
            ["gcc", "-O2", "-fPIC", "-shared", "-o", out, src],
            capture_output=True,
        )
        if r.returncode != 0:
            self.skipTest("gcc build failed")
        self.assertTrue(is_shared_library(out))


SHARED_SOURCE = """
__attribute__((noinline)) long shared_scale(long x) { return x * 7 + 3; }
"""
PRINTING_SOURCE = """
#include <stdio.h>
__attribute__((noinline)) long chatty(long n) {
  fprintf(stderr, "e%ld\\n", n);
  fprintf(stdout, "o%ld\\n", n);
  return n + 1;
}
int main(void) { return (int)chatty(1); }
"""
ARGS_SOURCE = """
volatile unsigned long args_sink;
volatile unsigned long *args_mark;

__attribute__((noinline)) long record_args(int argc, char **argv, char **envp) {
  unsigned long *mark = args_mark;
  mark[0] = (unsigned long)argc;
  mark[1] = (unsigned long)argv;
  mark[2] = (unsigned long)envp;
  mark[3] = (unsigned long)argv[0];
  mark[4] = (unsigned long)(argc > 1 ? argv[1] : 0);
  mark[5] = (unsigned long)(argc > 2 ? argv[2] : 0);
  mark[6] = (unsigned long)(envp ? envp[0] : 0);
  mark[7] = (unsigned long)(envp && envp[1] ? envp[1] : 0);
  mark[8] = *(unsigned long *)((char *)__builtin_frame_address(0) + 8 + 8 * 3);
  args_sink = mark[0];
  return mark[0];
}

int main(int argc, char **argv, char **envp) {
  return (int)record_args(argc, argv, envp);
}
"""


def _build_shared(d, name, source, compiler="gcc"):
    if shutil.which(compiler) is None:
        return None
    src = os.path.join(d, f"{name}.c")
    out = os.path.join(d, f"lib{name}.so")
    with open(src, "w") as fh:
        fh.write(source)
    r = subprocess.run(
        [
            compiler,
            "-O2",
            "-fPIC",
            "-shared",
            f"-Wl,-soname,lib{name}.so",
            "-o",
            out,
            src,
        ],
        capture_output=True,
    )
    return out if r.returncode == 0 and os.path.exists(out) else None


def _build_exe(d, name, source, *flags):
    if shutil.which("gcc") is None:
        return None
    src = os.path.join(d, f"{name}.c")
    out = os.path.join(d, name)
    with open(src, "w") as fh:
        fh.write(source)
    r = subprocess.run(["gcc", "-O2", *flags, "-o", out, src], capture_output=True)
    return out if r.returncode == 0 and os.path.exists(out) else None


def _mapped(path):
    import angr

    from perf.exec import Elf

    proj = angr.Project(path, auto_load_libs=False, load_debug_info=False)
    return Elf(proj.loader).map_elf()


class TestSharedLibraryTargets(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._dir, True)
        self.so = _build_shared(self._dir, "demo", SHARED_SOURCE)
        if self.so is None:
            self.skipTest("gcc build failed")

    def test_a_shared_library_is_recognised(self):
        from perf.exec import is_shared_library

        self.assertTrue(is_shared_library(self.so))
        self.assertFalse(is_shared_library(__file__))

    def test_an_executable_is_not_a_library(self):
        from perf.exec import is_shared_library

        exe = _build_exe(
            self._dir,
            "plain",
            "__attribute__((noinline)) int k(int x){return x;}\n"
            "int main(void){return k(1);}\n",
        )
        if exe is None:
            self.skipTest("gcc build failed")
        self.assertFalse(is_shared_library(exe))

    def test_copy_relocation_points_at_the_real_stream(self):
        import ctypes

        chatty = _build_exe(self._dir, "chatty", PRINTING_SOURCE)
        if chatty is None:
            self.skipTest("gcc build failed")
        obj = _mapped(chatty)
        copies = {
            rec["sym"]: obj.runtime_addr(int(rec["offset"]))
            for rec in obj.applied_relocs
            if rec["type"] == 5
        }
        self.assertTrue({"stdout", "stderr"} & set(copies), copies)
        for name, slot in copies.items():
            if name not in ("stdout", "stderr"):
                continue
            from perf.exec import _dlsym

            real = _dlsym(obj.libc, name)
            copied = ctypes.c_uint64.from_address(slot).value
            self.assertEqual(copied, ctypes.c_uint64.from_address(real).value)

    def test_a_shared_library_symbol_is_runnable(self):
        import ctypes

        obj = _mapped(self.so)
        fn = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_long)(
            obj.get_symbol("shared_scale")
        )
        self.assertEqual(fn(6), 45)

    def test_initializers_are_not_run_for_a_library(self):
        obj = _mapped(self.so)
        self.assertEqual(obj.run_initializers(), [])

    def test_relocated_addresses_cover_every_slot(self):
        obj = _mapped(self.so)
        self.assertTrue(obj.applied_relocs)
        self.assertEqual(len(obj.relocated_addresses()), len(obj.applied_relocs))


class TestNonPieImage(unittest.TestCase):
    def test_a_live_range_reports_its_own_overlap(self):
        from perf.exec import _live_overlap, _live_ranges

        ranges = _live_ranges()
        self.assertTrue(ranges)
        taken = ranges[0]
        self.assertEqual(_live_overlap(taken[0], taken[1] - taken[0]), taken)

    def test_map_elf_refuses_to_overwrite_the_process(self):
        from perf.exec import _live_overlap

        exe = _build_exe(
            self._dir(),
            "npie",
            "__attribute__((noinline)) int k(int x){return x;}\n"
            "int main(void){return k(1);}\n",
            "-no-pie",
        )
        if exe is None:
            self.skipTest("gcc build failed")
        from elftools.elf.elffile import ELFFile

        with open(exe, "rb") as fh:
            elf = ELFFile(fh)
            if elf.header["e_type"] == "ET_DYN":
                self.skipTest("linker produced a position-independent binary")
            low = min(
                int(seg["p_vaddr"])
                for seg in elf.iter_segments()
                if seg["p_type"] == "PT_LOAD"
            )
            high = max(
                int(seg["p_vaddr"]) + int(seg["p_memsz"])
                for seg in elf.iter_segments()
                if seg["p_type"] == "PT_LOAD"
            )
        if _live_overlap(low, high - low) is None:
            _mapped(exe)
            return
        import angr

        from perf.exec import Elf

        proj = angr.Project(exe, auto_load_libs=False, load_debug_info=False)
        with self.assertRaises(ValueError) as ctx:
            Elf(proj.loader).map_elf()
        self.assertIn("already", str(ctx.exception))
        self.assertIn("position-independent", str(ctx.exception))

    @staticmethod
    def _dir():
        import atexit

        d = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, d, True)
        return d


class TestOutputFaking(unittest.TestCase):
    def test_default_config_fakes_both_streams(self):
        from perf.bench import _DEFAULT_BENCH, _fake_stderr, _fake_stdout

        self.assertFalse(_DEFAULT_BENCH["external"]["stdout"])
        self.assertFalse(_DEFAULT_BENCH["external"]["stderr"])
        self.assertTrue(_fake_stdout(_DEFAULT_BENCH))
        self.assertTrue(_fake_stderr(_DEFAULT_BENCH))

    def test_a_missing_key_fakes(self):
        from perf.bench import _fake_stderr, _fake_stdout

        self.assertTrue(_fake_stdout({}))
        self.assertTrue(_fake_stderr({}))

    def test_true_lets_the_target_write(self):
        from perf.bench import _fake_stderr, _fake_stdout

        spec = {"external": {"stdout": True, "stderr": True}}
        self.assertFalse(_fake_stdout(spec))
        self.assertFalse(_fake_stderr(spec))

    def test_strings_follow_the_same_polarity(self):
        from perf.bench import _fake_stdout

        self.assertFalse(_fake_stdout({"external": {"stdout": "real"}}))
        self.assertFalse(_fake_stdout({"external": {"stdout": "true"}}))
        self.assertTrue(_fake_stdout({"external": {"stdout": "fake"}}))
        self.assertTrue(_fake_stdout({"external": {"stdout": "false"}}))

    def test_validation_rejects_an_unknown_key(self):
        from perf.bench import _validate_lib

        with self.assertRaises(ValueError) as ctx:
            _validate_lib({"external": {"nope": True}})
        self.assertIn("nope", str(ctx.exception))

    def test_validation_rejects_a_non_boolean(self):
        from perf.bench import _validate_lib

        with self.assertRaises(ValueError) as ctx:
            _validate_lib({"external": {"stdout": 1.5}})
        self.assertIn("stdout", str(ctx.exception))

    def test_validation_accepts_the_booleans(self):
        from perf.bench import _validate_lib

        _validate_lib({"external": {"lib": False, "stdout": False, "stderr": True}})

    def test_a_faked_stream_keeps_what_libc_buffered(self):
        import os
        import tempfile

        from perf.bench import _fake_output

        before = os.dup(1)
        try:
            with _fake_output({"external": {"stdout": False}}):
                os.write(1, b"perf-should-not-see-this\n")
                with tempfile.TemporaryFile() as sink:
                    os.dup2(sink.fileno(), 1)
                    os.write(1, b"perf-discards-this\n")
        finally:
            os.dup2(before, 1)
            os.close(before)


class TestReservedAddresses(unittest.TestCase):
    def test_relocation_slots_are_not_steered(self):
        from perf.bench import _reserved_addresses, _static_table_addrs

        exe = _build_exe(
            tempfile.mkdtemp(),
            "reserved",
            "#include <stdio.h>\n"
            '__attribute__((noinline)) long q(long x){ fprintf(stderr, "%ld\\n", x);'
            " return x+1; }\n"
            "int main(void){return (int)q(1);}\n",
        )
        if exe is None:
            self.skipTest("gcc build failed")
        import angr

        from perf.info import functions

        proj = angr.Project(exe, auto_load_libs=False, load_debug_info=False)
        obj = _mapped(exe)
        funcs, _ = functions(proj)
        start, end = funcs["q"]
        reserved = _reserved_addresses(obj)
        self.assertTrue(reserved)
        found = _static_table_addrs(proj, start, end, obj, solutions=None)
        self.assertEqual([a for a in found if a in reserved], [])


class TestIfuncSymbols(unittest.TestCase):
    def test_ifuncs_are_functions(self):
        from perf.info import _FUNCTION_TYPES, _is_function_type

        self.assertIn("STT_FUNC", _FUNCTION_TYPES)
        self.assertIn("STT_GNU_IFUNC", _FUNCTION_TYPES)
        self.assertIn("STT_LOOS", _FUNCTION_TYPES)
        self.assertTrue(_is_function_type("STT_GNU_IFUNC"))
        self.assertTrue(_is_function_type("STT_LOOS"))
        self.assertFalse(_is_function_type("STT_OBJECT"))

    def test_libc_ifuncs_resolve_to_the_installed_ones(self):
        from perf.exec import _ifunc_symbols

        libc = None
        for path in ("/lib/x86_64-linux-gnu/libc.so.6", "/lib64/libc.so.6"):
            if os.path.exists(path):
                libc = path
                break
        if libc is None:
            self.skipTest("no libc.so.6")
        names = {name for name, _v, _s in _ifunc_symbols(libc)}
        self.assertIn("strlen", names)
        self.assertIn("memcpy", names)


class TestProcessStack(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._dir, True)
        self.exe = _build_exe(self._dir, "args", ARGS_SOURCE, "-fPIE", "-pie")
        if self.exe is None:
            self.skipTest("gcc build failed")
        import angr

        from perf.exec import Elf

        proj = angr.Project(self.exe, auto_load_libs=True, load_debug_info=False)
        self.elf = Elf(proj.loader)
        self.elf.map_elf()
        self.elf.setup_stack()

    def _frame(self, argv=("prog", "one", "two"), env=("A=1", "B=2")):
        return self.elf.setup_argv(list(argv), list(env))

    def _string(self, addr):
        return self.elf.read_memory(addr, 64).split(b"\0")[0].decode()

    def _at(self, addr):
        import struct

        return struct.unpack("<Q", self.elf.read_memory(addr, 8))[0]

    def _header(self, key):
        from elftools.elf.elffile import ELFFile

        with open(self.exe, "rb") as fh:
            return int(ELFFile(fh).header[key])

    def test_argc_and_the_vectors_are_written(self):

        frame = self._frame()
        self.assertEqual(frame["argc"], 3)
        self.assertEqual(self._at(frame["sp"]), 3)
        for i, word in enumerate(("prog", "one", "two")):
            self.assertEqual(self._string(self._at(frame["argv"] + 8 * i)), word)
        self.assertEqual(self._at(frame["argv"] + 8 * 3), 0)
        for i, word in enumerate(("A=1", "B=2")):
            self.assertEqual(self._string(self._at(frame["envp"] + 8 * i)), word)
        self.assertEqual(self._at(frame["envp"] + 8 * 2), 0)

    def test_the_stack_pointer_is_the_kernel_layout(self):
        frame = self._frame()
        self.assertEqual(frame["sp"] % 16, 8)
        self.assertLess(frame["sp"], self.elf.stack_top)
        self.assertGreater(frame["sp"], self.elf.stack_base)
        self.assertEqual(frame["argv"], frame["sp"] + 8)
        self.assertEqual(frame["envp"], frame["sp"] + 8 * 5)
        self.assertEqual(frame["auxv"], frame["sp"] + 8 * 8)

    def test_the_auxiliary_vector_is_elf_derived_and_terminated(self):
        import struct

        frame = self._frame()
        auxv = []
        for i in range(8):
            key, value = struct.unpack(
                "<QQ", self.elf.read_memory(frame["auxv"] + 16 * i, 16)
            )
            auxv.append((key, value))
            if key == 0:
                break
        self.assertEqual(auxv[-1], (0, 0))
        self.assertEqual(len(auxv), len(self.elf.auxv()))
        entry = self.elf._offset_to_vaddr(self._header("e_entry"))
        self.assertEqual(auxv[-2], (9, self.elf.runtime_addr(entry)))
        phdr = self.elf._offset_to_vaddr(self._header("e_phoff"))
        self.assertEqual(auxv[-6], (3, self.elf.runtime_addr(phdr)))
        self.assertEqual(auxv[-5], (4, self._header("e_phentsize")))
        self.assertEqual(auxv[-4], (5, self._header("e_phnum")))
        self.assertEqual(auxv[-3], (6, mmap.PAGESIZE))

    def test_everything_lives_inside_the_mapped_stack(self):
        frame = self._frame(argv=("prog", "x" * 4096))
        self.assertGreaterEqual(frame["sp"], self.elf.stack_base)
        self.assertLess(frame["sp"], self.elf.stack_top)
        self.assertLessEqual(frame["save"] + 8, self.elf.stack_top)

    def test_the_frame_sits_above_the_stack_the_target_gets(self):
        frame = self._frame(argv=("prog", "x" * 4096))
        self.assertGreater(frame["strings"], frame["sp"])
        self.assertGreater(frame["save"], frame["strings"])

    def test_the_space_below_the_stack_the_target_gets_is_its_own(self):
        frame = self._frame()
        below = self.elf.read_memory(frame["sp"] - 4096, 4096)
        self.assertEqual(below, b"\0" * 4096)

    def test_a_save_slot_holds_the_old_stack_pointer(self):
        import struct

        frame = self._frame()
        saved = struct.unpack("<Q", self.elf.read_memory(frame["save"], 8))[0]
        self.assertEqual(saved, self.elf.stack_top)

    def test_argv_needs_a_program_name(self):
        with self.assertRaises(ValueError):
            self.elf.setup_argv([], [])

    def test_argv_before_the_stack_is_an_error(self):
        self.elf.stack_top = self.elf.stack_base + 0x100
        with self.assertRaises(ValueError):
            self.elf.setup_argv(["prog", "y" * 0x4000])

    def test_a_later_call_replaces_the_image(self):
        first = self._frame(argv=("prog", "one"))
        second = self._frame(argv=("prog", "two", "three"))
        self.assertEqual(second["argc"], 3)
        self.assertEqual(second["sp"], first["sp"])
        self.assertGreater(second["strings"], first["strings"])
        self.assertEqual(self._string(self._at(second["argv"] + 8)), "two")
        self.assertEqual(
            self.elf.read_memory(second["strings"], 16), b"prog\0two\0three\0A"
        )


if __name__ == "__main__":
    unittest.main()
