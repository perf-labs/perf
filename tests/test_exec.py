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
        page = ElfConst.PAGE_SIZE
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


class TestExecNoImportSideEffects(unittest.TestCase):
    def test_aslr_disabled_lazily(self):
        from perf.exec import Elf, ElfConst

        self.assertTrue(hasattr(Elf, "_disable_aslr"))
        src = inspect.getsource(Elf)
        self.assertIn("_disable_aslr", src)
        self.assertEqual(ElfConst.ASLR_DISABLED, 0x40000)


if __name__ == "__main__":
    unittest.main()
