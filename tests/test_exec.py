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
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


class TestOutputSinkThreadSafety(unittest.TestCase):
    def test_output_sink_holds_the_shared_redirection_lock(self):
        import perf.exec as exec_mod

        entered = threading.Event()
        release = threading.Event()

        def worker():
            with exec_mod._output_sink([]):
                entered.set()
                release.wait(5)

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertFalse(exec_mod._OUTPUT_LOCK.acquire(blocking=False))
        finally:
            release.set()
            thread.join(timeout=10)
        self.assertFalse(thread.is_alive())


class TestToObjectEdges(unittest.TestCase):
    def test_missing_file_raises(self):
        from perf.exec import to_object

        with self.assertRaises(ValueError):
            to_object(symbol="/nonexistent/a.out:func")

    def test_needs_a_binary_target(self):
        from perf.exec import to_object

        with self.assertRaises(ValueError):
            to_object(symbol="mov eax, 42")


class TestAsmSource(unittest.TestCase):
    def test_resolve_exec_leaves_assembly_alone(self):

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


class TestSymtabValuesFallback(unittest.TestCase):
    def test_empty_symtab_falls_through_to_dynsym(self):
        import importlib
        from unittest.mock import patch

        exec_mod = importlib.import_module("perf.exec")

        class _Sym:
            def __init__(self, name, value):
                self.name = name
                self.entry = {"st_value": value}

        class _Section:
            def __init__(self, symbols):
                self._symbols = symbols

            def iter_symbols(self):
                return iter(self._symbols)

        class _Elf:
            def get_section_by_name(self, name):
                if name == ".symtab":
                    return _Section([])
                if name == ".dynsym":
                    return _Section([_Sym("exported", 0x1234)])
                return None

        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.addCleanup(os.remove, path)
        with patch.object(exec_mod, "ELFFile", return_value=_Elf()):
            values, demangled = exec_mod._symtab_values(path)
        self.assertEqual(values, {"exported": 0x1234})
        self.assertEqual(demangled, {"exported": 0x1234})


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

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        src = os.path.join(d, "g.S")
        with open(src, "w") as f:
            f.write(".globl gfunc\n gfunc:\n  ret\n")
        self.assertEqual(resolve_exec(src), src)

    def test_linker_script_resolves_to_target(self):

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

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        script = os.path.join(d, "libc.so")
        with open(script, "w") as f:
            f.write("GROUP ( /nonexistent-perf-test-lib.so )")
        with self.assertRaises(ValueError) as ctx:
            resolve_exec(script)
        self.assertIn("linker script", str(ctx.exception))

    def test_non_elf_raises(self):

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "note.txt")
        with open(path, "w") as f:
            f.write("hello")
        with self.assertRaises(ValueError) as ctx:
            resolve_exec(path)
        self.assertIn("not an ELF", str(ctx.exception))

    def test_missing_file_passes_through(self):

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        missing = os.path.join(d, "nope")
        self.assertEqual(resolve_exec(missing), missing)


class TestNonPieImage(unittest.TestCase):
    def test_a_live_range_reports_its_own_overlap(self):
        from perf.exec import _live_overlap, _live_ranges

        ranges = _live_ranges()
        self.assertTrue(ranges)
        taken = ranges[0]
        self.assertEqual(_live_overlap(taken[0], taken[1] - taken[0]), taken)

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


if __name__ == "__main__":
    unittest.main()


class TestLinkObjectUnit(unittest.TestCase):
    def setUp(self):
        from perf import exec as exec_mod

        self.exec_mod = exec_mod
        self._before = set(exec_mod._OBJ_LINK_DEPS)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for d in set(self.exec_mod._OBJ_LINK_DEPS) - self._before:
            shutil.rmtree(d, ignore_errors=True)
        self.exec_mod._OBJ_LINK_DEPS[:] = [
            d for d in self.exec_mod._OBJ_LINK_DEPS if d in self._before
        ]

    def _run(self, path, *, archive=False, compilers=(("cc",),), outcomes=None):
        if outcomes is None:
            outcomes = [(0, "")]

        def fake_run(cmd, **kwargs):
            outcome = outcomes[min(len(self._calls), len(outcomes) - 1)]
            self._calls.append(list(cmd))
            if outcome[0] == 0:
                out = cmd[cmd.index("-o") + 1]
                Path(out).write_text("")
            return SimpleNamespace(returncode=outcome[0], stderr=outcome[1])

        self._calls = []
        with (
            patch.object(self.exec_mod, "needs_link", return_value=True),
            patch.object(self.exec_mod, "is_archive", return_value=archive),
            patch.object(
                self.exec_mod,
                "_candidate_compilers",
                return_value=[list(c) for c in compilers],
            ),
            patch.object(self.exec_mod.subprocess, "run", side_effect=fake_run),
        ):
            return self.exec_mod.link_object(path)

    def test_not_relocatable_is_returned_unchanged(self):
        with patch.object(self.exec_mod, "needs_link", return_value=False):
            self.assertEqual(self.exec_mod.link_object("/tmp/a.out"), "/tmp/a.out")

    def test_archive_uses_whole_archive(self):
        out = self._run("/tmp/lib.a", archive=True)
        self.assertTrue(os.path.exists(out))
        self.assertTrue(any("whole-archive" in arg for arg in self._calls[0]))
        self.assertIn("/tmp/lib.a", self._calls[0])

    def test_plain_object_is_linked_without_archive_flags(self):
        out = self._run("/tmp/lib.o")
        self.assertTrue(os.path.exists(out))
        self.assertNotIn("--whole-archive", self._calls[0])

    def test_tries_the_next_compiler_after_a_failure(self):
        out = self._run(
            "/tmp/lib.o",
            compilers=(("bad",), ("good",)),
            outcomes=[(1, "pie"), (1, "no-pie"), (0, "")],
        )
        self.assertTrue(os.path.exists(out))
        self.assertEqual(len(self._calls), 3)
        self.assertEqual(self._calls[0][0], "bad")
        self.assertEqual(self._calls[1][0], "bad")
        self.assertEqual(self._calls[2][0], "good")

    def test_pie_and_non_pie_are_both_tried(self):
        self._run("/tmp/lib.o", outcomes=[(1, "pie"), (0, "")])
        flags = [arg for cmd in self._calls for arg in cmd]
        self.assertIn("-pie", flags)
        self.assertIn("-fno-pie", flags)

    def test_all_failures_report_the_last_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run("/tmp/lib.o", outcomes=[(1, "first"), (1, "last")])
        self.assertIn("last", str(ctx.exception))

    def test_no_compiler_is_an_error(self):
        with (
            patch.object(self.exec_mod, "needs_link", return_value=True),
            patch.object(self.exec_mod, "_candidate_compilers", return_value=[]),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.exec_mod.link_object("/tmp/lib.o")
        self.assertIn("no compiler", str(ctx.exception))

    def test_link_name_edges(self):
        self.assertEqual(self.exec_mod._link_name("/tmp/a.o"), "a.o")
        self.assertEqual(self.exec_mod._link_name(""), "perf-linked")
        self.assertEqual(self.exec_mod._link_name("."), "perf-linked")


class TestSharedLibraryDetectionUnit(unittest.TestCase):
    def _check(self, header, sections, expected):
        elf = MagicMock()
        elf.header = {"e_type": header}
        elf.get_section_by_name.side_effect = lambda name: sections.get(name)
        with (
            patch("builtins.open", MagicMock()),
            patch.object(self.exec_mod, "ELFFile", return_value=elf),
        ):
            self.assertEqual(self.exec_mod.is_shared_library("/tmp/x"), expected)

    def setUp(self):
        from perf import exec as exec_mod

        self.exec_mod = exec_mod

    def test_soname_marks_a_library(self):
        dynamic = MagicMock()
        dynamic.iter_tags.return_value = [
            SimpleNamespace(entry=SimpleNamespace(d_tag="DT_SONAME"))
        ]
        self._check("ET_DYN", {".dynamic": dynamic}, True)

    def test_dyn_without_interp_is_a_library(self):
        dynamic = MagicMock()
        dynamic.iter_tags.return_value = []
        self._check("ET_DYN", {".dynamic": dynamic}, True)

    def test_dyn_with_interp_is_an_executable(self):
        dynamic = MagicMock()
        dynamic.iter_tags.return_value = []
        self._check("ET_DYN", {".dynamic": dynamic, ".interp": MagicMock()}, False)

    def test_non_dyn_is_not_a_library(self):
        self._check("ET_EXEC", {}, False)

    def test_missing_dynamic_is_not_a_library(self):
        self._check("ET_DYN", {}, False)

    def test_unreadable_file_is_not_a_library(self):
        with patch("builtins.open", side_effect=OSError("no")):
            self.assertFalse(self.exec_mod.is_shared_library("/tmp/nope"))


class TestElfRelocationsUnit(unittest.TestCase):
    def setUp(self):
        from perf import exec as exec_mod
        from perf.exec import Elf

        self.exec_mod = exec_mod
        self.Elf = Elf

    def _elf(self):
        obj = self.Elf.__new__(self.Elf)
        obj.applied_relocs = []
        obj.initializers_done = False
        obj.ran_initializers = []
        obj.loader = SimpleNamespace(main_object=SimpleNamespace(binary="/tmp/lib.so"))
        return obj

    def test_relocated_addresses_uses_applied_relocations(self):
        obj = self._elf()
        obj.applied_relocs = [{"offset": 0x100}, {"offset": 0x200}]
        obj.runtime_addr = lambda addr: addr + 0x1000
        self.assertEqual(obj.relocated_addresses(), {0x1100, 0x1200})

    def test_relocated_addresses_skips_bad_records(self):
        obj = self._elf()
        obj.applied_relocs = [{}, {"offset": "bad"}, {"offset": 1}]
        obj.runtime_addr = lambda addr: addr + 1
        self.assertEqual(obj.relocated_addresses(), {2})
        obj.applied_relocs = None
        self.assertEqual(obj.relocated_addresses(), set())

    def test_initializers_are_not_run_for_a_library(self):
        obj = self._elf()
        obj.initializers = MagicMock()
        with patch.object(self.exec_mod, "is_shared_library", return_value=True):
            self.assertEqual(obj.run_initializers(), [])
        obj.initializers.assert_not_called()

    def test_initializers_run_once_and_cache(self):
        obj = self._elf()
        obj.initializers = MagicMock(return_value=[0x1000, 0x2000])
        obj.runtime_addr = lambda addr: addr + 1
        with (
            patch.object(self.exec_mod, "is_shared_library", return_value=False),
            patch.object(self.exec_mod, "call_native") as call,
        ):
            self.assertEqual(obj.run_initializers(), [0x1001, 0x2001])
            self.assertEqual(obj.run_initializers(), [0x1001, 0x2001])
        self.assertEqual(call.call_count, 2)

    def test_initializer_failure_is_reported_and_not_cached(self):
        import io
        from contextlib import redirect_stderr

        obj = self._elf()
        obj.initializers = MagicMock(return_value=[0x1000])
        obj.runtime_addr = lambda addr: addr
        err = io.StringIO()
        with (
            patch.object(self.exec_mod, "is_shared_library", return_value=False),
            patch.object(
                self.exec_mod, "call_native", side_effect=RuntimeError("boom")
            ),
            redirect_stderr(err),
        ):
            self.assertEqual(obj.run_initializers(), [])
        self.assertIn("boom", err.getvalue())
        self.assertEqual(obj.ran_initializers, [])

    def test_live_overlap_detects_intersections(self):
        with patch.object(
            self.exec_mod, "_live_ranges", return_value=[(0x1000, 0x3000)]
        ):
            self.assertEqual(
                self.exec_mod._live_overlap(0x2000, 0x100), (0x1000, 0x3000)
            )
            self.assertEqual(
                self.exec_mod._live_overlap(0x2F00, 0x200), (0x1000, 0x3000)
            )
            self.assertIsNone(self.exec_mod._live_overlap(0x3000, 0x100))
            self.assertIsNone(self.exec_mod._live_overlap(0x0, 0x1000))


class TestProcessStackLayout(unittest.TestCase):
    def _elf(self, stack_top=0x700000, stack_base=0x600000):
        from perf.arch import x86_64
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.arch = x86_64
        obj.stack_top = stack_top
        obj.stack_base = stack_base
        obj.memory = {}
        obj.auxv = lambda: [(3, 0x5000), (0, 0)]

        def write(addr, data):
            if isinstance(data, str):
                data = data.encode()
            for i, byte in enumerate(data):
                obj.memory[int(addr) + i] = byte

        obj.write_memory = write
        return obj

    def _read(self, obj, addr, size):
        return bytes(obj.memory[addr + i] for i in range(size))

    def test_argc_and_vectors_are_written(self):
        obj = self._elf()
        frame = obj.setup_argv(["prog", "one"], ["A=1"])
        self.assertEqual(frame["argc"], 2)
        self.assertEqual(frame["sp"] % 16, 8)
        self.assertEqual(frame["argv"], frame["sp"] + 8)
        self.assertEqual(frame["envp"], frame["sp"] + 32)
        self.assertEqual(frame["auxv"], frame["sp"] + 48)
        self.assertEqual(self._read(obj, frame["sp"], 8), (2).to_bytes(8, "little"))
        self.assertEqual(
            self._read(obj, frame["argv"], 8), (frame["strings"]).to_bytes(8, "little")
        )
        self.assertEqual(
            self._read(obj, frame["envp"], 8),
            (frame["strings"] + 9).to_bytes(8, "little"),
        )
        self.assertEqual(self._read(obj, frame["strings"], 5), b"prog\0")
        self.assertEqual(self._read(obj, frame["strings"] + 5, 4), b"one\0")
        self.assertEqual(self._read(obj, frame["strings"] + 9, 4), b"A=1\0")

    def test_auxv_is_terminated_and_save_holds_stack_top(self):
        import struct

        obj = self._elf()
        frame = obj.setup_argv(["prog"])
        self.assertEqual(
            struct.unpack("<QQ", self._read(obj, frame["auxv"] + 16, 16)), (0, 0)
        )
        self.assertEqual(
            struct.unpack("<Q", self._read(obj, frame["save"], 8))[0], obj.stack_top
        )

    def test_empty_argv_is_an_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._elf().setup_argv([])
        self.assertIn("argv[0]", str(ctx.exception))

    def test_stack_too_small_is_an_error(self):
        obj = self._elf(stack_top=0x700000, stack_base=0x700000)
        with self.assertRaises(ValueError) as ctx:
            obj.setup_argv(["prog"])
        self.assertIn("raise --config.stack.size", str(ctx.exception))

    def test_a_later_call_replaces_the_image(self):
        obj = self._elf()
        first = obj.setup_argv(["a"])
        second = obj.setup_argv(["b", "c"])
        self.assertNotEqual(first["argc"], second["argc"])
        self.assertEqual(self._read(obj, second["strings"], 2), b"b\0")
        self.assertEqual(self._read(obj, second["strings"] + 2, 2), b"c\0")


class TestLayoutShuffleValidation(unittest.TestCase):
    def _elf(self):
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.runtime_base = 0x1000
        obj.is_pie = False
        obj.loader = SimpleNamespace(main_object=SimpleNamespace(mapped_base=0))
        obj.moved_functions = {"old": (1, 2)}
        obj.layout_names = {"old": 1}
        obj._text_range = lambda: None
        obj.segment_runtime_ranges = lambda: []
        return obj

    def test_map_is_required(self):
        from perf.exec import Elf

        with self.assertRaises(RuntimeError):
            Elf.__new__(Elf).randomize_layout({})

    def test_bad_align_is_rejected(self):
        obj = self._elf()
        for bad in (0, 3, 24, "many"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                obj.randomize_layout({}, align=bad)

    def test_unknown_layout_keys_are_rejected(self):
        obj = self._elf()
        with self.assertRaises(ValueError):
            obj.randomize_layout({}, alignment=16)
        with self.assertRaises(TypeError):
            obj.randomize_layout({}, nope=1)

    def test_no_functions_clears_the_previous_layout(self):
        obj = self._elf()
        self.assertEqual(obj.randomize_layout({}), {})
        self.assertEqual(obj.moved_functions, {})
        self.assertEqual(obj.layout_names, {})

    def _shuffle(self, seed):
        obj = self._elf()
        obj.mapped_regions = []
        obj.arch = SimpleNamespace(
            disassembler=lambda: SimpleNamespace(detail=True),
            layout_hazard=lambda *args, **kwargs: False,
            _FILL_BYTE=0x90,
            _BRANCH_REACH=1 << 31,
        )
        obj.segment_runtime_ranges = lambda: [(0x1000, 0x2000, 0x1000, "X")]
        obj.runtime_addr = lambda addr: addr
        obj.read_memory = lambda addr, size: b"\x90" * size
        obj.write_memory = lambda addr, data: None
        obj.mmap = lambda *args, **kwargs: 0x900000
        obj._relocate_blob = lambda code, fs, new_base, reloc_map, md, orig: code
        funcs = {f"f{i}": (0x1000 + i * 0x10, 0x1008 + i * 0x10) for i in range(6)}
        with patch("perf.exec.ctypes.memset"):
            obj.randomize_layout(funcs, seed=seed)
        return obj.layout_names

    def test_shuffle_is_deterministic_per_seed(self):
        first = self._shuffle(11)
        self.assertEqual(first, self._shuffle(11))
        self.assertNotEqual(first, self._shuffle(12))


class TestExecLinksThreadSafety(unittest.TestCase):
    def test_exec_links_bounded_and_thread_safe(self):
        import perf.exec as exec_mod

        self.assertTrue(hasattr(exec_mod, "_EXEC_LINKS_MAX"))
        errors = []

        def worker(n):
            try:
                exec_mod._exec_links_store(f"k{n}", n, f"v{n}")
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        with exec_mod._EXEC_LOCK:
            exec_mod._EXEC_LINKS.clear()


class TestElfLifecycleUnit(unittest.TestCase):
    def _elf(self):
        import ctypes

        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.mapped_regions = [
            (0x100000, 0x2000),
            (ctypes.c_void_p(0x300000), ctypes.c_size_t(0x1000)),
            ("not an address", "not a size"),
        ]
        obj.runtime_base = 0x100000
        obj.stack_base = 0x700000
        obj.stack_top = 0x700100
        obj.moved_functions = {"f": (0x1000, 0x2000)}
        obj.layout_names = {"f": 0x9000}
        return obj

    def test_close_unmaps_every_valid_region_and_resets_state(self):
        from perf.exec import Elf

        obj = self._elf()
        with patch.object(Elf, "munmap") as munmap:
            obj.close()
        self.assertEqual(munmap.call_count, 2)
        first = munmap.call_args_list[0].args
        self.assertEqual(first[0].value, 0x100000)
        self.assertEqual(first[1].value, 0x2000)
        self.assertEqual(obj.mapped_regions, [])
        self.assertEqual(obj.runtime_base, 0)
        self.assertEqual(obj.stack_base, 0)
        self.assertEqual(obj.stack_top, 0)
        self.assertEqual(obj.moved_functions, {})
        self.assertEqual(obj.layout_names, {})

    def test_close_ignores_broken_region_entries(self):
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.mapped_regions = [(None, None), (0, 0)]
        with patch.object(Elf, "munmap") as munmap:
            obj.close()
        munmap.assert_not_called()
        self.assertEqual(obj.mapped_regions, [])


class TestElfContextManagerUnit(unittest.TestCase):
    def test_context_manager_and_destructor_close(self):
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.close = MagicMock()
        self.assertFalse(obj.__exit__(None, None, None))
        obj.__del__()
        self.assertEqual(obj.close.call_count, 2)


class TestSetupStackUnit(unittest.TestCase):
    def _elf(self):
        from perf.arch import x86_64
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.arch = x86_64
        obj.mapped_regions = []
        obj.mmap = Elf.mmap
        return obj

    def test_valid_sizes_and_alignments_are_mapped(self):
        obj = self._elf()
        self.addCleanup(obj.close)
        top = obj.setup_stack(size="0x4000", align="16")
        self.assertEqual(top % 16, 0)
        self.assertEqual(len(obj.mapped_regions), 1)
        self.assertGreater(obj.stack_top, obj.stack_base)

    def test_bad_sizes_are_rejected(self):
        obj = self._elf()
        for size in (0, -1, "nope"):
            with self.subTest(size=size), self.assertRaises(ValueError):
                obj.setup_stack(size=size)

    def test_bad_alignments_are_rejected(self):
        obj = self._elf()
        for align in (0, 3, "nope"):
            with self.subTest(align=align), self.assertRaises(ValueError):
                obj.setup_stack(align=align)


class TestElfAuxvUnit(unittest.TestCase):
    def _elf(self, binary="/proc/self/exe"):
        from perf.exec import Elf

        main = SimpleNamespace(binary=binary, segments=[], mapped_base=0)
        obj = Elf.__new__(Elf)
        obj.loader = SimpleNamespace(main_object=main)
        obj.is_pie = False
        obj.runtime_addr = lambda value: int(value)
        return obj

    def test_auxv_contains_the_process_facts(self):
        import mmap

        from perf.exec import _AT_ENTRY, _AT_NULL, _AT_PAGESZ, _AT_PHDR

        entries = dict(self._elf().auxv())
        self.assertEqual(entries[_AT_PAGESZ], mmap.PAGESIZE)
        self.assertIn(_AT_PHDR, entries)
        self.assertIn(_AT_ENTRY, entries)
        self.assertEqual(entries[_AT_NULL], 0)

    def test_unreadable_binary_has_no_auxv(self):
        self.assertEqual(self._elf("/nonexistent/perf").auxv(), [])


class TestElfSegmentRangesUnit(unittest.TestCase):
    def _elf(self):
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.runtime_base = 0x400000
        obj.start = 0x1000
        obj.is_pie = False
        obj.loader = SimpleNamespace(
            main_object=SimpleNamespace(
                mapped_base=0,
                segments=[
                    SimpleNamespace(
                        type="PT_LOAD",
                        vaddr=0x1000,
                        memsize=0x2000,
                        flags=7,
                    ),
                    SimpleNamespace(
                        type="PT_LOAD",
                        vaddr=0x3000,
                        memsize=0x10,
                        header=SimpleNamespace(p_flags=6),
                    ),
                    SimpleNamespace(
                        type="PT_NOTE",
                        vaddr=0x5000,
                        memsize=0x10,
                        flags=4,
                    ),
                    SimpleNamespace(
                        vaddr=0x4000,
                        memsize=0x100,
                        header=SimpleNamespace(p_flags=1),
                    ),
                ],
            )
        )
        return obj

    def test_loadable_segments_report_runtime_addresses_and_flags(self):
        ranges = self._elf().segment_runtime_ranges()
        self.assertEqual(ranges[0], (0x1000, 0x2000, 0x400000, "RWX"))
        self.assertEqual(ranges[1], (0x3000, 0x10, 0x402000, "RW"))
        self.assertEqual(ranges[2], (0x4000, 0x100, 0x403000, "X"))
        self.assertEqual(len(ranges), 3)

    def test_a_missing_runtime_base_is_an_error(self):
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.runtime_base = 0
        with self.assertRaises(RuntimeError):
            obj.segment_runtime_ranges()


class TestElfTextRangeUnit(unittest.TestCase):
    def test_text_section_bounds_are_returned(self):
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.loader = SimpleNamespace(
            main_object=SimpleNamespace(binary="/proc/self/exe")
        )
        start, size = obj._text_range()
        self.assertGreaterEqual(start, 0)
        self.assertGreater(size, 0)

    def test_unreadable_or_non_elf_files_have_no_text_range(self):
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.loader = SimpleNamespace(
            main_object=SimpleNamespace(binary="/nonexistent/perf")
        )
        self.assertIsNone(obj._text_range())
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "note.txt")
        with open(path, "w") as fh:
            fh.write("hello")
        obj.loader.main_object.binary = path
        self.assertIsNone(obj._text_range())


class TestPerfMapEntriesUnit(unittest.TestCase):
    def _elf(self):
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.loader = SimpleNamespace(
            main_object=SimpleNamespace(binary="/proc/self/exe")
        )
        obj.segment_runtime_ranges = lambda: [
            (0x1000, 0x10, 0x400000, "RX"),
            (0x2000, 0, 0x500000, "RW"),
        ]
        obj.ifuncs = {}
        obj.moved_functions = {}
        obj.layout_names = {}
        obj.is_pie = False
        obj.runtime_base = 0
        return obj

    def test_segments_and_prefixes_are_written(self):
        obj = self._elf()
        entries = obj.perf_map_entries(include_symbols=False, prefix="p:")
        self.assertEqual(
            entries,
            [(0x400000, 0x10, "p:perf_exec:exe:seg0")],
        )

    def test_symbols_are_skipped_when_unresolvable(self):
        obj = self._elf()
        with patch.object(obj, "get_symbol", side_effect=ValueError("missing")):
            entries = obj.perf_map_entries(include_symbols=True)
        self.assertEqual(len(entries), 1)

    def test_relocated_functions_are_appended(self):
        obj = self._elf()
        obj.moved_functions = {0x1000: (0x1100, 0x900000)}
        obj.layout_names = {"hot": 0x900000, "cold": 0x910000}
        entries = obj.perf_map_entries(include_symbols=False)
        self.assertIn((0x900000, 0x100, "hot"), entries)
        self.assertNotIn((0x910000, 0x100, "cold"), entries)

    def test_an_unnamed_binary_still_reports_segments(self):
        from perf.exec import Elf

        class Loader:
            @property
            def main_object(self):
                raise OSError("unreadable")

        obj = Elf.__new__(Elf)
        obj.loader = Loader()
        obj.segment_runtime_ranges = lambda: [(0x1000, 0x10, 0x400000, "RX")]
        entries = obj.perf_map_entries(include_symbols=False)
        self.assertEqual(entries, [(0x400000, 0x10, "perf_exec:exec:seg0")])


class TestPerfMapWriteEdges(unittest.TestCase):
    def _path(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        return os.path.join(d, "perf.map")

    def test_entries_are_normalized_and_invalid_ones_skipped(self):
        path = self._path()
        write_perf_map(
            [
                ("0x1000", "0x20", "a\nb"),
                (0x2000, 0x10, "   "),
                ("not an address", 1, "x"),
                (1, "not a size", "y"),
            ],
            path=path,
            append=False,
        )
        with open(path) as fh:
            lines = fh.read().splitlines()
        self.assertEqual(lines, ["1000 20 a_b", "2000 10 jit"])

    def test_the_default_path_is_the_perf_map_path(self):
        import perf.exec as exec_mod

        path = self._path()
        with patch.object(exec_mod, "perf_map_path", return_value=path) as name:
            self.assertEqual(write_perf_map([(1, 2, "x")]), path)
        name.assert_called_once()


class TestOutputSinkEdges(unittest.TestCase):
    def test_redirection_hides_writes_and_restores_stdout(self):
        import perf.exec as exec_mod

        saved = os.dup(1)
        with tempfile.TemporaryFile() as capture:
            try:
                os.dup2(capture.fileno(), 1)
                with exec_mod._output_sink([1, 3]):
                    os.write(1, b"hidden\n")
                os.write(1, b"visible\n")
            finally:
                os.dup2(saved, 1)
                os.close(saved)
            capture.seek(0)
            data = capture.read()
        self.assertIn(b"visible", data)
        self.assertNotIn(b"hidden", data)

    def test_a_failed_save_leaves_the_stream_alone(self):
        import perf.exec as exec_mod

        with patch.object(exec_mod.os, "dup", side_effect=OSError("no fd")):
            with exec_mod._output_sink([1]):
                pass


class TestCompilerDetectionUnit(unittest.TestCase):
    def test_environment_compilers_are_split_and_deduplicated(self):
        import perf.exec as exec_mod

        def which(name):
            return f"/usr/bin/{name}" if name in ("cc", "gcc") else None

        with (
            patch.dict(os.environ, {"CXX": "cc -O2", "CC": "gcc"}),
            patch.object(exec_mod.shutil, "which", side_effect=which),
        ):
            found = exec_mod._candidate_compilers()
        self.assertIn(["cc", "-O2"], found)
        self.assertIn(["gcc"], found)
        self.assertEqual(found.count(["gcc"]), 1)

    def test_malformed_compiler_commands_fall_back_to_one_word(self):
        import perf.exec as exec_mod

        def which(name):
            return "/usr/bin/cc" if name in ("cc", '"unclosed') else None

        with (
            patch.dict(os.environ, {"CXX": '"unclosed', "CC": ""}),
            patch.object(exec_mod.shutil, "which", side_effect=which),
        ):
            found = exec_mod._candidate_compilers()
        self.assertEqual(found, [['"unclosed']])


class TestObjectHelpersUnit(unittest.TestCase):
    def test_sym_id_is_a_safe_identifier(self):
        from perf.exec import _sym_id

        self.assertEqual(_sym_id("2 bad-label!"), "f_2_bad_label")
        self.assertEqual(_sym_id("..."), "target")
        self.assertEqual(_sym_id("main"), "main")

    def test_object_runs_deduplicates_spans(self):
        from perf.exec import _object_runs

        funcs = {
            "b": (0x20, 0x30),
            "a": (0x10, 0x20),
            "dup": (0x10, 0x20),
            "bad": (1,),
        }
        self.assertEqual(
            _object_runs(None, None, funcs),
            [("a", 0x10, 0x20), ("b", 0x20, 0x30)],
        )
        self.assertEqual(_object_runs(None, "", funcs), _object_runs(None, None, funcs))

    def test_object_runs_resolves_a_region(self):
        from perf.exec import _object_runs

        with patch("perf.info.targets", return_value=[("r", 0x10, 0x20)]) as targets:
            self.assertEqual(_object_runs(None, ("a", "b"), {}), [("r", 0x10, 0x20)])
        targets.assert_called_once()
        self.assertEqual(targets.call_args.args[1], "a..b")

    def test_object_harness_falls_back_to_the_runtime_offset(self):
        from perf.arch import x86_64
        from perf.exec import _object_harness

        class Elf:
            def get_symbol(self, name):
                raise ValueError("missing")

            def raw_symbol(self, name):
                return None

        name, blob, raw = _object_harness(Elf(), x86_64, "foo", 0x1000, 0x500)
        self.assertEqual(name, "perf_bench_foo")
        self.assertTrue(blob)
        self.assertIsNone(raw)
        self.assertIsNone(_object_harness(Elf(), x86_64, "foo", 0x1000, None))

    def test_object_harness_keeps_the_raw_symbol(self):
        from perf.arch import x86_64
        from perf.exec import _object_harness

        class Elf:
            def get_symbol(self, name):
                return 0x2000

            def raw_symbol(self, name):
                return "_Z3foov"

        name, _blob, raw = _object_harness(Elf(), x86_64, "foo", 0x1000, 0)
        self.assertEqual(name, "perf_bench_foo")
        self.assertEqual(raw, "_Z3foov")
        _name, _blob, raw = _object_harness(Elf(), x86_64, "foo..bar", 0x1000, 0)
        self.assertIsNone(raw)

    def test_object_harness_reports_assembler_failures(self):
        from perf.exec import _object_harness

        class Elf:
            def get_symbol(self, name):
                return 0x2000

            def raw_symbol(self, name):
                return None

        class Arch:
            def call_seq_asm(self, target):
                return "call rax"

            def assemble(self, text):
                raise ValueError("bad assembly")

        self.assertIsNone(_object_harness(Elf(), Arch(), "foo", 0x1000, 0))


class TestElfSymbolHelpersUnit(unittest.TestCase):
    def _binary_and_symbol(self):
        import sys

        from perf.exec import _symtab_values

        values, _ = _symtab_values(sys.executable)
        name = next(
            (sym for sym, value in values.items() if value and not sym.startswith("_")),
            None,
        )
        if name is None:
            self.skipTest("no suitable symbol in the running interpreter")
        return sys.executable, name, values[name]

    def test_ifunc_override_short_circuits_the_symbol_table(self):
        from perf.exec import Elf

        binary, _name, _value = self._binary_and_symbol()
        obj = Elf.__new__(Elf)
        obj.loader = SimpleNamespace(main_object=SimpleNamespace(binary=binary))
        obj.ifuncs = {"strlen": 0x1234}
        self.assertEqual(obj.get_symbol("strlen"), 0x1234)

    def test_symbol_values_follow_pie_and_moved_ranges(self):
        from perf.exec import Elf

        binary, name, value = self._binary_and_symbol()
        obj = Elf.__new__(Elf)
        obj.loader = SimpleNamespace(main_object=SimpleNamespace(binary=binary))
        obj.ifuncs = {}
        obj.moved_functions = {}
        obj.is_pie = False
        obj.runtime_base = 0
        self.assertEqual(obj.get_symbol(name), value)
        obj.is_pie = True
        obj.runtime_base = 0x400000
        self.assertEqual(obj.get_symbol(name), value + 0x400000)
        obj.is_pie = False
        obj.moved_functions = {value: (value + 0x20, 0x900000)}
        self.assertEqual(obj.get_symbol(name), 0x900000)
        with self.assertRaises(ValueError):
            obj.get_symbol("__perf_missing_symbol__")

    def test_ifunc_names_are_indexed_by_address(self):
        from perf.exec import Elf, _ifunc_symbols

        libc = next(
            (
                path
                for path in (
                    "/lib/x86_64-linux-gnu/libc.so.6",
                    "/lib64/libc.so.6",
                )
                if os.path.exists(path)
            ),
            None,
        )
        if libc is None:
            self.skipTest("no libc.so.6")
        obj = Elf.__new__(Elf)
        obj.loader = SimpleNamespace(main_object=SimpleNamespace(binary=libc))
        names = obj.ifunc_names()
        self.assertTrue(names)
        self.assertTrue(all(isinstance(addr, int) for addr in names))
        self.assertIs(obj.ifunc_names(), names)
        expected = {}
        for name, value, _size in _ifunc_symbols(libc):
            expected.setdefault(int(value), name)
        for addr, name in list(expected.items())[:3]:
            self.assertEqual(names[addr], name)

    def test_file_vaddr_and_offset_translation(self):
        from perf.exec import Elf

        main = SimpleNamespace(
            mapped_base=0x400000,
            segments=[SimpleNamespace(offset=0x100, vaddr=0x401000, filesize=0x100)],
        )
        obj = Elf.__new__(Elf)
        obj.loader = SimpleNamespace(main_object=main)
        obj.is_pie = True
        self.assertEqual(obj._file_vaddr(0x401000), 0x1000)
        self.assertEqual(obj._offset_to_vaddr(0x150), 0x1050)
        self.assertEqual(obj._offset_to_vaddr(0x500), obj._file_vaddr(0x500))
        obj.is_pie = False
        self.assertEqual(obj._file_vaddr(0x401000), 0x401000)

    def test_ifunc_symbols_of_a_missing_file_are_empty(self):
        from perf.exec import _ifunc_symbols, _short_symbol

        self.assertEqual(_ifunc_symbols("/nonexistent/perf-lib.so"), [])
        self.assertEqual(_short_symbol("ns::foo(int)"), "ns::foo")
        self.assertIsNone(_short_symbol(""))


class TestApplyCopyUnit(unittest.TestCase):
    def _elf(self):
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.arch = SimpleNamespace(_POINTER_SIZE=8)
        return obj

    def test_a_resolved_object_is_copied(self):
        import ctypes

        import perf.exec as exec_mod

        source = ctypes.create_string_buffer(b"ABCDEFGH")
        place = ctypes.create_string_buffer(b"\x00" * 8)
        sym = SimpleNamespace(name="obj", entry={"st_size": 4})
        with patch.object(
            exec_mod,
            "_resolve_external",
            return_value=ctypes.addressof(source),
        ):
            self._elf()._apply_copy(ctypes.addressof(place), sym)
        self.assertEqual(place.raw[:4], b"ABCD")

    def test_a_missing_object_is_zeroed_and_reported(self):
        import ctypes
        import io

        import perf.exec as exec_mod

        place = ctypes.create_string_buffer(b"\xff" * 8)
        sym = SimpleNamespace(name="missing", entry={"st_size": 0})
        err = io.StringIO()
        with (
            patch.object(exec_mod, "_resolve_external", return_value=None),
            patch("sys.stderr", err),
        ):
            self._elf()._apply_copy(ctypes.addressof(place), sym)
        self.assertEqual(place.raw[:8], b"\x00" * 8)
        self.assertIn("no definition", err.getvalue())

    def test_a_failing_copy_is_zeroed_and_reported(self):
        import ctypes
        import io

        import perf.exec as exec_mod

        place = ctypes.create_string_buffer(b"\xff" * 8)
        sym = SimpleNamespace(name="obj", entry={"st_size": 8})
        err = io.StringIO()
        with (
            patch.object(exec_mod, "_resolve_external", return_value=0x1000),
            patch.object(
                exec_mod.ctypes,
                "memmove",
                side_effect=OSError("cannot copy"),
            ),
            patch("sys.stderr", err),
        ):
            self._elf()._apply_copy(ctypes.addressof(place), sym)
        self.assertEqual(place.raw[:8], b"\x00" * 8)
        self.assertIn("cannot copy", err.getvalue())


class TestDynamicLinkerHelpersUnit(unittest.TestCase):
    def _libc(self):
        import ctypes.util

        return ctypes.util.find_library("c")

    def test_dlopen_loads_a_library_and_reports_failures(self):
        import io

        import perf.exec as exec_mod

        libc = self._libc()
        if not libc:
            self.skipTest("no libc")
        self.assertIsNotNone(exec_mod._dlopen(libc))
        err = io.StringIO()
        with patch("sys.stderr", err):
            self.assertIsNone(exec_mod._dlopen("/nonexistent/perf-lib.so"))
        self.assertIn("cannot load", err.getvalue())

    def test_dlopen_quiet_caches_handles(self):
        import perf.exec as exec_mod

        libc = self._libc()
        if not libc:
            self.skipTest("no libc")
        handle = exec_mod._dlopen_quiet(libc)
        self.assertIsNotNone(handle)
        self.assertIs(exec_mod._dlopen_quiet(libc), handle)
        self.assertIs(exec_mod._SHARED_LIBS[libc], handle)
        self.assertIsNone(exec_mod._dlopen_quiet("/nonexistent/perf-lib.so"))

    def test_external_symbols_resolve_through_libc(self):
        import perf.exec as exec_mod

        self.assertIsNone(exec_mod._resolve_external(None))
        self.assertTrue(exec_mod._resolve_external("malloc"))
        self.assertIsNone(exec_mod._resolve_external("__perf_missing_symbol_xyz__"))
        self.assertIsNotNone(exec_mod._dlsym_entry())
        self.assertTrue(exec_mod._dlsym(exec_mod._main_lib(), "malloc"))
        self.assertIsNone(exec_mod._dlsym(None, "malloc"))


class TestExitHooksUnit(unittest.TestCase):
    def setUp(self):
        import perf.exec as exec_mod

        self.exec_mod = exec_mod
        exec_mod.retire_exit_hooks()
        self.addCleanup(exec_mod.retire_exit_hooks)

    def test_hooks_record_unique_entries_and_retire_clears_them(self):
        hooks = self.exec_mod.exit_hooks()
        self.assertEqual(set(hooks), {"atexit", "__cxa_atexit"})
        self.assertEqual(hooks["atexit"](0x401000), 0)
        self.assertEqual(hooks["__cxa_atexit"](0x401008, 1, 0), 0)
        self.assertEqual(hooks["atexit"](0x401000), 0)
        self.assertEqual(self.exec_mod.retire_exit_hooks(), 2)
        self.assertEqual(self.exec_mod.retire_exit_hooks(), 0)


class TestInitializersRealBinary(unittest.TestCase):
    def test_the_running_interpreter_has_initializers(self):
        import sys

        from perf.exec import initializers

        entries = initializers(sys.executable)
        self.assertEqual(entries, [value for value in entries if value])
        self.assertTrue(all(isinstance(value, int) for value in entries))

    def test_missing_files_raise_and_elf_initializers_swallow_it(self):
        from perf.exec import Elf, initializers

        with self.assertRaises(OSError):
            initializers("/nonexistent/perf-binary")
        obj = Elf.__new__(Elf)
        obj.loader = SimpleNamespace(
            main_object=SimpleNamespace(binary="/nonexistent/perf-binary")
        )
        self.assertEqual(obj.initializers(), [])


class TestLiveRangesParsingUnit(unittest.TestCase):
    def test_valid_lines_are_parsed_and_bad_lines_skipped(self):
        import io

        import perf.exec as exec_mod

        text = "1000-2000 r-xp 00000000\nnot-a-range\nzz-qq x\n3000-4000 rw-p\n"
        with patch("builtins.open", return_value=io.StringIO(text)):
            self.assertEqual(
                exec_mod._live_ranges(),
                [(0x1000, 0x2000), (0x3000, 0x4000)],
            )

    def test_unreadable_maps_yield_no_ranges(self):
        import perf.exec as exec_mod

        with patch("builtins.open", side_effect=OSError("no maps")):
            self.assertEqual(exec_mod._live_ranges(), [])


class TestCallStubUnit(unittest.TestCase):
    def test_the_stub_is_executable_and_maps_to_the_code_length(self):
        from perf.exec import _call_stub

        code, buffer = _call_stub(0x1234)
        try:
            self.assertTrue(code)
            self.assertEqual(len(buffer), len(code))
        finally:
            buffer.close()


class TestCleanupObjLinksUnit(unittest.TestCase):
    def test_temporary_link_directories_are_removed(self):
        import perf.exec as exec_mod

        d = tempfile.mkdtemp()
        Path(d, "driver.c").write_text("int main(void) { return 0; }\n")
        saved = list(exec_mod._OBJ_LINK_DEPS)
        self.addCleanup(exec_mod._OBJ_LINK_DEPS.__setitem__, slice(None), saved)
        exec_mod._OBJ_LINK_DEPS.append(d)
        exec_mod._cleanup_obj_links()
        self.assertFalse(os.path.exists(d))
        self.assertEqual(exec_mod._OBJ_LINK_DEPS, [])


class TestDumpRuntimeSectionsUnit(unittest.TestCase):
    def test_readable_sections_are_dumped(self):
        from perf.exec import Elf

        obj = Elf.__new__(Elf)
        obj.segment_runtime_ranges = lambda: [
            (0x1000, 4, 0x400000, "RX"),
            (0x2000, 0, 0x500000, ""),
            (0x3000, 4, 0x600000, "R"),
        ]

        def read(addr, size):
            if addr == 0x600000:
                raise OSError("unreadable")
            return b"ABCD"

        obj.read_memory = read
        sections = obj._dump_runtime_sections()
        self.assertEqual(len(sections), 1)
        self.assertEqual(
            sections[0],
            {
                "name": ".perf.exec0",
                "vaddr": 0x1000,
                "bytes": b"ABCD",
                "flags": "RX",
            },
        )
