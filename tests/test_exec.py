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
