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

import io
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from perf.info import _span, functions, perf_labels, targets


def _write_binary(case, blob):
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "wb") as f:
        f.write(blob)
    case.addCleanup(os.remove, path)
    return path


class TestLabels(unittest.TestCase):
    def _project(self, blob, label=".perf.label"):
        project = Mock()
        obj = project.loader.main_object
        obj.binary = "/nonexistent"
        obj.mapped_base = 0x400000
        obj.pic = True
        obj.sections_map = {label: Mock(offset=0, filesize=len(blob))}
        return project, label

    def test_no_section_returns_empty(self):
        project = Mock()
        project.loader.main_object.sections_map = {}
        self.assertEqual(perf_labels(project), [])

    def test_parses_labels(self):
        blob = (
            struct.pack("<Q", 0x1234)
            + b"main\0"
            + struct.pack("<Q", 0x5678)
            + b"helper\0"
        )
        project, label = self._project(blob)
        obj = project.loader.main_object
        obj.binary = _write_binary(self, blob)
        self.assertEqual(
            perf_labels(project),
            [("main", 0x400000 + 0x1234), ("helper", 0x400000 + 0x5678)],
        )

    def test_no_pie_labels_are_absolute(self):
        blob = (
            struct.pack("<Q", 0x401174)
            + b"hot_begin\0"
            + struct.pack("<Q", 0x401189)
            + b"hot_end\0"
        )
        project, _ = self._project(blob)
        obj = project.loader.main_object
        obj.pic = False
        obj.binary = _write_binary(self, blob)
        self.assertEqual(
            perf_labels(project),
            [("hot_begin", 0x401174), ("hot_end", 0x401189)],
        )

    def test_labels_are_sorted_by_address(self):
        blob = (
            struct.pack("<Q", 0x1135)
            + b"foo_begin\0"
            + struct.pack("<Q", 0x112E)
            + b"foo_end\0"
        )
        project, _ = self._project(blob)
        obj = project.loader.main_object
        obj.pic = False
        obj.binary = _write_binary(self, blob)
        self.assertEqual(
            perf_labels(project),
            [("foo_end", 0x112E), ("foo_begin", 0x1135)],
        )

    def test_parses_high_addresses(self):
        blob = (
            struct.pack("<Q", 0x100002000)
            + b"hot_begin\0"
            + struct.pack("<Q", 0x100003000)
            + b"hot_end\0"
        )
        project, _ = self._project(blob)
        obj = project.loader.main_object
        obj.binary = _write_binary(self, blob)
        self.assertEqual(
            perf_labels(project),
            [
                ("hot_begin", 0x400000 + 0x100002000),
                ("hot_end", 0x400000 + 0x100003000),
            ],
        )

    def test_no_pie_high_addresses(self):
        blob = struct.pack("<Q", 0x401174) + b"hot_begin\0"
        project, _ = self._project(blob)
        obj = project.loader.main_object
        obj.pic = False
        obj.binary = _write_binary(self, blob)
        self.assertEqual(perf_labels(project), [("hot_begin", 0x401174)])

    def test_truncated_address_raises(self):
        blob = b"\x34\x12"
        project, _ = self._project(blob)
        obj = project.loader.main_object
        obj.binary = _write_binary(self, blob)
        with self.assertRaises(ValueError):
            perf_labels(project)

    def test_missing_null_terminator_raises(self):
        blob = struct.pack("<Q", 0x1234) + b"hot_begin"
        project, _ = self._project(blob)
        obj = project.loader.main_object
        obj.binary = _write_binary(self, blob)
        with self.assertRaises(ValueError):
            perf_labels(project)


class TestFunctions(unittest.TestCase):
    def test_returns_function_ranges(self):
        project = Mock()
        cfg = project.analyses.CFGFast.return_value

        block_a = Mock(addr=0x1000, size=4)
        block_b = Mock(addr=0x1010, size=8)
        func = Mock()
        func.name = "foo"
        func.addr = 0x1000
        func.blocks = [block_a, block_b]
        func.prototype = None
        cfg.kb.functions.values.return_value = iter([func])

        funcs, protos = functions(project)
        self.assertEqual(funcs, {"foo": (0x1000, 0x1018)})
        self.assertEqual(protos, {})


class TestSymtabFallback(unittest.TestCase):
    def test_empty_symtab_falls_through_to_dynsym(self):
        import importlib

        info_mod = importlib.import_module("perf.info")

        class _Sym:
            def __init__(self, name, value, size):
                self.name = name
                self.entry = {
                    "st_info": {"type": "STT_FUNC"},
                    "st_shndx": "SHN_TEXT",
                    "st_value": value,
                    "st_size": size,
                }

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
                    return _Section([_Sym("exported", 0x1234, 8)])
                return None

        project = Mock()
        project.loader.main_object.binary = _write_binary(self, b"")
        project.loader.main_object.mapped_base = 0
        project.loader.main_object.pic = False

        with patch.object(info_mod, "ELFFile", return_value=_Elf()):
            self.assertEqual(
                list(info_mod._symtab_functions(project)),
                [(0x1234, 8, "exported")],
            )


class TestTargets(unittest.TestCase):
    @patch("perf.info.perf_labels", return_value=[])
    @patch("perf.info.functions", return_value=({"foo": (0x1122, 0x1133)}, {}))
    def test_regex_matches_function_names(self, mock_functions, mock_labels):
        self.assertEqual(list(targets(None, "foo")), [("foo", 0x1122, 0x1133)])

    @patch("perf.info.perf_labels", return_value=[])
    @patch("perf.info.functions", return_value=({"foo": (0x1122, 0x1133)}, {}))
    def test_partial_name_does_not_match(self, mock_functions, mock_labels):
        got = list(targets(None, "o"))
        self.assertEqual(got, [])

    @patch("perf.info.perf_labels", return_value=[])
    @patch("perf.info.functions", return_value=({}, {}))
    def test_no_match_is_empty(self, mock_functions, mock_labels):
        self.assertEqual(list(targets(None, "nope")), [])

    @patch(
        "perf.info.perf_labels",
        return_value=[("hot_begin", 0x1000), ("hot_end", 0x2000)],
    )
    def test_range_spec_uses_labels(self, mock_labels):
        self.assertEqual(
            list(targets(None, "hot_begin..hot_end")),
            [("hot_begin..hot_end", 0x1000, 0x2000)],
        )

    @patch("perf.info.perf_labels", return_value=[("hot_begin", 0x1000)])
    def test_range_spec_missing_end_is_empty(self, mock_labels):
        self.assertEqual(list(targets(None, "hot_begin..missing")), [])

    @patch(
        "perf.info.perf_labels",
        return_value=[("foo_end", 0x1000), ("foo_begin", 0x2000)],
    )
    def test_range_spec_normalizes_reordered_pair(self, mock_labels):
        self.assertEqual(
            list(targets(None, "foo_begin..foo_end")),
            [("foo_begin..foo_end", 0x1000, 0x2000)],
        )

    def test_range_spec_hex_addresses(self):
        self.assertEqual(
            list(targets(None, "0x1000..0x2000")),
            [("0x1000..0x2000", 0x1000, 0x2000)],
        )

    def test_range_spec_reversed_hex_addresses(self):
        self.assertEqual(
            list(targets(None, "0x2000..0x1000")),
            [("0x2000..0x1000", 0x1000, 0x2000)],
        )

    def test_range_spec_hex_and_decimal_with_spaces(self):
        self.assertEqual(
            list(targets(None, "  0x1000 .. 8192  ")),
            [("  0x1000 .. 8192  ", 0x1000, 8192)],
        )

    @patch("perf.info.perf_labels", return_value=[])
    def test_range_spec_func_endpoints(self, mock_labels):
        funcs = {"main": (0x1000, 0x1100), "foo": (0x2000, 0x2100)}
        self.assertEqual(
            list(targets(None, "main..foo", funcs=funcs)),
            [("main..foo", 0x1000, 0x2100)],
        )
        self.assertEqual(
            list(targets(None, "main..main", funcs=funcs)),
            [("main..main", 0x1000, 0x1100)],
        )

    @patch("perf.info.perf_labels", return_value=[])
    def test_range_spec_offsets_unsupported(self, mock_labels):
        funcs = {"main": (0x1000, 0x1100), "foo": (0x2000, 0x2100)}
        self.assertEqual(list(targets(None, "main+0x10..foo-0x10", funcs=funcs)), [])
        self.assertEqual(list(targets(None, "0x1000+0x10..0x2000-16", funcs=funcs)), [])

    @patch(
        "perf.info.perf_labels",
        return_value=[("hot_begin", 0x1000), ("hot_end", 0x2000)],
    )
    def test_range_spec_labels_kept_with_offsets_and_mixed_sides(self, mock_labels):
        funcs = {"foo": (0x3000, 0x3100)}
        self.assertEqual(
            list(targets(None, "hot_begin+0x10..hot_end-16", funcs=funcs)), []
        )
        self.assertEqual(
            list(targets(None, "hot_begin..0x3000", funcs=funcs)),
            [("hot_begin..0x3000", 0x1000, 0x3000)],
        )
        self.assertEqual(
            list(targets(None, "0x1000..foo", funcs=funcs)),
            [("0x1000..foo", 0x1000, 0x3100)],
        )
        self.assertEqual(
            list(targets(None, "hot_begin..foo", funcs=funcs)),
            [("hot_begin..foo", 0x1000, 0x3100)],
        )

    @patch("perf.info.perf_labels", return_value=[])
    def test_range_spec_partial_and_ambiguous(self, mock_labels):
        funcs = {"main": (0x1000, 0x1100)}
        self.assertEqual(list(targets(None, "mai..0x2000", funcs=funcs)), [])
        ambiguous = {"main": (0x1000, 0x1100), "maintain": (0x2000, 0x2100)}
        self.assertEqual(
            list(targets(None, "main..0x3000", funcs=ambiguous)),
            [("main..0x3000", 0x1000, 0x3000)],
        )
        self.assertEqual(list(targets(None, "mai..0x3000", funcs=ambiguous)), [])
        self.assertEqual(list(targets(None, "missing..0x3000", funcs=funcs)), [])

    @patch("perf.info.perf_labels", return_value=[])
    @patch("perf.info.functions", return_value=({"foo": (0x1000, 0x1005)}, {}))
    def test_precomputed_functions_avoid_cfg(self, mock_functions, mock_labels):
        targets(None, "foo", funcs={"foo": (0x1000, 0x1005)})
        mock_functions.assert_not_called()

    def test_variadic_function_name_is_not_a_region(self):
        funcs = {"foo(int, ...)": (0x1000, 0x1100)}
        self.assertEqual(
            list(targets(None, "foo(int, ...)", funcs=funcs)),
            [("foo(int, ...)", 0x1000, 0x1100)],
        )

    def test_span_normalizes_endpoints_before_adding_the_base(self):
        project = Mock()
        project.loader.main_object.mapped_base = 0x400000
        self.assertEqual(
            _span(project, "x", 0x4010B5, 0x3000), ("x", 0x4010B5, 0x403000)
        )
        self.assertEqual(
            _span(project, "x", 0x1000, 0x400000), ("x", 0x400000, 0x401000)
        )


class TestTargetsNamespace(unittest.TestCase):
    @patch("perf.info.perf_labels", return_value=[])
    def test_bare_name_matches_namespaced_symbol(self, mock_labels):
        funcs = {"ns::foo": (0x1000, 0x1010)}
        self.assertEqual(
            list(targets(None, "foo", funcs=funcs)),
            [("ns::foo", 0x1000, 0x1010)],
        )

    @patch("perf.info.perf_labels", return_value=[])
    def test_qualified_name_does_not_suffix_match(self, mock_labels):
        funcs = {"ns::foo": (0x1000, 0x1010)}
        self.assertEqual(list(targets(None, "other::foo", funcs=funcs)), [])

    @patch("perf.info.perf_labels", return_value=[])
    def test_ambiguous_suffix_is_empty(self, mock_labels):
        funcs = {"a::foo": (0x1000, 0x1010), "b::foo": (0x2000, 0x2010)}
        self.assertEqual(list(targets(None, "foo", funcs=funcs)), [])

    @patch("perf.info.perf_labels", return_value=[])
    def test_same_address_suffix_resolves(self, mock_labels):
        funcs = {"a::foo": (0x1000, 0x1010), "b::foo": (0x1000, 0x1010)}
        got = list(targets(None, "foo", funcs=funcs))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][1:], (0x1000, 0x1010))

    @patch("perf.info.perf_labels", return_value=[])
    def test_signature_suffix_matches(self, mock_labels):
        funcs = {"ns::foo(int)": (0x1000, 0x1010)}
        self.assertEqual(
            list(targets(None, "foo", funcs=funcs)),
            [("ns::foo(int)", 0x1000, 0x1010)],
        )


class TestTargetsFromLabelBlob(unittest.TestCase):
    @staticmethod
    def _blob(*addr_name):
        return b"".join(struct.pack("<Q", a) + n.encode() + b"\0" for a, n in addr_name)

    def _project(self, blob, pic=True):
        project = Mock()
        obj = project.loader.main_object
        obj.binary = "/nonexistent"
        obj.mapped_base = 0x400000
        obj.pic = pic
        obj.sections_map = {".perf.label": Mock(offset=0, filesize=len(blob))}
        return project

    def test_pie_range_addresses(self):
        blob = self._blob((0x1184, "hot_begin"), (0x1199, "hot_end"))
        project = self._project(blob, pic=True)
        project.loader.main_object.binary = _write_binary(self, blob)
        got = list(targets(project, "hot_begin..hot_end"))
        self.assertEqual(len(got), 1)
        start, end = got[0][1], got[0][2]
        self.assertEqual(start, 0x400000 + 0x1184)
        self.assertEqual(end, 0x400000 + 0x1199)
        self.assertLess(start, end)

    def test_no_pie_range_addresses(self):
        blob = self._blob((0x401174, "hot_begin"), (0x401189, "hot_end"))
        project = self._project(blob, pic=False)
        project.loader.main_object.binary = _write_binary(self, blob)
        got = list(targets(project, "hot_begin..hot_end"))
        self.assertEqual(len(got), 1)
        start, end = got[0][1], got[0][2]
        self.assertEqual(start, 0x401174)
        self.assertEqual(end, 0x401189)
        self.assertLess(start, end)

    def test_pie_large_range_span(self):
        blob = self._blob((0x1000, "func_begin"), (0x100000, "func_end"))
        project = self._project(blob, pic=True)
        project.loader.main_object.binary = _write_binary(self, blob)
        got = list(targets(project, "func_begin..func_end"))
        self.assertEqual(len(got), 1)
        start, end = got[0][1], got[0][2]
        self.assertEqual(end - start, 0x100000 - 0x1000)
        self.assertLess(start, end)

    def test_pie_single_byte_range(self):
        blob = self._blob((0x5000, "seq_begin"), (0x5001, "seq_end"))
        project = self._project(blob, pic=True)
        project.loader.main_object.binary = _write_binary(self, blob)
        got = list(targets(project, "seq_begin..seq_end"))
        self.assertEqual(len(got), 1)
        start, end = got[0][1], got[0][2]
        self.assertEqual(end - start, 1)
        self.assertLess(start, end)


class TestMetadata(unittest.TestCase):
    @patch("angr.Project")
    @patch("perf.info.perf_labels")
    @patch("perf.info.functions")
    def test_metadata_returns_all_kinds(
        self, mock_functions, mock_labels, mock_project
    ):
        import pandas as pd

        from perf.info import metadata

        mock_labels.return_value = [("main", 0x401000), ("helper", 0x402000)]
        mock_functions.return_value = (
            {"foo": (0x401000, 0x401017), "bar": (0x402000, 0x402010)},
            {},
        )

        df = metadata("a.out")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["kind", "begin", "end", "size", "name"])
        self.assertEqual(len(df), 4)
        kinds = df["kind"].tolist()
        self.assertEqual(kinds.count("label"), 2)
        self.assertEqual(kinds.count("func"), 2)
        self.assertNotIn("region", kinds)

    @patch("angr.Project")
    @patch("perf.info.perf_labels")
    @patch("perf.info.functions")
    def test_only_functions_have_size(self, mock_functions, mock_labels, mock_project):
        import pandas as pd

        from perf.info import metadata

        mock_labels.return_value = [("hot_begin", 0x401000)]
        mock_functions.return_value = ({"foo": (0x402000, 0x402010)}, {})

        df = metadata("a.out")

        sizes = dict(zip(df["name"], df["size"], strict=True))
        self.assertTrue(pd.isna(sizes["hot_begin"]))
        self.assertEqual(sizes["foo"], 16)

    @patch("angr.Project")
    @patch("perf.info.perf_labels", return_value=[])
    @patch("perf.info.functions", return_value=({}, {}))
    def test_metadata_empty_output(self, mock_functions, mock_labels, mock_project):
        import pandas as pd

        from perf.info import metadata

        df = metadata("a.out")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["kind", "begin", "end", "size", "name"])
        self.assertEqual(len(df), 0)

    @patch("angr.Project")
    @patch("perf.info.functions", return_value=({}, {}))
    def test_metadata_warns_on_reordered_pair(self, mock_functions, mock_project):
        from perf.info import metadata

        with patch(
            "perf.info.perf_labels",
            return_value=[("foo_begin", 0x401135), ("foo_end", 0x40112E)],
        ):
            buf = io.StringIO()
            with patch("sys.stderr", buf):
                metadata("a.out")
        err = buf.getvalue()
        self.assertIn("'foo_end' (0x40112e) precedes 'foo_begin' (0x401135)", err)

    @patch("angr.Project")
    @patch("perf.info.functions", return_value=({}, {}))
    def test_metadata_no_warning_for_ordered_pair(self, mock_functions, mock_project):
        from perf.info import metadata

        with patch(
            "perf.info.perf_labels",
            return_value=[("foo_begin", 0x40112E), ("foo_end", 0x401135)],
        ):
            buf = io.StringIO()
            with patch("sys.stderr", buf):
                metadata("a.out")
        self.assertEqual(buf.getvalue(), "")

    def test_list_exported(self):
        import perf

        self.assertTrue(callable(perf.metadata))
        self.assertTrue(callable(perf.cpuinfo))
        self.assertFalse(callable(perf.info))
        self.assertFalse(hasattr(perf, "cpu_hz"))
        self.assertFalse(hasattr(perf, "trace"))
        self.assertFalse(hasattr(perf, "asm"))


class TestCpuinfo(unittest.TestCase):
    def test_cpuinfo_keys(self):
        from perf.info import cpuinfo

        df = cpuinfo()
        self.assertTrue(len(df) >= 1)
        for key in (
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
        ):
            self.assertIn(key, df.columns)
            self.assertIn(key, df.iloc[0].to_dict())
        self.assertNotIn("hz", df.columns)

    def test_freq_is_a_single_numeric_column(self):
        from perf.info import _cpu_freq, cpuinfo

        df = cpuinfo(["freq"])
        self.assertEqual(list(df.columns), ["freq"])
        freq = df["freq"].iloc[0]
        if freq is not None:
            self.assertEqual(freq, _cpu_freq())
            self.assertGreater(freq, 0)

    def test_exposed_on_package(self):
        import perf

        self.assertTrue(callable(perf.cpuinfo))

    def test_cpuid_tsc_frequency_matches_calibration(self):
        from perf.arch import x86_64
        from perf.info import _tsc_calibrate

        hz = x86_64.cpuid_tsc_hz()
        calibrated = _tsc_calibrate()
        if hz is None or calibrated is None:
            self.skipTest("TSC frequency unavailable")
        self.assertAlmostEqual(hz, calibrated, delta=hz * 0.05)

    def test_frequency_falls_back_when_cpuid_is_unavailable(self):
        import importlib

        info_mod = importlib.import_module("perf.info")
        info_mod._cpu_freq.cache_clear()
        try:
            with (
                patch(
                    "perf.arch.x86_64.cpuid_tsc_hz",
                    return_value=None,
                ),
                patch.object(info_mod, "_tsc_calibrate", return_value=None),
                patch.object(info_mod, "_os_cpu_freq", return_value=2_500_000_000),
            ):
                self.assertEqual(info_mod._cpu_freq(), 2_500_000_000)
        finally:
            info_mod._cpu_freq.cache_clear()


class TestCpuinfoErrors(unittest.TestCase):
    def test_missing_cache_sizes_raise_runtime_error(self):
        import importlib

        info_mod = importlib.import_module("perf.info")

        with (
            patch.object(info_mod, "_online_cpus", return_value=[0]),
            patch.object(info_mod, "_sysfs_cache_sizes", return_value={}),
        ):
            with self.assertRaises(RuntimeError):
                info_mod.cpuinfo()

    def test_no_online_cpus_raise_runtime_error(self):
        import importlib

        info_mod = importlib.import_module("perf.info")

        with patch.object(info_mod, "_online_cpus", return_value=[]):
            with self.assertRaises(RuntimeError):
                info_mod.cpuinfo()


class TestCpuCoreId(unittest.TestCase):
    def setUp(self):
        import importlib

        info_mod = importlib.import_module("perf.info")
        for helper in (
            "_online_cpus",
            "_cpu_core_id",
            "_cpu_numa_node",
            "_sysfs_cache_sizes",
        ):
            try:
                getattr(info_mod, helper).cache_clear()
            except Exception:
                pass

    def test_siblings_list_fallback(self):
        import importlib

        info_mod = importlib.import_module("perf.info")

        def fake_read(path):
            if path.endswith("core_id"):
                return ""
            if path.endswith("core_siblings_list"):
                return "4-5"
            return ""

        with patch.object(info_mod, "_read_text", side_effect=fake_read):
            self.assertEqual(info_mod._cpu_core_id(4), 4)

    def test_core_id_preferred(self):
        import importlib

        info_mod = importlib.import_module("perf.info")

        def fake_read(path):
            if path.endswith("core_id"):
                return "7"
            return "0"

        with patch.object(info_mod, "_read_text", side_effect=fake_read):
            self.assertEqual(info_mod._cpu_core_id(3), 7)


class TestDemangledNames(unittest.TestCase):
    def test_plain(self):
        from perf.info import _demangled_names

        self.assertEqual(_demangled_names("foo"), {"foo"})

    def test_mangled(self):
        from perf.info import _demangled_names

        names = _demangled_names("_Z3fooi")
        self.assertIn("_Z3fooi", names)
        self.assertIn("foo(int)", names)


class TestMetadataNewlineSanitization(unittest.TestCase):
    @patch("angr.Project")
    @patch("perf.info.perf_labels")
    @patch("perf.info.functions")
    def test_names_have_no_newlines(self, mock_functions, mock_labels, mock_project):
        from perf.info import metadata

        mock_labels.return_value = [("bad\nlabel", 0x401000)]
        mock_functions.return_value = ({"bad\nfunc": (0x402000, 0x402010)}, {})
        df = metadata("a.out")
        for name in df["name"].tolist():
            self.assertNotIn("\n", str(name))
            self.assertNotIn("\r", str(name))


ASM_SOURCE = """
# a line comment
\t.text
\t.globl foo
\t.type foo, @function
foo:            /* the entry */
\tmov eax, 42
\tmov ebx, 1
\tret
\t.globl bar
bar:
\txor eax, eax
\tret
\t.long 1, 2, 3
.Llocal:
\tnop
loop:
\tadd eax, 1
\tcmp eax, 10
\tjne loop
\tret
"""


class TestAsmSource(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "f.s")
        with open(self.path, "w") as fh:
            fh.write(ASM_SOURCE)

    def test_is_asm_source(self):
        from perf.info import is_asm_source

        self.assertTrue(is_asm_source("a.s"))
        self.assertTrue(is_asm_source("a.asm"))
        self.assertTrue(is_asm_source("A.S"))
        self.assertFalse(is_asm_source("a.out"))
        self.assertFalse(is_asm_source("asm"))

    def test_labels_with_size_and_position(self):
        from perf.info import asm_labels

        text = Path(self.path).read_text()
        labels = asm_labels(self.path)
        self.assertEqual(
            [name for name, _pos, _size in labels], ["foo", "bar", ".Llocal", "loop"]
        )
        for name, position, size in labels:
            body = text[position : position + size]
            self.assertNotIn(":", body.splitlines()[0])
            self.assertTrue(body.strip())

    def test_body_of_a_label_ends_at_the_next_label(self):
        from perf.info import asm_labels

        text = Path(self.path).read_text()
        bodies = {
            name: text[position : position + size].strip()
            for name, position, size in asm_labels(self.path)
        }
        self.assertEqual(
            bodies["foo"],
            "/* the entry */\n\tmov eax, 42\n\tmov ebx, 1\n\tret\n\t.globl bar",
        )
        self.assertEqual(bodies["bar"], "xor eax, eax\n\tret\n\t.long 1, 2, 3")
        self.assertEqual(bodies[".Llocal"], "nop")

    def test_last_label_runs_to_the_end_of_the_file(self):
        from perf.info import asm_labels

        name, position, size = asm_labels(self.path)[-1]
        self.assertEqual(name, "loop")
        self.assertEqual(position + size, len(Path(self.path).read_text()))

    def test_missing_file(self):
        from perf.info import asm_labels

        with self.assertRaises(ValueError) as ctx:
            asm_labels(os.path.join(self.dir, "nope.s"))
        self.assertIn("does not exist", str(ctx.exception))

    def test_public_api(self):
        import perf

        self.assertEqual(
            [name for name, _pos, _size in perf.metadata.asm_labels(self.path)],
            ["foo", "bar", ".Llocal", "loop"],
        )
        self.assertTrue(callable(perf.metadata.perf_labels))
        self.assertTrue(callable(perf.metadata.asm_labels))
        self.assertTrue(callable(perf.metadata.functions))
        self.assertIn("metadata", perf.__all__)
        self.assertNotIn("asm_labels", perf.__all__)
        self.assertNotIn("perf_labels", perf.__all__)
        self.assertNotIn("functions", perf.__all__)
        self.assertFalse(hasattr(perf, "asm_labels"))
        self.assertFalse(hasattr(perf, "perf_labels"))
        self.assertFalse(hasattr(perf, "functions"))


class TestHexAddresses(unittest.TestCase):
    def test_formats_int_columns(self):
        import pandas as pd

        from perf.info import hex_addresses

        df = pd.DataFrame({"begin": [4096, 8192], "name": ["a", "b"]})
        out = hex_addresses(df)
        self.assertEqual(out["begin"].tolist(), ["0x1000", "0x2000"])
        self.assertEqual(out["name"].tolist(), ["a", "b"])

    def test_missing_column_skipped(self):
        import pandas as pd

        from perf.info import hex_addresses

        df = pd.DataFrame({"name": ["a"]})
        out = hex_addresses(df, columns=("begin",))
        self.assertEqual(out["name"].tolist(), ["a"])

    def test_does_not_mutate_input(self):
        import pandas as pd

        from perf.info import hex_addresses

        df = pd.DataFrame({"begin": [4096]})
        hex_addresses(df)
        self.assertEqual(df["begin"].tolist(), [4096])


class TestInfoHelpers(unittest.TestCase):
    def test_format_hz_units(self):
        from perf.info import format_hz

        self.assertEqual(format_hz(2700000000), "2.7Ghz")
        self.assertEqual(format_hz(1500000), "2Mhz")
        self.assertEqual(format_hz(500), "500Hz")
        self.assertIsNone(format_hz(0))
        self.assertIsNone(format_hz("bad"))

    def test_format_size(self):
        from perf.info import _format_size

        self.assertEqual(_format_size(2048), "2Kb")
        self.assertEqual(_format_size("4Kb"), "4Kb")
        self.assertIsNone(_format_size(0))
        self.assertIsNone(_format_size(None))

    def test_hostname(self):
        from perf.info import hostname

        self.assertTrue(hostname())

    def test_asm_read_missing(self):
        from perf.info import _asm_read

        with self.assertRaises(ValueError):
            _asm_read("/nonexistent-perf-test-file.s")


class TestFunctionsCacheConcurrency(unittest.TestCase):
    def test_functions_cache_releases_lock_during_compute(self):
        import perf.info as info

        info._FUNCTIONS_MEMO.clear()
        self.addCleanup(info._FUNCTIONS_MEMO.clear)
        held_during_compute = {}

        real_analyze = info._analyze_functions

        def fake_analyze(symtab, cfg):
            held_during_compute["held"] = info._FUNCTIONS_LOCK.acquire(blocking=False)
            if held_during_compute["held"]:
                info._FUNCTIONS_LOCK.release()
            return real_analyze(symtab, cfg)

        from unittest.mock import patch

        with (
            patch("angr.Project"),
            patch.object(
                info, "_analyze_functions", side_effect=fake_analyze
            ) as mock_analyze,
        ):
            mock_analyze.return_value = ({}, {})
            try:
                info._functions_cached(("fake",), ())
            except Exception:
                pass
        self.assertIn("held", held_during_compute)
        self.assertTrue(held_during_compute["held"])


class TestElfProject(unittest.TestCase):
    def test_unsupported_arch_is_a_clear_error(self):
        from perf.info import elf_project

        with patch("perf.info._arch_fn", side_effect=ValueError("bogus")):
            with self.assertRaisesRegex(ValueError, "no elf fallback"):
                elf_project("a.out")

    def test_elf_project_resolves_the_executable(self):
        from perf.info import ElfProject, elf_project

        with (
            patch("perf.info._arch_fn"),
            patch("perf.info.resolve_exec", return_value="/bin/true") as resolve,
            patch("perf.info._MainObject") as main,
        ):
            project = elf_project("prog")
        self.assertIsInstance(project, ElfProject)
        resolve.assert_called_once_with("prog")
        main.assert_called_once_with("/bin/true")

    def test_main_object_reads_elf_metadata(self):
        import importlib

        info_mod = importlib.import_module("perf.info")

        class Section:
            name = ".text"

            def __getitem__(self, key):
                return {"sh_offset": 0x1000, "sh_size": 0x200}[key]

        class Elf:
            header = {"e_type": "ET_DYN"}

            def iter_sections(self):
                return iter([Section()])

        with (
            patch.object(info_mod, "ELFFile", return_value=Elf()),
            tempfile.NamedTemporaryFile() as fh,
        ):
            obj = info_mod._MainObject(fh.name)
        self.assertTrue(obj.pic)
        self.assertEqual(obj.mapped_base, info_mod._ANGR_MAIN_BASE)
        self.assertEqual(obj.sections_map[".text"].offset, 0x1000)
        self.assertEqual(obj.sections_map[".text"].filesize, 0x200)

    def test_main_object_no_pie_uses_zero_base(self):
        import importlib

        info_mod = importlib.import_module("perf.info")
        elf = SimpleNamespace(
            header={"e_type": "ET_EXEC"},
            iter_sections=lambda: iter(()),
        )
        with (
            patch.object(info_mod, "ELFFile", return_value=elf),
            tempfile.NamedTemporaryFile() as fh,
        ):
            obj = info_mod._MainObject(fh.name)
        self.assertFalse(obj.pic)
        self.assertEqual(obj.mapped_base, 0)

    def test_pic_base(self):
        from perf.info import _pic_base

        with patch(
            "perf.info.elf_project",
            return_value=SimpleNamespace(
                loader=SimpleNamespace(
                    main_object=SimpleNamespace(pic=True, mapped_base=0x400000)
                )
            ),
        ):
            self.assertEqual(_pic_base("prog"), (True, 0x400000))
        with patch("perf.info.elf_project", side_effect=ValueError("bad")):
            self.assertEqual(_pic_base("prog"), (False, 0))

    def test_project_loads_without_libraries(self):
        import importlib

        info_mod = importlib.import_module("perf.info")
        project = object()
        with (
            patch("angr.Project", return_value=project) as load,
            patch("perf.info.resolve_exec", return_value="/bin/true"),
        ):
            self.assertIs(info_mod._project("prog"), project)
        load.assert_called_once_with(
            "/bin/true", auto_load_libs=False, load_debug_info=False
        )


class TestRelocations(unittest.TestCase):
    def setUp(self):
        import importlib

        self.info = importlib.import_module("perf.info")
        fd, self.binary = tempfile.mkstemp()
        os.close(fd)
        self.addCleanup(os.remove, self.binary)

        class RelocSection:
            name = ".rela.dyn"

            def __init__(self, relocs, rela=True):
                self._relocs = relocs
                self._rela = rela

            def is_RELA(self):
                return self._rela

            def iter_relocations(self):
                if isinstance(self._relocs, Exception):
                    raise self._relocs
                return iter(self._relocs)

        class Elf:
            def __init__(self, sections):
                self._sections = sections

            def iter_sections(self):
                return iter(self._sections)

        self.RelocSection = RelocSection
        self.Elf = Elf

    def test_rela_addends_filters_relocations(self):
        relocs = [
            {"r_info_type": 8, "r_offset": 0x1000, "r_addend": 1},
            {"r_info_type": 1, "r_info_sym": 0, "r_offset": 0x1008, "r_addend": 2},
            {"r_info_type": 1, "r_info_sym": 1, "r_offset": 0x1010, "r_addend": 3},
            {"r_info_type": 7, "r_offset": 0x1018, "r_addend": 4},
            {"r_info_type": "bad", "r_offset": 0x1020},
            {"r_info_type": 8, "r_offset": "bad"},
            {"r_info_type": 8, "r_offset": 0x2000, "r_addend": 5},
            {"r_info_type": 8, "r_offset": 0x1028, "r_addend": "bad"},
            {
                "r_info_type": 1,
                "r_info_sym": "bad",
                "r_offset": 0x1030,
                "r_addend": 6,
            },
        ]
        elf = self.Elf([self.RelocSection(relocs)])
        with (
            patch.object(self.info, "ELFFile", return_value=elf),
            patch.object(self.info, "RelocationSection", self.RelocSection),
        ):
            out = self.info._rela_addends(self.binary, 0x1000, 0x2000)
        self.assertEqual(out, {0x1000: 1, 0x1008: 2})

    def test_rela_addends_skips_unusable_sections(self):
        class BadName:
            @property
            def name(self):
                raise RuntimeError("bad")

            def is_RELA(self):
                return True

            def iter_relocations(self):
                return iter(())

        sections = [
            BadName(),
            SimpleNamespace(name=".rela.plt"),
            self.RelocSection([], rela=False),
            SimpleNamespace(name=".text"),
            self.RelocSection([], rela=True),
            self.RelocSection(RuntimeError("bad")),
        ]
        sections[1].is_RELA = lambda: True
        sections[1].iter_relocations = lambda: iter(())
        elf = self.Elf(sections)
        with (
            patch.object(self.info, "ELFFile", return_value=elf),
            patch.object(self.info, "RelocationSection", self.RelocSection),
        ):
            self.assertEqual(self.info._rela_addends(self.binary, 0, 1), {})

    def test_rela_addends_tolerates_bad_binaries(self):
        with patch.object(self.info, "ELFFile", side_effect=ValueError("not an elf")):
            self.assertEqual(self.info._rela_addends(self.binary, 0, 1), {})
        self.assertEqual(self.info._rela_addends("/nonexistent-perf-test", 0, 1), {})

    def test_label_section_range(self):
        header = {"sh_addr": 0x400000, "sh_offset": 0x10, "sh_size": 0x20}
        section = SimpleNamespace(header=header)
        elf = SimpleNamespace(get_section_by_name=lambda label: section)
        with patch.object(self.info, "ELFFile", return_value=elf):
            self.assertEqual(
                self.info._label_section_range(self.binary, ".perf.label"),
                (0x400000, 0x10, 0x20),
            )
        with patch.object(
            self.info,
            "ELFFile",
            return_value=SimpleNamespace(get_section_by_name=lambda label: None),
        ):
            self.assertIsNone(
                self.info._label_section_range(self.binary, ".perf.label")
            )
        with patch.object(self.info, "ELFFile", side_effect=ValueError("bad")):
            self.assertIsNone(
                self.info._label_section_range(self.binary, ".perf.label")
            )


class TestLabelRelocations(unittest.TestCase):
    def _project(self, blob):
        project = Mock()
        obj = project.loader.main_object
        obj.binary = _write_binary(self, blob)
        obj.mapped_base = 0
        obj.pic = False
        obj.sections_map = {".perf.label": Mock(offset=0, filesize=len(blob))}
        return project

    def test_relocated_label_addresses_use_their_addends(self):
        blob = struct.pack("<Q", 0x1234) + b"main\0"
        project = self._project(blob)
        with (
            patch("perf.info._label_section_range", return_value=(0x400000, 0, 9)),
            patch("perf.info._rela_addends", return_value={0x400000: 0x5000}),
        ):
            self.assertEqual(perf_labels(project), [("main", 0x5000)])

    def test_relocation_errors_fall_back_to_the_raw_address(self):
        blob = struct.pack("<Q", 0x1234) + b"main\0"
        project = self._project(blob)
        with patch("perf.info._label_section_range", side_effect=ValueError("bad")):
            self.assertEqual(perf_labels(project), [("main", 0x1234)])

        class BrokenAddends(dict):
            def get(self, key):
                raise RuntimeError("bad")

        with (
            patch("perf.info._label_section_range", return_value=(0x400000, 0, 9)),
            patch("perf.info._rela_addends", return_value=BrokenAddends()),
        ):
            self.assertEqual(perf_labels(project), [("main", 0x1234)])

    def test_rela_addends_errors_are_ignored(self):
        blob = struct.pack("<Q", 0x1234) + b"main\0"
        project = self._project(blob)
        with (
            patch("perf.info._label_section_range", return_value=(0x400000, 0, 9)),
            patch("perf.info._rela_addends", side_effect=ValueError("bad")),
        ):
            self.assertEqual(perf_labels(project), [("main", 0x1234)])


class TestFunctionTables(unittest.TestCase):
    def setUp(self):
        import importlib

        self.info = importlib.import_module("perf.info")
        self.info._FUNCTIONS_MEMO.clear()
        self.addCleanup(self.info._FUNCTIONS_MEMO.clear)

    @staticmethod
    def _project(binary="/bin/true"):
        project = Mock()
        project.loader.main_object.binary = binary
        return project

    def test_fast_mode_uses_the_symbol_table(self):
        with (
            patch.object(self.info, "_symtab_functions", return_value=[(1, 1, "f")]),
            patch.object(
                self.info,
                "_symtab_only_functions",
                return_value=({"f": (1, 2)}, {}),
            ) as fast,
        ):
            self.assertEqual(
                self.info.functions(self._project(), fast=True),
                ({"f": (1, 2)}, {}),
            )
        fast.assert_called_once()

    def test_large_binaries_use_the_symbol_table(self):
        project = self._project()
        with (
            patch.object(self.info.os.path, "getsize", return_value=2 * 1024 * 1024),
            patch.object(self.info, "_symtab_functions", return_value=[(1, 1, "f")]),
            patch.object(
                self.info,
                "_symtab_only_functions",
                return_value=({"f": (1, 2)}, {}),
            ) as fast,
        ):
            self.assertEqual(self.info.functions(project), ({"f": (1, 2)}, {}))
        fast.assert_called_once()

    def test_cache_key_and_memoized_path(self):
        project = self._project()
        stat = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4)
        with (
            patch.object(self.info.os.path, "getsize", return_value=10),
            patch.object(self.info.os, "stat", return_value=stat),
            patch.object(self.info, "_symtab_functions", return_value=[]),
            patch.object(
                self.info,
                "_functions_cached",
                return_value=({"f": (1, 2)}, {"f": "proto"}),
            ) as cached,
        ):
            self.assertEqual(
                self.info.functions(project), ({"f": (1, 2)}, {"f": "proto"})
            )
        cached.assert_called_once_with(("/bin/true", 1, 2, 3, 4), [])

    def test_functions_cached_hits_and_clears(self):
        self.info._FUNCTIONS_MEMO[("hit",)] = ({"x": (0, 1)}, {})
        self.assertEqual(self.info._functions_cached(("hit",), ()), ({"x": (0, 1)}, {}))
        cfg = Mock()
        with (
            patch("angr.Project") as load,
            patch.object(
                self.info,
                "_analyze_functions",
                return_value=({"f": (1, 2)}, {}),
            ),
        ):
            load.return_value.analyses.CFGFast.return_value = cfg
            self.assertEqual(
                self.info._functions_cached(("prog", 1, 2, 3, 4), ()),
                ({"f": (1, 2)}, {}),
            )
            self.assertEqual(
                self.info._functions_cached(("prog", 1, 2, 3, 4), ()),
                ({"f": (1, 2)}, {}),
            )
        self.assertEqual(load.call_count, 1)

    def test_functions_cache_is_bounded(self):
        self.info._FUNCTIONS_MEMO_MAX = 2
        self.addCleanup(setattr, self.info, "_FUNCTIONS_MEMO_MAX", 8)
        for index in range(2):
            self.info._FUNCTIONS_MEMO[(index,)] = ({}, {})
        with (
            patch("angr.Project"),
            patch.object(
                self.info, "_analyze_functions", return_value=({"f": (1, 2)}, {})
            ),
        ):
            self.info._functions_cached(("new",), ())
        self.assertEqual(set(self.info._FUNCTIONS_MEMO), {("new",)})

    def test_analyze_functions_merges_symbols_and_prototypes(self):
        class Block:
            def __init__(self, addr, size):
                self.addr = addr
                self.size = size

        class Func:
            def __init__(self, name, addr, blocks, prototype=None):
                self.name = name
                self.addr = addr
                self.blocks = blocks
                self.prototype = prototype

        good = Func(
            "good",
            0x1000,
            [Block(0x1000, 4), Block(0x1004, 4)],
            prototype="int good()",
        )
        broken = Func("broken", 0x2000, [])
        broken.blocks = Mock(side_effect=RuntimeError("bad"))
        cfg = SimpleNamespace(
            kb=SimpleNamespace(
                functions={
                    0x1000: good,
                    0x2000: broken,
                    0x3000: Func("", 0x3000, []),
                    0x4000: Func("Unresolvable_1", 0x4000, []),
                }
            )
        )
        symtab = [(0x1000, 8, "_Z4goodv"), (0x2000, 4, "raw")]
        found, prototypes = self.info._analyze_functions(symtab, cfg)
        self.assertIn("_Z4goodv", found)
        self.assertIn("good()", found)
        self.assertEqual(found["_Z4goodv"], (0x1000, 0x1008))
        self.assertEqual(found["raw"], (0x2000, 0x2004))
        self.assertNotIn("", found)
        self.assertNotIn("Unresolvable_1", found)
        self.assertEqual(prototypes, {"good": "int good()"})

    def test_symbol_table_function_filters(self):
        def sym(name, value, size, typ="STT_FUNC", shndx="SHN_TEXT"):
            return SimpleNamespace(
                name=name,
                entry={
                    "st_info": {"type": typ},
                    "st_shndx": shndx,
                    "st_value": value,
                    "st_size": size,
                },
            )

        symbols = [
            sym("ok", 0x1000, 8),
            sym("undef", 0x1000, 8, shndx="SHN_UNDEF"),
            sym("abs", 0x1000, 8, shndx="SHN_ABS"),
            sym("common", 0x1000, 8, shndx="SHN_COMMON"),
            sym("object", 0x1000, 8, typ="STT_OBJECT"),
            sym("zero", 0x1000, 0),
            sym("", 0x1000, 8),
            SimpleNamespace(name="bad", entry={}),
        ]
        section = SimpleNamespace(iter_symbols=lambda: iter(symbols))
        elf = SimpleNamespace(get_section_by_name=lambda name: section)
        project = self._project()
        project.loader.main_object.mapped_base = 0x400000
        project.loader.main_object.pic = True
        with patch.object(self.info, "ELFFile", return_value=elf):
            got = list(self.info._symtab_functions(project))
        self.assertEqual(got, [(0x401000, 8, "ok")])

    def test_symtab_only_functions_demangles_and_deduplicates(self):
        symtab = [
            (0x1000, 8, "_Z3fooi"),
            (0x1000, 8, "foo"),
            (0x2000, 4, "bar"),
        ]
        found, prototypes = self.info._symtab_only_functions(None, symtab)
        self.assertIn("_Z3fooi", found)
        self.assertIn("foo(int)", found)
        self.assertEqual(found["bar"], (0x2000, 0x2004))
        self.assertEqual(prototypes, {})
        self.assertTrue(self.info._is_function_type("STT_GNU_IFUNC"))
        self.assertFalse(self.info._is_function_type("STT_OBJECT"))


class TestTargetsRegions(unittest.TestCase):
    def test_range_falls_back_when_functions_fail(self):
        with (
            patch("perf.info.perf_labels", return_value=[]),
            patch("perf.info.functions", side_effect=RuntimeError("bad")),
        ):
            self.assertEqual(list(targets(None, "missing..0x2000")), [])

    def test_short_signature_matches(self):
        funcs = {"foo(int)": (0x1000, 0x1010)}
        self.assertEqual(
            list(targets(None, "foo", funcs=funcs)),
            [("foo(int)", 0x1000, 0x1010)],
        )

    def test_numeric_target_inside_a_function(self):
        funcs = {"f": (0x1000, 0x2000)}
        self.assertEqual(
            list(targets(None, "0x1500", funcs=funcs)),
            [("f", 0x1000, 0x2000)],
        )
        self.assertEqual(
            list(targets(None, "0x5000", funcs=funcs)),
            [("0x5000", 0x5000, 0x5001)],
        )

    def test_region_endpoint_resolution(self):
        from perf.info import _region_endpoint

        entries = {"f": (0x1000, 0x2000), "ns::foo(int)": (0x3000, 0x4000)}
        self.assertEqual(_region_endpoint("0x1234", {}, {}, True), 0x1234)
        self.assertEqual(_region_endpoint("label", {"label": 0x10}, {}, True), 0x10)
        self.assertEqual(_region_endpoint("f", {}, entries, True), 0x1000)
        self.assertEqual(_region_endpoint("f", {}, entries, False), 0x2000)
        self.assertEqual(_region_endpoint("ns::foo(int)", {}, entries, True), 0x3000)
        self.assertEqual(_region_endpoint("ns::foo(...)", {}, entries, False), 0x4000)
        self.assertEqual(_region_endpoint("ns::foo", {}, {}, False), None)
        self.assertEqual(_region_endpoint("", {}, {}, False), None)

    def test_region_endpoint_rejects_ambiguous_short_names(self):
        from perf.info import _region_endpoint

        entries = {"a::foo": (0x1000, 0x1100), "b::foo": (0x2000, 0x2100)}
        self.assertIsNone(_region_endpoint("foo", {}, entries, True))

    def test_metadata_skips_duplicate_function_ranges(self):
        from perf.info import metadata

        with (
            patch("perf.info._project"),
            patch("perf.info.perf_labels", return_value=[]),
            patch(
                "perf.info.functions",
                return_value=(
                    {"a": (0x401000, 0x401010), "b": (0x401000, 0x401010)},
                    {},
                ),
            ),
        ):
            df = metadata("a.out")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["name"], "a")

    def test_safe_labels_and_span_errors(self):
        from perf.info import _safe_labels, _span

        with patch("perf.info.perf_labels", side_effect=RuntimeError("bad")):
            self.assertEqual(_safe_labels(None), [])
        project = Mock()
        project.loader.main_object.mapped_base = Mock(side_effect=ValueError("bad"))
        self.assertEqual(_span(project, "x", 1, 2), ("x", 1, 2))

    def test_inverted_and_cloned_labels(self):
        from perf.info import _cloned_labels, _inverted_pairs

        entries = [
            ("foo_begin", 0x2000),
            ("foo_end", 0x1000),
            ("bar_begin", 0x1000),
            ("bar_end", 0x2000),
            ("dup", 0x3000),
            ("dup", 0x4000),
        ]
        self.assertEqual(_inverted_pairs(entries), [("foo", 0x2000, 0x1000)])
        self.assertEqual(_cloned_labels(entries), [("dup", [0x3000, 0x4000])])


class TestCpuInfoParsing(unittest.TestCase):
    def setUp(self):
        import importlib

        self.info = importlib.import_module("perf.info")
        for name in (
            "_cpuinfo",
            "_proc_cpuinfo",
            "hostname",
            "_os_cpu_freq",
            "_cpu_freq",
            "_online_cpus",
            "_cpu_core_id",
            "_cpu_numa_node",
            "_sysfs_cache_sizes",
        ):
            try:
                getattr(self.info, name).cache_clear()
            except AttributeError:
                pass

    def test_asm_read_reports_open_errors(self):
        with (
            patch("perf.info.os.path.isfile", return_value=True),
            patch("builtins.open", side_effect=OSError("denied")),
        ):
            with self.assertRaisesRegex(ValueError, "cannot be read"):
                self.info._asm_read("a.s")

    def test_hostname_error(self):
        with patch("socket.gethostname", side_effect=OSError("bad")):
            self.info.hostname.cache_clear()
            self.assertIsNone(self.info.hostname())
        self.info.hostname.cache_clear()

    def test_format_hz_khz_and_negative(self):
        self.assertEqual(self.info.format_hz(1500), "2Khz")
        self.assertIsNone(self.info.format_hz(-1))

    def test_format_size_string_and_numeric_forms(self):
        self.assertEqual(self.info._format_size("1.5 GiB"), "1.5Gb")
        self.assertEqual(self.info._format_size("4 Mb"), "4Mb")
        self.assertEqual(self.info._format_size("2048"), "2048")
        self.assertEqual(self.info._format_size("0x800"), "2Kb")
        self.assertEqual(self.info._format_size("16.0"), "16")
        self.assertEqual(self.info._format_size("2048b"), "2048b")
        self.assertEqual(self.info._format_size("1Tb"), "1Tb")
        self.assertEqual(self.info._format_size("abc"), "abc")
        self.assertEqual(self.info._format_size(""), None)
        self.assertEqual(self.info._format_size(2**40), "1024Gb")
        self.assertEqual(self.info._format_size(-5), None)
        self.assertIsNone(self.info._format_size(object()))

    def test_proc_cpuinfo_parses_and_casts(self):
        lines = [
            "vendor_id\t: GenuineIntel\n",
            "model name\t: Test CPU\n",
            "cpu family\t: 6\n",
            "model\t\t: 42\n",
            "stepping\t: bad\n",
            "cpu cores\t: 8\n",
            "vendor_id\t: Ignored\n",
            "unknown\t: value\n",
            "no colon line\n",
        ]
        fake = io.StringIO("".join(lines))
        with patch("builtins.open", return_value=fake):
            self.info._proc_cpuinfo.cache_clear()
            out = self.info._proc_cpuinfo()
        self.assertEqual(
            out,
            {
                "vendor_id": "GenuineIntel",
                "brand_raw": "Test CPU",
                "family": 6,
                "model": 42,
                "count": 8,
            },
        )

    def test_proc_cpuinfo_missing(self):
        with patch("builtins.open", side_effect=OSError("bad")):
            self.info._proc_cpuinfo.cache_clear()
            self.assertEqual(self.info._proc_cpuinfo(), {})

    def test_cpuinfo_prefers_proc_and_falls_back(self):
        with patch.object(
            self.info, "_proc_cpuinfo", return_value={"brand_raw": "CPU"}
        ):
            self.info._cpuinfo.cache_clear()
            info = self.info._cpuinfo()
        self.assertEqual(info["brand_raw"], "CPU")
        self.assertIn("arch_string_raw", info)
        with patch.object(self.info, "_proc_cpuinfo", return_value={}):
            with patch("cpuinfo.get_cpu_info", return_value={"brand_raw": "x"}):
                self.info._cpuinfo.cache_clear()
                self.assertEqual(self.info._cpuinfo(), {"brand_raw": "x"})
            with patch("cpuinfo.get_cpu_info", side_effect=RuntimeError("bad")):
                self.info._cpuinfo.cache_clear()
                self.assertEqual(self.info._cpuinfo(), {})

    def test_online_cpus_parsing_and_fallbacks(self):
        with patch.object(self.info, "_read_text", return_value="0-2,4"):
            self.info._online_cpus.cache_clear()
            self.assertEqual(self.info._online_cpus(), [0, 1, 2, 4])
        with (
            patch.object(self.info, "_read_text", return_value=""),
            patch(
                "perf.info.glob.glob",
                return_value=[
                    "/sys/devices/system/cpu/cpu2",
                    "/sys/devices/system/cpu/cpu0",
                    "/sys/devices/system/cpu/cpuX",
                ],
            ),
        ):
            self.info._online_cpus.cache_clear()
            self.assertEqual(self.info._online_cpus(), [0, 2])
        with (
            patch.object(self.info, "_read_text", return_value=""),
            patch("perf.info.glob.glob", return_value=[]),
            patch.object(self.info, "_proc_cpuinfo", return_value={"count": 3}),
        ):
            self.info._online_cpus.cache_clear()
            self.assertEqual(self.info._online_cpus(), [0, 1, 2])
        with (
            patch.object(self.info, "_read_text", return_value=""),
            patch("perf.info.glob.glob", return_value=[]),
            patch.object(self.info, "_proc_cpuinfo", return_value={"count": "bad"}),
        ):
            self.info._online_cpus.cache_clear()
            self.assertEqual(self.info._online_cpus(), [0])
        with (
            patch.object(self.info, "_read_text", return_value=""),
            patch("perf.info.glob.glob", side_effect=OSError("bad")),
            patch.object(self.info, "_proc_cpuinfo", return_value={}),
        ):
            self.info._online_cpus.cache_clear()
            self.assertEqual(self.info._online_cpus(), [0])

    def test_cpu_core_id_fallbacks(self):
        def fake_read(path):
            if path.endswith("core_id"):
                return "bad"
            if path.endswith("thread_siblings_list"):
                return "4-5"
            return ""

        with patch.object(self.info, "_read_text", side_effect=fake_read):
            self.info._cpu_core_id.cache_clear()
            self.assertEqual(self.info._cpu_core_id(9), 4)

        def bad_siblings(path):
            if path.endswith("thread_siblings_list"):
                return "bad"
            return ""

        with patch.object(self.info, "_read_text", side_effect=bad_siblings):
            self.info._cpu_core_id.cache_clear()
            self.assertEqual(self.info._cpu_core_id(9), 9)

    def test_cpu_numa_node_sources(self):
        def fake_read(path):
            if path.endswith("node0/cpulist"):
                return "0"
            if path.endswith("node1/cpulist"):
                return ""
            if path.endswith("node1/cpumap"):
                return "00000002"
            return ""

        with (
            patch(
                "perf.info.glob.glob",
                return_value=[
                    "/sys/devices/system/node/node0",
                    "/sys/devices/system/node/node1",
                ],
            ),
            patch.object(self.info, "_read_text", side_effect=fake_read),
            patch.object(self.info.os.path, "exists", return_value=False),
        ):
            self.info._cpu_numa_node.cache_clear()
            self.assertEqual(self.info._cpu_numa_node(1), 1)
        with (
            patch(
                "perf.info.glob.glob",
                return_value=["/sys/devices/system/node/nodeX"],
            ),
            patch.object(self.info, "_read_text", return_value=""),
            patch.object(self.info.os.path, "exists", return_value=False),
        ):
            self.info._cpu_numa_node.cache_clear()
            self.assertEqual(self.info._cpu_numa_node(1), 0)
        with (
            patch(
                "perf.info.glob.glob",
                return_value=["/sys/devices/system/node/node2"],
            ),
            patch.object(self.info, "_read_text", return_value=""),
            patch.object(self.info.os.path, "exists", return_value=True),
        ):
            self.info._cpu_numa_node.cache_clear()
            self.assertEqual(self.info._cpu_numa_node(1), 2)
        with patch("perf.info.glob.glob", side_effect=OSError("bad")):
            self.info._cpu_numa_node.cache_clear()
            self.assertEqual(self.info._cpu_numa_node(1), 0)

    def test_sysfs_cache_sizes(self):
        def fake_read(path):
            values = {
                "index0/level": "1\n",
                "index0/type": "Data\n",
                "index0/size": "32K\n",
                "index1/level": "1\n",
                "index1/type": "Instruction\n",
                "index1/size": "64K\n",
                "index2/level": "2\n",
                "index2/type": "Unified\n",
                "index2/size": "256K\n",
                "index3/level": "3\n",
                "index3/type": "Unified\n",
                "index3/size": "8M\n",
                "index4/level": "bad\n",
                "index4/type": "Unified\n",
                "index4/size": "1M\n",
                "index5/level": "1\n",
                "index5/type": "Data\n",
                "index5/size": "",
                "index6/level": "1\n",
                "index6/type": "Unified\n",
                "index6/size": "1M\n",
            }
            for suffix, value in values.items():
                if path.endswith(suffix):
                    return value
            return ""

        indices = [
            f"/sys/devices/system/cpu/cpu0/cache/index{index}" for index in range(7)
        ]
        with (
            patch("perf.info.glob.glob", return_value=indices),
            patch.object(self.info, "_read_text", side_effect=fake_read),
        ):
            self.info._sysfs_cache_sizes.cache_clear()
            sizes = self.info._sysfs_cache_sizes(0)
        self.assertEqual(
            sizes, {"L1d": "32Kb", "L1i": "64Kb", "L2": "256Kb", "L3": "8Mb"}
        )

    def test_compute_cpuinfo_rows(self):
        with (
            patch.object(self.info, "_chip_info", return_value={"arch": "x86_64"}),
            patch.object(self.info, "_online_cpus", return_value=[0, 2]),
            patch.object(
                self.info,
                "_sysfs_cache_sizes",
                side_effect=[{"L1d": "32Kb"}, {"L2": "1Mb"}],
            ),
            patch.object(self.info, "_cpu_core_id", side_effect=[0, 1]),
            patch.object(self.info, "_cpu_numa_node", side_effect=[0, 1]),
        ):
            rows = self.info._compute_cpuinfo_rows()
        self.assertEqual(
            rows,
            [
                {
                    "cpu": 0,
                    "core": 0,
                    "numa": 0,
                    "arch": "x86_64",
                    "L1i": None,
                    "L1d": "32Kb",
                    "L2": None,
                    "L3": None,
                },
                {
                    "cpu": 2,
                    "core": 1,
                    "numa": 1,
                    "arch": "x86_64",
                    "L1i": None,
                    "L1d": None,
                    "L2": "1Mb",
                    "L3": None,
                },
            ],
        )

    def test_cpuinfo_rows_uses_the_cached_key(self):
        with (
            patch.object(self.info, "_online_cpus", return_value=[0]),
            patch.object(
                self.info, "_cpuinfo_rows_cached", return_value=[{"cpu": 0}]
            ) as cached,
        ):
            self.assertEqual(self.info._cpuinfo_rows(), [{"cpu": 0}])
        cached.assert_called_once_with(((0,),))
        with (
            patch.object(self.info, "_online_cpus", side_effect=OSError("bad")),
            patch.object(
                self.info, "_compute_cpuinfo_rows", return_value=[{"cpu": 1}]
            ) as compute,
        ):
            self.assertEqual(self.info._cpuinfo_rows(), [{"cpu": 1}])
        compute.assert_called_once()
        with patch.object(
            self.info, "_compute_cpuinfo_rows", return_value=[{"cpu": 2}]
        ):
            self.info._cpuinfo_rows_cached.cache_clear()
            self.assertEqual(self.info._cpuinfo_rows_cached((0,)), [{"cpu": 2}])

    def test_os_cpu_freq_prefers_sysfs(self):
        with (
            patch.object(self.info, "_online_cpus", side_effect=RuntimeError("bad")),
            patch.object(
                self.info,
                "_read_text",
                side_effect=["bad", "2400000\n"],
            ),
        ):
            self.info._os_cpu_freq.cache_clear()
            self.assertEqual(self.info._os_cpu_freq(), 2_400_000_000)

    def test_os_cpu_freq_falls_back_to_proc(self):
        fake = io.StringIO("cpu MHz\t\t: 3200.000\n")
        with (
            patch.object(self.info, "_online_cpus", return_value=[0]),
            patch.object(self.info, "_read_text", return_value=""),
            patch("builtins.open", return_value=fake),
        ):
            self.info._os_cpu_freq.cache_clear()
            self.assertEqual(self.info._os_cpu_freq(), 3_200_000_000)

    def test_os_cpu_freq_missing(self):
        with (
            patch.object(self.info, "_online_cpus", return_value=[0]),
            patch.object(self.info, "_read_text", return_value=""),
            patch("builtins.open", side_effect=OSError("bad")),
        ):
            self.info._os_cpu_freq.cache_clear()
            self.assertIsNone(self.info._os_cpu_freq())

    def test_cpu_freq_all_probes_fail(self):
        arch = SimpleNamespace(cpuid_tsc_hz=Mock(side_effect=RuntimeError("bad")))
        with (
            patch.object(self.info, "_arch_fn", return_value=arch),
            patch.object(self.info, "_tsc_calibrate", return_value=None),
            patch.object(self.info, "_os_cpu_freq", return_value=None),
        ):
            self.info._cpu_freq.cache_clear()
            self.assertIsNone(self.info._cpu_freq())

    def test_tsc_calibration_errors_return_none(self):
        with patch.object(self.info, "_arch_fn", side_effect=RuntimeError("bad")):
            self.assertIsNone(self.info._tsc_calibrate())
        arch = SimpleNamespace(tsc_reader_bytes=lambda: b"\xc3")
        with (
            patch.object(self.info, "_arch_fn", return_value=arch),
            patch.object(self.info.mmap, "mmap", side_effect=OSError("bad")),
        ):
            self.assertIsNone(self.info._tsc_calibrate())
        with (
            patch.object(self.info, "_arch_fn", return_value=arch),
            patch.object(self.info.time, "perf_counter", side_effect=[1.0, 1.0]),
            patch.object(self.info.time, "sleep"),
        ):
            self.assertIsNone(self.info._tsc_calibrate())

    def test_chip_info(self):
        with (
            patch.object(
                self.info,
                "_cpuinfo",
                return_value={
                    "arch_string_raw": "x86_64",
                    "brand_raw": "CPU",
                    "model": 1,
                    "family": 2,
                    "stepping": 3,
                },
            ),
            patch.object(self.info, "_cpu_freq", return_value=1000),
            patch("platform.platform", return_value="Linux"),
        ):
            self.assertEqual(
                self.info._chip_info(),
                {
                    "arch": "x86_64",
                    "platform": "Linux",
                    "vendor": "CPU",
                    "model": 1,
                    "family": 2,
                    "stepping": 3,
                    "freq": 1000,
                },
            )

    def test_cpuinfo_with_selected_fields(self):
        df = self.info.cpuinfo(["cpu", "freq"])
        self.assertEqual(list(df.columns), ["cpu", "freq"])


class TestHexAddressFallbacks(unittest.TestCase):
    def test_numeric_conversion_fallback(self):
        import pandas as pd

        from perf.info import hex_addresses

        df = pd.DataFrame({"begin": ["4096"]})
        with patch.object(pd, "to_numeric", side_effect=ValueError("bad")):
            out = hex_addresses(df)
        self.assertEqual(out["begin"].tolist(), ["0x1000"])

    def test_unconvertible_values_are_left_alone(self):
        import pandas as pd

        from perf.info import hex_addresses

        df = pd.DataFrame({"begin": ["nope"]})
        with patch.object(pd, "to_numeric", side_effect=ValueError("bad")):
            out = hex_addresses(df)
        self.assertEqual(out["begin"].tolist(), ["nope"])


if __name__ == "__main__":
    unittest.main()
