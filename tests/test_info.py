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
import os
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

from perf.info import functions, labels, regions, targets


class TestRegions(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(regions([]), {})

    def test_begin_then_end_builds_region(self):
        entries = [("main_begin", 0x1000), ("main_end", 0x2000)]
        self.assertEqual(regions(entries), {"main": (0x1000, 0x2000)})

    def test_end_then_begin_builds_region(self):
        entries = [("main_end", 0x2000), ("main_begin", 0x1000)]
        self.assertEqual(regions(entries), {"main": (0x1000, 0x2000)})

    def test_multiple_regions(self):
        entries = [
            ("a_begin", 0x1000),
            ("a_end", 0x2000),
            ("b_begin", 0x3000),
            ("b_end", 0x4000),
        ]
        self.assertEqual(
            regions(entries),
            {
                "a": (0x1000, 0x2000),
                "b": (0x3000, 0x4000),
            },
        )

    def test_unmatched_begin_creates_no_region(self):
        self.assertEqual(regions([("main_begin", 0x1000)]), {})

    def test_skips_plain_labels(self):
        entries = [
            ("plain", 0x1000),
            ("hot_begin", 0x2000),
            ("hot_end", 0x3000),
        ]
        self.assertEqual(regions(entries), {"hot": (0x2000, 0x3000)})

    def test_names_with_inner_underscores(self):
        entries = [("hot_loop_begin", 0x1000), ("hot_loop_end", 0x2000)]
        self.assertEqual(regions(entries), {"hot_loop": (0x1000, 0x2000)})

    def test_nested_regions(self):
        entries = [
            ("outer_begin", 0x1000),
            ("inner_begin", 0x2000),
            ("inner_end", 0x3000),
            ("outer_end", 0x4000),
        ]
        self.assertEqual(
            regions(entries),
            {"inner": (0x2000, 0x3000), "outer": (0x1000, 0x4000)},
        )

    def test_interleaved_order(self):
        entries = [
            ("a_begin", 0x1000),
            ("b_begin", 0x2000),
            ("a_end", 0x3000),
            ("b_end", 0x4000),
        ]
        self.assertEqual(
            regions(entries),
            {"a": (0x1000, 0x3000), "b": (0x2000, 0x4000)},
        )

    def test_unmatched_end(self):
        self.assertEqual(regions([("main_end", 0x1000)]), {})

    def test_multiple_pairs_same_name(self):
        entries = [
            ("hot_begin", 0x1000),
            ("hot_end", 0x2000),
            ("hot_begin", 0x3000),
            ("hot_end", 0x4000),
        ]
        self.assertEqual(
            regions(entries),
            {"hot": (0x3000, 0x4000)},
        )


class TestLabels(unittest.TestCase):
    def _project(self, blob, label=".perf.label"):
        project = Mock()
        obj = project.loader.main_object
        obj.binary = "/nonexistent"
        obj.mapped_base = 0x400000
        obj.pic = True
        obj.sections_map = {label: Mock(offset=0, filesize=len(blob))}
        return project, label

    def _write_binary(self, blob):
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        self.addCleanup(os.remove, path)
        return path

    def test_no_section_returns_empty(self):
        project = Mock()
        project.loader.main_object.sections_map = {}
        self.assertEqual(labels(project), [])

    def test_parses_labels(self):
        blob = (
            struct.pack("<Q", 0x1234)
            + b"main\0"
            + struct.pack("<Q", 0x5678)
            + b"helper\0"
        )
        project, label = self._project(blob)
        obj = project.loader.main_object
        obj.binary = self._write_binary(blob)
        self.assertEqual(
            labels(project),
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
        obj.binary = self._write_binary(blob)
        self.assertEqual(
            labels(project),
            [("hot_begin", 0x401174), ("hot_end", 0x401189)],
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
        obj.binary = self._write_binary(blob)
        self.assertEqual(
            labels(project),
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
        obj.binary = self._write_binary(blob)
        self.assertEqual(labels(project), [("hot_begin", 0x401174)])

    def test_truncated_address_raises(self):
        blob = b"\x34\x12"
        project, _ = self._project(blob)
        obj = project.loader.main_object
        obj.binary = self._write_binary(blob)
        with self.assertRaises(ValueError):
            labels(project)

    def test_missing_null_terminator_raises(self):
        blob = struct.pack("<Q", 0x1234) + b"hot_begin"
        project, _ = self._project(blob)
        obj = project.loader.main_object
        obj.binary = self._write_binary(blob)
        with self.assertRaises(ValueError):
            labels(project)


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


class TestTargets(unittest.TestCase):
    @patch("perf.info.regions", return_value={})
    @patch("perf.info.labels", return_value=[])
    @patch("perf.info.functions", return_value=({"foo": (0x1122, 0x1133)}, {}))
    def test_regex_matches_function_names(
        self, mock_functions, mock_labels, mock_regions
    ):
        self.assertEqual(list(targets(None, "foo")), [("foo", 0x1122, 0x1133)])

    @patch("perf.info.regions", return_value={})
    @patch("perf.info.labels", return_value=[])
    @patch("perf.info.functions", return_value=({"foo": (0x1122, 0x1133)}, {}))
    def test_regex_is_partial(self, mock_functions, mock_labels, mock_regions):
        got = list(targets(None, "o"))
        self.assertEqual(got, [("foo", 0x1122, 0x1133)])

    @patch("perf.info.regions", return_value={})
    @patch("perf.info.labels", return_value=[])
    @patch("perf.info.functions", return_value=({}, {}))
    def test_no_match_is_empty(self, mock_functions, mock_labels, mock_regions):
        self.assertEqual(list(targets(None, "nope")), [])

    @patch(
        "perf.info.labels", return_value=[("hot_begin", 0x1000), ("hot_end", 0x2000)]
    )
    def test_range_spec_uses_labels(self, mock_labels):
        self.assertEqual(
            list(targets(None, "hot_begin..hot_end")),
            [("hot_begin..hot_end", 0x1000, 0x2000)],
        )

    @patch("perf.info.labels", return_value=[("hot_begin", 0x1000)])
    def test_range_spec_missing_end_is_empty(self, mock_labels):
        self.assertEqual(list(targets(None, "hot_begin..missing")), [])

    def test_range_spec_hex_addresses(self):
        self.assertEqual(
            list(targets(None, "0x1000..0x2000")),
            [("0x1000..0x2000", 0x1000, 0x2000)],
        )

    def test_range_spec_hex_and_decimal_with_spaces(self):
        self.assertEqual(
            list(targets(None, "  0x1000 .. 8192  ")),
            [("  0x1000 .. 8192  ", 0x1000, 8192)],
        )

    @patch("perf.info.regions", return_value={})
    @patch("perf.info.labels", return_value=[])
    def test_range_spec_func_endpoints(self, mock_labels, mock_regions):
        funcs = {"main": (0x1000, 0x1100), "foo": (0x2000, 0x2100)}
        self.assertEqual(
            list(targets(None, "main..foo", funcs=funcs)),
            [("main..foo", 0x1000, 0x2100)],
        )
        self.assertEqual(
            list(targets(None, "main..main", funcs=funcs)),
            [("main..main", 0x1000, 0x1100)],
        )

    @patch("perf.info.regions", return_value={})
    @patch("perf.info.labels", return_value=[])
    def test_range_spec_func_offsets(self, mock_labels, mock_regions):
        funcs = {"main": (0x1000, 0x1100), "foo": (0x2000, 0x2100)}
        self.assertEqual(
            list(targets(None, "main+0x10..foo-0x10", funcs=funcs)),
            [("main+0x10..foo-0x10", 0x1010, 0x20F0)],
        )
        self.assertEqual(
            list(targets(None, "0x1000+0x10..0x2000-16", funcs=funcs)),
            [("0x1000+0x10..0x2000-16", 0x1010, 0x1FF0)],
        )

    @patch("perf.info.regions", return_value={})
    @patch(
        "perf.info.labels",
        return_value=[("hot_begin", 0x1000), ("hot_end", 0x2000)],
    )
    def test_range_spec_labels_kept_with_offsets_and_mixed_sides(
        self, mock_labels, mock_regions
    ):
        funcs = {"foo": (0x3000, 0x3100)}
        self.assertEqual(
            list(targets(None, "hot_begin+0x10..hot_end-16", funcs=funcs)),
            [("hot_begin+0x10..hot_end-16", 0x1010, 0x1FF0)],
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

    @patch("perf.info.regions", return_value={})
    @patch("perf.info.labels", return_value=[])
    def test_range_spec_partial_and_ambiguous(self, mock_labels, mock_regions):
        funcs = {"main": (0x1000, 0x1100)}
        self.assertEqual(
            list(targets(None, "mai..0x2000", funcs=funcs)),
            [("mai..0x2000", 0x1000, 0x2000)],
        )
        ambiguous = {"main": (0x1000, 0x1100), "maintain": (0x2000, 0x2100)}
        self.assertEqual(
            list(targets(None, "main..0x3000", funcs=ambiguous)),
            [("main..0x3000", 0x1000, 0x3000)],
        )
        self.assertEqual(list(targets(None, "mai..0x3000", funcs=ambiguous)), [])
        self.assertEqual(list(targets(None, "missing..0x3000", funcs=funcs)), [])

    @patch("perf.info.regions", return_value={})
    @patch("perf.info.labels", return_value=[])
    @patch("perf.info.functions", return_value=({"foo": (0x1000, 0x1005)}, {}))
    def test_precomputed_functions_avoid_cfg(
        self, mock_functions, mock_labels, mock_regions
    ):
        targets(None, "foo", funcs={"foo": (0x1000, 0x1005)})
        mock_functions.assert_not_called()


class TestRegionsBinary(unittest.TestCase):
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

    def _write_binary(self, blob):
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        self.addCleanup(os.remove, path)
        return path

    def test_pie_region_addresses(self):
        blob = self._blob((0x1184, "hot_begin"), (0x1199, "hot_end"))
        project = self._project(blob, pic=True)
        project.loader.main_object.binary = self._write_binary(blob)
        regs = regions(labels(project))
        start, end = regs["hot"]
        self.assertEqual(start, 0x400000 + 0x1184)
        self.assertEqual(end, 0x400000 + 0x1199)
        self.assertLess(start, end)

    def test_no_pie_region_addresses(self):
        blob = self._blob((0x401174, "hot_begin"), (0x401189, "hot_end"))
        project = self._project(blob, pic=False)
        project.loader.main_object.binary = self._write_binary(blob)
        regs = regions(labels(project))
        start, end = regs["hot"]
        self.assertEqual(start, 0x401174)
        self.assertEqual(end, 0x401189)
        self.assertLess(start, end)

    def test_nested_regions_from_blob(self):
        blob = self._blob(
            (0x1000, "outer_begin"),
            (0x2000, "inner_begin"),
            (0x3000, "inner_end"),
            (0x4000, "outer_end"),
        )
        project = self._project(blob, pic=True)
        project.loader.main_object.binary = self._write_binary(blob)
        regs = regions(labels(project))
        self.assertEqual(
            regs,
            {
                "inner": (0x402000, 0x403000),
                "outer": (0x401000, 0x404000),
            },
        )
        for start, end in regs.values():
            self.assertLess(start, end)

    def test_no_pie_nested_regions_from_blob(self):
        blob = self._blob(
            (0x401000, "outer_begin"),
            (0x402000, "inner_begin"),
            (0x403000, "inner_end"),
            (0x404000, "outer_end"),
        )
        project = self._project(blob, pic=False)
        project.loader.main_object.binary = self._write_binary(blob)
        regs = regions(labels(project))
        self.assertEqual(
            regs,
            {
                "inner": (0x402000, 0x403000),
                "outer": (0x401000, 0x404000),
            },
        )
        for start, end in regs.values():
            self.assertLess(start, end)

    def test_pie_large_region_span(self):
        blob = self._blob(
            (0x1000, "func_begin"),
            (0x100000, "func_end"),
        )
        project = self._project(blob, pic=True)
        project.loader.main_object.binary = self._write_binary(blob)
        regs = regions(labels(project))
        start, end = regs["func"]
        self.assertEqual(end - start, 0x100000 - 0x1000)
        self.assertLess(start, end)

    def test_pie_single_byte_region(self):
        blob = self._blob((0x5000, "seq_begin"), (0x5001, "seq_end"))
        project = self._project(blob, pic=True)
        project.loader.main_object.binary = self._write_binary(blob)
        regs = regions(labels(project))
        start, end = regs["seq"]
        self.assertEqual(end - start, 1)
        self.assertLess(start, end)

    def test_targets_range_from_blob(self):
        blob = self._blob((0x1184, "hot_begin"), (0x1199, "hot_end"))
        project = self._project(blob, pic=True)
        project.loader.main_object.binary = self._write_binary(blob)
        got = list(targets(project, "hot_begin..hot_end"))
        self.assertEqual(len(got), 1)
        name, start, end = got[0]
        self.assertEqual(start, 0x400000 + 0x1184)
        self.assertEqual(end, 0x400000 + 0x1199)
        self.assertLess(start, end)


class TestMetadataAll(unittest.TestCase):
    @patch("perf.info.angr.Project")
    @patch("perf.info.labels")
    @patch("perf.info.functions")
    @patch("perf.info.regions")
    def test_metadata_returns_all_kinds(
        self, mock_regions, mock_functions, mock_labels, mock_project
    ):
        import pandas as pd

        from perf.info import metadata

        mock_labels.return_value = [("main", 0x401000), ("helper", 0x402000)]
        mock_functions.return_value = (
            {"foo": (0x401000, 0x401017), "bar": (0x402000, 0x402010)},
            {},
        )
        mock_regions.return_value = {"hot": (0x403000, 0x403100)}

        df = metadata("a.out")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["kind", "name", "start", "end", "size"])
        self.assertEqual(len(df), 5)
        kinds = df["kind"].tolist()
        self.assertEqual(kinds.count("label"), 2)
        self.assertEqual(kinds.count("func"), 2)
        self.assertEqual(kinds.count("region"), 1)

    @patch("perf.info.angr.Project")
    @patch("perf.info.labels", return_value=[])
    @patch("perf.info.functions", return_value=({}, {}))
    @patch("perf.info.regions", return_value={})
    def test_metadata_empty_output(
        self, mock_regions, mock_functions, mock_labels, mock_project
    ):
        import pandas as pd

        from perf.info import metadata

        df = metadata("a.out")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["kind", "name", "start", "end", "size"])
        self.assertEqual(len(df), 0)


class TestMetadata(unittest.TestCase):
    @patch("perf.info.angr.Project")
    @patch("perf.info.labels")
    @patch("perf.info.functions")
    @patch("perf.info.regions")
    def test_metadata_functions(
        self, mock_regions, mock_functions, mock_labels, mock_project
    ):
        import pandas as pd

        from perf.info import metadata

        mock_labels.return_value = [
            ("main", 0x401000),
            ("hot_begin", 0x403000),
            ("hot_end", 0x403100),
        ]
        mock_functions.return_value = (
            {"foo": (0x401000, 0x401017), "bar": (0x402000, 0x402010)},
            {},
        )
        mock_regions.return_value = {"hot": (0x403000, 0x403100)}

        df = metadata("a.out", "func")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["kind", "name", "start", "end", "size"])
        self.assertEqual(df["kind"].tolist(), ["func", "func"])
        self.assertEqual(df["name"].tolist(), ["foo", "bar"])

    @patch("perf.info.angr.Project")
    @patch("perf.info.labels")
    @patch("perf.info.functions")
    @patch("perf.info.regions")
    def test_metadata_regions(
        self, mock_regions, mock_functions, mock_labels, mock_project
    ):
        import pandas as pd

        from perf.info import metadata

        mock_labels.return_value = [("hot_begin", 0x403000), ("hot_end", 0x403100)]
        mock_functions.return_value = ({"foo": (0x401000, 0x401017)}, {})
        mock_regions.return_value = {"hot": (0x403000, 0x403100)}

        df = metadata("a.out", "region")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["kind", "name", "start", "end", "size"])
        self.assertIn("label", df["kind"].tolist())
        self.assertIn("region", df["kind"].tolist())
        self.assertNotIn("func", df["kind"].tolist())

    @patch("perf.info.angr.Project")
    @patch("perf.info.labels")
    @patch("perf.info.functions")
    @patch("perf.info.regions")
    def test_metadata_rejects_unknown_kind(
        self, mock_regions, mock_functions, mock_labels, mock_project
    ):
        from perf.info import metadata

        with self.assertRaises(ValueError):
            metadata("a.out", "asm")

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
            "hz",
            "L1i",
            "L1d",
            "L2",
            "L3",
        ):
            self.assertIn(key, df.columns)
            self.assertIn(key, df.iloc[0].to_dict())

    def test_exposed_on_package(self):
        import perf

        self.assertTrue(callable(perf.cpuinfo))


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


class TestRegionsPairing(unittest.TestCase):
    def test_double_begin_no_region(self):
        self.assertEqual(regions([("hot_begin", 0x1000), ("hot_begin", 0x2000)]), {})

    def test_double_end_no_region(self):
        self.assertEqual(regions([("hot_end", 0x1000), ("hot_end", 0x2000)]), {})

    def test_begin_begin_end_pairs_last(self):
        got = regions(
            [
                ("hot_begin", 0x1000),
                ("hot_begin", 0x3000),
                ("hot_end", 0x4000),
            ]
        )
        self.assertEqual(got, {"hot": (0x3000, 0x4000)})

    def test_ignores_other_suffix(self):
        self.assertEqual(regions([("hot_middle", 0x1000)]), {})


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
    @patch("perf.info.angr.Project")
    @patch("perf.info.labels")
    @patch("perf.info.functions")
    @patch("perf.info.regions")
    def test_names_have_no_newlines(
        self, mock_regions, mock_functions, mock_labels, mock_project
    ):
        from perf.info import metadata

        mock_labels.return_value = [("bad\nlabel", 0x401000)]
        mock_functions.return_value = ({"bad\nfunc": (0x402000, 0x402010)}, {})
        mock_regions.return_value = {"bad\nregion": (0x403000, 0x403100)}
        df = metadata("a.out")
        for name in df["name"].tolist():
            self.assertNotIn("\n", str(name))
            self.assertNotIn("\r", str(name))


if __name__ == "__main__":
    unittest.main()
