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
from unittest.mock import Mock, patch

from perf.info import functions, labels, targets


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
        obj.binary = _write_binary(self, blob)
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
        obj.binary = _write_binary(self, blob)
        self.assertEqual(
            labels(project),
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
            labels(project),
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
        obj.binary = _write_binary(self, blob)
        self.assertEqual(labels(project), [("hot_begin", 0x401174)])

    def test_truncated_address_raises(self):
        blob = b"\x34\x12"
        project, _ = self._project(blob)
        obj = project.loader.main_object
        obj.binary = _write_binary(self, blob)
        with self.assertRaises(ValueError):
            labels(project)

    def test_missing_null_terminator_raises(self):
        blob = struct.pack("<Q", 0x1234) + b"hot_begin"
        project, _ = self._project(blob)
        obj = project.loader.main_object
        obj.binary = _write_binary(self, blob)
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
    @patch("perf.info.labels", return_value=[])
    @patch("perf.info.functions", return_value=({"foo": (0x1122, 0x1133)}, {}))
    def test_regex_matches_function_names(self, mock_functions, mock_labels):
        self.assertEqual(list(targets(None, "foo")), [("foo", 0x1122, 0x1133)])

    @patch("perf.info.labels", return_value=[])
    @patch("perf.info.functions", return_value=({"foo": (0x1122, 0x1133)}, {}))
    def test_partial_name_does_not_match(self, mock_functions, mock_labels):
        got = list(targets(None, "o"))
        self.assertEqual(got, [])

    @patch("perf.info.labels", return_value=[])
    @patch("perf.info.functions", return_value=({}, {}))
    def test_no_match_is_empty(self, mock_functions, mock_labels):
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

    @patch(
        "perf.info.labels", return_value=[("foo_end", 0x1000), ("foo_begin", 0x2000)]
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

    @patch("perf.info.labels", return_value=[])
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

    @patch("perf.info.labels", return_value=[])
    def test_range_spec_offsets_unsupported(self, mock_labels):
        funcs = {"main": (0x1000, 0x1100), "foo": (0x2000, 0x2100)}
        self.assertEqual(list(targets(None, "main+0x10..foo-0x10", funcs=funcs)), [])
        self.assertEqual(list(targets(None, "0x1000+0x10..0x2000-16", funcs=funcs)), [])

    @patch(
        "perf.info.labels",
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

    @patch("perf.info.labels", return_value=[])
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

    @patch("perf.info.labels", return_value=[])
    @patch("perf.info.functions", return_value=({"foo": (0x1000, 0x1005)}, {}))
    def test_precomputed_functions_avoid_cfg(self, mock_functions, mock_labels):
        targets(None, "foo", funcs={"foo": (0x1000, 0x1005)})
        mock_functions.assert_not_called()


class TestTargetsNamespace(unittest.TestCase):
    @patch("perf.info.labels", return_value=[])
    def test_bare_name_matches_namespaced_symbol(self, mock_labels):
        funcs = {"ns::foo": (0x1000, 0x1010)}
        self.assertEqual(
            list(targets(None, "foo", funcs=funcs)),
            [("ns::foo", 0x1000, 0x1010)],
        )

    @patch("perf.info.labels", return_value=[])
    def test_qualified_name_does_not_suffix_match(self, mock_labels):
        funcs = {"ns::foo": (0x1000, 0x1010)}
        self.assertEqual(list(targets(None, "other::foo", funcs=funcs)), [])

    @patch("perf.info.labels", return_value=[])
    def test_ambiguous_suffix_is_empty(self, mock_labels):
        funcs = {"a::foo": (0x1000, 0x1010), "b::foo": (0x2000, 0x2010)}
        self.assertEqual(list(targets(None, "foo", funcs=funcs)), [])

    @patch("perf.info.labels", return_value=[])
    def test_same_address_suffix_resolves(self, mock_labels):
        funcs = {"a::foo": (0x1000, 0x1010), "b::foo": (0x1000, 0x1010)}
        got = list(targets(None, "foo", funcs=funcs))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][1:], (0x1000, 0x1010))

    @patch("perf.info.labels", return_value=[])
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
    @patch("perf.info.labels")
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
    @patch("perf.info.labels")
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
    @patch("perf.info.labels", return_value=[])
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
            "perf.info.labels",
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
            "perf.info.labels",
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
    @patch("perf.info.labels")
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
            [name for name, _pos, _size in perf.asm_labels(self.path)],
            ["foo", "bar", ".Llocal", "loop"],
        )
        self.assertIn("asm_labels", perf.__all__)


if __name__ == "__main__":
    unittest.main()
