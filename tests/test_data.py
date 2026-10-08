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

import importlib as _importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import perf
import perf.data as pr

pi = _importlib.import_module("perf.info")
_SAMPLES = (
    "# ========\n"
    "# captured on: 2026-09-08\n"
    "#\n"
    "       perf-exec 2631 425759.959363:          1 cpu_core/cycles/:  "
    "ffffffff9d140c27 [unknown] ([unknown])\n"
    "         python3 2632 425759.960681:    6169252 cycles:ppp:  "
    "7ffff7970e93 __memcmp_avx2_movbe (/usr/lib/x86_64-linux-gnu/libc.so.6)\n"
    "            perf 2633 425759.959502:    5519178 instructions:  "
    "4d1642 [unknown] (/usr/bin/python3.11)\n"
)


class TestIsRecord(unittest.TestCase):
    def test_magic_detected(self):
        with tempfile.TemporaryDirectory() as d:
            for magic in (b"PERFILE1", b"PERFILE2"):
                p = Path(d) / f"pd_{magic[:7].decode()}"
                p.write_bytes(magic + b"rest")
                self.assertTrue(pr.is_record(p))

    def test_non_record_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.json"
            p.write_bytes(b'[{"cycles": 1}]')
            self.assertFalse(pr.is_record(p))
            self.assertFalse(pr.is_record(Path(d) / "missing"))


class TestParseLines(unittest.TestCase):
    def test_parses_samples(self):
        df = pr._parse_lines(_SAMPLES)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), pr._FIELDS.split(","))
        self.assertEqual(df["comm"].tolist(), ["perf-exec", "python3", "perf"])
        self.assertEqual(df["pid"].tolist(), [2631, 2632, 2633])
        self.assertEqual(df["period"].tolist(), [1, 6169252, 5519178])
        self.assertEqual(df["event"].tolist(), ["cycles", "cycles", "instructions"])
        self.assertEqual(df["ip"].iloc[0], int("ffffffff9d140c27", 16))
        self.assertEqual(df["sym"].iloc[1], "__memcmp_avx2_movbe")
        self.assertEqual(df["dso"].iloc[1], "/usr/lib/x86_64-linux-gnu/libc.so.6")
        self.assertEqual(df["sym"].iloc[0], "[unknown]")

    def test_norm_event_strips_pmu_and_modifiers(self):
        self.assertEqual(pr._norm_event("cycles"), "cycles")
        self.assertEqual(pr._norm_event("cpu_core/cycles/"), "cycles")
        self.assertEqual(
            pr._norm_event("cpu_core/topdown-fe-bound/"), "topdown-fe-bound"
        )
        self.assertEqual(pr._norm_event("cycles:ppp"), "cycles")


class TestSamples(unittest.TestCase):
    @patch("perf.data.subprocess.run")
    def test_calls_perf_script(self, run):
        proc = Mock(returncode=0, stdout=_SAMPLES, stderr="")
        run.return_value = proc
        df = pr.samples("perf.data", bin="/usr/bin/perf")
        self.assertEqual(len(df), 3)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:5], ["/usr/bin/perf", "script", "-i", "perf.data", "-F"])
        self.assertEqual(df["event"].tolist(), ["cycles", "cycles", "instructions"])

    @patch("perf.data.subprocess.run")
    def test_raises_on_failure(self, run):
        proc = Mock(returncode=1, stdout="", stderr="boom")
        run.return_value = proc
        with self.assertRaises(RuntimeError):
            pr.samples("perf.data", bin="/usr/bin/perf")


class TestMetrics(unittest.TestCase):
    def test_widens_into_event_columns(self):
        long = pd.DataFrame(
            {
                "file": ["a", "a", "a", "a"],
                "time": [1, 2, 3, 4],
                "event": ["cycles", "cycles", "instructions", "cycles"],
                "period": [10, 20, 30, 40],
            }
        )
        wide = pr.metrics(long)
        self.assertIn("cycles", wide.columns)
        self.assertIn("instructions", wide.columns)
        self.assertEqual([v for v in wide["cycles"] if pd.notna(v)], [10.0, 20.0, 40.0])
        self.assertEqual([v for v in wide["instructions"] if pd.notna(v)], [30.0])

    def test_filters_requested_events(self):
        long = pd.DataFrame(
            {
                "time": [1, 2],
                "event": ["cycles", "instructions"],
                "period": [10, 20],
            }
        )
        wide = pr.metrics(long, ["cycles"])
        self.assertEqual(wide.columns.tolist(), ["time", "cycles"])
        self.assertEqual([v for v in wide["cycles"] if pd.notna(v)], [10.0])

    def test_noop_without_event(self):
        df = pd.DataFrame({"a": [1]})
        self.assertIs(pr.metrics(df), df)


class TestParse(unittest.TestCase):
    @patch("perf.data.samples")
    def test_concats_files_with_file_column(self, samples):
        samples.side_effect = [
            pr._parse_lines(_SAMPLES),
            pr._parse_lines(_SAMPLES),
        ]
        df = pr.parse(["perf.data", "perf.data.old"])
        self.assertEqual(len(df), 6)
        self.assertCountEqual(df["file"].unique(), ["perf.data", "perf.data.old"])
        self.assertIn("cycles", df.columns)
        self.assertIn("instructions", df.columns)


class TestNest(unittest.TestCase):
    def test_dotted_columns_become_nested_objects(self):
        self.assertEqual(
            pr.nest(
                {
                    "mode": "latency",
                    "config.backend.loop.probes": 3,
                    "config.dcache.L1d.hit_rate": 100,
                    "data.rdi": [1, 2, 3],
                    "cycles": 23.0,
                }
            ),
            {
                "mode": "latency",
                "config": {
                    "backend": {"loop": {"probes": 3}},
                    "dcache": {"L1d": {"hit_rate": 100}},
                },
                "data": {"rdi": [1, 2, 3]},
                "cycles": 23.0,
            },
        )

    def test_untouched_keys(self):
        self.assertEqual(
            pr.nest({"configlike.a": 1, "sample": 2, "cycles": 3}),
            {"configlike.a": 1, "sample": 2, "cycles": 3},
        )

    def test_a_scalar_prefix_keeps_the_dotted_keys_dotted(self):
        self.assertEqual(
            pr.nest({"config": 1, "config.a": 2}), {"config": 1, "config.a": 2}
        )
        self.assertEqual(
            pr.nest({"config.a": 2, "config": 1}), {"config.a": 2, "config": 1}
        )

    def test_empty(self):
        self.assertEqual(pr.nest({}), {})
        self.assertEqual(pr.nest(None), {})

    def test_custom_prefixes(self):
        self.assertEqual(pr.nest({"a.b": 1}, prefixes=("a",)), {"a": {"b": 1}})


class TestSpread(unittest.TestCase):
    def test_nested_objects_become_dotted_columns(self):
        self.assertEqual(
            pr.spread(
                {
                    "mode": "latency",
                    "config": {
                        "backend": {"loop": {"probes": 3}},
                    },
                    "data": {"rdi": 15},
                    "cycles": 23.0,
                }
            ),
            {
                "mode": "latency",
                "config.backend.loop.probes": 3,
                "data.rdi": 15,
                "cycles": 23.0,
            },
        )

    def test_regs_and_mem_buckets_are_flattened(self):
        self.assertEqual(
            pr.spread(
                {
                    "data": {
                        "regs": {"rdi": 15},
                        "mem": {"0x1000": 123},
                    }
                }
            ),
            {"data.rdi": 15, "data.0x1000": 123},
        )

    def test_non_dict_values_pass_through(self):
        self.assertEqual(pr.spread({"config": "loop"}), {"config": "loop"})

    def test_round_trip(self):
        row = {
            "mode": "latency",
            "config.backend.loop.runs": 10,
            "config.dcache.L1d.hit_rate": 100,
            "data.rdi": 15,
            "iterations": 9975,
            "samples": 99,
            "operations": 1,
            "cycles": 23.0,
        }
        self.assertEqual(pr.spread(pr.nest(row)), row)

    def test_exposed_on_package(self):
        self.assertIs(perf.samples.nest, pr.nest)
        self.assertIs(perf.samples.spread, pr.spread)


class TestQueryMembership(unittest.TestCase):
    def _df(self):
        return pd.DataFrame(
            {
                "assembly": ["a", "b", "c"],
                "sym": ["4 in cycles", "other", "x"],
                "data.rdi": [[15], [15], [16]],
                "data.0x10": [[1], [2], [3]],
                "cycles": [1.0, 2.0, 3.0],
                "size": [1, 2, 3],
            }
        )

    def test_membership_over_a_list_column(self):
        out = pr.query(self._df(), "15 in `data.rdi`")
        self.assertEqual(out["assembly"].tolist(), ["a", "b"])
        out = pr.query(self._df(), "16 in `data.rdi` and size > 0")
        self.assertEqual(out["assembly"].tolist(), ["c"])

    def test_membership_over_a_plain_column(self):
        out = pr.query(self._df(), "2 in cycles")
        self.assertEqual(out["assembly"].tolist(), ["b"])

    def test_float_and_hex_literals(self):
        self.assertTrue(pr.query(self._df(), "2.5 in `data.rdi`").empty)
        self.assertTrue(pr.query(self._df(), "1e3 in cycles").empty)
        self.assertTrue(pr.query(self._df(), "0x9 in cycles").empty)
        self.assertEqual(
            pr.query(self._df(), "0x2 in cycles")["assembly"].tolist(), ["b"]
        )

    def test_string_literals_are_untouched(self):
        out = pr.query(self._df(), "sym == '4 in cycles'")
        self.assertEqual(out["assembly"].tolist(), ["a"])

    def test_plain_filter_still_works(self):
        out = pr.query(self._df(), "size > 1")
        self.assertEqual(out["assembly"].tolist(), ["b", "c"])


class TestFindSystemPerf(unittest.TestCase):
    def test_searches_path_then_fallbacks(self):
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, "perf")
            Path(fake).touch()
            os.chmod(fake, 0o755)
            with patch.object(pr.os, "get_exec_path", return_value=[d]):
                self.assertEqual(pr.system_perf(), fake)

    def test_exposed_on_package(self):
        self.assertIs(perf.samples.is_record, pr.is_record)
        self.assertIs(perf.samples, pr.samples)
        self.assertIs(perf.samples.metrics, pr.metrics)
        self.assertIs(perf.samples.parse, pr.parse)


if __name__ == "__main__":
    unittest.main()
