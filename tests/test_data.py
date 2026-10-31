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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import perf
import perf.data as pr

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

    def test_norm_event_keeps_tracepoint_names(self):
        self.assertEqual(pr._norm_event("sched:sched_switch"), "sched:sched_switch")
        self.assertEqual(
            pr._norm_event("syscalls:sys_enter_openat"), "syscalls:sys_enter_openat"
        )

    def test_comm_with_spaces(self):
        df = pr._parse_lines(
            "Web Content 1234 100.000000: 1 cycles: "
            "ffffffff81000000 foo (/usr/bin/app)\n"
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df["comm"].iloc[0], "Web Content")

    def test_custom_fields_select_columns(self):
        df = pr._parse_lines(
            "       perf 2633 425759.959502: 5519178 instructions: "
            "4d1642 [unknown] (/usr/bin/python3.11)\n",
            "comm,pid,time,period,event,ip",
        )
        self.assertEqual(
            list(df.columns), ["comm", "pid", "time", "period", "event", "ip"]
        )
        self.assertEqual(df["event"].iloc[0], "instructions")


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


class TestQuoteColumns(unittest.TestCase):
    def test_plain_identifier_untouched(self):
        df = pd.DataFrame({"cycles": [1]})
        self.assertEqual(pr.quote_columns(df, "cycles > 0"), "cycles > 0")

    def test_dotted_column_quoted(self):
        df = pd.DataFrame({"data.rdi": [[15]]})
        self.assertEqual(pr.quote_columns(df, "data.rdi == 1"), "`data.rdi` == 1")

    def test_already_quoted_not_double_quoted(self):
        df = pd.DataFrame({"data.rdi": [[15]]})
        self.assertEqual(pr.quote_columns(df, "`data.rdi` == 1"), "`data.rdi` == 1")

    def test_unknown_name_untouched(self):
        df = pd.DataFrame({"cycles": [1]})
        self.assertEqual(pr.quote_columns(df, "nope > 0"), "nope > 0")

    def test_hyphenated_column_next_to_operator(self):
        df = pd.DataFrame({"operations": [1], "context-switches": [2]})
        self.assertEqual(
            pr.quote_columns(df, "context-switches/operations"),
            "`context-switches`/operations",
        )

    def test_string_literal_is_untouched(self):
        df = pd.DataFrame({"data.rdi": [[1]], "sym": ["see data.rdi"]})
        self.assertEqual(
            pr.quote_columns(df, 'sym == "see data.rdi"'),
            'sym == "see data.rdi"',
        )


class TestEnvelopeHelpers(unittest.TestCase):
    def test_envelope_columns_data_buckets(self):
        self.assertEqual(
            pr.envelope_columns({"data": {"regs": {"rdi": 1}, "x": 2}}),
            {"data.rdi": 1, "data.x": 2},
        )

    def test_envelope_frame_shared_only(self):
        df = pr.envelope_frame({"output": [], "shared": {"mode": "latency"}})
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["mode"], "latency")

    def test_envelope_frame_empty(self):
        self.assertTrue(pr.envelope_frame({"output": [], "shared": {}}).empty)

    def test_nest_all_dots_rest(self):
        self.assertEqual(pr.nest({"config...": 1}), {"config...": 1})

    def test_is_record_str_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rec"
            p.write_bytes(b"PERFILE1rest")
            self.assertTrue(pr.is_record(str(p)))
            self.assertTrue(pr.is_record(p))

    def test_metrics_empty_events(self):
        import pandas as pd

        df = pd.DataFrame({"event": ["cycles"], "period": [1]})
        self.assertTrue(pr.metrics(df, []).equals(df))
        self.assertTrue(pr.metrics(df, ["nope"]).equals(df))


class TestIsRecordEdges(unittest.TestCase):
    def test_is_record_rejects_none_and_missing(self):
        self.assertFalse(pr.is_record(None))
        self.assertFalse(pr.is_record(12345))
        self.assertFalse(pr.is_record("/nonexistent/path/perf.data"))


class TestParseFileColumn(unittest.TestCase):
    def test_parse_does_not_fail_when_file_column_present(self):
        from unittest.mock import patch

        base = pr._parse_lines(
            "  proc 1 1.0: 1 cycles: 1234 sym (dso)\n"
            "  proc 1 1.0: 2 instructions: 1234 sym (dso)\n"
        )
        with patch("perf.data.samples", return_value=base):
            df = pr.parse(["a", "b"])
        self.assertIn("file", df.columns)
        self.assertEqual(set(df["file"].unique()), {"a", "b"})


class TestEnvelopeFrameBranches(unittest.TestCase):
    def test_output_rows_are_flattened_with_shared_columns(self):
        payload = {
            "output": [
                {"mode": "latency", "cycles": 1, "data": {"regs": {"rdi": 5}}},
                {"mode": "latency", "cycles": 2, "data": {"mem": {"0x10": 3}}},
            ],
            "shared": {"id": "x", "mode": "shared"},
        }
        df = pr.envelope_frame(payload)
        self.assertEqual(df["cycles"].tolist(), [1, 2])
        self.assertEqual(df["data.rdi"].tolist()[0], 5)
        self.assertEqual(df["data.0x10"].tolist()[1], 3)
        self.assertEqual(df["id"].tolist(), ["x", "x"])
        self.assertEqual(df["mode"].tolist(), ["latency", "latency"])

    def test_missing_shared_columns_are_appended(self):
        payload = {
            "output": [{"cycles": 1}, {"cycles": 2}],
            "shared": {
                "scalar": 7,
                "values": [1, 2],
                "mapping": {0: 10, 1: 20},
                "series": pd.Series([4, 5]),
            },
        }
        df = pr.envelope_frame(payload)
        self.assertEqual(df["scalar"].tolist(), [7, 7])
        self.assertEqual(df["values"].tolist(), [1, 2])
        self.assertEqual(df["mapping.0"].tolist(), [10, 10])
        self.assertEqual(df["mapping.1"].tolist(), [20, 20])
        self.assertEqual(df["series"].tolist(), [4, 5])

    def test_missing_shared_columns_manual_fallback(self):
        payload = {
            "output": [{"cycles": 1}, {"cycles": 2}],
            "shared": {
                "scalar": 7,
                "values": [1, 2],
                "wrong_length": [1, 2, 3],
                "mapping": {0: 10, 1: 20},
                "series": pd.Series([4, 5]),
            },
        }
        df = pr.envelope_frame(payload)
        self.assertEqual(df["scalar"].tolist(), [7, 7])
        self.assertEqual(df["values"].tolist(), [1, 2])
        self.assertEqual(df["mapping.0"].tolist(), [10, 10])
        self.assertEqual(df["mapping.1"].tolist(), [20, 20])
        self.assertEqual(df["series"].tolist(), [4, 5])
        self.assertNotIn("wrong_length", df.columns)

    def test_output_frame_construction_fallback(self):
        real_frame = pd.DataFrame

        def fake_frame(*args, **kwargs):
            if args and isinstance(args[0], list):
                raise RuntimeError("cannot build")
            return real_frame(*args, **kwargs)

        rows = [{"a": 1}, {"a": 2, "b": 3}, {"c": 4}]
        with patch("pandas.DataFrame", side_effect=fake_frame):
            df = pr.envelope_frame({"output": rows, "shared": {}})
        self.assertEqual(df["a"].iloc[:2].tolist(), [1.0, 2.0])
        self.assertTrue(pd.isna(df["a"].iloc[2]))
        self.assertEqual(df["b"].iloc[1], 3.0)
        self.assertTrue(pd.isna(df["b"].iloc[0]))
        self.assertEqual(df["c"].iloc[2], 4.0)
        self.assertTrue(pd.isna(df["c"].iloc[0]))

    def test_envelope_output_without_rows_uses_shared(self):
        df = pr.envelope_frame({"output": [], "shared": {"config": {"a": 1}}})
        self.assertEqual(df["config.a"].tolist(), [1])


class TestDataParsingBranches(unittest.TestCase):
    def setUp(self):
        pr.system_perf.cache_clear()

    def tearDown(self):
        pr.system_perf.cache_clear()

    def test_system_perf_falls_back_to_standard_paths(self):
        with (
            patch("os.get_exec_path", return_value=[]),
            patch("os.path.isfile", side_effect=lambda p: p == "/usr/bin/perf"),
            patch("os.access", return_value=True),
        ):
            self.assertEqual(pr.system_perf(), "/usr/bin/perf")

    def test_system_perf_returns_none_when_missing(self):
        with (
            patch("os.get_exec_path", return_value=[]),
            patch("os.path.isfile", return_value=False),
        ):
            self.assertIsNone(pr.system_perf())

    def test_system_perf_skips_non_executable_path_entries(self):
        with (
            patch("os.get_exec_path", return_value=["/bin"]),
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=False),
        ):
            self.assertIsNone(pr.system_perf())

    def test_samples_without_perf_binary_raises(self):
        with patch.object(pr, "system_perf", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                pr.samples("perf.data")
        self.assertIn("system perf not found", str(ctx.exception))

    @patch("perf.data.subprocess.run")
    def test_samples_accepts_string_fields(self, run):
        run.return_value = Mock(returncode=0, stdout=_SAMPLES, stderr="")
        pr.samples("perf.data", bin="/usr/bin/perf", fields="comm,pid")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[-1], "comm,pid")

    @patch("perf.data.subprocess.run")
    def test_samples_failure_without_stderr_uses_stdout(self, run):
        run.return_value = Mock(returncode=1, stdout="stdout detail", stderr="")
        with self.assertRaises(RuntimeError) as ctx:
            pr.samples("perf.data", bin="/usr/bin/perf")
        self.assertIn("stdout detail", str(ctx.exception))

    def test_metrics_without_key_columns_returns_input(self):
        df = pd.DataFrame({"event": ["cycles", "cycles"], "period": [1, 2]})
        self.assertIs(pr.metrics(df), df)

    @patch("perf.data.samples")
    def test_parse_without_rows_returns_schema(self, samples):
        samples.return_value = pd.DataFrame()
        df = pr.parse(["a.data"])
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), ["file"] + pr._FIELDS.split(","))

    @patch("perf.data.samples")
    def test_parse_keeps_file_column_first(self, samples):
        frame = pd.DataFrame({"cycles": [1], "file": ["existing"]})
        samples.return_value = frame
        df = pr.parse(["new.data"])
        self.assertEqual(df["file"].tolist(), ["new.data"])
        self.assertEqual(df.columns[0], "file")

    def test_flatten_into_respects_data_buckets(self):
        out = {}
        pr._flatten_into(
            "data",
            {
                "regs": {"rdi": 1},
                "mem": {"0x10": 2},
                "plain": {"x": 3},
                "scalar": 4,
            },
            out,
            buckets=True,
        )
        self.assertEqual(
            out,
            {"data.rdi": 1, "data.0x10": 2, "data.plain.x": 3, "data.scalar": 4},
        )

    def test_scalar_conversions(self):
        self.assertEqual(pr._scalar("'abc'"), "abc")
        self.assertEqual(pr._scalar('"def"'), "def")
        self.assertEqual(pr._scalar("0x10"), 16)
        self.assertEqual(pr._scalar("12"), 12)
        self.assertEqual(pr._scalar("1.5"), 1.5)
        self.assertEqual(pr._scalar("plain"), "plain")

    def test_members_plain_and_container_columns(self):
        plain = pd.Series([1, 2, 3])
        self.assertEqual(pr._members(plain, "2").tolist(), [False, True, False])
        containers = pd.Series([[1, 2], [3], "x"])
        self.assertEqual(pr._members(containers, "1").tolist(), [True, False, False])
        self.assertEqual(
            pr._members(pd.Series(["None", 1]), "None").tolist(), [True, False]
        )

    def test_members_falls_back_when_isin_fails(self):
        frame = pd.Series([1, 2, 3])
        with patch.object(pd.Series, "isin", side_effect=RuntimeError("boom")):
            result = pr._members(frame, "2")
        self.assertEqual(result.tolist(), [False, True, False])

    def test_members_falls_back_when_mapping_fails(self):
        frame = pd.Series([1, 2, 3])
        with patch.object(pd.Series, "map", side_effect=RuntimeError("boom")):
            result = pr._members(frame, "2")
        self.assertEqual(result.tolist(), [False, True, False])

    def test_memberships_unknown_column_is_left_alone(self):
        df = pd.DataFrame({"cycles": [1, 2]})
        expr, local = pr._memberships(df, "3 in missing and cycles > 0")
        self.assertEqual(expr, "3 in missing and cycles > 0")
        self.assertEqual(local, {})

    def test_memberships_ignores_string_literals(self):
        df = pd.DataFrame({"sym": ["1 in cycles", "x"]})
        expr, local = pr._memberships(df, 'sym == "1 in cycles"')
        self.assertEqual(expr, 'sym == "1 in cycles"')
        self.assertEqual(local, {})

    def test_quote_columns_converts_non_string_expression(self):
        df = pd.DataFrame({"cycles": [1]})
        self.assertEqual(pr.quote_columns(df, 42), "42")

    def test_quote_columns_handles_bad_frame(self):
        class BadFrame:
            @property
            def columns(self):
                raise RuntimeError("boom")

        self.assertEqual(pr.quote_columns(BadFrame(), "cycles > 0"), "cycles > 0")

    def test_quote_columns_single_and_backtick_literals(self):
        df = pd.DataFrame({"data.rdi": [[1]], "sym": ["a data.rdi b"]})
        self.assertEqual(
            pr.quote_columns(df, "sym == 'a data.rdi b'"),
            "sym == 'a data.rdi b'",
        )
        self.assertEqual(pr.quote_columns(df, "`data.rdi` == [1]"), "`data.rdi` == [1]")

    def test_quote_columns_escapes_regex_characters(self):
        df = pd.DataFrame({"a+b": [1]})
        self.assertEqual(pr.quote_columns(df, "a+b > 0"), "`a+b` > 0")

    def test_quote_columns_ignores_non_string_column_names(self):
        df = pd.DataFrame({1: [1], "cycles": [2]})
        self.assertEqual(pr.quote_columns(df, "cycles > 0"), "cycles > 0")

    def test_norm_event_modifiers_and_nested_paths(self):
        self.assertEqual(pr._norm_event("cpu_core/cycles/u"), "cycles")
        self.assertEqual(pr._norm_event("a/b/c"), "b")
        self.assertEqual(pr._norm_event("cycles:zzzz"), "cycles:zzzz")
        self.assertEqual(pr._norm_event("sched:sched_switch"), "sched:sched_switch")

    def test_parse_lines_custom_field_suffixes(self):
        sym = pr._parse_lines(
            "1 2 3.0 4 cycles ff10 foo bar (/lib/x.so)\n",
            ["pid", "cpu", "time", "period", "event", "ip", "sym"],
        )
        self.assertEqual(sym["sym"].iloc[0], "foo bar (/lib/x.so)")

        dso = pr._parse_lines("1 2 (/lib/x.so)\n", ["pid", "cpu", "dso"])
        self.assertEqual(dso["dso"].iloc[0], "/lib/x.so")

        comm = pr._parse_lines("1 2 Web Content\n", ["pid", "cpu", "comm"])
        self.assertEqual(comm["comm"].iloc[0], "Web Content")

        event = pr._parse_lines("1 2 3.0 cycles:\n", ["pid", "cpu", "time", "event"])
        self.assertEqual(event["event"].iloc[0], "cycles")

        time = pr._parse_lines("1 2 3.0\n", ["pid", "cpu", "time"])
        self.assertEqual(time["time"].iloc[0], "3.0")

        extra = pr._parse_lines("1 2 3 extra\n", ["pid", "cpu", "unknown"])
        self.assertEqual(extra["unknown"].iloc[0], "3")

    def test_parse_lines_skips_comments_and_malformed_rows(self):
        text = "# comment\n\n  not a sample\n1 2 3.0 cycles:\n"
        df = pr._parse_lines(text, ["pid", "cpu", "time", "event"])
        self.assertEqual(len(df), 1)
        self.assertEqual(df["event"].iloc[0], "cycles")

    def test_parse_lines_custom_fields_convert_numbers(self):
        df = pr._parse_lines("1 2 3.5 4\n", ["pid", "cpu", "time", "period"])
        self.assertEqual(df["pid"].iloc[0], 1)
        self.assertEqual(df["cpu"].iloc[0], 2)
        self.assertEqual(df["period"].iloc[0], 4)

    def test_spread_nested_prefixes_and_memory_bucket(self):
        record = {
            "data": {"memory": {"0x20": 1}, "regs": {"rdi": 2}, "plain": 3},
            "config": {"nested": {"value": 4}},
        }
        self.assertEqual(
            pr.spread(record),
            {
                "data.0x20": 1,
                "data.rdi": 2,
                "data.plain": 3,
                "config.nested.value": 4,
            },
        )

    @patch("perf.data.samples")
    def test_parse_concatenates_empty_and_nonempty_frames(self, samples):
        samples.side_effect = [
            pd.DataFrame(),
            pr._parse_lines(_SAMPLES),
        ]
        df = pr.parse(["empty.data", "perf.data"])
        self.assertEqual(set(df["file"].unique()), {"perf.data"})
        self.assertIn("cycles", df.columns)


if __name__ == "__main__":
    unittest.main()
