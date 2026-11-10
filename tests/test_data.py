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

    def test_custom_fields_deduplicate_names(self):
        df = pr._parse_lines("1 Web Content\n", "pid,comm,comm")
        self.assertEqual(list(df.columns), ["pid", "comm"])
        self.assertEqual(df["comm"].iloc[0], "Web Content")


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

    def test_submodule_is_not_a_top_level_export(self):
        self.assertIs(perf.data, pr)
        self.assertIs(perf.data.samples, pr.samples)
        self.assertFalse(hasattr(perf, "samples"))
        self.assertNotIn("data", perf.__all__)
        self.assertNotIn("samples", perf.__all__)


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

    @patch("perf.data.samples")
    def test_parse_concatenates_empty_and_nonempty_frames(self, samples):
        samples.side_effect = [
            pd.DataFrame(),
            pr._parse_lines(_SAMPLES),
        ]
        df = pr.parse(["empty.data", "perf.data"])
        self.assertEqual(set(df["file"].unique()), {"perf.data"})
        self.assertIn("cycles", df.columns)


class TestEnvelopePaths(unittest.TestCase):
    def test_shared_columns_are_added_to_an_empty_frame(self):
        frame = pr.envelope_frame(
            {"output": [], "shared": {"mode": "latency", "data.rdi": 15}}
        )
        self.assertEqual(frame["mode"].tolist(), ["latency"])
        self.assertEqual(frame["data.rdi"].tolist(), [15])

    def test_shared_columns_are_added_to_rows(self):
        frame = pr.envelope_frame(
            {
                "output": [{"cycles": 1}, {"cycles": 2}],
                "shared": {"mode": "latency"},
            }
        )
        self.assertEqual(frame["mode"].tolist(), ["latency", "latency"])


class TestParseLinesPaths(unittest.TestCase):
    def test_bad_numeric_fields_are_skipped(self):
        text = "comm bad 1.0 2 cycles: 1000 sym (dso)\n"
        self.assertTrue(pr._parse_lines(text).empty)

    def test_dso_in_parentheses_is_split_from_the_symbol(self):
        text = "comm 1 1.0 2 cycles: 1000 sym with spaces (dso)\n"
        rows = pr._parse_lines(text)
        self.assertEqual(rows["sym"].iloc[0], "sym with spaces")
        self.assertEqual(rows["dso"].iloc[0], "dso")

    def test_symbol_without_dso_is_kept(self):
        text = "comm 1 1.0 2 cycles: 1000 sym\n"
        rows = pr._parse_lines(text)
        self.assertEqual(rows["sym"].iloc[0], "sym")
        self.assertEqual(rows["dso"].iloc[0], "")


class TestTableHelpers(unittest.TestCase):
    def test_duration_columns(self):
        self.assertTrue(pr.is_duration_col("duration_time"))
        self.assertTrue(pr.is_duration_col("duration_time/operations"))
        self.assertFalse(pr.is_duration_col("cycles"))

    def test_table_number(self):
        self.assertTrue(pd.isna(pr.table_number("  ")))
        self.assertEqual(pr.table_number("10"), 10.0)
        self.assertEqual(pr.table_number("1.50us", duration=True), 1500.0)
        self.assertEqual(pr.table_number("1.50", duration=True), 1.5)
        self.assertEqual(pr.table_number("hot"), "hot")

    def test_token_spans(self):
        self.assertEqual(pr.token_spans("  a  bb "), [(2, 3), (5, 7)])
        self.assertEqual(pr.token_spans("   "), [])


class TestTableFixedWidth(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame(
            {
                "file": ["", ""],
                "name": ["mov r11, rax", "mov r11, rax"],
                "mode": ["latency", "latency"],
                "duration_time": [1.5, 2250.0],
                "cycles": [10.0, 20.0],
            }
        )

    def test_round_trip(self):
        df = pr.table_to_df(self._frame().to_string(index=False))
        self.assertEqual(df["name"].tolist(), ["mov r11, rax"] * 2)
        self.assertEqual(df["duration_time"].tolist(), [1.5, 2250.0])
        self.assertEqual(df["cycles"].tolist(), [10.0, 20.0])
        self.assertTrue(df["file"].isna().all())

    def test_blank_rows_preserved(self):
        text = self._frame().to_string(index=False).splitlines()
        df = pr.table_to_df("\n".join(text[:1] + ["   " * 4, *text[1:]]))
        self.assertEqual(len(df), 3)
        self.assertTrue(df.iloc[0].isna().all())

    def test_left_aligned_layout(self):
        text = "a bb\n 1 34\n 2 56\n"
        df = pr.table_to_df(text)
        self.assertEqual(df.columns.tolist(), ["a", "bb"])
        self.assertEqual(df["a"].tolist(), [1.0, 2.0])
        self.assertEqual(df["bb"].tolist(), [34.0, 56.0])

    def test_large_tables_sample_the_layout(self):
        rows = [f"{i:6d} {i * 2:8.2f}" for i in range(600)]
        text = "count value\n" + "\n".join(rows) + "\n"
        df = pr.table_to_df(text)
        self.assertEqual(len(df), 600)
        self.assertEqual(df["count"].iloc[-1], 599.0)
        self.assertEqual(df["value"].iloc[-1], 1198.0)

    def test_repeated_text_is_kept(self):
        rows = ["fizz_buzz latency  10"] * 20
        text = "name      mode     cycles\n" + "\n".join(rows) + "\n"
        df = pr.table_to_df(text)
        self.assertEqual(df["name"].tolist(), ["fizz_buzz"] * 20)
        self.assertEqual(df["cycles"].tolist(), [10.0] * 20)

    def test_leading_and_trailing_blank_lines(self):
        df = pr.table_to_df("\n\nname  cycles\nfunc      10\n\n")
        self.assertEqual(df["name"].tolist(), ["func"])
        self.assertEqual(df["cycles"].tolist(), [10.0])

    def test_fixed_width_durations(self):
        df = pr.table_to_df("duration_time\n       1.50ns\n       2.00us\n")
        self.assertEqual(df["duration_time"].tolist(), [1.5, 2000.0])

    def test_text_column_with_blank(self):
        text = pd.DataFrame({"name": ["foo", "", "bar"]}).to_string(index=False)
        df = pr.table_to_df(text)
        self.assertEqual(df["name"].tolist()[0], "foo")
        self.assertTrue(pd.isna(df["name"].tolist()[1]))
        self.assertEqual(df["name"].tolist()[2], "bar")

    def test_empty_text_markers_become_nan(self):
        text = pd.DataFrame({"name": ["foo", "none", "null"]}).to_string(index=False)
        df = pr.table_to_df(text)
        self.assertTrue(pd.isna(df["name"].tolist()[1]))
        self.assertTrue(pd.isna(df["name"].tolist()[2]))

    def test_unexpected_text_in_numeric_column(self):
        rows = [f"{i:6d}" for i in range(514)]
        rows[512] = "  oops"
        text = "value\n" + "\n".join(rows) + "\n"
        df = pr.table_to_df(text)
        self.assertEqual(df["value"].tolist()[0], 0.0)
        self.assertEqual(df["value"].tolist()[512], "oops")

    def test_unexpected_text_in_duration_column(self):
        rows = [" 1.00ns"] * 514
        rows[512] = "  oops"
        text = "duration_time\n" + "\n".join(rows) + "\n"
        df = pr.table_to_df(text)
        self.assertEqual(df["duration_time"].tolist()[0], 1.0)
        self.assertEqual(df["duration_time"].tolist()[512], "oops")


class TestTableDelimited(unittest.TestCase):
    def test_csv(self):
        df = pr.table_to_df(
            "file,name,mode,operations,duration_time\n"
            "a.out,func,latency,1,3.00ns\n"
            "a.out,func,latency,1,1.50us\n"
        )
        self.assertEqual(df["duration_time"].tolist(), [3.0, 1500.0])
        self.assertEqual(df["operations"].tolist(), [1.0, 1.0])

    def test_tsv(self):
        df = pr.table_to_df("file\tname\tmode\na.out\tfunc\tlatency\n")
        self.assertEqual(df["name"].tolist(), ["func"])

    def test_mixed_numeric_column(self):
        df = pr.table_to_df("value,other\n1,a\nx,b\n,c\n2.5,d\n")
        self.assertEqual(df["value"].tolist()[0], 1.0)
        self.assertEqual(df["value"].tolist()[1], "x")
        self.assertTrue(pd.isna(df["value"].tolist()[2]))
        self.assertEqual(df["value"].tolist()[3], 2.5)

    def test_mixed_duration_column(self):
        df = pr.table_to_df("duration_time,other\n3.00ns,a\n1.50us,b\nplain,c\n")
        self.assertEqual(df["duration_time"].tolist(), [3.0, 1500.0, "plain"])

    def test_duration_column_with_plain_numbers(self):
        df = pr.table_to_df("duration_time,other\n3.00ns,a\n1.50,b\n")
        self.assertEqual(df["duration_time"].tolist(), [3.0, 1.5])

    def test_text_column_with_blank_from_csv(self):
        df = pr.table_to_df("name,other\nfoo,a\n,b\nbar,c\n")
        self.assertEqual(df["name"].tolist()[0], "foo")
        self.assertTrue(pd.isna(df["name"].tolist()[1]))
        self.assertEqual(df["name"].tolist()[2], "bar")

    def test_duration_conversion_of_non_strings(self):
        series = pd.Series([1.0, 2.0])
        out = pr._convert_column(series, True, "dur")
        self.assertEqual(out.tolist(), [1.0, 2.0])

    def test_blank_column_is_float(self):
        df = pr.table_to_df("a,b\n1,\n2,\n")
        self.assertTrue(pd.api.types.is_numeric_dtype(df["b"]))
        self.assertTrue(df["b"].isna().all())

    def test_engine_fallback(self):
        expected = pd.DataFrame({"a": [1.0], "b": [2.0]})

        def read_csv(_raw, **kwargs):
            if kwargs.get("engine") == "c":
                raise ValueError("no c engine")
            return expected

        with patch.object(pr.pd, "read_csv", side_effect=read_csv):
            df = pr.table_to_df("a,b\n1,2\n")
        self.assertEqual(df["a"].tolist(), [1.0])
        self.assertEqual(df["b"].tolist(), [2.0])

    def test_engine_failure_is_reported(self):
        with patch.object(pr.pd, "read_csv", side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                pr.table_to_df("a,b\n1,2\n")

    def test_large_csv_uses_a_sample(self):
        rows = [f"{i},{i * 2}" for i in range(600)]
        text = "a,b\n" + "\n".join(rows) + "\n"
        df = pr.table_to_df(text)
        self.assertEqual(len(df), 600)
        self.assertEqual(df["a"].tolist()[-1], 599.0)
        self.assertEqual(df["b"].tolist()[-1], 1198.0)

    def test_direct_empty_helper(self):
        self.assertTrue(pr._is_empty(None))
        self.assertTrue(pr._is_empty(" none "))
        self.assertFalse(pr._is_empty("hot"))
        self.assertFalse(pr._is_empty([]))


class TestTableInvalidInput(unittest.TestCase):
    def test_empty(self):
        self.assertTrue(pr.table_to_df("").empty)
        self.assertTrue(pr.table_to_df("   \n  \n").empty)

    def test_header_only(self):
        with self.assertRaises(ValueError):
            pr.table_to_df("file name mode")

    def test_garbage(self):
        with self.assertRaises(ValueError):
            pr.table_to_df("not a perf table at all")


if __name__ == "__main__":
    unittest.main()
