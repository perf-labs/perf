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

import importlib.machinery
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import perf
from perf.bench import bench

_pbench = importlib.import_module("perf.bench")
_FAST = {
    "iterations": 128,
    "samples": 2,
    "runs": 2,
    "branch": "unpredictable",
    "cache": "hot",
}


def _cli():
    path = Path(__file__).resolve().parent.parent / "bin" / "perf"
    loader = importlib.machinery.SourceFileLoader("perfcli_test_perf", str(path))
    spec = importlib.util.spec_from_loader("perfcli_test_perf", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


cli = _cli()


def _df():
    return pd.DataFrame(
        {
            "file": ["a", "a", "b", "b"],
            "name": ["f", "f", "g", "g"],
            "mode": ["latency"] * 4,
            "time": ["t0"] * 4,
            "samples": [0, 1, 0, 1],
            "duration_time": [10.0, 20.0, 30.0, 40.0],
            "cycles": [100.0, 200.0, 300.0, 400.0],
            "instructions": [50.0, 100.0, 150.0, 200.0],
        }
    )


def _env_df():
    df = pd.DataFrame(
        {
            "file": ["bin@abc"] * 2,
            "name": ["fizz"] * 2,
            "mode": ["latency"] * 2,
            "samples": [1, 2],
            "duration_time": [2.0, 3.0],
        }
    )
    df.attrs["config"] = {"samples": 100}
    df.attrs["data"] = {"regs": {"rdi": 5}, "mem": {}}
    df.attrs["info"] = {
        "cpu": {"arch": "x86_64"},
        "binary": {"path": "bin", "sha": "abc"},
    }
    return df


def _table_df():
    return pd.DataFrame(
        {
            "file": ["", ""],
            "name": ["mov r11, rax", "mov r11, rax"],
            "mode": ["latency", "latency"],
            "operations": [1, 1],
            "samples": [0, 1],
            "duration_time": [1.5, 2250.0],
            "cycles": [10.0, 20.0],
            "config.branch": ["unpredictable", "as-is"],
        }
    )


class TestModuleConstants(unittest.TestCase):
    def test_default_groupby(self):
        self.assertEqual(
            cli._DEFAULT_GROUPBY,
            ["file", "name", "mode"],
        )

    def test_default_columns(self):
        self.assertEqual(
            cli._DEFAULT_COLUMNS,
            ["time", "file", "name", "mode", "samples"],
        )

    def test_default_stats(self):
        self.assertEqual(
            cli._DEFAULT_STATS,
            ["min", "median", "p10", "p50", "p90", "p99", "max"],
        )

    def test_baseline_re_pattern(self):
        m = cli._BASELINE_RE.fullmatch("'group'.metric_name")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("key"), "group")
        self.assertEqual(m.group("metric"), "metric_name")

    def test_baseline_re_double_quotes(self):
        m = cli._BASELINE_RE.fullmatch('"group".metric')
        self.assertIsNotNone(m)
        self.assertEqual(m.group("key"), "group")

    def test_baseline_re_requires_dot(self):
        m = cli._BASELINE_RE.fullmatch("no-dot-here")
        self.assertIsNone(m)


class TestFormatDuration(unittest.TestCase):
    def test_ns(self):
        self.assertTrue(cli._format_duration(15.0).endswith("ns"))

    def test_us(self):
        self.assertTrue(cli._format_duration(15e3).endswith("us"))

    def test_ms(self):
        self.assertTrue(cli._format_duration(15e6).endswith("ms"))

    def test_seconds(self):
        self.assertTrue(cli._format_duration(5e9).endswith("s"))

    def test_nan_value(self):
        result = cli._format_duration(float("nan"))
        self.assertEqual(result, "")

    def test_non_numeric_returns_str(self):
        self.assertEqual(cli._format_duration("abc"), "abc")

    def test_none_returns_str(self):
        self.assertEqual(cli._format_duration(None), "")


class TestFormatNumber(unittest.TestCase):
    def test_rounds_to_two(self):
        self.assertEqual(cli._format_number(1.234), "1.23")

    def test_nan(self):
        result = cli._format_number(float("nan"))
        self.assertEqual(result, "")

    def test_non_numeric(self):
        self.assertEqual(cli._format_number("n/a"), "n/a")

    def test_integer(self):
        self.assertEqual(cli._format_number(42), "42.00")

    def test_none(self):
        self.assertEqual(cli._format_number(None), "")


class TestWrapperPath(unittest.TestCase):
    def test_returns_realpath(self):
        result = cli._wrapper_path()
        self.assertIsInstance(result, str)
        self.assertTrue(result.endswith("perf"))


class TestRewriteInnerPerf(unittest.TestCase):
    def test_passthrough_non_perf_commands(self):
        argv = ["record", "stat", "-e", "cycles"]
        result = cli._perf(argv)
        self.assertEqual(result, argv)

    def test_replaces_perf_after_dashes(self):
        wrapper = cli._wrapper_path()
        result = cli._perf(["--", "perf", "record"])
        self.assertEqual(result[1], wrapper)

    def test_passthrough_with_dashes(self):
        argv = ["--", "ls", "-l"]
        result = cli._perf(argv)
        self.assertEqual(result, argv)

    def test_empty_list(self):
        self.assertEqual(cli._perf([]), [])

    def test_non_passthrough_commands(self):
        argv = ["bench", "asm", "nop"]
        result = cli._perf(argv)
        self.assertEqual(result, argv)

    def test_perf_token_in_passthrough(self):
        wrapper = cli._wrapper_path()
        argv = ["trace", "record", "--", "perf"]
        result = cli._perf(argv)
        self.assertEqual(result[3], wrapper)


class TestExecSystemPerf(unittest.TestCase):
    @patch.object(cli.perf, "bin", return_value=None)
    def test_raises_when_not_found(self, mock_find):
        with self.assertRaises(SystemExit):
            cli._exec_system_perf(["record"])


class TestSplitList(unittest.TestCase):
    def test_none_returns_empty(self):
        from perf.core import _split_list

        self.assertEqual(_split_list(None), [])

    def test_empty_string(self):
        from perf.core import _split_list

        self.assertEqual(_split_list(""), [])

    def test_comma_separated(self):
        from perf.core import _split_list

        self.assertEqual(
            _split_list("a,b,c"),
            ["a", "b", "c"],
        )

    def test_single_string(self):
        from perf.core import _split_list

        self.assertEqual(_split_list(["x"]), ["x"])

    def test_list_of_strings(self):
        from perf.core import _split_list

        self.assertEqual(
            _split_list(["a,b", "c"]),
            ["a", "b", "c"],
        )

    def test_none_elements_skipped(self):
        from perf.core import _split_list

        self.assertEqual(
            _split_list([None, "a"]),
            ["a"],
        )

    def test_empty_list(self):
        from perf.core import _split_list

        self.assertEqual(_split_list([]), [])


class TestGroups(unittest.TestCase):
    def _groups(self, value):
        from perf.core import _split_groups

        return _split_groups(value, ["duration_time"])

    def test_none_returns_default(self):
        self.assertEqual(self._groups(None), [["duration_time"]])

    def test_empty_string(self):
        self.assertEqual(self._groups(""), [["duration_time"]])

    def test_single_string(self):
        self.assertEqual(
            self._groups("cycles"),
            [["cycles"]],
        )

    def test_comma_in_string(self):
        self.assertEqual(
            self._groups("cycles,instructions"),
            [["cycles", "instructions"]],
        )

    def test_list_of_strings(self):
        self.assertEqual(
            self._groups(["a,b", "c"]),
            [["a", "b"], ["c"]],
        )

    def test_list_of_lists(self):
        self.assertEqual(
            self._groups([["a", "b"], ["c"]]),
            [["a", "b"], ["c"]],
        )


class TestFlatEvents(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(cli._flat_events(None))

    def test_empty_string(self):
        self.assertIsNone(cli._flat_events(""))

    def test_splits(self):
        from perf.core import _split_list

        self.assertEqual(_split_list("a,b"), ["a", "b"])


class TestAsRecords(unittest.TestCase):
    def test_multiindex_reset(self):
        df = _df().set_index(["file", "name", "mode"])
        rec = cli._as_records(df)
        self.assertIn("file", rec.columns)

    def test_no_meta_index(self):
        df = pd.DataFrame({"cycles": [1, 2]})
        result = cli._as_records(df)
        self.assertEqual(list(result.columns), ["cycles"])


class TestMetricColumns(unittest.TestCase):
    def test_excludes_meta(self):
        df = _df()
        cols = cli._metric_columns(df)
        self.assertNotIn("file", cols)
        self.assertNotIn("name", cols)
        self.assertNotIn("mode", cols)
        self.assertNotIn("samples", cols)
        self.assertIn("cycles", cols)
        self.assertIn("duration_time", cols)

    def test_explicit_event_keeps_bookkeeping(self):
        df = _df()
        df["operations"] = [1] * 4
        df2, out = cli._eval_events(df, ["operations"])
        self.assertIn("operations", out)

    def test_excludes_data_columns(self):
        df = _df()
        df["data.rdi"] = [1] * 4
        df["config.foo"] = [2] * 4
        cols = cli._metric_columns(df)
        self.assertNotIn("data.rdi", cols)
        self.assertNotIn("config.foo", cols)

    def test_derived_metrics_present(self):
        df = _df()
        df["operations"] = [1, 2, 3, 4]
        self.assertEqual(
            set(cli._derived_metrics(df)),
            {
                "duration_time/operations",
                "cycles/operations",
                "instructions/operations",
                "instructions/cycles",
            },
        )

    def test_derived_metrics_partial(self):
        df = _df()
        self.assertEqual(cli._derived_metrics(df), ["instructions/cycles"])
        df2 = _df().drop(columns=["cycles"])
        df2["operations"] = [1, 2, 3, 4]
        self.assertEqual(
            cli._derived_metrics(df2),
            ["duration_time/operations", "instructions/operations"],
        )

    def test_derived_metrics_ignores_non_numeric(self):
        df = pd.DataFrame({"duration_time": ["x", "y"], "operations": ["a", "b"]})
        self.assertEqual(cli._derived_metrics(df), [])


class TestResolveOutput(unittest.TestCase):
    def test_measure_default_none(self):
        args = SimpleNamespace(output=None)
        out, to_stdout = cli._resolve_output(args)
        self.assertIsNone(out)
        self.assertFalse(to_stdout)

    def test_measure_dash(self):
        args = SimpleNamespace(output="-")
        out, to_stdout = cli._resolve_output(args)
        self.assertIsNone(out)
        self.assertFalse(to_stdout)

    def test_measure_file(self):
        args = SimpleNamespace(output="out.json")
        out, to_stdout = cli._resolve_output(args)
        self.assertEqual(out, "out.json")
        self.assertFalse(to_stdout)

    def test_asm_default_stdout(self):
        args = SimpleNamespace(output=None)
        out, to_stdout = cli._resolve_output(args, stage="S")
        self.assertIsNone(out)
        self.assertTrue(to_stdout)

    def test_asm_dash_stdout(self):
        args = SimpleNamespace(output="-")
        out, to_stdout = cli._resolve_output(args, stage="S")
        self.assertIsNone(out)
        self.assertTrue(to_stdout)

    def test_asm_file(self):
        args = SimpleNamespace(output="f.s")
        out, to_stdout = cli._resolve_output(args, stage="S")
        self.assertEqual(out, "f.s")
        self.assertFalse(to_stdout)

    def test_object_default_needs_file(self):
        args = SimpleNamespace(output=None)
        out, to_stdout = cli._resolve_output(args, stage="c")
        self.assertIsNone(out)
        self.assertFalse(to_stdout)

    def test_object_dash_raises(self):
        args = SimpleNamespace(output="-")
        with self.assertRaises(SystemExit):
            cli._resolve_output(args, stage="c")

    def test_object_file(self):
        args = SimpleNamespace(output="a.o")
        out, to_stdout = cli._resolve_output(args, stage="c")
        self.assertEqual(out, "a.o")
        self.assertFalse(to_stdout)

    def test_multi_output_rejected(self):
        args = SimpleNamespace(output=["a", "b"])
        with self.assertRaises(SystemExit):
            cli._resolve_output(args)

    def test_multi_output_single_unwrap(self):
        args = SimpleNamespace(output=["x"])
        out, to_stdout = cli._resolve_output(args)
        self.assertEqual(out, "x")


class TestBenchFileOutput(unittest.TestCase):
    def test_none_returns_none(self):
        args = SimpleNamespace(output=None)
        self.assertIsNone(cli._bench_file_output(args))

    def test_dash_returns_none(self):
        args = SimpleNamespace(output="-")
        self.assertIsNone(cli._bench_file_output(args))

    def test_string_passes_through(self):
        args = SimpleNamespace(output="out")
        self.assertEqual(cli._bench_file_output(args), "out")

    def test_single_element_list_unwraps(self):
        args = SimpleNamespace(output=["out"])
        self.assertEqual(cli._bench_file_output(args), "out")

    def test_multi_output_rejected(self):
        args = SimpleNamespace(output=["a", "b"])
        with self.assertRaises(SystemExit):
            cli._bench_file_output(args)


class TestWantJson(unittest.TestCase):
    def test_explicit_json_flag(self):
        args = SimpleNamespace(json=True)
        self.assertTrue(cli._want_json(args))

    def test_no_flag_means_table(self):
        args = SimpleNamespace(json=False)
        self.assertFalse(cli._want_json(args))


class TestEnvelope(unittest.TestCase):
    def test_structure_keys(self):
        env = cli._record_envelope(_env_df())
        self.assertEqual(
            list(env),
            [
                "file",
                "name",
                "id",
                "time",
                "info",
                "config",
                "mode",
                "data",
                "code",
                "state",
                "distribution",
                "output",
            ],
        )

    def test_file(self):
        env = cli._record_envelope(_env_df())
        self.assertEqual(env["file"], "bin@abc")

    def test_name_includes_id(self):
        from perf.bench import _id_hash

        df = _env_df()
        env = cli._record_envelope(df)
        binary = df.attrs["info"]["binary"]
        self.assertEqual(
            env["id"], _id_hash(df.attrs["data"], df.attrs["config"], binary)
        )
        self.assertEqual(env["name"], f"fizz-{env['id']}")

    def test_id_covers_binary(self):
        from perf.bench import _id_hash

        df = _env_df()
        env = cli._record_envelope(df)
        other = dict(df.attrs["info"]["binary"])
        other["sha"] = "different"
        self.assertNotEqual(
            env["id"], _id_hash(df.attrs["data"], df.attrs["config"], other)
        )

    def test_mode(self):
        env = cli._record_envelope(_env_df())
        self.assertEqual(env["mode"], "latency")

    def test_data(self):
        env = cli._record_envelope(_env_df())
        self.assertEqual(
            env["data"],
            {"regs": {"rdi": 5}, "mem": {}},
        )

    def test_info_has_cpu(self):
        env = cli._record_envelope(_env_df())
        self.assertIn("cpu", env["info"])

    def test_config(self):
        env = cli._record_envelope(_env_df())
        self.assertEqual(env["config"], {"samples": 100})

    def test_code_state_distribution_none(self):
        env = cli._record_envelope(_env_df())
        self.assertIsNone(env["code"])
        self.assertIsNone(env["state"])
        self.assertIsNone(env["distribution"])

    def test_output_records(self):
        env = cli._record_envelope(_env_df())
        self.assertEqual(len(env["output"]), 2)
        self.assertIn("duration_time", env["output"][0])

    def test_time_format(self):
        env = cli._record_envelope(_env_df())
        self.assertTrue(re.match(r"\d{4}-\d{2}-\d{2}", env["time"]))

    def test_no_id(self):
        df = _df()
        env = cli._record_envelope(df)
        self.assertEqual(env["name"], "f")

    def test_missing_file_label_not_baked_as_nan(self):
        import numpy as np

        df = pd.DataFrame(
            {
                "file": [np.nan, np.nan],
                "name": ["imul eax, 0", "imul eax, 0"],
                "mode": ["latency", "latency"],
                "samples": [0, 1],
                "cycles": [10.0, 20.0],
            }
        )
        df.attrs["config"] = {"samples": 2}
        df.attrs["data"] = {"regs": {}, "mem": {}}
        df.attrs["info"] = {"cpu": {"arch": "x86_64"}}
        env = cli._record_envelope(df)
        self.assertEqual(env["file"], "")
        self.assertTrue(env["name"].startswith("imul eax, 0-"))
        self.assertNotIn("nan", env["file"])
        self.assertNotIn("nan", env["name"])

    def test_missing_name_stays_empty(self):
        import numpy as np

        df = pd.DataFrame(
            {
                "file": [np.nan, np.nan],
                "name": [np.nan, np.nan],
                "mode": ["latency", "latency"],
                "samples": [0, 1],
                "cycles": [10.0, 20.0],
            }
        )
        df.attrs["config"] = {"samples": 2}
        df.attrs["data"] = {"regs": {}, "mem": {}}
        df.attrs["info"] = {"cpu": {"arch": "x86_64"}}
        env = cli._record_envelope(df)
        self.assertEqual(env["file"], "")
        self.assertEqual(env["name"], "")

    def test_text_or_empty(self):
        import numpy as np

        from perf.core import _text

        self.assertEqual(_text(np.nan), "")
        self.assertEqual(_text(None), "")
        self.assertEqual(_text("nan"), "")
        self.assertEqual(_text("None"), "")
        self.assertEqual(_text(""), "")
        self.assertEqual(_text("a.out@abc"), "a.out@abc")

    def test_wildcard_keeps_names(self):
        df = pd.DataFrame(
            {
                "file": ["x", "y"],
                "name": ["a", "b"],
                "mode": ["lat", "lat"],
                "samples": [0, 0],
                "duration_time": [1.0, 2.0],
            }
        )
        env = cli._record_envelope(df)
        self.assertIn("name", env["output"][0])


class TestOutDirname(unittest.TestCase):
    def test_basic(self):
        args = SimpleNamespace(file=None)
        d = cli._out_dirname(args, "name", "file")
        self.assertEqual(str(d), "file/name")

    def test_no_file_label(self):
        args = SimpleNamespace(file=None)
        d = cli._out_dirname(args, "name", "")
        self.assertEqual(str(d), "name/name")


class TestRequireModeArg(unittest.TestCase):
    def test_missing_raises(self):
        args = SimpleNamespace(mode=None)
        with self.assertRaises(SystemExit):
            cli._require_mode_arg(args)

    def test_present(self):
        args = SimpleNamespace(mode="latency")
        self.assertEqual(cli._require_mode_arg(args), ["latency"])

    def test_default_both(self):
        args = SimpleNamespace(mode=["latency", "throughput"])
        self.assertEqual(cli._require_mode_arg(args), ["latency", "throughput"])

    def test_comma_flattened(self):
        args = SimpleNamespace(mode=["latency,throughput"])
        self.assertEqual(cli._require_mode_arg(args), ["latency", "throughput"])

    def test_unknown_raises(self):
        args = SimpleNamespace(mode="bogus")
        with self.assertRaises(SystemExit):
            cli._require_mode_arg(args)


class TestStatFunc(unittest.TestCase):
    def test_median_passthrough(self):
        name, func = cli._stat_func("median")
        self.assertEqual(name, "median")
        self.assertEqual(func, "median")

    def test_min_passthrough(self):
        name, func = cli._stat_func("min")
        self.assertEqual(name, "min")

    def test_p50(self):
        name, func = cli._stat_func("p50")
        self.assertEqual(name, "p50")
        self.assertTrue(callable(func))
        s = pd.Series([1, 2, 3, 4, 5])
        self.assertEqual(func(s), 3.0)

    def test_p99(self):
        name, func = cli._stat_func("p99")
        self.assertEqual(name, "p99")
        s = pd.Series(list(range(100)))
        self.assertAlmostEqual(func(s), 98.01, places=1)

    def test_p0(self):
        name, func = cli._stat_func("p0")
        self.assertEqual(name, "p0")
        s = pd.Series([10, 20])
        self.assertEqual(func(s), 10.0)

    def test_p100(self):
        name, func = cli._stat_func("p100")
        s = pd.Series([10, 20])
        self.assertEqual(func(s), 20.0)

    def test_uppercase_p(self):
        name, func = cli._stat_func("P50")
        self.assertEqual(name, "p50")
        s = pd.Series([1, 2, 3])
        self.assertEqual(func(s), 2.0)

    def test_decimal_percentile(self):
        name, func = cli._stat_func("p33.3")
        self.assertEqual(name, "p33.3")
        self.assertTrue(callable(func))

    def test_invalid_over_100(self):
        with self.assertRaises(SystemExit):
            cli._stat_func("p101")


class TestAggregate(unittest.TestCase):
    def test_groupby_and_stat(self):
        df = _df()
        out = cli._aggregate(
            df,
            ["file", "name", "mode"],
            ["cycles"],
            ["min", "max"],
        )
        self.assertIn("stat", out.columns)
        self.assertEqual(set(out["stat"]), {"min", "max"})

    def test_missing_event_raises(self):
        df = _df()
        with self.assertRaises(SystemExit):
            cli._aggregate(
                df,
                ["file"],
                ["nonexistent"],
                ["min"],
            )

    def test_no_groupby_raises(self):
        df = pd.DataFrame({"cycles": [1, 2]})
        with self.assertRaises(SystemExit):
            cli._aggregate(df, [], ["cycles"], ["min"])

    def test_percentile_stat(self):
        df = _df()
        out = cli._aggregate(df, ["file", "name", "mode"], ["cycles"], ["p50"])
        self.assertEqual(set(out["stat"]), {"p50"})


class TestFormatTable(unittest.TestCase):
    def test_duration_time_formatted(self):
        df = pd.DataFrame(
            {
                "duration_time": [3.0],
            }
        )
        result = cli._format_table(df)
        val = result["duration_time"].iloc[0]
        self.assertTrue(val.endswith("s") or val.endswith("ms") or val.endswith("us"))
        self.assertTrue(val.endswith("ns"))

    def test_numeric_columns_formatted(self):
        df = pd.DataFrame({"value": [1.23456, 7.89012]})
        result = cli._format_table(df)
        self.assertEqual(result["value"].iloc[0], "1.23")

    def test_meta_columns_not_formatted(self):
        df = pd.DataFrame({"stat": ["min"], "cycles": [100.0]})
        result = cli._format_table(df)
        self.assertEqual(result["stat"].iloc[0], "min")

    def test_iterations_operations_not_float_formatted(self):
        df = pd.DataFrame(
            {
                "iterations": [30905],
                "operations": [1],
                "duration_time": [3.0],
            }
        )
        result = cli._format_table(df)
        self.assertEqual(str(result["iterations"].iloc[0]), "30905")
        self.assertEqual(str(result["operations"].iloc[0]), "1")
        self.assertTrue(str(result["duration_time"].iloc[0]).endswith("ns"))


class TestFormatTraceTable(unittest.TestCase):
    def test_empty_df(self):
        df = pd.DataFrame()
        result = cli._format_trace_table(df)
        self.assertIsInstance(result, str)

    def test_none_df(self):
        result = cli._format_trace_table(None)
        self.assertEqual(result, "")

    def test_basic_table(self):
        df = pd.DataFrame({"name": ["a", "b"], "val": [1, 2]})
        result = cli._format_trace_table(df)
        self.assertIn("name", result)
        self.assertIn("val", result)

    def test_numeric_right_justified(self):
        df = pd.DataFrame({"n": [1, 22, 333]})
        result = cli._format_trace_table(df)
        lines = result.strip().split("\n")
        self.assertTrue(len(lines) == 4)

    def test_string_left_justified(self):
        df = pd.DataFrame({"s": ["hi", "hello"]})
        result = cli._format_trace_table(df)
        self.assertIn("hi", result)

    def test_newlines_sanitized_no_embedded_blank_lines(self):
        df = pd.DataFrame({"name": ["a\nb", "c\rd"], "val": [1, 2]})
        result = cli._format_trace_table(df)
        self.assertEqual(len(result.strip().split("\n")), 3)
        self.assertNotIn("a\nb", result)
        self.assertIn("a b", result)


class TestInteractiveBenchDf(unittest.TestCase):
    def _bench_df(self):
        df = pd.DataFrame(
            {
                "iterations": [128, 128],
                "samples": [0, 1],
                "operations": [1, 1],
                "data.rdi": [15, 15],
                "duration_time": [1.0, 2.0],
            }
        )
        df = df.set_index(
            pd.MultiIndex.from_arrays(
                [["f@1", "f@1"], ["n", "n"], ["latency", "latency"]],
                names=["file", "name", "mode"],
            )
        )
        df.attrs["config"] = {"samples": 2}
        df.attrs["data"] = {"regs": {"rdi": 15}, "mem": {}}
        return df

    def test_mode_is_a_column(self):
        rec = cli._interactive_bench_df(self._bench_df(), [["duration_time"]])
        self.assertIn("mode", rec.columns)
        self.assertIn("file", rec.columns)
        self.assertIn("name", rec.columns)
        self.assertEqual(rec["mode"].tolist(), ["latency", "latency"])

    def test_data_columns_separated_before_events(self):
        rec = cli._interactive_bench_df(self._bench_df(), [["duration_time"]])
        cols = list(rec.columns)
        self.assertIn("data.rdi", cols)
        self.assertIn("duration_time", cols)
        self.assertLess(cols.index("data.rdi"), cols.index("duration_time"))

    def test_attrs_preserved(self):
        rec = cli._interactive_bench_df(self._bench_df(), [["duration_time"]])
        self.assertEqual(rec.attrs["data"], {"regs": {"rdi": 15}, "mem": {}})
        self.assertEqual(rec.attrs["config"], {"samples": 2})


class TestIsTraceNumeric(unittest.TestCase):
    def test_numeric(self):
        df = pd.DataFrame({"n": [1, 2]})
        self.assertTrue(cli._is_trace_numeric(df, "n"))

    def test_bool_not_numeric(self):
        df = pd.DataFrame({"b": [True, False]})
        self.assertFalse(cli._is_trace_numeric(df, "b"))

    def test_string_not_numeric(self):
        df = pd.DataFrame({"s": ["a", "b"]})
        self.assertFalse(cli._is_trace_numeric(df, "s"))


class TestEvalEvents(unittest.TestCase):
    def test_plain_column(self):
        df = _df()
        result, events = cli._eval_events(df, ["cycles"])
        self.assertEqual(events, ["cycles"])
        self.assertIn("cycles", result.columns)

    def test_expression(self):
        df = _df()
        result, events = cli._eval_events(df, ["cycles/instructions"])
        self.assertEqual(events, ["cycles/instructions"])
        self.assertIn("cycles/instructions", result.columns)

    def test_auto_detect_metrics(self):
        df = _df()
        _, events = cli._eval_events(df, None)
        self.assertIn("cycles", events)
        self.assertIn("instructions", events)

    def test_auto_detect_adds_derived(self):
        df = _df()
        df["operations"] = [1, 2, 3, 4]
        df2, events = cli._eval_events(df, None)
        self.assertIn("duration_time/operations", events)
        self.assertIn("cycles/operations", events)
        self.assertIn("instructions/operations", events)
        self.assertIn("instructions/cycles", events)
        self.assertIn("duration_time", events)
        self.assertEqual(df2["instructions/cycles"].tolist(), [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(
            df2["cycles/operations"].tolist(), [100.0, 100.0, 100.0, 100.0]
        )
        self.assertEqual(
            df2["instructions/operations"].tolist(), [50.0, 50.0, 50.0, 50.0]
        )
        self.assertEqual(
            df2["duration_time/operations"].tolist(), [10.0, 10.0, 10.0, 10.0]
        )

    def test_unknown_column_raises(self):
        df = _df()
        with self.assertRaises(SystemExit):
            cli._eval_events(df, ["nonexistent"])

    def test_no_metrics_raises(self):
        df = pd.DataFrame({"file": ["x"], "name": ["y"]})
        with self.assertRaises(SystemExit):
            cli._eval_events(df, ["nonexistent"])


class TestApplyFilter(unittest.TestCase):
    def test_none_filter(self):
        df = _df()
        result = cli._apply_filter(df, None)
        self.assertEqual(len(result), len(df))

    def test_empty_filter(self):
        df = _df()
        result = cli._apply_filter(df, "")
        self.assertEqual(len(result), len(df))

    def test_valid_filter(self):
        df = _df()
        result = cli._apply_filter(df, 'name == "f"')
        self.assertEqual(len(result), 2)

    def test_invalid_filter_raises(self):
        df = _df()
        with self.assertRaises(SystemExit):
            cli._apply_filter(df, "this is not valid python")


class TestResolveGroupby(unittest.TestCase):
    def test_default(self):
        df = _df()
        result = cli._resolve_groupby(df, None)
        self.assertEqual(result, ["file", "name", "mode"])

    def test_custom(self):
        df = _df()
        result = cli._resolve_groupby(df, "file,mode")
        self.assertEqual(result, ["file", "mode"])

    def test_nonexistent_column_filtered(self):
        df = _df()
        result = cli._resolve_groupby(df, "file,nonexistent")
        self.assertEqual(result, ["file"])

    def test_empty_string(self):
        df = _df()
        result = cli._resolve_groupby(df, "")
        self.assertEqual(result, [])


class TestResolveColumns(unittest.TestCase):
    def test_default(self):
        df = _df()
        result = cli._resolve_columns(df, None)
        self.assertEqual(
            result,
            ["time", "file", "name", "mode", "samples"],
        )

    def test_custom(self):
        df = _df()
        result = cli._resolve_columns(df, "file,cycles")
        self.assertEqual(result, ["file", "cycles"])

    def test_empty_string(self):
        df = _df()
        result = cli._resolve_columns(df, "")
        self.assertEqual(result, [])


class TestSplitSeq(unittest.TestCase):
    def test_comma(self):
        self.assertEqual(cli._split_seq("a,b,c"), ["a", "b", "c"])

    def test_space(self):
        self.assertEqual(cli._split_seq("a b c"), ["a", "b", "c"])

    def test_mixed(self):
        self.assertEqual(cli._split_seq("a, b  c"), ["a", "b", "c"])

    def test_empty(self):
        self.assertEqual(cli._split_seq(""), [])

    def test_leading_trailing_space(self):
        self.assertEqual(cli._split_seq("  a, b  "), ["a", "b"])


class TestParseAtom(unittest.TestCase):
    def test_int(self):
        self.assertEqual(cli._parse_atom("42"), 42)

    def test_hex(self):
        self.assertEqual(cli._parse_atom("0x10"), 0x10)

    def test_float(self):
        self.assertAlmostEqual(cli._parse_atom("3.14"), 3.14)

    def test_true(self):
        self.assertIs(cli._parse_atom("true"), True)

    def test_false(self):
        self.assertIs(cli._parse_atom("false"), False)

    def test_string(self):
        self.assertEqual(cli._parse_atom("hello"), "hello")

    def test_dict_literal(self):
        result = cli._parse_atom("{a:1}")
        self.assertEqual(result, {"a": 1})

    def test_list_literal(self):
        result = cli._parse_atom("[1,2,3]")
        self.assertEqual(result, [1, 2, 3])

    def test_comma_string(self):
        result = cli._parse_atom("1,2,3")
        self.assertEqual(result, [1, 2, 3])


class TestParseValue(unittest.TestCase):
    def test_int(self):
        self.assertEqual(cli._parse_value("42"), 42)

    def test_hex(self):
        self.assertEqual(cli._parse_value("0xFF"), 0xFF)

    def test_float(self):
        self.assertAlmostEqual(cli._parse_value("1.5"), 1.5)

    def test_bool_true(self):
        self.assertIs(cli._parse_value("true"), True)

    def test_bool_false(self):
        self.assertIs(cli._parse_value("false"), False)

    def test_string(self):
        self.assertEqual(cli._parse_value("hello"), "hello")

    def test_dict_simple(self):
        result = cli._parse_value("{a:1}")
        self.assertEqual(result, {"a": 1})

    def test_dict_nested(self):
        result = cli._parse_value("{branch:predictable}")
        self.assertEqual(result, {"branch": "predictable"})

    def test_list(self):
        result = cli._parse_value("[1,2,3]")
        self.assertEqual(result, [1, 2, 3])

    def test_empty_dict(self):
        self.assertEqual(cli._parse_value("{}"), {})

    def test_comma_separated(self):
        result = cli._parse_value("a,b,c")
        self.assertEqual(result, ["a", "b", "c"])

    def test_single_colon_makes_dict(self):
        result = cli._parse_value("k:v")
        self.assertEqual(result, {"k": "v"})


class TestResolvePlotConfig(unittest.TestCase):
    def test_overrides(self):
        args = SimpleNamespace(config=None)
        config = cli._resolve_plot_config(
            args,
            [
                "--config.style=dark_background",
                "--config.axes.grid=False",
            ],
        )
        self.assertEqual(config["style"], "dark_background")
        self.assertFalse(config["axes"]["grid"])

    def test_rejects_unknown(self):
        args = SimpleNamespace(config=None)
        with self.assertRaises(SystemExit):
            cli._resolve_plot_config(args, ["--bogus=1"])

    def test_file_load(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"style": "dark_background"}, f)
            f.flush()
            args = SimpleNamespace(config=f.name)
            config = cli._resolve_plot_config(args, [])
        os.unlink(f.name)
        self.assertEqual(config["style"], "dark_background")

    def test_dict_value(self):
        args = SimpleNamespace(config=None)
        config = cli._resolve_plot_config(
            args,
            ["--config.figure.figsize=[12,7]"],
        )
        self.assertEqual(config["figure"]["figsize"], [12, 7])

    def test_empty_override(self):
        args = SimpleNamespace(config=None)
        config = cli._resolve_plot_config(args, [])
        self.assertEqual(config, {})


class TestLoad(unittest.TestCase):
    def test_json_envelope(self):
        env = cli._record_envelope(_env_df())
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "x.json").write_text(json.dumps(env))
            loaded = cli._load(SimpleNamespace(data=[tmp]))
        self.assertEqual(
            loaded["file"].tolist(),
            ["bin@abc"] * 2,
        )

    def test_no_data_raises(self):
        with self.assertRaises(SystemExit):
            cli._load(SimpleNamespace(data=None))

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                cli._load(SimpleNamespace(data=[tmp]))

    def test_plain_json_rejected(self):
        records = [
            {"cycles": 10, "name": "a"},
            {"cycles": 20, "name": "b"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "x.json").write_text(json.dumps(records))
            with self.assertRaises(SystemExit):
                cli._load(SimpleNamespace(data=[tmp]))

    def test_multiple_envelopes_concat(self):
        env1 = cli._record_envelope(_env_df())
        env2 = cli._record_envelope(_env_df())
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(json.dumps(env1))
            Path(tmp, "b.json").write_text(json.dumps(env2))
            loaded = cli._load(SimpleNamespace(data=[tmp]))
        self.assertEqual(len(loaded), 4)


class TestPrintBenchVerbose(unittest.TestCase):
    def _run(self):
        args = SimpleNamespace(
            json=False,
            output=None,
        )
        df = _df()
        df.attrs["info"] = {"cpu": {"arch": "x86_64"}}
        df.attrs["config"] = {"samples": 10}
        df.attrs["data"] = {"regs": {"rdi": 5}}
        df.attrs["code"] = ["mov rax, 1"]
        df.attrs["state"] = [1, 2]
        fake = io.StringIO()
        with patch("sys.stdout", fake):
            cli._output_or_default(args, df, [["duration_time"]])
        return fake.getvalue()

    def test_prints_info_and_config(self):
        text = self._run()
        self.assertIn("cycles", text)

    def test_no_state(self):
        args = SimpleNamespace(
            json=False,
            output=None,
        )
        df = _df()
        df.attrs["info"] = {}
        df.attrs["config"] = {}
        df.attrs["data"] = None
        df.attrs["code"] = None
        df.attrs["state"] = None
        fake = io.StringIO()
        with patch("sys.stdout", fake):
            cli._output_or_default(args, df, [["duration_time"]])
        self.assertIn("duration_time", fake.getvalue())


class TestPrintBenchDf(unittest.TestCase):
    def test_prints_table(self):
        df = _df()
        args = SimpleNamespace(
            json=False,
            output=None,
        )
        fake = io.StringIO()
        with patch("sys.stdout", fake):
            cli._output_or_default(args, df, [["duration_time"]])
        text = fake.getvalue()
        self.assertIn("cycles", text)


class TestOutputOrDefault(unittest.TestCase):
    def _run(self, json_flag, output):
        args = SimpleNamespace(
            json=json_flag,
            output=output,
        )
        fake = io.StringIO()
        env = _env_df()
        with patch("sys.stdout", fake):
            cli._output_or_default(args, env, [["duration_time"]])
        return fake.getvalue()

    def test_no_output_prints_table(self):
        text = self._run(False, None)
        self.assertIn("duration_time", text)
        self.assertNotIn('"output"', text)

    def test_output_saves_no_print(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(False, tmp)
            files = list(Path(tmp).rglob("*.json"))
            self.assertEqual(len(files), 1)
            data = json.loads(files[0].read_text())
            self.assertEqual(data["file"], "bin@abc")
            self.assertTrue(data["name"].startswith("fizz-"))
            self.assertEqual(data["name"], f"fizz-{data['id']}")

    def test_json_prints_and_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = self._run(True, tmp)
            data = json.loads(text)
            self.assertTrue(data["name"].startswith("fizz-"))
            self.assertEqual(data["name"], f"fizz-{data['id']}")
            files = list(Path(tmp).rglob("*.json"))
            self.assertEqual(len(files), 1)

    def test_piped_json_without_output(self):
        text = self._run(True, None)
        data = json.loads(text)
        self.assertIn("output", data)


class TestWriteEnvelope(unittest.TestCase):
    def test_stdout_only(self):
        env = _env_df()
        args = SimpleNamespace(file=None)
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cli._write_envelope(
                args,
                env,
                [["duration_time"]],
                file_output=None,
                to_stdout=True,
            )
        text = buf.getvalue()
        data = json.loads(text)
        self.assertIn("output", data)

    def test_file_only(self):
        env = _env_df()
        args = SimpleNamespace(file=None)
        with tempfile.TemporaryDirectory() as tmp:
            cli._write_envelope(
                args,
                env,
                [["duration_time"]],
                file_output=tmp,
                to_stdout=False,
            )
            files = list(Path(tmp).rglob("*.json"))
            self.assertEqual(len(files), 1)

    def test_both(self):
        env = _env_df()
        args = SimpleNamespace(file=None)
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sys.stdout", buf):
                cli._write_envelope(
                    args,
                    env,
                    [["duration_time"]],
                    file_output=tmp,
                    to_stdout=True,
                )
            files = list(Path(tmp).rglob("*.json"))
            self.assertEqual(len(files), 1)
        self.assertIn("output", buf.getvalue())


class TestSanitizeName(unittest.TestCase):
    def test_basic(self):
        result = cli._sanitize_name("hello_world")
        self.assertEqual(result, "hello_world")

    def test_special_chars(self):
        result = cli._sanitize_name("hello@world!")
        self.assertNotIn("@", result)
        self.assertNotIn("!", result)

    def test_none(self):
        result = cli._sanitize_name(None)
        self.assertEqual(result, "bench")


class TestBenchSubcommand(unittest.TestCase):
    def test_first_non_flag(self):
        self.assertEqual(
            cli._bench_subcommand(["bench", "asm", "nop"]),
            "bench",
        )

    def test_no_flags(self):
        self.assertIsNone(cli._bench_subcommand(["--verbose"]))

    def test_empty(self):
        self.assertIsNone(cli._bench_subcommand([]))


class TestFindBaselineRows(unittest.TestCase):
    def test_found_by_file(self):
        df = _df()
        result, col = cli._find_baseline_rows(df, "a")
        self.assertEqual(col, "file")
        self.assertEqual(len(result), 2)

    def test_found_by_name(self):
        df = _df()
        result, col = cli._find_baseline_rows(df, "g")
        self.assertEqual(col, "name")
        self.assertEqual(len(result), 2)

    def test_not_found(self):
        df = _df()
        result, col = cli._find_baseline_rows(df, "nonexistent")
        self.assertIsNone(result)
        self.assertIsNone(col)


class TestBaselineValues(unittest.TestCase):
    def test_scalar_baseline(self):
        df = _df().drop(columns=["samples"])
        result = cli._baseline_values(df, "a", "cycles")
        self.assertAlmostEqual(result, 150.0)

    def test_sample_aligned_baseline(self):
        df = _df()
        result = cli._baseline_values(df, "a", "cycles")
        self.assertEqual(len(result), 4)

    def test_unknown_group_raises(self):
        df = _df()
        with self.assertRaises(ValueError):
            cli._baseline_values(df, "nope", "cycles")

    def test_unknown_metric_raises(self):
        df = _df()
        with self.assertRaises(ValueError):
            cli._baseline_values(df, "a", "nonexistent")


class TestInjectBaselines(unittest.TestCase):
    def test_temp_columns_cleaned(self):
        df = _df()
        df["name"] = ["a", "a", "b", "b"]
        counter = [0]
        rewritten, temps = cli._inject_baselines(df, "cycles", counter)
        self.assertIsInstance(rewritten, str)
        for t in temps:
            self.assertIn(t, df.columns)

    def test_unknown_group_raises(self):
        df = _df()
        counter = [0]
        with self.assertRaises(ValueError):
            cli._inject_baselines(df, "'nope'.cycles", counter)


class TestDataOverridesParse(unittest.TestCase):
    @staticmethod
    def _bench_parser():
        import argparse

        p = argparse.ArgumentParser(prog="perf")
        sub = p.add_subparsers(dest="command")
        bench = sub.add_parser("bench")
        bench.add_argument("code", nargs="?", default=None)
        bench.add_argument("-f", "--filter", dest="target", default=None)
        bench.add_argument("--list", action="store_true", default=False)
        bench.add_argument("-n", "--name", default=None)
        bench.add_argument(
            "-m",
            "--mode",
            choices=["latency", "throughput"],
            default=None,
        )
        bench.add_argument("--config", default=None)
        bench.add_argument("--data", default=None)
        bench.add_argument(
            "-e",
            "--event",
            action="append",
            default=None,
        )
        bench.add_argument("-o", "--output", dest="output", default=None)
        bench.add_argument("--json", action="store_true", default=False)
        bench.add_argument(
            "-i",
            "--interactive",
            action="store_true",
        )
        bench.add_argument("-v", "--verbose", action="store_true")
        bench.add_argument("-S", dest="emit_asm", action="store_true")
        bench.add_argument("-c", dest="emit_obj", action="store_true")
        bench.add_argument("--backend", default=None)
        bench.add_argument("--setup", dest="setup_asm", default=None)
        bench.add_argument(
            "--teardown",
            dest="teardown_asm",
            default=None,
        )
        return p

    def _parse(self, *extra):
        argv = ["perf", "bench", "nop"]
        argv.extend(extra)
        parser = self._bench_parser()
        with patch.object(cli.sys, "argv", argv):
            return cli.parse(parser)

    def test_reg_by_name(self):
        args = self._parse("--data.rdi=15")
        self.assertEqual(args.data, {"regs": {"rdi": 15}, "mem": {}})

    def test_reg_by_arg_alias(self):
        args = self._parse("--data.arg0=15")
        self.assertEqual(args.data, {"regs": {"rdi": 15}, "mem": {}})

    def test_arg_alias_canonicalized(self):
        args = self._parse("--data.arg0=15", "--data.rdi=16")
        self.assertEqual(args.data, {"regs": {"rdi": 16}, "mem": {}})

    def test_namespaced_mem_form_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("--data.mem[0x1000]=1")

    def test_bracket_addr(self):
        args = self._parse("--data[0x1000]=42")
        self.assertEqual(
            args.data,
            {"regs": {}, "mem": {"0x1000": 42}},
        )

    def test_bracket_reg(self):
        args = self._parse("--data[rdi]=99")
        self.assertEqual(
            args.data,
            {"regs": {"rdi": 99}, "mem": {}},
        )

    def test_json_string(self):
        args = self._parse("--data", '{"regs": {"rdi": 15}}')
        self.assertEqual(
            args.data,
            {"regs": {"rdi": 15}, "mem": {}},
        )

    def test_unknown_dotted_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("--data.regs.rdi=15")

    def test_config_override(self):
        args = self._parse("--config.branch=predictable")
        self.assertEqual(args.config["branch"], "predictable")

    def test_config_branch_uniform_alias_accepted(self):
        args = self._parse("--config.branch=unpredictable.uniform")
        self.assertEqual(args.config["branch"], "unpredictable.uniform")
        from perf.bench import _branch_config

        cfg = _branch_config({"branch": args.config["branch"]})
        self.assertEqual(cfg["prediction"], "unpredictable")

    def test_config_branch_invalid_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("--config.branch=uniform")

    def test_config_address_override(self):
        args = self._parse("--config.branch[0x401234]=predictable")
        self.assertEqual(
            args.config["branch"]["0x401234"],
            "predictable",
        )

    def test_config_cache_rate_string(self):
        args = self._parse("--config.cache.L1d=hit_rate:100")
        L1d = args.config["cache"]["L1d"]
        self.assertEqual(L1d["hit_rate"], 100)


class TestConfigFile(unittest.TestCase):
    @staticmethod
    def _bench_parser():
        import argparse

        p = argparse.ArgumentParser(prog="perf")
        sub = p.add_subparsers(dest="command")
        bench = sub.add_parser("bench")
        bench.add_argument("code", nargs="?", default=None)
        bench.add_argument("-f", "--filter", dest="target", default=None)
        bench.add_argument("--list", action="store_true", default=False)
        bench.add_argument("-n", "--name", default=None)
        bench.add_argument(
            "-m",
            "--mode",
            choices=["latency", "throughput"],
            default=None,
        )
        bench.add_argument("--config", default=None)
        bench.add_argument("--data", default=None)
        bench.add_argument(
            "-e",
            "--event",
            action="append",
            default=None,
        )
        bench.add_argument("-o", "--output", dest="output", default=None)
        bench.add_argument("--json", action="store_true", default=False)
        bench.add_argument(
            "-i",
            "--interactive",
            action="store_true",
        )
        bench.add_argument("-v", "--verbose", action="store_true")
        bench.add_argument("-S", dest="emit_asm", action="store_true")
        bench.add_argument("-c", dest="emit_obj", action="store_true")
        bench.add_argument("--backend", default=None)
        bench.add_argument("--setup", dest="setup_asm", default=None)
        bench.add_argument(
            "--teardown",
            dest="teardown_asm",
            default=None,
        )
        return p

    def _parse_config(self, config_dict, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp, "config.json")
            cfg_path.write_text(json.dumps(config_dict))
            argv = [
                "perf",
                "bench",
                "nop",
                f"--config={cfg_path}",
            ]
            argv.extend(extra)
            parser = self._bench_parser()
            with patch.object(cli.sys, "argv", argv):
                return cli.parse(parser)

    def test_per_address_branch_and_cache(self):
        config = {
            "branch": {"0x401000": "predictable"},
            "cache": {"L1d": {"0x10008000": {"hit_rate": 50}}},
        }
        args = self._parse_config(config)
        self.assertEqual(
            args.config["branch"]["0x401000"],
            "predictable",
        )
        self.assertEqual(
            args.config["cache"]["L1d"]["0x10008000"]["hit_rate"],
            50,
        )

    def test_config_with_data_rejected(self):
        config = {"data": {"rdi": 1}}
        with self.assertRaises(SystemExit):
            self._parse_config(config)

    def test_branch_address_override(self):
        args = self._parse_config({}, "--config.branch[0x401234]=predictable")
        self.assertEqual(
            args.config["branch"]["0x401234"],
            "predictable",
        )

    def test_cache_address_override(self):
        args = self._parse_config(
            {},
            "--config.cache.L1d[0x10008000]=cold",
        )
        self.assertEqual(
            args.config["cache"]["L1d"]["0x10008000"],
            "cold",
        )

    def test_cache_rate_string_override(self):
        args = self._parse_config({}, "--config.cache.L1d=hit_rate:100")
        L1d = args.config["cache"]["L1d"]
        self.assertEqual(L1d["hit_rate"], 100)

    def test_cache_shortcut(self):
        args = self._parse_config({}, "--config.cache=cold")
        self.assertEqual(args.config["cache"], "cold")

    def test_cache_shortcut_list(self):
        args = self._parse_config({}, "--config.cache=[cold,hot]")
        self.assertEqual(args.config["cache"], ["cold", "hot"])


class TestMemEntries(unittest.TestCase):
    def test_scalar(self):
        entries = list(_pbench._mem_entries("0x1000", 7))
        self.assertEqual(entries, [("0x1000", 7)])

    def test_list(self):
        entries = list(_pbench._mem_entries("0x1000", [1, 2]))
        self.assertEqual(entries, [("0x1000", [1, 2])])

    def test_range(self):
        entries = list(_pbench._mem_entries("0x1000:", [1, 2]))
        self.assertEqual(
            entries,
            [("0x1000", 1), ("0x1001", 2)],
        )

    def test_non_numeric_key(self):
        entries = list(_pbench._mem_entries("bogus", 1))
        self.assertEqual(entries, [])


class TestConfigData(unittest.TestCase):
    def test_none(self):
        result = _pbench._config_data(None)
        self.assertEqual(result, {"regs": {}, "mem": {}})

    def test_empty(self):
        result = _pbench._config_data({})
        self.assertEqual(result, {"regs": {}, "mem": {}})

    def test_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            _pbench._config_data({"data": [1, 2]})


class TestConfigParamColumns(unittest.TestCase):
    def test_flattens(self):
        cols = _pbench.config_param_columns({"branch": "predictable"})
        self.assertIn("config.branch", cols)
        self.assertEqual(cols["config.branch"], "predictable")

    def test_nested(self):
        cols = _pbench.config_param_columns({"cache": {"L1d": {"hit_rate": 100}}})
        self.assertIn("config.cache.L1d.hit_rate", cols)
        self.assertEqual(cols["config.cache.L1d.hit_rate"], 100)


class TestEmbed(unittest.TestCase):
    def test_function_exists(self):
        self.assertTrue(callable(cli.embed))


class TestCachedCpuInfo(unittest.TestCase):
    def test_caches_result(self):
        from perf.info import _cpuinfo

        info1 = _cpuinfo()
        info2 = _cpuinfo()
        self.assertIs(info1, info2)

    def test_returns_dict(self):
        from perf.info import _cpuinfo

        result = _cpuinfo()
        self.assertIsInstance(result, dict)


class TestCpuInfoHz(unittest.TestCase):
    def test_hz_included_in_cpuinfo(self):
        df = cli.perf.cpuinfo()
        self.assertIn("hz", df.columns)
        hz = df["hz"].iloc[0]
        if hz is not None:
            self.assertGreater(hz, 0)


class TestExecSystemPerfWithExecv(unittest.TestCase):
    @patch.object(
        cli.perf,
        "bin",
        return_value="/usr/bin/perf",
    )
    @patch.object(cli.os, "execv")
    def test_calls_execv(self, mock_execv, mock_find):
        with self.assertRaises(SystemExit):
            mock_execv.side_effect = SystemExit
            cli._exec_system_perf(["record", "-e", "cycles"])
        mock_execv.assert_called_once()
        args = mock_execv.call_args
        self.assertEqual(args[0][0], "/usr/bin/perf")


class TestPrintAvailable(unittest.TestCase):
    @patch.object(
        cli.perf,
        "metadata",
        return_value=pd.DataFrame(
            {
                "name": ["foo", "bar"],
                "start": [0x1000, 0x2000],
                "end": [0x1050, 0x2050],
            }
        ),
    )
    def test_prints_hex_addresses(self, mock_meta):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cli._print_available("/bin/foo", "func")
        text = buf.getvalue()
        self.assertIn("0x1000", text)
        self.assertIn("0x2000", text)

    @patch.object(
        cli.perf,
        "metadata",
        return_value=pd.DataFrame(),
    )
    def test_empty_metadata(self, mock_meta):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cli._print_available("/bin/foo", "func")


class TestViewCmd(unittest.TestCase):
    @patch.object(cli, "_load")
    @patch("builtins.print")
    def test_view_aggregates(self, mock_print, mock_load):
        mock_load.return_value = _df()
        args = SimpleNamespace(
            event=None,
            groupby="file,name,mode",
            stat="min,max",
            column="file,name",
            filter=None,
            interactive=False,
        )
        cli.view_cmd(args)
        mock_print.assert_called_once()

    @patch.object(cli, "_load")
    @patch("builtins.print")
    def test_view_no_groupby(self, mock_print, mock_load):
        mock_load.return_value = _df()
        args = SimpleNamespace(
            event=["cycles"],
            groupby="",
            stat="",
            column="file,name",
            filter=None,
            interactive=False,
        )
        cli.view_cmd(args)
        mock_print.assert_called_once()


class TestPlotCmd(unittest.TestCase):
    @patch.object(cli.perf, "plot")
    @patch.object(cli, "_load")
    def test_plot_dispatches(self, mock_load, mock_plot):
        mock_load.return_value = _df()
        args = SimpleNamespace(
            event=["cycles"],
            groupby="file,name,mode",
            type=None,
            filter=None,
            plot_config={},
            xaxis=None,
            output=None,
            logx=False,
            logy=False,
            interactive=False,
            config=None,
        )
        cli.plot_cmd(args)
        mock_plot.assert_called_once()

    @patch.object(cli.perf, "plot")
    @patch.object(cli, "_load")
    def test_plot_type(self, mock_load, mock_plot):
        mock_load.return_value = _df()
        args = SimpleNamespace(
            event=None,
            groupby="file,name,mode",
            type=["bar"],
            filter=None,
            plot_config={},
            xaxis=None,
            output=None,
            logx=False,
            logy=False,
            interactive=False,
            config=None,
        )
        cli.plot_cmd(args)
        call_args = mock_plot.call_args
        self.assertEqual(call_args[0][1], [["bar"]])


class TestPlotInteractive(unittest.TestCase):
    def test_plot_cmd_interactive_has_plt_defined(self):
        df = pd.DataFrame(
            {
                "file": ["a", "a"],
                "name": ["f", "f"],
                "mode": ["latency"] * 2,
                "samples": [0, 1],
                "duration_time": [1.0, 2.0],
            }
        )
        args = SimpleNamespace(
            event=None,
            groupby="file,name,mode",
            type=None,
            filter=None,
            plot_config={},
            xaxis=None,
            output=None,
            logx=False,
            logy=False,
            interactive=True,
            config=None,
        )
        with (
            patch.object(cli, "_load", return_value=df),
            patch.object(cli.perf, "plot") as mock_plot,
            patch.object(cli, "embed") as mock_embed,
            patch("matplotlib.pyplot.show"),
        ):
            cli.plot_cmd(args)
            mock_plot.assert_called()
            mock_embed.assert_called_once()
            ns = mock_embed.call_args[0][0]
            self.assertIn("plt", ns)
            self.assertIsNotNone(ns["plt"])


class TestBenchDispatch(unittest.TestCase):
    @patch.object(cli, "_stage_S")
    def test_emit_asm(self, mock_stage):
        args = SimpleNamespace(
            emit_asm=True,
            emit_obj=False,
            target="foo",
            file="a.out",
            asm=None,
            topdown=False,
        )
        cli._bench_dispatch(args)
        mock_stage.assert_called_once_with(args, is_asm=False)

    @patch.object(cli, "_stage_c")
    def test_emit_obj(self, mock_stage):
        args = SimpleNamespace(
            emit_asm=False,
            emit_obj=True,
            target="foo",
            file="a.out",
            asm=None,
            topdown=False,
        )
        cli._bench_dispatch(args)
        mock_stage.assert_called_once_with(args, is_asm=False)

    @patch.object(cli, "_bench_file")
    def test_measure_dispatch(self, mock_file):
        args = SimpleNamespace(emit_asm=False, emit_obj=False)
        cli._bench_dispatch(args)
        mock_file.assert_called_once_with(args)

    @patch.object(cli, "bench_asm_cmd_measure")
    def test_asm_measure_dispatch(self, mock_measure):
        args = SimpleNamespace(emit_asm=False, emit_obj=False)
        cli._bench_dispatch(args, is_asm=True)
        mock_measure.assert_called_once_with(args)

    def test_asm_object_stage_needs_binary(self):
        args = SimpleNamespace(asm="mov eax, 42", emit_asm=False, emit_obj=True)
        with self.assertRaises(SystemExit):
            cli._bench_dispatch(args, is_asm=True)


class TestBenchAsmCmd(unittest.TestCase):
    def test_requires_code(self):
        args = SimpleNamespace(asm=None)
        with self.assertRaises(SystemExit):
            cli.bench_asm_cmd(args)

    @patch.object(cli, "_bench_dispatch")
    def test_dispatches(self, mock_dispatch):
        args = SimpleNamespace(asm="mov eax, 42")
        cli.bench_asm_cmd(args)
        mock_dispatch.assert_called_once_with(args, is_asm=True)


class TestBenchCmd(unittest.TestCase):
    @patch.object(cli, "_bench_list")
    def test_list_dispatch(self, mock_list):
        args = SimpleNamespace(list=True, target=None, asm=None, file="a.out")
        cli.bench_cmd(args)
        mock_list.assert_called_once_with(args)

    @patch.object(cli, "bench_asm_cmd")
    def test_asm_dispatch(self, mock_asm):
        args = SimpleNamespace(list=False, target=None, asm="mov eax, 42", file=None)
        cli.bench_cmd(args)
        mock_asm.assert_called_once_with(args)

    @patch.object(cli, "_bench_dispatch")
    def test_target_dispatch(self, mock_dispatch):
        args = SimpleNamespace(list=False, target="foo", asm=None, file="a.out")
        cli.bench_cmd(args)
        mock_dispatch.assert_called_once_with(args, is_asm=False)

    @patch.object(cli, "_bench_dispatch")
    def test_missing_target_raises(self, mock_dispatch):
        args = SimpleNamespace(list=False, target=None, asm=None, file="a.out")
        with self.assertRaises(SystemExit):
            cli.bench_cmd(args)
        mock_dispatch.assert_not_called()

    def test_needs_code_or_binary(self):
        args = SimpleNamespace(list=False, target=None, asm=None, file=None)
        with self.assertRaises(SystemExit):
            cli.bench_cmd(args)
        args = SimpleNamespace(list=False, target=None, asm="", file=None)
        with self.assertRaises(SystemExit):
            cli.bench_cmd(args)


class _PerfconfigParser:
    @staticmethod
    def parser():
        import argparse

        p = argparse.ArgumentParser(prog="perf")
        sub = p.add_subparsers(dest="command", required=True)
        plot = sub.add_parser("plot")
        plot.add_argument("-e", "--event", dest="event", action="append")
        plot.add_argument("-x", "--xaxis", dest="xaxis", default=None)
        plot.add_argument("-o", "--output", dest="output", default=None)
        plot.add_argument("--logx", dest="logx", action="store_true")
        bench = sub.add_parser("bench")
        bench.add_argument("-m", "--mode", dest="mode", nargs="+")
        bench.add_argument("--config.cache", dest="config_cache", default=None)
        bench.add_argument("-e", "--event", dest="event", action="append")
        bench.add_argument("--backend", dest="backend", default=None)
        sub.add_parser("compare", aliases=["cmp"])
        return p

    def _write(self, text):
        fh = tempfile.NamedTemporaryFile("w", suffix=".perfconfig", delete=False)
        try:
            fh.write(text)
        finally:
            fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def apply(self, command, argv, text=None):
        parser = self.parser()
        if text is None:
            return cli._apply_perfconfig(parser, command, list(argv))
        return cli._apply_perfconfig(
            parser, command, list(argv), config_path=self._write(text)
        )


class TestLoadPerfconfig(unittest.TestCase):
    def test_missing_returns_empty(self):
        path = str(Path(tempfile.mkdtemp()) / "does-not-exist.perfconfig")
        self.assertEqual(cli._load_perfconfig(path), {})

    def test_reads_sections(self):
        fh = tempfile.NamedTemporaryFile("w", suffix=".perfconfig", delete=False)
        try:
            fh.write(
                "[plot]\nconfig.style = dark_background\n\n[bench]\nmode = throughput\n"
            )
        finally:
            fh.close()
        try:
            cfg = cli._load_perfconfig(fh.name)
        finally:
            os.unlink(fh.name)
        self.assertEqual(
            cfg,
            {
                "plot": {"config.style": "dark_background"},
                "bench": {"mode": "throughput"},
            },
        )

    def test_duplicate_option_last_wins(self):
        fh = tempfile.NamedTemporaryFile("w", suffix=".perfconfig", delete=False)
        try:
            fh.write("[plot]\nmode=a\nmode=b\n")
        finally:
            fh.close()
        try:
            cfg = cli._load_perfconfig(fh.name)
        finally:
            os.unlink(fh.name)
        self.assertEqual(cfg["plot"]["mode"], "b")

    def test_bad_syntax_returns_empty(self):
        fh = tempfile.NamedTemporaryFile("w", suffix=".perfconfig", delete=False)
        try:
            fh.write("[plot\nnope\n")
        finally:
            fh.close()
        try:
            cfg = cli._load_perfconfig(fh.name)
        finally:
            os.unlink(fh.name)
        self.assertEqual(cfg, {})


class TestApplyPerfconfig(_PerfconfigParser, unittest.TestCase):
    def test_no_file_returns_same_argv(self):
        argv = self.apply("plot", ["plot", "--", "data/"], "some-nonexistent@path")
        self.assertEqual(argv, ["plot", "--", "data/"])

    def test_empty_config_returns_same_argv(self):
        argv = self.apply("plot", ["plot", "--", "data/"], "")
        self.assertEqual(argv, ["plot", "--", "data/"])

    def test_plot_style_default(self):
        argv = self.apply(
            "plot",
            ["plot", "--", "data/"],
            "[plot]\nconfig.style=dark_background\n",
        )
        self.assertEqual(
            argv, ["plot", "--config.style=dark_background", "--", "data/"]
        )

    def test_cli_override_wins(self):
        argv = self.apply(
            "plot",
            ["plot", "--config.style=classic", "--", "data/"],
            "[plot]\nconfig.style=dark_background\n",
        )
        self.assertEqual(argv, ["plot", "--config.style=classic", "--", "data/"])

    def test_short_flag_override_wins(self):
        argv = self.apply(
            "bench",
            ["bench", "-m", "latency", "nop"],
            "[bench]\nmode=throughput\n",
        )
        self.assertEqual(argv, ["bench", "-m", "latency", "nop"])

    def test_bench_mode_default(self):
        argv = self.apply(
            "bench",
            ["bench", "nop"],
            "[bench]\nmode=throughput\n",
        )
        self.assertEqual(argv, ["bench", "--mode=throughput", "nop"])

    def test_default_section_then_command_overrides(self):
        argv = self.apply(
            "bench",
            ["bench", "nop"],
            "[default]\nmode=latency\n[bench]\nmode=throughput\n",
        )
        self.assertEqual(argv, ["bench", "--mode=throughput", "nop"])

    def test_default_key_not_applicable_skipped(self):
        argv = self.apply(
            "plot",
            ["plot", "--", "data/"],
            "[default]\nbackend=loop\n",
        )
        self.assertEqual(argv, ["plot", "--", "data/"])

    def test_unrelated_section_ignored(self):
        argv = self.apply(
            "bench",
            ["bench", "nop"],
            "[plot]\nconfig.style=dark_background\n",
        )
        self.assertEqual(argv, ["bench", "nop"])

    def test_config_dot_applies_for_plot(self):
        argv = self.apply(
            "plot",
            ["plot", "--", "data/"],
            "[default]\nconfig.style=dark_background\n",
        )
        self.assertEqual(
            argv, ["plot", "--config.style=dark_background", "--", "data/"]
        )

    def test_inserts_after_command(self):
        argv = self.apply("bench", ["bench", "sched"], "[bench]\nmode=throughput\n")
        self.assertEqual(argv, ["bench", "--mode=throughput", "sched"])

    def test_flows_into_plot_resolve(self):
        argv = self.apply(
            "plot",
            ["plot", "--", "data/"],
            "[plot]\nconfig.style=dark_background\n",
        )
        i = argv.index("--")
        parser = self.parser()
        args, unknown = parser.parse_known_args(argv[:i])
        cfg = cli._resolve_plot_config(args, unknown)
        self.assertEqual(cfg["style"], "dark_background")


class TestPresentOptions(unittest.TestCase):
    def test_long_and_short(self):
        opts = cli._present_options(
            ["bench", "-m", "latency", "--event=cycles", "--", "data/"]
        )
        self.assertIn("m", opts)
        self.assertIn("event", opts)

    def test_nothing_past_double_dash(self):
        opts = cli._present_options(["--", "--event=cycles"])
        self.assertNotIn("event", opts)


class TestOptionHelpers(unittest.TestCase):
    def test_find_subparser(self):
        parser = _PerfconfigParser.parser()
        self.assertEqual(cli._find_subparser(parser, "bench").prog, "perf bench")

    def test_find_subparser_alias(self):
        parser = _PerfconfigParser.parser()
        self.assertEqual(cli._find_subparser(parser, "cmp").prog, "perf compare")

    def test_find_subparser_unknown(self):
        parser = _PerfconfigParser.parser()
        self.assertIsNone(cli._find_subparser(parser, "nope"))

    def test_option_aliases_dest(self):
        parser = _PerfconfigParser.parser()
        sub = cli._find_subparser(parser, "bench")
        self.assertEqual(cli._option_aliases(sub, "mode"), {"mode", "m"})

    def test_option_aliases_config_dot(self):
        parser = _PerfconfigParser.parser()
        sub = cli._find_subparser(parser, "bench")
        self.assertEqual(cli._option_aliases(sub, "config.cache"), {"config.cache"})

    def test_key_applies_action(self):
        parser = _PerfconfigParser.parser()
        sub = cli._find_subparser(parser, "bench")
        self.assertTrue(cli._key_applies(sub, "bench", "backend"))

    def test_key_applies_config_dot_only_bench_plot(self):
        parser = _PerfconfigParser.parser()
        sub = cli._find_subparser(parser, "bench")
        self.assertTrue(cli._key_applies(sub, "bench", "config.cache"))
        sub_view = cli._find_subparser(parser, "compare")
        self.assertFalse(cli._key_applies(sub_view, "compare", "config.style"))

    def test_key_applies_unknown_false(self):
        parser = _PerfconfigParser.parser()
        sub = cli._find_subparser(parser, "bench")
        self.assertFalse(cli._key_applies(sub, "bench", "nope"))


class TestBenchList(unittest.TestCase):
    def _listing(self):
        return pd.DataFrame(
            {
                "kind": ["label", "func", "func"],
                "name": ["hot", "foo", "bar"],
                "start": [1, 2, 3],
                "end": [1, 10, 20],
                "size": [1, 8, 17],
            }
        )

    @patch.object(cli, "_print_available")
    @patch.object(cli.perf, "metadata")
    def test_lists_all_targets(self, mock_meta, mock_print):
        mock_meta.return_value = self._listing()
        args = SimpleNamespace(file="a.out", target=None)
        cli._bench_list(args)
        mock_meta.assert_called_once_with("a.out", None)
        printed = mock_print.call_args[0][2]
        self.assertEqual(len(printed), 3)

    @patch.object(cli, "_print_available")
    @patch.object(cli.perf, "metadata")
    def test_list_ignores_target(self, mock_meta, mock_print):
        mock_meta.return_value = self._listing()
        args = SimpleNamespace(file="a.out", target="foo")
        cli._bench_list(args)
        printed = mock_print.call_args[0][2]
        self.assertEqual(len(printed), 3)

    def test_requires_file(self):
        args = SimpleNamespace(file=None, target=None)
        with self.assertRaises(SystemExit):
            cli._bench_list(args)


class TestUnknownTargetListsAll(unittest.TestCase):
    @patch.object(cli.perf, "bench", side_effect=ValueError("cannot resolve 'nope'"))
    def test_unknown_target_exits(self, mock_bench):
        args = SimpleNamespace(
            file="a.out",
            target="nope",
            bench_name=None,
            mode=["latency"],
            event=None,
            topdown=False,
            config={},
            data=None,
            setup=None,
            teardown=None,
            backend=None,
            debug=False,
        )
        with self.assertRaises(SystemExit) as ctx:
            cli._bench_file(args)
        self.assertIn("cannot resolve 'nope'", str(ctx.exception))

    @patch.object(cli.perf, "bench", side_effect=ValueError("cannot resolve 'a..b'"))
    def test_unknown_region_exits(self, mock_bench):
        args = SimpleNamespace(
            file="a.out",
            target="a..b",
            bench_name=None,
            mode=["latency"],
            event=None,
            topdown=False,
            config={},
            data=None,
            setup=None,
            teardown=None,
            backend=None,
            debug=False,
        )
        with self.assertRaises(SystemExit):
            cli._bench_file(args)


class TestBenchAsmCmdMeasure(unittest.TestCase):
    @patch.object(cli, "_output_or_default")
    @patch.object(cli.perf, "bench")
    def test_requires_mode(self, mock_bench, mock_output):
        args = SimpleNamespace(
            mode=None,
            asm="nop",
            bench_name=None,
            event=None,
            config=None,
            data=None,
            setup=None,
            teardown=None,
            backend=None,
            interactive=False,
            topdown=False,
        )
        with self.assertRaises(SystemExit):
            cli.bench_asm_cmd_measure(args)
        mock_bench.assert_not_called()

    @patch.object(cli, "_output_or_default")
    @patch.object(cli.perf, "bench")
    def test_requires_code(self, mock_bench, mock_output):
        args = SimpleNamespace(
            mode=["latency"],
            asm="",
            bench_name=None,
            event=None,
            config=None,
            data=None,
            setup=None,
            teardown=None,
            backend=None,
            interactive=False,
            topdown=False,
        )
        with self.assertRaises(SystemExit):
            cli.bench_asm_cmd_measure(args)
        mock_bench.assert_not_called()

    @patch.object(cli, "_output_or_default")
    @patch.object(cli.perf, "bench")
    def test_output_called(self, mock_bench, mock_output):
        mock_bench.return_value = pd.DataFrame({"duration_time": [1.0], "samples": [0]})
        args = SimpleNamespace(
            mode=["latency"],
            asm="nop",
            bench_name=None,
            event=None,
            config={},
            data=None,
            setup=None,
            teardown=None,
            backend=None,
            interactive=False,
            topdown=False,
        )
        cli.bench_asm_cmd_measure(args)
        mock_output.assert_called_once()


class TestEmitText(unittest.TestCase):
    @patch.object(cli, "_resolve_output", return_value=(None, True))
    def test_stdout(self, mock_resolve):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cli._emit_text(SimpleNamespace(), "hello")
        self.assertEqual(buf.getvalue().strip(), "hello")

    @patch.object(
        cli,
        "_resolve_output",
        return_value=("out.s", False),
    )
    def test_file_output(self, mock_resolve):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace()
            path = Path(tmp, "out.s")
            with patch.object(
                cli,
                "_resolve_output",
                return_value=(str(path), False),
            ):
                cli._emit_text(args, "hello")
            self.assertEqual(path.read_text(), "hello\n")

    @patch.object(cli, "_resolve_output", return_value=(None, True))
    def test_newline_appended(self, mock_resolve):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cli._emit_text(SimpleNamespace(), "noend")
        self.assertTrue(buf.getvalue().endswith("\n"))


class TestRegexBasics(unittest.TestCase):
    def test_baseline_re_matches_simple(self):
        m = cli._BASELINE_RE.match("'foo'.bar")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("key"), "foo")
        self.assertEqual(m.group("metric"), "bar")

    def test_baseline_re_no_match_without_quotes(self):
        m = cli._BASELINE_RE.match("foo.bar")
        self.assertIsNone(m)

    def test_baseline_re_metric_with_hyphens(self):
        m = cli._BASELINE_RE.match("'g'.my-metric")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("metric"), "my-metric")


class TestMainSubcommandRouting(unittest.TestCase):
    def test_bench_subcommand(self):
        self.assertEqual(
            cli._bench_subcommand(["bench", "func", "main"]),
            "bench",
        )

    def test_view_subcommand(self):
        self.assertEqual(
            cli._bench_subcommand(["view", "--data", "x"]),
            "view",
        )

    def test_plot_subcommand(self):
        self.assertEqual(
            cli._bench_subcommand(["plot", "-e", "cycles"]),
            "plot",
        )

    def test_flags_only(self):
        self.assertIsNone(cli._bench_subcommand(["--json", "--verbose"]))


class TestTableToDf(unittest.TestCase):
    def _text(self):
        return cli._format_table(_table_df()).to_string(index=False)

    def test_round_trip_columns(self):
        parsed = cli._table_to_df(self._text())
        self.assertEqual(parsed.columns.tolist(), _table_df().columns.tolist())

    def test_multi_token_name_cells(self):
        parsed = cli._table_to_df(self._text())
        self.assertEqual(parsed["name"].tolist(), ["mov r11, rax"] * 2)

    def test_string_columns_preserved(self):
        parsed = cli._table_to_df(self._text())
        self.assertEqual(parsed["mode"].tolist(), ["latency"] * 2)
        self.assertEqual(
            parsed["config.branch"].tolist(),
            ["unpredictable", "as-is"],
        )
        self.assertFalse(pd.api.types.is_numeric_dtype(parsed["config.branch"]))

    def test_blank_file_cells(self):
        parsed = cli._table_to_df(self._text())
        self.assertTrue(parsed["file"].isna().all())

    def test_duration_denormalized(self):
        parsed = cli._table_to_df(self._text())
        self.assertTrue(pd.api.types.is_numeric_dtype(parsed["duration_time"]))
        self.assertAlmostEqual(parsed["duration_time"].iloc[0], 1.5, places=12)
        self.assertAlmostEqual(parsed["duration_time"].iloc[1], 2250.0, places=9)

    def test_numeric_columns_promoted(self):
        parsed = cli._table_to_df(self._text())
        self.assertTrue(pd.api.types.is_numeric_dtype(parsed["cycles"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(parsed["operations"]))
        self.assertIn("duration_time", cli._metric_columns(parsed))
        self.assertIn("cycles", cli._metric_columns(parsed))
        self.assertNotIn("config.branch", cli._metric_columns(parsed))

    def test_empty_text(self):
        self.assertTrue(cli._table_to_df("").empty)
        self.assertTrue(cli._table_to_df("   \n  \n").empty)

    def test_header_only_rejected(self):
        with self.assertRaises(ValueError):
            cli._table_to_df("file name mode")

    def test_garbage_rejected(self):
        with self.assertRaises(ValueError):
            cli._table_to_df("not a perf table at all")

    def test_blank_data_rows_preserved(self):
        text = self._text().splitlines()
        parsed = cli._table_to_df("\n".join(text[:1] + ["   " * 4, *text[1:]]))
        self.assertEqual(len(parsed), len(text))
        self.assertTrue(parsed.iloc[0].isna().all())

    def test_all_blank_rows_return_nan(self):
        text = self._text().splitlines()
        parsed = cli._table_to_df("\n".join(text[:1] + ["   " * 4] * 2))
        self.assertEqual(len(parsed), 2)
        self.assertTrue(parsed.isna().all().all())

    def test_wide_cells_parsed(self):
        df = _table_df().copy()
        df["name"] = [
            "sub rsp, 8 and rsp, r15",
            "mov  r11, rax and rsp, r15 xor rax, rax",
        ]
        parsed = cli._table_to_df(cli._format_table(df).to_string(index=False))
        self.assertEqual(parsed["name"].tolist(), df["name"].tolist())


class TestLoadFromStdin(unittest.TestCase):
    def test_envelope_json(self):
        env = cli._record_envelope(_env_df())
        raw = json.dumps(env)
        buf = io.StringIO(raw)
        args = SimpleNamespace(data=None)
        with patch.object(cli.sys, "stdin", buf):
            with patch.object(
                cli.sys.stdin,
                "isatty",
                return_value=False,
            ):
                loaded = cli._load(args)
        self.assertEqual(len(loaded), 2)

    def test_plain_records_rejected(self):
        records = [
            {"cycles": 10, "name": "a"},
            {"cycles": 20, "name": "b"},
        ]
        raw = json.dumps(records)
        buf = io.StringIO(raw)
        args = SimpleNamespace(data=None)
        with patch.object(cli.sys, "stdin", buf):
            with patch.object(
                cli.sys.stdin,
                "isatty",
                return_value=False,
            ):
                with self.assertRaises(SystemExit):
                    cli._load(args)

    def test_bench_table(self):
        raw = cli._format_table(_table_df()).to_string(index=False)
        buf = io.StringIO(raw)
        args = SimpleNamespace(data=None)
        with patch.object(cli.sys, "stdin", buf):
            with patch.object(
                cli.sys.stdin,
                "isatty",
                return_value=False,
            ):
                loaded = cli._load(args)
        self.assertEqual(loaded["name"].tolist(), ["mov r11, rax"] * 2)
        self.assertTrue(pd.api.types.is_numeric_dtype(loaded["duration_time"]))
        self.assertAlmostEqual(loaded["duration_time"].iloc[0], 1.5, places=12)

    def test_garbage_stdin_rejected(self):
        buf = io.StringIO("definitely not a table")
        args = SimpleNamespace(data=None)
        with patch.object(cli.sys, "stdin", buf):
            with patch.object(
                cli.sys.stdin,
                "isatty",
                return_value=False,
            ):
                with self.assertRaises(SystemExit):
                    cli._load(args)

    def test_empty_stdin_no_data_raises(self):
        buf = io.StringIO("")
        args = SimpleNamespace(data=None)
        with patch.object(cli.sys, "stdin", buf):
            with patch.object(
                cli.sys.stdin,
                "isatty",
                return_value=False,
            ):
                with self.assertRaises(SystemExit):
                    cli._load(args)


class TestParseEdgeCases(unittest.TestCase):
    @staticmethod
    def _bench_parser():
        import argparse

        p = argparse.ArgumentParser(prog="perf")
        sub = p.add_subparsers(dest="command")
        bench = sub.add_parser("bench")
        bench.add_argument("code", nargs="?", default=None)
        bench.add_argument("-f", "--filter", dest="target", default=None)
        bench.add_argument("--list", action="store_true", default=False)
        bench.add_argument("-n", "--name", default=None)
        bench.add_argument(
            "-m",
            "--mode",
            choices=["latency", "throughput"],
            default=None,
        )
        bench.add_argument("--config", default=None)
        bench.add_argument("--data", default=None)
        bench.add_argument(
            "-e",
            "--event",
            action="append",
            default=None,
        )
        bench.add_argument("-o", "--output", dest="output", default=None)
        bench.add_argument("--json", action="store_true", default=False)
        bench.add_argument(
            "-i",
            "--interactive",
            action="store_true",
        )
        bench.add_argument("-v", "--verbose", action="store_true")
        bench.add_argument("-S", dest="emit_asm", action="store_true")
        bench.add_argument("-c", dest="emit_obj", action="store_true")
        bench.add_argument("--backend", default=None)
        bench.add_argument("--setup", dest="setup_asm", default=None)
        bench.add_argument(
            "--teardown",
            dest="teardown_asm",
            default=None,
        )
        return p

    def _parse(self, *extra):
        argv = ["perf", "bench", "nop"]
        argv.extend(extra)
        parser = self._bench_parser()
        with patch.object(cli.sys, "argv", argv):
            return cli.parse(parser)

    def test_unknown_arg_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("--bogus=1")

    def test_no_args_is_valid(self):
        args = self._parse()
        self.assertIsNone(args.data)
        self.assertIsInstance(args.config, dict)

    def test_data_file_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"regs": {"rdi": 42}}, f)
            f.flush()
            args = self._parse("--data", f.name)
        os.unlink(f.name)
        self.assertEqual(
            args.data,
            {"regs": {"rdi": 42}, "mem": {}},
        )

    def test_empty_data_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{}")
            f.flush()
            args = self._parse("--data", f.name)
        os.unlink(f.name)
        self.assertIsNone(args.data)

    def test_invalid_json_data(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("not json")
            f.flush()
            with self.assertRaises(json.JSONDecodeError):
                self._parse("--data", f.name)
        os.unlink(f.name)

    def test_invalid_json_string_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("--data", "not json")


class TestGlobalCaches(unittest.TestCase):
    def test_hz_cache_respected(self):
        from perf.info import _cpu_hz

        first = _cpu_hz()
        try:
            self.assertEqual(_cpu_hz(), first)
            self.assertEqual(_cpu_hz.cache_info().hits >= 0, True)
        finally:
            pass

    def test_cpuinfo_cache_respected(self):
        from perf.info import _cpuinfo

        first = _cpuinfo()
        second = _cpuinfo()
        self.assertIs(first, second)


class TestLoadWalksDirectories(unittest.TestCase):
    def test_walks_subdirs_for_json(self):
        env = cli._record_envelope(_env_df())
        with tempfile.TemporaryDirectory() as tmp:
            subdir = Path(tmp, "sub")
            subdir.mkdir()
            subdir.joinpath("x.json").write_text(json.dumps(env))
            loaded = cli._load(SimpleNamespace(data=[tmp]))
        self.assertEqual(len(loaded), 2)

    def test_finds_perf_data_files(self):
        env = cli._record_envelope(_env_df())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "perf.data.abc")
            p.write_text(json.dumps(env))
            loaded = cli._load(SimpleNamespace(data=[tmp]))
            self.assertIsNotNone(loaded)

    def test_finds_dot_data_files(self):
        env = cli._record_envelope(_env_df())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "run1.data")
            p.write_text(json.dumps(env))
            loaded = cli._load(SimpleNamespace(data=[tmp]))
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), 2)


class TestPrintAvailableEmpty(unittest.TestCase):
    @patch.object(
        cli.perf,
        "metadata",
        return_value=pd.DataFrame(columns=["name", "start", "end"]),
    )
    def test_empty_metadata_prints(self, mock_meta):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            cli._print_available("/bin/x", "func")


def _binary_with_labels(tmp):
    src = os.path.join(tmp, "cmds.c")
    exe = os.path.join(tmp, "cmds")
    with open(src, "w") as fh:
        fh.write(
            '#include "perf/perf.hpp"\n'
            "int add42(int x){return x+42;}\n"
            "const char* fizz_buzz(int n){\n"
            '  if(n%15==0) return "FizzBuzz";\n'
            '  else if(n%3==0) return "Fizz";\n'
            '  else if(n%5==0) return "Buzz";\n'
            '  else return "Unknown";\n'
            "}\n"
            "int hot(int n){\n"
            "  PERF_LABEL(hot_begin);\n"
            "  int s=0;\n"
            "  for(int i=0;i<n;++i) s+=i;\n"
            "  PERF_LABEL(hot_end);\n"
            "  return s;\n"
            "}\n"
            "int main(){return 0;}\n"
        )
    repo = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        ["g++", "-O2", f"-I{repo}/lib", "-o", exe, src],
        capture_output=True,
    )
    if r.returncode != 0 or not os.path.exists(exe):
        return None
    return exe


class TestHarnessRegisterGuard(unittest.TestCase):
    def test_guard_asm_even_pushes(self):
        from perf.arch import x86_64

        pre, post = x86_64.guard_asm(["r8", "r9", "r10", "r11"])
        self.assertEqual(pre.count("push"), 4)
        self.assertEqual(post.count("pop"), 4)
        self.assertNotIn("sub rsp", pre)
        ks_pre, ks_post = pre, post
        import keystone

        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        enc, _ = ks.asm(ks_pre + "\nnop\n" + ks_post)
        self.assertTrue(enc)

    def test_guard_asm_odd_pushes_padded(self):
        from perf.arch import x86_64

        pre, post = x86_64.guard_asm(["r8", "r9", "r10", "rbx", "rbp"])
        self.assertEqual(pre.count("push"), 5)
        self.assertIn("sub rsp, 8", pre)
        self.assertIn("add rsp, 8", post)
        import keystone

        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        enc, _ = ks.asm(pre + "\nnop\n" + post)
        self.assertTrue(enc)

    def test_guard_asm_dedups(self):
        from perf.arch import x86_64

        pre, _ = x86_64.guard_asm(["r8", "r8", "r9"])
        self.assertEqual(pre.count("push r8"), 1)

    def test_timing_regs_match_timing_choice(self):
        from perf.arch import x86_64

        self.assertEqual(x86_64.timing_regs("duration_time"), ["r11"])
        regs = x86_64.timing_regs("duration_time", callee_saved=True, avoid={"r11"})
        self.assertEqual(regs, ["rbx"])
        t0, _ = x86_64.timing("duration_time", [None])
        self.assertIn("mov r11, rax", t0)

    def test_timed_guard_covers_loop_and_baselines(self):
        from perf.arch import x86_64

        pre, post = x86_64.timed_guard("duration_time")
        for reg in x86_64.HARNESS_LOOP_REGS:
            self.assertIn(f"push {reg}", pre)
            self.assertIn(f"pop {reg}", post)
        self.assertIn("push r11", pre)
        self.assertIn("pop r11", post)

    def test_loop_asm_preserves_harness_regs(self):
        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        for mode in ("latency", "throughput"):
            asm, _, _ = _build_loop_asm(
                128,
                mode,
                "mov r8, 42;",
                "",
                "",
                [],
                ["duration_time"],
                [None],
                {"L1d": 100, "L2": 100, "L3": 100},
                True,
                __import__("random").Random(0),
                arch,
            )
            for reg in arch.HARNESS_LOOP_REGS:
                self.assertIn(f"push {reg}", asm)
                self.assertIn(f"pop {reg}", asm)

    def test_clobbering_snippets_run(self):
        from perf.arch import arch as get_arch

        arch = get_arch()
        regs = list(arch.HARNESS_LOOP_REGS) + arch.timing_regs("duration_time")
        for reg in regs:
            for mode in ("latency", "throughput"):
                with self.subTest(reg=reg, mode=mode):
                    df = bench(
                        asm=f"mov {reg}, 42",
                        mode=mode,
                        config=dict(_FAST),
                    )
                    self.assertFalse(df.empty)
                    self.assertIn("duration_time", df.columns)


class TestAsmCommands(unittest.TestCase):
    @staticmethod
    def _bench_retry(**kw):
        last = None
        for _ in range(3):
            try:
                return bench(**kw)
            except (ValueError, TypeError):
                raise
            except Exception as ex:
                last = ex
        raise last

    def test_modes_and_backends(self):
        for mode in ("latency", "throughput"):
            for backend in ("loop", "unroll"):
                with self.subTest(mode=mode, backend=backend):
                    if mode == "throughput" and backend == "unroll":
                        with self.assertRaises(ValueError):
                            bench(
                                asm="mov eax, 42",
                                mode=mode,
                                backend=backend,
                                config=dict(_FAST),
                            )
                        continue
                    df = self._bench_retry(
                        asm="mov eax, 42",
                        mode=mode,
                        backend=backend,
                        config=dict(_FAST),
                    )
                    self.assertFalse(df.empty)
                    self.assertIn("duration_time", df.columns)
                    self.assertGreaterEqual(len(df), 1)
                    self.assertLessEqual(len(df), 2)

    def test_modes_default_backends(self):
        df = bench(asm="nop", mode="latency", config=dict(_FAST))
        self.assertFalse(df.empty)
        self.assertIn("operations", df.columns)
        self.assertEqual(df["operations"].tolist(), [1] * len(df))
        self.assertEqual(df.attrs["config"]["backend"], "unroll")

        df = bench(asm="nop", mode="throughput", config=dict(_FAST))
        self.assertFalse(df.empty)
        self.assertIn("operations", df.columns)
        self.assertEqual(df.attrs["config"]["backend"], "loop")
        if len(df) and not pd.isna(df["operations"].iloc[0]):
            self.assertGreater(df["operations"].iloc[0], 1)

    def test_setup_and_teardown(self):
        df = bench(
            asm="nop",
            mode="latency",
            setup=["mov ecx, 1"],
            teardown=["mov edx, 2"],
            config=dict(_FAST),
        )
        self.assertFalse(df.empty)

    def test_data_forms(self):
        for data in (
            {"rdi": 15},
            {"arg0": 15},
            {"eax": 7, "edx": 0, "ecx": 42},
            {"0x41000000000": 100},
            {"rdi": [3, 5, 15]},
        ):
            with self.subTest(data=data):
                df = bench(asm="nop", mode="latency", data=data, config=dict(_FAST))
                self.assertFalse(df.empty)

    def test_idiv_loop_backend(self):
        df = bench(
            asm="idiv ecx",
            mode="latency",
            backend="loop",
            data={"eax": 100, "edx": 0, "ecx": 42},
            config=dict(_FAST),
        )
        self.assertFalse(df.empty)

    def test_events(self):
        from perf import core

        try:
            c = core.open_event("cycles")
        except Exception as ex:
            self.skipTest(f"rdpmc unavailable: {ex}")
        try:
            c.enable()
            c.close()
        except Exception as ex:
            self.skipTest(f"rdpmc unavailable: {ex}")
        df = bench(
            asm="nop",
            mode="latency",
            event=["cycles", "instructions"],
            config=dict(_FAST),
        )
        self.assertIn("cycles", df.columns)
        self.assertIn("instructions", df.columns)
        df = bench(
            asm="nop",
            mode="latency",
            event=[["cycles", "instructions"], ["duration_time"]],
            config=dict(_FAST),
        )
        self.assertIn("cycles", df.columns)
        self.assertIn("instructions", df.columns)

    def test_single_instruction_zero_delta_reports_zero(self):
        from perf import core

        try:
            c = core.open_event("cycles")
        except Exception as ex:
            self.skipTest(f"rdpmc unavailable: {ex}")
        try:
            c.enable()
            c.close()
        except Exception as ex:
            self.skipTest(f"rdpmc unavailable: {ex}")
        df = bench(
            asm="imul eax, eax, 3",
            mode="latency",
            backend="loop",
            data={"eax": 7},
            event=["cycles", "instructions"],
            config=dict(_FAST),
        )
        self.assertFalse(df.empty)
        self.assertIn("cycles", df.columns)
        self.assertIn("instructions", df.columns)

    def test_disassemble_symmetry(self):
        text = perf.disassemble(asm="mov eax, 42")
        self.assertIn("mov eax", text)
        self.assertIn(".intel_syntax noprefix", text)

    def test_errors(self):
        df = bench(asm="nop", mode=None, config=dict(_FAST))
        rec = df.reset_index()
        self.assertEqual(
            sorted(rec["mode"].unique().tolist()), ["latency", "throughput"]
        )
        with self.assertRaises(ValueError):
            bench(asm="nop", mode="sideways", config=dict(_FAST))
        with self.assertRaises(ValueError):
            bench(asm="nop", mode="latency", backend="turbo", config=dict(_FAST))


class TestFuncCommands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp()
        cls.exe = _binary_with_labels(cls._tmp)
        if cls.exe is None:
            raise unittest.SkipTest("g++ unavailable")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_modes(self):
        for mode in ("latency", "throughput"):
            with self.subTest(mode=mode):
                df = bench(
                    file=self.exe,
                    target="fizz_buzz",
                    mode=mode,
                    data={"rdi": 15},
                    config=dict(_FAST),
                )
                self.assertFalse(df.empty)
                self.assertIn("duration_time", df.columns)

    def test_backends(self):
        for backend in ("loop", "unroll"):
            with self.subTest(backend=backend):
                df = bench(
                    file=self.exe,
                    target="add42",
                    mode="latency",
                    backend=backend,
                    data={"rdi": 1},
                    config=dict(_FAST),
                )
                self.assertFalse(df.empty)

    def test_arg_alias_symmetry(self):
        from perf.bench import data_param_value

        a = bench(
            file=self.exe,
            target="fizz_buzz",
            mode="latency",
            data={"rdi": 15},
            config=dict(_FAST),
        )
        b = bench(
            file=self.exe,
            target="fizz_buzz",
            mode="latency",
            data={"arg0": 15},
            config=dict(_FAST),
        )
        self.assertIn("data.rdi", a.columns)
        self.assertNotIn("data.arg0", a.columns)
        self.assertIn("data.rdi", b.columns)
        self.assertNotIn("data.arg0", b.columns)
        self.assertEqual(a["data.rdi"].tolist(), b["data.rdi"].tolist())
        self.assertEqual(
            data_param_value(a.attrs["data"], "data.rdi"),
            data_param_value(b.attrs["data"], "data.arg0"),
        )

    def test_branch_configs(self):
        for branch in ("predictable", "unpredictable"):
            with self.subTest(branch=branch):
                df = bench(
                    file=self.exe,
                    target="fizz_buzz",
                    mode="latency",
                    config=dict(_FAST, branch=branch),
                )
                self.assertFalse(df.empty)

    def test_disassemble_symmetry(self):
        text = perf.disassemble(file=self.exe, target="add42")
        self.assertIn(".intel_syntax noprefix", text)
        self.assertIn("ret", text)

    def test_to_object_symmetry(self):
        out = os.path.join(self._tmp, "func_bench.o")
        perf.to_object(file=self.exe, target="add42", path=out)
        self.assertTrue(os.path.exists(out))
        from elftools.elf.elffile import ELFFile

        with open(out, "rb") as fh:
            self.assertEqual(ELFFile(fh).header["e_type"], "ET_REL")

    def test_unknown_func_lists_available(self):
        import contextlib
        import io

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(ValueError):
                bench(
                    file=self.exe,
                    target="no_such_func",
                    mode="latency",
                    config=dict(_FAST),
                )
        text = err.getvalue()
        self.assertIn("available targets", text)
        self.assertIn("func", text)
        self.assertIn("label", text)
        self.assertEqual(text.count("available targets"), 1)


class TestRegionCommands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp()
        cls.exe = _binary_with_labels(cls._tmp)
        if cls.exe is None:
            raise unittest.SkipTest("g++ unavailable")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_bench_region(self):
        df = bench(
            file=self.exe,
            target="hot_begin..hot_end",
            mode="latency",
            data={"rdi": 50},
            config=dict(_FAST),
        )
        self.assertFalse(df.empty)
        self.assertIn("duration_time", df.columns)

    def test_disassemble_symmetry(self):
        text = perf.disassemble(file=self.exe, target="hot_begin..hot_end")
        self.assertIn(".intel_syntax noprefix", text)

    def test_to_object_symmetry(self):
        out = os.path.join(self._tmp, "region_bench.o")
        perf.to_object(file=self.exe, target="hot_begin..hot_end", path=out)
        self.assertTrue(os.path.exists(out))

    def test_unknown_region_errors(self):
        with self.assertRaises(ValueError):
            bench(
                file=self.exe,
                target="nope_begin..nope_end",
                mode="latency",
                config=dict(_FAST),
            )


class TestInfoCommands(unittest.TestCase):
    def test_cpuinfo(self):
        df = perf.cpuinfo()
        self.assertFalse(df.empty)
        for col in ("cpu", "arch", "L1d", "L2", "L3"):
            self.assertIn(col, df.columns)

    def test_metadata_kinds(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        exe = _binary_with_labels(tmp)
        if exe is None:
            self.skipTest("g++ unavailable")
        full = perf.metadata(exe)
        self.assertIn("func", full["kind"].tolist())
        funcs = perf.metadata(exe, "func")
        self.assertTrue((funcs["kind"] == "func").all())
        regions = perf.metadata(exe, "region")
        self.assertFalse(regions.empty)
        with self.assertRaises(ValueError):
            perf.metadata(exe, "asm")


class TestViewPlotCommands(unittest.TestCase):
    def test_envelope_view_plot_roundtrip(self):
        df = bench(asm="nop", mode="latency", config=dict(_FAST))
        env = cli._record_envelope(df)
        self.assertIn("output", env)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "x.json").write_text(json.dumps(env))
            loaded = cli._load(SimpleNamespace(data=[tmp]))
        self.assertEqual(len(loaded), len(df))
        agg = cli._aggregate(
            loaded, ["file", "name", "mode"], ["duration_time"], ["min", "max"]
        )
        self.assertEqual(set(agg["stat"]), {"min", "max"})
        with patch("perf.plot.plt.show"):
            perf.plot(loaded, ["ecdf"], ["duration_time"])

    def test_cli_plot_config_symmetry(self):
        args = SimpleNamespace(config=None)
        cfg = cli._resolve_plot_config(args, ["--config.style=dark_background"])
        perf.plot.config = cfg
        self.assertEqual(perf.plot.config["style"], "dark_background")
        perf.plot.config = None


class TestCliBenchMapping(unittest.TestCase):
    def _asm_args(self, **kw):
        base = dict(
            asm="nop",
            bench_name="n",
            mode=["latency"],
            event=None,
            topdown=False,
            config={},
            data=None,
            setup=None,
            teardown=None,
            backend=None,
            interactive=False,
            output=None,
            json=False,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def _func_args(self, **kw):
        base = dict(
            file="a.out",
            target="foo",
            bench_name="lbl",
            mode=["latency"],
            event=None,
            topdown=False,
            config={},
            data=None,
            setup="s",
            teardown="t",
            backend=None,
            debug=False,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    @patch.object(cli, "_output_or_default")
    @patch.object(cli.perf, "bench")
    def test_asm_maps_setup_lists(self, mock_bench, mock_out):
        mock_bench.return_value = pd.DataFrame({"duration_time": [1.0], "samples": [0]})
        cli.bench_asm_cmd_measure(
            self._asm_args(setup="mov eax, 1", teardown="mov ebx, 2")
        )
        _, kwargs = mock_bench.call_args
        self.assertEqual(kwargs["setup"], ["mov eax, 1"])
        self.assertEqual(kwargs["teardown"], ["mov ebx, 2"])
        self.assertEqual(kwargs.get("asm", kwargs.get("code")), "nop")

    def test_maps_file_and_target(self):
        args = self._func_args()
        with patch.object(cli.perf, "bench") as mock_bench:
            mock_bench.return_value = (pd.DataFrame(), [["duration_time"]])
            cli._bench_measure(args)
        _, kwargs = mock_bench.call_args
        self.assertEqual(kwargs["file"], "a.out")
        self.assertEqual(kwargs["target"], "foo")
        self.assertEqual(kwargs["name"], "lbl")
        self.assertEqual(kwargs["setup"], "s")

    @patch.object(cli.perf, "bench")
    def test_region_passes_target(self, mock_bench):
        mock_bench.return_value = (pd.DataFrame(), [["duration_time"]])
        args = SimpleNamespace(
            file="a.out",
            target="hot_begin..hot_end",
            bench_name=None,
            mode=["latency"],
            event=None,
            topdown=False,
            config={},
            data=None,
            setup=None,
            teardown=None,
            backend=None,
            debug=False,
        )
        cli._bench_measure(args)
        _, kwargs = mock_bench.call_args
        self.assertEqual(kwargs["file"], "a.out")
        self.assertEqual(kwargs["target"], "hot_begin..hot_end")
        self.assertIsNone(kwargs["name"])

    @patch.object(cli.perf, "bench")
    def test_region_passes_hex_with_spaces(self, mock_bench):
        mock_bench.return_value = (pd.DataFrame(), [["duration_time"]])
        args = SimpleNamespace(
            file="a.out",
            target=" 0x1000 .. 0x2000 ",
            bench_name=None,
            mode=["latency"],
            event=None,
            topdown=False,
            config={},
            data=None,
            setup=None,
            teardown=None,
            backend=None,
            debug=False,
        )
        cli._bench_measure(args)
        _, kwargs = mock_bench.call_args
        self.assertEqual(kwargs["file"], "a.out")
        self.assertEqual(kwargs["target"], " 0x1000 .. 0x2000 ")
        self.assertIsNone(kwargs["name"])

    def test_asm_object_stage_needs_binary(self):
        with self.assertRaises(SystemExit):
            cli._stage_c(self._asm_args(), is_asm=True)

    def test_bench_extract_subcommands(self):
        self.assertEqual(cli._bench_subcommand(["bench", "asm", "nop"]), "bench")
        self.assertEqual(cli._bench_subcommand(["asm", "nop"]), "asm")
        self.assertEqual(cli._bench_subcommand(["region", "a..b"]), "region")
        self.assertEqual(cli._bench_subcommand(["func", "foo"]), "func")
        self.assertIsNone(cli._bench_subcommand(["--json"]))


class TestEnvelopeKeepsFileMode(unittest.TestCase):
    def test_keeps_file_and_mode_when_multiple(self):
        df = pd.DataFrame(
            {
                "file": ["a", "b"],
                "name": ["n1", "n2"],
                "mode": ["latency", "throughput"],
                "samples": [0, 0],
                "duration_time": [1.0, 2.0],
            }
        )
        env = cli._record_envelope(df)
        self.assertIn("file", env["output"][0])
        self.assertIn("name", env["output"][0])
        self.assertIn("mode", env["output"][0])


if __name__ == "__main__":
    unittest.main()
