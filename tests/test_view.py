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
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import perf
from perf.bench import _as_records, _record_envelope


def _cli():
    path = Path(__file__).resolve().parent.parent / "bin" / "perf-view"
    loader = importlib.machinery.SourceFileLoader("perfcli_view", str(path))
    spec = importlib.util.spec_from_loader("perfcli_view", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_BENCH = None


def _bench():
    global _BENCH
    if _BENCH is None:
        path = Path(__file__).resolve().parent.parent / "bin" / "perf-benchmark"
        loader = importlib.machinery.SourceFileLoader("perfcli_bench", str(path))
        spec = importlib.util.spec_from_loader("perfcli_bench", loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        _BENCH = module
    return _BENCH


cli = _cli()


def _df():
    return pd.DataFrame(
        {
            "file": ["cur", "cur", "base@1", "base@1"],
            "name": ["cur", "cur", "base", "base"],
            "mode": ["latency"] * 4,
            "samples": [0, 1, 0, 1],
            "instructions": [20.0, 40.0, 5.0, 5.0],
            "cycles": [10.0, 10.0, 10.0, 20.0],
        }
    )


class TestBaselineRefs(unittest.TestCase):
    def test_speedup_sample_aligned(self):
        df, out = cli.eval_events(_df(), ["instructions/'base@1'.cycles"])
        self.assertEqual(out, ["instructions/'base@1'.cycles"])
        self.assertEqual(df[out[0]].tolist(), [2.0, 2.0, 0.5, 0.25])

    def test_baseline_ref_alone(self):
        df, out = cli.eval_events(_df(), ["'base@1'.cycles"])
        self.assertEqual(df[out[0]].tolist(), [10.0, 20.0, 10.0, 20.0])

    def test_name_fallback(self):
        df, out = cli.eval_events(_df(), ["instructions/'base'.cycles"])
        self.assertEqual(df[out[0]].tolist(), [2.0, 2.0, 0.5, 0.25])

    def test_double_quotes(self):
        df, out = cli.eval_events(_df(), ['instructions/"base@1".cycles'])
        self.assertEqual(df[out[0]].tolist(), [2.0, 2.0, 0.5, 0.25])

    def test_plain_expression_unchanged(self):
        df, out = cli.eval_events(_df(), ["cycles/instructions"])
        self.assertEqual(out, ["cycles/instructions"])
        self.assertEqual(df[out[0]].tolist(), [0.5, 0.25, 2.0, 4.0])

    def test_unknown_group_skipped(self):
        df, out = cli.eval_events(_df(), ["cycles/'nope'.cycles", "cycles"])
        self.assertEqual(out, ["cycles"])

    def test_unknown_metric_skipped(self):
        with self.assertRaises(SystemExit):
            cli.eval_events(_df(), ["cycles/'base@1'.nope"])

    def test_no_temp_columns_leaked(self):
        df, _ = cli.eval_events(_df(), ["instructions/'base@1'.cycles"])
        self.assertFalse([c for c in df.columns if "perf_base" in c])

    def test_no_temp_columns_leaked_on_failure(self):
        df, out = cli.eval_events(_df(), ["instructions/'nope'.cycles", "cycles"])
        self.assertEqual(out, ["cycles"])
        self.assertFalse([c for c in df.columns if "perf_base" in c])

    def test_duplicate_samples_use_mean(self):
        df = pd.DataFrame(
            {
                "file": ["c"] * 4 + ["d"] * 4,
                "name": ["c"] * 4 + ["d"] * 4,
                "mode": ["latency"] * 8,
                "samples": [0, 1, 0, 1, 0, 1, 0, 1],
                "cycles": [8.0, 10.0, 12.0, 10.0, 100.0, 100.0, 100.0, 100.0],
            }
        )
        _, out = cli.eval_events(df, ["cycles/'c'.cycles"])
        self.assertEqual(
            _df_col(df, out[0]),
            [0.8, 1.0, 1.2, 1.0, 10.0, 10.0, 10.0, 10.0],
        )

    def test_aggregate_speedup(self):
        from perf.core import _split_list

        df, evs = cli.eval_events(_df(), ["instructions/'base@1'.cycles"])
        out = cli._aggregate(df, ["file", "name", "mode"], evs, _split_list("min"))
        got = {(r["file"], r["stat"]): r[evs[0]] for _, r in out.iterrows()}
        self.assertAlmostEqual(got[("cur", "min")], 2.0)
        self.assertAlmostEqual(got[("base@1", "min")], 0.25)

    def test_aggregate_groups_by_mode(self):
        from perf.core import _split_list

        df = pd.DataFrame(
            {
                "file": ["b"] * 4,
                "name": ["f"] * 4,
                "mode": ["throughput", "latency", "throughput", "latency"],
                "samples": [0, 0, 1, 1],
                "duration_time": [100.0, 10.0, 110.0, 12.0],
            }
        )
        out = cli._aggregate(
            df, ["file", "name"], ["duration_time"], _split_list("min")
        )
        self.assertIn("mode", out.columns)
        got = {r["mode"]: r["duration_time"] for _, r in out.iterrows()}
        self.assertAlmostEqual(got["latency"], 10.0)
        self.assertAlmostEqual(got["throughput"], 100.0)

    def test_format_table_durations_per_operation(self):
        df = pd.DataFrame(
            {
                "file": ["b"],
                "name": ["f"],
                "mode": ["latency"],
                "stat": ["min"],
                "duration_time": [8.93],
                "duration_time/operations": [8.93],
            }
        )
        out = cli.format_table(df)
        self.assertTrue(str(out["duration_time"].iloc[0]).endswith("ns"))
        self.assertTrue(str(out["duration_time/operations"].iloc[0]).endswith("ns"))


def _df_col(df, expr):
    df2, out = cli.eval_events(df, [expr])
    return df2[out[0]].tolist()


class TestEnvelope(unittest.TestCase):
    def _df(self):
        df = pd.DataFrame(
            {
                "file": ["branch@abc12345"] * 2,
                "name": ["fizz"] * 2,
                "mode": ["latency"] * 2,
                "samples": [1, 99],
                "duration_time": [2.0, 2.0],
            }
        )
        df = df.set_index(["file", "name", "mode"])
        df.attrs["data"] = {"regs": {"rdi": 5}, "mem": {}}
        df.attrs["config"] = {"samples": 100}
        df.attrs["info"] = {
            "cpu": {"arch": "x86_64"},
            "binary": {"path": "branch", "sha": "abc12345"},
        }
        return df

    def test_envelope_structure(self):
        env = _record_envelope(self._df())
        self.assertEqual(
            list(env),
            [
                "file",
                "name",
                "id",
                "time",
                "info",
                "output",
            ],
        )
        self.assertIn("id", env)
        self.assertNotIn("results", env)
        self.assertEqual(env["file"], "branch@abc12345")
        from perf.bench import _id_hash

        df = self._df()
        expected_id = _id_hash(
            df.attrs["data"], df.attrs["config"], df.attrs["info"]["binary"]
        )
        self.assertEqual(env["id"], expected_id)
        self.assertEqual(env["name"], f"fizz-{expected_id}")
        self.assertEqual(set(env["info"]), {"cpu"})
        self.assertEqual(
            env["output"],
            [
                {"mode": "latency", "samples": 1, "duration_time": 2.0},
                {"mode": "latency", "samples": 99, "duration_time": 2.0},
            ],
        )

    def test_dirname(self):
        dname = _bench().out_dirname("fizz-95373155", "branch@abc12345")
        self.assertEqual(str(dname), "branch@abc12345/fizz-95373155")

    def test_load_reinflates(self):
        env = _record_envelope(self._df())
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "x.json").write_text(json.dumps(env))
            loaded = cli.load(SimpleNamespace(data=[tmp]))
        self.assertEqual(loaded["file"].tolist(), ["branch@abc12345"] * 2)
        self.assertEqual(loaded["name"].tolist(), [env["name"]] * 2)
        self.assertEqual(loaded["mode"].tolist(), ["latency"] * 2)
        self.assertTrue((loaded["time"] == env["time"]).all())

        payload = env["output"]
        self.assertEqual(set(payload[0]), {"mode", "samples", "duration_time"})


def _reject_constant(name):
    raise ValueError(f"non-JSON constant {name!r} in the envelope")


class TestOutputRoundTrip(unittest.TestCase):
    _SERIES = (
        ("latency", "config.backend.unroll.count", 5, 3.0, 1),
        ("latency", "config.backend.unroll.count", 5, 3.0, 1),
        ("throughput", "config.backend.loop.runs", 10, 3000.0, 1000),
        ("throughput", "config.backend.loop.runs", 10, 2600.0, 1000),
    )

    def _frame(self, series):
        rows = []
        for i, (mode, col, value, cycles, iterations) in enumerate(series):
            rows.append(
                {
                    "file": None,
                    "name": "imul eax, 42",
                    "mode": mode,
                    col: value,
                    "config.dcache": "hot",
                    "config.dtlb.hit_rate": 50,
                    "iterations": iterations,
                    "samples": i,
                    "operations": iterations,
                    "cycles": cycles,
                }
            )
        return pd.DataFrame(rows).set_index(["file", "name", "mode"])

    def _df(self):
        df = pd.concat([self._frame(self._SERIES[:2]), self._frame(self._SERIES[2:])])
        df.attrs["data"] = {"regs": {}, "mem": {}}
        df.attrs["config"] = {"dcache": "hot", "dtlb": {"hit_rate": 50}}
        df.attrs["info"] = {"cpu": {"arch": "x86_64"}}
        return df

    def _assert_same(self, got):
        want = self._df().reset_index()
        self.assertEqual(len(got), len(want))
        self.assertEqual(got["mode"].tolist(), want["mode"].tolist())
        for col in ("iterations", "operations", "cycles"):
            self.assertEqual(got[col].astype(float).tolist(), want[col].tolist(), col)
        for col in (
            "config.backend.unroll.count",
            "config.backend.loop.runs",
            "config.dcache",
            "config.dtlb.hit_rate",
        ):
            self._assert_column(got[col].tolist(), want[col].tolist(), col)
        self.assertNotIn("config.backend_options", got.columns)

    def _assert_column(self, got, want, col):
        for a, b in zip(got, want):
            if pd.isna(b):
                self.assertTrue(pd.isna(a), f"{col}: {a!r} != {b!r}")
            else:
                self.assertEqual(a, b, col)

    def test_table_round_trip(self):
        rec = _as_records(self._df())
        text = cli.format_table(rec).to_string(index=False)
        self._assert_same(cli.table_to_df(text))

    def test_envelope_round_trip(self):
        df = self._df()
        env = json.loads(perf.to_json(df))
        self.assertEqual(
            env["output"][0]["config"]["backend"], {"unroll": {"count": 5.0}}
        )
        self.assertEqual(
            env["output"][2]["config"]["backend"], {"loop": {"runs": 10.0}}
        )
        self.assertEqual(env["output"][0]["config"]["dtlb"], {"hit_rate": 50})
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "x.json").write_text(json.dumps(env))
            loaded = cli.load(SimpleNamespace(data=[tmp]))
        self._assert_same(loaded)

    def test_unread_counters_are_null_not_zero(self):
        df = self._df()
        df["cycles"] = [3.0, float("nan"), float("nan"), 2600.0]
        env = json.loads(perf.to_json(df))
        self.assertEqual(
            [row["cycles"] for row in env["output"]], [3.0, None, None, 2600.0]
        )
        text = perf.to_json(df)
        json.loads(text, parse_constant=_reject_constant)


if __name__ == "__main__":
    unittest.main()
