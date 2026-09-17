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


def _cli():
    path = Path(__file__).resolve().parent.parent / "bin" / "perf"
    loader = importlib.machinery.SourceFileLoader("perfcli_view", str(path))
    spec = importlib.util.spec_from_loader("perfcli_view", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


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
        df, out = cli._eval_events(_df(), ["instructions/'base@1'.cycles"])
        self.assertEqual(out, ["instructions/'base@1'.cycles"])
        self.assertEqual(df[out[0]].tolist(), [2.0, 2.0, 0.5, 0.25])

    def test_baseline_ref_alone(self):
        df, out = cli._eval_events(_df(), ["'base@1'.cycles"])
        self.assertEqual(df[out[0]].tolist(), [10.0, 20.0, 10.0, 20.0])

    def test_name_fallback(self):
        df, out = cli._eval_events(_df(), ["instructions/'base'.cycles"])
        self.assertEqual(df[out[0]].tolist(), [2.0, 2.0, 0.5, 0.25])

    def test_double_quotes(self):
        df, out = cli._eval_events(_df(), ['instructions/"base@1".cycles'])
        self.assertEqual(df[out[0]].tolist(), [2.0, 2.0, 0.5, 0.25])

    def test_plain_expression_unchanged(self):
        df, out = cli._eval_events(_df(), ["cycles/instructions"])
        self.assertEqual(out, ["cycles/instructions"])
        self.assertEqual(df[out[0]].tolist(), [0.5, 0.25, 2.0, 4.0])

    def test_unknown_group_skipped(self):
        df, out = cli._eval_events(_df(), ["cycles/'nope'.cycles", "cycles"])
        self.assertEqual(out, ["cycles"])

    def test_unknown_metric_skipped(self):
        with self.assertRaises(SystemExit):
            cli._eval_events(_df(), ["cycles/'base@1'.nope"])

    def test_no_temp_columns_leaked(self):
        df, _ = cli._eval_events(_df(), ["instructions/'base@1'.cycles"])
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
        _, out = cli._eval_events(df, ["cycles/'c'.cycles"])
        self.assertEqual(
            _df_col(df, cli, out[0]),
            [0.8, 1.0, 1.2, 1.0, 10.0, 10.0, 10.0, 10.0],
        )

    def test_aggregate_speedup(self):
        df, evs = cli._eval_events(_df(), ["instructions/'base@1'.cycles"])
        out = cli._aggregate(df, ["file", "name", "mode"], evs, cli._split_list("min"))
        got = {(r["file"], r["stat"]): r[evs[0]] for _, r in out.iterrows()}
        self.assertAlmostEqual(got[("cur", "min")], 2.0)
        self.assertAlmostEqual(got[("base@1", "min")], 0.25)


def _df_col(df, cli_mod, expr):
    df2, out = cli_mod._eval_events(df, [expr])
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
        env = cli._envelope(self._df())
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
        self.assertEqual(env["mode"], "latency")
        self.assertEqual(env["data"], {"regs": {"rdi": 5}, "mem": {}})
        self.assertIsNone(env["code"])
        self.assertIsNone(env["state"])
        self.assertIsNone(env["distribution"])
        self.assertEqual(set(env["info"]), {"cpu"})
        self.assertEqual(env["config"], {"samples": 100})
        self.assertEqual(
            env["output"],
            [
                {"samples": 1, "duration_time": 2.0},
                {"samples": 99, "duration_time": 2.0},
            ],
        )

    def test_dirname(self):
        args = SimpleNamespace(exec=None)
        dname = cli._out_dirname(args, "fizz-95373155", "branch@abc12345")
        self.assertEqual(str(dname), "branch@abc12345/fizz-95373155")

    def test_load_reinflates(self):
        env = cli._envelope(self._df())
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "x.json").write_text(json.dumps(env))
            loaded = cli._load(SimpleNamespace(data=[tmp]))
        self.assertEqual(loaded["file"].tolist(), ["branch@abc12345"] * 2)
        self.assertEqual(loaded["name"].tolist(), [env["name"]] * 2)
        self.assertEqual(loaded["mode"].tolist(), ["latency"] * 2)
        self.assertTrue((loaded["time"] == env["time"]).all())

        payload = env["output"]
        self.assertEqual(set(payload[0]), {"samples", "duration_time"})


if __name__ == "__main__":
    unittest.main()
