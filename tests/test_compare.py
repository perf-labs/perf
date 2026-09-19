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
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from perf.compare import clt_test, compare, gmean_ztest, norm_cdf, z_critical, ztest


def _cli():
    path = Path(__file__).resolve().parent.parent / "bin" / "perf"
    loader = importlib.machinery.SourceFileLoader("perfcli_cmp", str(path))
    spec = importlib.util.spec_from_loader("perfcli_cmp", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


cli = _cli()


def _df(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(100.0, 5.0, 100)
    same = rng.normal(100.0, 5.0, 100)
    faster = rng.normal(80.0, 5.0, 100)
    return pd.DataFrame(
        {
            "file": ["a.out@hash"] * 300,
            "name": ["base"] * 100 + ["same"] * 100 + ["fast"] * 100,
            "mode": ["latency"] * 300,
            "samples": list(range(100)) * 3,
            "cycles": np.concatenate([base, same, faster]),
        }
    )


def _file_df(seed=0):
    df = _df(seed=seed)
    df = df.copy()
    df["file"] = (["base"] * 100 + ["same"] * 100 + ["fast"] * 100).copy()
    df["name"] = "f"
    return df


class TestNorm(unittest.TestCase):
    def test_cdf(self):
        self.assertAlmostEqual(norm_cdf(0.0), 0.5)
        self.assertAlmostEqual(norm_cdf(1.96), 0.975, places=3)
        self.assertAlmostEqual(norm_cdf(-1.96), 0.025, places=3)

    def test_critical(self):
        self.assertAlmostEqual(z_critical(0.05), 1.96, places=2)
        self.assertGreater(z_critical(0.01), z_critical(0.05))
        with self.assertRaises(ValueError):
            z_critical(0.0)
        with self.assertRaises(ValueError):
            z_critical(1.5)


class TestZTest(unittest.TestCase):
    def test_equal_means_not_significant(self):
        rng = np.random.default_rng(1)
        a = rng.normal(50.0, 2.0, 200)
        b = rng.normal(50.0, 2.0, 200)
        r = ztest(a, b)
        self.assertAlmostEqual(r["diff"], 0.0, delta=1.0)
        self.assertGreater(r["p_value"], 0.05)
        self.assertFalse(r["significant"])

    def test_shifted_means_significant(self):
        rng = np.random.default_rng(2)
        a = rng.normal(100.0, 5.0, 200)
        b = rng.normal(110.0, 5.0, 200)
        r = ztest(a, b)
        self.assertAlmostEqual(r["diff"], 10.0, delta=2.0)
        self.assertLess(r["p_value"], 0.05)
        self.assertTrue(r["significant"])

    def test_identical_constants(self):
        r = ztest([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        self.assertEqual(r["diff"], 0.0)
        self.assertEqual(r["p_value"], 1.0)
        self.assertFalse(r["significant"])

    def test_different_constants(self):
        r = ztest([5.0, 5.0, 5.0], [6.0, 6.0, 6.0])
        self.assertEqual(r["p_value"], 0.0)
        self.assertTrue(r["significant"])

    def test_nan_guarded(self):
        r = ztest([], [])
        self.assertTrue(math.isnan(r["p_value"]))
        self.assertFalse(r["significant"])

    def test_diff_pct(self):
        r = ztest([100.0] * 10, [110.0] * 10)
        self.assertAlmostEqual(r["diff_pct"], 10.0)


class TestGmeanZTest(unittest.TestCase):
    def test_equal_geomeans_not_significant(self):
        rng = np.random.default_rng(1)
        a = rng.lognormal(4.0, 0.2, 200)
        b = rng.lognormal(4.0, 0.2, 200)
        r = gmean_ztest(a, b)
        self.assertGreater(r["p_value"], 0.05)
        self.assertFalse(r["significant"])

    def test_shifted_geomeans_significant(self):
        rng = np.random.default_rng(2)
        a = rng.lognormal(4.0, 0.1, 200)
        b = rng.lognormal(4.2, 0.1, 200)
        r = gmean_ztest(a, b)
        self.assertLess(r["p_value"], 0.05)
        self.assertTrue(r["significant"])
        self.assertAlmostEqual(r["diff_pct"], (np.e**0.2 - 1) * 100.0, delta=5.0)

    def test_non_positive_guarded(self):
        r = gmean_ztest([1.0, 2.0], [0.0, -1.0])
        self.assertTrue(math.isnan(r["p_value"]))
        self.assertFalse(r["significant"])

    def test_outlier_resistant(self):
        rng = np.random.default_rng(3)
        a = rng.normal(100.0, 2.0, 100)
        b = rng.normal(100.0, 2.0, 100)
        b = np.append(b, [1000.0])
        r_mean = ztest(a, b)
        r = clt_test(a, b)
        self.assertGreater(abs(r_mean["diff"]), 5.0)
        self.assertFalse(r["significant"])
        self.assertLess(abs(r["diff"]), 5.0)


class TestCltTest(unittest.TestCase):
    def test_same_distribution_not_significant(self):
        rng = np.random.default_rng(11)
        a = rng.normal(100.0, 5.0, 100)
        b = rng.normal(100.0, 5.0, 100)
        r = clt_test(a, b)
        self.assertGreater(r["p_value"], 0.05)
        self.assertFalse(r["significant"])

    def test_shifted_distribution_significant(self):
        rng = np.random.default_rng(12)
        a = rng.normal(100.0, 5.0, 100)
        b = rng.normal(120.0, 5.0, 100)
        r = clt_test(a, b)
        self.assertLess(r["p_value"], 0.05)
        self.assertTrue(r["significant"])
        self.assertAlmostEqual(r["diff"], 20.0, delta=3.0)

    def test_conservative_combination(self):
        rng = np.random.default_rng(13)
        a = rng.normal(100.0, 5.0, 100)
        b = rng.normal(100.0, 5.0, 100)
        r = clt_test(a, b)
        self.assertGreaterEqual(r["p_value"], r["p_mean"])
        self.assertGreaterEqual(r["p_value"], r["p_gmean"])

    def test_single_outlier_not_significant(self):
        rng = np.random.default_rng(14)
        a = rng.normal(100.0, 2.0, 100)
        b = rng.normal(100.0, 2.0, 100)
        b = np.append(b, [500.0])
        r = clt_test(a, b)
        self.assertFalse(r["significant"])


class TestStatEdgeCases(unittest.TestCase):
    def test_norm_cdf_bad_input(self):
        self.assertTrue(math.isnan(norm_cdf("nope")))
        self.assertTrue(math.isnan(norm_cdf(None)))

    def test_ndtri_invalid_and_tails(self):
        from perf.compare import _ndtri

        with self.assertRaises(ValueError):
            _ndtri(0.0)
        with self.assertRaises(ValueError):
            _ndtri(1.0)
        self.assertLess(_ndtri(0.001), -3.0)
        self.assertGreater(_ndtri(0.999), 3.0)

    def test_clean_rejects_non_numeric(self):
        from perf.compare import _clean

        self.assertEqual(_clean([["a"]]).size, 0)
        self.assertEqual(_clean(object()).size, 0)

    def test_clean_isfinite_failure(self):
        from perf.compare import _clean

        with patch("numpy.isfinite", side_effect=RuntimeError("boom")):
            self.assertEqual(_clean([1.0, 2.0]).size, 0)

    def test_summarize_nan_std_guarded(self):
        from perf.compare import _summarize

        with patch("perf.compare._clean", return_value=np.array([np.nan, np.nan])):
            s = _summarize([1.0, 2.0])
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["std"], 0.0)

    def test_ztest_from_stats_bad_inputs(self):
        from perf.compare import _ztest_from_stats

        r = _ztest_from_stats("a", 1.0, "b", 1.0)
        self.assertTrue(math.isnan(r["diff"]))
        r = _ztest_from_stats(0.0, "x", 0.0, "y")
        self.assertTrue(math.isnan(r["se_diff"]))

    def test_diff_ci_bad_alpha(self):
        from perf.compare import _diff_ci

        lo, hi = _diff_ci(1.0, 0.5, alpha="bad")
        self.assertTrue(math.isnan(lo) and math.isnan(hi))

    def test_diff_ci_overflow_guarded(self):
        from perf.compare import _diff_ci

        with patch("perf.compare.z_critical", return_value="x"):
            lo, hi = _diff_ci(1.0, 0.5)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))

    def test_pct_ci_edges(self):
        from perf.compare import _pct_ci_from_log

        lo, hi = _pct_ci_from_log("a", "b")
        self.assertTrue(math.isnan(lo))
        lo, hi = _pct_ci_from_log(float("nan"), 0.5)
        self.assertTrue(math.isnan(lo))
        self.assertEqual(_pct_ci_from_log(0.0, 0.0), (0.0, 0.0))
        lo, hi = _pct_ci_from_log(1000.0, 1e-10)
        self.assertTrue(math.isnan(lo))
        with patch("perf.compare.z_critical", return_value=float("nan")):
            lo, hi = _pct_ci_from_log(0.5, 0.5)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))

    def test_ztest_bad_alpha(self):
        r = ztest([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], alpha="bad")
        self.assertFalse(r["significant"])

    def test_gmean_overflow_and_bad_alpha(self):
        with patch("perf.compare.math.exp", side_effect=OverflowError("boom")):
            r = gmean_ztest([1.0, 2.0], [2.0, 3.0])
        self.assertTrue(math.isnan(r["gmean_a"]))
        r = gmean_ztest([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], alpha="bad")
        self.assertFalse(r["significant"])

    def test_clt_partial_and_empty(self):
        r = clt_test([1.0, 2.0, 3.0], [0.0, 3.0])
        self.assertTrue(math.isfinite(r["p_mean"]))
        self.assertTrue(math.isnan(r["p_gmean"]))
        r = clt_test([], [])
        self.assertTrue(math.isnan(r["p_value"]))
        self.assertFalse(r["significant"])

    def test_clt_zero_mean_and_bad_alpha(self):
        r = clt_test([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertFalse(r["significant"])
        r = clt_test([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], alpha="bad")
        self.assertFalse(r["significant"])

    def test_resolve_baseline_edges(self):
        from perf.compare import _resolve_baseline

        with self.assertRaises(ValueError):
            _resolve_baseline([], None)
        self.assertEqual(
            _resolve_baseline({"a/foo": True, "bar": True}, "x/foo"), "a/foo"
        )
        with patch("pathlib.Path", side_effect=RuntimeError("boom")):
            with self.assertRaises(ValueError):
                _resolve_baseline({"a": True}, "b/c")

    def test_can_compare_failure(self):
        from perf.compare import _can_compare

        with patch("perf.compare._clean", side_effect=RuntimeError("boom")):
            self.assertFalse(_can_compare([1.0], [2.0]))

    def test_compare_validation(self):
        with self.assertRaises(ValueError):
            compare("not-a-frame")
        df = pd.DataFrame({"mode": ["latency", "latency"], "cycles": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            compare(df)
        out = compare(_df(), events="cycles")
        self.assertGreater(len(out), 0)
        out = compare(_df(), on="name,mode")
        self.assertGreater(len(out), 0)

    def test_compare_label_fallback(self):
        with patch("perf.compare._text", side_effect=RuntimeError("boom")):
            out = compare(_df())
        self.assertGreater(len(out), 0)

    def test_compare_mode_branch(self):
        out = compare(_df(), on="name")
        self.assertGreater(len(out), 0)
        self.assertIn("latency", out["mode"].tolist())

    def test_compare_disjoint_modes_raises(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "file": ["a"] * 40,
                "name": ["base"] * 20 + ["fast"] * 20,
                "mode": ["latency"] * 20 + ["throughput"] * 20,
                "samples": list(range(20)) * 2,
                "cycles": np.concatenate(
                    [rng.normal(100, 5, 20), rng.normal(80, 5, 20)]
                ),
            }
        )
        with self.assertRaises(ValueError):
            compare(df, events=["cycles"], on="name")

    def test_compare_nan_mode_slice_skipped(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame(
            {
                "file": ["a"] * 80,
                "name": ["base"] * 40 + ["fast"] * 40,
                "mode": (["latency"] * 20 + ["throughput"] * 20) * 2,
                "samples": list(range(20)) * 4,
                "cycles": np.concatenate(
                    [
                        np.full(20, np.nan),
                        rng.normal(100, 5, 20),
                        rng.normal(100, 5, 20),
                        rng.normal(80, 5, 20),
                    ]
                ),
            }
        )
        out = compare(df, events=["cycles"], on="name")
        self.assertEqual(sorted(out["mode"].unique().tolist()), ["throughput"])


class TestCompare(unittest.TestCase):
    def test_baseline_default_first_sorted(self):
        out = compare(_df(), events=["cycles"])
        self.assertEqual(out["baseline"].unique().tolist(), ["base"])
        self.assertEqual(sorted(out["challenger"].unique().tolist()), ["fast", "same"])
        self.assertEqual(out["mode"].unique().tolist(), ["latency"])
        self.assertNotIn("name", out.columns)

    def test_same_not_significant_fast_significant(self):
        out = compare(_df(), events=["cycles"], baseline="base")
        by_challenger = {r["challenger"]: r for _, r in out.iterrows()}
        self.assertFalse(by_challenger["same"]["significant"])
        self.assertGreater(by_challenger["same"]["p_value"], 0.05)
        fast = by_challenger["fast"]
        self.assertTrue(fast["significant"])
        self.assertLess(fast["p_value"], 0.05)
        self.assertAlmostEqual(fast["diff"], -20.0, delta=3.0)

    def test_repeated_same_runs_stable(self):
        rng = np.random.default_rng(42)
        frames = []
        for name in ("run0", "run1", "run2"):
            frames.append(
                pd.DataFrame(
                    {
                        "file": ["a.out@hash"] * 100,
                        "name": [name] * 100,
                        "mode": ["latency"] * 100,
                        "samples": list(range(100)),
                        "cycles": rng.normal(100.0, 5.0, 100),
                    }
                )
            )
        out = compare(pd.concat(frames), events=["cycles"], baseline="run0")
        self.assertFalse(out["significant"].any())

    def test_explicit_baseline(self):
        out = compare(_df(), events=["cycles"], baseline="fast")
        self.assertEqual(out["baseline"].unique().tolist(), ["fast"])
        self.assertEqual(sorted(out["challenger"].unique().tolist()), ["base", "same"])

    def test_unknown_baseline_raises(self):
        with self.assertRaises(ValueError):
            compare(_df(), events=["cycles"], baseline="nope")

    def test_single_variant_raises(self):
        df = _df().query('name == "base"')
        with self.assertRaises(ValueError):
            compare(df, events=["cycles"])

    def test_file_variant_grouped_by_name(self):
        out = compare(_file_df(), events=["cycles"], baseline="base", variant="file")
        by_challenger = {r["challenger"]: r for _, r in out.iterrows()}
        self.assertNotIn("name", out.columns)
        self.assertFalse(by_challenger["same"]["significant"])
        self.assertTrue(by_challenger["fast"]["significant"])

    def test_file_variant_unrelated_names_never_compared(self):
        df = _file_df()
        df = df.copy()
        df.loc[df["file"] == "fast", "name"] = "other"
        out = compare(df, events=["cycles"], baseline="base", variant="file")
        self.assertEqual(out["challenger"].unique().tolist(), ["same"])

    def test_file_variant_unrelated_modes_never_compared(self):
        df = _file_df()
        df = df.copy()
        df.loc[df["file"] == "fast", "mode"] = "throughput"
        out = compare(df, events=["cycles"], baseline="base", variant="file")
        self.assertEqual(out["challenger"].unique().tolist(), ["same"])
        self.assertEqual(out["mode"].unique().tolist(), ["latency"])

    def test_per_mode_partitioning(self):
        rng = np.random.default_rng(9)
        df = pd.DataFrame(
            {
                "file": ["a.out@hash"] * 200,
                "name": ["a"] * 50 + ["b"] * 50 + ["a"] * 50 + ["b"] * 50,
                "mode": ["latency"] * 100 + ["throughput"] * 100,
                "samples": list(range(50)) * 4,
                "cycles": np.concatenate(
                    [
                        rng.normal(100.0, 5.0, 50),
                        rng.normal(100.0, 5.0, 50),
                        rng.normal(100.0, 5.0, 50),
                        rng.normal(200.0, 5.0, 50),
                    ]
                ),
            }
        )
        out = compare(df, events=["cycles"], baseline="a")
        by_mode = {r["mode"]: r for _, r in out.iterrows()}
        self.assertEqual(sorted(by_mode), ["latency", "throughput"])
        self.assertFalse(by_mode["latency"]["significant"])
        self.assertTrue(by_mode["throughput"]["significant"])

    def test_no_cooccurring_baseline_raises(self):
        df = _df()
        df = df.copy()
        df.loc[df["name"] != "base", "mode"] = "throughput"
        with self.assertRaises(ValueError):
            compare(df, events=["cycles"], baseline="base")

    def test_unknown_event_raises(self):
        with self.assertRaises(ValueError):
            compare(_df(), events=["nope"])

    def test_alpha_validation(self):
        with self.assertRaises(ValueError):
            compare(_df(), events=["cycles"], alpha=0.0)

    def test_missing_mode_matches_every_mode(self):
        rng = np.random.default_rng(31)
        df = pd.DataFrame(
            {
                "file": ["a.out@hash"] * 200,
                "name": ["bench"] * 100 + ["track"] * 100,
                "mode": ["latency"] * 50 + ["throughput"] * 50 + [None] * 100,
                "samples": list(range(50)) * 2 + list(range(100)),
                "cycles": rng.normal(100.0, 5.0, 200),
            }
        )
        out = compare(df, events=["cycles"], baseline="bench")
        self.assertEqual(
            sorted(out["mode"].unique().tolist()), ["latency", "throughput"]
        )
        self.assertEqual(out["baseline"].unique().tolist(), ["bench"])
        self.assertEqual(out["challenger"].unique().tolist(), ["track"])

    def test_all_modes_missing_compares_together(self):
        df = _df()
        df = df.copy()
        df["mode"] = None
        out = compare(df, events=["cycles"], baseline="base")
        self.assertEqual(sorted(out["challenger"].unique().tolist()), ["fast", "same"])

    def test_columns(self):
        out = compare(_df(), events=["cycles"])
        self.assertEqual(
            out.columns.tolist(),
            [
                "baseline",
                "challenger",
                "mode",
                "event",
                "diff",
                "ci_low",
                "ci_high",
                "p_value",
                "alpha",
                "significant",
                "result",
            ],
        )

    def test_result_labels(self):
        out = compare(_df(), events=["cycles"])
        for _, r in out.iterrows():
            if bool(r["significant"]):
                self.assertTrue(
                    str(r["result"]).startswith("faster")
                    or str(r["result"]).startswith("slower")
                )
            else:
                self.assertEqual(r["result"], "insignificant")

    def test_significant_matches_p_value_vs_alpha(self):
        out = compare(_df(), events=["cycles"], baseline="base")
        for _, r in out.iterrows():
            self.assertEqual(
                bool(r["significant"]), bool(float(r["p_value"]) < float(r["alpha"]))
            )
            self.assertAlmostEqual(float(r["alpha"]), 0.05)

    def test_ci_contains_diff_and_is_ordered(self):
        out = compare(_df(), events=["cycles"], baseline="base")
        for _, r in out.iterrows():
            self.assertLessEqual(r["ci_low"], r["diff"])
            self.assertLessEqual(r["diff"], r["ci_high"])

    def test_ci_level_follows_alpha(self):
        rng = np.random.default_rng(0)
        a = rng.normal(100.0, 5.0, 200)
        b = rng.normal(110.0, 5.0, 200)
        wide = clt_test(a, b, alpha=0.05)
        narrow = clt_test(a, b, alpha=0.01)
        self.assertLessEqual(wide["ci_low"], wide["diff"])
        self.assertLessEqual(wide["diff"], wide["ci_high"])
        self.assertLessEqual(narrow["ci_low"], wide["ci_low"])
        self.assertGreaterEqual(narrow["ci_high"], wide["ci_high"])

    def test_incomparable_modes_skipped(self):
        df = _df()
        df = df.copy()
        df.loc[df["name"] == "fast", "mode"] = "throughput"
        out = compare(df, events=["cycles"], baseline="base")
        self.assertEqual(out["challenger"].unique().tolist(), ["same"])
        self.assertEqual(out["mode"].unique().tolist(), ["latency"])

    def test_incomparable_empty_event_skipped(self):
        df = _df()
        df = df.copy()
        df["cycles"] = float("nan")
        with self.assertRaises(ValueError):
            compare(df, events=["cycles"], baseline="base")


class TestCompareCmd(unittest.TestCase):
    def test_compare_cmd_exists(self):
        self.assertTrue(callable(cli.compare_cmd))

    def test_variant_prefers_names(self):
        self.assertEqual(cli._compare_variant(_df()), "name")
        self.assertEqual(cli._compare_variant(_file_df()), "file")

    def _envelope(self, file_label, values, name="f", mode="latency", operations=1):
        return {
            "file": file_label,
            "name": name,
            "id": "x",
            "time": "2026-01-01 00:00:00",
            "info": {"cpu": {}},
            "config": {},
            "mode": mode,
            "data": None,
            "code": None,
            "state": None,
            "distribution": None,
            "output": [
                {"samples": i, "operations": operations, "cycles": v}
                for i, v in enumerate(values)
            ],
        }

    def _args(self, tmp, **kw):
        base = dict(
            data=list(tmp) if isinstance(tmp, (list, tuple)) else [tmp],
            event=kw.get("event", ["cycles"]),
            filter=kw.get("filter", None),
            baseline=kw.get("baseline", None),
            alpha=kw.get("alpha", 0.05),
            json=kw.get("json", False),
            interactive=False,
        )
        return SimpleNamespace(**base)

    def test_table_output(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(
                json.dumps(self._envelope("a", rng.normal(100, 5, 50).tolist()))
            )
            Path(tmp, "b.json").write_text(
                json.dumps(self._envelope("b", rng.normal(120, 5, 50).tolist()))
            )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.compare_cmd(self._args(tmp))
            text = buf.getvalue()
        self.assertIn("challenger", text)
        self.assertIn("p_value", text)

    def test_baseline_flag(self):
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(
                json.dumps(self._envelope("a", rng.normal(100, 5, 50).tolist()))
            )
            Path(tmp, "b.json").write_text(
                json.dumps(self._envelope("b", rng.normal(120, 5, 50).tolist()))
            )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.compare_cmd(self._args(tmp, baseline="b", json=True))
            records = json.loads(buf.getvalue())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["baseline"], "b")
        self.assertEqual(records[0]["challenger"], "a")

    def test_named_runs_compared_by_name(self):
        rng = np.random.default_rng(13)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "base.json").write_text(
                json.dumps(
                    self._envelope(
                        "a.out@hash", rng.normal(100, 5, 60).tolist(), name="base"
                    )
                )
            )
            Path(tmp, "fast.json").write_text(
                json.dumps(
                    self._envelope(
                        "a.out@hash", rng.normal(80, 5, 60).tolist(), name="fast"
                    )
                )
            )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.compare_cmd(self._args(tmp, json=True))
            records = json.loads(buf.getvalue())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["baseline"], "base")
        self.assertEqual(records[0]["challenger"], "fast")
        self.assertTrue(records[0]["significant"])

    def test_json_output(self):
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(
                json.dumps(self._envelope("a", rng.normal(100, 5, 50).tolist()))
            )
            Path(tmp, "b.json").write_text(
                json.dumps(self._envelope("b", rng.normal(100, 5, 50).tolist()))
            )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.compare_cmd(self._args(tmp, json=True))
            records = json.loads(buf.getvalue())
        self.assertEqual(len(records), 1)
        self.assertIn("p_value", records[0])
        self.assertIn("significant", records[0])
        self.assertIn("ci_low", records[0])
        self.assertIn("ci_high", records[0])
        self.assertIn("alpha", records[0])
        self.assertIn("diff", records[0])

    def test_duration_per_op_compared_by_default(self):
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as tmp:
            for name, vals in (
                ("a", rng.normal(100, 5, 50)),
                ("b", rng.normal(120, 5, 50)),
            ):
                Path(tmp, f"{name}.json").write_text(
                    json.dumps(
                        {
                            "file": name,
                            "name": "f",
                            "id": "x",
                            "time": "2026-01-01 00:00:00",
                            "info": {"cpu": {}},
                            "config": {},
                            "mode": "latency",
                            "data": None,
                            "code": None,
                            "state": None,
                            "distribution": None,
                            "output": [
                                {
                                    "samples": i,
                                    "operations": 2,
                                    "duration_time": float(v),
                                }
                                for i, v in enumerate(vals)
                            ],
                        }
                    )
                )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.compare_cmd(self._args(tmp, json=True, event=None))
            records = json.loads(buf.getvalue())
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {r["event"] for r in records},
            {"duration_time", "duration_time/operations"},
        )

    def test_expression_event(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(
                json.dumps(
                    self._envelope("a", rng.normal(100, 5, 60).tolist(), operations=10)
                )
            )
            Path(tmp, "b.json").write_text(
                json.dumps(
                    self._envelope("b", rng.normal(80, 5, 60).tolist(), operations=10)
                )
            )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.compare_cmd(self._args(tmp, event=["cycles/operations"], json=True))
            records = json.loads(buf.getvalue())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"], "cycles/operations")
        self.assertEqual(records[0]["baseline"], "a")
        self.assertEqual(records[0]["challenger"], "b")
        self.assertTrue(records[0]["significant"])

    def test_same_file_label_compared_by_source(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp, "a.json")
            base.write_text(
                json.dumps(
                    self._envelope("branch@b04f8237", rng.normal(100, 5, 60).tolist())
                )
            )
            Path(tmp, "b.json").write_text(
                json.dumps(
                    self._envelope("branch@b04f8237", rng.normal(120, 5, 60).tolist())
                )
            )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.compare_cmd(self._args(tmp, json=True))
            records = json.loads(buf.getvalue())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["baseline"], "a.json")
        self.assertEqual(records[0]["challenger"], "b.json")

    def test_distinct_file_labels_use_file_variant(self):
        rng = np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(
                json.dumps(
                    self._envelope("branch@aaa", rng.normal(100, 5, 60).tolist())
                )
            )
            Path(tmp, "b.json").write_text(
                json.dumps(
                    self._envelope("branch@bbb", rng.normal(120, 5, 60).tolist())
                )
            )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.compare_cmd(self._args(tmp, json=True))
            records = json.loads(buf.getvalue())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["baseline"], "branch@aaa")
        self.assertEqual(records[0]["challenger"], "branch@bbb")

    def _records_df(self, seed=20):
        rng = np.random.default_rng(seed)
        n = 40
        files = ["perf1.data"] * n + ["perf2.data"] * n
        syms = (["foo"] * 20 + ["bar"] * 20) * 2
        shift = np.array([0.0] * n + [30.0] * n)
        return pd.DataFrame(
            {
                "file": files,
                "sym": syms,
                "cycles": rng.normal(1000.0, 20.0, 2 * n) + shift,
                "instructions": rng.normal(500.0, 10.0, 2 * n),
                "pid": [111] * n + [222] * n,
                "ip": list(range(2 * n)),
            }
        )

    def _write_records(self, tmp):
        paths = []
        for name in ("perf1.data", "perf2.data"):
            path = Path(tmp, name)
            with open(path, "wb") as fh:
                fh.write(b"PERFILE1" + b"\x00" * 64)
            paths.append(str(path))
        return paths

    def test_records_compared_per_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_records(tmp)
            args = self._args(paths, json=True)
            args.event = None
            with patch.object(cli.perf, "parse", return_value=self._records_df()):
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    cli.compare_cmd(args)
                records = json.loads(buf.getvalue())
        self.assertNotIn("name", records[0])
        self.assertEqual(len(records), 4)
        self.assertTrue(all(r["event"] in ("cycles", "instructions") for r in records))
        cycles = [r for r in records if r["event"] == "cycles"]
        self.assertEqual(len(cycles), 2)
        self.assertTrue(all(r["significant"] for r in cycles))
        self.assertTrue(all(r["baseline"] == "perf1.data" for r in cycles))
        self.assertTrue(all(r["challenger"] == "perf2.data" for r in cycles))

    def test_records_explicit_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_records(tmp)
            args = self._args(paths, event=["cycles"], json=True)
            with patch.object(cli.perf, "parse", return_value=self._records_df()):
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    cli.compare_cmd(args)
                records = json.loads(buf.getvalue())
        self.assertTrue(records)
        self.assertTrue(all(r["event"] == "cycles" for r in records))

    def test_records_found_by_data_glob_in_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name in ("run1.data", "run2.data"):
                path = Path(tmp, name)
                with open(path, "wb") as fh:
                    fh.write(b"PERFILE1" + b"\x00" * 64)
                paths.append(str(path))
            args = self._args(tmp, event=["cycles"], json=True)
            with patch.object(cli.perf, "parse", return_value=self._records_df()):
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    cli.compare_cmd(args)
                records = json.loads(buf.getvalue())
        self.assertTrue(records)
        self.assertTrue(all(r["event"] == "cycles" for r in records))

    def test_records_require_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_records(tmp)
            df = self._records_df().drop(columns=["sym"])
            args = self._args(paths, json=True)
            args.event = None
            with patch.object(cli.perf, "parse", return_value=df):
                with self.assertRaises(SystemExit):
                    cli.compare_cmd(args)

    def test_names_compared_across_files(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(
                json.dumps(
                    self._envelope("a", rng.normal(100, 5, 30).tolist(), name="f")
                )
            )
            Path(tmp, "b.json").write_text(
                json.dumps(
                    self._envelope("b", rng.normal(200, 5, 30).tolist(), name="g")
                )
            )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.compare_cmd(self._args(tmp, json=True))
            records = json.loads(buf.getvalue())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["baseline"], "f")
        self.assertEqual(records[0]["challenger"], "g")
        self.assertTrue(records[0]["significant"])

    def test_event_and_filter(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(
                json.dumps(self._envelope("a", rng.normal(100, 5, 30).tolist()))
            )
            Path(tmp, "b.json").write_text(
                json.dumps(self._envelope("b", rng.normal(200, 5, 30).tolist()))
            )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.compare_cmd(
                    self._args(
                        tmp, event=["cycles"], filter='file == "a" or file == "b"'
                    )
                )
            self.assertIn("challenger", buf.getvalue())
            with self.assertRaises(SystemExit):
                cli.compare_cmd(self._args(tmp, filter="not valid @@ filter @@"))

    def test_filter(self):
        rng = np.random.default_rng(15)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(
                json.dumps(self._envelope("a", rng.normal(100, 5, 30).tolist()))
            )
            Path(tmp, "b.json").write_text(
                json.dumps(self._envelope("b", rng.normal(200, 5, 30).tolist()))
            )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.compare_cmd(self._args(tmp, filter='mode == "latency"'))
            self.assertIn("challenger", buf.getvalue())

    def test_bad_alpha_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                cli.compare_cmd(self._args(tmp, alpha=2.0))


if __name__ == "__main__":
    unittest.main()
