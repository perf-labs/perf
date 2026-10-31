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

import importlib
import threading
import time
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import perf

pl = importlib.import_module("perf.plot")


class TestChartRegistry(unittest.TestCase):
    def setUp(self):
        pl._ensure_backend("Agg")
        patcher = patch.object(pl, "_ensure_backend", return_value="Agg")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _df(self):
        return pd.DataFrame(
            {
                "name": ["a", "a", "b", "b"],
                "samples": [0, 1, 0, 1],
                "cycles": [10, 20, 30, 40],
            }
        )

    def test_known_charts_registered(self):
        for name in (
            "ecdf",
            "scatter",
            "line",
            "hist",
            "kde",
            "box",
            "violin",
            "strip",
            "swarm",
            "bar",
            "point",
            "boxen",
            "rug",
        ):
            self.assertIn(name, pl._CHART)
        self.assertEqual(len(pl._CHART), 13)

    def test_all_charts_execute(self):
        import matplotlib.pyplot as plt

        df = self._df()
        for name in pl._CHART:
            with self.subTest(chart=name):
                try:
                    with patch("perf.plot.plt.show"):
                        pl._CHART[name](df, "cycles")
                except TypeError as e:
                    self.fail(f"chart {name!r} raised TypeError: {e}")
                finally:
                    plt.close("all")

    def test_split_groups(self):
        self.assertEqual(pl._split_groups(None, ["ecdf"]), [["ecdf"]])
        self.assertEqual(pl._split_groups("line", ["ecdf"]), [["line"]])
        self.assertEqual(
            pl._split_groups("line,scatter", ["ecdf"]), [["line", "scatter"]]
        )
        self.assertEqual(
            pl._split_groups(["line,scatter"], ["ecdf"]), [["line", "scatter"]]
        )
        self.assertEqual(
            pl._split_groups(["line", "scatter"], ["ecdf"]), [["line"], ["scatter"]]
        )
        self.assertEqual(
            pl._split_groups(["cycles,instructions"], ["x"]),
            [["cycles", "instructions"]],
        )
        self.assertEqual(
            pl._split_groups(["cycles", "instructions"], ["x"]),
            [["cycles"], ["instructions"]],
        )

    @patch("perf.plot.plt.show")
    def test_comma_types_overlay_same_chart(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df()
        try:
            pl.plot(df, ["line,scatter"], ["cycles"])
            fig = plt.gcf()
            self.assertEqual(len(fig.axes), 1)
            self.assertEqual(fig.axes[0].get_title(), "")
            self.assertEqual(fig.axes[0].get_ylabel(), "cycles")
            mock_show.assert_called_once()
        finally:
            plt.close("all")

    def test_unknown_chart_lists_available(self):
        df = self._df()
        with self.assertRaises(KeyError) as ctx:
            pl.plot(df, ["bogus"], ["cycles"])
        msg = ctx.exception.args[0]
        for name in pl._CHART:
            self.assertIn(name, msg)

    @patch("perf.plot.plt.show")
    def test_repeated_types_make_new_charts(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df()
        try:
            pl.plot(df, ["line", "scatter"], ["cycles"])
            self.assertEqual(mock_show.call_count, 2)
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_comma_events_overlay_same_chart(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df().assign(inst=[1, 2, 3, 4])
        try:
            pl.plot(
                df, ["line"], ["cycles,instructions".replace("instructions", "inst")]
            )
            fig = plt.gcf()
            self.assertEqual(len(fig.axes), 1)
            self.assertEqual(fig.axes[0].get_title(), "")
            mock_show.assert_called_once()
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_repeated_events_side_by_side(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df().assign(inst=[1, 2, 3, 4])
        try:
            pl.plot(df, ["line"], [["cycles"], ["inst"]])
            fig = plt.gcf()
            self.assertEqual(len(fig.axes), 2)
            pos = sorted(ax.get_subplotspec().colspan.start for ax in fig.axes)
            self.assertEqual(pos, [0, 1])
            mock_show.assert_called_once()
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_types_events_combined_side_by_side(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df().assign(inst=[1, 2, 3, 4])
        orig_line = pl._CHART["line"]
        orig_scatter = pl._CHART["scatter"]
        line = Mock(wraps=orig_line)
        scatter = Mock(wraps=orig_scatter)
        pl._CHART["line"] = line
        pl._CHART["scatter"] = scatter
        try:
            pl.plot(df, ["line,scatter"], [["cycles"], ["inst"]])
            fig = plt.gcf()
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(line.call_count, 2)
            self.assertEqual(scatter.call_count, 2)
            mock_show.assert_called_once()
        finally:
            pl._CHART["line"] = orig_line
            pl._CHART["scatter"] = orig_scatter
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_log_scales_come_from_config(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df()
        try:
            pl.plot(
                df,
                ["line"],
                ["cycles"],
                "samples",
                None,
                None,
                {"logx": True, "logy": True},
            )
            fig = plt.gcf()
            for ax in fig.axes:
                self.assertEqual(ax.get_xscale(), "log")
                self.assertEqual(ax.get_yscale(), "log")
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_config_is_read_off_an_object(self, mock_show):
        from types import SimpleNamespace

        import matplotlib.pyplot as plt

        df = self._df()
        try:
            pl.plot(df, ["line"], ["cycles"], config=SimpleNamespace(logy=True))
            fig = plt.gcf()
            for ax in fig.axes:
                self.assertEqual(ax.get_yscale(), "log")
                self.assertEqual(ax.get_xscale(), "linear")
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_config_carries_rcparams(self, mock_show):
        import matplotlib.pyplot as plt

        old = plt.rcParams["axes.grid"]
        try:
            pl.plot(self._df(), ["line"], ["cycles"], config={"axes.grid": False})
            self.assertFalse(plt.rcParams["axes.grid"])
        finally:
            plt.rcParams["axes.grid"] = old
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_config_keeps_log_scales_out_of_rcparams(self, mock_show):
        import matplotlib.pyplot as plt

        try:
            pl.plot(self._df(), ["line"], ["cycles"], config={"logx": True})
            self.assertNotIn("logx", plt.rcParams)
            self.assertNotIn("logy", plt.rcParams)
            self.assertEqual(plt.gcf().axes[0].get_xscale(), "log")
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_yaxis_overrides_the_event(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df().assign(inst=[1, 2, 3, 4])
        try:
            pl.plot(df, ["line"], ["cycles"], yaxis="inst")
            fig = plt.gcf()
            self.assertEqual(fig.axes[0].get_ylabel(), "inst")
            plotted = sorted(
                y for line in fig.axes[0].get_lines() for y in line.get_ydata()
            )
            self.assertAlmostEqual(max(plotted), 4.0)
        finally:
            plt.close("all")

    def test_unknown_yaxis_is_reported(self):
        df = self._df()
        with self.assertRaises(KeyError) as ctx:
            pl.plot(df, ["line"], ["cycles"], yaxis="bogus")
        self.assertIn("y column not found in data: 'bogus'", ctx.exception.args[0])

    @patch("perf.plot.plt.show")
    def test_yaxis_and_event_keep_one_column_per_event(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df().assign(inst=[1, 2, 3, 4])
        try:
            pl.plot(df, ["line"], [["cycles"], ["inst"]], yaxis="inst")
            fig = plt.gcf()
            self.assertEqual(len(fig.axes), 2)
            for ax in fig.axes:
                self.assertEqual(ax.get_ylabel(), "inst")
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_errorbar_default(self, mock_show):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "name": ["a"] * 8 + ["b"] * 8,
                "samples": list(range(8)) * 2,
                "cycles": list(range(8)) * 2,
            }
        )
        for name in ("line", "bar", "point"):
            orig = pl._CHART[name]
            mocked = Mock(wraps=orig)
            pl._CHART[name] = mocked
            try:
                pl.plot(df, [name], ["cycles"])
                _, kw = mocked.call_args
                self.assertEqual(kw.get("errorbar"), pl._nonnegative_sd)
            finally:
                pl._CHART[name] = orig
                plt.close("all")

    def test_no_group_synthetic_column(self):
        df = self._df()
        out, label = pl.ensure_group(df, ["name"])
        self.assertEqual(label, "name")
        self.assertNotIn("_group", out.columns)

    def test_multi_key_label_is_descriptive(self):
        df = self._df().assign(file=["f"] * 4, mode=["latency"] * 4)
        out, label = pl.ensure_group(df, ["file", "name", "mode"])
        self.assertEqual(label, "file/name/mode")
        self.assertNotIn("_group", out.columns)
        self.assertIn("file/name/mode", out.columns)

    def test_default_group_expands_data_columns(self):
        df = self._df().assign(
            file=["f"] * 4,
            mode=["latency"] * 4,
            **{"data.rdi": [15, 15, 16, 16]},
        )
        out, label = pl.ensure_group(df)
        self.assertEqual(label, "file/name/mode/data.rdi")
        self.assertIn(label, out.columns)

    def test_wildcard_group_expands_data_columns(self):
        df = self._df().assign(
            file=["f"] * 4,
            mode=["latency"] * 4,
            **{"data.rdi": [15, 15, 16, 16]},
        )
        out, label = pl.ensure_group(df, ["file", "name", "mode", "data.*"])
        self.assertEqual(label, "file/name/mode/data.rdi")
        self.assertIn(label, out.columns)

    def test_empty_groupby_means_no_hue(self):
        df = self._df()
        _, label = pl.ensure_group(df, [])
        self.assertIsNone(label)

    @patch("perf.plot.plt.show")
    def test_no_titles_axis_labels_show_event(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df()
        try:
            pl.plot(df, ["line"], ["cycles"])
            fig = plt.gcf()
            for ax in fig.axes:
                self.assertEqual(ax.get_title(), "")
            self.assertEqual(fig.axes[0].get_ylabel(), "cycles")
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_legend_fontsize_medium(self, mock_show):
        import matplotlib.pyplot as plt

        self.assertEqual(pl._legend_kwargs()["fontsize"], "medium")
        df = self._df()
        try:
            pl.plot(df, ["line"], ["cycles"])
            fig = plt.gcf()
            self.assertTrue(len(fig.legends) >= 1)
            import matplotlib as mpl

            expected = (
                mpl.rcParams["font.size"] * mpl.font_manager.font_scalings["medium"]
            )
            for leg in fig.legends:
                for txt in leg.get_texts():
                    self.assertAlmostEqual(txt.get_fontsize(), expected)
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_duration_time_ns_label_and_values(self, mock_show):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "name": ["a", "a", "b", "b"],
                "samples": [0, 1, 0, 1],
                "duration_time": [1.0, 2.0, 3.0, 4.0],
            }
        )
        try:
            pl.plot(df, ["line"], ["duration_time"])
            fig = plt.gcf()
            self.assertEqual(fig.axes[0].get_ylabel(), "duration_time[ns]")
            plotted = sorted(
                y for line in fig.axes[0].get_lines() for y in line.get_ydata()
            )
            self.assertTrue(plotted)
            self.assertAlmostEqual(min(plotted), 1.0, places=5)
            self.assertAlmostEqual(max(plotted), 4.0, places=5)
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_duration_time_per_operation(self, mock_show):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "name": ["a", "a"],
                "samples": [0, 1],
                "operations": [2, 2],
                "duration_time": [4.0, 6.0],
            }
        )
        try:
            pl.plot(df, ["line"], ["duration_time"])
            fig = plt.gcf()
            self.assertEqual(fig.axes[0].get_ylabel(), "duration_time[ns]/operations")
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_duration_per_op_explicit_both_render(self, mock_show):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "name": ["a", "a"],
                "samples": [0, 1],
                "operations": [2, 2],
                "duration_time": [4.0, 6.0],
            }
        )
        df["duration_time/operations"] = df["duration_time"] / df["operations"]
        try:
            pl.plot(df, ["line"], ["duration_time", "duration_time/operations"])
            fig = plt.gcf()
            visible = [ax for ax in fig.axes if ax.get_visible()]
            self.assertEqual(len(visible), 2)
            self.assertEqual(visible[0].get_ylabel(), "duration_time[ns]/operations")
            self.assertEqual(visible[1].get_ylabel(), "duration_time[ns]/operations")
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_duration_per_op_alone_shows_ns(self, mock_show):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "name": ["a", "a"],
                "samples": [0, 1],
                "duration_time/operations": [2.0, 3.0],
            }
        )
        try:
            pl.plot(df, ["line"], ["duration_time/operations"])
            fig = plt.gcf()
            self.assertEqual(fig.axes[0].get_ylabel(), "duration_time[ns]/operations")
            plotted = sorted(
                y for line in fig.axes[0].get_lines() for y in line.get_ydata()
            )
            self.assertTrue(plotted)
            self.assertAlmostEqual(min(plotted), 2.0, places=5)
            self.assertAlmostEqual(max(plotted), 3.0, places=5)
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_bar_mean_dots(self, mock_show):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "name": ["a"] * 8 + ["b"] * 8,
                "samples": list(range(8)) * 2,
                "cycles": list(range(8)) * 2,
            }
        )
        try:
            pl.plot(df, ["bar"], ["cycles"])
            fig = plt.gcf()
            dots = [
                c
                for ax in fig.axes
                for c in ax.collections
                if "PathCollection" in type(c).__name__
            ]
            self.assertEqual(len(dots), 2)
            for d in dots:
                for x, y in d.get_offsets().tolist():
                    self.assertGreater(y, 0)
        finally:
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_errorbar_caps_and_markers(self, mock_show):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "name": ["a"] * 8 + ["b"] * 8,
                "samples": list(range(8)) * 2,
                "cycles": list(range(8)) * 2,
            }
        )
        for name in ("bar", "point"):
            orig = pl._CHART[name]
            mocked = Mock(wraps=orig)
            pl._CHART[name] = mocked
            try:
                pl.plot(df, [name], ["cycles"])
                _, kw = mocked.call_args
                self.assertEqual(kw.get("errorbar"), pl._nonnegative_sd)
                self.assertEqual(kw.get("capsize"), pl._ERRORBAR_CAPS)
                self.assertLessEqual(kw.get("capsize"), 0.1)
            finally:
                pl._CHART[name] = orig
                plt.close("all")
        orig = pl._CHART["line"]
        mocked = Mock(wraps=orig)
        pl._CHART["line"] = mocked
        try:
            pl.plot(df, ["line"], ["cycles"])
            _, kw = mocked.call_args
            self.assertEqual(kw.get("errorbar"), pl._nonnegative_sd)
            self.assertEqual(kw.get("err_style"), "bars")
            self.assertEqual(kw.get("marker"), "o")
            self.assertEqual(kw.get("err_kws", {}).get("capsize"), pl._LINE_ERR_CAPS)
            self.assertLessEqual(kw.get("err_kws", {}).get("capsize"), 3)
        finally:
            pl._CHART["line"] = orig
            plt.close("all")

    def test_errorbar_never_below_zero(self):
        lo, hi = pl._nonnegative_sd([0.5, 0.1, 0.0, 2.0])
        self.assertGreaterEqual(lo, 0.0)
        self.assertGreaterEqual(hi, lo)
        lo, hi = pl._nonnegative_sd([5.0, 5.0, 5.0])
        self.assertEqual((lo, hi), (5.0, 5.0))
        lo, hi = pl._nonnegative_sd([])
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_no_nan_in_group_labels(self):
        import numpy as np

        df = pd.DataFrame(
            {
                "file": [np.nan, np.nan],
                "name": ["t", "t"],
                "mode": ["latency", "latency"],
                "cycles": [10, 20],
            }
        )
        out, label = pl.ensure_group(df, ["file", "name", "mode"])
        self.assertEqual(label, "file/name/mode")
        self.assertEqual(out[label].tolist(), ["t/latency", "t/latency"])
        self.assertFalse(out[label].str.contains("nan").any())

    def test_group_cell_blanks_missing_strings(self):
        import numpy as np

        from perf.core import _text

        self.assertEqual(_text(np.nan), "")
        self.assertEqual(_text(None), "")
        self.assertEqual(_text("nan"), "")
        self.assertEqual(_text("None"), "")
        self.assertEqual(_text(""), "")
        self.assertEqual(_text("a.out@abc"), "a.out@abc")

    def test_single_key_nan_blanked(self):
        import numpy as np

        df = pd.DataFrame({"file": [np.nan, np.nan], "cycles": [10, 20]})
        out, label = pl.ensure_group(df, ["file"])
        self.assertEqual(label, "file")
        self.assertEqual(out[label].tolist(), ["", ""])


class TestPlotLayout(unittest.TestCase):
    def setUp(self):
        pl._ensure_backend("Agg")
        patcher = patch.object(pl, "_ensure_backend", return_value="Agg")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _df(self):
        return pd.DataFrame(
            {
                "name": ["a", "a", "b", "b"],
                "samples": [0, 1, 0, 1],
                "cycles": [10, 20, 30, 40],
            }
        )

    def test_single_chart_file_default_size(self):
        import sys

        import matplotlib.pyplot as plt

        old = plt.rcParams.get("figure.figsize")
        plt.rcParams["figure.figsize"] = (10, 5)
        orig_tty = sys.stdout.isatty
        sys.stdout.isatty = lambda: False
        try:
            rows, cols, w, h = pl._grid_for(1, "/tmp/out.png")
        finally:
            sys.stdout.isatty = orig_tty
            plt.rcParams["figure.figsize"] = old
        self.assertEqual((rows, cols, w, h), (1, 1, 10, 5))

    def test_single_chart_honors_config_figsize(self):
        import sys

        import matplotlib.pyplot as plt

        old = plt.rcParams.get("figure.figsize")
        plt.rcParams["figure.figsize"] = (12, 7)
        orig_tty = sys.stdout.isatty
        sys.stdout.isatty = lambda: False
        try:
            rows, cols, w, h = pl._grid_for(1, "/tmp/out.png")
        finally:
            sys.stdout.isatty = orig_tty
            plt.rcParams["figure.figsize"] = old
        self.assertEqual((rows, cols, w, h), (1, 1, 12, 7))

    def test_two_charts_side_by_side(self):
        import sys

        orig_tty = sys.stdout.isatty
        sys.stdout.isatty = lambda: False
        try:
            rows, cols, _, _ = pl._grid_for(2, "/tmp/out.png")
        finally:
            sys.stdout.isatty = orig_tty
        self.assertEqual((rows, cols), (1, 2))

    def test_many_charts_become_grid(self):
        import sys

        orig_tty = sys.stdout.isatty
        sys.stdout.isatty = lambda: False
        try:
            rows, cols, _, _ = pl._grid_for(6, "/tmp/out.png")
        finally:
            sys.stdout.isatty = orig_tty
        self.assertTrue(rows > 1)

    def test_grid_fits_available_screen(self):
        import os
        import sys

        orig_tty = sys.stdout.isatty
        orig_tsize = os.get_terminal_size
        sys.stdout.isatty = lambda: True
        os.get_terminal_size = lambda *a, **k: os.terminal_size((80, 24))
        try:
            rows, cols, w, h = pl._grid_for(6, None)
        finally:
            sys.stdout.isatty = orig_tty
            os.get_terminal_size = orig_tsize
        avail_w = 80 * pl._TERMINAL_CELL_W / 100.0 * pl._SCREEN_MARGIN
        avail_h = 24 * pl._TERMINAL_CELL_H / 100.0 * pl._SCREEN_MARGIN
        self.assertLessEqual(cols * w, avail_w + 1e-9)
        self.assertLessEqual(rows * h, avail_h + 1e-9)

    def test_duration_label_with_threads(self):
        df = pd.DataFrame(
            {
                "name": ["a", "a"],
                "samples": [0, 1],
                "operations": [2, 2],
                "threads": [2, 2],
                "duration_time": [8.0, 12.0],
            }
        )
        self.assertEqual(
            pl._duration_axis_label(df), "duration_time[ns]/operations/threads"
        )
        vals = pl._duration_ns_values(df).tolist()
        self.assertEqual(vals, [2.0, 3.0])

    def test_duration_per_op_divides_threads(self):
        df = pd.DataFrame(
            {
                "duration_time/operations": [4.0, 6.0],
                "threads": [2, 2],
            }
        )
        self.assertEqual(pl._duration_per_op_ns_values(df).tolist(), [2.0, 3.0])

    @patch("perf.plot.plt.show")
    def test_multi_event_grid_figure(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df().assign(inst=[1, 2, 3, 4])
        try:
            pl.plot(
                df,
                ["line"],
                [["cycles"], ["inst"], ["cycles"], ["inst"]],
            )
            fig = plt.gcf()
            self.assertEqual(len(fig.axes), 4)
            self.assertTrue(fig.axes[0].get_ylabel())
        finally:
            plt.close("all")

    def test_legend_below_stays_inside_figure(self):
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots()
        try:
            lines = []
            labels = []
            for i in range(6):
                line = plt.plot([0, 1], [i, i], label=f"grp {i}")[0]
                lines.append(line)
                labels.append(f"grp {i}")
            leg = pl._move_legend_below(fig, lines, labels)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = leg.get_window_extent(renderer)
            fig_bb = fig.get_window_extent(renderer)
            self.assertGreaterEqual(bb.y0, fig_bb.y0 - 0.5)
            self.assertLessEqual(bb.y1, fig_bb.y1 + 0.5)
            self.assertLessEqual(bb.x1, fig_bb.x1 + 0.5)
            self.assertLessEqual(bb.x0, fig_bb.x0 + 0.5)
        finally:
            plt.close("all")

    def test_legend_below_clears_axes_for_many_labels(self):
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots(figsize=(6, 4))
        try:
            labels = [f"group-{i:02d}-very-long-name" for i in range(20)]
            handles = [plt.Line2D([], [], label=lab) for lab in labels]
            leg = pl._move_legend_below(fig, handles, labels)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = leg.get_window_extent(renderer)
            top = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes)
            self.assertLessEqual(bb.y1, top + 0.5)
        finally:
            plt.close("all")

    def test_legend_below_reserves_ascii_rows(self):
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots(figsize=(10, 4.8))
        try:
            labels = [f"g{i}" for i in range(12)]
            handles = [plt.Line2D([], [], label=lab) for lab in labels]
            with patch.object(pl, "_ascii_canvas", return_value=True):
                pl._move_legend_below(fig, handles, labels)
            self.assertGreaterEqual(
                fig.axes[0].get_position().bounds[1],
                pl._ascii_legend_height(fig, labels),
            )
        finally:
            plt.close("all")


class TestPlotCfg(unittest.TestCase):
    def test_defaults(self):
        perf.plot.config = None
        params = perf.plot.config
        self.assertEqual(params["figure.figsize"], (10, 5))
        self.assertTrue(params["axes.grid"])

    def test_flat_override(self):
        perf.plot.config = {"axes.grid": False}
        params = perf.plot.config
        self.assertFalse(params["axes.grid"])
        self.assertEqual(params["figure.figsize"], (10, 5))

    def test_nested_flattens(self):
        self.assertEqual(
            pl.normalize_config({"figure": {"figsize": [6, 4]}}),
            {"figure.figsize": [6, 4]},
        )

    def test_container_alternatives_flatten_to_a_column_list(self):
        self.assertEqual(
            pl.normalize_config(
                {
                    "code": [{"align": 1}, {"align": 16}],
                    "thread": [[{"affinity": None, "priority": "normal"}]],
                }
            ),
            {
                "code.align": [1, 16],
                "thread.affinity": [None],
                "thread.priority": ["normal"],
            },
        )

    def test_single_element_list_unwraps(self):
        self.assertEqual(
            pl._flatten_list([16]),
            16,
        )
        self.assertEqual(
            pl._flatten_list([[0x200000]]),
            0x200000,
        )
        self.assertEqual(
            pl._flatten_list([1, 16]),
            [1, 16],
        )

    def test_figsize_list_becomes_tuple(self):
        perf.plot.config = {"figure": {"figsize": [6, 4]}}
        params = perf.plot.config
        self.assertEqual(params["figure.figsize"], (6, 4))

    def test_style_applied(self):
        with patch("perf.plot.plt.style.use") as use:
            perf.plot.config = {"style": "dark_background"}
            use.assert_called_once_with("dark_background")

    def test_rcparams_written(self):
        import matplotlib.pyplot as plt

        old = plt.rcParams["axes.grid"]
        try:
            perf.plot.config = {"axes.grid": False}
            self.assertFalse(plt.rcParams["axes.grid"])
        finally:
            plt.rcParams["axes.grid"] = old
            perf.plot.config = None


class TestPlotCliConfig(unittest.TestCase):
    @staticmethod
    def _cli():
        import importlib.machinery
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "bin" / "perf-plot"
        loader = importlib.machinery.SourceFileLoader("perfcli", str(path))
        spec = importlib.util.spec_from_loader("perfcli", loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module

    def test_resolve_plot_config_overrides(self):
        cli = self._cli()
        args = Mock(config=None)
        config = cli._resolve_plot_config(
            args,
            [
                "--config.figure.figsize=[12,7]",
                "--config.style=dark_background",
                "--config.axes.grid=False",
            ],
        )
        self.assertEqual(config["figure"]["figsize"], [12, 7])
        self.assertEqual(config["style"], "dark_background")
        self.assertEqual(config["axes"]["grid"], False)

    def test_resolve_plot_config_rejects_unknown(self):
        cli = self._cli()
        with self.assertRaises(SystemExit):
            cli._resolve_plot_config(Mock(config=None), ["--bogus=1"])
        with self.assertRaises(SystemExit) as ctx:
            cli._resolve_plot_config(Mock(config=None), ["--config.bogus=1"])
        self.assertIn("bogus", str(ctx.exception))

    def test_resolve_plot_config_file(self):
        import json
        import tempfile

        cli = self._cli()
        with tempfile.NamedTemporaryFile("w", suffix=".json") as fh:
            json.dump({"style": "dark_background"}, fh)
            fh.flush()
            args = Mock(config=fh.name)
            config = cli._resolve_plot_config(args, [])
        self.assertEqual(config["style"], "dark_background")

    def test_csv_duration_values_are_converted_to_ns(self):
        cli = self._cli()
        df = cli.table_to_df(
            "file,name,mode,operations,duration_time\n"
            "a.out,func,latency,1,3.00ns\n"
            "a.out,func,latency,1,1.50us\n"
        )
        self.assertEqual(df["duration_time"].tolist(), [3.0, 1500.0])

    def test_resolve_plot_config_missing_file(self):
        cli = self._cli()
        with self.assertRaises(SystemExit) as ctx:
            cli._resolve_plot_config(Mock(config="/nonexistent/plot.json"), [])
        self.assertIn("cannot read --config", str(ctx.exception))

    def test_resolve_plot_config_non_object_file(self):
        import tempfile

        cli = self._cli()
        with tempfile.NamedTemporaryFile("w", suffix=".json") as fh:
            fh.write("[1, 2]")
            fh.flush()
            with self.assertRaises(SystemExit) as ctx:
                cli._resolve_plot_config(Mock(config=fh.name), [])
        self.assertIn("JSON object", str(ctx.exception))

    def test_plot_config_round_trip(self):
        cli = self._cli()
        args = Mock(config=None)
        config = cli._resolve_plot_config(args, ["--config.axes.grid=False"])
        perf.plot.config = config
        params = perf.plot.config
        self.assertFalse(params["axes.grid"])


class TestBackendSelection(unittest.TestCase):
    def test_inline_backend_is_left_alone(self):
        import matplotlib

        with patch.object(
            matplotlib,
            "get_backend",
            return_value="module://matplotlib_inline.backend_inline",
        ):
            with patch.object(matplotlib, "use") as use:
                importlib.reload(pl)
                try:
                    pl._ensure_backend()
                except Exception:
                    pass
        use.assert_not_called()

    def test_console_backend_prefers_sixel_when_terminal_supports_it(self):
        import os

        import matplotlib

        env = {
            "TERM": "xterm-256color",
            "TERM_PROGRAM": "any-terminal",
            "LC_TERMINAL": "",
        }
        pl._terminal_capabilities.cache_clear()
        self.addCleanup(pl._terminal_capabilities.cache_clear)
        with patch.dict(os.environ, env, clear=True):
            with (
                patch.object(pl, "_stdout_isatty", return_value=True),
                patch.object(pl, "_query_sixel_support", return_value=True),
                patch.object(pl, "_module_available", return_value=True),
                patch.object(pl.shutil, "which", return_value="/usr/bin/convert"),
                patch.object(matplotlib, "get_backend", return_value="Agg"),
                patch.object(matplotlib, "use") as use,
            ):
                pl._ensure_backend()
        use.assert_called_once_with("module://matplotlib_backend_sixel", force=True)

    def test_console_backend_falls_back_to_ascii_without_sixel(self):
        import os

        import matplotlib

        env = {
            "TERM": "xterm-256color",
            "TERM_PROGRAM": "tmux",
            "LC_TERMINAL": "",
        }
        with patch.dict(os.environ, env, clear=True):
            with (
                patch.object(pl, "_module_available", return_value=True),
                patch.object(matplotlib, "get_backend", return_value="Agg"),
                patch.object(matplotlib, "use") as use,
            ):
                pl._ensure_backend()
        use.assert_called_once_with("module://mpl_ascii", force=True)

    def test_console_backend_falls_back_to_agg_without_terminal_backends(self):
        import os

        import matplotlib

        env = {"TERM": "xterm-256color", "TERM_PROGRAM": "tmux", "LC_TERMINAL": ""}
        with patch.dict(os.environ, env, clear=True):
            with (
                patch.object(pl, "_module_available", return_value=False),
                patch.object(matplotlib, "get_backend", return_value="Agg"),
                patch.object(matplotlib, "use") as use,
            ):
                pl._ensure_backend()
        use.assert_called_once_with("Agg", force=True)

    def test_terminal_capabilities(self):
        import os

        pl._terminal_capabilities.cache_clear()
        self.addCleanup(pl._terminal_capabilities.cache_clear)
        env = {"TERM": "xterm-256color", "TERM_PROGRAM": "", "LC_TERMINAL": ""}
        with patch.dict(os.environ, env, clear=True):
            with (
                patch.object(pl, "_stdout_isatty", return_value=True),
                patch.object(pl, "_query_sixel_support", return_value=True),
            ):
                self.assertTrue(pl._terminal_supports_sixel())
                self.assertTrue(pl._terminal_supports_ascii())
        env = {"TERM": "dumb", "TERM_PROGRAM": "", "LC_TERMINAL": ""}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(pl._terminal_supports_sixel())
            self.assertFalse(pl._terminal_supports_ascii())

    def test_sixel_capability_is_cached(self):
        pl._terminal_capabilities.cache_clear()
        self.addCleanup(pl._terminal_capabilities.cache_clear)
        with (
            patch.object(pl, "_stdout_isatty", return_value=True),
            patch.object(pl, "_query_sixel_support", return_value=True) as query,
        ):
            self.assertTrue(pl._terminal_supports_sixel())
            self.assertTrue(pl._terminal_supports_sixel())
        query.assert_called_once()

    def test_da1_reports_sixel_capability(self):
        self.assertTrue(pl._da1_supports_sixel(b"\x1b[?62;4;6;22c"))
        self.assertFalse(pl._da1_supports_sixel(b"\x1b[?62;6;22c"))
        self.assertFalse(pl._da1_supports_sixel(b"not a response"))

    def test_explicit_backend_uses_matplotlib_name(self):
        import matplotlib

        with patch.object(matplotlib, "use") as use:
            pl._ensure_backend("qtagg")
        use.assert_called_once_with("qtagg", force=True)

    def test_named_backends_expand_to_modules(self):
        import matplotlib

        for name, expected in (
            ("sixel", "module://matplotlib_backend_sixel"),
            ("ascii", "module://mpl_ascii"),
        ):
            with self.subTest(backend=name):
                with patch.object(matplotlib, "use") as use:
                    pl._ensure_backend(name)
                use.assert_called_once_with(expected, force=True)

    def test_invalid_backend_raises(self):
        import matplotlib

        with patch.object(matplotlib, "use", side_effect=RuntimeError("nope")):
            with self.assertRaises(ValueError) as ctx:
                pl._ensure_backend("bogus")
        self.assertIn("cannot use matplotlib backend 'bogus'", str(ctx.exception))

    def test_ascii_backend_detection(self):
        self.assertTrue(pl._is_ascii_backend("module://mpl_ascii"))
        self.assertFalse(pl._is_ascii_backend("agg"))

    def test_ascii_plot_keeps_legend(self):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "name": ["a", "a", "b", "b"],
                "cycles": [1, 2, 3, 4],
            }
        )
        with (
            patch.object(pl, "_ensure_backend", return_value="module://mpl_ascii"),
            patch.object(pl, "_show_or_save", return_value=None),
        ):
            pl.plot(df, ["line"], ["cycles"], groupby=["name"])
        fig = plt.gcf()
        self.assertEqual(len(fig.legends), 1)
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertEqual(labels, ["a", "b"])

    def test_ascii_distribution_plots_keep_legend_names(self):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "name": ["a"] * 8 + ["b"] * 8,
                "samples": list(range(8)) * 2,
                "cycles": list(range(8)) * 2,
            }
        )
        for chart in ("ecdf", "hist", "kde"):
            with self.subTest(chart=chart):
                try:
                    with (
                        patch.object(
                            pl, "_ensure_backend", return_value="module://mpl_ascii"
                        ),
                        patch.object(pl, "_show_or_save", return_value=None),
                    ):
                        pl.plot(df, [chart], ["cycles"], groupby=["name"])
                    fig = plt.gcf()
                    self.assertEqual(len(fig.legends), 1)
                    labels = [text.get_text() for text in fig.legends[0].get_texts()]
                    self.assertEqual(labels, ["a", "b"])
                finally:
                    plt.close("all")

    def test_ascii_categorical_plots_keep_group_names(self):
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "name": ["a"] * 8 + ["b"] * 8,
                "samples": list(range(8)) * 2,
                "cycles": list(range(8)) * 2,
            }
        )
        for chart in ("bar", "box", "violin"):
            with self.subTest(chart=chart):
                try:
                    with (
                        patch.object(
                            pl, "_ensure_backend", return_value="module://mpl_ascii"
                        ),
                        patch.object(pl, "_show_or_save", return_value=None),
                    ):
                        pl.plot(df, [chart], ["cycles"], groupby=["name"])
                    fig = plt.gcf()
                    self.assertEqual(len(fig.legends), 1)
                    labels = [text.get_text() for text in fig.legends[0].get_texts()]
                    self.assertEqual(labels, ["a", "b"])
                finally:
                    plt.close("all")

    def test_ascii_prepare_sanitizes_artists(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots()
        try:
            line = ax.plot([float("-inf"), 1, 2], [0, 0.5, 1])[0]
            coll = LineCollection([[(0, 0), (1, 1)]])
            ax.add_collection(coll)
            zero = plt.Rectangle((0, 0), 0, 0)
            ax.add_patch(zero)
            ax.legend([line], ["a"])
            fig.legend([line], ["a"])
            with patch.object(coll, "get_segments", return_value=[np.empty((0, 2))]):
                pl._ascii_prepare(fig)
            self.assertTrue(np.isfinite(line.get_xdata()).all())
            self.assertEqual(len(ax.collections), 0)
            self.assertNotIn(zero, ax.patches)
            self.assertIsNone(ax.get_legend())
            self.assertEqual(len(fig.legends), 0)
        finally:
            plt.close(fig)


class TestPlotValidation(unittest.TestCase):
    def _df(self):
        return pd.DataFrame({"name": ["a", "a"], "cycles": [1, 2]})

    def test_unknown_event_raises(self):
        with self.assertRaises(KeyError):
            pl.plot(self._df(), event=["nope"])

    def test_bogus_xaxis_raises(self):
        with self.assertRaises(KeyError):
            pl._xaxis(self._df(), "bogus")

    def test_nonnegative_sd_garbage(self):
        self.assertEqual(pl._nonnegative_sd(pd.Series(["a"])), (0.0, 0.0))
        self.assertEqual(pl._nonnegative_sd(pd.Series([float("nan")])), (0.0, 0.0))

    def test_legend_right_empty(self):
        self.assertEqual(pl._legend_right([]), 1.0)

    def test_resolve_plot_paths(self):
        paths = pl._resolve_plot_paths("o.png", [["line"], ["scatter"]])
        self.assertEqual(len(paths), 2)


class TestPlotThreadSafety(unittest.TestCase):
    def test_plot_calls_are_serialized(self):
        active = {"n": 0, "peak": 0}
        lock = threading.Lock()

        def fake_options(_config=None):
            with lock:
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])
            time.sleep(0.05)
            with lock:
                active["n"] -= 1
            raise RuntimeError("stop")

        def worker():
            try:
                pl.plot(pd.DataFrame({"cycles": [1.0]}))
            except RuntimeError:
                pass

        with (
            patch.object(pl, "_ensure_backend"),
            patch.object(pl, "_plot_options", side_effect=fake_options),
        ):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertEqual(active["peak"], 1)


class _PlotTestBase(unittest.TestCase):
    def setUp(self):
        import matplotlib.pyplot as plt

        pl._ensure_backend("Agg")
        patcher = patch.object(pl, "_ensure_backend", return_value="Agg")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(plt.close, "all")

    def _df(self):
        return pd.DataFrame(
            {
                "name": ["a", "a", "b", "b"],
                "samples": [0, 1, 0, 1],
                "cycles": [10.0, 20.0, 30.0, 40.0],
            }
        )


class TestPlotDataResolution(_PlotTestBase):
    @patch("perf.plot.plt.show")
    def test_filter_is_applied_before_plotting(self, mock_show):
        df = self._df()
        filtered = df[df["cycles"] > 15]
        with patch("perf.data.query", return_value=filtered) as query:
            pl.plot(df, ["line"], ["cycles"], filter="cycles > 15")
        query.assert_called_once_with(df, "cycles > 15")
        self.assertEqual(mock_show.call_count, 1)

    @patch("perf.plot.plt.show")
    def test_missing_event_falls_back_to_metric_columns(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df()
        pl.plot(df, ["line"])
        self.assertEqual(plt.gcf().axes[0].get_ylabel(), "cycles")

    @patch("perf.plot.plt.show")
    def test_xaxis_values_are_sorted_stably(self, mock_show):
        df = pd.DataFrame(
            {
                "name": ["a", "a", "a", "a"],
                "samples": [3, 1, 2, 0],
                "cycles": [30.0, 10.0, 20.0, 0.0],
            }
        )
        seen = []
        orig = pl._CHART["line"]
        pl._CHART["line"] = lambda data, *a, **k: seen.append(data.copy())
        try:
            pl.plot(df, ["line"], ["cycles"], xaxis="samples")
        finally:
            pl._CHART["line"] = orig
        self.assertEqual(seen[0]["samples"].tolist(), [0, 1, 2, 3])
        self.assertEqual(seen[0]["cycles"].tolist(), [0.0, 10.0, 20.0, 30.0])

    @patch("perf.plot.plt.show")
    def test_expression_axes_are_materialized(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df()
        pl.plot(df, ["line"], ["cycles"], xaxis="samples * 2", yaxis="cycles + 1")
        ax = plt.gcf().axes[0]
        self.assertEqual(ax.get_xlabel(), "samples * 2")
        self.assertEqual(ax.get_ylabel(), "cycles + 1")

    def test_plot_rejects_unknown_xaxis(self):
        with patch("perf.plot.plt.show"):
            with self.assertRaises(KeyError):
                pl.plot(self._df(), ["line"], ["cycles"], xaxis="nope")

    def test_default_xaxis_candidates(self):
        time_df = pd.DataFrame({"time": [1.0, 2.0], "cycles": [1, 2]})
        self.assertEqual(pl._xaxis(time_df).tolist(), [1.0, 2.0])
        indexed = pd.DataFrame({"cycles": [1, 2]}, index=pd.Index([4, 5], name="idx"))
        self.assertEqual(pl._xaxis(indexed).tolist(), [4, 5])
        plain = pd.DataFrame({"cycles": [1, 2]})
        self.assertEqual(list(pl._xaxis(plain)), [0, 1])

    def test_yaxis_none_returns_event(self):
        df = self._df()
        self.assertEqual(pl._yaxis(df, None, "cycles"), "cycles")
        self.assertEqual(pl._yaxis(df, "cycles", "other"), "cycles")

    def test_resolve_x_series_column_and_alias(self):
        from perf.arch import x86_64

        df = pd.DataFrame({"data.rdi": [1, 2], "cycles": [3, 4]})
        self.assertEqual(pl._resolve_x_series(df, "cycles").tolist(), [3, 4])
        with patch.object(pl, "arch", x86_64):
            alias = pl._resolve_x_series(df, "data.arg0")
            self.assertEqual(alias.tolist(), [1, 2])
            self.assertEqual(alias.name, "data.arg0")
            swapped = pd.DataFrame({"data.arg0": [1, 2], "cycles": [3, 4]})
            self.assertEqual(pl._resolve_x_series(swapped, "data.rdi").tolist(), [1, 2])
            prefixed = df.rename(columns={"data.rdi": "data.RDI"})
            self.assertEqual(
                pl._resolve_x_series(prefixed, "data.arg0").tolist(), [1, 2]
            )

    def test_resolve_x_series_data_and_config_attributes(self):
        df = pd.DataFrame({"cycles": [1, 2]})
        df.attrs["data"] = {"regs": {"rdi": 5}, "mem": {}}
        df.attrs["config"] = {"dcache": {"L1d": {"hit_rate": 100}}}
        self.assertEqual(pl._resolve_x_series(df, "data.rdi").tolist(), [5, 5])
        cfg = pl._resolve_x_series(df, "config.dcache.L1d.hit_rate")
        self.assertEqual(cfg.tolist(), [100, 100])

    def test_resolve_x_series_evaluates_expressions(self):
        df = self._df()
        result = pl._resolve_x_series(df, "cycles / 10")
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_resolve_x_series_raises_on_unknown(self):
        with self.assertRaises(KeyError):
            pl._resolve_x_series(self._df(), "missing")

    def test_data_reg_name_and_canonicalization(self):
        from perf.arch import x86_64

        self.assertEqual(pl._data_reg_name("data.rdi"), "rdi")
        self.assertIsNone(pl._data_reg_name("data.0x10"))
        self.assertIsNone(pl._data_reg_name("data.regs.rdi"))
        self.assertIsNone(pl._data_reg_name("data.mem.0x10"))
        self.assertIsNone(pl._data_reg_name("cycles"))
        self.assertIsNone(pl._data_reg_name(None))
        with patch.object(pl, "arch", x86_64):
            self.assertEqual(pl._canonical_data_col("data.arg0"), "data.rdi")
        self.assertEqual(pl._canonical_data_col("cycles"), "cycles")

    def test_alias_data_columns_adds_register_aliases(self):
        from perf.arch import x86_64

        df = pd.DataFrame({"data.arg0": [7, 8], "data.0x10": [1, 2]})

        class _FrameView:
            columns = ["data.arg0", "data.0x10"]

            def __getitem__(self, column):
                return df[column]

        with patch.object(pl, "arch", x86_64):
            aliases = pl._alias_data_columns(_FrameView())
        self.assertEqual(aliases["data.rdi"].tolist(), [7, 8])
        self.assertEqual(aliases["data.rdi"].tolist(), [7, 8])
        self.assertNotIn("data.arg0", aliases)
        self.assertNotIn("data.0x10", aliases)

    def test_materialize_axis(self):
        df = self._df()
        self.assertIs(pl._materialize_axis(df, "cycles"), df)
        out = pl._materialize_axis(df, "cycles * 2")
        self.assertIn("cycles * 2", out.columns)
        self.assertEqual(out["cycles * 2"].tolist(), [20.0, 40.0, 60.0, 80.0])
        with self.assertRaises(KeyError):
            pl._materialize_axis(df, "nope")

    def test_groupby_string_and_wildcard_expansion(self):
        df = self._df().assign(file=["f"] * 4, **{"data.rdi": [1, 1, 2, 2]})
        out, keys = pl._resolve_group_keys(df, "file, data.*")
        self.assertEqual(keys, ["file", "data.rdi"])
        self.assertEqual(list(out.columns), list(df.columns))

    def test_groupby_index_name_is_reset(self):
        df = self._df().set_index("name")
        out, keys = pl._resolve_group_keys(df, ["name"])
        self.assertIn("name", out.columns)
        self.assertEqual(keys, ["name"])

    def test_ensure_group_uses_vectorized_grouping_for_many_rows(self):
        df = pd.DataFrame(
            {
                "file": ["f"] * 120,
                "name": ["a", "b"] * 60,
                "mode": ["latency"] * 120,
                "cycles": range(120),
            }
        )
        out, label = pl.ensure_group(df, ["file", "name"])
        self.assertEqual(label, "file/name")
        self.assertEqual(out[label].nunique(), 2)
        self.assertEqual(out[label].tolist()[:2], ["f/a", "f/b"])

    def test_ensure_group_keeps_existing_combined_column(self):
        df = self._df().assign(
            file=["f", "f", "g", "g"], **{"file/name": ["x", "x", "y", "y"]}
        )
        out, label = pl.ensure_group(df, ["file", "name"])
        self.assertEqual(label, "file/name")
        self.assertEqual(out[label].tolist(), ["x", "x", "y", "y"])

    def test_ensure_group_falls_back_when_grouping_fails(self):
        df = self._df().assign(file=["f"] * 4)
        with patch("pandas.Series.nunique", side_effect=RuntimeError("boom")):
            out, label = pl.ensure_group(df, ["file", "name"])
        self.assertEqual(label, "file/name")
        self.assertEqual(out[label].tolist(), ["f/a", "f/a", "f/b", "f/b"])

    def test_ensure_group_normalizes_blank_and_repeated_separators(self):
        df = pd.DataFrame(
            {
                "file": ["a/", None, "b", "c"],
                "name": ["", "b", "", None],
                "cycles": [1, 2, 3, 4],
            }
        )
        out, label = pl.ensure_group(df, ["file", "name"])
        self.assertEqual(label, "file/name")
        self.assertEqual(out[label].tolist(), ["a", "b", "b", "c"])

    def test_group_series_cleans_missing_values(self):
        series = pd.Series(["a", None, "nan", "None", "", "b"])
        self.assertEqual(pl._group_series(series).tolist(), ["a", "", "", "", "", "b"])
        self.assertEqual(pl._group_series(pd.Series([1, 2])).tolist(), ["1", "2"])


class TestPlotConfigHelpers(_PlotTestBase):
    def test_split_config_from_dict_and_object(self):
        from types import SimpleNamespace

        flat, style = pl._split_config(
            {"logx": True, "backend": "agg", "style": "dark_background", "a.b": 1}
        )
        self.assertEqual(flat, {"a.b": 1})
        self.assertEqual(style, "dark_background")
        flat, style = pl._split_config(SimpleNamespace(logy=True, backend="agg"))
        self.assertEqual(flat, {})
        self.assertIsNone(style)

    def test_config_backend_and_flags(self):
        from types import SimpleNamespace

        self.assertEqual(pl._config_backend({"backend": "agg"}), "agg")
        self.assertIsNone(pl._config_backend({}))
        self.assertEqual(pl._config_backend(SimpleNamespace(backend="sixel")), "sixel")
        self.assertTrue(pl._config_flag({"logx": 1}, "logx"))
        self.assertFalse(pl._config_flag({}, "logx"))
        self.assertTrue(pl._config_flag(SimpleNamespace(logy=True), "logy"))

    def test_flatten_helpers(self):
        self.assertTrue(pl._has_dict([1, {"a": 1}]))
        self.assertFalse(pl._has_dict([1, 2]))
        self.assertEqual(pl._dict_entries({"a": 1}), [{"a": 1}])
        self.assertEqual(pl._dict_entries([[{"a": 1}], {"b": 2}]), [{"a": 1}, {"b": 2}])
        out = {"x": 1}
        pl._add_flat(out, "x", 2)
        pl._add_flat(out, "x", 3)
        self.assertEqual(out["x"], [1, 2, 3])
        pl._add_flat(out, "new", 1)
        self.assertEqual(out["new"], [1])
        self.assertEqual(
            pl._flatten_config("", {"a": [{"b": 1}, {"b": 2}]}, {}), {"a.b": [1, 2]}
        )

    def test_set_rcparams_converts_figsize(self):
        import matplotlib.pyplot as plt

        old = plt.rcParams["figure.figsize"]
        try:
            pl._set_rcparams({"figure.figsize": [6, 4]})
            self.assertEqual(tuple(plt.rcParams["figure.figsize"]), (6, 4))
        finally:
            plt.rcParams["figure.figsize"] = old

    def test_plot_cfg_and_options_apply_style(self):
        import matplotlib.pyplot as plt

        old_grid = plt.rcParams["axes.grid"]
        try:
            with patch("perf.plot.plt.style.use") as use:
                params = pl._plot_cfg({"style": "dark_background", "axes.grid": False})
                self.assertEqual(params["style"], "dark_background")
                self.assertFalse(params["axes.grid"])
                use.assert_called_once_with("dark_background")
            with patch("perf.plot.plt.style.use") as use:
                logx, logy = pl._plot_options(
                    {"style": "dark_background", "logx": True}
                )
                use.assert_called_once_with("dark_background")
            self.assertTrue(logx)
            self.assertFalse(logy)
        finally:
            plt.rcParams["axes.grid"] = old_grid

    def test_plot_config_object_is_callable(self):
        import matplotlib.pyplot as plt

        config = pl.PlotConfig()
        config.config = {"axes.grid": False}
        self.assertFalse(config.config["axes.grid"])
        with patch("perf.plot.plt.show"):
            config(self._df(), ["line"], ["cycles"])
        plt.close("all")
        config.config = None

    def test_figsize_falls_back_for_invalid_values(self):
        with patch.object(pl.plt, "rcParams", {"figure.figsize": None}):
            self.assertEqual(pl._figsize(), (10, 5))
        with patch.object(pl.plt, "rcParams", {"figure.figsize": (0, -1)}):
            self.assertEqual(pl._figsize(), (10, 5))
        with patch.object(pl.plt, "rcParams", {"figure.figsize": ("a", "b")}):
            self.assertEqual(pl._figsize(), (10, 5))

    def test_available_inches_guards(self):
        import sys

        self.assertIsNone(pl._available_inches("/tmp/out.png"))
        with patch.object(sys.stdout, "isatty", return_value=False):
            self.assertIsNone(pl._available_inches(None))
        with (
            patch.object(sys.stdout, "isatty", return_value=True),
            patch("os.get_terminal_size", side_effect=OSError("no tty")),
        ):
            self.assertIsNone(pl._available_inches(None))
        with (
            patch.object(sys.stdout, "isatty", return_value=True),
            patch("os.get_terminal_size", return_value=(0, 0)),
        ):
            self.assertIsNone(pl._available_inches(None))
        with (
            patch.object(sys.stdout, "isatty", return_value=True),
            patch("os.get_terminal_size", return_value=(100, 40)),
        ):
            width, height = pl._available_inches(None)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_grid_rows_cols_scales_down_to_fit(self):
        rows, cols, w, h = pl._grid_rows_cols(4, 4.0, 3.0, (5.0, 4.0))
        self.assertGreaterEqual(rows * cols, 4)
        self.assertLessEqual(cols * w, 4.0 + 1e-9)
        self.assertLessEqual(rows * h, 3.0 + 1e-9)

    def test_grid_for_tty_and_file_paths(self):
        import os
        import sys

        with (
            patch.object(sys.stdout, "isatty", return_value=True),
            patch("os.get_terminal_size", return_value=os.terminal_size((80, 24))),
        ):
            rows, cols, _, _ = pl._grid_for(4, None)
        self.assertGreaterEqual(rows * cols, 4)
        rows, cols, _, _ = pl._grid_for(4, "/tmp/out.png")
        self.assertGreaterEqual(rows * cols, 4)


class TestPlotSamplingAndDuration(_PlotTestBase):
    def _capture_plot_size(self, df, chart):
        seen = []
        orig = pl._CHART[chart]
        pl._CHART[chart] = lambda data, *a, **k: seen.append(len(data))
        try:
            with patch("perf.plot.plt.show"):
                pl.plot(df, [chart], ["cycles"])
        finally:
            pl._CHART[chart] = orig
        return seen[-1]

    def test_swarm_is_capped_at_1000(self):
        df = self._df().loc[[0, 1, 2, 3] * 300].reset_index(drop=True)
        df["cycles"] = range(len(df))
        self.assertLessEqual(self._capture_plot_size(df, "swarm"), 1000)

    def test_strip_is_capped_at_2000(self):
        df = self._df().loc[[0, 1, 2, 3] * 600].reset_index(drop=True)
        df["cycles"] = range(len(df))
        self.assertLessEqual(self._capture_plot_size(df, "strip"), 2000)

    def test_violin_is_capped_at_5000(self):
        df = self._df().loc[[0, 1, 2, 3] * 1300].reset_index(drop=True)
        df["cycles"] = range(len(df))
        self.assertLessEqual(self._capture_plot_size(df, "violin"), 5000)

    def test_default_chart_cap_samples_per_group(self):
        df = pd.DataFrame(
            {
                "name": ["a"] * 6000 + ["b"] * 6000,
                "samples": list(range(6000)) * 2,
                "cycles": range(12000),
            }
        )
        self.assertEqual(self._capture_plot_size(df, "line"), 10000)

    def test_ungrouped_sampling_failure_is_ignored(self):
        df = pd.DataFrame(
            {
                "samples": list(range(10001)),
                "cycles": range(10001),
            }
        )
        with patch.object(pd.DataFrame, "sample", side_effect=RuntimeError("no")):
            self.assertEqual(self._capture_plot_size(df, "line"), 10001)

    def test_grouped_sampling_failure_falls_back(self):
        df = pd.DataFrame(
            {
                "name": ["a", "b"] * 5001,
                "samples": list(range(10002)),
                "cycles": range(10002),
            }
        )
        with patch.object(pd.DataFrame, "sample", side_effect=RuntimeError("no")):
            self.assertEqual(self._capture_plot_size(df, "line"), 10002)

    def test_duration_axis_label_variants(self):
        self.assertIsNone(pl._duration_axis_label(pd.DataFrame({"cycles": [1]})))
        per_op = pd.DataFrame({"duration_time/operations": [1.0]})
        self.assertEqual(
            pl._duration_axis_label(per_op), "duration_time[ns]/operations"
        )
        units = pd.DataFrame({"duration_time": [1.0, 2.0], "operations": [1, 1]})
        self.assertEqual(pl._duration_axis_label(units), "duration_time[ns]")
        threaded = pd.DataFrame({"duration_time": [1.0, 2.0], "threads": [1, 2]})
        self.assertEqual(pl._duration_axis_label(threaded), "duration_time[ns]/threads")

    def test_duration_ns_values_handles_operations_and_threads(self):
        df = pd.DataFrame(
            {
                "duration_time": [8.0, 9.0, 10.0],
                "operations": [2.0, 0.0, None],
                "threads": [2.0, 3.0, 0.0],
            }
        )
        vals = pl._duration_ns_values(df)
        self.assertTrue(pd.isna(vals.iloc[1]))
        self.assertTrue(pd.isna(vals.iloc[2]))
        self.assertEqual(vals.iloc[0], 2.0)
        self.assertIsNone(pl._duration_ns_values(pd.DataFrame({"x": [1]})))

    def test_duration_per_op_values_without_threads_or_denominator(self):
        self.assertTrue(pl._duration_per_op_ns_values(pd.DataFrame({"x": [1]})).empty)
        plain = pd.DataFrame({"duration_time/operations": [4.0, 6.0]})
        self.assertEqual(pl._duration_per_op_ns_values(plain).tolist(), [4.0, 6.0])
        threaded = pd.DataFrame(
            {"duration_time/operations": [4.0, 6.0], "threads": [2.0, 0.0]}
        )
        vals = pl._duration_per_op_ns_values(threaded)
        self.assertEqual(vals.iloc[0], 2.0)
        self.assertTrue(pd.isna(vals.iloc[1]))

    def test_apply_duration_label_updates_matching_axes(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlabel("duration_time")
        ax.set_ylabel("duration_time/operations")
        pl._apply_duration_label(ax, ["duration_time"], "duration_time[ns]")
        self.assertEqual(ax.get_xlabel(), "duration_time[ns]")
        self.assertEqual(ax.get_ylabel(), "duration_time[ns]")
        ax.set_xlabel("cycles")
        pl._apply_duration_label(ax, ["cycles"], "duration_time[ns]")
        self.assertEqual(ax.get_xlabel(), "cycles")
        pl._apply_duration_label(ax, ["duration_time"], "")
        plt.close(fig)

    def test_apply_duration_label_swallows_errors(self):
        ax = Mock()
        ax.get_xlabel.side_effect = RuntimeError("boom")
        ax.get_ylabel.side_effect = RuntimeError("boom")
        pl._apply_duration_label(ax, ["duration_time"], "duration_time[ns]")

    def test_bar_mean_dots_skips_bad_patches(self):
        ax = Mock()
        zero = Mock()
        zero.get_width.return_value = 0
        zero.get_height.return_value = 1
        broken = Mock()
        broken.get_width.side_effect = RuntimeError("boom")
        good = Mock()
        good.get_width.return_value = 1.0
        good.get_height.return_value = 2.0
        good.get_x.return_value = 0.0
        ax.patches = [zero, broken, good]
        pl._bar_mean_dots(ax)
        ax.scatter.assert_called_once()
        self.assertEqual(ax.scatter.call_args.args[1], [2.0])


class TestPlotLayoutHelpers(_PlotTestBase):
    def test_visible_order_and_extra_event(self):
        df = self._df()
        self.assertIsNone(pl._visible_order(df, None, ["a"], "cycles"))
        self.assertIsNone(pl._visible_order(df, "name", None, "cycles"))
        self.assertEqual(pl._visible_order(df, "name", ["a"], "missing"), [])
        self.assertEqual(
            pl._visible_order(df, "name", ["b", "a"], "cycles"), ["b", "a"]
        )
        extra = pl._extra_for_event(
            df, "name", ["a", "b"], {"a": "red", "b": "blue", "c": "green"}, "cycles"
        )
        self.assertEqual(extra["hue_order"], ["a", "b"])
        self.assertEqual(extra["palette"], {"a": "red", "b": "blue"})
        self.assertEqual(pl._extra_for_event(df, None, None, None, "cycles"), {})

    def test_shared_palette_sizes_and_missing_labels(self):
        df = pd.DataFrame({"name": [f"n{i}" for i in range(11)]})
        hue_order, palette = pl._shared_palette(df, "name")
        self.assertEqual(len(hue_order), 11)
        self.assertEqual(set(palette), set(hue_order))
        self.assertEqual(pl._shared_palette(df, "missing"), (None, None))
        self.assertEqual(pl._shared_palette(df.iloc[:0], "name"), (None, None))

    def test_numeric_column_and_hue_defaults(self):
        df = self._df()
        self.assertIsNone(pl._numeric_column(df, "missing"))
        self.assertEqual(
            pl._numeric_column(pd.DataFrame({"x": ["1", "bad"]}), "x").tolist()[0],
            1.0,
        )
        self.assertEqual(pl._default_group(df, "name"), "name")
        self.assertEqual(pl._default_group(df, pl._UNSET), "name")
        no_name = pd.DataFrame({"cycles": [1]})
        self.assertIsNone(pl._default_group(no_name, pl._UNSET))
        self.assertEqual(pl._hue_kw(None, None, None, a=1), {"a": 1})
        self.assertEqual(
            pl._hue_kw("name", {"a": 1}, ["a"], a=1),
            {"a": 1, "hue": "name", "palette": {"a": 1}, "hue_order": ["a"]},
        )

    def test_plot_primitives_call_through(self):
        calls = []

        def fn(**kw):
            calls.append(kw)
            return kw

        self.assertIn("data", pl._dist([], "x", fn, group=None))
        self.assertEqual(calls[-1]["x"], "x")
        self.assertIn("y", pl._cat([], "x", fn, group=None))
        self.assertEqual(calls[-1]["y"], "x")
        df = self._df()
        pl._xy(df, "cycles", fn, group=None)
        self.assertIn("x", calls[-1])

    def test_legend_right_scales_with_label_length(self):
        self.assertAlmostEqual(pl._legend_right(["a"]), 1.0 - 0.12)
        self.assertLess(pl._legend_right(["x" * 100]), 0.6)

    def test_layout_with_legend_handles_render_failures(self):
        fig = Mock()
        fig.tight_layout.side_effect = RuntimeError("boom")
        leg = Mock()
        leg.get_window_extent.side_effect = RuntimeError("boom")
        right, overflow = pl._layout_with_legend(fig, leg, 0.75)
        self.assertEqual(right, 0.75)
        self.assertEqual(overflow, 0.0)

    def test_move_legend_below_wraps_for_narrow_figures(self):
        import matplotlib.pyplot as plt

        fig, _ = plt.subplots(figsize=(4, 3))
        try:
            handles = [plt.Line2D([], [], label=f"h{i}") for i in range(10)]
            labels = [f"long label number {i}" for i in range(10)]
            leg = pl._move_legend_below(fig, handles, labels)
            self.assertIsNotNone(leg)
            self.assertEqual(len(leg.get_texts()), 10)
        finally:
            plt.close(fig)
        fig, _ = plt.subplots()
        try:
            leg = pl._move_legend_below(fig, [plt.Line2D([], [])], ["x" * 60])
            self.assertEqual(len(leg.get_texts()), 1)
        finally:
            plt.close(fig)

    def test_single_legend_collects_and_dedupes(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="a")
        ax.plot([0, 1], [1, 0], label="b")
        ax.legend()
        try:
            self.assertEqual(pl._single_legend(fig, [ax], ["b", "a"]), 2)
            self.assertEqual(len(fig.legends), 1)
            self.assertEqual(
                [t.get_text() for t in fig.legends[0].get_texts()], ["b", "a"]
            )
        finally:
            plt.close(fig)
        fig, ax = plt.subplots()
        try:
            self.assertEqual(pl._single_legend(fig, [ax]), 0)
        finally:
            plt.close(fig)

    def test_single_legend_prefers_below_and_handles_overflow(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="a")
        try:
            with patch.object(pl, "_move_legend_below", return_value=None) as below:
                self.assertEqual(pl._single_legend(fig, [ax], prefer_below=True), 1)
                below.assert_called_once()
        finally:
            plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="a")
        try:
            with (
                patch.object(pl, "_layout_with_legend", return_value=(0.8, 2.0)),
                patch.object(pl, "_move_legend_below", return_value=None) as below,
            ):
                self.assertEqual(pl._single_legend(fig, [ax]), 1)
                below.assert_called_once()
        finally:
            plt.close(fig)

    def test_style_axes(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            pl._style_axes(ax)
            self.assertFalse(ax.spines["top"].get_visible())
            self.assertFalse(ax.spines["right"].get_visible())
            self.assertTrue(ax.get_axisbelow())
        finally:
            plt.close(fig)

    def test_plot_paths_for_directory_output(self):
        paths = pl._resolve_plot_paths("/tmp/plots", [["line"], ["scatter"]])
        self.assertEqual(
            [str(p) for p in paths],
            [
                "/tmp/plots/line.png",
                "/tmp/plots/scatter.png",
            ],
        )
        paths = pl._resolve_plot_paths("o.svg", [["line,scatter"]])
        self.assertEqual([str(p) for p in paths], ["o.svg"])

    def test_save_and_show_or_save(self):
        import tempfile
        from pathlib import Path

        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp:
            fig, _ = plt.subplots()
            path = pl._save_fig(fig, Path(tmp) / "nested" / "out.png")
            self.assertTrue(Path(path).exists())
            fig, _ = plt.subplots()
            with patch("perf.plot.plt.show"):
                self.assertIsNone(pl._show_or_save(fig, None))
            fig, _ = plt.subplots()
            saved = pl._show_or_save(fig, Path(tmp) / "saved.png")
            self.assertTrue(Path(saved).exists())


class TestPlotBackendAndAscii(unittest.TestCase):
    def setUp(self):
        import matplotlib.pyplot as plt

        pl._ensure_backend("Agg")
        self.addCleanup(plt.close, "all")

    def test_backend_name_aliases_and_gui(self):
        import matplotlib

        self.assertEqual(
            pl._backend_name(matplotlib, "sixel"),
            "module://matplotlib_backend_sixel",
        )
        self.assertEqual(pl._backend_name(matplotlib, "ascii"), "module://mpl_ascii")
        self.assertEqual(pl._backend_name(matplotlib, "Agg"), "Agg")
        gui = pl._backend_name(matplotlib, "window")
        self.assertTrue(gui)

    def test_stdout_isatty_and_module_available_guards(self):
        noisy = Mock()
        noisy.isatty.side_effect = RuntimeError("boom")
        with patch("sys.stdout", noisy):
            self.assertFalse(pl._stdout_isatty())
        self.assertFalse(pl._module_available("definitely_missing_module_xyz"))

    def test_da1_parser_variants(self):
        self.assertTrue(pl._da1_supports_sixel(b"\x1b[?4c"))
        self.assertTrue(pl._da1_supports_sixel(b"\x1b[?62;4;6c"))
        self.assertFalse(pl._da1_supports_sixel(b"\x1b[?62;6c"))
        self.assertFalse(pl._da1_supports_sixel(None))

    def _fake_termios(self):
        attrs = [0, 0, 0, 0, 0, 0, [0] * 7]
        return attrs

    def test_query_sixel_support_parses_reply(self):
        attrs = self._fake_termios()
        with (
            patch("os.open", return_value=99),
            patch("termios.tcgetattr", return_value=attrs),
            patch("termios.tcsetattr") as setattr,
            patch("os.write"),
            patch("select.select", return_value=([99], [], [])),
            patch("os.read", return_value=b"\x1b[?62;4;6c"),
            patch("os.close") as close,
        ):
            self.assertTrue(pl._query_sixel_support(timeout=1.0))
        self.assertGreaterEqual(setattr.call_count, 2)
        close.assert_called_once_with(99)

    def test_query_sixel_support_without_reply(self):
        attrs = self._fake_termios()
        with (
            patch("os.open", return_value=99),
            patch("termios.tcgetattr", return_value=attrs),
            patch("termios.tcsetattr"),
            patch("os.write"),
            patch("select.select", return_value=([], [], [])),
            patch("os.read", return_value=b""),
            patch("os.close"),
        ):
            self.assertFalse(pl._query_sixel_support(timeout=0.01))

    def test_query_sixel_support_open_and_write_failures(self):
        with patch("os.open", side_effect=OSError("no tty")):
            self.assertFalse(pl._query_sixel_support())
        attrs = self._fake_termios()
        with (
            patch("os.open", return_value=99),
            patch("termios.tcgetattr", return_value=attrs),
            patch("termios.tcsetattr"),
            patch("os.write", side_effect=OSError("boom")),
            patch("os.close"),
        ):
            self.assertFalse(pl._query_sixel_support())

    def test_terminal_identity_and_capabilities(self):
        import os

        pl._terminal_capabilities.cache_clear()
        self.addCleanup(pl._terminal_capabilities.cache_clear)
        env = {"TERM": "xterm", "TERM_PROGRAM": "tmux", "LC_TERMINAL": "iTerm"}
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(pl, "_stdout_isatty", return_value=False),
        ):
            self.assertEqual(pl._terminal_identity(), ("xterm", "tmux", "iterm", False))
            caps = pl._terminal_capabilities(*pl._terminal_identity())
        self.assertEqual(caps, {"sixel": False, "ascii": True})

    def test_auto_backend_respects_env_and_inline(self):
        import matplotlib

        with patch.dict("os.environ", {"MPLBACKEND": "Agg"}, clear=True):
            with patch.object(matplotlib, "get_backend", return_value="Agg"):
                self.assertEqual(pl._auto_backend(matplotlib), "Agg")
        inline = "module://matplotlib_inline.backend_inline"
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(matplotlib, "get_backend", return_value=inline),
        ):
            self.assertEqual(pl._auto_backend(matplotlib), inline)

    def test_auto_backend_candidate_fallbacks(self):
        import matplotlib

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(pl, "_terminal_supports_sixel", return_value=True),
            patch.object(pl, "_terminal_supports_ascii", return_value=True),
            patch.object(pl, "_module_available", return_value=True),
            patch.object(pl.shutil, "which", return_value="/usr/bin/convert"),
            patch.object(matplotlib, "get_backend", return_value="Agg"),
            patch.object(
                matplotlib, "use", side_effect=[RuntimeError("sixel"), None]
            ) as use,
        ):
            self.assertEqual(pl._auto_backend(matplotlib), "Agg")
        self.assertEqual(use.call_count, 2)
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(matplotlib, "get_backend", return_value="Agg"),
            patch.object(matplotlib, "use", side_effect=RuntimeError("boom")),
        ):
            self.assertEqual(pl._auto_backend(matplotlib), "Agg")

    def test_ensure_backend_without_matplotlib(self):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("no matplotlib")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            self.assertIsNone(pl._ensure_backend("Agg"))

    def test_ascii_prepare_filters_invalid_artists(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots()
        try:
            nan_line = ax.plot([0, np.nan], [0, 1])[0]
            broken_line = ax.plot([0, 1], [0, 1])[0]
            inf_line = ax.plot([0, np.inf], [1, 2])[0]
            scatter = ax.scatter([0, np.inf], [0, 1])
            good = LineCollection([[(0, 0), (1, 1), (2, 2)]])
            bad = LineCollection([[(np.inf, 0), (1, 1)]])
            ax.add_collection(good)
            ax.add_collection(bad)
            keep = plt.Rectangle((0, 0), 1, 1)
            drop = plt.Rectangle((0, 0), np.inf, 1)
            ax.add_patch(keep)
            ax.add_patch(drop)
            ax.legend([inf_line], ["a"])
            with patch.object(
                broken_line, "get_xdata", side_effect=RuntimeError("boom")
            ):
                with patch.object(ax, "get_xlim", side_effect=RuntimeError("boom")):
                    pl._ascii_prepare(fig)
            self.assertNotIn(nan_line, ax.lines)
            self.assertIn(broken_line, ax.lines)
            self.assertTrue(np.isfinite(inf_line.get_xdata()).all())
            self.assertTrue(np.isfinite(inf_line.get_ydata()).all())
            offsets = np.asarray(scatter.get_offsets(), dtype=float)
            self.assertTrue(np.isfinite(offsets).all())
            self.assertEqual(len(offsets), 1)
            self.assertIn(good, ax.collections)
            self.assertNotIn(bad, ax.collections)
            self.assertEqual(len(good.get_segments()), 2)
            self.assertIn(keep, ax.patches)
            self.assertNotIn(drop, ax.patches)
            self.assertIsNone(ax.get_legend())
        finally:
            plt.close(fig)

    def test_ascii_prepare_handles_empty_collections_and_bad_patches(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots()
        try:
            everywhere_bad = LineCollection([[(np.inf, 0), (np.nan, 1)]])
            ax.add_collection(everywhere_bad)
            scatter = ax.scatter([np.inf, np.nan], [np.nan, np.inf])
            broken = plt.Rectangle((0, 0), 1, 1)
            ax.add_patch(broken)
            with patch.object(broken, "get_x", side_effect=RuntimeError("boom")):
                with patch.object(
                    everywhere_bad, "get_segments", side_effect=RuntimeError("boom")
                ):
                    pl._ascii_prepare(fig)
            self.assertNotIn(everywhere_bad, ax.collections)
            self.assertNotIn(scatter, ax.collections)
            self.assertIn(broken, ax.patches)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
