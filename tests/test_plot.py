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
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import perf

pl = importlib.import_module("perf.plot")


class TestChartRegistry(unittest.TestCase):
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
            self.assertIn(name, pl.CHART)
        self.assertEqual(len(pl.CHART), 13)

    def test_all_charts_execute(self):
        import matplotlib.pyplot as plt

        df = self._df()
        for name in pl.CHART:
            with self.subTest(chart=name):
                try:
                    with patch("perf.plot.plt.show"):
                        pl.CHART[name](df, "cycles")
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
        for name in pl.CHART:
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
        orig_line = pl.CHART["line"]
        orig_scatter = pl.CHART["scatter"]
        line = Mock(wraps=orig_line)
        scatter = Mock(wraps=orig_scatter)
        pl.CHART["line"] = line
        pl.CHART["scatter"] = scatter
        try:
            pl.plot(df, ["line,scatter"], [["cycles"], ["inst"]])
            fig = plt.gcf()
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(line.call_count, 2)
            self.assertEqual(scatter.call_count, 2)
            mock_show.assert_called_once()
        finally:
            pl.CHART["line"] = orig_line
            pl.CHART["scatter"] = orig_scatter
            plt.close("all")

    @patch("perf.plot.plt.show")
    def test_log_scales(self, mock_show):
        import matplotlib.pyplot as plt

        df = self._df()
        try:
            pl.plot(df, ["line"], ["cycles"], logx=True, logy=True)
            fig = plt.gcf()
            for ax in fig.axes:
                self.assertEqual(ax.get_xscale(), "log")
                self.assertEqual(ax.get_yscale(), "log")
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
            orig = pl.CHART[name]
            mocked = Mock(wraps=orig)
            pl.CHART[name] = mocked
            try:
                pl.plot(df, [name], ["cycles"])
                _, kw = mocked.call_args
                self.assertEqual(kw.get("errorbar"), pl._nonnegative_sd)
            finally:
                pl.CHART[name] = orig
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
            orig = pl.CHART[name]
            mocked = Mock(wraps=orig)
            pl.CHART[name] = mocked
            try:
                pl.plot(df, [name], ["cycles"])
                _, kw = mocked.call_args
                self.assertEqual(kw.get("errorbar"), pl._nonnegative_sd)
                self.assertEqual(kw.get("capsize"), pl._ERRORBAR_CAPS)
                self.assertLessEqual(kw.get("capsize"), 0.1)
            finally:
                pl.CHART[name] = orig
                plt.close("all")
        orig = pl.CHART["line"]
        mocked = Mock(wraps=orig)
        pl.CHART["line"] = mocked
        try:
            pl.plot(df, ["line"], ["cycles"])
            _, kw = mocked.call_args
            self.assertEqual(kw.get("errorbar"), pl._nonnegative_sd)
            self.assertEqual(kw.get("err_style"), "bars")
            self.assertEqual(kw.get("marker"), "o")
            self.assertEqual(kw.get("err_kws", {}).get("capsize"), pl._LINE_ERR_CAPS)
            self.assertLessEqual(kw.get("err_kws", {}).get("capsize"), 3)
        finally:
            pl.CHART["line"] = orig
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

        path = Path(__file__).resolve().parent.parent / "bin" / "perf"
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

    def test_plot_config_round_trip(self):
        cli = self._cli()
        args = Mock(config=None)
        config = cli._resolve_plot_config(args, ["--config.axes.grid=False"])
        perf.plot.config = config
        params = perf.plot.config
        self.assertFalse(params["axes.grid"])


if __name__ == "__main__":
    unittest.main()
