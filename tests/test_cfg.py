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

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from perf import cfg as cg


class TestPerfconfigEntries(unittest.TestCase):
    def _write(self, text):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".perfconfig", delete=False, encoding="utf-8"
        )
        with handle:
            handle.write(text)
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_entries_are_scoped_to_the_section(self):
        path = self._write("[view]\nstat = p50\n# comment\n\n[plot]\nlogy = true\n")
        self.assertEqual(cg.perfconfig_entries("view", path), {"stat": "p50"})
        self.assertEqual(cg.perfconfig_entries("plot", path), {"logy": "true"})
        self.assertEqual(cg.perfconfig_entries("compare", path), {})

    def test_duplicate_entries_are_collected_in_order(self):
        path = self._write("[view]\nstat=a\nstat=b\nother=c\n")
        self.assertEqual(
            cg.perfconfig_entries("view", path),
            {"stat": ["a", "b"], "other": "c"},
        )

    def test_empty_values_and_inline_comments_are_stripped(self):
        path = self._write("[view]\nstat = min  # one row\nempty =\n; ignored\n")
        self.assertEqual(cg.perfconfig_entries("view", path), {"stat": "min"})

    def test_invalid_or_missing_file_returns_empty_entries(self):
        self.assertEqual(cg.perfconfig_entries("view", "/does/not/exist"), {})
        path = self._write("[view\nnope\n")
        self.assertEqual(cg.perfconfig_entries("view", path), {})

    def test_nearest_perfconfig_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "a" / "b"
            child.mkdir(parents=True)
            (root / ".perfconfig").write_text("[view]\nstat = root\n")
            (root / "a" / ".perfconfig").write_text("[view]\nstat = near\n")
            with patch.object(cg.Path, "cwd", return_value=child):
                self.assertEqual(
                    cg.perfconfig_entries("view"),
                    {"stat": "near"},
                )


class TestApplyPerfconfig(unittest.TestCase):
    def _write(self, text):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".perfconfig", delete=False, encoding="utf-8"
        )
        with handle:
            handle.write(text)
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_duplicate_events_are_injected_as_separate_flags(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--event", action="append")
        parser.add_argument("target", nargs="*")
        path = self._write("[analyze]\nevent=assembly\nevent=source\n")
        argv = cg.apply_perfconfig(parser, ["a.out:main"], "analyze", path)
        self.assertEqual(argv, ["--event=assembly", "--event=source", "a.out:main"])
        args = parser.parse_args(argv)
        self.assertEqual(args.event, ["assembly", "source"])

    def test_output_template_and_flags(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", action="append", nargs="?", const=True)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("target", nargs="*")
        path = self._write(
            "[view]\noutput = {name}.json\ndry-run = true\nunknown = 1\n"
        )
        argv = cg.apply_perfconfig(parser, ["data/"], "view", path)
        self.assertEqual(argv, ["--output={name}.json", "--dry-run", "data/"])
        args = parser.parse_args(argv)
        self.assertTrue(args.dry_run)
        self.assertEqual(args.output, ["{name}.json"])
        self.assertTrue(args._output_from_config)
        self.assertEqual(args._output_template, "{name}.json")

    def test_cli_wins_and_skip_and_prepend(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        tune = sub.add_parser("tune")
        tune.add_argument("--dry-run", action="store_true")
        tune.add_argument("--output", action="append", nargs="?", const=True)
        parser.add_argument("--output", action="append", nargs="?", const=True)
        path = self._write("[system]\ndry-run = true\noutput = out.json\n")
        argv = cg.apply_perfconfig(
            parser,
            ["tune", "--dry-run"],
            "system",
            path,
            options=tune,
            prepend=True,
        )
        self.assertEqual(argv, ["tune", "--output=out.json", "--dry-run"])

    def test_invalid_boolean_is_rejected(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dry-run", action="store_true")
        path = self._write("[view]\ndry-run = sometimes\n")
        with self.assertRaises(SystemExit):
            cg.apply_perfconfig(parser, [], "view", path)


class TestGenericConfig(unittest.TestCase):
    def test_normalize_flattens_dicts_and_alternatives(self):
        self.assertEqual(
            cg.normalize_config(
                {
                    "figure": {"figsize": [6, 4]},
                    "cache": [{"hit_rate": 50}, {"hit_rate": 100}],
                }
            ),
            {"figure.figsize": [6, 4], "cache.hit_rate": [50, 100]},
        )
        out = {}
        self.assertEqual(cg.flatten_config("", {"a": 1}, out), {"a": 1})
        self.assertEqual(cg.add_flat(out, "a", 2), None)
        self.assertEqual(out, {"a": [1, 2]})
        self.assertTrue(cg.has_dict([1, {"a": 1}]))
        self.assertFalse(cg.has_dict([1, 2]))
        self.assertEqual(cg.dict_entries([[{"a": 1}], {"b": 2}]), [{"a": 1}, {"b": 2}])

    def test_deep_merge_replaces_scalars_and_merges_dicts(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        self.assertEqual(
            cg.deep_merge(base, {"a": {"c": 4}, "d": 5}),
            {"a": {"b": 1, "c": 4}, "d": 5},
        )
        self.assertEqual(cg.deep_merge({"a": 1}, 2), 2)
        self.assertEqual(cg.deep_merge(1, {"a": 2}), {"a": 2})

    def test_config_backend_and_flag_accept_dicts_and_objects(self):
        obj = argparse.Namespace(backend="sixel", logx=True)
        self.assertEqual(cg.config_backend({"backend": "agg"}), "agg")
        self.assertIsNone(cg.config_backend({}))
        self.assertEqual(cg.config_backend(obj), "sixel")
        self.assertTrue(cg.config_flag({"logx": 1}, "logx"))
        self.assertFalse(cg.config_flag({}, "logx"))
        self.assertTrue(cg.config_flag(obj, "logx"))


class TestOutputHelpers(unittest.TestCase):
    def test_output_targets_and_table_visibility(self):
        self.assertEqual(cg.output_targets(None), [])
        self.assertEqual(cg.output_targets(True), [True])
        self.assertEqual(cg.output_targets("-"), [True])
        self.assertEqual(cg.output_targets("none"), [])
        self.assertEqual(
            cg.output_targets(["a.json", ["b.json", "off"]]), ["a.json", "b.json"]
        )
        self.assertFalse(cg.has_output(None))
        self.assertTrue(cg.has_output("a.json"))
        self.assertTrue(cg.show_table(None))
        self.assertFalse(cg.show_table("a.json"))
        self.assertTrue(cg.show_table("a.json", from_config=True))

    def test_output_template_only_accepts_a_relative_name(self):
        self.assertEqual(cg.output_template("{name}.json"), "{name}.json")
        self.assertIsNone(cg.output_template(None))
        self.assertIsNone(cg.output_template("none"))
        self.assertIsNone(cg.output_template("dir/"))
        self.assertIsNone(cg.output_template("/tmp/{name}.json"))
        self.assertIsNone(cg.output_template("sub/{name}.json"))

    def test_option_helpers(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", action="append", nargs="?", const=True)
        parser.add_argument("--dry-run", action="store_true")
        self.assertTrue(cg.is_known(parser, "output"))
        self.assertTrue(cg.is_known(parser, "config.style", ("config.",)))
        self.assertFalse(cg.is_known(parser, "config.style"))
        self.assertEqual(cg.aliases(parser, "dry_run"), {"dry-run", "dry_run"})
        self.assertTrue(cg.is_flag_only(parser, "dry-run"))
        self.assertFalse(cg.is_flag_only(parser, "output"))
        self.assertEqual(
            cg.given_options(["-m", "latency", "--event=cycles", "--", "--quiet"]),
            {"m", "event"},
        )
        self.assertTrue(cg.truthy("ON"))
        self.assertFalse(cg.truthy("off"))
        self.assertIsNone(cg.truthy("sometimes"))


class TestCoverageGaps(unittest.TestCase):
    def _write(self, text):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".perfconfig", delete=False, encoding="utf-8"
        )
        with handle:
            handle.write(text)
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_perfconfig_value_handles_none_list_and_str(self):
        self.assertEqual(cg._perfconfig_value(None), "")
        self.assertEqual(cg._perfconfig_value(["a", "b"]), "a\nb")
        self.assertEqual(cg._perfconfig_value("  x  "), "x")

    def test_perfconfig_path_handles_cwd_failure(self):
        with patch.object(cg.Path, "cwd", side_effect=OSError("denied")):
            self.assertIsNone(cg._perfconfig_path())

    def test_perfconfig_path_ignores_isfile_errors_and_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp) / "sub"
            child.mkdir()
            with (
                patch.object(cg.Path, "cwd", return_value=child),
                patch.object(cg.Path, "home", return_value=Path(tmp)),
                patch.object(cg.Path, "is_file", side_effect=OSError("denied")),
            ):
                self.assertIsNone(cg._perfconfig_path())
                self.assertEqual(cg.perfconfig_entries("view"), {})
            with (
                patch.object(cg.Path, "cwd", return_value=child),
                patch.object(cg.Path, "home", return_value=Path(tmp)),
            ):
                self.assertEqual(cg.perfconfig_entries("view"), {})

    def test_triple_duplicate_entries_stay_a_list(self):
        path = self._write("[view]\nstat=a\nstat=b\nstat=c\n")
        self.assertEqual(cg.perfconfig_entries("view", path), {"stat": ["a", "b", "c"]})

    def test_history_fallback_without_entries_attr(self):
        path = self._write("[view]\nstat = p50\n")
        with patch.object(cg, "_PerfconfigDict", dict):
            self.assertEqual(cg.perfconfig_entries("view", path), {"stat": "p50"})

    def test_output_template_with_existing_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = os.path.basename(tmp)
            parent = os.path.dirname(tmp)
            cwd = os.getcwd()
            os.chdir(parent)
            try:
                self.assertIsNone(cg.output_template(name))
            finally:
                os.chdir(cwd)

    def test_output_template_ignores_isdir_errors(self):
        with patch.object(cg.Path, "is_dir", side_effect=OSError("denied")):
            self.assertEqual(cg.output_template("{name}.json"), "{name}.json")

    def test_option_name_falls_back_to_short_flag(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--mode", default=None)
        self.assertEqual(cg._option_name(parser, "mode"), "mode")
        parser2 = argparse.ArgumentParser()
        parser2.add_argument("-x", dest="extra")
        self.assertEqual(cg._option_name(parser2, "extra"), "x")
        self.assertEqual(cg._option_name(parser2, "missing"), "missing")
        parser3 = argparse.ArgumentParser()
        parser3.add_argument("target", nargs="*")
        self.assertEqual(cg._option_name(parser3, "target"), "target")

    def test_event_skipped_when_list_given_without_event(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--event", action="append", default=None)
        parser.add_argument("--list", action="store_true")
        parser.add_argument("target", nargs="*")
        path = self._write("[analyze]\nevent = assembly\n")
        argv = cg.apply_perfconfig(parser, ["--list", "a.out:main"], "analyze", path)
        self.assertEqual(argv, ["--list", "a.out:main"])

    def test_false_flag_is_not_injected(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("target", nargs="*")
        path = self._write("[view]\ndry-run = false\n")
        argv = cg.apply_perfconfig(parser, ["data/"], "view", path)
        self.assertEqual(argv, ["data/"])

    def test_skip_callback_suppresses_entries(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--event", action="append", default=None)
        parser.add_argument("target", nargs="*")
        path = self._write("[analyze]\nevent = assembly\n")
        argv = cg.apply_perfconfig(
            parser,
            ["a.out:main"],
            "analyze",
            path,
            skip=lambda key, value, cmd, given: True,
        )
        self.assertEqual(argv, ["a.out:main"])
