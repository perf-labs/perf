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
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

tune_mod = importlib.import_module("perf.tune")


class TestProfiles(unittest.TestCase):
    def test_predefined_profiles(self):
        self.assertEqual(
            tune_mod.profiles(),
            ("default", "latency", "throughput"),
        )

    def test_default_profile_covers_every_setting(self):
        profile = tune_mod.resolve_profile()
        self.assertEqual(tuple(profile), tune_mod._TUNABLE)
        self.assertTrue(profile["aslr"])
        self.assertEqual(profile["swappiness"], 60)

    def test_latency_profile(self):
        profile = tune_mod.resolve_profile("latency")
        self.assertFalse(profile["aslr"])
        self.assertEqual(profile["governor"], "performance")
        self.assertEqual(profile["cpu_frequency"], "max")
        self.assertFalse(profile["smt"])
        self.assertEqual(profile["thp"], "never")
        self.assertEqual(profile["thp_defrag"], "never")
        self.assertEqual(profile["swappiness"], 0)
        self.assertEqual(profile["stat_interval"], 120)
        self.assertEqual(profile["perf_event_max_sample_rate"], 1)
        self.assertEqual(profile["perf_event_mlock_kb"], 65536)

    def test_throughput_profile(self):
        profile = tune_mod.resolve_profile("throughput")
        self.assertFalse(profile["aslr"])
        self.assertEqual(profile["cpu_frequency"], "max")
        self.assertTrue(profile["smt"])
        self.assertEqual(profile["thp"], "always")
        self.assertEqual(profile["thp_defrag"], "always")
        self.assertEqual(profile["swappiness"], 10)
        self.assertEqual(profile["perf_event_max_sample_rate"], 1)

    def test_unknown_profile(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("nope")
        self.assertIn("unknown profile 'nope'", str(ex.exception))
        self.assertIn("latency", str(ex.exception))

    def test_profile_names_are_not_mutated(self):
        tune_mod.resolve_profile("latency", {"aslr": True})
        self.assertFalse(tune_mod.resolve_profile("latency")["aslr"])


class TestProfileFiles(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, data):
        path = self.dir / "profile.json"
        path.write_text(json.dumps(data) if not isinstance(data, str) else data)
        return str(path)

    def test_json_merges_over_default(self):
        path = self.write({"aslr": False, "swappiness": 0})
        profile = tune_mod.resolve_profile(path)
        self.assertFalse(profile["aslr"])
        self.assertEqual(profile["swappiness"], 0)
        self.assertEqual(profile["governor"], "powersave")

    def test_json_without_suffix(self):
        path = self.dir / "custom"
        path.write_text('{"governor": "performance"}')
        profile = tune_mod.resolve_profile(str(path))
        self.assertEqual(profile["governor"], "performance")

    def test_missing_json(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile(str(self.dir / "missing.json"))
        self.assertIn("cannot read profile", str(ex.exception))

    def test_invalid_json(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile(self.write("{nope"))
        self.assertIn("invalid JSON", str(ex.exception))

    def test_json_must_be_an_object(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile(self.write([]))
        self.assertEqual(
            str(ex.exception),
            "profile file must contain a JSON object",
        )

    def test_unknown_json_key(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile(self.write({"nope": True}))
        self.assertIn("unknown profile key 'nope'", str(ex.exception))


class TestOverrides(unittest.TestCase):
    def test_boolean_override(self):
        profile = tune_mod.resolve_profile("latency", {"aslr": True})
        self.assertTrue(profile["aslr"])

    def test_string_boolean_override(self):
        profile = tune_mod.resolve_profile("latency", {"aslr": "false"})
        self.assertFalse(profile["aslr"])

    def test_prefixed_override(self):
        profile = tune_mod.resolve_profile("default", {"profile.aslr": False})
        self.assertFalse(profile["aslr"])

    def test_key_value_override(self):
        profile = tune_mod.resolve_profile("default", ["swappiness=10"])
        self.assertEqual(profile["swappiness"], 10)

    def test_override_beats_profile_file(self):
        path = Path(tempfile.mkdtemp()) / "custom.json"
        self.addCleanup(shutil.rmtree, path.parent, True)
        path.write_text('{"aslr": false}')
        profile = tune_mod.resolve_profile(str(path), {"aslr": True})
        self.assertTrue(profile["aslr"])

    def test_unknown_override(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"nope": True})
        self.assertIn("unknown profile key 'nope'", str(ex.exception))

    def test_invalid_override_shape(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", 3)
        self.assertIn("profile overrides must be a dict", str(ex.exception))

    def test_invalid_key_value_override(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", ["aslr"])
        self.assertIn("expected key=value", str(ex.exception))

    def test_boolean_validation(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"aslr": "sometimes"})
        self.assertIn("needs a boolean", str(ex.exception))

    def test_aslr_accepts_levels(self):
        self.assertEqual(
            tune_mod.resolve_profile("default", {"aslr": 1})["aslr"],
            1,
        )

    def test_thp_validation(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"thp": "sometimes"})
        self.assertIn("profile thp must be one of", str(ex.exception))

    def test_thp_defrag_validation(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"thp_defrag": "sometimes"})
        self.assertIn("profile thp_defrag must be one of", str(ex.exception))

    def test_cpu_frequency_aliases(self):
        for alias, expected in (
            ("auto", "range"),
            ("default", "range"),
            ("high", "max"),
            ("low", "min"),
            ("nominal", "base"),
        ):
            profile = tune_mod.resolve_profile("default", {"cpu_frequency": alias})
            self.assertEqual(profile["cpu_frequency"], expected)

    def test_cpu_frequency_accepts_khz(self):
        profile = tune_mod.resolve_profile("default", {"cpu_frequency": 2400000})
        self.assertEqual(profile["cpu_frequency"], 2400000)

    def test_cpu_frequency_validation(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"cpu_frequency": 0})
        self.assertIn("positive frequency in kHz", str(ex.exception))

    def test_smt_validation(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"smt": "sometimes"})
        self.assertIn("needs a boolean", str(ex.exception))

    def test_perf_event_max_sample_rate_validation(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"perf_event_max_sample_rate": 0})
        self.assertIn("integer in [1, 1000000000]", str(ex.exception))

    def test_integer_validation(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"swappiness": -1})
        self.assertIn("integer in [0, 200]", str(ex.exception))

    def test_text_validation(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"governor": ""})
        self.assertIn("must not be empty", str(ex.exception))


class TestTuner(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def read(self, rel):
        return (self.root / rel).read_text()

    def row(self, df, setting):
        return df[df["setting"] == setting].iloc[0]

    def test_dry_run_does_not_write(self):
        path = self.write("proc/sys/kernel/randomize_va_space", "2\n")
        df = tune_mod.tune(
            profile="latency",
            root=self.root,
            dry_run=True,
        )
        self.assertEqual(path.read_text(), "2\n")
        row = self.row(df, "aslr")
        self.assertEqual(row["current"], "on")
        self.assertEqual(row["target"], "off")
        self.assertEqual(row["status"], "planned")

    def test_applies_setting(self):
        path = self.write("proc/sys/kernel/randomize_va_space", "2\n")
        df = tune_mod.tune(
            profile="latency",
            overrides={"aslr": False},
            root=self.root,
        )
        self.assertEqual(path.read_text(), "0")
        row = self.row(df, "aslr")
        self.assertEqual(row["status"], "tuned")

    def test_unchanged_setting(self):
        self.write("proc/sys/kernel/randomize_va_space", "0\n")
        df = tune_mod.tune(
            profile="latency",
            overrides={"aslr": False},
            root=self.root,
        )
        self.assertEqual(self.row(df, "aslr")["status"], "unchanged")

    def test_missing_setting_is_unavailable(self):
        df = tune_mod.tune(profile="latency", root=self.root, dry_run=True)
        for setting in tune_mod._TUNABLE:
            self.assertEqual(self.row(df, setting)["status"], "unavailable")

    def test_every_cpu_governor_is_written(self):
        first = self.write(
            "sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
            "powersave\n",
        )
        second = self.write(
            "sys/devices/system/cpu/cpu1/cpufreq/scaling_governor",
            "schedutil\n",
        )
        df = tune_mod.tune(
            profile="default",
            overrides={"governor": "performance"},
            root=self.root,
        )
        self.assertEqual(first.read_text(), "performance")
        self.assertEqual(second.read_text(), "performance")
        row = self.row(df, "governor")
        self.assertEqual(row["current"], "powersave,schedutil")
        self.assertEqual(row["status"], "tuned")

    def test_unsupported_governor_is_not_written(self):
        path = self.write(
            "sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
            "performance\n",
        )
        self.write(
            "sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
            "performance schedutil\n",
        )
        df = tune_mod.tune(profile="default", root=self.root)
        self.assertEqual(path.read_text().strip(), "performance")
        row = self.row(df, "governor")
        self.assertEqual(row["target"], "powersave")
        self.assertEqual(row["status"], "unsupported")

    def test_unsupported_smt_is_not_written(self):
        path = self.write(
            "sys/devices/system/cpu/smt/control",
            "notsupported\n",
        )
        df = tune_mod.tune(profile="latency", root=self.root)
        self.assertEqual(path.read_text().strip(), "notsupported")
        self.assertEqual(self.row(df, "smt")["status"], "unsupported")

    def test_turbo_is_inverted(self):
        path = self.write(
            "sys/devices/system/cpu/intel_pstate/no_turbo",
            "0\n",
        )
        df = tune_mod.tune(
            profile="default",
            overrides={"turbo": False},
            root=self.root,
        )
        self.assertEqual(path.read_text(), "1")
        row = self.row(df, "turbo")
        self.assertEqual(row["current"], "on")
        self.assertEqual(row["target"], "off")
        self.assertEqual(row["status"], "tuned")

    def test_transparent_hugepage_current(self):
        self.write(
            "sys/kernel/mm/transparent_hugepage/enabled",
            "[always] madvise never\n",
        )
        df = tune_mod.tune(profile="latency", root=self.root, dry_run=True)
        row = self.row(df, "thp")
        self.assertEqual(row["current"], "always")
        self.assertEqual(row["target"], "never")

    def test_transparent_hugepage_verification_uses_the_selected_value(self):
        self.write(
            "sys/kernel/mm/transparent_hugepage/enabled",
            "[always] madvise never\n",
        )

        def write(path, text):
            path.write_text(f"[{text}] madvise always\n")

        with patch.object(tune_mod.Tuner, "_write", side_effect=write):
            df = tune_mod.tune(
                profile="latency",
                overrides={"thp": "never"},
                root=self.root,
            )
        row = self.row(df, "thp")
        self.assertEqual(row["current"], "always")
        self.assertEqual(row["status"], "tuned")

    def test_aslr_level_is_displayed(self):
        self.write("proc/sys/kernel/randomize_va_space", "2\n")
        df = tune_mod.tune(
            profile="default",
            overrides={"aslr": 1},
            root=self.root,
            dry_run=True,
        )
        row = self.row(df, "aslr")
        self.assertEqual(row["current"], "on")
        self.assertEqual(row["target"], "partial")

    def frequency(self, cpu=0, low=800000, high=3600000, base=2400000):
        directory = f"sys/devices/system/cpu/cpu{cpu}/cpufreq"
        self.write(f"{directory}/scaling_min_freq", str(low))
        self.write(f"{directory}/scaling_max_freq", str(high))
        self.write(f"{directory}/cpuinfo_min_freq", "800000")
        self.write(f"{directory}/cpuinfo_max_freq", "3600000")
        self.write(f"{directory}/base_frequency", str(base))
        return (
            self.root / directory / "scaling_min_freq",
            self.root / directory / "scaling_max_freq",
        )

    def test_cpu_frequency_max_pins_both_limits(self):
        minimum, maximum = self.frequency()
        df = tune_mod.tune(profile="latency", root=self.root)
        self.assertEqual(minimum.read_text(), "3600000")
        self.assertEqual(maximum.read_text(), "3600000")
        row = self.row(df, "cpu_frequency")
        self.assertEqual(row["current"], "800MHz..3.6GHz")
        self.assertEqual(row["target"], "max")
        self.assertEqual(row["status"], "tuned")

    def test_cpu_frequency_range_restores_the_limits(self):
        minimum, maximum = self.frequency(low=3600000, high=3600000)
        df = tune_mod.tune(profile="default", root=self.root)
        self.assertEqual(minimum.read_text(), "800000")
        self.assertEqual(maximum.read_text(), "3600000")
        self.assertEqual(self.row(df, "cpu_frequency")["status"], "tuned")

    def test_cpu_frequency_base(self):
        minimum, maximum = self.frequency()
        tune_mod.tune(
            profile="default",
            overrides={"cpu_frequency": "base"},
            root=self.root,
        )
        self.assertEqual(minimum.read_text(), "2400000")
        self.assertEqual(maximum.read_text(), "2400000")

    def test_cpu_frequency_khz_target(self):
        minimum, maximum = self.frequency()
        df = tune_mod.tune(
            profile="default",
            overrides={"cpu_frequency": 2000000},
            root=self.root,
        )
        self.assertEqual(minimum.read_text(), "2000000")
        self.assertEqual(maximum.read_text(), "2000000")
        self.assertEqual(self.row(df, "cpu_frequency")["target"], "2GHz")

    def test_cpu_frequency_lowering_writes_minimum_first(self):
        minimum, maximum = self.frequency(low=2000000, high=3600000)
        tune_mod.tune(
            profile="default",
            overrides={"cpu_frequency": 1000000},
            root=self.root,
        )
        self.assertEqual(minimum.read_text(), "1000000")
        self.assertEqual(maximum.read_text(), "1000000")

    def test_cpu_frequency_without_cpuinfo_fails(self):
        self.frequency()
        (self.root / "sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").unlink()
        df = tune_mod.tune(profile="latency", root=self.root)
        self.assertEqual(self.row(df, "cpu_frequency")["status"], "failed")

    def test_cpu_frequency_multiple_cpus(self):
        self.frequency(cpu=0, low=800000, high=3600000)
        self.frequency(cpu=1, low=1200000, high=3600000)
        df = tune_mod.tune(profile="latency", root=self.root)
        row = self.row(df, "cpu_frequency")
        self.assertEqual(
            row["current"],
            "800MHz..3.6GHz;1.2GHz..3.6GHz",
        )
        self.assertEqual(row["status"], "tuned")

    def test_smt_control(self):
        path = self.write("sys/devices/system/cpu/smt/control", "on\n")
        df = tune_mod.tune(profile="latency", root=self.root)
        self.assertEqual(path.read_text(), "off")
        row = self.row(df, "smt")
        self.assertEqual(row["current"], "on")
        self.assertEqual(row["target"], "off")
        self.assertEqual(row["status"], "tuned")

    def test_boost_flag(self):
        path = self.write(
            "sys/devices/system/cpu/cpufreq/boost",
            "0\n",
        )
        df = tune_mod.tune(
            profile="default",
            overrides={"boost": True},
            root=self.root,
        )
        self.assertEqual(path.read_text(), "1")
        self.assertEqual(self.row(df, "boost")["status"], "tuned")

    def test_perf_event_limits(self):
        rate = self.write(
            "proc/sys/kernel/perf_event_max_sample_rate",
            "100000\n",
        )
        mlock = self.write("proc/sys/kernel/perf_event_mlock_kb", "516\n")
        df = tune_mod.tune(profile="latency", root=self.root)
        self.assertEqual(rate.read_text(), "1")
        self.assertEqual(mlock.read_text(), "65536")
        self.assertEqual(
            self.row(df, "perf_event_max_sample_rate")["status"],
            "tuned",
        )
        self.assertEqual(
            self.row(df, "perf_event_mlock_kb")["status"],
            "tuned",
        )

    def test_thp_defrag_current(self):
        self.write(
            "sys/kernel/mm/transparent_hugepage/defrag",
            "[always] madvise never\n",
        )
        df = tune_mod.tune(profile="latency", root=self.root, dry_run=True)
        row = self.row(df, "thp_defrag")
        self.assertEqual(row["current"], "always")
        self.assertEqual(row["target"], "never")

    def test_khugepaged_defrag_and_timer_migration(self):
        defrag = self.write(
            "sys/kernel/mm/transparent_hugepage/khugepaged/defrag",
            "1\n",
        )
        migration = self.write("proc/sys/kernel/timer_migration", "1\n")
        tune_mod.tune(profile="latency", root=self.root)
        self.assertEqual(defrag.read_text(), "0")
        self.assertEqual(migration.read_text(), "0")

    def test_permission_denied(self):
        self.write("proc/sys/kernel/randomize_va_space", "2\n")
        with patch.object(
            tune_mod.Tuner,
            "_write",
            side_effect=PermissionError(13, "denied"),
        ):
            df = tune_mod.tune(
                profile="latency",
                overrides={"aslr": False},
                root=self.root,
            )
        self.assertEqual(self.row(df, "aslr")["status"], "denied")

    def test_write_failure(self):
        self.write("proc/sys/kernel/randomize_va_space", "2\n")
        with patch.object(
            tune_mod.Tuner,
            "_write",
            side_effect=OSError(5, "failed"),
        ):
            df = tune_mod.tune(
                profile="latency",
                overrides={"aslr": False},
                root=self.root,
            )
        self.assertEqual(self.row(df, "aslr")["status"], "failed")

    def test_columns(self):
        df = tune_mod.tune(profile="default", root=self.root, dry_run=True)
        self.assertEqual(list(df.columns), list(tune_mod._COLUMNS))
        self.assertEqual(len(df), len(tune_mod._TUNABLE))


if __name__ == "__main__":
    unittest.main()
