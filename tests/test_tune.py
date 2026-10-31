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

import errno
import importlib
import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

tune_mod = importlib.import_module("perf.tune")


class TestProfiles(unittest.TestCase):
    def test_predefined_profiles(self):
        self.assertEqual(
            tuple(tune_mod._PROFILES),
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

    def test_pyperf_tuning_is_covered(self):
        for name in (
            "aslr",
            "governor",
            "cpu_frequency",
            "turbo",
            "turbo_msr",
            "perf_event_max_sample_rate",
            "irqbalance",
            "irq_affinity",
            "power_supply",
            "isolcpus",
            "rcu_nocbs",
            "nohz_full",
        ):
            self.assertIn(name, tune_mod._TUNABLE)
        latency = tune_mod.resolve_profile("latency")
        self.assertEqual(latency["perf_event_max_sample_rate"], 1)
        self.assertEqual(latency["governor"], "performance")
        self.assertEqual(latency["cpu_frequency"], "max")
        self.assertFalse(latency["turbo"])
        self.assertFalse(latency["irqbalance"])
        self.assertTrue(latency["isolcpus"])
        self.assertTrue(latency["rcu_nocbs"])

    def test_latency_and_throughput_are_distinct(self):
        latency = tune_mod.resolve_profile("latency")
        throughput = tune_mod.resolve_profile("throughput")
        self.assertNotEqual(latency, throughput)
        self.assertNotEqual(latency["smt"], throughput["smt"])
        self.assertFalse(throughput["irqbalance"])
        self.assertNotEqual(latency["turbo"], throughput["turbo"])
        self.assertNotEqual(latency["thp"], throughput["thp"])
        self.assertNotEqual(latency["isolcpus"], throughput["isolcpus"])

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

    def test_service_row_plans_and_applies(self):
        tuner = tune_mod.Tuner("/")
        with (
            patch.object(tuner, "_service_state", return_value=True),
            patch.object(tuner, "_service_set") as run,
        ):
            row = tuner._service_row("irqbalance", False, True)
            self.assertEqual(row["current"], "on")
            self.assertEqual(row["target"], "off")
            self.assertEqual(row["status"], "planned")
            row = tuner._service_row("irqbalance", False, False)
            run.assert_called_once_with("irqbalance", False)
            self.assertEqual(row["status"], "failed")

    def test_power_supply_row(self):
        self.write("sys/class/power_supply/AC/type", "Mains\n")
        self.write("sys/class/power_supply/AC/online", "1\n")
        df = tune_mod.tune(profile="default", root=self.root, dry_run=True)
        row = self.row(df, "power_supply")
        self.assertEqual(row["current"], "ac")
        self.assertEqual(row["target"], "ac")
        self.assertEqual(row["status"], "unchanged")

    def test_kernel_arguments_are_checked(self):
        self.write("proc/cmdline", "isolcpus=2-3 rcu_nocbs=2,3\n")
        df = tune_mod.tune(profile="latency", root=self.root, dry_run=True)
        self.assertEqual(self.row(df, "isolcpus")["status"], "unchanged")
        self.assertEqual(self.row(df, "rcu_nocbs")["status"], "unchanged")

    def test_missing_kernel_argument_is_unsupported(self):
        self.write("proc/cmdline", "quiet\n")
        df = tune_mod.tune(profile="latency", root=self.root, dry_run=True)
        self.assertEqual(self.row(df, "isolcpus")["status"], "unsupported")

    def test_turbo_msr_fallback(self):
        self.write("proc/cpuinfo", "vendor_id\t: GenuineIntel\n")
        msr = self.root / "dev/cpu/0/msr"
        msr.parent.mkdir(parents=True)
        msr.write_bytes(b"\0" * (tune_mod._MSR_ENABLE + 8))
        df = tune_mod.tune(profile="latency", root=self.root)
        self.assertEqual(self.row(df, "turbo_msr")["status"], "tuned")
        at = tune_mod._MSR_ENABLE
        value = int.from_bytes(msr.read_bytes()[at : at + 8], "little")
        self.assertTrue(value & (1 << tune_mod._MSR_TURBO_BIT))

    def test_irq_affinity_check(self):
        self.write("proc/cmdline", "isolcpus=2-3\n")
        self.write("proc/irq/default_smp_affinity", "3\n")
        df = tune_mod.tune(profile="latency", root=self.root, dry_run=True)
        row = self.row(df, "irq_affinity")
        self.assertEqual(row["current"], "on")
        self.assertEqual(row["status"], "unchanged")

    def test_irq_affinity_checks_per_irq_masks(self):
        self.write("proc/cmdline", "isolcpus=1\n")
        self.write("proc/irq/default_smp_affinity", "1\n")
        self.write("proc/irq/7/smp_affinity", "3\n")
        df = tune_mod.tune(profile="latency", root=self.root, dry_run=True)
        row = self.row(df, "irq_affinity")
        self.assertEqual(row["current"], "off")
        self.assertEqual(row["status"], "planned")

    def test_affinity_mask_formats_wide_masks_in_32_bit_groups(self):
        self.assertEqual(
            tune_mod.Tuner._format_affinity_mask((1 << 40) | 3), "100,00000003"
        )

    def test_columns(self):
        df = tune_mod.tune(profile="default", root=self.root, dry_run=True)
        self.assertEqual(list(df.columns), list(tune_mod._COLUMNS))
        self.assertEqual(len(df), len(tune_mod._TUNABLE))


class TestNewKnobs(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def row(self, df, setting):
        return df[df["setting"] == setting].iloc[0]

    def test_new_profiles_cover_every_setting(self):
        for profile in ("default", "latency", "throughput"):
            resolved = tune_mod.resolve_profile(profile)
            self.assertEqual(tuple(resolved), tune_mod._TUNABLE)
        latency = tune_mod.resolve_profile("latency")
        self.assertEqual(latency["kptr_restrict"], 0)
        self.assertFalse(latency["watchdog"])
        self.assertFalse(latency["sched_autogroup_enabled"])
        self.assertEqual(latency["sched_rt_runtime_us"], -1)
        self.assertEqual(latency["perf_event_paranoid"], -1)

    def test_kptr_restrict_applies(self):
        path = self.write("proc/sys/kernel/kptr_restrict", "1\n")
        df = tune_mod.tune(profile="latency", root=self.root)
        self.assertEqual(path.read_text(), "0")
        self.assertEqual(self.row(df, "kptr_restrict")["status"], "tuned")

    def test_kptr_restrict_validation(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"kptr_restrict": 5})
        self.assertIn("integer in [0, 2]", str(ex.exception))

    def test_paranoid_validation(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"perf_event_paranoid": 5})
        self.assertIn("integer in [-1, 4]", str(ex.exception))

    def test_watchdog_applies(self):
        path = self.write("proc/sys/kernel/watchdog", "1\n")
        df = tune_mod.tune(profile="latency", root=self.root)
        self.assertEqual(path.read_text(), "0")
        self.assertEqual(self.row(df, "watchdog")["status"], "tuned")

    def test_sched_rt_runtime_minus_one(self):
        path = self.write("proc/sys/kernel/sched_rt_runtime_us", "950000\n")
        df = tune_mod.tune(profile="latency", root=self.root)
        self.assertEqual(path.read_text(), "-1")
        self.assertEqual(self.row(df, "sched_rt_runtime_us")["status"], "tuned")

    def test_power_validation(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"power_supply": "sometimes"})
        self.assertIn("must be one of", str(ex.exception))

    def test_power_any_missing_is_unavailable(self):
        df = tune_mod.tune(
            profile="default",
            overrides={"power_supply": "any"},
            root=self.root,
            dry_run=True,
        )
        self.assertEqual(self.row(df, "power_supply")["status"], "unavailable")

    def test_freq_ghz_suffix(self):
        profile = tune_mod.resolve_profile("default", {"cpu_frequency": "2.4GHz"})
        self.assertEqual(profile["cpu_frequency"], 2400000)
        profile = tune_mod.resolve_profile("default", {"cpu_frequency": "800MHz"})
        self.assertEqual(profile["cpu_frequency"], 800000)

    def test_freq_validation_mentions_units(self):
        with self.assertRaises(ValueError) as ex:
            tune_mod.resolve_profile("default", {"cpu_frequency": "fast"})
        self.assertIn("2.4GHz", str(ex.exception))

    def test_comma_affinity_mask(self):
        self.write("proc/cmdline", "isolcpus=1\n")
        self.write("proc/irq/default_smp_affinity", "ff,00000003\n")
        df = tune_mod.tune(profile="latency", root=self.root, dry_run=True)
        row = self.row(df, "irq_affinity")
        self.assertEqual(row["current"], "off")
        self.assertEqual(row["status"], "planned")

    def test_isolated_sysfs_source(self):
        self.write("sys/devices/system/cpu/isolated", "1-3\n")
        self.write("proc/cmdline", "quiet\n")
        self.write("proc/irq/default_smp_affinity", "f\n")
        df = tune_mod.tune(profile="latency", root=self.root, dry_run=True)
        self.assertEqual(self.row(df, "irq_affinity")["current"], "off")

    def test_irq_affinity_tunes_masks(self):
        self.write("proc/cmdline", "isolcpus=1\n")
        default = self.write("proc/irq/default_smp_affinity", "3\n")
        per_irq = self.write("proc/irq/7/smp_affinity", "3\n")
        df = tune_mod.tune(profile="latency", root=self.root)
        self.assertEqual(default.read_text(), "1")
        self.assertEqual(per_irq.read_text(), "1")
        self.assertEqual(self.row(df, "irq_affinity")["status"], "tuned")

    def test_irq_affinity_false_shows_actual_but_unchanged(self):
        self.write("proc/cmdline", "isolcpus=1\n")
        self.write("proc/irq/default_smp_affinity", "1\n")
        df = tune_mod.tune(profile="default", root=self.root, dry_run=True)
        row = self.row(df, "irq_affinity")
        self.assertEqual(row["current"], "on")
        self.assertEqual(row["status"], "unchanged")

    def test_nohz_null_is_off(self):
        self.write("sys/devices/system/cpu/nohz_full", " (null)\n")
        df = tune_mod.tune(profile="default", root=self.root, dry_run=True)
        self.assertEqual(self.row(df, "nohz_full")["status"], "unchanged")

    def test_dry_run_reports_unchanged_when_matching(self):
        self.write("proc/sys/kernel/randomize_va_space", "0\n")
        df = tune_mod.tune(
            profile="latency",
            overrides={"aslr": False},
            root=self.root,
            dry_run=True,
        )
        self.assertEqual(self.row(df, "aslr")["status"], "unchanged")

    def test_smt_forceoff_satisfies_off(self):
        self.write("sys/devices/system/cpu/smt/control", "forceoff\n")
        df = tune_mod.tune(profile="latency", root=self.root, dry_run=True)
        row = self.row(df, "smt")
        self.assertEqual(row["current"], "off")
        self.assertEqual(row["status"], "unchanged")

    def test_smt_forceoff_blocks_on(self):
        self.write("sys/devices/system/cpu/smt/control", "forceoff\n")
        df = tune_mod.tune(profile="throughput", root=self.root, dry_run=True)
        self.assertEqual(self.row(df, "smt")["status"], "unsupported")

    def test_msr_short_read_is_failure_not_none_label(self):
        self.write("proc/cpuinfo", "vendor_id\t: GenuineIntel\n")
        msr = self.root / "dev/cpu/0/msr"
        msr.parent.mkdir(parents=True)
        msr.write_bytes(b"\0" * tune_mod._MSR_ENABLE)
        df = tune_mod.tune(profile="latency", root=self.root)
        row = self.row(df, "turbo_msr")
        self.assertNotIn("None", row["current"])
        self.assertIn(row["status"], ("denied", "failed"))

    def test_bool_decode_variants(self):
        self.assertEqual(tune_mod._decode("ksm", "Y"), "on")
        self.assertEqual(tune_mod._decode("ksm", "N"), "off")
        self.assertEqual(tune_mod._decode("turbo", "0"), "on")
        self.assertEqual(tune_mod._decode("turbo", "1"), "off")


class TestTuneThreadSafety(unittest.TestCase):
    def test_tune_calls_are_serialized(self):
        active = {"n": 0, "peak": 0}
        lock = threading.Lock()

        class FakeTuner:
            def __init__(self, _root):
                pass

            def apply(self, **_kwargs):
                with lock:
                    active["n"] += 1
                    active["peak"] = max(active["peak"], active["n"])
                time.sleep(0.05)
                with lock:
                    active["n"] -= 1
                return tune_mod.pd.DataFrame()

        def worker():
            tune_mod.tune()

        with patch.object(tune_mod, "Tuner", FakeTuner):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertEqual(active["peak"], 1)


class TestTunerInternals(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.tuner = tune_mod.Tuner(self.root)

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_read_int_handles_missing_and_invalid_values(self):
        self.assertIsNone(self.tuner._read(self.root / "missing"))
        self.assertIsNone(self.tuner._read_int(self.root / "missing"))
        path = self.write("value", "not-a-number\n")
        self.assertIsNone(self.tuner._read_int(path))
        self.assertEqual(self.tuner._read(path), "not-a-number")
        self.assertEqual(self.tuner._current("swappiness", path), "not-a-number")

    def test_glob_ignores_oserror(self):
        with patch.object(tune_mod.glob, "glob", side_effect=OSError("nope")):
            self.assertEqual(self.tuner._glob("/proc/sys/*"), [])

    def test_paths_for_cpu_frequency_returns_min_max_pairs(self):
        self.write("sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq", "1")
        self.write("sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", "2")
        self.write("sys/devices/system/cpu/cpu1/cpufreq/scaling_min_freq", "1")
        paths = self.tuner.paths("cpu_frequency")
        self.assertEqual(
            [path.name for path in paths],
            ["scaling_min_freq", "scaling_max_freq"],
        )

    def test_compat_warning_for_nohz_full_with_intel_pstate(self):
        self.write("sys/devices/system/cpu/intel_pstate/no_turbo", "0")
        warnings = self.tuner._compat_warnings({"nohz_full": True})
        self.assertEqual(len(warnings), 1)
        self.assertIn("intel_pstate", warnings[0])
        self.assertIn("1378529", warnings[0])
        self.assertEqual(self.tuner._compat_warnings({"nohz_full": False}), [])

    def test_compat_warning_handles_path_errors(self):
        with patch.object(tune_mod.Path, "exists", side_effect=OSError("nope")):
            self.assertEqual(self.tuner._compat_warnings({"nohz_full": True}), [])

    def test_apply_attaches_compatibility_warnings(self):
        self.write("sys/devices/system/cpu/intel_pstate/no_turbo", "0")
        df = tune_mod.tune(
            profile="latency",
            overrides={"nohz_full": True},
            root=self.root,
            dry_run=True,
        )
        self.assertEqual(len(df.attrs["warnings"]), 1)
        plain = tune_mod.tune(profile="latency", root=self.root, dry_run=True)
        self.assertNotIn("warnings", plain.attrs)

    def test_freq_targets_fall_back_to_the_maximum_for_base(self):
        directory = self.root / "cpufreq"
        directory.mkdir()
        minimum = directory / "scaling_min_freq"
        maximum = directory / "scaling_max_freq"
        minimum.write_text("800000")
        maximum.write_text("3600000")
        (directory / "cpuinfo_max_freq").write_text("3600000")
        low, high = self.tuner._freq_targets("base", minimum, maximum)
        self.assertEqual((low, high), (3600000, 3600000))
        low, high = self.tuner._freq_targets("min", minimum, maximum)
        self.assertEqual((low, high), (None, None))
        self.assertEqual(
            self.tuner._freq_targets(2000000, minimum, maximum), (2000000, 2000000)
        )
        self.assertEqual(self.tuner._freq_current(minimum, maximum), "800MHz..3.6GHz")
        self.assertEqual(
            self.tuner._freq_current(directory / "nope", maximum), "..3.6GHz"
        )

    def test_frequency_row_is_unchanged_when_already_set(self):
        directory = self.root / "sys/devices/system/cpu/cpu0/cpufreq"
        for name, value in (
            ("scaling_min_freq", "3600000"),
            ("scaling_max_freq", "3600000"),
            ("cpuinfo_min_freq", "800000"),
            ("cpuinfo_max_freq", "3600000"),
        ):
            self.write(f"{directory.relative_to(self.root)}/{name}", value)
        row = self.tuner._frequency_row("max", dry_run=False)
        self.assertEqual(row["status"], "unchanged")

    def test_frequency_row_reports_denied_and_failed_writes(self):
        directory = self.root / "sys/devices/system/cpu/cpu0/cpufreq"
        for name, value in (
            ("scaling_min_freq", "800000"),
            ("scaling_max_freq", "3600000"),
            ("cpuinfo_min_freq", "800000"),
            ("cpuinfo_max_freq", "3600000"),
        ):
            self.write(f"{directory.relative_to(self.root)}/{name}", value)
        with patch.object(
            self.tuner, "_freq_write", side_effect=PermissionError(errno.EACCES, "no")
        ):
            self.assertEqual(
                self.tuner._frequency_row("max", False)["status"], "denied"
            )
        with patch.object(
            self.tuner, "_freq_write", side_effect=OSError(errno.EIO, "boom")
        ):
            self.assertEqual(
                self.tuner._frequency_row("max", False)["status"], "failed"
            )

    def test_failed_verification_after_write(self):
        directory = self.root / "sys/devices/system/cpu/cpu0/cpufreq"
        for name, value in (
            ("scaling_min_freq", "800000"),
            ("scaling_max_freq", "3600000"),
            ("cpuinfo_min_freq", "800000"),
            ("cpuinfo_max_freq", "3600000"),
        ):
            self.write(f"{directory.relative_to(self.root)}/{name}", value)
        with patch.object(
            self.tuner,
            "_read_int",
            side_effect=[800000, 3600000, 800000, 3600000, 0, 0, 0, 0],
        ):
            row = self.tuner._frequency_row("max", False)
        self.assertEqual(row["status"], "failed")

    def test_kernel_argument_sources(self):
        nohz = self.write("sys/devices/system/cpu/nohz_full", "0\n")
        self.assertEqual(self.tuner._kernel_arg_current("nohz_full"), "off")
        nohz.write_text("  \n")
        self.assertEqual(self.tuner._kernel_arg_current("nohz_full"), "off")
        nohz.write_text("1-3\n")
        self.assertEqual(self.tuner._kernel_arg_current("nohz_full"), "on")
        self.write("proc/cmdline", "quiet isolcpus=1-2\n")
        self.assertEqual(self.tuner._kernel_arg_current("isolcpus"), "on")
        self.write("proc/cmdline", "quiet isolcpus=\n")
        self.assertEqual(self.tuner._kernel_arg_current("isolcpus"), "off")
        self.write("proc/cmdline", "quiet isolcpus=0\n")
        self.assertEqual(self.tuner._kernel_arg_current("isolcpus"), "off")
        (self.root / "proc/cmdline").unlink()
        self.assertIsNone(self.tuner._kernel_arg_current("isolcpus"))
        row = self.tuner._kernel_arg_row("isolcpus", True)
        self.assertEqual(row["status"], "unavailable")

    def test_isolated_cpu_parsing(self):
        self.write(
            "proc/cmdline",
            "quiet isolcpus=1-3,5 nohz_full=7-8 malformed=9\n",
        )
        self.write("sys/devices/system/cpu/isolated", "2-4\n")
        self.assertEqual(self.tuner._isolated_cpus(), {1, 2, 3, 4, 5, 7, 8})

    def test_isolated_cpu_parsing_handles_bare_and_invalid_ranges(self):
        self.write("proc/cmdline", "1-2,4\n")
        self.assertEqual(self.tuner._isolated_cpus(), {1, 2, 4})
        self.write("proc/cmdline", "isolcpus=3,a-b\n")
        self.assertEqual(self.tuner._isolated_cpus(), {3})
        self.write("proc/cmdline", "isolcpus=0\n")
        self.assertEqual(self.tuner._isolated_cpus(), set())

    def test_isolated_cpu_parsing_without_files(self):
        self.assertEqual(self.tuner._isolated_cpus(), set())

    def test_affinity_mask_round_trip(self):
        self.assertIsNone(self.tuner._parse_affinity_mask(""))
        self.assertIsNone(self.tuner._parse_affinity_mask("zz"))
        self.assertEqual(self.tuner._parse_affinity_mask("3,1"), (0x3 << 32) | 0x1)
        self.assertEqual(self.tuner._format_affinity_mask(0), "0")
        self.assertEqual(
            self.tuner._format_affinity_mask((1 << 40) | 3), "100,00000003"
        )

    def test_irq_affinity_current_without_isolated_cpus(self):
        self.write("proc/irq/default_smp_affinity", "f\n")
        self.assertEqual(self.tuner._irq_affinity_current(), "on")

    def test_irq_affinity_current_without_masks(self):
        self.write("proc/cmdline", "isolcpus=1\n")
        self.write("proc/irq/default_smp_affinity", "not-a-mask\n")
        self.assertIsNone(self.tuner._irq_affinity_current())

    def test_irq_affinity_current_detects_isolated_irqs(self):
        self.write("proc/cmdline", "isolcpus=1\n")
        self.write("proc/irq/default_smp_affinity", "3\n")
        self.assertEqual(self.tuner._irq_affinity_current(), "off")

    def test_irq_affinity_row_unavailable_without_a_default_file(self):
        row = self.tuner._irq_affinity_row("irq_affinity", True)
        self.assertEqual(row["status"], "unavailable")
        self.write("proc/irq/default_smp_affinity", "zz\n")
        row = self.tuner._irq_affinity_row("irq_affinity", True)
        self.assertEqual(row["status"], "unavailable")

    def test_irq_affinity_row_only_reports_when_requested(self):
        self.write("proc/cmdline", "isolcpus=1\n")
        self.write("proc/irq/default_smp_affinity", "1\n")
        row = self.tuner._irq_affinity_row("irq_affinity", False)
        self.assertEqual((row["current"], row["status"]), ("on", "unchanged"))
        self.write("proc/irq/default_smp_affinity", "2\n")
        self.write("proc/cmdline", "isolcpus=2\n")
        row = self.tuner._irq_affinity_row("irq_affinity", True)
        self.assertEqual(row["status"], "unchanged")

    def test_irq_affinity_row_plans_and_applies(self):
        self.write("proc/cmdline", "isolcpus=1\n")
        self.write("proc/irq/default_smp_affinity", "3\n")
        planned = self.tuner._irq_affinity_row("irq_affinity", True, dry_run=True)
        self.assertEqual(planned["status"], "planned")
        applied = self.tuner._irq_affinity_row("irq_affinity", True)
        self.assertEqual(applied["status"], "tuned")
        self.assertEqual((self.root / "proc/irq/default_smp_affinity").read_text(), "1")

    def test_irq_affinity_row_skips_unusable_masks(self):
        self.write("proc/cmdline", "isolcpus=1\n")
        self.write("proc/irq/default_smp_affinity", "2\n")
        row = self.tuner._irq_affinity_row("irq_affinity", True)
        self.assertEqual(row["status"], "failed")
        row = self.tuner._irq_affinity_row("irq_affinity", True)
        self.assertEqual(row["status"], "failed")
        with patch.object(self.tuner, "_read", return_value=None):
            row = self.tuner._irq_affinity_row("irq_affinity", True)
        self.assertEqual((row["current"], row["status"]), ("", "unavailable"))

    def test_irq_affinity_row_reports_write_failures(self):
        self.write("proc/cmdline", "isolcpus=1\n")
        self.write("proc/irq/default_smp_affinity", "3\n")
        with patch.object(
            self.tuner, "_write", side_effect=PermissionError(errno.EACCES, "no")
        ):
            row = self.tuner._irq_affinity_row("irq_affinity", True)
        self.assertEqual(row["status"], "denied")

    def test_irq_affinity_row_rechecks_after_writing(self):
        self.write("proc/cmdline", "isolcpus=1\n")
        self.write("proc/irq/default_smp_affinity", "3\n")
        with patch.object(
            self.tuner, "_irq_affinity_current", side_effect=["off", None]
        ):
            row = self.tuner._irq_affinity_row("irq_affinity", True)
        self.assertEqual(row["status"], "unavailable")

    def test_power_current_reads_the_mains_supply(self):
        self.write("sys/class/power_supply/AC/type", "Mains\n")
        self.write("sys/class/power_supply/AC/online", "0\n")
        self.write("sys/class/power_supply/BAT/type", "Battery\n")
        self.assertEqual(self.tuner._power_current(), "battery")
        self.write("sys/class/power_supply/AC/online", "1\n")
        self.assertEqual(self.tuner._power_current(), "ac")
        self.write("sys/class/power_supply/AC/online", "?\n")
        self.assertIsNone(self.tuner._power_current())

    def test_power_current_without_a_supply_directory(self):
        self.assertIsNone(self.tuner._power_current())
        row = self.tuner._power_row("power_supply", "any")
        self.assertEqual(row["status"], "unavailable")

    def test_power_any_is_unchanged_when_known(self):
        self.write("sys/class/power_supply/AC/type", "Mains\n")
        self.write("sys/class/power_supply/AC/online", "1\n")
        row = self.tuner._power_row("power_supply", "any")
        self.assertEqual((row["current"], row["status"]), ("ac", "unchanged"))

    def test_service_row_with_unknown_state(self):
        tuner = tune_mod.Tuner("/")
        with patch.object(tuner, "_service_state", return_value=None):
            row = tuner._service_row("irqbalance", True, False)
        self.assertEqual((row["current"], row["status"]), ("", "unavailable"))

    def test_service_row_unchanged(self):
        tuner = tune_mod.Tuner("/")
        with (
            patch.object(tuner, "_service_state", return_value=True),
            patch.object(tuner, "_service_set") as run,
        ):
            row = tuner._service_row("irqbalance", True, False)
        self.assertEqual(row["status"], "unchanged")
        run.assert_not_called()

    def test_service_row_reports_permission_errors(self):
        tuner = tune_mod.Tuner("/")
        with (
            patch.object(tuner, "_service_state", return_value=False),
            patch.object(
                tuner,
                "_service_set",
                side_effect=PermissionError(errno.EACCES, "denied"),
            ),
        ):
            row = tuner._service_row("irqbalance", True, False)
        self.assertEqual(row["status"], "denied")

    def test_service_row_outside_root_is_unavailable(self):
        tuner = tune_mod.Tuner(self.root)
        row = tuner._service_row("irqbalance", True, False)
        self.assertEqual(row["status"], "unavailable")


class TestServiceCommands(unittest.TestCase):
    def setUp(self):
        self.tuner = tune_mod.Tuner("/")

    def _which(self, systemctl=True, service=True):
        def which(name):
            if name == "systemctl" and systemctl:
                return "/bin/systemctl"
            if name == "service" and service:
                return "/bin/service"
            return None

        return which

    def test_active_states(self):
        for text, expected in (
            ("active\n", True),
            ("activating\n", True),
            ("reloading\n", True),
            ("inactive\n", False),
            ("deactivating\n", False),
            ("failed\n", False),
            ("dead\n", False),
            ("unknown\n", None),
        ):
            results = [
                SimpleNamespace(returncode=0, stdout="loaded\n", stderr=""),
                SimpleNamespace(returncode=0, stdout=text, stderr=""),
            ]
            with (
                patch.object(tune_mod.shutil, "which", side_effect=self._which()),
                patch.object(tune_mod.subprocess, "run", side_effect=results),
            ):
                self.assertEqual(self.tuner._service_state("svc"), expected, text)

    def test_service_fallback_states(self):
        cases = [
            ("unrecognized service\n", "", 1, None),
            ("", "", 0, True),
            ("", "", 2, False),
        ]
        for stdout, stderr, returncode, expected in cases:
            load = SimpleNamespace(returncode=0, stdout="nothing\n", stderr="")
            status = SimpleNamespace(
                returncode=returncode, stdout=stdout, stderr=stderr
            )
            with (
                patch.object(tune_mod.shutil, "which", side_effect=self._which()),
                patch.object(tune_mod.subprocess, "run", side_effect=[load, status]),
            ):
                self.assertEqual(self.tuner._service_state("svc"), expected, stdout)

    def test_systemctl_errors_fall_back(self):
        svc = SimpleNamespace(returncode=0, stdout="running", stderr="")
        for load in (
            SimpleNamespace(returncode=1, stdout="", stderr="bad"),
            SimpleNamespace(returncode=0, stdout="not-found", stderr=""),
        ):
            with (
                patch.object(tune_mod.shutil, "which", side_effect=self._which()),
                patch.object(tune_mod.subprocess, "run", side_effect=[load, svc]),
            ):
                self.assertTrue(self.tuner._service_state("svc"))
        with (
            patch.object(tune_mod.shutil, "which", side_effect=self._which()),
            patch.object(
                tune_mod.subprocess,
                "run",
                side_effect=OSError(errno.ENOENT, "missing"),
            ),
        ):
            self.assertIsNone(self.tuner._service_state("svc"))

    def test_systemctl_active_errors_fall_back(self):
        load = SimpleNamespace(returncode=0, stdout="loaded", stderr="")
        svc = SimpleNamespace(returncode=0, stdout="", stderr="")
        with (
            patch.object(tune_mod.shutil, "which", side_effect=self._which()),
            patch.object(
                tune_mod.subprocess,
                "run",
                side_effect=[load, OSError(errno.EIO, "bad"), svc],
            ),
        ):
            self.assertTrue(self.tuner._service_state("svc"))
        with (
            patch.object(tune_mod.shutil, "which", side_effect=self._which()),
            patch.object(
                tune_mod.subprocess,
                "run",
                side_effect=[
                    load,
                    SimpleNamespace(returncode=1, stdout="active", stderr=""),
                    svc,
                ],
            ),
        ):
            self.assertTrue(self.tuner._service_state("svc"))
        with (
            patch.object(
                tune_mod.shutil, "which", side_effect=self._which(service=False)
            ),
            patch.object(
                tune_mod.subprocess,
                "run",
                side_effect=[
                    load,
                    SimpleNamespace(returncode=1, stdout="active", stderr=""),
                ],
            ),
        ):
            self.assertIsNone(self.tuner._service_state("svc"))

    def test_service_status_errors(self):
        with (
            patch.object(tune_mod.shutil, "which", side_effect=self._which(False)),
            patch.object(
                tune_mod.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("service", 10),
            ),
        ):
            self.assertIsNone(self.tuner._service_state("svc"))

    def test_no_service_manager(self):
        with patch.object(tune_mod.shutil, "which", return_value=None):
            self.assertIsNone(self.tuner._service_state("svc"))

    def test_service_set_succeeds_with_systemctl(self):
        with (
            patch.object(tune_mod.shutil, "which", side_effect=self._which()),
            patch.object(
                tune_mod.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ) as run,
        ):
            self.tuner._service_set("svc", True)
        self.assertEqual(run.call_args.args[0], ["/bin/systemctl", "start", "svc"])

    def test_service_set_permission_failure(self):
        with (
            patch.object(
                tune_mod.shutil, "which", side_effect=self._which(service=False)
            ),
            patch.object(
                tune_mod.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=1, stdout="", stderr="Access denied"
                ),
            ),
        ):
            with self.assertRaises(PermissionError):
                self.tuner._service_set("svc", False)

    def test_service_set_falls_back_to_service(self):
        oks = [
            SimpleNamespace(returncode=1, stdout="", stderr="failed"),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
        with (
            patch.object(tune_mod.shutil, "which", side_effect=self._which()),
            patch.object(tune_mod.subprocess, "run", side_effect=oks) as run,
        ):
            self.tuner._service_set("svc", True)
        self.assertEqual(run.call_args.args[0], ["/bin/service", "svc", "start"])

    def test_service_set_reports_the_first_failure(self):
        with (
            patch.object(tune_mod.shutil, "which", side_effect=self._which()),
            patch.object(
                tune_mod.subprocess,
                "run",
                side_effect=[
                    SimpleNamespace(returncode=3, stdout="", stderr="unknown"),
                    SimpleNamespace(returncode=4, stdout="", stderr="also bad"),
                ],
            ),
        ):
            with self.assertRaises(OSError) as ctx:
                self.tuner._service_set("svc", True)
        self.assertEqual(ctx.exception.errno, 3)

    def test_service_set_timeout_is_reported(self):
        with (
            patch.object(
                tune_mod.shutil, "which", side_effect=self._which(service=False)
            ),
            patch.object(
                tune_mod.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("systemctl", 30),
            ),
        ):
            with self.assertRaises(OSError):
                self.tuner._service_set("svc", True)

    def test_service_set_without_a_manager(self):
        with patch.object(tune_mod.shutil, "which", return_value=None):
            with self.assertRaises(OSError) as ctx:
                self.tuner._service_set("svc", True)
        self.assertEqual(ctx.exception.errno, errno.ENOENT)


class TestMsrAccess(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.tuner = tune_mod.Tuner(self.root)
        self.msr = self.root / "msr"
        self.msr.write_bytes(b"\0" * (tune_mod._MSR_ENABLE + 8))

    def _set_bit(self, enabled):
        data = bytearray(self.msr.read_bytes())
        at = tune_mod._MSR_ENABLE
        value = int.from_bytes(data[at : at + 8], "little")
        mask = 1 << tune_mod._MSR_TURBO_BIT
        value = value & ~mask if enabled else value | mask
        data[at : at + 8] = value.to_bytes(8, "little")
        self.msr.write_bytes(bytes(data))

    def _read_bit(self):
        data = self.msr.read_bytes()
        at = tune_mod._MSR_ENABLE
        return bool(int.from_bytes(data[at : at + 8], "little") & (1 << 38))

    def test_msr_value_and_write_round_trip(self):
        self.assertTrue(self.tuner._msr_value(self.msr))
        self.tuner._msr_write(self.msr, False)
        self.assertTrue(self._read_bit())
        self.assertFalse(self.tuner._msr_value(self.msr))
        self.tuner._msr_write(self.msr, True)
        self.assertFalse(self._read_bit())
        self.assertTrue(self.tuner._msr_value(self.msr))

    def test_msr_value_short_read(self):
        short = self.root / "short"
        short.write_bytes(b"\0" * 8)
        self.assertIsNone(self.tuner._msr_value(short))

    def test_msr_value_fallback_io(self):
        self._set_bit(False)
        with patch.object(tune_mod.os, "pread", side_effect=AttributeError):
            self.assertFalse(self.tuner._msr_value(self.msr))

    def test_msr_write_short_read_raises(self):
        short = self.root / "short-msr"
        short.write_bytes(b"\0" * 8)
        with self.assertRaises(OSError):
            self.tuner._msr_write(short, True)

    def test_msr_write_fallback_io(self):
        with (
            patch.object(tune_mod.os, "pread", side_effect=AttributeError),
            patch.object(tune_mod.os, "pwrite", side_effect=AttributeError),
        ):
            self.tuner._msr_write(self.msr, False)
        self.assertTrue(self._read_bit())

    def test_turbo_row_unavailable_without_intel(self):
        path = self.root / "proc/cpuinfo"
        path.parent.mkdir(parents=True)
        path.write_text("vendor_id\t: AuthenticAMD\n")
        row = self.tuner._msr_turbo_row("turbo_msr", True, True)
        self.assertEqual(row["status"], "unavailable")

    def test_turbo_row_unavailable_with_intel_pstate(self):
        path = self.root / "proc/cpuinfo"
        path.parent.mkdir(parents=True)
        path.write_text("vendor_id\t: GenuineIntel\n")
        (self.root / "sys/devices/system/cpu/intel_pstate").mkdir(parents=True)
        (self.root / "sys/devices/system/cpu/intel_pstate/no_turbo").write_text("0")
        row = self.tuner._msr_turbo_row("turbo_msr", True, True)
        self.assertEqual(row["status"], "unavailable")

    def test_turbo_row_without_msr_files(self):
        path = self.root / "proc/cpuinfo"
        path.parent.mkdir(parents=True)
        path.write_text("vendor_id\t: GenuineIntel\n")
        row = self.tuner._msr_turbo_row("turbo_msr", True, True)
        self.assertEqual(row["status"], "unavailable")

    def test_turbo_row_short_read_is_a_failure(self):
        path = self.root / "proc/cpuinfo"
        path.parent.mkdir(parents=True)
        path.write_text("vendor_id\t: GenuineIntel\n")
        msr = self.root / "dev/cpu/0/msr"
        msr.parent.mkdir(parents=True)
        msr.write_bytes(b"\0" * 8)
        row = self.tuner._msr_turbo_row("turbo_msr", True, True)
        self.assertNotIn("None", row["current"])
        self.assertIn(row["status"], ("failed", "denied"))

    def test_turbo_row_unchanged_and_planned(self):
        cpuinfo = self.root / "proc/cpuinfo"
        cpuinfo.parent.mkdir(parents=True)
        cpuinfo.write_text("vendor_id\t: GenuineIntel\n")
        msr = self.root / "dev/cpu/0/msr"
        msr.parent.mkdir(parents=True)
        msr.write_bytes(b"\0" * (tune_mod._MSR_ENABLE + 8))
        self._set_bit(True)
        self.msr = msr
        row = self.tuner._msr_turbo_row("turbo_msr", True, False)
        self.assertEqual(row["status"], "unchanged")
        self._set_bit(False)
        msr.write_bytes(self.msr.read_bytes())
        row = self.tuner._msr_turbo_row("turbo_msr", True, True)
        self.assertEqual(row["status"], "planned")
        row = self.tuner._msr_turbo_row("turbo_msr", True, False)
        self.assertEqual(row["status"], "tuned")
        self.assertFalse(self._read_bit())

    def test_turbo_row_reports_write_failures(self):
        cpuinfo = self.root / "proc/cpuinfo"
        cpuinfo.parent.mkdir(parents=True)
        cpuinfo.write_text("vendor_id\t: GenuineIntel\n")
        msr = self.root / "dev/cpu/0/msr"
        msr.parent.mkdir(parents=True)
        msr.write_bytes(b"\0" * (tune_mod._MSR_ENABLE + 8))
        self.msr = msr
        self._set_bit(False)
        with patch.object(
            self.tuner, "_msr_write", side_effect=PermissionError(errno.EACCES, "no")
        ):
            row = self.tuner._msr_turbo_row("turbo_msr", True, False)
        self.assertEqual(row["status"], "denied")


class TestNormalizationHelpers(unittest.TestCase):
    def test_as_bool_variants(self):
        self.assertTrue(tune_mod._as_bool(True))
        self.assertFalse(tune_mod._as_bool(False))
        self.assertTrue(tune_mod._as_bool(1))
        self.assertFalse(tune_mod._as_bool(0))
        self.assertIsNone(tune_mod._as_bool(2))
        self.assertTrue(tune_mod._as_bool("YES"))
        self.assertFalse(tune_mod._as_bool("disabled"))
        self.assertIsNone(tune_mod._as_bool("maybe"))
        self.assertIsNone(tune_mod._as_bool(None))

    def test_as_int_variants(self):
        self.assertIsNone(tune_mod._as_int(True))
        self.assertEqual(tune_mod._as_int("0x10"), 16)
        self.assertEqual(tune_mod._as_int("-1"), -1)
        self.assertIsNone(tune_mod._as_int("nope"))

    def test_encode_all_kinds(self):
        self.assertEqual(tune_mod._encode("aslr", True), "2")
        self.assertEqual(tune_mod._encode("aslr", False), "0")
        self.assertEqual(tune_mod._encode("aslr", 1), "1")
        self.assertEqual(tune_mod._encode("ksm", True), "1")
        self.assertEqual(tune_mod._encode("turbo", True), "0")
        self.assertEqual(tune_mod._encode("smt", False), "off")
        self.assertEqual(tune_mod._encode("turbo_msr", True), "0")
        self.assertEqual(tune_mod._encode("irqbalance", True), "active")
        self.assertEqual(tune_mod._encode("isolcpus", False), "0")
        self.assertEqual(tune_mod._encode("irq_affinity", True), "1")
        self.assertEqual(tune_mod._encode("governor", "performance"), "performance")

    def test_decode_aslr_and_unknown_bools(self):
        self.assertEqual(tune_mod._decode("aslr", "0"), "off")
        self.assertEqual(tune_mod._decode("aslr", "1"), "partial")
        self.assertEqual(tune_mod._decode("aslr", "2"), "on")
        self.assertEqual(tune_mod._decode("aslr", "9"), "9")
        self.assertEqual(tune_mod._decode("ksm", "maybe"), "maybe")
        self.assertEqual(tune_mod._decode("turbo", "maybe"), "maybe")

    def test_decode_smt_and_thp_choices(self):
        self.assertEqual(tune_mod._decode("smt", "forceoff"), "off")
        self.assertEqual(tune_mod._decode("thp", "[always] madvise never"), "always")
        self.assertEqual(tune_mod._decode("thp", "madvise"), "madvise")

    def test_decode_service_states(self):
        for raw in ("active", "activating", "reloading", "enabled", "1"):
            self.assertEqual(tune_mod._decode("irqbalance", raw), "on")
        for raw in ("inactive", "deactivating", "failed", "dead", "0"):
            self.assertEqual(tune_mod._decode("irqbalance", raw), "off")
        self.assertEqual(tune_mod._decode("irqbalance", "unknown"), "unknown")
        self.assertEqual(tune_mod._decode("irq_affinity", "unknown"), "unknown")

    def test_parse_frequency_units(self):
        self.assertIsNone(tune_mod._parse_freq_khz(True))
        self.assertIsNone(tune_mod._parse_freq_khz(0))
        self.assertEqual(tune_mod._parse_freq_khz(2400), 2400)
        self.assertEqual(tune_mod._parse_freq_khz(2.4), 2)
        self.assertEqual(tune_mod._parse_freq_khz("2.4GHz"), 2400000)
        self.assertEqual(tune_mod._parse_freq_khz("800 MHz"), 800000)
        self.assertEqual(tune_mod._parse_freq_khz("100khz"), 100)
        self.assertEqual(tune_mod._parse_freq_khz("1000hz"), 1)
        self.assertIsNone(tune_mod._parse_freq_khz("fast"))

    def test_frequency_text(self):
        self.assertEqual(tune_mod._freq_text(None), "")
        self.assertEqual(tune_mod._freq_text(2_000_000), "2GHz")
        self.assertEqual(tune_mod._freq_text(800_000), "800MHz")
        self.assertEqual(tune_mod._freq_text(500), "500kHz")

    def test_display_variants(self):
        self.assertEqual(tune_mod._display("ksm", True), "on")
        self.assertEqual(tune_mod._display("ksm", False), "off")
        self.assertEqual(tune_mod._display("aslr", 0), "off")
        self.assertEqual(tune_mod._display("aslr", 1), "partial")
        self.assertEqual(tune_mod._display("aslr", 2), "on")
        self.assertEqual(tune_mod._display("aslr", 3), "3")
        self.assertEqual(tune_mod._display("cpu_frequency", 2_000_000), "2GHz")
        self.assertEqual(tune_mod._display("governor", "performance"), "performance")

    def test_normalize_displays_numeric_frequency(self):
        self.assertEqual(tune_mod._normalize("cpu_frequency", "2400000"), 2400000)
        self.assertEqual(tune_mod._normalize("aslr", "yes"), True)
        with self.assertRaises(ValueError):
            tune_mod._normalize("aslr", "partial")

    def test_override_items_shapes(self):
        self.assertEqual(tune_mod._override_items(None), [])
        self.assertEqual(tune_mod._override_items({"aslr": False}), [("aslr", False)])
        self.assertEqual(tune_mod._override_items("aslr=false"), [("aslr", "false")])
        self.assertEqual(tune_mod._override_items([{"aslr": False}]), [("aslr", False)])
        with self.assertRaises(ValueError):
            tune_mod._override_items([3])
        with self.assertRaises(ValueError):
            tune_mod._override_items([["a", "b"]])


if __name__ == "__main__":
    unittest.main()
