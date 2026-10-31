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

import ctypes
import os
import unittest
from unittest.mock import Mock, patch

import perf.core as core


class TestConstants(unittest.TestCase):
    def test_hardware_type(self):
        self.assertEqual(core._PERF_TYPE_HARDWARE, 0)
        self.assertEqual(core._PERF_TYPE_SOFTWARE, 1)
        self.assertEqual(core._PERF_TYPE_HW_CACHE, 3)
        self.assertEqual(core._PERF_TYPE_RAW, 4)

    def test_hw_counters(self):
        self.assertEqual(core._HW_EVENTS["cycles"], 0x0)
        self.assertEqual(core._HW_EVENTS["instructions"], 0x1)

    def test_ioctl_commands(self):
        self.assertEqual(core.PerfCounter.PERF_EVENT_IOC_ENABLE, ord("$") << 8)


class TestPerfEventAttr(unittest.TestCase):
    def test_is_ctypes_structure(self):
        self.assertTrue(issubclass(core.PerfCounter.perf_event_attr, ctypes.Structure))
        fields = dict(core.PerfCounter.perf_event_attr._fields_)
        self.assertIn("config", fields)
        self.assertIn("type", fields)
        self.assertEqual(fields["config"], ctypes.c_uint64)


class TestPerfCounterFd(unittest.TestCase):
    def test_a_failed_meta_mmap_closes_the_fd(self):
        closed = []
        c = core.PerfCounter.__new__(core.PerfCounter)
        c.perf_event_open = lambda attr, pid=0, cpu=-1, group_fd=-1: 4242
        with (
            patch.object(core.mmap, "mmap", side_effect=OSError("no mmap")),
            patch.object(core.os, "close", side_effect=closed.append),
        ):
            with self.assertRaises(OSError):
                c.__init__(0x0, type=0, flags=0)
        self.assertEqual(closed, [4242])


class TestRdpmcIndex(unittest.TestCase):
    def _counter(self):
        c = core.PerfCounter.__new__(core.PerfCounter)
        c.meta = Mock()
        return c

    def test_returns_index_minus_one(self):
        c = self._counter()
        c.meta.index = 5
        c.refresh_meta = lambda: None
        self.assertEqual(c.rdpmc_index, 4)

    def test_raises_when_index_zero(self):
        c = self._counter()
        c.meta.index = 0
        c.refresh_meta = lambda: None
        with self.assertRaises(RuntimeError):
            c.rdpmc_index


class TestEventsMapping(unittest.TestCase):
    def test_cycles_and_instructions(self):
        self.assertEqual(core._HW_EVENTS["cycles"], 0x0)
        self.assertEqual(core._HW_EVENTS["instructions"], 0x1)

    def test_contains_hw_core_events(self):
        for name in (
            "cache-misses",
            "branch-instructions",
            "branch-misses",
            "ref-cycles",
        ):
            self.assertIn(name, core._HW_EVENTS)

    def test_idle_cycle_aliases(self):
        self.assertEqual(core._HW_EVENTS["idle-cycles-frontend"], 0x7)
        self.assertEqual(core._HW_EVENTS["idle-cycles-backend"], 0x8)
        self.assertEqual(core.resolve("idle-cycles-frontend")[:2], (0, 0x7))
        self.assertEqual(core.resolve("idle-cycles-backend")[:2], (0, 0x8))

    def test_hw_cache_events_cover_every_standard_cache_extension(self):
        expected = {
            "L1-dcache-loads",
            "L1-dcache-load-misses",
            "L1-dcache-stores",
            "L1-dcache-store-misses",
            "L1-dcache-prefetches",
            "L1-dcache-prefetch-misses",
            "L1-icache-loads",
            "L1-icache-load-misses",
            "L1-icache-prefetches",
            "L1-icache-prefetch-misses",
            "LLC-loads",
            "LLC-load-misses",
            "LLC-stores",
            "LLC-store-misses",
            "LLC-prefetches",
            "LLC-prefetch-misses",
            "dTLB-loads",
            "dTLB-load-misses",
            "dTLB-stores",
            "dTLB-store-misses",
            "dTLB-prefetches",
            "dTLB-prefetch-misses",
            "iTLB-loads",
            "iTLB-load-misses",
            "iTLB-prefetches",
            "iTLB-prefetch-misses",
            "branch-loads",
            "branch-load-misses",
            "node-loads",
            "node-load-misses",
            "node-stores",
            "node-store-misses",
            "node-prefetches",
            "node-prefetch-misses",
        }
        self.assertEqual(set(core._HW_CACHE_EVENTS), expected)

    def test_hw_cache_encoding(self):
        for name, config in core._HW_CACHE_EVENTS.items():
            typ, resolved, _flags = core.resolve(name)
            self.assertEqual(typ, core._PERF_TYPE_HW_CACHE, name)
            self.assertEqual(resolved, config, name)
        self.assertEqual(core.resolve("L1-dcache-loads")[1], 0x0)
        self.assertEqual(core.resolve("L1-dcache-load-misses")[1], 0x10000)
        self.assertEqual(core.resolve("L1-dcache-stores")[1], 0x100)
        self.assertEqual(core.resolve("L1-dcache-store-misses")[1], 0x10100)
        self.assertEqual(core.resolve("LLC-loads")[1], 0x2)
        self.assertEqual(core.resolve("dTLB-loads")[1], 0x3)
        self.assertEqual(core.resolve("iTLB-load-misses")[1], 0x10004)
        self.assertEqual(core.resolve("branch-loads")[1], 0x5)
        self.assertEqual(core.resolve("node-store-misses")[1], 0x10106)

    def test_hw_cache_events_accept_underscores_and_modifiers(self):
        self.assertEqual(
            core.resolve("L1_dcache_load_misses")[:2],
            core.resolve("L1-dcache-load-misses")[:2],
        )
        flags = core.resolve("L1-dcache-load-misses:u")[2]
        self.assertFalse(flags & (1 << 4))
        self.assertTrue(flags & (1 << 5))
        flags = core.resolve("branch-load-misses:p")[2]
        self.assertEqual((flags >> 15) & 0x3, 1)


class TestResolveEvents(unittest.TestCase):
    def test_hardware_events(self):
        self.assertEqual(core.resolve("cycles"), (0, 0x0, 0x61))
        self.assertEqual(core.resolve("instructions"), (0, 0x1, 0x61))
        self.assertEqual(core.resolve("cache-misses"), (0, 0x3, 0x61))

    def test_underscore_aliases(self):
        self.assertEqual(core.resolve("cache_references")[:2], (0, 0x2))
        self.assertEqual(core.resolve("branch_instructions")[:2], (0, 0x4))
        self.assertEqual(core.resolve("ref_cpu_cycles")[:2], (0, 0x9))

    def test_perf_aliases(self):
        self.assertEqual(core.resolve("cpu-cycles")[:2], (0, 0x0))
        self.assertEqual(core.resolve("branches")[:2], (0, 0x4))
        self.assertEqual(core.resolve("ref-cycles")[:2], (0, 0x9))

    def test_integer_config(self):
        self.assertEqual(core.resolve(0x1)[:2], (0, 0x1))

    def test_raw_config(self):
        self.assertEqual(core.resolve("r0123")[:2], (core._PERF_TYPE_RAW, 0x123))

    def test_raw_hex_uses_raw_type(self):
        self.assertEqual(core._PERF_TYPE_RAW, 4)
        typ, config, _flags = core.resolve("r0123")
        self.assertEqual(typ, 4)
        self.assertEqual(config, 0x123)

    def test_raw_hex_with_modifier(self):
        typ, config, _flags = core.resolve("r1A2B:u")
        self.assertEqual(typ, core._PERF_TYPE_RAW)
        self.assertEqual(config, 0x1A2B)

    def test_raw_uppercase(self):
        typ, config, _flags = core.resolve("rFF")
        self.assertEqual((typ, config), (core._PERF_TYPE_RAW, 0xFF))

    def test_software_events(self):
        self.assertEqual(core.resolve("cpu-clock")[:2], (1, 0x0))
        self.assertEqual(core.resolve("page-faults")[:2], (1, 0x2))

    def test_modifiers(self):
        flags = core.resolve("cycles:u")[2]
        self.assertTrue(flags & (1 << 5))
        self.assertFalse(flags & (1 << 4))
        self.assertTrue(flags & (1 << 6))

        flags = core.resolve("cycles:k")[2]
        self.assertFalse(flags & (1 << 5))
        self.assertTrue(flags & (1 << 4))

        flags = core.resolve("cycles:pp")[2]
        self.assertEqual((flags >> 15) & 0x3, 2)

    def test_unknown_event(self):
        with self.assertRaises(ValueError):
            core.resolve("definitely-not-an-event-xyz")

    def test_unsupported_modifier(self):
        with self.assertRaises(ValueError):
            core.resolve("cycles:z")

    def test_sysfs_named_event_underscore(self):
        fake = [
            (
                "cpu_core",
                4,
                {"frontend_bound": 0x100, "topdown-retiring": 0x400},
                {},
            )
        ]
        with patch("perf.core._sysfs_pmus", return_value=fake):
            self.assertEqual(core.resolve("cpu_core/frontend_bound/")[:2], (4, 0x100))
            self.assertEqual(core.resolve("frontend_bound")[:2], (4, 0x100))
            self.assertEqual(core.resolve("topdown_retiring")[:2], (4, 0x400))


class TestOpenEvent(unittest.TestCase):
    @patch("perf.core.PerfCounter")
    def test_open_event_passes_resolved_type_and_flags(self, mock_cls):
        core.open_event("cycles")
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["type"], core._PERF_TYPE_HARDWARE)
        self.assertEqual(kwargs["flags"], core._DEFAULT_FLAGS)

    @patch("perf.core.PerfCounter")
    def test_open_event_forwards_pid(self, mock_cls):
        core.open_event("cycles", pid=1234)
        _, kwargs = mock_cls.call_args
        self.assertEqual(kwargs["pid"], 1234)

    @patch("perf.core.PerfCounter")
    def test_open_counters_forwards_pid_and_skips_duration(self, mock_cls):
        mock_cls.return_value.rdpmc_index = 7
        counters, indices = core.open_counters(["duration_time", "cycles"], pid=1234)
        self.assertEqual(indices[0], None)
        self.assertEqual(indices[1], 7)
        self.assertEqual(len(counters), 1)
        for _, kwargs in mock_cls.call_args_list:
            self.assertEqual(kwargs["pid"], 1234)


class TestArchDelegation(unittest.TestCase):
    def test_syscall_nr_comes_from_arch(self):
        from perf.arch import arch as get_arch

        self.assertEqual(
            core.PerfCounter.SYS_perf_event_open, get_arch()._PERF_SYSCALL_NR
        )
        self.assertEqual(get_arch()._PERF_SYSCALL_NR, 298)


class TestPinPmuRestore(unittest.TestCase):
    def test_created_thread_entry_is_removed(self):
        import perf.core as core

        config = {}
        with (
            patch.object(core, "cpus_for_type", return_value=[0]),
            patch.object(core.os, "sched_getaffinity", return_value={0}),
            patch.object(core.os, "sched_setaffinity"),
        ):
            restore = core.pin_pmu(config, 4, requested=[0])
            self.assertEqual(config, {"thread": [{"affinity": [0]}]})
            restore()
            self.assertIsNone(config["thread"])

    def test_existing_affinity_is_restored(self):
        import perf.core as core

        config = {"thread": [{"affinity": [1]}]}
        with (
            patch.object(core, "cpus_for_type", return_value=[0, 1]),
            patch.object(core.os, "sched_getaffinity", return_value={0, 1}),
            patch.object(core.os, "sched_setaffinity"),
        ):
            restore = core.pin_pmu(config, 4, requested=[0])
            self.assertEqual(config["thread"][0]["affinity"], [0])
            restore()
            self.assertEqual(config["thread"][0]["affinity"], [1])

    def test_spec_form_affinity_is_restored(self):
        import perf.core as core

        config = {"thread": [[{"affinity": [1], "priority": "normal"}]]}
        with (
            patch.object(core, "cpus_for_type", return_value=[0, 1]),
            patch.object(core.os, "sched_getaffinity", return_value={0, 1}),
            patch.object(core.os, "sched_setaffinity"),
        ):
            restore = core.pin_pmu(config, 4, requested=[0])
            self.assertEqual(config["thread"][0][0]["affinity"], [0])
            self.assertEqual(config["thread"][0][0]["priority"], "normal")
            restore()
            self.assertEqual(config["thread"][0][0]["affinity"], [1])


class TestEventCpus(unittest.TestCase):
    def test_no_pmu_bound_events(self):
        self.assertIsNone(core.event_cpus(None))
        self.assertIsNone(core.event_cpus([]))
        self.assertIsNone(core.event_cpus([["duration_time"]]))

    def test_cpus_come_from_the_chosen_pmu(self):
        with (
            patch.object(core, "choose_type", return_value=8),
            patch.object(core, "cpus_for_type", return_value=[6, 7, 8, 9]),
        ):
            self.assertEqual(
                core.event_cpus([["topdown-retiring", "topdown-bad-spec"]]),
                [6, 7, 8, 9],
            )

    def test_a_flat_event_list_is_one_group(self):
        with (
            patch.object(core, "choose_type", return_value=4) as choose,
            patch.object(core, "cpus_for_type", return_value=[0, 1]),
        ):
            self.assertEqual(core.event_cpus(["cycles"]), [0, 1])
            self.assertEqual(choose.call_args.args[0], ["cycles"])

    def test_groups_intersect(self):
        with (
            patch.object(core, "choose_type", side_effect=[4, 8]),
            patch.object(core, "cpus_for_type", side_effect=[[0, 1, 2], [2, 3]]),
        ):
            self.assertEqual(core.event_cpus([["a"], ["b"]]), [2])

    def test_disjoint_groups_yield_an_empty_set(self):
        with (
            patch.object(core, "choose_type", side_effect=[4, 8]),
            patch.object(core, "pmu_cpus", side_effect=[[0], [1]]),
        ):
            self.assertEqual(core.event_cpus([["a"], ["b"]]), [])

    def test_pmu_cpus_falls_back_to_cpu_core_for_generic_events(self):
        with (
            patch.object(core, "cpus_for_type", return_value=None),
            patch.object(core, "_sysfs_pmus", return_value=[("cpu_core", 4, {}, {})]),
        ):
            self.assertEqual(core.pmu_cpus(0), None)
        with (
            patch.object(core, "cpus_for_type", side_effect=[None, [0, 1]]),
            patch.object(core, "_sysfs_pmus", return_value=[("cpu_core", 4, {}, {})]),
        ):
            self.assertEqual(core.pmu_cpus(0), [0, 1])

    def test_pmu_cpus_keeps_a_declared_list(self):
        with (
            patch.object(core, "cpus_for_type", return_value=[6, 7]),
            patch.object(core, "_sysfs_pmus", return_value=[("cpu_core", 4, {}, {})]),
        ):
            self.assertEqual(core.pmu_cpus(0), [6, 7])

    def test_pmu_cpus_leaves_other_types_alone(self):
        with (
            patch.object(core, "cpus_for_type", return_value=None),
            patch.object(core, "_sysfs_pmus", return_value=[("cpu_core", 4, {}, {})]),
        ):
            self.assertIsNone(core.pmu_cpus(8))

    def test_pmu_cpus_falls_back_to_cpu_core_for_hw_cache_events(self):
        with (
            patch.object(core, "cpus_for_type", side_effect=[None, [0, 1]]),
            patch.object(core, "_sysfs_pmus", return_value=[("cpu_core", 4, {}, {})]),
        ):
            self.assertEqual(core.pmu_cpus(core._PERF_TYPE_HW_CACHE), [0, 1])

    def test_pmus_without_a_cpu_list_are_skipped(self):
        with (
            patch.object(core, "choose_type", return_value=4),
            patch.object(core, "cpus_for_type", return_value=None),
        ):
            self.assertIsNone(core.event_cpus([["a"]]))

    def test_failures_are_swallowed(self):
        with patch.object(core, "choose_type", side_effect=RuntimeError("boom")):
            self.assertIsNone(core.event_cpus([["a"]]))


class TestDefaultAffinity(unittest.TestCase):
    def test_allowed_cpus_keeps_every_reachable_pmu_cpu(self):
        with (
            patch.object(core, "event_cpus", return_value=[6, 7, 8]),
            patch.object(core, "_current_affinity_list", return_value=[7, 8, 9]),
        ):
            self.assertEqual(core.allowed_cpus([["a"]]), [7, 8])

    def test_allowed_cpus_narrows_to_the_numa_node(self):
        with (
            patch.object(core, "event_cpus", return_value=[0, 1, 2, 3]),
            patch.object(core, "_current_affinity_list", return_value=[0, 1, 2, 3]),
            patch.object(core, "_numa_cpus", return_value=[2, 3]),
        ):
            self.assertEqual(core.allowed_cpus([["a"]], numa=1), [2, 3])

    def test_allowed_cpus_uses_the_numa_cpu_without_a_pmu(self):
        with (
            patch.object(core, "event_cpus", return_value=None),
            patch.object(core, "_numa_first_cpu", return_value=4),
        ):
            self.assertEqual(core.allowed_cpus([["a"]], numa=0), [4])

    def test_allowed_cpus_falls_back_to_the_current_affinity(self):
        with (
            patch.object(core, "event_cpus", return_value=None),
            patch.object(core, "_current_affinity_list", return_value=[3, 4]),
        ):
            self.assertEqual(core.allowed_cpus([["a"]]), [3, 4])

    def test_allowed_cpus_defaults_to_cpu_zero(self):
        with (
            patch.object(core, "event_cpus", return_value=None),
            patch.object(core, "_current_affinity_list", return_value=None),
        ):
            self.assertEqual(core.allowed_cpus([["a"]]), [0])

    def test_allowed_cpus_leaves_an_unreachable_pmu_unset(self):
        with (
            patch.object(core, "event_cpus", return_value=[6, 7]),
            patch.object(core, "_current_affinity_list", return_value=[0, 1]),
        ):
            self.assertIsNone(core.allowed_cpus([["a"]]))

    def test_first_pmu_cpu_reachable_from_here(self):
        with (
            patch.object(core, "event_cpus", return_value=[6, 7]),
            patch.object(core, "_current_affinity_list", return_value=[0, 6, 7]),
        ):
            self.assertEqual(core.default_affinity([["a"]]), 6)

    def test_numa_narrows_the_choice(self):
        with (
            patch.object(core, "event_cpus", return_value=[0, 1, 2, 3]),
            patch.object(core, "_current_affinity_list", return_value=[0, 1, 2, 3]),
            patch.object(core, "_numa_cpus", return_value=[2, 3]),
        ):
            self.assertEqual(core.default_affinity([["a"]], numa=1), 2)

    def test_numa_first_cpu_when_no_pmu(self):
        with (
            patch.object(core, "event_cpus", return_value=None),
            patch.object(core, "_numa_first_cpu", return_value=4),
        ):
            self.assertEqual(core.default_affinity([["a"]], numa=0), 4)

    def test_falls_back_to_the_current_cpu(self):
        with (
            patch.object(core, "event_cpus", return_value=None),
            patch.object(core, "_numa_first_cpu", return_value=None),
            patch.object(core, "_current_affinity_list", return_value=[0]),
            patch.object(core, "first_cpu", return_value=0),
        ):
            self.assertEqual(core.default_affinity([["a"]], numa=0), 0)
            self.assertEqual(core.default_affinity([["a"]]), 0)

    def test_an_unreachable_pmu_leaves_the_affinity_unset(self):
        with (
            patch.object(core, "event_cpus", return_value=[6, 7]),
            patch.object(core, "_current_affinity_list", return_value=[0, 1]),
        ):
            self.assertIsNone(core.default_affinity([["a"]]))

    def test_conflicting_pmus_leave_the_affinity_unset(self):
        with patch.object(core, "event_cpus", return_value=[]):
            self.assertIsNone(core.default_affinity([["a"], ["b"]]))
            self.assertIsNone(core.default_affinity([["a"]], numa=0))


class TestPinPmuCpus(unittest.TestCase):
    def test_generic_events_pin_to_the_core_pmu(self):
        config = {}
        with (
            patch.object(core, "pmu_cpus", return_value=[0, 1]) as pmu,
            patch.object(core.os, "sched_getaffinity", return_value={0, 1}),
            patch.object(core.os, "sched_setaffinity"),
        ):
            core.pin_pmu(config, 0)
        pmu.assert_called_once_with(0)
        self.assertEqual(config["thread"][0]["affinity"], [0, 1])

    def test_a_pmu_without_cpus_is_a_no_op(self):
        with patch.object(core, "pmu_cpus", return_value=None):
            self.assertEqual(core.pin_pmu({}, 8)(), None)

    def test_no_type_is_a_no_op(self):
        with patch.object(core, "pmu_cpus") as pmu:
            self.assertEqual(core.pin_pmu({}, None)(), None)
        pmu.assert_not_called()

    def test_an_unreachable_pmu_is_reported(self):
        with (
            patch.object(core, "pmu_cpus", return_value=[6, 7]),
            patch.object(core.os, "sched_getaffinity", return_value={0, 1}),
        ):
            with self.assertRaises(RuntimeError) as ex:
                core.pin_pmu({}, 8, requested=[0])
        self.assertIn("PMU CPUs [6, 7]", str(ex.exception))
        self.assertIn("PMU type 8", str(ex.exception))

    def test_track_target_cpus_uses_the_same_fallback(self):
        with patch.object(core, "pmu_cpus", return_value=[0, 1]) as pmu:
            self.assertEqual(core.track_target_cpus(0), [0, 1])
        pmu.assert_called_once_with(0)
        with patch.object(core, "pmu_cpus", return_value=None):
            self.assertIsNone(core.track_target_cpus(0))
        self.assertIsNone(core.track_target_cpus(None))


class TestDemangle(unittest.TestCase):
    def setUp(self):
        from perf.core import demangle

        demangle.cache_clear()

    def test_mangled_without_subprocess(self):
        from unittest.mock import patch

        from perf.core import demangle

        with patch("subprocess.run", side_effect=AssertionError("must not shell out")):
            self.assertEqual(demangle("_Z3fooi"), "foo(int)")

    def test_plain_unchanged(self):
        from perf.core import demangle

        self.assertEqual(demangle("plain"), "plain")
        self.assertEqual(demangle("hot_begin"), "hot_begin")
        self.assertEqual(demangle(""), "")

    def test_plain_name_newlines_stripped(self):
        from perf.core import demangle

        self.assertNotIn("\n", demangle("foo\nbar"))
        self.assertNotIn("\r", demangle("foo\rbar"))
        self.assertEqual(demangle("foo\nbar"), "foo bar")

    def test_mangled_newline_in_demangled_output_stripped(self):
        from perf.core import demangle

        with patch("cxxfilt.demangle", return_value="foo(int)\nbar"):
            out = demangle("_Z3fooi")
        self.assertNotIn("\n", out)
        self.assertIn("foo(int)", out)


class TestFuncPrototype(unittest.TestCase):
    def test_synthesized_from_data(self):
        from perf.core import _func_prototype

        empty = _func_prototype(None, None)
        self.assertIsNotNone(empty)
        self.assertEqual(len(empty.args), 0)

        two = _func_prototype(None, {"regs": {"rdi": 1, "rsi": 2}})
        self.assertEqual(len(two.args), 2)

        class Recovered:
            pass

        recovered = Recovered()
        self.assertIs(_func_prototype(recovered, {"regs": {"rdi": 1}}), recovered)

    def test_arg_alias_counts_as_first_arg(self):
        from perf.core import _func_prototype

        proto = _func_prototype(None, {"regs": {"arg0": 15}})
        self.assertEqual(len(proto.args), 1)

    def test_physical_reg_counts(self):
        from perf.core import _func_prototype

        self.assertEqual(len(_func_prototype(None, {"regs": {"rdi": 1}}).args), 1)
        self.assertEqual(len(_func_prototype(None, {"regs": {"rsi": 1}}).args), 2)

    def test_uppercase_and_mixed_aliases(self):
        from perf.core import _func_prototype

        self.assertEqual(len(_func_prototype(None, {"regs": {"RDI": 1}}).args), 1)
        self.assertEqual(len(_func_prototype(None, {"regs": {"ARG1": 1}}).args), 2)

    def test_empty_data_gives_zero_args(self):
        from perf.core import _func_prototype

        self.assertEqual(len(_func_prototype(None, {"regs": {}}).args), 0)


class TestParseAddrKey(unittest.TestCase):
    def test_plain_key(self):
        self.assertEqual(core._parse_addr_key("0x1000"), ("0x1000", False))
        self.assertEqual(core._parse_addr_key("  123  "), ("123", False))

    def test_range_key_strips_colon(self):
        self.assertEqual(core._parse_addr_key("0x1000:"), ("0x1000", True))
        self.assertEqual(core._parse_addr_key("  0x40: "), ("0x40", True))

    def test_missing_base_range_key(self):
        self.assertEqual(core._parse_addr_key(":"), ("", True))


class TestIsMemAddrKey(unittest.TestCase):
    def test_recognizes_integer_keys(self):
        for key in ("0x1000", "16", "0b11", "0o17", "0x1000:", " 42 "):
            self.assertTrue(core._is_mem_addr_key(key), key)

    def test_rejects_non_integer_keys(self):
        for key in ("rdi", "0x:", "", ":", "data.x", "7zz"):
            self.assertFalse(core._is_mem_addr_key(key), key)


class TestParseCpuList(unittest.TestCase):
    def test_single_values(self):
        self.assertEqual(core._parse_cpu_list("0,2,1"), [0, 1, 2])

    def test_ranges(self):
        self.assertEqual(core._parse_cpu_list("0-3"), [0, 1, 2, 3])
        self.assertEqual(core._parse_cpu_list("3-1"), [1, 2, 3])

    def test_skips_invalid_parts(self):
        self.assertEqual(core._parse_cpu_list("0,bogus,2"), [0, 2])
        self.assertEqual(core._parse_cpu_list(""), [])

    def test_cpumap_bitmaps(self):
        self.assertEqual(core._parse_cpu_bitmap("0000000f"), [0, 1, 2, 3])
        self.assertEqual(core._parse_cpu_bitmap("00000000,0000000f"), [0, 1, 2, 3])
        self.assertEqual(core._parse_cpu_bitmap("00000001,00000000"), [32])
        self.assertEqual(core._parse_cpu_bitmap("not-a-map"), [])

    def test_returns_fresh_lists(self):
        first = core._parse_cpu_list("0-3")
        first.append(99)
        self.assertEqual(core._parse_cpu_list("0-3"), [0, 1, 2, 3])
        first_map = core._parse_cpu_bitmap("0000000f")
        first_map.append(99)
        self.assertEqual(core._parse_cpu_bitmap("0000000f"), [0, 1, 2, 3])


class TestParseRangeList(unittest.TestCase):
    def test_invalid_part_raises_when_errors_requested(self):
        with self.assertRaises(ValueError):
            core._parse_range_list("0,bogus,2", err="bad cpu list")

    def test_no_error_message_skips_bad_parts(self):
        self.assertEqual(core._parse_range_list("0,bogus,2"), {0, 2})


class TestSplitHelpers(unittest.TestCase):
    def test_split_list_none_empty(self):
        self.assertEqual(core._split_list(None), [])
        self.assertEqual(core._split_list([]), [])
        self.assertEqual(core._split_list(""), [])

    def test_split_list_single_string(self):
        self.assertEqual(core._split_list("a,b,c"), ["a", "b", "c"])
        self.assertEqual(core._split_list("  a ,  b  "), ["a", "b"])

    def test_split_list_int(self):
        self.assertEqual(core._split_list(7), ["7"])
        self.assertEqual(core._split_list(0), ["0"])

    def test_split_list_skips_none(self):
        self.assertEqual(core._split_list([None, "a"]), ["a"])
        self.assertEqual(core._split_list([None, None]), [])

    def test_split_groups_none(self):
        self.assertEqual(
            core._split_groups(None, ["duration_time"]), [["duration_time"]]
        )

    def test_split_groups_string(self):
        self.assertEqual(core._split_groups("a,b", ["x"]), [["a", "b"]])
        self.assertEqual(core._split_groups("", ["x"]), [["x"]])
        self.assertEqual(core._split_groups("  ", ["x"]), [["x"]])

    def test_split_groups_nested_splits_comma(self):
        self.assertEqual(core._split_groups([["a,b", "c"]], ["x"]), [["a", "b", "c"]])
        self.assertEqual(core._split_groups(["a,b", "c"], ["x"]), [["a", "b"], ["c"]])

    def test_split_groups_int(self):
        self.assertEqual(core._split_groups(7, ["x"]), [["7"]])

    def test_split_groups_int_in_list(self):
        self.assertEqual(core._split_groups([7], ["x"]), [["7"]])
        self.assertEqual(core._split_groups(["a", 7], ["x"]), [["a"], ["7"]])

    def test_encode_ignores_unknown_word(self):
        self.assertEqual(core._encode({"x": "1"}, {"x": "foo:0-3"}), 0)
        self.assertEqual(core._encode({"x": "1"}, {"x": "config:0-3"}), 1)

    def test_modifier_flags_host_guest(self):
        self.assertTrue(core._modifier_flags("h") & (1 << 5))
        self.assertTrue(core._modifier_flags("G") & (1 << 19))
        self.assertFalse(core._modifier_flags("G") & (1 << 20))
        self.assertTrue(core._modifier_flags("H") & (1 << 20))
        self.assertFalse(core._modifier_flags("H") & (1 << 19))

    def test_candidates_empty_and_bad(self):
        with self.assertRaises(ValueError):
            core._candidates("")
        with self.assertRaises(ValueError):
            core._candidates("cpu/nope/")

    def test_split_groups_skips_none(self):
        self.assertEqual(core._split_groups([None, "a"], ["x"]), [["a"]])
        self.assertEqual(core._split_groups([None], ["x"]), [["x"]])

    def test_split_list_without_expansion(self):
        self.assertEqual(
            core._split_list("data*,name", expand=False), ["data*", "name"]
        )
        self.assertEqual(core._split_list("*", expand=False), ["*"])
        self.assertEqual(core._split_list(7, expand=False), ["7"])
        self.assertEqual(core._split_list(["a,b"], expand=False), ["a", "b"])

    def test_split_groups_without_expansion(self):
        self.assertEqual(
            core._split_groups("data*,cycles", ["x"], expand=False),
            [["data*", "cycles"]],
        )
        self.assertEqual(
            core._split_groups([["data*", "cycles"]], ["x"], expand=False),
            [["data*", "cycles"]],
        )
        self.assertEqual(core._split_groups(7, ["x"], expand=False), [["7"]])


class TestEventPatterns(unittest.TestCase):
    def test_no_alias_for_plain_topdown(self):
        self.assertEqual(core.expand_event_alias("topdown"), ["topdown"])

    def test_glob_expands_supported_events(self):
        self.assertEqual(
            core.expand_event_alias("topdown-*"),
            list(core._TOPDOWN_EVENTS),
        )
        self.assertEqual(
            core.expand_event_alias("cache-*"),
            ["cache-misses", "cache-references"],
        )

    def test_a_bare_string_is_one_event_not_a_sequence(self):
        self.assertEqual(
            core.expand_event_aliases("topdown-*"), list(core._TOPDOWN_EVENTS)
        )
        self.assertEqual(core.expand_event_aliases("cycles"), ["cycles"])

    def test_regex_expands_supported_events(self):
        self.assertEqual(
            core.expand_event_alias("topdown-(retiring|be-bound)"),
            ["topdown-retiring", "topdown-be-bound"],
        )

    def test_suffix_applies_to_every_expansion(self):
        self.assertEqual(
            core.expand_event_alias("topdown-*/operations"),
            [f"{e}/operations" for e in core._TOPDOWN_EVENTS],
        )

    def test_modifiers_survive_expansion(self):
        self.assertEqual(
            core.expand_event_alias("topdown-*:u"),
            [f"{e}:u" for e in core._TOPDOWN_EVENTS],
        )

    def test_unmatched_pattern_is_left_alone(self):
        self.assertEqual(core.expand_event_alias("nope-*"), ["nope-*"])

    def test_plain_names_are_left_alone(self):
        self.assertEqual(core.expand_event_alias("cycles"), ["cycles"])
        self.assertEqual(core.expand_event_alias("cpu/cycles/"), ["cpu/cycles/"])

    def test_every_expansion_resolves(self):
        for event in core.expand_event_alias("*"):
            with self.subTest(event=event):
                self.assertTrue(core._candidates(event))

    def test_split_list_expands(self):
        self.assertEqual(
            core._split_list("topdown-*,cycles"),
            list(core._TOPDOWN_EVENTS) + ["cycles"],
        )

    def test_split_groups_keeps_one_measurement_group(self):
        self.assertEqual(
            core._split_groups(["topdown-*"], ["duration_time"]),
            [list(core._TOPDOWN_EVENTS)],
        )

    def test_is_event_pattern(self):
        self.assertTrue(core.is_event_pattern("topdown-*"))
        self.assertTrue(core.is_event_pattern("topdown-*/operations"))
        self.assertFalse(core.is_event_pattern("topdown-retiring"))
        self.assertFalse(core.is_event_pattern(7))


class TestThreadConfig(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertEqual(core._thread_cfg({}), {})
        self.assertEqual(core._thread_cfg({"thread": None}), {})

    def test_dict_shorthand(self):
        self.assertEqual(core._thread_cfg({"thread": {"affinity": 1}}), {"affinity": 1})

    def test_spec_form(self):
        cfg = {"thread": [[{"affinity": 1, "priority": "normal"}]]}
        self.assertEqual(core._thread_cfg(cfg)["priority"], "normal")
        self.assertEqual(core._affinity_spec(cfg), 1)
        self.assertEqual(core._priority_spec(cfg), "normal")

    def test_expanded_form(self):
        cfg = {"thread": [{"affinity": [0, 1]}]}
        self.assertEqual(core._affinity_spec(cfg), [0, 1])

    def test_multiple_threads_rejected(self):
        cfg = {"thread": [[{"affinity": 1}, {"affinity": 2}]]}
        self.assertEqual(len(core._thread_group(cfg)), 2)
        self.assertEqual(core._thread_cfg(cfg), {"affinity": 1})

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            core._thread_cfg({"thread": [{"bogus": 1}]})

    def test_store_creates_and_drops(self):
        cfg = {}
        entry, created = core._thread_store(cfg)
        self.assertTrue(created)
        entry["affinity"] = [3]
        self.assertEqual(cfg, {"thread": [{"affinity": [3]}]})
        core._thread_drop(cfg, entry)
        self.assertIsNone(cfg["thread"])

    def test_store_keeps_existing(self):
        cfg = {"thread": [[{"priority": "low"}]]}
        entry, created = core._thread_store(cfg)
        self.assertFalse(created)
        entry["affinity"] = [3]
        self.assertEqual(cfg, {"thread": [[{"priority": "low", "affinity": [3]}]]})


class TestIntHelpers(unittest.TestCase):
    def test_to_int_or(self):
        self.assertEqual(core._to_int_or("0x10", 0), 16)
        self.assertEqual(core._to_int_or(42, 0), 42)
        self.assertEqual(core._to_int_or("bogus", 99), 99)
        self.assertEqual(core._to_int_or(None, 5), 5)

    def test_to_u64(self):
        self.assertEqual(core._to_u64(42), 42)
        self.assertEqual(core._to_u64("0x10"), 16)
        self.assertEqual(core._to_u64(-1), 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(core._to_u64("bogus"), 0)

    def test_is_duration_event(self):
        self.assertTrue(core._is_duration_event("duration_time"))
        self.assertTrue(core._is_duration_event("  duration_time  "))
        self.assertFalse(core._is_duration_event("cycles"))
        self.assertFalse(core._is_duration_event(""))
        self.assertFalse(core._is_duration_event(None))


class TestDurationNormalization(unittest.TestCase):
    def test_with_leader_skips_duration(self):
        self.assertEqual(core.with_leader(["duration_time"]), ["duration_time"])

    def test_choose_type_skips_duration(self):
        self.assertIsNone(core.choose_type(["duration_time"]))
        self.assertEqual(
            core.choose_type(["duration_time", "cycles"]),
            core.choose_type(["cycles"]),
        )

    def test_is_group_duration(self):
        self.assertFalse(core.is_group(["slots", "duration_time"]))
        self.assertFalse(core.is_group(["duration_time", "slots"]))
        self.assertFalse(core.is_group(["duration_time", "duration_time"]))


class TestOneLine(unittest.TestCase):
    def test_newlines_become_spaces(self):
        self.assertEqual(core.one_line("a\nb\rc"), "a b c")
        self.assertEqual(core.one_line(42), "42")


class TestConcreteEval(unittest.TestCase):
    def test_concrete_int_bvv(self):
        import claripy

        self.assertEqual(core.concrete_int(claripy.BVV(7, 64)), 7)
        self.assertIsNone(core.concrete_int(claripy.BVS("x", 64)))
        self.assertIsNone(core.concrete_int(None))
        self.assertIsNone(core.concrete_int(42))

    def test_eval_int_none_and_bvv(self):
        import claripy

        self.assertIsNone(core.eval_int(None, None))
        self.assertEqual(core.eval_int(None, claripy.BVV(9, 64)), 9)

    def test_eval_int_solver(self):
        state = Mock()
        state.solver.eval.return_value = 11
        import claripy

        self.assertEqual(core.eval_int(state, claripy.BVS("y", 64)), 11)
        state.solver.eval.side_effect = RuntimeError("boom")
        self.assertIsNone(core.eval_int(state, claripy.BVS("z", 64)))

    def test_eval_ints_dedupes(self):
        import claripy

        state = Mock()
        state.solver.eval.return_value = 5
        x = claripy.BVS("x", 64)
        self.assertEqual(core.eval_ints(state, [None, x, x]), [None, 5, 5])

    def test_min_int_bvv_and_none(self):
        import claripy

        self.assertIsNone(core.min_int(None, None))
        self.assertEqual(core.min_int(None, claripy.BVV(3, 64)), 3)
        state = Mock()
        state.solver.eval.return_value = 4
        with patch.object(core, "_any_in_range", return_value=None):
            self.assertEqual(core.min_int(state, claripy.BVS("q", 64)), 4)

    def test_min_ints_mixed(self):
        import claripy

        self.assertEqual(core.min_ints(None, [None, claripy.BVV(2, 64)]), [None, 2])

    def test_min_int_solver_error_falls_back(self):
        import claripy

        state = Mock()
        state.solver.eval.return_value = 7
        with patch.object(core, "_any_in_range", side_effect=RuntimeError("boom")):
            self.assertEqual(core.min_int(state, claripy.BVS("q", 64)), 7)


class TestResolveAll(unittest.TestCase):
    def test_resolve_all_lists_candidates(self):
        self.assertIn((0, 0x0, 0x61), core.resolve_all("cycles"))
        with self.assertRaises(ValueError):
            core.resolve_on("cycles", 9999)
        self.assertEqual(core.resolve_on("cycles", 0)[1], 0x0)

    def test_supported_and_catalog(self):
        self.assertIn("cycles", core.supported_events())
        self.assertIn("cycles", core.event_catalog())
        self.assertLess(
            list(core.event_catalog()).index("topdown-retiring"),
            list(core.event_catalog()).index("cycles"),
        )

    def test_hw_cache_events_are_supported_and_catalogued(self):
        for name in core._HW_CACHE_EVENTS:
            self.assertIn(name, core.supported_events())
            self.assertIn(name, core.event_catalog())
        self.assertIn("L1-dcache-load-misses", core.expand_event_alias("L1-dcache-*"))
        self.assertIn("LLC-load-misses", core.expand_event_alias("*-misses"))
        self.assertIn("dTLB-loads", core.expand_event_alias("dTLB-*"))

    def test_perf_attr_bytes_size(self):
        import ctypes as _ct

        blob = core.perf_attr_bytes(0, 0x0, 0x61)
        self.assertEqual(len(blob), _ct.sizeof(core.PerfCounter.perf_event_attr))

    def test_assert_rdpmc_unique(self):
        core.assert_rdpmc_unique(["cycles"], [3])
        with self.assertRaises(RuntimeError):
            core.assert_rdpmc_unique(["a", "b"], [3, 3])

    def test_enable_rdpmc_never_raises(self):
        got = core.enable_rdpmc()
        try:
            self.assertTrue(got is None or hasattr(got, "enable"))
        finally:
            try:
                if got is not None:
                    got.close()
            except Exception:
                pass


class TestPerfCounterLifecycle(unittest.TestCase):
    def _new_counter(self):
        c = core.PerfCounter.__new__(core.PerfCounter)
        c.fd = -1
        c.meta_mmap = None
        return c

    def test_init_maps_meta_and_enters_context(self):
        with (
            patch.object(core.PerfCounter, "perf_event_open", return_value=41),
            patch.object(core.mmap, "mmap", return_value=b"\x00" * core.mmap.PAGESIZE),
        ):
            c = core.PerfCounter(0x0, type=0, flags=0, pid=1, cpu=2, group_fd=3)
        self.assertEqual(c.fd, 41)
        self.assertEqual(c.meta.index, 0)
        self.assertIs(c.__enter__(), c)
        with patch.object(c, "close") as close:
            self.assertFalse(c.__exit__(None, None, None))
        close.assert_called_once_with()
        c.fd = -1

    def test_perf_event_open_returns_fd_and_reports_errno(self):
        c = self._new_counter()
        with patch.object(core.PerfCounter, "libc") as libc:
            libc.syscall.return_value = 9
            attr = core.PerfCounter.perf_event_attr()
            self.assertEqual(c.perf_event_open(attr), 9)
            libc.syscall.return_value = -1
            with (
                patch.object(core.ctypes, "get_errno", return_value=13),
                patch.object(core.os, "strerror", return_value="EACCES"),
            ):
                with self.assertRaises(OSError) as ex:
                    c.perf_event_open(attr)
        self.assertEqual(ex.exception.errno, 13)

    def test_refresh_meta_rereads_the_page(self):
        import struct

        c = self._new_counter()
        page = bytearray(core.mmap.PAGESIZE)
        struct.pack_into("<I", page, 12, 3)
        c.meta_mmap = bytes(page)
        c.refresh_meta()
        self.assertEqual(c.meta.index, 3)

    def test_enable_and_disable_issue_ioctls(self):
        c = self._new_counter()
        c.fd = 5
        with patch.object(core.fcntl, "ioctl") as ioctl:
            c.enable()
            c.disable()
        self.assertEqual(
            [call.args[1] for call in ioctl.call_args_list],
            [
                core.PerfCounter.PERF_EVENT_IOC_RESET,
                core.PerfCounter.PERF_EVENT_IOC_ENABLE,
                core.PerfCounter.PERF_EVENT_IOC_DISABLE,
            ],
        )
        c.fd = -1

    def test_close_swallows_mmap_and_fd_errors(self):
        class BadPage:
            def close(self):
                raise RuntimeError("mmap close failed")

        c = self._new_counter()
        c.meta_mmap = BadPage()
        c.fd = 5
        with patch.object(core.os, "close", side_effect=OSError("close failed")):
            c.close()
        self.assertIsNone(c.meta_mmap)
        self.assertEqual(c.fd, -1)

    def test_close_ignores_an_already_closed_counter(self):
        c = self._new_counter()
        c.fd = "not-an-fd"
        c.close()
        self.assertEqual(c.fd, "not-an-fd")

    def test_del_never_raises(self):
        c = self._new_counter()
        c.close = Mock(side_effect=RuntimeError("boom"))
        c.__del__()


class TestEvalHelpers(unittest.TestCase):
    def test_concrete_int_handles_malformed_bvv(self):
        from types import SimpleNamespace

        self.assertIsNone(core.concrete_int(SimpleNamespace(op="BVV", args=[])))
        self.assertIsNone(core.concrete_int(SimpleNamespace(op="BVV", args=["x"])))

    def test_eval_ints_handles_unhashable_expressions(self):
        state = Mock()
        state.solver.eval.return_value = 42
        self.assertEqual(core.eval_ints(state, [[1]]), [42])

    def test_min_int_cap_of_zero_falls_back_to_solver(self):
        import claripy

        state = Mock()
        state.solver.eval.return_value = 6
        self.assertEqual(core.min_int(state, claripy.BVS("cap", 64), cap=0), 6)

    def test_min_int_binary_searches_for_the_smallest_value(self):
        import claripy

        state = Mock()

        def any_in_range(_state, _expr, _bits, lo, hi):
            return 10 if lo <= 10 <= hi else None

        with patch.object(core, "_any_in_range", side_effect=any_in_range):
            self.assertEqual(core.min_int(state, claripy.BVS("small", 64)), 10)

    def test_min_ints_handles_unhashable_expressions(self):
        state = Mock()
        state.solver.eval.return_value = 9
        with patch.object(core, "_any_in_range", return_value=None):
            self.assertEqual(core.min_ints(state, [[1]]), [9])


class TestDemangleErrors(unittest.TestCase):
    def setUp(self):
        core.demangle.cache_clear()

    def tearDown(self):
        core.demangle.cache_clear()

    def test_demangler_failure_returns_the_original_name(self):
        core.demangle.cache_clear()
        with patch("cxxfilt.demangle", side_effect=RuntimeError("boom")):
            self.assertEqual(core.demangle("_Z3fooi"), "_Z3fooi")

    def test_demangler_empty_result_returns_the_original_name(self):
        core.demangle.cache_clear()
        with patch("cxxfilt.demangle", return_value=""):
            self.assertEqual(core.demangle("_Z3fooi"), "_Z3fooi")


class TestLeaderSelection(unittest.TestCase):
    CORE_PMU = (
        "cpu_core",
        4,
        {"topdown-retiring": 0x400, "slots": 0x500},
        {},
    )
    ATOM_PMU = ("cpu_atom", 8, {"topdown-retiring": 0x400}, {})

    def _patched(self, pmus, types=None, has=None):
        return (
            patch.object(core, "_sysfs_pmus", return_value=pmus),
            patch.object(
                core,
                "_pmu_for_type",
                side_effect=types or (lambda _t: "cpu_core"),
            ),
            patch.object(
                core,
                "_pmu_has",
                side_effect=has or (lambda _dev, _name: True),
            ),
            patch.object(core, "_candidates", return_value=[(4, 0, 0)]),
        )

    def test_topdown_event_gets_a_slots_leader(self):
        with (
            self._patched([self.CORE_PMU])[0],
            self._patched([self.CORE_PMU])[1],
            self._patched([self.CORE_PMU])[2],
            self._patched([self.CORE_PMU])[3],
        ):
            self.assertEqual(
                core.with_leader(["topdown-retiring"]),
                ["cpu_core/slots/", "topdown-retiring"],
            )

    def test_explicit_pmu_event_gets_a_leader(self):
        with (
            self._patched([self.CORE_PMU])[0],
            self._patched([self.CORE_PMU])[1],
            self._patched([self.CORE_PMU])[2],
            self._patched([self.CORE_PMU])[3],
        ):
            self.assertEqual(
                core.with_leader(["cpu_core/topdown-retiring"]),
                ["cpu_core/slots/", "cpu_core/topdown-retiring"],
            )

    def test_conflicting_pmus_leave_events_alone(self):
        with (
            patch.object(
                core, "_sysfs_pmus", return_value=[self.CORE_PMU, self.ATOM_PMU]
            ),
            patch.object(core, "_pmu_for_type", side_effect=lambda t: f"pmu{t}"),
            patch.object(
                core, "_pmu_has", side_effect=lambda dev, _name: dev == "pmu4"
            ),
            patch.object(
                core,
                "_candidates",
                return_value=[(4, 0, 0), (8, 0, 0)],
            ),
        ):
            self.assertEqual(
                core.with_leader(["topdown-retiring"]), ["topdown-retiring"]
            )

    def test_multiple_explicit_devices_are_left_alone(self):
        with patch.object(
            core, "_sysfs_pmus", return_value=[self.CORE_PMU, self.ATOM_PMU]
        ):
            names = ["cpu_core/topdown-retiring", "cpu_atom/topdown-retiring"]
            self.assertEqual(core.with_leader(names), names)

    def test_no_matching_pmu_leaves_events_alone(self):
        with (
            patch.object(core, "_sysfs_pmus", return_value=[self.CORE_PMU]),
            patch.object(core, "_pmu_for_type", return_value=None),
            patch.object(core, "_candidates", return_value=[(4, 0, 0)]),
        ):
            self.assertEqual(
                core.with_leader(["topdown-retiring"]), ["topdown-retiring"]
            )

    def test_non_leader_and_non_string_events_are_ignored(self):
        self.assertEqual(core.with_leader(None), [])
        self.assertEqual(core.with_leader(["cycles", 7]), ["cycles", 7])


class TestChooseType(unittest.TestCase):
    def test_no_common_type_returns_none(self):
        with patch.object(
            core,
            "_candidates",
            side_effect=[[(4, 0, 0)], [(8, 0, 0)]],
        ):
            self.assertIsNone(core.choose_type(["a", "b"]))

    def test_multiple_common_types_prefer_a_leader_free_pmu(self):
        with (
            patch.object(
                core,
                "_candidates",
                side_effect=[[(4, 0, 0), (8, 0, 0)], [(4, 0, 0), (8, 0, 0)]],
            ),
            patch.object(core, "_pmu_for_type", side_effect=lambda t: f"pmu{t}"),
            patch.object(core, "_pmu_has", side_effect=lambda dev, _ld: dev == "pmu4"),
        ):
            self.assertEqual(core.choose_type(["topdown-retiring"]), 8)

    def test_multiple_common_types_without_leaders_sort(self):
        with patch.object(
            core,
            "_candidates",
            side_effect=[[(8, 0, 0), (4, 0, 0)], [(8, 0, 0), (4, 0, 0)]],
        ):
            self.assertEqual(core.choose_type(["a", "b"]), 4)

    def test_empty_candidate_list_returns_none(self):
        with patch.object(core, "_candidates", return_value=[]):
            self.assertIsNone(core.choose_type(["a"]))

    def test_candidate_errors_return_none(self):
        with patch.object(core, "_candidates", side_effect=ValueError("bad")):
            self.assertIsNone(core.choose_type(["a"]))


class TestOpenEventRetries(unittest.TestCase):
    def test_retryable_errnos_fall_through_to_the_next_candidate(self):
        fake = Mock()
        with (
            patch.object(
                core,
                "_candidates",
                return_value=[(4, 0x1, 0), (8, 0x2, 0)],
            ),
            patch.object(
                core,
                "_make_counter",
                side_effect=[OSError(22, "EINVAL"), fake],
            ) as make,
        ):
            self.assertIs(core.open_event("candidate"), fake)
        self.assertEqual(make.call_count, 2)

    def test_non_retryable_errors_are_raised_immediately(self):
        with (
            patch.object(core, "_candidates", return_value=[(4, 0x1, 0)]),
            patch.object(
                core, "_make_counter", side_effect=OSError(13, "EACCES")
            ) as make,
        ):
            with self.assertRaises(OSError):
                core.open_event("candidate")
        self.assertEqual(make.call_count, 1)

    def test_open_event_on_uses_the_requested_type(self):
        fake = Mock()
        with patch.object(core, "_make_counter", return_value=fake) as make:
            self.assertIs(core.open_event_on("cycles", 0), fake)
        self.assertEqual(make.call_args.args[1], 0)

    def test_enable_rdpmc_returns_none_without_events(self):
        with patch.object(core, "open_event", side_effect=OSError("no perf")):
            self.assertIsNone(core.enable_rdpmc())

    def test_enable_rdpmc_keeps_a_counter_whose_enable_fails(self):
        counter = Mock()
        counter.enable.side_effect = OSError("denied")
        with patch.object(core, "open_event", return_value=counter):
            self.assertIs(core.enable_rdpmc(), counter)


class TestPmuCpuDiscovery(unittest.TestCase):
    def setUp(self):
        core.cpus_for_type.cache_clear()

    def tearDown(self):
        core.cpus_for_type.cache_clear()

    def test_reads_the_event_source_cpu_list(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "cpu_core"))
            with open(os.path.join(tmp, "cpu_core", "cpus"), "w") as fh:
                fh.write("0-3\n")
            with (
                patch.object(core, "_SYSFS_EVENT_SOURCES", tmp),
                patch.object(core, "_pmu_for_type", return_value="cpu_core"),
            ):
                self.assertEqual(core.cpus_for_type(4), [0, 1, 2, 3])

    def test_falls_back_to_the_device_cpu_list(self):
        import io
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(core, "_SYSFS_EVENT_SOURCES", tmp),
                patch.object(core, "_pmu_for_type", return_value="cpu_core"),
                patch(
                    "builtins.open",
                    side_effect=[FileNotFoundError(), io.StringIO("0-1\n")],
                ),
            ):
                self.assertEqual(core.cpus_for_type(5), [0, 1])

    def test_empty_or_invalid_lists_are_ignored(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "dev"))
            with open(os.path.join(tmp, "dev", "cpus"), "w") as fh:
                fh.write("bogus\n")
            with (
                patch.object(core, "_SYSFS_EVENT_SOURCES", tmp),
                patch.object(core, "_pmu_for_type", return_value="dev"),
            ):
                self.assertIsNone(core.cpus_for_type(6))

    def test_unknown_type_has_no_cpus(self):
        with patch.object(core, "_pmu_for_type", return_value=None):
            self.assertIsNone(core.cpus_for_type(7))

    def test_pmu_cpus_swallows_lookup_errors(self):
        with patch.object(core, "cpus_for_type", side_effect=RuntimeError("boom")):
            self.assertIsNone(core.pmu_cpus(4))

    def test_pmu_cpus_handles_a_cpu_core_fallback_error(self):
        with (
            patch.object(core, "cpus_for_type", side_effect=[None, RuntimeError("x")]),
            patch.object(core, "_sysfs_pmus", return_value=[("cpu_core", 4, {}, {})]),
        ):
            with patch.object(core, "cpus_for_type", side_effect=RuntimeError("x")):
                self.assertIsNone(core.pmu_cpus(0))


class TestGroupDetection(unittest.TestCase):
    def test_slots_with_topdown_is_a_group(self):
        self.assertTrue(core.is_group(["slots", "topdown-retiring"]))

    def test_empty_first_event_is_not_a_group(self):
        self.assertFalse(core.is_group(["", "topdown-retiring"]))

    def test_single_event_is_not_a_group(self):
        self.assertFalse(core.is_group(["cycles"]))


class TestPinPmuEdgeCases(unittest.TestCase):
    def test_affinity_lookup_failure_still_pins_config(self):
        config = {}
        with (
            patch.object(core, "cpus_for_type", return_value=[0]),
            patch.object(core.os, "sched_getaffinity", side_effect=OSError("no")),
            patch.object(core.os, "sched_setaffinity") as setaffinity,
        ):
            restore = core.pin_pmu(config, 4, requested=[0])
            restore()
        setaffinity.assert_not_called()
        self.assertIsNone(config["thread"])

    def test_invalid_thread_config_does_not_touch_config(self):
        config = {"thread": "not-a-list-of-threads"}
        with (
            patch.object(core, "cpus_for_type", return_value=[0]),
            patch.object(core.os, "sched_getaffinity", return_value={0}),
            patch.object(core.os, "sched_setaffinity"),
        ):
            restore = core.pin_pmu(config, 4, requested=[0])
            restore()
        self.assertEqual(config, {"thread": "not-a-list-of-threads"})

    def test_created_entry_with_extra_keys_keeps_the_entry(self):
        config = {}
        with (
            patch.object(core, "cpus_for_type", return_value=[0]),
            patch.object(core.os, "sched_getaffinity", return_value={0}),
            patch.object(core.os, "sched_setaffinity"),
        ):
            restore = core.pin_pmu(config, 4, requested=[0])
            config["thread"][0]["priority"] = 1
            restore()
        self.assertEqual(config, {"thread": [{"priority": 1}]})

    def test_restore_swallows_affinity_errors(self):
        config = {}
        with (
            patch.object(core, "cpus_for_type", return_value=[0]),
            patch.object(core.os, "sched_getaffinity", return_value={0}),
            patch.object(
                core.os,
                "sched_setaffinity",
                side_effect=[None, OSError("restore failed")],
            ),
        ):
            core.pin_pmu(config, 4, requested=[0])()

    def test_non_dict_config_is_ignored(self):
        with (
            patch.object(core, "cpus_for_type", return_value=[0]),
            patch.object(core.os, "sched_getaffinity", return_value={0}),
            patch.object(core.os, "sched_setaffinity"),
        ):
            core.pin_pmu([], 4, requested=[0])()


class TestOpenCountersBehaviors(unittest.TestCase):
    def _counter(self, index=3):
        c = Mock()
        c.fd = 10
        c.rdpmc_index = index
        return c

    def test_group_leader_is_opened_first_and_followers_join_it(self):
        first, second = self._counter(1), self._counter(2)
        with (
            patch.object(core, "open_event", return_value=first) as open_event,
            patch.object(core, "open_event_on", return_value=second) as open_on,
        ):
            counters, indices = core.open_counters(
                ["cycles", "instructions"], group=True
            )
        self.assertEqual(counters, [first, second])
        self.assertEqual(indices, [1, 2])
        open_event.assert_called_once()
        self.assertEqual(open_on.call_args.kwargs["group_fd"], first.fd)

    def test_force_type_group_leader_uses_open_event_on(self):
        first, second = self._counter(1), self._counter(2)
        with (
            patch.object(core, "open_event_on", side_effect=[first, second]),
            patch.object(core, "open_event") as open_event,
        ):
            counters, _ = core.open_counters(
                ["cycles", "instructions"], force_typ=4, group=True
            )
        self.assertEqual(counters, [first, second])
        open_event.assert_not_called()

    def test_force_type_falls_back_when_the_event_is_not_exported(self):
        first, second = self._counter(), self._counter()
        with (
            patch.object(core, "open_event_on", side_effect=ValueError("nope")),
            patch.object(core, "open_event", return_value=second) as open_event,
            patch.object(core, "_make_counter", return_value=first),
        ):
            counters, _ = core.open_counters(["cycles"], force_typ=4)
        self.assertEqual(counters, [second])
        self.assertEqual(open_event.call_count, 1)

    def test_require_indices_false_skips_rdpmc(self):
        counter = self._counter()
        counter.rdpmc_index = Mock(side_effect=AssertionError("must not read"))
        with patch.object(core, "open_event", return_value=counter):
            _, indices = core.open_counters(["cycles"], require_indices=False)
        self.assertEqual(indices, [None])

    def test_rdpmc_index_is_retried_until_available(self):
        attempts = iter([RuntimeError("not ready"), 4])

        class Counter:
            fd = 10

            @property
            def rdpmc_index(self):
                result = next(attempts)
                if isinstance(result, Exception):
                    raise result
                return result

        counter = Counter()
        with (
            patch.object(core, "open_event", return_value=counter),
            patch("time.sleep"),
        ):
            _, indices = core.open_counters(["cycles"])
        self.assertEqual(indices, [4])

    def test_missing_rdpmc_index_cleans_up_and_raises(self):
        closed = []

        class Counter:
            fd = 10

            @property
            def rdpmc_index(self):
                raise RuntimeError("not ready")

            def disable(self):
                pass

            def close(self):
                closed.append(True)

        with (
            patch.object(core, "open_event", return_value=Counter()),
            patch("time.sleep"),
        ):
            with self.assertRaises(RuntimeError):
                core.open_counters(["cycles"])
        self.assertEqual(closed, [True])

    def test_enable_errors_are_ignored(self):
        counter = self._counter()
        counter.enable.side_effect = OSError("enable failed")
        with patch.object(core, "open_event", return_value=counter):
            _, indices = core.open_counters(["cycles"])
        self.assertEqual(indices, [3])

    def test_duplicate_indices_clean_up_and_raise(self):
        one, two = self._counter(1), self._counter(1)
        with patch.object(core, "open_event", side_effect=[one, two]):
            with self.assertRaises(RuntimeError):
                core.open_counters(["cycles", "instructions"])
        for counter in (one, two):
            counter.close.assert_called_once_with()


class TestRefreshIndices(unittest.TestCase):
    def _counter(self, index):
        c = Mock()
        c.rdpmc_index = index
        return c

    def test_duration_events_keep_none_positions(self):
        one = self._counter(5)
        self.assertEqual(
            core.refresh_indices([one], ["duration_time", "cycles"]), [None, 5]
        )

    def test_missing_counters_fall_back_to_old_indices(self):
        self.assertEqual(core.refresh_indices([], ["cycles"], indices=[7]), [7])
        self.assertEqual(core.refresh_indices([], ["cycles"]), [None])

    def test_rdpmc_retry_then_success(self):
        attempts = iter([RuntimeError("wait"), 8])

        class Counter:
            @property
            def rdpmc_index(self):
                result = next(attempts)
                if isinstance(result, Exception):
                    raise result
                return result

        with patch("time.sleep"):
            self.assertEqual(core.refresh_indices([Counter()], ["cycles"]), [8])

    def test_rdpmc_failure_raises(self):
        class Counter:
            @property
            def rdpmc_index(self):
                raise RuntimeError("wait")

        with (
            patch("time.sleep"),
            self.assertRaises(RuntimeError),
        ):
            core.refresh_indices([Counter()], ["cycles"])

    def test_broken_counter_iteration_falls_back_to_indices(self):
        counters = Mock()
        counters.__len__ = Mock(return_value=1)
        counters.__getitem__ = Mock(side_effect=RuntimeError("boom"))
        self.assertEqual(core.refresh_indices(counters, ["cycles"], indices=[9]), [9])


class TestTrackTargetCpus(unittest.TestCase):
    def test_pmu_errors_are_swallowed(self):
        with patch.object(core, "pmu_cpus", side_effect=RuntimeError("boom")):
            self.assertIsNone(core.track_target_cpus(4))


class TestPerfAttrBytes(unittest.TestCase):
    def test_tuple_config_packs_all_words(self):
        import struct

        blob = core.perf_attr_bytes(4, (0x1, 0x2, 0x3), 0x40)
        typ, size, config, _period, _sample, _read, flags = struct.unpack_from(
            "<IIQQQQQ", blob, 0
        )
        self.assertEqual((typ, config, flags), (4, 1, 0x40))
        self.assertEqual(size, len(blob))
        self.assertEqual(struct.unpack_from("<Q", blob, 56)[0], 2)
        self.assertEqual(struct.unpack_from("<Q", blob, 64)[0], 3)


class TestRemoteOpenSelfCounters(unittest.TestCase):
    def test_group_plan_and_retryable_candidate(self):
        import types

        prof = types.ModuleType("perf.prof")
        prof.remote_mmap = Mock(return_value=0x5000)
        prof.remote_syscall = Mock(side_effect=[-22, 9])
        prof.signed_rax = lambda v: v
        prof.write_mem = Mock()
        import sys

        with (
            patch.dict(sys.modules, {"perf.prof": prof}),
            patch.object(
                core, "_candidates", return_value=[(4, 1, 0x61), (4, 2, 0x61)]
            ),
            patch.object(core, "resolve", return_value=(4, 1, 0x61)),
        ):
            fds = core.remote_open_self_counters(
                123, ["duration_time", "cycles"], None, True, 0x1000
            )
        self.assertEqual(fds, [9])
        self.assertEqual(prof.remote_mmap.call_count, 2)

    def test_exhausted_candidates_report_the_last_error(self):
        import sys
        import types

        prof = types.ModuleType("perf.prof")
        prof.remote_mmap = Mock(return_value=0x5000)
        prof.remote_syscall = Mock(return_value=-22)
        prof.signed_rax = lambda v: v
        prof.write_mem = Mock()
        with (
            patch.dict(sys.modules, {"perf.prof": prof}),
            patch.object(core, "_candidates", return_value=[(4, 1, 0x61)]),
        ):
            with self.assertRaises(OSError):
                core.remote_open_self_counters(123, ["cycles"], None, False, 0x1000)

    def test_non_retryable_candidate_error_stops_the_search(self):
        import sys
        import types

        prof = types.ModuleType("perf.prof")
        prof.remote_mmap = Mock(return_value=0x5000)
        prof.remote_syscall = Mock(return_value=-13)
        prof.signed_rax = lambda v: v
        prof.write_mem = Mock()
        with (
            patch.dict(sys.modules, {"perf.prof": prof}),
            patch.object(core, "_candidates", return_value=[(4, 1, 0x61)]),
        ):
            with self.assertRaises(OSError) as ex:
                core.remote_open_self_counters(123, ["cycles"], None, False, 0x1000)
        self.assertEqual(ex.exception.errno, 13)


class TestOpenTaskCounters(unittest.TestCase):
    def test_proxy_counters_are_closed_and_indices_refreshed(self):
        proxy = Mock()
        with (
            patch.object(core, "open_counters", return_value=([proxy], [1])),
            patch.object(core, "assert_rdpmc_unique"),
            patch.object(core, "remote_open_self_counters", return_value=[5]),
            patch.object(core, "refresh_indices", return_value=[2]),
        ):
            counters, indices = core.open_task_counters(["cycles"], 4, False, 123)
        self.assertEqual((counters, indices), ([], [2]))
        proxy.close.assert_called_once_with()

    def test_refresh_errors_keep_proxy_indices(self):
        proxy = Mock()
        with (
            patch.object(core, "open_counters", return_value=([proxy], [1])),
            patch.object(core, "assert_rdpmc_unique"),
            patch.object(core, "remote_open_self_counters", return_value=[5]),
            patch.object(core, "refresh_indices", side_effect=RuntimeError("boom")),
        ):
            _, indices = core.open_task_counters(["cycles"], 4, False, 123)
        self.assertEqual(indices, [1])


class TestFirstCpu(unittest.TestCase):
    def test_first_cpu_uses_affinity_or_zero(self):
        with patch.object(core, "_current_affinity_list", return_value=[4, 5]):
            self.assertEqual(core.first_cpu(), 4)
        with patch.object(core, "_current_affinity_list", return_value=None):
            self.assertEqual(core.first_cpu(), 0)


class TestEventAliasHelpers(unittest.TestCase):
    def test_non_string_and_empty_aliases(self):
        self.assertEqual(core.expand_event_alias(None), [None])
        self.assertEqual(core.expand_event_alias("  "), [])
        self.assertEqual(core.expand_event_aliases(None), [])

    def test_pattern_without_matches_is_left_alone(self):
        self.assertEqual(core.expand_event_alias("nothing-*"), ["nothing-*"])


class TestLogFilter(unittest.TestCase):
    def test_filters_known_noisy_messages(self):
        import logging

        filt = core._SizelessImportFilter()
        record = logging.LogRecord(
            "cle.loader",
            logging.WARNING,
            __file__,
            1,
            "Symbol imported without a known size: %s",
            ("foo",),
            None,
        )
        self.assertFalse(filt.filter(record))
        record = logging.LogRecord(
            "cle.loader", logging.WARNING, __file__, 1, "ordinary message", (), None
        )
        self.assertTrue(filt.filter(record))

    def test_unformattable_records_are_kept(self):
        record = Mock()
        record.getMessage.side_effect = ValueError("bad format")
        self.assertTrue(core._SizelessImportFilter().filter(record))


class TestBitHelpers(unittest.TestCase):
    def test_bv_bits_rejects_missing_or_out_of_range_sizes(self):
        self.assertIsNone(core._bv_bits(object()))
        self.assertIsNone(core._bv_bits(Mock(size=Mock(return_value=0))))
        self.assertIsNone(core._bv_bits(Mock(size=Mock(return_value=65))))
        self.assertEqual(core._bv_bits(Mock(size=Mock(return_value=64))), 64)

    def test_any_in_range_uses_solver_constraints(self):
        import claripy

        state = Mock()
        state.solver.eval.return_value = 7
        self.assertEqual(core._any_in_range(state, claripy.BVS("x", 64), 64, 1, 10), 7)
        state.solver.eval.side_effect = RuntimeError("unsat")
        self.assertIsNone(core._any_in_range(state, claripy.BVS("y", 64), 64, 1, 10))


class TestReadText(unittest.TestCase):
    def test_reads_stripped_text_and_missing_files(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", delete=False) as fh:
            fh.write("  hello \n")
            path = fh.name
        try:
            self.assertEqual(core._read_text(path), "hello")
        finally:
            os.unlink(path)
        self.assertIsNone(core._read_text("/nonexistent/perf/file"))


class TestSpecParsing(unittest.TestCase):
    def test_split_modifiers(self):
        self.assertEqual(core._split_modifiers("cycles"), ("cycles", ""))
        self.assertEqual(core._split_modifiers(" cycles : up "), ("cycles", " up "))

    def test_modifier_flags_precise_is_capped(self):
        self.assertEqual((core._modifier_flags("pppp") >> 15) & 0x3, 3)

    def test_parse_spec_skips_bare_parts(self):
        self.assertEqual(core._parse_spec(" a = 1 , bare, b=2 "), {"a": "1", "b": "2"})

    def test_parse_bitspec_handles_words_ranges_and_junk(self):
        self.assertEqual(
            core._parse_bitspec("config:0-3,config1:4,foo:1,config2:8-9%x,,bad:x"),
            [
                ("config", 0, 3),
                ("config1", 4, 4),
                ("foo", 1, 1),
                ("config2", 8, 9),
            ],
        )
        self.assertEqual(core._parse_bitspec(":%"), [])

    def test_encode_respects_formats_and_reports_config_words(self):
        self.assertEqual(core._encode({"a": "0x3", "b": "bad"}, {"a": "config:0-1"}), 3)
        self.assertEqual(
            core._encode(
                {"a": "1", "b": "2"},
                {"a": "config:0-1", "b": "config2:0-1"},
            ),
            (1, 0, 2),
        )


class TestSysfsDiscovery(unittest.TestCase):
    def setUp(self):
        core._sysfs_pmus.cache_clear()

    def tearDown(self):
        core._sysfs_pmus.cache_clear()

    def test_sysfs_formats_reads_regular_files(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "event"), "w") as fh:
                fh.write("event:0-7\n")
            open(os.path.join(tmp, "empty"), "w").close()
            self.assertEqual(core._sysfs_formats(tmp), {"event": "event:0-7"})
        self.assertEqual(core._sysfs_formats("/nonexistent/perf/formats"), {})

    def test_sysfs_pmus_skips_incomplete_devices(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            core_dev = os.path.join(tmp, "cpu_core")
            os.makedirs(os.path.join(core_dev, "events"))
            os.makedirs(os.path.join(core_dev, "format"))
            with open(os.path.join(core_dev, "type"), "w") as fh:
                fh.write("4\n")
            with open(os.path.join(core_dev, "events", "good"), "w") as fh:
                fh.write("event=0x10\n")
            open(os.path.join(core_dev, "events", "empty"), "w").close()
            with open(os.path.join(core_dev, "format", "event"), "w") as fh:
                fh.write("config:0-7\n")

            bad_type = os.path.join(tmp, "bad_type")
            os.makedirs(os.path.join(bad_type, "events"))
            with open(os.path.join(bad_type, "type"), "w") as fh:
                fh.write("nope\n")

            no_events = os.path.join(tmp, "no_events")
            os.makedirs(no_events)
            with open(os.path.join(no_events, "type"), "w") as fh:
                fh.write("8\n")

            with patch.object(core, "_SYSFS_EVENT_SOURCES", tmp):
                pmus = core._sysfs_pmus()
        self.assertEqual(len(pmus), 1)
        self.assertEqual(pmus[0][0], "cpu_core")
        self.assertEqual(pmus[0][1], 4)
        self.assertEqual(pmus[0][2], {"good": 0x10})

    def test_named_device_events_and_attribute_encoding(self):
        fake = [("cpu_core", 4, {"front": 0x100}, {"event": "config:0-7"})]
        with patch.object(core, "_sysfs_pmus", return_value=fake):
            self.assertEqual(core._candidates("cpu_core/front/"), [(4, 0x100, 0x61)])
            self.assertEqual(
                core._candidates("cpu_core/event=0x12/"), [(4, 0x12, 0x61)]
            )
            with self.assertRaises(ValueError):
                core._candidates("cpu_core/missing/")

    def test_pmu_helpers_normalize_names(self):
        fake = [("cpu_core", 4, {"topdown-retiring": 1}, {})]
        with patch.object(core, "_sysfs_pmus", return_value=fake):
            self.assertEqual(core._pmu_for_type(4), "cpu_core")
            self.assertIsNone(core._pmu_for_type(9))
            self.assertTrue(core._pmu_has("cpu-core", "topdown_retiring"))
            self.assertFalse(core._pmu_has("cpu_atom", "topdown-retiring"))


class TestEventBaseHelpers(unittest.TestCase):
    def test_event_base(self):
        self.assertEqual(core._event_base(7), "")
        self.assertEqual(core._event_base("cycles:u"), "cycles")
        self.assertEqual(core._event_base("cpu/cycles/"), "cycles")
        self.assertEqual(core._event_base("cpu/event=0x1/"), "")

    def test_explicit_pmu(self):
        self.assertIsNone(core._explicit_pmu(7))
        self.assertIsNone(core._explicit_pmu("cycles"))
        self.assertIsNone(core._explicit_pmu("/cycles/"))
        self.assertEqual(core._explicit_pmu("cpu/cycles/"), "cpu")

    def test_event_pattern_errors_mention_wildcards(self):
        with patch.object(core, "_match_events", return_value=[]):
            with self.assertRaises(ValueError) as ex:
                core._candidates("does-not-exist-*")
        self.assertIn("wildcard", str(ex.exception))


class TestNumberParsing(unittest.TestCase):
    def test_parse_range_list_rejects_bad_ranges(self):
        with self.assertRaises(ValueError):
            core._parse_range_list("1-bogus", err="bad range")

    def test_parse_range_list_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            core._parse_range_list("bogus", err="bad value")

    def test_parse_cpu_bitmap_skips_invalid_words(self):
        self.assertEqual(core._parse_cpu_bitmap("not-a-word,00000001"), [0])

    def test_text_normalizes_missing_values(self):
        self.assertEqual(core._text(float("nan")), "")
        self.assertEqual(core._text(" nan "), "")
        self.assertEqual(core._text("NaT"), "")
        self.assertEqual(core._text(None), "")
        self.assertEqual(core._text(12), "12")

    def test_text_handles_objects_without_str(self):
        class Bad:
            def __str__(self):
                raise RuntimeError("no str")

        self.assertEqual(core._text(Bad()), "")


class TestThreadSpecs(unittest.TestCase):
    def test_check_thread_rejects_bad_shapes_and_keys(self):
        with self.assertRaises(ValueError):
            core._check_thread("nope")
        with self.assertRaises(ValueError):
            core._check_thread({"bogus": 1})

    def test_thread_entries_forms(self):
        self.assertEqual(core._thread_entries(None), [])
        self.assertEqual(core._thread_entries({"affinity": 1}), [{"affinity": 1}])
        self.assertEqual(
            core._thread_entries([{"affinity": 1}, [{"affinity": 2}]]),
            [{"affinity": 1}, {"affinity": 2}],
        )
        with self.assertRaises(ValueError):
            core._thread_entries("bad")
        with self.assertRaises(ValueError):
            core._thread_entries([[]])
        with self.assertRaises(ValueError):
            core._thread_entries(["bad"])

    def test_thread_store_creates_for_missing_or_empty_threads(self):
        config = {"thread": []}
        entry, created = core._thread_store(config)
        self.assertTrue(created)
        self.assertEqual(config["thread"], [entry])

    def test_thread_drop_variants(self):
        entry = {"affinity": 1}
        config = {"thread": entry}
        core._thread_drop(config, entry)
        self.assertIsNone(config["thread"])
        config = {"thread": entry}
        core._thread_drop(config, {"affinity": 2})
        self.assertIs(config["thread"], entry)
        config = {"thread": "not-a-list"}
        core._thread_drop(config, entry)
        self.assertEqual(config["thread"], "not-a-list")

    def test_thread_drop_cleans_nested_groups(self):
        one, two = {"affinity": 1}, {"affinity": 2}
        config = {"thread": [[one], [two]]}
        core._thread_drop(config, one)
        self.assertEqual(config["thread"], [[two]])
        config = {"thread": [[one]]}
        core._thread_drop(config, one)
        self.assertIsNone(config["thread"])

    def test_thread_group_and_specs(self):
        self.assertEqual(core._thread_group({}), [{}])
        self.assertEqual(core._thread_group({"thread": []}), [{}])
        self.assertIsNone(core._numa_spec({}))
        self.assertIsNone(core._affinity_spec({"thread": None}))


class TestAffinityParsing(unittest.TestCase):
    def test_parse_affinity_forms(self):
        self.assertIsNone(core._parse_affinity(None))
        self.assertIsNone(core._parse_affinity("off"))
        self.assertIsNone(core._parse_affinity(""))
        self.assertEqual(core._parse_affinity("0,2-3"), [0, 2, 3])
        self.assertEqual(core._parse_affinity(2), [2])
        self.assertEqual(core._parse_affinity([2, "1"]), [1, 2])
        self.assertIsNone(core._parse_affinity([]))

    def test_parse_affinity_rejects_bad_forms(self):
        with self.assertRaises(ValueError):
            core._parse_affinity(True)
        with self.assertRaises(ValueError):
            core._parse_affinity("cpu-one")
        with self.assertRaises(ValueError):
            core._parse_affinity(["bad"])
        with self.assertRaises(ValueError):
            core._parse_affinity(object())

    def test_affinity_guard_sets_and_restores(self):
        calls = []
        with (
            patch.object(core.os, "sched_getaffinity", return_value={0, 1}),
            patch.object(
                core.os,
                "sched_setaffinity",
                side_effect=lambda _pid, cpus: calls.append(set(cpus)),
            ),
        ):
            with core._affinity_guard({"thread": {"affinity": [1]}}):
                self.assertEqual(calls, [{1}])
        self.assertEqual(calls, [{1}, {0, 1}])

    def test_affinity_guard_is_a_noop_without_a_spec(self):
        with patch.object(core.os, "sched_getaffinity") as getaffinity:
            with core._affinity_guard({}):
                pass
        getaffinity.assert_not_called()

    def test_affinity_guard_survives_os_errors(self):
        with patch.object(core.os, "sched_getaffinity", side_effect=OSError("no")):
            with core._affinity_guard({"thread": {"affinity": [0]}}):
                pass
        with (
            patch.object(core.os, "sched_getaffinity", return_value={0}),
            patch.object(core.os, "sched_setaffinity", side_effect=OSError("no")),
        ):
            with self.assertWarns(UserWarning):
                with core._affinity_guard({"thread": {"affinity": [0]}}):
                    pass
        calls = []
        with (
            patch.object(core.os, "sched_getaffinity", return_value={0}),
            patch.object(
                core.os,
                "sched_setaffinity",
                side_effect=lambda _pid, cpus: (
                    calls.append(set(cpus))
                    if len(calls) == 0
                    else (_ for _ in ()).throw(OSError("restore failed"))
                ),
            ),
        ):
            with core._affinity_guard({"thread": {"affinity": [0]}}):
                pass
        self.assertEqual(len(calls), 1)


class TestPriorityParsing(unittest.TestCase):
    def test_parse_priority_forms(self):
        self.assertIsNone(core._parse_priority(None))
        self.assertIsNone(core._parse_priority("default"))
        self.assertEqual(core._parse_priority(5), 5)
        self.assertEqual(core._parse_priority(3.9), 3)
        self.assertEqual(core._parse_priority("high"), -10)
        self.assertEqual(core._parse_priority("0x10"), 16)

    def test_parse_priority_rejects_bad_forms(self):
        with self.assertRaises(ValueError):
            core._parse_priority(True)
        with self.assertRaises(ValueError):
            core._parse_priority("impossible")
        with self.assertRaises(ValueError):
            core._parse_priority(object())

    def test_priority_guard_sets_and_restores(self):
        calls = []
        with (
            patch.object(core.os, "getpriority", return_value=5),
            patch.object(
                core.os,
                "setpriority",
                side_effect=lambda _which, _who, nice: calls.append(nice),
            ),
        ):
            with core._priority_guard({"thread": {"priority": "high"}}):
                self.assertEqual(calls, [-10])
        self.assertEqual(calls, [-10, 5])

    def test_priority_guard_noop_and_errors(self):
        with patch.object(core.os, "setpriority") as setpriority:
            with core._priority_guard({}):
                pass
        setpriority.assert_not_called()
        with (
            patch.object(core.os, "getpriority", side_effect=OSError("no")),
            patch.object(core.os, "setpriority"),
        ):
            with core._priority_guard({"thread": {"priority": 1}}):
                pass
        with (
            patch.object(core.os, "getpriority", return_value=0),
            patch.object(core.os, "setpriority", side_effect=OSError("no")),
        ):
            with self.assertWarns(UserWarning):
                with core._priority_guard({"thread": {"priority": 1}}):
                    pass
        with (
            patch.object(core.os, "getpriority", return_value=0),
            patch.object(core.os, "setpriority", side_effect=ValueError("bad")),
        ):
            with self.assertRaises(ValueError):
                with core._priority_guard({"thread": {"priority": 1}}):
                    pass

    def test_current_priority_value(self):
        with patch.object(core.os, "getpriority", return_value=3):
            self.assertEqual(core._current_priority_value(), 3)
        with patch.object(core.os, "getpriority", side_effect=OSError("no")):
            self.assertEqual(core._current_priority_value(), 0)


class TestNumaHelpers(unittest.TestCase):
    def test_parse_numa_forms(self):
        self.assertIsNone(core._parse_numa(None))
        self.assertIsNone(core._parse_numa("default"))
        self.assertEqual(core._parse_numa("0x1"), 1)
        self.assertEqual(core._parse_numa(2), 2)

    def test_parse_numa_rejects_bad_forms(self):
        with self.assertRaises(ValueError):
            core._parse_numa("node-one")
        with self.assertRaises(ValueError):
            core._parse_numa(-1)
        with self.assertRaises(ValueError):
            core._parse_numa(True)
        with self.assertRaises(ValueError):
            core._parse_numa(object())

    def test_numa_nodes_and_cpu_lists(self):
        with patch.object(
            core.os, "listdir", return_value=["node0", "node10", "cpu0", "nodeX"]
        ):
            self.assertEqual(core._numa_nodes(), [0, 10])
        with patch.object(core.os, "listdir", side_effect=OSError("no")):
            self.assertEqual(core._numa_nodes(), [])

        import unittest.mock

        with patch("builtins.open", unittest.mock.mock_open(read_data="0-1\n")):
            self.assertEqual(core._numa_cpus(0), [0, 1])
        with patch("builtins.open", side_effect=OSError("no")):
            self.assertEqual(core._numa_cpus(0), [])
        with patch.object(core, "_numa_cpus", return_value=[4]):
            self.assertEqual(core._numa_first_cpu(0), 4)
        with patch.object(core, "_numa_cpus", return_value=[]):
            self.assertIsNone(core._numa_first_cpu(0))

    def test_node_mask_covers_the_node(self):
        mask, maxnode = core._node_mask(0)
        self.assertEqual(maxnode, 1)
        self.assertEqual(bytes(mask), b"\x01" + b"\x00" * 7)
        mask, maxnode = core._node_mask(64)
        self.assertEqual(maxnode, 65)
        self.assertEqual(len(mask), 16)
        self.assertEqual(mask[8], 0x01)

    def test_libc_syscall_failure_is_cached_as_none(self):
        core._libc_syscall.cache_clear()
        with patch.object(core.ctypes, "CDLL", side_effect=OSError("no libc")):
            self.assertIsNone(core._libc_syscall())
        core._libc_syscall.cache_clear()

    def test_set_and_get_mempolicy(self):
        syscall = Mock(return_value=0)
        with (
            patch.object(core, "_libc_syscall", return_value=None),
            self.assertRaises(OSError),
        ):
            core._set_mempolicy(core._MPOL_DEFAULT)
        with (
            patch.object(core, "_libc_syscall", return_value=None),
            self.assertRaises(OSError),
        ):
            core._get_mempolicy()
        with patch.object(core, "_libc_syscall", return_value=syscall):
            self.assertEqual(core._set_mempolicy(core._MPOL_DEFAULT), 0)
            self.assertEqual(core._set_mempolicy(core._MPOL_BIND, 0), 0)
        self.assertEqual(syscall.call_count, 2)
        syscall.reset_mock()
        syscall.return_value = 0
        with patch.object(core, "_libc_syscall", return_value=syscall):
            self.assertEqual(core._get_mempolicy(), 0)
        syscall.return_value = -1
        with (
            patch.object(core, "_libc_syscall", return_value=syscall),
            patch.object(core.ctypes, "get_errno", return_value=22),
        ):
            with self.assertRaises(OSError):
                core._get_mempolicy()

    def test_numa_guard_noop_and_warnings(self):
        with core._numa_guard({}):
            pass
        with (
            patch.object(core, "_parse_numa", return_value=5),
            patch.object(core, "_numa_nodes", return_value=[0]),
        ):
            with self.assertWarns(UserWarning):
                with core._numa_guard({"thread": {"numa": 5}}):
                    pass
        with (
            patch.object(core, "_parse_numa", return_value=0),
            patch.object(core, "_numa_nodes", return_value=[0]),
        ):
            with core._numa_guard({"thread": {"numa": 0}}):
                pass

    def test_numa_guard_bind_and_restore(self):
        calls = []
        with (
            patch.object(core, "_parse_numa", return_value=0),
            patch.object(core, "_numa_nodes", return_value=[0, 1]),
            patch.object(core, "_get_mempolicy", return_value=1),
            patch.object(
                core,
                "_set_mempolicy",
                side_effect=lambda mode, node=None: calls.append((mode, node)) or 0,
            ),
        ):
            with core._numa_guard({"thread": {"numa": 0}}):
                self.assertEqual(calls, [(core._MPOL_BIND, 0)])
        self.assertEqual(calls, [(core._MPOL_BIND, 0), (1, None)])

    def test_numa_guard_handles_bind_failures(self):
        with (
            patch.object(core, "_parse_numa", return_value=0),
            patch.object(core, "_numa_nodes", return_value=[0, 1]),
            patch.object(core, "_get_mempolicy", side_effect=OSError("no")),
            patch.object(core, "_set_mempolicy", return_value=-1),
            patch.object(core.ctypes, "get_errno", return_value=22),
        ):
            with self.assertWarns(UserWarning):
                with core._numa_guard({"thread": {"numa": 0}}):
                    pass
        with (
            patch.object(core, "_parse_numa", return_value=0),
            patch.object(core, "_numa_nodes", return_value=[0, 1]),
            patch.object(core, "_get_mempolicy", return_value=0),
            patch.object(core, "_set_mempolicy", side_effect=[0, -1]),
            patch.object(core.ctypes, "get_errno", return_value=22),
        ):
            with self.assertWarns(UserWarning):
                with core._numa_guard({"thread": {"numa": 0}}):
                    pass


class TestNameHelpers(unittest.TestCase):
    def test_resolves_reports_success(self):
        self.assertTrue(core._resolves("cycles"))
        self.assertFalse(core._resolves("not-an-event"))

    def test_split_pattern_suffix(self):
        self.assertEqual(core._split_pattern_suffix("cycles"), ("cycles", ""))
        self.assertEqual(core._split_pattern_suffix("topdown-*/x"), ("topdown-*", "/x"))
        self.assertEqual(
            core._split_pattern_suffix("cpu/cycles/x"), ("cpu", "/cycles/x")
        )
        self.assertEqual(
            core._split_pattern_suffix("cpu/event=1/x"), ("cpu", "/event=1/x")
        )
        self.assertEqual(core._split_pattern_suffix("/x"), ("/x", ""))

    def test_match_events_glob_and_regex(self):
        core._match_events.cache_clear()
        with (
            patch.object(core, "event_catalog", return_value=("a", "b", "c")),
            patch.object(core, "supported_events", return_value=("a", "b")),
        ):
            self.assertEqual(core._match_events("a*"), ["a"])
            self.assertEqual(core._match_events("(a|b)"), ["a", "b"])
            self.assertEqual(core._match_events("*"), ["a", "b"])

    def test_match_events_invalid_regex_is_treated_as_a_glob(self):
        with patch.object(core, "event_catalog", return_value=("a",)):
            self.assertEqual(core._match_events("(["), [])

    def test_is_bare_pattern(self):
        self.assertTrue(core._is_bare_pattern("*"))
        self.assertFalse(core._is_bare_pattern("a*"))
        self.assertFalse(core._is_bare_pattern(""))


class TestSplitListExtra(unittest.TestCase):
    def test_tuple_set_and_generator_values(self):
        self.assertEqual(core._split_list(("a,b", "c")), ["a", "b", "c"])
        self.assertEqual(core._split_list({"a"}), ["a"])
        self.assertEqual(core._split_list(iter(["a,b"])), ["a", "b"])

    def test_non_iterable_values_use_str(self):
        class One:
            def __str__(self):
                return " one "

        self.assertEqual(core._split_list(One()), ["one"])

    def test_nested_list_values_are_flattened(self):
        self.assertEqual(core._split_list([["a,b"], "c"]), ["a", "b", "c"])

    def test_split_groups_tuple_and_scalar_forms(self):
        self.assertEqual(core._split_groups(("a", "b"), ["x"]), [["a"], ["b"]])
        self.assertEqual(core._split_groups([["a,b"], 7], ["x"]), [["a", "b"], ["7"]])
        self.assertEqual(core._split_groups(("a,b", None), ["x"]), [["a", "b"]])
        self.assertEqual(core._split_groups([], ["x"]), [["x"]])

    def test_is_duration_event_swallows_string_errors(self):
        class Bad:
            def __str__(self):
                raise RuntimeError("no")

        self.assertFalse(core._is_duration_event(Bad()))


if __name__ == "__main__":
    unittest.main()
