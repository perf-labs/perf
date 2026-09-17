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
import unittest
from unittest.mock import Mock, patch

import perf.core as core


class TestConstants(unittest.TestCase):
    def test_hardware_type(self):
        self.assertEqual(core._PERF_TYPE_HARDWARE, 0)
        self.assertEqual(core._PERF_TYPE_SOFTWARE, 1)

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
        self.assertEqual(core.resolve("r0123")[:2], (0, 0x123))

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
        self.assertEqual((flags >> 18) & 0x3, 2)

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


class TestArchDelegation(unittest.TestCase):
    def test_syscall_nr_comes_from_arch(self):
        from perf.arch import arch as get_arch

        self.assertEqual(
            core.PerfCounter.SYS_perf_event_open, get_arch().PERF_SYSCALL_NR
        )
        self.assertEqual(get_arch().PERF_SYSCALL_NR, 298)


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
            self.assertEqual(config, {"thread": {"affinity": [0]}})
            restore()
            self.assertIsNone(config["thread"])

    def test_existing_affinity_is_restored(self):
        import perf.core as core

        config = {"thread": {"affinity": [1]}}
        with (
            patch.object(core, "cpus_for_type", return_value=[0, 1]),
            patch.object(core.os, "sched_getaffinity", return_value={0, 1}),
            patch.object(core.os, "sched_setaffinity"),
        ):
            restore = core.pin_pmu(config, 4, requested=[0])
            self.assertEqual(config["thread"]["affinity"], [0])
            restore()
            self.assertEqual(config["thread"]["affinity"], [1])


class TestDemangle(unittest.TestCase):
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

    def test_split_groups_skips_none(self):
        self.assertEqual(core._split_groups([None, "a"], ["x"]), [["a"]])
        self.assertEqual(core._split_groups([None], ["x"]), [["x"]])


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


if __name__ == "__main__":
    unittest.main()
