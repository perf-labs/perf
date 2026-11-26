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


import copy
import inspect
import json
import math
import os
import random
import shutil
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import perf.bench as bench
import perf.cfg as cfg_mod
import perf.core as core
import perf.mca as mca_mod
from perf.bench import _contains_call, _plan_iterations, benchmark
from perf.cfg import _DEFAULT_BENCH, parse_code, region


def _forget_current_measure_thread():
    core._measure_threads_forget([SimpleNamespace(ident=threading.get_ident())])


ASM_SOURCE = """
# a line comment
\t.text
\t.globl foo
foo:            /* the entry */
\tmov eax, 42
\tmov ebx, 1
\tret
\t.globl bar
bar:
\txor eax, eax
\tret
\t.long 1, 2, 3
.Llocal:
\tnop
loop:
\tadd eax, 1
\tcmp eax, 10
\tjne loop
\tret
"""


def _ticks_to_ns(df, ticks):
    freq = df.attrs["info"]["cpu"]["freq"]
    return [t * 1e9 / freq for t in ticks]


class TestModes(unittest.TestCase):
    @patch("perf.bench._bench_one")
    def test_mode_defaults_to_both(self, mock_one):
        def fake(**kw):
            return pd.DataFrame([{"mode": kw["mode"]}])

        mock_one.side_effect = fake
        result = benchmark(
            target={"asm": "nop"}, name="t", config={"branch": ["predictable"]}
        )
        self.assertEqual(sorted(result["mode"].unique()), ["latency", "throughput"])

    @patch("perf.bench._bench_one")
    def test_modes_run_with_each_config_combo(self, mock_one):
        def fake(**kw):
            return pd.DataFrame(
                [{"mode": kw["mode"], "align": kw["config"]["code"]["align"]}]
            )

        mock_one.side_effect = fake
        result = benchmark(
            target={"asm": "nop"},
            name="t",
            mode=["latency", "throughput"],
            config={
                "code": [{"align": 1}, {"align": 32}],
                "icache": "hot",
                "itlb": "hot",
            },
        )
        self.assertEqual(sorted(result["mode"].unique()), ["latency", "throughput"])
        self.assertEqual(sorted(result["align"].unique()), [1, 32])
        self.assertEqual(len(mock_one.call_args_list), 4)

    @patch("perf.bench._bench_one")
    def test_mode_list_runs_with_each_config_combo(self, mock_one):
        def fake(**kw):
            return pd.DataFrame([{"mode": kw["mode"]}])

        mock_one.side_effect = fake
        benchmark(
            target={"asm": "nop"},
            name="t",
            mode=["throughput", "latency"],
            config={"branch": ["predictable", "unpredictable"]},
        )
        modes = [c.kwargs["mode"] for c in mock_one.call_args_list]
        self.assertEqual(modes[0], "throughput")
        self.assertEqual(modes[-1], "latency")

    @patch("perf.bench._bench_one")
    def test_auto_backend_steers_non_resident_combos_to_loop(self, mock_one):
        def fake(**kw):
            return pd.DataFrame(
                [
                    {
                        "mode": kw["mode"],
                        "dcache": kw["config"]["dcache"],
                        "backend": kw["backend"],
                    }
                ]
            )

        mock_one.side_effect = fake
        config = {
            "dcache": ["hot", "cold"],
            "icache": "hot",
            "dtlb": "hot",
            "itlb": "hot",
            "branch": "predictable",
            "code": {"align": 16},
        }
        benchmark(
            target={"asm": "mov rax, [rdi]"},
            mode=["latency"],
            backend={"latency": "unroll"},
            auto_backend={"dcache", "dtlb", "icache", "itlb"},
            config=config,
        )
        got = {
            c.kwargs["config"]["dcache"]: c.kwargs["backend"]
            for c in mock_one.call_args_list
        }
        self.assertEqual(got, {"hot": "unroll", "cold": "loop"})

        mock_one.reset_mock()
        benchmark(
            target={"asm": "mov rax, [rdi]"},
            mode=["latency"],
            backend={"latency": "unroll"},
            config=config,
        )
        self.assertEqual(
            {c.kwargs["backend"] for c in mock_one.call_args_list}, {"unroll"}
        )

    @patch("perf.bench._bench_one")
    def test_auto_backend_ignores_default_instruction_tiers(self, mock_one):
        def fake(**kw):
            return pd.DataFrame([{"mode": kw["mode"], "backend": kw["backend"]}])

        mock_one.side_effect = fake
        benchmark(
            target={"asm": "imul eax, 42"},
            mode=["latency"],
            backend={"latency": "unroll"},
            auto_backend={"dcache", "dtlb"},
            config={
                "dcache": "hot",
                "dtlb": "hot",
                "icache": ["hot", "cold"],
                "itlb": ["hot", "cold"],
                "branch": "predictable",
                "code": {"align": 16},
            },
        )
        self.assertEqual(
            {c.kwargs["backend"] for c in mock_one.call_args_list}, {"unroll"}
        )

    def test_modes_must_be_a_list(self):
        from perf.cfg import _modes

        with self.assertRaises(TypeError):
            _modes("latency")
        with self.assertRaises(TypeError):
            _modes(None)
        self.assertEqual(_modes(["latency", "latency"]), ["latency"])
        with self.assertRaises(ValueError):
            _modes([])
        with self.assertRaises(ValueError):
            _modes(["bogus"])


class TestCodeSpec(unittest.TestCase):
    _FAST = {"branch": "predictable"}

    def test_parse_code_text(self):
        self.assertEqual(parse_code(None), (None, None, None))
        self.assertEqual(parse_code("  "), (None, None, None))
        self.assertEqual(parse_code("a.out:foo"), ("a.out", "foo", None))
        self.assertEqual(parse_code("a.out:"), ("a.out", None, None))
        self.assertEqual(parse_code("a.s:label"), ("a.s", "label", None))
        self.assertEqual(
            parse_code("a.out:Counter::member(long)"),
            ("a.out", "Counter::member(long)", None),
        )
        self.assertEqual(
            parse_code("a.out:begin..end"), ("a.out", ("begin", "end"), None)
        )
        self.assertEqual(parse_code("mov eax, 42"), (None, None, "mov eax, 42"))
        self.assertEqual(parse_code("L1: nop"), (None, None, "L1: nop"))
        self.assertEqual(parse_code(".L1: nop"), (None, None, ".L1: nop"))
        self.assertEqual(parse_code("loop.1: dec eax"), (None, None, "loop.1: dec eax"))
        self.assertEqual(parse_code("nop # note: x"), (None, None, "nop # note: x"))
        self.assertEqual(parse_code("a.out: func"), ("a.out", "func", None))

    def test_parse_code_pair(self):
        self.assertEqual(parse_code(("a.out", "foo")), ("a.out", "foo", None))
        self.assertEqual(parse_code(["a.out", "foo"]), ("a.out", "foo", None))
        self.assertEqual(
            parse_code(["a.out", ("begin", "end")]), ("a.out", ("begin", "end"), None)
        )
        self.assertEqual(parse_code(["a.out"]), ("a.out", None, None))
        self.assertEqual(parse_code([]), (None, None, None))

    def test_parse_code_bad_region(self):
        for bad in ("a.out:a..", "a.out:..b", "a.out:.."):
            with self.subTest(code=bad), self.assertRaises(ValueError):
                parse_code(bad)

    def test_region(self):
        self.assertIsNone(region(None))
        self.assertIsNone(region("  "))
        self.assertEqual(region(" foo "), "foo")
        self.assertEqual(region(" 0x1000 .. 0x2000 "), ("0x1000", "0x2000"))
        self.assertEqual(region(("a", " b ")), ("a", "b"))

    def _one(self, mock_one):
        mock_one.side_effect = lambda **kw: pd.DataFrame([{"mode": kw["mode"]}])

    @patch("perf.bench._bench_one")
    def test_code_kwarg_pair(self, mock_one):
        self._one(mock_one)
        benchmark(target=("a.out", "foo"), mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertEqual((kwargs["file"], kwargs["target"]), ("a.out", "foo"))
        self.assertIsNone(kwargs["code"])

    @patch("perf.bench._bench_one")
    def test_code_kwarg_text(self, mock_one):
        self._one(mock_one)
        benchmark(target="a.out:begin..end", mode=["latency"], config=self._FAST)
        self.assertEqual(mock_one.call_args.kwargs["target"], ("begin", "end"))

    @patch("perf.bench._bench_one")
    def test_code_positional(self, mock_one):
        self._one(mock_one)
        benchmark(target=["a.out", "foo"], mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertEqual((kwargs["file"], kwargs["target"]), ("a.out", "foo"))

    @patch("perf.bench._bench_one")
    def test_code_positional_asm(self, mock_one):
        self._one(mock_one)
        benchmark(target={"asm": "mov eax, 42"}, mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertIsNone(kwargs["file"])
        self.assertEqual(kwargs["code"], "mov eax, 42")

    @patch("perf.bench._bench_one")
    def test_target_and_code_forms(self, mock_one):
        self._one(mock_one)
        benchmark(target=("a.out", "main"), mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertEqual((kwargs["file"], kwargs["target"]), ("a.out", "main"))
        self.assertIsNone(kwargs["code"])
        benchmark(target={"asm": "mov eax, 42"}, mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertIsNone(kwargs["file"])
        self.assertEqual(kwargs["code"], "mov eax, 42")

    def test_target_conflicts(self):
        for target in (
            {"file": "a.out", "symbol": "foo", "region": ["a", "b"]},
            {"file": "a.out", "symbol": "foo", "asm": "nop"},
            {"file": "a.out", "asm": "nop"},
            {"file": "a.out", "unknown": "foo"},
            {"symbol": "foo"},
            {"region": ["a", "b"]},
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                benchmark(target=target, mode=["latency"], config=self._FAST)

    def test_needs_target(self):
        with self.assertRaises(TypeError):
            benchmark(mode=["latency"], config=self._FAST)


class TestBenchCodePath(unittest.TestCase):
    @patch("perf.bench._bench")
    def test_code_path_builds_dataframe(self, mock_bench):
        mock_bench.return_value = pd.Series([10, 20, 30])

        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="mytarget",
            backend="unroll",
        )

        self.assertEqual(mock_bench.call_count, 2)
        self.assertEqual(
            list(result.index.names),
            ["file", "name", "mode"],
        )
        self.assertEqual(
            result["duration_time"].tolist(),
            _ticks_to_ns(result, [2.0, 4.0, 6.0]),
        )
        self.assertNotIn("iteration", result.columns)
        self.assertNotIn("samples", result.columns)

    @patch("perf.bench._bench", return_value=pd.Series([100]))
    def test_default_config_applied_when_none(self, mock_bench):
        result = benchmark(
            target={"asm": "nop"}, name="t", mode=["latency"], config=None
        )
        cfg = mock_bench.call_args.kwargs["config"]

        for key in ("samples",):
            self.assertEqual(cfg[key], _DEFAULT_BENCH[key])
        self.assertEqual(
            cfg["backend"],
            {"loop": _DEFAULT_BENCH["backend"]["loop"]},
        )
        self.assertEqual(
            cfg["iterations"],
            _DEFAULT_BENCH["iterations"]["min"],
        )

        self.assertIsInstance(cfg["seed"], int)
        self.assertEqual(cfg["thread"][0]["affinity"], 0)
        self.assertEqual(cfg["thread"][0]["priority"], "normal")
        self.assertEqual(cfg["func"]["order"], "as-is")
        self.assertIsInstance(cfg["iterations"], int)

        spec = result.attrs["config"]
        self.assertEqual(spec["branch"], ["predictable", "unpredictable"])
        self.assertEqual(spec["dcache"], ["hot", "warm", "cool", "cold"])
        self.assertEqual(spec["icache"], ["hot", "cold"])
        self.assertEqual(spec["dtlb"], ["hot", "cold"])
        self.assertEqual(spec["itlb"], ["hot", "cold"])
        self.assertEqual(
            spec["external"],
            {"lib": False, "stdout": False, "stderr": False},
        )
        self.assertEqual(spec["iterations"], {"min": 100, "max": 1_000_000})
        self.assertNotIn("data", spec)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_duration_time_modifier_is_scaled_to_nanoseconds(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="mytarget",
            event="duration_time:u",
        )
        self.assertIn("duration_time:u", result.columns)
        self.assertEqual(
            result["duration_time:u"].tolist(),
            _ticks_to_ns(result, [10.0, 20.0, 30.0]),
        )

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_event_propagates_and_names_column(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="mytarget",
            event="instructions",
        )

        for call in mock_bench.call_args_list:
            self.assertEqual(call.kwargs["events"], ["instructions"])

        self.assertIn("instructions", result.columns)
        self.assertNotIn("duration_time", result.columns)

    @patch("perf.bench._bench")
    def test_combined_events_single_measurement(self, mock_bench):
        mock_bench.return_value = pd.DataFrame(
            {
                "cycles": [10.0, 20.0],
                "instructions": [4.0, 6.0],
            }
        )

        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="mytarget",
            backend="unroll",
            event=["cycles", "instructions"],
        )

        self.assertEqual(mock_bench.call_count, 2)
        for call in mock_bench.call_args_list:
            self.assertEqual(call.kwargs["events"], ["cycles", "instructions"])
        self.assertIn("cycles", result.columns)
        self.assertIn("instructions", result.columns)
        self.assertEqual(result["cycles"].tolist(), [2.0, 4.0])
        self.assertEqual(result["instructions"].tolist(), [0.8, 1.2])

    @patch("perf.bench._bench")
    def test_separate_events_are_combined(self, mock_bench):
        mock_bench.return_value = pd.Series([10.0, 20.0])

        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="mytarget",
            backend="unroll",
            event=[["cycles"], ["instructions"]],
        )

        self.assertEqual(mock_bench.call_count, 4)
        groups = [call.kwargs["events"] for call in mock_bench.call_args_list]
        self.assertEqual(groups.count(["cycles"]), 2)
        self.assertEqual(groups.count(["instructions"]), 2)
        self.assertIn("cycles", result.columns)
        self.assertIn("instructions", result.columns)
        self.assertEqual(result["cycles"].tolist(), [2.0, 4.0])
        self.assertEqual(result["instructions"].tolist(), [2.0, 4.0])

    @patch("perf.bench._measure_group")
    @patch("perf.mca._explore_asm_cached", return_value=[])
    def test_separate_event_groups_share_calibrated_iterations(
        self, _explore, mock_group
    ):
        seen = []

        def fake_group(config, _mode, _setup, _teardown, _solutions, group, *a, **kw):
            spec = config.get("iterations")
            pinned = spec if not isinstance(spec, dict) else None
            seen.append(spec)
            count = (
                int(pinned)
                if pinned is not None
                else (100 if group == ["cycles"] else 200)
            )
            return pd.Series([1.0]), count

        mock_group.side_effect = fake_group
        result = benchmark(
            config={
                "iterations": {"min": 1, "max": 10000},
                "samples": 1,
                "branch": "predictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["throughput"],
            target={"asm": "nop"},
            name="mytarget",
            event=[["cycles"], ["instructions"]],
        )
        configs = [call.args[0] for call in mock_group.call_args_list]
        self.assertEqual(len(configs), 2, configs)
        self.assertEqual(seen[1], 100)
        self.assertEqual(result["operations"].unique().tolist(), [1600])

    @patch("perf.bench._bench")
    def test_zero_counter_reports_zero(self, mock_bench):
        mock_bench.side_effect = [
            pd.Series([1.0, 2.0]),
            pd.Series([10.0, 20.0]),
            pd.Series([0.0, 0.0]),
            pd.Series([0.0, 0.0]),
        ]

        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="mytarget",
            backend="unroll",
            event=[["cycles"], ["cache-misses"]],
        )

        self.assertIn("cycles", result.columns)
        self.assertIn("cache-misses", result.columns)
        self.assertEqual(result["cycles"].tolist(), [2.0, 4.0])
        self.assertEqual(result["cache-misses"].tolist(), [0.0, 0.0])

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_data_propagates(self, mock_bench):
        benchmark(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="mytarget",
            data={"rax": 32, "0x44000000000": 7},
        )

        for call in mock_bench.call_args_list:
            self.assertEqual(
                call.kwargs["data"],
                {"regs": {"rax": 32}, "mem": {"0x44000000000": 7}},
            )


class TestBenchDataSetup(unittest.TestCase):
    def test_data_registers_generate_movs(self):
        from perf.arch.x86_64 import data_setup_asm

        data = {"regs": {"rax": 32, "rbx": 16}, "mem": {}}
        setup_asm = data_setup_asm(data)
        self.assertIn("mov rax, 0x20", setup_asm)
        self.assertIn("mov rbx, 0x10", setup_asm)

    def test_data_memory_generates_stores(self):
        from perf.arch.x86_64 import data_setup_asm

        data = {
            "regs": {},
            "mem": {
                "0x430000000000000": 32,
                "0x41000000000": 0x12345678,
                "0x42000000000": 0x123456789012345,
            },
        }
        setup_asm = data_setup_asm(data)
        self.assertIn("mov byte ptr [r12], 0x20", setup_asm)
        self.assertIn("mov dword ptr [r12], 0x12345678", setup_asm)
        self.assertIn("mov rax, 0x123456789012345", setup_asm)
        self.assertIn("mov qword ptr [r12], rax", setup_asm)

    def test_map_data_pages_maps_and_writes(self):
        import ctypes

        from perf.core import _map_data_pages

        data = {"regs": {}, "mem": {"0x43000000000": 0xDEADBEEF}}
        _map_data_pages(data)

        ctypes.c_uint64.from_address(0x43000000000).value = 0xDEADBEEF
        self.assertEqual(ctypes.c_uint64.from_address(0x43000000000).value, 0xDEADBEEF)

    def test_map_data_pages_reports_a_taken_page(self):
        import ctypes

        mapped = set(core._MAPPED_DATA_PAGES)
        unmappable = set(core._UNMAPPABLE_DATA_PAGES)

        def _restore():
            core._MAPPED_DATA_PAGES.clear()
            core._MAPPED_DATA_PAGES.update(mapped)
            core._UNMAPPABLE_DATA_PAGES.clear()
            core._UNMAPPABLE_DATA_PAGES.update(unmappable)

        self.addCleanup(_restore)
        buf = ctypes.create_string_buffer(4096)
        page = (ctypes.addressof(buf) + 4095) & ~4095
        with self.assertRaises(OSError) as ctx:
            core._map_data_pages({"regs": {}, "mem": {hex(page): 1}})
        self.assertNotIsInstance(ctx.exception, NameError)
        self.assertTrue(ctx.exception.errno)
        self.assertIn("address already mapped", str(ctx.exception))

    def test_data_assembly_round_trips(self):
        import keystone

        from perf.arch.x86_64 import data_setup_asm
        from perf.core import _map_data_pages

        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        data = {"regs": {"rcx": 0x45000000000}, "mem": {"0x41000000000": 32}}
        _map_data_pages(data)
        asm = data_setup_asm(data)
        enc, _ = ks.asm(asm)
        self.assertTrue(enc)

    def test_reserved_registers_are_rejected(self):
        from perf.cfg import check_data

        for reg in ("r8", "r9", "r10", "rsp", "esp", "rip", "eip", "ip", "flags"):
            with self.assertRaises(ValueError) as ctx:
                check_data({"regs": {reg: 1}, "mem": {}})
            self.assertIn("reserved by the measurement harness", str(ctx.exception))

    def test_harness_registers_reject_per_iteration_lists(self):
        from perf.bench import get_arch
        from perf.cfg import _per_iter_data, check_data

        for reg in ("rbx", "r11", "r12", "r14"):
            with self.subTest(reg=reg), self.assertRaises(ValueError):
                check_data({"regs": {reg: [1, 2, 3]}})
            with self.subTest(reg=reg), self.assertRaises(ValueError):
                _per_iter_data(
                    [],
                    4,
                    None,
                    True,
                    random.Random(0),
                    get_arch(),
                    explicit={"regs": {reg: [1, 2]}},
                )
        check_data({"regs": {"rbx": 1}})

    def test_argument_registers_are_allowed(self):
        from perf.cfg import check_data

        for reg in ("rdi", "rsi", "rdx", "rcx", "rax", "rbx", "rbp", "r12", "r15"):
            check_data({"regs": {reg: 1}, "mem": {}})

    def test_bench_rejects_reserved_registers(self):
        with self.assertRaises(ValueError):
            benchmark(
                target={"asm": "mov eax, 42"},
                mode=["latency"],
                config={"iterations": 8, "samples": 1},
                data={"r8": 1},
            )

    def test_asm_source_label_becomes_a_snippet(self):
        import tempfile
        from pathlib import Path

        from perf.cfg import asm_source_code

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "f.s")
        Path(path).write_text(
            "foo:\n\tmov eax, 42\n\tret\nbar:\n\txor eax, eax\n\tret\n"
        )
        self.assertEqual(asm_source_code(path, "foo"), "mov eax, 42")
        self.assertEqual(asm_source_code(path, "foo..bar"), "mov eax, 42")
        self.assertIsNone(asm_source_code(path, "nope"))
        self.assertIsNone(asm_source_code("/nope/x.s", "foo"))
        self.assertIsNone(asm_source_code("/bin/true", "main"))

    def test_asm_source_falls_back_when_not_assemblable(self):
        from perf.cfg import asm_source_code

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "att.s")
        with open(path, "w") as fh:
            fh.write("att:\n\tmovl $42, %eax\n\tretl\n")
        self.assertIsNone(asm_source_code(path, "att"))

    def test_asm_source_region_semantics(self):
        from pathlib import Path

        from perf.cfg import _asm_region, asm_source_target

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "f.s")
        Path(path).write_text(ASM_SOURCE)
        self.assertEqual(_asm_region(path, "foo"), "mov eax, 42; mov ebx, 1")
        self.assertEqual(_asm_region(path, "bar"), "xor eax, eax")
        self.assertEqual(_asm_region(path, "foo..bar"), "mov eax, 42; mov ebx, 1")
        self.assertEqual(_asm_region(path, "bar..loop"), "xor eax, eax; .Llocal:; nop")
        self.assertNotIn("ret", _asm_region(path, "loop"))
        self.assertEqual(asm_source_target(("a", "b")), "a..b")
        self.assertEqual(asm_source_target("a"), "a")
        self.assertEqual(asm_source_target(None), "")
        for target in ("nope", "foo..nope", "foo..foo", "loop..foo", None):
            with self.assertRaises(ValueError):
                _asm_region(path, target)
        with self.assertRaises(ValueError) as ctx:
            _asm_region(path, "nope")
        self.assertIn("foo, bar, loop", str(ctx.exception))


class TestSolverDataModels(unittest.TestCase):
    def test_cache_levels_parsing(self):
        from perf.arch.x86_64 import data_cache_levels

        cfg = {
            "dcache": {
                "L1": {"hit_rate": 100},
                "L2": {"hit_rate": 50},
                "L3": {"hit_rate": 100},
            }
        }
        levels = data_cache_levels(cfg)
        self.assertEqual(levels["L1d"], 100)
        self.assertEqual(levels["L2"], 50)
        self.assertEqual(levels["L3"], 100)

        self.assertEqual(
            data_cache_levels({"dcache": "hot"}), {"L1d": 100, "L2": 0, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"dcache": "warm"}), {"L1d": 0, "L2": 100, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"dcache": "cool"}), {"L1d": 0, "L2": 0, "L3": 100}
        )
        self.assertEqual(
            data_cache_levels({"dcache": "cold"}), {"L1d": 0, "L2": 0, "L3": 0}
        )
        self.assertIsNone(data_cache_levels({}))

    def test_cache_levels_top_level(self):
        from perf.arch.x86_64 import data_cache_levels

        cfg = {"dcache": {"L1": {"hit_rate": 50}, "L2": {"hit_rate": 100}}}
        levels = data_cache_levels(cfg)
        self.assertEqual(levels["L1d"], 50)
        self.assertEqual(levels["L2"], 100)

    def test_cache_levels_default(self):
        from perf.arch.x86_64 import _MEMORY_SHORTCUTS, data_cache_levels
        from perf.cfg import _DEFAULT_BENCH

        self.assertEqual(_DEFAULT_BENCH["dcache"], ["hot", "warm", "cool", "cold"])
        self.assertEqual(_DEFAULT_BENCH["icache"], ["hot", "cold"])
        self.assertEqual(_DEFAULT_BENCH["dtlb"], ["hot", "cold"])
        self.assertEqual(_DEFAULT_BENCH["itlb"], ["hot", "cold"])
        self.assertEqual(_DEFAULT_BENCH["code"], [{"align": 1}, {"align": 16}])
        self.assertEqual(
            _DEFAULT_BENCH["external"],
            {"lib": False, "stdout": False, "stderr": False},
        )
        self.assertEqual(
            data_cache_levels({"dcache": "hot"}), dict(_MEMORY_SHORTCUTS["hot"])
        )
        levels = data_cache_levels({"dcache": _DEFAULT_BENCH["dcache"][0]})
        self.assertEqual(levels["L1d"], 100)

    def test_cache_state_sensitivity(self):
        from perf.cfg import _DEFAULT_BENCH, _cache_state_sensitive

        self.assertFalse(_cache_state_sensitive(None))
        self.assertFalse(_cache_state_sensitive({}))
        self.assertTrue(_cache_state_sensitive(_DEFAULT_BENCH))
        self.assertFalse(
            _cache_state_sensitive(
                {
                    "dcache": "hot",
                    "icache": "hot",
                    "dtlb": "hot",
                    "itlb": "hot",
                }
            )
        )
        self.assertTrue(_cache_state_sensitive({"dcache": "warm"}))
        self.assertTrue(_cache_state_sensitive({"dcache": ["hot", "cold"]}))
        self.assertTrue(_cache_state_sensitive({"icache": "cold"}))
        self.assertTrue(_cache_state_sensitive({"dtlb": "cold"}))
        self.assertTrue(_cache_state_sensitive({"itlb": "cold"}))
        self.assertTrue(_cache_state_sensitive({"dcache": {"L1": {"hit_rate": 50}}}))
        self.assertTrue(_cache_state_sensitive({"dcache": {"L1": {"0x1000": "cold"}}}))
        self.assertFalse(_cache_state_sensitive({"dcache": {"L1": {"0x1000": "hot"}}}))
        self.assertFalse(
            _cache_state_sensitive({"icache": "cold"}, keys={"dcache", "dtlb"})
        )
        self.assertTrue(_cache_state_sensitive({"dcache": "cold"}, keys={"dcache"}))
        self.assertTrue(_cache_state_sensitive({"icache": "cold"}, keys={"icache"}))

    def test_topdown_group_detection(self):
        from perf.bench import _topdown_group

        events = ["topdown-retiring", "topdown-bad-spec"]
        self.assertEqual(_topdown_group(events), events)
        self.assertEqual(_topdown_group(["slots", *events]), events)
        self.assertEqual(_topdown_group(["topdown-retiring", "cycles"]), [])
        self.assertEqual(_topdown_group(["cycles"]), [])

    def test_topdown_percentages(self):
        from perf.bench import _topdown_percentages

        events = ["topdown-retiring", "topdown-bad-spec", "topdown-be-bound"]
        result = _topdown_percentages([[10, 10, 80]], events, events)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0][0], 10.0)
        self.assertAlmostEqual(result[0][1], 10.0)
        self.assertAlmostEqual(result[0][2], 80.0)

        result = _topdown_percentages([[10, 10, 80]], events, ["topdown-be-bound"])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0][0], 80.0)

        with_slots = ["slots", "topdown-retiring", "topdown-be-bound"]
        result = _topdown_percentages(
            [[100, 25, 75]], with_slots, ["topdown-retiring", "topdown-be-bound"]
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0][0], 25.0)
        self.assertAlmostEqual(result[0][1], 75.0)

        self.assertIsNone(_topdown_percentages([[0, 0, 0]], events, events))

    def test_records_keeps_topdown_percentages_undivided(self):
        from perf.bench import _records

        df = pd.DataFrame({"topdown-be-bound": [50.0], "cycles": [10.0]})
        recs = _records(
            "f", "n", "latency", df, ["topdown-be-bound", "cycles"], divisor=5
        )
        self.assertEqual(recs[0]["topdown-be-bound"], 50.0)
        self.assertEqual(recs[0]["cycles"], 2.0)

    def test_records_keeps_scalar_topdown_percentages_undivided(self):
        from perf.bench import _records

        recs = _records(
            "f",
            "n",
            "latency",
            [[50.0, 10.0]],
            ["topdown-be-bound", "cycles"],
            slots={"slot": [7]},
            divisor=5,
        )
        self.assertEqual(recs[0]["topdown-be-bound"], 50.0)
        self.assertEqual(recs[0]["cycles"], 2.0)
        self.assertEqual(recs[0]["slot"], 7)

        partial = _records(
            "f",
            "n",
            "latency",
            [[50.0]],
            ["topdown-be-bound", "cycles"],
            slots={"slot": [7]},
            divisor=5,
        )
        self.assertEqual(partial[0]["topdown-be-bound"], 50.0)
        self.assertNotIn("cycles", partial[0])
        self.assertEqual(partial[0]["slot"], 7)

        one = _records(
            "f",
            "n",
            "latency",
            [50.0],
            ["topdown-be-bound"],
            slots={"slot": [7]},
            divisor=5,
        )
        self.assertEqual(one[0]["topdown-be-bound"], 50.0)
        self.assertEqual(one[0]["slot"], 7)

        counted = _records("f", "n", "latency", [50.0], ["cycles"], divisor=5)
        self.assertEqual(counted[0]["cycles"], 10.0)

    def test_topdown_leader_is_not_a_result_column(self):
        from perf.bench import benchmark

        frame = pd.DataFrame(
            {
                "topdown-retiring": [50.0],
                "topdown-bad-spec": [10.0],
                "topdown-fe-bound": [20.0],
                "topdown-be-bound": [20.0],
            }
        )
        with (
            patch("perf.mca._explore_asm_cached", return_value=[]),
            patch("perf.bench._measure_group", return_value=(frame, 128)),
        ):
            df = benchmark(
                target={"asm": "nop"},
                mode=["latency"],
                event=["topdown-*"],
                config={"iterations": 8, "samples": 1},
            )
        self.assertTrue(
            any(name.startswith("topdown-") for name in df.columns),
            list(df.columns),
        )
        self.assertFalse(
            [name for name in df.columns if "slots" in str(name)],
            list(df.columns),
        )

    def test_latency_steers_before_timing_and_primes_after(self):
        import random

        from perf.bench import _build_loop_asm, get_arch

        addr = 0x4100003000
        models = [{"regs": {"rdi": addr}, "reads": [(addr, 8, 0)], "writes": []}]
        full, _meta, _buf = _build_loop_asm(
            10,
            "latency",
            "mov rax, [rdi];",
            "",
            "",
            models,
            ["cycles"],
            [0],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            get_arch(),
        )
        loop = full.split(".perfloop:", 1)[1]
        steer = loop.index(f"mov rax, [0x{addr:x}]")
        t0 = loop.index("lfence")
        prime = loop.index("mov rdi, [r14 + r8*8]")
        self.assertLess(steer, t0)
        self.assertLess(t0, prime)
        self.assertLess(loop.index("push rax"), steer)
        self.assertLess(steer, loop.index("pop rax"))
        self.assertEqual(full.count("mov rdi, [r14 + r8*8]"), 1)

    def test_latency_keeps_setup_rdi_and_has_no_slot_load(self):
        from perf.bench import _build_loop_asm, get_arch

        full, _meta, _buf = _build_loop_asm(
            4,
            "latency",
            "mov rax, rdi;",
            "mov rdi, 0x1234",
            "",
            [],
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            get_arch(),
        )
        self.assertNotIn("[r9 + 8*r8]", full)
        self.assertIn("mov rdi, 0x1234", full.split(".perfloop:")[0])

    def test_harness_zeroes_rdi_before_setup(self):
        import random

        from perf.bench import _build_loop_asm, get_arch

        for mode in ("latency", "throughput"):
            with self.subTest(mode=mode):
                full, _meta, _buf = _build_loop_asm(
                    4,
                    mode,
                    "nop;",
                    "mov rax, rdi;",
                    "",
                    [],
                    ["duration_time"],
                    [None],
                    {"L1d": 100, "L2": 0, "L3": 0},
                    True,
                    random.Random(0),
                    get_arch(),
                )
                prologue = full.split(".perfloop:", 1)[0]
                self.assertIn("xor edi, edi", prologue)
                self.assertLess(
                    prologue.index("xor edi, edi"),
                    prologue.index("mov rax, rdi"),
                )

    def test_latency_reloads_model_registers_after_timing(self):
        import random

        from perf.bench import _build_loop_asm, get_arch

        addr = 0x4100003000
        models = [{"regs": {"rdx": addr}, "reads": [], "writes": []}]
        full, _meta, _buf = _build_loop_asm(
            1,
            "latency",
            "call rax;",
            "",
            "",
            models,
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            get_arch(),
        )
        loop = full.split(".perfloop:", 1)[1]
        t0_end = loop.index("mov rbx, rax")
        reload = loop.rindex("mov rdx, [r14 + r8*8]")
        self.assertGreater(reload, t0_end)

    def test_latency_dereferences_a_model_pointer_after_timing(self):
        from perf.bench import benchmark

        df = benchmark(
            target={"asm": "mov rax, [rdx]; ret"},
            mode=["latency"],
            event=["duration_time"],
            config={
                "samples": 1,
                "iterations": 1,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
        )
        self.assertFalse(df.empty)
        self.assertIn("duration_time", df.columns)

    def test_per_iter_predictable_cycles_models_in_order(self):
        import random

        from perf.cfg import _per_iter_data

        models = [
            {"regs": {"rdi": 15, "rcx": 3}, "reads": [], "writes": []},
            {"regs": {"rdi": 3, "rcx": 15}, "reads": [], "writes": []},
        ]
        buf, meta = _per_iter_data(models, 6, None, True, random.Random(1))
        k = int(meta["k"])
        self.assertEqual(
            [int(buf[0, k - 1 - i]) for i in range(6)], [15, 3, 15, 3, 15, 3]
        )
        self.assertEqual(
            [int(buf[1, k - 1 - i]) for i in range(6)], [3, 15, 3, 15, 3, 15]
        )

    def test_per_iter_unpredictable_varies(self):
        import random

        from perf.cfg import _per_iter_data

        models = [
            {"regs": {"rdi": 15}, "reads": [], "writes": []},
            {"regs": {"rdi": 3}, "reads": [], "writes": []},
        ]
        buf, meta = _per_iter_data(models, 200, None, False, random.Random(7))
        self.assertEqual(set(buf[0, 1:201]), {3, 15})

    def test_per_iter_heap_modes(self):
        import random

        from perf.cfg import _per_iter_data

        heap = 0x41000000000
        other = 0x42000000000
        data = {
            "mem": {f"0x{heap:x}": 7, f"0x{other:x}": 9},
            "heap": {f"0x{heap:x}": 8},
        }
        buf, meta = _per_iter_data(
            [],
            16,
            None,
            True,
            random.Random(1),
            explicit=data,
            heap_mode="random",
        )
        heap_vals = [int(v) for v in buf[meta["val_col"][heap], 1:17]]
        other_vals = [int(v) for v in buf[meta["val_col"][other], 1:17]]
        self.assertGreater(len(set(heap_vals)), 1)
        self.assertEqual(set(other_vals), {9})

        zero, zero_meta = _per_iter_data(
            [],
            16,
            None,
            True,
            random.Random(1),
            explicit=data,
            heap_mode="zero",
        )
        self.assertEqual(set(zero[zero_meta["val_col"][heap], 1:17]), {0})

        fixed, fixed_meta = _per_iter_data(
            [],
            16,
            None,
            True,
            random.Random(1),
            explicit=data,
            heap_mode="fixed",
        )
        self.assertIn(heap, fixed_meta["mem_addrs"])
        self.assertNotIn(heap, fixed_meta["write_addrs"])

    def test_harness_regs_excluded(self):
        import random

        from perf.cfg import _per_iter_data

        models = [{"regs": {"r8": 99, "rdi": 5}, "reads": [], "writes": []}]
        buf, meta = _per_iter_data(models, 10, None, True, random.Random(1))
        self.assertNotIn("r8", meta["reg_col"])
        self.assertIn("rdi", meta["reg_col"])

    def test_data_script_assembles(self):
        import random

        import keystone

        from perf.bench import _arch_asm
        from perf.cfg import _fill_evict_tables, _per_iter_data

        models = [
            {
                "regs": {"rdi": 15, "rcx": 3},
                "reads": [(0x41000000000, 4, 42)],
                "writes": [],
            },
            {
                "regs": {"rdi": 3, "rcx": 15},
                "reads": [(0x41000000000, 4, 7)],
                "writes": [],
            },
        ]
        buf, meta = _per_iter_data(
            models, 20, {"L1d": 100, "L2": 50, "L3": None}, False, random.Random(1)
        )
        _fill_evict_tables(buf, meta, buf.ctypes.data)
        asm = "\n".join(
            p
            for p in (
                _arch_asm("steer_asm", meta, buf.ctypes.data),
                _arch_asm("prime_asm", meta, buf.ctypes.data),
            )
            if p
        )
        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        enc, _ = ks.asm(asm)
        self.assertTrue(enc)

    def test_data_script_loop_over_many_addresses(self):
        import random

        import keystone

        from perf.bench import _arch_asm
        from perf.cfg import _fill_evict_tables, _per_iter_data

        addrs = [0x41000000000 + i * 0x1000 for i in range(4)]
        models = [
            {"regs": {"rdi": 15}, "reads": [(a, 4, 42) for a in addrs], "writes": []},
            {"regs": {"rdi": 3}, "reads": [(a, 4, 7) for a in addrs], "writes": []},
        ]
        buf, meta = _per_iter_data(models, 8, None, False, random.Random(1))
        _fill_evict_tables(buf, meta, buf.ctypes.data)
        asm = "\n".join(
            p
            for p in (
                _arch_asm("steer_asm", meta, buf.ctypes.data),
                _arch_asm("prime_asm", meta, buf.ctypes.data),
            )
            if p
        )
        self.assertLessEqual(asm.count("dec r15"), 1)
        self.assertEqual(asm.count(".perfwrite:"), 1)
        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        enc, _ = ks.asm(asm)
        self.assertTrue(enc)

    def test_fill_evict_tables_bakes_absolute_bases(self):
        import random

        from perf.cfg import _fill_evict_tables, _per_iter_data

        addr = 0x41000000000
        models = [
            {"regs": {}, "reads": [(addr, 4, 1)], "writes": []},
        ]
        buf, meta = _per_iter_data(models, 10, None, True, random.Random(1))
        data_ptr = buf.ctypes.data
        _fill_evict_tables(buf, meta, data_ptr)
        n = len(meta["mem_addrs"])
        et = meta["evict_table_col"]
        self.assertEqual(buf[et, 0], addr)
        self.assertEqual(buf[et, n], data_ptr + meta["val_col"][addr] * meta["K"] * 8)
        self.assertEqual(list(buf[et][2 * n :]), [0] * (len(buf[et]) - 2 * n))

    def test_fill_evict_tables_many_addresses_small_iterations(self):
        import random

        from perf.cfg import _fill_evict_tables, _per_iter_data

        addrs = [0x41000000000 + i * 0x1000 for i in range(50)]
        models = [
            {"regs": {"rdi": 7}, "reads": [(a, 4, 1) for a in addrs], "writes": []},
        ]
        buf, meta = _per_iter_data(
            models, 8, {"L1d": 0, "L2": 0, "L3": 0}, True, random.Random(1)
        )
        self.assertGreater(meta["K"], meta["k"])
        data_ptr = buf.ctypes.data
        _fill_evict_tables(buf, meta, data_ptr)
        n = len(meta["mem_addrs"])
        et = meta["evict_table_col"]
        for i, a in enumerate(addrs):
            self.assertEqual(buf[et, i], a)
            self.assertEqual(
                buf[et, n + i], data_ptr + meta["val_col"][a] * meta["K"] * 8
            )
        self.assertEqual(list(buf[et][2 * n :]), [0] * (len(buf[et]) - 2 * n))


class TestStatisticalIterations(unittest.TestCase):
    def test_plan_iterations_scales_with_noise(self):
        import numpy as np

        from perf.bench import _plan_iterations

        rng = np.random.default_rng(0)
        noisy = rng.exponential(100, 200)
        n_noisy = _plan_iterations(noisy, 0.005, 128, 5_000_000)

        n_loose = _plan_iterations(noisy, 0.02, 128, 5_000_000)
        self.assertGreater(n_noisy, n_loose)

        self.assertEqual(
            _plan_iterations(np.full(200, 50.0), 0.005, 128, 5_000_000), 128
        )

        self.assertEqual(_plan_iterations([1, 2], 0.005, 128, 5_000_000), 128)

        self.assertEqual(_plan_iterations(noisy, 1e-9, 128, 10_000), 10_000)

    def test_plan_iterations_defaults(self):
        from perf.bench import _probe_plan

        target, lo, hi = _probe_plan({})
        self.assertEqual(target, 0.01)
        self.assertEqual(lo, 100)
        self.assertEqual(hi, 1_000_000)

        target, lo, hi = _probe_plan(
            {
                "backend": {"loop": {"target": 0.01}},
                "iterations": {"min": 64, "max": 9999},
            }
        )
        self.assertEqual((target, lo, hi), (0.01, 64, 9999))
        target, lo, hi = _probe_plan(
            {
                "backend": {"loop": {"target": 0.01}, "unroll": {"target": 0.5}},
                "iterations": {"min": 64, "max": 9999},
            },
            "unroll",
        )
        self.assertEqual((target, lo, hi), (0.5, 64, 9999))


class TestBenchFilePath(unittest.TestCase):
    def _binary(self):
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        self.addCleanup(Path(path).unlink, missing_ok=True)
        return path

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.mca.explore", return_value=[])
    @patch("perf.mca.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.mca.Elf")
    @patch("angr.Project")
    def test_file_path_builds_dataframe(
        self,
        _mock_project,
        mock_elf,
        mock_get_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_get_arch.return_value = x86_64
        path = self._binary()

        result = benchmark(
            config={
                "iterations": 10,
                "samples": 2,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target=f"{path}:func",
            name="func",
        )

        self.assertEqual(
            list(result.index.names),
            ["file", "name", "mode"],
        )
        self.assertEqual(
            result["duration_time"].tolist(),
            _ticks_to_ns(result, [50, 60]),
        )
        self.assertNotIn("iteration", result.columns)
        self.assertNotIn("samples", result.columns)

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.mca.explore", return_value=[])
    @patch("perf.mca.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.mca.Elf")
    @patch("angr.Project")
    def test_data_regs_map_to_prototype(
        self,
        _mock_project,
        mock_elf,
        mock_get_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_get_arch.return_value = x86_64
        path = self._binary()

        benchmark(
            config={"iterations": 10, "samples": 2},
            mode=["latency"],
            target=f"{path}:func",
            name="func",
            data={"rdi": 42, "rsi": 7},
        )

        self.assertTrue(mock_explore.called)
        call = mock_explore.call_args

        self.assertEqual(call.kwargs["data"]["regs"]["rdi"], 42)
        self.assertEqual(call.kwargs["data"]["regs"]["rsi"], 7)

        proto = call.kwargs["prototype"]
        self.assertIsNotNone(proto)
        self.assertEqual(len(proto.args), 2)

    def test_data_arg_aliases_map_to_sysv_registers(self):
        from perf.arch.x86_64 import arg_alias, canonical_data_reg, data_reg_keys

        self.assertEqual(canonical_data_reg("arg0"), "rdi")
        self.assertEqual(canonical_data_reg("arg5"), "r9")
        self.assertEqual(canonical_data_reg("arg6"), "arg6")
        self.assertEqual(canonical_data_reg("rdi"), "rdi")
        self.assertEqual(arg_alias("rdi"), "arg0")
        self.assertEqual(arg_alias("rsi"), "arg1")
        self.assertIsNone(arg_alias("rax"))
        self.assertIn("arg0", data_reg_keys("rdi"))
        self.assertIn("rdi", data_reg_keys("arg0"))

    def test_data_arg_aliases_roundtrip(self):
        from perf.arch.x86_64 import (
            _ARG_REGS,
            arg_alias,
            canonical_data_reg,
            data_reg_keys,
        )

        for idx, reg in enumerate(_ARG_REGS):
            alias = f"arg{idx}"
            self.assertEqual(canonical_data_reg(alias), reg)
            self.assertEqual(arg_alias(reg), alias)
            self.assertEqual(canonical_data_reg(arg_alias(reg)), reg)
            self.assertEqual(arg_alias(canonical_data_reg(alias)), alias)
            self.assertEqual(data_reg_keys(alias), data_reg_keys(reg))

    def test_data_param_alias_symmetric(self):
        from perf.cfg import data_param_columns, data_param_value

        pairs = [
            ("arg0", "rdi"),
            ("arg1", "rsi"),
            ("arg2", "rdx"),
            ("arg3", "rcx"),
            ("arg4", "r8"),
            ("arg5", "r9"),
        ]
        for alias, canon in pairs:
            for stored in (canon, alias, canon.upper()):
                data = {"regs": {stored: 7}}
                for query in (f"data.{canon}", f"data.{alias}"):
                    self.assertEqual(data_param_value(data, query), 7, query)
                cols = data_param_columns(data)
                self.assertEqual(cols[f"data.{canon}"], 7)
                self.assertNotIn(f"data.{alias}", cols)
                self.assertNotIn(f"data.regs.{canon}", cols)
                self.assertNotIn(f"data.regs.{alias}", cols)

    def test_data_param_alias_shared_key_canonicalizes(self):
        from perf.cfg import data_param_columns, data_param_value

        data = {"regs": {"rdi": 1, "arg0": 2}}
        cols = data_param_columns(data)
        self.assertEqual(cols["data.rdi"], 2)
        self.assertNotIn("data.arg0", cols)
        self.assertEqual(data_param_value(data, "data.rdi"), 2)
        self.assertEqual(data_param_value(data, "data.arg0"), 2)

    def test_data_param_value_mem_fallback_preserved(self):
        from perf.cfg import data_param_value

        data = {"regs": {"rdi": 1}, "mem": {"0x1000": 5}}
        self.assertEqual(data_param_value(data, "data.0x1000"), 5)
        self.assertIsNone(data_param_value(data, "data.mem.0x1000"))
        self.assertIsNone(data_param_value(data, "data.memory.0x1000"))
        self.assertIsNone(data_param_value(data, "data.regs.rdi"))
        self.assertEqual(data_param_value(data, "data.rdi"), 1)
        self.assertEqual(data_param_value(data, "data.arg0"), 1)
        self.assertEqual(data_param_value(data, "data.rax"), None)

    def test_data_param_columns_keep_distribution(self):
        from perf.cfg import data_param_columns, data_param_value

        data = {"regs": {"rdi": [1, 3, 5]}, "mem": {"0x1000": [7, 8]}}
        cols = data_param_columns(data)
        self.assertEqual(cols["data.rdi"], [1, 3, 5])
        self.assertNotIn("data.arg0", cols)
        self.assertEqual(cols["data.0x1000"], [7, 8])
        self.assertEqual(data_param_value(data, "data.rdi"), 1)

    def test_solver_data_collects_state_lists(self):
        from perf.mca import _MODEL_CACHE

        _MODEL_CACHE.clear()
        from perf.mca import explored_data_dict

        models = [
            {"regs": {"rdi": 1}, "reads": [(0x1000, 8, 7)], "writes": []},
            {"regs": {"rdi": 2}, "reads": [(0x1000, 8, 8)], "writes": []},
            {"regs": {"rdi": 3}, "reads": [(0x1000, 8, 7)], "writes": []},
        ]
        with patch("perf.mca.solve", return_value=models):
            d = explored_data_dict(["fake"])
        self.assertEqual(d["regs"]["rdi"], [1, 2, 3])
        self.assertEqual(d["mem"]["0x1000"], [7, 8])

    def test_solver_data_single_value_is_scalar(self):
        from perf.mca import _MODEL_CACHE

        _MODEL_CACHE.clear()
        from perf.mca import explored_data_dict

        models = [
            {"regs": {"rsi": 42}, "reads": [(0x2000, 8, 5)], "writes": []},
        ]
        with patch("perf.mca.solve", return_value=models):
            d = explored_data_dict(["fake"])
        self.assertEqual(d["regs"]["rsi"], 42)
        self.assertEqual(d["mem"]["0x2000"], 5)

    def test_solver_data_skips_harness_regs(self):
        from perf.mca import _MODEL_CACHE

        _MODEL_CACHE.clear()
        from perf.mca import explored_data_dict

        models = [
            {"regs": {"rdi": 1, "r8": 99}, "reads": [], "writes": []},
            {"regs": {"rdi": 2, "r8": 100}, "reads": [], "writes": []},
        ]
        with patch("perf.mca.solve", return_value=models):
            d = explored_data_dict(["fake"])
        self.assertEqual(d["regs"]["rdi"], [1, 2])
        self.assertNotIn("r8", d["regs"])

    def test_solver_data_canonicalizes_no_alias_columns(self):
        from perf.mca import _MODEL_CACHE

        _MODEL_CACHE.clear()
        from perf.cfg import data_param_columns
        from perf.mca import explored_data_dict

        models = [
            {"regs": {"rdi": 1}, "reads": [], "writes": []},
            {"regs": {"rdi": 2}, "reads": [], "writes": []},
        ]
        with patch("perf.mca.solve", return_value=models):
            d = explored_data_dict(["fake"])
        cols = data_param_columns({"regs": d["regs"], "mem": d["mem"]})
        self.assertEqual(cols["data.rdi"], [1, 2])
        self.assertNotIn("data.arg0", cols)

    def test_func_prototype_imports_from_core(self):
        from perf.core import _func_prototype

        self.assertIsNotNone(_func_prototype(None, None))

    def test_scratch_value_filters_hex_strings(self):
        from perf.cfg import _is_scratch_value, data_param_columns
        from perf.core import _page_mappable

        self.assertTrue(_is_scratch_value(0x4100001000))
        self.assertTrue(_is_scratch_value("0x4100001000"))
        self.assertTrue(_is_scratch_value(["0x4100001000"]))
        self.assertFalse(_is_scratch_value(123))
        self.assertFalse(_is_scratch_value("0x1000"))
        self.assertEqual(data_param_columns({"regs": {"rdi": 0x4100001000}}), {})
        self.assertEqual(data_param_columns({"regs": {"rdi": "0x4100001000"}}), {})
        self.assertTrue(_page_mappable("0x4100001000"))
        self.assertTrue(_page_mappable(0x4100001000))


class TestConfigSeedAffinityPriority(unittest.TestCase):
    def test_seed_int_parsing(self):
        from perf.cfg import _config_seed_int

        self.assertIsNone(_config_seed_int({}))
        self.assertIsNone(_config_seed_int(None))
        self.assertEqual(_config_seed_int({"seed": 42}), 42)
        self.assertIsInstance(_config_seed_int({"seed": "random"}), int)
        self.assertEqual(
            _config_seed_int({"seed": "abc"}), _config_seed_int({"seed": "abc"})
        )

    def test_seeded_rng_reproducible(self):
        from perf.cfg import _make_rng

        a = _make_rng({"seed": 7})
        b = _make_rng({"seed": 7})
        self.assertEqual([a.random() for _ in range(5)], [b.random() for _ in range(5)])

    def test_per_iter_data_deterministic_with_seed(self):
        from perf.cfg import _make_rng, _per_iter_data

        models = [
            {"regs": {"rdi": 15}, "reads": [], "writes": []},
            {"regs": {"rdi": 3}, "reads": [], "writes": []},
        ]
        buf1, _ = _per_iter_data(models, 64, None, False, _make_rng({"seed": 9}))
        buf2, _ = _per_iter_data(models, 64, None, False, _make_rng({"seed": 9}))
        self.assertTrue((buf1 == buf2).all())

    def test_affinity_forms(self):
        from perf.cfg import _parse_affinity

        self.assertIsNone(_parse_affinity(None))
        self.assertIsNone(_parse_affinity("none"))
        self.assertEqual(_parse_affinity(2), [2])
        self.assertEqual(_parse_affinity([1, 0]), [0, 1])
        self.assertEqual(_parse_affinity("0,1"), [0, 1])
        self.assertEqual(_parse_affinity("0-1"), [0, 1])
        with self.assertRaises(ValueError):
            _parse_affinity("bogus")
        with self.assertRaises(ValueError):
            _parse_affinity({"cpu": 1})

    def test_priority_forms(self):
        from perf.cfg import _parse_priority

        self.assertIsNone(_parse_priority(None))
        self.assertEqual(_parse_priority(5), 5)
        self.assertEqual(_parse_priority("-3"), -3)
        self.assertEqual(_parse_priority("-20"), -20)
        self.assertEqual(_parse_priority("lowest"), 19)
        self.assertEqual(_parse_priority("low"), 10)
        self.assertEqual(_parse_priority("normal"), 0)
        self.assertEqual(_parse_priority("high"), -10)
        self.assertEqual(_parse_priority("highest"), -20)
        with self.assertRaises(ValueError):
            _parse_priority("bogus")
        with self.assertRaises(ValueError):
            _parse_priority({"policy": "fifo"})

    def test_numa_forms(self):
        from perf.cfg import _numa_spec, _parse_numa

        self.assertIsNone(_parse_numa(None))
        self.assertIsNone(_parse_numa("none"))
        self.assertEqual(_parse_numa(0), 0)
        self.assertEqual(_parse_numa(1), 1)
        self.assertEqual(_parse_numa("1"), 1)
        self.assertEqual(_parse_numa("0x1"), 1)
        with self.assertRaises(ValueError):
            _parse_numa(-1)
        with self.assertRaises(ValueError):
            _parse_numa("bogus")
        with self.assertRaises(ValueError):
            _parse_numa([0, 1])
        self.assertIsNone(_numa_spec({}))
        self.assertEqual(_numa_spec({"thread": {"numa": 2}}), 2)
        self.assertEqual(_numa_spec({"thread": [[{"numa": 3, "affinity": 4}]]}), 3)

    def test_numa_is_a_valid_thread_key(self):
        from perf.cfg import _check_thread, _validate_thread_top

        self.assertEqual(_check_thread({"numa": 0}), {"numa": 0})
        _validate_thread_top([[{"numa": 0, "affinity": 4, "priority": "normal"}]])
        with self.assertRaises(ValueError):
            _check_thread({"node": 0})

    def test_numa_mask(self):
        import ctypes

        from perf.core import _node_mask

        mask, maxnode = _node_mask(0)
        self.assertEqual(ctypes.sizeof(mask), 8)
        self.assertEqual(maxnode, 1)
        self.assertEqual(list(mask)[0], 1)
        mask, maxnode = _node_mask(130)
        self.assertEqual(ctypes.sizeof(mask), 24)
        self.assertEqual(maxnode, 131)
        self.assertEqual(list(mask)[130 // 8], 1 << (130 % 8))

    def test_numa_guard_binds_and_restores(self):

        calls = []

        def _set(mode, node=None):
            calls.append((mode, node))
            return 0

        with (
            patch.object(cfg_mod, "_set_mempolicy", side_effect=_set),
            patch.object(cfg_mod, "_get_mempolicy", return_value=0),
            patch.object(cfg_mod, "_numa_nodes", return_value=[0, 1]),
        ):
            with cfg_mod._numa_guard({"thread": {"numa": 1}}):
                pass
            with cfg_mod._numa_guard({}):
                pass
        self.assertEqual(calls, [(cfg_mod._MPOL_BIND, 1), (0, None)])

    def test_numa_guard_warns_and_continues(self):

        with (
            patch.object(cfg_mod, "_set_mempolicy", return_value=-1),
            patch.object(cfg_mod, "_get_mempolicy", return_value=0),
            patch.object(cfg_mod, "_numa_nodes", return_value=[0, 1]),
            patch("warnings.warn") as warn,
        ):
            with cfg_mod._numa_guard({"thread": {"numa": 1}}):
                pass
        self.assertTrue(warn.called)

    def test_numa_guard_skips_the_only_online_node(self):

        with (
            patch.object(cfg_mod, "_set_mempolicy") as setter,
            patch.object(cfg_mod, "_numa_nodes", return_value=[0]),
            patch("warnings.warn") as warn,
        ):
            with cfg_mod._numa_guard({"thread": {"numa": 0}}):
                pass
        setter.assert_not_called()
        warn.assert_not_called()

    def test_numa_guard_skips_an_offline_node(self):

        with (
            patch.object(cfg_mod, "_set_mempolicy") as setter,
            patch.object(cfg_mod, "_numa_nodes", return_value=[0]),
            patch("warnings.warn") as warn,
        ):
            with cfg_mod._numa_guard({"thread": {"numa": 3}}):
                pass
        setter.assert_not_called()
        self.assertIn("not online", str(warn.call_args.args[0]))

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_numa_affinity_defaults_to_the_node(self, mock_bench):
        with patch("perf.core._numa_cpus", return_value=[8, 9]):
            result = benchmark(
                config={"iterations": 8, "thread": {"numa": 1}},
                mode=["latency"],
                target={"asm": "nop"},
                name="t",
            )
        cfg = mock_bench.call_args.kwargs["config"]
        self.assertEqual(cfg["thread"][0]["numa"], 1)
        self.assertEqual(cfg["thread"][0]["affinity"], 8)
        self.assertEqual(result.attrs["config"]["thread"][0][0]["numa"], 1)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_numa_leaves_an_explicit_affinity_alone(self, mock_bench):
        with patch("perf.core._numa_cpus", return_value=[8, 9]):
            benchmark(
                config={"iterations": 8, "thread": {"numa": 1, "affinity": 4}},
                mode=["latency"],
                target={"asm": "nop"},
                name="t",
            )
        cfg = mock_bench.call_args.kwargs["config"]
        self.assertEqual(cfg["thread"][0]["numa"], 1)
        self.assertEqual(cfg["thread"][0]["affinity"], 4)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_affinity_defaults_to_the_pmu_that_schedules_the_events(self, mock_bench):
        with (
            patch.object(core, "event_cpus", return_value=[6, 7, 8, 9]),
            patch.object(core, "_current_affinity_list", return_value=[6, 7, 8, 9]),
        ):
            result = benchmark(
                config={"iterations": 8},
                mode=["latency"],
                target={"asm": "nop"},
                name="t",
                event=["topdown-*"],
            )
        cfg = mock_bench.call_args.kwargs["config"]
        self.assertEqual(cfg["thread"][0]["affinity"], 6)
        self.assertEqual(result.attrs["config"]["thread"][0][0]["affinity"], 6)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_affinity_default_survives_every_event_group(self, mock_bench):
        with (
            patch.object(core, "event_cpus", return_value=[2, 3]),
            patch.object(core, "_current_affinity_list", return_value=[2, 3]),
        ):
            benchmark(
                config={"iterations": 8},
                mode=["latency"],
                target={"asm": "nop"},
                name="t",
                event=["cycles", "instructions", "cache-misses"],
            )
        self.assertEqual(
            mock_bench.call_args.kwargs["config"]["thread"][0]["affinity"], 2
        )

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_an_explicit_affinity_is_never_replaced(self, mock_bench):
        with patch.object(core, "event_cpus", return_value=[6, 7, 8, 9]):
            benchmark(
                config={"iterations": 8, "thread": {"affinity": 0}},
                mode=["latency"],
                target={"asm": "nop"},
                name="t",
                event=["topdown-*"],
            )
        self.assertEqual(
            mock_bench.call_args.kwargs["config"]["thread"][0]["affinity"], 0
        )

    def test_pin_default_affinity_leaves_swept_affinities_alone(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{"affinity": 1}], [{"affinity": 2}]]}
        with patch.object(core, "allowed_cpus") as fallback:
            _pin_default_affinity(config, ["cycles"])
        fallback.assert_not_called()
        self.assertEqual(config["thread"], [[{"affinity": 1}], [{"affinity": 2}]])

    def test_pin_default_affinity_fills_every_alternative(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{"priority": "high"}], [{"priority": "low"}]]}
        with patch.object(core, "allowed_cpus", return_value=[3]) as fallback:
            _pin_default_affinity(config, ["duration_time"])
        self.assertEqual(fallback.call_count, 2, fallback.call_args_list)
        fallback.assert_any_call([["duration_time"]], None)
        self.assertEqual(
            config["thread"],
            [
                [{"priority": "high", "affinity": 3}],
                [{"priority": "low", "affinity": 3}],
            ],
        )

    def test_pin_default_affinity_spreads_concurrent_threads_over_cpus(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{}, {}, {}]]}
        with patch.object(core, "allowed_cpus", return_value=[2, 3, 4]) as fallback:
            _pin_default_affinity(config, ["cycles"])
        fallback.assert_any_call([["cycles"]], None)
        self.assertEqual(
            config["thread"],
            [[{"affinity": 2}, {"affinity": 3}, {"affinity": 4}]],
        )

    def test_pin_default_affinity_wraps_when_cpus_run_out(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{}, {}, {}]]}
        with patch.object(core, "allowed_cpus", return_value=[7, 2]):
            _pin_default_affinity(config, ["cycles"])
        self.assertEqual(
            config["thread"],
            [[{"affinity": 7}, {"affinity": 2}, {"affinity": 7}]],
        )

    def test_pin_default_affinity_reserves_explicit_affinities(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{"affinity": 2, "numa": 1}, {}, {}]]}
        with patch.object(core, "allowed_cpus", return_value=[2, 3, 4]) as fallback:
            _pin_default_affinity(config, ["cycles"])
        self.assertEqual(fallback.call_count, 2)
        self.assertEqual(
            [call.args for call in fallback.call_args_list],
            [([["cycles"]], None), ([["cycles"]], None)],
        )
        self.assertEqual(
            config["thread"],
            [[{"numa": 1, "affinity": 2}, {"affinity": 3}, {"affinity": 4}]],
        )

    def test_pin_default_affinity_limits_threads_to_their_numa_node(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{"numa": 0}, {"numa": 1}]]}
        with patch.object(
            core,
            "allowed_cpus",
            side_effect=lambda _groups, numa: [1, 2] if numa == 0 else [3, 4],
        ) as fallback:
            _pin_default_affinity(config, ["cycles"])
        self.assertEqual(fallback.call_count, 2)
        self.assertEqual(
            config["thread"],
            [[{"numa": 0, "affinity": 1}, {"numa": 1, "affinity": 3}]],
        )

    def test_pin_default_affinity_swallows_cpu_discovery_failures(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{}, {}]]}
        with patch.object(core, "allowed_cpus", side_effect=RuntimeError("boom")):
            _pin_default_affinity(config, ["cycles"])
        self.assertEqual(config["thread"], [[{}, {}]])

    def test_pin_default_affinity_without_a_thread_config(self):
        from perf.bench import _pin_default_affinity

        config = {}
        with patch.object(core, "allowed_cpus", return_value=[1]):
            _pin_default_affinity(config, ["cycles"])
        self.assertEqual(config, {})

    def test_pin_default_affinity_leaves_an_unreachable_pmu_alone(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{"priority": "normal"}]]}
        with patch.object(core, "allowed_cpus", return_value=None):
            _pin_default_affinity(config, [["topdown-retiring"], ["cycles"]])
        self.assertEqual(config["thread"], [[{"priority": "normal"}]])

    def test_pin_default_affinity_reads_the_numa_node(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{"numa": 1}]]}
        with patch.object(core, "allowed_cpus", return_value=[9]) as fallback:
            _pin_default_affinity(config, ["topdown-*"])
        self.assertEqual(fallback.call_args.args[1], 1)
        self.assertEqual(config["thread"], [[{"numa": 1, "affinity": 9}]])

    def test_pin_default_affinity_passes_the_leader_along(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{}]]}
        with (
            patch.object(core, "with_leader", return_value=["p/slots/", "e"]),
            patch.object(core, "allowed_cpus", return_value=[1]) as fallback,
        ):
            _pin_default_affinity(config, [["topdown-retiring"]])
        fallback.assert_called_once_with([["p/slots/", "e"]], None)

    def test_numa_measures_and_is_recorded(self):
        from perf.core import _numa_nodes

        nodes = _numa_nodes()
        if not nodes:
            self.skipTest("no numa nodes under sysfs")
        df = benchmark(
            config={
                "iterations": 64,
                "samples": 2,
                "thread": [{"numa": nodes[0]}],
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="numa",
        )
        self.assertTrue(len(df) > 0)
        self.assertEqual(df.attrs["config"]["thread"][0][0]["numa"], nodes[0])

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bad_numa_fails_fast(self, mock_bench):
        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 10, "thread": {"numa": "bogus"}},
                mode=["latency"],
                target={"asm": "nop"},
                name="t",
            )
        mock_bench.assert_not_called()

    def test_normalize_groups(self):
        from perf.bench import _normalize_groups

        self.assertEqual(_normalize_groups(None), [["duration_time"]])
        self.assertEqual(_normalize_groups(""), [["duration_time"]])
        self.assertEqual(_normalize_groups("cycles"), [["cycles"]])
        self.assertEqual(_normalize_groups([]), [["duration_time"]])
        self.assertEqual(_normalize_groups(["cycles", "inst"]), [["cycles", "inst"]])
        self.assertEqual(_normalize_groups([["a", "b"], "c"]), [["a", "b"], ["c"]])
        self.assertEqual(_normalize_groups(7), [["7"]])

    def test_guards_restore_process_state(self):
        import os

        from perf.cfg import _affinity_guard, _priority_guard

        with _affinity_guard({}):
            pass
        with _priority_guard({}):
            pass
        before = sorted(os.sched_getaffinity(0))
        with _affinity_guard({"thread": {"affinity": [before[0]]}}):
            self.assertEqual([before[0]], sorted(os.sched_getaffinity(0)))
        self.assertEqual(sorted(os.sched_getaffinity(0)), before)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bad_affinity_fails_fast(self, mock_bench):
        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 10, "thread": {"affinity": ["bogus"]}},
                mode=["latency"],
                target={"asm": "nop"},
                name="t",
            )
        mock_bench.assert_not_called()

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bad_priority_fails_fast(self, mock_bench):
        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 10, "thread": {"priority": ["bogus"]}},
                mode=["latency"],
                target={"asm": "nop"},
                name="t",
            )
        mock_bench.assert_not_called()


def imm_of(value):
    from perf.arch import x86_64

    return x86_64.imm(value)


class TestExplicitMemSteering(unittest.TestCase):
    def test_explicit_mem_merged_without_models(self):
        import random

        from perf.cfg import _per_iter_data

        buf, meta = _per_iter_data(
            [],
            16,
            {"L1d": 100, "L2": 100, "L3": 100},
            True,
            random.Random(0),
            explicit={"regs": {}, "mem": {"0x400000": 100}},
        )
        self.assertIn(0x400000, meta["mem_addrs"])
        self.assertIn(0x400000, meta["val_col"])
        col = meta["val_col"][0x400000]
        self.assertTrue((buf[col, 1:17] == 100).all())

    def test_repeated_writes_keep_the_last_value(self):
        from perf.mca import _collapse_access

        accesses, index = [], {}
        for value in (1, 2, 3):
            _collapse_access(accesses, index, 0x1000, 4, value, last=True)
        self.assertEqual(accesses[0][2], 3)

    def test_initial_read_wins_over_a_later_write_for_priming(self):
        import random

        from perf.cfg import _per_iter_data

        addr = 0x41000000000
        models = [
            {
                "regs": {},
                "reads": [(addr, 8, 10)],
                "writes": [(addr, 8, 99)],
            }
        ]
        buf, meta = _per_iter_data(models, 4, None, True, random.Random(0))
        col = meta["val_col"][addr]
        self.assertTrue((buf[col, 1:5] == 10).all())

    def test_static_and_cache_only_addresses_are_not_written(self):
        import random

        from perf.cfg import _fill_evict_tables, _per_iter_data

        buf, meta = _per_iter_data(
            [],
            8,
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            explicit={"regs": {}, "mem": {"0x400000": 9}},
            extra_addrs={0x401000},
        )
        self.assertIn(0x400000, meta["write_addrs"])
        self.assertIn(0x401000, meta["mem_addrs"])
        self.assertNotIn(0x401000, meta["write_addrs"])
        self.assertTrue((buf[meta["val_col"][0x400000], 1:9] == 9).all())
        _fill_evict_tables(buf, meta, buf.ctypes.data)
        self.assertIn(0x401000, buf[meta["evict_table_col"]].tolist())
        self.assertEqual(
            buf[meta["write_table_col"]][: len(meta["write_addrs"])].tolist(),
            meta["write_addrs"],
        )

    def test_explicit_value_overrides_solver(self):
        import random

        from perf.cfg import _per_iter_data

        models = [{"regs": {}, "reads": [(0x400000, 8, 0)], "writes": []}]
        buf, meta = _per_iter_data(
            models,
            8,
            {"L1d": 100, "L2": 100, "L3": 100},
            True,
            random.Random(0),
            explicit={"regs": {}, "mem": {"0x400000": 100}},
        )
        col = meta["val_col"][0x400000]
        self.assertTrue((buf[col, 1:9] == 100).all())

    def test_steer_prime_split(self):
        import random

        import keystone

        from perf.bench import _arch_asm
        from perf.cfg import _fill_evict_tables, _per_iter_data

        models = [
            {
                "regs": {"rax": 0x4100001000},
                "reads": [(0x4100001000, 8, 0)],
                "writes": [],
            }
        ]
        buf, meta = _per_iter_data(
            models, 8, {"L1d": 100, "L2": 100, "L3": 100}, True, random.Random(0)
        )
        _fill_evict_tables(buf, meta, buf.ctypes.data)
        steer = _arch_asm("steer_asm", meta, buf.ctypes.data)
        prime = _arch_asm("prime_asm", meta, buf.ctypes.data)
        self.assertIn(".perfwrite:", steer)
        self.assertIn("mov rax, [r14 + r8*8]", prime)
        self.assertIn("mfence", steer)
        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        enc, _ = ks.asm("\n".join(p for p in (steer, prime) if p))
        self.assertTrue(enc)


class TestResultHash(unittest.TestCase):
    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bench_attrs_and_label(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            data={"rdi": 1},
            mode=["latency"],
            target={"asm": "nop"},
            name="t",
        )
        self.assertIn("config", result.attrs)
        self.assertIn("info", result.attrs)
        self.assertIn("id", result.attrs)
        self.assertIn("data", result.attrs)
        self.assertNotIn("hash", result.attrs)
        self.assertNotIn("config_hash", result.attrs)
        self.assertNotIn("data_hash", result.attrs)
        self.assertNotIn("data", result.attrs["config"])
        self.assertEqual(result.attrs["data"]["regs"]["rdi"], 1)
        from perf.bench import _id_hash

        binary = result.attrs["info"]["binary"]
        self.assertEqual(binary["name"], "t")
        self.assertIn("asm", binary)
        self.assertEqual(
            result.attrs["id"],
            _id_hash(result.attrs["data"], result.attrs["config"], binary),
        )
        self.assertIsNotNone(result.attrs["id"])
        file_label = result.index.get_level_values("file").tolist()
        self.assertTrue(all(pd.isna(f) for f in file_label))
        self.assertIsNone(result.attrs["file"])


class TestIdHash(unittest.TestCase):
    def test_empty_returns_none(self):
        from perf.bench import _id_hash

        self.assertIsNone(_id_hash(None, None))
        self.assertIsNone(_id_hash({}, {}))
        self.assertIsNone(_id_hash({"regs": {}, "mem": {}}, {}))
        self.assertIsNone(_id_hash(None, {}))
        self.assertIsNone(_id_hash(None, None, None))
        self.assertIsNone(_id_hash(None, None, {}))

    def test_stable_and_sensitive(self):
        from perf.bench import _id_hash

        d = {"regs": {"rdi": 1}, "mem": {}}
        c = {"samples": 10}
        self.assertEqual(_id_hash(d, c), _id_hash(d, c))
        self.assertEqual(len(_id_hash(d, c)), 8)
        self.assertNotEqual(
            _id_hash(d, c), _id_hash({"regs": {"rdi": 2}, "mem": {}}, c)
        )
        self.assertNotEqual(_id_hash(d, c), _id_hash(d, {"samples": 11}))

    def test_binary_sensitive(self):
        from perf.bench import _id_hash

        d = {"regs": {"rdi": 1}, "mem": {}}
        c = {"samples": 10}
        b1 = {"path": "a.out", "sha": "abc"}
        b2 = {"path": "b.out", "sha": "def"}
        self.assertEqual(_id_hash(d, c, b1), _id_hash(d, c, b1))
        self.assertEqual(len(_id_hash(d, c, b1)), 8)
        self.assertNotEqual(_id_hash(d, c, b1), _id_hash(d, c, b2))
        self.assertNotEqual(_id_hash(d, c, b1), _id_hash(d, c))
        self.assertIsNotNone(_id_hash(None, None, b1))

    def test_data_only_and_config_only(self):
        from perf.bench import _id_hash

        self.assertIsNotNone(_id_hash({"regs": {"rdi": 1}, "mem": {}}, {}))
        self.assertIsNotNone(_id_hash(None, {"samples": 10}))
        self.assertIsNotNone(_id_hash({"regs": {}, "mem": {}}, {"samples": 10}))

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bench_id_covers_data_and_config(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 1000,
                "seed": [42],
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            data={"rdi": 1},
            mode=["latency"],
            target={"asm": "nop"},
            name="t",
        )
        result2 = benchmark(
            config={
                "iterations": 1000,
                "seed": [42],
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            data={"rdi": 2},
            mode=["latency"],
            target={"asm": "nop"},
            name="t",
        )
        self.assertNotEqual(result.attrs["id"], result2.attrs["id"])
        result3 = benchmark(
            config={
                "iterations": 1000,
                "seed": [43],
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            data={"rdi": 1},
            mode=["latency"],
            target={"asm": "nop"},
            name="t",
        )
        self.assertNotEqual(result.attrs["id"], result3.attrs["id"])


class TestConfigPruning(unittest.TestCase):
    def _combos(self, **override):
        from perf.cfg import _concrete_configs

        spec = {
            "branch": ["predictable", "unpredictable"],
            "dcache": ["hot", "cold"],
            "dtlb": ["hot", "cold"],
            "icache": "hot",
            "itlb": "hot",
            "code": [{"align": 1}, {"align": 16}],
        }
        spec.update(override)
        _s, combos = _concrete_configs(spec)
        return combos

    def test_no_branch_no_memory_keeps_one_of_each(self):
        from perf.mca import _prune_combos

        combos = self._combos()
        pruned = _prune_combos(combos, {"branches": False, "memory": False})
        self.assertEqual(len(pruned), 2)
        self.assertEqual({c["branch"] for c in pruned}, {"predictable"})
        self.assertEqual({c["dcache"] for c in pruned}, {"hot"})
        self.assertEqual({c["dtlb"] for c in pruned}, {"hot"})
        self.assertEqual({c["code"]["align"] for c in pruned}, {1, 16})

    def test_memory_keeps_the_cache_sweep(self):
        from perf.mca import _prune_combos

        pruned = _prune_combos(self._combos(), {"branches": False, "memory": True})
        self.assertEqual({c["dcache"] for c in pruned}, {"hot", "cold"})
        self.assertEqual({c["branch"] for c in pruned}, {"predictable"})

    def test_branches_keep_the_branch_sweep(self):
        from perf.mca import _prune_combos

        pruned = _prune_combos(self._combos(), {"branches": True, "memory": False})
        self.assertEqual(
            {c["branch"] for c in pruned}, {"predictable", "unpredictable"}
        )
        self.assertEqual({c["dcache"] for c in pruned}, {"hot"})

    def test_nothing_pruned_when_no_axis_varies(self):
        from perf.mca import _prune_combos

        combos = self._combos(branch="predictable", dcache="hot", dtlb="hot")
        self.assertEqual(
            _prune_combos(combos, {"branches": False, "memory": False}), combos
        )

    def test_features_of_a_plain_snippet(self):
        from perf.mca import _target_features

        combos = self._combos()
        features = _target_features(
            None, None, "mov eax, 42;", combos[0], {"regs": {}, "mem": {}}
        )
        self.assertEqual(features, {"branches": False, "memory": False})

    def test_features_of_a_load(self):
        from perf.mca import _target_features

        combos = self._combos()
        features = _target_features(
            None, None, "mov rax, [rdi]", combos[0], {"regs": {}, "mem": {}}
        )
        self.assertEqual(features, {"branches": False, "memory": True})

    def test_features_of_a_store(self):
        from perf.mca import _target_features

        combos = self._combos()
        features = _target_features(
            None, None, "mov [rdi], rax;", combos[0], {"regs": {}, "mem": {}}
        )
        self.assertTrue(features["memory"])

    def test_features_of_a_conditional_branch(self):
        from perf.mca import _target_features

        combos = self._combos()
        features = _target_features(
            None,
            None,
            "cmp rax, 5\njl .L1\nnop\n.L1:",
            combos[0],
            {"regs": {}, "mem": {}},
        )
        self.assertTrue(features["branches"])
        self.assertFalse(features["memory"])

    def test_features_of_a_swept_branch(self):
        from perf.mca import _target_features

        combos = self._combos()
        features = _target_features(
            None,
            None,
            "cmp rdi, 5\njl .L1\nnop\n.L1:",
            combos[0],
            {"regs": {"rdi": [1, 9]}, "mem": {}},
        )
        self.assertTrue(features["branches"])

    def test_explicit_memory_data_counts_as_memory(self):
        from perf.mca import _target_features

        combos = self._combos()
        features = _target_features(
            None, None, "mov eax, 42;", combos[0], {"regs": {}, "mem": {"0x1000": 1}}
        )
        self.assertTrue(features["memory"])

    def test_explore_records_branches_and_memory(self):
        from perf.arch import arch as get_arch
        from perf.mca import explore_asm

        arch = get_arch()
        plain = explore_asm("mov eax, 42;", arch=arch)
        self.assertEqual([s["branches"] for s in plain], [0])
        self.assertEqual([s["reads"] for s in plain], [[]])
        self.assertEqual([s["writes"] for s in plain], [[]])

        branch = explore_asm("cmp rax, 5\njl .L1\nnop\n.L1:", arch=arch)
        self.assertGreaterEqual(sum(s["branches"] for s in branch), 1)

        load = explore_asm("mov rax, [rdi]", arch=arch)
        self.assertEqual(len(load[0]["reads"]), 1)

        store = explore_asm("mov [rdi], rax", arch=arch)
        self.assertEqual(len(store[0]["writes"]), 1)

    def test_features_come_from_the_explore_result(self):
        from perf.mca import _branches_seen, _memory_seen

        self.assertTrue(_branches_seen([{"branches": 0}, {"branches": 1}]))
        self.assertTrue(_branches_seen([{"branches": 0}, {"branches": 0}]))
        self.assertFalse(_branches_seen([{"branches": 0}]))
        self.assertFalse(_branches_seen([]))
        self.assertTrue(_memory_seen([{"reads": [], "writes": [(1, 8, 0)]}]))
        self.assertTrue(_memory_seen([{"reads": [(1, 8, 0)], "writes": []}]))
        self.assertFalse(_memory_seen([{"reads": [], "writes": []}]))
        self.assertFalse(_memory_seen(None))

    def test_exploration_is_shared_across_combinations(self):
        from perf.mca import _EXPLORE_CACHE, _explore_cached

        _EXPLORE_CACHE.clear()
        calls = []

        def factory():
            calls.append(1)
            return []

        self.assertEqual(_explore_cached("k", factory), [])
        self.assertEqual(_explore_cached("k", factory), [])
        self.assertEqual(len(calls), 1)
        _EXPLORE_CACHE.clear()

    def test_function_exploration_ignores_the_stack_seed(self):
        from unittest.mock import patch

        mca_mod._EXPLORE_CACHE.clear()
        self.addCleanup(mca_mod._EXPLORE_CACHE.clear)
        funcs = {"f": (0x1000, 0x1010)}
        with patch.object(mca_mod, "explore", return_value=[]) as explored:
            for seed in (1, 2):
                mca_mod._explore_target_cached(
                    None,
                    "id",
                    "f",
                    0x1000,
                    0x1010,
                    None,
                    funcs,
                    None,
                    None,
                    0,
                    16,
                    "random",
                    seed,
                )
        self.assertEqual(explored.call_count, 1)

    def test_bench_prunes_combinations(self):
        from perf.mca import _prune_for_target

        combos = self._combos()
        pruned = _prune_for_target(
            None, None, "mov eax, 42;", combos, {"regs": {}, "mem": {}}
        )
        self.assertEqual(len(pruned), 2)

    def test_region_features_normalize_the_target_pair(self):
        from unittest.mock import patch

        with (
            patch.object(mca_mod, "resolve_exec", return_value="/tmp/a.out"),
            patch.object(mca_mod, "_bench_project", return_value=(None, None, {}, {})),
            patch.object(
                mca_mod, "resolve_targets", return_value=[("a..b", 0x1000, 0x1010)]
            ) as resolve,
            patch.object(mca_mod, "_explore_target_cached", return_value=[]),
        ):
            features = mca_mod._target_features(
                "/tmp/a.out",
                ("a", "b"),
                None,
                {"stack": {"size": 256, "align": 16, "mode": "random"}},
                {"regs": {}, "mem": {}},
            )
        self.assertEqual(features, {"branches": False, "memory": False})
        resolve.assert_called_once_with(None, "a..b", {})


class TestTargetApi(unittest.TestCase):
    def test_symbol_region_and_asm_are_not_accepted(self):
        from perf.bench import benchmark

        for kw in (
            {"symbol": ("a.out", "foo")},
            {"region": ("a.out", ("foo", "bar"))},
            {"asm": "nop"},
        ):
            with self.subTest(**kw), self.assertRaises(TypeError):
                benchmark(mode=["latency"], **kw)

    def test_file_and_func_are_not_accepted(self):
        from perf.bench import benchmark

        with self.assertRaises(TypeError):
            benchmark(file="a.out", mode=["latency"])
        with self.assertRaises(TypeError):
            benchmark(func=["a.out", "foo"], mode=["latency"])

    def test_symbol_target_is_accepted(self):
        from unittest.mock import patch

        import pandas as pd

        from perf.bench import benchmark

        with patch("perf.bench._bench_one") as mock_one:
            mock_one.side_effect = lambda **kw: pd.DataFrame([{"mode": kw["mode"]}])
            benchmark(
                target={"file": "a.out", "symbol": "foo"},
                mode=["latency"],
                config={"branch": "predictable"},
            )
            kwargs = mock_one.call_args.kwargs
            self.assertEqual((kwargs["file"], kwargs["target"]), ("a.out", "foo"))

    def test_asm_target_is_accepted(self):
        from unittest.mock import patch

        import pandas as pd

        from perf.bench import benchmark

        with patch("perf.bench._bench_one") as mock_one:
            mock_one.side_effect = lambda **kw: pd.DataFrame([{"mode": kw["mode"]}])
            benchmark(target={"asm": "nop"}, mode=["latency"])
            self.assertEqual(mock_one.call_args.kwargs["code"], "nop")

    def test_region_target_is_accepted(self):
        from unittest.mock import patch

        import pandas as pd

        from perf.bench import benchmark

        with patch("perf.bench._bench_one") as mock_one:
            mock_one.side_effect = lambda **kw: pd.DataFrame([{"mode": kw["mode"]}])
            benchmark(
                target={"file": "a.out", "region": ["foo", "bar"]},
                mode=["latency"],
                config={"branch": "predictable"},
            )
            kwargs = mock_one.call_args.kwargs
            self.assertEqual(kwargs["target"], ("foo", "bar"))

    def test_code_is_not_accepted(self):
        from perf.bench import benchmark

        with self.assertRaises(TypeError):
            benchmark(code="nop", mode=["latency"])


class TestNormalizeTarget(unittest.TestCase):
    def test_name(self):
        from perf.cfg import _normalize_target as norm

        self.assertEqual(norm("foo"), "foo")
        self.assertIsNone(norm(None))
        self.assertIsNone(norm("  "))

    def test_pair_becomes_region(self):
        from perf.cfg import _normalize_target as norm

        self.assertEqual(norm(("hot_begin", "hot_end")), "hot_begin..hot_end")
        self.assertEqual(norm(["a", "b"]), "a..b")
        self.assertEqual(norm((" a ", " b ")), "a..b")

    def test_region_string_rejected(self):
        from perf.cfg import _normalize_target as norm

        with self.assertRaises(ValueError):
            norm("hot_begin..hot_end")

    def test_bad_pairs_rejected(self):
        from perf.cfg import _normalize_target as norm

        for bad in (("only_one",), ("a", "b", "c"), ("", "b")):
            with self.subTest(target=bad), self.assertRaises(ValueError):
                norm(bad)


class TestProjectCacheKey(unittest.TestCase):
    def test_code_alignment_is_shared(self):
        from perf.mca import _project_cache_id

        self.assertEqual(
            _project_cache_id("a.out", {"code": {"align": 1}}),
            _project_cache_id("a.out", {"code": {"align": 16}}),
        )

    def test_stack_and_func_layouts_are_not_shared(self):
        from perf.mca import _project_cache_id

        self.assertNotEqual(
            _project_cache_id("a.out", {"stack": {"size": 0x1000}}),
            _project_cache_id("a.out", {"stack": {"size": 0x2000}}),
        )
        self.assertNotEqual(
            _project_cache_id("a.out", {"func": {"order": "as-is"}}),
            _project_cache_id("a.out", {"func": {"order": "random"}}),
        )

    def test_fixed_stack_seed_is_not_shared(self):
        from perf.mca import _project_cache_id

        self.assertNotEqual(
            _project_cache_id("a.out", {"seed": 1, "stack": {"mode": "fixed"}}),
            _project_cache_id("a.out", {"seed": 2, "stack": {"mode": "fixed"}}),
        )
        self.assertEqual(
            _project_cache_id("a.out", {"seed": 1, "stack": {"mode": "random"}}),
            _project_cache_id("a.out", {"seed": 2, "stack": {"mode": "random"}}),
        )

    def test_external_output_settings_are_not_shared(self):
        from perf.mca import _project_cache_id

        self.assertNotEqual(
            _project_cache_id(
                "a.out",
                {"external": {"stdout": False, "stderr": False}},
            ),
            _project_cache_id(
                "a.out",
                {"external": {"stdout": True, "stderr": True}},
            ),
        )


class TestBenchTargetSpec(unittest.TestCase):
    def test_analyze_rejects_file(self):
        from perf.mca import analyze

        with self.assertRaises(TypeError):
            analyze(file="a.out", target="foo")

    def test_analyze_rejects_code(self):
        from perf.mca import analyze

        with self.assertRaises(TypeError):
            analyze(code="mov eax, 42")

    def test_analyze_accepts_target_only(self):
        import inspect

        from perf.bench import benchmark
        from perf.mca import analyze

        for fn in (benchmark, analyze):
            sig = inspect.signature(fn)
            self.assertIn("target", sig.parameters)
            for old in ("symbol", "region", "asm", "code", "file", "func"):
                self.assertNotIn(old, sig.parameters)
        df = analyze(target={"asm": "nop"})
        self.assertEqual(len(df), 1)
        with self.assertRaises(ValueError):
            analyze(target={"file": "a.out", "symbol": "foo", "asm": "nop"})
        with self.assertRaises(TypeError):
            analyze()


class TestSeededRng(unittest.TestCase):
    def test_string_seed_is_deterministic(self):
        from perf.cfg import _make_rng

        a = _make_rng({"seed": "abc"})
        b = _make_rng({"seed": "abc"})
        self.assertEqual([a.random() for _ in range(5)], [b.random() for _ in range(5)])

    def test_string_seed_matches_seed_int_stream(self):
        from perf.cfg import _config_seed_int, _make_rng

        rng = _make_rng({"seed": "abc"})
        ref = random.Random(_config_seed_int({"seed": "abc"}))
        self.assertEqual(
            [rng.random() for _ in range(5)], [ref.random() for _ in range(5)]
        )


class TestModelPicks(unittest.TestCase):
    def test_predictable_without_overrides_cycles_in_order(self):
        from perf.cfg import _model_picks

        models = [
            {"regs": {"rdi": 1}, "reads": [], "writes": []},
            {"regs": {"rdi": 2}, "reads": [], "writes": []},
        ]
        picks = _model_picks(models, True, random.Random(0), None, 6)
        self.assertEqual(picks, [0, 1] * 3)

    def test_predictable_without_overrides_pins_first(self):
        from perf.cfg import _model_picks

        models = [
            {"regs": {"rdi": 1}, "reads": [], "writes": []},
            {"regs": {"rdi": 2}, "reads": [], "writes": []},
        ]
        picks = _model_picks(models, True, random.Random(0), None, 1)
        self.assertEqual(picks, [0])


class TestExplicitDistributionBranching(unittest.TestCase):
    def test_predictable_cycles_explicit_reg_list_in_exec_order(self):
        from perf.cfg import _per_iter_data

        models = [
            {"regs": {"rdi": 15}, "reads": [], "writes": []},
            {"regs": {"rdi": 3}, "reads": [], "writes": []},
            {"regs": {"rdi": 5}, "reads": [], "writes": []},
        ]
        buf, meta = _per_iter_data(
            models,
            9,
            {"L1d": 100, "L2": 100, "L3": 100},
            True,
            random.Random(1),
            explicit={"regs": {"rdi": [1, 3, 5]}},
        )
        col = meta["reg_col"]["rdi"]
        k = int(meta["k"])
        seq = [int(buf[col, k - 1 - i]) for i in range(9)]
        self.assertEqual(seq, [1, 3, 5, 1, 3, 5, 1, 3, 5])

    def test_unpredictable_samples_explicit_reg_list(self):
        from perf.cfg import _per_iter_data

        models = [
            {"regs": {"rdi": 15}, "reads": [], "writes": []},
            {"regs": {"rdi": 3}, "reads": [], "writes": []},
            {"regs": {"rdi": 5}, "reads": [], "writes": []},
        ]
        buf, meta = _per_iter_data(
            models,
            120,
            {"L1d": 100, "L2": 100, "L3": 100},
            False,
            random.Random(7),
            explicit={"regs": {"rdi": [1, 3, 5]}},
        )
        col = meta["reg_col"]["rdi"]
        k = int(meta["k"])
        vals = {int(buf[col, i]) for i in range(1, k)}
        self.assertEqual(vals, {1, 3, 5})

    def test_build_loop_asm_cycles_explicit_list(self):
        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        models = [
            {"regs": {"rdi": 15}, "reads": [], "writes": []},
            {"regs": {"rdi": 3}, "reads": [], "writes": []},
            {"regs": {"rdi": 5}, "reads": [], "writes": []},
        ]
        asm, meta, buf = _build_loop_asm(
            9,
            "latency",
            "call rax",
            "",
            "",
            models,
            ["duration_time"],
            [None],
            None,
            True,
            random.Random(1),
            arch,
            data={"regs": {"rdi": [1, 3, 5]}},
        )
        self.assertIn("mov rdi, [r14 + r8*8]", asm)
        self.assertNotIn("mov rdi, 0x1", asm)
        col = meta["reg_col"]["rdi"]
        k = int(meta["k"])
        seq = [int(buf[col, k - 1 - i]) for i in range(9)]
        self.assertEqual(seq, [1, 3, 5, 1, 3, 5, 1, 3, 5])

    def test_benchmark_keeps_explicit_values_for_unpredictable_branch(self):
        from perf.bench import benchmark

        df = benchmark(
            target={"asm": "mov eax, edi; ret"},
            mode=["latency"],
            event=["duration_time"],
            data={"rdi": [1, 10, 100]},
            config={
                "samples": 1,
                "iterations": 6,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
        )
        self.assertEqual(
            sorted(df["data.rdi"].astype(int).unique().tolist()), [1, 10, 100]
        )

    def test_build_loop_asm_keeps_reload_for_unmanaged_reg(self):
        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        models = [{"regs": {"rdi": 9}, "reads": [], "writes": []}]
        asm, meta, buf = _build_loop_asm(
            6,
            "latency",
            "call rax",
            "",
            "",
            models,
            ["duration_time"],
            [None],
            None,
            True,
            random.Random(1),
            arch,
            data={"regs": {"r8": 7}},
        )
        self.assertIn("mov r8, 0x7", asm)


class TestBenchGroupsComma(unittest.TestCase):
    def test_normalize_splits_comma_string(self):
        from perf.bench import _normalize_groups

        self.assertEqual(
            _normalize_groups("cycles,instructions"),
            [["cycles", "instructions"]],
        )

    def test_normalize_flat_splits_comma(self):
        from perf.bench import _normalize_groups

        self.assertEqual(
            _normalize_groups(["cycles,instructions"]),
            [["cycles", "instructions"]],
        )
        self.assertEqual(
            _normalize_groups(["cycles", "inst"]),
            [["cycles", "inst"]],
        )

    def test_normalize_nested_splits_comma(self):
        from perf.bench import _normalize_groups

        self.assertEqual(
            _normalize_groups([["cycles,instructions"]]),
            [["cycles", "instructions"]],
        )

    def test_names_splits_comma(self):
        from perf.cfg import _names

        self.assertEqual(_names("cycles,instructions"), ["cycles", "instructions"])
        self.assertEqual(_names(["a,b", "c"]), ["a", "b", "c"])
        self.assertEqual(_names(7), [7])


class TestToJson(unittest.TestCase):
    def _df(self, **cols):
        df = pd.DataFrame(
            {
                "samples": [10],
                "iterations": [1000],
                "operations": [1],
                "time": [0.5],
                "duration_time": [4.2],
                **cols,
            }
        )
        df.index = pd.MultiIndex.from_tuples(
            [("a.out", "foo", "latency")], names=["file", "name", "mode"]
        )
        df.attrs["config"] = {"samples": 10, "dcache": "hot"}
        df.attrs["info"] = {
            "cpu": {"freq": 3000000000.0, "arch": "x86_64"},
            "binary": {"name": "t"},
        }
        df.attrs["data"] = {"regs": {"rdi": 3}, "mem": {}}
        return df

    def test_to_json_envelope_structure(self):
        from perf.bench import to_json
        from perf.info import hostname

        payload = json.loads(to_json(self._df()))
        self.assertEqual(
            list(payload), ["file", "name", "id", "time", "info", "output"]
        )
        self.assertEqual(payload["file"], "a.out")
        self.assertTrue(payload["name"].startswith("foo-"))
        self.assertEqual(len(payload["id"]), 8)
        self.assertEqual(payload["name"], f"foo-{payload['id']}")
        self.assertEqual(
            payload["info"],
            {
                "cpu": {"freq": 3000000000.0, "arch": "x86_64"},
                "hostname": hostname(),
            },
        )
        self.assertEqual(len(payload["output"]), 1)
        row = payload["output"][0]
        self.assertEqual(row["mode"], "latency")
        self.assertEqual(row["samples"], 10)
        self.assertEqual(row["iterations"], 1000)
        self.assertNotIn("time", row)
        self.assertNotIn("file", row)
        self.assertNotIn("name", row)

    def test_to_json_keeps_config_and_data_columns(self):
        from perf.bench import to_json

        payload = json.loads(
            to_json(
                self._df(
                    **{
                        "config.dcache": ["hot"],
                        "data.rdi": [3],
                        "extra": [1],
                    }
                )
            )
        )
        row = payload["output"][0]
        self.assertEqual(row["config"], {"dcache": "hot"})
        self.assertEqual(row["data"], {"rdi": 3})
        self.assertIn("extra", row)

    def test_to_json_keeps_multiple_unique_meta_columns(self):
        from perf.bench import to_json

        df = self._df()
        extra = pd.DataFrame({"samples": [5]})
        extra.index = pd.MultiIndex.from_tuples(
            [("a.out", "foo", "throughput")], names=["file", "name", "mode"]
        )
        df = pd.concat([df, extra])
        payload = json.loads(to_json(df))
        self.assertEqual(len(payload["output"]), 2)
        self.assertEqual(
            sorted({r["mode"] for r in payload["output"]}),
            ["latency", "throughput"],
        )

    def test_to_json_indent_and_compact(self):
        from perf.bench import to_json

        text = to_json(self._df())
        json.loads(text)
        self.assertIn('\n  "', text)
        compact = to_json(self._df(), indent=None)
        self.assertNotIn("\n", compact)
        json.loads(compact)

    def test_to_json_compact_handles_non_string_columns(self):
        from perf.bench import to_json

        df = pd.DataFrame({0: [1.0], 1: [2.0]})
        payload = json.loads(to_json(df, indent=None))
        self.assertEqual(payload["output"], [{"0": 1.0, "1": 2.0}])

    def test_to_json_matches_id_hash(self):
        from perf.bench import _id_hash, to_json

        df = self._df()
        payload = json.loads(to_json(df))
        self.assertEqual(
            payload["id"],
            _id_hash(df.attrs["data"], df.attrs["config"], df.attrs["info"]["binary"]),
        )

    def test_to_json_handles_mixed_integer_and_string_config_keys(self):
        from perf.bench import _id_hash, to_json

        df = self._df()
        df.attrs["config"] = {
            "branch": {"prediction": "predictable", 0x401000: "unpredictable"}
        }
        payload = json.loads(to_json(df))
        self.assertEqual(len(payload["id"]), 8)
        self.assertEqual(
            payload["id"],
            _id_hash(df.attrs["data"], df.attrs["config"], df.attrs["info"]["binary"]),
        )

    def test_to_json_is_not_exported(self):
        import perf

        self.assertFalse(hasattr(perf, "to_json"))
        self.assertFalse(hasattr(perf, "to_object"))

    @patch("perf.bench._bench", return_value=pd.Series([10.0]))
    def test_benchmark_frame_has_to_json(self, _mock_bench):
        df = benchmark(
            target={"asm": "nop"},
            name="t",
            mode=["latency"],
            config={"iterations": 64, "samples": 1},
        )
        payload = json.loads(df.to_json())
        self.assertTrue(payload["output"])
        self.assertIn("duration_time", payload["output"][0])
        rows = json.loads(df.to_json(orient="records"))
        self.assertEqual(len(rows), len(payload["output"]))


class TestOverheadFallback(unittest.TestCase):
    def test_all_nan_falls_back_to_raw_latency(self):
        import warnings

        import numpy as np

        from perf.bench import _fallback_all_nan_to_raw

        diffs = np.full((4, 8, 1), np.nan)
        raw_all = np.full((4, 8, 1), 500.0)
        result = np.full((8, 1), np.nan)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fallback_all_nan_to_raw(
                diffs, raw_all, result, np.array([600.0]), ["duration_time"]
            )
        self.assertEqual(float(result[0, 0]), 0.0)
        self.assertTrue(bool((result == 0.0).all()))
        self.assertEqual(float(diffs[0, 0, 0]), 0.0)
        self.assertEqual(len(caught), 1)

    def test_all_nan_falls_back_to_raw_throughput(self):
        import warnings

        import numpy as np

        from perf.bench import _fallback_all_nan_to_raw

        diffs = np.full((4, 1), np.nan)
        raw_all = np.full((4, 1), 1234.0)
        result = np.array([np.nan])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fallback_all_nan_to_raw(
                diffs, raw_all, result, np.array([2000.0]), ["duration_time"]
            )
        self.assertEqual(float(result[0]), 0.0)
        self.assertEqual(float(diffs[0, 0]), 0.0)
        self.assertEqual(len(caught), 1)

    def test_partial_nan_keeps_finite_result(self):
        import warnings

        import numpy as np

        from perf.bench import _fallback_all_nan_to_raw

        diffs = np.array([[[1.0], [np.nan]]])
        raw_all = np.array([[[10.0], [20.0]]])
        result = np.array([[1.0], [np.nan]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fallback_all_nan_to_raw(
                diffs, raw_all, result, np.array([0.0]), ["duration_time"]
            )
        self.assertEqual(float(result[0, 0]), 1.0)
        self.assertEqual(len(caught), 0)

    def test_all_nan_with_no_finite_raw_stays_nan(self):
        import warnings

        import numpy as np

        from perf.bench import _fallback_all_nan_to_raw

        diffs = np.full((2, 4, 1), np.nan)
        raw_all = np.full((2, 4, 1), np.nan)
        result = np.full((4, 1), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _fallback_all_nan_to_raw(
                diffs, raw_all, result, np.array([10.0]), ["duration_time"]
            )
        self.assertTrue(bool(np.isnan(result[0, 0])))

    def test_raw_min_fallback_matches_raw_all_fallback(self):
        import warnings

        import numpy as np

        from perf.bench import _fallback_all_nan_to_raw_min

        diffs = np.full((4, 8, 1), np.nan)
        result = np.full((8, 1), np.nan)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fallback_all_nan_to_raw_min(
                diffs, np.array([500.0]), result, np.array([600.0]), ["duration_time"]
            )
        self.assertEqual(float(result[0, 0]), 0.0)
        self.assertEqual(float(diffs[0, 0, 0]), 0.0)
        self.assertEqual(len(caught), 1)

    def test_bench_measure_recalibrates_throughput_overhead(self):
        from unittest.mock import patch

        import numpy as np

        from perf.arch import arch as get_arch
        from perf.bench import _bench_measure

        overhead_runs = np.array(
            [[10.0], [30.0], [20.0], [5.0], [25.0], [15.0], [8.0], [28.0], [18.0]]
        )
        main_runs = np.array(
            [[100.0], [50.0], [75.0], [80.0], [40.0], [60.0], [90.0], [70.0], [55.0]]
        )
        with patch(
            "perf.bench._collect", side_effect=[overhead_runs, main_runs]
        ) as collect:
            df, iterations = _bench_measure(
                {"iterations": 500, "seed": 1},
                "throughput",
                "add eax, 1;",
                "",
                "",
                [],
                100.0,
                ["cycles"],
                None,
                get_arch(),
                "m",
                [],
                set(),
                None,
                False,
                lambda: None,
                lambda: None,
                "loop",
                [Mock(rdpmc_index=None)],
                3,
                3,
                [None],
                None,
                overhead_iterations=100,
                overhead_code="nop;",
            )
        self.assertEqual(iterations, 500)
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(collect.call_args_list[0].args[4], "nop;")
        self.assertEqual(
            float(np.asarray(collect.call_args_list[0].args[10]).reshape(-1)[0]), 0.0
        )
        self.assertEqual(
            float(np.asarray(collect.call_args_list[1].args[10]).reshape(-1)[0]), 5.0
        )
        self.assertEqual(collect.call_args_list[1].args[2], 9)
        self.assertEqual(df["cycles"].tolist(), [50.0, 40.0, 55.0])

    def test_bench_measure_propagates_refresh_failures(self):
        from unittest.mock import patch

        from perf.arch import arch as get_arch
        from perf.bench import _bench_measure

        with patch(
            "perf.core.refresh_indices", side_effect=RuntimeError("stale indices")
        ):
            with self.assertRaises(RuntimeError):
                _bench_measure(
                    {"iterations": 500, "seed": 1},
                    "throughput",
                    "mov eax, 42;",
                    "",
                    "",
                    [],
                    0.0,
                    ["duration_time"],
                    None,
                    get_arch(),
                    "m",
                    [],
                    set(),
                    None,
                    False,
                    lambda: None,
                    lambda: None,
                    "loop",
                    [],
                    3,
                    3,
                    [None],
                    None,
                )

    def test_bench_measure_propagates_overhead_calibration_failures(self):
        from unittest.mock import patch

        from perf.arch import arch as get_arch
        from perf.bench import _bench_measure

        with patch(
            "perf.bench._collect", side_effect=RuntimeError("calibration failed")
        ):
            with self.assertRaises(RuntimeError):
                _bench_measure(
                    {"iterations": 500, "seed": 1},
                    "throughput",
                    "mov eax, 42;",
                    "",
                    "",
                    [],
                    0.0,
                    ["duration_time"],
                    None,
                    get_arch(),
                    "m",
                    [],
                    set(),
                    None,
                    False,
                    lambda: None,
                    lambda: None,
                    "loop",
                    [],
                    3,
                    3,
                    [None],
                    None,
                    overhead_iterations=100,
                    overhead_code="nop;",
                )

    def test_bench_measure_prefers_finite_rows(self):
        import pandas as pd

        df = pd.DataFrame({"duration_time": [10.0, float("nan"), 30.0]})
        finite = df.dropna()
        self.assertFalse(finite.empty)
        sampled = finite.sample(n=min(1, len(finite)), random_state=0)
        self.assertTrue(sampled["duration_time"].notna().all())

    def test_bench_measure_throughput_reports_min_only(self):
        from unittest.mock import patch

        import numpy as np

        from perf.arch import arch as get_arch
        from perf.bench import _bench_measure

        diffs = np.array(
            [
                [100.0, 200.0],
                [50.0, 300.0],
                [75.0, 250.0],
                [40.0, 100.0],
                [60.0, 120.0],
                [80.0, 140.0],
                [90.0, 210.0],
                [70.0, 190.0],
                [95.0, 180.0],
            ]
        )
        with patch("perf.bench._collect", return_value=diffs):
            df, iterations = _bench_measure(
                {"iterations": 500, "seed": 1},
                "throughput",
                "mov eax, 42;",
                "",
                "",
                [],
                0.0,
                ["duration_time", "cycles"],
                None,
                get_arch(),
                "m",
                [],
                set(),
                None,
                False,
                lambda: None,
                lambda: None,
                "loop",
                [Mock(rdpmc_index=None)],
                3,
                3,
                [None, None],
                None,
            )
        self.assertEqual(len(df), 3)
        self.assertEqual(df["duration_time"].tolist(), [50.0, 40.0, 70.0])
        self.assertEqual(df["cycles"].tolist(), [200.0, 100.0, 180.0])
        self.assertEqual(iterations, 500)

    def test_bench_measure_throughput_all_nan_keeps_every_sample(self):
        from unittest.mock import patch

        import numpy as np

        from perf.arch import arch as get_arch
        from perf.bench import _bench_measure

        with patch("perf.bench._collect", return_value=np.full((3, 1), np.nan)):
            df, _ = _bench_measure(
                {"iterations": 500, "seed": 1},
                "throughput",
                "mov eax, 42;",
                "",
                "",
                [],
                0.0,
                ["duration_time"],
                None,
                get_arch(),
                "m",
                [],
                set(),
                None,
                False,
                lambda: None,
                lambda: None,
                "loop",
                [],
                3,
                3,
                [None],
                None,
            )
        self.assertEqual(len(df), 3)
        self.assertTrue(bool(df["duration_time"].isna().all()))

    def test_bench_measure_keeps_cache_levels_with_register_levels(self):
        from unittest.mock import patch

        import numpy as np

        from perf.arch import arch as get_arch
        from perf.bench import _bench_measure

        addr = 0x4200000000
        models = [{"regs": {"rdi": addr}, "reads": [(addr, 4, 1)], "writes": []}]
        config = {
            "iterations": 10,
            "seed": 1,
            "dcache": {"hit_rate": 100, "L1": {"rdi": "cold"}},
        }
        with (
            patch("perf.bench._models_cached", return_value=models),
            patch("perf.bench._runtable_models", side_effect=lambda m, *a, **k: m),
            patch("perf.bench._collect", return_value=np.array([[10.0]])) as collect,
        ):
            _bench_measure(
                config,
                "latency",
                "mov eax, [rdi];",
                "",
                "",
                [{}],
                0.0,
                ["duration_time"],
                None,
                get_arch(),
                "m",
                [],
                set(),
                None,
                False,
                lambda: None,
                lambda: None,
                "loop",
                [],
                1,
                1,
                [None],
                None,
            )
        self.assertEqual(collect.call_args.args[11], {"L1d": 100, "L2": 0, "L3": 0})


class TestThreads(unittest.TestCase):
    def _pinned(self):
        return {
            "branch": "predictable",
            "dcache": "hot",
            "icache": "hot",
            "dtlb": "hot",
            "itlb": "hot",
            "code": {"align": 16},
            "samples": 2,
            "iterations": 8,
        }

    def test_validate_examples(self):
        from perf.cfg import validate_spec

        ex1 = {"thread": [[{"numa": None, "affinity": None, "priority": "normal"}]]}
        ex2 = {
            "thread": [
                [{"numa": None, "affinity": None, "priority": "normal"}],
                [
                    {"numa": None, "affinity": 1, "priority": "normal"},
                    {"numa": None, "affinity": 2, "priority": "normal"},
                ],
            ]
        }
        validate_spec(ex1)
        validate_spec(ex2)
        with self.assertRaises(ValueError):
            validate_spec({"thread": []})
        with self.assertRaises(ValueError):
            validate_spec({"thread": [[{"bogus": 1}]]})
        for entry in (
            {"affinity": "bogus"},
            {"priority": "bogus"},
            {"numa": "bogus"},
        ):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                validate_spec({"thread": [[entry]]})

    def test_concrete_thread_alternatives(self):
        from perf.cfg import _concrete_configs, _thread_group

        cfg = dict(self._pinned())
        cfg["thread"] = [
            [{"numa": None, "affinity": None, "priority": "normal"}],
            [
                {"numa": None, "affinity": 1, "priority": "normal"},
                {"numa": None, "affinity": 2, "priority": "normal"},
            ],
        ]
        _, combos = _concrete_configs(cfg)
        counts = sorted(len(_thread_group(c)) for c in combos)
        self.assertEqual(counts, [1, 2])

    def test_records_threads_column(self):
        from perf.bench import _records

        recs = _records(
            "f",
            "n",
            "latency",
            pd.DataFrame([{"cycles": 10.0}]),
            ["cycles"],
            operations=1,
        )
        self.assertNotIn("threads", recs[0])
        self.assertEqual(recs[0]["operations"], 1)
        self.assertNotIn("iteration", recs[0])
        self.assertNotIn("samples", recs[0])

    def test_measure_group_threaded_combines(self):
        from perf.bench import _measure_group

        calls = []

        def fake_single(
            config,
            mode,
            setup,
            teardown,
            solutions,
            group,
            arch,
            map_name,
            tracker,
            data,
            overhead_code,
            main_code,
            extra=None,
            debug=False,
            models_tracker=None,
            fallback_tracker=None,
            **kw,
        ):
            calls.append(list(_thread_group_for(config)))
            import pandas as pd

            return pd.DataFrame({"cycles": [10.0, 20.0]}), 8

        import perf.bench as bench
        from perf.cfg import _thread_group as _thread_group_for

        orig = bench._measure_group_single
        bench._measure_group_single = fake_single
        try:
            cfg = dict(self._pinned())
            cfg["thread"] = [
                {"affinity": None, "priority": "normal"},
                {"affinity": None, "priority": "normal"},
            ]
            res, it = _measure_group(
                cfg,
                "latency",
                "",
                "",
                [],
                ["cycles"],
                None,
                "m",
                [],
                {"regs": {}, "mem": {}},
                "nop;",
                "nop;",
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(res), 4)
            self.assertEqual(it, 8)
        finally:
            bench._measure_group_single = orig

    def test_benchmark_reports_threads_and_per_thread_operations(self):
        from perf.bench import benchmark

        def fake_group(*args, **kwargs):
            df = pd.DataFrame({"duration_time": [1.0, 2.0]})
            df.attrs["_perf_operations"] = [10, 20]
            return df, 20

        cfg = dict(self._pinned())
        cfg["thread"] = [[{}, {}]]
        with (
            patch.object(bench, "_measure_group", side_effect=fake_group),
            patch.object(mca_mod, "_explore_asm_cached", return_value=[]),
        ):
            df = benchmark(
                target={"asm": "nop"},
                name="t",
                mode=["throughput"],
                event=["duration_time"],
                config=cfg,
            )
        rec = df.reset_index()
        self.assertEqual(rec["threads"].tolist(), [2, 2])
        self.assertEqual(rec["operations"].tolist(), [160, 320])

    def test_benchmark_threads_column_single(self):
        df = benchmark(
            target={"asm": "nop"},
            name="t",
            mode=["latency"],
            config=dict(self._pinned()),
            event=["duration_time"],
        )
        rec = df.reset_index()
        self.assertNotIn("threads", rec.columns)
        thread = df.attrs["config"]["thread"]
        if thread and isinstance(thread[0], list):
            count = len(thread[0])
        else:
            count = len(thread)
        self.assertEqual(count, 1)


class TestPerIterationRows(unittest.TestCase):
    def test_records_carry_samples_and_iteration(self):
        import pandas as pd

        from perf.bench import _records

        result = pd.DataFrame({"duration_time": [1.0, 2.0]})
        slots = [{"data.rdi": 7}, {"data.rdi": 8}]
        recs = _records(
            "f",
            "n",
            "latency",
            result,
            ["duration_time"],
            slots,
        )
        self.assertNotIn("samples", recs[0])
        self.assertNotIn("iteration", recs[0])
        self.assertNotIn("threads", recs[0])
        self.assertEqual([r["data.rdi"] for r in recs], [7, 8])
        self.assertEqual([r["duration_time"] for r in recs], [1.0, 2.0])

    def test_records_without_slots_still_carry_iteration(self):
        import pandas as pd

        from perf.bench import _records

        result = pd.DataFrame({"duration_time": [1.0, 2.0]})
        recs = _records("f", "n", "throughput", result, ["duration_time"])
        self.assertNotIn("samples", recs[0])
        self.assertNotIn("iteration", recs[0])
        self.assertNotIn("threads", recs[0])

    def test_records_carry_the_thread_count_only_when_concurrent(self):
        import pandas as pd

        from perf.bench import _records

        result = pd.DataFrame({"duration_time": [1.0]})
        recs = _records("f", "n", "throughput", result, ["duration_time"], threads=3)
        self.assertEqual(recs[0]["threads"], 3)
        recs = _records("f", "n", "throughput", result, ["duration_time"], threads=1)
        self.assertNotIn("threads", recs[0])

    def test_records_use_the_operations_of_each_thread(self):
        import pandas as pd

        from perf.bench import _records

        result = pd.DataFrame({"duration_time": [10.0, 20.0]})
        result.attrs["_perf_operations"] = [100, 200]
        recs = _records(
            "f",
            "n",
            "throughput",
            result,
            ["duration_time"],
            operations=999,
            threads=2,
        )
        self.assertEqual([r["operations"] for r in recs], [100, 200])
        self.assertEqual([r["threads"] for r in recs], [2, 2])

    def test_to_json_carries_iteration(self):
        import json

        import pandas as pd

        from perf.bench import to_json

        df = pd.DataFrame(
            {
                "mode": ["latency", "latency"],
                "iterations": [8, 8],
                "operations": [1, 1],
                "threads": [1, 1],
                "duration_time": [5.0, 6.0],
                "data.rdi": [1, 2],
            }
        )
        df.attrs["data"] = {"regs": {}, "mem": {}}
        df.attrs["config"] = {"samples": 2}
        df.attrs["info"] = {"cpu": {"arch": "x86_64"}}
        env = json.loads(to_json(df))
        self.assertEqual(len(env["output"]), 2)
        self.assertEqual(
            [r["data"]["rdi"] for r in env["output"]],
            [1, 2],
        )

    @patch("perf.bench._bench", return_value=pd.Series([10.0, 20.0]))
    def test_benchmark_rows_carry_iteration(self, _mock_bench):
        df = benchmark(
            config={
                "iterations": 10,
                "samples": 2,
                "branch": "predictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="t",
        )
        rec = df.reset_index()
        self.assertNotIn("iteration", rec.columns)
        self.assertNotIn("samples", rec.columns)
        self.assertEqual(len(rec), 2)

    def test_benchmark_dumps_all_iterations(self):
        df = benchmark(
            target={"asm": "mov eax, 42"},
            mode=["latency"],
            event=["duration_time"],
            data={"rdi": [1, 2]},
            config={
                "samples": 2,
                "iterations": 8,
                "branch": "predictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
        )
        rec = df.reset_index()
        self.assertEqual(len(rec), 8)
        self.assertNotIn("iteration", rec.columns)
        self.assertNotIn("samples", rec.columns)
        self.assertEqual(rec["data.rdi"].tolist(), [1, 2, 1, 2, 1, 2, 1, 2])


class TestBenchProjectCache(unittest.TestCase):
    def setUp(self):
        self.bench = mca_mod
        mca_mod._PROJECTS.clear()
        self.addCleanup(mca_mod._PROJECTS.clear)

    def test_project_is_reused_for_the_same_configuration(self):
        from unittest.mock import MagicMock

        project = MagicMock()
        project.loader = MagicMock()
        obj = MagicMock()
        funcs = {"f": (0x1000, 0x1100)}
        with (
            patch.object(self.bench, "resolve_exec", return_value="/tmp/a.out"),
            patch("angr.Project", return_value=project) as mk,
            patch.object(self.bench, "Elf", return_value=obj),
            patch.object(self.bench, "functions", return_value=(funcs, {})),
        ):
            first = mca_mod._bench_project(
                "/tmp/a.out",
                {},
                {"size": 0x1000, "align": 16, "mode": "random"},
                {"align": 16, "order": "as-is"},
            )
            second = mca_mod._bench_project(
                "/tmp/a.out",
                {},
                {"size": 0x1000, "align": 16, "mode": "random"},
                {"align": 16, "order": "as-is"},
            )
        self.assertIs(first, second)
        self.assertEqual(mk.call_count, 1)
        obj.map_elf.assert_called_once_with()
        obj.setup_stack.assert_called_once_with(
            size=0x1000,
            align=16,
            mode="random",
            seed=None,
        )

    def test_different_file_misses_the_cache(self):
        from unittest.mock import MagicMock

        project = MagicMock()
        project.loader = MagicMock()
        obj = MagicMock()
        with (
            patch.object(self.bench, "resolve_exec", side_effect=lambda p: p),
            patch("angr.Project", return_value=project) as mk,
            patch.object(self.bench, "Elf", return_value=obj),
            patch.object(self.bench, "functions", return_value=({}, {})),
        ):
            mca_mod._bench_project(
                "/tmp/a.out",
                {},
                {"size": 0x1000, "align": 16},
                {"align": 16, "order": "as-is"},
            )
            mca_mod._bench_project(
                "/tmp/b.out",
                {},
                {"size": 0x1000, "align": 16},
                {"align": 16, "order": "as-is"},
            )
        self.assertEqual(mk.call_count, 2)

    def test_random_order_shuffles_layout(self):
        from unittest.mock import MagicMock

        project = MagicMock()
        project.loader = MagicMock()
        obj = MagicMock()
        with (
            patch.object(self.bench, "resolve_exec", return_value="/tmp/a.out"),
            patch("angr.Project", return_value=project),
            patch.object(self.bench, "Elf", return_value=obj),
            patch.object(self.bench, "functions", return_value=({}, {})),
        ):
            mca_mod._bench_project(
                "/tmp/a.out",
                {"seed": 9},
                {"size": 0x1000, "align": 16},
                {"align": 32, "order": "random"},
            )
        obj.randomize_layout.assert_called_once_with({}, seed=9, align=32)

    def test_as_is_order_keeps_layout(self):
        from unittest.mock import MagicMock

        project = MagicMock()
        project.loader = MagicMock()
        obj = MagicMock()
        with (
            patch.object(self.bench, "resolve_exec", return_value="/tmp/a.out"),
            patch("angr.Project", return_value=project),
            patch.object(self.bench, "Elf", return_value=obj),
            patch.object(self.bench, "functions", return_value=({}, {})),
        ):
            mca_mod._bench_project(
                "/tmp/a.out",
                {},
                {"size": 0x1000, "align": 16},
                {"align": 16, "order": "as-is"},
            )
        obj.randomize_layout.assert_not_called()

    def test_initializer_failure_is_contained(self):
        from unittest.mock import MagicMock

        project = MagicMock()
        project.loader = MagicMock()
        obj = MagicMock()
        obj.run_initializers.side_effect = RuntimeError("boom")
        with (
            patch.object(self.bench, "resolve_exec", return_value="/tmp/a.out"),
            patch("angr.Project", return_value=project),
            patch.object(self.bench, "Elf", return_value=obj),
            patch.object(self.bench, "functions", return_value=({}, {})),
        ):
            out = mca_mod._bench_project(
                "/tmp/a.out",
                {},
                {"size": 0x1000, "align": 16},
                {"align": 16, "order": "as-is"},
            )
        self.assertIs(out[1], obj)


class TestMeasureGroupThreadSafety(unittest.TestCase):
    def _pinned(self):
        return {
            "seed": 0,
            "thread": [[{"numa": None, "affinity": None, "priority": "normal"}]],
            "branch": "predictable",
            "dcache": "hot",
            "dtlb": "hot",
            "icache": "hot",
            "itlb": "hot",
            "code": [{"align": 1}],
            "stack": [{"size": 2097152, "align": 16}],
            "func": [{"align": 16, "order": "as-is"}],
        }

    def _install_fake_single(self, behavior):
        import perf.bench as bench

        orig = bench._measure_group_single
        bench._measure_group_single = behavior
        self.addCleanup(setattr, bench, "_measure_group_single", orig)
        return orig

    def test_threaded_combines_all_slots_entries(self):
        from perf.bench import _measure_group

        def fake_single(
            config,
            mode,
            setup,
            teardown,
            solutions,
            group,
            arch,
            map_name,
            tracker,
            data,
            overhead_code,
            main_code,
            extra=None,
            debug=False,
            models_tracker=None,
            fallback_tracker=None,
            **kw,
        ):
            slots = kw.get("slots_out")
            if slots is not None:
                slots.append({"ev": [1]})
                slots.append({"ev": [2]})
            return pd.DataFrame({"cycles": [float(len(slots or []))]}), 4

        self._install_fake_single(fake_single)
        cfg = dict(self._pinned())
        cfg["thread"] = [
            {"affinity": None, "priority": "normal"},
            {"affinity": None, "priority": "normal"},
        ]
        slots_out = []
        res, it = _measure_group(
            cfg,
            "latency",
            "",
            "",
            [],
            ["cycles"],
            None,
            "m",
            [],
            {"regs": {}, "mem": {}},
            "nop;",
            "nop;",
            slots_out=slots_out,
        )
        self.assertEqual(len(res), 2)
        self.assertEqual(it, 4)
        self.assertEqual(len(slots_out), 1)
        self.assertEqual(sorted(slots_out[0]["ev"]), [1, 1, 2, 2])

    def test_threaded_throughput_keeps_each_threads_operations(self):
        from perf.bench import _measure_group, _records

        def fake_single(
            config,
            mode,
            setup,
            teardown,
            solutions,
            group,
            arch,
            map_name,
            tracker,
            data,
            overhead_code,
            main_code,
            extra=None,
            debug=False,
            models_tracker=None,
            fallback_tracker=None,
            **kw,
        ):
            affinity = int(config["thread"][0]["affinity"])
            return pd.DataFrame({"cycles": [float(affinity)]}), affinity * 10

        self._install_fake_single(fake_single)
        cfg = dict(self._pinned())
        cfg["thread"] = [
            {"affinity": 1, "priority": "normal"},
            {"affinity": 2, "priority": "normal"},
        ]
        res, it = _measure_group(
            cfg,
            "throughput",
            "",
            "",
            [],
            ["cycles"],
            None,
            "m",
            [],
            {"regs": {}, "mem": {}},
            "nop;",
            "nop;",
        )
        self.assertEqual(it, 20)
        recs = _records(
            "f",
            "m",
            "throughput",
            res,
            ["cycles"],
            operations=it,
            threads=2,
        )
        self.assertEqual(sorted(r["operations"] for r in recs), [10, 20])
        self.assertEqual([r["threads"] for r in recs], [2, 2])
        self.assertEqual(sorted(r["cycles"] for r in recs), [1.0, 2.0])

    def test_threaded_error_propagates(self):
        from perf.bench import _measure_group

        def fake_single(*a, **kw):
            raise RuntimeError("boom")

        self._install_fake_single(fake_single)
        cfg = dict(self._pinned())
        cfg["thread"] = [
            {"affinity": None, "priority": "normal"},
            {"affinity": None, "priority": "normal"},
        ]
        with self.assertRaises(RuntimeError):
            _measure_group(
                cfg,
                "latency",
                "",
                "",
                [],
                ["cycles"],
                None,
                "m",
                [],
                {"regs": {}, "mem": {}},
                "nop;",
                "nop;",
            )

    def test_threaded_worker_error_propagates_within_barrier_timeout(self):
        import time as _time

        from perf.bench import _measure_group

        def broken_single(*a, **kw):
            raise RuntimeError("boom")

        self._install_fake_single(broken_single)
        cfg = dict(self._pinned())
        cfg["thread"] = [
            {"affinity": None, "priority": "normal"},
            {"affinity": None, "priority": "normal"},
        ]
        start = _time.monotonic()
        with self.assertRaises(RuntimeError):
            _measure_group(
                cfg,
                "latency",
                "",
                "",
                [],
                ["cycles"],
                None,
                "m",
                [],
                {"regs": {}, "mem": {}},
                "nop;",
                "nop;",
            )
        self.assertLess(_time.monotonic() - start, 5)

    def test_threaded_timeout_abandons_blocked_workers(self):
        import time as _time
        from unittest.mock import patch as _patch

        from perf.bench import _measure_group

        release = threading.Event()

        def hung_single(*a, **kw):
            release.wait(timeout=30)
            return pd.DataFrame({"cycles": [1.0]}), 1

        self._install_fake_single(hung_single)
        cfg = dict(self._pinned())
        cfg["thread"] = [
            {"affinity": None, "priority": "normal"},
            {"affinity": None, "priority": "normal"},
        ]
        released = {"ok": False}

        def teardown():
            release.set()
            released["ok"] = True

        self.addCleanup(teardown)
        start = _time.monotonic()
        with _patch.object(bench, "_MEASURE_JOIN_TIMEOUT", 1):
            with self.assertRaises(TimeoutError):
                _measure_group(
                    cfg,
                    "latency",
                    "",
                    "",
                    [],
                    ["cycles"],
                    None,
                    "m",
                    [],
                    {"regs": {}, "mem": {}},
                    "nop;",
                    "nop;",
                )
        self.assertLess(_time.monotonic() - start, 10)
        release.set()

    def test_threaded_timeout_keeps_the_worker_error_as_cause(self):
        from unittest.mock import patch as _patch

        from perf.bench import _measure_group

        release = threading.Event()
        entered = threading.Event()
        calls = {"n": 0}
        lock = threading.Lock()

        def mixed_single(*a, **kw):
            with lock:
                index = calls["n"]
                calls["n"] += 1
            if index == 0:
                entered.wait(timeout=5)
                raise RuntimeError("worker failed first")
            entered.set()
            release.wait(timeout=30)
            return pd.DataFrame({"cycles": [1.0]}), 1

        self._install_fake_single(mixed_single)
        cfg = dict(self._pinned())
        cfg["thread"] = [
            {"affinity": None, "priority": "normal"},
            {"affinity": None, "priority": "normal"},
        ]

        def teardown():
            release.set()

        self.addCleanup(teardown)
        with _patch.object(bench, "_MEASURE_JOIN_TIMEOUT", 0.2):
            with self.assertRaises(TimeoutError) as ctx:
                _measure_group(
                    cfg,
                    "latency",
                    "",
                    "",
                    [],
                    ["cycles"],
                    None,
                    "m",
                    [],
                    {"regs": {}, "mem": {}},
                    "nop;",
                    "nop;",
                )
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertEqual(str(ctx.exception.__cause__), "worker failed first")
        release.set()

    def test_threaded_data_isolated_per_thread(self):
        from perf.bench import _measure_group

        seen = []
        lock = threading.Lock()

        def fake_single(
            config,
            mode,
            setup,
            teardown,
            solutions,
            group,
            arch,
            map_name,
            tracker,
            data,
            overhead_code,
            main_code,
            extra=None,
            debug=False,
            models_tracker=None,
            fallback_tracker=None,
            **kw,
        ):
            data["regs"]["touched"] = threading.get_ident()
            with lock:
                seen.append(copy.deepcopy(data))
            return pd.DataFrame({"cycles": [1.0]}), 2

        self._install_fake_single(fake_single)
        cfg = dict(self._pinned())
        cfg["thread"] = [
            {"affinity": None, "priority": "normal"},
            {"affinity": None, "priority": "normal"},
        ]
        base_data = {"regs": {}, "mem": {}}
        _measure_group(
            cfg,
            "latency",
            "",
            "",
            [],
            ["cycles"],
            None,
            "m",
            [],
            base_data,
            "nop;",
            "nop;",
        )
        self.assertEqual(len(seen), 2)
        self.assertNotIn("touched", base_data["regs"])
        self.assertEqual(len({d["regs"]["touched"] for d in seen}), 2)

    def test_differing_priorities_run_sequentially(self):
        from perf.bench import _measure_group

        active = {"n": 0, "peak": 0}
        lock = threading.Lock()

        def fake_single(*a, **kw):
            with lock:
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])
            import time

            time.sleep(0.05)
            with lock:
                active["n"] -= 1
            return pd.DataFrame({"cycles": [1.0]}), 3

        self._install_fake_single(fake_single)
        cfg = dict(self._pinned())
        cfg["thread"] = [
            {"affinity": None, "priority": "low"},
            {"affinity": None, "priority": "high"},
        ]
        tracker = []
        res, it = _measure_group(
            cfg,
            "latency",
            "",
            "",
            [],
            ["cycles"],
            None,
            "m",
            tracker,
            {"regs": {}, "mem": {}},
            "nop;",
            "nop;",
        )
        self.assertEqual(len(res), 2)
        self.assertEqual(it, 3)
        self.assertEqual(tracker, [3])
        self.assertEqual(active["peak"], 1)

    def test_differing_priorities_combine_all_slots_entries(self):
        from perf.bench import _measure_group

        def fake_single(*a, **kw):
            slots = kw.get("slots_out")
            if slots is not None:
                slots.append({"ev": [len(slots) + 1]})
            return pd.DataFrame({"cycles": [1.0]}), 3

        self._install_fake_single(fake_single)
        cfg = dict(self._pinned())
        cfg["thread"] = [
            {"affinity": None, "priority": "low"},
            {"affinity": None, "priority": "high"},
        ]
        slots_out = []
        res, _it = _measure_group(
            cfg,
            "latency",
            "",
            "",
            [],
            ["cycles"],
            None,
            "m",
            [],
            {"regs": {}, "mem": {}},
            "nop;",
            "nop;",
            slots_out=slots_out,
        )
        self.assertEqual(len(res), 2)
        self.assertEqual(len(slots_out), 1)
        self.assertEqual(slots_out[0]["ev"], [1, 1])

    def test_same_priority_runs_concurrently(self):
        from perf.bench import _measure_group

        entered = threading.Barrier(2, timeout=10)
        left = {"ok": False}

        def fake_single(*a, **kw):
            try:
                entered.wait(timeout=10)
            except Exception:
                return pd.DataFrame({"cycles": [1.0]}), 1
            left["ok"] = True
            return pd.DataFrame({"cycles": [1.0]}), 1

        self._install_fake_single(fake_single)
        cfg = dict(self._pinned())
        cfg["thread"] = [
            {"affinity": None, "priority": "normal"},
            {"affinity": None, "priority": "normal"},
        ]
        _measure_group(
            cfg,
            "latency",
            "",
            "",
            [],
            ["cycles"],
            None,
            "m",
            [],
            {"regs": {}, "mem": {}},
            "nop;",
            "nop;",
        )
        self.assertTrue(left["ok"])

    def test_apply_seed_has_no_global_side_effects(self):
        import random

        import numpy as np

        from perf.cfg import _apply_seed

        random.seed(1234)
        np.random.seed(1234)
        before_py = random.random()
        random.seed(1234)
        np.random.seed(1234)
        _apply_seed({"seed": 999})
        after_py = random.random()
        self.assertEqual(before_py, after_py)

    def test_models_cache_is_content_addressed(self):
        from unittest.mock import patch

        from perf.mca import _MODEL_CACHE, _models_cached

        _MODEL_CACHE.clear()
        self.addCleanup(_MODEL_CACHE.clear)
        with patch("perf.mca.solve", return_value=[{"a": 1}]) as m1:
            out1 = _models_cached([{"x": 1}], None, None)
            self.assertEqual(m1.call_count, 1)
            out2 = _models_cached([{"x": 1}], None, None)
            self.assertEqual(m1.call_count, 1)
            self.assertEqual(out1, out2)
            out1.append({"mutated": True})
            out3 = _models_cached([{"x": 1}], None, None)
            self.assertNotIn({"mutated": True}, out3)

    def test_tag_config_columns_copies_list_values(self):

        df = pd.DataFrame({"cycles": [1.0, 2.0]})
        out = cfg_mod._tag_config_columns(df, {"code": [{"align": 1}]})
        self.assertIn("cycles", out.columns)


class TestRecordEnvelopeCompaction(unittest.TestCase):
    def _frame(self, rows=40, **columns):
        lists = [len(v) for v in columns.values() if isinstance(v, (list, tuple))]
        if lists:
            rows = lists[0]
        df = pd.DataFrame(
            {
                "samples": [10] * rows,
                "iterations": [1000] * rows,
                "duration_time": [float(i) for i in range(rows)],
                **columns,
            }
        )
        df.index = pd.MultiIndex.from_tuples(
            [("a.out", "foo", "latency")] * rows,
            names=["file", "name", "mode"],
        )
        df.attrs["config"] = {"samples": 10, "code": [{"align": 1}]}
        df.attrs["info"] = {"cpu": {"arch": "x86_64"}}
        df.attrs["data"] = {"regs": {"rdi": 3}, "mem": {}}
        return df

    def test_constant_columns_move_into_shared(self):
        from perf.bench import _record_envelope

        df = self._frame(
            **{
                "extra": [7] * 40,
                "config.samples": [10] * 40,
                "config.code.align": [1 if i % 2 else 16 for i in range(40)],
                "config.code.order": ["as-is"] * 40,
                "data.rdi": [3] * 40,
                "data.rcx": list(range(40)),
            }
        )
        env = _record_envelope(df)
        self.assertEqual(len(env["output"]), 40)
        shared = env["shared"]
        self.assertEqual(shared["extra"], 7)
        self.assertEqual(shared["config"]["samples"], 10)
        self.assertEqual(shared["config"]["code"]["order"], "as-is")
        self.assertEqual(shared["data"], {"rdi": 3})
        self.assertNotIn("duration_time", shared)
        first = env["output"][0]
        self.assertEqual(first["config"]["code"]["align"], 16)
        self.assertEqual(first["data"]["rcx"], 0)
        self.assertEqual(env["output"][1]["config"]["code"]["align"], 1)

    def test_varying_nested_columns_stay_with_the_rows(self):
        from perf.bench import _record_envelope

        df = self._frame(
            **{
                "config.samples": [10] * 40,
                "config.branch": [
                    "predictable" if i % 2 else "unpredictable" for i in range(40)
                ],
                "data.rdi": list(range(40)),
            }
        )
        env = _record_envelope(df)
        self.assertEqual(env["shared"]["config"]["samples"], 10)
        self.assertEqual(env["output"][0]["config"]["branch"], "unpredictable")
        self.assertEqual(env["output"][1]["config"]["branch"], "predictable")
        self.assertEqual(env["output"][39]["data"]["rdi"], 39)

    def test_a_frame_of_only_constants_has_empty_rows(self):
        from perf.bench import _record_envelope

        df = self._frame(duration_time=[1.0] * 40)
        env = _record_envelope(df)
        self.assertEqual(len(env["output"]), 40)
        self.assertTrue(all(row == {} for row in env["output"]))
        self.assertEqual(env["shared"]["duration_time"], 1.0)
        self.assertEqual(env["shared"]["mode"], "latency")

    def test_non_finite_values_become_null(self):
        from perf.bench import _record_envelope

        df = self._frame(
            samples=2,
            iterations=2,
            duration_time=[float("nan"), float("inf")],
            **{"data.rdi": [float("-inf"), 2], "extra": [1.0, 2.0]},
        )
        env = _record_envelope(df)
        self.assertIsNone(env["output"][0]["duration_time"])
        self.assertIsNone(env["output"][1]["duration_time"])
        self.assertNotIn("data", env["output"][0])
        self.assertEqual(env["output"][1]["data"]["rdi"], 2.0)

    def test_hostname_failure_leaves_it_empty(self):
        from perf.bench import _record_envelope

        df = self._frame(samples=1, iterations=1, duration_time=[1.0])
        with patch("perf.info.hostname", side_effect=OSError("no utsname")):
            env = _record_envelope(df)
        self.assertIsNone(env["info"]["hostname"])
        self.assertEqual(env["info"]["cpu"], {"arch": "x86_64"})

    def test_unhashable_values_do_not_disable_shared_constants(self):
        from perf.bench import _record_envelope

        df = self._frame(
            samples=40,
            iterations=40,
            **{
                "config.samples": [10] * 40,
                "config.branch": ["predictable"] * 40,
                "data.rdi": [15] * 39 + [[1, 2]],
            },
        )
        env = _record_envelope(df)
        shared = env["shared"]
        self.assertEqual(shared["config"]["samples"], 10)
        self.assertEqual(shared["config"]["branch"], "predictable")
        self.assertEqual(env["output"][0]["data"]["rdi"], 15)
        self.assertEqual(env["output"][-1]["data"]["rdi"], [1, 2])


class TestJsonScalars(unittest.TestCase):
    def test_json_safe_normalizes_nested_numpy_and_non_finite(self):
        import numpy as np

        from perf.bench import _json_safe

        value = {
            "a": np.float64(1.5),
            "b": [np.int64(2), float("nan"), float("inf")],
            "c": np.nan,
            "d": {"e": np.uint32(7)},
        }
        self.assertEqual(
            _json_safe(value),
            {"a": 1.5, "b": [2, None, None], "c": None, "d": {"e": 7}},
        )

    def test_json_safe_serializes_integers_outside_64_bits(self):
        from perf.bench import _json_safe

        value = {
            "wide": 1 << 100,
            "wide_negative": -(1 << 100),
            "limit": [(1 << 64) - 1, -(1 << 63)],
        }
        self.assertEqual(
            _json_safe(value),
            {
                "wide": hex(1 << 100),
                "wide_negative": hex(-(1 << 100)),
                "limit": [(1 << 64) - 1, -(1 << 63)],
            },
        )

    def test_json_scalar_handles_pandas_and_containers(self):
        import numpy as np
        import pandas as pd

        from perf.bench import _json_scalar

        self.assertIsNone(_json_scalar(None))
        self.assertEqual(_json_scalar("x"), "x")
        self.assertEqual(_json_scalar(3), 3)
        self.assertEqual(_json_scalar(2.5), 2.5)
        self.assertIsNone(_json_scalar(float("nan")))
        self.assertIsNone(_json_scalar(float("inf")))
        self.assertIsNone(_json_scalar(float("-inf")))
        self.assertEqual(_json_scalar(np.int64(4)), 4)
        self.assertEqual(_json_scalar(1 << 70), hex(1 << 70))
        self.assertIs(_json_scalar(True), True)
        self.assertIsNone(_json_scalar(pd.NA))
        self.assertEqual(_json_scalar({"a": np.nan}), {"a": None})


class TestRecordsEdges(unittest.TestCase):
    def test_list_results_become_records(self):
        from perf.bench import _records

        recs = _records(
            "f",
            "n",
            "latency",
            [1.0, 2.0],
            ["cycles"],
            slots=[{"data.rdi": 7}, {"data.rdi": 8}],
            divisor=2,
            operations=5,
            threads=2,
        )
        self.assertEqual([r["cycles"] for r in recs], [0.5, 1.0])
        self.assertEqual([r["data.rdi"] for r in recs], [7, 8])
        self.assertEqual({r["operations"] for r in recs}, {5})
        self.assertEqual({r["threads"] for r in recs}, {2})

    def test_list_results_skip_slots_with_the_wrong_length(self):
        from perf.bench import _records

        recs = _records(
            "f",
            "n",
            "latency",
            [1.0, 2.0],
            ["cycles"],
            slots={"data.rdi": [7]},
        )
        self.assertEqual(len(recs), 2)
        self.assertNotIn("data.rdi", recs[0])
        self.assertEqual([r["cycles"] for r in recs], [1.0, 2.0])

    def test_list_results_keep_unconvertible_values(self):
        from perf.bench import _records

        class Opaque:
            def __float__(self):
                raise TypeError("not a number")

        value = Opaque()
        recs = _records("f", "n", "latency", [1.0, value], ["cycles"])
        self.assertEqual(recs[0]["cycles"], 1.0)
        self.assertIs(recs[1]["cycles"], value)

    def test_non_iterable_results_yield_no_records(self):
        from perf.bench import _records

        class NotIterable:
            def __iter__(self):
                raise TypeError("not iterable")

        self.assertEqual(_records("f", "n", "latency", NotIterable(), ["cycles"]), [])

    def test_objects_with_a_broken_to_numpy_fall_back_to_iteration(self):
        from perf.bench import _records

        class ArrayLike:
            def to_numpy(self, *args, **kwargs):
                raise RuntimeError("cannot convert")

            def __iter__(self):
                return iter([1.0, 2.0])

        recs = _records("f", "n", "latency", ArrayLike(), ["cycles"])
        self.assertEqual([r["cycles"] for r in recs], [1.0, 2.0])

    def test_array_results_without_columns_fall_back_to_the_row(self):
        import numpy as np

        from perf.bench import _records

        class EmptyRows:
            def to_numpy(self, *args, **kwargs):
                return np.empty((2, 0))

        recs = _records(
            "f",
            "n",
            "latency",
            EmptyRows(),
            ["cycles"],
            slots=[{"data.rdi": 1}, {"data.rdi": 2}],
        )
        self.assertEqual([r["data.rdi"] for r in recs], [1, 2])
        self.assertNotIn("cycles", recs[0])

    def test_dataframe_group_fallback_and_metadata_edges(self):
        from perf.bench import _records

        df = pd.DataFrame({"cycles": [1.0, 2.0], "other": [3.0, 4.0]})
        df.attrs["_perf_operations"] = ["not an int"]
        recs = _records(
            "f",
            "n",
            "latency",
            df,
            ["duration_time"],
            operations=9,
            threads="many",
        )
        self.assertEqual([r["operations"] for r in recs], [9, 9])
        self.assertNotIn("threads", recs[0])
        self.assertNotIn("duration_time", recs[0])

    def test_row_operations_length_mismatch_is_ignored(self):
        from perf.bench import _records

        df = pd.DataFrame({"cycles": [1.0, 2.0]})
        df.attrs["_perf_operations"] = [100]
        recs = _records("f", "n", "latency", df, ["cycles"], operations=9)
        self.assertEqual([r["operations"] for r in recs], [9, 9])

    def test_short_dataframe_group_keeps_the_available_columns(self):
        from perf.bench import _records

        df = pd.DataFrame({"cycles": [1.0, 2.0]})
        recs = _records("f", "n", "latency", df, ["cycles", "missing"])
        self.assertEqual([r["cycles"] for r in recs], [1.0, 2.0])
        self.assertNotIn("missing", recs[0])


class TestCombineResults(unittest.TestCase):
    def test_empty_records_make_an_empty_frame(self):
        from perf.bench import _combine_results

        self.assertTrue(_combine_results([]).empty)

    def test_records_for_the_same_target_are_joined_by_position(self):
        from perf.bench import _combine_results

        first = [
            {"file": "f", "name": "n", "mode": "latency", "cycles": 1.0},
            {"file": "f", "name": "n", "mode": "latency", "cycles": 2.0},
        ]
        second = [{"file": "f", "name": "n", "mode": "latency", "insns": 3.0}]
        out = _combine_results([first, second])
        self.assertEqual(out.index.names, ["file", "name", "mode"])
        self.assertEqual(out["cycles"].tolist(), [1.0, 2.0])
        self.assertEqual(out["insns"].tolist()[:1], [3.0])
        self.assertTrue(pd.isna(out["insns"].iloc[1]))

    def test_different_targets_are_concatenated(self):
        from perf.bench import _combine_results

        first = [{"file": "f", "name": "n", "mode": "latency", "cycles": 1.0}]
        second = [{"file": "g", "name": "n", "mode": "latency", "insns": 2.0}]
        out = _combine_results([first, second])
        self.assertEqual(len(out), 2)
        self.assertEqual(out.index.get_level_values("file").tolist(), ["f", "g"])

    def test_shared_meta_columns_are_not_duplicated(self):
        from perf.bench import _combine_results

        first = [
            {
                "file": "f",
                "name": "n",
                "mode": "latency",
                "operations": 2,
                "cycles": 1.0,
            }
        ]
        second = [
            {
                "file": "f",
                "name": "n",
                "mode": "latency",
                "operations": 9,
                "insns": 2.0,
            }
        ]
        out = _combine_results([first, second])
        self.assertEqual(out["operations"].tolist(), [2])
        self.assertEqual(out["insns"].tolist(), [2.0])


class TestResolveTargetEdges(unittest.TestCase):
    def _asm_file(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "target.s")
        Path(path).write_text("foo:\n\tmov eax, 42\n\tret\n")
        return path

    def test_dict_forms(self):
        from perf.cfg import _resolve_target

        path = self._asm_file()
        self.assertEqual(
            _resolve_target(target={"file": path, "symbol": "foo"}),
            (
                None,
                None,
                "mov eax, 42",
                "foo",
                {"file": path, "symbol": "foo"},
            ),
        )
        self.assertEqual(
            _resolve_target(target={"asm": "nop"})[:4],
            (None, None, "nop", None),
        )
        self.assertEqual(
            _resolve_target(target={"file": "a.out", "symbol": "func"})[:4],
            ("a.out", "func", None, None),
        )
        self.assertEqual(
            _resolve_target(target={"file": "a.out", "region": ["a", "b"]})[:4],
            ("a.out", ("a", "b"), None, None),
        )
        with self.assertRaises(ValueError):
            _resolve_target(target={"file": "a.out", "region": "a"})
        with self.assertRaises(ValueError):
            _resolve_target(target={"file": "a.out", "symbol": ["a", "b"]})
        with self.assertRaises(ValueError):
            _resolve_target(target={"file": "a.out", "symbol": "a..b"})
        with self.assertRaises(ValueError):
            _resolve_target(target={"file": "", "symbol": "func"})
        with self.assertRaises(ValueError):
            _resolve_target(target={"asm": "nop", "file": "a.out"})
        with self.assertRaises(ValueError):
            _resolve_target(target={})

    def test_short_forms(self):
        from perf.cfg import _resolve_target

        path = self._asm_file()
        self.assertEqual(_resolve_target(target=f"{path}:foo")[2], "mov eax, 42")
        self.assertEqual(
            _resolve_target(target="a.out:func")[:4],
            ("a.out", "func", None, None),
        )
        self.assertEqual(
            _resolve_target(target="a.out:a..b")[:4],
            ("a.out", ("a", "b"), None, None),
        )
        self.assertEqual(
            _resolve_target(target="rdtsc")[:4],
            (None, None, "rdtsc", None),
        )
        self.assertEqual(
            _resolve_target(target=["a.out", ("a", "b")])[:4],
            ("a.out", ("a", "b"), None, None),
        )

    def test_errors(self):
        from perf.cfg import _resolve_target

        path = self._asm_file()
        with self.assertRaises(ValueError):
            _resolve_target(target=f"{path}:missing")
        with self.assertRaises(ValueError):
            _resolve_target(target={"file": "a.out", "unknown": "foo"})
        with self.assertRaises(ValueError):
            _resolve_target(target={"file": "a.out"})
        with self.assertRaises(ValueError):
            _resolve_target(target="  ")
        with self.assertRaises(TypeError):
            _resolve_target()

    def test_asm_source_name_defaults_to_the_label(self):
        from perf.cfg import _resolve_target

        path = self._asm_file()
        _, _, _, name, raw = _resolve_target(target=f"{path}:foo", name="given")
        self.assertEqual(name, "given")
        self.assertEqual(raw, f"{path}:foo")


class TestDefaultThreadCpus(unittest.TestCase):
    def test_free_cpus_are_assigned_around_the_reserved_ones(self):
        from perf.bench import _default_thread_cpus

        entries = [
            {"affinity": 0},
            {"affinity": None},
            {"affinity": None},
            "not an entry",
        ]
        with patch.object(core, "allowed_cpus", return_value=[0, 1, 2]):
            _default_thread_cpus(entries, ["cycles"])
        self.assertEqual(entries[1]["affinity"], 1)
        self.assertEqual(entries[2]["affinity"], 2)

    def test_numa_and_allowed_cpu_errors_leave_the_entry_alone(self):
        from perf.bench import _default_thread_cpus

        entries = [{"affinity": None}]
        with patch.object(cfg_mod, "_parse_numa", side_effect=ValueError("bad")):
            with patch.object(core, "allowed_cpus", return_value=[1]):
                _default_thread_cpus(entries, None)
        self.assertEqual(entries[0]["affinity"], 1)

        entries = [{"affinity": None}]
        with patch.object(core, "allowed_cpus", return_value=[]):
            _default_thread_cpus(entries, None)
        self.assertIsNone(entries[0]["affinity"])

    def test_thread_alternatives_accepts_all_documented_forms(self):
        from perf.bench import _thread_alternatives

        self.assertEqual(_thread_alternatives(None), [])
        self.assertEqual(_thread_alternatives({}), [])
        self.assertEqual(_thread_alternatives({"thread": "no"}), [])
        self.assertEqual(
            _thread_alternatives(
                {
                    "thread": [
                        "ignored",
                        {"affinity": 1},
                        [{"affinity": 2}, "ignored"],
                        [],
                    ]
                }
            ),
            [[{"affinity": 1}], [{"affinity": 2}]],
        )


class TestSlotColumns(unittest.TestCase):
    def _buffer(self):
        import random

        from perf.cfg import _per_iter_data

        models = [
            {
                "regs": {"rdi": 15, "rcx": 3},
                "reads": [(0x41000000000, 4, 42)],
                "writes": [],
            },
            {
                "regs": {"rdi": 3, "rcx": 15},
                "reads": [(0x41000000000, 4, 7)],
                "writes": [],
            },
        ]
        data = {
            "regs": {"rsi": [5, 6]},
            "mem": {"0x42000000000": [1, 2]},
        }
        buf, meta = _per_iter_data(
            models,
            20,
            None,
            True,
            random.Random(1),
            explicit=data,
        )
        return buf, meta, models, data

    def test_registers_and_memory_are_dumped_per_iteration(self):
        from perf.bench import _slot_columns

        buf, meta, models, data = self._buffer()
        slots = _slot_columns(buf, meta, models, data, indices=range(5))
        self.assertEqual(slots["data.rdi"], [15, 3, 15, 3, 15])
        self.assertEqual(slots["data.rcx"], [3, 15, 3, 15, 3])
        self.assertEqual(slots["data.rsi"], [5, 6, 5, 6, 5])
        self.assertEqual(slots["data.0x41000000000"], [42, 7, 42, 7, 42])
        self.assertEqual(slots["data.0x42000000000"], [1, 2, 1, 2, 1])

    def test_indices_are_filtered_to_the_available_range(self):
        from perf.bench import _slot_columns

        buf, meta, models, data = self._buffer()
        slots = _slot_columns(buf, meta, models, data, indices=[0, 1, 99, -1])
        self.assertEqual(slots["data.rdi"], [15, 3])
        self.assertIsNone(_slot_columns(buf, meta, models, data, indices=[99, 98]))

    def test_trailing_colons_are_accepted_for_memory_keys(self):
        from perf.bench import _slot_columns

        buf, meta, models, data = self._buffer()
        slots = _slot_columns(buf, meta, models, {"mem": {"0x41000000000:": None}})
        self.assertIn("data.0x41000000000", slots)
        self.assertNotIn("data.0x42000000000", slots)

    def test_malformed_meta_yields_no_columns(self):
        from perf.bench import _slot_columns

        buf, meta, models, data = self._buffer()
        self.assertIsNone(_slot_columns(None, meta, models, data))
        self.assertIsNone(_slot_columns(buf, {}, models, data))
        self.assertIsNone(_slot_columns(buf, {"k": 0}, models, data))
        self.assertIsNone(
            _slot_columns(buf, {"k": 5, "reg_col": {}, "val_col": {}}, models, data)
        )
        self.assertIsNone(
            _slot_columns(
                buf,
                {"k": "many", "reg_col": {}, "val_col": {}},
                models,
                data,
            )
        )
        self.assertIsNone(
            _slot_columns(
                buf,
                {"k": 5, "reg_col": [], "val_col": []},
                models,
                data,
            )
        )
        self.assertIsNone(
            _slot_columns(
                buf,
                {"k": 5, "reg_col": {"rdi": 99}, "val_col": {}},
                models,
                data,
            )
        )

    def test_heap_addresses_are_not_emitted_as_data_columns(self):
        from perf.bench import _slot_columns

        buf, meta, models, data = self._buffer()
        heap_data = {**data, "heap": {"0x41000000000": 4096}}
        slots = _slot_columns(buf, meta, models, heap_data, indices=range(5))
        self.assertIn("data.rdi", slots)
        self.assertNotIn("data.0x41000000000", slots)
        self.assertIn("data.0x42000000000", slots)

    def test_code_addresses_are_never_used_as_data(self):
        from perf.bench import _slot_columns

        buf, meta, models, data = self._buffer()
        with_l1i = dict(meta)
        with_l1i["l1i_addrs"] = list(meta["val_col"])
        slots = _slot_columns(
            buf,
            with_l1i,
            models,
            {"mem": {"0x42000000000": None}},
        )
        self.assertIn("data.rdi", slots)
        self.assertNotIn("data.0x42000000000", slots)


class TestFallbackNanEdges(unittest.TestCase):
    def test_three_dimensional_results_are_filled_per_event(self):
        import warnings

        import numpy as np

        from perf.bench import _fallback_all_nan_to_raw_min

        diffs = np.full((2, 3, 2), np.nan)
        result = np.full((3, 2), np.nan)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fallback_all_nan_to_raw_min(
                diffs,
                np.array([10.0, np.nan]),
                result,
                np.array([np.nan, 3.0]),
                ["cycles", "insns"],
                debug=True,
            )
        self.assertTrue(bool((result[:, 0] == 10.0).all()))
        self.assertTrue(bool(np.isnan(result[:, 1]).all()))
        self.assertEqual(float(diffs[0, 0, 0]), 10.0)
        self.assertEqual(len(caught), 1)

    def test_overhead_larger_than_raw_clamps_to_zero(self):
        import warnings

        import numpy as np

        from perf.bench import _fallback_all_nan_to_raw_min

        diffs = np.full((1, 2, 1), np.nan)
        result = np.full((2, 1), np.nan)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _fallback_all_nan_to_raw_min(
                diffs,
                np.array([1.0]),
                result,
                np.array([5.0]),
                ["cycles"],
            )
        self.assertTrue(bool((result == 0.0).all()))

    def test_flat_results_are_filled_by_event(self):
        import warnings

        import numpy as np

        from perf.bench import _fallback_all_nan_to_raw_min

        diffs = np.full((1, 3, 2), np.nan)
        result = np.array([np.nan, np.nan])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _fallback_all_nan_to_raw_min(
                diffs,
                np.array([3.0, 4.0]),
                result,
                np.array([1.0, 1.0]),
                [],
            )
        self.assertEqual(result.tolist(), [2.0, 3.0])
        self.assertEqual(len(caught), 2)

    def test_events_beyond_the_raw_minimum_are_left_alone(self):
        import warnings

        import numpy as np

        from perf.bench import _fallback_all_nan_to_raw_min

        diffs = np.full((1, 2), np.nan)
        result = np.array([np.nan, np.nan])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _fallback_all_nan_to_raw_min(
                diffs,
                np.array([5.0]),
                result,
                np.array([0.0, 0.0]),
                ["cycles", "insns"],
            )
        self.assertEqual(float(result[0]), 5.0)
        self.assertTrue(bool(np.isnan(result[1])))


class TestSetupAsmFor(unittest.TestCase):
    def test_call_setup_for_a_single_symbol(self):
        import sys

        import angr

        from perf.exec import Elf, _symtab_values
        from perf.mca import _setup_asm_for

        values, _ = _symtab_values(sys.executable)
        name = next(
            (sym for sym, value in values.items() if value and not sym.startswith("_")),
            None,
        )
        if name is None:
            self.skipTest("no suitable symbol in the running interpreter")
        project = angr.Project(
            sys.executable, auto_load_libs=False, load_debug_info=False
        )
        obj = Elf.__new__(Elf)
        obj.loader = project.loader
        obj.runtime_base = 0
        obj.is_pie = True
        obj.ifuncs = {}
        obj.moved_functions = {}
        from perf.arch import arch as get_arch

        arch = get_arch()
        asm = _setup_asm_for(project, obj, arch, name)
        self.assertIsNotNone(asm)
        self.assertIn("call", asm)
        combined = _setup_asm_for(project, obj, arch, [name, "__perf_missing__"])
        self.assertIn("__perf_missing__", combined)
        self.assertIsNone(_setup_asm_for(project, obj, arch, None))
        literal = _setup_asm_for(project, obj, arch, "__perf_missing__")
        self.assertEqual(literal, "__perf_missing__")


class TestBranchMeasurement(unittest.TestCase):
    def test_timing_events_cover_cycles_and_clocks(self):
        from perf.bench import _is_timing_event

        for event in (
            "cycles",
            "cpu-cycles",
            "bus-cycles",
            "ref-cycles",
            "stalled-cycles-frontend",
            "idle-cycles-backend",
            "cpu-clock",
            "task-clock",
            "duration_time",
        ):
            self.assertTrue(_is_timing_event(event), event)
        for event in (
            "instructions",
            "branches",
            "branch-misses",
            "cache-misses",
            "L1-dcache-load-misses",
        ):
            self.assertFalse(_is_timing_event(event), event)

    def test_count_events_average_over_runs_while_cycles_take_the_minimum(self):
        from perf.bench import _aggregate_over_runs

        values = np.array(
            [
                [[10.0, 0.0], [20.0, 1.0]],
                [[14.0, 2.0], [30.0, 3.0]],
            ]
        )
        result = _aggregate_over_runs(
            values, ["cycles", "branch-misses"], minimize=True
        )
        self.assertEqual(result[:, 0].tolist(), [10.0, 20.0])
        self.assertEqual(result[:, 1].tolist(), [1.0, 2.0])

        averaged = _aggregate_over_runs(
            values, ["cycles", "branch-misses"], minimize=False
        )
        self.assertEqual(averaged[:, 0].tolist(), [12.0, 25.0])
        self.assertEqual(averaged[:, 1].tolist(), [1.0, 2.0])

    def test_unpredictable_data_is_refreshed_between_runs(self):
        import random

        from perf.arch import arch as get_arch
        from perf.cfg import _per_iter_data, _refresh_per_iter_data

        arch = get_arch()
        branch_cfg = {"prediction": "unpredictable", "regs": {}, "mem": {}}
        rng = random.Random(0)
        buf, meta = _per_iter_data(
            [{"regs": {}, "reads": [], "writes": []}],
            32,
            None,
            False,
            rng,
            arch,
            explicit={"regs": {"rdi": [0, 1]}, "mem": {}},
            branch_cfg=branch_cfg,
        )
        column = meta["reg_col"]["rdi"]
        before = buf[column, 1:33].tolist()
        _refresh_per_iter_data(
            buf,
            meta,
            32,
            [{"regs": {}, "reads": [], "writes": []}],
            None,
            False,
            rng,
            arch,
            data={"regs": {"rdi": [0, 1]}, "mem": {}},
            branch_cfg=branch_cfg,
        )
        after = buf[column, 1:33].tolist()
        self.assertNotEqual(before, after)

    def test_predictable_data_is_not_refreshed(self):
        import random

        from perf.arch import arch as get_arch
        from perf.cfg import _per_iter_data, _refresh_per_iter_data

        arch = get_arch()
        branch_cfg = {"prediction": "predictable", "regs": {}, "mem": {}}
        rng = random.Random(0)
        buf, meta = _per_iter_data(
            [{"regs": {}, "reads": [], "writes": []}],
            16,
            None,
            True,
            rng,
            arch,
            explicit={"regs": {"rdi": [0, 1]}, "mem": {}},
            branch_cfg=branch_cfg,
        )
        column = meta["reg_col"]["rdi"]
        before = buf[column, 1:17].tolist()
        _refresh_per_iter_data(
            buf,
            meta,
            16,
            [{"regs": {}, "reads": [], "writes": []}],
            None,
            True,
            rng,
            arch,
            data={"regs": {"rdi": [0, 1]}, "mem": {}},
            branch_cfg=branch_cfg,
        )
        self.assertEqual(before, buf[column, 1:17].tolist())


class TestMeasurementCalibrationCost(unittest.TestCase):
    def test_latency_auto_calibration_uses_all_measurement_runs(self):
        from perf.bench import _measure_runs

        self.assertEqual(_measure_runs("latency", 100, 10), 10)
        self.assertEqual(_measure_runs("latency", 100, 2), 2)
        self.assertEqual(_measure_runs("throughput", 7, 10), 70)

    def test_calibrated_overhead_uses_the_median_run(self):
        from perf.bench import _calibrated_overhead

        overhead = _calibrated_overhead(
            np.array([[80.0, 30.0], [75.0, 10.0], [78.0, 20.0]])
        )
        self.assertEqual(overhead.tolist(), [78.0, 20.0])

        frame = pd.DataFrame({"cycles": [80.0, np.nan, 76.0]})
        self.assertEqual(_calibrated_overhead(frame).tolist(), [78.0])

    def test_throughput_overhead_uses_the_minimum_sample(self):
        from perf.bench import _calibrated_overhead

        overhead = _calibrated_overhead(
            np.array([[80.0, 30.0], [75.0, 10.0], [78.0, 20.0]]),
            mode="throughput",
        )
        self.assertEqual(overhead.tolist(), [75.0, 10.0])

    def test_throughput_groups_samples_before_taking_minima(self):
        from perf.bench import _throughput_sample_rows

        runs = np.array(
            [
                [100.0, 10.0],
                [50.0, 20.0],
                [75.0, 30.0],
                [80.0, 40.0],
                [40.0, 50.0],
                [60.0, 60.0],
            ]
        )
        samples = _throughput_sample_rows(runs, 2, 3, ["duration_time", "instructions"])
        self.assertEqual(samples.tolist(), [[50.0, 20.0], [40.0, 50.0]])

    def test_unroll_latency_subtracts_the_median_n_calibration(self):
        from perf.arch import arch as get_arch
        from perf.bench import _measure_group_single

        overhead_code = "imul eax, 42;" * 5
        main_code = "imul eax, 42;" * 10
        overhead = pd.DataFrame({"cycles": [80.0, 80.0, 80.0, 70.0]})

        def fake_bench(**kwargs):
            if kwargs["code"] == overhead_code:
                return overhead, 100
            baseline = kwargs["overhead"][0]
            return pd.DataFrame({"cycles": [95.0 - baseline] * 3}), 100

        with patch("perf.bench._bench", side_effect=fake_bench):
            result, iterations = _measure_group_single(
                {"backend": {"unroll": {"count": 5}}},
                "latency",
                "",
                "",
                [],
                ["cycles"],
                get_arch(),
                "imul eax, 42",
                [],
                None,
                overhead_code,
                main_code,
            )
        self.assertEqual(iterations, 100)
        self.assertEqual(result["cycles"].tolist(), [15.0, 15.0, 15.0])

    def test_latency_auto_calibration_measures_every_run(self):
        from perf.arch import arch as get_arch
        from perf.bench import _bench_measure

        probe = np.array([[10.0], [12.0]])
        main = np.array([[10.0], [11.0], [12.0], [9.0]])
        config = {
            "seed": 1,
            "backend": {"loop": {"probes": 2, "runs": 10, "target": 0.005}},
            "iterations": {"min": 2, "max": 100},
        }
        with patch("perf.bench._collect", side_effect=[probe, main]) as collect:
            _bench_measure(
                config,
                "latency",
                "mov eax, 42;",
                "",
                "",
                [],
                0.0,
                ["duration_time"],
                None,
                get_arch(),
                "m",
                [],
                set(),
                None,
                False,
                lambda: None,
                lambda: None,
                "loop",
                [],
                1,
                10,
                [None],
                None,
            )
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(collect.call_args_list[0].args[2], 2)
        self.assertEqual(collect.call_args_list[1].args[2], 10)
        self.assertNotIn("target_rel_se", collect.call_args_list[1].kwargs)

    def test_collect_measures_every_run(self):
        from perf.arch import arch as get_arch
        from perf.bench import _collect

        page = bytearray(64)
        with (
            patch("perf.bench._build_loop_asm", return_value=("nop;", {}, None)),
            patch("perf.bench._jit_page", return_value=page),
            patch("perf.bench._jit_release"),
            patch("perf.bench._write_jit_map"),
            patch("perf.bench._check_target_runnable"),
            patch("perf.bench._run_measured") as run,
        ):
            _collect(
                {},
                4,
                5,
                "latency",
                "nop;",
                "",
                "",
                [],
                ["duration_time"],
                [None],
                0,
                None,
                True,
                random.Random(0),
                get_arch(),
            )
        self.assertEqual(run.call_count, 5)

    def test_slots_reflect_the_final_per_iteration_buffer(self):
        from perf.arch import arch as get_arch
        from perf.bench import _collect

        page = bytearray(64)
        buf = np.zeros(2, dtype=np.uint64)
        state = {"run": 0}

        def refresh(*_args, **_kwargs):
            state["run"] += 1
            buf[0] = state["run"]

        with (
            patch("perf.bench._build_loop_asm", return_value=("nop;", {}, buf)),
            patch("perf.bench._refresh_per_iter_data", side_effect=refresh),
            patch(
                "perf.bench._slot_columns",
                side_effect=lambda b, *_a, **_k: {"data.rdi": [int(b[0])]},
            ),
            patch("perf.bench._jit_page", return_value=page),
            patch("perf.bench._jit_release"),
            patch("perf.bench._write_jit_map"),
            patch("perf.bench._check_target_runnable"),
            patch("perf.bench._run_measured"),
        ):
            slots = []
            _collect(
                {},
                4,
                3,
                "latency",
                "nop;",
                "",
                "",
                [],
                ["duration_time"],
                [None],
                0,
                None,
                True,
                random.Random(0),
                get_arch(),
                data={"regs": {"rdi": [1, 2]}},
                slots_out=slots,
            )
        self.assertEqual(slots, [{"data.rdi": [3]}])


class TestBenchNormalization(unittest.TestCase):
    def test_bucket_data_is_normalized_like_flat_data(self):
        from perf.cfg import _merge_data

        merged = _merge_data(
            None,
            {
                "regs": {"arg0": ["0x10"]},
                "mem": {"0x1000:": [1, 2], "4096": 3},
            },
        )
        self.assertEqual(merged["regs"], {"rdi": [16]})
        self.assertEqual(merged["mem"], {"0x1000": 3, "0x1001": 2})

    def test_records_never_map_an_absent_event_to_another_column(self):
        from perf.bench import _records

        frame = pd.DataFrame({"cycles": [1.0, 2.0], "other": [3.0, 4.0]})
        recs = _records("f", "n", "latency", frame, ["cycles", "missing"])
        self.assertEqual([row["cycles"] for row in recs], [1.0, 2.0])
        self.assertNotIn("missing", recs[0])

    def test_topdown_percentages_accept_numpy_rows(self):
        from perf.bench import _topdown_percentages

        events = ["topdown-a", "topdown-b"]
        out = _topdown_percentages(np.array([[25, 75]]), events, events)
        self.assertEqual(out, [[25.0, 75.0]])


class TestBenchHelpers(unittest.TestCase):
    def test_run_measured_propagates_raw_counter_enable_failures(self):
        from perf.bench import _run_measured

        counter = Mock()
        counter.enable.side_effect = OSError("cannot enable")
        fn = Mock()
        with self.assertRaises(OSError):
            _run_measured(
                fn,
                SimpleNamespace(iterations=1),
                raw_counters=[counter],
            )
        fn.assert_not_called()

    def test_raw_counters_are_read_per_fd(self):
        first = Mock(fd=10)
        second = Mock(fd=11)
        with patch.object(
            bench.os,
            "read",
            side_effect=[(7).to_bytes(8, "little"), OSError("x")],
        ):
            self.assertEqual(core._read_raw_counters([first, second]), [7, None])
        first.disable.assert_called_once()
        second.disable.assert_called_once()

    def test_fallback_fills_flat_and_stacked_results(self):
        diffs = np.full((1, 3, 2), np.nan)
        result = np.array([np.nan, np.nan])
        with patch.object(bench.warnings, "warn"):
            bench._fallback_all_nan_to_raw_min(
                diffs,
                np.array([3.0, 4.0]),
                result,
                np.array([1.0, 1.0]),
                [],
            )
        self.assertEqual(result.tolist(), [2.0, 3.0])

    def test_fallback_ignores_non_finite_raw_values(self):
        diffs = np.full((1, 2), np.nan)
        result = np.array([np.nan, np.nan])
        bench._fallback_all_nan_to_raw_min(
            diffs,
            np.array([np.nan, 5.0]),
            result,
            np.array([0.0, 0.0]),
            ["cycles", "cycles"],
        )
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 5.0)

    def test_records_apply_divisor_and_slot_mismatch(self):
        frame = pd.DataFrame({"topdown-retiring": [10.0], "cycles": [20.0]})
        recs = bench._records(
            "f",
            "n",
            "latency",
            frame,
            ["topdown-retiring", "cycles"],
            divisor=2,
        )
        self.assertEqual(recs[0]["topdown-retiring"], 10.0)
        self.assertEqual(recs[0]["cycles"], 10.0)
        recs = bench._records(
            "f",
            "n",
            "latency",
            pd.DataFrame({"cycles": [1.0, 2.0]}),
            ["cycles"],
            slots={"extra": [1]},
        )
        self.assertNotIn("extra", recs[0])

    def test_envelope_keeps_constant_containers(self):
        rows = 40
        frame = pd.DataFrame(
            {
                "cycles": [float(i) for i in range(rows)],
                "config.branch": ["predictable"] * rows,
                "config.extra": [{"a": 1}] * rows,
                "data.rdi": [[1, 2]] * rows,
            }
        )
        env = bench._record_envelope(frame)
        self.assertEqual(len(env["output"]), rows)
        self.assertEqual(env["shared"]["config"]["branch"], "predictable")
        self.assertIn("config", env["output"][0])

    def test_ret_addrs_at_uses_function_blocks(self):
        class Insn:
            address = 0x1004
            mnemonic = "ret"

        class Block:
            capstone = types.SimpleNamespace(insns=[Insn()])

        class Func:
            addr = 0x1000
            blocks = [Block()]

        project = types.SimpleNamespace(
            kb=types.SimpleNamespace(functions={1: Func()}),
            loader=types.SimpleNamespace(memory=Mock()),
        )
        arch = types.SimpleNamespace(is_return_mnemonic=lambda m: m == "ret")
        with patch.object(mca_mod, "_tail_jump_targets", return_value=[]):
            self.assertEqual(
                mca_mod._ret_addrs_at(project, arch, 0x1000, 0x1010, {}, set()),
                {0x1004},
            )

    def test_validate_spec_rejects_bad_shapes(self):
        for config in (
            {"data": {}},
            {"thread": []},
            {"code": []},
            {"iterations": {"min": 0}},
            {"backend": "bogus"},
            {"external": {"bogus": True}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                cfg_mod.validate_spec(config)

    def test_validate_spec_accepts_swept_branch_and_backend_leaves(self):
        for config in (
            {"branch": {"0x401000": ["predictable", "unpredictable"]}},
            {"branch": {"mem": {"0x401000": ["predictable", "unpredictable"]}}},
            {"backend": ["loop", "unroll"]},
            {"backend": {"loop": {"runs": ["3", "5"]}}},
            {"backend": {"unroll": {"count": [2, 3], "target": [0.01, 0.1]}}},
            {"external": {"lib": [True, False], "stdout": [False, True]}},
        ):
            with self.subTest(config=config):
                cfg_mod.validate_spec(config)

    def test_validate_spec_rejects_invalid_swept_values(self):
        for config in (
            {"branch": {"0x401000": ["predictable", "sometimes"]}},
            {"backend": ["loop", "turbo"]},
            {"backend": {"loop": {"runs": ["3", "nope"]}}},
            {"external": {"lib": [True, object()]}},
            {"func": {"order": "bogus"}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                cfg_mod.validate_spec(config)

    def test_register_level_helpers_skip_non_matching_keys(self):
        from perf.cfg import (
            _collect_mem_addrs,
            _reg_level_specs,
            _reg_level_values,
            _resolve_reg_levels,
        )

        self.assertEqual(
            _reg_level_specs(
                {
                    True: {"L1d": 0},
                    0x1000: {"L1d": 0},
                    "0x1000": {"L1d": 0},
                    "": {"L1d": 0},
                    "rdi": "cold",
                    "arg0": {"L1d": 0},
                }
            ),
            {"rdi": [{"L1d": 0}]},
        )
        self.assertEqual(
            _reg_level_values({"rdi": ["bad", 0, 0x1000], "rsi": "0x2000"}),
            {"rdi": {0x1000}, "rsi": {0x2000}},
        )
        mem_levels = {0x1000: {"L1d": 0}}
        self.assertIs(_resolve_reg_levels(mem_levels, []), mem_levels)
        register_levels = {"rdi": {"L1d": 0}}
        self.assertIs(_resolve_reg_levels(register_levels, []), register_levels)
        self.assertEqual(
            _resolve_reg_levels(
                register_levels,
                [],
                {"rdi": 0x1000},
                {"0x1000:": 1},
            ),
            {0x1000: {"L1d": 0}},
        )
        self.assertEqual(
            _resolve_reg_levels(
                {"rdi": {"L1d": 0}, "0x2000": {"L2": 100}},
                [],
                {"rdi": 0x1000},
            ),
            {0x1000: {"L1d": 0}, 0x2000: {"L2": 100}},
        )
        self.assertEqual(_collect_mem_addrs([]), set())

    def test_validate_backend_rejects_empty_option_sweeps(self):
        for config in (
            {"backend": []},
            {"backend": {"loop": {"runs": []}}},
            {"backend": {"loop": {"probes": []}}},
            {"backend": {"loop": {"target": []}}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                cfg_mod.validate_spec(config)
        cfg_mod.validate_spec({"backend": [{"loop": {"runs": [3, 5]}}, "unroll"]})

    def test_validate_external_rejects_empty_sweeps(self):
        for config in (
            {"external": {"lib": []}},
            {"external": {"stdout": []}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                cfg_mod.validate_spec(config)

    def test_concrete_configs_expand_swept_backend_and_branch(self):
        _spec, combos = cfg_mod._concrete_configs(
            {
                "branch": {"0x401000": ["predictable", "unpredictable"]},
                "backend": {"loop": {"runs": ["3", "5"]}},
                "external": {"stdout": [False, True]},
            }
        )
        self.assertEqual(
            {combo["branch"]["0x401000"] for combo in combos},
            {"predictable", "unpredictable"},
        )
        self.assertEqual(
            sorted({int(combo["backend"]["loop"]["runs"]) for combo in combos}),
            [3, 5],
        )
        self.assertEqual(
            {combo["external"]["stdout"] for combo in combos},
            {False, True},
        )

    def test_benchmark_rejects_empty_alternatives(self):
        with self.assertRaises(ValueError):
            bench.benchmark(
                target={"asm": "mov eax, 42"},
                mode=["latency"],
                config={"branch": []},
                event=["duration_time"],
            )

    def test_prune_for_target_skips_func_for_snippets(self):
        combos = [{"func": [{"align": 16}, {"align": 32}]}]
        with patch.object(mca_mod, "_target_features", return_value={}):
            out = mca_mod._prune_for_target(None, None, "nop", combos, {})
        self.assertEqual(len(out), 1)


class TestRecordsMissingEventRegression(unittest.TestCase):
    def test_ndarray_short_row_does_not_fabricate(self):
        import numpy as np

        from perf.bench import _records

        out = _records(
            "f",
            "n",
            "latency",
            np.array([[1.0]]),
            ["cycles", "instructions"],
        )
        self.assertIn("cycles", out[0])
        self.assertNotIn("instructions", out[0])
        self.assertEqual(out[0]["cycles"], 1.0)

    def test_list_rows_map_events_by_position(self):
        from perf.bench import _records

        out = _records(
            "f",
            "n",
            "latency",
            [[1.0, 2.0], [3.0, 4.0]],
            ["cycles", "instructions"],
        )
        self.assertEqual([row["cycles"] for row in out], [1.0, 3.0])
        self.assertEqual([row["instructions"] for row in out], [2.0, 4.0])


class TestBackend(unittest.TestCase):
    def test_resolve_defaults(self):
        from perf.bench import _resolve_backend

        self.assertEqual(_resolve_backend(None, {}, "loop"), "loop")
        self.assertEqual(_resolve_backend(None, {}, "unroll"), "unroll")
        self.assertEqual(
            _resolve_backend(None, {"backend": "unroll"}, "loop"), "unroll"
        )
        self.assertEqual(
            _resolve_backend("loop", {"backend": "unroll"}, "unroll"), "loop"
        )

    def test_resolve_rejects_unknown(self):
        from perf.bench import _resolve_backend

        with self.assertRaises(ValueError):
            _resolve_backend("turbo", {}, "loop")
        with self.assertRaises(ValueError):
            _resolve_backend(None, {"backend": "turbo"}, "loop")

    def test_used_options_are_fully_resolved(self):
        from perf.bench import _select_backend, _used_backend_options

        cfg = {
            "backend": {"loop": {"runs": 3, "target": 0.5}},
            "iterations": {"min": 64, "max": 1000},
        }
        self.assertEqual(
            _used_backend_options(cfg, "loop"),
            {"count": 16, "probes": 3, "runs": 3, "target": 0.5},
        )
        _select_backend(cfg, "loop")
        self.assertEqual(
            cfg["backend"],
            {"loop": {"count": 16, "probes": 3, "runs": 3, "target": 0.5}},
        )
        self.assertNotIn("backend_options", cfg)

    def test_used_options_include_the_unroll_factor(self):
        from perf.bench import _used_backend_options

        used = _used_backend_options({"backend": {"unroll": {"count": 7}}}, "unroll")
        self.assertEqual(used["count"], 7)
        self.assertEqual(_used_backend_options({}, "loop")["count"], 16)

    def test_selected_options_stay_effective(self):
        from perf.bench import _resolve_runs, _select_backend

        cfg = {"backend": {"loop": {"runs": 3}}}
        self.assertEqual(_resolve_runs(cfg, "loop"), 3)
        _select_backend(cfg, "loop")
        self.assertEqual(_resolve_runs(cfg, "loop"), 3)
        self.assertEqual(_resolve_runs(cfg, "unroll"), 10)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_config_backend_mode_map_selects_per_mode(self, mock_bench):
        benchmark(
            config={
                "iterations": 64,
                "samples": 2,
                "backend": {"latency": "unroll", "throughput": "loop"},
            },
            mode=["latency", "throughput"],
            target={"asm": "nop"},
            name="t",
        )
        selected = {
            next(iter(call.kwargs["config"]["backend"]))
            for call in mock_bench.call_args_list
        }
        self.assertEqual(selected, {"loop", "unroll"})

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_backend_is_in_the_output(self, mock_bench):
        result = benchmark(
            config={"iterations": 64, "samples": 2, "branch": "unpredictable"},
            mode=["latency"],
            target={"asm": "nop"},
            name="t",
            backend="unroll",
        )
        rec = result.reset_index()
        self.assertEqual(rec["config.backend.unroll.count"].tolist(), [5] * len(rec))
        self.assertEqual(rec["config.backend.unroll.runs"].tolist(), [10] * len(rec))
        self.assertEqual(rec["config.backend.unroll.probes"].tolist(), [3] * len(rec))
        self.assertNotIn("config.backend_options", rec.columns)
        self.assertFalse([c for c in rec.columns if ".backend_options." in str(c)])
        cols = list(result.columns)
        self.assertEqual(cols[0], "config.backend.unroll.count")
        self.assertLess(
            cols.index("config.backend.unroll.count"), cols.index("config.branch")
        )

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_output_backend_follows_the_explicit_option(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 64,
                "samples": 2,
                "backend": {"loop": {"runs": 4, "target": 0.5}},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="t",
        )
        rec = result.reset_index()
        self.assertEqual(rec["config.backend.loop.runs"].tolist(), [4] * len(rec))
        self.assertEqual(rec["config.backend.loop.target"].tolist(), [0.5] * len(rec))
        self.assertEqual(
            mock_bench.call_args.kwargs["config"]["backend"],
            {"loop": {"count": 16, "probes": 3, "runs": 4, "target": 0.5}},
        )

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_json_output_repeats_the_backend(self, mock_bench):
        result = benchmark(
            config={"iterations": 64, "samples": 2},
            mode=["latency"],
            target={"asm": "nop"},
            name="t",
            backend="loop",
        )
        from perf.bench import to_json

        row = json.loads(to_json(result))["output"][0]
        self.assertEqual(
            row["config"]["backend"],
            {
                "loop": {
                    "count": 16,
                    "probes": 3,
                    "runs": 10,
                    "target": 0.01,
                }
            },
        )
        self.assertNotIn("backend_options", row["config"])
        self.assertIn("code", row["config"])

    def test_default_asm_backend(self):
        from unittest.mock import patch

        import pandas as pd

        from perf.bench import benchmark

        with patch("perf.bench._bench", return_value=pd.Series([10, 20, 30])):
            result = benchmark(
                config={
                    "iterations": 1000,
                    "samples": 3,
                    "branch": "unpredictable",
                    "dcache": "hot",
                    "icache": "hot",
                    "dtlb": "hot",
                    "itlb": "hot",
                    "code": {"align": 16},
                },
                mode=["latency"],
                target={"asm": "nop"},
                name="t",
            )
        self.assertEqual(
            result.attrs["config"]["backend"],
            {
                "loop": {
                    "count": 16,
                    "probes": 3,
                    "runs": 10,
                    "target": 0.01,
                }
            },
        )

    def test_resolve_unroll_n(self):
        from perf.bench import _resolve_unroll_n
        from perf.cfg import _DEFAULT_BENCH as _DB3

        self.assertEqual(
            _resolve_unroll_n(None, {}), _DB3["backend"]["unroll"]["count"]
        )
        self.assertEqual(
            _resolve_unroll_n(None, {"backend": {"unroll": {"count": 3}}}), 3
        )
        self.assertEqual(_resolve_unroll_n(7, {"backend": {"unroll": {"count": 3}}}), 7)
        with self.assertRaises(ValueError):
            _resolve_unroll_n(0, {})
        with self.assertRaises(ValueError):
            _resolve_unroll_n("many", {})

    def test_records_carries_operations(self):
        from perf.bench import _records

        recs = _records(
            "f@1", "n", "latency", pd.Series([10.0, 20.0]), ["cycles"], operations=1
        )
        self.assertEqual([r["operations"] for r in recs], [1, 1])
        self.assertEqual([r["cycles"] for r in recs], [10.0, 20.0])

        recs = _records(
            "f@1",
            "n",
            "throughput",
            pd.Series([100.0]),
            ["cycles"],
            operations=1000,
        )
        self.assertEqual(recs[0]["operations"], 1000)
        self.assertEqual(recs[0]["cycles"], 100.0)

    @patch("perf.bench._bench", return_value=(pd.Series([10, 20, 30]), 1000))
    def test_operations_is_one_for_latency(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="mytarget",
            backend="unroll",
        )
        self.assertEqual(result["operations"].tolist(), [1, 1, 1])
        self.assertEqual(
            result["duration_time"].tolist(),
            _ticks_to_ns(result, [2.0, 4.0, 6.0]),
        )

    @patch("perf.bench._bench", return_value=(pd.Series([10, 20]), 1000))
    def test_operations_is_iterations_for_throughput(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["throughput"],
            target={"asm": "nop"},
            name="mytarget",
        )
        calls = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertTrue(all("nop" in c for c in calls) or len(calls) == 2)
        self.assertEqual(result["operations"].tolist(), [16000, 16000])
        self.assertEqual(
            result["duration_time"].tolist(),
            _ticks_to_ns(result, [10, 20]),
        )

    @patch("perf.bench._bench", return_value=(pd.Series([10, 20]), 1000))
    def test_asm_both_modes_resolve_backend_per_mode(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency", "throughput"],
            target={"asm": "mov eax, 42"},
            name="mytarget",
        )
        rec = result.reset_index()
        self.assertEqual(
            sorted(rec["mode"].unique().tolist()), ["latency", "throughput"]
        )
        self.assertEqual(rec[rec["mode"] == "latency"]["operations"].tolist(), [1, 1])
        self.assertEqual(
            rec[rec["mode"] == "throughput"]["operations"].tolist(),
            [16000, 16000],
        )

    @patch("perf.bench._bench", return_value=(pd.Series([10, 20]), 1000))
    def test_asm_defaults_to_loop_for_throughput(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["throughput"],
            target={"asm": "nop"},
            name="mytarget",
        )
        self.assertEqual(
            result.attrs["config"]["backend"],
            {
                "loop": {
                    "count": 16,
                    "probes": 3,
                    "runs": 10,
                    "target": 0.01,
                }
            },
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertEqual(codes[1], ".align 16\n" + "nop;" * 16)

    def test_unroll_rejected_for_throughput(self):
        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 100, "samples": 1},
                mode=["throughput"],
                target={"asm": "nop"},
                name="t",
                backend="unroll",
            )

    @patch("perf.bench._measure_group")
    def test_loop_low_signal_falls_back_to_the_differential(self, mock_group):
        low = pd.DataFrame({"cycles": [0.0, 0.0]})
        high = pd.DataFrame({"cycles": [15.0, 15.0]})
        mock_group.side_effect = [(low, 10), (high, 10)]

        result = benchmark(
            config={
                "iterations": 10,
                "samples": 1,
                "branch": "predictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            event=["cycles"],
            target={"asm": "imul eax, 42"},
            name="mytarget",
            backend="loop",
        )

        self.assertEqual(mock_group.call_count, 2)
        overhead_code = mock_group.call_args_list[1].args[10]
        main_code = mock_group.call_args_list[1].args[11]
        self.assertEqual(overhead_code.count("imul"), 5)
        self.assertEqual(main_code.count("imul"), 10)
        self.assertEqual(result["cycles"].tolist(), [3.0, 3.0])

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_asm_unroll_is_2N_minus_N(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="mytarget",
            backend="unroll",
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertEqual(len(codes), 2)
        self.assertEqual(codes[0], ".align 16\n" + "nop;" * 5)
        self.assertEqual(codes[1], ".align 16\n" + "nop;" * 10)
        self.assertEqual(
            result["duration_time"].tolist(),
            _ticks_to_ns(result, [2.0, 4.0, 6.0]),
        )

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_asm_loop_single_copy(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="mytarget",
            backend="loop",
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertEqual(codes[1], ".align 16\nnop;")
        self.assertEqual(
            result["duration_time"].tolist(),
            _ticks_to_ns(result, [10, 20, 30]),
        )

    def test_unroll_renames_local_labels(self):
        from perf.bench import _unroll_code, get_arch

        code = _unroll_code(".L1: dec eax\njnz .L1", 2)
        self.assertEqual(code.count(".L1:"), 0)
        self.assertIn(".L1__unroll0:", code)
        self.assertIn(".L1__unroll1:", code)
        self.assertTrue(get_arch().assemble(code))

    def test_unroll_renames_inline_labels(self):
        from perf.bench import _unroll_code, get_arch

        code = _unroll_code("mov eax, 1; jne .L1; nop; .L1:;", 2)
        self.assertEqual(code.count(".L1:"), 0)
        self.assertIn(".L1__unroll0:", code)
        self.assertIn(".L1__unroll1:", code)
        self.assertIn("jne .L1__unroll1", code)
        self.assertTrue(get_arch().assemble(code))

    def test_unroll_separates_unlabeled_multiline_copies(self):
        from perf.bench import _unroll_code, get_arch

        code = _unroll_code("mov eax, 1\nnop", 2)
        self.assertEqual(code, "mov eax, 1\nnop\nmov eax, 1\nnop")
        self.assertTrue(get_arch().assemble(code))

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_asm_unroll_custom_n(self, mock_bench):
        result = benchmark(
            config={
                "iterations": 1000,
                "samples": 3,
                "backend": {"unroll": {"count": 2}},
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target={"asm": "nop"},
            name="mytarget",
            backend="unroll",
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertEqual(codes[0], ".align 16\n" + "nop;" * 2)
        self.assertEqual(codes[1], ".align 16\n" + "nop;" * 4)
        self.assertEqual(
            result["duration_time"].tolist(),
            _ticks_to_ns(result, [5.0, 10.0, 15.0]),
        )

    def test_asm_rejects_bad_backend(self):
        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 1},
                mode=["latency"],
                target={"asm": "nop"},
                name="t",
                backend="turbo",
            )

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.mca.explore", return_value=[])
    @patch("perf.mca.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.mca.Elf")
    @patch("angr.Project")
    def test_file_unroll_subtracts_the_call_baseline(
        self,
        _mock_project,
        mock_elf,
        mock_get_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_get_arch.return_value = x86_64
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        self.addCleanup(Path(path).unlink, missing_ok=True)

        result = benchmark(
            config={
                "iterations": 10,
                "samples": 2,
                "backend": {"unroll": {"count": 3}},
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target=f"{path}:func",
            name="func",
            backend="unroll",
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertEqual(len(codes), 2)
        n_copies = codes[0].count("call rax")
        self.assertEqual(n_copies, 3)
        self.assertEqual(codes[1].count("call rax"), 3)
        self.assertNotEqual(codes[0], codes[1])
        freq = result.attrs["info"]["cpu"]["freq"]
        self.assertAlmostEqual(
            result["duration_time"].tolist()[0], (50 / 3) * 1e9 / freq
        )

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.mca.explore", return_value=[])
    @patch("perf.mca.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.mca.Elf")
    @patch("angr.Project")
    def test_file_loop_is_default_single_call(
        self,
        _mock_project,
        mock_elf,
        mock_get_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_get_arch.return_value = x86_64
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        self.addCleanup(Path(path).unlink, missing_ok=True)

        benchmark(
            config={
                "iterations": 10,
                "samples": 2,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target=f"{path}:func",
            name="func",
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        measured = [c for c in codes if "call rax" in c]
        self.assertEqual(len(measured), 2)
        self.assertEqual([c.count("call rax") for c in measured], [1, 1])
        self.assertNotEqual(measured[0], measured[1])

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.mca.explore", return_value=[])
    @patch("perf.mca.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.mca.Elf")
    @patch("angr.Project")
    def test_file_throughput_uses_a_single_call_sequence(
        self,
        _mock_project,
        mock_elf,
        mock_get_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_get_arch.return_value = x86_64
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        self.addCleanup(Path(path).unlink, missing_ok=True)

        benchmark(
            config={
                "iterations": 10,
                "samples": 2,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["throughput"],
            target=f"{path}:func",
            name="func",
            data={"rdi": 0x5000, "rsi": 16},
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        measured = [c for c in codes if "call rax" in c]
        self.assertEqual(len(measured), 2)
        self.assertEqual([c.count("call rax") for c in measured], [16, 16])
        self.assertNotEqual(measured[0], measured[1])
        self.assertNotIn("mov rdi", measured[1])

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.mca.explore", return_value=[])
    @patch("perf.mca.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.mca.Elf")
    @patch("angr.Project")
    def test_file_latency_keeps_regs_out_of_loop_code(
        self,
        _mock_project,
        mock_elf,
        mock_get_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_get_arch.return_value = x86_64
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        self.addCleanup(Path(path).unlink, missing_ok=True)

        benchmark(
            config={
                "iterations": 10,
                "samples": 2,
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            target=f"{path}:func",
            name="func",
            data={"rdi": 0x5000, "rsi": 16},
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        measured = [c for c in codes if "call rax" in c]
        self.assertEqual(len(measured), 2)
        for code in measured:
            self.assertNotIn("mov rdi", code)

    def test_data_script_uses_combined_steer(self):
        import random

        import keystone

        from perf.bench import _arch_asm
        from perf.cfg import _fill_evict_tables, _per_iter_data

        models = [
            {"regs": {"rdi": 15}, "reads": [(0x41000000000, 4, 42)], "writes": []},
        ]
        buf, meta = _per_iter_data(
            models, 20, {"L1d": 100, "L2": 50, "L3": None}, False, random.Random(1)
        )
        _fill_evict_tables(buf, meta, buf.ctypes.data)
        asm = "\n".join(
            p
            for p in (
                _arch_asm("steer_asm", meta, buf.ctypes.data),
                _arch_asm("prime_asm", meta, buf.ctypes.data),
            )
            if p
        )
        self.assertEqual(asm.count(".perfwrite:"), 1)
        self.assertIn("mov rax, [0x41000000000]", asm)
        self.assertIn("mfence", asm)
        self.assertNotIn(".Sev_done:", asm)
        self.assertIn("pause", asm)
        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        enc, _ = ks.asm(asm)
        self.assertTrue(enc)


class TestHarnessTimingRegisters(unittest.TestCase):
    def test_multi_event_throughput_keeps_saved_scratch_regs_available(self):
        import random

        from perf.bench import _build_loop_asm, get_arch

        models = [
            {
                "regs": {},
                "reads": [(0x41000000000, 8, 1)],
                "writes": [(0x41000000000, 8, 1)],
            }
        ]
        full, _, _ = _build_loop_asm(
            4,
            "throughput",
            "mov rax, [rdi];",
            "",
            "",
            models,
            ["cycles", "instructions", "branches", "branch-misses"],
            [0, 1, 2, 3],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            get_arch(),
        )
        self.assertIn(".perfwrite:", full)

    def test_throughput_scalar_data_is_reloaded_each_iteration(self):
        import random

        from perf.bench import _build_loop_asm, get_arch

        full, _, _ = _build_loop_asm(
            4,
            "throughput",
            "mov eax, ebx;",
            "",
            "",
            [],
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            get_arch(),
            data={"regs": {"rbx": 5}},
        )
        before, after = full.split(".perfloop:")
        self.assertIn("mov rbx", before)
        self.assertIn("mov rbx", after)

    def test_throughput_reloads_scalars_not_in_the_per_iter_table(self):
        import random

        from perf.bench import _build_loop_asm, get_arch

        full, _, _ = _build_loop_asm(
            4,
            "throughput",
            "mov eax, ebx;",
            "",
            "",
            [{"regs": {}, "reads": [], "writes": []}],
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            get_arch(),
            data={"regs": {"rdi": 1, "r12": 2}},
        )
        _before, after = full.split(".perfloop:")
        self.assertIn("mov rdi", after)
        self.assertIn("mov r12, 0x2", after)


class TestModelProbeAlignment(unittest.TestCase):
    def test_model_probe_calls_the_target_with_the_abi_stack_alignment(self):
        import ctypes

        from perf.bench import _model_returns, get_arch
        from perf.core import _jit_page, _jit_release

        arch = get_arch()
        code = arch.assemble(
            arch.normalize_asm(
                """
                mov rax, rsp
                and eax, 0xf
                cmp eax, 0x8
                je .aligned
                ud2
                .aligned:
                ret
                """
            )
        )
        page = _jit_page(code, in_use=True)
        try:
            target = ctypes.addressof(ctypes.c_char.from_buffer(page))
            self.assertTrue(
                _model_returns(
                    arch,
                    arch.call_seq_asm(target),
                    "",
                    "",
                    {"regs": {}, "reads": [], "writes": []},
                )
            )
        finally:
            _jit_release(page)


class TestBackendUnrollN(unittest.TestCase):
    def test_backend_rejects_suffix(self):
        from perf.bench import _resolve_backend

        with self.assertRaises(ValueError):
            _resolve_backend("unroll:1000", {}, "loop")
        with self.assertRaises(ValueError):
            _resolve_backend(None, {"backend": "unroll:1000"}, "loop")
        with self.assertRaises(ValueError):
            _resolve_backend("loop:2", {}, "loop")

    def test_resolve_unroll_n_from_config_only(self):
        from perf.bench import _resolve_unroll_n
        from perf.cfg import _DEFAULT_BENCH as _DB3

        self.assertEqual(
            _resolve_unroll_n(None, {}), _DB3["backend"]["unroll"]["count"]
        )
        self.assertEqual(
            _resolve_unroll_n(None, {"backend": {"unroll": {"count": 1000}}}), 1000
        )
        self.assertEqual(
            _resolve_unroll_n(7, {"backend": {"unroll": {"count": 1000}}}), 7
        )


class TestLoopAsm(unittest.TestCase):
    def test_trailing_return_is_dropped(self):
        from perf.bench import _loop_asm

        self.assertEqual(_loop_asm("mov eax, 0x2a; ret;"), "mov eax, 0x2a;")
        self.assertEqual(_loop_asm("mov eax, 0x2a;\nret;"), "mov eax, 0x2a;")

    def test_only_a_lone_return_is_kept(self):
        from perf.bench import _loop_asm

        self.assertEqual(_loop_asm("ret;"), "ret;")
        self.assertEqual(_loop_asm(""), "")

    def test_interior_return_is_kept(self):
        from perf.bench import _loop_asm

        self.assertEqual(
            _loop_asm("mov eax, 0x2a; ret; mov ebx, 0x1;"),
            "mov eax, 0x2a; ret; mov ebx, 0x1;",
        )


class TestBuildLoopAsmNoModels(unittest.TestCase):
    def test_empty_models_and_no_data_has_no_meta(self):
        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        asm, meta, buf = _build_loop_asm(
            8,
            "latency",
            "nop;",
            "",
            "",
            [],
            ["duration_time"],
            [None],
            None,
            True,
            random.Random(0),
            arch,
            data=None,
        )
        self.assertIsNone(meta)
        self.assertIsNone(buf)
        self.assertIn("nop", asm)

    def test_throughput_reloads_scalar_data_each_iteration(self):
        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        full, _meta, _buf = _build_loop_asm(
            4,
            "throughput",
            "mov eax, ebx;",
            "",
            "",
            [],
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            get_arch(),
            data={"regs": {"rbx": 5}},
        )
        before, after = full.split(".perfloop:")
        self.assertIn("mov rbx, 0x5", before)
        self.assertIn("mov rbx, 0x5", after)


class TestBuildLoopAsmTlbLevels(unittest.TestCase):
    def _asm(self, **kw):
        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        addr = 0x42000000000
        models = [{"regs": {"rdi": addr}, "reads": [(addr, 8, 123)], "writes": []}]
        return _build_loop_asm(
            4,
            "latency",
            "mov rax, [rdi]",
            "",
            "",
            models,
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            arch,
            data={"mem": {"0x42000000000": 123}},
            **kw,
        )

    def test_tlb_levels_reach_the_steer_asm(self):
        asm, meta, _buf = self._asm(tlb_levels={"TLBd": 0})
        self.assertIn("mov rax, 10", asm)
        self.assertEqual(meta["tlb_levels"], {"TLBd": 0})

    def test_a_resident_tlb_emits_no_protection_toggle(self):
        cold, _m, _b = self._asm(tlb_levels={"TLBd": 0})
        hot, _m, _b = self._asm(tlb_levels={"TLBd": 100})
        self.assertIn("mov rax, 10", cold)
        self.assertNotIn("mov rax, 10", hot)

    def test_unconfigured_tlb_axes_emit_no_protection_toggle(self):
        asm, meta, _buf = self._asm()
        self.assertNotIn("mov rax, 10", asm)
        self.assertEqual(meta["tlb_levels"], {})

    def test_code_addresses_are_evicted_through_the_i_tlb(self):
        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        code = 0x401000
        models = [{"regs": {"rdi": 0x42000000000}, "reads": [], "writes": []}]
        asm, meta, _buf = _build_loop_asm(
            4,
            "latency",
            "call rax",
            "",
            "",
            models,
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            arch,
            data={"regs": {"rdi": 0x42000000000}},
            mem_levels={code: {"L1i": 0}},
            extra_addrs=(code,),
            tlb_levels={"TLBi": 0},
        )
        self.assertIn(code, meta["l1i_addrs"])
        self.assertIn(f"mov rdi, 0x{code:x}", asm)

    def test_tlb_avoid_never_contains_code_pages(self):
        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        code = 0x401000
        models = [{"regs": {"rdi": 0x42000000000}, "reads": [], "writes": []}]
        _asm, meta, _buf = _build_loop_asm(
            4,
            "latency",
            "call rax",
            "",
            "",
            models,
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            arch,
            data={"regs": {"rdi": 0x42000000000}},
            mem_levels={code: {"L1i": 0}},
            extra_addrs=(code,),
            tlb_levels={"TLBi": 0},
        )
        avoid_pages = {int(a) & ~0xFFF for a in meta["tlb_avoid"]}
        for addr in meta["l1i_addrs"]:
            self.assertNotIn(int(addr) & ~0xFFF, avoid_pages)


class TestAsmNormalization(unittest.TestCase):
    def test_rewrites_bare_decimal_immediates(self):
        from perf.arch.x86_64 import normalize_asm

        self.assertEqual(normalize_asm("add eax, 42;"), "add eax, 0x2a;")
        self.assertEqual(normalize_asm("sub eax, 42;"), "sub eax, 0x2a;")
        self.assertEqual(normalize_asm("imul eax, eax, 42;"), "imul eax, eax, 0x2a;")

    def test_leaves_hex_regs_and_memory(self):
        from perf.arch.x86_64 import normalize_asm

        self.assertEqual(normalize_asm("add eax, 0x2a;"), "add eax, 0x2a;")
        self.assertEqual(normalize_asm("add r11, [rax];"), "add r11, [rax];")
        self.assertEqual(normalize_asm("nop;"), "nop;")
        self.assertEqual(normalize_asm("mov rax, 0x400000;"), "mov rax, 0x400000;")

    def test_normalized_assembles_to_same_bytes(self):
        from perf.arch.x86_64 import assemble, normalize_asm

        cases = (
            ("add eax, 42", "add eax, 0x2a;"),
            ("sub eax, 42", "sub eax, 0x2a;"),
            ("imul eax, eax, 42", "imul eax, eax, 0x2a;"),
        )
        for code, expected_hex in cases:
            self.assertEqual(normalize_asm(code + ";"), expected_hex)
            self.assertEqual(
                bytes(assemble(normalize_asm(code + ";"))),
                bytes(assemble(expected_hex)),
            )
        self.assertEqual(bytes(assemble("add eax, 0x2a;")), b"\x83\xc0*")

    def test_timed_regs_avoid_covers_data(self):
        from perf.arch import x86_64
        from perf.bench import _timed_regs_avoid

        avoid = _timed_regs_avoid(
            "add r11, [rax]",
            None,
            None,
            {"regs": {"rax": 0x400000}, "mem": {"0x400000": 100}},
            x86_64,
        )
        self.assertIn("r11", avoid)
        self.assertIn("rax", avoid)

    def test_collect_rejects_invalid_idiv_syntax(self):
        import random

        from perf.arch import arch as get_arch
        from perf.bench import _collect

        arch = get_arch()
        with self.assertRaises(ValueError) as ctx:
            _collect(
                {},
                8,
                1,
                "latency",
                "idiv eax, 42;",
                "",
                "",
                [],
                ["duration_time"],
                [None],
                0,
                None,
                True,
                random.Random(0),
                arch,
                data=None,
            )
        self.assertIn("idiv", str(ctx.exception))


class TestAsmNormalizationExtended(unittest.TestCase):
    def test_default_radix_is_decimal(self):
        import keystone

        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        self.assertEqual(list(ks.asm("add eax, 42")[0]), [0x83, 0xC0, 0x2A])
        self.assertEqual(list(ks.asm("add eax, 0x2a")[0]), [0x83, 0xC0, 0x2A])

    def test_negative_immediates_preserve_bytes(self):
        from perf.arch.x86_64 import assemble, normalize_asm

        self.assertEqual(normalize_asm("add eax, -42;"), "add eax, 0xffffffffffffffd6;")
        self.assertEqual(
            bytes(assemble(normalize_asm("add eax, -42;"))),
            bytes(assemble("add eax, -42")),
        )

    def test_normalize_leaves_explicit_hex_and_memory(self):
        from perf.arch.x86_64 import assemble, normalize_asm

        for code in (
            "add eax, 0x2a;",
            "mov rax, 0x400000;",
            "add r11, [rax];",
            "nop;",
        ):
            self.assertEqual(normalize_asm(code), code)
            self.assertTrue(bytes(assemble(code)))


class TestAsmAccuracyAlderLake(unittest.TestCase):
    def test_snippets_assemble(self):
        import keystone

        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        for code in (
            "add eax, 42",
            "sub eax, 42",
            "imul eax, eax, 42",
            "nop",
            "add r11, [rax]",
            "idiv ecx",
        ):
            enc, _ = ks.asm(code)
            self.assertTrue(enc, code)

    def test_invalid_idiv_syntax_raises(self):
        from perf.bench import benchmark

        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 8, "samples": 2},
                mode=["latency"],
                target={"asm": "idiv eax, 42"},
                name="idiv-bad",
            )


class TestFuncCallPreservation(unittest.TestCase):
    def test_call_wrapper_preserves_loop_regs(self):
        import inspect

        from perf.arch import x86_64
        from perf.bench import _bench_one

        seq = x86_64.call_seq_asm(0x401000)
        self.assertIn("push r8", seq)
        self.assertIn("push r9", seq)
        self.assertIn("push r10", seq)
        self.assertIn("call rax", seq)
        self.assertIn("pop r10", seq)

        fenced = x86_64.call_seq_asm(0x401000, fence=True)
        self.assertIn("call rax\nlfence", fenced)

        src = inspect.getsource(_bench_one)
        self.assertIn("arch.call_seq_asm(target, fence=", src)


class TestCallStubBaseline(unittest.TestCase):
    def test_call_stub_code_calls_a_ret_stub(self):
        import ctypes
        import re

        from perf.arch import x86_64
        from perf.bench import _call_stub_code

        with _call_stub_code(x86_64, fence=True, copies=2) as code:
            self.assertEqual(code.count("call rax"), 2)
            self.assertEqual(code.count("lfence"), 2)
            match = re.search(r"mov rax, (0x[0-9a-f]+)", code)
            self.assertIsNotNone(match)
            target = int(match.group(1), 0)
            self.assertEqual(ctypes.c_ubyte.from_address(target).value, 0xC3)
            ctypes.CFUNCTYPE(None)(target)()

    def test_binary_target_overhead_uses_the_call_stub(self):
        from perf.bench import _bench_one

        src = inspect.getsource(_bench_one)
        self.assertIn("_call_stub_code(", src)
        self.assertNotIn("call_seq_nop_asm", src)


class TestHarnessHelperEdges(unittest.TestCase):
    def test_scale_row_operations(self):
        from perf.bench import _scale_row_operations

        frame = pd.DataFrame({"cycles": [1, 2]})
        items = []
        self.assertIs(_scale_row_operations(frame, 1), frame)
        self.assertIs(_scale_row_operations(items, 2), items)
        self.assertIs(_scale_row_operations(frame, 2), frame)
        frame.attrs["_perf_operations"] = [10, 20]
        _scale_row_operations(frame, 4)
        self.assertEqual(frame.attrs["_perf_operations"], [40, 80])
        frame.attrs["_perf_operations"] = [None]
        _scale_row_operations(frame, 2)
        self.assertEqual(frame.attrs["_perf_operations"], [None])

    def test_result_low_signal(self):
        from perf.bench import _result_low_signal

        self.assertFalse(_result_low_signal([], ["cycles"]))
        self.assertFalse(
            _result_low_signal(pd.DataFrame({"instructions": [0]}), ["instructions"])
        )
        with patch("perf.bench._is_timing_event", return_value=True):
            self.assertFalse(
                _result_low_signal(
                    pd.DataFrame({"topdown-be-bound": [0.0]}), ["topdown-be-bound"]
                )
            )
        self.assertTrue(
            _result_low_signal(pd.DataFrame({"cycles": [0.0, 1.0]}), ["cycles"])
        )
        self.assertFalse(
            _result_low_signal(pd.DataFrame({"cycles": [np.nan]}), ["cycles"])
        )

    def test_differential_safe_dispatch(self):
        from perf.bench import _differential_safe

        self.assertFalse(_differential_safe(object(), "nop"))

        def broken(_code):
            raise RuntimeError("bad")

        self.assertFalse(
            _differential_safe(SimpleNamespace(differential_safe=broken), "nop")
        )
        self.assertTrue(
            _differential_safe(
                SimpleNamespace(differential_safe=lambda code: code == "nop"), "nop"
            )
        )

    def test_needs_guard_skips_broken_chunks(self):
        from perf.bench import _needs_guard

        arch = SimpleNamespace(
            _HARNESS_LOOP_REGS=["r8"],
            used_regs=Mock(side_effect=RuntimeError("bad")),
            _canonical_reg=lambda reg: reg,
        )
        self.assertFalse(_needs_guard(arch, "mov r8, 1"))

    def test_model_picks_without_randbytes(self):
        from perf.cfg import _model_picks

        class Rng:
            def randrange(self, _n):
                return 0

        models = [{"regs": {}, "reads": [], "writes": []}]
        self.assertEqual(_model_picks(models, False, Rng(), None, 3), [0, 0, 0])

    @patch("perf.bench.get_arch", side_effect=RuntimeError("bad"))
    def test_align_directive_failure(self, _mock):
        from perf.bench import _align_directive

        self.assertEqual(_align_directive(16), "")

    @patch("perf.bench.get_arch", side_effect=RuntimeError("bad"))
    def test_asm_error_ignores_missing_hint(self, _mock):
        from perf.bench import _asm_error

        with self.assertRaisesRegex(ValueError, "original"):
            _asm_error("nop", ValueError("original"))

    def test_refresh_per_iter_data_fallback(self):
        from perf.arch import arch as get_arch
        from perf.cfg import _per_iter_data, _refresh_per_iter_data

        arch = get_arch()
        branch_cfg = {"prediction": "unpredictable", "regs": {}, "mem": {}}
        models = [{"regs": {}, "reads": [], "writes": []}]
        rng = random.Random(0)
        data = {"regs": {"rdi": [0, 1]}, "mem": {}}
        buf, meta = _per_iter_data(
            models, 8, None, False, rng, arch, explicit=data, branch_cfg=branch_cfg
        )
        real = cfg_mod._fill_dynamic
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("fail")
            return real(*args, **kwargs)

        with patch("perf.cfg._fill_dynamic", side_effect=flaky):
            _refresh_per_iter_data(
                buf,
                meta,
                8,
                models,
                None,
                False,
                rng,
                arch,
                data=data,
                branch_cfg=branch_cfg,
            )
        self.assertGreaterEqual(len(calls), 2)


class TestIterationPlan(unittest.TestCase):
    def test_exact_relative_standard_error_formula(self):
        values = [95.0, 105.0, 97.0, 103.0, 100.0, 99.0, 101.0]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        expected = int(math.ceil((std / (0.005 * mean)) ** 2))
        self.assertEqual(_plan_iterations(values, 0.005, 2, 1000000), expected)

    def test_degenerate_inputs_return_the_minimum(self):
        self.assertEqual(_plan_iterations([0, 0, 0, 0], 0.005, 100, 1000), 100)
        self.assertEqual(_plan_iterations([5, 5, 5, 5], 0.005, 100, 1000), 100)
        self.assertEqual(_plan_iterations([1, 2], 0.005, 100, 1000), 100)
        self.assertEqual(
            _plan_iterations([0, 0, 0, 0, 0, 0, 1, 1], 0.005, 100, 1000), 100
        )


class TestCallDetection(unittest.TestCase):
    def test_words_not_comments(self):
        self.assertTrue(_contains_call("push r8\ncall rax\npop r8"))
        self.assertTrue(_contains_call("CALL rax"))
        self.assertTrue(_contains_call("  call   rax  "))
        self.assertFalse(_contains_call("mov eax, 42"))
        self.assertFalse(_contains_call("mov eax, 1 ; call this comment"))
        self.assertFalse(_contains_call("mov eax, 1 # call this comment"))
        self.assertFalse(_contains_call("; call rax"))
        self.assertTrue(_contains_call("; comment\ncall rax"))
        self.assertFalse(_contains_call("recall_counter:\nmov eax, 1"))
        self.assertFalse(_contains_call("mov eax, recall"))
        self.assertFalse(_contains_call(""))
        self.assertTrue(_contains_call("nop; call rax;"))
        self.assertTrue(_contains_call("nop; callq rax;"))
        self.assertFalse(_contains_call("mov eax, 1 ; call this comment"))


class TestThroughputBatching(unittest.TestCase):
    def test_static_read_only_values_are_written_before_the_timed_loop(self):
        import random

        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        addr = 0x42000000000
        models = [
            {
                "regs": {"rdi": addr},
                "reads": [(addr, 8, 123)],
                "writes": [],
            }
        ]
        asm, meta, _ = _build_loop_asm(
            8,
            "throughput",
            "mov rax, [rdi];",
            "",
            "",
            models,
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            arch,
        )
        self.assertTrue(meta["values_static"])
        before, after = asm.split(".perfloop:")
        self.assertIn(".perfwrite", before)
        self.assertNotIn(".perfwrite", after)

    def test_read_only_values_that_change_stay_in_the_timed_loop(self):
        import random

        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        addr = 0x42000000000
        models = [
            {
                "regs": {"rdi": addr},
                "reads": [(addr, 8, 123)],
                "writes": [],
            }
        ]
        asm, meta, _ = _build_loop_asm(
            8,
            "throughput",
            "mov rax, [rdi];",
            "",
            "",
            models,
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            False,
            random.Random(0),
            arch,
        )
        self.assertFalse(meta["values_static"])
        before, after = asm.split(".perfloop:")
        self.assertNotIn(".perfwrite", before)
        self.assertIn(".perfwrite", after)

    def test_target_writes_keep_the_value_write_in_the_timed_loop(self):
        import random

        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        addr = 0x42000000000
        models = [
            {
                "regs": {"rdi": addr},
                "reads": [],
                "writes": [(addr, 8, 123)],
            }
        ]
        asm, meta, _ = _build_loop_asm(
            8,
            "throughput",
            "mov [rdi], rax;",
            "",
            "",
            models,
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            arch,
        )
        self.assertTrue(meta["target_writes"])
        self.assertFalse(meta["values_static"])
        _before, after = asm.split(".perfloop:")
        self.assertIn(".perfwrite", after)

    def test_missing_model_keeps_the_value_write_in_the_timed_loop(self):
        import random

        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        addr = 0x42000000000
        asm, meta, _ = _build_loop_asm(
            8,
            "throughput",
            f"mov rax, [{hex(addr)}];",
            "",
            "",
            [],
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            arch,
            data={"mem": {hex(addr): 123}},
        )
        self.assertFalse(meta["values_static"])
        before, after = asm.split(".perfloop:")
        self.assertNotIn(".perfwrite", before)
        self.assertIn(".perfwrite", after)

    def test_loop_count_default_and_override(self):
        from perf.bench import _resolve_loop_count

        self.assertEqual(_resolve_loop_count({}), 16)
        self.assertEqual(_resolve_loop_count({"backend": {"loop": {"count": 4}}}), 4)
        with self.assertRaises(ValueError):
            _resolve_loop_count({"backend": {"loop": {"count": 0}}})

    def test_alignment_only_has_no_instruction(self):
        from perf.bench import _align_only

        self.assertEqual(_align_only(16), ".align 16\n")
        self.assertEqual(_align_only(1), "")

    def test_throughput_snippet_skips_the_guard_without_a_call(self):
        import random

        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        asm, _, _ = _build_loop_asm(
            16,
            "throughput",
            "mov eax, 42;",
            "",
            "",
            [],
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 100, "L3": 100},
            True,
            random.Random(0),
            arch,
        )
        self.assertNotIn("push r8", asm)
        self.assertNotIn("lea rsp, [rsp-128]", asm)
        self.assertIn("dec r8", asm)

        forced, _, _ = _build_loop_asm(
            16,
            "throughput",
            "mov eax, 42;",
            "",
            "",
            [],
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 100, "L3": 100},
            True,
            random.Random(0),
            arch,
            force_guard=True,
        )
        self.assertIn("push r8", forced)
        self.assertIn("lea rsp, [rsp-128]", forced)

    def test_harness_register_snippet_enables_the_guard(self):
        import random

        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        for mode in ("latency", "throughput"):
            asm, _, _ = _build_loop_asm(
                8,
                mode,
                "mov r8, 42;",
                "",
                "",
                [],
                ["duration_time"],
                [None],
                {"L1d": 100, "L2": 100, "L3": 100},
                True,
                random.Random(0),
                arch,
            )
            self.assertIn("push r8", asm, mode)

    def test_latency_steering_preserves_timing_baselines(self):
        import random

        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        addr = 0x42000000000
        models = [
            {
                "regs": {},
                "reads": [(addr, 8, 123)],
                "writes": [],
            }
        ]
        asm, _, _ = _build_loop_asm(
            8,
            "latency",
            f"mov rax, [{hex(addr)}];",
            "",
            "",
            models,
            ["cycles", "instructions", "branch-misses"],
            [1, 2, 3],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            arch,
        )
        loop = asm.split(".perfloop:", 1)[1]
        self.assertIn("push rax", loop)
        self.assertTrue(arch.assemble(asm))


class TestRuntableModelsUnit(unittest.TestCase):
    def setUp(self):
        from perf import bench

        self.bench = bench
        bench._RUNTABLE_MODELS.clear()
        self.addCleanup(bench._RUNTABLE_MODELS.clear)

    @staticmethod
    def _world(model):
        return {
            "regs": dict(model.get("regs") or {}),
            "reads": list(model.get("reads") or []),
            "writes": list(model.get("writes") or []),
        }

    def _keep(self, models, predicate, probe="ret", setup=""):
        with (
            patch.object(bench, "_model_world", side_effect=self._world),
            patch.object(bench, "_model_returns", side_effect=predicate) as mock,
        ):
            kept = bench._runtable_models(models, object(), probe, setup_asm=setup)
        return kept, mock

    def test_no_probe_keeps_the_models(self):
        models = [{"regs": {"rdi": 1}, "reads": [], "writes": []}]
        with patch.object(bench, "_model_returns") as mock:
            self.assertIs(bench._runtable_models(models, object(), None), models)
        mock.assert_not_called()

    def test_worlds_that_do_not_return_are_dropped(self):
        models = [
            {"regs": {"rdi": 1}, "reads": [(0x1000, 8, 0)], "writes": []},
            {"regs": {"rdi": 2}, "reads": [], "writes": []},
        ]
        kept, _ = self._keep(
            models,
            lambda _arch, _probe, _setup, _teardown, world, config=None: (
                not world["reads"]
            ),
        )
        self.assertEqual([w["regs"] for w in kept], [{"rdi": 2}])

    def test_all_blocked_falls_back_to_regs_only_world(self):
        models = [{"regs": {"rdi": 7}, "reads": [(0x1000, 8, 0)], "writes": []}]
        kept, _ = self._keep(
            models,
            lambda _arch, _probe, _setup, _teardown, world, config=None: (
                not world["reads"]
            ),
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["regs"], {"rdi": 7})
        self.assertEqual(kept[0]["reads"], [])

    def test_every_world_kept_when_none_returns(self):
        models = [
            {"regs": {"rdi": 1}, "reads": [(0x1000, 8, 0)], "writes": []},
            {"regs": {"rdi": 2}, "reads": [(0x2000, 8, 0)], "writes": []},
        ]
        kept, _ = self._keep(models, lambda *args, **kwargs: False)
        self.assertEqual(len(kept), 2)

    def test_verdict_is_cached_for_the_same_key(self):
        models = [{"regs": {"rdi": 1}, "reads": [], "writes": []}]
        with (
            patch.object(bench, "_model_world", side_effect=self._world),
            patch.object(bench, "_model_returns", return_value=True) as mock,
        ):
            bench._runtable_models(models, object(), "ret")
            bench._runtable_models(models, object(), "ret")
        self.assertEqual(mock.call_count, 1)

    def test_setup_change_breaks_the_cache(self):
        models = [{"regs": {"rdi": 1}, "reads": [], "writes": []}]
        with (
            patch.object(bench, "_model_world", side_effect=self._world),
            patch.object(bench, "_model_returns", return_value=True) as mock,
        ):
            bench._runtable_models(models, object(), "ret", setup_asm="a")
            bench._runtable_models(models, object(), "ret", setup_asm="b")
        self.assertEqual(mock.call_count, 2)

    def test_empty_models(self):
        self.assertEqual(bench._runtable_models([], object(), "ret"), [])


if __name__ == "__main__":
    unittest.main()
