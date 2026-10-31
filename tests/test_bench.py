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
import os
import random
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

import perf.bench as bench
import perf.core as core
from perf.bench import _DEFAULT_BENCH, benchmark, parse_code, region

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
        result = benchmark(asm="nop", name="t", config={"branch": ["predictable"]})
        self.assertEqual(sorted(result["mode"].unique()), ["latency", "throughput"])

    @patch("perf.bench._bench_one")
    def test_modes_run_with_each_config_combo(self, mock_one):
        def fake(**kw):
            return pd.DataFrame(
                [{"mode": kw["mode"], "align": kw["config"]["code"]["align"]}]
            )

        mock_one.side_effect = fake
        result = benchmark(
            asm="nop",
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
            asm="nop",
            name="t",
            mode=["throughput", "latency"],
            config={"branch": ["predictable", "unpredictable"]},
        )
        modes = [c.kwargs["mode"] for c in mock_one.call_args_list]
        self.assertEqual(modes[0], "throughput")
        self.assertEqual(modes[-1], "latency")

    def test_modes_must_be_a_list(self):
        from perf.bench import _modes

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
        self.assertEqual(parse_code("nop # note: x"), (None, None, "nop # note: x"))
        self.assertEqual(parse_code("a.out: func"), ("a.out", "func", None))

    def test_parse_code_pair(self):
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
        benchmark(symbol=["a.out", "foo"], mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertEqual((kwargs["file"], kwargs["target"]), ("a.out", "foo"))
        self.assertIsNone(kwargs["code"])

    @patch("perf.bench._bench_one")
    def test_code_kwarg_text(self, mock_one):
        self._one(mock_one)
        benchmark(region="a.out:begin..end", mode=["latency"], config=self._FAST)
        self.assertEqual(mock_one.call_args.kwargs["target"], ("begin", "end"))

    @patch("perf.bench._bench_one")
    def test_code_positional(self, mock_one):
        self._one(mock_one)
        benchmark(["a.out", "foo"], mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertEqual((kwargs["file"], kwargs["target"]), ("a.out", "foo"))

    @patch("perf.bench._bench_one")
    def test_code_positional_asm(self, mock_one):
        self._one(mock_one)
        benchmark(asm="mov eax, 42", mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertIsNone(kwargs["file"])
        self.assertEqual(kwargs["code"], "mov eax, 42")

    @patch("perf.bench._bench_one")
    def test_target_and_code_forms(self, mock_one):
        self._one(mock_one)
        benchmark(symbol=["a.out", "main"], mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertEqual((kwargs["file"], kwargs["target"]), ("a.out", "main"))
        self.assertIsNone(kwargs["code"])
        benchmark(asm="mov eax, 42", mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertIsNone(kwargs["file"])
        self.assertEqual(kwargs["code"], "mov eax, 42")

    def test_code_conflicts(self):
        for kw in (
            {"symbol": "a.out:foo", "asm": "mov eax, 42"},
            {"symbol": "a.out:foo", "file": "a.out"},
            {"code": "mov eax, 42", "asm": "mov eax, 42"},
            {"asm": "mov eax, 42", "symbol": "foo"},
        ):
            with self.subTest(**kw), self.assertRaises(TypeError):
                benchmark(mode=["latency"], config=self._FAST, **kw)

    def test_needs_target_or_asm(self):
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
            asm="nop",
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
        result = benchmark(asm="nop", name="t", mode=["latency"], config=None)
        cfg = mock_bench.call_args.kwargs["config"]

        for key in ("samples",):
            self.assertEqual(cfg[key], _DEFAULT_BENCH[key])
        self.assertEqual(
            cfg["backend"],
            {"loop": _DEFAULT_BENCH["backend"]["loop"]},
        )
        self.assertEqual(
            cfg["iterations"],
            {
                **_DEFAULT_BENCH["iterations"],
                "count": _DEFAULT_BENCH["iterations"]["min"],
            },
        )

        self.assertIsInstance(cfg["seed"], int)
        self.assertEqual(cfg["thread"][0]["affinity"], 0)
        self.assertEqual(cfg["thread"][0]["priority"], "normal")
        self.assertEqual(cfg["func"]["order"], "as-is")
        self.assertIsInstance(cfg["iterations"]["count"], int)

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
            asm="nop",
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
            asm="nop",
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
            asm="nop",
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
            asm="nop",
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
            asm="nop",
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

        from perf.bench import _map_data_pages

        data = {"regs": {}, "mem": {"0x43000000000": 0xDEADBEEF}}
        _map_data_pages(data)

        ctypes.c_uint64.from_address(0x43000000000).value = 0xDEADBEEF
        self.assertEqual(ctypes.c_uint64.from_address(0x43000000000).value, 0xDEADBEEF)

    def test_map_data_pages_reports_a_taken_page(self):
        import ctypes

        import perf.bench as bench

        mapped = set(bench._MAPPED_DATA_PAGES)
        unmappable = set(bench._UNMAPPABLE_DATA_PAGES)

        def _restore():
            bench._MAPPED_DATA_PAGES.clear()
            bench._MAPPED_DATA_PAGES.update(mapped)
            bench._UNMAPPABLE_DATA_PAGES.clear()
            bench._UNMAPPABLE_DATA_PAGES.update(unmappable)

        self.addCleanup(_restore)
        buf = ctypes.create_string_buffer(4096)
        page = (ctypes.addressof(buf) + 4095) & ~4095
        with self.assertRaises(OSError) as ctx:
            bench._map_data_pages({"regs": {}, "mem": {hex(page): 1}})
        self.assertNotIsInstance(ctx.exception, NameError)
        self.assertTrue(ctx.exception.errno)
        self.assertIn("address already mapped", str(ctx.exception))

    def test_data_assembly_round_trips(self):
        import keystone

        from perf.arch.x86_64 import data_setup_asm
        from perf.bench import _map_data_pages

        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        data = {"regs": {"rcx": 0x45000000000}, "mem": {"0x41000000000": 32}}
        _map_data_pages(data)
        asm = data_setup_asm(data)
        enc, _ = ks.asm(asm)
        self.assertTrue(enc)

    def test_reserved_registers_are_rejected(self):
        from perf.bench import check_data

        for reg in ("r8", "r9", "r10", "rsp", "esp", "rip", "eip", "ip", "flags"):
            with self.assertRaises(ValueError) as ctx:
                check_data({"regs": {reg: 1}, "mem": {}})
            self.assertIn("reserved by the measurement harness", str(ctx.exception))

    def test_argument_registers_are_allowed(self):
        from perf.bench import check_data

        for reg in ("rdi", "rsi", "rdx", "rcx", "rax", "rbx", "rbp", "r12", "r15"):
            check_data({"regs": {reg: 1}, "mem": {}})

    def test_bench_rejects_reserved_registers(self):
        with self.assertRaises(ValueError):
            benchmark(
                asm="mov eax, 42",
                mode=["latency"],
                config={"iterations": 8, "samples": 1},
                data={"r8": 1},
            )

    def test_asm_source_label_becomes_a_snippet(self):
        import tempfile
        from pathlib import Path

        from perf.bench import asm_source_code

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
        from perf.bench import asm_source_code

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "att.s")
        with open(path, "w") as fh:
            fh.write("att:\n\tmovl $42, %eax\n\tretl\n")
        self.assertIsNone(asm_source_code(path, "att"))

    def test_asm_source_region_semantics(self):
        from pathlib import Path

        from perf.bench import _asm_region, asm_source_target

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
                "L1d": {"hit_rate": 100},
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

        cfg = {"dcache": {"L1d": {"hit_rate": 50}, "L2": {"hit_rate": 100}}}
        levels = data_cache_levels(cfg)
        self.assertEqual(levels["L1d"], 50)
        self.assertEqual(levels["L2"], 100)

    def test_cache_levels_default(self):
        from perf.arch.x86_64 import _MEMORY_SHORTCUTS, data_cache_levels
        from perf.bench import _DEFAULT_BENCH

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

    def test_per_iter_predictable_cycles_models_in_order(self):
        import random

        from perf.bench import _per_iter_data

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

        from perf.bench import _per_iter_data

        models = [
            {"regs": {"rdi": 15}, "reads": [], "writes": []},
            {"regs": {"rdi": 3}, "reads": [], "writes": []},
        ]
        buf, meta = _per_iter_data(models, 200, None, False, random.Random(7))
        self.assertEqual(set(buf[0, 1:201]), {3, 15})

    def test_harness_regs_excluded(self):
        import random

        from perf.bench import _per_iter_data

        models = [{"regs": {"r8": 99, "rdi": 5}, "reads": [], "writes": []}]
        buf, meta = _per_iter_data(models, 10, None, True, random.Random(1))
        self.assertNotIn("r8", meta["reg_col"])
        self.assertIn("rdi", meta["reg_col"])

    def test_data_script_assembles(self):
        import random

        import keystone

        from perf.bench import (
            _arch_asm,
            _fill_evict_tables,
            _per_iter_data,
        )

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

        from perf.bench import (
            _arch_asm,
            _fill_evict_tables,
            _per_iter_data,
        )

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

        from perf.bench import _fill_evict_tables, _per_iter_data

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

        from perf.bench import _fill_evict_tables, _per_iter_data

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
        self.assertEqual(target, 0.005)
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


class TestSolveReturnValueSplit(unittest.TestCase):
    def _solution(self, rax_expr, x=None):
        import angr

        proj = angr.load_shellcode(b"\x90", arch="x86_64", load_address=0x1000)
        state = proj.factory.blank_state(addr=0x1000)
        if x is None:
            import claripy

            x = claripy.BVS("x_1_64", 64)
        state.regs.rdi = x
        state.regs.rax = rax_expr
        return {"state": state, "reg": {"rdi": x}, "reads": [], "writes": []}

    def test_conditional_return_splits_into_two(self):
        import claripy

        from perf.bench import solve

        x = claripy.BVS("rdi_1_64", 64)
        sol = self._solution(
            claripy.If(x == 0, claripy.BVV(0x1111, 64), claripy.BVV(0x2222, 64)), x
        )
        models = solve([sol])
        self.assertEqual(len(models), 2)
        vals = {m["regs"]["rdi"] for m in models}
        self.assertIn(0, vals)
        self.assertEqual(len(vals), 2)

    def test_linear_return_not_split(self):
        import claripy

        from perf.bench import solve

        x = claripy.BVS("x_1_64", 64)
        models = solve([self._solution(x + 5)])
        self.assertEqual(len(models), 1)

    def test_concrete_return_not_split(self):
        from perf.bench import solve

        models = solve([self._solution(0x4242)])
        self.assertEqual(len(models), 1)

    def test_incomplete_memory_resolution_is_skipped(self):
        import claripy

        from perf.bench import solve

        solution = self._solution(0x4242)
        solution["reads"] = [
            (claripy.BVV(0x1000, 64), claripy.BVV(4, 64), claripy.BVV(1, 64))
        ]
        with patch("perf.bench.eval_ints", return_value=[0x1000, 4]):
            models = solve([solution])
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["reads"], [])


class TestBenchFilePath(unittest.TestCase):
    def _binary(self):
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        self.addCleanup(Path(path).unlink, missing_ok=True)
        return path

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_file_path_builds_dataframe(
        self,
        mock_project,
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
            symbol=f"{path}:func",
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
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_data_regs_map_to_prototype(
        self,
        mock_project,
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
            symbol=f"{path}:func",
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
        from perf.bench import data_param_columns, data_param_value

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
        from perf.bench import data_param_columns, data_param_value

        data = {"regs": {"rdi": 1, "arg0": 2}}
        cols = data_param_columns(data)
        self.assertEqual(cols["data.rdi"], 2)
        self.assertNotIn("data.arg0", cols)
        self.assertEqual(data_param_value(data, "data.rdi"), 2)
        self.assertEqual(data_param_value(data, "data.arg0"), 2)

    def test_data_param_value_mem_fallback_preserved(self):
        from perf.bench import data_param_value

        data = {"regs": {"rdi": 1}, "mem": {"0x1000": 5}}
        self.assertEqual(data_param_value(data, "data.0x1000"), 5)
        self.assertIsNone(data_param_value(data, "data.mem.0x1000"))
        self.assertIsNone(data_param_value(data, "data.memory.0x1000"))
        self.assertIsNone(data_param_value(data, "data.regs.rdi"))
        self.assertEqual(data_param_value(data, "data.rdi"), 1)
        self.assertEqual(data_param_value(data, "data.arg0"), 1)
        self.assertEqual(data_param_value(data, "data.rax"), None)

    def test_data_param_columns_keep_distribution(self):
        from perf.bench import data_param_columns, data_param_value

        data = {"regs": {"rdi": [1, 3, 5]}, "mem": {"0x1000": [7, 8]}}
        cols = data_param_columns(data)
        self.assertEqual(cols["data.rdi"], [1, 3, 5])
        self.assertNotIn("data.arg0", cols)
        self.assertEqual(cols["data.0x1000"], [7, 8])
        self.assertEqual(data_param_value(data, "data.rdi"), 1)

    def test_solver_data_collects_state_lists(self):
        from perf.bench import _MODEL_CACHE

        _MODEL_CACHE.clear()
        from perf.bench import explored_data_dict

        models = [
            {"regs": {"rdi": 1}, "reads": [(0x1000, 8, 7)], "writes": []},
            {"regs": {"rdi": 2}, "reads": [(0x1000, 8, 8)], "writes": []},
            {"regs": {"rdi": 3}, "reads": [(0x1000, 8, 7)], "writes": []},
        ]
        with patch("perf.bench.solve", return_value=models):
            d = explored_data_dict(["fake"])
        self.assertEqual(d["regs"]["rdi"], [1, 2, 3])
        self.assertEqual(d["mem"]["0x1000"], [7, 8])

    def test_solver_data_single_value_is_scalar(self):
        from perf.bench import _MODEL_CACHE

        _MODEL_CACHE.clear()
        from perf.bench import explored_data_dict

        models = [
            {"regs": {"rsi": 42}, "reads": [(0x2000, 8, 5)], "writes": []},
        ]
        with patch("perf.bench.solve", return_value=models):
            d = explored_data_dict(["fake"])
        self.assertEqual(d["regs"]["rsi"], 42)
        self.assertEqual(d["mem"]["0x2000"], 5)

    def test_solver_data_skips_harness_regs(self):
        from perf.bench import _MODEL_CACHE

        _MODEL_CACHE.clear()
        from perf.bench import explored_data_dict

        models = [
            {"regs": {"rdi": 1, "r8": 99}, "reads": [], "writes": []},
            {"regs": {"rdi": 2, "r8": 100}, "reads": [], "writes": []},
        ]
        with patch("perf.bench.solve", return_value=models):
            d = explored_data_dict(["fake"])
        self.assertEqual(d["regs"]["rdi"], [1, 2])
        self.assertNotIn("r8", d["regs"])

    def test_solver_data_canonicalizes_no_alias_columns(self):
        from perf.bench import _MODEL_CACHE

        _MODEL_CACHE.clear()
        from perf.bench import data_param_columns, explored_data_dict

        models = [
            {"regs": {"rdi": 1}, "reads": [], "writes": []},
            {"regs": {"rdi": 2}, "reads": [], "writes": []},
        ]
        with patch("perf.bench.solve", return_value=models):
            d = explored_data_dict(["fake"])
        cols = data_param_columns({"regs": d["regs"], "mem": d["mem"]})
        self.assertEqual(cols["data.rdi"], [1, 2])
        self.assertNotIn("data.arg0", cols)

    def test_func_prototype_imports_from_core(self):
        from perf.core import _func_prototype

        self.assertIsNotNone(_func_prototype(None, None))

    def test_scratch_value_filters_hex_strings(self):
        from perf.bench import _is_scratch_value, _page_mappable, data_param_columns

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
        from perf.bench import _config_seed_int

        self.assertIsNone(_config_seed_int({}))
        self.assertIsNone(_config_seed_int(None))
        self.assertEqual(_config_seed_int({"seed": 42}), 42)
        self.assertEqual(
            _config_seed_int({"seed": "abc"}), _config_seed_int({"seed": "abc"})
        )

    def test_seeded_rng_reproducible(self):
        from perf.bench import _make_rng

        a = _make_rng({"seed": 7})
        b = _make_rng({"seed": 7})
        self.assertEqual([a.random() for _ in range(5)], [b.random() for _ in range(5)])

    def test_per_iter_data_deterministic_with_seed(self):
        from perf.bench import _make_rng, _per_iter_data

        models = [
            {"regs": {"rdi": 15}, "reads": [], "writes": []},
            {"regs": {"rdi": 3}, "reads": [], "writes": []},
        ]
        buf1, _ = _per_iter_data(models, 64, None, False, _make_rng({"seed": 9}))
        buf2, _ = _per_iter_data(models, 64, None, False, _make_rng({"seed": 9}))
        self.assertTrue((buf1 == buf2).all())

    def test_affinity_forms(self):
        from perf.core import _parse_affinity

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
        from perf.core import _parse_priority

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
        from perf.core import _numa_spec, _parse_numa

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
        from perf.bench import _validate_thread_top
        from perf.core import _check_thread

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
        import perf.core as core

        calls = []

        def _set(mode, node=None):
            calls.append((mode, node))
            return 0

        with (
            patch.object(core, "_set_mempolicy", side_effect=_set),
            patch.object(core, "_get_mempolicy", return_value=0),
            patch.object(core, "_numa_nodes", return_value=[0, 1]),
        ):
            with core._numa_guard({"thread": {"numa": 1}}):
                pass
            with core._numa_guard({}):
                pass
        self.assertEqual(calls, [(core._MPOL_BIND, 1), (0, None)])

    def test_numa_guard_warns_and_continues(self):
        import perf.core as core

        with (
            patch.object(core, "_set_mempolicy", return_value=-1),
            patch.object(core, "_get_mempolicy", return_value=0),
            patch.object(core, "_numa_nodes", return_value=[0, 1]),
            patch("warnings.warn") as warn,
        ):
            with core._numa_guard({"thread": {"numa": 1}}):
                pass
        self.assertTrue(warn.called)

    def test_numa_guard_skips_the_only_online_node(self):
        import perf.core as core

        with (
            patch.object(core, "_set_mempolicy") as setter,
            patch.object(core, "_numa_nodes", return_value=[0]),
            patch("warnings.warn") as warn,
        ):
            with core._numa_guard({"thread": {"numa": 0}}):
                pass
        setter.assert_not_called()
        warn.assert_not_called()

    def test_numa_guard_skips_an_offline_node(self):
        import perf.core as core

        with (
            patch.object(core, "_set_mempolicy") as setter,
            patch.object(core, "_numa_nodes", return_value=[0]),
            patch("warnings.warn") as warn,
        ):
            with core._numa_guard({"thread": {"numa": 3}}):
                pass
        setter.assert_not_called()
        self.assertIn("not online", str(warn.call_args.args[0]))

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_numa_affinity_defaults_to_the_node(self, mock_bench):
        with patch("perf.core._numa_cpus", return_value=[8, 9]):
            result = benchmark(
                config={"iterations": 8, "thread": {"numa": 1}},
                mode=["latency"],
                asm="nop",
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
                asm="nop",
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
                asm="nop",
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
                asm="nop",
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
                asm="nop",
                name="t",
                event=["topdown-*"],
            )
        self.assertEqual(
            mock_bench.call_args.kwargs["config"]["thread"][0]["affinity"], 0
        )

    def test_pin_default_affinity_leaves_swept_affinities_alone(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{"affinity": 1}], [{"affinity": 2}]]}
        with patch.object(core, "default_affinity") as fallback:
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
            asm="nop",
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
                asm="nop",
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

        from perf.core import _affinity_guard, _priority_guard

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
                asm="nop",
                name="t",
            )
        mock_bench.assert_not_called()

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bad_priority_fails_fast(self, mock_bench):
        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 10, "thread": {"priority": ["bogus"]}},
                mode=["latency"],
                asm="nop",
                name="t",
            )
        mock_bench.assert_not_called()


class TestFunctionOrder(unittest.TestCase):
    def test_resolve_orders(self):
        from perf.bench import _container

        self.assertEqual(_container({}, "func")["order"], "as-is")
        self.assertEqual(_container(None, "func")["order"], "as-is")
        self.assertEqual(
            _container({"func": {"order": "random"}}, "func")["order"], "random"
        )
        self.assertEqual(_container({"func": "random"}, "func")["order"], "random")
        self.assertEqual(
            _container({"func": {"order": "as-is"}}, "func")["order"], "as-is"
        )
        with self.assertRaises(ValueError):
            _container({"func": {"order": "bogus"}}, "func")

    def test_resolve_alignment(self):
        from perf.bench import _container

        self.assertEqual(_container({}, "func")["align"], 16)
        self.assertEqual(_container(None, "func")["align"], 16)
        self.assertEqual(_container({"func": {"align": 64}}, "func")["align"], 64)
        with self.assertRaises(ValueError):
            _container({"func": {"align": 24}}, "func")
        with self.assertRaises(ValueError):
            _container({"func": {"align": "many"}}, "func")

    def test_every_alternative_is_covered(self):
        from perf.bench import _container, _project_cache_id

        for key in ("code", "stack", "func"):
            with self.assertRaises(ValueError):
                _container({key: [{}, {}]}, key)
        ids = {
            _project_cache_id(
                "a.out",
                {"stack": {"size": size}, "code": {"align": align}},
            )
            for size in (0x1000, 0x2000)
            for align in (8, 16)
        }
        self.assertEqual(len(ids), 2)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bad_order_fails_fast(self, mock_bench):
        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 10, "func": {"order": ["bogus"]}},
                mode=["latency"],
                asm="nop",
                name="t",
            )
        mock_bench.assert_not_called()

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_random_order_shuffles_layout(
        self,
        mock_project,
        mock_elf,
        mock_get_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        import tempfile
        from pathlib import Path

        from perf.arch import x86_64

        mock_get_arch.return_value = x86_64
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        self.addCleanup(Path(path).unlink, missing_ok=True)

        benchmark(
            config={
                "iterations": 10,
                "samples": 2,
                "seed": [9],
                "func": {"order": ["random"], "align": [32]},
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
            mode=["latency"],
            symbol=f"{path}:func",
            name="func",
        )

        obj = mock_elf.return_value
        obj.randomize_layout.assert_called_once()
        _, kwargs = obj.randomize_layout.call_args
        self.assertEqual(kwargs["seed"], 9)
        self.assertEqual(kwargs["align"], 32)
        obj.write_perf_map.assert_called_once_with()

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_asis_order_keeps_layout(
        self,
        mock_project,
        mock_elf,
        mock_get_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        import tempfile
        from pathlib import Path

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
            symbol=f"{path}:func",
            name="func",
        )

        obj = mock_elf.return_value
        obj.randomize_layout.assert_not_called()
        obj.write_perf_map.assert_called_once_with()


class TestBranchSetup(unittest.TestCase):
    def test_global_distribution(self):
        from perf.bench import _branch_global_distribution

        self.assertEqual(_branch_global_distribution(None, True), "predictable")
        self.assertEqual(_branch_global_distribution(None, False), "unpredictable")

    def test_sample_index_predictable_cycles_in_order(self):
        from perf.bench import _branch_sample_index

        idx = [_branch_sample_index(3, None, "predictable", i) for i in range(7)]
        self.assertEqual(idx, [0, 1, 2, 0, 1, 2, 0])

    def test_sample_index_unpredictable_varies(self):
        import random

        from perf.bench import _branch_sample_index

        rng = random.Random(1)
        got = {_branch_sample_index(4, rng, "unpredictable", i) for i in range(60)}
        self.assertGreater(len(got), 1)

    def test_sample_values_cycles_when_predictable(self):
        from perf.bench import _sample_values

        vals = [1, 2, 3]
        got = [_sample_values(vals, "predictable", i, None) for i in range(4)]
        self.assertEqual(got, [1, 2, 3, 1])

    def test_config_prediction_reaches_the_sampler(self):
        import random

        from perf.bench import _branch_config, _per_iter_data

        models = [
            {"regs": {"rdi": 15}, "reads": [], "writes": []},
            {"regs": {"rdi": 3}, "reads": [], "writes": []},
        ]
        for prediction, expected in (
            ("predictable", [15, 3, 15, 3]),
            ("unpredictable", None),
        ):
            cfg = _branch_config({"branch": prediction})
            buf, meta = _per_iter_data(
                models, 4, None, None, random.Random(1), branch_cfg=cfg
            )
            k = int(meta["k"])
            got = [int(buf[0, k - 1 - i]) for i in range(4)]
            if expected is not None:
                self.assertEqual(got, expected)
            else:
                self.assertEqual(set(got), {3, 15})

    def test_per_address_override_wins_for_that_address(self):
        import random

        from perf.bench import _branch_config, _per_iter_data

        models = [
            {
                "regs": {"rdi": 15},
                "reads": [],
                "writes": [],
                "blocks": [0x401000],
            }
        ]
        cfg = _branch_config(
            {
                "branch": {
                    "prediction": "unpredictable",
                    "mem": {"0x401000": "predictable"},
                }
            }
        )
        self.assertEqual(cfg["mem"][0x401000], "predictable")
        buf, meta = _per_iter_data(
            models, 6, None, None, random.Random(1), branch_cfg=cfg
        )
        k = int(meta["k"])
        got = [int(buf[0, k - 1 - i]) for i in range(6)]
        self.assertTrue(all(v == 15 for v in got))

    def test_unknown_prediction_rejected(self):
        from perf.bench import _branch_config

        for bad in ("sometimes", "always", ""):
            with self.assertRaises(ValueError):
                _branch_config({"branch": bad})


class TestCodeAlign(unittest.TestCase):
    @staticmethod
    def _offset(align):
        from perf.bench import _with_align, get_arch

        arch = get_arch()
        prefix = "push rbx; push rbp; mov r8, [rdi];"
        code = "add eax, 42;"
        stream = prefix + _with_align(code, align)
        blob = arch.assemble(stream)
        return blob.find(arch.assemble(code), len(arch.assemble(prefix)) - 1)

    def test_align_directive(self):
        from perf.bench import _align_directive

        self.assertEqual(_align_directive(1), "")
        self.assertEqual(_align_directive(2), ".align 2")
        self.assertEqual(_align_directive(16), ".align 16")
        self.assertEqual(_align_directive(64), ".align 64")
        self.assertEqual(_align_directive(None), "")

    def test_align_one_emits_no_padding(self):
        from perf.bench import _with_align

        self.assertEqual(_with_align("nop;", 1), "nop;")
        self.assertEqual(_with_align("nop;", 16), ".align 16\nnop;")

    def test_measured_code_lands_on_the_boundary(self):
        for align in (2, 4, 8, 16, 32, 64):
            with self.subTest(align=align):
                self.assertEqual(self._offset(align) % align, 0)

    def test_align_one_is_the_unpadded_native_offset(self):
        prefix_len = self._offset(1)
        self.assertEqual(prefix_len, self._offset(1))
        self.assertNotEqual(prefix_len % 16, 0)

    def test_default_sweeps_one_and_sixteen(self):
        from perf.bench import _DEFAULT_BENCH, _container

        self.assertEqual(_DEFAULT_BENCH["code"], [{"align": 1}, {"align": 16}])
        self.assertEqual(_container({"code": {"align": 1}}, "code")["align"], 1)
        self.assertEqual(_container({"code": {"align": 16}}, "code")["align"], 16)

    def test_align_must_be_power_of_two(self):
        from perf.bench import _container

        for bad in (0, 3, 24, -1):
            with self.assertRaises(ValueError):
                _container({"code": {"align": bad}}, "code")


def imm_of(value):
    from perf.arch import x86_64

    return x86_64.imm(value)


class TestCodeTierDefaults(unittest.TestCase):
    CODE = [0x401000, 0x401100]
    DATA = [0x402000, 0x402100]

    def _steer(self, cfg, code=None, data=None):
        import random

        from perf.bench import (
            _apply_code_tier_defaults,
            _fill_evict_tables,
            _per_iter_data,
            get_arch,
        )

        arch = get_arch()
        code = self.CODE if code is None else code
        data = self.DATA if data is None else data
        mem_levels = _apply_code_tier_defaults(arch, cfg, {}, data, code)
        buf, meta = _per_iter_data(
            [{"regs": {}, "reads": [], "writes": []}],
            4,
            None,
            False,
            random.Random(1),
            mem_levels=mem_levels,
            extra_addrs=set(data),
        )
        _fill_evict_tables(buf, meta, buf.ctypes.data)
        return arch.steer_asm(meta, buf.ctypes.data)

    def test_icache_maps_to_l1i_on_code_addresses(self):
        cold = self._steer({"icache": "cold"})
        hot = self._steer({"icache": "hot"})
        self.assertEqual(cold.count("clflushopt"), len(self.CODE))
        self.assertEqual(hot.count("clflushopt"), 0)

    def test_icache_never_touches_data_references(self):
        cold = self._steer({"icache": "cold"})
        for addr in self.DATA:
            self.assertIn(f"mov rax, [{imm_of(addr)}]", cold)
            self.assertNotIn(f"clflushopt [{imm_of(addr)}]", cold)
        for addr in self.CODE:
            self.assertIn(f"mov r12, {imm_of(addr)}", cold)

    def test_itlb_steers_code_addresses_and_dtlb_data_references(self):
        code = self._steer({"itlb": "cold"})
        self.assertIn("syscall", code)
        self.assertNotIn("syscall", self._steer({"itlb": "hot"}))
        data = self._steer({"dtlb": "cold"})
        self.assertIn("syscall", data)
        self.assertNotIn("syscall", self._steer({"dtlb": "hot"}))

    def test_dcache_steers_data_references(self):
        cold = self._steer({"dcache": "cold"})
        self.assertIn(imm_of(self.DATA[0]), cold)
        self.assertNotIn(imm_of(self.CODE[0]), cold)

    def test_no_tier_config_emits_no_extra_eviction(self):
        asm = self._steer({})
        self.assertNotIn("clflushopt", asm)

    def test_code_addresses_are_not_evicted_twice(self):
        asm = self._steer({"icache": "cold", "dcache": "cold"})
        self.assertEqual(asm.count("clflushopt"), len(self.CODE) + len(self.DATA))

    def test_code_addresses_are_not_stored_to(self):
        import random

        from perf.bench import _apply_code_tier_defaults, _per_iter_data, get_arch

        arch = get_arch()
        mem_levels = _apply_code_tier_defaults(
            arch, {"icache": "cold"}, {}, self.DATA, self.CODE
        )
        _buf, meta = _per_iter_data(
            [{"regs": {}, "reads": [], "writes": []}],
            4,
            None,
            False,
            random.Random(1),
            mem_levels=mem_levels,
            extra_addrs=set(self.DATA),
        )
        self.assertEqual(list(meta["l1i_addrs"]), self.CODE)
        self.assertEqual(list(meta["mem_addrs"]), self.DATA)


class TestStaticTableAddrs(unittest.TestCase):
    ASM = (
        "lea rax, [rip + 0x20]\nmov rbx, [rip + 0x28]\njmp qword ptr [rip + 0x30]\nret"
    )
    BASE = 0x1000

    def _project(self):
        import angr

        from perf.arch import arch as get_arch

        arch = get_arch()
        return angr.load_shellcode(
            arch.assemble(self.ASM, self.BASE),
            arch=arch._ANGR_ARCH_NAME,
            load_address=self.BASE,
        )

    def _refs(self):
        from perf.arch import arch as get_arch

        arch = get_arch()
        md = arch.disassembler()
        md.detail = True
        out = {}
        for insn in md.disasm(arch.assemble(self.ASM, self.BASE), self.BASE):
            for ref in arch.mem_refs(insn):
                out[insn.mnemonic] = ref
        return out

    def _addrs(self, solutions):
        from perf.bench import _static_table_addrs

        return set(
            _static_table_addrs(
                self._project(),
                self.BASE,
                self.BASE + len(self.ASM),
                None,
                solutions=solutions,
            )
        )

    def test_only_recorded_data_is_kept(self):
        data, control = self._refs()["mov"], self._refs()["jmp"]
        sol = [{"data_addrs": {data}, "control_addrs": {control}}]
        self.assertEqual(self._addrs(sol), {data})

    def test_a_data_read_wins_over_a_control_flow_read(self):
        data, control = self._refs()["mov"], self._refs()["jmp"]
        sol = [{"data_addrs": {data, control}, "control_addrs": set()}]
        self.assertEqual(self._addrs(sol), {data, control})

    def test_nothing_recorded_keeps_nothing(self):
        self.assertEqual(self._addrs(None), set())
        self.assertEqual(
            self._addrs([{"data_addrs": set(), "control_addrs": set()}]), set()
        )

    def test_recorded_stats_split_data_from_control_flow(self):
        import angr

        from perf.arch import arch as get_arch
        from perf.bench import (
            _recorded_stats,
            _track_branches,
            _track_mem_access,
        )

        arch = get_arch()
        blob = arch.assemble(self.ASM, self.BASE)
        end = self.BASE + len(blob)
        proj = angr.load_shellcode(
            blob, arch=arch._ANGR_ARCH_NAME, load_address=self.BASE
        )
        state = proj.factory.blank_state(addr=self.BASE)
        arch.map_stack(state)
        state.memory.store(self._refs()["jmp"], end.to_bytes(8, "little"))
        _track_mem_access(state)
        _track_branches(state)
        simgr = proj.factory.simgr(state)
        simgr.explore(find=[end])
        found = simgr.found + simgr.active + simgr.deadended
        self.assertTrue(found)
        data, control, branches = _recorded_stats(found[0])
        refs = self._refs()
        self.assertIn(refs["mov"], data)
        self.assertNotIn(refs["mov"], control)
        self.assertIn(refs["jmp"], control)
        self.assertNotIn(refs["jmp"], data)
        self.assertIn(end, branches)
        self.assertNotIn(refs["lea"], data | control)


class TestConfigShapes(unittest.TestCase):
    def test_seed_defaults_to_zero_and_is_scalar(self):
        from perf.bench import _DEFAULT_BENCH, _config_seed_int

        self.assertEqual(_DEFAULT_BENCH["seed"], 0)
        self.assertNotIsInstance(_DEFAULT_BENCH["seed"], list)
        self.assertEqual(_config_seed_int({"seed": 0}), 0)
        self.assertIsNone(_config_seed_int({"seed": None}))

    def test_iterations_config_forms(self):
        from perf.bench import _iterations_config

        self.assertEqual(_iterations_config({"iterations": 100}), (100, None, None))
        self.assertEqual(
            _iterations_config({"iterations": {"min": 64, "max": 9999}}),
            (None, 64, 9999),
        )
        self.assertEqual(_iterations_config({}), (None, None, None))

    def test_backend_keys_are_per_backend(self):
        from perf.bench import _backend_option, _resolve_probe_runs

        cfg = {
            "backend": {
                "loop": {"probes": 9, "target": 0.01},
                "unroll": {"count": 4, "probes": 2, "target": 0.5},
            }
        }
        self.assertEqual(_backend_option(cfg, "loop", "probes"), 9)
        self.assertEqual(_backend_option(cfg, "unroll", "probes"), 2)
        self.assertEqual(_backend_option(cfg, "unroll", "target"), 0.5)
        self.assertIsNone(_backend_option(cfg, "loop", "count"))
        self.assertEqual(_resolve_probe_runs(cfg, "loop", 4), 4)
        self.assertEqual(_resolve_probe_runs(cfg, "unroll", 10), 2)

    def test_external_lib_is_a_switch(self):
        from perf.bench import _resolve_lib, _validate_lib

        self.assertFalse(_resolve_lib({"external": {"lib": False}}))
        self.assertTrue(_resolve_lib({"external": {"lib": True}}))
        self.assertFalse(_resolve_lib({}))
        self.assertFalse(_resolve_lib({"external": {}}))
        _validate_lib({"external": {"lib": "yes"}})
        with self.assertRaises(ValueError):
            _validate_lib({"external": {"nope": 1}})
        with self.assertRaises(ValueError):
            _validate_lib({"external": {"lib": 1.5}})


class TestDefaultBench(unittest.TestCase):
    def test_all_keys_have_defaults(self):
        from perf.bench import _DEFAULT_BENCH

        self.assertEqual(_DEFAULT_BENCH["seed"], 0)
        self.assertEqual(
            _DEFAULT_BENCH["thread"],
            [[{"numa": None, "affinity": None, "priority": "normal"}]],
        )
        self.assertEqual(_DEFAULT_BENCH["branch"], ["predictable", "unpredictable"])
        self.assertEqual(_DEFAULT_BENCH["dcache"], ["hot", "warm", "cool", "cold"])
        self.assertEqual(_DEFAULT_BENCH["icache"], ["hot", "cold"])
        self.assertEqual(_DEFAULT_BENCH["dtlb"], ["hot", "cold"])
        self.assertEqual(_DEFAULT_BENCH["itlb"], ["hot", "cold"])
        self.assertEqual(_DEFAULT_BENCH["code"], [{"align": 1}, {"align": 16}])
        self.assertEqual(
            _DEFAULT_BENCH["external"],
            {"lib": False, "stdout": False, "stderr": False},
        )
        self.assertEqual(_DEFAULT_BENCH["func"], [{"align": 16, "order": "as-is"}])
        self.assertEqual(_DEFAULT_BENCH["samples"], 100)
        self.assertEqual(_DEFAULT_BENCH["iterations"], {"min": 100, "max": 1_000_000})
        self.assertEqual(
            _DEFAULT_BENCH["backend"],
            {
                "loop": {"probes": 3, "runs": 10, "target": 0.005},
                "unroll": {"count": 5, "probes": 3, "runs": 10, "target": 0.005},
            },
        )
        self.assertNotIn("data", _DEFAULT_BENCH)
        for gone in (
            "lib",
            "runs",
            "min_iterations",
            "max_iterations",
            "target_rel_se",
            "probe_runs",
            "unroll_n",
        ):
            self.assertNotIn(gone, _DEFAULT_BENCH)

    def test_branch_default_sweeps_both_alternatives(self):
        from perf.bench import _BRANCH_CHOICES, _concrete_configs

        _spec, defaults = _concrete_configs({})
        self.assertEqual(_DEFAULT_BENCH["branch"], list(_BRANCH_CHOICES))
        self.assertEqual({c["branch"] for c in defaults}, set(_BRANCH_CHOICES))
        _spec, explicit = _concrete_configs({"branch": list(_BRANCH_CHOICES)})
        self.assertEqual(defaults, explicit)

    def test_none_backend_and_unroll_fall_back(self):
        from perf.bench import _DEFAULT_BENCH as _DB2
        from perf.bench import _resolve_backend, _resolve_unroll_n

        self.assertEqual(_resolve_backend(None, {"backend": None}, "loop"), "loop")
        self.assertEqual(_resolve_backend(None, {"backend": None}, "unroll"), "unroll")
        self.assertEqual(
            _resolve_unroll_n(None, {"backend": {"unroll": {"count": None}}}),
            _DB2["backend"]["unroll"]["count"],
        )


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
            {"probes": 3, "runs": 3, "target": 0.5},
        )
        _select_backend(cfg, "loop")
        self.assertEqual(
            cfg["backend"],
            {"loop": {"probes": 3, "runs": 3, "target": 0.5}},
        )
        self.assertNotIn("backend_options", cfg)

    def test_used_options_include_the_unroll_factor(self):
        from perf.bench import _used_backend_options

        used = _used_backend_options({"backend": {"unroll": {"count": 7}}}, "unroll")
        self.assertEqual(used["count"], 7)
        self.assertNotIn("count", _used_backend_options({}, "loop"))

    def test_selected_options_stay_effective(self):
        from perf.bench import _resolve_runs, _select_backend

        cfg = {"backend": {"loop": {"runs": 3}}}
        self.assertEqual(_resolve_runs(cfg, "loop"), 3)
        _select_backend(cfg, "loop")
        self.assertEqual(_resolve_runs(cfg, "loop"), 3)
        self.assertEqual(_resolve_runs(cfg, "unroll"), 10)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_backend_is_in_the_output(self, mock_bench):
        result = benchmark(
            config={"iterations": 64, "samples": 2, "branch": "unpredictable"},
            mode=["latency"],
            asm="nop",
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
            asm="nop",
            name="t",
        )
        rec = result.reset_index()
        self.assertEqual(rec["config.backend.loop.runs"].tolist(), [4] * len(rec))
        self.assertEqual(rec["config.backend.loop.target"].tolist(), [0.5] * len(rec))
        self.assertEqual(
            mock_bench.call_args.kwargs["config"]["backend"],
            {"loop": {"probes": 3, "runs": 4, "target": 0.5}},
        )

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_json_output_repeats_the_backend(self, mock_bench):
        result = benchmark(
            config={"iterations": 64, "samples": 2},
            mode=["latency"],
            asm="nop",
            name="t",
            backend="loop",
        )
        from perf.bench import to_json

        row = json.loads(to_json(result))["output"][0]
        self.assertEqual(
            row["config"]["backend"],
            {"loop": {"probes": 3, "runs": 10, "target": 0.005}},
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
                asm="nop",
                name="t",
            )
        self.assertEqual(
            result.attrs["config"]["backend"],
            {"loop": {"probes": 3, "runs": 10, "target": 0.005}},
        )

    def test_resolve_unroll_n(self):
        from perf.bench import _DEFAULT_BENCH as _DB3
        from perf.bench import _resolve_unroll_n

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
            asm="nop",
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
            asm="nop",
            name="mytarget",
        )
        calls = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertTrue(all("nop" in c for c in calls) or len(calls) == 2)
        self.assertEqual(result["operations"].tolist(), [1000, 1000])
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
            asm="mov eax, 42",
            name="mytarget",
        )
        rec = result.reset_index()
        self.assertEqual(
            sorted(rec["mode"].unique().tolist()), ["latency", "throughput"]
        )
        self.assertEqual(rec[rec["mode"] == "latency"]["operations"].tolist(), [1, 1])
        self.assertEqual(
            rec[rec["mode"] == "throughput"]["operations"].tolist(),
            [1000, 1000],
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
            asm="nop",
            name="mytarget",
        )
        self.assertEqual(
            result.attrs["config"]["backend"],
            {"loop": {"probes": 3, "runs": 10, "target": 0.005}},
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertEqual(codes[1], ".align 16\nnop;")

    def test_unroll_rejected_for_throughput(self):
        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 100, "samples": 1},
                mode=["throughput"],
                asm="nop",
                name="t",
                backend="unroll",
            )

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
            asm="nop",
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
            asm="nop",
            name="mytarget",
            backend="loop",
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertEqual(codes[1], ".align 16\nnop;")
        self.assertEqual(
            result["duration_time"].tolist(),
            _ticks_to_ns(result, [10, 20, 30]),
        )

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
            asm="nop",
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
                asm="nop",
                name="t",
                backend="turbo",
            )

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_file_unroll_duplicates_call(
        self,
        mock_project,
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
            symbol=f"{path}:func",
            name="func",
            backend="unroll",
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertEqual(len(codes), 2)
        n_copies = codes[0].count("call rax")
        self.assertEqual(n_copies, 3)
        self.assertEqual(codes[1].count("call rax"), 6)
        freq = result.attrs["info"]["cpu"]["freq"]
        self.assertAlmostEqual(
            result["duration_time"].tolist()[0], (50 / 3) * 1e9 / freq
        )

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_file_loop_is_default_single_call(
        self,
        mock_project,
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
            symbol=f"{path}:func",
            name="func",
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        measured = [c for c in codes if "call rax" in c]
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0].count("call rax"), 1)

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_file_throughput_restores_regs_before_each_call(
        self,
        mock_project,
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
            symbol=f"{path}:func",
            name="func",
            data={"rdi": 0x5000, "rsi": 16},
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        measured = [c for c in codes if "call rax" in c]
        self.assertEqual(len(measured), 1)
        self.assertIn("mov rdi, 0x5000", measured[0])
        self.assertIn("mov rsi, 0x10", measured[0])
        self.assertLess(measured[0].index("mov rdi"), measured[0].index("call rax"))

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_file_latency_keeps_regs_out_of_loop_code(
        self,
        mock_project,
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
            symbol=f"{path}:func",
            name="func",
            data={"rdi": 0x5000, "rsi": 16},
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        measured = [c for c in codes if "call rax" in c]
        self.assertEqual(len(measured), 1)
        self.assertNotIn("mov rdi", measured[0])

    def test_data_script_uses_combined_steer(self):
        import random

        import keystone

        from perf.bench import (
            _arch_asm,
            _fill_evict_tables,
            _per_iter_data,
        )

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


class TestExploreAsm(unittest.TestCase):
    def test_solver_maps_scratch_without_data(self):
        from perf.bench import explore_asm, solve

        sols = explore_asm("add r11, [rax];", data={"regs": {}, "mem": {}})
        self.assertTrue(sols)
        models = solve(sols)
        self.assertTrue(models)
        addrs = [a for m in models for a, _, _ in m["reads"] + m["writes"]]
        self.assertTrue(addrs)
        for a in addrs:
            self.assertGreater(a, 0x10000)

    def test_solver_respects_explicit_rax(self):
        from perf.bench import explore_asm, solve

        data = {"regs": {"rax": 0x400000}, "mem": {"0x400000": 100}}
        sols = explore_asm("add r11, [rax];", data=data)
        self.assertTrue(sols)
        models = solve(sols)
        addrs = [a for m in models for a, _, _ in m["reads"] + m["writes"]]
        self.assertIn(0x400000, addrs)

    def test_nop_needs_no_models(self):
        from perf.bench import explore_asm

        sols = explore_asm("nop;", data={"regs": {}, "mem": {}})
        self.assertTrue(
            all(
                not s.get("reg") and not s.get("reads") and not s.get("writes")
                for s in sols
            )
        )

    def test_idiv_needs_explicit_state(self):
        from perf.bench import explore_asm

        data = {"regs": {"eax": 100, "edx": 0, "ecx": 42}, "mem": {}}
        sols = explore_asm("idiv ecx;", data=data)
        self.assertTrue(sols)


class TestExplicitMemSteering(unittest.TestCase):
    def test_explicit_mem_merged_without_models(self):
        import random

        from perf.bench import _per_iter_data

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

    def test_explicit_value_overrides_solver(self):
        import random

        from perf.bench import _per_iter_data

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

        from perf.bench import (
            _arch_asm,
            _fill_evict_tables,
            _per_iter_data,
        )

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
                asm="idiv eax, 42",
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
        from perf.bench import _DEFAULT_BENCH as _DB3
        from perf.bench import _resolve_unroll_n

        self.assertEqual(
            _resolve_unroll_n(None, {}), _DB3["backend"]["unroll"]["count"]
        )
        self.assertEqual(
            _resolve_unroll_n(None, {"backend": {"unroll": {"count": 1000}}}), 1000
        )
        self.assertEqual(
            _resolve_unroll_n(7, {"backend": {"unroll": {"count": 1000}}}), 7
        )


class TestConfigData(unittest.TestCase):
    def test_normalize_regs_and_mem(self):
        from perf.bench import _config_data

        data = _config_data({"data": {"rdi": 21, "0x1234": 5}})
        self.assertEqual(data["regs"]["rdi"], 21)
        self.assertEqual(data["mem"]["0x1234"], 5)

    def test_flat_rdi_maps_to_regs(self):
        from perf.bench import _config_data

        data = _config_data({"data": {"rdi": 21}})
        self.assertEqual(data["regs"]["rdi"], 21)

    def test_memory_array_expands(self):
        from perf.bench import _config_data

        data = _config_data({"data": {"0x1234": [1, 2, 3]}})
        self.assertEqual(data["mem"]["0x1234"], [1, 2, 3])

        data = _config_data({"data": {"0x1234:": [1, 2, 3]}})
        self.assertEqual(data["mem"]["0x1234"], 1)
        self.assertEqual(data["mem"]["0x1235"], 2)
        self.assertEqual(data["mem"]["0x1236"], 3)

    def test_memory_namespace_array_expands(self):
        from perf.bench import _config_data

        data = _config_data({"data": {"0x1234:": [1, 2, 3, 3, 4, 4, 5]}})
        self.assertEqual(len(data["mem"]), 7)
        self.assertEqual(data["mem"]["0x1234"], 1)
        data = _config_data({"data": {"0x1234": [1, 2, 3]}})
        self.assertEqual(data["mem"]["0x1234"], [1, 2, 3])

    def test_memory_byte_list_expands(self):
        from perf.bench import _config_data

        data = _config_data({"data": {"0x10000000:": [3, 1, 2, "0x4", "5"]}})
        self.assertEqual(
            data["mem"],
            {
                "0x10000000": 3,
                "0x10000001": 1,
                "0x10000002": 2,
                "0x10000003": 4,
                "0x10000004": 5,
            },
        )

    def test_flat_mem_distribution_and_range(self):
        from perf.bench import _config_data

        data = _config_data({"data": {"0x1234": [1, 2, 3]}})
        self.assertEqual(data["mem"]["0x1234"], [1, 2, 3])
        data = _config_data({"data": {"0x1234:": [1, 2, 3]}})
        self.assertEqual(data["mem"]["0x1235"], 2)
        data = _config_data({"data": {"rdi": [1, 2, 3]}})
        self.assertEqual(data["regs"]["rdi"], [1, 2, 3])

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bench_data_flows(self, mock_bench):
        benchmark(
            config={
                "iterations": 1000,
                "samples": 3,
            },
            data={"rdi": 21},
            mode=["latency"],
            asm="nop",
            name="t",
        )
        for call in mock_bench.call_args_list:
            self.assertEqual(call.kwargs["data"]["regs"]["rdi"], 21)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bench_config_data_rejected(self, mock_bench):
        with self.assertRaises(ValueError):
            benchmark(
                config={
                    "iterations": 1000,
                    "samples": 3,
                    "data": {"regs": {"rdi": 21}},
                },
                mode=["latency"],
                asm="nop",
                name="t",
            )
        mock_bench.assert_not_called()


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
            asm="nop",
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
            asm="nop",
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
            asm="nop",
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
            asm="nop",
            name="t",
        )
        self.assertNotEqual(result.attrs["id"], result3.attrs["id"])


class TestBranchConfig(unittest.TestCase):
    def test_string_form(self):
        from perf.bench import _branch_config

        cfg = _branch_config({"branch": "predictable"})
        self.assertEqual(cfg, {"prediction": "predictable", "mem": {}, "regs": {}})

    def test_dict_form(self):
        from perf.bench import _branch_config

        cfg = _branch_config(
            {"branch": {"prediction": "unpredictable", "mem": {"0x401234": True}}}
        )
        self.assertEqual(cfg["prediction"], "unpredictable")
        self.assertEqual(cfg["mem"], {0x401234: "predictable"})

    def test_hex_and_decimal_addresses(self):
        from perf.bench import _branch_config

        cfg = _branch_config(
            {"branch": {"mem": {"0x401234": "predictable", "4200004": False}}}
        )
        self.assertIn(0x401234, cfg["mem"])
        self.assertIn(4200004, cfg["mem"])
        self.assertEqual(cfg["mem"][4200004], "unpredictable")

    def test_unset_returns_none(self):
        from perf.bench import _branch_config

        self.assertIsNone(_branch_config({}))
        self.assertIsNone(_branch_config(None))

    def test_default_with_address_overrides(self):
        from perf.bench import _branch_config

        cfg = _branch_config(
            {
                "branch": {
                    "prediction": "predictable",
                    "0x03132": "predictable",
                    "0x03134": "unpredictable",
                }
            }
        )
        self.assertEqual(cfg["prediction"], "predictable")
        self.assertEqual(cfg["mem"], {0x3132: "predictable", 0x3134: "unpredictable"})

    def test_default_with_prediction_combined(self):
        from perf.bench import _branch_config

        cfg = _branch_config(
            {
                "branch": {
                    "prediction": "unpredictable",
                    "mem": {"0x401234": True},
                }
            }
        )
        self.assertEqual(cfg["prediction"], "unpredictable")
        self.assertEqual(cfg["mem"], {0x401234: "predictable"})

    def test_unknown_key_rejected(self):
        from perf.bench import _branch_config

        with self.assertRaises(ValueError):
            _branch_config({"branch": {"default": "predictable"}})

    def test_invalid_rejected(self):
        from perf.bench import _branch_config

        with self.assertRaises(ValueError):
            _branch_config({"branch": "sometimes"})
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"prediction": "sometimes"}})
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"mem": {"0x1": "maybe"}}})
        with self.assertRaises(ValueError):
            _branch_config({"branch": [1, 2]})


class TestCondBranchAnalysis(unittest.TestCase):
    def test_finds_controlling_regs_and_mem(self):
        import capstone

        from perf.bench import _cond_branch_analysis

        code = bytes.fromhex("488b4d004989c04839d97f05")
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        insns = list(md.disasm(code, 0x401175))
        regs, mems = _cond_branch_analysis(insns)
        self.assertIn("rcx", regs)
        self.assertIn("rbx", regs)
        self.assertTrue(mems)
        base, index, scale, disp = mems[0]

        self.assertTrue(base)

    def test_unconditional_block_returns_empty(self):
        import capstone

        from perf.bench import _cond_branch_analysis

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        insns = list(md.disasm(bytes.fromhex("4883c101ebf8"), 0x1000))
        regs, mems = _cond_branch_analysis(insns)
        self.assertEqual(regs, set())
        self.assertEqual(mems, [])


class TestPerIterDataBranchPinning(unittest.TestCase):
    def test_pins_regs_for_predictable_address(self):
        import random

        from perf.bench import _per_iter_data

        models = [
            {
                "regs": {"rcx": 5, "rsi": 1},
                "reads": [],
                "writes": [],
                "blocks": [0x401000],
                "branch_deps": {
                    0x401000: {"regs": {"rcx"}, "mem": set()},
                },
            },
            {
                "regs": {"rcx": 99, "rsi": 2},
                "reads": [],
                "writes": [],
                "blocks": [0x401000],
                "branch_deps": {
                    0x401000: {"regs": {"rcx"}, "mem": set()},
                },
            },
        ]
        branch_cfg = {"prediction": "unpredictable", "mem": {0x401000: "predictable"}}
        buf, meta = _per_iter_data(
            models,
            10,
            {"L1d": 100, "L2": 100, "L3": 100},
            False,
            random.Random(7),
            branch_cfg=branch_cfg,
        )

        col = meta["reg_col"]["rcx"]
        vals = {int(buf[col, i]) for i in range(1, int(meta["k"]))}
        self.assertEqual(vals, {5})

        col_si = meta["reg_col"]["rsi"]
        vals_si = {int(buf[col_si, i]) for i in range(1, int(meta["k"]))}
        self.assertGreater(len(vals_si), 1)

    def test_pins_mem_values_for_predictable_address(self):
        import random

        from perf.bench import _per_iter_data

        models = [
            {
                "regs": {"rax": 0x41000000000},
                "reads": [(0x41000000000, 8, 1)],
                "writes": [],
                "blocks": [0x401000],
                "branch_deps": {
                    0x401000: {
                        "regs": {"rax"},
                        "mem": {0x41000000000},
                    },
                },
            },
            {
                "regs": {"rax": 0x41000000000},
                "reads": [(0x41000000000, 8, 42)],
                "writes": [],
                "blocks": [0x401000],
                "branch_deps": {
                    0x401000: {
                        "regs": {"rax"},
                        "mem": {0x41000000000},
                    },
                },
            },
        ]
        branch_cfg = {"prediction": "unpredictable", "mem": {0x401000: "predictable"}}
        buf, meta = _per_iter_data(
            models,
            10,
            {"L1d": 100, "L2": 100, "L3": 100},
            False,
            random.Random(7),
            branch_cfg=branch_cfg,
        )
        a = 0x41000000000
        col = meta["val_col"][a]
        vals = {int(buf[col, i]) for i in range(1, int(meta["k"]))}
        self.assertEqual(vals, {1})

    def test_unpinned_branches_keep_sampling(self):
        import random

        from perf.bench import _per_iter_data

        models = [
            {
                "regs": {"rax": 1},
                "reads": [],
                "writes": [],
                "blocks": [0x401000],
                "branch_deps": {0x401000: {"regs": {"rax"}, "mem": set()}},
            },
            {
                "regs": {"rax": 2},
                "reads": [],
                "writes": [],
                "blocks": [0x401000],
                "branch_deps": {0x401000: {"regs": {"rax"}, "mem": set()}},
            },
        ]
        branch_cfg = {"prediction": "unpredictable", "mem": {0x401000: "unpredictable"}}
        buf, meta = _per_iter_data(
            models,
            10,
            {"L1d": 100, "L2": 100, "L3": 100},
            False,
            random.Random(7),
            branch_cfg=branch_cfg,
        )
        col = meta["reg_col"]["rax"]
        vals = {int(buf[col, i]) for i in range(1, int(meta["k"]))}
        self.assertGreater(len(vals), 1)


class TestDeepMergeConfig(unittest.TestCase):
    def test_nested_per_address_cache_preserves_defaults(self):
        from perf.bench import _branch_config, _merge_config

        cfg = _merge_config(
            {
                "dcache": {
                    "L1d": {"hit_rate": 100},
                    "0x41000000000": "hot",
                },
                "branch": {"mem": {"0x401000": "predictable"}},
            }
        )
        mc = cfg["dcache"]
        self.assertEqual(mc["L1d"], {"hit_rate": 100})
        self.assertEqual(mc["0x41000000000"], "hot")
        cfgb = _branch_config(cfg)
        self.assertEqual(cfgb["prediction"], "unpredictable")
        self.assertEqual(cfgb["mem"], {0x401000: "predictable"})

    def test_scalar_override_wins(self):
        from perf.bench import _merge_config

        cfg = _merge_config({"branch": "predictable", "samples": 5})
        self.assertEqual(cfg["branch"], "predictable")
        self.assertEqual(cfg["samples"], 5)
        self.assertEqual(cfg["dcache"], ["hot", "warm", "cool", "cold"])
        self.assertEqual(cfg["icache"], ["hot", "cold"])

    def test_cache_shortcut_forms(self):
        from perf.bench import _expand_config_spec, _merge_config

        self.assertEqual(_merge_config({"dcache": "cold"})["dcache"], "cold")
        self.assertEqual(
            _merge_config({"dcache": ["cold", "hot"]})["dcache"], ["cold", "hot"]
        )
        combos = _expand_config_spec(
            _merge_config(
                {
                    "dcache": "cold",
                    "branch": "predictable",
                    "icache": "hot",
                    "dtlb": "hot",
                    "itlb": "hot",
                    "code": {"align": 16},
                }
            )
        )
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0]["dcache"], "cold")

    def test_non_dict_config_rejected(self):
        from perf.bench import _merge_config, validate_spec

        with self.assertRaises(ValueError):
            _merge_config(["bad"])
        with self.assertRaises(ValueError):
            validate_spec(["bad"])

    def test_unknown_top_key_rejected(self):
        from perf.bench import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"bogus": 1})

    def test_bench_rejects_non_dict_config(self):
        with self.assertRaises(ValueError):
            benchmark(asm="mov eax, 42", mode=["latency"], config=["bad"])

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            benchmark(symbol=["/nonexistent/a.out", "func"], mode=["latency"], event=[])

    def test_binary_without_target_raises(self):
        with self.assertRaises(ValueError):
            benchmark(symbol=["/bin/true"], mode=["latency"], event=[])


class TestContainerConfigFormat(unittest.TestCase):
    def test_defaults_keep_the_list_on_the_first_level(self):
        from perf.bench import _DEFAULT_BENCH

        self.assertEqual(
            _DEFAULT_BENCH["thread"],
            [[{"numa": None, "affinity": None, "priority": "normal"}]],
        )
        self.assertEqual(_DEFAULT_BENCH["code"], [{"align": 1}, {"align": 16}])
        self.assertEqual(_DEFAULT_BENCH["func"], [{"align": 16, "order": "as-is"}])
        self.assertEqual(_DEFAULT_BENCH["stack"], [{"size": 0x200000, "align": 16}])

    def test_shorthands_normalize_to_the_list_form(self):
        from perf.bench import normalize_container

        self.assertEqual(normalize_container("code", 32), [{"align": 32}])
        self.assertEqual(normalize_container("code", {"align": 32}), [{"align": 32}])
        self.assertEqual(normalize_container("func", "random"), [{"order": "random"}])
        self.assertEqual(
            normalize_container("stack", {"size": 4096, "align": 16}),
            [{"size": 4096, "align": 16}],
        )
        self.assertEqual(
            normalize_container("thread", {"affinity": 1}), [[{"affinity": 1}]]
        )
        self.assertEqual(
            normalize_container("thread", [{"affinity": 1}, {"affinity": 2}]),
            [[{"affinity": 1}], [{"affinity": 2}]],
        )
        self.assertEqual(
            normalize_container("thread", [[{"affinity": 1}]]), [[{"affinity": 1}]]
        )

    def test_shorthand_dict_form_still_works(self):
        from perf.bench import _concrete_configs

        _spec, combos = _concrete_configs(
            {
                "code": {"align": 32},
                "func": {"order": "random"},
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
            }
        )
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0]["code"], {"align": 32})
        self.assertEqual(combos[0]["func"], {"order": "random"})

    def test_concrete_thread_is_the_run_thread_list(self):
        from perf.bench import _concrete_configs

        _spec, combos = _concrete_configs(None)
        self.assertEqual(
            combos[0]["thread"],
            [{"numa": None, "affinity": None, "priority": "normal"}],
        )

    def test_list_leaves_of_an_entry_still_sweep(self):
        from perf.bench import _concrete_configs

        _spec, combos = _concrete_configs(
            {
                "code": [{"align": 1}, {"align": 16}, {"align": 32}],
                "stack": [
                    {"size": 0x1000, "align": 8},
                    {"size": 0x1000, "align": 16},
                    {"size": 0x2000, "align": 8},
                    {"size": 0x2000, "align": 16},
                ],
                "branch": "unpredictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
            }
        )
        self.assertEqual(len(combos), 3 * 2 * 2)
        self.assertEqual(
            sorted(
                {
                    (c["code"]["align"], c["stack"]["size"], c["stack"]["align"])
                    for c in combos
                }
            ),
            [
                (1, 0x1000, 8),
                (1, 0x1000, 16),
                (1, 0x2000, 8),
                (1, 0x2000, 16),
                (16, 0x1000, 8),
                (16, 0x1000, 16),
                (16, 0x2000, 8),
                (16, 0x2000, 16),
                (32, 0x1000, 8),
                (32, 0x1000, 16),
                (32, 0x2000, 8),
                (32, 0x2000, 16),
            ],
        )

    def test_multiple_threads_per_run_rejected(self):
        from perf.bench import validate_spec
        from perf.core import _thread_group

        cfg = validate_spec({"thread": [[{"affinity": 1}, {"affinity": 2}]]})
        self.assertEqual(len(_thread_group(cfg)), 2)

    def test_unknown_container_key_rejected(self):
        from perf.bench import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"code": [{"alignment": 16}]})
        with self.assertRaises(ValueError):
            validate_spec({"stack": [{"bogus": 1}]})
        with self.assertRaises(ValueError):
            validate_spec({"code": []})

    def test_single_element_list_leaf_kept_in_the_spec(self):
        from perf.bench import _merge_config

        spec = _merge_config({"stack": [{"size": 0x1000, "align": 16}]})
        self.assertEqual(spec["stack"], [{"size": 0x1000, "align": 16}])

    def test_config_param_columns_read_the_list_form(self):
        from perf.bench import config_param_columns

        cols = config_param_columns(
            {
                "code": [{"align": 1}, {"align": 16}],
                "thread": [[{"affinity": 0, "priority": "normal"}]],
            }
        )
        self.assertEqual(cols["config.code.align"], [1, 16])
        self.assertEqual(cols["config.thread.priority"], ["normal"])


class TestConfigPruning(unittest.TestCase):
    def _combos(self, **override):
        from perf.bench import _concrete_configs

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
        from perf.bench import _prune_combos

        combos = self._combos()
        pruned = _prune_combos(combos, {"branches": False, "memory": False})
        self.assertEqual(len(pruned), 2)
        self.assertEqual({c["branch"] for c in pruned}, {"predictable"})
        self.assertEqual({c["dcache"] for c in pruned}, {"hot"})
        self.assertEqual({c["dtlb"] for c in pruned}, {"hot"})
        self.assertEqual({c["code"]["align"] for c in pruned}, {1, 16})

    def test_memory_keeps_the_cache_sweep(self):
        from perf.bench import _prune_combos

        pruned = _prune_combos(self._combos(), {"branches": False, "memory": True})
        self.assertEqual({c["dcache"] for c in pruned}, {"hot", "cold"})
        self.assertEqual({c["branch"] for c in pruned}, {"predictable"})

    def test_branches_keep_the_branch_sweep(self):
        from perf.bench import _prune_combos

        pruned = _prune_combos(self._combos(), {"branches": True, "memory": False})
        self.assertEqual(
            {c["branch"] for c in pruned}, {"predictable", "unpredictable"}
        )
        self.assertEqual({c["dcache"] for c in pruned}, {"hot"})

    def test_nothing_pruned_when_no_axis_varies(self):
        from perf.bench import _prune_combos

        combos = self._combos(branch="predictable", dcache="hot", dtlb="hot")
        self.assertEqual(
            _prune_combos(combos, {"branches": False, "memory": False}), combos
        )

    def test_features_of_a_plain_snippet(self):
        from perf.bench import _target_features

        combos = self._combos()
        features = _target_features(
            None, None, "mov eax, 42;", combos[0], {"regs": {}, "mem": {}}
        )
        self.assertEqual(features, {"branches": False, "memory": False})

    def test_features_of_a_load(self):
        from perf.bench import _target_features

        combos = self._combos()
        features = _target_features(
            None, None, "mov rax, [rdi]", combos[0], {"regs": {}, "mem": {}}
        )
        self.assertEqual(features, {"branches": False, "memory": True})

    def test_features_of_a_store(self):
        from perf.bench import _target_features

        combos = self._combos()
        features = _target_features(
            None, None, "mov [rdi], rax;", combos[0], {"regs": {}, "mem": {}}
        )
        self.assertTrue(features["memory"])

    def test_features_of_a_conditional_branch(self):
        from perf.bench import _target_features

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
        from perf.bench import _target_features

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
        from perf.bench import _target_features

        combos = self._combos()
        features = _target_features(
            None, None, "mov eax, 42;", combos[0], {"regs": {}, "mem": {"0x1000": 1}}
        )
        self.assertTrue(features["memory"])

    def test_explore_records_branches_and_memory(self):
        from perf.arch import arch as get_arch
        from perf.bench import explore_asm

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
        from perf.bench import _branches_seen, _memory_seen

        self.assertTrue(_branches_seen([{"branches": 0}, {"branches": 1}]))
        self.assertTrue(_branches_seen([{"branches": 0}, {"branches": 0}]))
        self.assertFalse(_branches_seen([{"branches": 0}]))
        self.assertFalse(_branches_seen([]))
        self.assertTrue(_memory_seen([{"reads": [], "writes": [(1, 8, 0)]}]))
        self.assertTrue(_memory_seen([{"reads": [(1, 8, 0)], "writes": []}]))
        self.assertFalse(_memory_seen([{"reads": [], "writes": []}]))
        self.assertFalse(_memory_seen(None))

    def test_exploration_is_shared_across_combinations(self):
        from perf.bench import _EXPLORE_CACHE, _explore_cached

        _EXPLORE_CACHE.clear()
        calls = []

        def factory():
            calls.append(1)
            return []

        self.assertEqual(_explore_cached("k", factory), [])
        self.assertEqual(_explore_cached("k", factory), [])
        self.assertEqual(len(calls), 1)
        _EXPLORE_CACHE.clear()

    def test_bench_prunes_combinations(self):
        from perf.bench import _prune_for_target

        combos = self._combos()
        pruned = _prune_for_target(
            None, None, "mov eax, 42;", combos, {"regs": {}, "mem": {}}
        )
        self.assertEqual(len(pruned), 2)


class TestBenchLegacyApiRemoved(unittest.TestCase):
    def test_file_is_not_accepted(self):
        from perf.bench import benchmark

        with self.assertRaises(TypeError):
            benchmark(file="a.out", mode=["latency"])

    def test_target_is_not_accepted(self):
        from perf.bench import benchmark

        with self.assertRaises(TypeError):
            benchmark(target=["a.out", "foo"], mode=["latency"])

    def test_func_is_not_accepted(self):
        from perf.bench import benchmark

        with self.assertRaises(TypeError):
            benchmark(func=["a.out", "foo"], mode=["latency"])

    def test_symbol_is_accepted(self):
        from unittest.mock import patch

        import pandas as pd

        from perf.bench import benchmark

        with patch("perf.bench._bench_one") as mock_one:
            mock_one.side_effect = lambda **kw: pd.DataFrame([{"mode": kw["mode"]}])
            benchmark(
                symbol=["a.out", "foo"],
                mode=["latency"],
                config={"branch": "predictable"},
            )
            kwargs = mock_one.call_args.kwargs
            self.assertEqual((kwargs["file"], kwargs["target"]), ("a.out", "foo"))

    def test_asm_is_accepted(self):
        from unittest.mock import patch

        import pandas as pd

        from perf.bench import benchmark

        with patch("perf.bench._bench_one") as mock_one:
            mock_one.side_effect = lambda **kw: pd.DataFrame([{"mode": kw["mode"]}])
            benchmark(asm="nop", mode=["latency"])
            self.assertEqual(mock_one.call_args.kwargs["code"], "nop")

    def test_region_is_accepted(self):
        from unittest.mock import patch

        import pandas as pd

        from perf.bench import benchmark

        with patch("perf.bench._bench_one") as mock_one:
            mock_one.side_effect = lambda **kw: pd.DataFrame([{"mode": kw["mode"]}])
            benchmark(
                region=["a.out", ("foo", "bar")],
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
        from perf.bench import _normalize_target as norm

        self.assertEqual(norm("foo"), "foo")
        self.assertIsNone(norm(None))
        self.assertIsNone(norm("  "))

    def test_pair_becomes_region(self):
        from perf.bench import _normalize_target as norm

        self.assertEqual(norm(("hot_begin", "hot_end")), "hot_begin..hot_end")
        self.assertEqual(norm(["a", "b"]), "a..b")
        self.assertEqual(norm((" a ", " b ")), "a..b")

    def test_region_string_rejected(self):
        from perf.bench import _normalize_target as norm

        with self.assertRaises(ValueError):
            norm("hot_begin..hot_end")

    def test_bad_pairs_rejected(self):
        from perf.bench import _normalize_target as norm

        for bad in (("only_one",), ("a", "b", "c"), ("", "b")):
            with self.subTest(target=bad), self.assertRaises(ValueError):
                norm(bad)


class TestProjectCacheKey(unittest.TestCase):
    def test_code_alignment_is_shared(self):
        from perf.bench import _project_cache_id

        self.assertEqual(
            _project_cache_id("a.out", {"code": {"align": 1}}),
            _project_cache_id("a.out", {"code": {"align": 16}}),
        )

    def test_stack_and_func_layouts_are_not_shared(self):
        from perf.bench import _project_cache_id

        self.assertNotEqual(
            _project_cache_id("a.out", {"stack": {"size": 0x1000}}),
            _project_cache_id("a.out", {"stack": {"size": 0x2000}}),
        )
        self.assertNotEqual(
            _project_cache_id("a.out", {"func": {"order": "as-is"}}),
            _project_cache_id("a.out", {"func": {"order": "random"}}),
        )


class TestBenchTargetSpec(unittest.TestCase):
    def test_analyze_rejects_file(self):
        from perf.mca import analyze

        with self.assertRaises(TypeError):
            analyze(file="a.out", symbol="foo")

    def test_analyze_rejects_code(self):
        from perf.mca import analyze

        with self.assertRaises(TypeError):
            analyze(code="mov eax, 42")

    def test_analyze_accepts_symbol_region_and_asm(self):
        import inspect

        from perf.bench import benchmark
        from perf.mca import analyze

        for fn in (benchmark, analyze):
            sig = inspect.signature(fn)
            self.assertIn("symbol", sig.parameters)
            self.assertIn("region", sig.parameters)
            self.assertIn("asm", sig.parameters)
            self.assertNotIn("code", sig.parameters)
            self.assertNotIn("file", sig.parameters)
            self.assertNotIn("target", sig.parameters)
            self.assertNotIn("func", sig.parameters)
        df = analyze(asm="nop")
        self.assertEqual(len(df), 1)
        with self.assertRaises(TypeError):
            analyze(symbol="a.out:foo", asm="nop")
        with self.assertRaises(TypeError):
            analyze()


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


class TestSeededRng(unittest.TestCase):
    def test_string_seed_is_deterministic(self):
        from perf.bench import _make_rng

        a = _make_rng({"seed": "abc"})
        b = _make_rng({"seed": "abc"})
        self.assertEqual([a.random() for _ in range(5)], [b.random() for _ in range(5)])

    def test_string_seed_matches_seed_int_stream(self):
        from perf.bench import _config_seed_int, _make_rng

        rng = _make_rng({"seed": "abc"})
        ref = random.Random(_config_seed_int({"seed": "abc"}))
        self.assertEqual(
            [rng.random() for _ in range(5)], [ref.random() for _ in range(5)]
        )


class TestSolveSignature(unittest.TestCase):
    def test_no_unused_rng_params(self):
        from perf.bench import solve

        params = set(inspect.signature(solve).parameters)
        self.assertNotIn("rng", params)
        self.assertNotIn("n", params)


class TestModelPicks(unittest.TestCase):
    def test_predictable_without_overrides_cycles_in_order(self):
        from perf.bench import _model_picks

        models = [
            {"regs": {"rdi": 1}, "reads": [], "writes": []},
            {"regs": {"rdi": 2}, "reads": [], "writes": []},
        ]
        picks = _model_picks(models, True, random.Random(0), None, 6)
        self.assertEqual(picks, [0, 1] * 3)

    def test_predictable_without_overrides_pins_first(self):
        from perf.bench import _model_picks

        models = [
            {"regs": {"rdi": 1}, "reads": [], "writes": []},
            {"regs": {"rdi": 2}, "reads": [], "writes": []},
        ]
        picks = _model_picks(models, True, random.Random(0), None, 1)
        self.assertEqual(picks, [0])


class TestExplicitDistributionBranching(unittest.TestCase):
    def test_predictable_cycles_explicit_reg_list_in_exec_order(self):
        from perf.bench import _per_iter_data

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
        from perf.bench import _per_iter_data

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


class TestValidateNumerics(unittest.TestCase):
    def test_bad_samples(self):
        from perf.bench import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"samples": 0})
        with self.assertRaises(ValueError):
            validate_spec({"samples": "many"})

    def test_bad_runs(self):
        from perf.bench import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"backend": {"loop": {"runs": 0}}})
        with self.assertRaises(ValueError):
            validate_spec({"backend": {"loop": {"runs": "many"}}})

    def test_bad_backend_target(self):
        from perf.bench import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"backend": {"loop": {"target": 0}}})
        with self.assertRaises(ValueError):
            validate_spec({"backend": {"loop": {"target": -1}}})
        with self.assertRaises(ValueError):
            validate_spec({"backend": {"loop": {"target": "many"}}})
        with self.assertRaises(ValueError):
            validate_spec({"backend": {"loop": {"probes": 0}}})
        with self.assertRaises(ValueError):
            validate_spec({"backend": {"loop": {"nope": 1}}})
        with self.assertRaises(ValueError):
            validate_spec({"backend": {"turbo": {"probes": 1}}})

    def test_bad_backend(self):
        from perf.bench import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"backend": "turbo"})

    def test_min_max_order(self):
        from perf.bench import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"iterations": {"min": 1000, "max": 10}})

    def test_bad_iterations(self):
        from perf.bench import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"iterations": 0})
        with self.assertRaises(ValueError):
            validate_spec({"iterations": {"min": 0}})
        with self.assertRaises(ValueError):
            validate_spec({"iterations": {"max": -5}})
        with self.assertRaises(ValueError):
            validate_spec({"iterations": {"nope": 5}})

    def test_branch_string_validated(self):
        from perf.bench import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"branch": "sometimes"})

    def test_valid_numerics(self):
        from perf.bench import validate_spec

        cfg = validate_spec(
            {
                "samples": 10,
                "iterations": {"min": 64, "max": 1000},
                "seed": 0,
                "external": {"lib": False},
                "backend": {
                    "loop": {"probes": 2, "runs": 2, "target": 0.01},
                    "unroll": {
                        "count": 3,
                        "probes": 2,
                        "runs": 2,
                        "target": 0.01,
                    },
                },
            }
        )
        self.assertEqual(cfg["samples"], 10)
        self.assertEqual(cfg["iterations"], {"min": 64, "max": 1000})
        self.assertEqual(cfg["backend"]["unroll"]["count"], 3)
        self.assertEqual(validate_spec({"backend": "unroll"})["backend"], "unroll")
        self.assertEqual(validate_spec({"iterations": 100})["iterations"], 100)


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
        from perf.bench import _names

        self.assertEqual(_names("cycles,instructions"), ["cycles", "instructions"])
        self.assertEqual(_names(["a,b", "c"]), ["a", "b", "c"])
        self.assertEqual(_names(7), ["7"])


class TestBranchString(unittest.TestCase):
    def test_valid_strings(self):
        from perf.bench import _branch_config

        self.assertEqual(
            _branch_config({"branch": "predictable"})["prediction"], "predictable"
        )
        self.assertEqual(
            _branch_config({"branch": "PREDICTABLE"})["prediction"], "predictable"
        )

    def test_invalid_string_raises(self):
        from perf.bench import _branch_config

        with self.assertRaises(ValueError):
            _branch_config({"branch": "sometimes"})


class TestMapStateMemPages(unittest.TestCase):
    def test_maps_pages_and_dedups(self):
        from perf.bench import _map_state_mem_pages

        state = Mock()
        _map_state_mem_pages(state, {"0x1000": 1, "0x1001": 2, "bogus": 3})
        self.assertEqual(state.memory.map_region.call_count, 1)
        args, _ = state.memory.map_region.call_args
        self.assertEqual(args[0], 0x1000)
        self.assertEqual(args[1], 0x1000)

    def test_range_suffix(self):
        from perf.bench import _map_state_mem_pages

        state = Mock()
        _map_state_mem_pages(state, {"0x2000:": 1})
        state.memory.map_region.assert_called_once()
        args, _ = state.memory.map_region.call_args
        self.assertEqual(args[0], 0x2000)

    def test_empty(self):
        from perf.bench import _map_state_mem_pages

        state = Mock()
        _map_state_mem_pages(state, {})
        _map_state_mem_pages(state, None)
        state.memory.map_region.assert_not_called()


class TestNormalizeDataRegsMasking(unittest.TestCase):
    def test_negative_masked(self):
        from perf.bench import _normalize_data_regs

        out = _normalize_data_regs({"rdi": -1})
        self.assertEqual(out["rdi"], 0xFFFFFFFFFFFFFFFF)

    def test_invalid_skipped(self):
        from perf.bench import _normalize_data_regs

        out = _normalize_data_regs({"rdi": "bogus"})
        self.assertNotIn("rdi", out)


class TestBranchValueChoices(unittest.TestCase):
    def test_all_choices_accepted_case_insensitive(self):
        from perf.bench import _BRANCH_CHOICES, _branch_value

        for choice in _BRANCH_CHOICES:
            self.assertEqual(_branch_value(choice), choice)
            self.assertEqual(_branch_value(choice.upper()), choice)
            self.assertEqual(_branch_value(f"  {choice}  "), choice)

    def test_bool_mapping(self):
        from perf.bench import _branch_value

        self.assertEqual(_branch_value(True), "predictable")
        self.assertEqual(_branch_value(False), "unpredictable")

    def test_invalid_returns_none(self):
        from perf.bench import _branch_value

        self.assertIsNone(_branch_value("sometimes"))
        self.assertIsNone(_branch_value("maybe"))


class TestBranchConfigChoices(unittest.TestCase):
    def test_string_form_each_choice(self):
        from perf.bench import _branch_config

        for choice in ("predictable", "unpredictable"):
            cfg = _branch_config({"branch": choice})
            self.assertEqual(cfg["prediction"], choice)

    def test_choice_is_case_and_space_insensitive(self):
        from perf.bench import _branch_config, _branch_value

        self.assertEqual(_branch_value("  UNPREDICTABLE "), "unpredictable")
        self.assertEqual(_branch_value("Predictable"), "predictable")
        cfg = _branch_config({"branch": "  UNPREDICTABLE "})
        self.assertEqual(cfg, {"prediction": "unpredictable", "mem": {}, "regs": {}})

    def test_only_two_choices_exist(self):
        from perf.bench import _BRANCH_CHOICES

        self.assertEqual(_BRANCH_CHOICES, ("predictable", "unpredictable"))

    def test_gaussian_rejected(self):
        from perf.bench import _branch_config

        with self.assertRaises(ValueError):
            _branch_config({"branch": "gaussian"})

    def test_removed_choices_rejected(self):
        from perf.bench import _branch_config

        for choice in (
            "unpredictable.exponential",
            "unpredictable.uniform",
            "predictable.exponential",
            "predictable.uniform",
            "uniform",
            "random",
            "shuffle",
            "normal",
            "exponential",
        ):
            with self.subTest(choice=choice):
                with self.assertRaises(ValueError):
                    _branch_config({"branch": choice})
                with self.assertRaises(ValueError):
                    _branch_config({"branch": {"prediction": choice}})
                with self.assertRaises(ValueError):
                    _branch_config({"branch": {"mem": {"0x1": choice}}})
                with self.assertRaises(ValueError):
                    _branch_config({"branch": {"regs": {"rdi": choice}}})

    def test_dict_prediction_and_default(self):
        from perf.bench import _branch_config

        cfg = _branch_config({"branch": {"prediction": "unpredictable"}})
        self.assertEqual(cfg["prediction"], "unpredictable")
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"default": "unpredictable"}})

    def test_bool_prediction_in_dict(self):
        from perf.bench import _branch_config

        self.assertEqual(
            _branch_config({"branch": {"prediction": True}})["prediction"],
            "predictable",
        )
        self.assertEqual(
            _branch_config({"branch": {"prediction": False}})["prediction"],
            "unpredictable",
        )

    def test_per_address_and_reg_choices(self):
        from perf.bench import _branch_config

        cfg = _branch_config(
            {
                "branch": {
                    "prediction": "unpredictable",
                    "mem": {"0x401000": "predictable"},
                    "regs": {"rdi": "predictable"},
                }
            }
        )
        self.assertEqual(cfg["mem"], {0x401000: "predictable"})
        self.assertEqual(cfg["regs"], {"rdi": "predictable"})

    def test_invalid_still_rejected(self):
        from perf.bench import _branch_config

        with self.assertRaises(ValueError):
            _branch_config({"branch": "sometimes"})
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"prediction": "sometimes"}})
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"mem": {"0x1": "sometimes"}}})
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"regs": {"rdi": "sometimes"}}})

    def test_validate_accepts_new_choices(self):
        from perf.bench import validate_spec

        for choice in ("predictable", "unpredictable"):
            validate_spec({"branch": choice})
        for choice in ("unpredictable.exponential", "unpredictable.uniform"):
            with self.assertRaises(ValueError):
                validate_spec({"branch": choice})


class TestBranchDistributions(unittest.TestCase):
    def test_global_distribution_precedence(self):
        from perf.bench import _branch_global_distribution

        self.assertEqual(
            _branch_global_distribution({"prediction": "unpredictable"}, False),
            "unpredictable",
        )
        self.assertEqual(
            _branch_global_distribution({"prediction": "predictable"}, False),
            "predictable",
        )
        self.assertEqual(_branch_global_distribution(None, True), "predictable")
        self.assertEqual(_branch_global_distribution(None, False), "unpredictable")

    def test_reg_and_mem_overrides(self):
        from perf.bench import _branch_mem_distribution, _branch_reg_distribution

        cfg = {
            "prediction": "unpredictable",
            "mem": {0x1000: "predictable"},
            "regs": {"rdi": "predictable"},
        }
        self.assertEqual(_branch_reg_distribution("rdi", cfg, False), "predictable")
        self.assertEqual(_branch_reg_distribution("rsi", cfg, False), "unpredictable")
        self.assertEqual(_branch_mem_distribution(0x1000, cfg, False), "predictable")
        self.assertEqual(_branch_mem_distribution(0x2000, cfg, False), "unpredictable")

    def test_overrides_are_case_insensitive(self):
        from perf.bench import _branch_mem_distribution, _branch_reg_distribution

        cfg = {
            "prediction": "unpredictable",
            "mem": {0x1000: "  PREDICTABLE "},
            "regs": {"rdi": "Predictable"},
        }
        self.assertEqual(_branch_reg_distribution("rdi", cfg, False), "predictable")
        self.assertEqual(_branch_mem_distribution(0x1000, cfg, False), "predictable")

    def test_sample_index_bounds(self):
        from perf.bench import _branch_sample_index

        for dist in ("predictable", "unpredictable"):
            rng = random.Random(0)
            for it in range(50):
                idx = _branch_sample_index(4, rng, dist, it)
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, 4)
        self.assertEqual(
            _branch_sample_index(1, random.Random(0), "unpredictable", 7), 0
        )
        self.assertEqual(
            _branch_sample_index(0, random.Random(0), "unpredictable", 3), 0
        )

    def test_sample_index_exponential_biased_to_first(self):
        from perf.bench import _branch_sample_index

        rng = random.Random(0)
        idxs = [
            _branch_sample_index(4, rng, "unpredictable.exponential", i)
            for i in range(400)
        ]
        self.assertGreater(idxs.count(0), idxs.count(3))

    def test_sample_index_predictable_cycles_incrementally(self):
        from perf.bench import _branch_sample_index

        rng = random.Random(0)
        self.assertEqual(
            [_branch_sample_index(3, rng, "predictable", it=i) for i in range(7)],
            [0, 1, 2, 0, 1, 2, 0],
        )

    def test_sample_index_deterministic(self):
        from perf.bench import _branch_sample_index

        for dist in (
            "unpredictable",
            "unpredictable.uniform",
            "unpredictable.exponential",
        ):
            a = [_branch_sample_index(4, random.Random(42), dist, i) for i in range(20)]
            b = [_branch_sample_index(4, random.Random(42), dist, i) for i in range(20)]
            self.assertEqual(a, b)

    def test_uniform_alias_matches_unpredictable_sampling(self):
        from perf.bench import _branch_sample_index

        expected = [
            _branch_sample_index(4, random.Random(9), "unpredictable", i)
            for i in range(20)
        ]
        actual = [
            _branch_sample_index(4, random.Random(9), "unpredictable.uniform", i)
            for i in range(20)
        ]
        self.assertEqual(actual, expected)

    def test_sample_values_predictable_cycles(self):
        from perf.bench import _sample_values

        vals = [10, 20, 30]
        self.assertEqual(
            [
                _sample_values(vals, "predictable", i, random.Random(0))
                for i in range(5)
            ],
            [10, 20, 30, 10, 20],
        )

    def test_sample_values_stays_in_set(self):
        from perf.bench import _sample_values

        for dist in (
            "unpredictable",
            "unpredictable.uniform",
            "unpredictable.exponential",
        ):
            rng = random.Random(1)
            for i in range(30):
                self.assertIn(_sample_values([1, 2, 3, 4], dist, i, rng), [1, 2, 3, 4])
        self.assertEqual(
            _sample_values([], "unpredictable.exponential", 0, random.Random(0)), 0
        )
        self.assertEqual(
            _sample_values([7], "unpredictable.exponential", 2, random.Random(0)), 7
        )


class TestBranchModelsAndPerIter(unittest.TestCase):
    def test_models_predictable_cycles(self):
        from perf.bench import _model_picks

        models = [
            {"regs": {"rdi": 1}, "reads": [], "writes": []},
            {"regs": {"rdi": 2}, "reads": [], "writes": []},
        ]
        cfg = {"prediction": "predictable", "mem": {}, "regs": {}}
        picked = [
            models[i]["regs"]["rdi"]
            for i in _model_picks(models, False, random.Random(0), cfg, 4)
        ]
        self.assertEqual(picked, [1, 2, 1, 2])

    def test_models_new_distributions_stay_in_set(self):
        from perf.bench import _model_picks

        models = [{"regs": {"rdi": i}, "reads": [], "writes": []} for i in (1, 2, 3, 4)]
        for dist in ("unpredictable", "unpredictable.exponential"):
            rng = random.Random(3)
            cfg = {"prediction": dist, "mem": {}, "regs": {}}
            for index in _model_picks(models, False, rng, cfg, 20):
                self.assertIn(models[index]["regs"]["rdi"], [1, 2, 3, 4])

    def test_per_iter_explicit_predictable_cycles(self):
        from perf.bench import _per_iter_data

        models = [{"regs": {"rdi": 0}, "reads": [], "writes": []}]
        buf, meta = _per_iter_data(
            models,
            6,
            None,
            False,
            random.Random(0),
            explicit={"regs": {"rdi": [1, 2, 3]}},
            branch_cfg={"prediction": "predictable", "mem": {}, "regs": {}},
        )
        col = meta["reg_col"]["rdi"]
        k = int(meta["k"])
        seq = [int(buf[col, k - 1 - i]) for i in range(6)]
        self.assertEqual(seq, [1, 2, 3, 1, 2, 3])

    def test_per_iter_explicit_new_distributions_cover_values(self):
        from perf.bench import _per_iter_data

        models = [{"regs": {"rdi": 0}, "reads": [], "writes": []}]
        for dist in ("unpredictable", "unpredictable.exponential"):
            buf, meta = _per_iter_data(
                models,
                60,
                None,
                False,
                random.Random(0),
                explicit={"regs": {"rdi": [1, 2, 3]}},
                branch_cfg={"prediction": dist, "mem": {}, "regs": {}},
            )
            col = meta["reg_col"]["rdi"]
            k = int(meta["k"])
            vals = {int(buf[col, i]) for i in range(1, k)}
            self.assertTrue(vals <= {1, 2, 3})
            self.assertTrue(len(vals) >= 1)


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
        self.assertIn('\n    "', text)
        compact = to_json(self._df(), indent=None)
        self.assertNotIn("\n", compact)
        json.loads(compact)

    def test_to_json_matches_id_hash(self):
        from perf.bench import _id_hash, to_json

        df = self._df()
        payload = json.loads(to_json(df))
        self.assertEqual(
            payload["id"],
            _id_hash(df.attrs["data"], df.attrs["config"], df.attrs["info"]["binary"]),
        )

    def test_to_json_is_exported(self):
        import perf
        from perf.bench import to_json as module_to_json

        self.assertTrue(callable(perf.to_json))
        self.assertEqual(perf.to_json, module_to_json)


class TestFlatDataWithBuckets(unittest.TestCase):
    def test_regs_survive_a_flat_dict_that_also_has_mem(self):
        from perf.bench import _merge_data

        merged = _merge_data(None, {"rdi": 7, "mem": {"0x1000": 1}})
        self.assertEqual(merged["regs"], {"rdi": 7})
        self.assertEqual(merged["mem"], {"0x1000": 1})

    def test_bucket_keys_win_over_flat_keys_on_a_conflict(self):
        from perf.bench import _merge_data

        merged = _merge_data(None, {"rdi": 7, "regs": {"rdi": 8}, "mem": {"0x1000": 1}})
        self.assertEqual(merged["regs"], {"rdi": 8})
        self.assertEqual(merged["mem"], {"0x1000": 1})

    def test_memory_alias_is_accepted(self):
        from perf.bench import _merge_data

        merged = _merge_data(None, {"memory": {"0x1000": 5}})
        self.assertEqual(merged["mem"], {"0x1000": 5})

    def test_flat_dict_without_buckets_is_parsed(self):
        from perf.bench import _merge_data

        merged = _merge_data(None, {"rdi": 7, "0x1000": 1})
        self.assertEqual(merged["regs"], {"rdi": 7})
        self.assertEqual(merged["mem"], {"0x1000": 1})

    def test_merge_overrides_the_base(self):
        from perf.bench import _merge_data

        merged = _merge_data(
            {"regs": {"rdi": 1}, "mem": {"0x10": 1}},
            {"regs": {"rdi": 2}, "mem": {"0x20": 2}},
        )
        self.assertEqual(merged["regs"], {"rdi": 2})
        self.assertEqual(merged["mem"], {"0x10": 1, "0x20": 2})


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

        overhead_runs = np.array([[10.0], [30.0], [20.0]])
        main_runs = np.array([[100.0], [50.0], [75.0]])
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
                [],
                None,
                [],
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
            float(np.asarray(collect.call_args_list[1].args[10]).reshape(-1)[0]), 10.0
        )
        self.assertEqual(df["cycles"].tolist(), [50.0])

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

        diffs = np.array([[100.0, 200.0], [50.0, 300.0], [np.nan, np.nan]])
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
                [],
                None,
                [],
                3,
                3,
                [None, None],
                None,
            )
        self.assertEqual(len(df), 1)
        self.assertEqual(df["duration_time"].tolist(), [50.0])
        self.assertEqual(df["cycles"].tolist(), [200.0])
        self.assertEqual(iterations, 500)

    def test_bench_measure_throughput_all_nan_stays_single_nan_row(self):
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
                None,
                [],
                3,
                3,
                [None],
                None,
            )
        self.assertEqual(len(df), 1)
        self.assertTrue(bool(np.isnan(df["duration_time"].iloc[0])))


class TestPinUnknownRegsConcrete(unittest.TestCase):
    def _arch(self):
        from perf.arch import arch as get_arch

        return get_arch()

    def test_concrete_false_adds_constraint_only(self):
        import claripy

        from perf.bench import _ASM_SCRATCH_BASE, _pin_unknown_regs

        arch = self._arch()
        state = Mock()
        state.solver.symbolic.return_value = True
        sym = claripy.BVS("rdi_sym", 64)
        pins = _pin_unknown_regs(state, {"rdi": sym}, set(), arch, concrete=False)
        self.assertEqual(pins, {"rdi"})
        state.solver.add.assert_called_once()
        state.registers.store.assert_not_called()
        state.memory.map_region.assert_called_once()
        constraint = state.solver.add.call_args.args[0]
        self.assertIn("rdi_sym", str(constraint))
        self.assertIn(hex(_ASM_SCRATCH_BASE), str(constraint).lower())

    def test_concrete_true_stores_scratch_concretely(self):
        import claripy

        from perf.bench import _ASM_SCRATCH_BASE, _pin_unknown_regs

        arch = self._arch()
        state = Mock()
        state.solver.symbolic.return_value = True
        syms = {
            "rdi": claripy.BVS("rdi_sym", 64),
            "rsi": claripy.BVS("rsi_sym", 64),
        }
        pins = _pin_unknown_regs(state, syms, set(), arch, concrete=True)
        self.assertEqual(pins, {"rdi", "rsi"})
        self.assertEqual(state.solver.add.call_count, 2)
        self.assertEqual(state.registers.store.call_count, 2)
        stored = {
            call.args[0]: call.args[1] for call in state.registers.store.call_args_list
        }
        self.assertEqual(set(stored), {"rdi", "rsi"})
        values = sorted(bvv.args[0] for bvv in stored.values())
        self.assertEqual(values, [_ASM_SCRATCH_BASE, _ASM_SCRATCH_BASE + 0x1000])
        for bvv in stored.values():
            self.assertEqual(bvv.size(), 64)

    def test_skips_explicit_and_harness(self):
        import claripy

        from perf.bench import _pin_unknown_regs

        arch = self._arch()
        state = Mock()
        state.solver.symbolic.return_value = True
        syms = {
            "rdi": claripy.BVS("rdi_sym", 64),
            "rax": claripy.BVS("rax_sym", 64),
            "r8": claripy.BVS("r8_sym", 64),
        }
        pins = _pin_unknown_regs(state, syms, {"rdi"}, arch, concrete=True)
        self.assertEqual(pins, {"rax"})
        self.assertEqual(state.solver.add.call_count, 1)
        self.assertEqual(state.registers.store.call_count, 1)

    def test_bits_fallback_to_64(self):
        from perf.bench import _ASM_SCRATCH_BASE, _pin_unknown_regs

        arch = self._arch()
        state = Mock()
        state.solver.symbolic.return_value = True

        class _NoSize:
            def size(self):
                raise RuntimeError("no size")

        pins = _pin_unknown_regs(state, {"rdi": _NoSize()}, set(), arch, concrete=True)
        self.assertEqual(pins, {"rdi"})
        _name, bvv = state.registers.store.call_args.args
        self.assertEqual(_name, "rdi")
        self.assertEqual(bvv.size(), 64)
        self.assertEqual(bvv.args[0], _ASM_SCRATCH_BASE)

    def test_store_failure_still_pins(self):
        import claripy

        from perf.bench import _pin_unknown_regs

        arch = self._arch()
        state = Mock()
        state.solver.symbolic.return_value = True
        state.registers.store.side_effect = RuntimeError("boom")
        pins = _pin_unknown_regs(
            state, {"rdi": claripy.BVS("x", 64)}, set(), arch, concrete=True
        )
        self.assertEqual(pins, {"rdi"})

    def test_symbolic_and_solver_failures_skipped(self):
        import claripy

        from perf.bench import _pin_unknown_regs

        arch = self._arch()
        state = Mock()
        state.solver.symbolic.side_effect = RuntimeError("boom")
        pins = _pin_unknown_regs(
            state, {"rdi": claripy.BVS("x", 64)}, set(), arch, concrete=True
        )
        self.assertEqual(pins, set())
        state2 = Mock()
        state2.solver.symbolic.return_value = True
        state2.solver.add.side_effect = RuntimeError("boom")
        pins2 = _pin_unknown_regs(
            state2, {"rdi": claripy.BVS("x", 64)}, set(), arch, concrete=True
        )
        self.assertEqual(pins2, set())


class TestExploreExplicitConcretePinning(unittest.TestCase):
    def _explore_with(self, sym_regs, explicit_regs, use_funcs=True, **kw):
        from perf import bench as bench_mod

        proj = Mock()
        state = Mock()
        state.solver.add = Mock()
        state.registers.store = Mock()
        state.memory.map_region = Mock()
        proj.factory.call_state.return_value = state
        patches = [
            patch.object(bench_mod, "_symbolic_options"),
            patch.object(bench_mod, "_symbolic_regs", return_value=sym_regs),
            patch.object(bench_mod, "_normalize_data_regs", return_value=explicit_regs),
            patch.object(bench_mod, "_map_state_mem_pages"),
            patch.object(bench_mod, "_nondet_options"),
            patch.object(bench_mod, "_ret_addrs", return_value={0x2000}),
            patch.object(bench_mod, "_target_ranges", return_value=[]),
            patch.object(bench_mod, "_setup_ranges", return_value=()),
            patch.object(bench_mod, "_explore_states"),
            patch.object(bench_mod, "_reached", return_value=kw.pop("reached", [])),
            patch.object(
                bench_mod,
                "_needs_reg_pinning",
                return_value=kw.pop("needs_pinning", False),
            ),
            patch.object(
                bench_mod, "_solution_records", return_value=kw.pop("records", ["ok"])
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        call_kw = {"funcs": {"f": (0x1000, 0x2000)}} if use_funcs else {}
        return proj, state, bench_mod.explore(proj, 0x1000, 0x2000, **call_kw, **kw)

    def test_missing_funcs_uses_info_functions(self):
        from perf import bench as bench_mod

        with patch.object(
            bench_mod,
            "functions",
            return_value=({"f": (0x1000, 0x2000)}, {}),
        ) as mock_functions:
            _proj, _state, out = self._explore_with({}, {}, use_funcs=False)
        self.assertEqual(out, ["ok"])
        mock_functions.assert_called_once()

    def test_explicit_list_value_skipped(self):
        import claripy

        sym_regs = {"rdi": claripy.BVS("rdi_sym", 64)}
        _proj, state, out = self._explore_with(sym_regs, {"rdi": [1, 2]})
        self.assertEqual(out, ["ok"])
        state.solver.add.assert_not_called()
        state.registers.store.assert_not_called()

    def test_explicit_values_masked_and_stored(self):
        import claripy

        sym_regs = {"rdi": claripy.BVS("rdi_sym", 64)}
        _proj, state, _out = self._explore_with(sym_regs, {"rdi": -1})
        state.solver.add.assert_called_once()
        state.registers.store.assert_called_once()
        name, bvv = state.registers.store.call_args.args
        self.assertEqual(name, "rdi")
        self.assertEqual(bvv.args[0], 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(bvv.size(), 64)

    def test_explicit_overflow_masked(self):
        import claripy

        sym_regs = {"rdi": claripy.BVS("rdi_sym", 64)}
        _proj, state, _out = self._explore_with(sym_regs, {"rdi": (1 << 70) + 5})
        _name, bvv = state.registers.store.call_args.args
        self.assertEqual(bvv.args[0], ((1 << 70) + 5) & 0xFFFFFFFFFFFFFFFF)

    def test_explicit_bits_fallback(self):
        class _NoSize:
            def __eq__(self, other):
                return ("eq", other)

            def size(self):
                raise RuntimeError("no size")

        _proj, state, _out = self._explore_with({"rsi": _NoSize()}, {"rsi": 99})
        state.registers.store.assert_called_once()
        _name, bvv = state.registers.store.call_args.args
        self.assertEqual(bvv.size(), 64)
        self.assertEqual(bvv.args[0], 99)

    def test_explicit_store_failure_ignored(self):
        import claripy

        sym_regs = {"rdi": claripy.BVS("rdi_sym", 64)}
        proj = Mock()
        state = Mock()
        state.solver.add = Mock()
        state.registers.store = Mock(side_effect=RuntimeError("boom"))
        state.memory.map_region = Mock()
        proj.factory.call_state.return_value = state
        from perf import bench as bench_mod

        with (
            patch.object(bench_mod, "_symbolic_options"),
            patch.object(bench_mod, "_symbolic_regs", return_value=sym_regs),
            patch.object(bench_mod, "_normalize_data_regs", return_value={"rdi": 7}),
            patch.object(bench_mod, "_map_state_mem_pages"),
            patch.object(bench_mod, "_nondet_options"),
            patch.object(bench_mod, "_ret_addrs", return_value={0x2000}),
            patch.object(bench_mod, "_target_ranges", return_value=[]),
            patch.object(bench_mod, "_setup_ranges", return_value=()),
            patch.object(bench_mod, "_explore_states"),
            patch.object(bench_mod, "_reached", return_value=[]),
            patch.object(bench_mod, "_needs_reg_pinning", return_value=False),
            patch.object(bench_mod, "_solution_records", return_value=["ok"]),
        ):
            out = bench_mod.explore(proj, 0x1000, 0x2000, funcs={"f": (0x1000, 0x2000)})
        self.assertEqual(out, ["ok"])
        state.solver.add.assert_called_once()

    def test_explicit_invalid_value_ignored(self):
        import claripy

        sym_regs = {"rcx": claripy.BVS("rcx_sym", 64)}
        _proj, state, out = self._explore_with(sym_regs, {"rcx": "bad"})
        self.assertEqual(out, ["ok"])
        state.solver.add.assert_not_called()
        state.registers.store.assert_not_called()

    def test_explicit_solver_type_error_ignored(self):
        import claripy

        sym_regs = {"rdi": claripy.BVS("rdi_sym", 64)}
        proj = Mock()
        state = Mock()
        state.solver.add = Mock(side_effect=TypeError("bad constraint"))
        state.registers.store = Mock()
        state.memory.map_region = Mock()
        proj.factory.call_state.return_value = state
        from perf import bench as bench_mod

        with (
            patch.object(bench_mod, "_symbolic_options"),
            patch.object(bench_mod, "_symbolic_regs", return_value=sym_regs),
            patch.object(bench_mod, "_normalize_data_regs", return_value={"rdi": 1}),
            patch.object(bench_mod, "_map_state_mem_pages"),
            patch.object(bench_mod, "_nondet_options"),
            patch.object(bench_mod, "_ret_addrs", return_value={0x2000}),
            patch.object(bench_mod, "_target_ranges", return_value=[]),
            patch.object(bench_mod, "_setup_ranges", return_value=()),
            patch.object(bench_mod, "_explore_states"),
            patch.object(bench_mod, "_reached", return_value=[]),
            patch.object(bench_mod, "_needs_reg_pinning", return_value=False),
            patch.object(bench_mod, "_solution_records", return_value=["ok"]),
        ):
            out = bench_mod.explore(proj, 0x1000, 0x2000, funcs={"f": (0x1000, 0x2000)})
        self.assertEqual(out, ["ok"])

    def test_second_pass_pins_concretely_with_fallbacks_and_teardown(self):
        import claripy

        from perf import bench as bench_mod

        proj = Mock()
        state = Mock()
        state.solver.add = Mock()
        state.registers.store = Mock()
        state.memory.map_region = Mock()
        proj.factory.call_state.return_value = state
        sym_regs = {"rdi": claripy.BVS("rdi_sym", 64)}
        fake_arch = Mock()
        fake_arch._canonical_reg.side_effect = RuntimeError("no canon")
        fake_arch._HARNESS_REGS = set()

        def _get_arch(arg=None):
            if arg is proj:
                raise RuntimeError("no arch for proj")
            return fake_arch

        with (
            patch.object(bench_mod, "_symbolic_options"),
            patch.object(bench_mod, "_symbolic_regs", return_value=sym_regs),
            patch.object(bench_mod, "_normalize_data_regs", return_value={"rdi": 1}),
            patch.object(bench_mod, "_map_state_mem_pages"),
            patch.object(bench_mod, "_nondet_options"),
            patch.object(bench_mod, "_ret_addrs", return_value={0x2000}),
            patch.object(bench_mod, "_target_ranges", return_value=[]),
            patch.object(bench_mod, "_setup_ranges", return_value=()),
            patch.object(bench_mod, "_explore_states"),
            patch.object(bench_mod, "_reached", return_value=["s1"]),
            patch.object(bench_mod, "_needs_reg_pinning", return_value=True),
            patch.object(bench_mod, "get_arch", side_effect=_get_arch),
            patch.object(
                bench_mod, "_pin_unknown_regs", return_value={"rsi"}
            ) as mock_pin,
            patch.object(bench_mod, "_solution_records", return_value=["rec"]),
            patch.object(
                bench_mod, "_explore_teardown", return_value=["torn"]
            ) as mock_teardown,
        ):
            out = bench_mod.explore(
                proj,
                0x1000,
                0x2000,
                funcs={"f": (0x1000, 0x2000)},
                teardown_asm="ret",
            )
        self.assertEqual(out, ["torn"])
        mock_pin.assert_called_once()
        _args, kwargs = mock_pin.call_args
        self.assertTrue(kwargs.get("concrete", _args[4] if len(_args) > 4 else None))
        self.assertEqual(_args[2], {"rdi"})
        mock_teardown.assert_called_once()


class TestExplicitBoundFallback(unittest.TestCase):
    def test_explicit_bound_values_filters_pointers(self):
        from perf.bench import _explicit_bound_values

        self.assertEqual(
            _explicit_bound_values({"regs": {"rcx": 100}, "mem": {}}), [100]
        )
        self.assertEqual(
            _explicit_bound_values(
                {"regs": {"rdi": 0x5000000000, "rcx": 100}, "mem": {}}
            ),
            [100],
        )
        self.assertEqual(_explicit_bound_values({"regs": {}, "mem": {}}), [])
        self.assertEqual(
            _explicit_bound_values({"regs": {"rcx": [1, 100]}, "mem": {}}), [1, 100]
        )

    def test_augment_maps_requested_pages(self):
        import ctypes

        from perf.bench import _augment_data_with_fallback

        data = {"regs": {"rcx": 100}, "mem": {}}
        out = _augment_data_with_fallback(dict(data), pages=2)
        self.assertEqual(out["regs"]["rcx"], 100)
        for reg in ("rdi", "rsi", "rdx"):
            self.assertIn(reg, out["regs"])
            base = int(out["regs"][reg])
            self.assertEqual(base & 0xFFF, 0)
            ctypes.c_uint8.from_address(base).value = 0xAB
            self.assertEqual(ctypes.c_uint8.from_address(base).value, 0xAB)
            ctypes.c_uint8.from_address(base + 4096).value = 0xCD
            self.assertEqual(ctypes.c_uint8.from_address(base + 4096).value, 0xCD)
        self.assertIn(f"0x{int(out['regs']['rdi']):x}", out["mem"])
        self.assertIn(f"0x{int(out['regs']['rdi']) + 4096:x}", out["mem"])

    def test_is_harness_fault_only_on_fault(self):
        from perf.bench import _is_harness_fault

        self.assertTrue(
            _is_harness_fault(
                ValueError("'matmul' faulted with a segmentation fault (SIGSEGV)")
            )
        )
        self.assertFalse(
            _is_harness_fault(
                ValueError("'matmul' did not return from 1 call(s) on its own")
            )
        )
        self.assertFalse(_is_harness_fault(ValueError("boom")))

    def test_grow_doubles_until_probe_passes(self):
        from perf.bench import _grow_fallback_data

        data = {"regs": {"rcx": 100}, "mem": {}}
        seen = []

        def _probe(augmented):
            seen.append(len(augmented["mem"]))
            if len(augmented["mem"]) < 3 * 4:
                raise ValueError("faulted with a segmentation fault (SIGSEGV)")
            return None

        out = _grow_fallback_data(data, None, _probe)
        self.assertGreaterEqual(len(out["mem"]), 3 * 4)
        self.assertGreaterEqual(len(seen), 3)
        self.assertEqual(seen, sorted(seen))

    def test_grow_propagates_non_fault(self):
        from perf.bench import _grow_fallback_data

        data = {"regs": {"rcx": 100}, "mem": {}}
        calls = []

        def _probe(augmented):
            calls.append(augmented)
            raise ValueError("did not return from 1 call(s) on its own")

        with self.assertRaises(ValueError):
            _grow_fallback_data(data, None, _probe)
        self.assertEqual(len(calls), 1)

    def test_grow_stops_at_budget(self):
        from unittest.mock import patch

        from perf import bench as bench_mod
        from perf.bench import _grow_fallback_data

        data = {"regs": {"rcx": 100}, "mem": {}}

        def _probe(augmented):
            raise ValueError("faulted with a segmentation fault (SIGSEGV)")

        with (
            patch.object(bench_mod, "_available_memory", return_value=1 << 20),
            patch.object(bench_mod, "_run_memory", return_value=0),
        ):
            with self.assertRaises(ValueError) as ctx:
                _grow_fallback_data(data, None, _probe)
        self.assertIn("faulted with", str(ctx.exception))

    def test_missing_arg_regs_excludes_explicit_and_harness(self):
        from perf.bench import _missing_arg_regs

        missing = _missing_arg_regs({"regs": {"rcx": 100}, "mem": {}})
        self.assertIn("rdi", missing)
        self.assertIn("rsi", missing)
        self.assertIn("rdx", missing)
        self.assertNotIn("rcx", missing)
        self.assertNotIn("r8", missing)
        self.assertNotIn("r9", missing)
        self.assertEqual(
            _missing_arg_regs(
                {
                    "regs": {
                        "rdi": 1,
                        "rsi": 2,
                        "rdx": 3,
                        "rcx": 100,
                    },
                    "mem": {},
                }
            ),
            [],
        )

    def test_augment_respects_explicit_and_maps(self):
        import ctypes

        from perf.bench import _augment_data_with_fallback

        data = {"regs": {"rcx": 100}, "mem": {}}
        out = _augment_data_with_fallback(dict(data))
        self.assertEqual(out["regs"]["rcx"], 100)
        for reg in ("rdi", "rsi", "rdx"):
            self.assertIn(reg, out["regs"])
            base = int(out["regs"][reg])
            self.assertNotEqual(base & 0xFFF, 0xFFF + 1)
            self.assertEqual(base & 0xFFF, 0)
            ctypes.c_uint8.from_address(base).value = 0xAB
            self.assertEqual(ctypes.c_uint8.from_address(base).value, 0xAB)
        self.assertIn(f"0x{int(out['regs']['rdi']):x}", out["mem"])

    def test_augment_needs_explicit_bounds(self):
        from perf.bench import _augment_data_with_fallback

        data = {"regs": {}, "mem": {}}
        inp = dict(data)
        self.assertIs(_augment_data_with_fallback(inp), inp)
        data = {"regs": {"rdi": 0x5000000000}, "mem": {}}
        inp = dict(data)
        self.assertIs(_augment_data_with_fallback(inp), inp)

    def _fast_config():
        return {
            "branch": "predictable",
            "dcache": "hot",
            "dtlb": "hot",
            "icache": "hot",
            "itlb": "hot",
            "code": {"align": 1},
            "stack": {"size": 2097152, "align": 16},
            "func": {"align": 16, "order": "as-is"},
            "samples": 1,
            "iterations": 1,
            "backend": {"loop": {"runs": 1, "probes": 1}},
            "seed": 0,
        }


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
        from perf.bench import validate_spec

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

    def test_concrete_thread_alternatives(self):
        from perf.bench import _concrete_configs
        from perf.core import _thread_group

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
            iterations=8,
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
        from perf.core import _thread_group as _thread_group_for

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
            patch.object(bench, "_explore_asm_cached", return_value=[]),
        ):
            df = benchmark(
                asm="nop",
                name="t",
                mode=["throughput"],
                event=["duration_time"],
                config=cfg,
            )
        rec = df.reset_index()
        self.assertEqual(rec["threads"].tolist(), [2, 2])
        self.assertEqual(rec["operations"].tolist(), [10, 20])

    def test_benchmark_threads_column_single(self):
        df = benchmark(
            asm="nop",
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


class TestAddParamColumns(unittest.TestCase):
    def test_empty_frame_passes_through(self):
        import pandas as pd

        from perf.bench import add_param_columns

        df = pd.DataFrame({"cycles": []})
        self.assertTrue(add_param_columns(df, {"regs": {"rdi": 1}}).empty)
        self.assertIsNone(add_param_columns(None, {"regs": {"rdi": 1}}))

    def test_scalar_data_becomes_column(self):
        import pandas as pd

        from perf.bench import add_param_columns

        df = pd.DataFrame({"cycles": [10.0, 20.0]})
        out = add_param_columns(df, {"regs": {"rdi": 7}, "mem": {}})
        self.assertIn("data.rdi", out.columns)
        self.assertEqual(out["data.rdi"].tolist(), [7, 7])

    def test_existing_column_not_duplicated(self):
        import pandas as pd

        from perf.bench import add_param_columns

        df = pd.DataFrame({"cycles": [1.0], "data.rdi": [5]})
        out = add_param_columns(df, {"regs": {"rdi": 5}, "mem": {}})
        self.assertEqual(out["data.rdi"].tolist(), [5])


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
            iterations=2,
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

        import perf

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
        env = json.loads(perf.to_json(df))
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
            asm="nop",
            name="t",
        )
        rec = df.reset_index()
        self.assertNotIn("iteration", rec.columns)
        self.assertNotIn("samples", rec.columns)
        self.assertEqual(len(rec), 2)

    def test_benchmark_dumps_all_iterations(self):
        df = benchmark(
            asm="mov eax, 42",
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


if __name__ == "__main__":
    unittest.main()


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
            patch.object(self.bench, "_model_world", side_effect=self._world),
            patch.object(self.bench, "_model_returns", side_effect=predicate) as mock,
        ):
            kept = self.bench._runtable_models(models, object(), probe, setup_asm=setup)
        return kept, mock

    def test_no_probe_keeps_the_models(self):
        models = [{"regs": {"rdi": 1}, "reads": [], "writes": []}]
        with patch.object(self.bench, "_model_returns") as mock:
            self.assertIs(self.bench._runtable_models(models, object(), None), models)
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
            patch.object(self.bench, "_model_world", side_effect=self._world),
            patch.object(self.bench, "_model_returns", return_value=True) as mock,
        ):
            self.bench._runtable_models(models, object(), "ret")
            self.bench._runtable_models(models, object(), "ret")
        self.assertEqual(mock.call_count, 1)

    def test_setup_change_breaks_the_cache(self):
        models = [{"regs": {"rdi": 1}, "reads": [], "writes": []}]
        with (
            patch.object(self.bench, "_model_world", side_effect=self._world),
            patch.object(self.bench, "_model_returns", return_value=True) as mock,
        ):
            self.bench._runtable_models(models, object(), "ret", setup_asm="a")
            self.bench._runtable_models(models, object(), "ret", setup_asm="b")
        self.assertEqual(mock.call_count, 2)

    def test_empty_models(self):
        self.assertEqual(self.bench._runtable_models([], object(), "ret"), [])


class TestBenchProjectCache(unittest.TestCase):
    def setUp(self):
        from perf import bench

        self.bench = bench
        bench._PROJECTS.clear()
        self.addCleanup(bench._PROJECTS.clear)

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
            first = self.bench._bench_project(
                "/tmp/a.out",
                {},
                {"size": 0x1000, "align": 16},
                {"align": 16, "order": "as-is"},
            )
            second = self.bench._bench_project(
                "/tmp/a.out",
                {},
                {"size": 0x1000, "align": 16},
                {"align": 16, "order": "as-is"},
            )
        self.assertIs(first, second)
        self.assertEqual(mk.call_count, 1)
        obj.map_elf.assert_called_once_with()
        obj.setup_stack.assert_called_once_with(size=0x1000, align=16)

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
            self.bench._bench_project(
                "/tmp/a.out",
                {},
                {"size": 0x1000, "align": 16},
                {"align": 16, "order": "as-is"},
            )
            self.bench._bench_project(
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
            self.bench._bench_project(
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
            self.bench._bench_project(
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
            out = self.bench._bench_project(
                "/tmp/a.out",
                {},
                {"size": 0x1000, "align": 16},
                {"align": 16, "order": "as-is"},
            )
        self.assertIs(out[1], obj)


class TestScratchValues(unittest.TestCase):
    def test_scratch_value_accepts_a_single_value(self):
        from perf.bench import _ASM_SCRATCH_BASE, _is_scratch_value

        self.assertTrue(_is_scratch_value(_ASM_SCRATCH_BASE))
        self.assertTrue(_is_scratch_value(_ASM_SCRATCH_BASE + 0xFFFFF))
        self.assertTrue(_is_scratch_value(str(_ASM_SCRATCH_BASE)))

    def test_scratch_value_rejects_outside_and_invalid(self):
        from perf.bench import _ASM_SCRATCH_BASE, _is_scratch_value

        self.assertFalse(_is_scratch_value(_ASM_SCRATCH_BASE - 1))
        self.assertFalse(_is_scratch_value(_ASM_SCRATCH_BASE + 0x100000))
        self.assertFalse(_is_scratch_value("nope"))
        self.assertFalse(_is_scratch_value([]))

    def test_scratch_value_scans_lists(self):
        from perf.bench import _ASM_SCRATCH_BASE, _is_scratch_value

        self.assertTrue(_is_scratch_value([1, _ASM_SCRATCH_BASE + 1]))
        self.assertFalse(_is_scratch_value([1, 2]))

    def test_scratch_pages_are_mapped_for_scratch_regs(self):
        from perf.bench import _ASM_SCRATCH_BASE, _map_scratch_reg_pages

        with patch("perf.bench._map_data_pages") as mock:
            _map_scratch_reg_pages(
                [{"regs": {"rdi": _ASM_SCRATCH_BASE + 0x1234}}, {"regs": {"rsi": 1}}]
            )
        mem = mock.call_args.args[0]["mem"]
        self.assertTrue(mem)
        self.assertTrue(all(int(a, 0) >= (_ASM_SCRATCH_BASE & ~0xFFF) for a in mem))

    def test_no_scratch_regs_do_not_map(self):
        from perf.bench import _map_scratch_reg_pages

        with patch("perf.bench._map_data_pages") as mock:
            _map_scratch_reg_pages([{"regs": {"rdi": 1}}, {"regs": {}}])
        mock.assert_not_called()


class TestReservedAddressesUnit(unittest.TestCase):
    def test_reserved_addresses_come_from_relocations(self):
        from perf.bench import _reserved_addresses

        class _Obj:
            def relocated_addresses(self):
                return {0x1000, 0x2000}

        self.assertEqual(_reserved_addresses(_Obj()), {0x1000, 0x2000})

    def test_reserved_addresses_are_empty_when_unavailable(self):
        from perf.bench import _reserved_addresses

        class _Obj:
            def relocated_addresses(self):
                raise RuntimeError("no load")

        self.assertEqual(_reserved_addresses(_Obj()), set())
        self.assertEqual(_reserved_addresses(object()), set())


class TestDataPageMappingUnit(unittest.TestCase):
    PAGE = 0x42000000000

    def setUp(self):
        from perf import bench

        self.bench = bench
        self._mapped = set(bench._MAPPED_DATA_PAGES)
        self._unmappable = set(bench._UNMAPPABLE_DATA_PAGES)
        bench._MAPPED_DATA_PAGES.clear()
        bench._UNMAPPABLE_DATA_PAGES.clear()

    def tearDown(self):
        self.bench._MAPPED_DATA_PAGES.clear()
        self.bench._MAPPED_DATA_PAGES.update(self._mapped)
        self.bench._UNMAPPABLE_DATA_PAGES.clear()
        self.bench._UNMAPPABLE_DATA_PAGES.update(self._unmappable)

    def _arch(self):
        return SimpleNamespace(
            page_runs=lambda pages: [(pages[0], len(pages))] if pages else []
        )

    def test_successful_mapping_is_not_marked_unmappable(self):
        with (
            patch.object(self.bench, "get_arch", return_value=self._arch()),
            patch.object(
                self.bench, "_map_range", return_value=(True, 0, self.PAGE)
            ) as mapping,
        ):
            self.bench._map_data_pages({"mem": {str(self.PAGE): 1}})
        self.assertIn(self.PAGE, self.bench._MAPPED_DATA_PAGES)
        self.assertNotIn(self.PAGE, self.bench._UNMAPPABLE_DATA_PAGES)
        mapping.assert_called_once_with(self.PAGE, 4096)

    def test_non_strict_failure_marks_the_page_unmappable(self):
        with (
            patch.object(self.bench, "get_arch", return_value=self._arch()),
            patch.object(self.bench, "_map_range", return_value=(False, 12, -1)),
        ):
            self.bench._map_data_pages({"mem": {str(self.PAGE): 1}}, strict=False)
        self.assertIn(self.PAGE, self.bench._UNMAPPABLE_DATA_PAGES)
        self.assertNotIn(self.PAGE, self.bench._MAPPED_DATA_PAGES)

    def test_strict_failure_raises_an_address_error(self):
        with (
            patch.object(self.bench, "get_arch", return_value=self._arch()),
            patch.object(
                self.bench, "_map_range", return_value=(False, 12, (1 << 64) - 1)
            ),
        ):
            with self.assertRaises(OSError) as ctx:
                self.bench._map_data_pages({"mem": {str(self.PAGE): 1}})
        self.assertIn("address already mapped", str(ctx.exception))

    def test_already_known_pages_are_not_remapped(self):
        self.bench._MAPPED_DATA_PAGES.add(self.PAGE)
        with (
            patch.object(self.bench, "get_arch", return_value=self._arch()),
            patch.object(self.bench, "_map_range") as mapping,
        ):
            self.bench._map_data_pages({"mem": {str(self.PAGE): 1}})
        mapping.assert_not_called()


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
        self.assertEqual(len(res), 2)
        self.assertEqual(it, 3)
        self.assertEqual(active["peak"], 1)

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

        from perf.bench import _apply_seed

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

        from perf.bench import _MODEL_CACHE, _models_cached

        _MODEL_CACHE.clear()
        self.addCleanup(_MODEL_CACHE.clear)
        with patch("perf.bench.solve", return_value=[{"a": 1}]) as m1:
            out1 = _models_cached([{"x": 1}], None, None)
            self.assertEqual(m1.call_count, 1)
            out2 = _models_cached([{"x": 1}], None, None)
            self.assertEqual(m1.call_count, 1)
            self.assertEqual(out1, out2)
            out1.append({"mutated": True})
            out3 = _models_cached([{"x": 1}], None, None)
            self.assertNotIn({"mutated": True}, out3)

    def test_tag_config_columns_copies_list_values(self):
        import perf.bench as bench

        df = pd.DataFrame({"cycles": [1.0, 2.0]})
        out = bench._tag_config_columns(df, {"code": [{"align": 1}]})
        self.assertIn("cycles", out.columns)


class TestThreadSharedState(unittest.TestCase):
    def test_bench_and_exec_share_the_fd_redirection_lock(self):
        import perf.bench as bench_mod
        import perf.exec as exec_mod

        self.assertIs(bench_mod._OUTPUT_LOCK, exec_mod._OUTPUT_LOCK)

    def test_bench_and_exec_share_the_fork_lock(self):
        import perf.bench as bench_mod
        import perf.exec as exec_mod

        self.assertIs(bench_mod._FORK_LOCK, exec_mod._FORK_LOCK)

    def test_runnable_cache_is_updated_after_releasing_the_fork_lock(self):
        import perf.bench as bench_mod

        fork_free = {}

        def fake_locked(*args, **kwargs):
            return None

        def fake_remember(key):
            acquired = bench_mod._FORK_LOCK.acquire(blocking=False)
            fork_free["value"] = acquired
            if acquired:
                bench_mod._FORK_LOCK.release()

        with (
            patch.object(bench_mod, "_is_runnable", return_value=False),
            patch.object(bench_mod, "_check_target_runnable_locked", fake_locked),
            patch.object(bench_mod, "_remember_runnable", side_effect=fake_remember),
        ):
            bench_mod._check_target_runnable(
                lambda _args: None,
                SimpleNamespace(inputs=0, outputs=0, data=0),
                key=("key", 1),
            )
        self.assertTrue(fork_free["value"])

    def test_simulated_ranges_are_passed_to_each_measurement_thread(self):
        import perf.bench as bench_mod

        seen = []
        lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=10)
        ranges = ((0x1000, 0x2000),)

        def fake_single(*args, **kwargs):
            barrier.wait(timeout=10)
            with lock:
                seen.append(kwargs.get("simulated_ranges"))
            return pd.DataFrame({"cycles": [1.0]}), 2

        with patch.object(bench_mod, "_measure_group_single", fake_single):
            bench_mod._measure_group(
                {"thread": [{"priority": "normal"}, {"priority": "normal"}]},
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
                simulated_ranges=ranges,
            )
        self.assertEqual(seen, [ranges, ranges])

    def test_memory_state_is_isolated_per_thread(self):
        import perf.bench as bench_mod

        bench_mod._MEMORY_STATE.value = {"calls": 7, "per_call": 3.5}
        self.addCleanup(lambda: setattr(bench_mod._MEMORY_STATE, "value", None))
        seen = []

        def worker():
            seen.append(bench_mod._memory_snapshot())
            bench_mod._MEMORY_STATE.value = {"calls": 2, "per_call": 1.0}
            seen.append(bench_mod._memory_snapshot())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        self.assertEqual(seen, [(0.0, 0), (1.0, 2)])
        self.assertEqual(bench_mod._memory_snapshot(), (3.5, 7))

    def test_measure_thread_registry_counts_nested_workers(self):
        import perf.bench as bench_mod

        bench_mod._measure_threads_reset()
        self.addCleanup(bench_mod._measure_threads_reset)
        self.assertEqual(bench_mod._measure_threads_count(), 0)
        bench_mod._measure_threads_enter()
        bench_mod._measure_threads_enter()
        self.assertEqual(bench_mod._measure_threads_count(), 1)
        bench_mod._measure_threads_exit()
        self.assertEqual(bench_mod._measure_threads_count(), 1)
        bench_mod._measure_threads_exit()
        self.assertEqual(bench_mod._measure_threads_count(), 0)
        bench_mod._measure_threads_exit()
        self.assertEqual(bench_mod._measure_threads_count(), 0)
        holder = SimpleNamespace(ident=threading.get_ident())
        bench_mod._measure_threads_enter()
        bench_mod._measure_threads_forget([holder])
        self.assertEqual(bench_mod._measure_threads_count(), 0)

    def test_projected_memory_growth_scales_with_active_workers(self):
        import perf.bench as bench_mod

        bench_mod._measure_threads_reset()
        bench_mod._MEMORY_STATE.value = {"calls": 20, "per_call": 1024.0}
        self.addCleanup(bench_mod._measure_threads_reset)
        self.addCleanup(lambda: setattr(bench_mod._MEMORY_STATE, "value", None))
        projected, per_call, threads, calls = bench_mod._projected_memory_growth(10)
        self.assertEqual(
            (projected, per_call, threads, calls), (10240.0, 1024.0, 1, 20)
        )
        entered = threading.Event()
        release = threading.Event()
        bench_mod._measure_threads_enter()
        self.addCleanup(bench_mod._measure_threads_exit)

        def worker():
            bench_mod._measure_threads_enter()
            entered.set()
            release.wait(timeout=10)
            bench_mod._measure_threads_exit()

        thread = threading.Thread(target=worker)
        thread.start()
        self.addCleanup(release.set)
        self.addCleanup(thread.join, 10)
        self.assertTrue(entered.wait(timeout=10))
        projected, per_call, threads, calls = bench_mod._projected_memory_growth(10)
        self.assertEqual(
            (projected, per_call, threads, calls), (20480.0, 1024.0, 2, 20)
        )
        release.set()
        thread.join(timeout=10)
        bench_mod._measure_threads_exit()
        self.assertEqual(bench_mod._measure_threads_count(), 0)

    def test_projected_memory_growth_ignores_invalid_iterations(self):
        import perf.bench as bench_mod

        bench_mod._measure_threads_reset()
        bench_mod._MEMORY_STATE.value = {"calls": 3, "per_call": 2.0}
        self.addCleanup(bench_mod._measure_threads_reset)
        self.addCleanup(lambda: setattr(bench_mod._MEMORY_STATE, "value", None))
        for value in (None, "bad", -5):
            projected, _per_call, threads, calls = bench_mod._projected_memory_growth(
                value
            )
            self.assertEqual(projected, 0.0)
            self.assertEqual(threads, 1)
            self.assertEqual(calls, 3)
        with patch.object(
            bench_mod, "_measure_threads_count", side_effect=RuntimeError("boom")
        ):
            projected, _per_call, threads, calls = bench_mod._projected_memory_growth(
                10
            )
            self.assertEqual(projected, 20.0)
            self.assertEqual(threads, 1)

    def test_check_growth_accounts_for_every_active_worker(self):
        import perf.bench as bench_mod

        with (
            patch.object(bench_mod, "_memory_snapshot", return_value=(1024.0, 5)),
            patch.object(bench_mod, "_measure_threads_count", return_value=3),
        ):
            with self.assertRaises(MemoryError):
                bench_mod._check_growth(100, used=0, budget=204800)
        with (
            patch.object(bench_mod, "_memory_snapshot", return_value=(1024.0, 5)),
            patch.object(bench_mod, "_measure_threads_count", return_value=1),
        ):
            bench_mod._check_growth(100, used=0, budget=204800)


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
        self.assertIsNone(_json_scalar(pd.NA))
        self.assertEqual(_json_scalar({"a": np.nan}), {"a": None})

    def test_prune_nulls_removes_empty_branches(self):
        from perf.bench import _prune_nulls

        self.assertEqual(
            _prune_nulls(
                {
                    "a": None,
                    "b": {},
                    "c": {"d": None, "e": 1},
                    "f": [None, 1],
                }
            ),
            {"c": {"e": 1}, "f": [None, 1]},
        )


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
        self.assertEqual([r["duration_time"] for r in recs], [1.0, 2.0])

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


class TestConcreteAddrValue(unittest.TestCase):
    def test_values_and_symbolic_expressions(self):
        from types import SimpleNamespace

        from perf.bench import _concrete_addr_value

        self.assertIsNone(_concrete_addr_value(True))
        self.assertEqual(_concrete_addr_value(0x1000), 0x1000)
        self.assertEqual(_concrete_addr_value("0xff"), 0xFF)
        self.assertIsNone(_concrete_addr_value("nope"))
        self.assertIsNone(_concrete_addr_value(None))
        concrete = SimpleNamespace(symbolic=False, args=(0x1234, 64))
        self.assertEqual(_concrete_addr_value(concrete), 0x1234)
        symbolic = SimpleNamespace(symbolic=True, args=(0x1234, 64))
        self.assertIsNone(_concrete_addr_value(symbolic))
        no_args = SimpleNamespace(symbolic=False, args=None)
        self.assertIsNone(_concrete_addr_value(no_args))
        empty_args = SimpleNamespace(symbolic=False, args=())
        self.assertIsNone(_concrete_addr_value(empty_args))
        no_symbolic = SimpleNamespace(args=(5, 64))
        self.assertIsNone(_concrete_addr_value(no_symbolic))

    def test_objects_without_a_numeric_value(self):
        from perf.bench import _concrete_addr_value

        class Opaque:
            def __int__(self):
                raise TypeError("no")

        self.assertIsNone(_concrete_addr_value(Opaque()))


class TestResolveTargetEdges(unittest.TestCase):
    def _asm_file(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "target.s")
        Path(path).write_text("foo:\n\tmov eax, 42\n\tret\n")
        return path

    def test_symbol_forms(self):
        from perf.bench import _resolve_target

        path = self._asm_file()
        self.assertEqual(
            _resolve_target(symbol=f"{path}:foo"),
            (None, None, "mov eax, 42", "foo", f"{path}:foo"),
        )
        with self.assertRaises(ValueError):
            _resolve_target(symbol="mov eax, 42")
        with self.assertRaises(ValueError):
            _resolve_target(symbol=f"{path}:missing")
        with self.assertRaises(ValueError):
            _resolve_target(symbol=["a.out", ("a", "b")])
        with self.assertRaises(ValueError):
            _resolve_target(symbol="a.out:begin..end")

    def test_region_forms(self):
        from perf.bench import _resolve_target

        path = self._asm_file()
        self.assertEqual(_resolve_target(region=f"{path}:foo")[2], "mov eax, 42")
        self.assertEqual(
            _resolve_target(region=["a.out", ("a", "b")])[:4],
            ("a.out", ("a", "b"), None, None),
        )
        with self.assertRaises(ValueError):
            _resolve_target(region="mov eax, 42")
        with self.assertRaises(ValueError):
            _resolve_target(region=f"{path}:missing")
        with self.assertRaises(ValueError):
            _resolve_target(region="a.out:foo")

    def test_asm_forms(self):
        from perf.bench import _resolve_target

        path = self._asm_file()
        self.assertEqual(_resolve_target(asm="nop")[:4], (None, None, "nop", None))
        self.assertEqual(_resolve_target(asm=f"{path}:foo")[2], "mov eax, 42")
        with self.assertRaises(ValueError):
            _resolve_target(asm=f"{path}:missing")
        with self.assertRaises(ValueError):
            _resolve_target(asm="a.out:foo")
        with self.assertRaises(ValueError):
            _resolve_target(asm=["a.out", ("a", "b")])

    def test_asm_source_name_defaults_to_the_label(self):
        from perf.bench import _resolve_target

        path = self._asm_file()
        _, _, _, name, raw = _resolve_target(symbol=f"{path}:foo", name="given")
        self.assertEqual(name, "given")
        self.assertEqual(raw, f"{path}:foo")


class TestTagConfigColumnsEdges(unittest.TestCase):
    def test_list_values_are_repeated_per_row(self):
        from perf.bench import _tag_config_columns

        df = pd.DataFrame({"cycles": [1.0, 2.0]})
        out = _tag_config_columns(
            df, {"code": [{"align": 1}], "branch": ["hot", "cold"]}
        )
        self.assertEqual(out["config.code.align"].tolist(), [1, 1])
        self.assertEqual(
            out["config.branch"].tolist(),
            [["hot", "cold"], ["hot", "cold"]],
        )
        self.assertEqual(list(out.attrs), list(df.attrs))

    def test_empty_frames_and_registry_failures_pass_through(self):
        from perf.bench import _tag_config_columns

        self.assertTrue(
            _tag_config_columns(pd.DataFrame(), {"code": [{"align": 1}]}).empty
        )

        df = pd.DataFrame({"cycles": [1.0]})
        with patch.object(
            bench, "config_param_columns", side_effect=RuntimeError("boom")
        ):
            out = _tag_config_columns(df, {"code": [{"align": 1}]})
        self.assertIs(out, df)

    def test_missing_columns_are_added_next_to_the_frame(self):
        from perf.bench import _tag_config_columns

        df = pd.DataFrame({"cycles": [1.0, 2.0]}, index=[5, 6])
        out = _tag_config_columns(df, {"iterations": {"min": 100, "max": 200}})
        self.assertEqual(out["config.iterations.min"].tolist(), [100, 100])
        self.assertEqual(out.index.tolist(), [5, 6])


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
        with patch.object(core, "_parse_numa", side_effect=ValueError("bad")):
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

        from perf.bench import _per_iter_data

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


class TestFillDynamic(unittest.TestCase):
    def test_zero_iterations_leaves_the_buffer_untouched(self):
        import numpy as np

        from perf.bench import _fill_dynamic

        buf = np.zeros((2, 3), dtype=np.uint64)
        _fill_dynamic(
            buf,
            0,
            [{"regs": {}, "reads": [], "writes": []}],
            [],
            ["rdi"],
            {"rdi": 0},
            {},
            [],
            {},
            {},
            set(),
            set(),
            {},
            None,
            True,
            random.Random(1),
        )
        self.assertFalse(buf.any())

    def test_explicit_lists_follow_the_branch_distribution(self):
        import numpy as np

        from perf.bench import _fill_dynamic

        models = [
            {"regs": {"rdi": 7}, "reads": [], "writes": []},
            {"regs": {"rdi": 9}, "reads": [], "writes": []},
        ]
        buf = np.zeros((2, 6), dtype=np.uint64)
        _fill_dynamic(
            buf,
            5,
            models,
            [0],
            ["rdi"],
            {"rdi": 0},
            {0: 1},
            [0],
            {"rdi": [1, 2]},
            {0: [3, 4]},
            set(),
            set(),
            {},
            {"prediction": "predictable", "mem": {}, "regs": {}},
            True,
            random.Random(1),
        )
        self.assertEqual(buf[0, 1:6].tolist(), [1, 2, 1, 2, 1])
        self.assertEqual(buf[1, 1:6].tolist(), [3, 4, 3, 4, 3])

    def test_unpredictable_explicit_lists_sample_the_values(self):
        import numpy as np

        from perf.bench import _fill_dynamic

        models = [
            {"regs": {"rdi": 7}, "reads": [], "writes": []},
        ]
        buf = np.zeros((2, 9), dtype=np.uint64)
        _fill_dynamic(
            buf,
            8,
            models,
            [0] * 8,
            ["rdi"],
            {"rdi": 0},
            {0: 1},
            [0],
            {"rdi": [1, 2]},
            {0: [3, 4]},
            set(),
            set(),
            {},
            {"prediction": "unpredictable", "mem": {}, "regs": {}},
            False,
            random.Random(3),
        )
        self.assertLessEqual(set(buf[0, 1:9].tolist()), {1, 2})
        self.assertLessEqual(set(buf[1, 1:9].tolist()), {3, 4})
        self.assertGreater(len(set(buf[0, 1:9].tolist())), 1)

    def test_pinned_registers_and_memory_keep_the_first_model(self):
        import numpy as np

        from perf.bench import _fill_dynamic

        models = [
            {"regs": {"rdi": 7}, "reads": [], "writes": []},
            {"regs": {"rdi": 9}, "reads": [], "writes": []},
        ]
        buf = np.zeros((2, 4), dtype=np.uint64)
        _fill_dynamic(
            buf,
            3,
            models,
            [0, 1, 0],
            ["rdi"],
            {"rdi": 0},
            {0: 1},
            [0],
            {},
            {},
            {"rdi"},
            {0},
            {0: 11},
            {"prediction": "predictable", "mem": {}, "regs": {}},
            True,
            random.Random(1),
        )
        self.assertEqual(buf[0, 1:4].tolist(), [7, 7, 7])
        self.assertEqual(buf[1, 1:4].tolist(), [11, 11, 11])

    def test_malformed_picks_fall_back_to_the_first_model(self):
        import numpy as np

        from perf.bench import _fill_dynamic

        class BadPicks:
            def __iter__(self):
                raise TypeError("not a list")

        models = [
            {"regs": {"rdi": 7}, "reads": [], "writes": []},
            {"regs": {"rdi": 9}, "reads": [], "writes": []},
        ]
        buf = np.zeros((2, 7), dtype=np.uint64)
        _fill_dynamic(
            buf,
            6,
            models,
            BadPicks(),
            ["rdi"],
            {"rdi": 0},
            {},
            [],
            {},
            {},
            set(),
            set(),
            {},
            None,
            True,
            random.Random(1),
        )
        self.assertEqual(buf[0, 1:7].tolist(), [7] * 6)


class TestBranchDeps(unittest.TestCase):
    def test_memory_operands_record_register_and_address_dependencies(self):
        import angr

        from perf.bench import _branch_deps

        code = bytes.fromhex("48837f0800750290c3")
        project = angr.load_shellcode(code, arch="x86_64", load_address=0x400000)
        state = project.factory.blank_state(addr=0x400000)
        deps = _branch_deps(state, {"regs": {"rdi": 0x500000}}, {0x400000})
        self.assertEqual(deps[0x400000]["regs"], {"rdi"})
        self.assertEqual(deps[0x400000]["mem"], {0x500008})

    def test_non_branch_and_broken_blocks_are_skipped(self):
        import angr

        from perf.bench import _branch_deps

        code = bytes.fromhex("4885ff7401c3")
        project = angr.load_shellcode(code, arch="x86_64", load_address=0x500000)
        state = project.factory.blank_state(addr=0x500000)
        deps = _branch_deps(
            state,
            {"regs": {"rdi": 1}},
            {0x500005, 0x700000},
        )
        self.assertEqual(deps, {})

    def test_a_state_without_a_project_has_no_dependencies(self):
        from perf.bench import _branch_deps

        self.assertEqual(_branch_deps(SimpleNamespace(), {"regs": {}}, {0x1000}), {})


class TestFastProbePaths(unittest.TestCase):
    def test_run_once_reports_a_returning_callback(self):
        from perf.bench import _run_once

        calls = []

        def target():
            calls.append(True)

        if not hasattr(os, "fork"):
            self.skipTest("requires fork")
        self.assertTrue(_run_once(target, []))
        self.assertEqual(calls, [])

    def test_run_once_reports_a_failing_callback(self):
        from perf.bench import _run_once

        def target():
            raise RuntimeError("boom")

        if not hasattr(os, "fork"):
            self.skipTest("requires fork")
        self.assertFalse(_run_once(target, []))

    def test_store_model_bytes_writes_only_valid_sized_scalars(self):
        import ctypes

        from perf.bench import _store_model_bytes

        buf = ctypes.create_string_buffer(b"\x00" * 8)
        addr = ctypes.addressof(buf)
        _store_model_bytes(addr, 1, 0x1234)
        self.assertEqual(buf.raw[:1], b"\x34")
        _store_model_bytes(addr, 4, 0xDEADBEEF)
        self.assertEqual(buf.raw[:4], b"\xef\xbe\xad\xde")
        before = buf.raw
        _store_model_bytes(addr, 0, 1)
        _store_model_bytes(addr, 9, 1)
        _store_model_bytes(addr, "many", 1)
        _store_model_bytes(addr, 4, "not a number")
        self.assertEqual(buf.raw, before)

    def test_merge_fallback_data_keeps_explicit_values(self):
        from perf.bench import _merge_fallback_data

        base = {"regs": {"rdi": 1}, "mem": {"0x1000": 2}}
        grown = [
            {"regs": {"rdi": 9, "rsi": 3}, "mem": {"0x1000": 8, "0x2000": 4}},
            {"regs": {"rdx": 5}, "mem": {"0x3000": 6}},
            "not a dict",
        ]
        merged = _merge_fallback_data(base, grown)
        self.assertEqual(merged["regs"], {"rdi": 1, "rsi": 3, "rdx": 5})
        self.assertEqual(merged["mem"], {"0x1000": 2, "0x2000": 4, "0x3000": 6})
        self.assertEqual(_merge_fallback_data(None, None), {"regs": {}, "mem": {}})

    def test_run_memory_reads_self_and_handles_a_missing_pid(self):
        from perf.bench import _run_memory

        self.assertGreater(_run_memory(), 0)
        self.assertEqual(_run_memory("no-such-perf-pid"), 0)

    def test_available_memory_reads_meminfo_and_falls_back_to_zero(self):
        from perf.bench import _available_memory

        self.assertGreater(_available_memory(), 0)
        with patch("builtins.open", side_effect=OSError("no meminfo")):
            self.assertEqual(_available_memory(), 0)

    def test_fallback_budget_subtracts_the_used_memory(self):
        import perf.bench as bench_mod

        with (
            patch.object(bench_mod, "_available_memory", return_value=1 << 21),
            patch.object(bench_mod, "_run_memory", return_value=1 << 19),
        ):
            self.assertEqual(bench_mod._fallback_budget_bytes(), 1 << 19)
        with patch.object(bench_mod, "_available_memory", return_value=0):
            self.assertIsNone(bench_mod._fallback_budget_bytes())

    def test_reset_fatal_handlers_restores_the_defaults(self):
        import signal

        import perf.bench as bench_mod

        saved = {}
        for signum in range(1, 32):
            try:
                saved[signum] = signal.getsignal(signum)
            except (OSError, ValueError):
                continue

        def restore():
            for signum, handler in saved.items():
                try:
                    signal.signal(signum, handler)
                except (OSError, ValueError):
                    continue

        self.addCleanup(restore)
        signal.signal(signal.SIGUSR1, lambda *args: None)
        bench_mod._reset_fatal_handlers()
        self.assertEqual(signal.getsignal(signal.SIGUSR1), signal.SIG_DFL)

    def test_runnable_probe_accepts_a_returning_callback(self):
        import ctypes
        from types import SimpleNamespace

        import perf.bench as bench_mod

        if not hasattr(os, "fork"):
            self.skipTest("requires fork")
        callback = ctypes.CFUNCTYPE(None, ctypes.POINTER(bench_mod._Args))(
            lambda _args: None
        )
        args = SimpleNamespace(inputs=0, outputs=0, data=0)
        self.assertIsNone(bench_mod._check_target_runnable_locked(callback, args))

    def test_runnable_probe_reports_a_signal(self):
        import ctypes
        import signal
        from types import SimpleNamespace

        import perf.bench as bench_mod

        if not hasattr(os, "fork"):
            self.skipTest("requires fork")
        callback = ctypes.CFUNCTYPE(None, ctypes.POINTER(bench_mod._Args))(
            lambda _args: os.kill(os.getpid(), signal.SIGSEGV)
        )
        args = SimpleNamespace(inputs=0, outputs=0, data=0)
        with self.assertRaises(ValueError) as ctx:
            bench_mod._check_target_runnable_locked(callback, args)
        self.assertIn("faulted with a segmentation fault", str(ctx.exception))

    def test_runnable_probe_reports_a_hang(self):
        import ctypes
        import time
        from types import SimpleNamespace

        import perf.bench as bench_mod

        if not hasattr(os, "fork"):
            self.skipTest("requires fork")
        old_timeout = bench_mod._PROBE_TIMEOUT
        bench_mod._PROBE_TIMEOUT = 0.1
        self.addCleanup(setattr, bench_mod, "_PROBE_TIMEOUT", old_timeout)
        callback = ctypes.CFUNCTYPE(None, ctypes.POINTER(bench_mod._Args))(
            lambda _args: time.sleep(1)
        )
        args = SimpleNamespace(inputs=0, outputs=0, data=0)
        with self.assertRaises(ValueError) as ctx:
            bench_mod._check_target_runnable_locked(callback, args)
        self.assertIn("did not return", str(ctx.exception))

    def test_runnable_probe_closes_the_pipe_when_fork_fails(self):
        from types import SimpleNamespace

        import perf.bench as bench_mod

        if not hasattr(os, "fork"):
            self.skipTest("requires fork")
        args = SimpleNamespace(inputs=0, outputs=0, data=0)
        with patch.object(bench_mod.os, "fork", side_effect=OSError("no fork")):
            with self.assertRaises(OSError):
                bench_mod._check_target_runnable_locked(lambda _args: None, args)


class TestRetAddrs(unittest.TestCase):
    def test_returns_are_found_in_the_code(self):
        import angr

        from perf.bench import _ret_addrs

        project = angr.load_shellcode(b"\x90\xc3", arch="x86_64", load_address=0x1000)
        self.assertIn(0x1001, _ret_addrs(project, 0x1000, 0x1002))

    def test_a_region_without_a_return_uses_its_end(self):
        import angr

        from perf.bench import _ret_addrs

        project = angr.load_shellcode(b"\x90\x90", arch="x86_64", load_address=0x1000)
        self.assertEqual(_ret_addrs(project, 0x1000, 0x1002), {0x1002})


class TestAsmTextHelpers(unittest.TestCase):
    def test_left_align_asm_strips_indentation(self):
        from perf.bench import _left_align_asm

        self.assertEqual(_left_align_asm(""), "")
        self.assertEqual(_left_align_asm(None), "")
        self.assertEqual(
            _left_align_asm("  mov eax, 1\n    ret\n\n"),
            "mov eax, 1\nret\n\n",
        )

    def test_executed_bbl_addresses_are_deduplicated(self):
        from perf.bench import _executed_bbl_addrs

        history = SimpleNamespace(bbl_addrs=[0x1000, 0x1001, 0x1000, "bad"])
        solution = {"state": SimpleNamespace(history=history)}
        self.assertEqual(_executed_bbl_addrs([solution]), [0x1000, 0x1001])
        direct = SimpleNamespace(history=SimpleNamespace(bbl_addrs=[0x2000, 0x2000]))
        self.assertEqual(_executed_bbl_addrs([direct]), [0x2000])
        self.assertEqual(_executed_bbl_addrs([{}]), [])
        self.assertEqual(_executed_bbl_addrs([None]), [])
        broken = SimpleNamespace(
            history=SimpleNamespace(bbl_addrs=SimpleNamespace(__iter__=None))
        )
        self.assertEqual(_executed_bbl_addrs([broken]), [])

    def test_sub_intel_labels_replaces_known_addresses(self):
        from perf.bench import _sub_intel_labels

        self.assertEqual(_sub_intel_labels("0x1000", {0x1000: "loop"}), "loop")
        self.assertEqual(
            _sub_intel_labels("0x10, 0x1000", {0x1000: "loop"}),
            "0x10, loop",
        )
        self.assertEqual(_sub_intel_labels("0xzz", {0x1000: "loop"}), "0xzz")
        self.assertEqual(_sub_intel_labels(None, {}), "")


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


class TestSlowProbeAsm(unittest.TestCase):
    def test_teardown_snippet_is_explored(self):
        from perf.bench import explore_asm

        sols = explore_asm("mov eax, 42", teardown_asm="mov ebx, 7")
        self.assertTrue(sols)
        self.assertEqual([sol["branches"] for sol in sols], [0])

    def test_string_and_invalid_register_values_are_filtered(self):
        from perf.bench import explore_asm

        sols = explore_asm(
            "mov eax, 42",
            data={"regs": {"rax": "0x10", "rsi": "bad"}, "mem": {}},
        )
        self.assertTrue(sols)

    def test_invalid_assembly_returns_no_solutions(self):
        from perf.bench import explore_asm

        self.assertEqual(explore_asm("this is not assembly"), [])
        self.assertEqual(explore_asm(""), [])


class TestSetupAsmFor(unittest.TestCase):
    def test_call_setup_for_a_single_symbol(self):
        import sys

        import angr

        from perf.bench import _setup_asm_for
        from perf.exec import Elf, _symtab_values

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
