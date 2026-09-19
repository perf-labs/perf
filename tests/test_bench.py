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

import inspect
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

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
    def test_mode_is_required(self, mock_one):
        with self.assertRaises(TypeError):
            benchmark(code="nop", name="t", config={"branch": ["predictable"]})

    @patch("perf.bench._bench_one")
    def test_modes_run_with_each_config_combo(self, mock_one):
        def fake(**kw):
            return pd.DataFrame(
                [{"mode": kw["mode"], "align": kw["config"]["code"]["align"]}]
            )

        mock_one.side_effect = fake
        result = benchmark(
            code="nop",
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
            code="nop",
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
        benchmark(code=["a.out", "foo"], mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertEqual((kwargs["file"], kwargs["target"]), ("a.out", "foo"))
        self.assertIsNone(kwargs["code"])

    @patch("perf.bench._bench_one")
    def test_code_kwarg_text(self, mock_one):
        self._one(mock_one)
        benchmark(code="a.out:begin..end", mode=["latency"], config=self._FAST)
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
        benchmark("mov eax, 42", mode=["latency"], config=self._FAST)
        kwargs = mock_one.call_args.kwargs
        self.assertIsNone(kwargs["file"])
        self.assertEqual(kwargs["code"], "mov eax, 42")

    def test_code_conflicts(self):
        for kw in (
            {"code": "a.out:foo", "file": "a.out"},
            {"code": "mov eax, 42", "asm": "mov eax, 42"},
            {"code": "a.out:foo", "target": "foo"},
            {"asm": "mov eax, 42", "target": "foo"},
        ):
            with self.subTest(**kw), self.assertRaises(TypeError):
                benchmark(mode=["latency"], config=self._FAST, **kw)


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
            code="nop",
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
        self.assertEqual(result["samples"].tolist(), [0, 1, 2])

    @patch("perf.bench._bench", return_value=pd.Series([100]))
    def test_default_config_applied_when_none(self, mock_bench):
        result = benchmark(code="nop", name="t", mode=["latency"], config=None)
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
        self.assertEqual(
            cfg["backend"],
            {"loop": _DEFAULT_BENCH["backend"]["loop"]},
        )
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
            code="nop",
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
            code="nop",
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
            code="nop",
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
            code="nop",
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
            code="nop",
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
                code="mov eax, 42",
                mode=["latency"],
                config={"iterations": 8, "samples": 1},
                data={"r8": 1},
            )

    def test_asm_source_label_becomes_a_snippet(self):
        import shutil
        import tempfile
        from pathlib import Path

        from perf.bench import asm_source_code

        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
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

        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
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

    def test_asm_source_is_measured_as_a_snippet(self):
        import shutil
        import tempfile
        from pathlib import Path

        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "f.s")
        Path(path).write_text("foo:\n\tmov eax, 42\n\tret\n")
        df = benchmark(
            code=f"{path}:foo",
            mode=["latency"],
            config={"iterations": 16, "samples": 1, "code": {"align": 1}},
        )
        self.assertTrue(len(df) > 0)
        self.assertEqual(list(df.index.get_level_values("name").unique()), ["foo"])
        self.assertTrue(df.index.get_level_values("file").isna().all())


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

    def test_default_config_includes_cache(self):
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


class TestExploreArgumentDiscovery(unittest.TestCase):
    @staticmethod
    def _make_binary():
        import subprocess
        import tempfile

        if shutil.which("g++") is None:
            return None
        d = tempfile.mkdtemp()
        src = os.path.join(d, "fb.cpp")
        out = os.path.join(d, "fb")
        with open(src, "w") as f:
            f.write(
                "const char* fizz_buzz(int n){\n"
                '  if(n%15==0) return "FizzBuzz";\n'
                '  else if(n%3==0) return "Fizz";\n'
                '  else if(n%5==0) return "Buzz";\n'
                '  else return "Unknown";\n'
                "}\n"
                "int main(){}\n"
            )
        r = subprocess.run(["g++", "-O3", "-o", out, src], capture_output=True)
        if r.returncode != 0 or not os.path.exists(out):
            return None
        return out

    @unittest.skipIf(os.environ.get("PERF_SKIP_ANGREXPLORE"), "angr explore skipped")
    def test_fizz_buzz_discovered_without_data(self):

        import angr

        from perf.bench import explore
        from perf.info import functions

        out = self._make_binary()
        if out is None:
            self.skipTest("g++ unavailable")
        d = os.path.dirname(out)
        try:
            proj = angr.Project(out, auto_load_libs=False, load_debug_info=False)
            funcs, _ = functions(proj)
            target = next(name for name in funcs if re.search("fizz_buzz", name))
            start, end = funcs[target]
            results = explore(proj, start, end, funcs=funcs)
            values = sorted({r["state"].solver.eval(r["reg"]["rdi"]) for r in results})
            self.assertGreaterEqual(len(values), 3)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    @unittest.skipIf(os.environ.get("PERF_SKIP_ANGREXPLORE"), "angr explore skipped")
    def test_fizz_buzz_pinned_data_stays_single(self):

        import angr

        from perf.bench import explore
        from perf.info import functions

        out = self._make_binary()
        if out is None:
            self.skipTest("g++ unavailable")
        d = os.path.dirname(out)
        try:
            proj = angr.Project(out, auto_load_libs=False, load_debug_info=False)
            funcs, _ = functions(proj)
            target = next(name for name in funcs if re.search("fizz_buzz", name))
            start, end = funcs[target]
            results = explore(proj, start, end, funcs=funcs, data={"regs": {"rdi": 64}})
            values = {r["state"].solver.eval(r["reg"]["rdi"]) for r in results}
            self.assertEqual(values, {64})
        finally:
            shutil.rmtree(d, ignore_errors=True)

    @unittest.skipIf(os.environ.get("PERF_SKIP_ANGREXPLORE"), "angr explore skipped")
    def test_fizz_buzz_folds_to_four_paths(self):
        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")

        import subprocess
        import tempfile

        import angr

        from perf.bench import explore
        from perf.info import functions

        d = tempfile.mkdtemp()
        try:
            src = os.path.join(d, "fb.c")
            out = os.path.join(d, "fb")
            with open(src, "w") as f:
                f.write(
                    'const char* buzz(int n){return n%5==0?"Buzz":"No";}\n'
                    "const char* fizz_buzz(int n){\n"
                    '  if(n%15==0) return "FizzBuzz";\n'
                    '  if(n%3==0) return "Fizz";\n'
                    "  return buzz(n);\n"
                    "}\n"
                    "int main(){return 0;}\n"
                )
            r = subprocess.run(["gcc", "-O2", "-o", out, src], capture_output=True)
            if r.returncode != 0 or not os.path.exists(out):
                self.skipTest("gcc -O2 build failed")
            proj = angr.Project(out, auto_load_libs=False, load_debug_info=False)
            funcs, _ = functions(proj)
            start, end = funcs["fizz_buzz"]

            def outcome(n):
                n &= 0xFFFFFFFF
                if n >= 0x80000000:
                    n -= 0x100000000
                if n % 15 == 0:
                    return "FizzBuzz"
                if n % 3 == 0:
                    return "Fizz"
                if n % 5 == 0:
                    return "Buzz"
                return "Unknown"

            from perf.bench import explore, solve

            results = solve(explore(proj, start, end, funcs=funcs))
            outs = {outcome(m["regs"]["rdi"]) for m in results}
            self.assertEqual(outs, {"FizzBuzz", "Fizz", "Buzz", "Unknown"})
        finally:
            shutil.rmtree(d, ignore_errors=True)

    @unittest.skipIf(os.environ.get("PERF_SKIP_ANGREXPLORE"), "angr explore skipped")
    def test_struct_return_gets_a_scratch_return_slot(self):
        if shutil.which("g++") is None:
            self.skipTest("g++ unavailable")

        import subprocess
        import tempfile

        import angr

        from perf.arch import arch as get_arch
        from perf.bench import explore
        from perf.info import functions

        d = tempfile.mkdtemp()
        try:
            src = os.path.join(d, "sret.cpp")
            out = os.path.join(d, "sret")
            with open(src, "w") as f:
                f.write(
                    "#include <string>\n"
                    "std::string to_str(int v){return std::to_string(v);}\n"
                    "int main(){return 0;}\n"
                )
            r = subprocess.run(["g++", "-O2", "-o", out, src], capture_output=True)
            if r.returncode != 0 or not os.path.exists(out):
                self.skipTest("g++ -O2 build failed")
            proj = angr.Project(out, auto_load_libs=False, load_debug_info=False)
            funcs, _ = functions(proj)
            target = next(name for name in funcs if "to_str" in name)
            start, end = funcs[target]

            results = explore(proj, start, end, funcs=funcs)
            self.assertTrue(results, "no solutions: rdi left pointing at nothing")
            scratch = get_arch()._ASM_SCRATCH_BASE
            for sol in results:
                rdi = int(sol["state"].solver.eval(sol["reg"]["rdi"]))
                self.assertTrue(
                    scratch <= rdi < scratch + 0x10000,
                    f"rdi {rdi:#x} is not a scratch page",
                )
        finally:
            shutil.rmtree(d, ignore_errors=True)


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


class TestBenchFilePath(unittest.TestCase):
    def _binary(self):
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        self.addCleanup(Path(path).unlink, missing_ok=True)
        return path

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.load_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_file_path_builds_dataframe(
        self,
        mock_project,
        mock_elf,
        mock_load_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_load_arch.return_value = x86_64
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
            code=f"{path}:func",
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
        self.assertEqual(result["samples"].tolist(), [0, 1])

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.load_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_data_regs_map_to_prototype(
        self,
        mock_project,
        mock_elf,
        mock_load_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_load_arch.return_value = x86_64
        path = self._binary()

        benchmark(
            config={"iterations": 10, "samples": 2},
            mode=["latency"],
            code=f"{path}:func",
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
        from perf.bench import explored_data_dict

        models = [
            {"regs": {"rsi": 42}, "reads": [(0x2000, 8, 5)], "writes": []},
        ]
        with patch("perf.bench.solve", return_value=models):
            d = explored_data_dict(["fake"])
        self.assertEqual(d["regs"]["rsi"], 42)
        self.assertEqual(d["mem"]["0x2000"], 5)

    def test_solver_data_skips_harness_regs(self):
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
        self.assertEqual(maxnode, 65)
        self.assertEqual(list(mask)[0], 1)
        mask, _ = _node_mask(130)
        self.assertEqual(ctypes.sizeof(mask), 24)
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
                code="nop",
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
                code="nop",
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
                code="nop",
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
                code="nop",
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
                code="nop",
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
        with patch.object(core, "default_affinity", return_value=3) as fallback:
            _pin_default_affinity(config, ["duration_time"])
        fallback.assert_called_once_with([["duration_time"]], None)
        self.assertEqual(
            config["thread"],
            [
                [{"priority": "high", "affinity": 3}],
                [{"priority": "low", "affinity": 3}],
            ],
        )

    def test_pin_default_affinity_without_a_thread_config(self):
        from perf.bench import _pin_default_affinity

        config = {}
        with patch.object(core, "default_affinity", return_value=1):
            _pin_default_affinity(config, ["cycles"])
        self.assertEqual(config, {})

    def test_pin_default_affinity_leaves_an_unreachable_pmu_alone(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{"priority": "normal"}]]}
        with patch.object(core, "default_affinity", return_value=None):
            _pin_default_affinity(config, [["topdown-retiring"], ["cycles"]])
        self.assertEqual(config["thread"], [[{"priority": "normal"}]])

    def test_pin_default_affinity_reads_the_numa_node(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{"numa": 1}]]}
        with patch.object(core, "default_affinity", return_value=9) as fallback:
            _pin_default_affinity(config, ["topdown-*"])
        self.assertEqual(fallback.call_args.args[1], 1)
        self.assertEqual(config["thread"], [[{"numa": 1, "affinity": 9}]])

    def test_pin_default_affinity_passes_the_leader_along(self):
        from perf.bench import _pin_default_affinity

        config = {"thread": [[{}]]}
        with (
            patch.object(core, "with_leader", return_value=["p/slots/", "e"]),
            patch.object(core, "default_affinity", return_value=1) as fallback,
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
            code="nop",
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
                code="nop",
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
                code="nop",
                name="t",
            )
        mock_bench.assert_not_called()

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bad_priority_fails_fast(self, mock_bench):
        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 10, "thread": {"priority": ["bogus"]}},
                mode=["latency"],
                code="nop",
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
        self.assertEqual(len(ids), 4)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bad_order_fails_fast(self, mock_bench):
        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 10, "func": {"order": ["bogus"]}},
                mode=["latency"],
                code="nop",
                name="t",
            )
        mock_bench.assert_not_called()

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.load_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_random_order_shuffles_layout(
        self,
        mock_project,
        mock_elf,
        mock_load_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        import tempfile
        from pathlib import Path

        from perf.arch import x86_64

        mock_load_arch.return_value = x86_64
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
            code=f"{path}:func",
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
    @patch("perf.bench.load_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_asis_order_keeps_layout(
        self,
        mock_project,
        mock_elf,
        mock_load_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        import tempfile
        from pathlib import Path

        from perf.arch import x86_64

        mock_load_arch.return_value = x86_64
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
            code=f"{path}:func",
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

    @unittest.skipIf(os.environ.get("PERF_SKIP_ANGREXPLORE"), "angr explore skipped")
    def test_jump_table_is_not_treated_as_data(self):
        import subprocess

        import angr
        import capstone

        from perf.arch import arch as get_arch
        from perf.bench import _static_extra_addrs, _static_table_addrs, explore
        from perf.exec import Elf
        from perf.info import functions

        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
        asm = "\n".join(
            [
                "        .text",
                "        .globl pick",
                "        .type pick, @function",
                "pick:",
                "        and    $0x7, %edi",
                "        jmp    *tbl(%rip)",
                *(
                    f"c{i}:\n        mov    ${11 * (i + 1)}, %eax\n        ret"
                    for i in range(8)
                ),
                "        .size pick, .-pick",
                "        .section .rodata",
                "        .align 8",
                "tbl:",
                "        .quad " + ", ".join(f"c{i}" for i in range(8)),
                "        .text",
                "        .globl main",
                "main:",
                "        xor    %edi, %edi",
                "        call   pick",
                "        ret",
                '        .section .note.GNU-stack,"",@progbits',
            ]
        )
        d = tempfile.mkdtemp()
        try:
            src = os.path.join(d, "jt.s")
            out = os.path.join(d, "jt")
            with open(src, "w") as f:
                f.write(asm)
            r = subprocess.run(
                ["gcc", "-O2", "-no-pie", "-o", out, src], capture_output=True
            )
            if r.returncode != 0 or not os.path.exists(out):
                self.skipTest(f"gcc build failed: {r.stderr.decode()[:200]}")
            proj = angr.Project(out, auto_load_libs=False, load_debug_info=False)
            funcs, _ = functions(proj)
            start, end = funcs["pick"]
            blob = proj.loader.memory.load(start, end - start)
            md = get_arch().disassembler()
            md.detail = True
            table = None
            for insn in md.disasm(bytes(blob), start):
                if insn.mnemonic == "jmp" and any(
                    op.type == capstone.x86.X86_OP_MEM for op in insn.operands
                ):
                    table = get_arch().mem_refs(insn)[0]
            self.assertIsNotNone(table, "expected an indirect jump through a table")
            solutions = explore(proj, start, end, funcs=funcs)
            self.assertTrue(solutions)
            found = _static_table_addrs(
                proj, start, end, Elf(proj.loader), solutions=solutions
            )
            self.assertNotIn(table, found)
            self.assertEqual(found, [])
            self.assertEqual(_static_extra_addrs(found), set())
        finally:
            shutil.rmtree(d, ignore_errors=True)


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
            code="nop",
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
            code="nop",
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
            code="nop",
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
        self.assertNotIn("code", row["config"])

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
                code="nop",
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
            code="nop",
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
            code="nop",
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
            code="mov eax, 42",
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
            code="nop",
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
                code="nop",
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
            code="nop",
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
            code="nop",
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
            code="nop",
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
                code="nop",
                name="t",
                backend="turbo",
            )

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.load_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_file_unroll_duplicates_call(
        self,
        mock_project,
        mock_elf,
        mock_load_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_load_arch.return_value = x86_64
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
            code=f"{path}:func",
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
    @patch("perf.bench.load_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_file_loop_is_default_single_call(
        self,
        mock_project,
        mock_elf,
        mock_load_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_load_arch.return_value = x86_64
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
            code=f"{path}:func",
            name="func",
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        measured = [c for c in codes if "call rax" in c]
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0].count("call rax"), 1)

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.bench.explore", return_value=[])
    @patch("perf.bench.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.load_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_file_throughput_restores_regs_before_each_call(
        self,
        mock_project,
        mock_elf,
        mock_load_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_load_arch.return_value = x86_64
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
            code=f"{path}:func",
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
    @patch("perf.bench.load_arch")
    @patch("perf.bench.Elf")
    @patch("angr.Project")
    def test_file_latency_keeps_regs_out_of_loop_code(
        self,
        mock_project,
        mock_elf,
        mock_load_arch,
        mock_functions,
        mock_explore,
        mock_bench,
    ):
        from perf.arch import x86_64

        mock_load_arch.return_value = x86_64
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
            code=f"{path}:func",
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
        import keystone

        from perf.arch.x86_64 import normalize_asm

        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        for code in ("add eax, 42", "sub eax, 42", "imul eax, eax, 42"):
            raw, _ = ks.asm(code)
            norm, _ = ks.asm(normalize_asm(code + ";"))
            self.assertEqual(bytes(raw), bytes(norm))

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
    @staticmethod
    def _bench_retry(**kw):
        from perf.bench import benchmark

        last = None
        for _ in range(3):
            try:
                return benchmark(**kw)
            except (ValueError, TypeError):
                raise
            except Exception as ex:
                last = ex
        raise last

    @staticmethod
    def _mca_latency(snippet):
        import subprocess

        if shutil.which("llvm-mca") is None:
            return None
        inp = ".intel_syntax noprefix\n" + snippet + "\n"
        try:
            out = subprocess.run(
                ["llvm-mca", "-mcpu=alderlake"],
                input=inp,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            return None
        if out.returncode != 0:
            return None
        import re

        rows = []
        in_table = False
        for line in out.stdout.splitlines():
            if "Instructions:" in line:
                in_table = True
                continue
            if in_table:
                mm = re.match(r"\s*(\d+)\s+(\d+)\s+([\d.]+)", line)
                if mm:
                    rows.append((int(mm.group(1)), int(mm.group(2))))
                elif rows:
                    break
        return rows[0][1] if rows else None

    def test_llvm_mca_reference_latencies(self):

        if shutil.which("llvm-mca") is None:
            self.skipTest("llvm-mca unavailable")
        self.assertEqual(self._mca_latency("add eax, 42"), 1)
        self.assertEqual(self._mca_latency("sub eax, 42"), 1)
        self.assertEqual(self._mca_latency("imul eax, eax, 42"), 3)

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

    def test_latency_ordering_duration_time(self):
        codes = ("nop", "add eax, 42", "sub eax, 42", "imul eax, eax, 42")
        try:
            cur = sorted(os.sched_getaffinity(0))
        except Exception:
            cur = []
        pin = [cur[0]] if cur else None
        last = None
        for _ in range(3):
            cfg = {
                "iterations": 1024,
                "samples": 11,
                "backend": {"unroll": {"count": 20, "runs": 10}},
            }
            if pin is not None:
                cfg["thread"] = {"affinity": list(pin)}
            try:
                self._bench_retry(
                    config=dict(cfg), mode=["latency"], code="nop", name="warmup"
                )
            except Exception:
                pass
            med = {}
            try:
                for code in codes:
                    df = self._bench_retry(
                        config=dict(cfg), mode=["latency"], code=code, name=code
                    )
                    vals = df["duration_time"]
                    self.assertTrue(len(vals) > 0, code)
                    med[code] = float(vals.median())
            except Exception as ex:
                last = ex
                continue
            try:
                self.assertLessEqual(med["nop"], med["add eax, 42"] + 10.0, med)
                self.assertAlmostEqual(
                    med["add eax, 42"], med["sub eax, 42"], delta=10.0
                )
                self.assertGreaterEqual(
                    med["imul eax, eax, 42"] + 1.0, med["add eax, 42"], med
                )
                return
            except AssertionError as ex:
                last = ex
                continue
        raise last

    def test_cycles_match_mca_when_rdpmc_available(self):
        from perf import core

        try:
            c = core.open_event("cycles")
        except Exception as ex:
            self.skipTest(f"rdpmc unavailable: {ex}")
        try:
            c.enable()
            c.close()
        except Exception as ex:
            self.skipTest(f"rdpmc unavailable: {ex}")

        cfg = {"iterations": 512, "samples": 5}
        last = None
        for _ in range(3):
            try:
                add = self._bench_retry(
                    config=dict(cfg),
                    mode=["latency"],
                    code="add eax, 42",
                    name="add",
                    event="cycles",
                    backend="unroll",
                )
                imul = self._bench_retry(
                    config=dict(cfg),
                    mode=["latency"],
                    code="imul eax, eax, 42",
                    name="imul",
                    event="cycles",
                    backend="unroll",
                )
            except Exception as ex:
                last = ex
                continue
            try:
                self.assertAlmostEqual(float(add["cycles"].median()), 1.0, delta=1.0)
                self.assertAlmostEqual(float(imul["cycles"].median()), 3.0, delta=1.0)
                return
            except AssertionError as ex:
                last = ex
                continue
        raise last

    def test_invalid_idiv_syntax_raises(self):
        from perf.bench import benchmark

        with self.assertRaises(ValueError):
            benchmark(
                config={"iterations": 8, "samples": 2},
                mode=["latency"],
                code="idiv eax, 42",
                name="idiv-bad",
            )

    def test_valid_idiv_does_not_crash(self):
        df = self._bench_retry(
            config={"iterations": 128, "samples": 5},
            mode=["latency"],
            code="idiv ecx",
            name="idiv ecx",
            backend="loop",
            data={"eax": 100, "edx": 0, "ecx": 42},
        )
        self.assertTrue(len(df) > 0)
        self.assertGreaterEqual(float(df["duration_time"].median()), 0)

    def test_memory_case_add_r11_mem(self):
        try:
            cur = sorted(os.sched_getaffinity(0))
        except Exception:
            cur = []
        pin = [cur[0]] if cur else None
        last = None
        for _ in range(3):
            cfg = {
                "iterations": 128,
                "samples": 11,
                "backend": {"unroll": {"runs": 10}},
            }
            if pin is not None:
                cfg["thread"] = {"affinity": list(pin)}
            try:
                self._bench_retry(
                    config=dict(cfg), mode=["latency"], code="nop", name="warmup"
                )
            except Exception:
                pass
            try:
                df = self._bench_retry(
                    config=dict(cfg),
                    mode=["latency"],
                    code="add r11, [rax]",
                    name="add r11, [rax]",
                    backend="loop",
                    data={"rax": 0x1000000000, "0x1000000000": 100},
                )
            except Exception as ex:
                last = ex
                continue
            self.assertTrue(len(df) > 0)
            med = float(df["duration_time"].median())
            try:
                self.assertGreater(med, 0)
                return
            except AssertionError as ex:
                last = ex
                continue
        raise last

    def test_idiv_setup_survives_timing(self):
        df = self._bench_retry(
            config={"iterations": 16, "samples": 2},
            mode=["latency"],
            code="cdq; idiv ecx;",
            name="idiv-setup",
            setup=["mov ecx, 2"],
        )
        self.assertTrue(len(df) > 0)

    def test_idiv_setup_survives_cycles(self):
        from perf import core

        try:
            c = core.open_event("cycles")
        except Exception as ex:
            self.skipTest(f"rdpmc unavailable: {ex}")
        try:
            c.enable()
            c.close()
        except Exception as ex:
            self.skipTest(f"rdpmc unavailable: {ex}")

        df = self._bench_retry(
            config={"iterations": 16, "samples": 2},
            mode=["latency"],
            code="cdq; idiv ecx;",
            name="idiv-setup-cycles",
            setup=["mov ecx, 2"],
            event="cycles",
        )
        self.assertTrue(len(df) > 0)
        self.assertGreaterEqual(float(df["cycles"].median()), 0)


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
            code="nop",
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
                code="nop",
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
            code="nop",
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
            code="nop",
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
            code="nop",
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
            code="nop",
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
            benchmark(code="mov eax, 42", mode=["latency"], config=["bad"])

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            benchmark(code=["/nonexistent/a.out", "func"], mode=["latency"], event=[])

    def test_binary_without_target_raises(self):
        with self.assertRaises(ValueError):
            benchmark(code=["/bin/true"], mode=["latency"], event=[])


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

        with self.assertRaises(ValueError) as ex:
            validate_spec({"thread": [[{"affinity": 1}, {"affinity": 2}]]})
        self.assertIn("not supported yet", str(ex.exception))

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
            benchmark(target="foo", mode=["latency"])

    def test_asm_is_not_accepted(self):
        from perf.bench import benchmark

        with self.assertRaises(TypeError):
            benchmark(asm="nop", mode=["latency"])


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


class TestBenchTargetSpec(unittest.TestCase):
    def test_analyze_rejects_file(self):
        from perf.code import analyze

        with self.assertRaises(TypeError):
            analyze(file="a.out", target="foo")

    def test_analyze_rejects_asm(self):
        from perf.code import analyze

        with self.assertRaises(TypeError):
            analyze(asm="nop")


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

        payload = json.loads(to_json(self._df()))
        self.assertEqual(
            list(payload), ["file", "name", "id", "time", "info", "output"]
        )
        self.assertEqual(payload["file"], "a.out")
        self.assertTrue(payload["name"].startswith("foo-"))
        self.assertEqual(len(payload["id"]), 8)
        self.assertEqual(payload["name"], f"foo-{payload['id']}")
        self.assertEqual(
            payload["info"], {"cpu": {"freq": 3000000000.0, "arch": "x86_64"}}
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


class _CompiledCase(unittest.TestCase):
    SOURCE = ""

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.mkdtemp()
        cls._src = os.path.join(cls._dir, "case.c")
        with open(cls._src, "w") as fh:
            fh.write(cls.SOURCE)
        cls._exe = os.path.join(cls._dir, "case")
        rc = subprocess.run(
            ["gcc", "-O2", "-o", cls._exe, cls._src], capture_output=True
        )
        if rc.returncode != 0:
            raise unittest.SkipTest("gcc unavailable")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir, ignore_errors=True)

    def config(self, **kw):
        return {"samples": 1, "iterations": {"count": 8}, **kw}


class TestTlbEviction(_CompiledCase):
    SOURCE = (
        "long touch(const long* p, long n){long s=0;for(long i=0;i<n;i++)"
        "s+=p[i];return s;}\n"
        "int main(void){return 0;}\n"
    )

    def _avoided(self):
        from perf import bench

        return set(bench._UNMAPPABLE_DATA_PAGES)

    def _data(self):
        return {
            "regs": {"rdi": 0x42000000000, "rsi": 1},
            "mem": {"0x42000000000": 5},
        }

    def test_data_pages_steered_cold_are_not_left_unmappable(self):
        before = self._avoided()
        benchmark(
            code="mov rax, [rdi]",
            mode=["latency"],
            data={"regs": {"rdi": 0x42000000000}, "mem": {"0x42000000000": 123}},
            config=self.config(dtlb="cold"),
        )
        self.assertEqual(self._avoided(), before)

    def test_code_pages_are_never_registered_as_unmappable(self):
        before = self._avoided()
        benchmark(
            code=[self._exe, "touch"],
            mode=["latency"],
            data=self._data(),
            config=self.config(itlb="cold"),
        )
        self.assertEqual(self._avoided(), before)

    def test_harness_mprotects_the_code_pages_of_a_cold_itlb(self):
        seen = []
        from perf.arch import arch as get_arch

        original = get_arch().steer_asm

        def spy(*a, **kw):
            out = original(*a, **kw)
            seen.append((list(a[0]["l1i_addrs"]), list(a[0]["tlb_avoid"])))
            return out

        with patch.object(get_arch(), "steer_asm", spy):
            benchmark(
                code=[self._exe, "touch"],
                mode=["latency"],
                data=self._data(),
                config=self.config(itlb="cold"),
            )
        self.assertTrue(seen)
        self.assertTrue(any(l1i_addrs for l1i_addrs, _ in seen))
        for l1i_addrs, avoid in seen:
            for addr in l1i_addrs:
                self.assertNotIn(addr & ~0xFFF, avoid)


class TestTargetRunsInIsolation(_CompiledCase):
    SOURCE = (
        "#include <stdlib.h>\n"
        "#include <unistd.h>\n"
        "int stay(int n){return n*2;}\n"
        "int leave(int n){ exit(0); return n; }\n"
        "int crash(int n){ return *(volatile int *)0 + n; }\n"
        "int sleeper(int n){ usleep(n); return n; }\n"
        "int main(void){return stay(1);}\n"
    )

    def test_a_plain_function_is_measured(self):
        df = benchmark(
            code=[self._exe, "stay"],
            mode=["latency"],
            data={"regs": {"rdi": 3}},
            config=self.config(),
        )
        self.assertFalse(df.empty)

    def test_a_target_that_exits_the_process_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            benchmark(
                code=[self._exe, "leave"],
                mode=["latency"],
                data={"regs": {"rdi": 3}},
                config=self.config(),
            )
        self.assertIn("did not return", str(ctx.exception))

    def test_a_target_that_faults_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            benchmark(
                code=[self._exe, "crash"],
                mode=["latency"],
                data={"regs": {"rdi": 3}},
                config=self.config(),
            )
        self.assertIn("cannot be measured in isolation", str(ctx.exception))

    def test_a_target_that_does_not_finish_is_refused(self):
        from perf import bench

        with patch.object(bench, "_PROBE_TIMEOUT", 0.5):
            with self.assertRaises(ValueError) as ctx:
                benchmark(
                    code=[self._exe, "sleeper"],
                    mode=["latency"],
                    data={"regs": {"rdi": 60_000_000}},
                    config=self.config(),
                )
        self.assertIn("did not return", str(ctx.exception))


class TestRunnableModels(_CompiledCase):
    SOURCE = (
        "struct node { struct node *next; };\n"
        "int depth(struct node *n){int d=0;while(n){d++;n=n->next;}return d;}\n"
        "int spin(void){for(;;){}return 0;}\n"
        "int main(void){return 0;}\n"
    )
    THIS = 0x46000000000

    def _arch(self):
        from perf.arch import load as load_arch
        from perf.bench import _bench_project

        project, obj, _funcs, _protos = _bench_project(
            self._exe,
            {},
            {"size": 0x200000, "align": 16},
            {"align": 16, "order": "as-is"},
        )
        return project, obj, load_arch(project)

    def _probe(self, obj, arch, name):
        return arch.call_seq_asm(obj.get_symbol(name))

    def _model(self, nxt):
        return {
            "regs": {"rdi": self.THIS},
            "reads": [(self.THIS, 8, nxt)],
            "writes": [],
        }

    def test_a_model_whose_walk_never_ends_is_dropped(self):
        from perf.bench import _runtable_models

        _project, obj, arch = self._arch()
        keep = _runtable_models(
            [self._model(self.THIS), self._model(0)],
            arch,
            self._probe(obj, arch, "depth"),
            "",
            "",
            None,
        )
        self.assertEqual(len(keep), 1)
        self.assertEqual(keep[0]["reads"], [(self.THIS, 8, 0)])

    def test_the_same_addresses_with_nothing_in_them_are_tried_next(self):
        from perf.bench import _runtable_models

        _project, obj, arch = self._arch()
        keep = _runtable_models(
            [self._model(self.THIS)],
            arch,
            self._probe(obj, arch, "depth"),
            "",
            "",
            None,
        )
        self.assertEqual(len(keep), 1)
        self.assertEqual(keep[0]["reads"], [])
        self.assertEqual(keep[0]["regs"], {"rdi": self.THIS})

    def test_a_target_that_never_returns_keeps_every_model(self):
        from perf import bench

        _project, obj, arch = self._arch()
        with patch.object(bench, "_MODEL_PROBE_TIMEOUT", 0.2):
            keep = bench._runtable_models(
                [self._model(0)],
                arch,
                self._probe(obj, arch, "spin"),
                "",
                "",
                None,
            )
        self.assertEqual(len(keep), 1)
        self.assertEqual(keep[0]["reads"], [(self.THIS, 8, 0)])

    def test_a_verdict_is_reused_for_the_same_data(self):
        from perf import bench

        _project, obj, arch = self._arch()
        models = [self._model(0)]
        probe = self._probe(obj, arch, "depth")
        first = bench._runtable_models(models, arch, probe, "", "", None)
        with patch.object(bench, "_run_once", Mock(return_value=False)):
            second = bench._runtable_models(models, arch, probe, "", "", None)
        self.assertEqual(first, second)

    def test_no_probe_leaves_the_models_alone(self):
        from perf import bench

        _project, _obj, arch = self._arch()
        models = [self._model(self.THIS)]
        self.assertIs(bench._runtable_models(models, arch, None), models)


class TestTargetOnUnbuiltData(unittest.TestCase):
    SOURCE = (
        "struct entry { entry *next; int key; int value; };\n"
        "struct map {\n"
        "  __attribute__((noinline)) bool find(int key) const {\n"
        "    for (const entry *e = head; e; e = e->next) {\n"
        "      if (e->key == key) {\n"
        "        return true;\n"
        "      }\n"
        "    }\n"
        "    return false;\n"
        "  }\n"
        "  entry *head;\n"
        "};\n"
        "__attribute__((noinline)) int deref(int **p) { return **p; }\n"
        "auto kept = &map::find;\n"
        "auto kept2 = &deref;\n"
        "int main(void) { return 0; }\n"
    )

    @classmethod
    def setUpClass(cls):
        if shutil.which("g++") is None:
            raise unittest.SkipTest("g++ unavailable")
        cls._dir = tempfile.mkdtemp()
        cls._src = os.path.join(cls._dir, "case.cpp")
        with open(cls._src, "w") as fh:
            fh.write(cls.SOURCE)
        cls._exe = os.path.join(cls._dir, "case")
        rc = subprocess.run(
            ["g++", "-O2", "-o", cls._exe, cls._src], capture_output=True
        )
        if rc.returncode != 0:
            shutil.rmtree(cls._dir, ignore_errors=True)
            raise unittest.SkipTest("g++ -O2 build failed")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir, ignore_errors=True)

    def config(self, **kw):
        return {"samples": 1, "iterations": {"count": 8}, **kw}

    def explored(self):
        cls = type(self)
        if getattr(cls, "_explored", None) is None:
            cls._explored = benchmark(
                code=[cls._exe, "map::find(int) const"],
                mode=["latency"],
                config=self.config(),
            )
        return cls._explored

    def _data_columns(self, df):
        return {c for c in df.columns if str(c).startswith("data.0x")}

    def test_a_method_on_a_container_nobody_built_is_measured(self):
        self.assertFalse(self.explored().empty)

    def test_the_containers_addresses_are_found_without_setup(self):
        self.assertTrue(self._data_columns(self.explored()))

    def test_a_pointer_behind_a_pointer_is_measured(self):
        df = benchmark(
            code=[type(self)._exe, "deref(int**)"],
            mode=["latency"],
            config=self.config(),
        )
        self.assertFalse(df.empty)

    def test_an_explicit_setup_still_wins(self):
        this = 0x42000000000
        df = benchmark(
            code=[type(self)._exe, "map::find(int) const"],
            mode=["latency"],
            data={"regs": {"rdi": this, "rsi": 1}, "mem": {f"{this:#x}": 0}},
            config=self.config(),
        )
        self.assertIn("data.rdi", df.columns)
        self.assertEqual(df["data.rdi"].tolist()[0], this)


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


class TestFakedOutput(_CompiledCase):
    SOURCE = (
        "#include <stdio.h>\n"
        "__attribute__((noinline)) long shout(long n){\n"
        '  fprintf(stderr, "err%ld\\n", n);\n'
        '  fprintf(stdout, "out%ld\\n", n);\n'
        "  return n + 1;\n"
        "}\n"
        "int main(void){return (int)shout(1);}\n"
    )

    def _run(self, stdout=None, stderr=None):
        import contextlib
        import io
        import tempfile

        external = {"lib": False, "stdout": False, "stderr": False}
        if stdout is not None:
            external["stdout"] = stdout
        if stderr is not None:
            external["stderr"] = stderr
        saved = os.dup(2)
        out = io.StringIO()
        try:
            with tempfile.TemporaryFile() as sink:
                os.dup2(sink.fileno(), 2)
                try:
                    df = benchmark(
                        code=[self._exe, "shout"],
                        mode=["latency"],
                        config={
                            "samples": 1,
                            "iterations": {"count": 4},
                            "external": external,
                            "branch": "predictable",
                            "dcache": "hot",
                            "dtlb": "hot",
                            "icache": "hot",
                            "itlb": "hot",
                            "backend": {"loop": {"runs": 1, "probes": 1}},
                        },
                    )
                finally:
                    os.dup2(saved, 2)
                sink.seek(0)
                out.write(sink.read().decode("utf-8", "replace"))
        finally:
            os.close(saved)
        with contextlib.suppress(Exception):
            os.close(saved)
        return df, out.getvalue()

    def test_the_target_is_measured_with_faked_streams(self):
        df, err = self._run()
        self.assertFalse(df.empty)
        self.assertNotIn("err", err)

    def test_a_stream_can_be_let_through(self):
        df, err = self._run(stderr=True)
        self.assertFalse(df.empty)
        self.assertIn("err", err)

    def test_the_config_is_reported(self):
        df, _ = self._run()
        self.assertEqual(
            df.attrs["config"]["external"],
            {"lib": False, "stdout": False, "stderr": False},
        )


class TestSharedLibraryBenchmark(unittest.TestCase):
    SOURCE = (
        "long scale_iters = 256;\n"
        "__attribute__((noinline)) long scaled(long x){\n"
        "  long s = 0;\n"
        "  for (long i = 0; i < scale_iters; i++) s += x * (i + 1);\n"
        "  return s;\n"
        "}\n"
    )

    def setUp(self):
        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
        self._dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._dir, True)
        src = os.path.join(self._dir, "lib.c")
        self.so = os.path.join(self._dir, "libdemo.so")
        with open(src, "w") as fh:
            fh.write(self.SOURCE)
        rc = subprocess.run(
            [
                "gcc",
                "-O2",
                "-fPIC",
                "-shared",
                "-Wl,-soname,libdemo.so",
                "-o",
                self.so,
                src,
            ],
            capture_output=True,
        )
        if rc.returncode != 0 or not os.path.exists(self.so):
            self.skipTest("gcc build failed")

    def test_a_shared_library_target_is_measured(self):
        df = benchmark(
            code=[self.so, "scaled"],
            mode=["latency"],
            config={
                "samples": 1,
                "iterations": {"count": 8},
                "branch": "predictable",
                "dcache": "hot",
                "dtlb": "hot",
                "icache": "hot",
                "itlb": "hot",
                "code": {"align": 16},
                "backend": {"loop": {"runs": 4, "probes": 1}},
            },
        )
        self.assertFalse(df.empty)
        self.assertEqual(set(df.index.get_level_values("name")), {"scaled"})
        self.assertIn("duration_time", df.columns)
        self.assertTrue(df["duration_time"].notna().all(), df["duration_time"].tolist())
        self.assertTrue((df["duration_time"] > 0).all(), df["duration_time"].tolist())


if __name__ == "__main__":
    unittest.main()
