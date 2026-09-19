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
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from perf.bench import _DEFAULT_BENCH, bench


def _ticks_to_ns(df, ticks):
    hz = df.attrs["info"]["cpu"]["hz"]
    return [t * 1e9 / hz for t in ticks]


class TestModes(unittest.TestCase):
    @patch("perf.bench._bench_one")
    def test_mode_defaults_to_both(self, mock_one):
        def fake(**kw):
            return pd.DataFrame(
                [{"mode": kw["mode"], "branch": kw["config"].get("branch")}]
            )

        mock_one.side_effect = fake
        result = bench(
            asm="nop",
            name="t",
            config={"branch": ["predictable", "unpredictable"]},
        )
        self.assertEqual(sorted(result["mode"].unique()), ["latency", "throughput"])
        self.assertEqual(
            sorted(result["branch"].unique()), ["predictable", "unpredictable"]
        )

    @patch("perf.bench._bench_one")
    def test_mode_list_runs_with_each_config_combo(self, mock_one):
        def fake(**kw):
            return pd.DataFrame([{"mode": kw["mode"]}])

        mock_one.side_effect = fake
        bench(
            asm="nop",
            name="t",
            mode=["throughput", "latency"],
            config={"branch": ["predictable", "unpredictable"]},
        )
        modes = [c.kwargs["mode"] for c in mock_one.call_args_list]
        self.assertEqual(modes[0], "throughput")
        self.assertEqual(modes[-1], "latency")

    def test_normalize_modes(self):
        from perf.bench import _normalize_modes as norm

        self.assertEqual(norm(None), ["latency", "throughput"])
        self.assertEqual(norm("throughput"), ["throughput"])
        self.assertEqual(norm("latency,throughput"), ["latency", "throughput"])
        self.assertEqual(norm(["latency", "latency"]), ["latency"])
        with self.assertRaises(ValueError):
            norm("bogus")


class TestBenchCodePath(unittest.TestCase):
    @patch("perf.bench._bench")
    def test_code_path_builds_dataframe(self, mock_bench):
        mock_bench.return_value = pd.Series([10, 20, 30])

        result = bench(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            asm="nop",
            name="mytarget",
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
        result = bench(asm="nop", name="t", mode="latency", config=None)
        cfg = mock_bench.call_args.kwargs["config"]

        for key in (
            "samples",
            "runs",
            "target_rel_se",
            "min_iterations",
            "max_iterations",
            "probe_runs",
            "unroll_n",
        ):
            self.assertEqual(cfg[key], _DEFAULT_BENCH[key])

        self.assertIsInstance(cfg["seed"], int)
        self.assertIsInstance(cfg["thread"]["affinity"], list)
        self.assertEqual(cfg["thread"]["priority"], "normal")
        self.assertEqual(cfg["backend"], "unroll")
        self.assertEqual(cfg["func"]["order"], "as-is")
        self.assertIsInstance(cfg["iterations"], int)

        spec = result.attrs["config"]
        self.assertEqual(spec["branch"], ["predictable", "unpredictable"])
        self.assertEqual(spec["cache"], ["hot", "warm", "cool", "cold"])
        self.assertIsNone(spec["iterations"])
        self.assertNotIn("data", spec)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_event_propagates_and_names_column(self, mock_bench):
        result = bench(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
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

        result = bench(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            asm="nop",
            name="mytarget",
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

        result = bench(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            asm="nop",
            name="mytarget",
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

        result = bench(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            asm="nop",
            name="mytarget",
            event=[["cycles"], ["cache-misses"]],
        )

        self.assertIn("cycles", result.columns)
        self.assertIn("cache-misses", result.columns)
        self.assertEqual(result["cycles"].tolist(), [2.0, 4.0])
        self.assertEqual(result["cache-misses"].tolist(), [0.0, 0.0])

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_data_propagates(self, mock_bench):
        bench(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
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


class TestSolverDataModels(unittest.TestCase):
    def test_cache_levels_parsing(self):
        from perf.arch.x86_64 import data_cache_levels

        cfg = {
            "cache": {
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
            data_cache_levels({"cache": "hot"}), {"L1d": 100, "L2": 0, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"cache": "warm"}), {"L1d": 0, "L2": 100, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"cache": "cool"}), {"L1d": 0, "L2": 0, "L3": 100}
        )
        self.assertEqual(
            data_cache_levels({"cache": "cold"}), {"L1d": 0, "L2": 0, "L3": 0}
        )
        self.assertIsNone(data_cache_levels({}))

    def test_cache_levels_top_level(self):
        from perf.arch.x86_64 import data_cache_levels

        cfg = {"cache": {"L1d": {"hit_rate": 50}, "L2": {"hit_rate": 100}}}
        levels = data_cache_levels(cfg)
        self.assertEqual(levels["L1d"], 50)
        self.assertEqual(levels["L2"], 100)

    def test_cache_levels_default(self):
        from perf.arch.x86_64 import _MEMORY_SHORTCUTS, data_cache_levels
        from perf.bench import _DEFAULT_BENCH

        self.assertEqual(_DEFAULT_BENCH["cache"], ["hot", "warm", "cool", "cold"])
        self.assertEqual(
            data_cache_levels({"cache": "hot"}), dict(_MEMORY_SHORTCUTS["hot"])
        )
        levels = data_cache_levels({"cache": _DEFAULT_BENCH["cache"][0]})
        self.assertEqual(levels["L1d"], 100)

    def test_default_config_includes_cache(self):
        from perf.bench import _DEFAULT_BENCH

        self.assertEqual(_DEFAULT_BENCH["cache"], ["hot", "warm", "cool", "cold"])

    def test_sample_tier_respects_hit_rate(self):
        import random

        from perf.arch.x86_64 import sample_tier

        rng = random.Random(1)
        levels = {"L1d": 100, "L2": None, "L3": None}
        tiers = {sample_tier(levels, rng) for _ in range(200)}
        self.assertEqual(tiers, {"L1d"})

        self.assertEqual(sample_tier(None, random.Random(0)), "L1d")

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
        self.assertEqual(asm.count(".Ld:"), 1)
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
        self.assertEqual(
            buf[et, 2 * n], data_ptr + meta["tier_col"][addr] * meta["K"] * 8
        )

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
            self.assertEqual(
                buf[et, 2 * n + i], data_ptr + meta["tier_col"][a] * meta["K"] * 8
            )


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
        self.assertEqual(lo, 128)
        self.assertEqual(hi, 5_000_000)

        target, lo, hi = _probe_plan(
            {"target_rel_se": 0.01, "min_iterations": 64, "max_iterations": 9999}
        )
        self.assertEqual((target, lo, hi), (0.01, 64, 9999))


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

        result = bench(
            config={
                "iterations": 10,
                "samples": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            file=path,
            target="func",
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

        bench(
            config={"iterations": 10, "samples": 2},
            mode="latency",
            file=path,
            target="func",
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
            ARG_REGS,
            arg_alias,
            canonical_data_reg,
            data_reg_keys,
        )

        for idx, reg in enumerate(ARG_REGS):
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
            bench(
                config={"iterations": 10, "thread": {"affinity": ["bogus"]}},
                mode="latency",
                asm="nop",
                name="t",
            )
        mock_bench.assert_not_called()

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bad_priority_fails_fast(self, mock_bench):
        with self.assertRaises(ValueError):
            bench(
                config={"iterations": 10, "thread": {"priority": ["bogus"]}},
                mode="latency",
                asm="nop",
                name="t",
            )
        mock_bench.assert_not_called()


class TestFunctionOrder(unittest.TestCase):
    def test_resolve_orders(self):
        from perf.bench import _resolve_function_order

        self.assertEqual(_resolve_function_order({}), "as-is")
        self.assertEqual(_resolve_function_order(None), "as-is")
        self.assertEqual(
            _resolve_function_order({"func": {"order": "random"}}), "random"
        )
        self.assertEqual(_resolve_function_order({"func": "random"}), "random")
        self.assertEqual(_resolve_function_order({"func": {"order": "as-is"}}), "as-is")
        with self.assertRaises(ValueError):
            _resolve_function_order({"func": {"order": "bogus"}})

    def test_resolve_alignment(self):
        from perf.bench import _resolve_function_alignment

        self.assertEqual(_resolve_function_alignment({}), 16)
        self.assertEqual(_resolve_function_alignment(None), 16)
        self.assertEqual(_resolve_function_alignment({"func": {"align": 64}}), 64)
        with self.assertRaises(ValueError):
            _resolve_function_alignment({"func": {"align": 24}})
        with self.assertRaises(ValueError):
            _resolve_function_alignment({"func": {"align": "many"}})

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bad_order_fails_fast(self, mock_bench):
        with self.assertRaises(ValueError):
            bench(
                config={"iterations": 10, "func": {"order": ["bogus"]}},
                mode="latency",
                asm="nop",
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

        bench(
            config={
                "iterations": 10,
                "samples": 2,
                "seed": [9],
                "func": {"order": ["random"], "align": [32]},
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            file=path,
            target="func",
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

        bench(
            config={
                "iterations": 10,
                "samples": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            file=path,
            target="func",
            name="func",
        )

        obj = mock_elf.return_value
        obj.randomize_layout.assert_not_called()
        obj.write_perf_map.assert_called_once_with()


class TestDefaultBench(unittest.TestCase):
    def test_all_keys_have_defaults(self):
        from perf.bench import _DEFAULT_BENCH

        self.assertEqual(_DEFAULT_BENCH["seed"], [None])
        self.assertEqual(
            _DEFAULT_BENCH["thread"], {"affinity": [None], "priority": ["normal"]}
        )
        self.assertEqual(_DEFAULT_BENCH["branch"], ["predictable", "unpredictable"])
        self.assertEqual(_DEFAULT_BENCH["cache"], ["hot", "warm", "cool", "cold"])
        self.assertEqual(_DEFAULT_BENCH["func"], {"align": [16], "order": ["as-is"]})
        self.assertEqual(_DEFAULT_BENCH["samples"], 100)
        self.assertEqual(_DEFAULT_BENCH["runs"], 10)
        self.assertIsNone(_DEFAULT_BENCH["iterations"])
        self.assertEqual(_DEFAULT_BENCH["target_rel_se"], 0.005)
        self.assertEqual(_DEFAULT_BENCH["min_iterations"], 128)
        self.assertEqual(_DEFAULT_BENCH["max_iterations"], 5_000_000)
        self.assertEqual(_DEFAULT_BENCH["probe_runs"], 3)
        self.assertIsNone(_DEFAULT_BENCH["backend"])
        self.assertEqual(_DEFAULT_BENCH["unroll_n"], 5)
        self.assertNotIn("data", _DEFAULT_BENCH)

    def test_none_backend_and_unroll_fall_back(self):
        from perf.bench import _DEFAULT_BENCH as _DB2
        from perf.bench import _resolve_backend, _resolve_unroll_n

        self.assertEqual(_resolve_backend(None, {"backend": None}, "loop"), "loop")
        self.assertEqual(_resolve_backend(None, {"backend": None}, "unroll"), "unroll")
        self.assertEqual(_resolve_unroll_n(None, {"unroll_n": None}), _DB2["unroll_n"])


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

    def test_default_asm_backend(self):
        from unittest.mock import patch

        import pandas as pd

        from perf.bench import bench

        with patch("perf.bench._bench", return_value=pd.Series([10, 20, 30])):
            result = bench(
                config={
                    "iterations": 1000,
                    "samples": 3,
                    "branch": "unpredictable",
                    "cache": "hot",
                },
                mode="latency",
                asm="nop",
                name="t",
            )
        self.assertEqual(result.attrs["config"]["backend"], "unroll")

    def test_resolve_unroll_n(self):
        from perf.bench import _DEFAULT_BENCH as _DB3
        from perf.bench import _resolve_unroll_n

        self.assertEqual(_resolve_unroll_n(None, {}), _DB3["unroll_n"])
        self.assertEqual(_resolve_unroll_n(None, {"unroll_n": 3}), 3)
        self.assertEqual(_resolve_unroll_n(7, {"unroll_n": 3}), 7)
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
        result = bench(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            asm="nop",
            name="mytarget",
        )
        self.assertEqual(result["operations"].tolist(), [1, 1, 1])
        self.assertEqual(
            result["duration_time"].tolist(),
            _ticks_to_ns(result, [2.0, 4.0, 6.0]),
        )

    @patch("perf.bench._bench", return_value=(pd.Series([10, 20]), 1000))
    def test_operations_is_iterations_for_throughput(self, mock_bench):
        result = bench(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="throughput",
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
        result = bench(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "cache": "hot",
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
        result = bench(
            config={
                "iterations": 1000,
                "samples": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="throughput",
            asm="nop",
            name="mytarget",
        )
        self.assertEqual(result.attrs["config"]["backend"], "loop")
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertEqual(codes[1], ".align 16\nnop;")

    def test_unroll_rejected_for_throughput(self):
        with self.assertRaises(ValueError):
            bench(
                config={"iterations": 100, "samples": 1},
                mode="throughput",
                asm="nop",
                name="t",
                backend="unroll",
            )

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_asm_unroll_is_2N_minus_N(self, mock_bench):
        result = bench(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            asm="nop",
            name="mytarget",
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
        result = bench(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
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
        result = bench(
            config={
                "iterations": 1000,
                "samples": 3,
                "unroll_n": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
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
            bench(
                config={"iterations": 1},
                mode="latency",
                asm="nop",
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

        result = bench(
            config={
                "iterations": 10,
                "samples": 2,
                "unroll_n": 3,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            file=path,
            target="func",
            name="func",
            backend="unroll",
        )
        codes = [c.kwargs["code"] for c in mock_bench.call_args_list]
        self.assertEqual(len(codes), 2)
        n_copies = codes[0].count("call rax")
        self.assertEqual(n_copies, 3)
        self.assertEqual(codes[1].count("call rax"), 6)
        hz = result.attrs["info"]["cpu"]["hz"]
        self.assertAlmostEqual(result["duration_time"].tolist()[0], (50 / 3) * 1e9 / hz)

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

        bench(
            config={
                "iterations": 10,
                "samples": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            file=path,
            target="func",
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

        bench(
            config={
                "iterations": 10,
                "samples": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="throughput",
            file=path,
            target="func",
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

        bench(
            config={
                "iterations": 10,
                "samples": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
            file=path,
            target="func",
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
        self.assertEqual(asm.count(".Ld:"), 1)
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
        self.assertIn(".Ld:", steer)
        self.assertIn("mov rax, [r14 + r8*8]", prime)
        self.assertIn("mfence", steer)
        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        enc, _ = ks.asm("\n".join(p for p in (steer, prime) if p))
        self.assertTrue(enc)


class TestAsmAccuracyAlderLake(unittest.TestCase):
    @staticmethod
    def _bench_retry(**kw):
        from perf.bench import bench

        last = None
        for _ in range(3):
            try:
                return bench(**kw)
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
            cfg = {"iterations": 1024, "samples": 11, "runs": 10, "unroll_n": 20}
            if pin is not None:
                cfg["thread"] = {"affinity": list(pin)}
            try:
                self._bench_retry(
                    config=dict(cfg), mode="latency", asm="nop", name="warmup"
                )
            except Exception:
                pass
            med = {}
            try:
                for code in codes:
                    df = self._bench_retry(
                        config=dict(cfg), mode="latency", asm=code, name=code
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
                    mode="latency",
                    asm="add eax, 42",
                    name="add",
                    event="cycles",
                )
                imul = self._bench_retry(
                    config=dict(cfg),
                    mode="latency",
                    asm="imul eax, eax, 42",
                    name="imul",
                    event="cycles",
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
        from perf.bench import bench

        with self.assertRaises(ValueError):
            bench(
                config={"iterations": 8, "samples": 2},
                mode="latency",
                asm="idiv eax, 42",
                name="idiv-bad",
            )

    def test_valid_idiv_does_not_crash(self):
        df = self._bench_retry(
            config={"iterations": 128, "samples": 5},
            mode="latency",
            asm="idiv ecx",
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
            cfg = {"iterations": 128, "samples": 11, "runs": 10}
            if pin is not None:
                cfg["thread"] = {"affinity": list(pin)}
            try:
                self._bench_retry(
                    config=dict(cfg), mode="latency", asm="nop", name="warmup"
                )
            except Exception:
                pass
            try:
                df = self._bench_retry(
                    config=dict(cfg),
                    mode="latency",
                    asm="add r11, [rax]",
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
            mode="latency",
            asm="cdq; idiv ecx;",
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
            mode="latency",
            asm="cdq; idiv ecx;",
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

        self.assertEqual(_resolve_unroll_n(None, {}), _DB3["unroll_n"])
        self.assertEqual(_resolve_unroll_n(None, {"unroll_n": 1000}), 1000)
        self.assertEqual(_resolve_unroll_n(7, {"unroll_n": 1000}), 7)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_asm_unroll_n_from_config(self, mock_bench):
        result = bench(
            config={
                "iterations": 1000,
                "samples": 3,
                "unroll_n": 2,
                "branch": "unpredictable",
                "cache": "hot",
            },
            mode="latency",
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
        bench(
            config={
                "iterations": 1000,
                "samples": 3,
            },
            data={"rdi": 21},
            mode="latency",
            asm="nop",
            name="t",
        )
        for call in mock_bench.call_args_list:
            self.assertEqual(call.kwargs["data"]["regs"]["rdi"], 21)

    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bench_config_data_rejected(self, mock_bench):
        with self.assertRaises(ValueError):
            bench(
                config={
                    "iterations": 1000,
                    "samples": 3,
                    "data": {"regs": {"rdi": 21}},
                },
                mode="latency",
                asm="nop",
                name="t",
            )
        mock_bench.assert_not_called()


class TestResultHash(unittest.TestCase):
    @patch("perf.bench._bench", return_value=pd.Series([10, 20, 30]))
    def test_bench_attrs_and_label(self, mock_bench):
        result = bench(
            config={
                "iterations": 1000,
                "samples": 3,
                "branch": "unpredictable",
                "cache": "hot",
            },
            data={"rdi": 1},
            mode="latency",
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
        result = bench(
            config={
                "iterations": 1000,
                "seed": [42],
                "branch": "unpredictable",
                "cache": "hot",
            },
            data={"rdi": 1},
            mode="latency",
            asm="nop",
            name="t",
        )
        result2 = bench(
            config={
                "iterations": 1000,
                "seed": [42],
                "branch": "unpredictable",
                "cache": "hot",
            },
            data={"rdi": 2},
            mode="latency",
            asm="nop",
            name="t",
        )
        self.assertNotEqual(result.attrs["id"], result2.attrs["id"])
        result3 = bench(
            config={
                "iterations": 1000,
                "seed": [43],
                "branch": "unpredictable",
                "cache": "hot",
            },
            data={"rdi": 1},
            mode="latency",
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
                "cache": {
                    "L1d": {"hit_rate": 100},
                    "0x41000000000": "hot",
                },
                "branch": {"mem": {"0x401000": "predictable"}},
            }
        )
        mc = cfg["cache"]
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
        self.assertEqual(cfg["cache"], ["hot", "warm", "cool", "cold"])

    def test_cache_shortcut_forms(self):
        from perf.bench import _expand_config_spec, _merge_config

        self.assertEqual(_merge_config({"cache": "cold"})["cache"], "cold")
        self.assertEqual(
            _merge_config({"cache": ["cold", "hot"]})["cache"], ["cold", "hot"]
        )
        combos = _expand_config_spec(
            _merge_config({"cache": "cold", "branch": "predictable"})
        )
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0]["cache"], "cold")


class TestBenchTargetExclusivity(unittest.TestCase):
    def test_code_and_filter_rejected(self):
        from perf.bench import bench

        with self.assertRaises(TypeError):
            bench(
                asm="nop",
                target="foo",
                mode="latency",
                config={"iterations": 8, "samples": 2},
            )

    def test_code_and_filter_rejected_region_form(self):
        from perf.bench import bench

        with self.assertRaises(TypeError):
            bench(
                asm="nop",
                target="a..b",
                mode="latency",
                config={"iterations": 8, "samples": 2},
            )


class TestNormalizeTarget(unittest.TestCase):
    def test_tuple_becomes_region(self):
        from perf.bench import _normalize_target

        self.assertEqual(
            _normalize_target(("hot_begin", "hot_end")), "hot_begin..hot_end"
        )
        self.assertEqual(_normalize_target(["a", "b"]), "a..b")
        self.assertEqual(_normalize_target("foo"), "foo")
        self.assertIsNone(_normalize_target(None))
        with self.assertRaises(ValueError):
            _normalize_target(("only_one",))
        with self.assertRaises(ValueError):
            _normalize_target(("", "b"))

    def test_bench_rejects_asm_and_target(self):
        from perf.bench import bench

        with self.assertRaises(TypeError):
            bench(asm="nop", target="foo", mode="latency")


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


class TestModelsForIteration(unittest.TestCase):
    def test_predictable_without_overrides_cycles_in_order(self):
        from perf.bench import _models_for_iteration

        models = [
            {"regs": {"rdi": 1}, "reads": [], "writes": []},
            {"regs": {"rdi": 2}, "reads": [], "writes": []},
        ]
        picked = [
            _models_for_iteration(
                models, True, random.Random(0), branch_cfg=None, it=it
            )[0]
            for it in range(6)
        ]
        self.assertEqual(picked, [models[0], models[1]] * 3)

    def test_predictable_without_overrides_pins_first(self):
        from perf.bench import _models_for_iteration

        models = [
            {"regs": {"rdi": 1}, "reads": [], "writes": []},
            {"regs": {"rdi": 2}, "reads": [], "writes": []},
        ]
        for _ in range(10):
            picked = _models_for_iteration(models, True, random.Random(0))
            self.assertEqual(picked, [models[0]])


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


class TestValidateNumerics(unittest.TestCase):
    def test_bad_samples(self):
        from perf.bench import validate

        with self.assertRaises(ValueError):
            validate({"samples": 0})
        with self.assertRaises(ValueError):
            validate({"samples": "many"})

    def test_bad_runs(self):
        from perf.bench import validate

        with self.assertRaises(ValueError):
            validate({"runs": 0})

    def test_bad_target_rel_se(self):
        from perf.bench import validate

        with self.assertRaises(ValueError):
            validate({"target_rel_se": 0})
        with self.assertRaises(ValueError):
            validate({"target_rel_se": -1})
        with self.assertRaises(ValueError):
            validate({"target_rel_se": "many"})

    def test_bad_backend(self):
        from perf.bench import validate

        with self.assertRaises(ValueError):
            validate({"backend": "turbo"})

    def test_min_max_order(self):
        from perf.bench import validate

        with self.assertRaises(ValueError):
            validate({"min_iterations": 1000, "max_iterations": 10})

    def test_branch_string_validated(self):
        from perf.bench import validate

        with self.assertRaises(ValueError):
            validate({"branch": "sometimes"})

    def test_valid_numerics(self):
        from perf.bench import validate

        cfg = validate(
            {
                "samples": 10,
                "runs": 2,
                "iterations": 128,
                "min_iterations": 64,
                "max_iterations": 1000,
                "target_rel_se": 0.01,
                "probe_runs": 2,
                "unroll_n": 3,
                "backend": "loop",
            }
        )
        self.assertEqual(cfg["samples"], 10)


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


class TestDisassemblePrototype(unittest.TestCase):
    def _run_disassemble(self, data):
        import importlib
        import sys

        bench_mod = sys.modules.get("perf.bench") or importlib.import_module(
            "perf.bench"
        )

        captured = {}
        fake_project = Mock()
        fake_project.loader.find_symbol.return_value = None
        fake_arch = Mock()
        fake_arch.call_asm.side_effect = AssertionError("no setup expected")
        fake_arch.SETUP_BASE = 0x1000000

        def _fake_explore(*args, **kwargs):
            captured.update(kwargs)
            return []

        with (
            patch("angr.Project", return_value=fake_project),
            patch.object(
                bench_mod, "functions", return_value=({"foo": (0x1000, 0x1010)}, {})
            ),
            patch.object(bench_mod, "load_arch", return_value=fake_arch),
            patch.object(
                bench_mod, "resolve_targets", return_value=[("foo", 0x1000, 0x1010)]
            ),
            patch.object(bench_mod, "explore", side_effect=_fake_explore),
            patch.object(
                bench_mod,
                "_disasm_target",
                return_value=".intel_syntax noprefix\nret\n",
            ),
        ):
            text = bench_mod.disassemble(file="/tmp/fake", target="foo", data=data)
        return text, captured

    def test_data_arg_passes_concrete_prototype(self):
        text, captured = self._run_disassemble({"regs": {"arg0": 15}})
        self.assertIn("ret", text)
        self.assertIsNotNone(captured.get("prototype"))
        self.assertEqual(len(captured["prototype"].args), 1)

    def test_no_data_still_passes_synthesized_prototype(self):
        text, captured = self._run_disassemble(None)
        self.assertIn("ret", text)
        self.assertIsNotNone(captured.get("prototype"))
        self.assertEqual(len(captured["prototype"].args), 0)


class TestDisasmExecutedAsm(unittest.TestCase):
    def _project_with_blocks(self, blocks):
        project = Mock()

        def _block(addr):
            return SimpleNamespace(
                capstone=SimpleNamespace(insns=list(blocks[int(addr)]))
            )

        project.factory.block.side_effect = _block
        return project

    def _insn(self, addr, text):
        mnemonic, _, op_str = text.partition(" ")
        return SimpleNamespace(address=addr, mnemonic=mnemonic, op_str=op_str)

    def test_blocks_rendered_in_address_order(self):
        from perf.bench import _disasm_executed_asm

        arch = SimpleNamespace(SETUP_BASE=0x1000000)
        blocks = {
            0x1000: [self._insn(0x1000, "mov rax, 1")],
            0x2000: [self._insn(0x2000, "mov rbx, 2")],
            0x3000: [self._insn(0x3000, "mov rcx, 3")],
        }
        project = self._project_with_blocks(blocks)
        text = _disasm_executed_asm(project, [0x3000, 0x1000, 0x2000], arch)
        self.assertLess(text.index("rax, 1"), text.index("rbx, 2"))
        self.assertLess(text.index("rbx, 2"), text.index("rcx, 3"))

    def test_overlapping_insns_emitted_once(self):
        from perf.bench import _disasm_executed_asm

        arch = SimpleNamespace(SETUP_BASE=0x1000000)
        shared = [
            self._insn(0x1000, "mov rax, 1"),
            self._insn(0x1001, "mov rbx, 2"),
        ]
        blocks = {
            0x1000: shared,
            0x1001: [shared[1]],
        }
        project = self._project_with_blocks(blocks)
        text = _disasm_executed_asm(project, [0x1000, 0x1001], arch)
        self.assertEqual(text.count("mov rbx, 2"), 1)


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

        for choice in (
            "predictable",
            "unpredictable",
            "unpredictable.exponential",
        ):
            cfg = _branch_config({"branch": choice})
            self.assertEqual(cfg["prediction"], choice)

    def test_prefixed_choice_case_insensitive(self):
        from perf.bench import _branch_config

        cfg = _branch_config({"branch": "  UNPREDICTABLE.Exponential "})
        self.assertEqual(cfg["prediction"], "unpredictable.exponential")

    def test_uniform_alias_normalizes_to_unpredictable(self):
        from perf.bench import _branch_config, _branch_value

        self.assertEqual(_branch_value("unpredictable.uniform"), "unpredictable")
        self.assertEqual(_branch_value("  UNPREDICTABLE.Uniform "), "unpredictable")
        cfg = _branch_config({"branch": "unpredictable.uniform"})
        self.assertEqual(cfg, {"prediction": "unpredictable", "mem": {}, "regs": {}})
        cfg = _branch_config(
            {
                "branch": {
                    "prediction": "unpredictable.uniform",
                    "mem": {"0x401000": "UNPREDICTABLE.UNIFORM"},
                    "regs": {"rdi": "unpredictable.uniform"},
                }
            }
        )
        self.assertEqual(cfg["prediction"], "unpredictable")
        self.assertEqual(cfg["mem"], {0x401000: "unpredictable"})
        self.assertEqual(cfg["regs"], {"rdi": "unpredictable"})

    def test_gaussian_rejected(self):
        from perf.bench import _branch_config

        with self.assertRaises(ValueError):
            _branch_config({"branch": "gaussian"})

    def test_removed_choices_rejected(self):
        from perf.bench import _branch_config

        for choice in (
            "uniform",
            "random",
            "shuffle",
            "normal",
            "exponential",
            "predictable.exponential",
            "predictable.uniform",
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

        cfg = _branch_config({"branch": {"prediction": "unpredictable.exponential"}})
        self.assertEqual(cfg["prediction"], "unpredictable.exponential")
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"default": "unpredictable.exponential"}})

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
                    "mem": {"0x401000": "unpredictable.exponential"},
                    "regs": {"rdi": "predictable"},
                }
            }
        )
        self.assertEqual(cfg["mem"], {0x401000: "unpredictable.exponential"})
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
        from perf.bench import validate

        for choice in (
            "predictable",
            "unpredictable",
            "unpredictable.exponential",
        ):
            validate({"branch": choice})


class TestBranchDistributions(unittest.TestCase):
    def test_global_distribution_precedence(self):
        from perf.bench import _branch_global_distribution

        self.assertEqual(
            _branch_global_distribution(
                {"prediction": "unpredictable.exponential"}, False
            ),
            "unpredictable.exponential",
        )
        self.assertEqual(
            _branch_global_distribution(None, "unpredictable.exponential"),
            "unpredictable.exponential",
        )
        self.assertEqual(_branch_global_distribution(None, True), "predictable")
        self.assertEqual(_branch_global_distribution(None, False), "unpredictable")

    def test_global_distribution_resolves_alias(self):
        from perf.bench import _branch_global_distribution

        self.assertEqual(
            _branch_global_distribution(None, "unpredictable.uniform"),
            "unpredictable",
        )
        self.assertEqual(
            _branch_global_distribution({"prediction": "unpredictable.uniform"}, False),
            "unpredictable",
        )

    def test_reg_and_mem_overrides(self):
        from perf.bench import _branch_mem_distribution, _branch_reg_distribution

        cfg = {
            "prediction": "unpredictable",
            "mem": {0x1000: "unpredictable.exponential"},
            "regs": {"rdi": "predictable"},
        }
        self.assertEqual(_branch_reg_distribution("rdi", cfg, False), "predictable")
        self.assertEqual(_branch_reg_distribution("rsi", cfg, False), "unpredictable")
        self.assertEqual(
            _branch_mem_distribution(0x1000, cfg, False), "unpredictable.exponential"
        )
        self.assertEqual(_branch_mem_distribution(0x2000, cfg, False), "unpredictable")

    def test_reg_and_mem_overrides_resolve_alias(self):
        from perf.bench import (
            _branch_global_distribution,
            _branch_mem_distribution,
            _branch_reg_distribution,
        )

        cfg = {
            "prediction": "unpredictable.uniform",
            "mem": {0x1000: "UNPREDICTABLE.UNIFORM"},
            "regs": {"rdi": "unpredictable.uniform"},
        }
        self.assertEqual(_branch_reg_distribution("rdi", cfg, False), "unpredictable")
        self.assertEqual(_branch_mem_distribution(0x1000, cfg, False), "unpredictable")
        self.assertEqual(_branch_global_distribution(cfg, False), "unpredictable")

    def test_sample_index_bounds(self):
        from perf.bench import _branch_sample_index

        for dist in (
            "predictable",
            "unpredictable",
            "unpredictable.exponential",
        ):
            rng = random.Random(0)
            for it in range(50):
                idx = _branch_sample_index(4, rng, dist, it)
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, 4)
        self.assertEqual(
            _branch_sample_index(1, random.Random(0), "unpredictable.exponential", 7),
            0,
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

    def test_sample_choice_predictable_cycles(self):
        from perf.bench import _branch_sample_choice

        vals = [10, 20, 30]
        self.assertEqual(
            [
                _branch_sample_choice(vals, random.Random(0), "predictable", it=i)
                for i in range(5)
            ],
            [10, 20, 30, 10, 20],
        )

    def test_sample_choice_stays_in_set(self):
        from perf.bench import _branch_sample_choice

        for dist in (
            "unpredictable",
            "unpredictable.uniform",
            "unpredictable.exponential",
        ):
            rng = random.Random(1)
            for i in range(30):
                self.assertIn(
                    _branch_sample_choice([1, 2, 3, 4], rng, dist, i), [1, 2, 3, 4]
                )
        self.assertIsNone(
            _branch_sample_choice([], random.Random(0), "unpredictable.exponential", 0)
        )
        self.assertEqual(
            _branch_sample_choice(
                [7], random.Random(0), "unpredictable.exponential", 2
            ),
            7,
        )


class TestBranchModelsAndPerIter(unittest.TestCase):
    def test_models_predictable_cycles(self):
        from perf.bench import _models_for_iteration

        models = [
            {"regs": {"rdi": 1}, "reads": [], "writes": []},
            {"regs": {"rdi": 2}, "reads": [], "writes": []},
        ]
        cfg = {"prediction": "predictable", "mem": {}, "regs": {}}
        picked = []
        for i in range(4):
            choice = _models_for_iteration(
                models, False, random.Random(0), branch_cfg=cfg, it=i
            )[0]
            picked.append(choice["regs"]["rdi"])
        self.assertEqual(picked, [1, 2, 1, 2])

    def test_models_new_distributions_stay_in_set(self):
        from perf.bench import _models_for_iteration

        models = [{"regs": {"rdi": i}, "reads": [], "writes": []} for i in (1, 2, 3, 4)]
        for dist in ("unpredictable", "unpredictable.exponential"):
            rng = random.Random(3)
            cfg = {"prediction": dist, "mem": {}, "regs": {}}
            for it in range(20):
                choice = _models_for_iteration(
                    models, False, rng, branch_cfg=cfg, it=it
                )[0]
                self.assertIn(choice["regs"]["rdi"], [1, 2, 3, 4])

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
        df.attrs["config"] = {"samples": 10, "cache": "hot"}
        df.attrs["info"] = {
            "cpu": {"hz": 3000000000.0, "arch": "x86_64"},
            "binary": {"name": "t"},
        }
        df.attrs["data"] = {"regs": {"rdi": 3}, "mem": {}}
        df.attrs["code"] = "nop"
        return df

    def test_to_json_envelope_structure(self):
        from perf.bench import to_json

        payload = json.loads(to_json(self._df()))
        self.assertEqual(payload["file"], "a.out")
        self.assertEqual(payload["mode"], "latency")
        self.assertTrue(payload["name"].startswith("foo-"))
        self.assertEqual(len(payload["id"]), 8)
        self.assertEqual(payload["name"], f"foo-{payload['id']}")
        self.assertEqual(payload["data"]["regs"]["rdi"], 3)
        self.assertEqual(payload["code"], "nop")
        self.assertEqual(payload["config"]["cache"], "hot")
        self.assertEqual(
            payload["info"], {"cpu": {"hz": 3000000000.0, "arch": "x86_64"}}
        )
        self.assertEqual(len(payload["output"]), 1)
        row = payload["output"][0]
        self.assertEqual(row["samples"], 10)
        self.assertEqual(row["iterations"], 1000)
        self.assertNotIn("time", row)
        self.assertNotIn("file", row)
        self.assertNotIn("name", row)
        self.assertNotIn("mode", row)

    def test_to_json_drops_config_and_data_columns(self):
        from perf.bench import to_json

        payload = json.loads(
            to_json(
                self._df(
                    **{
                        "config.cache": ["hot"],
                        "data.rdi": [3],
                        "extra": [1],
                    }
                )
            )
        )
        row = payload["output"][0]
        self.assertNotIn("config.cache", row)
        self.assertNotIn("data.rdi", row)
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


if __name__ == "__main__":
    unittest.main()
