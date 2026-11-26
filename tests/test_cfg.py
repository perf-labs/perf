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


import os
import random
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import perf.bench as bench_impl
import perf.cfg as cfg_mod
import perf.core as core
from perf.bench import benchmark
from perf.cfg import _DEFAULT_BENCH


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


def imm_of(value):
    from perf.arch import x86_64

    return x86_64.imm(value)


class TestFunctionOrder(unittest.TestCase):
    def test_resolve_orders(self):
        from perf.cfg import _container

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
        from perf.cfg import _container

        self.assertEqual(_container({}, "func")["align"], 16)
        self.assertEqual(_container(None, "func")["align"], 16)
        self.assertEqual(_container({"func": {"align": 64}}, "func")["align"], 64)
        with self.assertRaises(ValueError):
            _container({"func": {"align": 24}}, "func")
        with self.assertRaises(ValueError):
            _container({"func": {"align": "many"}}, "func")

    def test_every_alternative_is_covered(self):
        from perf.cfg import _container
        from perf.mca import _project_cache_id

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
                target={"asm": "nop"},
                name="t",
            )
        mock_bench.assert_not_called()

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.mca.explore", return_value=[])
    @patch("perf.mca.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.mca.Elf")
    @patch("angr.Project")
    def test_random_order_shuffles_layout(
        self,
        _mock_project,
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
            target=f"{path}:func",
            name="func",
        )

        obj = mock_elf.return_value
        obj.randomize_layout.assert_called_once()
        _, kwargs = obj.randomize_layout.call_args
        self.assertEqual(kwargs["seed"], 9)
        self.assertEqual(kwargs["align"], 32)
        obj.write_perf_map.assert_called_once_with()

    @patch("perf.bench._bench", return_value=pd.Series([50, 60]))
    @patch("perf.mca.explore", return_value=[])
    @patch("perf.mca.functions", return_value=({"func": (0x1000, 0x1100)}, {}))
    @patch("perf.bench.get_arch")
    @patch("perf.mca.Elf")
    @patch("angr.Project")
    def test_asis_order_keeps_layout(
        self,
        _mock_project,
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
            target=f"{path}:func",
            name="func",
        )

        obj = mock_elf.return_value
        obj.randomize_layout.assert_not_called()
        obj.write_perf_map.assert_called_once_with()


class TestBranchSetup(unittest.TestCase):
    def test_global_distribution(self):
        from perf.cfg import _branch_global_distribution

        self.assertEqual(_branch_global_distribution(None, True), "predictable")
        self.assertEqual(_branch_global_distribution(None, False), "unpredictable")

    def test_sample_index_predictable_cycles_in_order(self):
        from perf.cfg import _branch_sample_index

        idx = [_branch_sample_index(3, None, "predictable", i) for i in range(7)]
        self.assertEqual(idx, [0, 1, 2, 0, 1, 2, 0])

    def test_sample_index_unpredictable_varies(self):
        import random

        from perf.cfg import _branch_sample_index

        rng = random.Random(1)
        got = {_branch_sample_index(4, rng, "unpredictable", i) for i in range(60)}
        self.assertGreater(len(got), 1)

    def test_sample_values_cycles_when_predictable(self):
        from perf.cfg import _sample_values

        vals = [1, 2, 3]
        got = [_sample_values(vals, "predictable", i, None) for i in range(4)]
        self.assertEqual(got, [1, 2, 3, 1])

    def test_config_prediction_reaches_the_sampler(self):
        import random

        from perf.cfg import _branch_config, _per_iter_data

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

        from perf.cfg import _branch_config, _per_iter_data

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
        from perf.cfg import _branch_config

        for bad in ("sometimes", "always", ""):
            with self.assertRaises(ValueError):
                _branch_config({"branch": bad})


class TestCodeAlign(unittest.TestCase):
    @staticmethod
    def _offset(align):
        from perf.bench import _with_align
        from perf.cfg import get_arch

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
        from perf.cfg import _DEFAULT_BENCH, _container

        self.assertEqual(_DEFAULT_BENCH["code"], [{"align": 1}, {"align": 16}])
        self.assertEqual(_container({"code": {"align": 1}}, "code")["align"], 1)
        self.assertEqual(_container({"code": {"align": 16}}, "code")["align"], 16)

    def test_align_must_be_power_of_two(self):
        from perf.cfg import _container

        for bad in (0, 3, 24, -1):
            with self.assertRaises(ValueError):
                _container({"code": {"align": bad}}, "code")


class TestCodeTierDefaults(unittest.TestCase):
    CODE = [0x401000, 0x401100]
    DATA = [0x402000, 0x402100]

    def _steer(self, cfg, code=None, data=None):
        import random

        from perf.bench import get_arch
        from perf.cfg import (
            _apply_code_tier_defaults,
            _fill_evict_tables,
            _per_iter_data,
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

        from perf.cfg import _apply_code_tier_defaults, _per_iter_data, get_arch

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


class TestRegisterCacheLevels(unittest.TestCase):
    ADDR = 0x4200000000

    def _models(self):
        return [
            {
                "regs": {"rdi": self.ADDR},
                "reads": [(self.ADDR, 4, 1)],
                "writes": [],
            }
        ]

    def test_register_key_resolves_to_the_accessed_pointer(self):
        from perf.cfg import _resolve_reg_levels

        resolved = _resolve_reg_levels(
            {"rdi": {"L1d": 0}}, self._models(), {"rdi": self.ADDR}
        )
        self.assertEqual(resolved, {self.ADDR: {"L1d": 0}})

    def test_arg_alias_and_model_registers_resolve(self):
        from perf.cfg import _resolve_reg_levels

        resolved = _resolve_reg_levels({"arg0": {"TLBd": 0}}, self._models())
        self.assertEqual(resolved, {self.ADDR: {"TLBd": 0}})

    def test_explicit_address_levels_win_over_register_levels(self):
        from perf.cfg import _resolve_reg_levels

        resolved = _resolve_reg_levels(
            {"rdi": {"L1d": 0}, self.ADDR: {"L1d": 100}},
            self._models(),
            {"rdi": self.ADDR},
        )
        self.assertEqual(resolved, {self.ADDR: {"L1d": 100}})

    def test_per_iter_data_includes_register_level_addresses(self):
        import random

        from perf.cfg import _per_iter_data

        _buf, meta = _per_iter_data(
            self._models(),
            4,
            None,
            True,
            random.Random(1),
            mem_levels={"rdi": {"L1d": 0, "L2": 0, "L3": 0}},
            explicit={"regs": {"rdi": self.ADDR}},
        )
        self.assertEqual(meta["mem_levels"], {self.ADDR: {"L1d": 0, "L2": 0, "L3": 0}})
        self.assertIn(self.ADDR, meta["mem_addrs"])

    def test_build_loop_asm_steers_register_level_addresses(self):
        import random

        from perf.bench import _build_loop_asm
        from perf.cfg import get_arch

        asm, _meta, _buf = _build_loop_asm(
            4,
            "latency",
            "mov eax, [rdi]",
            "",
            "",
            self._models(),
            ["duration_time"],
            [None],
            None,
            True,
            random.Random(1),
            get_arch(),
            data={"regs": {}, "mem": {}},
            mem_levels={"rdi": {"L1d": 0, "L2": 0, "L3": 0}},
        )
        self.assertIn("clflushopt", asm)


class TestConfigShapes(unittest.TestCase):
    _ONE = {
        "branch": "predictable",
        "dcache": "hot",
        "icache": "hot",
        "dtlb": "hot",
        "itlb": "hot",
        "code": {"align": 16},
    }

    @patch("perf.bench._bench_one")
    def test_benchmark_defaults_to_seed_zero(self, mock_one):
        mock_one.return_value = pd.DataFrame([{"mode": "latency"}])
        benchmark(target={"asm": "nop"}, mode=["latency"], config=dict(self._ONE))
        benchmark(target={"asm": "nop"}, mode=["latency"], config=dict(self._ONE))
        self.assertEqual(
            [call.kwargs["config"]["seed"] for call in mock_one.call_args_list],
            [0, 0],
        )

    @patch("perf.bench._bench_one")
    def test_benchmark_keeps_an_explicit_seed(self, mock_one):
        mock_one.return_value = pd.DataFrame([{"mode": "latency"}])
        benchmark(
            target={"asm": "nop"},
            mode=["latency"],
            config={**self._ONE, "seed": 7},
        )
        self.assertEqual(mock_one.call_args.kwargs["config"]["seed"], 7)

    @patch("perf.bench._bench_one")
    def test_benchmark_none_seed_defaults_to_zero(self, mock_one):
        mock_one.return_value = pd.DataFrame([{"mode": "latency"}])
        benchmark(
            target={"asm": "nop"},
            mode=["latency"],
            config={**self._ONE, "seed": None},
        )
        self.assertEqual(mock_one.call_args.kwargs["config"]["seed"], 0)

    @patch("perf.bench._bench_one")
    def test_benchmark_random_seed_is_generated(self, mock_one):
        mock_one.return_value = pd.DataFrame([{"mode": "latency"}])
        with patch("perf.bench._generate_seed", return_value=23) as make_seed:
            benchmark(
                target={"asm": "nop"},
                mode=["latency"],
                config={**self._ONE, "seed": "random"},
            )
        make_seed.assert_called_once_with()
        self.assertEqual(mock_one.call_args.kwargs["config"]["seed"], 23)

    def test_seed_defaults_to_zero_and_is_scalar(self):
        from perf.cfg import _DEFAULT_BENCH, _config_seed_int

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
        from perf.cfg import _resolve_lib, _validate_lib

        self.assertFalse(_resolve_lib({"external": {"lib": False}}))
        self.assertTrue(_resolve_lib({"external": {"lib": True}}))
        self.assertFalse(_resolve_lib({}))
        self.assertFalse(_resolve_lib({"external": {}}))
        _validate_lib({"external": {"lib": "yes"}})
        with self.assertRaises(ValueError):
            _validate_lib({"external": {"nope": 1}})
        with self.assertRaises(ValueError):
            _validate_lib({"external": {"lib": 1.5}})


class TestHeapConfig(unittest.TestCase):
    def _data(self):
        from perf.cfg import _merge_data

        return _merge_data(
            None,
            {
                "regs": {
                    "rsi": [1, 10],
                    "rdx": 4,
                    "rcx": 0x4200000000,
                },
                "mem": {"0x4200000000:": b"\x90"},
            },
        )

    def test_heap_defaults_to_random(self):
        from perf.cfg import _DEFAULT_BENCH, _container, _heap_settings

        self.assertEqual(
            _DEFAULT_BENCH["heap"],
            [{"size": "auto", "mode": "random"}],
        )
        self.assertEqual(
            _container({"heap": {"size": "auto", "mode": "random"}}, "heap"),
            {"size": "auto", "mode": "random"},
        )
        self.assertEqual(_heap_settings({}), ("auto", "random"))
        self.assertEqual(
            _heap_settings({"heap": {"size": "auto", "mode": "zero"}}),
            ("auto", "zero"),
        )
        self.assertEqual(
            _heap_settings({"heap": {"size": 64, "mode": "fixed"}}),
            (64, "fixed"),
        )

    def test_heap_validation(self):
        from perf.cfg import _merge_config

        for value in (
            {"size": "auto", "mode": "random"},
            {"size": "auto", "mode": "zero"},
            {"size": 64, "mode": "random"},
            {"mode": "zero"},
        ):
            self.assertEqual(_merge_config({"heap": value})["heap"], [value])
        for value in (
            "random",
            "zero",
            "off",
            "none",
            True,
            False,
            8,
            "8192",
            "0x1000",
            {"bogus": 1},
            {"size": "bogus", "mode": "random"},
            {"size": 4, "mode": "random"},
            {"size": "auto", "mode": "bogus"},
            {"size": "auto", "mode": True},
            [],
        ):
            with self.assertRaises(ValueError):
                _merge_config({"heap": value})

    def test_heap_fills_missing_leading_args_with_random_data(self):
        from perf.cfg import _apply_heap_data, _heap_covers, data_param_columns
        from perf.core import _FALLBACK_DATA_BASE

        data = self._data()
        out = _apply_heap_data(
            data, {"seed": 7, "heap": {"size": "auto", "mode": "random"}}
        )
        base = _FALLBACK_DATA_BASE
        size = 10 * 4
        self.assertEqual(out["regs"]["rdi"], base)
        self.assertNotIn("r8", out["regs"])
        self.assertNotIn("r9", out["regs"])
        self.assertEqual(out["heap"], {f"0x{base:x}": size})
        added = {
            int(addr, 0): value
            for addr, value in out["mem"].items()
            if _heap_covers(out, addr)
        }
        self.assertEqual(len(added), size // 8)
        self.assertGreater(len(set(added.values())), 1)
        self.assertTrue(all(base <= addr < base + size for addr in added))
        cols = data_param_columns(out)
        self.assertIn("data.rdi", cols)
        self.assertIn("data.0x4200000000", cols)
        self.assertFalse(any(str(col).startswith("data.0x5200") for col in cols))

    def test_heap_helpers_cover_edge_forms(self):
        from perf.cfg import _apply_heap_data, _heap_arg_regs, _heap_covers, _heap_rng
        from perf.core import _FALLBACK_DATA_BASE

        data = self._data()
        self.assertIn("rdi", _heap_arg_regs(data))
        self.assertFalse(_heap_covers(data, "not-an-address"))
        self.assertIsInstance(_heap_rng({"seed": None}).random(), float)
        a = _heap_rng({"seed": 7}, 0).getrandbits(64)
        b = _heap_rng({"seed": 7}, 1).getrandbits(64)
        self.assertNotEqual(a, b)
        out = _apply_heap_data(
            data, {"seed": 7, "heap": {"size": "auto", "mode": "random"}}, debug=True
        )
        self.assertEqual(out["regs"]["rdi"], _FALLBACK_DATA_BASE)

    def test_heap_settings_rejects_aliases_and_bad_values(self):
        from perf.cfg import _heap_settings

        for value in ("random", "zero", "off", "none", True, False, 8, []):
            with self.assertRaises(ValueError):
                _heap_settings({"heap": value})
        with self.assertRaises(ValueError):
            _heap_settings({"heap": {"size": "bogus", "mode": "random"}})
        with self.assertRaises(ValueError):
            _heap_settings({"heap": {"size": "auto", "mode": "bogus"}})
        with self.assertRaises(ValueError):
            _heap_settings({"heap": {"size": "auto", "mode": True}})

    def test_heap_size_hint_uses_explicit_bounds_only(self):
        from perf.cfg import _heap_size_hint

        arch = SimpleNamespace(_ARG_REGS=("rdi", "rsi", "rdx", "rcx"))
        self.assertIsNone(_heap_size_hint(None, arch))
        self.assertIsNone(_heap_size_hint({"regs": {}}, arch))
        self.assertIsNone(_heap_size_hint({"regs": {"rdx": 4}}, arch))
        self.assertIsNone(_heap_size_hint({"regs": {"rsi": "bad"}}, arch))
        self.assertIsNone(_heap_size_hint({"regs": {"rsi": 2**40}}, arch))
        self.assertEqual(_heap_size_hint({"regs": {"rsi": 2, "rdx": 4}}), 8)
        self.assertEqual(_heap_size_hint({"regs": {"rsi": 3}}, arch), 24)
        self.assertEqual(
            _heap_size_hint(
                {"regs": {"rsi": [0, "bad", 10], "rdx": [4, 8]}},
                arch,
            ),
            80,
        )
        self.assertEqual(
            _heap_size_hint(
                {"regs": {"rsi": [10], "esi": 3, "rdx": 4}},
                arch,
            ),
            40,
        )

    def test_heap_arg_regs_only_covers_missing_leading_arguments(self):
        from perf.cfg import _heap_arg_regs

        arch = SimpleNamespace(_ARG_REGS=("rdi", "rsi", "rdx", "rcx", "r8"))
        self.assertEqual(_heap_arg_regs({"regs": {}}, arch), [])
        self.assertEqual(
            _heap_arg_regs({"regs": {"rsi": 10, "rdx": 4}}, arch),
            ["rdi"],
        )
        self.assertEqual(
            _heap_arg_regs({"regs": {"rdx": 4}}, arch),
            ["rdi", "rsi"],
        )
        self.assertEqual(
            _heap_arg_regs({"regs": {"rdi": 1, "rsi": 10}}, arch),
            [],
        )

    def test_heap_covers_handles_malformed_markers(self):
        from perf.cfg import _heap_covers

        self.assertFalse(_heap_covers({"heap": {"bad": 64}}, 1))
        self.assertFalse(_heap_covers({"heap": {"0x1000": "bad"}}, 0x1000))
        self.assertFalse(_heap_covers({"heap": {"0x1000": 8}}, 0x1008))
        self.assertFalse(_heap_covers({"heap": {"0x1000": 8}}, object()))
        self.assertTrue(_heap_covers({"heap": {"0x1000:": 8}}, 0x1004))

    def test_heap_allocates_one_buffer_per_leading_argument(self):
        from perf.cfg import _HEAP_RANDOM_LIMIT, _apply_heap_data, _merge_data
        from perf.core import _FALLBACK_DATA_BASE, _FALLBACK_DATA_STRIDE

        data = _merge_data(
            None,
            {
                "regs": {"rdx": 4},
                "mem": {"0x4200000000:": b"\x90"},
            },
        )
        out = _apply_heap_data(
            data, {"seed": 7, "heap": {"size": "auto", "mode": "random"}}
        )
        self.assertEqual(
            [out["regs"]["rdi"], out["regs"]["rsi"]],
            [_FALLBACK_DATA_BASE, _FALLBACK_DATA_BASE + _FALLBACK_DATA_STRIDE],
        )
        self.assertEqual(len(out["heap"]), 2)
        self.assertEqual(sorted(out["heap"].values()), [_HEAP_RANDOM_LIMIT] * 2)

    def test_heap_apply_is_idempotent(self):
        from perf.cfg import _apply_heap_data

        out = _apply_heap_data(
            self._data(), {"seed": 7, "heap": {"size": "auto", "mode": "random"}}
        )
        self.assertIs(
            _apply_heap_data(
                out, {"seed": 7, "heap": {"size": "auto", "mode": "random"}}
            ),
            out,
        )

    def test_heap_marker_survives_entry_data(self):
        from perf.cfg import _entry_data

        data = self._data()
        data["heap"] = {"0x5200000000": 64}
        out = _entry_data("main", "a.out", data, argv=["a.out"], env=[])
        self.assertEqual(out["heap"], {"0x5200000000": 64})
        self.assertIn("rdi", out["regs"])

    def test_heap_marker_survives_fallback_merge(self):
        from perf.cfg import _merge_fallback_data

        base = {
            "regs": {"rdi": 1},
            "mem": {"0x5200000000": 1},
            "heap": {"0x5200000000": 64},
            "code": [0x400000],
        }
        grown = {
            "regs": {"rsi": 2},
            "mem": {"0x5200000000": 9, "0x5300000000": 3},
            "heap": {"0x5300000000": 32},
            "code": [0x401000],
        }
        out = _merge_fallback_data(base, [grown])
        self.assertEqual(out["regs"], {"rdi": 1, "rsi": 2})
        self.assertEqual(out["mem"], {"0x5200000000": 1, "0x5300000000": 3})
        self.assertEqual(
            out["heap"],
            {"0x5200000000": 64, "0x5300000000": 32},
        )
        self.assertEqual(out["code"], [0x400000, 0x401000])

    def test_heap_normalizes_and_ignores_bad_markers(self):
        from perf.cfg import _as_normalized_data, _heap_covers

        out = _as_normalized_data(
            {
                "regs": {},
                "mem": {},
                "heap": {
                    "0x5200000000:": "64",
                    "nope": 8,
                    "0x5300000000": None,
                    0x5400000000: 32,
                },
            }
        )
        self.assertEqual(
            out["heap"],
            {"0x5200000000": 64, "0x5400000000": 32},
        )
        self.assertTrue(_heap_covers(out, "0x5200000020:"))

    def test_heap_debug_logs_filled_and_skipped_buffers(self):
        from perf.cfg import _apply_heap_data, _merge_data

        with patch("perf.cfg._debug_log") as log:
            _apply_heap_data(
                self._data(),
                {"seed": 7, "heap": {"size": "auto", "mode": "random"}},
                debug=True,
            )
        messages = [str(call.args[0]) for call in log.call_args_list]
        self.assertTrue(any("heap buffer for rdi" in message for message in messages))

        data = _merge_data(
            None,
            {
                "regs": {"rsi": 100000, "rdx": 4},
                "mem": {"0x4200000000:": b"\x90"},
            },
        )
        with patch("perf.cfg._debug_log") as log:
            self.assertIs(
                _apply_heap_data(
                    data,
                    {"seed": 7, "heap": {"size": "auto", "mode": "random"}},
                    debug=True,
                ),
                data,
            )
        self.assertIn("above the", str(log.call_args_list[0].args[0]))

    def test_heap_does_not_touch_unknown_data(self):
        from perf.cfg import _apply_heap_data

        self.assertIsNone(
            _apply_heap_data(
                None, {"seed": 7, "heap": {"size": "auto", "mode": "random"}}
            )
        )

    def test_heap_seed_is_stable(self):
        from perf.cfg import _apply_heap_data

        data = self._data()
        a = _apply_heap_data(
            data, {"seed": 7, "heap": {"size": "auto", "mode": "random"}}
        )
        b = _apply_heap_data(
            data, {"seed": 7, "heap": {"size": "auto", "mode": "random"}}
        )
        c = _apply_heap_data(
            data, {"seed": 8, "heap": {"size": "auto", "mode": "random"}}
        )
        self.assertEqual(a["mem"], b["mem"])
        self.assertNotEqual(a["mem"], c["mem"])

    def test_heap_marker_survives_data_normalization(self):
        from perf.cfg import _as_normalized_data, _merge_data

        heap = {"0x5200000000": 64}
        out = _as_normalized_data({"regs": {}, "mem": {}, "heap": heap})
        self.assertEqual(out["heap"], heap)
        merged = _merge_data(out, {"heap": {"0x5300000000": 32}})
        self.assertEqual(merged["heap"], {"0x5200000000": 64, "0x5300000000": 32})

    def test_heap_zero_allocates_a_zeroed_buffer(self):
        from perf.cfg import _apply_heap_data, _heap_covers

        data = self._data()
        out = _apply_heap_data(
            data, {"seed": 7, "heap": {"size": "auto", "mode": "zero"}}
        )
        self.assertIn("rdi", out["regs"])
        self.assertIn("heap", out)
        added = {
            int(addr, 0): value
            for addr, value in out["mem"].items()
            if _heap_covers(out, addr)
        }
        self.assertTrue(added)
        self.assertEqual(set(added.values()), {0})
        with self.assertRaises(ValueError):
            _apply_heap_data(data, {"seed": 7, "heap": False})

    def test_heap_grows_to_fit_leading_bounds(self):
        from perf.cfg import _HEAP_RANDOM_LIMIT_MAX, _apply_heap_data, _merge_data

        data = _merge_data(
            None,
            {
                "regs": {"rsi": 2000, "rdx": 4, "rcx": 0x4200000000},
                "mem": {"0x4200000000:": b"\x90"},
            },
        )
        out = _apply_heap_data(
            data, {"seed": 7, "heap": {"size": "auto", "mode": "random"}}
        )
        self.assertEqual(next(iter(out["heap"].values())), 8000)
        self.assertLessEqual(8000, _HEAP_RANDOM_LIMIT_MAX)

    def test_heap_skips_bounds_beyond_the_limit(self):
        from perf.cfg import _apply_heap_data, _merge_data

        data = _merge_data(
            None,
            {
                "regs": {"rsi": 100000, "rdx": 4, "rcx": 0x4200000000},
                "mem": {"0x4200000000:": b"\x90"},
            },
        )
        self.assertIs(
            _apply_heap_data(
                data, {"seed": 7, "heap": {"size": "auto", "mode": "random"}}
            ),
            data,
        )

    def test_heap_does_not_overwrite_explicit_memory(self):
        from perf.cfg import _FALLBACK_DATA_BASE, _apply_heap_data

        data = self._data()
        key = f"0x{_FALLBACK_DATA_BASE:x}"
        data["mem"][key] = 1234
        out = _apply_heap_data(
            data, {"seed": 7, "heap": {"size": "auto", "mode": "random"}}
        )
        self.assertEqual(out["mem"][key], 1234)

    def test_heap_does_not_fill_explicit_pointers(self):
        from perf.cfg import _apply_heap_data, _merge_data

        data = _merge_data(
            None,
            {
                "regs": {
                    "rdi": 0x4300000000,
                    "rsi": 10,
                    "rdx": 4,
                    "rcx": 0x4200000000,
                },
                "mem": {"0x4200000000:": b"\x90"},
            },
        )
        self.assertIs(
            _apply_heap_data(
                data, {"seed": 7, "heap": {"size": "auto", "mode": "random"}}
            ),
            data,
        )

    @patch("perf.bench._bench", return_value=pd.Series([10]))
    def test_benchmark_applies_heap_before_measurement(self, mock_bench):
        benchmark(
            target={"asm": "nop"},
            name="heap",
            mode=["latency"],
            config={
                "heap": {"size": "auto", "mode": "random"},
                "seed": 7,
                "iterations": 1,
                "samples": 1,
                "branch": "predictable",
                "code": {"align": 16},
            },
            data={"regs": {"rsi": 10}},
        )
        self.assertTrue(mock_bench.call_args_list)
        data = mock_bench.call_args_list[-1].kwargs["data"]
        self.assertIn("rdi", data["regs"])
        self.assertIn("heap", data)


class TestStackConfig(unittest.TestCase):
    def test_auto_stack_settings_resolve_to_defaults(self):
        from perf.cfg import _DEFAULT_STACK_SIZE, _container

        self.assertEqual(
            _container(
                {"stack": {"size": "auto", "align": "auto", "mode": "zero"}},
                "stack",
            ),
            {"size": _DEFAULT_STACK_SIZE, "align": 16, "mode": "zero"},
        )

    def test_explicit_stack_settings(self):
        from perf.cfg import _container

        self.assertEqual(
            _container(
                {"stack": {"size": 0x4000, "align": 64, "mode": "zero"}},
                "stack",
            ),
            {"size": 0x4000, "align": 64, "mode": "zero"},
        )
        self.assertEqual(
            _container(
                {"stack": {"align": 32, "mode": "random"}},
                "stack",
            )["align"],
            32,
        )

    def test_invalid_stack_settings(self):
        from perf.cfg import _merge_config

        for value in (
            {"size": 0},
            {"size": 64},
            {"size": "bad"},
            {"align": 3},
            {"align": "bad"},
            {"mode": "off"},
            {"mode": "none"},
            {"mode": True},
            {"bogus": 1},
        ):
            with self.assertRaises(ValueError):
                _merge_config({"stack": value})

    def test_prepare_stack_info_refreshes_random_and_zero(self):
        import random

        from perf.bench import _prepare_stack_info
        from perf.cfg import get_arch

        arch = get_arch()
        info = {
            "base": 0x700000,
            "top": 0x800000,
            "size": 0x10000,
            "mode": "random",
            "frame": {"save": 0x700000, "sp": 0x800000},
        }
        out = _prepare_stack_info(info, 10, "latency", random.Random(1), arch)
        self.assertIsNotNone(out)
        self.assertGreater(out["src"], 0)
        self.assertEqual(out["_buffer"].shape, (10, out["size"]))
        fixed = _prepare_stack_info(
            {**info, "mode": "fixed"}, 10, "latency", random.Random(1), arch
        )
        self.assertNotIn("src", fixed)
        zero = _prepare_stack_info(
            {**info, "mode": "zero"}, 10, "latency", random.Random(1), arch
        )
        self.assertNotIn("src", zero)
        self.assertIsNone(
            _prepare_stack_info(info, 10, "throughput", random.Random(1), arch)
        )

    def test_heap_settings_accepts_singleton_list_and_rejects_sweep(self):
        from perf.cfg import _heap_settings

        self.assertEqual(
            _heap_settings({"heap": [{"size": 64, "mode": "random"}]}),
            (64, "random"),
        )
        for bad in (
            {"heap": [{"size": 64}, {"size": 128}]},
            {"heap": [None, None]},
        ):
            with self.assertRaises(ValueError):
                _heap_settings(bad)

    def test_heap_fixed_size_limit_falls_back_to_zeroed_buffer(self):
        from unittest.mock import patch

        from perf.cfg import _apply_heap_data, _merge_data

        data = _merge_data(
            None,
            {
                "regs": {"rsi": 1000, "rdx": 4},
                "mem": {"0x4200000000:": b"\x90"},
            },
        )
        out = _apply_heap_data(
            data, {"seed": 7, "heap": {"size": 64, "mode": "random"}}
        )
        self.assertIs(out, data)
        self.assertNotIn("heap", out)
        with patch("perf.cfg._debug_log") as log:
            out = _apply_heap_data(
                data,
                {"seed": 7, "heap": {"size": 64, "mode": "random"}},
                debug=True,
            )
        self.assertIs(out, data)
        self.assertIn("above the", str(log.call_args_list[0].args[0]))


class TestDefaultBench(unittest.TestCase):
    def test_all_keys_have_defaults(self):
        from perf.cfg import _DEFAULT_BENCH

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
            _DEFAULT_BENCH["stack"],
            [{"size": "auto", "align": "auto", "mode": "random"}],
        )
        self.assertEqual(
            _DEFAULT_BENCH["heap"],
            [{"size": "auto", "mode": "random"}],
        )
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
                "loop": {
                    "count": 16,
                    "probes": 3,
                    "runs": 10,
                    "target": 0.01,
                },
                "unroll": {"count": 5, "probes": 3, "runs": 10, "target": 0.01},
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
        from perf.cfg import _BRANCH_CHOICES, _concrete_configs

        _spec, defaults = _concrete_configs({})
        self.assertEqual(_DEFAULT_BENCH["branch"], list(_BRANCH_CHOICES))
        self.assertEqual({c["branch"] for c in defaults}, set(_BRANCH_CHOICES))
        _spec, explicit = _concrete_configs({"branch": list(_BRANCH_CHOICES)})
        self.assertEqual(defaults, explicit)

    def test_none_backend_and_unroll_fall_back(self):
        from perf.bench import _resolve_backend, _resolve_unroll_n
        from perf.cfg import _DEFAULT_BENCH as _DB2

        self.assertEqual(_resolve_backend(None, {"backend": None}, "loop"), "loop")
        self.assertEqual(_resolve_backend(None, {"backend": None}, "unroll"), "unroll")
        self.assertEqual(
            _resolve_unroll_n(None, {"backend": {"unroll": {"count": None}}}),
            _DB2["backend"]["unroll"]["count"],
        )


class TestConfigData(unittest.TestCase):
    def test_normalize_regs_and_mem(self):
        from perf.cfg import _config_data

        data = _config_data({"data": {"rdi": 21, "0x1234": 5}})
        self.assertEqual(data["regs"]["rdi"], 21)
        self.assertEqual(data["mem"]["0x1234"], 5)

    def test_flat_rdi_maps_to_regs(self):
        from perf.cfg import _config_data

        data = _config_data({"data": {"rdi": 21}})
        self.assertEqual(data["regs"]["rdi"], 21)

    def test_memory_array_expands(self):
        from perf.cfg import _config_data

        data = _config_data({"data": {"0x1234": [1, 2, 3]}})
        self.assertEqual(data["mem"]["0x1234"], [1, 2, 3])

        data = _config_data({"data": {"0x1234:": [1, 2, 3]}})
        self.assertEqual(data["mem"]["0x1234"], 1)
        self.assertEqual(data["mem"]["0x1235"], 2)
        self.assertEqual(data["mem"]["0x1236"], 3)

    def test_memory_namespace_array_expands(self):
        from perf.cfg import _config_data

        data = _config_data({"data": {"0x1234:": [1, 2, 3, 3, 4, 4, 5]}})
        self.assertEqual(len(data["mem"]), 7)
        self.assertEqual(data["mem"]["0x1234"], 1)
        data = _config_data({"data": {"0x1234": [1, 2, 3]}})
        self.assertEqual(data["mem"]["0x1234"], [1, 2, 3])

    def test_memory_byte_list_expands(self):
        from perf.cfg import _config_data

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
        from perf.cfg import _config_data

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
            target={"asm": "nop"},
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
                target={"asm": "nop"},
                name="t",
            )
        mock_bench.assert_not_called()


class TestBranchConfig(unittest.TestCase):
    def test_string_form(self):
        from perf.cfg import _branch_config

        cfg = _branch_config({"branch": "predictable"})
        self.assertEqual(cfg, {"prediction": "predictable", "mem": {}, "regs": {}})

    def test_dict_form(self):
        from perf.cfg import _branch_config

        cfg = _branch_config(
            {"branch": {"prediction": "unpredictable", "mem": {"0x401234": True}}}
        )
        self.assertEqual(cfg["prediction"], "unpredictable")
        self.assertEqual(cfg["mem"], {0x401234: "predictable"})

    def test_hex_and_decimal_addresses(self):
        from perf.cfg import _branch_config

        cfg = _branch_config(
            {"branch": {"mem": {"0x401234": "predictable", "4200004": False}}}
        )
        self.assertIn(0x401234, cfg["mem"])
        self.assertIn(4200004, cfg["mem"])
        self.assertEqual(cfg["mem"][4200004], "unpredictable")

    def test_unset_returns_none(self):
        from perf.cfg import _branch_config

        self.assertIsNone(_branch_config({}))
        self.assertIsNone(_branch_config(None))

    def test_default_with_address_overrides(self):
        from perf.cfg import _branch_config

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
        from perf.cfg import _branch_config

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
        from perf.cfg import _branch_config

        with self.assertRaises(ValueError):
            _branch_config({"branch": {"default": "predictable"}})

    def test_invalid_rejected(self):
        from perf.cfg import _branch_config

        with self.assertRaises(ValueError):
            _branch_config({"branch": "sometimes"})
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"prediction": "sometimes"}})
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"mem": {"0x1": "maybe"}}})
        with self.assertRaises(ValueError):
            _branch_config({"branch": [1, 2]})


class TestPerIterDataBranchPinning(unittest.TestCase):
    def test_pins_regs_for_predictable_address(self):
        import random

        from perf.cfg import _per_iter_data

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

        from perf.cfg import _per_iter_data

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

        from perf.cfg import _per_iter_data

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
        from perf.cfg import _branch_config, _merge_config

        cfg = _merge_config(
            {
                "dcache": {
                    "L1": {"hit_rate": 100},
                    "0x41000000000": "hot",
                },
                "branch": {"mem": {"0x401000": "predictable"}},
            }
        )
        mc = cfg["dcache"]
        self.assertEqual(mc["L1"], {"hit_rate": 100})
        self.assertEqual(mc["0x41000000000"], "hot")
        cfgb = _branch_config(cfg)
        self.assertEqual(cfgb["prediction"], "unpredictable")
        self.assertEqual(cfgb["mem"], {0x401000: "predictable"})

    def test_scalar_override_wins(self):
        from perf.cfg import _merge_config

        cfg = _merge_config({"branch": "predictable", "samples": 5})
        self.assertEqual(cfg["branch"], "predictable")
        self.assertEqual(cfg["samples"], 5)
        self.assertEqual(cfg["dcache"], ["hot", "warm", "cool", "cold"])
        self.assertEqual(cfg["icache"], ["hot", "cold"])

    def test_nested_cache_and_tlb_lists_expand(self):
        from perf.cfg import _expand_config_spec, _merge_config

        cfg = _merge_config(
            {
                "dcache": {"L1": [{"hit_rate": 100}, {"hit_rate": 50}]},
                "dtlb": {"0x41000000000": ["hot", "cold"]},
                "branch": "predictable",
                "icache": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            }
        )
        combos = _expand_config_spec(cfg)
        self.assertEqual(len(combos), 4)
        self.assertEqual(
            {combo["dcache"]["L1"]["hit_rate"] for combo in combos}, {100, 50}
        )
        self.assertEqual(
            {combo["dtlb"]["0x41000000000"] for combo in combos}, {"hot", "cold"}
        )

    def test_iterations_min_list_expands(self):
        from perf.cfg import _expand_config_spec, _merge_config

        cfg = _merge_config(
            {
                "iterations": {"min": [10, 20]},
                "branch": "predictable",
                "dcache": "hot",
                "dtlb": "hot",
                "icache": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            }
        )
        combos = _expand_config_spec(cfg)
        self.assertEqual({combo["iterations"]["min"] for combo in combos}, {10, 20})

    def test_iterations_max_tuple_expands(self):
        from perf.cfg import _expand_config_spec, _merge_config

        cfg = _merge_config(
            {
                "iterations": {"max": (2000, 3000)},
                "branch": "predictable",
                "dcache": "hot",
                "dtlb": "hot",
                "icache": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            }
        )
        combos = _expand_config_spec(cfg)
        self.assertEqual({combo["iterations"]["max"] for combo in combos}, {2000, 3000})

    def test_cache_shortcut_forms(self):
        from perf.cfg import _expand_config_spec, _merge_config

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
        from perf.cfg import _merge_config, validate_spec

        with self.assertRaises(ValueError):
            _merge_config(["bad"])
        with self.assertRaises(ValueError):
            validate_spec(["bad"])

    def test_unknown_top_key_rejected(self):
        from perf.cfg import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"bogus": 1})

    def test_bench_rejects_non_dict_config(self):
        with self.assertRaises(ValueError):
            benchmark(target={"asm": "mov eax, 42"}, mode=["latency"], config=["bad"])

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            benchmark(target=("/nonexistent/a.out", "func"), mode=["latency"], event=[])

    def test_binary_without_target_raises(self):
        with self.assertRaises(ValueError):
            benchmark(target=["/bin/true"], mode=["latency"], event=[])


class TestContainerConfigFormat(unittest.TestCase):
    def test_defaults_keep_the_list_on_the_first_level(self):
        from perf.cfg import _DEFAULT_BENCH

        self.assertEqual(
            _DEFAULT_BENCH["thread"],
            [[{"numa": None, "affinity": None, "priority": "normal"}]],
        )
        self.assertEqual(_DEFAULT_BENCH["code"], [{"align": 1}, {"align": 16}])
        self.assertEqual(_DEFAULT_BENCH["func"], [{"align": 16, "order": "as-is"}])
        self.assertEqual(
            _DEFAULT_BENCH["stack"],
            [{"size": "auto", "align": "auto", "mode": "random"}],
        )
        self.assertEqual(
            _DEFAULT_BENCH["heap"],
            [{"size": "auto", "mode": "random"}],
        )

    def test_shorthands_normalize_to_the_list_form(self):
        from perf.cfg import normalize_container

        self.assertEqual(normalize_container("code", 32), [{"align": 32}])
        self.assertEqual(normalize_container("code", {"align": 32}), [{"align": 32}])
        self.assertEqual(normalize_container("func", "random"), [{"order": "random"}])
        self.assertEqual(
            normalize_container("stack", {"size": 4096, "align": 16}),
            [{"size": 4096, "align": 16}],
        )
        self.assertEqual(
            normalize_container("heap", {"size": "auto", "mode": "zero"}),
            [{"size": "auto", "mode": "zero"}],
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
        from perf.cfg import _concrete_configs

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
        from perf.cfg import _concrete_configs

        _spec, combos = _concrete_configs(None)
        self.assertEqual(
            combos[0]["thread"],
            [{"numa": None, "affinity": None, "priority": "normal"}],
        )

    def test_list_leaves_of_an_entry_still_sweep(self):
        from perf.cfg import _concrete_configs

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
        from perf.cfg import _thread_group, validate_spec

        cfg = validate_spec({"thread": [[{"affinity": 1}, {"affinity": 2}]]})
        self.assertEqual(len(_thread_group(cfg)), 2)

    def test_unknown_container_key_rejected(self):
        from perf.cfg import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"code": [{"alignment": 16}]})
        with self.assertRaises(ValueError):
            validate_spec({"stack": [{"bogus": 1}]})
        with self.assertRaises(ValueError):
            validate_spec({"code": []})

    def test_single_element_list_leaf_kept_in_the_spec(self):
        from perf.cfg import _merge_config

        spec = _merge_config({"stack": [{"size": 0x1000, "align": 16}]})
        self.assertEqual(spec["stack"], [{"size": 0x1000, "align": 16}])

    def test_config_param_columns_read_the_list_form(self):
        from perf.cfg import config_param_columns

        cols = config_param_columns(
            {
                "code": [{"align": 1}, {"align": 16}],
                "thread": [[{"affinity": 0, "priority": "normal"}]],
            }
        )
        self.assertEqual(cols["config.code.align"], [1, 16])
        self.assertEqual(cols["config.thread.priority"], ["normal"])


class TestValidateNumerics(unittest.TestCase):
    def test_bad_samples(self):
        from perf.cfg import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"samples": 0})
        with self.assertRaises(ValueError):
            validate_spec({"samples": "many"})

    def test_bad_runs(self):
        from perf.cfg import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"backend": {"loop": {"runs": 0}}})
        with self.assertRaises(ValueError):
            validate_spec({"backend": {"loop": {"runs": "many"}}})

    def test_bad_backend_target(self):
        from perf.cfg import validate_spec

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
        from perf.cfg import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"backend": "turbo"})

    def test_min_max_order(self):
        from perf.cfg import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"iterations": {"min": 1000, "max": 10}})

    def test_bad_iterations(self):
        from perf.cfg import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"iterations": 0})
        with self.assertRaises(ValueError):
            validate_spec({"iterations": {"min": 0}})
        with self.assertRaises(ValueError):
            validate_spec({"iterations": {"max": -5}})
        with self.assertRaises(ValueError):
            validate_spec({"iterations": {"min": "many"}})
        with self.assertRaises(ValueError):
            validate_spec({"iterations": {"nope": 5}})
        with self.assertRaises(ValueError):
            validate_spec({"iterations": {"count": 10}})

    def test_branch_string_validated(self):
        from perf.cfg import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"branch": "sometimes"})

    def test_empty_branch_alternatives_rejected(self):
        from perf.cfg import validate_spec

        with self.assertRaises(ValueError):
            validate_spec({"branch": []})

    def test_valid_numerics(self):
        from perf.cfg import validate_spec

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


class TestBranchString(unittest.TestCase):
    def test_valid_strings(self):
        from perf.cfg import _branch_config

        self.assertEqual(
            _branch_config({"branch": "predictable"})["prediction"], "predictable"
        )
        self.assertEqual(
            _branch_config({"branch": "PREDICTABLE"})["prediction"], "predictable"
        )

    def test_invalid_string_raises(self):
        from perf.cfg import _branch_config

        with self.assertRaises(ValueError):
            _branch_config({"branch": "sometimes"})


class TestBranchValueChoices(unittest.TestCase):
    def test_all_choices_accepted_case_insensitive(self):
        from perf.cfg import _BRANCH_CHOICES, _branch_value

        for choice in _BRANCH_CHOICES:
            self.assertEqual(_branch_value(choice), choice)
            self.assertEqual(_branch_value(choice.upper()), choice)
            self.assertEqual(_branch_value(f"  {choice}  "), choice)

    def test_bool_mapping(self):
        from perf.cfg import _branch_value

        self.assertEqual(_branch_value(True), "predictable")
        self.assertEqual(_branch_value(False), "unpredictable")

    def test_invalid_returns_none(self):
        from perf.cfg import _branch_value

        self.assertIsNone(_branch_value("sometimes"))
        self.assertIsNone(_branch_value("maybe"))


class TestBranchConfigChoices(unittest.TestCase):
    def test_string_form_each_choice(self):
        from perf.cfg import _branch_config

        for choice in ("predictable", "unpredictable"):
            cfg = _branch_config({"branch": choice})
            self.assertEqual(cfg["prediction"], choice)

    def test_choice_is_case_and_space_insensitive(self):
        from perf.cfg import _branch_config, _branch_value

        self.assertEqual(_branch_value("  UNPREDICTABLE "), "unpredictable")
        self.assertEqual(_branch_value("Predictable"), "predictable")
        cfg = _branch_config({"branch": "  UNPREDICTABLE "})
        self.assertEqual(cfg, {"prediction": "unpredictable", "mem": {}, "regs": {}})

    def test_only_two_choices_exist(self):
        from perf.cfg import _BRANCH_CHOICES

        self.assertEqual(_BRANCH_CHOICES, ("predictable", "unpredictable"))

    def test_gaussian_rejected(self):
        from perf.cfg import _branch_config

        with self.assertRaises(ValueError):
            _branch_config({"branch": "gaussian"})

    def test_removed_choices_rejected(self):
        from perf.cfg import _branch_config

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
        from perf.cfg import _branch_config

        cfg = _branch_config({"branch": {"prediction": "unpredictable"}})
        self.assertEqual(cfg["prediction"], "unpredictable")
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"default": "unpredictable"}})

    def test_bool_prediction_in_dict(self):
        from perf.cfg import _branch_config

        self.assertEqual(
            _branch_config({"branch": {"prediction": True}})["prediction"],
            "predictable",
        )
        self.assertEqual(
            _branch_config({"branch": {"prediction": False}})["prediction"],
            "unpredictable",
        )

    def test_per_address_and_reg_choices(self):
        from perf.cfg import _branch_config

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
        from perf.cfg import _branch_config

        with self.assertRaises(ValueError):
            _branch_config({"branch": "sometimes"})
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"prediction": "sometimes"}})
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"mem": {"0x1": "sometimes"}}})
        with self.assertRaises(ValueError):
            _branch_config({"branch": {"regs": {"rdi": "sometimes"}}})

    def test_validate_accepts_new_choices(self):
        from perf.cfg import validate_spec

        for choice in ("predictable", "unpredictable"):
            validate_spec({"branch": choice})
        for choice in ("unpredictable.exponential", "unpredictable.uniform"):
            with self.assertRaises(ValueError):
                validate_spec({"branch": choice})


class TestBranchDistributions(unittest.TestCase):
    def test_global_distribution_precedence(self):
        from perf.cfg import _branch_global_distribution

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
        from perf.cfg import _branch_mem_distribution, _branch_reg_distribution

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
        from perf.cfg import _branch_mem_distribution, _branch_reg_distribution

        cfg = {
            "prediction": "unpredictable",
            "mem": {0x1000: "  PREDICTABLE "},
            "regs": {"rdi": "Predictable"},
        }
        self.assertEqual(_branch_reg_distribution("rdi", cfg, False), "predictable")
        self.assertEqual(_branch_mem_distribution(0x1000, cfg, False), "predictable")

    def test_sample_index_bounds(self):
        from perf.cfg import _branch_sample_index

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
        from perf.cfg import _branch_sample_index

        rng = random.Random(0)
        idxs = [
            _branch_sample_index(4, rng, "unpredictable.exponential", i)
            for i in range(400)
        ]
        self.assertGreater(idxs.count(0), idxs.count(3))

    def test_sample_index_predictable_cycles_incrementally(self):
        from perf.cfg import _branch_sample_index

        rng = random.Random(0)
        self.assertEqual(
            [_branch_sample_index(3, rng, "predictable", it=i) for i in range(7)],
            [0, 1, 2, 0, 1, 2, 0],
        )

    def test_sample_index_deterministic(self):
        from perf.cfg import _branch_sample_index

        for dist in (
            "unpredictable",
            "unpredictable.uniform",
            "unpredictable.exponential",
        ):
            a = [_branch_sample_index(4, random.Random(42), dist, i) for i in range(20)]
            b = [_branch_sample_index(4, random.Random(42), dist, i) for i in range(20)]
            self.assertEqual(a, b)

    def test_uniform_alias_matches_unpredictable_sampling(self):
        from perf.cfg import _branch_sample_index

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
        from perf.cfg import _sample_values

        vals = [10, 20, 30]
        self.assertEqual(
            [
                _sample_values(vals, "predictable", i, random.Random(0))
                for i in range(5)
            ],
            [10, 20, 30, 10, 20],
        )

    def test_sample_values_stays_in_set(self):
        from perf.cfg import _sample_values

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
        from perf.cfg import _model_picks

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
        from perf.cfg import _model_picks

        models = [{"regs": {"rdi": i}, "reads": [], "writes": []} for i in (1, 2, 3, 4)]
        for dist in ("unpredictable", "unpredictable.exponential"):
            rng = random.Random(3)
            cfg = {"prediction": dist, "mem": {}, "regs": {}}
            for index in _model_picks(models, False, rng, cfg, 20):
                self.assertIn(models[index]["regs"]["rdi"], [1, 2, 3, 4])

    def test_per_iter_explicit_predictable_cycles(self):
        from perf.cfg import _per_iter_data

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
        from perf.cfg import _per_iter_data

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


class TestFlatDataWithBuckets(unittest.TestCase):
    def test_regs_survive_a_flat_dict_that_also_has_mem(self):
        from perf.cfg import _merge_data

        merged = _merge_data(None, {"rdi": 7, "mem": {"0x1000": 1}})
        self.assertEqual(merged["regs"], {"rdi": 7})
        self.assertEqual(merged["mem"], {"0x1000": 1})

    def test_bucket_keys_win_over_flat_keys_on_a_conflict(self):
        from perf.cfg import _merge_data

        merged = _merge_data(None, {"rdi": 7, "regs": {"rdi": 8}, "mem": {"0x1000": 1}})
        self.assertEqual(merged["regs"], {"rdi": 8})
        self.assertEqual(merged["mem"], {"0x1000": 1})

    def test_flat_dict_without_buckets_is_parsed(self):
        from perf.cfg import _merge_data

        merged = _merge_data(None, {"rdi": 7, "0x1000": 1})
        self.assertEqual(merged["regs"], {"rdi": 7})
        self.assertEqual(merged["mem"], {"0x1000": 1})

    def test_merge_overrides_the_base(self):
        from perf.cfg import _merge_data

        merged = _merge_data(
            {"regs": {"rdi": 1}, "mem": {"0x10": 1}},
            {"regs": {"rdi": 2}, "mem": {"0x20": 2}},
        )
        self.assertEqual(merged["regs"], {"rdi": 2})
        self.assertEqual(merged["mem"], {"0x10": 1, "0x20": 2})


class TestExplicitBoundFallback(unittest.TestCase):
    def test_explicit_bound_values_filters_pointers(self):
        from perf.cfg import _explicit_bound_values

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

    def test_pointer_fallback_only_for_missing_earlier_arguments(self):
        from perf.cfg import _needs_pointer_fallback

        self.assertTrue(_needs_pointer_fallback({"regs": {"rsi": 1000}}))
        self.assertTrue(_needs_pointer_fallback({"regs": {"rsi": [10, 100]}}))
        self.assertFalse(_needs_pointer_fallback({"regs": {"rdi": 15}}))
        self.assertFalse(_needs_pointer_fallback({"regs": {"rdi": 0x41000000000}}))
        self.assertFalse(_needs_pointer_fallback({"regs": {"rsi": 0x41000000000}}))

    def test_sequence_probe_uses_the_largest_bound(self):
        from perf.cfg import _sequence_probe_regs

        self.assertEqual(
            _sequence_probe_regs(
                {
                    "regs": {
                        "rsi": [1, 10, 1000, 100000],
                        "rdi": [0x41000000000, 0x42000000000],
                        "rdx": 5,
                    }
                }
            ),
            {"rsi": 100000},
        )

    def test_augment_maps_requested_pages(self):
        import ctypes

        from perf.cfg import _augment_data_with_fallback

        data = {"regs": {"rcx": 100}, "mem": {}, "code": [0x4200000000]}
        out = _augment_data_with_fallback(dict(data), pages=2)
        self.assertEqual(out["regs"]["rcx"], 100)
        self.assertEqual(out["code"], [0x4200000000])
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
        from perf.cfg import _is_harness_fault

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
        from perf.cfg import _grow_fallback_data

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
        from perf.cfg import _grow_fallback_data

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

        from perf.cfg import _grow_fallback_data

        data = {"regs": {"rcx": 100}, "mem": {}}

        def _probe(augmented):
            raise ValueError("faulted with a segmentation fault (SIGSEGV)")

        with (
            patch.object(core, "_available_memory", return_value=1 << 20),
            patch.object(core, "_run_memory", return_value=0),
        ):
            with self.assertRaises(ValueError) as ctx:
                _grow_fallback_data(data, None, _probe)
        self.assertIn("faulted with", str(ctx.exception))

    def test_missing_arg_regs_excludes_explicit_and_harness(self):
        from perf.cfg import _missing_arg_regs

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

        from perf.cfg import _augment_data_with_fallback

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
        from perf.cfg import _augment_data_with_fallback

        data = {"regs": {}, "mem": {}}
        inp = dict(data)
        self.assertIs(_augment_data_with_fallback(inp), inp)
        data = {"regs": {"rdi": 0x5000000000}, "mem": {}}
        inp = dict(data)
        self.assertIs(_augment_data_with_fallback(inp), inp)

    def test_benchmark_supplies_a_pointer_for_an_explicit_sequence(self):
        from perf.bench import benchmark
        from perf.cfg import _FALLBACK_DATA_BASE

        df = benchmark(
            target={"asm": "mov rax, [rdi]; ret"},
            mode=["latency"],
            event=["duration_time"],
            data={"rsi": [0, 1]},
            config={
                "samples": 1,
                "iterations": 2,
                "branch": "predictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
        )
        self.assertEqual(set(df["data.rdi"].astype(int)), {_FALLBACK_DATA_BASE})
        self.assertEqual(df["data.rsi"].astype(int).tolist(), [0, 1])

    def test_benchmark_supplies_a_pointer_for_an_explicit_bound(self):
        from perf.bench import benchmark
        from perf.cfg import _FALLBACK_DATA_BASE

        df = benchmark(
            target={"asm": "mov rax, [rdi]; ret"},
            mode=["latency"],
            event=["duration_time"],
            data={"rsi": 1000},
            config={
                "samples": 1,
                "iterations": 2,
                "branch": "predictable",
                "dcache": "hot",
                "icache": "hot",
                "dtlb": "hot",
                "itlb": "hot",
                "code": {"align": 16},
            },
        )
        self.assertEqual(set(df["data.rdi"].astype(int)), {_FALLBACK_DATA_BASE})


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


class TestTagConfigColumnsEdges(unittest.TestCase):
    def test_list_values_are_repeated_per_row(self):
        from perf.cfg import _tag_config_columns

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
        from perf.cfg import _tag_config_columns

        self.assertTrue(
            _tag_config_columns(pd.DataFrame(), {"code": [{"align": 1}]}).empty
        )

        df = pd.DataFrame({"cycles": [1.0]})
        with patch.object(
            cfg_mod, "config_param_columns", side_effect=RuntimeError("boom")
        ):
            out = _tag_config_columns(df, {"code": [{"align": 1}]})
        self.assertIs(out, df)

    def test_missing_columns_are_added_next_to_the_frame(self):
        from perf.cfg import _tag_config_columns

        df = pd.DataFrame({"cycles": [1.0, 2.0]}, index=[5, 6])
        out = _tag_config_columns(df, {"iterations": {"min": 100, "max": 200}})
        self.assertEqual(out["config.iterations.min"].tolist(), [100, 100])
        self.assertEqual(out.index.tolist(), [5, 6])


class TestFillDynamic(unittest.TestCase):
    def test_zero_iterations_leaves_the_buffer_untouched(self):

        from perf.cfg import _fill_dynamic

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

        from perf.cfg import _fill_dynamic

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

        from perf.cfg import _fill_dynamic

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

        from perf.cfg import _fill_dynamic

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

        from perf.cfg import _fill_dynamic

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


class TestEntryData(unittest.TestCase):
    def test_unrelated_register_keeps_synthetic_argv_env(self):
        from perf.cfg import _entry_data

        data = _entry_data(
            "main",
            "a.out",
            {"regs": {"rax": 5}, "mem": {}},
            argv=["a.out", "arg"],
            env=["K=V"],
        )
        self.assertEqual(data["regs"]["rax"], 5)
        self.assertIn("rdi", data["regs"])
        self.assertIn("rsi", data["regs"])
        self.assertIn("rdx", data["regs"])
        self.assertTrue(any(str(k).startswith("0x4200000000") for k in data["mem"]))

    def test_non_entry_target_is_left_unchanged(self):
        from perf.cfg import _entry_data

        data = {"regs": {"rax": 5}, "mem": {"0x1000": 1}}
        self.assertIs(_entry_data("work", "a.out", data), data)

    def test_large_argv_env_does_not_overlap_the_pointer_table(self):
        from perf.cfg import _entry_data

        argv = [f"a{i}" for i in range(40)]
        env = [f"K{i}=V" for i in range(40)]
        data = _entry_data(
            "main",
            "prog",
            {"regs": {}, "mem": {}},
            argv=argv,
            env=env,
        )
        mem = {int(k, 16): int(v) for k, v in data["mem"].items()}

        def read_cstr(addr):
            out = bytearray()
            while True:
                word = mem.get(addr, 0)
                chunk = word.to_bytes(8, "little")
                if b"\0" in chunk:
                    out += chunk[: chunk.index(0)]
                    return out.decode()
                out += chunk
                addr += 8

        argv_at = data["regs"]["rsi"]
        envp_at = data["regs"]["rdx"]
        self.assertEqual(
            [read_cstr(mem[argv_at + 8 * i]) for i in range(len(argv))],
            argv,
        )
        self.assertEqual(
            [read_cstr(mem[envp_at + 8 * i]) for i in range(len(env))],
            env,
        )


class TestBenchRegression(unittest.TestCase):
    def test_string_seed_is_accepted(self):
        cfg_mod.validate_spec({"seed": "myseed"})
        cfg_mod.validate_spec({"seed": "random"})
        cfg_mod.validate_spec({"seed": 42})
        with self.assertRaises(ValueError):
            cfg_mod.validate_spec({"seed": ["bad"]})

    def test_code_level_addrs_ignore_register_keys(self):
        from perf.cfg import _code_level_addrs

        self.assertEqual(
            _code_level_addrs(
                {
                    "rdi": {"L1i": 0},
                    "0x1000": {"L1i": 0},
                    0x2000: {"TLBi": 25},
                }
            ),
            {0x1000, 0x2000},
        )

    def test_static_global_data_is_not_zeroed(self):
        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "guard.c"
            exe = Path(tmp) / "guard.out"
            src.write_text(
                "long count = 7;\n"
                "__attribute__((noinline)) long myfunc(long x) {\n"
                "    if (count != 7) __builtin_trap();\n"
                "    return x + count;\n"
                "}\n"
                "int main(void) { return 0; }\n"
            )
            built = subprocess.run(
                ["gcc", "-O2", "-fPIE", "-pie", str(src), "-o", str(exe)],
                capture_output=True,
                text=True,
            )
            if built.returncode != 0:
                self.skipTest(built.stderr)
            df = benchmark(
                target=f"{exe}:myfunc",
                mode=["latency"],
                event=["duration_time"],
                data={"rdi": 1},
                config={
                    "samples": 1,
                    "iterations": 1,
                    "branch": "predictable",
                    "dcache": "hot",
                    "dtlb": "hot",
                    "icache": "hot",
                    "itlb": "hot",
                    "code": {"align": 16},
                },
            )
            self.assertFalse(df.empty)

    def test_cache_shortcut_is_validated_fail_fast(self):
        with self.assertRaises(ValueError):
            cfg_mod.validate_spec({"dcache": "bogus"})
        with self.assertRaises(ValueError):
            cfg_mod.validate_spec({"dcache": True})
        cfg_mod.validate_spec({"dcache": "hot"})
        cfg_mod.validate_spec({"dcache": "cold"})

    def test_iterations_count_key_is_rejected(self):
        with self.assertRaises(ValueError):
            cfg_mod.validate_spec({"iterations": {"min": 100, "max": 200, "count": 10}})
        with self.assertRaises(ValueError):
            cfg_mod.validate_spec({"iterations": {"count": [10, 20]}})

    def test_list_tops_reject_leaf_lists_in_dict_form(self):
        with self.assertRaises(ValueError):
            cfg_mod.validate_spec({"code": {"align": [1, 16]}})
        with self.assertRaises(ValueError):
            cfg_mod.validate_spec({"func": {"align": [16, 32]}})

    def test_list_top_shorthands_are_validated(self):
        cfg_mod.validate_spec({"func": "random"})
        cfg_mod.validate_spec({"code": 16})
        cfg_mod.validate_spec({"code": "32"})
        for config in (
            {"func": "bogus"},
            {"code": 3},
            {"code": "nope"},
            {"stack": "auto"},
            {"heap": 64},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                cfg_mod.validate_spec(config)

    def test_bench_one_rejects_leaf_lists(self):
        with self.assertRaises(ValueError):
            bench_impl._bench_one(
                mode="latency", code="nop", config={"code": {"align": [1, 16]}}
            )

    def test_throughput_uses_direct_tlb_steer(self):
        import random

        from perf.bench import _build_loop_asm
        from perf.cfg import get_arch

        meta_models = [
            {"regs": {}, "reads": [(0x400000, 8, 0)], "writes": []},
        ]
        full, _, _ = _build_loop_asm(
            10,
            "throughput",
            "nop;",
            "",
            "",
            meta_models,
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            get_arch(),
            mem_levels={0x400000: {"TLBd": 0}},
            tlb_levels={"TLBd": 0},
        )
        self.assertNotIn("mov rax, r8", full)

    def test_overhead_iterations_do_not_leak_into_result_metadata(self):
        from perf.bench import _measure_group_single

        overhead_iterations = 855
        target_iterations = 100
        trackers = []

        def fake_bench(**kwargs):
            tracker = kwargs.get("iterations_tracker")
            trackers.append(tracker)
            if len(trackers) == 1:
                if tracker is not None:
                    tracker.append(overhead_iterations)
                return pd.DataFrame({"cycles": [10.0]}), overhead_iterations
            if tracker is not None:
                tracker.append(target_iterations)
            return pd.DataFrame({"cycles": [5.0]}), target_iterations

        tracker = []
        with patch.object(bench_impl, "_bench", side_effect=fake_bench):
            result, iterations = _measure_group_single(
                {"samples": 1},
                "throughput",
                "",
                "",
                [],
                ["cycles"],
                None,
                "target",
                tracker,
                {"regs": {}, "mem": {}},
                "nop;",
                "call target;",
            )

        self.assertEqual(iterations, target_iterations)
        self.assertEqual(tracker, [target_iterations])
        self.assertIsNot(trackers[0], tracker)
        self.assertEqual(trackers[0], [overhead_iterations])
        self.assertIs(trackers[1], tracker)
        self.assertEqual(result["cycles"].tolist(), [5.0])


class TestByteRangeRegression(unittest.TestCase):
    def test_mem_entries_string_range_expands_to_bytes(self):
        from perf.cfg import _mem_entries

        self.assertEqual(
            list(_mem_entries("0x1000:", "hi")),
            [("0x1000", 104), ("0x1001", 105)],
        )
        self.assertEqual(
            list(_mem_entries("0x1000:", b"hi")),
            [("0x1000", 104), ("0x1001", 105)],
        )

    def test_solution_code_pages_only_mark_executed_explicit_pages(self):
        from types import SimpleNamespace

        from perf.mca import _solution_code_pages

        page = 0x4200000000
        other = 0x4300000000

        def state(addr, blocks=()):
            return SimpleNamespace(addr=addr, history=SimpleNamespace(bbl_addrs=blocks))

        data = {"mem": {f"0x{page:x}:": b"\x8b\x07\xc3"}}
        self.assertEqual(_solution_code_pages([{"state": state(page)}], data), [page])
        self.assertEqual(
            _solution_code_pages([{"state": state(0x5000, [page])}], data), [page]
        )
        self.assertEqual(_solution_code_pages([{"state": state(other)}], data), [])
        self.assertEqual(_solution_code_pages([], data), [])
        self.assertEqual(_solution_code_pages([{"state": state(page)}], {}), [])
        self.assertEqual(_solution_code_pages(["not-a-solution"], data), [])
        self.assertEqual(
            _solution_code_pages([{"state": SimpleNamespace(addr=None)}], data), []
        )
        self.assertEqual(
            _solution_code_pages(
                [
                    {
                        "state": SimpleNamespace(
                            addr="bad", history=SimpleNamespace(bbl_addrs=["bad"])
                        )
                    }
                ],
                data,
            ),
            [],
        )
        self.assertEqual(
            _solution_code_pages(
                [{"state": SimpleNamespace(addr=page, history=None)}], data
            ),
            [page],
        )
        with patch("perf.mca._iter_mem_addrs", side_effect=RuntimeError):
            self.assertEqual(_solution_code_pages([{"state": state(page)}], data), [])

    def test_bytes_mem_is_not_code_without_execution_evidence(self):
        from perf.cfg import _merge_data

        page = 0x4200000000
        merged = _merge_data(None, {"mem": {f"0x{page:x}:": b"\x8b\x07"}})
        self.assertNotIn("code", merged)
        explicit = _merge_data(
            None, {"mem": {f"0x{page:x}:": b"\x8b\x07"}, "code": [page]}
        )
        self.assertEqual(explicit["code"], [page])
        carried = _merge_data(explicit, {"regs": {"rdx": 4}})
        self.assertEqual(carried["code"], [page])
        normalized = _merge_data(
            None, {"mem": {}, "code": ["bad", page + 0x1001, page]}
        )
        self.assertEqual(normalized["code"], [page, page + 0x1000])

    def test_merge_code_pages_adds_executed_pages_to_data(self):
        from types import SimpleNamespace

        from perf.mca import _merge_code_pages

        page = 0x4200000000
        state = SimpleNamespace(addr=page, history=SimpleNamespace(bbl_addrs=[]))
        data = {"mem": {f"0x{page:x}:": b"\x8b\x07"}, "code": [0x4300000000]}
        merged = _merge_code_pages(data, [{"state": state}])
        self.assertEqual(merged["code"], [page, 0x4300000000])
        self.assertIsNot(merged, data)
        self.assertIs(_merge_code_pages(data, []), data)

    def test_host_target_only_uses_the_default_layout(self):
        from perf.cfg import _host_target

        page = 0x4200000000
        with patch("perf.cfg.host_symbol", return_value=page) as target:
            self.assertEqual(_host_target({}, "libc.so.6", "qsort"), page)
            target.assert_called_once_with("libc.so.6", "qsort")
        with patch("perf.cfg.host_symbol", return_value=page):
            self.assertIsNone(
                _host_target({"func": {"order": "random"}}, "libc.so.6", "qsort")
            )
            self.assertIsNone(
                _host_target({"func": {"align": 32}}, "libc.so.6", "qsort")
            )
        with patch("perf.cfg.host_symbol", side_effect=RuntimeError):
            self.assertIsNone(_host_target({}, "libc.so.6", "qsort"))

    def test_host_target_tries_the_requested_symbol_before_the_demangled_label(self):
        from perf.cfg import _host_target

        page = 0x4200000000
        mangled = "_ZNSt3__19to_stringEi"
        demangled = "std::__1::to_string(int)"

        def _resolve(_file, name):
            return page if name == mangled else None

        with patch("perf.cfg.host_symbol", side_effect=_resolve) as target:
            self.assertEqual(_host_target({}, "libc++.so.1", demangled, mangled), page)
            target.assert_called_once_with("libc++.so.1", mangled)

    def test_address_window_uses_one_read_address_per_state(self):
        from perf.mca import _install_address_window

        marker = object()
        memory = SimpleNamespace(
            read_strategies=[marker],
            write_strategies=[marker],
        )
        _install_address_window(SimpleNamespace(memory=memory))
        self.assertEqual(
            [type(s).__name__ for s in memory.read_strategies],
            [
                "_AddressWindow",
                "SimConcretizationStrategyAny",
                "object",
            ],
        )
        self.assertEqual(
            [type(s).__name__ for s in memory.write_strategies],
            ["_AddressWindow", "object"],
        )

    def test_cold_dtlb_keeps_plain_data_pages_mprotectable(self):
        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        page = 0x4200000000
        _asm, meta, _buf = _build_loop_asm(
            1,
            "latency",
            "nop",
            "",
            "",
            [],
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            arch,
            data={"mem": {f"0x{page:x}:": b"\x01\x02"}},
            tlb_levels={"TLBd": 0},
        )
        self.assertNotIn(page, meta["tlb_avoid"])

    def test_cold_dtlb_does_not_mprotect_explicit_code_pages(self):
        from perf.arch import arch as get_arch
        from perf.bench import _build_loop_asm

        arch = get_arch()
        page = 0x4200000000
        target = 0x440A50
        models = [
            {
                "regs": {"rdi": 0x5200000000, "rsi": 10, "rdx": 4, "rcx": page},
                "reads": [],
                "writes": [],
            }
        ]
        asm, meta, _buf = _build_loop_asm(
            1,
            "latency",
            f"mov rax, {target:#x}; call rax",
            "",
            "",
            models,
            ["duration_time"],
            [None],
            {"L1d": 100, "L2": 0, "L3": 0},
            True,
            random.Random(0),
            arch,
            data={"mem": {f"0x{page:x}:": b"\x8b\x07\xc3"}, "code": [page]},
            tlb_levels={"TLBd": 0},
        )
        self.assertIn(page, meta["tlb_avoid"])
        self.assertNotIn(f"mov rdi, {page:#x}", asm)

    def test_mem_entries_range_rejects_wide_values(self):
        from perf.cfg import _mem_entries

        with self.assertRaisesRegex(ValueError, "0\\.\\.255"):
            list(_mem_entries("0x1000:", [0x1234]))
        with self.assertRaisesRegex(ValueError, "0\\.\\.255"):
            list(_mem_entries("0x1000:", 0x1234))


if __name__ == "__main__":
    unittest.main()
