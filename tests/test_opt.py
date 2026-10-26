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

import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import perf
from perf.arch import x86_64
from perf.opt import (
    PASSES,
    _branch_function,
    _has_jump_table,
    _hotness_from_envelope,
    _match_hotness,
    _peephole_function,
    collect_hotness,
    describe,
    discover_functions,
    merge_config,
    plan,
    report_json,
)


def _cli():
    path = Path(__file__).resolve().parent.parent / "bin" / "perf-optimize"
    loader = importlib.machinery.SourceFileLoader("perfcli_optimize", str(path))
    spec = importlib.util.spec_from_loader("perfcli_optimize", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


cli = _cli()
HAVE_GCC = shutil.which("gcc") is not None
HAVE_GXX = shutil.which("g++") is not None


def _build_c(tmp, src, name="a.out", extra=None):
    src_p = Path(tmp) / "a.c"
    out_p = Path(tmp) / name
    src_p.write_text(src)
    cmd = ["gcc", "-O2", "-fPIE", "-pie", str(src_p), "-o", str(out_p)]
    if extra:
        cmd.extend(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return str(out_p)


def _asm_bytes(asm, addr=0x1000):
    from perf.arch import arch as get_arch

    return bytes(get_arch().assemble(asm, addr))


class TestOptimalNops(unittest.TestCase):
    def test_lengths(self):
        for size in range(1, 20):
            blob = x86_64.optimal_nops(size)
            self.assertEqual(len(blob), size)

    def test_single_bytes(self):
        self.assertEqual(x86_64.optimal_nops(1), b"\x90")
        self.assertEqual(x86_64.optimal_nops(2), b"\x66\x90")

    def test_disassembles_to_nop(self):
        import capstone

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        for size in (1, 2, 3, 5, 8, 9, 15):
            blob = x86_64.optimal_nops(size)
            insns = list(md.disasm(blob, 0x1000))
            self.assertTrue(insns)
            total = sum(i.size for i in insns)
            self.assertEqual(total, size)
            for insn in insns:
                self.assertEqual(insn.mnemonic, "nop")

    def test_zero(self):
        self.assertEqual(x86_64.optimal_nops(0), b"")


class TestPeephole(unittest.TestCase):
    def test_mov_self_64_removed(self):
        blob = _asm_bytes("mov rax, rax", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["mov_self"], 1)
        self.assertEqual(len(new), len(blob))
        self.assertEqual(new, x86_64.optimal_nops(len(blob)))

    def test_mov_self_32_preserved(self):

        blob = _asm_bytes("mov eax, eax", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["mov_self"], 0)
        self.assertEqual(new, blob)

    def test_mov_different_kept(self):
        blob = _asm_bytes("mov rax, rbx", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["mov_self"], 0)
        self.assertEqual(new, blob)

    def test_movabs_shrink_unsigned(self):
        blob = _asm_bytes("movabs rax, 0x1234", 0x1000)
        self.assertEqual(len(blob), 10)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["movabs"], 1)
        self.assertEqual(len(new), 10)

        import capstone

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        insns = list(md.disasm(new, 0x1000))
        self.assertEqual(insns[0].mnemonic, "mov")

    def test_movabs_shrink_signed(self):
        blob = _asm_bytes("movabs rax, -1", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["movabs"], 1)
        self.assertEqual(len(new), len(blob))

    def test_movabs_large_kept(self):
        blob = _asm_bytes("movabs rax, 0x1122334455667788", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["movabs"], 0)
        self.assertEqual(new, blob)

    def test_lea_self_removed(self):
        blob = _asm_bytes("lea rax, [rax]", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["lea_self"], 1)
        self.assertEqual(new, x86_64.optimal_nops(len(blob)))

    def test_lea_other_kept(self):
        blob = _asm_bytes("lea rax, [rbx]", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["lea_self"], 0)

    def test_lea_index_self_removed(self):
        blob = _asm_bytes("lea rax, [rax*1]", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["lea_self"], 1)
        self.assertEqual(new, x86_64.optimal_nops(len(blob)))

    def test_xchg_self_removed(self):
        blob = _asm_bytes("xchg rbx, rbx", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["xchg_self"], 1)
        self.assertEqual(new, x86_64.optimal_nops(len(blob)))

    def test_shift_by_zero_removed(self):
        blob = _asm_bytes("shl rax, 0", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["shift_zero"], 1)
        self.assertEqual(new, x86_64.optimal_nops(len(blob)))

    def test_shift_by_nonzero_kept(self):
        blob = _asm_bytes("shl rax, 1", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["shift_zero"], 0)
        self.assertEqual(new, blob)


class TestBranch(unittest.TestCase):
    def test_jmp_next_removed(self):
        blob = _asm_bytes("jmp 0x1002\nnop", 0x1000)

        new, stats = _branch_function(blob, 0x1000)
        self.assertEqual(stats["jmp_next"], 1)
        self.assertEqual(new[:2], x86_64.optimal_nops(2))

    def test_jcc_next_removed(self):
        blob = _asm_bytes("je 0x1002\nnop", 0x1000)
        new, stats = _branch_function(blob, 0x1000)
        self.assertEqual(stats["jcc_next"], 1)

    def test_jmp_chain_folded(self):

        blob = _asm_bytes("jmp 0x1005\nnop\nnop\nnop\njmp 0x1010", 0x1000)
        new, stats = _branch_function(blob, 0x1000)
        self.assertGreaterEqual(stats["jmp_fold"], 1)
        self.assertEqual(len(new), len(blob))

    def test_no_self_loop(self):
        blob = _asm_bytes("jmp 0x1000", 0x1000)
        new, stats = _branch_function(blob, 0x1000)
        self.assertEqual(new, blob)

    def test_size_preserved(self):
        for asm in ("jmp 0x1010", "je 0x1010", "jne 0x1005"):
            blob = _asm_bytes(asm, 0x1000)
            new, _ = _branch_function(blob, 0x1000)
            self.assertEqual(len(new), len(blob))

    def test_conditional_branch_inverted_over_jump(self):
        blob = _asm_bytes(
            "je 0x1004\njmp 0x1020\nnop\nnop",
            0x1000,
        )
        new, stats = _branch_function(blob, 0x1000)
        self.assertEqual(stats["jcc_invert"], 1)
        self.assertEqual(new[0], 0x75)
        self.assertEqual(new[2:], b"\x66\x90\x90\x90")
        self.assertEqual(len(new), len(blob))


class TestPlan(unittest.TestCase):
    def test_defaults(self):
        selected, align, order = plan()
        self.assertEqual(selected, list(PASSES))
        self.assertEqual(align, 16)
        self.assertEqual(order, "hot-first")

    def test_subset(self):
        selected, _, _ = plan({"passes": ["peephole"]})
        self.assertEqual(selected, ["peephole"])

    def test_comma(self):
        selected, _, _ = plan({"passes": "peephole,branch"})
        self.assertEqual(selected, ["peephole", "branch"])

    def test_align_implies_layout(self):
        selected, _, _ = plan({"passes": ["align"], "layout": {"align": 32}})
        self.assertIn("layout", selected)
        self.assertIn("align", selected)

    def test_unknown(self):
        with self.assertRaises(ValueError):
            plan({"passes": ["bogus"]})

    def test_bad_align(self):
        with self.assertRaises(ValueError):
            plan({"layout": {"align": 7}})

    def test_extended_align(self):
        selected, align, _ = plan({"passes": ["align"], "layout": {"align": 64}})
        self.assertEqual(align, 64)
        self.assertIn("layout", selected)

    def test_bad_order(self):
        with self.assertRaises(ValueError):
            plan({"layout": {"order": "sideways"}})

    def test_flat_aliases(self):
        cfg = merge_config({"align": 32, "order": "as-is"})
        self.assertEqual(cfg["layout"], {"align": 32, "order": "as-is"})

    def test_opt_alias(self):
        cfg = merge_config({"opt": "peephole,branch"})
        self.assertEqual(cfg["passes"], ["peephole", "branch"])

    def test_unknown_config(self):
        with self.assertRaises(ValueError):
            merge_config({"bogus": True})

    def test_passes_override(self):
        selected, _, _ = plan({"passes": ["layout"]}, passes=["peephole"])
        self.assertEqual(selected, ["peephole"])


class TestHotness(unittest.TestCase):
    def test_benchmark_envelope(self):
        payload = {
            "file": "a.out@abc",
            "name": "hot-12345678",
            "output": [
                {"mode": "latency", "operations": 2, "cycles": 100.0},
                {"mode": "latency", "operations": 2, "cycles": 200.0},
            ],
        }
        scores = _hotness_from_envelope(payload)
        self.assertIn("hot", scores)
        self.assertGreater(scores["hot"], 0)

    def test_profile_envelope(self):
        payload = {
            "file": "a.out@abc",
            "name": "track-12345678",
            "output": [
                {"name": "hot", "iteration": 0, "operations": 1, "cycles": 50},
                {"name": "hot", "iteration": 1, "operations": 1, "cycles": 60},
                {"name": "cold", "iteration": 0, "operations": 1, "cycles": 5},
            ],
        }
        scores = _hotness_from_envelope(payload)
        self.assertGreater(scores.get("hot", 0), scores.get("cold", 0))

    def test_analyze_envelope(self):
        payload = {
            "file": "a.out@abc",
            "name": "func",
            "output": [
                {"address": 0x1000, "latency": 2.0},
                {"address": 0x1005, "latency": 3.0},
            ],
        }
        scores = _hotness_from_envelope(payload)
        self.assertIn("func", scores)
        self.assertAlmostEqual(scores["func"], 5.0)

    def test_empty(self):
        self.assertEqual(_hotness_from_envelope({"output": []}), {})

    def test_collect_json_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            hot = {
                "file": "a.out@x",
                "name": "hot-11111111",
                "output": [{"mode": "latency", "operations": 1, "cycles": 100}],
            }
            cold = {
                "file": "a.out@x",
                "name": "cold-22222222",
                "output": [{"mode": "latency", "operations": 1, "cycles": 10}],
            }
            Path(tmp, "hot.json").write_text(json.dumps(hot))
            Path(tmp, "cold.json").write_text(json.dumps(cold))
            scores, details = collect_hotness([tmp])

            self.assertIn("hot", scores)
            self.assertIn("cold", scores)

    def test_collect_single_benchmark_ranks(self):

        payload = {
            "file": "a.out@x",
            "name": "track",
            "output": [
                {"name": "hot", "iteration": 0, "operations": 1, "cycles": 100},
                {"name": "cold", "iteration": 0, "operations": 1, "cycles": 10},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "profile.json")
            Path(path).write_text(json.dumps(payload))
            scores, _ = collect_hotness([path])
            self.assertGreater(scores.get("hot", 0), scores.get("cold", 0))

    def test_collect_dataframe(self):
        df = pd.DataFrame(
            [{"mode": "latency", "operations": 1, "cycles": 42.0}],
        )
        df.attrs["name"] = "myfunc-12345678"
        df.attrs["file"] = "a.out@abc"
        scores, _ = collect_hotness([df])
        self.assertIn("myfunc", scores)

    def test_collect_perf_data_mock(self):
        with patch("perf.opt._hotness_from_perf_data", return_value={"hot": 90.0}):
            with tempfile.TemporaryDirectory() as tmp:
                fake = os.path.join(tmp, "perf.data")
                Path(fake).write_bytes(b"PERFILE2" + b"\x00" * 64)
                scores, _ = collect_hotness([fake])
                self.assertIn("hot", scores)

    def test_profile_envelope_with_addresses(self):
        payload = {
            "file": "a.out@abc",
            "name": "track-12345678",
            "output": [
                {
                    "name": "hot",
                    "iteration": 0,
                    "address": 0x1000,
                    "cycles": 50,
                },
                {
                    "name": "cold",
                    "iteration": 0,
                    "address": 0x2000,
                    "cycles": 5,
                },
            ],
        }
        scores = _hotness_from_envelope(payload)
        self.assertEqual(scores["hot"], 50.0)
        self.assertEqual(scores["cold"], 5.0)

    def test_benchmark_envelope_with_row_names(self):
        payload = {
            "file": "a.out@abc",
            "name": "track",
            "output": [
                {"name": "hot", "operations": 1, "cycles": 50},
                {"name": "cold", "operations": 1, "cycles": 500},
            ],
        }
        scores = _hotness_from_envelope(payload)
        self.assertEqual(scores["hot"], 50.0)
        self.assertEqual(scores["cold"], 500.0)

    def test_mangled_names_match_demangled_scores(self):
        with patch("perf.core.demangle", return_value="ns::foo()"):
            self.assertEqual(_match_hotness("_ZN2ns3fooEv", {"ns::foo": 10.0}), 10.0)

    def test_match_hotness(self):
        scores = {"hot": 2.0}
        self.assertEqual(_match_hotness("hot", scores), 2.0)
        self.assertEqual(_match_hotness("hot(int)", scores), 2.0)
        self.assertEqual(_match_hotness("unknown", scores), 0.0)
        self.assertEqual(_match_hotness("hot", {}), 0.0)


class TestJumpHeuristic(unittest.TestCase):
    def test_mem_indirect_flagged(self):
        blob = _asm_bytes("jmp qword ptr [rax]", 0x1000)
        self.assertTrue(_has_jump_table(blob, 0x1000))

    def test_reg_indirect_without_table_allowed(self):
        blob = _asm_bytes("jmp rax", 0x1000)
        self.assertFalse(_has_jump_table(blob, 0x1000))

    def test_scaled_table_flagged(self):
        blob = _asm_bytes("mov eax, [rdi + rax*4]\njmp rax", 0x1000)
        self.assertTrue(_has_jump_table(blob, 0x1000))

    def test_direct_not_flagged(self):
        blob = _asm_bytes("jmp 0x1010", 0x1000)
        self.assertFalse(_has_jump_table(blob, 0x1000))


@unittest.skipUnless(HAVE_GCC, "gcc required")
class TestDiscover(unittest.TestCase):
    def test_finds_user_functions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(
                tmp,
                "int foo(void){return 1;}\nint main(void){return foo();}\n",
            )
            funcs = discover_functions(out)
            self.assertIn("main", funcs)
            self.assertIn("foo", funcs)

    def test_size_zero_helpers_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(tmp, "int main(void){return 0;}\n")
            funcs = discover_functions(out)

            self.assertIn("main", funcs)
            for name, (start, end) in funcs.items():
                self.assertGreater(end, start, name)

    def test_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(
                tmp,
                "int foo(void){return 1;}\nint main(void){return foo();}\n",
            )
            df = perf.optimize(
                target=out,
                output=os.path.join(tmp, "b.out"),
                config={"passes": ["peephole"]},
                filter="foo",
            )
            names = df["name"].tolist()
            self.assertEqual(names, ["foo"])

    def test_no_match_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(tmp, "int main(void){return 0;}\n")
            with self.assertRaises(ValueError):
                perf.optimize(
                    target=out,
                    output=os.path.join(tmp, "b.out"),
                    filter="does_not_exist_xyz",
                )

    def test_target_must_be_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(tmp, "int main(void){return 0;}\n")
            with self.assertRaises(ValueError) as ctx:
                perf.optimize(
                    target=[out, "main"],
                    output=os.path.join(tmp, "b.out"),
                )
            self.assertIn("filter=", str(ctx.exception))


@unittest.skipUnless(HAVE_GCC, "gcc required")
class TestOptimizeIntegration(unittest.TestCase):
    def test_peephole_preserves_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = (
                "__attribute__((noinline)) long load_const(void){"
                ' long r; __asm__ volatile("movabs $0x1234, %0" : "=r"(r));'
                " return r; }\n"
                "int main(void){ volatile long a = load_const();"
                " return (int)(a & 0xff); }\n"
            )
            out = _build_c(tmp, src)
            opt = os.path.join(tmp, "b.out")
            df = perf.optimize(target=out, output=opt, config={"passes": ["peephole"]})
            self.assertGreater(df.attrs["peephole"].get("movabs", 0), 0)
            for path in (out, opt):
                r = subprocess.run([path], capture_output=True)
                self.assertEqual(r.returncode, 0x34)

    def test_branch_preserves_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = (
                "__attribute__((noinline)) void jn(void){"
                ' __asm__ volatile("jmp 1f\\n1:"); }\n'
                "int main(void){ jn(); return 7; }\n"
            )
            out = _build_c(tmp, src)
            opt = os.path.join(tmp, "b.out")
            df = perf.optimize(target=out, output=opt, config={"passes": ["branch"]})
            self.assertGreater(df.attrs["branch"].get("jmp_next", 0), 0)
            for path in (out, opt):
                r = subprocess.run([path], capture_output=True)
                self.assertEqual(r.returncode, 7)

    def test_layout_reorders_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = (
                "int hot(int n){int s=0;for(int i=0;i<n;i++)s+=i;return s;}\n"
                "int cold(int n){return n+1;}\n"
                "int main(void){volatile int h=hot(100);"
                "volatile int c=cold(1);return h+c;}\n"
            )
            out = _build_c(tmp, src)
            payload = {
                "file": "a.out@x",
                "name": "track",
                "output": [
                    {
                        "name": "cold",
                        "iteration": 0,
                        "operations": 1,
                        "cycles": 1000,
                    },
                    {
                        "name": "hot",
                        "iteration": 0,
                        "operations": 1,
                        "cycles": 10,
                    },
                ],
            }
            data = os.path.join(tmp, "profile.json")
            Path(data).write_text(json.dumps(payload))
            opt = os.path.join(tmp, "b.out")
            df = perf.optimize(
                target=out,
                output=opt,
                data=[data],
                config={"passes": ["layout"]},
            )
            self.assertTrue(bool(df.attrs.get("layout")))
            hot_new = df[df["name"] == "hot"]["new_address"].iloc[0]
            cold_new = df[df["name"] == "cold"]["new_address"].iloc[0]
            self.assertLess(cold_new, hot_new)
            for path in (out, opt):
                r = subprocess.run([path], capture_output=True)
                self.assertEqual(r.returncode, 88)

    def test_all_data_sources_combine(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = "int hot(void){return 1;}\nint main(void){return hot();}\n"
            out = _build_c(tmp, src)
            bench = {
                "file": "a.out@x",
                "name": "hot-11111111",
                "output": [{"mode": "latency", "operations": 1, "cycles": 50}],
            }
            bench_p = os.path.join(tmp, "bench.json")
            Path(bench_p).write_text(json.dumps(bench))
            prof = {
                "file": "a.out@x",
                "name": "track",
                "output": [
                    {
                        "name": "hot",
                        "iteration": 0,
                        "operations": 1,
                        "cycles": 70,
                    }
                ],
            }
            prof_p = os.path.join(tmp, "profile.json")
            Path(prof_p).write_text(json.dumps(prof))
            frame = pd.DataFrame([{"mode": "latency", "operations": 1, "cycles": 30.0}])
            frame.attrs["name"] = "hot-22222222"
            frame.attrs["file"] = "a.out@x"
            opt = os.path.join(tmp, "b.out")
            df = perf.optimize(
                target=out,
                output=opt,
                data=[bench_p, prof_p, frame],
                config={"passes": ["peephole"]},
            )
            self.assertIn("hot", df["name"].tolist())
            self.assertTrue(os.path.exists(opt))

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(tmp, "int main(void){return 0;}\n")
            opt = os.path.join(tmp, "b.out")
            df = perf.optimize(
                target=out,
                output=opt,
                config={"passes": ["peephole"]},
                dry_run=True,
            )
            self.assertFalse(os.path.exists(opt))
            self.assertIn("name", df.columns)

    def test_same_input_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(tmp, "int main(void){return 0;}\n")
            with self.assertRaises(ValueError):
                perf.optimize(target=out, output=out)

    def test_missing_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(tmp, "int main(void){return 0;}\n")
            with self.assertRaises(ValueError):
                perf.optimize(target=out, output=None)

    def test_non_pie_skips_layout_safely(self):
        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            src_p = Path(tmp) / "a.c"
            out = str(Path(tmp) / "a.out")
            src_p.write_text("int main(void){return 0;}\n")
            r = subprocess.run(
                ["gcc", "-O2", "-fno-pie", "-no-pie", str(src_p), "-o", out],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0 or not os.path.exists(out):
                self.skipTest("non-PIE build failed")
            opt = os.path.join(tmp, "b.out")
            df = perf.optimize(target=out, output=opt, config={"passes": ["layout"]})
            self.assertFalse(bool(df.attrs.get("layout")))
            self.assertIn("non-PIE", df.attrs.get("layout_skipped", ""))
            for path in (out, opt):
                proc = subprocess.run([path], capture_output=True)
                self.assertEqual(proc.returncode, 0)

    def test_report_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(tmp, "int main(void){return 0;}\n")
            opt = os.path.join(tmp, "b.out")
            df = perf.optimize(target=out, output=opt, config={"passes": ["peephole"]})
            payload = report_json(df)
            self.assertIn("passes", payload)
            self.assertIn("functions", payload)
            self.assertIn("peephole", payload)

    def test_describe(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(tmp, "int main(void){return 0;}\n")
            opt = os.path.join(tmp, "b.out")
            df = perf.optimize(
                target=out,
                output=opt,
                config={"passes": ["peephole"]},
                dry_run=True,
            )
            text = describe(df)
            self.assertIn("passes:", text)
            self.assertIn("main", text)


class TestCli(unittest.TestCase):
    def _args(self, **kw):
        return SimpleNamespace(
            target=kw.get("target", None),
            output=kw.get("output", None),
            opt=kw.get("opt", None),
            config=kw.get("config", None),
            filter=kw.get("filter", None),
            dry_run=kw.get("dry_run", False),
            list_targets=kw.get("list_targets", False),
            json=kw.get("json", False),
            interactive=False,
        )

    def test_output_required(self):
        with self.assertRaises(SystemExit):
            cli.main(self._args(target="a.out", output=None), [])

    def test_target_required(self):
        with self.assertRaises(SystemExit):
            cli.main(self._args(target=None, output="b.out"), [])

    def test_unknown_pass(self):
        with self.assertRaises(ValueError):
            perf.optimize(
                target="a.out",
                output="b.out",
                config={"passes": ["bogus"]},
            )
        with tempfile.TemporaryDirectory() as tmp:
            if not HAVE_GCC:
                self.skipTest("gcc unavailable")
            out = _build_c(tmp, "int main(void){return 0;}\n")
            with self.assertRaises(SystemExit):
                cli.main(
                    self._args(
                        target=out,
                        output=os.path.join(tmp, "b.out"),
                        opt=["bogus"],
                    ),
                    [],
                )

    def test_cli_opt_parsing(self):
        self.assertEqual(cli._parse_opts(None), None)
        self.assertEqual(cli._parse_opts("a,b"), ["a", "b"])
        self.assertEqual(cli._parse_opts(["a,b", "c"]), ["a", "b", "c"])

    def test_cli_config_overrides(self):
        args = cli.parse_target_args(
            cli._build_parser(),
            [
                "a.out",
                "--config.layout.align=64",
                "--config.layout.order=as-is",
                "--config.passes=branch",
            ],
        )
        self.assertEqual(args.config["layout"], {"align": 64, "order": "as-is"})
        self.assertEqual(args.config["passes"], ["branch"])

    def test_cli_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "optimize.json"
            path.write_text(json.dumps({"layout": {"align": 32}}))
            args = cli.parse_target_args(
                cli._build_parser(),
                ["a.out", "--config", str(path), "--config.layout.order=as-is"],
            )
            self.assertEqual(args.config["layout"], {"align": 32, "order": "as-is"})
            self.assertEqual(args.config["passes"], list(PASSES))

    @unittest.skipUnless(HAVE_GCC, "gcc required")
    def test_cli_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(tmp, "int main(void){return 0;}\n")
            opt = os.path.join(tmp, "b.out")
            cli.main(self._args(target=out, output=opt, dry_run=True), [])
            self.assertFalse(os.path.exists(opt))

    @unittest.skipUnless(HAVE_GCC, "gcc required")
    def test_cli_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(tmp, "int main(void){return 0;}\n")
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main(self._args(target=out, list_targets=True), [])
            self.assertIn("main", buf.getvalue())

    def test_cli_filter_parsing(self):
        args = cli.parse_target_args(
            cli._build_parser(),
            ["a.out", "--filter", "foo"],
        )
        self.assertEqual(args.filter, "foo")

    @unittest.skipUnless(HAVE_GCC, "gcc required")
    def test_cli_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(
                tmp,
                "int foo(void){return 1;}\nint main(void){return foo();}\n",
            )
            opt = os.path.join(tmp, "b.out")
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main(
                    self._args(
                        target=out,
                        output=opt,
                        config={"passes": ["peephole"]},
                        filter="foo",
                    ),
                    [],
                )
            text = buf.getvalue()
            self.assertIn("foo", text)
            self.assertNotIn("main", text)
            self.assertTrue(os.path.exists(opt))

    @unittest.skipUnless(HAVE_GCC, "gcc required")
    def test_cli_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(tmp, "int main(void){return 0;}\n")
            opt = os.path.join(tmp, "b.out")
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main(self._args(target=out, output=opt, json=True), [])
            payload = json.loads(buf.getvalue())
            self.assertIn("functions", payload)


class TestPeepholeSafety(unittest.TestCase):
    def test_xchg_subregs_preserved(self):
        blob = _asm_bytes("xchg al, ah", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["xchg_self"], 0)
        self.assertEqual(new, blob)

    def test_xchg_same_preserved_or_removed_exact(self):
        blob = _asm_bytes("xchg rbx, rbx", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["xchg_self"], 1)
        self.assertEqual(new, x86_64.optimal_nops(len(blob)))

    def test_lea_32bit_address_preserved(self):
        blob = _asm_bytes("lea rax, [eax]", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["lea_self"], 0)
        self.assertEqual(new, blob)

    def test_movabs_r8_shrinks(self):
        blob = _asm_bytes("movabs r8, 0x1234", 0x1000)
        self.assertEqual(len(blob), 10)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["movabs"], 1)
        self.assertEqual(len(new), 10)

    def test_push_pop_same_removed(self):
        blob = _asm_bytes("push rbx\npop rbx", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["push_pop"], 0)
        self.assertEqual(new, blob)

    def test_push_pop_different_kept(self):
        blob = _asm_bytes("push rbx\npop rcx", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["push_pop"], 0)
        self.assertEqual(new, blob)

    def test_rcl_zero_kept(self):
        blob = _asm_bytes("rcl rax, 0", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["shift_zero"], 0)
        self.assertEqual(new, blob)

    def test_32bit_self_xchg_preserved(self):
        for asm in ("xchg ebx, ebx", "xchg eax, eax", "xchg ecx, ecx"):
            blob = _asm_bytes(asm, 0x1000)
            new, stats = _peephole_function(blob, 0x1000)
            self.assertEqual(stats["xchg_self"], 0, asm)
            self.assertEqual(new, blob, asm)

    def test_64bit_self_xchg_still_removed(self):
        blob = _asm_bytes("xchg rbx, rbx", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["xchg_self"], 1)
        self.assertNotEqual(new, blob)

    def test_32bit_shift_zero_preserved(self):
        for asm in (
            "shl eax, 0",
            "shr eax, 0",
            "sar ebx, 0",
            "rol ecx, 0",
            "ror edx, 0",
        ):
            blob = _asm_bytes(asm, 0x1000)
            new, stats = _peephole_function(blob, 0x1000)
            self.assertEqual(stats["shift_zero"], 0, asm)
            self.assertEqual(new, blob, asm)

    def test_64bit_shift_zero_still_removed(self):
        blob = _asm_bytes("shl rax, 0", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["shift_zero"], 1)
        self.assertNotEqual(new, blob)

    def test_push_pop_preserved(self):
        blob = _asm_bytes("push rbx\npop rbx", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["push_pop"], 0)
        self.assertEqual(new, blob)


class TestBranchSafety(unittest.TestCase):
    def test_transitive_chain_folded(self):
        blob = _asm_bytes(
            "jmp 0x1005\nnop\nnop\nnop\njmp 0x1009\nnop\nnop\njmp 0x1010",
            0x1000,
        )
        new, stats = _branch_function(blob, 0x1000)
        self.assertGreaterEqual(stats["jmp_fold"], 2)
        self.assertEqual(len(new), len(blob))
        first = list(x86_64.detailed_disassembler().disasm(new, 0x1000))[0]
        self.assertEqual(x86_64.immediate_target(first), 0x1010)

    def test_jcc_self_loop_not_inverted(self):
        blob = _asm_bytes("je 0x1004\njmp 0x1002\nnop\nnop", 0x1000)
        new, stats = _branch_function(blob, 0x1000)
        self.assertEqual(stats["jcc_invert"], 0)
        self.assertEqual(len(new), len(blob))

    def test_hotness_substring_not_misattributed(self):
        self.assertEqual(_match_hotness("domain", {"main": 2.0}), 0.0)
        self.assertEqual(_match_hotness("main", {"main": 2.0}), 2.0)

    def test_verify_accepts_valid_text(self):
        blob = _asm_bytes("jmp 0x1002\nnop", 0x1000)
        x86_64.verify_text(blob, 0x1000, {})


if __name__ == "__main__":
    unittest.main()
