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

    def test_removed_inversion_jump_is_not_patched_again(self):
        blob = _asm_bytes(
            "jmp 0x1002\nnop\njns 0x1007\njmp 0x1009\njns 0x100c\njmp 0x100b\nnop\nret",
            0x1000,
        )
        new, stats = _branch_function(blob, 0x1000)
        self.assertEqual(stats["jcc_invert"], 1)
        self.assertEqual(new[5:7], x86_64.optimal_nops(2))

    def test_indirect_entry_disables_jump_inversion(self):
        blob = _asm_bytes(
            "jmp rax\nje 0x1006\njmp 0x1020\nnop\nnop",
            0x1000,
        )
        new, stats = _branch_function(blob, 0x1000)
        self.assertEqual(stats["jcc_invert"], 0)
        self.assertEqual(new[2:4], x86_64.assemble("je 0x1006", 0x1002))

    def test_long_jump_chain_is_not_folded_to_an_intermediate_jump(self):
        lines = []
        for i in range(20):
            lines.append(f".L{i}:")
            lines.append(f"jmp .L{i + 1}")
            lines.append("nop")
        lines.append(".L20:")
        lines.append("mov eax, 1")
        lines.append("ret")
        blob = _asm_bytes("\n".join(lines), 0x1000)
        new, _stats = _branch_function(blob, 0x1000)
        first = next(x86_64.disassembler().disasm(new, 0x1000))
        self.assertEqual(x86_64.immediate_target(first), 0x1003)


class TestPlan(unittest.TestCase):
    def test_defaults(self):
        selected, align, order = plan()
        self.assertEqual(selected, list(PASSES))
        self.assertEqual(align, 16)
        self.assertEqual(order, "hot-first")

    def test_subset(self):
        selected, _, _ = plan({"opt": ["peephole"]})
        self.assertEqual(selected, ["peephole"])

    def test_comma(self):
        selected, _, _ = plan({"opt": "peephole,branch"})
        self.assertEqual(selected, ["peephole", "branch"])

    def test_align_implies_layout(self):
        selected, _, _ = plan({"opt": ["align"], "layout": {"align": 32}})
        self.assertIn("layout", selected)
        self.assertIn("align", selected)

    def test_unknown(self):
        with self.assertRaises(ValueError):
            plan({"opt": ["bogus"]})

    def test_bad_align(self):
        with self.assertRaises(ValueError):
            plan({"layout": {"align": 7}})

    def test_extended_align(self):
        selected, align, _ = plan({"opt": ["align"], "layout": {"align": 64}})
        self.assertEqual(align, 64)
        self.assertIn("layout", selected)

    def test_bad_order(self):
        with self.assertRaises(ValueError):
            plan({"layout": {"order": "sideways"}})

    def test_flat_aliases(self):
        cfg = merge_config({"align": 32, "order": "as-is"})
        self.assertEqual(cfg["layout"], {"align": 32, "order": "as-is"})

    def test_opt_string(self):
        cfg = merge_config({"opt": "peephole,branch"})
        self.assertEqual(cfg["opt"], ["peephole", "branch"])

    def test_unknown_config(self):
        with self.assertRaises(ValueError):
            merge_config({"bogus": True})

    def test_opt_override(self):
        selected, _, _ = plan({"opt": ["layout"]}, opt=["peephole"])
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


class TestCli(unittest.TestCase):
    def _args(self, **kw):
        return SimpleNamespace(
            exec=kw.get("exec", None),
            output=kw.get("output", None),
            opt=kw.get("opt", None),
            config=kw.get("config", None),
            select=kw.get("select", None),
            dry_run=kw.get("dry_run", False),
            list_targets=kw.get("list_targets", False),
            json=kw.get("json", False),
            interactive=False,
        )

    def test_output_required(self):
        with self.assertRaises(SystemExit):
            cli.main(self._args(exec="a.out", output=None), [])

    def test_target_required(self):
        with self.assertRaises(SystemExit):
            cli.main(self._args(exec=None, output="b.out"), [])

    def test_unknown_pass(self):
        with self.assertRaises(ValueError):
            perf.optimize(
                exec="a.out",
                output="b.out",
                config={"opt": ["bogus"]},
            )
        with tempfile.TemporaryDirectory() as tmp:
            if not HAVE_GCC:
                self.skipTest("gcc unavailable")
            out = _build_c(tmp, "int main(void){return 0;}\n")
            with self.assertRaises(SystemExit):
                cli.main(
                    self._args(
                        exec=out,
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
                "--config.opt=branch",
            ],
        )
        self.assertEqual(args.config["layout"], {"align": 64, "order": "as-is"})
        self.assertEqual(args.config["opt"], ["branch"])

    def test_cli_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "optimize.json"
            path.write_text(json.dumps({"layout": {"align": 32}}))
            args = cli.parse_target_args(
                cli._build_parser(),
                ["a.out", "--config", str(path), "--config.layout.order=as-is"],
            )
            self.assertEqual(args.config["layout"], {"align": 32, "order": "as-is"})
            self.assertEqual(args.config["opt"], list(PASSES))

    def test_cli_filter_parsing(self):
        args = cli.parse_target_args(
            cli._build_parser(),
            ["a.out", "--select", "foo"],
        )
        self.assertEqual(args.select, "foo")


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

    def test_jcc_inversion_keeps_targeted_jump(self):
        blob = _asm_bytes(
            ".org 0x0F90\n"
            "mov eax, 1\n"
            "ret\n"
            ".org 0x1000\n"
            "je 0x1004\n"
            "jmp 0x0F90\n"
            "mov eax, 2\n"
            "ret\n"
            ".org 0x1080\n"
            "jmp 0x1002\n",
            0x0F80,
        )
        new, stats = _branch_function(blob, 0x0F80)
        self.assertEqual(stats["jcc_invert"], 0)
        self.assertEqual(new, blob)

    def test_hotness_substring_not_misattributed(self):
        self.assertEqual(_match_hotness("domain", {"main": 2.0}), 0.0)
        self.assertEqual(_match_hotness("main", {"main": 2.0}), 2.0)

    def test_verify_accepts_valid_text(self):
        blob = _asm_bytes("jmp 0x1002\nnop", 0x1000)
        x86_64.verify_text(blob, 0x1000, {})


if __name__ == "__main__":
    unittest.main()


class TestDiscoverUnit(unittest.TestCase):
    TEXT = {"addr": 0x1000, "offset": 0, "size": 0x100}

    def _discover(self, symbols, angr=None, has_symtab=True):
        with (
            patch("perf.opt._text_section", return_value=self.TEXT),
            patch("perf.opt._symtab_functions", return_value=symbols),
            patch("perf.opt._has_symtab", return_value=has_symtab),
            patch("perf.opt._angr_functions", return_value=angr or {}),
        ):
            return discover_functions("/tmp/a.out")

    def test_size_zero_helper_is_inferred_from_the_next_symbol(self):
        out = self._discover(
            {
                "a": (0x1000, 0x1000),
                "b": (0x1010, 0x1020),
            }
        )
        self.assertEqual(out["a"], (0x1000, 0x1010))
        self.assertEqual(out["b"], (0x1010, 0x1020))

    def test_symbols_outside_text_are_filtered(self):
        out = self._discover({"inside": (0x1000, 0x1010), "outside": (0x2000, 0x2010)})
        self.assertIn("inside", out)
        self.assertNotIn("outside", out)

    def test_duplicate_ranges_are_deduped(self):
        out = self._discover({"first": (0x1000, 0x1010), "second": (0x1000, 0x1010)})
        self.assertEqual(len(out), 1)

    def test_overlapping_functions_are_dropped(self):
        out = self._discover(
            {
                "big": (0x1000, 0x1050),
                "inside": (0x1010, 0x1020),
                "after": (0x1050, 0x1060),
            }
        )
        self.assertEqual(set(out), {"big", "after"})

    def test_angr_fallback_is_used_without_a_symtab(self):
        out = self._discover(
            {"sym": (0x1000, 0x1010)},
            angr={"found": (0x1020, 0x1030)},
            has_symtab=False,
        )
        self.assertIn("found", out)
        self.assertIn("sym", out)

    def test_no_functions_is_empty(self):
        self.assertEqual(self._discover({}), {})

    def test_text_section_error_propagates(self):
        with patch("perf.opt._text_section", side_effect=ValueError("no text")):
            with self.assertRaises(ValueError):
                discover_functions("/tmp/a.out")


class TestOptimizeReportUnit(unittest.TestCase):
    def test_report_json_contains_the_envelope(self):
        df = pd.DataFrame([{"name": "main", "address": 0x1000}])
        df.attrs.update(
            {
                "file": "a.out@1",
                "output": "b.out",
                "opt": ["peephole"],
                "peephole": {"mov_self": 1},
                "branch": {"jmp_next": 1},
                "fixups": {"rel32": 2},
                "layout": True,
                "layout_skipped": "",
            }
        )
        payload = report_json(df)
        self.assertEqual(payload["file"], "a.out@1")
        self.assertEqual(payload["output"], "b.out")
        self.assertEqual(payload["opt"], ["peephole"])
        self.assertEqual(payload["functions"], [{"name": "main", "address": 0x1000}])

    def test_report_json_handles_missing_attrs(self):
        payload = report_json(pd.DataFrame())
        self.assertEqual(payload["opt"], [])
        self.assertEqual(payload["functions"], [])
        self.assertFalse(payload["layout"])

    def test_describe_prints_the_plan(self):
        df = pd.DataFrame(
            [
                {
                    "name": "main",
                    "address": 0x1000,
                    "new_address": 0x2000,
                    "size": 16,
                    "moved": False,
                    "hotness": 1.5,
                }
            ]
        )
        df.attrs["opt"] = ["layout"]
        text = describe(df)
        self.assertIn("opt: layout", text)
        self.assertIn("main 0x1000->0x2000", text)
        self.assertIn("kept", text)
        self.assertIn("hotness 1.500", text)

    def test_describe_handles_empty_and_invalid_frames(self):
        self.assertIn("opt: ", describe(pd.DataFrame()))
