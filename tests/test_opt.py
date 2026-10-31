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
from unittest.mock import Mock, patch

import pandas as pd

import perf
from perf.arch import x86_64
from perf.opt import (
    _PASSES,
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
        self.assertEqual(selected, list(_PASSES))
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

    def test_collect_does_not_double_count_a_file_reached_twice(self):
        payload = {
            "file": "a.out@x",
            "name": "hot",
            "output": [{"name": "hot", "operations": 1, "cycles": 100}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hot.json")
            Path(path).write_text(json.dumps(payload))
            once, _ = collect_hotness([path])
            twice, _ = collect_hotness([tmp, path])
            self.assertEqual(once, twice)
            self.assertEqual(collect_hotness([tmp, tmp])[0], once)

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
    def test_interactive_option_is_gone(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            cli._build_parser().parse_args(["-i"])

    def _args(self, **kw):
        return SimpleNamespace(
            exec=kw.get("exec", None),
            output=kw.get("output", None),
            opt=kw.get("opt", None),
            config=kw.get("config", None),
            dry_run=kw.get("dry_run", False),
            list_targets=kw.get("list_targets", False),
            json=kw.get("json", False),
        )

    def test_default_output_is_derived(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.out"
            path.write_bytes(b"\x7fELF")
            expected = str(Path(tmp) / "a.opt.out")
            df = pd.DataFrame([{"name": "main", "address": 0x1000}])
            df.attrs.update({"output": expected, "layout": False})
            with (
                patch.object(cli.perf, "optimize", return_value=df) as opt,
                patch("sys.stdout"),
            ):
                cli.main(self._args(exec=str(path), output=None), [])
            self.assertEqual(opt.call_args.kwargs["output"], expected)

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
        args = cli.parse_exec_args(
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
            args = cli.parse_exec_args(
                cli._build_parser(),
                ["a.out", "--config", str(path), "--config.layout.order=as-is"],
            )
            self.assertEqual(args.config["layout"], {"align": 32, "order": "as-is"})
            self.assertEqual(args.config["opt"], list(_PASSES))


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

    def test_interior_branch_target_is_not_rewritten(self):
        blob = (
            _asm_bytes("movabs rax, 0x1234", 0x1000)
            + b"\xc3"
            + _asm_bytes("jmp 0x1004", 0x100B)
        )
        new, stats = x86_64.peephole_function(blob, 0x1000, protected={0x1004})
        self.assertEqual(stats["movabs"], 0)
        self.assertEqual(new, blob)

    def test_unreachable_embedded_data_is_not_rewritten(self):
        blob = (
            _asm_bytes("movabs rax, 0x1234", 0x1000)
            + b"\xc3"
            + _asm_bytes("movabs rax, 0x5678", 0x100B)
        )
        new, stats = x86_64.peephole_function(blob, 0x1000)
        self.assertEqual(stats["movabs"], 1)
        self.assertEqual(new[11:], blob[11:])


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
        x86_64.verify_text(blob, 0x1000)

    def test_interior_branch_target_blocks_rewrites(self):
        blob = _asm_bytes("je 0x1002\nnop\nret", 0x1000)
        new, stats = _branch_function(blob, 0x1000, protected={0x1001})
        self.assertEqual(stats["jcc_next"], 0)
        self.assertEqual(new, blob)

    def test_unreachable_branch_bytes_are_not_rewritten(self):
        blob = _asm_bytes("ret", 0x1000) + _asm_bytes("jmp 0x1000", 0x1001)
        new, stats = _branch_function(blob, 0x1000)
        self.assertEqual(new, blob)
        self.assertEqual(sum(stats.values()), 0)

    def test_branch_to_unreachable_data_target_is_kept(self):
        blob = _asm_bytes("jmp 0x1003\nnop", 0x1000)
        new, stats = _branch_function(blob, 0x1000)
        self.assertEqual(stats["jmp_fold"], 0)
        self.assertEqual(new, blob)


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


class TestOptimizeIntegration(unittest.TestCase):
    def test_optimize_api_runs_all_passes(self):
        if not HAVE_GCC:
            self.skipTest("gcc unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            out = _build_c(
                tmp,
                "int hot(int n){int s=0;for(int i=0;i<n;i++)s+=i;return s;}\n"
                "int cold(int n){return n+1;}\n"
                "int main(void){volatile int h=hot(100);"
                "volatile int c=cold(1);(void)(h+c);return 0;}\n",
            )
            data = Path(tmp) / "data.json"
            data.write_text(
                json.dumps(
                    {
                        "file": "a.out",
                        "name": "track",
                        "id": "x",
                        "time": "t",
                        "info": {},
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
                )
            )
            target = Path(tmp) / "b.out"
            df = perf.optimize(
                exec=out,
                output=str(target),
                data=[str(data)],
                opt=["peephole", "branch", "layout"],
            )
            self.assertTrue(target.exists())
            self.assertTrue(df.attrs.get("layout"))
            self.assertIn("hot", list(df["name"]))
            result = subprocess.run([str(target)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_layout_patches_calls_in_other_executable_sections(self):
        if not HAVE_GCC:
            self.skipTest("gcc unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.s"
            src.write_text(
                "    .text\n"
                "    .globl main\n"
                "    .type main, @function\n"
                "main:\n"
                "    call cold_second\n"
                "    cmp $7, %eax\n"
                "    jne .fail\n"
                "    xor %eax, %eax\n"
                "    ret\n"
                ".fail:\n"
                "    mov $1, %eax\n"
                "    ret\n"
                "    .size main, .-main\n"
                "    .globl hot\n"
                "    .type hot, @function\n"
                "hot:\n"
                "    mov $7, %eax\n"
                "    ret\n"
                "    .size hot, .-hot\n"
                '    .section .coldcode,"ax",@progbits\n'
                "    .globl cold_first\n"
                "    .type cold_first, @function\n"
                "cold_first:\n"
                "    ret\n"
                "    .size cold_first, .-cold_first\n"
                "    .globl cold_second\n"
                "    .type cold_second, @function\n"
                "cold_second:\n"
                "    call hot\n"
                "    ret\n"
                "    .size cold_second, .-cold_second\n"
                '    .section .note.GNU-stack,"",@progbits\n'
            )
            out = Path(tmp) / "a.out"
            build = subprocess.run(
                ["gcc", "-fPIE", "-pie", str(src), "-o", str(out)],
                capture_output=True,
                text=True,
            )
            if build.returncode != 0:
                self.skipTest("assembler does not support custom executable sections")
            data = Path(tmp) / "data.json"
            data.write_text(
                json.dumps(
                    {
                        "file": "a.out",
                        "name": "track",
                        "id": "x",
                        "time": "t",
                        "info": {},
                        "output": [
                            {
                                "name": "hot",
                                "iteration": 0,
                                "operations": 1,
                                "cycles": 1,
                            },
                            {
                                "name": "main",
                                "iteration": 0,
                                "operations": 1,
                                "cycles": 1000,
                            },
                        ],
                    }
                )
            )
            target = Path(tmp) / "b.out"
            df = perf.optimize(
                exec=str(out),
                output=str(target),
                data=[str(data)],
                opt=["layout"],
            )
            self.assertTrue(df.attrs["layout"], df.attrs["layout_skipped"])
            result = subprocess.run([str(target)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_relr_layout_updates_packed_relative_relocations(self):
        if not HAVE_GCC:
            self.skipTest("gcc unavailable")
        from elftools.elf.elffile import ELFFile

        with tempfile.TemporaryDirectory() as tmp:
            try:
                out = _build_c(
                    tmp,
                    "static int hot(int n){int s=0;for(int i=0;i<n;i++)s+=i;"
                    "return s;}\n"
                    "static int cold(int n){return n*3+1;}\n"
                    "typedef int (*fn)(int);\n"
                    "static fn table[2] = {hot, cold};\n"
                    "__attribute__((constructor)) static void ctor(void)"
                    '{__builtin_puts("ctor");}\n'
                    "__attribute__((destructor)) static void dtor(void)"
                    '{__builtin_puts("dtor");}\n'
                    "int main(void){return table[0](3)==3 && table[1](2)==7"
                    " ? 0 : 1;}\n",
                    extra=["-Wl,-z,pack-relative-relocs"],
                )
            except AssertionError:
                self.skipTest("linker does not support packed relative relocations")
            with open(out, "rb") as fh:
                sec = ELFFile(fh).get_section_by_name(".relr.dyn")
                if sec is None or int(sec.header["sh_size"]) == 0:
                    self.skipTest("no RELR relocations in the test binary")
            data = Path(tmp) / "data.json"
            data.write_text(
                json.dumps(
                    {
                        "file": "a.out",
                        "name": "track",
                        "id": "x",
                        "time": "t",
                        "info": {},
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
                )
            )
            expected = subprocess.run([out], capture_output=True, text=True)
            target = Path(tmp) / "b.out"
            df = perf.optimize(
                exec=out,
                output=str(target),
                data=[str(data)],
                opt=["layout"],
            )
            self.assertTrue(df.attrs["layout"], df.attrs["layout_skipped"])
            self.assertGreater(df.attrs["fixups"]["relr"], 0)
            result = subprocess.run([str(target)], capture_output=True, text=True)
            self.assertEqual(result.returncode, expected.returncode, result.stderr)
            self.assertEqual(result.stdout, expected.stdout)

    def test_layout_skips_absolute_text_references(self):
        if not HAVE_GCC:
            self.skipTest("gcc unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "abs.s"
            src.write_text(
                "    .text\n"
                "    .globl hot\n"
                "    .type hot, @function\n"
                "hot:\n"
                "    ret\n"
                "    .size hot, .-hot\n"
                "    .globl use_abs\n"
                "    .type use_abs, @function\n"
                "use_abs:\n"
                "    movabs $hot, %rax\n"
                "    ret\n"
                "    .size use_abs, .-use_abs\n"
                "    .globl main\n"
                "    .type main, @function\n"
                "main:\n"
                "    call use_abs\n"
                "    xor %eax, %eax\n"
                "    ret\n"
                "    .size main, .-main\n"
                '    .section .note.GNU-stack,"",@progbits\n'
            )
            out = Path(tmp) / "a.out"
            build = subprocess.run(
                [
                    "gcc",
                    "-fPIE",
                    "-pie",
                    "-Wl,-z,notext",
                    str(src),
                    "-o",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            if build.returncode != 0:
                self.skipTest("assembler does not support text relocations here")
            expected = subprocess.run([str(out)], capture_output=True, text=True)
            target = Path(tmp) / "b.out"
            df = perf.optimize(
                exec=str(out),
                output=str(target),
                opt=["layout"],
            )
            self.assertFalse(df.attrs["layout"])
            self.assertIn("absolute addresses", df.attrs["layout_skipped"])
            result = subprocess.run([str(target)], capture_output=True, text=True)
            self.assertEqual(result.returncode, expected.returncode)


class TestOptConfigHelpers(unittest.TestCase):
    def test_parse_opt_forms(self):
        from perf.opt import _parse_opt

        self.assertEqual(_parse_opt(None), list(_PASSES))
        self.assertEqual(_parse_opt("peephole, branch"), ["peephole", "branch"])
        self.assertEqual(
            _parse_opt(("peephole,branch", "align")), ["peephole", "branch", "align"]
        )

    def test_parse_opt_rejects_bad_types_and_names(self):
        from perf.opt import _parse_opt

        with self.assertRaises(ValueError):
            _parse_opt(7)
        with self.assertRaises(ValueError):
            _parse_opt("peephole,bogus")

    def test_merge_config_none_and_scalar_override(self):
        from perf.opt import _deep_merge, merge_config

        cfg = merge_config(None)
        self.assertEqual(cfg["layout"], {"align": 16, "order": "hot-first"})
        self.assertEqual(_deep_merge({"a": 1}, 2), 2)
        self.assertEqual(_deep_merge(1, {"a": 2}), {"a": 2})

    def test_merge_config_rejects_bad_shapes(self):
        from perf.opt import merge_config

        with self.assertRaises(ValueError):
            merge_config([])
        with self.assertRaises(ValueError):
            merge_config({"layout": "nope"})
        with self.assertRaises(ValueError):
            merge_config({"layout": {"bogus": 1}})
        with self.assertRaises(ValueError):
            merge_config({"align": "wide"})

    def test_plan_without_opt_override_and_unknown(self):
        with self.assertRaises(ValueError):
            plan({"opt": ["bogus"]})


class TestOptFileHelpers(unittest.TestCase):
    def test_same_file_compares_paths_and_identity(self):
        from perf.opt import _same_file

        with tempfile.TemporaryDirectory() as tmp:
            one = Path(tmp) / "one"
            two = Path(tmp) / "two"
            one.write_text("x")
            two.write_text("y")
            self.assertTrue(_same_file(str(one), str(one)))
            self.assertFalse(_same_file(str(one), str(two)))
            self.assertFalse(_same_file(str(one), str(Path(tmp) / "missing")))

    def test_pie_and_exception_table_detection(self):
        from types import SimpleNamespace

        from perf.opt import _has_exception_tables, _is_pie_elf

        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        self.addCleanup(os.unlink, path)
        with patch("perf.opt.ELFFile") as elf:
            elf.return_value.header = {"e_type": "ET_DYN"}
            self.assertTrue(_is_pie_elf(path))
            elf.return_value.header = {"e_type": "ET_EXEC"}
            self.assertFalse(_is_pie_elf(path))
            elf.side_effect = OSError("not an elf")
            self.assertFalse(_is_pie_elf(path))

        section = SimpleNamespace(header={"sh_size": 4})
        with patch("perf.opt.ELFFile") as elf:
            elf.return_value.get_section_by_name.return_value = section
            self.assertTrue(_has_exception_tables(path))
            section.header["sh_size"] = 0
            self.assertFalse(_has_exception_tables(path))
            elf.return_value.get_section_by_name.return_value = None
            self.assertFalse(_has_exception_tables(path))
            elf.side_effect = OSError("not an elf")
            self.assertTrue(_has_exception_tables(path))

    def test_text_section_errors_without_text(self):
        from perf.opt import _text_section

        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        self.addCleanup(os.unlink, path)
        with patch("perf.opt.ELFFile") as elf:
            elf.return_value.get_section_by_name.return_value = None
            with self.assertRaises(ValueError):
                _text_section(path)

    def test_text_section_returns_offsets(self):
        from types import SimpleNamespace

        from perf.opt import _text_section

        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        self.addCleanup(os.unlink, path)
        section = SimpleNamespace(
            header={"sh_addr": 0x1000, "sh_offset": 0x200, "sh_size": 0x40}
        )
        with patch("perf.opt.ELFFile") as elf:
            elf.return_value.get_section_by_name.return_value = section
            self.assertEqual(
                _text_section(path),
                {"addr": 0x1000, "offset": 0x200, "size": 0x40},
            )

    def test_has_symtab(self):
        from perf.opt import _has_symtab

        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        self.addCleanup(os.unlink, path)
        with patch("perf.opt.ELFFile") as elf:
            elf.return_value.get_section_by_name.return_value = object()
            self.assertTrue(_has_symtab(path))
            elf.side_effect = OSError("not an elf")
            self.assertFalse(_has_symtab(path))


class TestSymtabFunctions(unittest.TestCase):
    def test_function_symbols_are_filtered_and_deduplicated(self):
        from types import SimpleNamespace

        from perf.opt import _symtab_functions

        def sym(name, typ, shndx, value, size):
            return SimpleNamespace(
                name=name,
                entry={
                    "st_info": {"type": typ},
                    "st_shndx": shndx,
                    "st_size": size,
                    "st_value": value,
                },
            )

        symbols = [
            sym("foo", "STT_FUNC", 1, 0x1000, 8),
            sym("foo", "STT_FUNC", 1, 0x1000, 8),
            sym("bar", "STT_GNU_IFUNC", 1, 0x2000, 0),
            sym("obj", "STT_OBJECT", 1, 0x3000, 8),
            sym("undef", "STT_FUNC", "SHN_UNDEF", 0x4000, 8),
            sym("", "STT_FUNC", 1, 0x5000, 8),
        ]
        section = SimpleNamespace(iter_symbols=lambda: iter(symbols))
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        self.addCleanup(os.unlink, path)
        with patch("perf.opt.ELFFile") as elf:
            elf.return_value.get_section_by_name.side_effect = lambda name: (
                section if name == ".symtab" else None
            )
            found = _symtab_functions(path)
        self.assertEqual(found, {"foo": (0x1000, 0x1008), "bar": (0x2000, 0x2000)})

    def test_distinct_duplicate_names_keep_both_ranges(self):
        from types import SimpleNamespace

        from perf.opt import _match_hotness, _symtab_functions

        def sym(name, value, size):
            return SimpleNamespace(
                name=name,
                entry={
                    "st_info": {"type": "STT_FUNC"},
                    "st_shndx": 1,
                    "st_size": size,
                    "st_value": value,
                },
            )

        section = SimpleNamespace(
            iter_symbols=lambda: iter([sym("foo", 0x1000, 8), sym("foo", 0x2000, 16)])
        )
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        self.addCleanup(os.unlink, path)
        with patch("perf.opt.ELFFile") as elf:
            elf.return_value.get_section_by_name.side_effect = lambda name: (
                section if name == ".symtab" else None
            )
            found = _symtab_functions(path)
        self.assertEqual(
            found,
            {"foo": (0x1000, 0x1008), "foo#2000": (0x2000, 0x2010)},
        )
        self.assertEqual(_match_hotness("foo#2000", {"foo": 3.0}), 3.0)

    def test_bad_symbol_entries_are_skipped(self):
        from perf.opt import _symtab_functions

        class BadSymbol:
            name = "bad"

            @property
            def entry(self):
                raise RuntimeError("bad entry")

        section = SimpleNamespace(iter_symbols=lambda: iter([BadSymbol()]))
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            path = fh.name
        self.addCleanup(os.unlink, path)
        with patch("perf.opt.ELFFile") as elf:
            elf.return_value.get_section_by_name.return_value = section
            self.assertEqual(_symtab_functions(path), {})


class TestAngrFunctions(unittest.TestCase):
    def test_project_functions_are_rebased_for_pie(self):
        import sys
        import types

        from perf.opt import _angr_functions

        fake_angr = types.ModuleType("angr")
        project = SimpleNamespace(
            loader=SimpleNamespace(
                main_object=SimpleNamespace(pic=True, mapped_base=0x400000)
            )
        )
        fake_angr.Project = lambda *_a, **_k: project
        with (
            patch.dict(sys.modules, {"angr": fake_angr}),
            patch(
                "perf.info.functions",
                return_value=({"f": (0x401000, 0x401010), "bad": ("x", 1)}, {}),
            ),
        ):
            self.assertEqual(_angr_functions("/tmp/a.out"), {"f": (0x1000, 0x1010)})

    def test_project_errors_return_empty(self):
        import sys
        import types

        from perf.opt import _angr_functions

        fake_angr = types.ModuleType("angr")
        fake_angr.Project = Mock(side_effect=RuntimeError("boom"))
        with patch.dict(sys.modules, {"angr": fake_angr}):
            self.assertEqual(_angr_functions("/tmp/a.out"), {})

    def test_non_pie_functions_are_not_rebased(self):
        import sys
        import types

        from perf.opt import _angr_functions

        fake_angr = types.ModuleType("angr")
        project = SimpleNamespace(
            loader=SimpleNamespace(
                main_object=SimpleNamespace(pic=False, mapped_base=0x400000)
            )
        )
        fake_angr.Project = lambda *_a, **_k: project
        with (
            patch.dict(sys.modules, {"angr": fake_angr}),
            patch(
                "perf.info.functions", return_value=({"f": (0x401000, 0x401010)}, {})
            ),
        ):
            self.assertEqual(_angr_functions("/tmp/a.out"), {"f": (0x401000, 0x401010)})


class TestEnvelopeHelpers(unittest.TestCase):
    def test_load_json_envelope(self):
        from perf.opt import _load_json_envelope

        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.json"
            good.write_text(json.dumps({"output": []}))
            self.assertEqual(_load_json_envelope(str(good)), {"output": []})
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps([1, 2]))
            with self.assertRaises(ValueError):
                _load_json_envelope(str(bad))
            bad.write_text(json.dumps({"input": []}))
            with self.assertRaises(ValueError):
                _load_json_envelope(str(bad))

    def test_strip_run_id(self):
        from perf.opt import _strip_run_id

        self.assertEqual(_strip_run_id("hot-1234abcd"), "hot")
        self.assertEqual(_strip_run_id("hot-1234abcg"), "hot-1234abcg")
        self.assertEqual(_strip_run_id("hot"), "hot")
        self.assertEqual(_strip_run_id("hot-1234abcd-extra"), "hot-1234abcd-extra")

    def test_hotness_adds_composite_and_part_names(self):
        payload = {
            "name": "outer-1234abcd",
            "output": [
                {
                    "name": "a..b",
                    "iteration": 0,
                    "operations": 2,
                    "cycles": 10,
                }
            ],
        }
        scores = _hotness_from_envelope(payload)
        self.assertEqual(scores["a..b"], 5.0)
        self.assertEqual(scores["a"], 5.0)
        self.assertEqual(scores["b"], 5.0)

    def test_hotness_ignores_bad_rows_and_metrics(self):
        payload = {
            "output": [
                "not-a-row",
                {"name": "", "iteration": 0},
                {"name": "x", "iteration": 0, "operations": "bad"},
            ]
        }
        scores = _hotness_from_envelope(payload)
        self.assertEqual(scores["x"], 1.0)

    def test_hotness_address_rows_skip_bad_values(self):
        payload = {
            "name": "func",
            "output": [
                "not-a-row",
                {"address": 0x1000, "latency": "bad", "throughput": 4},
                {"address": 0x1001, "latency": 3, "throughput": 99},
                {"address": 0x1002},
            ],
        }
        self.assertEqual(_hotness_from_envelope(payload), {"func": 7.0})

    def test_hotness_address_rows_without_metrics_are_empty(self):
        payload = {"output": [{"address": 0x1000, "latency": 0, "throughput": 0}]}
        self.assertEqual(_hotness_from_envelope(payload), {})

    def test_hotness_named_rows_use_top_name_for_empty_rows(self):
        payload = {
            "name": "fallback-1234abcd",
            "output": [
                {"name": "", "operations": 1, "cycles": 2},
                {"name": "", "instructions": 3},
            ],
        }
        scores = _hotness_from_envelope(payload)
        self.assertEqual(scores["fallback"], 5.0)

    def test_hotness_unnamed_rows_sum_metrics(self):
        payload = {
            "output": [
                {"operations": 1, "cycles": 2},
                {"operations": 1, "duration_time": 3},
            ]
        }
        self.assertEqual(_hotness_from_envelope(payload), {"unknown": 5.0})
        self.assertEqual(_hotness_from_envelope({"output": [{}, {}]}), {"unknown": 2.0})

    def test_hotness_metric_priority(self):
        payload = {
            "name": "n",
            "output": [{"operations": 1, "cycles": 5, "instructions": 99}],
        }
        self.assertEqual(_hotness_from_envelope(payload), {"n": 5.0})

    def test_dataframe_hotness_error_path(self):
        from perf.opt import _hotness_from_dataframe

        df = Mock()
        df.to_dict.side_effect = RuntimeError("boom")
        self.assertEqual(_hotness_from_dataframe(df), {})

    def test_match_hotness_demangle_and_namespace(self):
        from perf.opt import _match_hotness

        with patch("perf.core.demangle", side_effect=RuntimeError("boom")):
            self.assertEqual(_match_hotness("ns::foo(int)", {"foo": 3.0}), 3.0)
        with patch("perf.core.demangle", return_value="ns::foo()"):
            self.assertEqual(_match_hotness("_ZN2ns3fooEv", {"foo": 4.0}), 4.0)
        self.assertEqual(_match_hotness("foo-1234abcd", {"foo": 5.0}), 5.0)


class TestPerfDataHotness(unittest.TestCase):
    def test_record_failures_return_empty(self):
        import pandas as pd

        import perf.data as data
        from perf.opt import _hotness_from_perf_data

        with patch.object(data, "is_record", return_value=False):
            self.assertEqual(_hotness_from_perf_data("/tmp/perf.data"), {})
        with patch.object(data, "is_record", side_effect=OSError("no")):
            self.assertEqual(_hotness_from_perf_data("/tmp/perf.data"), {})
        with (
            patch.object(data, "is_record", return_value=True),
            patch.object(data, "samples", side_effect=RuntimeError("bad")),
            patch("sys.stderr"),
        ):
            self.assertEqual(_hotness_from_perf_data("/tmp/perf.data"), {})
        with (
            patch.object(data, "is_record", return_value=True),
            patch.object(data, "samples", return_value=pd.DataFrame()),
        ):
            self.assertEqual(_hotness_from_perf_data("/tmp/perf.data"), {})
        with (
            patch.object(data, "is_record", return_value=True),
            patch.object(data, "samples", return_value=pd.DataFrame({"x": [1]})),
        ):
            self.assertEqual(_hotness_from_perf_data("/tmp/perf.data"), {})

    def test_record_grouped_and_value_count_fallbacks(self):
        import pandas as pd

        import perf.data as data
        from perf.opt import _hotness_from_perf_data

        frame = pd.DataFrame({"sym": ["foo", "foo", "bar"], "period": [2, 3, 4]})
        with (
            patch.object(data, "is_record", return_value=True),
            patch.object(data, "samples", return_value=frame),
        ):
            self.assertEqual(
                _hotness_from_perf_data("/tmp/perf.data"), {"foo": 5.0, "bar": 4.0}
            )

        class BadGroup:
            columns = ["sym"]

            def groupby(self, _col):
                raise RuntimeError("no group")

            def __getitem__(self, _col):
                return SimpleNamespace(
                    value_counts=Mock(return_value={"foo": 7, "bar": "bad"})
                )

        with (
            patch.object(data, "is_record", return_value=True),
            patch.object(data, "samples", return_value=BadGroup()),
        ):
            self.assertEqual(_hotness_from_perf_data("/tmp/perf.data"), {"foo": 7.0})


class TestFallthrough(unittest.TestCase):
    def _insn(self, mnemonic="ret", raw=b"\xc3", size=1):
        return SimpleNamespace(mnemonic=mnemonic, bytes=raw, size=size, address=0x1000)

    def _arch(self, insns):
        md = Mock()
        md.disasm.return_value = list(insns)
        arch = Mock()
        arch.detailed_disassembler.return_value = md
        arch.strip_trailing_nops.return_value = (b"\x90", 0)
        return arch

    def test_fallthrough_detected(self):
        from perf.opt import _has_fallthrough

        arch = self._arch([self._insn("mov")])
        with patch("perf.opt.get_arch", return_value=arch):
            self.assertTrue(
                _has_fallthrough(
                    {"a": (0x1000, 0x1001), "b": (0x1001, 0x1002)},
                    {"a": b"\x90", "b": b"\x90"},
                )
            )

    def test_last_function_falling_off_is_rejected(self):
        from perf.opt import _has_fallthrough

        arch = self._arch([self._insn("mov")])
        with patch("perf.opt.get_arch", return_value=arch):
            self.assertTrue(_has_fallthrough({"a": (0x1000, 0x1001)}, {"a": b"\x90"}))

    def test_terminators_and_cc_padding_are_allowed(self):
        from perf.opt import _has_fallthrough

        for mnemonic in ("ret", "retf", "retq", "jmp", "hlt", "ud1", "ud2"):
            arch = self._arch([self._insn(mnemonic)])
            with patch("perf.opt.get_arch", return_value=arch):
                self.assertFalse(
                    _has_fallthrough(
                        {"a": (0x1000, 0x1001), "b": (0x1001, 0x1002)},
                        {"a": b"\x90", "b": b"\x90"},
                    ),
                    mnemonic,
                )
        arch = self._arch([self._insn("hlt", raw=b"\xcc", size=1)])
        with patch("perf.opt.get_arch", return_value=arch):
            self.assertFalse(
                _has_fallthrough(
                    {"a": (0x1000, 0x1001), "b": (0x1001, 0x1002)},
                    {"a": b"\x90", "b": b"\x90"},
                )
            )

    def test_arch_and_disassembly_errors_are_conservative(self):
        from perf.opt import _has_fallthrough

        with patch("perf.opt.get_arch", side_effect=RuntimeError("no arch")):
            self.assertTrue(_has_fallthrough({"a": (0x1000, 0x1001)}, {"a": b"x"}))
        arch = Mock()
        arch.detailed_disassembler.side_effect = RuntimeError("no md")
        with patch("perf.opt.get_arch", return_value=arch):
            self.assertTrue(_has_fallthrough({"a": (0x1000, 0x1001)}, {"a": b"x"}))
        arch = self._arch([])
        with patch("perf.opt.get_arch", return_value=arch):
            self.assertTrue(_has_fallthrough({"a": (0x1000, 0x1001)}, {"a": b"x"}))
        arch = self._arch([self._insn("ret")])
        arch.detailed_disassembler.return_value.disasm.side_effect = RuntimeError("bad")
        with patch("perf.opt.get_arch", return_value=arch):
            self.assertTrue(_has_fallthrough({"a": (0x1000, 0x1001)}, {"a": b"x"}))
        arch = self._arch([self._insn("ret")])
        arch.strip_trailing_nops.side_effect = RuntimeError("bad")
        with patch("perf.opt.get_arch", return_value=arch):
            self.assertFalse(_has_fallthrough({"a": (0x1000, 0x1001)}, {"a": b"x"}))

    def test_bad_mnemonic_and_bytes_are_conservative(self):
        from perf.opt import _has_fallthrough

        class BadMnemonic:
            @property
            def mnemonic(self):
                raise RuntimeError("bad")

        arch = self._arch([BadMnemonic()])
        with patch("perf.opt.get_arch", return_value=arch):
            self.assertTrue(_has_fallthrough({"a": (0x1000, 0x1001)}, {"a": b"x"}))

        class BadBytes:
            mnemonic = "mov"

            @property
            def bytes(self):
                raise RuntimeError("bad")

        arch = self._arch([BadBytes()])
        with patch("perf.opt.get_arch", return_value=arch):
            self.assertTrue(
                _has_fallthrough(
                    {"a": (0x1000, 0x1001), "b": (0x1001, 0x1002)},
                    {"a": b"x", "b": b"y"},
                )
            )


class TestTextCoverage(unittest.TestCase):
    def test_empty_range_is_fully_covered(self):
        from perf.opt import _text_coverage

        self.assertEqual(_text_coverage({}, 0x1000, 0x1000), 1.0)

    def test_gaps_reduce_coverage(self):
        from perf.opt import _text_coverage

        funcs = {"a": (0x1000, 0x1010)}
        self.assertEqual(_text_coverage(funcs, 0x1000, 0x1020), 0.5)

    def test_overlaps_and_invalid_spans(self):
        from perf.opt import _text_coverage

        funcs = {
            "a": (0x1000, 0x1010),
            "b": (0x1008, 0x1020),
            "bad": ("x", 1),
            "empty": (0x1100, 0x1100),
        }
        self.assertEqual(_text_coverage(funcs, 0x1000, 0x1020), 1.0)

    def test_nop_gaps_are_not_counted_as_missing(self):
        from perf.opt import _text_coverage

        funcs = {"a": (0x1000, 0x1010)}
        with patch("perf.opt._strip_trailing_nops", return_value=(b"", 0)):
            self.assertEqual(
                _text_coverage(funcs, 0x1000, 0x1020, text=b"\x90" * 0x20), 1.0
            )

    def test_nop_stripping_errors_keep_the_gap(self):
        from perf.opt import _text_coverage

        funcs = {"a": (0x1000, 0x1010)}
        with patch("perf.opt._strip_trailing_nops", side_effect=RuntimeError("bad")):
            self.assertEqual(
                _text_coverage(funcs, 0x1000, 0x1020, text=b"\x90" * 0x20), 0.5
            )


def _uleb(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _sleb(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if (value == 0 and not byte & 0x40) or (value == -1 and byte & 0x40):
            out.append(byte)
            return bytes(out)
        out.append(byte | 0x80)


def _cie(aug=b"zR", enc=0x03, version=1):
    import struct

    body = bytes([version]) + aug + b"\x00"
    body += _uleb(1)
    body += _sleb(0)
    if version == 1:
        body += b"\x00"
    else:
        body += _uleb(1)
    if aug.startswith(b"z"):
        payload = b""
        if b"P" in aug:
            payload += b"\x03" + b"\x00" * 4
        if b"L" in aug:
            payload += b"\x00"
        if b"R" in aug:
            payload += bytes([enc])
        body += _uleb(len(payload)) + payload
    length = 4 + len(body)
    return struct.pack("<I", length) + struct.pack("<I", 0) + body


def _fde(cie_len, pc=0x1000, span=0x20, enc=0x03):
    import struct

    body = struct.pack("<II", pc, span)
    fde_off = cie_len
    cid = fde_off + 4
    length = 4 + len(body)
    return struct.pack("<I", length) + struct.pack("<I", cid) + body


class TestEhFrameParsing(unittest.TestCase):
    def test_parse_eh_frame_cie_and_fde(self):
        from perf.opt import _parse_eh_frame

        cie = _cie()
        fde = _fde(len(cie))
        entries = _parse_eh_frame(cie + fde)
        self.assertEqual([entry["kind"] for entry in entries], ["cie", "fde"])
        self.assertEqual(entries[1]["cie_off"], 0)

    def test_parse_eh_frame_stops_on_bad_lengths(self):
        import struct

        from perf.opt import _parse_eh_frame

        self.assertEqual(_parse_eh_frame(struct.pack("<I", 0) + b"x" * 8), [])
        with self.assertRaises(ValueError):
            _parse_eh_frame(struct.pack("<I", 0xFFFFFFFF) + b"x" * 8)
        with self.assertRaises(ValueError):
            _parse_eh_frame(struct.pack("<I", 100) + b"x" * 4)
        with self.assertRaises(ValueError):
            _parse_eh_frame(struct.pack("<I", 3) + b"x" * 8)

    def test_leb128_readers(self):
        from perf.opt import _read_sleb, _read_uleb

        self.assertEqual(_read_uleb(b"\x81\x01", 0), (129, 2))
        self.assertEqual(_read_sleb(b"\x7f", 0), (-1, 1))
        self.assertEqual(_read_sleb(_sleb(-65), 0), (-65, 2))
        with self.assertRaises(ValueError):
            _read_uleb(b"", 0)
        with self.assertRaises(ValueError):
            _read_uleb(b"\x80" * 20, 0)
        with self.assertRaises(ValueError):
            _read_sleb(b"\x80" * 20, 0)

    def test_encoded_size_table(self):
        from perf.opt import _encoded_size

        self.assertEqual(
            [_encoded_size(fmt) for fmt in (0x00, 0x02, 0x03, 0x04, 0x0A, 0x0B, 0x0C)],
            [8, 2, 4, 8, 2, 4, 8],
        )
        self.assertIsNone(_encoded_size(0x05))


class TestEncodedValues(unittest.TestCase):
    def test_read_all_formats(self):
        import struct

        from perf.opt import _read_encoded

        for fmt, value in (
            (0x00, 7),
            (0x02, 7),
            (0x03, 7),
            (0x04, 7),
            (0x0A, 7),
            (0x0B, 7),
            (0x0C, 7),
        ):
            with self.subTest(fmt=fmt):
                data = struct.pack("<Q", value) + b"\x00" * 8
                got, pos = _read_encoded(data, 0, fmt)
                self.assertEqual(got, value)

        self.assertEqual(
            _read_encoded(_sleb(-9), 0, 0x09), (-9 & 0xFFFFFFFFFFFFFFFF, 1)
        )
        self.assertEqual(
            _read_encoded(struct.pack("<I", 5), 0, 0x13, section_addr=0x1000),
            (0x1005, 4),
        )
        self.assertEqual(
            _read_encoded(struct.pack("<I", 5), 0, 0x33, base=0x2000),
            (0x2005, 4),
        )

    def test_pcrel_udata_round_trips_negative_offsets(self):
        from perf.opt import _read_encoded, _write_encoded

        sec = 0x400000
        for target in (sec - 4, sec - 2, sec + 0x100):
            out = bytearray(4)
            self.assertTrue(_write_encoded(out, 0, 0x13, target, section_addr=sec))
            self.assertEqual(
                _read_encoded(bytes(out), 0, 0x13, section_addr=sec),
                (target, 4),
            )
        out = bytearray(2)
        self.assertTrue(_write_encoded(out, 0, 0x12, sec - 2, section_addr=sec))
        self.assertEqual(
            _read_encoded(bytes(out), 0, 0x12, section_addr=sec),
            (sec - 2, 2),
        )
        self.assertFalse(
            _write_encoded(bytearray(2), 0, 0x12, sec + 0x8000, section_addr=sec)
        )
        self.assertFalse(
            _write_encoded(bytearray(4), 0, 0x13, sec + 0x80000000, section_addr=sec)
        )

    def test_read_encoded_rejects_bad_encodings(self):
        import struct

        from perf.opt import _read_encoded

        data = struct.pack("<I", 1)
        for enc in (None, 0xFF, 0x80):
            self.assertIsNone(_read_encoded(data, 0, enc))
        self.assertIsNone(_read_encoded(data, 0, 0x05))
        self.assertIsNone(_read_encoded(b"", 0, 0x03))
        self.assertIsNone(_read_encoded(data, 0, 0x03, size=2))

    def test_write_all_formats(self):
        import struct

        from perf.opt import _write_encoded

        for fmt, value, fmtchar in (
            (0x00, 7, "<Q"),
            (0x02, 7, "<H"),
            (0x03, 7, "<I"),
            (0x04, 7, "<Q"),
            (0x0A, -7, "<h"),
            (0x0B, -7, "<i"),
            (0x0C, -7, "<q"),
        ):
            with self.subTest(fmt=fmt):
                out = bytearray(16)
                self.assertTrue(_write_encoded(out, 0, fmt, value))
                self.assertEqual(struct.unpack_from(fmtchar, out, 0)[0], value)

    def test_write_encoded_rejects_bad_values(self):
        from perf.opt import _write_encoded

        for enc, value in (
            (None, 1),
            (0xFF, 1),
            (0x80, 1),
            (0x05, 1),
            (0x43, 1),
            (0x53, 1),
            (0x02, 0x10000),
            (0x03, -1),
            (0x0A, 2**15),
            (0x0B, -(2**31) - 1),
            (0x0C, 2**63),
        ):
            with self.subTest(enc=enc, value=value):
                self.assertFalse(_write_encoded(bytearray(16), 0, enc, value))
        self.assertFalse(_write_encoded(bytearray(2), 0, 0x03, 1))

    def test_write_encoded_applies_relative_bases(self):
        import struct

        from perf.opt import _write_encoded

        out = bytearray(16)
        self.assertTrue(_write_encoded(out, 0, 0x10, 0x1010, section_addr=0x1000))
        self.assertEqual(struct.unpack_from("<Q", out, 0)[0], 0x10)
        self.assertTrue(_write_encoded(out, 0, 0x30, 0x1010, base=0x1000))
        self.assertEqual(struct.unpack_from("<Q", out, 0)[0], 0x10)


class TestEhFramePointerEncoding(unittest.TestCase):
    def test_reads_r_encoding(self):
        from perf.opt import _eh_frame_pointer_encoding

        self.assertEqual(_eh_frame_pointer_encoding(_cie(enc=0x1B), 0), 0x1B)

    def test_version_three_reads_return_register(self):
        from perf.opt import _eh_frame_pointer_encoding

        self.assertEqual(_eh_frame_pointer_encoding(_cie(enc=0x03, version=3), 0), 0x03)

    def test_version_four_skips_address_and_segment_sizes(self):
        import struct

        from perf.opt import _eh_frame_pointer_encoding

        body = bytes([4]) + b"zR\x00" + _uleb(1) + _sleb(0) + _uleb(1)
        body += b"\x08\x00" + _uleb(1) + b"\x03"
        cie = struct.pack("<I", 4 + len(body)) + struct.pack("<I", 0) + body
        self.assertEqual(_eh_frame_pointer_encoding(cie, 0), 0x03)

    def test_missing_r_or_bad_cie_returns_none(self):
        from perf.opt import _eh_frame_pointer_encoding

        self.assertIsNone(_eh_frame_pointer_encoding(_cie(aug=b"z"), 0))
        self.assertIsNone(_eh_frame_pointer_encoding(b"\x00" * 8, 0))
        self.assertIsNone(_eh_frame_pointer_encoding(_cie(), 100))

    def test_personality_and_lsda_fields(self):
        from perf.opt import _eh_frame_pointer_encoding

        cie = _cie(aug=b"zPLRS")
        self.assertEqual(_eh_frame_pointer_encoding(cie, 0), 0x03)


class TestEhFramePatching(unittest.TestCase):
    def test_patch_eh_frame_remaps_fde(self):
        import struct

        from perf.opt import _patch_eh_frame

        cie = _cie()
        fde = _fde(len(cie))
        out, updated = _patch_eh_frame(cie + fde, {(0x1000, 0x1020): (0x2000, 0x2020)})
        self.assertEqual(updated, 1)
        body = len(cie) + 8
        self.assertEqual(struct.unpack_from("<I", out, body)[0], 0x2000)
        self.assertEqual(struct.unpack_from("<I", out, body + 4)[0], 0x20)

    def test_unmapped_fde_is_left_alone(self):
        from perf.opt import _patch_eh_frame

        cie = _cie()
        fde = _fde(len(cie))
        out, updated = _patch_eh_frame(cie + fde, {(0x3000, 0x3020): (0x4000, 0x4020)})
        self.assertEqual(updated, 0)
        self.assertEqual(out, cie + fde)

    def test_span_is_clipped_to_the_new_function(self):
        import struct

        from perf.opt import _patch_eh_frame

        cie = _cie()
        fde = _fde(len(cie), span=0x100)
        out, _ = _patch_eh_frame(cie + fde, {(0x1000, 0x1020): (0x2000, 0x2010)})
        body = len(cie) + 8
        self.assertEqual(struct.unpack_from("<I", out, body + 4)[0], 0x10)

    def test_unparseable_frames_raise(self):
        from perf.opt import _patch_eh_frame

        with self.assertRaises(ValueError):
            _patch_eh_frame(_cie(aug=b"z") + _fde(0), {})
        with self.assertRaises(ValueError):
            _patch_eh_frame(_cie(enc=0x09) + _fde(0), {})

    def test_datarel_fde_uses_the_section_base(self):
        import struct

        from perf.opt import _patch_eh_frame

        sec = 0x1000
        cie = _cie(enc=0x33)
        fde = _fde(len(cie), pc=0x10, span=0x20)
        out, updated = _patch_eh_frame(
            cie + fde,
            {(0x1010, 0x1030): (0x2010, 0x2030)},
            section_addr=sec,
        )
        self.assertEqual(updated, 1)
        body = len(cie) + 8
        self.assertEqual(struct.unpack_from("<I", out, body)[0], 0x1010)
        self.assertEqual(struct.unpack_from("<I", out, body + 4)[0], 0x20)


class TestEhFrameHdrPatching(unittest.TestCase):
    def _hdr(self, pc=0x1000, fde=0x2000, count=1, table=0x03, ptr=0x03, cnt=0x03):
        import struct

        return (
            bytes([1, ptr, cnt, table])
            + struct.pack("<I", 0x100)
            + struct.pack("<I", count)
            + struct.pack("<I", pc)
            + struct.pack("<I", fde)
        )

    def test_header_entry_is_remapped_and_sorted(self):
        import struct

        from perf.opt import _patch_eh_frame_hdr

        out, updated = _patch_eh_frame_hdr(
            self._hdr(), {(0x1000, 0x1020): (0x3000, 0x3020)}
        )
        self.assertEqual(updated, 1)
        self.assertEqual(struct.unpack_from("<I", out, 12)[0], 0x3000)
        self.assertEqual(struct.unpack_from("<I", out, 16)[0], 0x2000)

    def test_header_keeps_entries_past_the_stripped_tail(self):
        import struct

        from perf.opt import _patch_eh_frame_hdr

        out, updated = _patch_eh_frame_hdr(
            self._hdr(pc=0x1015), {(0x1000, 0x1020): (0x3000, 0x3010)}
        )
        self.assertEqual(updated, 0)
        self.assertEqual(struct.unpack_from("<I", out, 12)[0], 0x1015)

    def test_header_without_a_table_is_unchanged(self):
        from perf.opt import _patch_eh_frame_hdr

        data = self._hdr(table=0xFF)
        self.assertEqual(_patch_eh_frame_hdr(data, {}), (data, 0))
        data = self._hdr(count=0)
        self.assertEqual(_patch_eh_frame_hdr(data, {}), (data, 0))
        data = b"\x02" + self._hdr()[1:]
        self.assertEqual(_patch_eh_frame_hdr(data, {}), (data, 0))
        self.assertEqual(_patch_eh_frame_hdr(b"\x01", {}), (b"\x01", 0))

    def test_header_decode_and_encode_errors(self):
        from perf.opt import _patch_eh_frame_hdr

        with self.assertRaises(ValueError):
            _patch_eh_frame_hdr(self._hdr(ptr=0xFF), {})
        with self.assertRaises(ValueError):
            _patch_eh_frame_hdr(self._hdr(cnt=0xFF), {})
        with self.assertRaises(ValueError):
            _patch_eh_frame_hdr(self._hdr(table=0x09), {})
        with self.assertRaises(ValueError):
            _patch_eh_frame_hdr(self._hdr(table=0x05), {})
        with self.assertRaises(ValueError):
            _patch_eh_frame_hdr(self._hdr()[:-4], {})
        with self.assertRaises(ValueError):
            _patch_eh_frame_hdr(
                self._hdr(table=0x02),
                {(0x1000, 0x1020): (0x10000, 0x10020)},
            )

    def test_header_write_errors(self):
        from perf.opt import _patch_eh_frame_hdr

        with self.assertRaises(ValueError):
            _patch_eh_frame_hdr(
                self._hdr(),
                {(0x1000, 0x1020): (0x1_0000_0000, 0x1_0000_0020)},
            )


class TestRemapHelpers(unittest.TestCase):
    def test_remap_factory(self):
        from perf.opt import _remap_factory

        mapping = {(0x1000, 0x1010): (0x2000, 0x2010)}
        remap = _remap_factory(mapping)
        self.assertEqual(remap(0x1008), 0x2008)
        self.assertIsNone(remap(0x1100))

    def test_remap_factory_skips_the_stripped_tail(self):
        from perf.opt import _remap_factory

        remap = _remap_factory({(0x1000, 0x1020): (0x2000, 0x2010)})
        self.assertEqual(remap(0x1005), 0x2005)
        self.assertIsNone(remap(0x1015))
        self.assertIsNone(remap(0x1030))


class TestRelrHelpers(unittest.TestCase):
    def test_relr_decodes_direct_and_bitmap_entries(self):
        import struct

        from perf.opt import _relr_relocations

        data = b"".join(struct.pack("<Q", entry) for entry in (0x3D70, 0xF, 0x100001))
        self.assertEqual(
            _relr_relocations(data),
            [0x3D70, 0x3D78, 0x3D80, 0x3D88, 0x4008],
        )
        self.assertEqual(_relr_relocations(struct.pack("<Q", 0x4000)), [0x4000])

    def test_relr_chained_bitmaps_advance_by_63_slots(self):
        import struct

        from perf.opt import _relr_relocations

        data = b"".join(struct.pack("<Q", entry) for entry in (0x1000, 0x3, 0x3))
        self.assertEqual(
            _relr_relocations(data),
            [0x1000, 0x1008, 0x1008 + 63 * 8],
        )

    def test_relr_rejects_truncated_or_baseless_data(self):
        import struct

        from perf.opt import _relr_relocations

        with self.assertRaises(ValueError):
            _relr_relocations(b"\x00" * 4)
        with self.assertRaises(ValueError):
            _relr_relocations(struct.pack("<Q", 0x1))

    def test_section_file_offset_maps_allocated_ranges(self):
        from perf.opt import _section_file_offset

        sections = [(0x1000, 0x200, 0x40), (0x2000, 0x400, 0x10)]
        self.assertEqual(_section_file_offset(sections, 0x1008), 0x208)
        self.assertEqual(_section_file_offset(sections, 0x200F), 0x40F)
        self.assertIsNone(_section_file_offset(sections, 0x3000))
        self.assertIsNone(_section_file_offset(sections, None))

    def test_layout_hazard_detects_only_text_addresses(self):
        from perf.opt import _layout_hazard

        blob = _asm_bytes("movabs rax, 0x1000", 0x1000)
        self.assertTrue(_layout_hazard(blob, 0x1000, 0x1000, 0x2000))
        self.assertFalse(_layout_hazard(blob, 0x1000, 0x2000, 0x3000))
        self.assertTrue(_layout_hazard(b"\x0f", 0x1000, 0x1000, 0x2000))
        with patch("perf.opt.get_arch", side_effect=RuntimeError("no arch")):
            self.assertTrue(_layout_hazard(blob, 0x1000, 0x1000, 0x2000))


class TestOptimizeGuards(unittest.TestCase):
    def test_input_validation(self):
        from perf.opt import optimize

        with self.assertRaises(ValueError):
            optimize()
        with self.assertRaises(ValueError):
            optimize(exec=["a.out"], output="b.out")
        with self.assertRaises(ValueError):
            optimize(exec="  ", output="b.out")
        with self.assertRaises(ValueError):
            optimize(exec="/nonexistent/perf/input", output="b.out")

    def test_default_output_and_must_differ(self):
        from perf.opt import default_output, optimize

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.out"
            path.write_bytes(b"\x7fELF")
            self.assertEqual(default_output(str(path)), str(Path(tmp) / "a.opt.out"))
            self.assertEqual(
                default_output(str(Path(tmp) / "tool")),
                str(Path(tmp) / "tool.opt"),
            )
            with self.assertRaises(ValueError):
                optimize(exec=str(path), output=str(path))

    def test_no_functions_is_an_error(self):
        from perf.opt import optimize

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.out"
            path.write_bytes(b"\x7fELF" + b"\x00" * 64)
            with (
                patch(
                    "perf.opt._text_section",
                    return_value={"addr": 0x1000, "offset": 0, "size": 0x10},
                ),
                patch("perf.opt._is_pie_elf", return_value=False),
                patch("perf.opt.discover_functions", return_value={}),
            ):
                with self.assertRaises(ValueError):
                    optimize(exec=str(path), output=str(Path(tmp) / "b.out"))


class TestOptimizeMocked(unittest.TestCase):
    TEXT = {"addr": 0x1000, "offset": 0x10, "size": 0x40}
    FUNCS = {"a": (0x1000, 0x1008), "b": (0x1008, 0x1010)}

    def _run(
        self,
        tmp,
        config=None,
        opt=None,
        dry_run=True,
        pie=False,
        output="b.out",
        **overrides,
    ):
        from perf.opt import optimize

        path = Path(tmp) / "a.out"
        path.write_bytes(b"\x7fELF" + b"\x00" * 0x0F + b"\x90" * 0x40)
        output = None if output is None else str(Path(tmp) / output)
        patches = {
            "_text_section": Mock(return_value=dict(self.TEXT)),
            "_is_pie_elf": Mock(return_value=pie),
            "_has_exception_tables": Mock(return_value=False),
            "_has_fallthrough": Mock(return_value=False),
            "_text_coverage": Mock(return_value=1.0),
            "_has_jump_table": Mock(return_value=False),
            "_strip_trailing_nops": Mock(side_effect=lambda blob: (bytes(blob), 0)),
            "_branch_targets": Mock(return_value=set()),
            "_verify_text": Mock(),
            "_fixup_layout_image": Mock(return_value=b"\x7fELF" + b"\x00" * 0x4F),
        }
        patches.update(overrides)
        stack = [patch("perf.opt.discover_functions", return_value=dict(self.FUNCS))]
        stack += [patch(f"perf.opt.{name}", value) for name, value in patches.items()]
        for ctx in stack:
            ctx.start()
            self.addCleanup(ctx.stop)
        return optimize(
            exec=str(path),
            output=output,
            data=None,
            config=config,
            opt=opt,
            dry_run=dry_run,
        )

    def test_layout_applied_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(tmp, config={"opt": ["layout", "align"]}, pie=True)
        self.assertTrue(df.attrs["layout"])
        self.assertEqual(df.attrs["opt"], ["layout", "align"])
        self.assertEqual(set(df["name"]), {"a", "b"})
        self.assertTrue(df["moved"].any())

    def test_non_pie_layout_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(tmp, config={"opt": ["layout"]}, pie=False)
        self.assertFalse(df.attrs["layout"])
        self.assertIn("non-PIE", df.attrs["layout_skipped"])

    def test_exception_tables_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(
                tmp,
                config={"opt": ["layout"]},
                pie=True,
                _has_exception_tables=Mock(return_value=True),
            )
        self.assertIn("exception tables", df.attrs["layout_skipped"])

    def test_fallthrough_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(
                tmp,
                config={"opt": ["layout"]},
                pie=True,
                _has_fallthrough=Mock(return_value=True),
            )
        self.assertIn("fall through", df.attrs["layout_skipped"])

    def test_coverage_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(
                tmp,
                config={"opt": ["layout"]},
                pie=True,
                _text_coverage=Mock(return_value=0.5),
            )
        self.assertIn("does not cover", df.attrs["layout_skipped"])

    def test_jump_table_pins_functions(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(
                tmp,
                config={"opt": ["layout"]},
                pie=True,
                _has_jump_table=Mock(return_value=True),
            )
        self.assertIn("indirect jumps", df.attrs["layout_skipped"])

    def test_absolute_text_addresses_pin_functions(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(
                tmp,
                config={"opt": ["layout"]},
                pie=True,
                _layout_hazard=Mock(return_value=True),
            )
        self.assertIn("absolute addresses", df.attrs["layout_skipped"])
        self.assertFalse(df.attrs["layout"])

    def test_aligned_layout_that_does_not_fit_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(
                tmp,
                config={"layout": {"align": 4096}},
                opt=["layout", "align"],
                pie=True,
                collect_hotness=Mock(return_value=({"a": 1.0, "b": 1.0}, {})),
            )
        self.assertIn("does not fit", df.attrs["layout_skipped"])

    def test_as_is_order_and_hot_first_ordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(
                tmp,
                config={"opt": ["layout"], "layout": {"order": "as-is"}},
                pie=True,
            )
        self.assertTrue(df.attrs["layout"])
        self.assertEqual(list(df.sort_values("address")["name"]), ["a", "b"])

    def test_peephole_and_branch_stats_are_summed(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(
                tmp,
                opt=["peephole", "branch"],
                pie=False,
                _peephole_function=Mock(return_value=(b"\x90" * 8, {"mov_self": 2})),
                _branch_function=Mock(return_value=(b"\x90" * 8, {"jmp_next": 1})),
            )
        self.assertEqual(df.attrs["peephole"]["mov_self"], 4)
        self.assertEqual(df.attrs["branch"]["jmp_next"], 2)

    def test_restore_original_resets_peephole_and_branch_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(
                tmp,
                opt=["peephole", "branch"],
                pie=False,
                _verify_text=Mock(side_effect=RuntimeError("bad")),
                _peephole_function=Mock(return_value=(b"\x90" * 8, {"mov_self": 2})),
                _branch_function=Mock(return_value=(b"\x90" * 8, {"jmp_next": 1})),
            )
        self.assertEqual(set(df.attrs["peephole"].values()), {0})
        self.assertEqual(set(df.attrs["branch"].values()), {0})

    def test_default_output_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(tmp, opt=["peephole"], pie=False, output=None)
        self.assertEqual(df.attrs["output"], str(Path(tmp) / "a.opt.out"))

    def test_layout_fixup_failure_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(
                tmp,
                config={"opt": ["peephole", "layout"]},
                pie=True,
                _fixup_layout_image=Mock(
                    side_effect=ValueError("short branch out of range")
                ),
            )
        self.assertFalse(df.attrs["layout"])
        self.assertIn("fixups failed", df.attrs["layout_skipped"])
        self.assertIn("safe fallback", df.attrs["layout_skipped"])
        self.assertEqual(set(df["name"]), {"a", "b"})

    def test_identity_verification_failure_restores_the_original_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.out"
            path.write_bytes(b"\x7fELF" + b"\x00" * 0x0F + b"\x90" * 0x40)
            before = path.read_bytes()
            df = self._run(
                tmp,
                opt=["peephole"],
                pie=False,
                dry_run=False,
                _verify_text=Mock(side_effect=ValueError("bad text")),
            )
            self.assertEqual(Path(df.attrs["output"]).read_bytes(), before)
            self.assertFalse(df.attrs["layout"])

    def test_writes_output_and_marks_it_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self._run(tmp, opt=["peephole"], pie=False, dry_run=False)
            out = Path(df.attrs["output"])
            self.assertTrue(out.exists())
            self.assertTrue(out.stat().st_mode & 0o111)


class TestReportErrors(unittest.TestCase):
    def test_report_json_survives_bad_frames(self):
        from perf.opt import report_json

        df = Mock()
        df.attrs = {}
        df.to_dict.side_effect = RuntimeError("bad")
        self.assertEqual(report_json(df)["functions"], [])

    def test_describe_survives_bad_frames(self):
        from perf.opt import describe

        df = Mock()
        df.attrs = Mock()
        df.attrs.get.side_effect = RuntimeError("bad")
        df.to_dict.side_effect = RuntimeError("bad")
        self.assertEqual(describe(df), "")

    def test_describe_skips_malformed_rows(self):
        from perf.opt import describe

        df = pd.DataFrame(
            [{"name": "bad", "address": "x", "new_address": 0x1000, "size": 1}]
        )
        self.assertEqual(describe(df), "opt: ")


class TestOptimizeOptions(unittest.TestCase):
    def test_layout_alignment_applies_without_the_align_pass(self):
        from perf.opt import plan

        selected, align, _order = plan({"opt": ["layout"], "layout": {"align": 64}})
        self.assertEqual(selected, ["layout"])
        self.assertEqual(align, 64)

    def test_align_rejects_bools_and_fractional_values(self):
        from perf.opt import merge_config

        for value in (True, 16.5, float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                merge_config({"layout": {"align": value}})

    def test_collect_hotness_accepts_a_dataframe(self):
        from perf.opt import collect_hotness

        df = pd.DataFrame({"name": ["func"], "cycles": [10.0], "operations": [1]})
        scores, details = collect_hotness(df)
        self.assertEqual(scores, {"func": 1.0})
        self.assertIn("func", details)

    def test_describe_skips_only_the_bad_row(self):
        from perf.opt import describe

        df = pd.DataFrame(
            [
                {"name": "bad", "address": "x", "new_address": 0x1000, "size": 1},
                {
                    "name": "good",
                    "address": 0x1000,
                    "new_address": 0x1000,
                    "size": 1,
                    "moved": False,
                    "hotness": 0.0,
                },
            ]
        )
        self.assertIn("good", describe(df))
        self.assertNotIn("bad", describe(df))


class TestBugFixes(unittest.TestCase):
    def test_lea_segment_override_is_not_removed(self):
        for asm in ("lea rax, fs:[rax]", "lea rax, gs:[rax]"):
            blob = _asm_bytes(asm, 0x1000)
            new, stats = _peephole_function(blob, 0x1000)
            self.assertEqual(stats["lea_self"], 0, asm)
            self.assertEqual(new, blob, asm)

    def test_lea_plain_self_is_still_removed(self):
        blob = _asm_bytes("lea rax, [rax]", 0x1000)
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["lea_self"], 1)
        self.assertNotEqual(new, blob)

    def test_peephole_auto_protects_in_blob_jump_target(self):
        blob = _asm_bytes("movabs rax, 0x1234", 0x1000) + _asm_bytes(
            "jmp 0x1004", 0x100A
        )
        new, stats = _peephole_function(blob, 0x1000)
        self.assertEqual(stats["movabs"], 0)
        self.assertEqual(new, blob)
        guarded, guarded_stats = _peephole_function(blob, 0x1000, protected={0x1004})
        self.assertEqual(guarded, blob)
        self.assertEqual(guarded_stats["movabs"], 0)

    def test_has_fallthrough_empty_blob_is_conservative(self):
        from perf.opt import _has_fallthrough

        self.assertTrue(_has_fallthrough({"f": (0x1000, 0x1001)}, {"f": b""}))
        self.assertTrue(_has_fallthrough({"f": (0x1000, 0x1001)}, {}))

    def test_text_coverage_rejects_short_text(self):
        from perf.opt import _text_coverage

        funcs = {"a": (0x1000, 0x1010)}
        full = _text_coverage(funcs, 0x1000, 0x1020, text=b"\x90" * 0x20)
        longer = _text_coverage(funcs, 0x1000, 0x1020, text=b"\x90" * 0x24)
        short = _text_coverage(funcs, 0x1000, 0x1020, text=b"\x90" * 2)
        self.assertEqual(full, 1.0)
        self.assertEqual(longer, 1.0)
        self.assertEqual(short, 0.5)

    def test_leb_truncated_raises(self):
        from perf.opt import _read_sleb, _read_uleb

        with self.assertRaises(ValueError):
            _read_uleb(b"\x80", 0)
        with self.assertRaises(ValueError):
            _read_sleb(b"\x80", 0)
        with self.assertRaises(ValueError):
            _read_uleb(b"", 0)

        from perf.opt import _read_encoded

        self.assertIsNone(_read_encoded(b"\x80", 0, 0x09))

    def test_read_encoded_rejects_textrel_funcrel(self):
        import struct

        from perf.opt import _read_encoded

        data = struct.pack("<I", 0x1000)
        self.assertIsNone(_read_encoded(data, 0, 0x23))
        self.assertIsNone(_read_encoded(data, 0, 0x43))
        self.assertIsNone(_read_encoded(data, 0, 0x53))

    def test_hotness_ignores_non_list_output(self):
        self.assertEqual(_hotness_from_envelope({"output": {"a": 1}}), {})
        self.assertEqual(_hotness_from_envelope({"output": "junk"}), {})

    def test_hotness_nan_operations_do_not_poison(self):
        payload = {
            "name": "t",
            "output": [
                {
                    "name": "f",
                    "iteration": 0,
                    "operations": float("nan"),
                    "cycles": 100.0,
                },
                {"name": "g", "iteration": 0, "operations": 1, "cycles": 10.0},
            ],
        }
        scores = _hotness_from_envelope(payload)
        self.assertGreater(scores.get("f", 0), 0)
        self.assertGreater(scores.get("g", 0), 0)

    def test_collect_hotness_skips_inf_and_negative(self):
        scores, _ = collect_hotness(
            [
                pd.DataFrame(
                    [{"mode": "latency", "operations": 1, "cycles": float("inf")}],
                )
            ]
        )
        self.assertEqual(scores, {})
        self.assertEqual(_match_hotness("f", {"f": float("nan")}), 0.0)
        self.assertEqual(_match_hotness("f", {"f": float("inf")}), 0.0)
        self.assertEqual(_match_hotness("f", {"f": -5.0}), 0.0)
        scores, _ = collect_hotness(
            [
                pd.DataFrame(
                    [
                        {"name": "good", "operations": 1, "cycles": 10.0},
                        {"name": "bad", "operations": 1, "cycles": float("inf")},
                    ]
                )
            ]
        )
        self.assertEqual(scores, {"good": 1.0})

    def test_discover_prefers_symtab_over_angr(self):
        from perf.opt import discover_functions

        text = {"addr": 0x1000, "offset": 0, "size": 0x100}
        sym = {"f": (0x1000, 0x1010)}
        angr = {"f": (0x1000, 0x1020), "extra": (0x1020, 0x1030)}
        with (
            patch("perf.opt._text_section", return_value=text),
            patch("perf.opt._symtab_functions", return_value=sym),
            patch("perf.opt._has_symtab", return_value=False),
            patch("perf.opt._angr_functions", return_value=angr),
        ):
            out = discover_functions("/tmp/a.out")
        self.assertEqual(out["f"], (0x1000, 0x1010))

    def test_eh_frame_hdr_rejects_huge_count(self):
        import struct

        from perf.opt import _patch_eh_frame_hdr

        hdr = bytearray(bytes([1, 0x00, 0x03, 0x03]))
        hdr += struct.pack("<Q", 0x1000)
        hdr += struct.pack("<I", 0xFFFFFFFF)
        hdr += b"\x00" * 16
        with self.assertRaises(ValueError):
            _patch_eh_frame_hdr(bytes(hdr), {(0x1000, 0x1010): (0x1000, 0x1010)})

    def test_eh_frame_hdr_unmapped_entries_are_not_counted(self):
        import struct

        from perf.opt import _patch_eh_frame_hdr

        hdr = bytearray(b"\x01\x00\x03\x03")
        hdr += struct.pack("<Q", 0x5000)
        hdr += struct.pack("<I", 1)
        hdr += struct.pack("<I", 0x2000) + struct.pack("<I", 0x10)
        hdr += struct.pack("<I", 0x3000)
        out, updated = _patch_eh_frame_hdr(
            bytes(hdr), {(0x1000, 0x1010): (0x2000, 0x2010)}
        )
        self.assertEqual(updated, 0)
        self.assertEqual(out, bytes(hdr))

    def test_eh_frame_hdr_keeps_unmapped_entries_when_remapping(self):
        import struct

        from perf.opt import _patch_eh_frame_hdr

        hdr = bytearray(b"\x01\x00\x03\x03")
        hdr += struct.pack("<Q", 0x5000)
        hdr += struct.pack("<I", 3)
        hdr += struct.pack("<I", 0x1000) + struct.pack("<I", 0x7000)
        hdr += struct.pack("<I", 0x1050) + struct.pack("<I", 0x7008)
        hdr += struct.pack("<I", 0x1080) + struct.pack("<I", 0x7010)
        out, updated = _patch_eh_frame_hdr(
            bytes(hdr),
            {
                (0x1000, 0x1010): (0x2000, 0x2010),
                (0x1080, 0x1090): (0x2100, 0x2110),
            },
        )
        self.assertEqual(updated, 2)
        self.assertEqual(
            struct.unpack_from("<IIIIII", out, 16),
            (0x1050, 0x7008, 0x2000, 0x7000, 0x2100, 0x7010),
        )

    def test_format_report_moved_flag(self):
        mod = cli
        df = pd.DataFrame(
            [
                {
                    "name": "a",
                    "address": 0x1000,
                    "new_address": 0x1000,
                    "size": 8,
                    "hotness": 1.0,
                    "moved": True,
                },
                {
                    "name": "b",
                    "address": 0x2000,
                    "new_address": 0x2000,
                    "size": 8,
                    "hotness": 1.0,
                    "moved": float("nan"),
                },
                {
                    "name": "c",
                    "address": 0x3000,
                    "new_address": 0x3000,
                    "size": 8,
                    "hotness": 1.0,
                    "moved": "false",
                },
                {
                    "name": "d",
                    "address": 0x4000,
                    "new_address": 0x4000,
                    "size": 8,
                    "hotness": 1.0,
                    "moved": pd.NA,
                },
                {
                    "name": "e",
                    "address": -1,
                    "new_address": 0x5000,
                    "size": 8,
                    "hotness": 1.0,
                    "moved": False,
                },
            ]
        )
        text = mod.format_report(df)
        self.assertIn("moved", text)
        lines = [line for line in text.splitlines() if line.strip()]
        data_lines = lines[1:]
        moved_lines = [line for line in data_lines if "moved" in line]
        self.assertEqual(len(moved_lines), 1)
        self.assertTrue(moved_lines[0].strip().startswith("a"))

    def test_format_report_blanks_nonfinite_numbers(self):
        df = pd.DataFrame(
            [
                {
                    "name": "a",
                    "address": float("inf"),
                    "new_address": float("-inf"),
                    "size": 8,
                    "hotness": float("inf"),
                    "moved": False,
                }
            ]
        )
        text = cli.format_report(df)
        self.assertNotIn("inf", text)
        self.assertNotIn("nan", text)


class TestOptRegression(unittest.TestCase):
    def test_text_section_rejects_non_elf_as_value_error(self):
        from perf.opt import _text_section

        with tempfile.NamedTemporaryFile(suffix=".s", delete=False, mode="w") as fh:
            fh.write("not an elf")
            path = fh.name
        self.addCleanup(os.unlink, path)
        with self.assertRaises(ValueError):
            _text_section(path)

    def test_optimize_rejects_asm_source_as_value_error(self):
        import perf

        with tempfile.NamedTemporaryFile(suffix=".s", delete=False, mode="w") as fh:
            fh.write("int main(void){return 0;}")
            path = fh.name
        self.addCleanup(os.unlink, path)
        with self.assertRaises(ValueError):
            perf.optimize(exec=path, output=path + ".out", dry_run=True)

    def test_collect_hotness_dedups_same_dataframe(self):
        df = pd.DataFrame(
            [{"mode": "latency", "operations": 1, "cycles": 42.0}],
        )
        df.attrs["name"] = "myfunc-12345678"
        df.attrs["file"] = "a.out@abc"
        once, _ = collect_hotness([df])
        twice, _ = collect_hotness([df, df])
        self.assertEqual(once, twice)


class TestOptCorrectnessRegression(unittest.TestCase):
    def test_leb_truncated_raises(self):
        from perf.opt import _read_sleb, _read_uleb

        with self.assertRaises(ValueError):
            _read_uleb(b"\x80", 0)
        with self.assertRaises(ValueError):
            _read_uleb(b"", 0)
        with self.assertRaises(ValueError):
            _read_sleb(b"\x80", 0)

    def test_branch_targets_include_absolute_refs(self):
        from perf.arch import x86_64
        from perf.opt import _branch_targets

        blob = x86_64.assemble("mov eax, 0x12345678; ret", 0x1000)
        targets = _branch_targets({"f": (0x1000, 0x1000 + len(blob))}, {"f": blob})
        self.assertIn(0x12345678, targets)

    def test_read_encoded_truncated_returns_none(self):
        from perf.opt import _read_encoded

        self.assertIsNone(_read_encoded(b"\x80", 0, 0x09))
        self.assertIsNone(_read_encoded(b"", 0, 0x09))


if __name__ == "__main__":
    unittest.main()
