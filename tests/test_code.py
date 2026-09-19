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
import io
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
from common import cli as _shared
from perf.arch import arch as get_arch
from perf.code import _instructions

BRANCHY = """
#include <string.h>
long __attribute__((noinline)) pick(long x) {
    char buf[8];
    if (x == 0) { memset(buf, 1, 8); return buf[3]; }
    if (x == 1) { memset(buf, 2, 8); return buf[3]; }
    return x * 3;
}
int main(int argc, char**argv){ return (int)pick(argc); }
"""

SOURCE = """
long myfunc(long x) {
  long sum = 0;
  for (int i = 0; i < 4; i++) {
    if (x > 2) sum += x * i;
    else sum -= i;
  }
  return sum;
}
long other(long y) { return y + 1; }
int main() { return (int)myfunc(3); }
"""


def _cli():
    path = Path(__file__).resolve().parent.parent / "bin" / "perf-analyze"
    loader = importlib.machinery.SourceFileLoader("perfcli_analyze", str(path))
    spec = importlib.util.spec_from_loader("perfcli_analyze", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


cli = _cli()


def _insn_columns(df):
    return [c for c in df.columns if not str(c).startswith("data.")]


def _args(**kw):
    code = kw.pop("code", None)
    file = kw.pop("file", None)
    target = kw.pop("target", None)
    if code is None:
        code = f"{file}:{target}" if file and target else (file or target)
    base = dict(
        code=code,
        analyze_name=kw.get("name", None),
        config=kw.get("config", None),
        data=kw.get("data", None),
        event=kw.get("event", None),
        data_paths=kw.get("data_paths", None),
        json=kw.get("json", False),
        interactive=False,
        emit_asm=kw.get("emit_asm", False),
        setup=kw.get("setup", None),
        teardown=kw.get("teardown", None),
        filter=kw.get("filter", None),
    )
    return SimpleNamespace(**base)


def _run(args):
    buf = io.StringIO()
    paths = getattr(args, "data_paths", None) or []
    with patch("sys.stdout", buf):
        cli.main(args, paths)
    return buf.getvalue()


class TestInstructions(unittest.TestCase):
    def _code(self, asm, base=0x1000000):
        arch = get_arch()
        return arch, arch.assemble(asm, base)

    def test_follows_branch_over_undecodable_bytes(self):
        arch, blob = self._code(
            "jmp 0x1000007; .byte 0xff, 0xff, 0xff, 0xff, 0xff; nop"
        )
        followed = [i.address for i in _instructions(arch, blob, 0x1000000)]
        linear = [i.address for i in arch.disassembler().disasm(blob, 0x1000000)]
        self.assertEqual(followed, [0x1000000, 0x1000007])
        self.assertEqual(linear, [0x1000000])

    def test_backward_branch_terminates(self):
        arch, blob = self._code("nop; jmp 0x1000000")
        got = [i.address for i in _instructions(arch, blob, 0x1000000)]
        self.assertEqual(got, [0x1000000, 0x1000001])

    def test_out_of_range_branch_is_not_followed(self):
        arch, blob = self._code("jmp 0x2000000; .byte 0xff, 0xff")
        got = [i.address for i in _instructions(arch, blob, 0x1000000)]
        self.assertEqual(got, [0x1000000])

    def test_sweep_continues_after_branch(self):
        arch, blob = self._code("jmp 0x2000000; nop")
        got = [i.address for i in _instructions(arch, blob, 0x1000000)]
        self.assertEqual(got, [0x1000000, 0x1000005])

    def test_branch_imm_targets_only_for_branches(self):
        arch, blob = self._code("mov rax, 0x1000000; jmp 0x1000011; nop")
        insns = list(_instructions(arch, blob, 0x1000000))
        self.assertEqual(arch.branch_imm_targets(insns[0]), ())
        self.assertEqual(arch.branch_imm_targets(insns[1]), [0x1000011])

    def test_memory_operand_is_not_a_branch_target(self):
        arch, blob = self._code("mov rax, [0x1000000]; ret")
        insns = list(_instructions(arch, blob, 0x1000000))
        self.assertEqual(arch.branch_imm_targets(insns[0]), ())


class TestAsm(unittest.TestCase):
    def test_columns_and_rows(self):
        df = perf.analyze(code="mov eax, 42; add eax, ebx")
        self.assertEqual(
            _insn_columns(df),
            [
                "file",
                "name",
                "index",
                "address",
                "encoding",
                "size",
                "latency",
                "throughput",
                "assembly",
            ],
        )
        self.assertEqual(df["index"].tolist(), [0, 1])
        self.assertEqual(len(df), 2)
        self.assertNotIn("mode", df.columns)
        self.assertEqual(df["assembly"].tolist(), ["mov eax, 0x2a", "add eax, ebx"])
        self.assertEqual(df["encoding"].tolist(), ["b8 2a 00 00 00", "01 d8"])
        self.assertEqual(df["size"].tolist(), [5, 2])
        self.assertEqual(df["address"].tolist(), [0x1000000, 0x1000005])

    def test_attrs(self):
        df = perf.analyze(code="mov eax, 42", name="snip")
        self.assertEqual(df.attrs["name"], "snip")
        self.assertEqual(df.attrs["code"], "mov eax, 0x2a;")
        self.assertIsNone(df.attrs["file"])
        self.assertNotIn("data", df.attrs)

    def test_osaca_latency_and_throughput(self):
        df = perf.analyze(code="mov eax, 42; imul ecx, edx; nop")
        self.assertEqual(df["latency"].tolist()[0:2], [1.0, 3.0])
        self.assertEqual(df["throughput"].tolist()[0:2], [0.2, 1.0])

    def test_osaca_reports_unknown_forms_as_nan(self):
        df = perf.analyze(code="nop")
        self.assertTrue(df["latency"].isna().all())
        self.assertTrue(df["throughput"].isna().all())

    def test_osaca_columns_follow_the_instructions(self):
        df = perf.analyze(code="mov eax, 42", event=["assembly,latency"])
        self.assertEqual(_insn_columns(df)[-1], "latency")

    def test_name_defaults_to_normalized_code(self):
        df = perf.analyze(code="mov eax, 42")
        self.assertEqual(df["name"].unique().tolist(), ["mov eax, 0x2a;"])

    def test_branch_is_labelled(self):
        df = perf.analyze(code="mov eax, 1; jl 0x1000007; mov ebx, 2")
        self.assertIn("jl .L1000007", df["assembly"].tolist())

    def test_state_columns_hold_the_explored_state(self):
        df = perf.analyze(code="mov rax, rdi; add rax, 1; ret")
        self.assertEqual(df["assembly"].tolist(), ["mov rax, rdi", "add rax, 1", "ret"])
        self.assertFalse(pd.isna(df["data.rdi"].iloc[0]))
        self.assertNotEqual(df["data.rax"].iloc[0], df["data.rdi"].iloc[0])
        self.assertTrue(pd.isna(df["data.rax"].iloc[2]))

    def test_state_columns_cover_every_state(self):
        df = perf.analyze(code="mov rax, rdi; add rax, 1; ret")
        self.assertEqual(len(df), 3)
        self.assertEqual(df["index"].tolist(), [0, 1, 2])

    def test_harness_registers_are_not_state(self):
        df = perf.analyze(code="mov rax, rdi; add rax, 1; ret")
        self.assertNotIn("data.rsp", df.columns)
        self.assertNotIn("data.rbp", df.columns)
        self.assertNotIn("data.rip", df.columns)

    def test_filter_keeps_matching_instructions(self):
        df = perf.analyze(code="mov rax, rdi; add rax, 1; ret", filter="size > 2")
        self.assertEqual(df["assembly"].tolist(), ["mov rax, rdi", "add rax, 1"])

    def test_filter_uses_state_columns(self):
        df = perf.analyze(code="mov rax, rdi; add rax, 1; ret")
        keep = int(df["data.rdi"].iloc[0])
        out = perf.analyze(
            code="mov rax, rdi; add rax, 1; ret", filter=f"`data.rdi` == {keep}"
        )
        self.assertEqual(out["assembly"].tolist(), ["mov rax, rdi"])

    def test_filter_accepts_plain_columns(self):
        df = perf.analyze(code="mov rax, rdi; add rax, 1; ret", filter="size <= 3")
        self.assertEqual(df["assembly"].tolist(), ["mov rax, rdi", "ret"])

    def test_bad_filter(self):
        with self.assertRaises(ValueError) as ctx:
            perf.analyze(code="mov eax, 1", filter="nope == 1")
        self.assertIn("invalid filter", str(ctx.exception))

    def test_event_selects_columns(self):
        df = perf.analyze(code="mov eax, 42; ret", event=["assembly", "encoding"])
        self.assertEqual(
            _insn_columns(df),
            ["file", "name", "index", "address", "assembly", "encoding"],
        )
        self.assertEqual([c for c in df.columns if c.startswith("data.")], ["data.rax"])

    def test_event_keeps_data_columns(self):
        df = perf.analyze(code="mov rax, rdi", event=["assembly", "encoding"])
        self.assertEqual(
            _insn_columns(df),
            [
                "file",
                "name",
                "index",
                "address",
                "assembly",
                "encoding",
            ],
        )
        self.assertEqual(
            [c for c in df.columns if c.startswith("data.")], ["data.rax", "data.rdi"]
        )

    def test_event_is_comma_separated(self):
        df = perf.analyze(code="mov eax, 42; ret", event="assembly,encoding")
        self.assertEqual(_insn_columns(df)[-2:], ["assembly", "encoding"])

    def test_unknown_event(self):
        with self.assertRaises(ValueError) as ctx:
            perf.analyze(code="mov eax, 42", event=["nope"])
        self.assertIn("unknown columns: nope", str(ctx.exception))

    def test_bad_asm(self):
        with self.assertRaises(ValueError):
            perf.analyze(code="not_an_instruction")

    def test_requires_code(self):
        with self.assertRaises(TypeError):
            perf.analyze()

    def test_json_envelope(self):
        df = perf.analyze(code="mov eax, 42")
        payload = json.loads(perf.to_json(df))
        self.assertIn("output", payload)
        row = payload["output"][0]
        self.assertEqual(row["assembly"], "mov eax, 0x2a")
        self.assertIn("rax", row["data"])


def _build(tmp, source, name="prog", *flags):
    src = os.path.join(tmp, f"{name}.c")
    exe = os.path.join(tmp, name)
    Path(src).write_text(source)
    if shutil.which("gcc") is None:
        return None
    r = subprocess.run(
        ["gcc", "-O2", "-fno-inline", *flags, "-o", exe, src], capture_output=True
    )
    return exe if r.returncode == 0 and os.path.exists(exe) else None


class TestBinary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp()
        cls.exe = _build(cls._tmp, SOURCE)
        cls.branchy = _build(cls._tmp, BRANCHY, "branchy")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def setUp(self):
        if self.exe is None:
            self.skipTest("gcc unavailable")

    def _df(self, target="myfunc", **kw):
        if isinstance(target, (list, tuple)):
            code = [self.exe, tuple(target)]
        else:
            code = f"{self.exe}:{target}"
        return perf.analyze(code, **kw)

    def test_identity_and_instructions(self):
        df = self._df()
        self.assertEqual(df["name"].unique().tolist(), ["myfunc"])
        self.assertEqual(
            df["file"].unique().tolist(), [f"prog@{df.attrs['file'].split('@')[1]}"]
        )
        self.assertTrue(df.attrs["file"].startswith("prog@"))
        self.assertGreater(len(df), 5)
        self.assertEqual(df["address"].tolist(), sorted(df["address"].tolist()))
        self.assertTrue(all(df["assembly"].str.len() > 0))
        self.assertTrue(
            all(len(e.split()) == df["size"][i] for i, e in enumerate(df["encoding"]))
        )

    def test_internal_branch_is_labelled(self):
        df = self._df()
        self.assertTrue(any(".L" in a for a in df["assembly"]), df["assembly"].tolist())

    def test_region_target(self):
        one = self._df("myfunc")
        region = self._df(("myfunc", "other"))
        self.assertEqual(region["name"].unique().tolist(), ["myfunc..other"])
        self.assertGreater(len(region), len(one))
        self.assertLessEqual(region["address"].min(), one["address"].min())

    def test_region_spans_functions(self):
        one = self._df("myfunc")
        region = self._df(("myfunc", "other"))
        self.assertGreater(len(region), len(one))
        self.assertIn("other", region.attrs["name"])
        self.assertTrue((region["address"] > one["address"].max()).any())

    def test_region_word_is_a_region(self):
        region = self._df("myfunc..other")
        self.assertEqual(region["name"].unique().tolist(), ["myfunc..other"])

    def test_empty_region_raises(self):
        with self.assertRaises(ValueError):
            self._df("myfunc..")

    def test_name_override(self):
        self.assertEqual(self._df(name="custom")["name"].unique().tolist(), ["custom"])

    def test_unknown_target(self):
        with self.assertRaises(ValueError):
            self._df("nosuchfunc")

    def test_per_ip_join(self):
        df = self._df()
        ip = int(df["address"].iloc[1])
        state = {c: df[c].tolist() for c in df.columns if str(c).startswith("data.")}
        results = pd.DataFrame(
            {
                "ip": [ip, ip, 0x10],
                "uops": [1.0, 2.0, 3.0],
                "assembly": ["x", "y", "z"],
                "data.rdi": [4.0, 4.0, 4.0],
            }
        )
        out = self._df(results=[results])
        self.assertIn("uops", out.columns)
        self.assertNotIn("assembly_x", out.columns)
        self.assertEqual(len(out), len(df))
        self.assertEqual(out["uops"].iloc[1], 3.0)
        self.assertTrue(pd.isna(out["uops"].iloc[0]))
        for column, values in state.items():
            self.assertEqual(out[column].tolist(), values, column)

    def test_event_period_pivot(self):
        df = self._df()
        ip = int(df["address"].iloc[0])
        results = pd.DataFrame(
            {
                "ip": [ip, ip],
                "event": ["cycles", "instructions"],
                "period": [10.0, 20.0],
            }
        )
        out = self._df(results=[results])
        self.assertNotIn("mode", out.columns)
        self.assertIn("cycles", out.columns)
        self.assertIn("instructions", out.columns)
        self.assertEqual(out["cycles"].iloc[0], 10.0)
        self.assertEqual(out["cycles"].iloc[0], 10.0)
        self.assertEqual(out["instructions"].iloc[0], 20.0)

    def test_results_without_address_only_group(self):
        results = pd.DataFrame(
            {"cycles": [10.0, 12.0], "samples": [0, 1], "operations": [1, 1]}
        )
        out = self._df(results=[results])
        self.assertNotIn("cycles", out.columns)
        self.assertEqual(len(out), len(self._df()))

    def test_results_merge_into_one_table(self):
        bench = pd.DataFrame(
            {
                "file": ["prog@1", "prog@1"],
                "name": ["myfunc", "myfunc"],
                "mode": ["latency", "latency"],
                "cycles": [10.0, 12.0],
            }
        )
        track = pd.DataFrame(
            {
                "file": ["prog@1", "prog@1"],
                "name": ["myfunc", "hot_begin..hot_end"],
                "mode": ["record", "record"],
                "cycles": [30.0, 5.0],
            }
        )
        out = self._df(results=[bench, track])
        self.assertEqual(out["index"].nunique(), len(out))
        self.assertEqual(out["name"].unique().tolist(), ["myfunc"])
        text = cli._format_analyze_table(out)
        self.assertEqual(len(text.strip().split("\n\n")), 1)
        self.assertEqual(len(text.strip().splitlines()), len(out) + 1)

    def test_per_instruction_values_are_summed_per_group(self):
        ip = int(self._df()["address"].iloc[0])
        first = pd.DataFrame({"ip": [ip], "uops": [1.0]})
        second = pd.DataFrame({"ip": [ip], "uops": [2.0]})
        out = self._df(results=[first, second])
        self.assertEqual(out["uops"].iloc[0], 3.0)

    def test_event_selection_on_joined_columns(self):
        ip = int(self._df()["address"].iloc[0])
        results = pd.DataFrame({"ip": [ip], "uops": [1.0]})
        out = self._df(results=[results], event=["assembly", "uops"])
        self.assertEqual(
            _insn_columns(out),
            ["file", "name", "index", "address", "assembly", "uops"],
        )

    def test_config_attrs(self):
        out = self._df(config={"external": {"lib": False}})
        self.assertFalse(out.attrs["config"]["external"]["lib"])

    def test_index_numbers_every_instruction(self):
        df = self._df()
        self.assertEqual(df["index"].tolist(), list(range(len(df))))
        self.assertEqual(str(df["index"].dtype), "int64")

    def test_every_state_is_analyzed(self):
        if self.branchy is None:
            self.skipTest("gcc unavailable")
        df = perf.analyze(f"{self.branchy}:pick")
        joined = " ".join(df["assembly"].tolist())
        self.assertIn("je", joined)
        self.assertIn(".L", joined)
        self.assertEqual(df["index"].tolist(), list(range(len(df))))

    def test_state_columns_show_the_explored_state(self):
        df = self._df()
        columns = [c for c in df.columns if c.startswith("data.")]
        self.assertTrue(columns, list(df.columns))
        for column in columns:
            values = df[column].dropna().tolist()
            self.assertTrue(values, column)
            self.assertGreater(len({repr(v) for v in values}), 1, column)
        self.assertTrue(
            any("rdi" in str(v) for v in df["data.rax"].tolist()),
            df.to_dict(orient="records"),
        )

    def test_no_harness_instructions_are_listed(self):
        df = self._df()
        joined = " ".join(df["assembly"].tolist())
        for harness in ("rdtsc", "rdpmc", "lfence", "rdtscp", "clflushopt", "mfence"):
            self.assertNotIn(harness, joined)

    def test_harness_registers_are_not_state(self):
        df = self._df()
        for column in df.columns:
            if not str(column).startswith("data."):
                continue
            self.assertNotIn(
                str(column).split(".", 1)[1], get_arch()._HARNESS_REGS, column
            )

    def test_asm_source_label(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "f.s")
        Path(path).write_text(
            "foo:\n\tmov eax, 42\n\tret\nbar:\n\txor eax, eax\n\tret\n"
        )
        df = perf.analyze(f"{path}:foo")
        self.assertEqual(df["name"].unique().tolist(), ["foo"])
        self.assertEqual(df["assembly"].tolist(), ["mov eax, 0x2a"])
        region = perf.analyze(f"{path}:foo..bar")
        self.assertEqual(region["assembly"].tolist(), ["mov eax, 0x2a"])

    def test_asm_source_unknown_label(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "f.s")
        Path(path).write_text("foo:\n\tnop\n")
        with self.assertRaises(ValueError):
            perf.analyze(f"{path}:nope")

    def test_json_envelope_has_addresses(self):
        payload = json.loads(perf.to_json(self._df(event=["assembly"])))
        self.assertEqual(payload["file"], self._df().attrs["file"])
        self.assertEqual(len(payload["output"]), len(self._df()))
        self.assertIn("assembly", payload["output"][0])


class TestCli(unittest.TestCase):
    def test_table_is_left_aligned(self):
        text = _run(_args(code="mov eax, 42; add eax, ebx"))
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        start = lines[0].index("assembly")
        for line, asm in zip(lines[1:], ["mov eax, 0x2a", "add eax, ebx"]):
            self.assertEqual(line.index(asm, start), start, line)

    def test_encoding_is_left_aligned(self):
        text = _run(_args(code="mov eax, 42; add eax, ebx"))
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        start = lines[0].index("encoding")
        for line, encoding in zip(lines[1:], ["b8 2a 00 00 00", "01 d8"]):
            self.assertEqual(line.index(encoding, start), start, line)

    def test_state_values_are_printed_as_text(self):
        asm = "mov rax, rdi; add rax, 1; ret"
        value = int(perf.analyze(code=asm)["data.rax"].iloc[0])
        text = _run(_args(code=asm))
        self.assertIn("data.rax", text)
        self.assertIn(str(value), text)
        self.assertNotIn(f"{value}.00", text)

    def test_table_hex_addresses(self):
        text = _run(_args(code="mov eax, 42"))
        self.assertIn("0x1000000", text)
        self.assertNotIn("16777216", text)

    def test_filter_matching_nothing_prints_empty_table(self):
        text = _run(_args(code="mov eax, 42; add eax, ebx", filter="size > 1000"))
        self.assertIn("Empty DataFrame", text)
        self.assertNotIn("mov eax", text)

    def test_json_flag(self):
        payload = json.loads(_run(_args(code="mov eax, 42", json=True)))
        self.assertEqual(payload["output"][0]["assembly"], "mov eax, 0x2a")

    def test_missing_target(self):
        with self.assertRaises(SystemExit):
            _run(_args())
        with self.assertRaises(SystemExit):
            _run(_args(code="/bin/true:"))

    def test_unknown_event(self):
        with self.assertRaises(SystemExit):
            _run(_args(code="mov eax, 42", event=["nope"]))

    def test_event_comma_split(self):
        text = _run(_args(code="mov eax, 42", event=["assembly,encoding"]))
        self.assertIn("encoding", text)
        self.assertNotIn("size", text)

    def test_column_flag_is_an_alias_of_event(self):
        parser = cli._build_parser()
        for flag in ("-c", "--column", "-e", "--event"):
            options = parser.parse_args(["mov eax, 42", flag, "assembly,encoding"])
            self.assertEqual(options.event, ["assembly,encoding"], flag)

    def test_the_table_has_no_mode_column(self):
        text = _run(_args(code="mov eax, 42"))
        self.assertNotIn("mode", text)

    def test_emit_asm_prints_the_executed_path(self):
        args = _args(code="mov eax, 42; add eax, ebx", emit_asm=True)
        text = _run(args)
        self.assertIn(".intel_syntax noprefix", text)
        self.assertIn("mov eax, 0x2a", text)
        self.assertIn("add eax, ebx", text)

    def test_emit_asm_omits_the_table(self):
        text = _run(_args(code="mov eax, 42", emit_asm=True))
        self.assertNotIn("encoding", text)

    def test_data_columns_shown(self):
        text = _run(_args(code="mov rax, rdi", data={"rdi": 5}))
        self.assertIn("data.rdi", text)

    def test_filter_selects_instructions(self):
        text = _run(
            _args(
                code="mov eax, 1; add eax, 2; ret", data={"eax": 5}, filter="size < 4"
            )
        )
        self.assertIn("add eax, 2", text)
        self.assertNotIn("mov eax, 1", text)

    def test_filter_uses_state_columns(self):
        df = perf.analyze(code="mov rax, rdi; add rax, 1; ret")
        keep = int(df["data.rdi"].iloc[0])
        text = _run(
            _args(code="mov rax, rdi; add rax, 1; ret", filter=f"`data.rdi` == {keep}")
        )
        rows = [line for line in text.strip().splitlines() if line.strip()]
        self.assertEqual(len(rows), 2)

    def test_bad_filter_exits(self):
        with self.assertRaises(SystemExit):
            _run(_args(code="mov eax, 1", filter="nope == 1"))


class TestCliDataPaths(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self.exe = os.path.join(self._tmp, "prog")
        src = os.path.join(self._tmp, "prog.c")
        Path(src).write_text(SOURCE)
        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
        r = subprocess.run(
            ["gcc", "-O2", "-fno-inline", "-o", self.exe, src], capture_output=True
        )
        if r.returncode != 0:
            self.skipTest("compile failed")

    def _path(self, name, payload):
        path = os.path.join(self._tmp, name)
        Path(path).write_text(json.dumps(payload))
        return path

    def test_data_paths_become_columns(self):
        address = int(perf.analyze(f"{self.exe}:myfunc")["address"].iloc[0])
        bench = self._path(
            "bench.json",
            {
                "file": "prog@1",
                "name": "myfunc-1",
                "output": [
                    {"mode": "latency", "samples": 0, "cycles": 10.0},
                    {"mode": "latency", "samples": 1, "cycles": 12.0},
                ],
            },
        )
        record = self._path(
            "perf.json",
            {
                "output": [
                    {"ip": address, "event": "cycles", "period": 7.0},
                    {"ip": address, "event": "cycles", "period": 3.0},
                ]
            },
        )
        text = _run(_args(file=self.exe, target="myfunc", data_paths=[bench, record]))
        tables = text.strip().split("\n\n")
        self.assertEqual(len(tables), 1)
        self.assertIn("myfunc-1", text)
        self.assertIn("10.00", text)

    def test_load_paths_keeps_files_separate(self):
        a = self._path("a.json", {"file": "a@1", "name": "a", "output": [{"m": 1}]})
        b = self._path("b.json", {"file": "b@1", "name": "b", "output": [{"m": 2}]})
        frames = _shared.load_paths([a, b])
        self.assertEqual(len(frames), 2)
        self.assertEqual([f["name"].unique().tolist() for f in frames], [["a"], ["b"]])

    def test_missing_data_path(self):
        with self.assertRaises(SystemExit):
            _run(_args(file=self.exe, target="myfunc", data_paths=["/nope/x.json"]))


class TestPerfConfig(unittest.TestCase):
    def _subparser(self):
        import argparse

        parser = argparse.ArgumentParser(prog="perf")
        analyze = parser.add_subparsers(dest="command").add_parser("analyze")
        analyze.add_argument("file", nargs="?", default=None)
        analyze.add_argument("target", nargs="?", default=None)
        analyze.add_argument("asm", nargs="?", default=None)
        analyze.add_argument("-e", "--event", action="append", default=None)
        return analyze

    def test_analyze_is_a_perfconfig_command(self):
        self.assertIn("analyze", _shared.PERFCONFIG_COMMANDS)

    def test_data_and_config_keys_apply(self):
        sub = self._subparser()
        self.assertTrue(_shared._key_applies(sub, "analyze", "config.func.align"))
        self.assertTrue(_shared._key_applies(sub, "analyze", "data.rdi"))

    def test_flat_events_splits_commas(self):
        self.assertEqual(
            _shared.flat_events(["assembly,encoding"]), ["assembly", "encoding"]
        )
        self.assertEqual(_shared.flat_events(["a", "b,c"]), ["a", "b", "c"])
        self.assertIsNone(_shared.flat_events(None))
