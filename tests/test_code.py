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
    return [
        c
        for c in df.columns
        if not str(c).startswith("data.") and c not in ("file", "name")
    ]


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
        column=kw.get("column", None),
        data_paths=kw.get("data_paths", None),
        json=kw.get("json", False),
        interactive=False,
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
        df = perf.analyze(asm="mov eax, 42; add eax, ebx")
        self.assertEqual(
            df.columns.tolist(),
            [
                "index",
                "file",
                "name",
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

    def test_default_events(self):
        self.assertEqual(
            perf.code.DEFAULT_EVENTS,
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
                "data*",
            ],
        )

    def test_default_takes_the_state_columns(self):
        df = perf.analyze(asm="mov rax, rdi", data={"regs": {"rdi": 5}})
        self.assertIn("data.rdi", df.columns)

    def test_attrs(self):
        df = perf.analyze(asm="mov eax, 42", name="snip")
        self.assertEqual(df.attrs["name"], "snip")
        self.assertEqual(df.attrs["code"], "mov eax, 0x2a;")
        self.assertIsNone(df.attrs["file"])
        self.assertNotIn("data", df.attrs)

    def test_osaca_latency_and_throughput(self):
        df = perf.analyze(asm="mov eax, 42; imul ecx, edx; nop")
        self.assertEqual(df["latency"].tolist()[0:2], [1.0, 3.0])
        self.assertEqual(df["throughput"].tolist()[0:2], [0.2, 1.0])

    def test_osaca_reports_unknown_forms_as_nan(self):
        df = perf.analyze(asm="nop")
        self.assertTrue(df["latency"].isna().all())
        self.assertTrue(df["throughput"].isna().all())

    def test_osaca_columns_follow_the_instructions(self):
        df = perf.analyze(asm="mov eax, 42", column=["assembly,latency"])
        self.assertEqual(_insn_columns(df)[-1], "latency")

    def test_name_defaults_to_normalized_code(self):
        df = perf.analyze(asm="mov eax, 42")
        self.assertEqual(df["name"].unique().tolist(), ["mov eax, 0x2a;"])

    def test_branch_is_labelled(self):
        df = perf.analyze(asm="mov eax, 1; jl 0x1000007; mov ebx, 2")
        self.assertIn("jl .L1000007", df["assembly"].tolist())

    def test_state_columns_hold_the_explored_data(self):
        code = "mov rax, rdi; add rax, 1; ret"
        df = perf.analyze(asm=code, data={"regs": {"rdi": 15}})
        self.assertEqual(df["assembly"].tolist(), ["mov rax, rdi", "add rax, 1", "ret"])
        self.assertEqual(df["data.rdi"].tolist(), [[15], [15], [15]])

    def test_only_the_data_the_target_reads_is_state(self):
        code = "mov rax, rdi; add rax, 1; ret"
        df = perf.analyze(asm=code, data={"regs": {"rdi": 15}})
        self.assertEqual([c for c in df.columns if c.startswith("data.")], ["data.rdi"])
        self.assertNotIn("data.rax", df.columns)

    def test_state_columns_cover_every_state(self):
        df = perf.analyze(asm="mov rax, rdi; add rax, 1; ret")
        self.assertEqual(len(df), 3)
        self.assertEqual(df["index"].tolist(), [0, 1, 2])

    def test_harness_registers_are_not_state(self):
        df = perf.analyze(asm="mov rax, rdi; add rax, 1; ret")
        self.assertNotIn("data.rsp", df.columns)
        self.assertNotIn("data.rbp", df.columns)
        self.assertNotIn("data.rip", df.columns)

    def test_filter_keeps_matching_instructions(self):
        df = perf.analyze(asm="mov rax, rdi; add rax, 1; ret", filter="size > 2")
        self.assertEqual(df["assembly"].tolist(), ["mov rax, rdi", "add rax, 1"])

    def test_filter_uses_state_columns(self):
        out = perf.analyze(
            asm="mov rax, rdi; add rax, 1; ret",
            data={"regs": {"rdi": 15}},
            filter="15 in `data.rdi`",
        )
        self.assertEqual(
            out["assembly"].tolist(), ["mov rax, rdi", "add rax, 1", "ret"]
        )
        out = perf.analyze(
            asm="mov rax, rdi; add rax, 1; ret",
            data={"regs": {"rdi": 15}},
            filter="16 in `data.rdi`",
        )
        self.assertTrue(out.empty)

    def test_state_columns_are_always_lists(self):
        df = perf.analyze(asm="mov rax, rdi", data={"regs": {"rdi": 1}})
        self.assertEqual(df["data.rdi"].tolist(), [[1]])

    def test_filter_accepts_plain_columns(self):
        df = perf.analyze(asm="mov rax, rdi; add rax, 1; ret", filter="size <= 3")
        self.assertEqual(df["assembly"].tolist(), ["mov rax, rdi", "ret"])

    def test_bad_filter(self):
        with self.assertRaises(ValueError) as ctx:
            perf.analyze(asm="mov eax, 1", filter="nope == 1")
        self.assertIn("invalid filter", str(ctx.exception))

    def test_column_selects_columns(self):
        df = perf.analyze(asm="mov eax, 42; ret", column=["assembly", "encoding"])
        self.assertEqual(df.columns.tolist(), ["assembly", "encoding"])

    def test_index_is_hidden_unless_selected(self):
        df = perf.analyze(asm="mov eax, 42; ret", column=["assembly"])
        self.assertNotIn("index", df.columns)
        df = perf.analyze(asm="mov eax, 42; ret", column=["assembly", "index"])
        self.assertEqual(df.columns.tolist(), ["index", "assembly"])

    def test_index_is_first_when_default(self):
        df = perf.analyze(asm="mov eax, 42; ret")
        self.assertEqual(df.columns.tolist()[0], "index")

    def test_column_selects_data(self):
        df = perf.analyze(
            asm="mov rax, rdi",
            data={"regs": {"rdi": 15}},
            column=["assembly", "data*"],
        )
        self.assertEqual(df.columns.tolist(), ["assembly", "data.rdi"])

    def test_column_expands_a_bare_wildcard(self):
        every = perf.analyze(asm="mov eax, 42; ret", column=["*"])
        default = perf.analyze(asm="mov eax, 42; ret")
        self.assertEqual(
            every.columns.tolist(),
            [
                "index",
                "address",
                "encoding",
                "size",
                "latency",
                "throughput",
                "assembly",
                "file",
                "name",
            ],
        )
        only = perf.analyze(asm="mov eax, 42", column=["size"])
        self.assertNotIn("index", only.columns)
        self.assertTrue(set(default.columns).issubset(set(every.columns)))

    def test_column_selects_the_identity(self):
        df = perf.analyze(asm="mov eax, 42; ret", column=["name", "file"])
        self.assertEqual(df.columns.tolist(), ["name", "file"])

    def test_column_is_comma_separated(self):
        df = perf.analyze(asm="mov eax, 42; ret", column="assembly,encoding")
        self.assertEqual(df.columns.tolist(), ["assembly", "encoding"])

    def test_column_takes_an_expression(self):
        df = perf.analyze(asm="mov eax, 42; ret", column=["size/latency"])
        self.assertEqual(df.columns.tolist(), ["size/latency"])
        self.assertEqual(df["size/latency"].iloc[0], 5.0)

    def test_unknown_column(self):
        with self.assertRaises(ValueError) as ctx:
            perf.analyze(asm="mov eax, 42", column=["nope"])
        self.assertIn("unknown columns: nope", str(ctx.exception))

    def test_memory_columns_hold_the_data(self):
        code = "mov rax, [rsi]; mov rcx, [rsi+8]; ret"
        data = {"regs": {"rsi": 0x3232}, "mem": {"0x3232": 99, "0x323a": 7}}
        df = perf.analyze(asm=code, data=data)
        self.assertEqual(df["data.0x3232"].iloc[0], [99])
        self.assertEqual(df["data.0x323a"].iloc[1], [7])
        self.assertEqual([c for c in df.columns if str(c).startswith("mem-")], [])

    def test_a_store_overrides_the_data_for_the_next_load(self):
        code = "mov qword ptr [rdi], rsi; mov rax, [rdi]; ret"
        data = {"regs": {"rdi": 0x3232, "rsi": 7}, "mem": {"0x3232": 0}}
        df = perf.analyze(asm=code, data=data)
        store = df[df["assembly"].str.startswith("mov qword")]
        load = df[df["assembly"].str.startswith("mov rax")]
        self.assertEqual(store["data.0x3232"].tolist(), [[7]])
        self.assertEqual(load["data.0x3232"].tolist(), [[7]])

    def test_bad_asm(self):
        with self.assertRaises(ValueError):
            perf.analyze(asm="not_an_instruction")

    def test_requires_code(self):
        with self.assertRaises(TypeError):
            perf.analyze()

    def test_json_envelope(self):
        df = perf.analyze(asm="mov eax, 42", data={"regs": {"rax": 15}})
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


def _build_object(tmp, source, name="reloc"):
    src = os.path.join(tmp, f"{name}.c")
    obj = os.path.join(tmp, f"{name}.o")
    Path(src).write_text(source)
    if shutil.which("gcc") is None:
        return None
    r = subprocess.run(["gcc", "-O2", "-c", src, "-o", obj], capture_output=True)
    return obj if r.returncode == 0 and os.path.exists(obj) else None


class TestBinary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp()
        cls.exe = _build(cls._tmp, SOURCE)
        cls.branchy = _build(cls._tmp, BRANCHY, "branchy")
        cls.obj = _build_object(cls._tmp, SOURCE)

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
        return perf.analyze(target=code, **kw)

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

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            perf.analyze("/nonexistent/a.out:func")

    def test_missing_target_raises(self):
        with self.assertRaises(ValueError):
            perf.analyze(self.exe)

    def test_a_relocatable_object_keeps_its_name(self):
        if self.obj is None:
            self.skipTest("gcc unavailable")
        df = perf.analyze(f"{self.obj}:myfunc")
        self.assertTrue(df.attrs["file"].startswith("reloc.o@"), df.attrs["file"])
        self.assertEqual(df["file"].unique().tolist(), [df.attrs["file"]])

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
        out = self._df(results=[results], column=["*"])
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
        out = self._df(results=[results], column=["*"])
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
        out = self._df(results=[results], column=["*"])
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
        out = self._df(results=[bench, track], column=["*"])
        self.assertEqual(out["index"].nunique(), len(out))
        self.assertEqual(out["name"].unique().tolist(), ["myfunc"])
        text = cli.format_analyze_table(out)
        self.assertEqual(len(text.strip().split("\n\n")), 1)
        self.assertEqual(len(text.strip().splitlines()), len(out) + 1)

    def test_per_instruction_values_are_summed_per_group(self):
        ip = int(self._df()["address"].iloc[0])
        first = pd.DataFrame({"ip": [ip], "uops": [1.0]})
        second = pd.DataFrame({"ip": [ip], "uops": [2.0]})
        out = self._df(results=[first, second], column=["*"])
        self.assertEqual(out["uops"].iloc[0], 3.0)

    def test_column_selection_on_joined_columns(self):
        ip = int(self._df()["address"].iloc[0])
        results = pd.DataFrame({"ip": [ip], "uops": [1.0], "cycles": [2.0]})
        out = self._df(results=[results], column=["assembly", "uops"])
        self.assertEqual(out.columns.tolist(), ["assembly", "uops"])

    def test_column_expression_over_joined_counters(self):
        ip = int(self._df()["address"].iloc[0])
        results = pd.DataFrame({"ip": [ip], "cycles": [2.0], "instructions": [8.0]})
        out = self._df(results=[results], column=["instructions/cycles"])
        self.assertEqual(out.columns.tolist(), ["instructions/cycles"])
        self.assertEqual(out["instructions/cycles"].iloc[0], 4.0)

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

    def test_state_columns_show_the_explored_data(self):
        df = self._df(data={"regs": {"rdi": 7}})
        columns = [c for c in df.columns if c.startswith("data.")]
        self.assertEqual(columns, ["data.rdi"])
        self.assertEqual({tuple(v) for v in df["data.rdi"]}, {(7,)})

    def test_pinned_registers_are_not_data(self):
        df = self._df()
        self.assertEqual([c for c in df.columns if c.startswith("data.")], [])

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

    def test_every_instruction_of_the_function_is_listed(self):
        import angr

        from perf.arch import load as load_arch
        from perf.code import _instructions
        from perf.info import functions

        proj = angr.Project(self.exe, auto_load_libs=False, load_debug_info=False)
        funcs, _ = functions(proj)
        start, end = funcs["myfunc"]
        blob = proj.loader.memory.load(start, end - start)
        expected = _instructions(load_arch(proj), blob, start)
        got = self._df()
        self.assertEqual(got["address"].tolist(), [int(i.address) for i in expected])

    def test_a_call_does_not_end_the_listing(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        src = os.path.join(d, "calls.c")
        exe = os.path.join(d, "calls")
        Path(src).write_text(
            "#include <string.h>\n"
            "__attribute__((noinline)) void helper(char* p){ memset(p, 1, 8); }\n"
            "__attribute__((noinline)) int caller(char* p){\n"
            "  helper(p);\n"
            "  return (int)strlen(p);\n"
            "}\n"
            "int main(void){ char b[8] = {0}; return caller(b); }\n"
        )
        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
        rc = subprocess.run(
            ["gcc", "-O2", "-fno-inline", "-o", exe, src], capture_output=True
        )
        if rc.returncode != 0 or not os.path.exists(exe):
            self.skipTest("gcc build failed")
        df = perf.analyze(f"{exe}:caller")
        asm = df["assembly"].tolist()
        calls = [i for i, a in enumerate(asm) if a.startswith("call ")]
        self.assertTrue(calls, asm)
        self.assertLess(max(calls), len(asm) - 1)
        self.assertIn("ret", asm[-1])

    def test_instructions_the_explorer_never_reaches_are_kept(self):
        df = self._df("myfunc")
        self.assertTrue((df["address"] > df["address"].iloc[0]).any())

    def test_json_envelope_has_addresses(self):
        payload = json.loads(perf.to_json(self._df(column=["assembly"])))
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
        text = _run(_args(code=asm, data={"regs": {"rdi": 15}}))
        self.assertIn("data.rdi", text)
        self.assertIn("[15]", text)
        self.assertNotIn("15.00", text)

    def test_memory_values_are_printed_as_text(self):
        code = "mov rax, [rsi]; mov rcx, [rsi+8]; ret"
        data = {"regs": {"rsi": 0x3232}, "mem": {"0x3232": 99, "0x323a": 7}}
        text = _run(_args(code=code, data=data))
        self.assertIn("data.0x3232", text)
        self.assertIn("[99]", text)
        self.assertNotIn("99.00", text)
        self.assertNotIn("mem-loads", text)
        self.assertNotIn("mem-stores", text)

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

    def test_unknown_column(self):
        with self.assertRaises(SystemExit):
            _run(_args(code="mov eax, 42", column=["nope"]))

    def test_column_comma_split(self):
        text = _run(_args(code="mov eax, 42", column=["assembly,encoding"]))
        self.assertIn("encoding", text)
        self.assertNotIn("size", text)

    def test_column_wildcard(self):
        text = _run(_args(code="mov eax, 42", column=["assembly,data*"]))
        self.assertIn("mov eax, 0x2a", text)
        self.assertNotIn("encoding", text)
        text = _run(_args(code="mov eax, 42", column=["*"]))
        self.assertIn("encoding", text)
        self.assertIn("file", text)

    def test_lone_assembly_column_is_an_llvm_mca_script(self):
        text = _run(_args(code="mov eax, 42; add eax, ebx", column=["assembly"]))
        self.assertEqual(
            text.strip().splitlines(),
            [".intel_syntax", "mov eax, 0x2a", "add eax, ebx"],
        )
        for line in text.splitlines():
            self.assertEqual(line, line.rstrip(), line)

    def test_assembly_stays_reachable_as_a_column(self):
        text = _run(_args(code="mov eax, 42", column=["assembly,size"]))
        self.assertIn("assembly", text)
        text = _run(_args(code="mov eax, 42", column=["assembly"], json=True))
        self.assertEqual(
            json.loads(text)["output"][0]["assembly"],
            "mov eax, 0x2a",
        )

    def test_event_selects_the_columns(self):
        parser = cli._build_parser()
        for flag in ("-e", "--event"):
            options = parser.parse_args(["mov eax, 42", flag, "assembly,encoding"])
            self.assertEqual(options.column, ["assembly,encoding"], flag)

    def test_column_flag_is_gone(self):
        parser = cli._build_parser()
        for flag in ("-c", "--column"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["mov eax, 42", flag, "assembly"])

    def test_the_table_has_no_mode_column(self):
        text = _run(_args(code="mov eax, 42"))
        self.assertNotIn("mode", text)

    def test_data_columns_shown(self):
        text = _run(_args(code="mov rax, rdi", data={"regs": {"rdi": 5}}))
        self.assertIn("data.rdi", text)

    def test_filter_selects_instructions(self):
        text = _run(
            _args(
                code="mov eax, 1; add eax, 2; ret",
                data={"regs": {"eax": 5}},
                filter="size < 4",
            )
        )
        self.assertIn("add eax, 2", text)
        self.assertNotIn("mov eax, 1", text)

    def test_filter_uses_state_columns(self):
        asm = "mov rax, rdi; add rax, 1; ret"
        text = _run(
            _args(code=asm, data={"regs": {"rdi": 15}}, filter="15 in `data.rdi`")
        )
        rows = [line for line in text.strip().splitlines() if line.strip()]
        self.assertEqual(len(rows), 4)
        text = _run(
            _args(code=asm, data={"regs": {"rdi": 15}}, filter="16 in `data.rdi`")
        )
        self.assertIn("Empty DataFrame", text)

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
        text = _run(
            _args(
                file=self.exe,
                target="myfunc",
                data_paths=[bench, record],
                column=["name,assembly,cycles"],
            )
        )
        tables = text.strip().split("\n\n")
        self.assertEqual(len(tables), 1)
        self.assertIn("myfunc-1", text)
        self.assertIn("10.00", text)

    def test_load_paths_keeps_files_separate(self):
        a = self._path("a.json", {"file": "a@1", "name": "a", "output": [{"m": 1}]})
        b = self._path("b.json", {"file": "b@1", "name": "b", "output": [{"m": 2}]})
        frames = cli.load_paths([a, b])
        self.assertEqual(len(frames), 2)
        self.assertEqual([f["name"].unique().tolist() for f in frames], [["a"], ["b"]])

    def test_missing_data_path(self):
        with self.assertRaises(SystemExit):
            _run(_args(file=self.exe, target="myfunc", data_paths=["/nope/x.json"]))


class TestPerfConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._cwd = os.getcwd()
        self.addCleanup(os.chdir, self._cwd)
        os.chdir(self._tmp)
        Path(".perfconfig").write_text(
            "; a comment\n"
            "[default]\nfilter = default\n"
            "[analyze]\nfilter = analyze\nconfig.func.align = 32 # inline\n"
            "[view]\nstat = p50\n"
        )
        self.entries = cli._config_entries()

    def _subparser(self):
        import argparse

        parser = argparse.ArgumentParser(prog="perf")
        analyze = parser.add_subparsers(dest="command").add_parser("analyze")
        analyze.add_argument("file", nargs="?", default=None)
        analyze.add_argument("target", nargs="?", default=None)
        analyze.add_argument("asm", nargs="?", default=None)
        analyze.add_argument("-e", "--event", action="append", default=None)
        return analyze

    def test_only_the_own_section_and_default_are_read(self):
        self.assertEqual(self.entries, {"filter": "analyze", "config.func.align": "32"})

    def test_dotted_config_and_data_keys_apply(self):
        sub = self._subparser()
        self.assertTrue(cli._known(sub, "config.func.align"))
        self.assertTrue(cli._known(sub, "data.rdi"))

    def test_given_flags_win_over_the_file(self):
        parser = self._subparser()
        argv = cli.apply_perfconfig(parser, ["-e", "assembly", "mov eax, 42"])
        self.assertEqual(
            argv, ["--config.func.align=32", "-e", "assembly", "mov eax, 42"]
        )

    def test_dotted_config_values_are_parsed(self):
        self.assertEqual(cli.parse_value("hit_rate:100"), {"hit_rate": 100})
        self.assertEqual(cli.parse_value("[1, 2,3]"), [1, 2, 3])


def _debug_exe(test, source=None):
    tmp = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, tmp, True)
    cwd = os.getcwd()
    test.addCleanup(os.chdir, cwd)
    os.chdir(tmp)
    Path("prog.c").write_text(
        source
        or (
            "long myfunc(long x){\n"
            "  long sum = 0;\n"
            "  for (int i = 0; i < 4; i++) {\n"
            "    if (x > 2) sum += x * i;\n"
            "    else sum -= i;\n"
            "  }\n"
            "  return sum;\n"
            "}\n"
            "int main() { return (int)myfunc(3); }\n"
        )
    )
    if shutil.which("gcc") is None:
        test.skipTest("gcc unavailable")
    done = subprocess.run(
        ["gcc", "-O2", "-g", "-fno-inline", "-o", "prog", "prog.c"],
        capture_output=True,
    )
    exe = os.path.join(tmp, "prog")
    if done.returncode != 0 or not os.path.exists(exe):
        test.skipTest("gcc build failed")
    return exe


def _debug_seen(test, exe, **kw):
    import angr

    from perf import bench as bench_module

    bench_module._PROJECTS.clear()
    test.addCleanup(bench_module._PROJECTS.clear)
    seen = []
    real = angr.Project

    def wrap(*args, **kwargs):
        seen.append(kwargs.get("load_debug_info"))
        return real(*args, **kwargs)

    with patch("angr.Project", side_effect=wrap):
        bench_module._PROJECTS.clear()
        df = perf.analyze(f"{exe}:myfunc", **kw)
    return seen, df


class TestDebugLoad(unittest.TestCase):
    def test_without_debug_never_loads_debug_info(self):
        exe = _debug_exe(self)
        seen, _ = _debug_seen(self, exe)
        self.assertTrue(seen)
        self.assertNotIn(True, seen)

    def test_with_debug_loads_debug_info_for_the_target(self):
        exe = _debug_exe(self)
        seen, _ = _debug_seen(self, exe, debug=True)
        self.assertIn(True, seen)

    def test_snippet_never_loads_debug_info(self):
        import angr

        seen = []
        real = angr.Project

        def wrap(*args, **kwargs):
            seen.append(kwargs.get("load_debug_info"))
            return real(*args, **kwargs)

        with patch("angr.Project", side_effect=wrap):
            perf.analyze(asm="mov eax, 42", debug=True)
        self.assertEqual(seen, [])


class TestDebugColumns(unittest.TestCase):
    def test_debug_adds_no_line_or_code_columns(self):
        df = perf.analyze(asm="mov eax, 42", debug=True)
        self.assertNotIn("line", df.columns)
        self.assertNotIn("code", df.columns)

    def test_binary_debug_adds_no_line_or_code_columns(self):
        exe = _debug_exe(self)
        df = perf.analyze(f"{exe}:myfunc", debug=True)
        self.assertNotIn("line", df.columns)
        self.assertNotIn("code", df.columns)
        plain = perf.analyze(f"{exe}:myfunc")
        self.assertEqual(df.columns.tolist(), plain.columns.tolist())

    def test_line_and_code_are_unknown_columns(self):
        with self.assertRaises(ValueError) as ctx:
            perf.analyze(asm="mov eax, 42", column=["line"])
        self.assertIn("unknown columns: line", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            perf.analyze(asm="mov eax, 42", column=["code"])
        self.assertIn("unknown columns: code", str(ctx.exception))

    def test_event_help_names_no_line_or_code(self):
        text = cli._build_parser().format_help()
        self.assertNotIn("line,code", text)


class TestDebugTable(unittest.TestCase):
    def test_debug_attr_holds_one_entry_per_row(self):
        exe = _debug_exe(self)
        df = perf.analyze(f"{exe}:myfunc", debug=True)
        debug = df.attrs.get("debug")
        self.assertIsNotNone(debug)
        self.assertEqual(len(debug), len(df))
        self.assertTrue(any(entry is not None for entry in debug))
        for entry in debug:
            if entry is None:
                continue
            src, num, code = entry
            self.assertTrue(src.endswith("prog.c"))
            self.assertGreater(num, 0)
            self.assertIsInstance(code, str)

    def test_without_debug_there_is_no_debug_attr(self):
        exe = _debug_exe(self)
        df = perf.analyze(f"{exe}:myfunc")
        self.assertNotIn("debug", df.attrs)

    def test_source_is_printed_before_its_instructions(self):
        exe = _debug_exe(self)
        df = perf.analyze(f"{exe}:myfunc", debug=True)
        text = cli.format_analyze_table(df)
        lines = text.splitlines()
        debug = df.attrs["debug"]
        for entry in debug:
            if entry is not None:
                self.assertNotIn(f"{entry[0]}:{entry[1]}", text)
        for entry in debug:
            if entry is not None and entry[2]:
                self.assertIn(entry[2], text)
        self.assertIn("else sum -= i;", text)
        asm_rows = [str(v) for v in df["assembly"].tolist()]
        pos = 0
        for asm in asm_rows:
            found = next((i for i in range(pos, len(lines)) if asm in lines[i]), None)
            self.assertIsNotNone(found, f"missing instruction {asm!r}")
            pos = found + 1

    def test_instructions_sharing_a_line_share_one_source_row(self):
        df = pd.DataFrame(
            {
                "index": [0, 1, 2],
                "address": [0x1000, 0x1005, 0x1008],
                "assembly": ["mov eax, 1", "add eax, 2", "ret"],
            }
        )
        df.attrs["debug"] = [
            ("a.c", 10, "int x;"),
            ("a.c", 10, "int x;"),
            ("a.c", 11, "return;"),
        ]
        text = cli.format_analyze_table(df)
        self.assertNotIn("a.c:10", text)
        self.assertNotIn("a.c:11", text)
        self.assertEqual(text.count("int x;"), 1)
        self.assertEqual(text.count("return;"), 1)
        rows = text.splitlines()
        first = next(i for i, line in enumerate(rows) if "int x;" in line)
        second = next(i for i, line in enumerate(rows) if "return;" in line)
        self.assertLess(first, second)
        self.assertIn("mov eax, 1", rows[first + 1])
        self.assertIn("add eax, 2", rows[first + 2])
        self.assertIn("ret", rows[second + 1])

    def test_table_without_debug_has_no_source_rows(self):
        exe = _debug_exe(self)
        df = perf.analyze(f"{exe}:myfunc")
        text = cli.format_analyze_table(df)
        self.assertNotIn("prog.c:", text)
        self.assertNotIn("long sum = 0;", text)

    def test_debug_flag_prints_source_in_the_table(self):
        exe = _debug_exe(self)
        args = _args(file=exe, target="myfunc")
        args.debug = True
        text = _run(args)
        self.assertNotIn("prog.c:", text)
        self.assertIn("long sum = 0;", text)
        plain = _run(_args(file=exe, target="myfunc"))
        self.assertNotIn("prog.c:", plain)
        self.assertNotIn("long sum = 0;", plain)

    def test_debug_with_only_assembly_still_prints_source(self):
        exe = _debug_exe(self)
        args = _args(file=exe, target="myfunc", column=["assembly"])
        args.debug = True
        text = _run(args)
        self.assertNotIn("prog.c:", text)
        self.assertIn("long sum = 0;", text)
        self.assertIn(".intel_syntax", text.splitlines()[0])
