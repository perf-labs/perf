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
import threading
import time
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

import perf
from perf.arch import arch as get_arch
from perf.mca import _instructions


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
        event=kw.get("event", None),
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


class TestInstructionFrame(unittest.TestCase):
    def test_many_data_columns_are_added_without_fragmentation(self):
        from perf.mca import _instruction_frame

        arch = get_arch()
        blob = arch.assemble("nop", 0x1000)
        insns = list(arch.disassembler().disasm(blob, 0x1000))
        records = {0x1000: {f"data.r{i}": i for i in range(40)}}
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            df = _instruction_frame(insns, records)
        self.assertEqual(df["data.r0"].tolist(), [0])
        self.assertEqual(df["data.r39"].tolist(), [39])
        self.assertEqual(len(df), 1)

    def test_osaca_cycles_are_serialized(self):
        import perf.mca as mca

        active = {"n": 0, "peak": 0}
        lock = threading.Lock()

        def fake_semantics(_parser, _model):
            with lock:
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])
            time.sleep(0.05)
            with lock:
                active["n"] -= 1
            return SimpleNamespace(
                normalize_instruction_forms=lambda _forms: None,
                add_semantics=lambda _forms: None,
            )

        errors = []

        def worker():
            try:
                mca._osaca_cycles(["nop"])
            except BaseException as ex:
                errors.append(ex)

        with patch.object(mca, "_osaca_semantics", side_effect=fake_semantics):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertEqual(active["peak"], 1)


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
            perf.mca.DEFAULT_EVENTS,
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
        df = perf.analyze(asm="mov eax, 42", event=["assembly,latency"])
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
        df = perf.analyze(asm="mov eax, 42; ret", event=["assembly", "encoding"])
        self.assertEqual(df.columns.tolist(), ["assembly", "encoding"])

    def test_index_is_hidden_unless_selected(self):
        df = perf.analyze(asm="mov eax, 42; ret", event=["assembly"])
        self.assertNotIn("index", df.columns)
        df = perf.analyze(asm="mov eax, 42; ret", event=["assembly", "index"])
        self.assertEqual(df.columns.tolist(), ["assembly", "index"])

    def test_index_is_first_when_default(self):
        df = perf.analyze(asm="mov eax, 42; ret")
        self.assertEqual(df.columns.tolist()[0], "index")

    def test_column_selects_data(self):
        df = perf.analyze(
            asm="mov rax, rdi",
            data={"regs": {"rdi": 15}},
            event=["assembly", "data*"],
        )
        self.assertEqual(df.columns.tolist(), ["assembly", "data.rdi"])

    def test_column_expands_a_bare_wildcard(self):
        every = perf.analyze(asm="mov eax, 42; ret", event=["*"])
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
        only = perf.analyze(asm="mov eax, 42", event=["size"])
        self.assertNotIn("index", only.columns)
        self.assertTrue(set(default.columns).issubset(set(every.columns)))

    def test_column_selects_the_identity(self):
        df = perf.analyze(asm="mov eax, 42; ret", event=["name", "file"])
        self.assertEqual(df.columns.tolist(), ["name", "file"])

    def test_column_is_comma_separated(self):
        df = perf.analyze(asm="mov eax, 42; ret", event="assembly,encoding")
        self.assertEqual(df.columns.tolist(), ["assembly", "encoding"])

    def test_column_takes_an_expression(self):
        df = perf.analyze(asm="mov eax, 42; ret", event=["size/latency"])
        self.assertEqual(df.columns.tolist(), ["size/latency"])
        self.assertEqual(df["size/latency"].iloc[0], 5.0)

    def test_unknown_column(self):
        with self.assertRaises(ValueError) as ctx:
            perf.analyze(asm="mov eax, 42", event=["nope"])
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

    def test_printed_table_round_trips_through_the_reader(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "index": [0, 1],
                "assembly": ["mov eax, 0x2a", "add eax, ebx"],
                "encoding": ["b8 2a 00 00 00", "01 d8"],
                "size": [5, 2],
            }
        )
        text = cli.format_analyze_table(df)
        parsed = cli.table_to_df(text)
        self.assertEqual(parsed["assembly"].tolist(), df["assembly"].tolist())
        self.assertEqual(parsed["encoding"].tolist(), df["encoding"].tolist())
        self.assertEqual(parsed["size"].tolist(), [5, 2])

    def test_event_selects_the_columns(self):
        parser = cli._build_parser()
        for flag in ("-e", "--event"):
            options = parser.parse_args(["mov eax, 42", flag, "assembly,encoding"])
            self.assertEqual(options.event, ["assembly,encoding"], flag)

    def test_missing_config_file_is_a_clean_cli_error(self):
        with self.assertRaises(SystemExit):
            cli.parse_target_args(
                cli._build_parser(),
                ["mov eax, 42", "--config", "/nonexistent/analyze.json"],
            )

    def test_csv_duration_values_are_converted_to_ns(self):
        df = cli.table_to_df(
            "file,name,mode,operations,duration_time\n"
            "a.out,func,latency,1,3.00ns\n"
            "a.out,func,latency,1,1.50us\n"
        )
        self.assertEqual(df["duration_time"].tolist(), [3.0, 1500.0])


class TestCliDataPaths(unittest.TestCase):
    def test_load_paths_keeps_files_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.json")
            b = os.path.join(tmp, "b.json")
            Path(a).write_text(
                json.dumps({"file": "a@1", "name": "a", "output": [{"m": 1}]})
            )
            Path(b).write_text(
                json.dumps({"file": "b@1", "name": "b", "output": [{"m": 2}]})
            )
            frames = cli.load_paths([a, b])
        self.assertEqual(len(frames), 2)
        self.assertEqual([f["name"].unique().tolist() for f in frames], [["a"], ["b"]])


class TestPerfConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self._cwd = os.getcwd()
        self.addCleanup(os.chdir, self._cwd)
        os.chdir(self._tmp)
        Path(".perfconfig").write_text(
            "; a comment\n"
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

    def test_only_the_own_section_is_read(self):
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


_DEBUG_EXE_CACHE = {}
_DEBUG_EXE_DIR = None


def _debug_exe(test, source=None):
    global _DEBUG_EXE_DIR
    key = source or "default"
    cached = _DEBUG_EXE_CACHE.get(key)
    if cached is not None and os.path.exists(cached):
        return cached
    if _DEBUG_EXE_DIR is None:
        _DEBUG_EXE_DIR = tempfile.mkdtemp(prefix="perf-debug-")
    tmp = tempfile.mkdtemp(dir=_DEBUG_EXE_DIR)
    Path(os.path.join(tmp, "prog.c")).write_text(
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
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        done = subprocess.run(
            ["gcc", "-O2", "-g", "-fno-inline", "-o", "prog", "prog.c"],
            capture_output=True,
        )
    finally:
        os.chdir(cwd)
    exe = os.path.join(tmp, "prog")
    if done.returncode != 0 or not os.path.exists(exe):
        test.skipTest("gcc build failed")
    _DEBUG_EXE_CACHE[key] = exe
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
            perf.analyze(asm="mov eax, 42", event=["line"])
        self.assertIn("unknown columns: line", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            perf.analyze(asm="mov eax, 42", event=["code"])
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


class TestCodeHelpers(unittest.TestCase):
    def test_bracket_to_dot(self):
        from perf.mca import _bracket_to_dot

        self.assertEqual(_bracket_to_dot("data[0x1000]"), "data.0x1000")
        self.assertEqual(_bracket_to_dot("cycles"), "cycles")

    def test_expand_bracket(self):
        import pandas as pd

        from perf.mca import _expand

        df = pd.DataFrame({"data.0x1000": [1], "cycles": [2]})
        self.assertEqual(
            _expand(df, ["data[0x1000]", "cycles"]), ["data.0x1000", "cycles"]
        )

    def test_alias_memory(self):
        import pandas as pd

        from perf.mca import _alias_memory

        df = pd.DataFrame({"data.x": [5]})
        _alias_memory(df, ["data[x]"])
        self.assertEqual(df["data[x]"].tolist(), [5])

    def test_derive_strict(self):
        import pandas as pd

        from perf.mca import _derive

        df = pd.DataFrame({"a": [1, 2]})
        self.assertIn("a", _derive(df, ["a/b"], strict=False).columns)
        with self.assertRaises(ValueError):
            _derive(pd.DataFrame({"a": [1, 2]}), ["a/b"], strict=True)


class TestAnalyzeJoins(unittest.TestCase):
    @staticmethod
    def _df():
        return pd.DataFrame(
            {
                "index": [0, 1],
                "address": [0x1000, 0x1010],
                "assembly": ["mov eax, 42", "ret"],
                "file": ["prog@1", "prog@1"],
                "name": ["myfunc", "myfunc"],
            }
        )

    def test_per_ip_values_are_joined_and_summed(self):
        from perf.mca import _merge_results

        results = pd.DataFrame(
            {
                "address": [0x1010, 0x1010, 0xDEAD0000],
                "uops": [1.0, 2.0, 3.0],
                "assembly": ["x", "y", "z"],
                "data.rdi": [4.0, 4.0, 4.0],
            }
        )
        out = _merge_results(self._df(), [results], "prog@1", "myfunc")
        self.assertIn("uops", out.columns)
        self.assertNotIn("assembly_x", out.columns)
        self.assertEqual(len(out), 2)
        self.assertTrue(pd.isna(out["uops"].iloc[0]))
        self.assertEqual(out["uops"].iloc[1], 3.0)

    def test_event_period_values_become_columns(self):
        from perf.mca import _merge_results

        results = pd.DataFrame(
            {
                "ip": [0x1000, 0x1000],
                "event": ["cycles", "instructions"],
                "period": [10.0, 20.0],
            }
        )
        out = _merge_results(self._df(), [results], "prog@1", "myfunc")
        self.assertEqual(out["cycles"].iloc[0], 10.0)
        self.assertEqual(out["instructions"].iloc[0], 20.0)
        self.assertNotIn("mode", out.columns)

    def test_results_without_an_address_are_not_joined(self):
        from perf.mca import _merge_results

        results = pd.DataFrame(
            {"cycles": [10.0, 12.0], "samples": [0, 1], "operations": [1, 1]}
        )
        out = _merge_results(self._df(), [results], "prog@1", "myfunc")
        self.assertNotIn("cycles", out.columns)
        self.assertEqual(len(out), 2)

    def test_multiple_frames_are_summed_per_instruction(self):
        from perf.mca import _merge_results

        first = pd.DataFrame({"ip": [0x1000], "uops": [1.0]})
        second = pd.DataFrame({"ip": [0x1000], "uops": [2.0]})
        out = _merge_results(self._df(), [first, second], "prog@1", "myfunc")
        self.assertEqual(out["uops"].iloc[0], 3.0)

    def test_column_expressions_over_joined_counters(self):
        from perf.mca import _select

        df = self._df()
        df["cycles"] = [2.0, 4.0]
        df["instructions"] = [8.0, 8.0]
        out = _select(df, ["instructions/cycles"])
        self.assertEqual(out.columns.tolist(), ["instructions/cycles"])
        self.assertEqual(out["instructions/cycles"].tolist(), [4.0, 2.0])

    def test_harness_registers_are_not_recorded_as_state(self):
        from perf.mca import _record_regs

        records = {}
        with patch("perf.mca.min_ints", return_value=[1, 2]):
            _record_regs(
                records,
                {"state": object(), "inputs": {"rdi": object(), "r11": object()}},
                [0x1000],
                {"r11"},
            )
        self.assertEqual(records, {0x1000: {"data.rdi": {1}}})

    def test_asm_analysis_numbers_and_addresses_every_instruction(self):
        df = perf.analyze(
            asm="mov eax, 42; add eax, 1; ret",
            event=["index", "address", "assembly"],
        )
        self.assertEqual(df["index"].tolist(), [0, 1, 2])
        self.assertEqual(df["address"].tolist(), sorted(df["address"].tolist()))
        self.assertEqual(
            df["assembly"].tolist(), ["mov eax, 0x2a", "add eax, 1", "ret"]
        )


class TestMcaParsingHelpers(unittest.TestCase):
    def test_split_columns(self):
        from perf.mca import _split_columns

        self.assertEqual(_split_columns(None), [])
        self.assertEqual(
            _split_columns("assembly, encoding,assembly"), ["assembly", "encoding"]
        )
        self.assertEqual(_split_columns(["a,b", "", None, "b"]), ["a", "b"])

    def test_memory_image_parses_keys_and_first_values(self):
        from perf.mca import _memory_image

        image = _memory_image(
            {
                "mem": {
                    "0x1234:": 5,
                    "bad": 6,
                    "0x2000": [7, 8],
                    "0x3000": [None],
                }
            }
        )
        self.assertEqual(image, {0x1234: 5, 0x2000: 7, 0x3000: 0})
        self.assertEqual(_memory_image(None), {})

    def test_data_value_normalizes_sets(self):
        from perf.mca import _data_value

        self.assertEqual(_data_value({3, 1, 2}), [1, 2, 3])
        self.assertIsNone(_data_value(set()))
        self.assertEqual(_data_value([1, 2]), [1, 2])

    def test_file_label_without_content_hash(self):
        from perf.mca import _file_label

        with patch("perf.mca._content_hash8", return_value=""):
            self.assertEqual(_file_label("/tmp/prog"), "prog")
        with patch("perf.mca._content_hash8", return_value="deadbeef"):
            self.assertEqual(_file_label("/tmp/prog"), "prog@deadbeef")

    def test_record_memory_tracks_stores_and_loads(self):
        from perf.mca import _record_memory

        solution = {
            "state": object(),
            "data_addrs": {0x2000},
            "accesses": [
                (object(), 0x1000, "mem-stores", object()),
                (object(), 0x1000, "mem-loads", object()),
                (object(), 0x1001, "mem-loads", object()),
                (object(), 0x1002, "mem-stores", object()),
                (object(), 0x1003, "mem-loads", object()),
                (object(), 0x9999, "mem-loads", object()),
            ],
        }
        records = {}
        image = {}
        with patch(
            "perf.mca.eval_ints",
            return_value=[
                0x2000,
                9,
                0x2000,
                None,
                0x3000,
                4,
                0x2000,
                None,
                0x2000,
                1,
            ],
        ):
            _record_memory(records, solution, {0x1000, 0x1001, 0x1002, 0x1003}, image)
        self.assertEqual(image, {0x2000: 9})
        self.assertEqual(records[0x1000]["data.0x2000"], {9})
        self.assertEqual(records[0x1003]["data.0x2000"], {1})
        self.assertNotIn(0x1001, records)
        self.assertEqual(records[0x1002], {})

    def test_record_registers_skip_missing_values(self):
        from perf.mca import _record_regs

        records = {}
        with patch("perf.mca.min_ints", return_value=[None, 7]):
            _record_regs(
                records,
                {
                    "state": object(),
                    "inputs": {"rdi": object(), "rsi": object()},
                },
                [0x1000],
                set(),
            )
        self.assertEqual(records, {0x1000: {"data.rsi": {7}}})
        _record_regs({}, {"inputs": {}}, [0x1000], set())

    def test_merge_records_merges_sets(self):
        from perf.mca import _merge_records

        into = {0x1000: {"data.rdi": {1}}}
        out = _merge_records(
            into, {0x1000: {"data.rdi": {2}, "data.rsi": {3}}, 0x2000: {}}
        )
        self.assertIs(out, into)
        self.assertEqual(out[0x1000]["data.rdi"], {1, 2})
        self.assertEqual(out[0x1000]["data.rsi"], {3})

    def test_explored_ranges_include_contained_functions(self):
        from perf.mca import _explored_ranges

        funcs = {
            "main": (0x1000, 0x1100),
            "inner": (0x1020, 0x1040),
            "outer": (0x0F00, 0x1080),
            "same": (0x1000, 0x1100),
        }
        self.assertEqual(
            _explored_ranges("main", 0x1000, 0x1100, funcs),
            [
                ("main", 0x1000, 0x1100),
                ("inner", 0x1020, 0x1040),
            ],
        )
        self.assertEqual(
            _explored_ranges("x", 0x1000, 0x2000, None), [("x", 0x1000, 0x2000)]
        )

    def test_stack_spec_parsing(self):
        from perf.mca import _stack_align, _stack_size

        self.assertEqual(_stack_size({"stack": {"size": "4096"}}), 4096)
        self.assertIsNone(_stack_size({}))
        self.assertIsNone(_stack_size({"stack": {"size": "bad"}}))
        self.assertEqual(_stack_align({"stack": {"align": "32"}}), 32)
        self.assertEqual(_stack_align({}), 16)
        self.assertEqual(_stack_align({"stack": {"align": "bad"}}), 16)

    def test_form_value(self):
        from perf.mca import _form_value

        self.assertEqual(_form_value(SimpleNamespace(latency=3.0), "latency"), 3.0)
        self.assertIsNone(_form_value(object(), "latency"))
        self.assertIsNone(_form_value(None, "latency"))

    def test_osaca_cycles_empty(self):
        from perf.mca import _osaca_cycles

        self.assertEqual(_osaca_cycles([]), [])

    def test_osaca_cycles_partial_parse(self):
        from perf.mca import _osaca_cycles

        cycles = _osaca_cycles(["nop", "definitely-not-an-instruction"])
        self.assertEqual(len(cycles), 2)
        self.assertTrue(
            all(isinstance(value, float) for pair in cycles for value in pair)
        )

    def test_att_lines_survives_decode_errors(self):
        from perf.mca import _att_lines

        arch = get_arch()
        blob = arch.assemble("mov eax, 42", 0x1000)
        insns = list(arch.disassembler().disasm(blob, 0x1000))
        self.assertEqual(_att_lines(insns), ["movl $0x2a, %eax"])
        broken = Mock(bytes=b"", address=0)
        with patch("perf.mca.get_arch") as mocked:
            mocked.return_value.att_disassembler.return_value.disasm.side_effect = (
                RuntimeError("bad")
            )
            self.assertEqual(_att_lines([broken]), [None])


class TestMcaDebugHelpers(unittest.TestCase):
    def test_source_text_reads_and_caches(self):
        from perf.mca import _source_text

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.c"
            path.write_text("one\ntwo\n")
            cache = {}
            self.assertEqual(_source_text(str(path), 2, cache), "two")
            self.assertIn(str(path), cache)
            self.assertEqual(_source_text(str(path), "1", cache), "one")
            self.assertEqual(_source_text(str(path), 0, cache), "")
            self.assertEqual(_source_text(str(path), "bad", cache), "")
            self.assertEqual(_source_text(str(path), 10, cache), "")
            self.assertEqual(_source_text(str(Path(tmp) / "missing"), 1, {}), "")

    def test_debug_rows(self):
        from perf.mca import _debug_rows

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.c"
            path.write_text("one\ntwo\n")
            rows = _debug_rows(
                [0x1000, "bad", 0x2000, 0x3000],
                {
                    0x1000: (str(path), 1),
                    0x3000: (str(path), 99),
                    0x4000: (str(path), "bad"),
                },
            )
        self.assertEqual(rows[0], (str(path), 1, "one"))
        self.assertIsNone(rows[1])
        self.assertIsNone(rows[2])
        self.assertEqual(rows[3], (str(path), 99, ""))

    def test_debug_entries_filters_bad_entries(self):
        from perf.mca import _debug_entries

        def project_with(table):
            return SimpleNamespace(
                loader=SimpleNamespace(main_object=SimpleNamespace(addr_to_line=table))
            )

        project = project_with(
            {
                0x1000: [("a.c", 3)],
                "bad": [("b.c", 4)],
                0x2000: [],
                0x3000: [("c.c", "bad")],
                0x4000: [("d.c", 5)],
            }
        )
        self.assertEqual(
            _debug_entries(project),
            {0x1000: ("a.c", 3), 0x4000: ("d.c", 5)},
        )
        self.assertEqual(_debug_entries(SimpleNamespace(loader=None)), {})
        broken = project_with(
            SimpleNamespace(items=lambda: (_ for _ in ()).throw(ValueError()))
        )
        self.assertEqual(_debug_entries(broken), {})


class TestMcaAddressMapping(unittest.TestCase):
    def test_target_lowmap_groups_by_page_offset(self):
        from perf.mca import _target_lowmap

        self.assertEqual(
            _target_lowmap([0x401010, 0x402010, "bad", None]),
            {0x10: [0x401010, 0x402010]},
        )
        self.assertEqual(_target_lowmap(None), {})

    def test_infer_biases_and_threshold(self):
        from perf.mca import _infer_biases

        targets = [0x401010, 0x402020]
        self.assertEqual(_infer_biases(["0x501010"], targets), [(0x100000, 1)])
        self.assertEqual(
            _infer_biases([0x501010, "bad", None], targets), [(0x100000, 1)]
        )
        self.assertEqual(_infer_biases([], targets), [])
        self.assertEqual(_infer_biases([0x501010], []), [])

    def test_translate_frame_ips_by_page_bias(self):
        from perf.mca import _translate_frame_ips

        frame = pd.DataFrame(
            {"ip": [0x501010, 0x501010, 0x999999], "cycles": [1, 2, 3]}
        )
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            out = _translate_frame_ips(frame, [0x401010])
        self.assertEqual(out["ip"].tolist()[:2], [0x401010, 0x401010])
        self.assertTrue(pd.isna(out["ip"].iloc[2]))
        self.assertIn("cannot map 1 sample(s)", buf.getvalue())

    def test_translate_frame_ips_filters_by_dso(self):
        from perf.mca import _translate_frame_ips

        frame = pd.DataFrame(
            {
                "ip": [0x501010, 0x501020, 0x888888],
                "dso": ["prog", "prog", "lib"],
                "cycles": [1, 2, 3],
            }
        )
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            out = _translate_frame_ips(frame, [0x401010, 0x402020])
        self.assertEqual(out["ip"].tolist()[:2], [0x401010, 0x402020])
        self.assertTrue(pd.isna(out["ip"].iloc[2]))
        self.assertIn("cannot map 1 sample(s)", buf.getvalue())

    def test_translate_frame_ips_skips_non_ip_frames(self):
        from perf.mca import _translate_frame_ips

        frame = pd.DataFrame({"cycles": [1]})
        self.assertIs(_translate_frame_ips(frame, [0x401010]), frame)
        frame = pd.DataFrame({"address": [0x1000]})
        self.assertIs(_translate_frame_ips(frame, [0x401010]), frame)
        frame = pd.DataFrame({"ip": [0x1000]})
        self.assertIs(_translate_frame_ips(frame, []), frame)

    def test_per_ip_table_rejects_unusable_frames(self):
        from perf.mca import _per_ip_table

        self.assertIsNone(_per_ip_table(None))
        self.assertIsNone(_per_ip_table(pd.DataFrame()))
        self.assertIsNone(_per_ip_table(pd.DataFrame({"cycles": [1]})))
        self.assertIsNone(_per_ip_table(pd.DataFrame({"ip": ["bad"], "cycles": [1]})))
        self.assertIsNone(_per_ip_table(pd.DataFrame({"ip": [0x1000], "name": ["x"]})))

    def test_per_ip_table_translation_errors_are_ignored(self):
        from perf.mca import _per_ip_table

        frame = pd.DataFrame({"ip": [0x1000], "cycles": [1.0]})
        with patch("perf.mca._translate_frame_ips", side_effect=ValueError("bad")):
            out = _per_ip_table(frame, [0x1000])
        self.assertEqual(out["cycles"].tolist(), [1.0])

    def test_result_identity_and_identities(self):
        from perf.mca import _identities, _result_identity

        no_addresses = pd.DataFrame({"name": ["x"], "mode": ["latency"]})
        self.assertEqual(
            _identities(no_addresses, "prog@1", "fallback"),
            [{"file": "prog@1", "name": "x"}],
        )
        self.assertEqual(
            _identities(pd.DataFrame({"x": [1]}), "prog@1", "fallback"),
            [{"file": "prog@1", "name": "fallback"}],
        )
        keyed = pd.DataFrame({"name": ["x", None], "mode": ["latency", "latency"]})
        self.assertEqual(
            _identities(keyed, "prog@1", "fallback"),
            [
                {"file": "prog@1", "name": "x"},
                {"file": "prog@1", "name": "fallback"},
            ],
        )
        self.assertEqual(
            _result_identity([no_addresses], "prog@1", "fallback"),
            {"file": "prog@1", "name": "x"},
        )
        self.assertEqual(
            _result_identity([pd.DataFrame({"cycles": [1]})], "prog@1", "fallback"),
            {"file": "prog@1", "name": "fallback"},
        )

    def test_result_table_combines_frames(self):
        from perf.mca import _result_table

        self.assertIsNone(_result_table(None))
        first = pd.DataFrame({"ip": [0x1000], "uops": [1.0]})
        second = pd.DataFrame({"ip": [0x1000], "uops": [2.0]})
        table = _result_table([first, second])
        self.assertEqual(table.loc[0x1000, "uops"], 3.0)

    def test_merge_results_warns_about_duplicate_columns(self):
        from perf.mca import _merge_results

        df = pd.DataFrame(
            {
                "index": [0],
                "address": [0x1000],
                "assembly": ["mov"],
                "uops": [99.0],
            }
        )
        results = pd.DataFrame({"ip": [0x1000], "uops": [1.0]})
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            out = _merge_results(df, [results], "prog@1", "f")
        self.assertEqual(out["uops"].tolist(), [99.0])
        self.assertIn("measured column(s) uops", buf.getvalue())
        self.assertEqual(out["file"].tolist(), ["prog@1"])
        self.assertEqual(out["name"].tolist(), ["f"])


class TestMcaExecutionHelpers(unittest.TestCase):
    def test_state_rip_handles_solver_errors(self):
        from perf.mca import _state_rip

        state = SimpleNamespace(
            solver=SimpleNamespace(eval=lambda _r: 0x1000),
            regs=SimpleNamespace(rip=None),
        )
        self.assertEqual(_state_rip(state), 0x1000)
        broken = SimpleNamespace(
            solver=SimpleNamespace(eval=lambda _r: (_ for _ in ()).throw(ValueError())),
            regs=SimpleNamespace(rip=None),
        )
        self.assertIsNone(_state_rip(broken))

    def test_block_insn_addrs_filters_and_skips_bad_blocks(self):
        from perf.mca import _block_insn_addrs

        class Block:
            def __init__(self, addrs):
                self.capstone = SimpleNamespace(
                    insns=[SimpleNamespace(address=addr) for addr in addrs]
                )

        project = SimpleNamespace(
            factory=SimpleNamespace(
                block=lambda addr: Block([0x1000, 0x1001, 0x1002, "bad"])
            )
        )
        self.assertEqual(_block_insn_addrs(project, [0x1000], 0x1001, 0x1002), {0x1001})

        class Broken:
            @property
            def factory(self):
                raise RuntimeError("bad")

        self.assertEqual(_block_insn_addrs(Broken(), [0x1000], None, None), set())

    def test_executed_by_solution_adds_the_final_rip(self):
        from perf.mca import _executed_by_solution

        solution = {"state": object()}
        with (
            patch("perf.mca._executed_bbl_addrs", return_value=[0x1000]),
            patch("perf.mca._state_rip", return_value=0x2000),
            patch("perf.mca._block_insn_addrs", return_value={0x1000}) as blocks,
        ):
            out = _executed_by_solution(None, [solution], 0x1000, 0x3000)
        self.assertEqual(out, [{0x1000}])
        blocks.assert_called_once_with(None, [0x1000, 0x2000], 0x1000, 0x3000)

    def test_executed_by_solution_without_a_rip(self):
        from perf.mca import _executed_by_solution

        with (
            patch("perf.mca._executed_bbl_addrs", return_value=[]),
            patch("perf.mca._state_rip", return_value=None),
            patch("perf.mca._block_insn_addrs", return_value=set()) as blocks,
        ):
            self.assertEqual(
                _executed_by_solution(None, [{"state": object()}], 0, None),
                [set()],
            )
        blocks.assert_called_once_with(None, [], 0, None)

    def test_asm_instructions_reports_assembly_failures(self):
        from perf.mca import _asm_instructions

        arch = SimpleNamespace(
            normalize_asm=lambda code: code,
            _SETUP_BASE=0x1000,
            assemble=Mock(side_effect=RuntimeError("bad")),
        )
        with patch("perf.mca.get_arch", return_value=arch):
            with self.assertRaisesRegex(ValueError, "cannot assemble"):
                _asm_instructions("nop")
        arch.assemble = Mock(return_value=b"")
        with patch("perf.mca.get_arch", return_value=arch):
            with self.assertRaisesRegex(ValueError, "no instructions"):
                _asm_instructions("nop")

    def test_asm_instructions_supports_setup_and_teardown_lists(self):
        from perf.mca import _asm_instructions

        code, insns = _asm_instructions(
            "mov eax, 1",
            setup=["mov ebx, 2"],
            teardown=["nop"],
            records={},
            data=None,
        )
        self.assertEqual(code, "mov eax, 0x1;")
        self.assertEqual(insns[0].mnemonic, "mov")

    def test_explored_asm_addrs_warns_on_exploration_errors(self):
        from perf.mca import _explored_asm_addrs

        arch = get_arch()
        buf = io.StringIO()
        with (
            patch("perf.mca.explore_asm", side_effect=RuntimeError("boom")),
            patch("sys.stderr", buf),
        ):
            keep, records = _explored_asm_addrs(arch, "nop", b"\x90", 0x1000, None)
        self.assertEqual((keep, records), (set(), {}))
        self.assertIn("cannot explore", buf.getvalue())

    def test_explored_asm_addrs_collects_blocks(self):
        from perf.mca import _explored_asm_addrs

        arch = get_arch()
        solution = {"state": object()}
        with (
            patch("perf.mca.explore_asm", return_value=[solution]),
            patch("perf.mca._executed_bbl_addrs", return_value=[0x1000]),
            patch("perf.mca._state_rip", return_value=0x1001),
            patch("perf.mca._data_records", return_value={0x1000: {"data.x": {1}}}),
        ):
            keep, records = _explored_asm_addrs(arch, "nop", b"\x90\x90", 0x1000, None)
        self.assertEqual(keep, {0x1000, 0x1001})
        self.assertEqual(records, {0x1000: {"data.x": {1}}})

    def test_explored_range_falls_back_then_warns(self):
        from perf.mca import _explored_range

        spec = {"stack": {"size": 16, "align": 8}}
        buf = io.StringIO()
        with (
            patch("perf.mca._setup_asm_for", return_value="nop"),
            patch("perf.mca.explore", return_value=[]) as explore,
            patch("sys.stderr", buf),
        ):
            keep, records = _explored_range(
                None, get_arch(), spec, "f", 0, 4, {}, {}, None
            )
        self.assertEqual((keep, records), (set(), {}))
        self.assertEqual(explore.call_count, 1)
        with (
            patch("perf.mca._setup_asm_for", return_value="nop"),
            patch("perf.mca.explore", side_effect=RuntimeError("boom")),
            patch("sys.stderr", buf),
        ):
            self.assertEqual(
                _explored_range(None, get_arch(), spec, "f", 0, 4, {}, {}, "setup"),
                (set(), {}),
            )
        self.assertIn("cannot explore", buf.getvalue())

    def test_explored_range_uses_the_original_setup_string(self):
        from perf.mca import _explored_range

        calls = []

        def fake_explore(*_args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("first")
            return []

        with (
            patch("perf.mca._setup_asm_for", return_value="derived"),
            patch("perf.mca.explore", side_effect=fake_explore),
        ):
            self.assertEqual(
                _explored_range(
                    self._project(), get_arch(), {}, "f", 0, 4, {}, {}, "raw"
                ),
                (set(), {}),
            )
        self.assertEqual(calls[0]["setup_asm"], "derived")
        self.assertEqual(calls[1]["setup_asm"], "raw")
        self.assertNotIn("stack_size", calls[1])

    @staticmethod
    def _project():
        return SimpleNamespace(loader=SimpleNamespace(main_object=object()))

    def test_binary_instructions_validation(self):
        from perf.mca import _binary_instructions

        with (
            patch("perf.mca.resolve_exec", return_value="/nope"),
            patch("perf.mca.os.path.exists", return_value=False),
        ):
            with self.assertRaisesRegex(ValueError, "does not exist"):
                _binary_instructions("/nope", "main", None, None)

        project = SimpleNamespace(loader=SimpleNamespace(main_object=SimpleNamespace()))
        with (
            patch("perf.mca.resolve_exec", return_value="/prog"),
            patch("perf.mca.os.path.exists", return_value=True),
            patch("angr.Project", return_value=project),
            patch("perf.mca._merge_config", return_value={}),
        ):
            with self.assertRaisesRegex(ValueError, "symbol or region"):
                _binary_instructions("/prog", None, None, None)
            with (
                patch("perf.mca.functions", return_value=({}, {})),
                patch("perf.mca.resolve_targets", return_value=[]),
                patch(
                    "perf.mca._unresolved", return_value=ValueError("unresolved")
                ) as unresolved,
            ):
                with self.assertRaisesRegex(ValueError, "unresolved"):
                    _binary_instructions("/prog", "main", None, None)
                unresolved.assert_called_once()
            with (
                patch("perf.mca.functions", return_value=({}, {})),
                patch(
                    "perf.mca.resolve_targets",
                    return_value=[("a", 0x1000, 0x1010), ("b", 0x2000, 0x2010)],
                ),
                patch(
                    "perf.mca._unresolved", return_value=ValueError("ambiguous")
                ) as unresolved,
            ):
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    _binary_instructions("/prog", "main", None, None)
                self.assertEqual(unresolved.call_args.kwargs, {"ambiguous": True})
            with (
                patch("perf.mca.functions", return_value=({}, {})),
                patch(
                    "perf.mca.resolve_targets",
                    return_value=[("a", 0x1000, 0x1000)],
                ),
            ):
                with self.assertRaisesRegex(ValueError, "empty region"):
                    _binary_instructions("/prog", "main", None, None)
            project.loader.memory = SimpleNamespace(
                load=Mock(side_effect=RuntimeError("unmapped"))
            )
            with (
                patch("perf.mca.functions", return_value=({}, {})),
                patch(
                    "perf.mca.resolve_targets",
                    return_value=[("a", 0x1000, 0x1010)],
                ),
            ):
                with self.assertRaisesRegex(ValueError, "cannot read"):
                    _binary_instructions("/prog", "main", None, None)


class TestMcaDataRecords(unittest.TestCase):
    def test_data_records_tolerates_a_broken_harness(self):
        from perf.mca import _data_records

        class Arch:
            @property
            def _HARNESS_REGS(self):
                raise RuntimeError("bad")

        solution = {"state": object(), "inputs": {}}
        self.assertEqual(_data_records(Arch(), [solution], [{0x1000}]), {})

    def test_data_records_skips_empty_execution_sets(self):
        from perf.mca import _data_records

        arch = SimpleNamespace(_HARNESS_REGS=())
        with (
            patch("perf.mca._record_regs") as regs,
            patch("perf.mca._record_memory") as mem,
        ):
            self.assertEqual(_data_records(arch, [{}], [set()]), {})
        regs.assert_not_called()
        mem.assert_not_called()
