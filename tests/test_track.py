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
import signal
import struct
import subprocess
import sys
import sys as _sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from perf.arch.x86_64 import (
    JMP_OP,
    PATCH_SIZE,
    patch_jmp,
)
from perf.arch.x86_64 import (
    TRACK_RING_HDR_SIZE as RING_HDR_SIZE,
)
from perf.arch.x86_64 import (
    TRACK_RING_MAGIC as RING_MAGIC,
)
from perf.arch.x86_64 import (
    build_track_detour as build_trampoline,
)
from perf.arch.x86_64 import (
    track_ring_layout as ring_layout,
)
from perf.core import _split_list as split_events
from perf.track import (
    DEFAULT_BUFFER_SIZE,
    MAP_ANONYMOUS,
    MAP_FIXED,
    MAP_PRIVATE,
    MAP_SHARED,
    PROT_EXEC,
    PROT_READ,
    PROT_WRITE,
    PTRACE_CONT,
    PTRACE_DETACH,
    PTRACE_GETREGS,
    PTRACE_PEEKTEXT,
    PTRACE_POKETEXT,
    PTRACE_SETREGS,
    PTRACE_TRACEME,
    RingReader,
    UserRegs,
    _libc,
    _resolve_single_side,
    getregs,
    peek,
    poke,
    ptrace,
    read_mem,
    remote_mmap,
    remote_syscall,
    setregs,
    signed_rax,
    track,
    waitpid,
    write_mem,
)

_track_mod = _sys.modules.get("perf.track")


def _cli():
    path = Path(__file__).resolve().parent.parent / "bin" / "perf"
    loader = importlib.machinery.SourceFileLoader("perfcli_track", str(path))
    spec = importlib.util.spec_from_loader("perfcli_track", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


cli = _cli()
LIB = Path(__file__).resolve().parent.parent / "lib" / "perf"

C_SRC = r"""
#include "perf.hpp"
#include <stdio.h>
int work(int n) {
    volatile int s = 0;
    PERF_LABEL(hot);
    for (int i = 0; i < n; i++) s += i;
    PERF_LABEL(cold);
    return s;
}
int main(void) {
    PERF_LABEL(main_begin);
    int r = work(100);
    printf("%d\n", r);
    PERF_LABEL(main_end);
    return 0;
}
"""

HAVE_CXX = shutil.which("g++") is not None


def _needs_perf(event="cycles"):
    from perf import core

    try:
        c = core.open_event(event)
    except Exception as ex:
        raise unittest.SkipTest(f"perf event {event!r} unavailable: {ex}")
    try:
        c.close()
    except Exception:
        pass


C_SRC_TWICE = C_SRC.replace("int r = work(100);", "int r = work(100) + work(100);")


def _build(tmp, src=C_SRC, name="a.out"):
    src_p = Path(tmp) / "a.c"
    out_p = Path(tmp) / name
    src_p.write_text(src)
    r = subprocess.run(
        ["g++", "-O2", "-I", str(LIB), str(src_p), "-o", str(out_p)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return str(out_p)


class TestLabelLibs(unittest.TestCase):
    def test_hpp_label(self):
        text = (LIB / "perf.hpp").read_text()
        self.assertIn("#define PERF_LABEL(name)", text)
        self.assertIn(".perf.label", text)
        self.assertNotIn("PERF_MARKER", text)

    def test_rs_label(self):
        text = (LIB / "perf.rs").read_text()
        self.assertIn("perf_label", text)
        self.assertIn(".perf.label", text)
        self.assertNotIn("perf_marker", text)

    def test_zig_label(self):
        text = (LIB / "perf.zig").read_text()
        self.assertIn("perf_label", text)
        self.assertIn(".perf.label", text)
        self.assertNotIn("perf_marker", text)


class TestPatch(unittest.TestCase):
    def test_jmp_roundtrip(self):
        for patch_addr, tramp in ((0x40118C, 0x7000000), (0x7000000, 0x40118C)):
            p = patch_jmp(patch_addr, tramp)
            self.assertEqual(len(p), PATCH_SIZE)
            self.assertEqual(p[0], JMP_OP)
            (rel,) = struct.unpack("<i", bytes(p)[1:])
            self.assertEqual(patch_addr + PATCH_SIZE + rel, tramp)

    def test_jmp_range(self):
        with self.assertRaises(ValueError):
            patch_jmp(0x0, 2**31 + 5)
        self.assertNotEqual((b"\x90" * 5)[0], JMP_OP)

    def test_split_events(self):
        self.assertEqual(
            split_events(["cycles,branch-misses", "instructions"]),
            ["cycles", "branch-misses", "instructions"],
        )
        self.assertEqual(split_events(None), [])


class TestRing(unittest.TestCase):
    def test_layout_pow2(self):
        lay = ring_layout(2, slots=100)
        self.assertEqual(lay["capacity"], 128)
        self.assertEqual(lay["stride"], 8 * (2 + 2))
        self.assertEqual(lay["total"], lay["data_off"] + 128 * lay["stride"])
        self.assertEqual(ring_layout(1)["capacity"], DEFAULT_BUFFER_SIZE)

    def test_reader_roundtrip(self):
        events = ["cycles", "branch-misses"]
        lay = ring_layout(len(events), slots=8)
        blob = bytearray(lay["total"])
        struct.pack_into(
            "<QQQQQ",
            blob,
            0,
            RING_MAGIC,
            3,
            lay["capacity"],
            lay["stride"],
            len(events),
        )
        for seq, mid, vals in ((0, 0, (10, 20)), (1, 1, (11, 21)), (2, 0, (12, 22))):
            off = RING_HDR_SIZE + (seq % lay["capacity"]) * lay["stride"]
            struct.pack_into("<IIQ", blob, off, mid, 0, seq)
            for i, v in enumerate(vals):
                struct.pack_into("<Q", blob, off + 16 + 8 * i, v)
        recs = RingReader(bytes(blob), events).records()
        self.assertEqual(len(recs), 3)
        self.assertEqual(recs[0]["point_id"], 0)
        self.assertEqual(recs[0]["cycles"], 10)
        self.assertEqual(recs[2]["branch-misses"], 22)

    def test_reader_bad_magic(self):
        self.assertEqual(RingReader(b"\x00" * 128, ["cycles"]).records(), [])


class TestTrampoline(unittest.TestCase):
    def test_hw_events_use_rdpmc(self):
        lay = ring_layout(2)
        blob = build_trampoline(
            3, ["cycles", "branch-misses"], [1, 2], 0x6000000, 0x7000000, 0x401191, lay
        )
        self.assertIn(b"\x0f\x33", blob)
        want = b"\xe9" + struct.pack("<i", 0x401191 - (0x7000000 + len(blob)))
        self.assertTrue(blob.endswith(want))
        import capstone

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        mnems = [i.mnemonic for i in md.disasm(blob, 0x7000000)]
        self.assertIn("rdpmc", mnems)
        self.assertIn("jmp", mnems)
        self.assertEqual(mnems.count("rdpmc"), 2)

    def test_duration_time_uses_rdtsc(self):
        lay = ring_layout(1)
        blob = build_trampoline(
            0, ["duration_time"], [None], 0x6000000, 0x7000000, 0x401191, lay
        )
        self.assertIn(b"\x0f\x31", blob)
        self.assertNotIn(b"\x0f\x33", blob)

    def test_missing_index_raises(self):
        with self.assertRaises(ValueError):
            build_trampoline(
                0, ["cycles"], [None], 0x6000000, 0x7000000, 0x401191, ring_layout(1)
            )


@unittest.skipUnless(HAVE_CXX, "g++ required to build the label fixture")
class TestLabelsElf(unittest.TestCase):
    def test_find_labels(self):
        from perf.track import _find_trackable

        with tempfile.TemporaryDirectory() as tmp:
            exe = _build(tmp)
            tb = _find_trackable(exe)
            names = [m["name"] for m in tb["labels"]]
            self.assertIn("hot", names)
            self.assertIn("cold", names)
            self.assertIn("main_begin", names)
            self.assertIn("main_end", names)

    def test_no_labels(self):
        from perf.track import _find_trackable

        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "plain.c"
            plain.write_text("int main(void){return 0;}\n")
            exe = str(Path(tmp) / "plain")
            r = subprocess.run(
                ["g++", "-O2", str(plain), "-o", exe], capture_output=True, text=True
            )
            assert r.returncode == 0, r.stderr
            tb = _find_trackable(exe)
            self.assertEqual(tb["labels"], [])
            self.assertNotEqual(tb["functions"], {})


class TestTrackCmd(unittest.TestCase):
    def test_cmd_registered(self):
        self.assertTrue(callable(cli.track_cmd))

    def _args(self, **kw):
        return SimpleNamespace(
            command=kw.get("command", []),
            event=kw.get("event", None),
            filter=kw.get("filter", None),
            output=kw.get("output", "track.json"),
            buffer_size=kw.get("buffer_size", 64),
            interactive=False,
        )

    def test_missing_command(self):
        with self.assertRaises(SystemExit):
            cli.track_cmd(self._args(command=[]))

    def test_bad_slots(self):
        with self.assertRaises(SystemExit):
            cli.track_cmd(self._args(command=["x"], buffer_size=0))

    @unittest.skipUnless(HAVE_CXX, "g++ required to build the label fixture")
    def test_cli(self):
        _needs_perf("cycles")
        with tempfile.TemporaryDirectory() as tmp:
            exe = _build(tmp)
            out = str(Path(tmp) / "track.json")
            buf = io.StringIO()
            with patch("sys.stderr", buf):
                cli.track_cmd(
                    self._args(
                        command=[exe], output=out, event=["cycles,branch-misses"]
                    )
                )
            self.assertIn("tracked ", buf.getvalue())
            payload = json.loads(Path(out).read_text())
            self.assertGreaterEqual(len(payload["output"]), 2)
            names = [r.get("name") for r in payload["output"]]
            self.assertIn("hot..cold", names)

    @unittest.skipUnless(HAVE_CXX, "g++ required to build the label fixture")
    def test_cli_region_writes_file(self):
        _needs_perf("cycles")
        with tempfile.TemporaryDirectory() as tmp:
            exe = _build(tmp, src=C_SRC_TWICE)
            out = str(Path(tmp) / "track.json")
            buf = io.StringIO()
            with patch("sys.stderr", buf):
                cli.track_cmd(
                    self._args(
                        command=[exe],
                        output=out,
                        event=["cycles"],
                        filter=["hot", "cold"],
                    )
                )
            self.assertIn(f"tracked 2 samples -> {out}", buf.getvalue())
            payload = json.loads(Path(out).read_text())
            self.assertEqual(len(payload["output"]), 2)
            self.assertEqual(payload["output"][0]["name"], "hot..cold")

    @unittest.skipUnless(HAVE_CXX, "g++ required to build the label fixture")
    def test_cli_output_none_prints_table(self):
        _needs_perf("cycles")
        with tempfile.TemporaryDirectory() as tmp:
            exe = _build(tmp, src=C_SRC_TWICE)
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                cli.track_cmd(
                    self._args(
                        command=[exe],
                        output="none",
                        event=["cycles"],
                        filter=["hot", "cold"],
                    )
                )
            self.assertIn("hot..cold", buf.getvalue())

    @unittest.skipUnless(HAVE_CXX, "g++ required to build the label fixture")
    def test_cli_region_nomatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = _build(tmp)
            with self.assertRaises(SystemExit):
                cli.track_cmd(
                    self._args(
                        command=[exe],
                        output="none",
                        event=["cycles"],
                        filter=["nomatch"],
                    )
                )


class TestIncludeLabels(unittest.TestCase):
    def _labels(self):
        return [
            {"name": "hot", "addr": 0x1000, "kind": "label"},
            {"name": "cold", "addr": 0x2000, "kind": "label"},
            {"name": "hot_loop", "addr": 0x3000, "kind": "label"},
        ]

    def test_exact_name_included(self):
        got = _resolve_single_side("cold", self._labels())
        self.assertEqual(got["name"], "cold")

    def test_pattern_is_exact_only(self):
        self.assertIsNone(_resolve_single_side("hot.*", self._labels()))

    def test_blank_ignored(self):
        self.assertIsNone(_resolve_single_side("", self._labels()))
        self.assertIsNone(_resolve_single_side("  ", self._labels()))

    def test_hex_address(self):
        got = _resolve_single_side("0x401000", self._labels())
        self.assertEqual(got["addr"], 0x401000)
        self.assertEqual(got["kind"], "label")

    @unittest.skipUnless(HAVE_CXX, "g++ required to build the label fixture")
    def test_track_include(self):
        _needs_perf("cycles")
        with tempfile.TemporaryDirectory() as tmp:
            exe = _build(tmp, src=C_SRC_TWICE)
            out = str(Path(tmp) / "track.json")
            df = track(
                [exe],
                event=["cycles"],
                filter=["hot"],
                output=out,
            )
            self.assertEqual(df["name"].tolist(), ["hot..hot"])
            self.assertEqual(df["operations"].tolist(), [1])
            payload = json.loads(Path(out).read_text())
            self.assertEqual(len(payload["output"]), 1)
            self.assertEqual(payload["output"][0]["name"], "hot..hot")
            self.assertEqual(payload["output"][0]["operations"], 1)

    @unittest.skipUnless(HAVE_CXX, "g++ required to build the label fixture")
    def test_track_include_nomatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = _build(tmp)
            with self.assertRaises(SystemExit):
                track(
                    [exe],
                    event=["cycles"],
                    filter=["nomatch"],
                    output=None,
                )


class TestPairDeltas(unittest.TestCase):
    def test_begin_end_deltas(self):
        from perf.track import _pair_deltas

        hits = [
            {"name": "foo_begin", "seq": 0, "cycles": 100, "instructions": 50},
            {"name": "foo_end", "seq": 1, "cycles": 150, "instructions": 80},
            {"name": "foo_begin", "seq": 2, "cycles": 200, "instructions": 100},
            {"name": "foo_end", "seq": 3, "cycles": 260, "instructions": 140},
        ]
        rows = _pair_deltas(hits, ["cycles", "instructions"])
        self.assertEqual(
            rows,
            [
                {
                    "name": "foo_begin..foo_end",
                    "samples": 0,
                    "operations": 1,
                    "cycles": 50,
                    "instructions": 30,
                },
                {
                    "name": "foo_begin..foo_end",
                    "samples": 1,
                    "operations": 1,
                    "cycles": 60,
                    "instructions": 40,
                },
            ],
        )

    def test_unmatched_end_skipped(self):
        from perf.track import _pair_deltas

        hits = [
            {"name": "foo_end", "seq": 5, "cycles": 150},
            {"name": "foo_begin", "seq": 6, "cycles": 200},
            {"name": "foo_end", "seq": 7, "cycles": 260},
        ]
        rows = _pair_deltas(hits, ["cycles"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "foo_begin..foo_end")
        self.assertEqual(rows[0]["operations"], 1)
        self.assertEqual(rows[0]["cycles"], 60)

    def test_counter_wrap(self):
        from perf.track import _pair_deltas

        hits = [
            {"name": "a", "seq": 0, "cycles": (1 << 64) - 5},
            {"name": "b", "seq": 1, "cycles": 10},
        ]
        rows = _pair_deltas(hits, ["cycles"])
        self.assertEqual(rows[0]["cycles"], 15)
        self.assertEqual(rows[0]["operations"], 1)

    def test_plain_labels_pair_consecutive(self):
        from perf.track import _pair_deltas

        hits = [
            {"name": "hot", "seq": 0, "cycles": 10},
            {"name": "cold", "seq": 1, "cycles": 25},
        ]
        rows = _pair_deltas(hits, ["cycles"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "hot..cold")
        self.assertEqual(rows[0]["cycles"], 15)

    def test_function_region_named_by_symbol(self):
        from perf.track import _pair_deltas

        hits = [
            {"name": "fizz_buzz(int)", "kind": "func_entry", "seq": 0, "cycles": 100},
            {"name": "fizz_buzz(int)", "kind": "func_exit", "seq": 1, "cycles": 150},
            {"name": "fizz_buzz(int)", "kind": "func_entry", "seq": 2, "cycles": 200},
            {"name": "fizz_buzz(int)", "kind": "func_exit", "seq": 3, "cycles": 260},
        ]
        rows = _pair_deltas(hits, ["cycles"])
        self.assertEqual([r["name"] for r in rows], ["fizz_buzz(int)"] * 2)
        self.assertEqual([r["samples"] for r in rows], [0, 1])
        self.assertEqual([r["cycles"] for r in rows], [50, 60])

    def test_function_nested_lifo(self):
        from perf.track import _pair_deltas

        hits = [
            {"name": "outer", "kind": "func_entry", "seq": 0, "cycles": 10},
            {"name": "inner", "kind": "func_entry", "seq": 1, "cycles": 20},
            {"name": "inner", "kind": "func_exit", "seq": 2, "cycles": 35},
            {"name": "outer", "kind": "func_exit", "seq": 3, "cycles": 60},
        ]
        rows = _pair_deltas(hits, ["cycles"])
        self.assertEqual(
            {r["name"]: r["cycles"] for r in rows}, {"inner": 15, "outer": 50}
        )

    def test_perkind_overhead_subtraction(self):
        from perf.track import _pair_deltas

        hits = [
            {"name": "a", "kind": "label", "seq": 0, "cycles": 200},
            {"name": "b", "kind": "label", "seq": 1, "cycles": 300},
            {"name": "f", "kind": "func_entry", "seq": 2, "cycles": 400},
            {"name": "f", "kind": "func_exit", "seq": 3, "cycles": 600},
        ]
        overhead = {"label": {"cycles": 20}, "func": {"cycles": 40}}
        rows = _pair_deltas(hits, ["cycles"], overhead=overhead)
        self.assertEqual([r["cycles"] for r in rows], [80, 160])


class TestTrackEnvelope(unittest.TestCase):
    def test_payload_bench_like(self):
        from perf.track import _payload

        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "a.out"
            exe.write_bytes(b"\x7fELF-track-test")
            rows = [
                {
                    "name": "foo_begin..foo_end",
                    "samples": 0,
                    "operations": 1,
                    "cycles": 50,
                }
            ]
            payload = _payload(str(exe), [str(exe)], ["cycles"], rows)
        self.assertIn("@", payload["file"])
        self.assertTrue(payload["file"].startswith("a.out@"))
        self.assertIn("foo_begin..foo_end", payload["name"])
        self.assertIn("time", payload)
        self.assertIn("cpu", payload["info"])
        self.assertIn("output", payload)
        self.assertEqual(payload["output"], rows)
        self.assertEqual(payload.get("mode"), "record")

    def test_dump_results_writes_given_path_with_file_and_mode(self):
        from perf.track import _dump_results

        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "a.out"
            exe.write_bytes(b"\x7fELF-track-test")
            ring = str(Path(tmp) / "empty.ring")
            Path(ring).write_bytes(b"\x00" * 64)
            out = str(Path(tmp) / "foo.json")
            df = _dump_results(
                ring, ["cycles"], str(exe), [str(exe)], [], out, overhead=None
            )
            payload = json.loads(Path(out).read_text())
            self.assertIn("output", payload)
            self.assertEqual(payload.get("mode"), "record")
            self.assertIn("file", df.columns)
            self.assertIn("mode", df.columns)

    @unittest.skipUnless(HAVE_CXX, "g++ required to build the label fixture")
    def test_run_begin_end_region(self):
        _needs_perf("cycles")
        src = (
            C_SRC.replace("PERF_LABEL(hot);", "PERF_LABEL(foo_begin);")
            .replace("PERF_LABEL(cold);", "PERF_LABEL(foo_end);")
            .replace("    PERF_LABEL(main_begin);\n", "")
            .replace("    PERF_LABEL(main_end);\n", "")
        )
        with tempfile.TemporaryDirectory() as tmp:
            exe = _build(tmp, src=src)
            out = str(Path(tmp) / "track.json")
            df = track(
                [exe],
                event=["cycles"],
                filter=[("foo_begin", "foo_end")],
                output=out,
            )
            self.assertEqual(df["name"].tolist(), ["foo_begin..foo_end"])
            payload = json.loads(Path(out).read_text())
            self.assertEqual(payload["output"][0]["name"], "foo_begin..foo_end")


LONG_SRC = r"""
#include "perf.hpp"
#include <stdio.h>
int main(void) {
    for (long i = 0; i < 20000000L; i++) {
        PERF_LABEL(tick);
        volatile long s = 0;
        for (long j = 0; j < 2000; j++) s += j;
    }
    printf("done\n");
    return 0;
}
"""


def _ring_head(path):
    with open(path, "rb") as fh:
        blob = fh.read(40)
    if len(blob) < 40:
        return 0
    return struct.unpack_from("<Q", blob, 8)[0]


@unittest.skipUnless(HAVE_CXX, "g++ required to build the label fixture")
class TestTrackInterrupt(unittest.TestCase):
    def _run_and_interrupt(self, sig):
        import glob

        bin_perf = str(Path(__file__).resolve().parent.parent / "bin" / "perf")
        with tempfile.TemporaryDirectory() as tmp:
            exe = _build(tmp, src=LONG_SRC, name="long.out")
            out = str(Path(tmp) / "track.json")
            known_rings = set(glob.glob("/tmp/perf-track-*.ring"))
            proc = subprocess.Popen(
                [
                    sys.executable,
                    bin_perf,
                    "track",
                    "-e",
                    "duration_time",
                    "-o",
                    out,
                    "--",
                    exe,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            signaled = False
            try:

                def _fresh_rings():
                    return sorted(
                        set(glob.glob("/tmp/perf-track-*.ring")) - known_rings,
                        key=os.path.getmtime,
                    )

                deadline = time.time() + 180
                while time.time() < deadline:
                    if proc.poll() is not None:
                        break
                    rings = _fresh_rings()
                    if rings:
                        try:
                            if _ring_head(rings[-1]) > 5000:
                                break
                        except OSError:
                            pass
                    time.sleep(0.5)
                if proc.poll() is None:
                    hits = 0
                    rings = _fresh_rings()
                    if rings:
                        try:
                            hits = _ring_head(rings[-1])
                        except OSError:
                            hits = 0
                    if hits > 5000:
                        proc.send_signal(sig)
                        signaled = True
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                for stream in (proc.stdout, proc.stderr):
                    try:
                        if stream is not None:
                            stream.close()
                    except Exception:
                        pass
            if not signaled and proc.returncode == 0:
                self.skipTest("tracked run finished before interrupt")
            if not os.path.exists(out):
                self.skipTest("live ptrace track not supported in this env")
            payload = json.loads(Path(out).read_text())
            self.assertIn("output", payload)
            self.assertGreater(len(payload["output"]), 0)
            if signaled:
                self.assertEqual(proc.returncode, 130)

    def test_sigint_dumps_partial_json(self):
        self._run_and_interrupt(signal.SIGINT)

    def test_sigterm_dumps_partial_json(self):
        self._run_and_interrupt(signal.SIGTERM)


class _FakeRegs(UserRegs):
    def __init__(self, **kw):
        super().__init__()
        for k, v in kw.items():
            setattr(self, k, v)


class TestPtraceConstants(unittest.TestCase):
    def test_ptrace_requests(self):
        self.assertEqual(PTRACE_TRACEME, 0)
        self.assertEqual(PTRACE_PEEKTEXT, 2)
        self.assertEqual(PTRACE_POKETEXT, 4)
        self.assertEqual(PTRACE_CONT, 7)
        self.assertEqual(PTRACE_GETREGS, 12)
        self.assertEqual(PTRACE_SETREGS, 13)
        self.assertEqual(PTRACE_DETACH, 17)

    def test_memory_protection(self):
        self.assertEqual(PROT_READ | PROT_WRITE | PROT_EXEC, 7)
        self.assertEqual(MAP_SHARED, 0x01)
        self.assertEqual(MAP_PRIVATE, 0x02)
        self.assertEqual(MAP_ANONYMOUS, 0x20)
        self.assertEqual(MAP_FIXED, 0x10)

    def test_user_regs_layout(self):
        import ctypes as _ct

        self.assertEqual(len(UserRegs._fields_), 27)
        self.assertEqual(UserRegs._fields_[0][0], "r15")
        self.assertEqual(_ct.sizeof(UserRegs), 27 * 8)


class TestPtraceLibc(unittest.TestCase):
    def test_cached(self):
        first = _libc()
        self.assertIs(first, _libc())
        self.assertTrue(hasattr(first, "ptrace"))


class TestPtraceCalls(unittest.TestCase):
    def test_int_data_success(self):
        from unittest.mock import Mock

        libc = Mock()
        libc.ptrace.return_value = 0
        with patch.object(_track_mod, "_libc", return_value=libc):
            self.assertEqual(ptrace(7, 123, 0, 0), 0)

    def test_int_data_error(self):
        from unittest.mock import Mock

        libc = Mock()
        libc.ptrace.return_value = -1
        with (
            patch.object(_track_mod, "_libc", return_value=libc),
            patch.object(_track_mod.ctypes, "get_errno", return_value=22),
        ):
            with self.assertRaises(OSError):
                ptrace(7, 123, 0, 0)

    def test_peek_masks_to_u64(self):
        from unittest.mock import Mock

        libc = Mock()
        libc.ptrace.return_value = -1
        with (
            patch.object(_track_mod, "_libc", return_value=libc),
            patch.object(_track_mod.ctypes, "get_errno", return_value=0),
        ):
            self.assertEqual(peek(123, 0x1000), 0xFFFFFFFFFFFFFFFF)

    def test_peek_error(self):
        from unittest.mock import Mock

        libc = Mock()
        libc.ptrace.return_value = -1
        with (
            patch.object(_track_mod, "_libc", return_value=libc),
            patch.object(_track_mod.ctypes, "get_errno", return_value=5),
        ):
            with self.assertRaises(OSError):
                peek(123, 0x1000)

    def test_poke_masks_word(self):
        with patch.object(_track_mod, "ptrace", return_value=0) as pt:
            poke(123, 0x1000, 1 << 70)
            _, _, _, word = pt.call_args[0]
            self.assertEqual(word, 0)


class TestPtraceMem(unittest.TestCase):
    def test_read_mem_alignment(self):
        words = {0x1000: 0x1122334455667788, 0x1008: 0x99AABBCCDDEEFF00}
        with patch.object(_track_mod, "peek", side_effect=lambda p, a: words[a]):
            out = read_mem(123, 0x1002, 10)
        self.assertEqual(out, bytes.fromhex("66554433221100ffeedd"))

    def test_write_mem_roundtrip(self):
        store = {0x1000: 0x1122334455667788, 0x1008: 0x99AABBCCDDEEFF00}
        with (
            patch.object(_track_mod, "peek", side_effect=lambda p, a: store[a]),
            patch.object(
                _track_mod, "poke", side_effect=lambda p, a, w: store.__setitem__(a, w)
            ),
        ):
            write_mem(123, 0x1002, b"\xde\xad\xbe\xef")
        blob = b"".join(store[a].to_bytes(8, "little") for a in (0x1000, 0x1008))[2:14]
        self.assertEqual(blob[0:4], b"\xde\xad\xbe\xef")


class TestPtraceRegs(unittest.TestCase):
    def test_getregs_setregs(self):
        with patch.object(_track_mod, "ptrace", return_value=0) as pt:
            regs = getregs(123)
            self.assertIsInstance(regs, UserRegs)
            setregs(123, regs)
        reqs = [c[0][0] for c in pt.call_args_list]
        self.assertEqual(reqs, [PTRACE_GETREGS, PTRACE_SETREGS])

    def test_waitpid(self):
        with patch("os.waitpid", return_value=(123, 0x7F)) as w:
            self.assertEqual(waitpid(123), 0x7F)
            w.assert_called_once_with(123, 0)

    def test_signed_rax(self):
        self.assertEqual(signed_rax(42), 42)
        self.assertEqual(signed_rax((1 << 64) - 1), -1)
        self.assertEqual(signed_rax(1 << 63), -(1 << 63))


class TestPtraceRemote(unittest.TestCase):
    def test_remote_syscall(self):
        entry = _FakeRegs(rip=0x400000)
        result = _FakeRegs(rip=0x400000, rax=7)
        written = {}
        with (
            patch.object(_track_mod, "getregs", side_effect=[entry, result]),
            patch.object(_track_mod, "setregs") as sr,
            patch.object(_track_mod, "read_mem", return_value=b"\x90" * 8),
            patch.object(
                _track_mod,
                "write_mem",
                side_effect=lambda p, a, d: written.setdefault(a, d),
            ),
            patch.object(_track_mod, "ptrace", return_value=0),
            patch.object(_track_mod, "waitpid", return_value=0),
        ):
            self.assertEqual(remote_syscall(123, 9, rdi=1, anchor=0x400000), 7)
        self.assertEqual(sr.call_count, 2)
        self.assertIn(0x400000, written)

    def test_remote_mmap(self):
        entry = _FakeRegs(rip=0x400000)
        result = _FakeRegs(rip=0x400000, rax=0x7000000)
        with (
            patch.object(_track_mod, "getregs", side_effect=[entry, result]),
            patch.object(_track_mod, "setregs"),
            patch.object(_track_mod, "read_mem", return_value=b"\x90" * 8),
            patch.object(_track_mod, "write_mem"),
            patch.object(_track_mod, "ptrace", return_value=0),
            patch.object(_track_mod, "waitpid", return_value=0),
        ):
            self.assertEqual(remote_mmap(123, 8192, anchor=0x400000), 0x7000000)

    def test_remote_mmap_failure(self):
        entry = _FakeRegs(rip=0x400000)
        result = _FakeRegs(rip=0x400000, rax=(1 << 64) - 14)
        with (
            patch.object(_track_mod, "getregs", side_effect=[entry, result]),
            patch.object(_track_mod, "setregs"),
            patch.object(_track_mod, "read_mem", return_value=b"\x90" * 8),
            patch.object(_track_mod, "write_mem"),
            patch.object(_track_mod, "ptrace", return_value=0),
            patch.object(_track_mod, "waitpid", return_value=0),
        ):
            with self.assertRaises(OSError):
                remote_mmap(123, 4096, anchor=0x400000)


class TestTrackOverheadCalib(unittest.TestCase):
    def test_live_layout_changes_blob(self):

        live = ring_layout(1, 65536)
        tiny = ring_layout(1, 2)
        self.assertEqual(live["stride"], tiny["stride"])
        b_live = build_trampoline(
            0, ["cycles"], [0], 0x6000000, 0x7000000, 0x401191, live
        )
        b_tiny = build_trampoline(
            0, ["cycles"], [0], 0x6000000, 0x7000000, 0x401191, tiny
        )
        self.assertEqual(len(b_live), len(b_tiny) + 3)

    def _needs_rdpmc(self):
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

    def test_calibrate_with_live_layout(self):
        self._needs_rdpmc()
        tmod = _sys.modules.get("perf.track")
        live = ring_layout(1, 65536)
        oh = tmod.measure_track_overhead(["cycles"], layout=live)
        self.assertIn("label", oh)
        self.assertIn("func", oh)
        self.assertIn("cycles", oh["label"])
        self.assertIn("cycles", oh["func"])
        self.assertGreater(oh["label"]["cycles"], 0)
        self.assertLess(oh["label"]["cycles"], 1000)
        self.assertGreater(oh["func"]["cycles"], 0)
        self.assertLess(oh["func"]["cycles"], 2000)

    def test_calibrate_default_layout(self):
        self._needs_rdpmc()
        tmod = _sys.modules.get("perf.track")
        oh = tmod.measure_track_overhead(["cycles"])
        self.assertIn("label", oh)
        self.assertIn("func", oh)
        self.assertGreater(oh["label"]["cycles"], 0)
        self.assertLess(oh["label"]["cycles"], 1000)
        self.assertGreater(oh["func"]["cycles"], 0)
        self.assertLess(oh["func"]["cycles"], 2000)


class TestDetourArch(unittest.TestCase):
    def test_patch_len_covers_whole_insns(self):
        from perf.arch import x86_64 as _a

        code = bytes.fromhex("554889e59090")
        plen, insns = _a.detour_patch_len(code, 0x401000)
        self.assertGreaterEqual(plen, 5)
        self.assertEqual(plen, sum(i.size for i in insns))

    def test_rip_relative_fixup(self):
        from perf.arch import x86_64 as _a

        code = _a.assemble("lea rax, [rip+0x1234]", 0x401000)
        fixed = _a.relocate_detour_bytes(code, 0x401000, 0x7000000)
        self.assertEqual(len(fixed), len(code))
        import capstone

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        insns = list(md.disasm(fixed, 0x7000000))
        self.assertEqual(insns[0].mnemonic, "lea")

        self.assertEqual(
            insns[0].operands[1].mem.disp + 0x7000000 + 7, 0x401000 + 7 + 0x1234
        )

    def test_short_branch_expansion(self):
        from perf.arch import x86_64 as _a

        live = bytes.fromhex("85ff7e1931c0")
        plen, _ = _a.detour_patch_len(live, 0x401000)
        self.assertEqual(plen, 6)
        fixed = _a.relocate_detour_bytes(live[:plen], 0x401000, 0x7000000)
        self.assertGreater(len(fixed), plen)
        import capstone

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        mnems = [i.mnemonic for i in md.disasm(fixed, 0x7000000)]
        self.assertIn("jle", mnems)

    def test_detour_blob_runs_record_then_resumes(self):
        from perf.arch import x86_64 as _a

        lay = ring_layout(1)
        live = bytes.fromhex("9090909090")
        blob = _a.build_track_detour_raw(
            0, ["duration_time"], [None], 0x6000000, 0x7000000, 0x401000, 5, live, lay
        )
        import capstone

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        mnems = [i.mnemonic for i in md.disasm(blob, 0x7000000)]
        self.assertIn("rdtsc", mnems)
        self.assertEqual(mnems[-1], "jmp")

    def test_func_entry_exit_blobs(self):
        from perf.arch import x86_64 as _a

        lay = ring_layout(1)
        live = bytes.fromhex("f30f1efa554889e5")
        plen, _ = _a.detour_patch_len(live, 0x401000)
        entry = _a.build_track_func_entry_raw(
            0,
            ["duration_time"],
            [None],
            0x6000000,
            0x7000000,
            0x401000,
            plen,
            live,
            lay,
            shadow_top=0x8000000,
            exit_addr=0x7001000,
        )
        ex = _a.build_track_func_exit(
            1,
            ["duration_time"],
            [None],
            0x6000000,
            0x7001000,
            lay,
            shadow_top=0x8000000,
        )
        import capstone

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        entry_mnems = [i.mnemonic for i in md.disasm(entry, 0x7000000)]
        exit_mnems = [i.mnemonic for i in md.disasm(ex, 0x7001000)]

        self.assertIn("jmp", entry_mnems)

        self.assertEqual(exit_mnems[-1], "jmp")
        self.assertNotIn("pushfq", entry_mnems)


FUNC_LABEL_SRC = r"""
#include "perf.hpp"
#include <stdio.h>
__attribute__((noinline)) int fizz_buzz(int n) {
    volatile int s = 0;
    for (int i = 0; i < n; i++) s += i;
    return s;
}
__attribute__((noinline)) int lab_work(int n) {
    volatile int s = 0;
    PERF_LABEL(lab_begin);
    for (int i = 0; i < n; i++) s += i;
    PERF_LABEL(lab_end);
    return s;
}
int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 100;
    int r = fizz_buzz(n) + lab_work(n);
    printf("%d\n", r);
    return 0;
}
"""


def _build_with_stdlib(tmp, src, name="a.out"):
    if "#include <stdlib.h>" not in src:
        src = src.replace(
            "#include <stdio.h>", "#include <stdio.h>\n#include <stdlib.h>"
        )
    return _build(tmp, src=src, name=name)


class TestResolveTrackPoints(unittest.TestCase):
    @unittest.skipUnless(HAVE_CXX, "g++ required")
    def test_function_expands_to_entry_exit(self):
        from perf.track import _resolve_track_points

        with tempfile.TemporaryDirectory() as tmp:
            exe = _build_with_stdlib(tmp, FUNC_LABEL_SRC)
            pts = _resolve_track_points(exe, ["fizz_buzz"])
            kinds = {p["kind"] for p in pts}
            self.assertEqual(kinds, {"func_entry", "func_exit"})
            self.assertEqual({p["name"] for p in pts}, {"fizz_buzz(int)"})

    @unittest.skipUnless(HAVE_CXX, "g++ required")
    def test_function_is_not_a_region_endpoint(self):
        from perf.track import _resolve_track_points

        with tempfile.TemporaryDirectory() as tmp:
            exe = _build_with_stdlib(tmp, FUNC_LABEL_SRC)

            self.assertEqual(_resolve_track_points(exe, [("fizz_buzz", "lab_end")]), [])
            self.assertEqual(
                _resolve_track_points(exe, [("lab_begin", "fizz_buzz")]), []
            )

    @unittest.skipUnless(HAVE_CXX, "g++ required")
    def test_label_and_region(self):
        from perf.track import _resolve_track_points

        with tempfile.TemporaryDirectory() as tmp:
            exe = _build_with_stdlib(tmp, FUNC_LABEL_SRC)
            single = _resolve_track_points(exe, ["lab_begin"])
            self.assertEqual(len(single), 1)
            self.assertEqual(single[0]["kind"], "label")
            region = _resolve_track_points(exe, [("lab_begin", "lab_end")])
            self.assertEqual(len(region), 2)
            self.assertTrue(all(p["kind"] == "label" for p in region))

    @unittest.skipUnless(HAVE_CXX, "g++ required")
    def test_hex_address(self):
        from perf.track import _resolve_track_points

        with tempfile.TemporaryDirectory() as tmp:
            exe = _build_with_stdlib(tmp, FUNC_LABEL_SRC)
            pts = _resolve_track_points(exe, ["0x401000"])
            self.assertEqual(len(pts), 1)
            self.assertEqual(pts[0]["addr"], 0x401000)

    @unittest.skipUnless(HAVE_CXX, "g++ required")
    def test_default_is_labels(self):
        from perf.track import _resolve_track_points

        with tempfile.TemporaryDirectory() as tmp:
            exe = _build(tmp)
            pts = _resolve_track_points(exe, None)
            names = sorted(p["name"] for p in pts)
            for want in ["hot", "cold", "main_begin", "main_end"]:
                self.assertIn(want, names)
            kinds = {p["kind"] for p in pts}
            self.assertIn("label", kinds)
            self.assertIn("func_entry", kinds)
            self.assertIn("func_exit", kinds)

    @unittest.skipUnless(HAVE_CXX, "g++ required")
    def test_default_empty_list_is_labels(self):
        from perf.track import _resolve_track_points

        with tempfile.TemporaryDirectory() as tmp:
            exe = _build(tmp)
            pts = _resolve_track_points(exe, [])
            names = sorted(p["name"] for p in pts)
            for want in ["hot", "cold", "main_begin", "main_end"]:
                self.assertIn(want, names)
            kinds = {p["kind"] for p in pts}
            self.assertIn("label", kinds)
            self.assertIn("func_entry", kinds)

    @unittest.skipUnless(HAVE_CXX, "g++ required")
    def test_func_exit_uses_end_addr(self):
        from perf.track import _find_trackable, _resolve_track_points

        with tempfile.TemporaryDirectory() as tmp:
            exe = _build_with_stdlib(tmp, FUNC_LABEL_SRC)
            funcs = _find_trackable(exe, with_functions=True)["functions"]
            pts = _resolve_track_points(exe, ["fizz_buzz"])
            by_kind = {p["kind"]: p for p in pts}
            entry = by_kind["func_entry"]
            exit_pt = by_kind["func_exit"]
            self.assertNotEqual(entry["addr"], exit_pt["addr"])
            match = next(v for k, v in funcs.items() if "fizz_buzz" in k)
            self.assertEqual(entry["addr"], int(match[0]))
            self.assertEqual(exit_pt["addr"], int(match[1]))


@unittest.skipUnless(HAVE_CXX, "g++ required to build the fixture")
class TestTrackFunctionsLabels(unittest.TestCase):
    def test_track_function_entry_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = _build_with_stdlib(tmp, FUNC_LABEL_SRC)
            out = str(Path(tmp) / "track.json")
            df = track(
                [exe, "150"],
                event=["duration_time"],
                filter=["fizz_buzz"],
                output=out,
            )
            self.assertEqual(df["name"].tolist(), ["fizz_buzz(int)"])
            self.assertGreater(int(df["duration_time"].iloc[0]), 0)

    def test_track_label_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = _build_with_stdlib(tmp, FUNC_LABEL_SRC)
            out = str(Path(tmp) / "track.json")
            df = track(
                [exe, "150"],
                event=["duration_time"],
                filter=[("lab_begin", "lab_end")],
                output=out,
            )
            self.assertEqual(df["name"].tolist(), ["lab_begin..lab_end"])
            self.assertGreater(int(df["duration_time"].iloc[0]), 0)

    def test_track_function_cycles(self):
        _needs_perf("cycles")
        with tempfile.TemporaryDirectory() as tmp:
            exe = _build_with_stdlib(tmp, FUNC_LABEL_SRC)
            df = track(
                [exe, "150"], event=["cycles"], filter=["fizz_buzz"], output=None
            )
            self.assertEqual(df["name"].tolist(), ["fizz_buzz(int)"])
            self.assertGreater(int(df["cycles"].iloc[0]), 0)


if __name__ == "__main__":
    unittest.main()
