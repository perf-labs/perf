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
import struct
import subprocess
import sys
import sys as _sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from perf.arch.x86_64 import (
    _JMP_OP,
    _PATCH_SIZE,
    patch_jmp,
)
from perf.arch.x86_64 import (
    _TRACK_RING_HDR_SIZE as RING_HDR_SIZE,
)
from perf.arch.x86_64 import (
    _TRACK_RING_MAGIC as RING_MAGIC,
)
from perf.arch.x86_64 import (
    build_profile_detour as build_trampoline,
)
from perf.arch.x86_64 import (
    profile_ring_layout as ring_layout,
)
from perf.core import _split_list as split_events
from perf.prof import (
    _DEFAULT_BUFFER_SIZE,
    _PTRACE_CONT,
    _PTRACE_DETACH,
    _PTRACE_GETREGS,
    _PTRACE_PEEKTEXT,
    _PTRACE_POKETEXT,
    _PTRACE_SETREGS,
    _PTRACE_TRACEME,
    RingReader,
    UserRegs,
    _libc,
    _resolve_single_side,
    getregs,
    peek,
    poke,
    profile,
    ptrace,
    read_mem,
    remote_mmap,
    remote_syscall,
    setregs,
    signed_rax,
    waitpid,
    write_mem,
)

_track_mod = _sys.modules.get("perf.prof")


def _cli():
    path = Path(__file__).resolve().parent.parent / "bin" / "perf-profile"
    loader = importlib.machinery.SourceFileLoader("perfcli_profile", str(path))
    spec = importlib.util.spec_from_loader("perfcli_profile", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


cli = _cli()
LIB = Path(__file__).resolve().parent.parent / "lib" / "perf"


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


class TestLabelLibs(unittest.TestCase):
    def test_hpp_label(self):
        text = (LIB / "perf.h").read_text()
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
            self.assertEqual(len(p), _PATCH_SIZE)
            self.assertEqual(p[0], _JMP_OP)
            (rel,) = struct.unpack("<i", bytes(p)[1:])
            self.assertEqual(patch_addr + _PATCH_SIZE + rel, tramp)

    def test_jmp_range(self):
        with self.assertRaises(ValueError):
            patch_jmp(0x0, 2**31 + 5)
        self.assertNotEqual((b"\x90" * 5)[0], _JMP_OP)

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
        self.assertEqual(ring_layout(1)["capacity"], _DEFAULT_BUFFER_SIZE)

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

    def test_reader_follows_a_wrapped_ring(self):
        events = ["cycles"]
        lay = ring_layout(len(events), slots=8)
        blob = bytearray(lay["total"])
        for seq in range(8):
            off = RING_HDR_SIZE + seq * lay["stride"]
            struct.pack_into("<IIQ", blob, off, seq, 0, seq)
            struct.pack_into("<Q", blob, off + 16, 100 + seq)
        struct.pack_into(
            "<QQQQQ",
            blob,
            0,
            RING_MAGIC,
            13,
            lay["capacity"],
            lay["stride"],
            len(events),
        )
        recs = RingReader(bytes(blob), events).records()
        self.assertEqual([r["seq"] for r in recs], [5, 6, 7, 0, 1, 2, 3, 4])
        self.assertEqual(
            [r["cycles"] for r in recs], [105, 106, 107, 100, 101, 102, 103, 104]
        )

    def test_reader_reads_only_the_occupied_slots(self):
        events = ["cycles"]
        lay = ring_layout(len(events), slots=64)
        blob = bytearray(lay["total"])
        for seq in range(3):
            off = RING_HDR_SIZE + seq * lay["stride"]
            struct.pack_into("<IIQ", blob, off, 0, 0, seq)
            struct.pack_into("<Q", blob, off + 16, 7 + seq)
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
        with tempfile.NamedTemporaryFile(suffix=".ring", delete=False) as fh:
            fh.write(bytes(blob))
            path = fh.name
        self.addCleanup(os.unlink, path)
        self.assertEqual(len(RingReader(path, events).records()), 3)


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


class TestProfileCmd(unittest.TestCase):
    def test_cmd_registered(self):
        self.assertTrue(callable(cli.main))

    def _args(self, **kw):
        return SimpleNamespace(
            command=kw.get("command", []),
            event=kw.get("event", None),
            select=kw.get("select", None),
            output=kw.get("output", "profile.json"),
            buffer_size=kw.get("buffer_size", 64),
            interactive=False,
            list=kw.get("list", False),
            paths=kw.get("command", []),
        )

    def test_buffer_size_is_validated(self):
        for value in (0, -1, "many"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile(["/bin/true"], buffer_size=value)

    def test_executable_is_not_rejected_as_library(self):
        with (
            patch("perf.prof.resolve_exec", return_value="/tmp/a.out"),
            patch("perf.prof.is_shared_library", return_value=False),
            patch("perf.prof._resolve_track_points", return_value=[]),
        ):
            with self.assertRaises(SystemExit) as ctx:
                profile(["/tmp/a.out"], event=["duration_time"], output=None)
        self.assertNotIn("shared library", str(ctx.exception))


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


class TestCalibrationLayout(unittest.TestCase):
    def _check_blob(self, blob, base):
        from perf.arch import x86_64 as arch

        md = arch.disassembler()
        size = len(bytes(blob))
        self.assertGreater(size, 0)
        for insn in md.disasm(bytes(blob), base):
            if insn.mnemonic != "jmp":
                continue
            try:
                dest = int(insn.op_str.strip(), 0)
            except (TypeError, ValueError):
                continue
            self.assertGreaterEqual(
                dest, base, f"jmp at {insn.address:#x} escapes the blob"
            )
            self.assertLess(
                dest, base + size + 8, f"jmp at {insn.address:#x} escapes the blob"
            )

    def test_label_layout_stable_for_event_counts(self):
        from perf.arch import x86_64 as arch
        from perf.prof import _build_label_calibration

        for count in (1, 2, 3):
            events = ["cycles", "instructions", "branch-misses"][:count]
            lay = arch.profile_ring_layout(len(events))
            blob = _build_label_calibration(events, [0] * count, 0x1000, 0x2000, lay)
            self._check_blob(blob, 0x2000)

    def test_func_layout_stable_for_event_counts(self):
        from perf.arch import x86_64 as arch
        from perf.prof import _build_func_calibration

        for count in (1, 2, 3):
            events = ["cycles", "instructions", "branch-misses"][:count]
            lay = arch.profile_ring_layout(len(events))
            blob = _build_func_calibration(
                events, [0] * count, 0x1000, 0x2000, lay, 0x3000
            )
            self._check_blob(blob, 0x2000)

    def test_measure_pairs_reads_fresh_slots(self):
        import struct

        from perf.arch import x86_64 as arch
        from perf.prof import _measure_pairs

        stride, trials = 40, 6
        total = arch._TRACK_RING_DATA_OFF + 2 * trials * stride
        ring = bytearray(total)
        struct.pack_into("<QQQQQ", ring, 0, arch._TRACK_RING_MAGIC, 0, 65536, stride, 1)
        state = {"head": 0}

        def fn():
            for _ in range(2):
                slot = state["head"] & 65535
                off = arch._TRACK_RING_DATA_OFF + slot * stride
                struct.pack_into("<I", ring, off, 7)
                struct.pack_into(
                    "<Q",
                    ring,
                    off + arch._TRACK_SLOT_EVENTS_OFF,
                    1000 + 10 * state["head"],
                )
                state["head"] += 1
            struct.pack_into("<Q", ring, arch._TRACK_RING_HEAD_OFF, state["head"])

        out = _measure_pairs(fn, ["cycles"], ring, stride, trials)
        self.assertEqual(out["cycles"], 10)

    def test_three_event_calibration_survives(self):
        _needs_perf("cycles")
        prog = (
            "from perf.prof import _calibrate_overhead\n"
            "import json\n"
            "for layout in (None,):\n"
            "    oh = _calibrate_overhead(\n"
            "        ['cycles', 'instructions', 'branch-misses'],\n"
            "        None, False, trials=8, layout=layout,\n"
            "    )\n"
            "    print(json.dumps(oh))\n"
        )
        r = subprocess.run(
            [sys.executable, "-u", "-c", prog],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        for line in r.stdout.strip().splitlines():
            oh = json.loads(line)
            for kind in ("label", "func"):
                self.assertGreater(oh[kind]["cycles"], 0)
                self.assertLess(oh[kind]["cycles"], 100000)


class TestPairDeltas(unittest.TestCase):
    def test_begin_end_deltas(self):
        from perf.prof import _pair_deltas

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
                    "iteration": 0,
                    "operations": 1,
                    "cycles": 50,
                    "instructions": 30,
                },
                {
                    "name": "foo_begin..foo_end",
                    "iteration": 1,
                    "operations": 1,
                    "cycles": 60,
                    "instructions": 40,
                },
            ],
        )

    def test_explicit_region_pair_without_suffix_names(self):
        from perf.prof import _pair_deltas

        hits = [
            {
                "name": "lab_begin",
                "pair": "t0",
                "pair_side": "begin",
                "seq": 0,
                "cycles": 100,
            },
            {
                "name": "fizz_buzz",
                "pair": "t0",
                "pair_side": "end",
                "seq": 1,
                "cycles": 160,
            },
        ]
        rows = _pair_deltas(hits, ["cycles"])
        self.assertEqual(rows[0]["name"], "lab_begin..fizz_buzz")
        self.assertEqual(rows[0]["cycles"], 60)

    def test_nested_suffix_regions_use_a_stack(self):
        from perf.prof import _pair_deltas

        hits = [
            {"name": "x_begin", "seq": 0, "cycles": 10},
            {"name": "x_begin", "seq": 1, "cycles": 20},
            {"name": "x_end", "seq": 2, "cycles": 50},
            {"name": "x_end", "seq": 3, "cycles": 80},
        ]
        rows = _pair_deltas(hits, ["cycles"])
        self.assertEqual([row["cycles"] for row in rows], [30, 70])

    def test_unmatched_end_skipped(self):
        from perf.prof import _pair_deltas

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
        from perf.prof import _pair_deltas

        hits = [
            {"name": "a", "seq": 0, "cycles": (1 << 64) - 5},
            {"name": "b", "seq": 1, "cycles": 10},
        ]
        rows = _pair_deltas(hits, ["cycles"])
        self.assertEqual(rows[0]["cycles"], 15)
        self.assertEqual(rows[0]["operations"], 1)

    def test_plain_labels_pair_consecutive(self):
        from perf.prof import _pair_deltas

        hits = [
            {"name": "hot", "seq": 0, "cycles": 10},
            {"name": "cold", "seq": 1, "cycles": 25},
        ]
        rows = _pair_deltas(hits, ["cycles"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "hot..cold")
        self.assertEqual(rows[0]["cycles"], 15)

    def test_function_region_named_by_symbol(self):
        from perf.prof import _pair_deltas

        hits = [
            {"name": "fizz_buzz(int)", "kind": "func_entry", "seq": 0, "cycles": 100},
            {"name": "fizz_buzz(int)", "kind": "func_exit", "seq": 1, "cycles": 150},
            {"name": "fizz_buzz(int)", "kind": "func_entry", "seq": 2, "cycles": 200},
            {"name": "fizz_buzz(int)", "kind": "func_exit", "seq": 3, "cycles": 260},
        ]
        rows = _pair_deltas(hits, ["cycles"])
        self.assertEqual([r["name"] for r in rows], ["fizz_buzz(int)"] * 2)
        self.assertEqual([r["iteration"] for r in rows], [0, 1])
        self.assertEqual([r["cycles"] for r in rows], [50, 60])

    def test_function_nested_lifo(self):
        from perf.prof import _pair_deltas

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
        from perf.prof import _pair_deltas

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
        from perf.prof import _payload

        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "a.out"
            exe.write_bytes(b"\x7fELF-profile-test")
            rows = [
                {
                    "name": "foo_begin..foo_end",
                    "iteration": 0,
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

    def test_payload_names_the_command_not_its_link(self):
        from perf.prof import _payload

        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / "bundle.o"
            obj.write_bytes(b"\x7fELF-profile-test")
            link = Path(tmp) / "q.o"
            link.write_bytes(b"\x7fELF-profile-test-linked")
            rows = [
                {"name": "foo", "iteration": 0, "operations": 1, "cycles": 50},
            ]
            payload = _payload(str(link), [str(obj)], ["cycles"], rows)
        self.assertTrue(payload["file"].startswith("bundle.o@"), payload["file"])

    def test_dump_results_writes_given_path_with_file_and_mode(self):
        from perf.prof import _dump_results

        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "a.out"
            exe.write_bytes(b"\x7fELF-profile-test")
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

    def test_dump_results_drop_last_ignores_torn_sample(self):
        from perf.prof import _dump_results

        events = ["cycles"]
        lay = ring_layout(len(events), slots=8)
        blob = bytearray(lay["total"])
        struct.pack_into(
            "<QQQQQ",
            blob,
            0,
            RING_MAGIC,
            2,
            lay["capacity"],
            lay["stride"],
            len(events),
        )
        for seq, mid, vals in ((0, 0, (100,)), (1, 1, (50,))):
            off = RING_HDR_SIZE + (seq % lay["capacity"]) * lay["stride"]
            struct.pack_into("<IIQ", blob, off, mid, 0, seq)
            for i, v in enumerate(vals):
                struct.pack_into("<Q", blob, off + 16 + 8 * i, v)
        selected = [
            {"name": "foo_begin", "kind": "label", "addr": 0x401000},
            {"name": "foo_end", "kind": "label", "addr": 0x401020},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "a.out"
            exe.write_bytes(b"\x7fELF-profile-test")
            ring = str(Path(tmp) / "torn.ring")
            out = str(Path(tmp) / "profile.json")
            Path(ring).write_bytes(bytes(blob))
            df = _dump_results(ring, events, str(exe), [str(exe)], selected, out)
            self.assertEqual(len(df), 1)
            self.assertGreater(int(df["cycles"].iloc[0]), 2**63)
            Path(ring).write_bytes(bytes(blob))
            df = _dump_results(
                ring, events, str(exe), [str(exe)], selected, out, drop_last=True
            )
            self.assertEqual(len(df), 0)

    def _dump_open_ended(self, tmp, head, out=None):
        from perf.prof import _dump_results

        events = ["cycles"]
        lay = ring_layout(len(events), slots=8)
        blob = bytearray(lay["total"])
        struct.pack_into(
            "<QQQQQ",
            blob,
            0,
            RING_MAGIC,
            head,
            lay["capacity"],
            lay["stride"],
            len(events),
        )
        for seq in range(head):
            mid = 0 if seq % 2 == 0 else 1
            off = RING_HDR_SIZE + (seq % lay["capacity"]) * lay["stride"]
            struct.pack_into("<IIQ", blob, off, mid, 0, seq)
            struct.pack_into("<Q", blob, off + 16, 1000 + 10 * seq)
        exe = Path(tmp) / "a.out"
        exe.write_bytes(b"\x7fELF-profile-test")
        ring = str(Path(tmp) / "open.ring")
        Path(ring).write_bytes(bytes(blob))
        selected = [
            {"name": "loop_begin", "kind": "label"},
            {"name": "loop_end", "kind": "label"},
        ]
        return _dump_results(
            ring,
            events,
            str(exe),
            [str(exe)],
            selected,
            out,
            drop_last=True,
            open_ended=True,
        )

    def test_interrupt_keeps_inflight_sample_open(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            df = self._dump_open_ended(tmp, 1)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["name"], "loop_begin..loop_end")
        self.assertTrue(pd.isna(df.iloc[0]["cycles"]))

    def test_interrupt_pairs_closed_and_opens_inflight(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            df = self._dump_open_ended(tmp, 3)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["cycles"], 10)
        self.assertTrue(pd.isna(df.iloc[1]["cycles"]))

    def test_interrupt_dump_is_strict_json_with_null(self):
        def _reject_constant(name):
            raise ValueError(f"non-JSON constant {name!r}")

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "profile.json"
            self._dump_open_ended(tmp, 3, str(dest))
            payload = json.loads(dest.read_text(), parse_constant=_reject_constant)
        self.assertEqual(len(payload["output"]), 2)
        self.assertIsNone(payload["output"][1]["cycles"])


class _FakeRegs(UserRegs):
    def __init__(self, **kw):
        super().__init__()
        for k, v in kw.items():
            setattr(self, k, v)


class TestPtraceConstants(unittest.TestCase):
    def test_ptrace_requests(self):
        self.assertEqual(_PTRACE_TRACEME, 0)
        self.assertEqual(_PTRACE_PEEKTEXT, 2)
        self.assertEqual(_PTRACE_POKETEXT, 4)
        self.assertEqual(_PTRACE_CONT, 7)
        self.assertEqual(_PTRACE_GETREGS, 12)
        self.assertEqual(_PTRACE_SETREGS, 13)
        self.assertEqual(_PTRACE_DETACH, 17)

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
        self.assertEqual(reqs, [_PTRACE_GETREGS, _PTRACE_SETREGS])

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
        tmod = _sys.modules.get("perf.prof")
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
        tmod = _sys.modules.get("perf.prof")
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
        blob = _a.build_profile_detour_raw(
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
        entry = _a.build_profile_func_entry_raw(
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
        ex = _a.build_profile_func_exit(
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

    def test_func_entry_exit_preserve_caller_regs(self):
        import ctypes
        import mmap

        from perf.arch import x86_64 as arch
        from perf.prof import _build_func_calibration

        ev = ["duration_time"]
        lay = arch.profile_ring_layout(len(ev))
        stride = lay["stride"]
        ring = ctypes.create_string_buffer(arch._TRACK_RING_DATA_OFF + 4 * stride + 64)
        ring_addr = ctypes.addressof(ring)
        struct.pack_into(
            "<QQQQQ",
            ring,
            0,
            arch._TRACK_RING_MAGIC,
            0,
            lay["capacity"],
            lay["stride"],
            len(ev),
        )
        shpage = mmap.mmap(-1, 0x10000, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        try:
            shadow_top = ctypes.addressof(ctypes.c_char.from_buffer(shpage))
            struct.pack_into("<Q", shpage, 0, shadow_top + 8)
            mm = mmap.mmap(
                -1,
                16384,
                prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
            )
            try:
                base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                blob = _build_func_calibration(
                    ev, [None], ring_addr, base, lay, shadow_top
                )
                caddr = base + ((len(blob) + 15) & ~15)
                out = ctypes.create_string_buffer(16)
                out_addr = ctypes.addressof(out)
                check = arch.assemble(
                    f"movabs r11, 0x1111111111111111\n"
                    f"movabs rcx, 0x2222222222222222\n"
                    f"movabs rax, 0x{base:x}\n"
                    "call rax\n"
                    f"movabs rdx, 0x{out_addr:x}\n"
                    "mov [rdx], r11\n"
                    "mov [rdx+8], rcx\n"
                    "ret",
                    caddr,
                )
                mm.seek(caddr - base)
                mm.write(check)
                mm.seek(0)
                mm.write(blob)
                ctypes.CFUNCTYPE(None)(caddr)()
                got_r11, got_rcx = struct.unpack_from("<QQ", out, 0)
                self.assertEqual(got_r11, 0x1111111111111111)
                self.assertEqual(got_rcx, 0x2222222222222222)
                head = struct.unpack_from("<Q", ring, arch._TRACK_RING_HEAD_OFF)[0]
                self.assertEqual(head, 2)
            finally:
                mm.close()
        finally:
            shpage.close()

    def test_detour_prefix_uses_lea_outside_flag_window(self):
        from perf.arch import x86_64

        lay = x86_64.profile_ring_layout(1)
        lines = x86_64._detour_prefix_lines(
            0, ["duration_time"], [None], 0x1000, lay, True, None
        )
        text = "\n".join(lines)
        self.assertNotIn("sub rsp", text)
        import re

        self.assertIsNone(re.search(r"(?<![a-z])add rsp", text))
        self.assertIn("lea rsp, [rsp-128]", text)
        self.assertIn("lea rsp, [rsp+128]", text)
        self.assertIn("pushfq", text)
        self.assertIn("popfq", text)
        self.assertLess(text.index("pushfq"), text.index("lock xadd"))
        self.assertLess(text.index("lock xadd"), text.index("popfq"))
        x86_64.assemble(text, 0x2000)

    def test_func_entry_prefix_uses_lea(self):
        from perf.arch import x86_64

        lay = x86_64.profile_ring_layout(1)
        lines = x86_64._func_entry_prefix_lines(
            0, ["duration_time"], [None], 0x1000, lay, 0x2000, 0x3000, None
        )
        text = "\n".join(lines)
        self.assertIn("lea rsp", text)
        self.assertNotIn("sub rsp", text)
        import re

        self.assertIsNone(re.search(r"(?<![a-z])add rsp", text))
        x86_64.assemble(text, 0x2000)

    def test_func_exit_uses_lea(self):
        from perf.arch import x86_64

        lay = x86_64.profile_ring_layout(1)
        blob = x86_64.build_profile_func_exit(
            1,
            ["duration_time"],
            [None],
            0x1000,
            0x2000,
            lay,
            shadow_top=0x3000,
            tid_off=None,
        )
        self.assertTrue(blob)
        import capstone

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        for insn in md.disasm(blob, 0x2000):
            text = f"{insn.mnemonic} {insn.op_str}".strip().lower()
            self.assertNotIn("sub rsp", text)
            self.assertNotIn("add rsp", text)
        mnems = [i.mnemonic for i in md.disasm(blob, 0x2000)]
        self.assertIn("lea", mnems)

    def test_record_uses_lock_xadd(self):
        from perf.arch import x86_64

        lay = x86_64.profile_ring_layout(1)
        lines = x86_64._profile_record_lines(
            3, ["duration_time"], [None], 0x6000000, lay
        )
        text = "\n".join(lines).lower()
        self.assertIn("lock xadd", text)
        self.assertNotIn("mov rax, [r11", text)
        self.assertNotIn("mov [r11 + 8], rdx", text)
        self.assertNotIn("lea rdx, [rax+1]", text)

    def test_record_publishes_seq_last(self):
        from perf.arch import x86_64

        lay = x86_64.profile_ring_layout(1)
        lines = x86_64._profile_record_lines(
            3, ["duration_time"], [None], 0x6000000, lay
        )
        seq_idx = next(
            i for i, line in enumerate(lines) if "mov qword ptr [r11+8], rax" in line
        )
        event_idx = next(
            i for i, line in enumerate(lines) if "mov [r11+16], rax" in line
        )
        id_idx = next(
            i for i, line in enumerate(lines) if "mov dword ptr [r11]" in line
        )
        self.assertLess(id_idx, event_idx)
        self.assertLess(event_idx, seq_idx)
        self.assertLess(id_idx, seq_idx)

    def test_record_assembles_and_detour_builds(self):
        from perf.arch import x86_64

        lay = x86_64.profile_ring_layout(2)
        lines = x86_64._profile_record_lines(
            7, ["duration_time", "cycles"], [None, 0], 0x6000000, lay
        )
        x86_64.assemble("\n".join(lines), 0x7000000)
        blob = x86_64.build_profile_detour(
            7,
            ["duration_time", "cycles"],
            [None, 0],
            0x6000000,
            0x7000000,
            0x401191,
            lay,
        )
        self.assertTrue(blob)


class TestTidCapture(unittest.TestCase):
    def test_record_lines_with_tid(self):
        from perf.arch import x86_64 as arch

        lay = arch.profile_ring_layout(1)
        lines = arch._profile_record_lines(
            3, ["duration_time"], [None], 0x6000000, lay, tid_off=0x2D0
        )
        text = "\n".join(lines)
        self.assertIn("fs:[0x2d0]", text.lower())
        self.assertIn("[r11+4]", text)
        blob = arch.build_profile_detour(
            3,
            ["duration_time"],
            [None],
            0x6000000,
            0x7000000,
            0x401191,
            lay,
            tid_off=0x2D0,
        )
        import capstone

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        mnems = [i.mnemonic for i in md.disasm(blob, 0x7000000)]
        self.assertIn("rdtsc", mnems)

    def test_record_with_tid_still_seq_last(self):
        from perf.arch import x86_64 as arch

        lay = arch.profile_ring_layout(1)
        lines = arch._profile_record_lines(
            3, ["duration_time"], [None], 0x6000000, lay, tid_off=0x2D0
        )
        text = "\n".join(lines)
        self.assertIn("fs:[0x2d0]", text.lower())
        seq_idx = next(
            i for i, line in enumerate(lines) if "mov qword ptr [r11+8], rax" in line
        )
        self.assertEqual(lines[-2:], ["pop rax", "mov qword ptr [r11+8], rax"])
        self.assertLess(seq_idx, len(lines))

    def test_ring_reader_tid_roundtrip(self):
        from perf.arch import x86_64 as arch

        events = ["cycles"]
        lay = arch.profile_ring_layout(len(events), slots=8)
        blob = bytearray(lay["total"])
        struct.pack_into(
            "<QQQQQ",
            blob,
            0,
            RING_MAGIC,
            2,
            lay["capacity"],
            lay["stride"],
            len(events),
        )
        for seq, mid, tid, val in ((0, 0, 111, 100), (1, 1, 222, 150)):
            off = RING_HDR_SIZE + seq * lay["stride"]
            struct.pack_into("<I", blob, off, mid)
            struct.pack_into("<I", blob, off + 4, tid)
            struct.pack_into("<Q", blob, off + 8, seq)
            struct.pack_into("<Q", blob, off + 16, val)
        recs = RingReader(bytes(blob), events).records()
        self.assertEqual([r["tid"] for r in recs], [111, 222])

    def test_pair_deltas_tid_aware(self):
        from perf.prof import _pair_deltas, _rows_to_df

        same = [
            {"name": "a", "seq": 0, "tid": 7, "cycles": 100},
            {"name": "b", "seq": 1, "tid": 7, "cycles": 150},
        ]
        rows = _pair_deltas(same, ["cycles"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tid"], 7)
        df = _rows_to_df(rows, ["cycles"])
        self.assertIn("tid", df.columns)
        self.assertIn("threads", df.columns)
        mixed = [
            {"name": "a", "seq": 0, "tid": 1, "cycles": 100},
            {"name": "b", "seq": 1, "tid": 2, "cycles": 150},
        ]
        self.assertEqual(_pair_deltas(mixed, ["cycles"]), [])

    def test_func_pairing_per_tid(self):
        from perf.prof import _pair_deltas

        hits = [
            {"name": "f", "kind": "func_entry", "seq": 0, "tid": 1, "cycles": 10},
            {"name": "f", "kind": "func_entry", "seq": 1, "tid": 2, "cycles": 20},
            {"name": "f", "kind": "func_exit", "seq": 2, "tid": 2, "cycles": 35},
            {"name": "f", "kind": "func_exit", "seq": 3, "tid": 1, "cycles": 60},
        ]
        rows = _pair_deltas(hits, ["cycles"])
        self.assertEqual({(r["tid"], r["cycles"]) for r in rows}, {(2, 15), (1, 50)})

    def test_discover_tls_offset(self):
        from perf.prof import _discover_tls_tid_offset

        base = 0x7000000
        blob = bytearray(0x1000)
        struct.pack_into("<I", blob, 0x2D0, 4242)
        with patch.object(_track_mod, "read_mem", return_value=bytes(blob)):
            self.assertEqual(_discover_tls_tid_offset(4242, base, 4242), 0x2D0)
        with patch.object(_track_mod, "read_mem", return_value=b"\x00" * 0x1000):
            self.assertIsNone(_discover_tls_tid_offset(1, base, 2))


class TestLoadBias(unittest.TestCase):
    def test_non_pie_returns_zero(self):
        import subprocess

        from perf.prof import load_bias

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        if shutil.which("gcc") is None:
            self.skipTest("gcc unavailable")
        src = os.path.join(d, "b.c")
        exe = os.path.join(d, "b")
        Path(src).write_text("int main(void){return 0;}\n")
        r = subprocess.run(
            ["gcc", "-O2", "-fno-pie", "-no-pie", "-o", exe, src],
            capture_output=True,
        )
        if r.returncode != 0:
            self.skipTest("gcc build failed")
        from elftools.elf.elffile import ELFFile

        with open(exe, "rb") as fh:
            if ELFFile(fh).header["e_type"] == "ET_DYN":
                self.skipTest("linker produced PIE")
        self.assertEqual(load_bias(os.getpid(), exe), 0)

    def test_missing_file_raises(self):
        from perf.prof import load_bias

        with self.assertRaises(OSError):
            load_bias(os.getpid(), "/nonexistent-perf-test-binary")


class TestProfHelpers(unittest.TestCase):
    def test_parse_hex_addr(self):
        from perf.prof import _parse_hex_addr

        self.assertIsNone(_parse_hex_addr(""))
        self.assertEqual(_parse_hex_addr("401000"), 401000)
        self.assertEqual(_parse_hex_addr("0x401000"), 0x401000)
        self.assertIsNone(_parse_hex_addr("xyz"))

    def test_waitpid_exited_and_signaled(self):
        import os

        from perf.prof import waitpid

        with patch("os.waitpid", return_value=(1, os.waitstatus_to_exitcode(0) and 0)):
            with patch("os.WIFEXITED", return_value=True):
                with patch("os.WEXITSTATUS", return_value=3):
                    with self.assertRaises(SystemExit):
                        waitpid(1)
        with patch("os.waitpid", return_value=(1, 9)):
            with patch("os.WIFEXITED", return_value=False):
                with patch("os.WIFSIGNALED", return_value=True):
                    with patch("os.WTERMSIG", return_value=9):
                        with self.assertRaises(SystemExit):
                            waitpid(1)

    def test_measure_pairs_empty(self):
        from perf.prof import _measure_pairs

        self.assertEqual(_measure_pairs(lambda: None, [], b"", 8, 1), {})

    def test_pair_deltas_unmatched_exit(self):
        from perf.prof import _pair_deltas

        self.assertEqual(_pair_deltas([{"kind": "func_exit"}], {}), [])

    def test_point_address_none(self):
        from perf.prof import _point_address

        self.assertIsNone(_point_address(None))

    def test_subtract_overhead_passthrough(self):
        from perf.prof import _subtract_overhead

        self.assertEqual(_subtract_overhead(10, "bad"), 10)
        self.assertEqual(_subtract_overhead(10, 0), 10)


if __name__ == "__main__":
    unittest.main()


class TestFindTrackableUnit(unittest.TestCase):
    def test_labels_are_mapped(self):
        from perf.prof import _find_trackable

        with (
            patch("perf.prof._info.elf_project", return_value=object()),
            patch(
                "perf.prof.perf_labels",
                return_value=[("hot", 0x1000), ("cold", 0x2000)],
            ),
        ):
            out = _find_trackable("/tmp/a.out", with_functions=False)
        self.assertEqual(
            out["labels"],
            [{"name": "hot", "addr": 0x1000}, {"name": "cold", "addr": 0x2000}],
        )
        self.assertEqual(out["functions"], {})

    def test_functions_are_demangled_and_deduplicated(self):
        from perf.prof import _find_trackable

        with (
            patch("angr.Project", return_value=object()),
            patch("perf.prof.perf_labels", return_value=[]),
            patch(
                "perf.prof.functions",
                return_value=(
                    {
                        "_Z3fooi": (0x1000, 0x1010),
                        "alias": (0x1000, 0x1010),
                        "bad": (0x1010, 0x1000),
                    },
                    {},
                ),
            ),
            patch("perf.prof._demangle_sym", side_effect=lambda name: "foo(int)"),
        ):
            out = _find_trackable("/tmp/a.out", with_functions=True)
        self.assertEqual(out["functions"], {"foo(int)": (0x1000, 0x1010)})

    def test_function_errors_are_contained(self):
        from perf.prof import _find_trackable

        with (
            patch("angr.Project", return_value=object()),
            patch("perf.prof.perf_labels", return_value=[]),
            patch("perf.prof.functions", side_effect=RuntimeError("boom")),
        ):
            out = _find_trackable("/tmp/a.out", with_functions=True)
        self.assertEqual(out, {"labels": [], "functions": {}})

    def test_without_functions_does_not_open_angr(self):
        from perf.prof import _find_trackable

        with (
            patch("perf.prof._info.elf_project", return_value=object()) as elf,
            patch("angr.Project") as project,
            patch("perf.prof.perf_labels", return_value=[]),
        ):
            _find_trackable("/tmp/a.out", with_functions=False)
        elf.assert_called_once_with("/tmp/a.out")
        project.assert_not_called()


class TestResolveTrackPointsUnit(unittest.TestCase):
    LABELS = [
        {"name": "lab_begin", "addr": 0x1000},
        {"name": "lab_end", "addr": 0x1010},
    ]
    FUNCS = {"fizz_buzz": (0x2000, 0x2040), "main": (0x3000, 0x3040)}

    def _resolve(self, patterns):
        from perf.prof import _resolve_track_points

        with (
            patch("perf.prof.resolve_exec", return_value="/tmp/a.out"),
            patch(
                "perf.prof._find_trackable",
                side_effect=lambda _path, with_functions: {
                    "labels": self.LABELS,
                    "functions": self.FUNCS if with_functions else {},
                },
            ),
        ):
            return _resolve_track_points("/tmp/a.out", patterns)

    def test_function_expands_to_entry_and_exit(self):
        pts = self._resolve(["fizz_buzz"])
        self.assertEqual([p["kind"] for p in pts], ["func_entry", "func_exit"])
        self.assertEqual([p["addr"] for p in pts], [0x2000, 0x2040])

    def test_label_and_region_resolution(self):
        single = self._resolve(["lab_begin"])
        self.assertEqual(
            single, [{"name": "lab_begin", "addr": 0x1000, "kind": "label"}]
        )
        region = self._resolve([("lab_begin", "lab_end")])
        self.assertEqual([p["name"] for p in region], ["lab_begin", "lab_end"])
        self.assertEqual([p["pair_side"] for p in region], ["begin", "end"])

    def test_hex_address_is_a_point(self):
        self.assertEqual(
            self._resolve(["0x401000"]),
            [{"name": "0x401000", "addr": 0x401000, "kind": "label"}],
        )

    def test_default_and_empty_patterns_track_main(self):
        for patterns in (None, []):
            pts = self._resolve(patterns)
            self.assertEqual({p["name"] for p in pts}, {"main"})
            self.assertEqual({p["kind"] for p in pts}, {"func_entry", "func_exit"})

    def test_invalid_region_syntax_raises(self):
        with self.assertRaises(ValueError):
            self._resolve(["a..b"])
        with self.assertRaises(ValueError):
            self._resolve([("a",)])

    def test_unknown_pattern_is_ignored(self):
        self.assertEqual(self._resolve(["nope"]), [])

    def test_duplicate_points_are_removed(self):
        pts = self._resolve(["0x401000", "0x401000"])
        self.assertEqual(len(pts), 1)


class TestProfileCleanup(unittest.TestCase):
    def _profile(self, init_side_effect=None, exc=OSError):
        from perf.prof import profile

        with (
            patch("perf.prof.resolve_exec", return_value="/bin/true"),
            patch("perf.prof.is_shared_library", return_value=False),
            patch(
                "perf.prof._resolve_track_points",
                return_value=[{"name": "main", "addr": 0x1000}],
            ),
            patch(
                "perf.prof.tempfile.mkstemp",
                return_value=(99, "/tmp/perf-track-x.ring"),
            ),
            patch("perf.prof.os.set_inheritable"),
            patch("perf.prof.os.close") as close,
            patch("perf.prof.os.unlink") as unlink,
            patch(
                "perf.prof._init_ring_file",
                side_effect=init_side_effect,
                return_value={"total": 64},
            ),
            patch("perf.prof.os.fork", side_effect=OSError("fork failed")),
        ):
            with self.assertRaises(exc):
                profile(["/bin/true"], event=["duration_time"], output=None)
        return close, unlink

    def test_ring_is_unlinked_when_init_fails(self):
        close, unlink = self._profile(
            init_side_effect=RuntimeError("init failed"), exc=RuntimeError
        )
        unlink.assert_called_once_with("/tmp/perf-track-x.ring")
        close.assert_called_once_with(99)

    def test_ring_is_unlinked_when_fork_fails(self):
        _close, unlink = self._profile()
        unlink.assert_called_once_with("/tmp/perf-track-x.ring")


class TestProfileThreadSafety(unittest.TestCase):
    def test_profile_shares_the_process_fork_lock(self):
        import perf.exec as exec_mod

        prof_mod = importlib.import_module("perf.prof")
        self.assertIs(prof_mod._FORK_LOCK, exec_mod._FORK_LOCK)

    def test_profile_calls_are_serialized(self):
        prof_mod = importlib.import_module("perf.prof")
        active = {"n": 0, "peak": 0}
        lock = threading.Lock()

        def fake_resolve(_path):
            with lock:
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])
            time.sleep(0.05)
            with lock:
                active["n"] -= 1
            raise RuntimeError("stop")

        def worker():
            try:
                prof_mod.profile(["unused"])
            except RuntimeError:
                pass

        with patch.object(prof_mod, "resolve_exec", side_effect=fake_resolve):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertEqual(active["peak"], 1)
