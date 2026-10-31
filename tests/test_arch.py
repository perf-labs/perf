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

import struct
import threading
import unittest

from perf.arch import x86_64

_CACHE_RESOLVERS = {
    "dcache": x86_64.data_cache_levels,
    "icache": x86_64.instruction_cache_levels,
    "dtlb": x86_64.data_tlb_levels,
    "itlb": x86_64.instruction_tlb_levels,
}


class TestBenchTemplates(unittest.TestCase):
    def test_modes_present(self):
        self.assertIn("latency", x86_64._BENCH)
        self.assertIn("throughput", x86_64._BENCH)

    def test_placeholders_format(self):
        t0, t1 = x86_64.timing()
        for mode, templ in x86_64._BENCH.items():
            out = templ.format(
                code="nop",
                setup="",
                teardown="",
                data="",
                data2="",
                data_iter="",
                t0=t0,
                t1=t1,
                tlb="",
            )
            self.assertNotIn("{setup}", out)
            self.assertNotIn("{code}", out)
            self.assertNotIn("{teardown}", out)
            self.assertNotIn("{data}", out)
            self.assertNotIn("{data2}", out)
            self.assertNotIn("{data_iter}", out)
            self.assertNotIn("{t0}", out)
            self.assertNotIn("{t1}", out)
            self.assertNotIn("{tlb}", out)

    def test_template_delegates_measurement(self):
        for mode in ("latency", "throughput"):
            self.assertIn("{t0}", x86_64._BENCH[mode])
            self.assertIn("{t1}", x86_64._BENCH[mode])
            self.assertIn("{data}", x86_64._BENCH[mode])
            self.assertIn("{data2}", x86_64._BENCH[mode])

    def test_throughput_wraps_loop_in_timing(self):
        lat = x86_64._BENCH["latency"]
        thr = x86_64._BENCH["throughput"]
        self.assertLess(thr.index("{t0}"), thr.index(".perfloop:"))
        self.assertGreater(thr.index("{t1}"), thr.index("jnz .perfloop"))
        self.assertLess(lat.index(".perfloop:"), lat.index("{t0}"))

    def test_timing_duration_time_uses_tsc(self):
        start, end = x86_64.timing("duration_time", None)
        self.assertIn("rdtsc", start)
        self.assertIn("rdtscp", end)
        self.assertNotIn("rdpmc", start)

    def test_timing_duration_time_reads_full_64bit_tsc(self):
        start, end = x86_64.timing("duration_time", None)
        for snippet, read in ((start, "rdtsc"), (end, "rdtscp")):
            self.assertIn("shl rdx, 32", snippet)
            self.assertIn("or rax, rdx", snippet)
            self.assertLess(snippet.index(read), snippet.index("shl rdx, 32"))
            self.assertLess(snippet.index("shl rdx, 32"), snippet.index("or rax, rdx"))
        self.assertLess(start.index("or rax, rdx"), start.index("mov r11, rax"))
        self.assertLess(end.index("or rax, rdx"), end.index("sub rax, r11"))

    def test_latency_and_throughput_measure_full_64bit_tsc(self):
        t0, t1 = x86_64.timing("duration_time", None)
        for mode in ("latency", "throughput"):
            out = x86_64._BENCH[mode].format(
                code="nop",
                setup="",
                teardown="",
                data="",
                data2="",
                data_iter="",
                t0=t0,
                t1=t1,
                tlb="",
            )
            self.assertEqual(out.count("shl rdx, 32"), 2, mode)
            self.assertEqual(out.count("or rax, rdx"), 2, mode)
            self.assertEqual(out.count("rdtscp"), 1, mode)
            loop = out.index(".perfloop:")
            if mode == "latency":
                self.assertLess(loop, out.index("rdtsc"))
                self.assertLess(out.index("rdtscp"), out.index("dec r8"))
            else:
                self.assertLess(out.index("rdtsc"), loop)
                self.assertGreater(out.index("rdtscp"), out.index("jnz .perfloop"))

    def test_timing_cycles_uses_rdpmc(self):
        start, end = x86_64.timing("cycles", 3)
        self.assertIn("rdpmc", start)
        self.assertIn("rdpmc", end)
        self.assertIn("mov ecx, 0x3", start)
        self.assertNotIn("rdtsc", start)

    def test_timing_pmc_uses_rdpmc_with_index(self):
        start, end = x86_64.timing("instructions", 7)
        self.assertIn("rdpmc", start)
        self.assertIn("rdpmc", end)
        self.assertIn("mov ecx, 0x7", start)
        self.assertNotIn("rdtsc", start)

    def test_timing_pmc_requires_index(self):
        with self.assertRaises(ValueError):
            x86_64.timing("instructions", None)

    def test_timing_multi_event_reads_in_order(self):
        start, end = x86_64.timing(["cycles", "instructions"], [4, 7])
        self.assertIn("mov ecx, 0x4", start)
        self.assertIn("mov ecx, 0x7", start)
        self.assertLess(start.index("mov ecx, 0x4"), start.index("mov ecx, 0x7"))
        self.assertIn("sub rax, rbx", end)
        self.assertIn("sub rax, rbp", end)
        self.assertIn("mov [r10], rax", end)
        self.assertIn("mov [r10 + 0x8], rax", end)
        self.assertIn("add r10, 0x10", end)

    def test_timing_mixed_duration_and_pmc(self):
        start, end = x86_64.timing(["duration_time", "cycles"], [None, 5])
        self.assertIn("rdtsc", start)
        self.assertIn("rdpmc", start)
        self.assertIn("mov ecx, 0x5", end)
        self.assertIn("add r10, 0x10", end)

    def test_timing_callee_saved_uses_rbx(self):
        start, end = x86_64.timing("duration_time", None, callee_saved=True)
        self.assertIn("mov rbx, rax", start)
        self.assertIn("sub rax, rbx", end)

    def test_timing_callee_saved_rejects_volatile_fallback(self):
        start, _ = x86_64.timing(["cycles"] * 6, list(range(6)), callee_saved=True)
        self.assertNotIn("r11", start)
        with self.assertRaises(ValueError):
            x86_64.timing(["cycles"] * 7, list(range(7)), callee_saved=True)

    def test_timing_default_single_event_keeps_r11(self):
        start, _ = x86_64.timing("duration_time", None)
        self.assertIn("mov r11, rax", start)

    def test_timing_too_many_events_fails(self):
        with self.assertRaises(ValueError):
            x86_64.timing(["cycles"] * 8, list(range(8)))

    def test_timing_preserves_rcx_only(self):
        for events, indices in [
            ("duration_time", None),
            ("cycles", 0),
            ("cycles", 3),
            (["cycles", "instructions"], [0, 1]),
        ]:
            start, end = x86_64.timing(events, indices)
            for block in (start, end):
                self.assertIn("push rcx", block)
                self.assertIn("pop rcx", block)
                self.assertNotIn("push rax", block)
                self.assertNotIn("push rdx", block)
                self.assertLess(block.index("push rcx"), block.index("pop rcx"))
            self.assertTrue(start.strip().startswith("push rcx"))
            self.assertTrue(start.strip().endswith("pop rcx"))
            self.assertTrue(end.strip().endswith("pop rcx"))

    def test_evict_proportions(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(16)]
        hot = x86_64.evict(_addrs, {"L1d": 100, "L2": 100, "L3": 100})
        self.assertEqual(hot.count("mov rax, ["), 16)
        self.assertNotIn("clflushopt", hot)
        self.assertNotIn("prefetch", hot)
        cold = x86_64.evict(_addrs, {"L1d": 0, "L2": 0, "L3": 0})
        self.assertEqual(cold.count("clflushopt"), 16)
        self.assertNotIn("mov rax, [", cold)
        self.assertNotIn("prefetch", cold)
        self.assertIn("mov rax, [", x86_64.evict(_addrs[:2]))
        self.assertEqual(x86_64.evict([]), "")

    def test_evict_cascade_partition(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(16)]
        warm = x86_64.evict(_addrs, {"L1d": 50, "L2": 50, "L3": 100}, cldemote=False)
        self.assertEqual(warm.count("mov rax, ["), 8)
        self.assertEqual(warm.count("prefetcht1"), 4)
        self.assertEqual(warm.count("prefetcht2"), 4)
        self.assertEqual(warm.count("clflushopt"), 8)
        self.assertEqual(warm.count("mfence"), 16)
        mixed = x86_64.evict(_addrs, {"L1d": 10, "L2": 20, "L3": 30}, cldemote=False)
        self.assertEqual(mixed.count("mov rax, ["), 1)
        self.assertEqual(mixed.count("prefetcht1"), 3)
        self.assertEqual(mixed.count("prefetcht2"), 3)
        self.assertEqual(mixed.count("clflushopt"), 15)
        self.assertEqual(mixed.count("mfence"), 16)

    def test_evict_fences_per_spec(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(16)]
        for levels in (
            {"L1d": 100, "L2": 100, "L3": 100},
            {"L1d": 50, "L2": 50, "L3": 100},
            {"L1d": 0, "L2": 0, "L3": 0},
        ):
            asm = x86_64.evict(_addrs, levels, cldemote=False)
            self.assertNotIn("lfence", asm)
            self.assertNotIn("sfence", asm)
            self.assertEqual(asm.count("mfence"), len(_addrs))

    def test_evict_settle_pauses(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(4)]
        for levels in (
            {"L1d": 100, "L2": 100, "L3": 100},
            {"L1d": 50, "L2": 50, "L3": 100},
            {"L1d": 0, "L2": 0, "L3": 0},
        ):
            asm = x86_64.evict(_addrs, levels, cldemote=False)
            self.assertEqual(asm.count("pause"), x86_64._SETTLE_PAUSES)
            bare = x86_64.evict(_addrs, levels, settle=0, cldemote=False)
            self.assertNotIn("pause", bare)

    def test_evict_assembles(self):
        import keystone

        _addrs = [0x400000 + i * 0x1000 for i in range(16)]
        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        for levels in (
            {"L1d": 100, "L2": 100, "L3": 100},
            {"L1d": 50, "L2": 50, "L3": 100},
            {"L1d": 0, "L2": 0, "L3": 0},
            {"L1d": 10, "L2": 20, "L3": 30},
        ):
            for cldemote in (False, True):
                enc, _ = ks.asm(x86_64.evict(_addrs, levels, cldemote=cldemote))
                self.assertTrue(enc)

    def test_evict_l2_orders_flush_before_prefetch(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(8)]
        asm = x86_64.evict(_addrs, {"L1d": 0, "L2": 100, "L3": 0}, cldemote=False)
        self.assertLess(asm.index("clflushopt"), asm.index("prefetcht1"))

    def test_evict_single_tier_patterns(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(8)]
        l1 = x86_64.evict(_addrs, {"L1d": 100, "L2": 100, "L3": 100}, cldemote=False)
        self.assertEqual(l1.count("mov rax, ["), 8)
        self.assertEqual(l1.count("mfence"), 8)
        self.assertNotIn("clflushopt", l1)
        self.assertNotIn("prefetch", l1)
        l2 = x86_64.evict(_addrs, {"L1d": 0, "L2": 100, "L3": 0}, cldemote=False)
        self.assertEqual(l2.count("prefetcht1"), 8)
        self.assertEqual(l2.count("prefetcht2"), 0)
        self.assertEqual(l2.count("clflushopt"), 8)
        self.assertNotIn("mov rax, [", l2)
        l3 = x86_64.evict(_addrs, {"L1d": 0, "L2": 0, "L3": 100}, cldemote=False)
        self.assertEqual(l3.count("prefetcht2"), 8)
        self.assertEqual(l3.count("prefetcht1"), 0)
        self.assertEqual(l3.count("clflushopt"), 8)
        dram = x86_64.evict(_addrs, {"L1d": 0, "L2": 0, "L3": 0}, cldemote=False)
        self.assertEqual(dram.count("clflushopt"), 8)
        self.assertNotIn("prefetch", dram)
        self.assertNotIn("mov rax, [", dram)

    def test_evict_l1_to_dram_cascade(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(16)]
        asm = x86_64.evict(
            _addrs, {"L1d": 25, "L2": 25, "L3": 25}, settle=0, cldemote=False
        )
        self.assertEqual(asm.count("mov rax, ["), 4)
        self.assertEqual(asm.count("prefetcht1"), 3)
        self.assertEqual(asm.count("prefetcht2"), 2)
        self.assertEqual(asm.count("clflushopt"), 12)
        self.assertEqual(asm.count("mfence"), 16)
        self.assertNotIn("pause", asm)
        for a in (_addrs[0], _addrs[3], _addrs[7], _addrs[15]):
            self.assertIn(f"{a:x}", asm)

    def test_evict_cascade_assembles_full_tier_chain(self):
        import keystone

        _addrs = [0x400000 + i * 0x1000 for i in range(16)]
        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        asm = x86_64.evict(
            _addrs, {"L1d": 25, "L2": 25, "L3": 25}, settle=1, cldemote=False
        )
        enc, _ = ks.asm(asm)
        self.assertTrue(enc)

    def test_evict_tlb_tiers_use_mprotect(self):
        _addrs = [0x400000, 0x400000 + 64 * 0x1000, 0x400000 + 128 * 0x1000]
        asm = x86_64.evict(_addrs, mem_levels={a: {"TLBd": 0} for a in _addrs})
        self.assertEqual(asm.count("syscall"), 1)
        self.assertNotIn("clflushopt", asm)
        self.assertNotIn("invlpg", asm)
        asm = x86_64.evict(_addrs, mem_levels={a: {"TLBi": 0} for a in _addrs})
        self.assertEqual(asm.count("syscall"), 1)
        self.assertNotIn("clflushopt", asm)
        self.assertNotIn("invlpg", asm)

    def test_evict_tlb_runs_are_bounded(self):
        _addrs = [0x400000 + i * 64 * 0x1000 for i in range(32)]
        asm = x86_64.evict(
            _addrs,
            mem_levels={a: {"TLBd": 0} for a in _addrs},
            settle=0,
        )
        self.assertEqual(asm.count("syscall"), 4)

    def test_evict_resident_tlb_tier_is_left_alone(self):
        _addrs = [0x400000, 0x410000]
        for tier in ("TLBd", "TLBi"):
            asm = x86_64.evict(_addrs, mem_levels={a: {tier: 100} for a in _addrs})
            self.assertNotIn("syscall", asm)
            self.assertNotIn("clflushopt", asm)

    def test_evict_resident_l1i_is_left_alone(self):
        _addrs = [0x401000, 0x401080]
        hot = x86_64.evict(_addrs, mem_levels={a: {"L1i": 100} for a in _addrs})
        self.assertNotIn("clflushopt", hot)
        cold = x86_64.evict(_addrs, mem_levels={a: {"L1i": 0} for a in _addrs})
        self.assertEqual(cold.count("clflushopt"), len(_addrs))

    def test_evict_tlb_tiers_combine_with_the_cache_tier(self):
        _addrs = [0x400000, 0x410000]
        hot = x86_64.evict(
            _addrs,
            {"L1d": 100, "L2": 0, "L3": 0},
            mem_levels={a: {"L1d": 100, "TLBd": 0} for a in _addrs},
            cldemote=False,
        )
        self.assertEqual(hot.count("mov rax, ["), len(_addrs))
        self.assertEqual(hot.count("syscall"), 1)
        self.assertNotIn("clflushopt", hot)
        cold = x86_64.evict(
            _addrs,
            {"L1d": 100, "L2": 0, "L3": 0},
            mem_levels={a: {"L1d": 100, "TLBd": 100} for a in _addrs},
            cldemote=False,
        )
        self.assertEqual(cold.count("mov rax, ["), len(_addrs))
        self.assertNotIn("syscall", cold)

    def test_evict_tlb_levels_apply_without_a_per_address_spec(self):
        _addrs = [0x400000, 0x410000]
        asm = x86_64.evict(
            _addrs,
            {"L1d": 0, "L2": 0, "L3": 0},
            tlb_levels={"TLBd": 0},
            cldemote=False,
        )
        self.assertEqual(asm.count("clflushopt"), len(_addrs))
        self.assertEqual(asm.count("syscall"), 1)
        resident = x86_64.evict(
            _addrs,
            {"L1d": 0, "L2": 0, "L3": 0},
            tlb_levels={"TLBd": 100},
            cldemote=False,
        )
        self.assertNotIn("syscall", resident)

    def test_evict_per_address_tlb_rate_beats_the_global_one(self):
        _addrs = [0x400000]
        asm = x86_64.evict(
            _addrs,
            {"L1d": 100},
            mem_levels={0x400000: {"TLBd": 100}},
            tlb_levels={"TLBd": 0},
            cldemote=False,
        )
        self.assertNotIn("syscall", asm)

    def test_steer_asm_tlb_tiers_follow_the_address_kind(self):
        data, code = 0x400000, 0x401000
        meta = {
            "mem_addrs": [data, code],
            "l1i_addrs": [code],
            "levels": {},
            "K": 8,
        }
        mem_levels = {data: {"L1d": 100}, code: {"L1i": 100, "TLBi": 0}}
        asm = x86_64.steer_asm(
            meta,
            0,
            mem_levels=mem_levels,
            tlb={"TLBd": 0, "TLBi": 100},
            write_values=False,
            settle=0,
        )
        self.assertIn(x86_64.tlb_inval_asm(data, 1, x86_64._PROT_RW), asm)
        self.assertIn(x86_64.tlb_inval_asm(code, 1, x86_64._PROT_RX), asm)
        self.assertEqual(asm.count("syscall"), 2)

    def test_steer_asm_resident_tlb_tiers_emit_nothing(self):
        data, code = 0x400000, 0x401000
        meta = {
            "mem_addrs": [data, code],
            "l1i_addrs": [code],
            "levels": {},
            "K": 8,
        }
        mem_levels = {data: {"L1d": 100}, code: {"L1i": 100}}
        asm = x86_64.steer_asm(
            meta,
            0,
            mem_levels=mem_levels,
            tlb={"TLBd": 100, "TLBi": 100},
            write_values=False,
            settle=0,
        )
        self.assertNotIn("syscall", asm)
        self.assertNotIn("clflushopt", asm)

    def test_steer_asm_code_addresses_are_never_loaded_as_data(self):
        data, code = 0x400000, 0x401000
        meta = {
            "mem_addrs": [data, code],
            "l1i_addrs": [code],
            "levels": {},
            "K": 8,
        }
        asm = x86_64.steer_asm(
            meta,
            0,
            mem_levels={data: {"L1d": 100}, code: {"L1i": 0, "TLBi": 0}},
            tlb={"TLBd": 100, "TLBi": 0},
            write_values=False,
            settle=0,
        )
        self.assertEqual(asm.count("syscall"), 1)
        self.assertIn(f"mov r12, 0x{code:x}", asm)
        self.assertIn("clflushopt [r12]", asm)
        self.assertNotIn(f"mov rax, [0x{code:x}]", asm)

    def test_steer_asm_code_addresses_default_to_a_resident_l1i(self):
        data, code = 0x400000, 0x401000
        meta = {
            "mem_addrs": [data, code],
            "l1i_addrs": [code],
            "levels": {},
            "K": 8,
        }
        asm = x86_64.steer_asm(
            meta,
            0,
            mem_levels={data: {"L1d": 100}, code: {"TLBi": 0}},
            tlb={"TLBd": 100, "TLBi": 0},
            write_values=False,
            settle=0,
        )
        self.assertEqual(asm.count("syscall"), 1)
        self.assertNotIn("clflushopt", asm)
        self.assertNotIn(f"mov rax, [0x{code:x}]", asm)

    def test_evict_tlb_inval_pages_aligned(self):
        _addrs = [0x400000 + i * 0x100000 + 0x123 for i in range(2)]
        mem_levels = {a: {"TLBd": 0} for a in _addrs}
        asm = x86_64.evict(_addrs, mem_levels=mem_levels)
        for a in _addrs:
            page = a & ~0xFFF
            self.assertIn(f"mov rdi, 0x{page:x}", asm)
        self.assertIn(x86_64.tlb_inval_asm(_addrs[0], prologue=False), asm)
        self.assertEqual(asm.count("syscall"), 2)
        self.assertEqual(asm.count("push r11"), 1)
        self.assertEqual(asm.count("pop r11"), 1)

    def test_evict_tlb_never_covers_an_avoided_page(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(4)]
        mem_levels = {a: {"TLBd": 0} for a in _addrs}
        asm = x86_64.evict(_addrs, mem_levels=mem_levels, avoid=[0x401000])
        self.assertEqual(asm.count("syscall"), 2)
        self.assertIn("mov rsi, 0x1000", asm)
        self.assertIn("mov rsi, 0x2000", asm)

    def test_page_runs_coalesce_adjacent_pages(self):
        self.assertEqual(
            x86_64.page_runs(
                [0x400010, 0x400020, 0x410000, 0x420000, 0x421000], merge_gap=4
            ),
            [(0x400000, 1), (0x410000, 1), (0x420000, 2)],
        )
        self.assertEqual(x86_64.page_runs([]), [])
        self.assertEqual(
            x86_64.page_runs([0x400010, 0x400008, 0x400018]), [(0x400000, 1)]
        )

    def test_page_runs_merge_a_bounded_gap(self):
        self.assertEqual(
            x86_64.page_runs([0x400010, 0x402000, 0x404000], merge_gap=4),
            [(0x400000, 5)],
        )
        self.assertEqual(
            x86_64.page_runs([0x400010, 0x402000, 0x410000], merge_gap=4),
            [(0x400000, 3), (0x410000, 1)],
        )
        self.assertEqual(
            x86_64.page_runs([0x410000, 0x400000], merge_gap=4),
            [(0x400000, 1), (0x410000, 1)],
        )

    def test_page_runs_cap_the_length_of_a_run(self):
        pages = [0x400000 + i * 0x1000 for i in range(8)]
        self.assertEqual(
            x86_64.page_runs(pages, max_pages=4),
            [
                (0x400000, 4),
                (0x404000, 4),
            ],
        )

    def test_page_runs_split_around_an_avoided_page(self):
        self.assertEqual(
            x86_64.page_runs([0x400000, 0x401000, 0x402000], avoid=[0x401000]),
            [(0x400000, 1), (0x402000, 1)],
        )
        self.assertEqual(
            x86_64.page_runs([0x400000, 0x401000], avoid=[0x400000]),
            [(0x401000, 1)],
        )

    def test_tlb_inval_asm_covers_a_range(self):
        asm = x86_64.tlb_inval_asm(0x400000, 3)
        self.assertIn("mov rdi, 0x400000", asm)
        self.assertIn("mov rsi, 0x3000", asm)
        self.assertEqual(asm.count("syscall"), 1)
        one = x86_64.tlb_inval_asm(0x400000)
        self.assertIn("mov rsi, 0x1000", one)
        self.assertEqual(one.count("syscall"), 1)
        aligned = x86_64.tlb_inval_asm(0x400123)
        self.assertIn("mov rdi, 0x400000", aligned)

    def test_evict_coalesces_adjacent_tlb_pages(self):
        adjacent = [0x400000 + i * 0x1000 for i in range(4)]
        spec = {a: {"TLBd": 0} for a in adjacent}
        asm = x86_64.evict(adjacent, mem_levels=spec, settle=0)
        self.assertEqual(asm.count("syscall"), 1)
        self.assertIn("mov rsi, 0x4000", asm)

    def test_evict_merges_nearby_tlb_pages_into_one_call(self):
        nearby = [0x400000 + i * 16 * 0x1000 for i in range(4)]
        spec = {a: {"TLBd": 0} for a in nearby}
        asm = x86_64.evict(nearby, mem_levels=spec, settle=0)
        self.assertEqual(asm.count("syscall"), 1)
        self.assertIn("mov rsi, 0x31000", asm)

    def test_harness_reserved_registers(self):
        self.assertEqual(
            set(x86_64._HARNESS_RESERVED_REGS),
            {"r8", "r9", "r10", "rsp", "rip", "flags"},
        )
        for alias in ("esp", "eip", "ip"):
            self.assertIn(x86_64._canonical_reg(alias), x86_64._HARNESS_RESERVED_REGS)

    def test_protection_constants_are_the_kernel_values(self):
        self.assertEqual(x86_64._PROT_R, 0x1)
        self.assertEqual(x86_64._PROT_W, 0x2)
        self.assertEqual(x86_64._PROT_X, 0x4)
        self.assertEqual(x86_64._PROT_RW, 0x3)
        self.assertEqual(x86_64._PROT_RX, 0x5)
        self.assertEqual(x86_64._PROT_RWX, 0x7)

    def test_tlb_inval_asm_toggles_prot(self):
        for cold, shift in ((x86_64._PROT_RW, 2), (x86_64._PROT_RX, 1)):
            asm = x86_64.tlb_inval_asm(0x4100001000, 1, cold)
            self.assertIn("mov rdi, 0x4100001000", asm)
            self.assertIn("mov rsi, 0x1000", asm)
            self.assertIn(f"mov rdx, 0x{x86_64._PROT_RWX:x}", asm)
            self.assertIn(f"shl rax, {shift}", asm)
            self.assertIn("sub rdx, rax", asm)
            self.assertEqual(asm.count("mov rax, 10"), 1)
            self.assertEqual(asm.count("syscall"), 1)

    def test_tlb_toggle_alternates_between_two_distinct_protections(self):
        for cold in (x86_64._PROT_RW, x86_64._PROT_RX):
            delta = x86_64._PROT_RWX ^ cold
            self.assertTrue(delta and not delta & (delta - 1))
            self.assertNotEqual(
                x86_64._pte_protection(x86_64._PROT_RWX),
                x86_64._pte_protection(cold),
            )

    def test_tlb_toggle_rejects_a_protection_without_a_distinct_pte(self):
        for prot in (
            0x0,
            x86_64._PROT_R,
            x86_64._PROT_W,
            x86_64._PROT_X,
            x86_64._PROT_W | x86_64._PROT_X,
        ):
            with self.assertRaises(ValueError):
                x86_64.tlb_inval_asm(0x4100001000, 1, prot)

    def test_tlb_toggle_leaves_the_page_readable_and_writable(self):
        self.assertTrue(x86_64._PROT_RW & x86_64._PROT_RWX)
        self.assertTrue(x86_64._PROT_RX & x86_64._PROT_RWX)
        self.assertFalse(x86_64._PROT_RX & x86_64._PROT_W)
        self.assertFalse(x86_64._PROT_RW & x86_64._PROT_X)

    def test_tlb_restore_asm_is_read_write_exec(self):
        asm = x86_64.tlb_restore_call(0x4100001000, 2)
        self.assertIn("mov rsi, 0x2000", asm)
        self.assertIn(f"mov rdx, 0x{x86_64._PROT_RWX:x}", asm)
        self.assertEqual(asm.count("syscall"), 1)

    def test_tlb_restore_asm_covers_the_steered_tiers(self):
        data, code = 0x400000, 0x401000
        meta = {
            "mem_addrs": [data, code],
            "l1i_addrs": [code],
            "mem_levels": {data: {"L1d": 100, "TLBd": 0}, code: {"L1i": 100}},
        }
        asm = x86_64.tlb_restore_asm(meta, tlb={"TLBd": 100, "TLBi": 100})
        self.assertEqual(asm.count("syscall"), 1)
        self.assertIn(f"mov rdi, 0x{data:x}", asm)
        cold = x86_64.tlb_restore_asm(meta, tlb={"TLBd": 0, "TLBi": 0})
        self.assertEqual(cold.count("syscall"), 2)
        self.assertIn(f"mov rdi, 0x{code:x}", cold)
        self.assertEqual(x86_64.tlb_restore_asm({}, tlb={"TLBd": 0}), "")

    def test_evict_tlb_tier_assembles(self):
        import keystone

        _addrs = [0x400000 + i * 0x1000 for i in range(4)]
        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        for tier in ("TLBd", "TLBi"):
            asm = x86_64.evict(_addrs, mem_levels={a: {tier: 100} for a in _addrs})
            enc, _ = ks.asm(asm)
            self.assertTrue(enc)

    def test_evict_never_emits_privileged_invlpg(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(4)]
        for levels in (
            {"L1d": 100, "L2": 100, "L3": 100},
            {"L1d": 0, "L2": 0, "L3": 0},
        ):
            self.assertNotIn("invlpg", x86_64.evict(_addrs, levels, cldemote=False))
        for tier in ("TLBd", "TLBi"):
            asm = x86_64.evict(_addrs, mem_levels={a: {tier: 0} for a in _addrs})
            self.assertNotIn("invlpg", asm)
            self.assertIn("syscall", asm)

    def test_evict_interleaves_tiers(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(8)]
        asm = x86_64.evict(_addrs, {"L1d": 50, "L2": 50, "L3": 100}, cldemote=False)
        self.assertIn("mov rax, [0x407000]", asm)
        self.assertNotIn("mov rax, [0x401000]", asm)
        self.assertIn("0x401000", asm)
        self.assertIn("0x402000", asm)

    def test_per_iter_preserves_baselines(self):
        meta = {
            "K": 8,
            "evict_table_col": 4,
            "mem_addrs": [0x400000 + i * 0x1000 for i in range(2)],
            "reg_col": {"rdi": 0},
        }
        loop = x86_64._value_write_loop(meta, 0)
        for r in ("r12", "r13", "r14", "r15"):
            self.assertIn(f"push {r}", loop)
            self.assertIn(f"pop {r}", loop)
        prime = x86_64.prime_asm(meta, 0)
        self.assertIn("push r14", prime)
        self.assertIn("pop r14", prime)

    def test_resolve_mem_cache_tlb_tags(self):
        cfg = {"dtlb": {"0x1800": 100}}
        out = x86_64.resolve_mem_cache(cfg)
        self.assertEqual(out[0x1800], {"TLBd": 100})
        cfg = {"itlb": {"0x2000": "hit_rate=75"}}
        out = x86_64.resolve_mem_cache(cfg)
        self.assertEqual(out[0x2000], {"TLBi": 75})
        for tier in x86_64._TLB_TIERS:
            self.assertIn(tier, x86_64._CACHE_TIER_TAGS)

    def test_tier_names_are_not_config_keys(self):
        for key, tag in (
            ("dtlb", "TLBd"),
            ("itlb", "TLBi"),
            ("icache", "L1i"),
            ("dcache", "L1i"),
            ("dcache", "TLBd"),
            ("dtlb", "L1d"),
        ):
            with self.subTest(key=key, tag=tag):
                cfg = {key: {tag: 50}}
                resolver = _CACHE_RESOLVERS[key]
                with self.assertRaises(ValueError):
                    x86_64.resolve_mem_cache(cfg)
                with self.assertRaises(ValueError):
                    resolver(cfg)

    def test_hit_rate_is_the_single_tier_config(self):
        self.assertEqual(
            x86_64.data_tlb_levels({"dtlb": {"hit_rate": 50}}), {"TLBd": 50}
        )
        self.assertEqual(
            x86_64.instruction_tlb_levels({"itlb": {"hit_rate": 0}}), {"TLBi": 0}
        )
        self.assertEqual(
            x86_64.instruction_cache_levels({"icache": {"hit_rate": 25}}), {"L1i": 25}
        )
        self.assertEqual(x86_64.data_tlb_levels({"dtlb": 50}), {"TLBd": 50})
        self.assertEqual(x86_64.data_tlb_levels({"dtlb": "hot"}), {"TLBd": 100})
        self.assertEqual(x86_64.data_tlb_levels({"dtlb": "cold"}), {"TLBd": 0})
        self.assertIsNone(x86_64.data_tlb_levels({"dtlb": {}}))
        self.assertEqual(
            x86_64.data_cache_levels({"dcache": {"hit_rate": 50}}),
            {"L1d": 50, "L2": 0, "L3": 0},
        )
        with self.assertRaises(ValueError):
            x86_64.data_tlb_levels({"dtlb": "lukewarm"})

    def test_used_regs_canonicalizes_subregs(self):
        self.assertEqual(x86_64.used_regs("add eax, 42"), {"rax"})
        self.assertEqual(x86_64.used_regs("mov r11, [rax]"), {"r11", "rax"})
        self.assertEqual(x86_64.used_regs("idiv ecx"), {"rcx"})
        self.assertEqual(x86_64.used_regs("nop"), set())
        self.assertEqual(x86_64.used_regs(""), set())

    def test_used_regs_extended_subregs(self):
        self.assertEqual(x86_64.used_regs("mov r8d, eax"), {"r8", "rax"})
        self.assertEqual(x86_64.used_regs("mov r15w, ax"), {"r15", "rax"})
        self.assertEqual(x86_64.used_regs("mov rip, rax"), {"rip", "rax"})
        self.assertEqual(x86_64.used_regs("push rflags"), {"flags"})
        self.assertEqual(x86_64._canonical_reg("r8d"), "r8")
        self.assertEqual(x86_64._canonical_reg("r15w"), "r15")
        self.assertEqual(x86_64._canonical_reg("eflags"), "flags")

    def test_pick_baselines_avoids_timed_regs(self):
        regs = x86_64.pick_baselines(1, avoid={"r11", "rax"})
        self.assertNotIn("r11", regs)
        self.assertNotIn("rax", regs)
        regs = x86_64.pick_baselines(1, avoid={"rbx"})
        self.assertNotIn("rbx", regs)
        with self.assertRaises(ValueError):
            x86_64.pick_baselines(
                1, avoid={"rbx", "rbp", "r12", "r13", "r14", "r15", "r11"}
            )

    def test_timing_avoid_r11_uses_callee_saved(self):
        start, end = x86_64.timing("duration_time", None, avoid={"r11"})
        self.assertNotIn("r11", start)
        self.assertIn("rbx", start)
        start, _ = x86_64.timing("duration_time", None)
        self.assertIn("mov r11, rax", start)

    def test_timing_multi_event_avoids_baselines(self):
        start, end = x86_64.timing(
            ["cycles", "instructions"], [4, 7], avoid={"rbx", "rbp"}
        )
        self.assertNotIn("mov rbx, rax", start)
        self.assertNotIn("mov rbp, rax", start)
        self.assertIn("mov r12, rax", start)
        self.assertIn("mov r13, rax", start)

    def test_evict_l3_cldemote_path(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(8)]
        asm = x86_64.evict(_addrs, {"L1d": 0, "L2": 0, "L3": 100}, cldemote=True)
        self.assertIn(".byte", asm)
        self.assertNotIn("clflushopt", asm)
        self.assertNotIn("prefetch", asm)
        self.assertEqual(asm.count("mov rax, ["), 8)
        self.assertEqual(asm.count("mfence"), 8)
        self.assertEqual(asm.count("pause"), x86_64._SETTLE_PAUSES)

    def test_cldemote_asm_round_trip(self):
        import capstone
        import keystone

        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        for reg in (
            "rax",
            "rbx",
            "rcx",
            "rdx",
            "rsi",
            "rdi",
            "rbp",
            "rsp",
            "r8",
            "r12",
            "r13",
            "r15",
        ):
            enc, _ = ks.asm(x86_64.cldemote_asm(reg))
            self.assertTrue(enc)
            insns = list(cs.disasm(bytes(enc), 0))
            self.assertEqual(len(insns), 1)
            self.assertEqual(insns[0].mnemonic, "cldemote")
            self.assertIn(reg, insns[0].op_str)
        with self.assertRaises(ValueError):
            x86_64.cldemote_asm("bogus")

    def test_has_cldemote_override(self):
        from unittest.mock import patch

        _addrs = [0x400000 + i * 0x1000 for i in range(4)]
        with patch.object(x86_64, "has_cldemote", return_value=True):
            self.assertTrue(x86_64.has_cldemote())
            self.assertIn(
                ".byte",
                x86_64.evict(_addrs, {"L1d": 0, "L2": 0, "L3": 100}),
            )
        with patch.object(x86_64, "has_cldemote", return_value=False):
            self.assertFalse(x86_64.has_cldemote())
            self.assertIn(
                "prefetcht2",
                x86_64.evict(_addrs, {"L1d": 0, "L2": 0, "L3": 100}),
            )

    def test_memory_shortcuts(self):
        from perf.arch.x86_64 import data_cache_levels

        self.assertEqual(
            data_cache_levels({"dcache": "hot"}), {"L1d": 100, "L2": 0, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"dcache": "warm"}), {"L1d": 0, "L2": 100, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"dcache": "cool"}), {"L1d": 0, "L2": 0, "L3": 100}
        )
        self.assertEqual(
            data_cache_levels({"dcache": "cold"}), {"L1d": 0, "L2": 0, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"dcache": "HOT"}), {"L1d": 100, "L2": 0, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"dcache": "cool"}),
            {"L1d": 0, "L2": 0, "L3": 100},
        )
        self.assertEqual(
            data_cache_levels({"dcache": "cold"}),
            {"L1d": 0, "L2": 0, "L3": 0},
        )
        self.assertEqual(
            data_cache_levels({"dcache": "warm"}),
            {"L1d": 0, "L2": 100, "L3": 0},
        )
        with self.assertRaises(ValueError):
            data_cache_levels({"dcache": "lukewarm"})


class TestMovedX86Helpers(unittest.TestCase):
    def test_perf_syscall_nr(self):
        self.assertEqual(x86_64._PERF_SYSCALL_NR, 298)

    def test_angr_names_and_bases(self):
        self.assertEqual(x86_64._ANGR_ARCH_NAME, "AMD64")
        self.assertEqual(x86_64._SETUP_BASE, 0x1000000)
        self.assertEqual(x86_64._ASM_SCRATCH_BASE, 0x4100001000)
        self.assertEqual(x86_64._STACK_ADDR, 0x7FFF00000000)
        self.assertEqual(x86_64._STACK_SIZE, 0x100000)
        self.assertEqual(x86_64._PRIME_SCRATCH_REG, "r14")

    def test_imm_formats_values(self):
        for v in (0, 42, 0x401000, 2**64 - 1):
            self.assertEqual(x86_64.imm(v), hex(v))
        self.assertEqual(x86_64.imm(42), "0x2a")
        self.assertEqual(x86_64.imm("bogus"), "bogus")

    def test_normalize_asm_decimal_immediates(self):
        self.assertEqual(x86_64.normalize_asm("add eax, 42;"), "add eax, 0x2a;")
        self.assertEqual(x86_64.normalize_asm("sub eax, 42;"), "sub eax, 0x2a;")
        self.assertEqual(x86_64.normalize_asm("nop;"), "nop;")
        self.assertEqual(x86_64.normalize_asm("add r11, [rax];"), "add r11, [rax];")
        self.assertEqual(
            x86_64.normalize_asm("mov rax, 0x400000;"), "mov rax, 0x400000;"
        )

    def test_tsc_reader(self):
        code = x86_64.tsc_reader_bytes()
        self.assertIsInstance(code, bytes)
        self.assertTrue(len(code) > 0)
        enc, _ = x86_64.assembler().asm(x86_64._TSC_READER_ASM)
        self.assertEqual(bytes(enc), code)

    def test_cpuid_tsc_frequency_is_exact_or_absent(self):
        hz = x86_64.cpuid_tsc_hz()
        self.assertTrue(hz is None or hz > 0)

    def test_call_helpers(self):
        call = x86_64.call_asm(0x401000)
        self.assertIn("call rax", call)
        self.assertIn("0x401000", call)

        seq = x86_64.call_seq_asm(0x401000)
        for reg in ("r8", "r9", "r10"):
            self.assertIn(f"push {reg}", seq)
            self.assertIn(f"pop {reg}", seq)
        self.assertIn("call rax", seq)

        nop = x86_64.call_seq_nop_asm()
        self.assertIn("push r8", nop)
        self.assertNotIn("call", nop)

        import keystone

        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        for asm in (call, seq, nop):
            enc, _ = ks.asm(asm)
            self.assertTrue(enc)

    def test_jump_helpers(self):
        self.assertIn("0x401000", x86_64.jump_asm(0x401000))
        chained = x86_64.setup_jump_asm("mov eax, 1", 0x401000)
        self.assertIn("mov eax, 1", chained)
        self.assertIn("jmp", chained)
        import keystone

        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        enc, _ = ks.asm(chained)
        self.assertTrue(enc)

    def test_map_stack_and_set_ip(self):
        from unittest.mock import Mock

        state = Mock()
        x86_64.map_stack(state)
        state.memory.map_region.assert_called_once_with(
            x86_64._STACK_ADDR, x86_64._STACK_SIZE, 7
        )

        state = Mock()
        x86_64.set_ip(state, 0x1234)
        self.assertEqual(state.regs.rip, 0x1234)

    def test_mnemonic_predicates(self):
        self.assertTrue(x86_64.is_return_mnemonic("ret"))
        self.assertTrue(x86_64.is_return_mnemonic("RETQ"))
        self.assertFalse(x86_64.is_return_mnemonic("jmp"))
        self.assertFalse(x86_64.is_branch_mnemonic("mov"))
        self.assertTrue(x86_64.is_branch_mnemonic("jne"))

    def test_mem_refs_rip_relative_and_absolute(self):
        md = x86_64.disassembler()
        md.detail = True
        blob = x86_64.assemble("lea rax, [rip + 0x10]\nmov rbx, [0x402000]", 0x401000)
        insns = list(md.disasm(blob, 0x401000))
        self.assertEqual(x86_64.mem_refs(insns[0]), (insns[0].address + 7 + 0x10,))
        self.assertEqual(x86_64.mem_refs(insns[1]), (insns[1].address + 7 + 0x402000,))
        absolute = bytes.fromhex("488b1c25") + (0x8048000).to_bytes(4, "little")
        insn = list(md.disasm(absolute, 0x401000))[0]
        self.assertEqual(x86_64.mem_refs(insn), (0x8048000,))
        reg = list(md.disasm(x86_64.assemble("mov rax, [rdi]", 0x401000), 0x401000))
        self.assertEqual(x86_64.mem_refs(reg[0]), ())

    def test_mem_refs_reports_indirect_branches(self):
        md = x86_64.disassembler()
        md.detail = True
        blob = x86_64.assemble("jmp qword ptr [rip + 0x10]", 0x401000)
        insn = list(md.disasm(blob, 0x401000))[0]
        self.assertEqual(x86_64.mem_refs(insn), (insn.address + insn.size + 0x10,))
        self.assertFalse(hasattr(x86_64, "MEM_FREE_MNEMONICS"))
        self.assertFalse(hasattr(x86_64, "is_mem_free_mnemonic"))

    def test_steer_prime_match_bench(self):
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
        self.assertEqual(
            _arch_asm("steer_asm", meta, buf.ctypes.data),
            x86_64.steer_asm(meta, buf.ctypes.data),
        )
        self.assertEqual(
            _arch_asm("prime_asm", meta, buf.ctypes.data),
            x86_64.prime_asm(meta, buf.ctypes.data),
        )
        combined = "\n".join(
            p
            for p in (
                x86_64.steer_asm(meta, buf.ctypes.data),
                x86_64.prime_asm(meta, buf.ctypes.data),
            )
            if p
        )
        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        enc, _ = ks.asm(combined)
        self.assertTrue(enc)


class TestElfArchitectureBoundary(unittest.TestCase):
    def test_elf_reloc_types_are_the_architecture_module_values(self):
        from perf.exec import ElfConst

        self.assertEqual(ElfConst.R_X86_64_64, x86_64._R_X86_64_64)
        self.assertEqual(ElfConst.R_X86_64_COPY, x86_64._R_X86_64_COPY)
        self.assertEqual(ElfConst.R_X86_64_GLOB_DAT, x86_64._R_X86_64_GLOB_DAT)
        self.assertEqual(ElfConst.R_X86_64_JUMP_SLOT, x86_64._R_X86_64_JUMP_SLOT)
        self.assertEqual(ElfConst.R_X86_64_RELATIVE, x86_64._R_X86_64_RELATIVE)
        self.assertEqual(ElfConst.R_X86_64_IRELATIVE, x86_64._R_X86_64_IRELATIVE)

    def test_word_and_stack_shape_are_the_architecture_modules(self):
        self.assertEqual(x86_64._POINTER_SIZE, 8)
        self.assertEqual(x86_64._STACK_ENTRY_ALIGN, 16)
        self.assertEqual(x86_64.process_stack_top(0x1000, 0x200000), 0x200FF0)
        self.assertEqual(x86_64.entry_sp(0x200FF0) % 16, 8)
        self.assertEqual(x86_64.entry_sp(0x200FF0, 0x10000) % 16, 8)
        self.assertEqual(
            x86_64.pack_pointer(2**64 + 5), b"\x05\x00\x00\x00\x00\x00\x00\x00"
        )
        self.assertEqual(
            x86_64.unpack_pointer(b"xx" + x86_64.pack_pointer(0x401000), 2), 0x401000
        )

    def test_call_native_asm_clears_the_argument_registers(self):
        asm = x86_64.call_native_asm(0x401000)
        for reg in ("eax", "ecx", "edx", "esi", "edi", "r8d", "r9d"):
            self.assertIn(f"xor {reg}, {reg}", asm)
        self.assertNotIn("ebx", asm)
        self.assertNotIn("rbx", asm)
        self.assertIn("mov r11, 0x401000", asm)
        self.assertIn("call r11", asm)
        self.assertTrue(bytes(x86_64.assemble(asm)))

    def test_movabs64_bytes_widens_the_immediate(self):
        narrow = bytes(x86_64.assemble("mov rax, 0x1234;"))
        self.assertIn(b"\x48\xc7\xc0", narrow)
        wide = x86_64.movabs64_bytes(narrow)
        self.assertEqual(len(wide), len(narrow) + 3)
        self.assertIn(b"\x48\xb8", wide)
        self.assertEqual(int.from_bytes(wide[2:10], "little", signed=True), 0x1234)
        self.assertEqual(x86_64.movabs64_bytes(b"\x90\x90"), b"\x90\x90")

    def test_clear_movabs_zeroes_the_field_and_reports_it(self):
        wide = x86_64.movabs64_bytes(bytes(x86_64.assemble("mov rax, 0x401000;")))
        out, at = x86_64.clear_movabs(wide)
        self.assertEqual(at, 2)
        self.assertEqual(out[:at], wide[:at])
        self.assertEqual(out[at : at + 8], b"\x00" * 8)
        self.assertEqual(out[at + 8 :], wide[at + 8 :])
        self.assertEqual(x86_64.clear_movabs(b"\x90\x90"), (b"\x90\x90", -1))

    def test_layout_hazard_spots_absolute_moves_and_addresses(self):
        moffs = b"\xa1" + bytes(x86_64.pack_pointer(0x401000))
        self.assertTrue(x86_64.layout_hazard(moffs, lambda v: False))
        self.assertFalse(
            x86_64.layout_hazard(bytes(x86_64.assemble("nop; ret;")), lambda v: True)
        )
        absolute = bytes(x86_64.assemble("mov rax, 0x401000;"))
        self.assertTrue(x86_64.layout_hazard(absolute, lambda v: v - 0x400000 < 0x2000))
        self.assertFalse(x86_64.layout_hazard(absolute, lambda v: False))

    def test_relocate_code_leaves_a_branch_into_the_same_code_alone(self):
        code = bytes(x86_64.assemble("jmp 0x400003;", 0x400000))
        out = x86_64.relocate_code(
            code, 0x500000, 0x400000, lambda fv: 0x500000, span=(0x400000, 0x400005)
        )
        self.assertEqual(out, code)

    def test_relocate_code_rewrites_a_moved_branch(self):
        code = bytes(x86_64.assemble("jmp 0x500000;"))
        out = x86_64.relocate_code(
            code, 0x900000, 0x400000, lambda fv: 0x500000, span=(0, 0)
        )
        self.assertEqual(
            int.from_bytes(out[1:5], "little", signed=True), 0x500000 - 0x900005
        )

    def test_relocate_code_rejects_a_branch_it_cannot_reach(self):
        code = bytes(x86_64.assemble("jmp 0x500000;"))
        with self.assertRaises(x86_64.Unrelocatable):
            x86_64.relocate_code(
                code,
                0x500000,
                0x400000,
                lambda fv: 0x500000 + 2**32,
                span=(0, 0),
            )


class TestPerAddressCache(unittest.TestCase):
    def test_normalize_cache_spec_shortcut(self):
        self.assertEqual(
            x86_64.normalize_cache_spec("hot"), {"L1d": 100, "L2": 0, "L3": 0}
        )
        self.assertEqual(
            x86_64.normalize_cache_spec("warm"), {"L1d": 0, "L2": 100, "L3": 0}
        )
        self.assertEqual(
            x86_64.normalize_cache_spec("cool"), {"L1d": 0, "L2": 0, "L3": 100}
        )
        self.assertEqual(
            x86_64.normalize_cache_spec("cold"), {"L1d": 0, "L2": 0, "L3": 0}
        )

    def test_normalize_cache_spec_number_and_dict(self):
        self.assertEqual(x86_64.normalize_cache_spec(80), {"L1d": 80, "L2": 0, "L3": 0})
        self.assertEqual(
            x86_64.normalize_cache_spec({"L1d": 50, "L2": 25}),
            {"L1d": 50, "L2": 25},
        )
        self.assertEqual(
            x86_64.normalize_cache_spec({"hit_rate": {"L1d": 10, "L3": 90}}),
            {"L1d": 10, "L3": 90},
        )
        self.assertIsNone(x86_64.normalize_cache_spec("bogus"))
        self.assertIsNone(x86_64.normalize_cache_spec({"L1d": "nope"}))

    def test_normalize_cache_spec_rate_string(self):
        self.assertEqual(
            x86_64.normalize_cache_spec("hit_rate:100"),
            {"L1d": 100, "L2": 0, "L3": 0},
        )
        self.assertEqual(
            x86_64.normalize_cache_spec("hit_rate: 50"),
            {"L1d": 50, "L2": 0, "L3": 0},
        )
        self.assertIsNone(x86_64.normalize_cache_spec("hit=0"))
        self.assertIsNone(x86_64.normalize_cache_spec("hit:100"))
        self.assertIsNone(x86_64.normalize_cache_spec("hit_rate:abc"))
        self.assertIsNone(x86_64.normalize_cache_spec("l1:50"))

    def test_cache_config_default_with_addresses(self):
        cfg = {
            "dcache": {
                "L1d": {
                    "default": {"hit_rate": 100},
                    "0x321321": {"hit_rate": 100},
                    "0x321121": {"hit_rate": 100},
                }
            }
        }
        self.assertEqual(
            x86_64.data_cache_levels(cfg),
            {"L1d": 100, "L2": None, "L3": None},
        )
        got = x86_64.resolve_mem_cache(cfg)
        self.assertEqual(got[0x321321], {"L1d": 100})
        self.assertEqual(got[0x321121], {"L1d": 100})

    def test_cache_config_top_level_rate_string(self):
        cfg = {
            "dcache": {
                "L1d": "hit_rate:100",
                "L2": {
                    "default": {"hit_rate": 50},
                    "0x321321": {"hit_rate": 100},
                },
            }
        }
        self.assertEqual(
            x86_64.data_cache_levels(cfg),
            {"L1d": 100, "L2": 50, "L3": None},
        )
        got = x86_64.resolve_mem_cache(cfg)
        self.assertEqual(got[0x321321], {"L2": 100})

    def test_cache_config_merged_over_defaults(self):
        from perf.bench import _merge_config

        cfg = _merge_config(
            {
                "dcache": {
                    "L1d": {
                        "default": {"hit_rate": 50},
                        "0x321321": {"hit_rate": 100},
                    }
                }
            }
        )
        self.assertEqual(
            x86_64.data_cache_levels(cfg), {"L1d": 50, "L2": None, "L3": None}
        )
        got = x86_64.resolve_mem_cache(cfg)
        self.assertEqual(got[0x321321]["L1d"], 100)

    def test_resolve_mem_cache(self):
        config = {
            "dcache": {
                "L1d": {"hit_rate": 100},
                "0x41000000000": "hot",
                "0x20000000": "cool",
            }
        }
        got = x86_64.resolve_mem_cache(config)
        self.assertIn(0x41000000000, got)
        self.assertEqual(got[0x41000000000]["L1d"], 100)
        self.assertIn(0x20000000, got)
        self.assertEqual(got[0x20000000]["L3"], 100)

    def test_resolve_mem_cache_ignores_bad(self):
        self.assertEqual(x86_64.resolve_mem_cache(None), {})
        self.assertEqual(x86_64.resolve_mem_cache({}), {})
        self.assertEqual(
            x86_64.resolve_mem_cache({"dcache": {"zzz": "bogus"}}),
            {},
        )

    def test_cache_tier(self):
        self.assertEqual(x86_64._cache_tier({"L1d": 100, "L2": 0, "L3": 0}), "L1d")
        self.assertEqual(x86_64._cache_tier({"L1d": 0, "L2": 100, "L3": 0}), "L2")
        self.assertEqual(x86_64._cache_tier({"L1d": 0, "L2": 0, "L3": 100}), "L3")
        self.assertEqual(x86_64._cache_tier({"L1d": 0, "L2": 0, "L3": 0}), "DRAM")
        self.assertEqual(x86_64._cache_tier({"L1d": 100, "L2": 100}), "L1d")

    def test_resolve_mem_cache_l1i(self):
        cfg = {"icache": {"0x401000": {"hit_rate": 0}}}
        got = x86_64.resolve_mem_cache(cfg)
        self.assertEqual(got[0x401000], {"L1i": 0})
        self.assertNotIn("default", got)
        self.assertIsNone(x86_64.data_cache_levels(cfg))
        self.assertIsNone(x86_64.instruction_cache_levels({"icache": {"0x401000": 0}}))

    def test_evict_l1i_flushes_instruction_addrs(self):
        addrs = [0x401000, 0x401080]
        mem_levels = {a: {"L1i": 0} for a in addrs}
        asm = x86_64.evict(
            addrs,
            {"L1d": 100, "L2": 100, "L3": 100},
            mem_levels=mem_levels,
        )
        self.assertEqual(asm.count("clflushopt"), 2)
        self.assertEqual(asm.count("mfence"), 2)
        self.assertNotIn("mov rax, [", asm)
        for a in addrs:
            self.assertIn(f"mov r12, {x86_64.imm(a)}", asm)

    def test_steer_asm_flushes_l1i(self):
        import random

        from perf.bench import _fill_evict_tables, _per_iter_data

        models = [{"regs": {}, "reads": [], "writes": []}]
        cfg = {"icache": {"0x401234": {"hit_rate": 0}}}
        mem_levels = x86_64.resolve_mem_cache(cfg)
        buf, meta = _per_iter_data(
            models,
            4,
            {"L1d": 100, "L2": 100, "L3": 100},
            False,
            random.Random(1),
            mem_levels=mem_levels,
        )
        _fill_evict_tables(buf, meta, buf.ctypes.data)
        asm = x86_64.steer_asm(meta, buf.ctypes.data)
        self.assertIn("clflushopt [r12]", asm)
        self.assertIn(f"mov r12, {x86_64.imm(0x401234)}", asm)

    def test_evict_groups_by_per_address_levels(self):
        addrs = [0x41000000000, 0x20000000, 0x30000000]
        mem_levels = {
            0x41000000000: {"L1d": 100, "L2": 0, "L3": 0},
            0x20000000: {"L1d": 0, "L2": 0, "L3": 100},
        }
        asm = x86_64.evict(
            addrs,
            levels={"L1d": 100, "L2": 100, "L3": 100},
            mem_levels=mem_levels,
            cldemote=False,
        )
        l1 = asm.find("mov rax, [0x41000000000]")
        l3 = asm.find("clflushopt [r12]")
        self.assertGreaterEqual(l1, 0)
        self.assertGreaterEqual(l3, 0)

        lines = asm.splitlines()
        cold_idx = next(i for i, ln in enumerate(lines) if "0x20000000" in ln)
        self.assertIn("prefetcht2", "\n".join(lines[cold_idx : cold_idx + 4]))

    def test_steer_asm_uses_mem_levels(self):
        import random
        from unittest.mock import patch

        from perf.bench import _fill_evict_tables, _per_iter_data

        models = [{"regs": {}, "reads": [(0x41000000000, 8, 1)], "writes": []}]
        cfg = {
            "dcache": {
                "L1d": {"hit_rate": 100},
                "0x41000000000": "cool",
            }
        }
        mem_levels = x86_64.resolve_mem_cache(cfg)
        buf, meta = _per_iter_data(
            models,
            4,
            {"L1d": 100, "L2": 100, "L3": 100},
            False,
            random.Random(1),
            mem_levels=mem_levels,
        )
        _fill_evict_tables(buf, meta, buf.ctypes.data)
        with patch.object(x86_64, "has_cldemote", return_value=False):
            asm = x86_64.steer_asm(meta, buf.ctypes.data)

        self.assertIn("clflushopt [r12]", asm)
        self.assertIn("prefetcht2 [r12]", asm)


class TestArchConstants(unittest.TestCase):
    def test_harness_regs_have_no_bogus_entries(self):
        from perf.arch import x86_64

        self.assertNotIn("rspd", x86_64._HARNESS_REGS)

    def test_not_does_not_write_flags(self):
        from perf.arch import x86_64

        self.assertNotIn("not", x86_64._FLAG_WRITERS)


class TestArchUnsupported(unittest.TestCase):
    def test_unsupported_arch_raises(self):
        from perf.arch import arch as get_arch

        with self.assertRaises(ValueError):
            get_arch("bogus-arch-xyz")


class TestTierTagCache(unittest.TestCase):
    def test_case_insensitive_tiers(self):
        self.assertEqual(x86_64._tier_tag("l1d"), "L1d")
        self.assertEqual(x86_64._tier_tag("L2"), "L2")
        self.assertEqual(x86_64._tier_tag("tlbd"), "TLBd")
        self.assertEqual(x86_64._tier_tag("L1I"), "L1i")
        self.assertEqual(x86_64._tier_tag("dram"), "DRAM")
        self.assertIsNone(x86_64._tier_tag("bogus"))
        cfg = {"dcache": {"l1d": {"hit_rate": 100}}}
        self.assertEqual(
            x86_64.data_cache_levels(cfg), {"L1d": 100, "L2": None, "L3": None}
        )

    def test_rate_clamping(self):
        self.assertEqual(x86_64.normalize_cache_spec({"L1d": 150}), {"L1d": 100})
        self.assertEqual(x86_64.normalize_cache_spec({"L1d": -20}), {"L1d": 0})
        self.assertEqual(x86_64.normalize_cache_spec(150)["L1d"], 100)
        addrs = [0x400000 + i * 0x1000 for i in range(4)]
        asm = x86_64.evict(addrs, {"L1d": 150, "L2": -10, "L3": 0}, cldemote=False)
        self.assertEqual(asm.count("mov rax, ["), 4)
        self.assertNotIn("prefetch", asm)

    def test_tlb_preserves_regs(self):
        asm = x86_64.tlb_inval_asm(0x4100001000)
        for reg in ("rax", "rdi", "rsi", "rdx", "rcx", "r11"):
            self.assertIn(f"push {reg}", asm)
            self.assertIn(f"pop {reg}", asm)
        self.assertLess(asm.index("push rax"), asm.index("syscall"))
        self.assertGreater(asm.rindex("pop rax"), asm.rindex("syscall"))


class TestTlbToggleRuns(unittest.TestCase):
    def _protection(self, addr):
        with open("/proc/self/maps") as fh:
            for line in fh:
                fields = line.split()
                low, high = (int(v, 16) for v in fields[0].split("-"))
                if low <= addr < high:
                    return fields[1]
        return None

    def _run(self, asm, parity):
        import ctypes
        import mmap

        slot = ctypes.c_uint64(parity)
        code = x86_64.assemble(
            f"mov r8, {x86_64.imm(ctypes.addressof(slot))}\nmov r8, [r8]\n{asm}\nret"
        )
        page = mmap.mmap(
            -1,
            len(code),
            prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        )
        page.write(code)
        ctypes.CFUNCTYPE(None)(ctypes.addressof(ctypes.c_char.from_buffer(page)))()

    def test_toggle_really_changes_the_kernel_protection(self):
        for prot in (x86_64._PROT_RW, x86_64._PROT_RX):
            import mmap as _mmap

            page = _mmap.mmap(
                -1,
                _mmap.PAGESIZE,
                prot=_mmap.PROT_READ | _mmap.PROT_WRITE | _mmap.PROT_EXEC,
                flags=_mmap.MAP_PRIVATE | _mmap.MAP_ANONYMOUS,
            )
            addr = _addr(page)
            asm = x86_64.tlb_inval_asm(addr, 1, prot, False)
            seen = []
            for parity in (0, 1, 0):
                self._run(asm, parity)
                seen.append(self._protection(addr))
            self.assertEqual(len(set(seen)), 2, f"{prot:#x} never changed")
            self.assertEqual(seen[0], seen[2])
            self.assertTrue(seen[0].startswith("rw"))

    def test_toggle_keeps_the_page_usable(self):
        import mmap as _mmap

        page = _mmap.mmap(
            -1,
            _mmap.PAGESIZE,
            prot=_mmap.PROT_READ | _mmap.PROT_WRITE | _mmap.PROT_EXEC,
            flags=_mmap.MAP_PRIVATE | _mmap.MAP_ANONYMOUS,
        )
        addr = _addr(page)
        for prot in (x86_64._PROT_RW, x86_64._PROT_RX):
            for parity in (0, 1):
                self._run(x86_64.tlb_inval_asm(addr, 1, prot, False), parity)
                current = self._protection(addr)
                self.assertIn("r", current)
                if prot == x86_64._PROT_RW:
                    self.assertIn("w", current)

    def test_restore_asm_puts_the_page_back(self):
        import mmap as _mmap

        page = _mmap.mmap(
            -1,
            _mmap.PAGESIZE,
            prot=_mmap.PROT_READ | _mmap.PROT_WRITE | _mmap.PROT_EXEC,
            flags=_mmap.MAP_PRIVATE | _mmap.MAP_ANONYMOUS,
        )
        addr = _addr(page)
        self._run(x86_64.tlb_inval_asm(addr, 1, x86_64._PROT_RW, False), 1)
        self._run(x86_64.tlb_restore_call(addr, 1, False), 0)
        restored = self._protection(addr)
        self.assertTrue(restored.startswith("rw"))
        self.assertIn("x", restored)


def _addr(page):
    import ctypes

    return ctypes.addressof(ctypes.c_char.from_buffer(page))


class TestDisassemblerCache(unittest.TestCase):
    def test_disassembler_is_shared(self):
        self.assertIs(x86_64.disassembler(), x86_64.disassembler())
        self.assertIs(x86_64.att_disassembler(), x86_64.att_disassembler())
        self.assertTrue(x86_64.disassembler().detail)
        self.assertTrue(x86_64.att_disassembler().detail)

    def test_cached_decoders_still_decode(self):
        blob = x86_64.assemble("mov eax, 42; ret", 0x1000000)
        insns = list(x86_64.disassembler().disasm(bytes(blob), 0x1000000))
        self.assertEqual([i.mnemonic for i in insns], ["mov", "ret"])


class TestArchHelpers(unittest.TestCase):
    def test_mem_regs(self):
        self.assertEqual(x86_64.mem_regs(""), set())
        self.assertEqual(x86_64.mem_regs("mov rax,[rdi+8]"), {"rdi"})
        self.assertEqual(x86_64.mem_regs("mov eax, 42"), set())

    def test_settle_asm_bad(self):
        self.assertEqual(x86_64.settle_asm("bad"), "")

    def test_validate_cache_key(self):
        with self.assertRaises(ValueError):
            x86_64.validate_cache_key("dtlb", {"L1d": 50})

    def test_data_setup_empty(self):
        self.assertEqual(x86_64.data_setup_asm(None), "")
        self.assertEqual(x86_64.data_setup_asm({}), "")

    def test_data_setup_reloads_registers_after_memory_stores(self):
        data = {
            "regs": {"rax": 0x1234, "r12": 0x5678},
            "mem": {"0x42000000000": 1},
        }
        asm = x86_64.data_setup_asm(data)
        self.assertLess(asm.index("mov byte ptr [r12]"), asm.rindex("mov rax"))
        self.assertLess(asm.index("mov byte ptr [r12]"), asm.rindex("mov r12"))
        self.assertTrue(bytes(x86_64.assemble(asm)))

    def test_guard_reserves_the_red_zone_below_saved_registers(self):
        pre, post = x86_64.guard_asm(["rax", "rcx", "rdx"])
        self.assertLess(pre.index("push rdx"), pre.index("lea rsp, [rsp-128]"))
        self.assertIn("lea rsp, [rsp+128]", post)
        self.assertLess(
            post.index("lea rsp, [rsp+128]"), post.index("lea rsp, [rsp+8]")
        )
        self.assertEqual(x86_64.guard_asm([]), ("", ""))
        self.assertTrue(bytes(x86_64.assemble(f"{pre}\n{post}")))

    def test_loop_and_jecxz_count_as_jumps(self):
        for asm in ("loop 0x1004", "jecxz 0x1004"):
            insn = next(
                x86_64.disassembler().disasm(
                    bytes(x86_64.assemble(asm, 0x1000)), 0x1000
                )
            )
            self.assertTrue(x86_64.is_jump(insn))

    def test_patch_branch_handles_prefixed_short_branch(self):
        blob = bytearray(x86_64.assemble("jecxz 0x1008", 0x1000))
        self.assertTrue(x86_64.patch_branch(blob, 0, 3, 4))
        self.assertEqual(blob[-1], 4)

    def test_rip_relative_locates_disp_before_an_immediate(self):
        code = bytes(x86_64.assemble("mov dword ptr [rip + 8], 0x12345678", 0x400000))
        insn = next(x86_64.detailed_disassembler().disasm(code, 0x400000))
        disp_off, disp_size, disp = x86_64.rip_relative(insn)
        self.assertEqual(disp_size, 4)
        self.assertEqual(disp, 8)
        self.assertEqual(struct.unpack_from("<i", code, disp_off)[0], 8)
        self.assertLess(disp_off + 4, len(code))

    def test_detour_relocates_an_internal_branch_to_the_moved_block(self):
        old, new = 0x400000, 0x900000
        code = bytes(x86_64.assemble("jmp 0x400004; nop; nop; ret", old))
        out = x86_64.relocate_detour_bytes(code, old, new)
        insn = next(x86_64.disassembler().disasm(out, new))
        self.assertEqual(x86_64.immediate_target(insn), new + 4)
        self.assertEqual(len(out), len(code))

    def test_detour_maps_internal_targets_after_an_expansion(self):
        old, new = 0x400000, 0x900000
        code = bytes(x86_64.assemble("jle 0x400010; jmp 0x400006; nop; nop; ret", old))
        out = x86_64.relocate_detour_bytes(code, old, new)
        insns = list(x86_64.disassembler().disasm(out, new))
        self.assertEqual(insns[0].mnemonic, "jle")
        self.assertEqual(insns[1].mnemonic, "jmp")
        self.assertEqual(x86_64.immediate_target(insns[1]), new + 10)

    def test_detour_relocates_near_loops_and_rejects_far_ones(self):
        old, new = 0x400000, 0x400008
        code = bytes(x86_64.assemble("loop 0x400010; nop", old))
        out = x86_64.relocate_detour_bytes(code, old, new)
        insn = next(x86_64.disassembler().disasm(out, new))
        self.assertEqual(x86_64.immediate_target(insn), 0x400010)
        with self.assertRaises(x86_64._Undetourable):
            x86_64.relocate_detour_bytes(code, old, 0x900000)

    def test_relocate_code_moves_near_loops_and_rejects_far_ones(self):
        old = 0x400000
        code = bytes(x86_64.assemble("loop 0x400010; nop", old))
        out = x86_64.relocate_code(
            code,
            0x500008,
            old,
            lambda _addr: 0x500020,
            span=(old, old + 0xB),
        )
        insn = next(x86_64.disassembler().disasm(out, 0x500008))
        self.assertEqual(x86_64.immediate_target(insn), 0x500020)
        with self.assertRaises(x86_64.Unrelocatable):
            x86_64.relocate_code(
                code,
                0x900008,
                old,
                lambda _addr: 0x500020,
                span=(old, old + 0xB),
            )


class TestArchToolState(unittest.TestCase):
    def test_assembler_and_disassembler_are_thread_local(self):
        main_assembler = x86_64.assembler()
        main_disassembler = x86_64.disassembler()
        main_att = x86_64.att_disassembler()
        seen = {}

        def worker():
            seen["assembler"] = x86_64.assembler()
            seen["disassembler"] = x86_64.disassembler()
            seen["att"] = x86_64.att_disassembler()
            seen["assembler_again"] = x86_64.assembler()
            seen["disassembler_again"] = x86_64.disassembler()
            seen["att_again"] = x86_64.att_disassembler()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)

        self.assertIs(seen["assembler"], seen["assembler_again"])
        self.assertIs(seen["disassembler"], seen["disassembler_again"])
        self.assertIs(seen["att"], seen["att_again"])
        self.assertIsNot(seen["assembler"], main_assembler)
        self.assertIsNot(seen["disassembler"], main_disassembler)
        self.assertIsNot(seen["att"], main_att)
        self.assertIs(x86_64.assembler(), main_assembler)
        self.assertIs(x86_64.disassembler(), main_disassembler)
        self.assertIs(x86_64.att_disassembler(), main_att)

    def test_concurrent_assembly_and_disassembly(self):
        errors = []
        barrier = threading.Barrier(4, timeout=10)

        def worker(worker_id):
            try:
                barrier.wait(timeout=10)
                for i in range(20):
                    value = worker_id * 100 + i
                    code = bytes(x86_64.assemble(f"mov eax, {value}", 0x1000))
                    insn = next(x86_64.disassembler().disasm(code, 0x1000))
                    try:
                        decoded = int(insn.op_str.split(",")[1].strip(), 0)
                    except (IndexError, ValueError):
                        decoded = None
                    if insn.mnemonic != "mov" or decoded != value:
                        raise AssertionError((insn.mnemonic, insn.op_str))
            except BaseException as ex:
                errors.append(ex)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(errors, [])


class TestCondBranchAnalysis(unittest.TestCase):
    def _insns(self, asm, base=0x400000):
        blob = bytes(x86_64.assemble(asm, base))
        return list(x86_64.disassembler().disasm(blob, base))

    def test_no_conditional_branch_has_no_dependencies(self):
        regs, mems = x86_64.cond_branch_analysis(self._insns("mov rax, rdi; ret"))
        self.assertEqual((regs, mems), (set(), []))
        self.assertEqual(x86_64.cond_branch_analysis([]), (set(), []))

    def test_register_and_memory_dependencies_are_collected(self):
        regs, mems = x86_64.cond_branch_analysis(
            self._insns("mov rax, [rdi]; cmp eax, 5; jne 0x400000")
        )
        self.assertIn("rdi", regs)
        self.assertIn("rax", regs)
        self.assertEqual(len(mems), 1)
        self.assertEqual(x86_64.disassembler().reg_name(mems[0][0]), "rdi")
        self.assertEqual(mems[0][1:], (0, 1, 0))

    def test_backward_scan_follows_written_registers(self):
        regs, mems = x86_64.cond_branch_analysis(
            self._insns("mov rax, rdi; mov rbx, rax; cmp rbx, 5; jne 0x400000")
        )
        self.assertTrue({"rax", "rbx", "rdi"} <= regs)
        self.assertEqual(mems, [])

    def test_memory_operand_is_reported_with_its_displacement(self):
        regs, mems = x86_64.cond_branch_analysis(
            self._insns("cmp [rdi + 8], rax; je 0x400000")
        )
        self.assertIn("rax", regs)
        self.assertEqual(len(mems), 1)
        self.assertEqual(x86_64.disassembler().reg_name(mems[0][0]), "rdi")
        self.assertEqual(mems[0][2:], (1, 8))

    def test_flags_are_carried_back_through_a_chain(self):
        regs, _ = x86_64.cond_branch_analysis(
            self._insns(
                "movzx rax, byte ptr [rdi]; add rax, rdx; test rax, rax; jz 0x400000"
            )
        )
        self.assertTrue({"rdi", "rdx", "rax"} <= regs)


class TestPeepholeFunction(unittest.TestCase):
    def _peephole(self, asm_or_bytes, base=0x400000):
        blob = (
            asm_or_bytes
            if isinstance(asm_or_bytes, bytes)
            else bytes(x86_64.assemble(asm_or_bytes, base))
        )
        return x86_64.peephole_function(blob, base)

    def test_self_moves_become_nops(self):
        out, stats = self._peephole("mov rax, rax; mov rbx, rbx; ret")
        self.assertTrue(out.startswith(x86_64.optimal_nops(3)))
        self.assertEqual(stats["mov_self"], 2)
        self.assertEqual(
            list(x86_64.disassembler().disasm(out, 0x400000))[0].mnemonic, "nop"
        )

    def test_self_exchange_uses_its_own_counter(self):
        out, stats = self._peephole("xchg r8, r8; ret")
        self.assertTrue(out.startswith(x86_64.optimal_nops(3)))
        self.assertEqual(stats["xchg_self"], 1)
        _, stats = self._peephole("xchg edi, edi; ret")
        self.assertEqual(stats["xchg_self"], 0)

    def test_self_lea_with_base_and_index(self):
        out, stats = self._peephole("lea rax, [rax]; lea rbx, [rbx*1]; ret")
        self.assertEqual(stats["lea_self"], 2)
        self.assertTrue(out.startswith(x86_64.optimal_nops(3)))

    def test_zero_shift_is_removed_only_outside_32_bit_registers(self):
        out, stats = self._peephole("shl rax, 0; shl eax, 0; ret")
        self.assertEqual(stats["shift_zero"], 1)
        self.assertTrue(out.startswith(x86_64.optimal_nops(4)))
        self.assertIn(b"\xc1\xe0\x00", out)

    def test_non_optimal_nop_encodings_are_canonicalized(self):
        out, stats = self._peephole(b"\x66\x66\x66\x66\x90")
        self.assertEqual(out, x86_64.optimal_nops(5))
        self.assertEqual(stats["nop_norm"], 1)

    def test_wide_self_move_uses_the_shortest_encoding(self):
        blob = (b"\x48\xb8" + struct.pack("<Q", 0x12345678)) + bytes(
            x86_64.assemble("ret", 0x40000A)
        )
        out, stats = self._peephole(blob)
        self.assertEqual(stats["movabs"], 1)
        self.assertEqual(out[:5], b"\xb8\x78\x56\x34\x12")
        self.assertEqual(len(out), len(blob))

    def test_negative_self_move_becomes_a_sign_extended_immediate(self):
        blob = b"\x48\xb8" + struct.pack("<Q", 2**64 - 1)
        out, stats = self._peephole(blob)
        self.assertEqual(stats["movabs"], 1)
        self.assertEqual(out[:7], b"\x48\xc7\xc0\xff\xff\xff\xff")
        self.assertEqual(out[7:], x86_64.optimal_nops(3))

    def test_uncompressible_immediate_is_left_alone(self):
        blob = b"\x48\xb8" + struct.pack("<Q", 0x1_0000_0000)
        out, stats = self._peephole(blob)
        self.assertEqual(out, blob)
        self.assertEqual(stats["movabs"], 0)

    def test_instructions_outside_the_blob_are_ignored(self):
        out, stats = self._peephole(b"", base=0x400000)
        self.assertEqual(out, b"")
        self.assertEqual(sum(stats.values()), 0)


class TestBranchFunction(unittest.TestCase):
    def _branch(self, asm, **kwargs):
        blob = bytes(x86_64.assemble(asm, 0x400000))
        out, stats = x86_64.branch_function(blob, 0x400000, **kwargs)
        return out, stats

    def test_jump_to_the_fallthrough_is_removed(self):
        out, stats = self._branch("jmp 0x400002; nop; nop")
        self.assertEqual(stats["jmp_next"], 1)
        self.assertTrue(out.startswith(x86_64.optimal_nops(2)))

    def test_conditional_jump_to_the_fallthrough_is_removed(self):
        out, stats = self._branch("je 0x400002; nop; nop")
        self.assertEqual(stats["jcc_next"], 1)
        self.assertTrue(out.startswith(x86_64.optimal_nops(2)))

    def test_jump_chain_is_folded_to_the_final_target(self):
        out, stats = self._branch("jmp L2; nop; L2: jmp L3; nop; nop; L3: ret")
        self.assertEqual(stats["jmp_fold"], 1)
        first = next(x86_64.disassembler().disasm(out, 0x400000))
        self.assertEqual(x86_64.immediate_target(first), 0x400007)

    def test_conditional_jump_chain_is_folded(self):
        out, stats = self._branch("je L2; nop; nop; L2: jmp L3; nop; nop; L3: ret")
        self.assertEqual(stats["jcc_fold"], 1)
        first = next(x86_64.disassembler().disasm(out, 0x400000))
        self.assertEqual(x86_64.immediate_target(first), 0x400008)

    def test_inverted_conditional_jump_skips_the_following_jump(self):
        out, stats = self._branch("jne Lend; jmp Ltarget; Lend: nop; Ltarget: ret")
        self.assertEqual(stats["jcc_invert"], 1)
        insns = list(x86_64.disassembler().disasm(out, 0x400000))
        self.assertEqual(insns[0].mnemonic, "je")
        self.assertEqual(x86_64.immediate_target(insns[0]), 0x400005)
        self.assertEqual(insns[1].mnemonic, "nop")

    def test_indirect_jump_protects_every_instruction(self):
        _, stats = self._branch(
            "jmp rax; jne Lend; jmp Ltarget; Lend: nop; Ltarget: ret"
        )
        self.assertEqual(stats["jcc_invert"], 0)

    def test_forward_jump_out_of_the_blob_is_left_alone(self):
        out, stats = self._branch("jmp 0x400010; nop; nop; nop")
        self.assertEqual(stats, dict.fromkeys(stats, 0))
        self.assertTrue(out.startswith(b"\xeb\x0e"))


class TestStripAndVerify(unittest.TestCase):
    def test_strip_trailing_nops_and_int3_padding(self):
        self.assertEqual(x86_64.strip_trailing_nops(b""), (b"", 0))
        self.assertEqual(x86_64.strip_trailing_nops(b"\x90\x90"), (b"", 2))
        self.assertEqual(x86_64.strip_trailing_nops(b"\x90\xc3"), (b"\x90\xc3", 0))
        self.assertEqual(x86_64.strip_trailing_nops(b"\xc3\x90\xcc"), (b"\xc3", 2))
        self.assertEqual(
            x86_64.strip_trailing_nops(bytes(x86_64.assemble("ret; nop; nop", 0))),
            (bytes(x86_64.assemble("ret", 0)), 2),
        )

    def test_verify_accepts_a_valid_layout(self):
        code = bytes(x86_64.assemble("nop; ret", 0x400000))
        x86_64.verify_text(code, 0x400000, {})
        x86_64.verify_text(
            code, 0x400000, {}, old_to_new={(0, 2): (0x400000, 0x400002)}
        )

    def test_verify_rejects_empty_code(self):
        with self.assertRaisesRegex(ValueError, "no instructions"):
            x86_64.verify_text(b"", 0x400000, {})

    def test_verify_rejects_a_function_starting_mid_instruction(self):
        code = bytes(x86_64.assemble("mov eax, 42; ret", 0x400000))
        with self.assertRaisesRegex(ValueError, "not an instruction boundary"):
            x86_64.verify_text(
                code, 0x400000, {}, old_to_new={(0, 6): (0x400001, 0x400006)}
            )

    def test_verify_rejects_a_function_not_ending_on_a_boundary(self):
        code = bytes(x86_64.assemble("mov eax, 42; ret", 0x400000))
        with self.assertRaisesRegex(ValueError, "does not end on an"):
            x86_64.verify_text(
                code, 0x400000, {}, old_to_new={(0, 6): (0x400000, 0x400003)}
            )

    def test_verify_rejects_unsupported_rip_relative_width(self):
        from unittest.mock import patch

        code = bytes(x86_64.assemble("mov eax, [rip + 8]; ret", 0x400000))
        with patch.object(x86_64, "rip_relative", return_value=(0, 2, 0)):
            with self.assertRaisesRegex(ValueError, "unsupported RIP"):
                x86_64.verify_text(code, 0x400000, {})


class TestDetourHelpers(unittest.TestCase):
    def test_expand_short_branch(self):
        self.assertEqual(
            x86_64._expand_short_branch(b"\xeb\x00", 0x1234),
            b"\xe9" + struct.pack("<i", 0x1234),
        )
        self.assertEqual(
            x86_64._expand_short_branch(b"\x74\x00", 0x1234),
            b"\x0f\x84" + struct.pack("<i", 0x1234),
        )
        with self.assertRaises(x86_64._Undetourable):
            x86_64._expand_short_branch(b"", 0)
        with self.assertRaisesRegex(x86_64._Undetourable, "loop"):
            x86_64._expand_short_branch(b"\xe0\x00", 0x10)

    def test_detour_patch_len_measures_whole_instructions(self):
        total, insns = x86_64.detour_patch_len(b"\x90" * 10, 0x400000, 5)
        self.assertEqual(total, 5)
        self.assertEqual(len(insns), 5)
        total, _ = x86_64.detour_patch_len(
            bytes(x86_64.assemble("mov eax, 42", 0x400000)), 0x400000, 5
        )
        self.assertEqual(total, 5)
        with self.assertRaisesRegex(x86_64._Undetourable, "only 3 bytes"):
            x86_64.detour_patch_len(b"\x90" * 3, 0x400000, 5)
        with self.assertRaisesRegex(x86_64._Undetourable, "only 6 disassembled"):
            x86_64.detour_patch_len(b"\x90" * 6 + b"\x66", 0x400000, 7)

    def test_relocate_empty_bytes(self):
        self.assertEqual(x86_64.relocate_detour_bytes(b"", 0x400000, 0x900000), b"")

    def test_relocate_rip_relative_operand_to_the_new_address(self):
        old, new = 0x400000, 0x900000
        code = bytes(x86_64.assemble("mov eax, [rip + 0x10]; ret", old))
        out = x86_64.relocate_detour_bytes(code, old, new)
        insn = next(x86_64.disassembler().disasm(out, new))
        self.assertEqual(_rip_target(insn), old + 6 + 0x10)

    def test_relocate_internal_rip_reference(self):
        old, new = 0x400000, 0x900000
        code = bytes(x86_64.assemble("mov eax, [rip + 2]; nop; nop; ret", old))
        out = x86_64.relocate_detour_bytes(code, old, new)
        insn = next(x86_64.disassembler().disasm(out, new))
        self.assertEqual(_rip_target(insn), new + 8)

    def test_relocate_short_conditional_branch_far_away(self):
        old = 0x400000
        code = bytes(x86_64.assemble("je 0x400010; nop", old))
        out = x86_64.relocate_detour_bytes(code, old, 0x900000)
        self.assertEqual(len(out), 7)
        self.assertTrue(out.startswith(b"\x0f\x84"))

    def test_relocate_call_and_short_jump_far_away(self):
        old = 0x400000
        call = bytes(x86_64.assemble("call 0x400010; nop", old))
        out = x86_64.relocate_detour_bytes(call, old, 0x900000)
        self.assertEqual(len(out), len(call))
        self.assertEqual(
            x86_64.immediate_target(next(x86_64.disassembler().disasm(out, 0x900000))),
            0x400010,
        )
        jump = bytes(x86_64.assemble("jmp 0x400006; nop", old))
        out = x86_64.relocate_detour_bytes(jump, old, 0x900000)
        self.assertTrue(out.startswith(b"\xe9"))

    def test_relocate_rejects_a_branch_into_an_instruction(self):
        with self.assertRaisesRegex(x86_64._Undetourable, "middle"):
            x86_64.relocate_detour_bytes(b"\xeb\x01\x66\x90", 0x400000, 0x900000)

    def test_patch_jmp_encodes_a_relative_trampoline(self):
        blob = x86_64.patch_jmp(0x1000, 0x1100)
        self.assertEqual(blob[:1], b"\xe9")
        self.assertEqual(
            int.from_bytes(blob[1:5], "little", signed=True), 0x1100 - 0x1005
        )
        with self.assertRaises(ValueError):
            x86_64.patch_jmp(0x1000, 0x1000 + 2**40)

    def test_profile_ring_layout_rounds_up_to_a_power_of_two(self):
        layout = x86_64.profile_ring_layout(2, slots=100)
        self.assertEqual(layout["capacity"], 128)
        self.assertEqual(layout["mask"], 127)
        self.assertEqual(layout["stride"], 32)
        self.assertEqual(layout["total"], 64 + 128 * 32)
        self.assertEqual(x86_64.profile_ring_layout(0, slots=0)["capacity"], 1)
        self.assertEqual(x86_64.profile_ring_layout(-1, slots=-4)["stride"], 16)


def _rip_target(insn):
    disp = x86_64.rip_relative(insn)[2]
    return int(insn.address) + int(insn.size) + disp


class TestProfileBuilders(unittest.TestCase):
    EVENTS = ["duration_time"]

    def _layout(self):
        return x86_64.profile_ring_layout(len(self.EVENTS))

    def test_scale_slot_lines(self):
        self.assertEqual(x86_64._scale_slot_lines(1), [])
        self.assertEqual(x86_64._scale_slot_lines(16), ["shl rcx, 4"])
        self.assertEqual(
            x86_64._scale_slot_lines(24), ["lea rcx, [rcx+rcx*2]", "shl rcx, 3"]
        )
        self.assertEqual(
            x86_64._scale_slot_lines(40), ["lea rcx, [rcx+rcx*4]", "shl rcx, 3"]
        )
        self.assertEqual(
            x86_64._scale_slot_lines(72), ["lea rcx, [rcx+rcx*8]", "shl rcx, 3"]
        )
        self.assertEqual(x86_64._scale_slot_lines(10), ["imul rcx, rcx, 10"])

    def test_record_lines_for_duration_time(self):
        lines = x86_64._profile_record_lines(
            7, self.EVENTS, [None], 0x500000, self._layout()
        )
        text = "\n".join(lines)
        self.assertIn("mov r11, 0x500000", text)
        self.assertIn("rdtsc", text)
        self.assertIn("mov [r11+16], rax", text)
        self.assertNotIn("rdpmc", text)
        self.assertTrue(bytes(x86_64.assemble(text, 0x900000)))

    def test_record_lines_for_pmc_and_tid(self):
        layout = x86_64.profile_ring_layout(2)
        lines = x86_64._profile_record_lines(
            7, ["cycles", "instructions"], [3, 4], 0x500000, layout, tid_off=0x10
        )
        text = "\n".join(lines)
        self.assertIn("mov ecx, 0x3", text)
        self.assertIn("mov ecx, 0x4", text)
        self.assertIn("rdpmc", text)
        self.assertIn("fs:[0x10]", text)
        self.assertIn("mov [r11+16], rax", text)
        self.assertIn("mov [r11+24], rax", text)

    def test_record_lines_require_an_index_for_pmc_events(self):
        with self.assertRaisesRegex(ValueError, "rdpmc counter index"):
            x86_64._profile_record_lines(
                7, ["cycles"], [None], 0x500000, self._layout()
            )
        with self.assertRaisesRegex(ValueError, "rdpmc counter index"):
            x86_64._profile_record_lines(7, ["cycles"], [], 0x500000, self._layout())

    def test_prefix_helpers_assemble(self):
        layout = self._layout()
        plain = x86_64._detour_prefix_lines(
            7, self.EVENTS, [None], 0x500000, layout, save_flags=False
        )
        full = x86_64._detour_prefix_lines(
            7, self.EVENTS, [None], 0x500000, layout, save_flags=True
        )
        self.assertIn("pushfq", full)
        self.assertNotIn("pushfq", plain)
        self.assertGreater(x86_64._prefix_len(full, 0x900000), len(plain))
        for tid_off in (None, 0x10):
            lines = x86_64._func_entry_prefix_lines(
                7, self.EVENTS, [None], 0x500000, layout, 0x600000, 0x700000, tid_off
            )
            encoded = bytes(x86_64.assemble("\n".join(lines), 0x900000))
            self.assertTrue(encoded)
            self.assertIn("movabs r11, 0x700000", "\n".join(lines))
            if tid_off is not None:
                self.assertIn("fs:[0x10]", "\n".join(lines))

    def test_build_detour_copies_and_jumps_back(self):
        orig = bytes(x86_64.assemble("mov eax, 42", 0x400000))
        ret = 0x400000 + len(orig)
        out = x86_64.build_profile_detour(
            7, self.EVENTS, [None], 0x500000, 0x900000, ret, orig_bytes=orig
        )
        self.assertIn(orig, out)
        last = list(x86_64.disassembler().disasm(out, 0x900000))[-2:]
        self.assertEqual(last[0].mnemonic, "mov")
        self.assertEqual(x86_64.immediate_target(last[1]), ret)
        raw = x86_64.build_profile_detour_raw(
            7, self.EVENTS, [None], 0x500000, 0x900000, 0x400000, len(orig), orig
        )
        self.assertEqual(raw, out)

    def test_build_func_entry_and_exit(self):
        orig = bytes(x86_64.assemble("mov eax, 42", 0x400000))
        ret = 0x400000 + len(orig)
        entry = x86_64.build_profile_func_entry(
            7,
            self.EVENTS,
            [None],
            0x500000,
            0x900000,
            ret,
            orig_bytes=orig,
            shadow_top=0x600000,
            exit_addr=0x700000,
        )
        self.assertIn(orig, entry)
        self.assertIn(bytes(x86_64.assemble("movabs r11, 0x700000")), entry)
        raw = x86_64.build_profile_func_entry_raw(
            7,
            self.EVENTS,
            [None],
            0x500000,
            0x900000,
            0x400000,
            len(orig),
            orig,
            shadow_top=0x600000,
            exit_addr=0x700000,
        )
        self.assertEqual(raw, entry)
        for tid_off in (None, 0x10):
            exit_blob = x86_64.build_profile_func_exit(
                7,
                self.EVENTS,
                [None],
                0x500000,
                0x900000,
                shadow_top=0x600000,
                tid_off=tid_off,
            )
            self.assertTrue(exit_blob)
            self.assertIn(bytes(x86_64.assemble("movabs rcx, 0x600000")), exit_blob)
            tail = list(x86_64.disassembler().disasm(exit_blob, 0x900000))[-1]
            self.assertEqual(tail.mnemonic, "jmp")
            self.assertIn("rsp - 8", tail.op_str)
            if tid_off is not None:
                self.assertIn(
                    bytes(x86_64.assemble("mov eax, dword ptr fs:[0x10]")),
                    exit_blob,
                )


class TestMemoryAndCacheHelpers(unittest.TestCase):
    def test_optimal_nops_cover_every_size(self):
        for size in range(0, 21):
            blob = x86_64.optimal_nops(size)
            self.assertEqual(len(blob), size)
            if size:
                decoded = list(x86_64.disassembler().disasm(blob, 0))
                self.assertTrue(decoded)
                self.assertTrue(all(insn.mnemonic == "nop" for insn in decoded))
        self.assertEqual(x86_64.optimal_nops(-1), b"")
        self.assertEqual(
            x86_64.optimal_nops(10),
            x86_64._NOP_PATTERNS[5] + x86_64._NOP_PATTERNS[5],
        )

    def test_patch_branch(self):
        blob = bytearray(10)
        self.assertTrue(x86_64.patch_branch(blob, 0, 2, 4))
        self.assertEqual(blob[1], 4)
        blob = bytearray(10)
        self.assertTrue(x86_64.patch_branch(blob, 0, 3, 4))
        self.assertEqual(blob[2], 4)
        blob = bytearray(10)
        self.assertTrue(x86_64.patch_branch(blob, 0, 5, 0x1234))
        self.assertEqual(struct.unpack_from("<i", blob, 1)[0], 0x1234)
        blob = bytearray(10)
        self.assertTrue(x86_64.patch_branch(blob, 0, 6, 0x1234))
        self.assertEqual(struct.unpack_from("<i", blob, 2)[0], 0x1234)
        self.assertFalse(x86_64.patch_branch(bytearray(10), 0, 2, 200))
        self.assertFalse(x86_64.patch_branch(bytearray(10), 0, 5, 2**40))
        self.assertFalse(x86_64.patch_branch(bytearray(10), 0, 4, 1))
        self.assertFalse(x86_64.patch_branch(bytearray(4), 2, 5, 1))
        self.assertFalse(x86_64.patch_branch(bytearray(4), -1, 2, 1))

    def test_has_jump_table(self):
        cases = {
            "jmp [rax]": True,
            "mov eax, [rdi + rax*4]; jmp rax": True,
            "mov eax, [rdi + rax*4]; nop; ret": False,
            "jmp rax": False,
            "nop; ret": False,
        }
        for asm, expected in cases.items():
            blob = bytes(x86_64.assemble(asm, 0x400000))
            self.assertEqual(x86_64.has_jump_table(blob, 0x400000), expected, asm)

    def test_tier_family_and_shortcuts(self):
        self.assertEqual(x86_64._tier_family("DRAM"), x86_64._DATA_TIERS)
        self.assertEqual(x86_64._tier_family("L1i"), ("L1i",))
        self.assertEqual(x86_64._tier_family("TLBd"), ("TLBd",))
        self.assertEqual(x86_64._tier_family("bogus"), ("bogus",))
        self.assertEqual(x86_64._shortcut_levels(("L1i",), "hot"), {"L1i": 100})
        self.assertEqual(x86_64._shortcut_levels(("TLBd",), "cold"), {"TLBd": 0})
        self.assertIsNone(x86_64._shortcut_levels(("L1i",), "bogus"))
        self.assertIsNone(x86_64._shortcut_levels(("L1i",), 3))

    def test_tier_levels_variants(self):
        self.assertEqual(
            x86_64._tier_levels({"icache": {"hit_rate": 25}}, "icache", ("L1i",)),
            {"L1i": 25},
        )
        self.assertEqual(
            x86_64._tier_levels({"icache": 50}, "icache", ("L1i",)), {"L1i": 50}
        )
        with self.assertRaisesRegex(ValueError, "unknown cache shortcut"):
            x86_64._tier_levels({"icache": "bogus"}, "icache", ("L1i",))
        self.assertIsNone(x86_64._tier_levels({"icache": {}}, "icache", ("L1i",)))
        self.assertIsNone(x86_64._tier_levels({}, "icache", ("L1i",)))
        with self.assertRaises(ValueError):
            x86_64._tier_levels({"icache": {"L1d": 50}}, "icache", ("L1i",))

    def test_entry_rate_variants(self):
        self.assertEqual(x86_64._entry_rate("L1d", {"hit_rate": 30}), 30)
        self.assertEqual(x86_64._entry_rate("L1d", 40), 40)
        self.assertEqual(x86_64._entry_rate("L1d", "hot"), 100)
        self.assertEqual(x86_64._entry_rate("L2", "hot"), 0)
        self.assertEqual(x86_64._entry_rate("L1d", "hit_rate: 60"), 60)
        self.assertEqual(x86_64._entry_rate("L1d", " "), None)
        self.assertEqual(x86_64._entry_rate("L1d", None), None)
        self.assertEqual(x86_64._entry_rate("L1d", True), None)
        self.assertEqual(x86_64._entry_rate("L1d", {"L1d": 70}), 70)
        self.assertEqual(x86_64._entry_rate("L1d", {"other": 1}), None)
        self.assertEqual(
            x86_64._entry_rate(
                "L1d", {"default": {"hit_rate": 20}, "0x1000": {"hit_rate": 90}}
            ),
            20,
        )

    def test_merge_mem_cache(self):
        out = {}
        x86_64._merge_mem_cache(out, 0x1000, {"L1d": 50})
        self.assertEqual(out, {0x1000: {"L1d": 50}})
        x86_64._merge_mem_cache(out, 0x1000, {"L2": 25})
        self.assertEqual(out, {0x1000: {"L1d": 50, "L2": 25}})
        x86_64._merge_mem_cache(out, 0x2000, None)
        self.assertEqual(set(out), {0x1000})

    def test_collect_cache_dicts(self):
        config = {"dcache": {"a": 1}, "icache": {"b": 2}, "dtlb": 3}
        self.assertEqual(x86_64._collect_cache_dicts(config), [{"a": 1}, {"b": 2}])
        self.assertEqual(x86_64._collect_cache_dicts(None), [])
        self.assertEqual(x86_64._collect_cache_dicts({}), [])

    def test_address_and_register_key_predicates(self):
        self.assertTrue(x86_64._is_addr_key("0x1000"))
        self.assertTrue(x86_64._is_addr_key(4096))
        self.assertFalse(x86_64._is_addr_key("rdi"))
        self.assertTrue(x86_64._is_reg_key("rdi"))
        self.assertFalse(x86_64._is_reg_key("0x1000"))
        self.assertFalse(x86_64._is_reg_key("L1d"))
        self.assertFalse(x86_64._is_reg_key("default"))
        self.assertFalse(x86_64._is_reg_key("hit_rate"))
        self.assertFalse(x86_64._is_reg_key(""))
        self.assertEqual(
            x86_64._split_default_addrs(
                {"default": {"hit_rate": 50}, "0x1000": {"hit_rate": 100}}
            ),
            ({"hit_rate": 50}, {0x1000: {"hit_rate": 100}}),
        )
        self.assertEqual(x86_64._split_default_addrs("hot"), ("hot", {}))
        self.assertEqual(
            x86_64._split_default_regs({"rdi": "hot", "0x1000": 1}), {"rdi": "hot"}
        )
        self.assertEqual(x86_64._split_default_regs("hot"), {})

    def test_evict_groups_partition_addresses(self):
        addresses = [0x1000, 0x2000, 0x3000, 0x4000]
        mem_levels = {
            0x1000: {"L1d": 100, "TLBd": 100},
            0x2000: {"L1i": 0, "TLBi": 100},
            0x3000: {"L1d": 0, "L2": 0, "L3": 0, "TLBd": 0},
            0x4000: {"TLBd": 0},
        }
        groups = x86_64._evict_groups(addresses, lambda _tier: 0, mem_levels=mem_levels)
        self.assertEqual(groups["L1d"], [0x1000])
        self.assertEqual(groups["L1i"], [0x2000])
        self.assertEqual(groups["TLBd"], [0x3000, 0x4000])
        self.assertEqual(groups["DRAM"], [0x3000, 0x4000])

    def test_rate_parsing(self):
        self.assertEqual(x86_64._rate(True), None)
        self.assertEqual(x86_64._rate(120), 100)
        self.assertEqual(x86_64._rate(-5), 0)
        self.assertEqual(x86_64._rate("hit_rate = 42"), 42)
        self.assertEqual(x86_64._rate("hit_rate: 420"), 100)
        self.assertIsNone(x86_64._rate("bogus"))
        self.assertIsNone(x86_64._rate(None))
        self.assertIsNone(x86_64._memory_shortcut(3))

    def test_store_and_first_value_helpers(self):
        self.assertEqual(x86_64._first_val([]), 0)
        self.assertEqual(x86_64._first_val([1, 2]), 1)
        self.assertEqual(x86_64._first_val(5), 5)
        self.assertEqual(
            x86_64._store_asm(0x1000, 1),
            ["mov r12, 0x1000", "mov byte ptr [r12], 0x1"],
        )
        self.assertEqual(
            x86_64._store_asm(0x1000, 0x1234)[1], "mov dword ptr [r12], 0x1234"
        )
        self.assertEqual(
            x86_64._store_asm(0x1000, 2**40),
            ["mov r12, 0x1000", "mov rax, 0x10000000000", "mov qword ptr [r12], rax"],
        )
        self.assertEqual(
            x86_64._store_asm(0x1000, "bad"),
            ["mov r12, 0x1000", "mov byte ptr [r12], 0x0"],
        )
        self.assertEqual(x86_64._is_pow2(0), False)
        self.assertEqual(x86_64._is_pow2("bad"), False)
        self.assertEqual(x86_64._is_pow2(64), True)


if __name__ == "__main__":
    unittest.main()
