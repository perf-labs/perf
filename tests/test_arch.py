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
        import keystone

        ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
        enc, _ = ks.asm(x86_64._TSC_READER_ASM)
        self.assertEqual(bytes(enc), code)

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
        for reg in ("eax", "ebx", "ecx", "edx", "esi", "edi", "r8d", "r9d"):
            self.assertIn(f"xor {reg}, {reg}", asm)
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


if __name__ == "__main__":
    unittest.main()
