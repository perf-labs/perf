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


class TestBenchTemplates(unittest.TestCase):
    def test_modes_present(self):
        self.assertIn("latency", x86_64.BENCH)
        self.assertIn("throughput", x86_64.BENCH)

    def test_placeholders_format(self):
        t0, t1 = x86_64.timing()
        for mode, templ in x86_64.BENCH.items():
            out = templ.format(
                code="nop",
                setup="",
                teardown="",
                data="",
                data2="",
                data_iter="",
                t0=t0,
                t1=t1,
            )
            self.assertNotIn("{setup}", out)
            self.assertNotIn("{code}", out)
            self.assertNotIn("{teardown}", out)
            self.assertNotIn("{data}", out)
            self.assertNotIn("{data2}", out)
            self.assertNotIn("{data_iter}", out)
            self.assertNotIn("{t0}", out)
            self.assertNotIn("{t1}", out)

    def test_template_delegates_measurement(self):
        for mode in ("latency", "throughput"):
            self.assertIn("{t0}", x86_64.BENCH[mode])
            self.assertIn("{t1}", x86_64.BENCH[mode])
            self.assertIn("{data}", x86_64.BENCH[mode])
            self.assertIn("{data2}", x86_64.BENCH[mode])

    def test_throughput_wraps_loop_in_timing(self):
        lat = x86_64.BENCH["latency"]
        thr = x86_64.BENCH["throughput"]
        self.assertLess(thr.index("{t0}"), thr.index(".loop:"))
        self.assertGreater(thr.index("{t1}"), thr.index("jnz .loop"))
        self.assertLess(lat.index(".loop:"), lat.index("{t0}"))

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
            self.assertEqual(asm.count("pause"), x86_64.SETTLE_PAUSES)
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

    def test_evict_tlb_tiers_use_invlpg(self):
        _addrs = [0x400000 + i * 0x1000 for i in range(2)]
        mem_levels = {a: {"TLBd": 100} for a in _addrs}
        asm = x86_64.evict(_addrs, mem_levels=mem_levels)
        self.assertEqual(asm.count("syscall"), 4)
        self.assertNotIn("clflushopt", asm)
        self.assertNotIn("invlpg", asm)
        asm = x86_64.evict(_addrs, mem_levels={a: {"TLBi": 100} for a in _addrs})
        self.assertEqual(asm.count("syscall"), 4)
        self.assertNotIn("clflushopt", asm)
        self.assertNotIn("invlpg", asm)

    def test_evict_tlb_inval_pages_aligned(self):
        _addrs = [0x400000 + i * 0x1000 + 0x123 for i in range(2)]
        mem_levels = {a: {"TLBd": 100} for a in _addrs}
        asm = x86_64.evict(_addrs, mem_levels=mem_levels)
        for a in _addrs:
            page = a & ~0xFFF
            self.assertIn(f"mov rdi, 0x{page:x}", asm)
        expected = x86_64.tlb_inval_asm(_addrs[0])
        self.assertIn(expected, asm)

    def test_tlb_inval_asm_toggles_prot(self):
        asm = x86_64.tlb_inval_asm(0x4100001000)
        self.assertIn("mov rdi, 0x4100001000", asm)
        self.assertIn("mov rsi, 0x1000", asm)
        self.assertIn("mov rdx, 0", asm)
        self.assertIn("mov rax, 10", asm)
        self.assertIn("syscall", asm)
        self.assertIn("mov rdx, 0x7", asm)
        self.assertGreaterEqual(asm.count("syscall"), 2)
        self.assertGreaterEqual(asm.count("mov rax, 10"), 2)

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
            asm = x86_64.evict(_addrs, mem_levels={a: {tier: 100} for a in _addrs})
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

    def test_addr_tier_routes_tlb(self):
        self.assertEqual(x86_64.addr_tier({"TLBd": 100}), "TLBd")
        self.assertEqual(x86_64.addr_tier({"TLBi": 50, "L1d": 40}), "TLBi")
        self.assertEqual(x86_64.addr_tier({"TLBd": 0}), "TLBd")

    def test_resolve_mem_cache_tlb_tags(self):
        cfg = {"caches": None}
        cfg = {"cache": {"TLBd": {"0x1800": 100}}}
        out = x86_64.resolve_mem_cache(cfg)
        self.assertEqual(out[0x1800], {"TLBd": 100})
        cfg = {"cache": {"TLBi": {"0x2000": "hit_rate=75"}}}
        out = x86_64.resolve_mem_cache(cfg)
        self.assertEqual(out[0x2000], {"TLBi": 75})
        for tier in x86_64._TLB_TIERS:
            self.assertIn(tier, x86_64._CACHE_TIER_TAGS)

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
        self.assertEqual(asm.count("pause"), x86_64.SETTLE_PAUSES)

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
            data_cache_levels({"cache": "hot"}), {"L1d": 100, "L2": 0, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"cache": "warm"}), {"L1d": 0, "L2": 100, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"cache": "cool"}), {"L1d": 0, "L2": 0, "L3": 100}
        )
        self.assertEqual(
            data_cache_levels({"cache": "cold"}), {"L1d": 0, "L2": 0, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"cache": "HOT"}), {"L1d": 100, "L2": 0, "L3": 0}
        )
        self.assertEqual(
            data_cache_levels({"cache": "cool"}),
            {"L1d": 0, "L2": 0, "L3": 100},
        )
        self.assertEqual(
            data_cache_levels({"cache": "cold"}),
            {"L1d": 0, "L2": 0, "L3": 0},
        )
        self.assertEqual(
            data_cache_levels({"cache": "warm"}),
            {"L1d": 0, "L2": 100, "L3": 0},
        )
        with self.assertRaises(ValueError):
            data_cache_levels({"cache": "lukewarm"})

    def test_sample_tier_cascade(self):
        import random

        from perf.arch.x86_64 import sample_tier

        rng = random.Random(0)
        tiers = {sample_tier({"L1d": 50, "L2": 50, "L3": 100}, rng) for _ in range(500)}
        self.assertEqual(tiers, {"L1d", "L2", "L3"})
        self.assertEqual(
            {
                sample_tier({"L1d": 0, "L2": 0, "L3": 0}, random.Random(1))
                for _ in range(50)
            },
            {"DRAM"},
        )


class TestMovedX86Helpers(unittest.TestCase):
    def test_perf_syscall_nr(self):
        self.assertEqual(x86_64.PERF_SYSCALL_NR, 298)

    def test_angr_names_and_bases(self):
        self.assertEqual(x86_64.ANGR_ARCH_NAME, "AMD64")
        self.assertEqual(x86_64.SETUP_BASE, 0x1000000)
        self.assertEqual(x86_64.ASM_SCRATCH_BASE, 0x4100001000)
        self.assertEqual(x86_64.STACK_ADDR, 0x7FFF00000000)
        self.assertEqual(x86_64.STACK_SIZE, 0x100000)
        self.assertEqual(x86_64.PRIME_SCRATCH_REG, "r14")

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
        enc, _ = ks.asm(x86_64.TSC_READER_ASM)
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
            x86_64.STACK_ADDR, x86_64.STACK_SIZE, 7
        )

        state = Mock()
        x86_64.set_ip(state, 0x1234)
        self.assertEqual(state.regs.rip, 0x1234)

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
            "cache": {
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
            "cache": {
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
                "cache": {
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
            "cache": {
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
            x86_64.resolve_mem_cache({"cache": {"zzz": "bogus"}}),
            {},
        )

    def test_addr_tier(self):
        self.assertEqual(x86_64.addr_tier({"L1d": 100, "L2": 0, "L3": 0}), "L1d")
        self.assertEqual(x86_64.addr_tier({"L1d": 0, "L2": 100, "L3": 0}), "L2")
        self.assertEqual(x86_64.addr_tier({"L1d": 0, "L2": 0, "L3": 100}), "L3")
        self.assertEqual(x86_64.addr_tier({"L1d": 0, "L2": 0, "L3": 0}), "DRAM")

        self.assertEqual(x86_64.addr_tier({"L1d": 100, "L2": 100}), "L1d")
        self.assertEqual(x86_64.addr_tier(None), "DRAM")

    def test_addr_tier_l1i(self):
        self.assertEqual(x86_64.addr_tier({"L1i": 0}), "L1i")
        self.assertEqual(x86_64.addr_tier({"L1d": 100, "L1i": 0}), "L1i")
        self.assertEqual(x86_64.addr_tier({"L1i": 100}), "L1i")
        self.assertEqual(x86_64.addr_tier({"L1d": 100, "L2": 0, "L3": 0}), "L1d")

    def test_resolve_mem_cache_l1i(self):
        cfg = {
            "cache": {
                "L1i": {
                    "default": {"hit_rate": 100},
                    "0x401000": {"hit_rate": 0},
                }
            }
        }
        got = x86_64.resolve_mem_cache(cfg)
        self.assertEqual(got[0x401000], {"L1i": 0})
        self.assertNotIn("default", got)
        self.assertIsNone(x86_64.data_cache_levels(cfg))

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
        cfg = {"cache": {"L1i": {"0x401234": {"hit_rate": 0}}}}
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
            "cache": {
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

        self.assertNotIn("rspd", x86_64.HARNESS_REGS)

    def test_not_does_not_write_flags(self):
        from perf.arch import x86_64

        self.assertNotIn("not", x86_64.FLAG_WRITERS)


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
        cfg = {"cache": {"l1d": {"hit_rate": 100}}}
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


if __name__ == "__main__":
    unittest.main()
