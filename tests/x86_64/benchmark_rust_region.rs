macro_rules! perf_label {
    ($name:ident) => {{
        unsafe {
            core::arch::asm!(
                concat!(
                    ".pushsection .perf.label, \"awR?\", @progbits\n",
                    ".quad 2f\n",
                    ".asciz \"",
                    stringify!($name),
                    "\"\n",
                    ".popsection\n",
                    "2:\n",
                ),
                options(nostack, preserves_flags),
            );
        }
    }};
}
#[inline(never)]
#[no_mangle]
pub extern "C" fn work_rs(n: i32) -> i32 {
    let mut s: i32 = 1;
    perf_label!(loop_begin);
    let mut i = 0;
    while i < n { s = s * 3 + i; i += 1; }
    perf_label!(loop_end);
    s
}
fn main() { println!("{}", work_rs(10)); }
