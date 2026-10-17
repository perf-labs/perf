#[inline(never)]
#[no_mangle]
pub extern "C" fn work_rs(n: i32) -> i32 {
    let mut s: i32 = 1;
    let mut i = 0;
    while i < n { s = s * 3 + i; i += 1; }
    s
}
fn main() { println!("{}", work_rs(10)); }
