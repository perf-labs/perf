export fn work_zig(n: i32) i32 {
    var s: i32 = 1;
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        s = s * 3 + i;
    }
    return s;
}
pub fn main() void {
    _ = work_zig(10);
}
