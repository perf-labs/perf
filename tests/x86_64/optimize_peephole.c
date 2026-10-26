__attribute__((noinline)) long load_const(void) {
  long r;
  __asm__ volatile("movabs $0x1234, %0" : "=r"(r));
  return r;
}

__attribute__((noinline)) long movself(long x) {
  __asm__ volatile("mov %0, %0" : "+r"(x));
  return x;
}

__attribute__((noinline)) long leaself(long x) {
  __asm__ volatile("lea (%0), %0" : "+r"(x));
  return x;
}

__attribute__((noinline)) long xchgself(long x) {
  __asm__ volatile(".byte 0x48, 0x87, 0xc0" : "+r"(x));
  return x;
}

__attribute__((noinline)) long shiftzero(long x) {
  __asm__ volatile("shl $0, %0" : "+r"(x));
  return x;
}

__attribute__((noinline)) void nopnorm(void) {
  __asm__ volatile(".byte 0x66, 0x66, 0x90");
}

__attribute__((noinline)) void jmpnext(void) { __asm__ volatile("jmp 1f\n1:"); }

int main(void) {
  volatile long a = load_const();
  volatile long b = movself(a);
  volatile long c = leaself(b);
  volatile long d = xchgself(c);
  volatile long e = shiftzero(d);
  volatile long f = a + b + c + d + e;
  jmpnext();
  nopnorm();
  (void)f;
  return 0;
}
