__attribute__((noinline)) long jmp_next(long x) {
  __asm__ volatile("jmp 1f\n1:" : "+r"(x) : : "cc");
  return x + 1;
}

__attribute__((noinline)) long jcc_next(long x) {
  __asm__ volatile("test %0, %0\nje 1f\n1:" : "+r"(x) : : "cc");
  return x + 2;
}

__attribute__((noinline)) long jmp_fold(long x) {
  __asm__ volatile(
      "jmp 1f\n"
      "nop\n"
      "1:\n"
      "jmp 2f\n"
      "nop\n"
      "2:\n"
      : "+r"(x)
      :
      : "cc");
  return x + 3;
}

__attribute__((noinline)) long jcc_fold(long x) {
  __asm__ volatile(
      "test %0, %0\n"
      "je 1f\n"
      "nop\n"
      "1:\n"
      "jmp 2f\n"
      "2:\n"
      : "+r"(x)
      :
      : "cc");
  return x + 4;
}

__attribute__((noinline)) long jcc_invert(long x) {
  __asm__ volatile(
      "test %0, %0\n"
      "jne 1f\n"
      "jmp 2f\n"
      "1:\n"
      "nop\n"
      "2:\n"
      : "+r"(x)
      :
      : "cc");
  return x + 5;
}

int main(void) {
  volatile long r = jmp_next(1) + jcc_next(1) + jmp_fold(1) + jcc_fold(1) +
                    jcc_invert(1);
  (void)r;
  return 0;
}
