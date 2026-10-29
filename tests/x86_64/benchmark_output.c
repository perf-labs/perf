#include <stdio.h>

__attribute__((noinline)) long shout(long n) {
    fprintf(stderr, "err%ld\n", n);
    fprintf(stdout, "out%ld\n", n);
    return n + 1;
}

int main(void) { return (int)shout(1); }
