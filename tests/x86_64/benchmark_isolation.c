#include <stdlib.h>
#include <unistd.h>

int stay(int n) { return n * 2; }

int leave(int n) {
    exit(0);
    return n;
}

int crash(int n) {
    __builtin_trap();
    return n;
}

int sleeper(int n) {
    usleep(n);
    return n;
}

int main(void) { return stay(1); }
