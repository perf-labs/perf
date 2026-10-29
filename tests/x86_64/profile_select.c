#include "perf/perf.h"
#include <stdio.h>

int work(int n) {
    int s = 0;
    PERF_LABEL(hot);
    for (int i = 0; i < n; i++) s += i;
    PERF_LABEL(cold);
    return s;
}

int main() {
    int r = work(100) + work(100);
    printf("%d\n", r);
    return 0;
}
