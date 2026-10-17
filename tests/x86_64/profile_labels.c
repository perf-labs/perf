#include "perf/perf.h"
#include <stdio.h>
int work(int n) {
    int s = 0;
    PERF_LABEL(loop_begin);
    for (int i = 0; i < n; i++) s += i;
    PERF_LABEL(loop_end);
    return s;
}
int main() { printf("%d\n", work(100)); return 0; }
