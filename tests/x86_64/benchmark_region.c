#include "perf/perf.h"
long radd(long a, long b) {
  PERF_LABEL(hot_begin);
  long r = a + b;
  PERF_LABEL(hot_end);
  return r;
}
int main() { return 0; }
