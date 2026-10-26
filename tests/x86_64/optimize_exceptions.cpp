#include <cstdio>
#include <stdexcept>

__attribute__((noinline)) int cold(int n) {
  if (n < 0)
    throw std::runtime_error("cold");
  return n + 1;
}

__attribute__((noinline)) int hot(int n) {
  int s = 0;
  for (int i = 0; i < n; i++)
    s += cold(i);
  return s;
}

int main() {
  int r = 0;
  try {
    r = hot(5);
  } catch (...) {
    r = -1;
  }
  try {
    cold(-1);
  } catch (const std::runtime_error &) {
    r += 100;
  }
  std::printf("%d\n", r);
  return 0;
}
