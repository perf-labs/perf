#include <stdio.h>
#include <string.h>

static const char *message = "layout fixups";
static int table[8] = {3, 1, 4, 1, 5, 9, 2, 6};

__attribute__((noinline)) int cold(int n) {
  int s = 0;
  for (int i = 0; i < n; i++)
    s += table[i & 7];
  return s;
}

__attribute__((noinline)) int hot(int n) {
  int s = 0;
  for (int i = 0; i < n; i++)
    s += (int)strlen(message) + cold(1);
  return s;
}

int main(void) {
  printf("%d %d\n", hot(3), cold(4));
  return 0;
}
