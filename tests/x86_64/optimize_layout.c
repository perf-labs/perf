int hot(int n) {
  int s = 0;
  for (int i = 0; i < n; i++)
    s += i;
  return s;
}
int cold(int n) { return n + 1; }
int main(void) {
  volatile int h = hot(100);
  volatile int c = cold(1);
  volatile int d = h + c;
  (void)d;
  return 0;
}
