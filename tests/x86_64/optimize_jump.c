extern int hot(int);

int main(void) {
  volatile int r = 0;
  for (int i = 0; i < 16; i++)
    r += hot(i);
  return r == 1503 ? 0 : 1;
}
