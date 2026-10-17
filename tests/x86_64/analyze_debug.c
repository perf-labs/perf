long myfunc(long x){
  long sum = 0;
  for (int i = 0; i < 4; i++) {
    if (x > 2) sum += x * i;
    else sum -= i;
  }
  return sum;
}
int main() { return (int)myfunc(3); }
