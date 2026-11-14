#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define HEAP_BASE 0x5200000000ULL
#define HEAP_END 0x5300000000ULL

static int *last_data;
static int last_size;
static int saw_unsorted;

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (x > y) - (x < y);
}

static int is_sorted(const int *data, int size) {
    for (int i = 1; i < size; ++i) {
        if (data[i - 1] > data[i]) {
            return 0;
        }
    }
    return 1;
}

static int is_heap(const int *data) {
    uintptr_t addr = (uintptr_t)data;
    return addr >= HEAP_BASE && addr < HEAP_END;
}

int heap_sort(int *data, int size) {
    if (is_heap(data)) {
        last_data = data;
        last_size = size;
        if (!is_sorted(data, size)) {
            saw_unsorted = 1;
        }
    }
    qsort(data, size, sizeof(int), cmp_int);
    return 0;
}

void check_heap_sort(void) {
    if (saw_unsorted && last_data != NULL && is_sorted(last_data, last_size)) {
        fprintf(stderr, "HEAP_SORT_OK\n");
    } else {
        fprintf(stderr, "HEAP_SORT_BAD\n");
    }
}

int main(void) { return 0; }
