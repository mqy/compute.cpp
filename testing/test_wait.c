#include <emmintrin.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

atomic_int n_wait;
atomic_int n_waiting;

int64_t inline time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

int64_t inline time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + (int64_t)ts.tv_nsec;
}

void *thread_runner(void *arg) {
    int id = *(int *)arg;

    while (n_wait >= 0) {
        if (id < n_wait) {
            pthread_mutex_lock(&mutex);
            n_waiting++;
            // printf("#-%d: waiting\n", id);
            pthread_cond_wait(&cond, &mutex);
            // printf("#-%d: wakeup\n", id);
            n_waiting--;
            pthread_mutex_unlock(&mutex);

            if (n_wait < 0) {
                return 0;
            }
        }
    }

    return 0;
}

// gcc -O3 -std=c11 test_wait.c -o test_wait && ./test_wait
int main() {
    const int n_threads = 6; // 3
    const int n_loops = 1;   // 10

    pthread_t pids[n_threads];
    int ids[n_threads];

    n_wait = 0;
    for (int i = 0; i < n_threads; i++) {
        ids[i] = i;
        pthread_create(&pids[i], NULL, thread_runner, &ids[i]);
    }

    int64_t total_0 = 0;
    int64_t total_1 = 0;

    for (int i = 0; i < n_loops; i++) {
        int64_t t0 = time_ns();

        // wait.
        {
            n_wait = n_threads;
            while (n_waiting != n_wait) {
            }
        }

        int64_t t1 = time_ns();
        total_0 += (t1 - t0);

        // wake up.
        {
            pthread_mutex_lock(&mutex);
            n_wait = 0;
            pthread_cond_broadcast(&cond);
            pthread_mutex_unlock(&mutex);

            while (n_waiting != n_wait) {
            }
        }

        total_1 += (time_ns() - t1);
    }

    printf("n_threads: %d, n_loops: %d\n", n_threads, n_loops);
    printf("    avg_wait:   %6.3f us\n", 1.0 * total_0 / (1000 * n_loops));
    printf("    avg_wakeup: %6.3f us\n", 1.0 * total_1 / (1000 * n_loops));

    n_wait = -1;
    for (int i = 0; i < n_threads; i++) {
        pthread_join(pids[i], NULL);
    }

    return 0;
}