// #define USE_ACCELERATE
// #define USE_OPENBLAS
// #define USE_BLIS

#if defined(USE_OPENBLAS)
#include <cblas.h>
#elif defined(USE_BLIS)
#include <blis.h>
#elif defined(USE_ACCELERATE) || defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#error "cblas vendor not found"
#endif

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int64_t time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

int64_t time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + (int64_t)ts.tv_nsec;
}

void cblas_sgemm_wrapper(int M, int N, int K) {
    int64_t sizeA = sizeof(float) * M * K;
    int64_t sizeB = sizeof(float) * N * K;
    int64_t sizeC = sizeof(float) * M * N;

    float *A = malloc(sizeA);
    float *B = malloc(sizeB);
    float *C = malloc(sizeC);

    memset(A, 0, sizeA);
    memset(B, 0, sizeB);
    memset(C, 0, sizeC);

    const int lda = K;
    const int ldb = K;
    const int ldc = N;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, lda,
                B, ldb, 0.0f, C, ldc);
    int64_t t1 = time_us();

    free(A);
    free(B);
    free(C);
}

struct thread_data_T {
    atomic_int M;
    atomic_int N;
    atomic_int K;

    atomic_int *n_ready;
    atomic_int *n_tasks;
    atomic_int *n_done;
    atomic_bool *stop;

    int ith;

    int time_us;
};

void *multi_threads_N_chunks_runner(void *arg) {
    struct thread_data_T *v = (struct thread_data_T *)arg;

    int n_tasks = 0;

    atomic_fetch_add(v->n_ready, 1);

    printf("#-%d is ready to run\n", v->ith);

    while (!atomic_load(v->stop)) {
        while (atomic_load(v->n_tasks) == n_tasks) {
            if (atomic_load(v->stop)) {
                printf("#-%d stopped\n", v->ith);
                return NULL;
            }
        }

        ++n_tasks;

        int M = atomic_load(&v->M);
        int N = atomic_load(&v->N);
        int K = atomic_load(&v->K);

        // printf("#-%d is computing the %d-th task\n", v->ith, n_tasks);

        int64_t t0 = time_us();
        cblas_sgemm_wrapper(M, N, K);
        v->time_us = (time_us() - t0);
        printf("%d-th: %5.3f ms\n", v->ith, 1e-3 * v->time_us);
        atomic_fetch_add(v->n_done, 1);
    }

    printf("#-%d stopped\n", v->ith);

    return NULL;
}

// split N as trunks.
// B: trans
void bench_multi_threads_N_chunks(void) {
    // Radeon Pro 560X 4 GB
    // K(8192) * N(8192) * M(64) = 4GB, suddenly go slow when M > 64.

    const int max_N = 4096;
    const int K = 4096;

    const int n_threads = 4;

    pthread_t pids[n_threads];
    struct thread_data_T args[n_threads];

    atomic_int n_ready = 0;
    atomic_int n_tasks = 0;
    atomic_int n_done = 0;
    atomic_bool stop = false;

    const int n_workers = n_threads - 1;

    // args[0]: main involves in computing.
    for (int i = 0; i < n_threads; i++) {
        args[i].ith = i;
        args[i].n_ready = &n_ready;
        args[i].n_tasks = &n_tasks;
        args[i].n_done = &n_done;
        args[i].stop = &stop;

        if (i > 0) {
            pthread_create(&pids[i], NULL, multi_threads_N_chunks_runner,
                           &args[i]);
        }
    }

    while (atomic_load(&n_ready) != n_workers) {
    }

    printf("main saw %d workers ready to run\n", n_workers);

    // for (int j = 5; j < 12; j++) {
    for (int j = 0; j < 12; j++) {
        int M = 1 << j;

        int64_t original_time_us = 0;
        {
            int N = max_N;

            {
                // warm up
                cblas_sgemm_wrapper(M, N, K);
            }

            int64_t t0 = time_us();
            cblas_sgemm_wrapper(M, N, K);

            int64_t t1 = time_us();
            original_time_us = (time_us() - t0);
        }

        const int min_chunk_size = 32;
        const int per_thread_max_N = (max_N + n_threads - 1) / n_threads;

        printf("\nM: %d, K: %d, max_N=%d, n_threads: %d, per_thread_max_N: %dn",
               M, K, max_N, n_threads, per_thread_max_N);

        for (int i = 0;; ++i) {
            int N = min_chunk_size * (i + 1);
            if (N > per_thread_max_N) {
                break;
            }

            printf("==== M: %d, K: %d, N=%d ===\n", M, K, N);

            for (int i = 0; i < n_threads; i++) {
                atomic_store(&args[i].M, M);
                atomic_store(&args[i].N, N);
                atomic_store(&args[i].K, K);
            }

            atomic_fetch_add(&n_tasks, 1);

            int64_t t0 = time_us();
            cblas_sgemm_wrapper(M, N, K);
            args[0].time_us = (time_us() - t0);
            printf("%d-th: %5.3f ms\n", 0, 1e-3 * args[0].time_us);

            atomic_fetch_add(&n_done, 1);

            while (atomic_load(&n_done) != n_threads) {
            }

            // NOTE: avg time totally useless, we'd use max time. Why?
            // - OpenBLAS serialize runs partly or totally
            // - We have to wait for the slowest thread before next run.
            int64_t slowest_time = 0;
            for (int i = 0; i < n_threads; i++) {
                if (args[i].time_us > slowest_time) {
                    slowest_time = args[i].time_us;
                }
            }

            int64_t equivalent_time_us =
                slowest_time * ((per_thread_max_N + N - 1) / N);

            double ratio = 1.0 * equivalent_time_us / original_time_us;

            printf("== M: %d, K: %d, max_N=%d: %6.3f ms, chunk_N=%d: slowest "
                   "%6.3f ms. Equivalent: %6.3f ms, new/old: %4.2f%% ==\n\n",
                   M, K, max_N, 1e-3 * original_time_us, N, 1e-3 * slowest_time,
                   1e-3 * equivalent_time_us, 100.0 * ratio);

            atomic_store(&n_done, 0);
        }
    }

    atomic_store(&stop, true);

    for (int i = 1; i <= n_workers; i++) {
        pthread_join(pids[i], NULL);
    }
}

int main() {

    // Conclusions:
    // - ACCELERATE is the fastest. Multi-threading almost does no help.
    // - OpenBLAS slightly slower. Simillar to ACCELERATE, looks like it also
    //   serializes concurrent runs.
    // - BLIS is the slowest one, but looks like it parallels concurrent runs.
    //   Multi-threading (4 of 6 physical cores) speedup significantly, but does
    //   not compete others: far too slow at 1 thread.
    bench_multi_threads_N_chunks();

    // TODO: evaluate BLIS multi-threading,
    // https://github.com/flame/blis/issues/644
    return 0;
}