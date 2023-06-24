#include <atomic>
#include <chrono>

#include "ggml-thread.hpp"

// 98% cpu, heavy context switch.
// real	0m4.372s
// user	0m3.051s
// sys	0m1.209s
void test_spin_yield_forever() {
#ifdef SPIN_YIELD
    fprintf(stderr, "%s:\n", __func__);
    while (true) {
        std::this_thread::yield();
    }
#else
    fprintf(stderr, "%s: skipped\n", __func__);
#endif
}

static void test_spin_yield() {
#ifdef SPIN_YIELD
    fprintf(stderr, "%s:\n", __func__);
    for (int i = 0; i < 1000000; i++) {
        std::this_thread::yield();
    }
#else
    fprintf(stderr, "%s: skipped\n", __func__);
#endif
}

// 100% cpu
// real	0m5.357s
// user	0m5.280s
// sys	0m0.014s
void test_spin_nop_forever() {
#ifdef SPIN_NOP
    const int multiplier = 32;
    fprintf(stderr, "%s (multiplier = %d):\n", __func__, multiplier);
    while (true) {
        spin_nop_32_x_(multiplier);
    }
#else
    fprintf(stderr, "%s: skipped\n", __func__);
#endif
}

// watch CPU load for a while
static void test_spin_nop() {
#ifdef SPIN_NO
    const int multiplier = 32;
    fprintf(stderr, "%s (multiplier = %d):\n", __func__, multiplier);
    for (int i = 0; i < 1000000; i++) {
        spin_nop_32_x_(multiplier);
    }
#else
    fprintf(stderr, "%s: skipped\n", __func__);
#endif
}

// 100% cpu
// real	0m5.039s
// user	0m4.978s
// sys	0m0.010s
#ifdef SPIN_MEM_PAUSE
static void test_mem_pause_forever() {
    fprintf(stderr, "%s:\n", __func__);
    while (true) {
        mem_pause();
    }
}
#endif

static void test_mem_pause() {
#ifdef SPIN_MEM_PAUSE
    fprintf(stderr, "%s:\n", __func__);
    for (int i = 0; i < 1000000; i++) {
        mem_pause();
    }
#else
    fprintf(stderr, "%s: skipped\n", __func__);
#endif
}

static void test_spin_pause() {
    printf("%s()\n", __func__);

    for (int i = 0; i < 3; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            test_spin_nop();
        } else if (i == 1) {
            test_mem_pause();
        } else {
            test_spin_yield();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapse = (end_time - start_time) / std::chrono::milliseconds(1);
        fprintf(stderr, "   elapsed: %d us\n", (int)elapse);
    }
}

static std::atomic<int> compute_counter;
struct DemoWork {
    void compute() {
        compute_counter.fetch_add(1, std::memory_order_relaxed);
        // std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
};

void test_threading() {
    printf("%s()\n", __func__);

    atomic_store(&compute_counter, 0);
    constexpr int n_workers = 6;
    auto m = new ggml::Producer<DemoWork>(n_workers);
    struct DemoWork works[n_workers];

    m->start();

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        m->suspend();
        m->resume();
    }
    m->suspend();

    for (int i = 0; i < 100; i++) {
        m->resume();
    }
    m->suspend();

    int expected = 0;
    for (int i = 0; i < 100; i++) {
        if (i % 10 == 0) {
            m->resume();
            m->assign(works, n_workers);
            m->suspend();
            expected += n_workers;
        } else {
            m->assign(works, n_workers);
            expected += n_workers;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapse = 1e-3 * (end_time - start_time) / std::chrono::milliseconds(1);
    int actual = atomic_load(&compute_counter);

    fprintf(stderr, "%d workers, scheduled %d times, elapsed: %.3f ms\n",
            n_workers, actual, elapse);

    if (actual != expected) {
        fprintf(stderr, "actual: %d, expected: %d, failed\n", actual, expected);
    } else {
        fprintf(stderr, "pass\n");
    }

    m->stop();
}

// make clean && make
// $ time ./bin/test-ggml-thread
int main() {
    print_build_options();
    fprintf(stderr, "\n");

    // test_spin_nop_forever();
    // test_mem_pause_forever();
    // test_spin_yield_forever();

    test_spin_pause();
    fprintf(stderr, "\n");

    test_threading();
}
