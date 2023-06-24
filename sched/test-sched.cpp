#include <atomic>
#include <chrono>

#include "sched.hpp"

// 98% cpu, heavy context switch.
void test_spin_yield_forever() {
    fprintf(stderr, "%s:\n", __func__);
    while (true) {
        std::this_thread::yield();
    }
}

static void test_spin_yield() {
    fprintf(stderr, "%s:\n", __func__);
    for (int i = 0; i < 1000000; i++) {
        std::this_thread::yield();
    }
}

// 100% cpu
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
#ifdef SPIN_NOP
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
void test_mem_pause_forever() {
    fprintf(stderr, "%s:\n", __func__);
    while (true) {
        mem_pause();
    }
}

static void test_mem_pause() {
    fprintf(stderr, "%s:\n", __func__);
    for (int i = 0; i < 1000000; i++) {
        mem_pause();
    }
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
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
};

void test_sched() {
    printf("%s()\n", __func__);

    atomic_store(&compute_counter, 0);
    constexpr int n_workers = 4;
    auto m = new sched::Scheduler<DemoWork>(n_workers);
    struct DemoWork works[n_workers + 1];

    m->start();
    auto start_time = std::chrono::high_resolution_clock::now();
    const int n_loops = 100;

    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_loops; i++) {
            m->suspend();
            m->resume();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto elapsed = 1e-3 * (t1 - t0) / std::chrono::milliseconds(1);
        fprintf(
            stderr,
            "%d workers, suspend + resume for %d times, elapsed: %6.3f ms\n",
            n_workers, 2 * n_loops, elapsed);
    }

    m->suspend();
    // m->suspend(); // trigger assert error
    m->resume();
    // m->resume(); // trigger assert error

    {
        atomic_store(&compute_counter, 0);
        int compute_expected = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_loops; i++) {
            m->suspend();
            works[0].compute();
            works[0].compute();
            m->resume();
            m->compute(works + 1, /*suspend*/ false);
            compute_expected += 2 + n_workers;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto elapsed = 1e-3 * (t1 - t0) / std::chrono::milliseconds(1);
        fprintf(
            stderr,
            "%d workers, passive suspend before compute, elapsed: %6.3f ms\n",
            n_workers, elapsed);
        int actual = atomic_load(&compute_counter);
        if (actual != compute_expected) {
            fprintf(stderr, "actual compute: %d, expected: %d, failed\n",
                    actual, compute_expected);
        }
    }

    if (worker_compute_suspend_enable()) {
        atomic_store(&compute_counter, 0);
        int compute_expected = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_loops; i++) {
            m->resume();
            m->compute(works + 1, /*suspend*/ true);
            works[0].compute();
            works[0].compute();
            compute_expected += 2 + n_workers;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto elapsed = 1e-3 * (t1 - t0) / std::chrono::milliseconds(1);
        fprintf(
            stderr,
            "%d workers, active suspend after compute,   elapsed: %6.3f ms\n",
            n_workers, elapsed);
        int actual = atomic_load(&compute_counter);
        if (actual != compute_expected) {
            fprintf(stderr, "actual compute: %d, expected: %d, failed\n",
                    actual, compute_expected);
        }
    } else {
        fprintf(stderr, "    active suspend after compute, skip\n");
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed =
        1e-3 * (end_time - start_time) / std::chrono::milliseconds(1);
    fprintf(stderr, "%d workers, total time: %.3f ms\n", n_workers, elapsed);

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

    test_sched();
}
