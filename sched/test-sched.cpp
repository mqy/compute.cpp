#include <atomic>
#include <chrono>

#include "sched.hpp"

// 98% cpu, heavy context switch.
void test_yield_forever() {
    fprintf(stderr, "%s:\n", __func__);
    while (true) {
        std::this_thread::yield();
    }
}

static void test_yield() {
    fprintf(stderr, "%s:\n", __func__);
    for (int i = 0; i < 1000000; i++) {
        std::this_thread::yield();
    }
}

// 100% cpu
void test_spin_nop_forever() {
    const int multiplier = 32;
    fprintf(stderr, "%s (multiplier = %d):\n", __func__, multiplier);
    while (true) {
        sched::spin_nop_32_x_(multiplier);
    }
}

static void test_spin_nop() {
    const int multiplier = 32;
    fprintf(stderr, "%s (multiplier = %d):\n", __func__, multiplier);
    for (int i = 0; i < 1000000; i++) {
        sched::spin_nop_32_x_(multiplier);
    }
}

static void test_pause() {
    printf("%s()\n", __func__);

    for (int i = 0; i < 2; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        if (i == 0) {
            test_spin_nop();
        } else {
            test_yield();
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapse = (end_time - start_time) / std::chrono::milliseconds(1);
        fprintf(stderr, "   elapsed: %d us\n", (int)elapse);
    }
}

namespace demo {
static std::atomic<int> compute_counter;

struct JobConfig {
    bool suspend_before;
    bool suspend_after;
    int nth;
    int workload;
};

struct Job {
    // the i-th part
    int ith;
    // number of parallels
    int nth;
    // simulate compute workload
    int workload; // 2^n

    Job(int ith = 0, int nth = 1, int workload = 0)
        : ith(), nth(), workload(){};

    void compute() {
        if (workload > 0) {
            int n = (1 << workload) / nth;

            volatile int64_t x = 0;
            for (int i = 0; i < n; i++) {
                ++x;
            }
            UNUSED(x);
        }
        compute_counter.fetch_add(1, std::memory_order_relaxed);
    }
};

static void compute_job(sched::Scheduler<Job> *scheduler,
                        struct JobConfig cfg) {
    ASSERT(cfg.nth >= 1);

    if (cfg.nth == 1) {
        if (cfg.suspend_before) {
            scheduler->suspend_workers();
        }
        auto job = Job(0, cfg.nth, cfg.workload);
        job.compute();
        return;
    }

    if (scheduler->is_workers_suspending()) {
        scheduler->resume_workers();
    }

    struct Job parts[cfg.nth];
    for (int i = 0; i < cfg.nth; i++) {
        parts[i].ith = i;
        parts[i].nth = cfg.nth;
        parts[i].workload = cfg.workload;
    }

    scheduler->compute(&parts[1], cfg.suspend_after);
    parts[0].compute();
}
} // namespace demo

void test_sched_suspend_resume(int n_workers, int loops) {
    auto m = new sched::Scheduler<demo::Job>(n_workers, false, true);
    m->start_workers();

    m->suspend_workers();
    // m->suspend(); // trigger assert error
    m->resume_workers();
    // m->resume(); // trigger assert error

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loops; i++) {
        m->suspend_workers();
        m->resume_workers();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed = 1.0 * (t1 - t0) / std::chrono::milliseconds(1);
    fprintf(stderr, "%s: %2d workers, loops: %3d, elapsed: %6.3f ms\n",
            __func__, n_workers, loops, elapsed);
    m->stop_workers();
}

static void test_sched(int n_workers, int loops, bool scheduler_suspend,
                       bool worker_suspend, bool worker_compute_suspend,
                       int workload) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    auto scheduler = new sched::Scheduler<demo::Job>(
        n_workers, scheduler_suspend, worker_suspend);
    int n_threads = n_workers + 1;

    scheduler->start_workers();

    atomic_store(&demo::compute_counter, 0);
    int compute_expected = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < loops; i++) {
        bool parallel = (i % 2 == 0);

        bool can_suspend = !parallel && worker_suspend;
        demo::JobConfig cfg = {
            .nth = parallel ? n_threads : 1,
            .suspend_before = can_suspend && !worker_compute_suspend,
            .suspend_after = can_suspend && worker_compute_suspend,
        };
        demo::compute_job(scheduler, cfg);

        compute_expected += cfg.nth;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    auto elapsed = 1.0 * (t1 - t0) / std::chrono::milliseconds(1);
    int actual = atomic_load(&demo::compute_counter);
    ASSERT(actual == compute_expected);

    fprintf(stderr,
            "%s: computed: %4d, elapsed: %8.3f ms. "
            "(NW: %2d, SS: %d, WS: %d, WCS: %d, loops: %3d, workload: 2^%d)\n",
            __func__, actual, elapsed, n_workers, scheduler_suspend,
            worker_suspend, worker_compute_suspend, loops, workload);

    scheduler->stop_workers();
}

// make clean && make
// $ time ./bin/test-sched
int main() {
    if (false) {
        // test_spin_nop_forever();
        // test_yield_forever();
    }

    test_pause();

    if (false) {
        auto m = new sched::Scheduler<demo::Job>(1, false, false);
        m->start_workers();
        m->suspend_workers(); // assert error
        m->resume_workers();  // assert error

        struct demo::Job parts[1];
        m->compute(parts, true); // assert error
        m->stop_workers();
    }

    int loops = 100;

    // 2^n
    int n_workers_arr[] = {1, 2, 4, 8, 16};
    int n_workers_arr_len = sizeof(n_workers_arr) / sizeof(n_workers_arr[0]);

    // 2^n
    int work_load_arr[] = {0, 18, 19, 20};
    int work_load_arr_len = sizeof(work_load_arr) / sizeof(work_load_arr[0]);

    for (int i = 0; i < n_workers_arr_len; i++) {
        int n_workers = n_workers_arr[i];
        fprintf(stderr, "\n");

        for (int j = 0; j < work_load_arr_len; j++) {
            int work_load = work_load_arr[j];
            test_sched_suspend_resume(n_workers, loops);

            test_sched(n_workers, loops, false, false, false, work_load);
            test_sched(n_workers, loops, false, true, true, work_load);
            test_sched(n_workers, loops, false, true, false, work_load);

            test_sched(n_workers, loops, true, false, false, work_load);
            test_sched(n_workers, loops, true, true, true, work_load);
            test_sched(n_workers, loops, true, true, false, work_load);
        }
    }
}
