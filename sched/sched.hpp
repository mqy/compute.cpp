#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace sched {

#define UNUSED(x) (void)(x)
#define ASSERT(x)                                                              \
    do {                                                                       \
        if (!(x)) {                                                            \
            fprintf(stderr, "ASSERT FAILED: line %d: %s\n", __LINE__, #x);     \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#ifndef NDEBUG
#define DEBUG(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG(...)
#endif

// Ref: https://github.com/google/marl/blob/main/src/scheduler.cpp
inline void nop() {
#if defined(_WIN32)
    __nop();
#else
    __asm__ __volatile__("nop");
#endif
}

static inline void spin_nop_32_x_(int n) {
    for (int i = 0; i < n; i++) {
        // clang-format off
        nop(); nop(); nop(); nop(); nop(); nop(); nop(); nop();
        nop(); nop(); nop(); nop(); nop(); nop(); nop(); nop();
        nop(); nop(); nop(); nop(); nop(); nop(); nop(); nop();
        nop(); nop(); nop(); nop(); nop(); nop(); nop(); nop();
    }
}

#define spin_nop(loops, break_if)  spin_nop_32_x_((loops)); if ((break_if)) { break; }

#define CALLER_THREAD_ID 1
thread_local int32_t thread_local_id;

enum worker_cmd {
    worker_cmd_start   = 1 << 0,
    worker_cmd_stop    = 1 << 1,
    worker_cmd_compute = 1 << 2,
    worker_cmd_suspend = 1 << 3,
    worker_cmd_resume  = 1 << 4,
    worker_cmd_compute_suspend = worker_cmd_compute | worker_cmd_suspend,
};

// Worker holds this interface.
class ICaller {
  public:
    virtual void(receive_ack)(enum worker_cmd cmd, int worker_id) = 0;
};

template <typename T> struct Task {
    enum worker_cmd cmd;
    T work; // valid only when cmd == worker_cmd_compute;
};

// https://en.cppreference.com/w/cpp/atomic/memory_order
template <typename T> class Worker {
  private:
    bool enable_suspend;
    std::mutex mutex;
    std::condition_variable cv; // suspend/resume
    std::atomic<bool> suspending;;

    Task<T> task;
    // false: reader is expecting for task, true: task set.
    // reader:
    // - wait for `task_ready` becomes true
    // - copy the `task` away
    // - set `task_ready` as false
    // writer:
    // - wait for `task_ready` becomes false
    // - set `task`
    // - set `task_ready` as true
    std::atomic<bool> task_ready;

    int worker_id;
    std::thread thread;
    ICaller *caller;

  public:

    Worker(ICaller *caller, int worker_id, bool enable_suspend) {
        this->caller = caller;
        this->worker_id = worker_id;
        this->enable_suspend = enable_suspend;
    }

    void attach_thread() { thread = std::thread(thread_runner, this); }

    ~Worker() {
        if (enable_suspend) {
            std::lock_guard<std::mutex> lk(mutex);
            cv.notify_one();
        }
        thread.join();
    }

    // spin blocking read.
    struct Task<T> read_task() {
        DEBUG("[%d] %s(): enter\n", thread_local_id, __func__);
        while(true) {
            spin_nop_32_x_(256);
            if (task_ready.load(std::memory_order_acquire)) {
                 break;
            }
            // this yield is critical to performance.
            std::this_thread::yield();
            if (task_ready.load(std::memory_order_acquire)) {
                 break;
            }
#ifdef ENABLE_SPIN_PAUSE
#endif
        }
        Task<T> task_copy = task;
        task_ready.store(false, std::memory_order_release);
        DEBUG("[%d] %s(): exit\n", thread_local_id, __func__);
        return task_copy;
    }

    // spin blocking write.
    void write_task(struct Task<T> task) {
        DEBUG("[%d] %s(): enter\n", thread_local_id, __func__);
        while(task_ready.load(std::memory_order_acquire)) {
            spin_nop_32_x_(256);
            if (!task_ready.load(std::memory_order_acquire)) {
                 break;
            }
#ifdef ENABLE_SPIN_PAUSE
            // this yield is critical to performance.
            std::this_thread::yield();
#endif
        }
        this->task = task;
        task_ready.store(true, std::memory_order_release);

        if (enable_suspend) {
            std::lock_guard<std::mutex> lk(mutex);
            if (suspending.load(std::memory_order_relaxed)) {
                cv.notify_one();
            }
        }
        DEBUG("[%d] %s(): exit\n", thread_local_id, __func__);
    }

    inline void ack(enum worker_cmd cmd) {
        caller->receive_ack(cmd, worker_id);
    }

    inline void wait() {
        DEBUG("[%d] %s(): enter\n", thread_local_id, __func__);
        ASSERT(enable_suspend);
        constexpr auto timeout = std::chrono::milliseconds(1);

        std::unique_lock<std::mutex> lk(mutex);
        suspending.store(true, std::memory_order_relaxed);
        while (!task_ready.load(std::memory_order_relaxed)) {
            cv.wait_for(lk, timeout);
        }
        suspending.store(false, std::memory_order_relaxed);
        lk.unlock();
        DEBUG("[%d] %s(): exit\n", thread_local_id, __func__);
    }

    static void thread_runner(Worker *w) {
        ASSERT(w->caller);
        thread_local_id = w->worker_id;
        w->ack(worker_cmd_start);

        while (true) {
            struct Task<T> e = w->read_task();

            if (e.cmd == worker_cmd_compute) {
                e.work.compute();
                w->ack(e.cmd); // computed.
            } else if (e.cmd == worker_cmd_stop) {
                break;
            } else if (e.cmd == worker_cmd_suspend) {
                w->ack(e.cmd); // to be suspended
                w->wait();
            } else if (e.cmd == worker_cmd_resume) {
                w->ack(e.cmd); // resumed
            } else if (e.cmd == worker_cmd_compute_suspend) {
                e.work.compute();
                w->ack(e.cmd); // computed, to be suspended
                w->wait();
            } else {
                ASSERT(false);
            }
        }

        w->ack(worker_cmd_stop); // exiting
    }
};

template <class T> class Scheduler : ICaller {
  private:
    // enable only if scheduler does not involve in computing.
    bool enable_scheduler_suspend;

    // enable only if estimated per-worker compute time is at least 5x of the
    // overhead of wait-notify (e.g., 10 us).
    bool enable_worker_suspend;

    bool workers_suspending;

    int n_workers;
    std::vector<Worker<T> *> workers;
    std::atomic<int> n_acks;

    std::mutex mutex;
    std::condition_variable cv; // cmd ack

    inline void wait_for_acks() {
        constexpr auto timeout = std::chrono::milliseconds(1);
        while(n_acks.load(std::memory_order_relaxed) != n_workers) {
            spin_nop_32_x_(256);
            if (n_acks.load(std::memory_order_relaxed) == n_workers) {
                 break;
            }
            
            std::this_thread::yield();
            if (n_acks.load(std::memory_order_relaxed) == n_workers) {
                 break;
            }
#ifdef ENABLE_SPIN_PAUSE
#endif

            if (enable_scheduler_suspend) {
                // quite slow when n_workers > n_physical cores.
                std::unique_lock<std::mutex> lk(mutex);
                cv.wait_for(lk, timeout, [this] {
                    return n_acks.load(std::memory_order_relaxed) == n_workers;
                });
                lk.unlock();
            }
        }
    }

    // dispatch without passive suspending.
    inline void dispatch(enum worker_cmd cmd, T works[] = nullptr) {
        ASSERT(n_workers > 0);
        atomic_store(&n_acks, 0);

        for (int i = 0; i < n_workers; i++) {
            Worker<T> *w = workers[i];
            Task<T> e = {.cmd = cmd};
            if (works != nullptr) {
                e.work = works[i];
            }
            // blocking write, this should be ok because by design the commands
            // are processed one by one.
            w->write_task(e);
        }
    }

    inline void dispatch_wait(enum worker_cmd cmd, T works[] = nullptr) {
        if (n_workers == 0) {
            return;
        }
        dispatch(cmd, works);
        wait_for_acks();
    }

  public:

    Scheduler(int n_workers = 4,
              bool enable_scheduler_suspend = false,
              bool enable_worker_suspend = false){
        this->n_workers = n_workers;
        this->enable_scheduler_suspend = enable_scheduler_suspend;
        this->enable_worker_suspend = enable_worker_suspend;
        this->n_acks = 0;
        this->workers_suspending = false;
        thread_local_id = CALLER_THREAD_ID;

        for (int i = 0; i < n_workers; i++) {
            Worker<T> *w = new Worker<T>(this, CALLER_THREAD_ID + i + 1,
                                         enable_worker_suspend);
            workers.push_back(w);
        }
    }

    ~Scheduler() {
        for (int i = 0; i < workers.size(); i++) {
            delete workers[i];
        }
        workers.clear();
    }

    void receive_ack(enum worker_cmd cmd, int worker_id) {
        UNUSED(cmd);
        UNUSED(worker_id);
        n_acks.fetch_add(1, std::memory_order_relaxed);
        if (enable_scheduler_suspend) {
            std::lock_guard<std::mutex> lk(mutex);
            cv.notify_one();
        }
    }

    void compute(T works[], bool suspend_after_compute = false) {
        if (n_workers == 0) {
            return;
        }

        ASSERT(works);
        for (int i = 0; i < n_workers; i++) {
            ASSERT(&works[i]);
        }

        enum worker_cmd cmd = worker_cmd_compute;
        if (suspend_after_compute) {
            ASSERT(enable_worker_suspend);
            cmd = worker_cmd_compute_suspend;
        }

        dispatch_wait(cmd, works);
        if (cmd == worker_cmd_compute_suspend) {
            workers_suspending = true;
        }
    }

    void start_workers() {
        for (int i = 0; i < n_workers; i++) {
            workers[i]->attach_thread();
        }
        wait_for_acks();
    }

    void suspend_workers() {
        ASSERT(enable_worker_suspend);
        dispatch_wait(worker_cmd_suspend);
        workers_suspending = true;
    }

    void resume_workers() {
        ASSERT(enable_worker_suspend);
        dispatch_wait(worker_cmd_resume);
        workers_suspending = false;
    }

    bool is_workers_suspending() {
        return workers_suspending;
    }

    void stop_workers() { dispatch_wait(worker_cmd_stop); }
};

} // namespace sched
