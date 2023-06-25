#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "pause.h"

namespace sched {

#define CALLER_THREAD_ID 1
thread_local int32_t thread_local_id;

enum worker_cmd {
    worker_cmd_start = 1 << 0,
    worker_cmd_stop = 1 << 1,
    worker_cmd_compute = 1 << 2,
#ifdef WORKER_WAIT
    worker_cmd_suspend = 1 << 3,
    worker_cmd_resume = 1 << 4,
#ifdef WORKER_COMPUTE_SUSPEND
    worker_cmd_compute_suspend = worker_cmd_compute | worker_cmd_suspend,
#endif
#endif
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

// a tiny task queue: drop in replacement of std:vector.
// TODO: try lock-free implementation.
template <typename T, size_t cap> class Ring {
    static_assert(std::is_trivial<T>::value, "type T must be trivial");
    static_assert(cap > 1, "The cap must be bigger than 1");

  private:
    std::atomic_flag flag;
    std::atomic<int> len; // number of valid slots
    int head;             // index to pop from
    int tail;             // index to push at
    T buf[cap];           // buffer

    inline void spin_lock() {
        while (flag.test_and_set(std::memory_order_acquire)) {
#ifdef SPIN_NOP
            spin_nop_32_x_(32);
#endif
        }
    }

    inline void spin_unlock() { flag.clear(std::memory_order_release); }

  public:
    Ring() : len(0), head(0), tail(0){};
    inline bool empty() { return len.load(std::memory_order_relaxed) == 0; }
    inline bool full() { return len.load(std::memory_order_relaxed) == cap; }

    void push_back(T e) {
        spin_lock();
        ASSERT(!full());
        buf[tail] = e;
        ++tail;
        if (tail == cap) {
            tail = 0;
        }
        len.fetch_add(1, std::memory_order_relaxed);
        spin_unlock();
    }

    T pop_front() {
        spin_lock();
        ASSERT(!empty());
        T e = buf[head];
        ++head;
        if (head == cap) {
            head = 0;
        }
        len.fetch_sub(1, std::memory_order_relaxed);
        spin_unlock();
        return e;
    }
};

template <typename T> class Worker {
  private:
#ifdef WORKER_WAIT
    std::mutex mutex;
    std::condition_variable cv; // suspend/resume
    std::atomic<bool> suspending;
#endif
    // a SPSC task queue shared with caller, it's enough to set capacity as 2.
    Ring<Task<T>, 2> queue;

    int worker_id;
    std::thread thread;
    ICaller *caller;

    inline Task<T> spinning_deque() {
        while (queue.empty()) {
            spin_nop(32, !queue.empty());
            spin_mem_pause(!queue.empty());
            spin_yield(!queue.empty());
        }
        return queue.pop_front();
    }

  public:
    Worker(ICaller *caller, int worker_id) {
        this->caller = caller;
        this->worker_id = worker_id;
    }

    void attach_thread() { thread = std::thread(thread_runner, this); }

    ~Worker() {
#ifdef WORKER_WAIT
        std::unique_lock<std::mutex> lk(mutex);
        cv.notify_one();
        lk.unlock();
#endif
        thread.join();
    }

    void enqueue(struct Task<T> e) {
        queue.push_back(e);
#ifdef WORKER_WAIT
        std::unique_lock<std::mutex> lk(mutex);
        if (atomic_load(&suspending)) {
            // ASSERT((e.cmd & worker_cmd_suspend) == 0); // may fail
            cv.notify_one();
        } else {
            // ASSERT(e.cmd != worker_cmd_resume); // may fail
        }
        lk.unlock();
#endif
    }

    inline void ack(enum worker_cmd cmd) {
        caller->receive_ack(cmd, worker_id);
    }

#ifdef WORKER_WAIT
    inline void wait() {
        std::unique_lock<std::mutex> lk(mutex);
        if (queue.empty()) {
            atomic_store(&suspending, true);
            cv.wait(lk);
            atomic_store(&suspending, false);
        }
        lk.unlock();
    }
#endif

    static void thread_runner(Worker *w) {
        ASSERT(w->caller);
        thread_local_id = w->worker_id;
        w->ack(worker_cmd_start);

        while (true) {
            struct Task<T> e = w->spinning_deque();
            if (e.cmd == worker_cmd_compute) {
                e.work.compute();
                w->ack(e.cmd); // computed.
            } else if (e.cmd == worker_cmd_stop) {
                break;
#ifdef WORKER_WAIT
            } else if (e.cmd == worker_cmd_suspend) {
                w->ack(e.cmd); // to be suspended
                w->wait();
            } else if (e.cmd == worker_cmd_resume) {
                w->ack(e.cmd); // resumed
#ifdef WORKER_COMPUTE_SUSPEND
            } else if (e.cmd == worker_cmd_compute_suspend) {
                e.work.compute();
                w->ack(e.cmd); // computed, to be suspended
                w->wait();
#endif
#endif
            } else {
                ASSERT(false);
            }
        }

        w->ack(worker_cmd_stop); // exiting
    }
};

template <class T> class Scheduler : ICaller {
  private:
    std::vector<Worker<T> *> workers;
    int n_workers;
    std::atomic<int> n_acks;

#ifdef SCHEDULER_WAIT
    std::mutex mutex;
    std::condition_variable cv; // cmd ack
#else
    inline void spin_wait_acks() {
        while (atomic_load(&n_acks) != n_workers) {
            spin_nop(32, atomic_load(&n_acks) == n_workers);
            spin_mem_pause(atomic_load(&n_acks) == n_workers);
            spin_yield(atomic_load(&n_acks) == n_workers);
        }
    }
#endif

    // dispatch to workers
    inline void dispatch(enum worker_cmd cmd, T works[] = nullptr) {
        ASSERT(n_workers > 0);
        atomic_store(&n_acks, 0);

        for (int i = 0; i < n_workers; i++) {
            Worker<T> *w = workers[i];
            Task<T> e = {.cmd = cmd};
            if (works != nullptr) {
                e.work = works[i];
            }
            w->enqueue(e);
        }
    }

  public:
    Scheduler(int n_workers) {
        n_acks = 0;
        this->n_workers = n_workers;
        thread_local_id = CALLER_THREAD_ID;
        for (int i = 0; i < n_workers; i++) {
            Worker<T> *w = new Worker<T>(this, CALLER_THREAD_ID + i + 1);
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
#ifdef SCHEDULER_WAIT
        std::unique_lock<std::mutex> lk(mutex);
#endif
        n_acks.fetch_add(1, std::memory_order_relaxed);
#ifdef SCHEDULER_WAIT
        cv.notify_one();
        lk.unlock();
#endif
    }

    void compute(T works[], bool suspend_after_compute = false) {
        if (n_workers == 0) {
            return;
        }

        enum worker_cmd cmd = worker_cmd_compute;
#ifdef WORKER_WAIT
#ifdef WORKER_COMPUTE_SUSPEND
        if (suspend_after_compute) {
            cmd = worker_cmd_compute_suspend;
        }
#endif
#endif

#ifdef SCHEDULER_WAIT
        std::unique_lock<std::mutex> lk(mutex);
#endif
        dispatch(cmd, works);
#ifdef SCHEDULER_WAIT
        cv.wait(lk, [this] { return atomic_load(&n_acks) == n_workers; });
        lk.unlock();
#else
        spin_wait_acks();
#endif
    }

    // suspend, resume, stop
    void command(enum worker_cmd cmd) {
        if (n_workers == 0) {
            return;
        }

#ifdef SCHEDULER_WAIT
        std::unique_lock<std::mutex> lk(mutex);
#endif
        dispatch(cmd, nullptr);
#ifdef SCHEDULER_WAIT
        cv.wait(lk, [this] { return atomic_load(&n_acks) == n_workers; });
        lk.unlock();
#else
        spin_wait_acks();
#endif
    }

    void start() {
#ifdef SCHEDULER_WAIT
        std::unique_lock<std::mutex> lk(mutex);
#endif
        for (int i = 0; i < n_workers; i++) {
            workers[i]->attach_thread();
        }
#ifdef SCHEDULER_WAIT
        cv.wait(lk, [this] { return atomic_load(&n_acks) == n_workers; });
        lk.unlock();
#else
        spin_wait_acks();
#endif
    }

#ifdef WORKER_WAIT
    void suspend() { command(worker_cmd_suspend); }
    void resume() { command(worker_cmd_resume); }
#else
    void suspend() {}
    void resume() {}
#endif

    void stop() { command(worker_cmd_stop); }
};

} // namespace sched
