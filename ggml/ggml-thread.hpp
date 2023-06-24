#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "ggml-thread.h"

namespace ggml {

#define PRODUCER_ID 1
// thread_local int32_t thread_local_id;

enum worker_cmd {
    worker_cmd_start = 1,
    worker_cmd_stop = 2,
#ifdef WORKER_WAIT
    worker_cmd_suspend = 3,
    worker_cmd_resume = 4,
#endif
    worker_cmd_compute = 5,
};

// Worker holds this interface.
class IProducer {
public:
    virtual void(receive_ack)(enum worker_cmd cmd, int worker_id) = 0;
};

template <typename T> struct Task {
    enum worker_cmd cmd;
    T work; // valid only when cmd == worker_cmd_compute;
};

// The ring is used as tiny task queue: a drop in replacement of std:vector.
// TODO: try lock-free implementation.
template <typename T, size_t cap> class Ring {
  static_assert(std::is_trivial<T>::value, "type T must be trivial");
  static_assert(cap > 1, "The cap must be bigger than 1");

  private:
    std::atomic_flag flag;
    std::atomic<int> len;            // number of valid slots
    int head;                        // index to pop from
    int tail;                        // index to push at
    T buf[cap]; // buffer

    inline void spin_lock() {
        while(flag.test_and_set(std::memory_order_acquire)) {
#ifdef SPIN_NOP
            spin_nop_32_x_(32);
#endif
        }
    }

    inline void spin_unlock(){
        flag.clear(std::memory_order_release);
    }

  public:
    Ring() : len(0), head(0), tail(0) {};
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

template <typename T>
class Worker {
private:
#ifdef WORKER_WAIT
    std::mutex mutex;
    std::condition_variable cv; // suspend/resume
    std::atomic<bool> suspending;
#endif
    // a SPSC task queue shared with producer, it's enough to set capacity as 2.
    Ring<Task<T>, 2> queue;

    int worker_id;
    std::thread thread;
    IProducer *producer;

public:
    Worker(IProducer *task_producer, int id) {
        producer = task_producer;
        worker_id = id;
    }

    void attach_thread() {
        thread = std::thread(thread_runner, this);
    }

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
        if (e.cmd != suspending) {
            std::unique_lock<std::mutex> lk(mutex);
            if (atomic_load(&suspending)) {
                cv.notify_one();
            }
            lk.unlock();
        }
#endif
    }

    inline void ack(enum worker_cmd cmd) {
        producer->receive_ack(cmd, worker_id);
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
        ASSERT(w->producer);
        // thread_local_id = w->worker_id;

        w->ack(worker_cmd_start);

        while (true) {
            while (w->queue.empty()) {
                spin_nop(32, !w->queue.empty());
                spin_mem_pause(!w->queue.empty());
                spin_yield(!w->queue.empty());
            }

            struct Task<T> e = w->queue.pop_front();

            switch (e.cmd) {
                case worker_cmd_compute: {
                    e.work.compute();
                    w->ack(e.cmd); // computed.
                } break;
                case worker_cmd_stop: {
                    w->ack(e.cmd); // exiting
                } break;
#ifdef WORKER_WAIT
                case worker_cmd_suspend: {
                    w->ack(e.cmd); // to be suspended
                    w->wait();
                } break;
                case worker_cmd_resume: {
                    w->ack(e.cmd); // resumed
                } break;
#endif
                default: {
                    ASSERT(false);
                } break;
            }
        }
    }
};

template <class T>
class Producer: IProducer {
private:
    std::mutex mutex;
    std::condition_variable cv; // cmd ack
    std::atomic<int> n_done;

    std::vector<Worker<T> *> workers;
    int n_workers;

    inline void send_cmd(enum worker_cmd cmd, T works[] = nullptr, int n_tasks = 0) {
#ifdef PRODUCER_WAIT
        std::unique_lock<std::mutex> lk(mutex);
#endif
        atomic_store(&n_done, 0);
        for (int i = 0; i < n_workers; i++) {
            Worker<T> *w = workers[i];
            Task<T> e = {.cmd = cmd};
            if (n_tasks > 0 && works != nullptr) {
                e.work = works[i];
            }
            w->enqueue(e);
        }
#ifdef PRODUCER_WAIT
        cv.wait(lk, [this]{return atomic_load(&n_done) == n_workers;});
        lk.unlock();
#else
        int pause_counter;
        UNUSED(pause_counter);
        while (atomic_load(&n_done) != n_workers) {
            spin_nop(32, atomic_load(&n_done) == n_workers);
            spin_mem_pause(atomic_load(&n_done) == n_workers);
            spin_yield(atomic_load(&n_done) == n_workers);
        }
#endif
    }

public:
    Producer(int n_workers) {
        n_done = 0;
        this->n_workers = n_workers;
        // thread_local_id = PRODUCER_ID;
        for (int i = 0; i < n_workers; i++) {
            Worker<T> *w = new Worker<T>(this, PRODUCER_ID + i + 1);
            workers.push_back(w);
        }
    }

    ~Producer() {
        for (int i = 0; i < workers.size(); i++) {
            delete workers[i];
        }
        workers.clear();
    }

    void receive_ack(enum worker_cmd cmd, int worker_id) {
        UNUSED(cmd);
        UNUSED(worker_id);
#ifdef PRODUCER_WAIT
        std::unique_lock<std::mutex> lk(mutex);
        n_done.fetch_add(1, std::memory_order_relaxed);
        cv.notify_one();
        lk.unlock();
#else
        n_done.fetch_add(1, std::memory_order_relaxed);
#endif
    }

    void assign(T works[], int n_tasks) {
        send_cmd(worker_cmd_compute, works, n_tasks);
    }

    void start() {
#ifdef PRODUCER_WAIT
        std::unique_lock<std::mutex> lk(mutex);
        for (int i = 0; i < n_workers; i++) {
            workers[i]->attach_thread();
        }
        cv.wait(lk, [this]{return atomic_load(&n_done) == n_workers;});
        lk.unlock();
#else
        for (int i = 0; i < n_workers; i++) {
            workers[i]->attach_thread();
        }

        int pause_counter;
        UNUSED(pause_counter);
        while (atomic_load(&n_done) != n_workers) {
            spin_nop(32, atomic_load(&n_done) == n_workers);
            spin_mem_pause(atomic_load(&n_done) == n_workers);
            spin_yield(atomic_load(&n_done) == n_workers);
        }
#endif
    }

#ifdef WORKER_WAIT
    void suspend() { send_cmd(worker_cmd_suspend); }
    void resume() { send_cmd(worker_cmd_resume); }
#else
    void suspend() {}
    void resume() {}
#endif

    void stop() { send_cmd(worker_cmd_stop); }
};

} // namespace: ggml
