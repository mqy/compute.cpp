#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace ggml {

// compile options:
// #define WORKER_WAIT
// #define PRODUCER_WAIT
#define SPIN_NOP
// #define SPIN_MEM_PAUSE
#define SPIN_YIELD

#define NDEBUG 1

#define RING_MAX_CAP 8
#define PRODUCER_ID 1
// thread_local int32_t thread_local_id;

#define UNUSED(x) (void)(x)
#define ASSERT(x)                                                              \
    do {                                                                       \
        if (!(x)) {                                                            \
            fprintf(stderr, "ASSERT FAILED: line %d: %s\n", __LINE__, #x);     \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

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

#ifdef SPIN_NOP
#define spin_nop(loops, break_if) {spin_nop_32_x_((loops)); if ((break_if)) break; }
#else
#define spin_nop(loops, break_if)
#endif // SPIN_NOP

#ifdef SPIN_MEM_PAUSE

#if defined(__x86_64__)
#include <emmintrin.h>
static inline void mem_pause() { _mm_pause(); }
#else // !__x86_64__
static inline void mem_pause()
#endif // __x86_64__
#define spin_mem_pause(break_if) { mem_pause(); if ((break_if)) break; }
#else // !SPIN_MEM_PAUSE
#define spin_mem_pause(break_if)
#endif // SPIN_MEM_PAUSE

#ifdef SPIN_YIELD
#define spin_yield(break_if) { std::this_thread::yield(); if ((break_if)) break; }
#else
#define spin_yield(break_if)
#endif // SPIN_YIELD

enum worker_cmd {
    worker_cmd_start = 1,
    worker_cmd_stop = 2,
#ifdef WORKER_WAIT
    worker_cmd_suspend = 3,
    worker_cmd_resume = 4,
#endif
    worker_cmd_compute = 5,
};

// The ring is used as tiny task queue: a drop in replacement of std:vector.
template <typename T> class Ring {
  private:
    std::atomic_flag flag;
    std::atomic<int> len;            // number of valid slots
    int cap;                         // capacity
    int head;                        // index to pop from
    int tail;                        // index to push at
    T buf[RING_MAX_CAP]; // ring buffer

    inline void spin_lock( void ) {
        while(flag.test_and_set( std::memory_order_acquire)) {
            spin_nop_32_x_(32);
        }
    }

    inline void spin_unlock( void ){
        flag.clear( std::memory_order_release );
    }

  public:
    Ring<T>(int _cap = RING_MAX_CAP) {
        ASSERT(_cap > 0 && _cap <= RING_MAX_CAP);
        cap = _cap;
    };

    inline bool empty() { return atomic_load(&len) == 0; }
    inline bool full() { return atomic_load(&len) == cap; }

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

template <typename T> struct Task {
    enum worker_cmd cmd;
    T work; // valid only when cmd == worker_cmd_compute;
};

class IProducer {
public:
    virtual void(receive_ack)(enum worker_cmd cmd, int worker_id) = 0;
};

template <typename T>
class Worker {
private:
#ifdef WORKER_WAIT
    std::mutex mutex;
    std::condition_variable cv; // suspend/resume
    std::atomic<bool> suspending;
#endif
    Ring<Task<T>> queue;
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
                spin_nop(256, !w->queue.empty());
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
            spin_nop(256, atomic_load(&n_done) == n_workers);
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
            spin_nop(256, atomic_load(&n_done) == n_workers);
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

} // namespace

// build with bg++ or clang++:
// debug:
//    clang++ -O0 -g -Wall -fsanitize=address --std c++11 ggml-thread.cpp -o ggml-thread
// release:
//    clang++ -O3 -Wall --std c++11 ggml-thread.cpp -o ggml-thread
// options:
//    -DWORKER_WAIT=1
//    -DPRODUCER_WAIT=1
//    -DWORKER_WAIT=1 -DPRODUCER_WAIT=1
//
// $ time ./bin/ggml-thread
static std::atomic<int> compute_counter;
struct DemoWork {
    void compute(){
        compute_counter.fetch_add(1, std::memory_order_relaxed);
        // std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
};

static inline bool producer_wait_enable() {
#ifdef PRODUCER_WAIT
    return true;
#else
    return false;
#endif
}

inline bool worker_wait_enable() {
#ifdef WORKER_WAIT
    return true;
#else
    return false;
#endif
}

// 98% cpu, heavy context switch.
// real	0m4.372s
// user	0m3.051s
// sys	0m1.209s
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
// real	0m5.357s
// user	0m5.280s
// sys	0m0.014s
void test_spin_nop_forever() {
    const int multiplier = 32;
    fprintf(stderr, "%s (multiplier = %d):\n", __func__, multiplier);
    while (true) {
        ggml::spin_nop_32_x_(multiplier);
    }
}

// watch CPU load for a while
static void test_spin_nop() {
    const int multiplier = 32;
    fprintf(stderr, "%s (multiplier = %d):\n", __func__, multiplier);
    for (int i = 0; i < 1000000; i++) {
        ggml::spin_nop_32_x_(multiplier);
    }
}

#ifdef SPIN_MEM_PAUSE
// 100% cpu
// real	0m5.039s
// user	0m4.978s
// sys	0m0.010s
static void test_mem_pause_forever() {
    fprintf(stderr, "%s:\n", __func__);
    while (true) {
        ggml::mem_pause();
    }
}

static void test_mem_pause() {
    fprintf(stderr, "%s:\n", __func__);
    for (int i = 0; i < 1000000; i++) {
        ggml::mem_pause();
    }
}
#endif

static void test_spin_pause() {
    for (int i = 0; i < 3; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        if (i == 0) test_spin_nop();
        else if (i == 1) {
#ifdef SPIN_MEM_PAUSE
            test_mem_pause();
#else
            fprintf(stderr, "test_mem_pause: skip (NOT enabled)\n");
#endif
        } else test_spin_yield();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapse = (end_time - start_time) / std::chrono::milliseconds(1);
        fprintf(stderr, "   elapsed: %d us\n", (int)elapse);
    }
}

void test_threading()
{
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
    for (int i = 0; i < 300; i++) {
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

    fprintf(stderr, "worker wait enable: %d\nproducer wait enable: %d\n",
        worker_wait_enable(), producer_wait_enable());

    fprintf(stderr, "%d calls, elapsed: %.3f ms\n", actual, elapse);

    if (actual != expected) {
        fprintf(stderr, "actual: %d, expected: %d, failed\n", actual, expected);
    } else {
        fprintf(stderr, "pass\n");
    }

    m->stop();
}

int main() {
    // test_spin_nop_forever();
    // test_mem_pause_forever();
    // test_spin_yield_forever();

    test_threading();
    test_spin_pause();
}