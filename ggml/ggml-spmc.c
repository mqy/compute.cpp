#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GGML_ASSERT(x)                                                         \
    do {                                                                       \
        if (!(x)) {                                                            \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__,    \
                    #x);                                                       \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define UNUSED(x) (void)(x)

// see https://github.com/ggerganov/llama.cpp/pull/1314
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#include <emmintrin.h>
static inline void ggml_mem_pause(void) { _mm_pause(); }
#else
static inline void ggml_mem_pause(void) {}
#endif

#if defined(_WIN32)

#include <windows.h>

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef LONG atomic_flag;

typedef CRITICAL_SECTION pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef void pthread_mutexattr_t;
typedef void pthread_condattr_t;

typedef HANDLE pthread_t;
typedef int ggml_thread_ret_t;

static void atomic_store(atomic_int *ptr, LONG val) {
    Intechan_lockedExchange(ptr, val);
}

static LONG atomic_load(atomic_int *ptr) {
    return Intechan_lockedCompareExchange(ptr, 0, 0);
}

static LONG atomic_fetch_add(atomic_int *ptr, LONG inc) {
    return Intechan_lockedExchangeAdd(ptr, inc);
}

static LONG atomic_fetch_sub(atomic_int *ptr, LONG dec) {
    return atomic_fetch_add(ptr, -(dec));
}

static inline LONG atomic_flag_test_and_set(volatile atomic_flag *ptr) {
    return Intechan_lockedCompareExchange(ptr, 1, 0);
}

static inline LONG atomic_flag_clear(volatile atomic_flag *ptr) {
    return Intechan_lockedExchange(ptr, 0);
}

static int pthread_create(pthread_t *out, void *unused,
                          ggml_thread_ret_t (*func)(void *), void *arg) {
    (void)unused;
    HANDLE handle =
        CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, arg, 0, NULL);
    if (handle == NULL) {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static pthread_t pthread_self(void) { return GetCurrentThread(); }

static int pthread_join(pthread_t thread, void *unused) {
    (void)unused;
    return (int)WaitForSingleObject(thread, INFINITE);
}

static int pthread_mutex_init(pthread_mutex_t *mutex,
                              pthread_mutexattr_t *attr) {
    (void)attr;
    InitializeCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_destroy(pthread_mutex_t *mutex) {
    DeleteCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_lock(pthread_mutex_t *mutex) {
    EnterCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_unlock(pthread_mutex_t *mutex) {
    LeaveCriticalSection(mutex);
    return 0;
}

static int pthread_cond_init(pthread_cond_t *cond, pthread_condattr_t *attr) {
    (void)attr;
    InitializeConditionVariable(cond);
    return 0;
}

static int pthread_cond_destroy(pthread_cond_t *cond) {
    (void)cond;
    return 0;
}

static int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
    SleepConditionVariableCS(cond, mutex, INFINITE);
    return 0;
}

static int pthread_cond_signal(pthread_cond_t *cond) {
    WakeConditionVariable(cond);
    return 0;
}

static int pthread_cond_broadcast(pthread_cond_t *cond) {
    WakeAllConditionVariable(cond);
    return 0;
}

static int sched_yield(void) {
    // https://learn.microsoft.com/en-us/windows/win32/api/winnt/nf-winnt-yieldprocessor
    YieldProcessor();
    return 0;
}

#else // ! _WIN32

typedef void *ggml_thread_ret_t;

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>

#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
static int64_t timer_freq, timer_start;
void ggml_time_init(void) {
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if
    // timer_freq and the uptime is high enough. We subtract the program start
    // time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_start = t.QuadPart;
}
int64_t ggml_time_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart - timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart - timer_start) * 1000000) / timer_freq;
}
#else
void ggml_time_init(void) {}
int64_t ggml_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

int64_t ggml_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}
#endif

//-----------------------------------------------------------------------------
/// Most of the above codes are taken from
/// https://github.com/ggerganov/llama.cpp/tree/master/ggml.c
/// Copyright original authors.
//-----------------------------------------------------------------------------

#define GGML_SPMC_PRODUCER_ID 1
#define MAX_SPIN_COUNT 1000000
#define MAX_NOTIFY_COUNT 10
#define MIN_NOTIFY_INTERVAL_US 100
#define MAX_RING_CAP 128

#define DEBUG 1

#ifdef DEBUG
#define PRINT_DEBUG(...)                                                       \
    fprintf(stdout, "[self: %d] %s\n", ggml_thread_local_id, __func__);        \
    fprintf(stdout, __VA_ARGS__)
#else
#define PRINT_DEBUG(...)
#endif

struct ggml_ring {
    atomic_flag spin;
    atomic_int len;                   // number of valid slots
    int cap;                          // capacity
    int head;                         // index to pop from.
    int tail;                         // index to push at.
    void *buf[GGML_SPMC_PRODUCER_ID]; // ring buffer.
};

// recursive spin lock.
typedef struct ggml_spin {
    volatile atomic_flag flag;
    atomic_int owner_id; // `ggml_thread_local_id`
} ggml_spin_t;

typedef int(ggml_spmc_handler)(void *);

struct ggml_spmc_consumer {
    int id;
    pthread_t thread_id;
    ggml_spmc_handler *handler; // task handler.
    struct ggml_spmc *chan;
    atomic_bool waiting; // updated by consumers.
};

// SPMC: a tailored Single Producer Multi-Consumers channel.
typedef struct ggml_spmc {
    ggml_spin_t spin;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    atomic_int n_wait; // updated by consumers.

    struct ggml_ring ring;

    atomic_int thread_counter; // increase/decrease by thread runners
    atomic_int task_counter;   // done counter

    atomic_bool closing; // updated by producer.

    // blocking read, volatile, can be updated on the fly.
    atomic_bool blocking; // updated by producer.

    int n_consumers;
    struct ggml_spmc_consumer *consumers;
} ggml_spmc_t;

enum ggml_spmc_wait_for_type {
    GGML_WAIT_FOR_RUNNING_THREADS = 1,
    GGML_WAIT_FOR_INACTIVE_CONSUMERS = 2,
    GGML_WAIT_FOR_FINISHED_TASKS = 3,
};

_Thread_local int32_t ggml_thread_local_id;
_Thread_local int32_t ggml_thread_local_spin_counter;
_Thread_local int32_t ggml_thread_local_pause_counter;

static void ggml_spin_init(ggml_spin_t *l) {
    memset(l, 0, sizeof(ggml_spin_t));
    atomic_flag_clear(&l->flag);
}

static inline void ggml_spin_lock(atomic_flag *flag) {
    while (atomic_flag_test_and_set(flag)) {
        if (ggml_thread_local_spin_counter % 2 == 0) {
            ggml_mem_pause();
        } else {
            sched_yield();
        }
    }
}

static inline void ggml_spin_unlock(atomic_flag *flag) {
    atomic_flag_clear(flag);
}

static inline void ggml_spin_checked_lock(ggml_spin_t *l) {
    if (ggml_thread_local_id == atomic_load(&l->owner_id)) {
        fprintf(stderr, "[self: %d] %s:already spin locked, deadlock!\n",
                ggml_thread_local_id, __func__);
        abort();
        return;
    }

    while (atomic_flag_test_and_set(&l->flag)) {
        if (ggml_thread_local_spin_counter % 2 == 0) {
            ggml_mem_pause();
        } else {
            sched_yield();
        }
        if (++ggml_thread_local_spin_counter == MAX_SPIN_COUNT) {
            fprintf(stderr,
                    "[self: %d] %s: spinning lock for %d times, deadlock!\n",
                    ggml_thread_local_id, __func__, MAX_SPIN_COUNT);
            abort();
        }
    }
    atomic_store(&l->owner_id, ggml_thread_local_id);
}

static inline void ggml_spin_checked_unlock(ggml_spin_t *l) {
    GGML_ASSERT(ggml_thread_local_id == atomic_load(&l->owner_id));
    atomic_store(&l->owner_id, 0);
    ggml_thread_local_spin_counter = 0;
    atomic_flag_clear(&l->flag);
}

static inline bool ggml_ring_empty(struct ggml_ring *r) { return r->len == 0; }

static inline bool ggml_ring_full(struct ggml_ring *r) {
    return atomic_load(&r->len) == r->cap;
}

static inline void ggml_ring_push(struct ggml_ring *r, void *v) {
    ggml_spin_lock(&r->spin);
    GGML_ASSERT(r->len < r->cap);
    GGML_ASSERT(v);
    r->buf[r->tail] = v;
    ++r->len;
    ++r->tail;
    if (r->tail == r->cap) {
        r->tail = 0;
    }
    atomic_thread_fence(memory_order_release);
    ggml_spin_unlock(&r->spin);
}

static inline void ggml_ring_reset(struct ggml_ring *r) {
    ggml_spin_lock(&r->spin);
    r->len = 0;
    r->head = 0;
    r->tail = 0;
    ggml_spin_unlock(&r->spin);
}

static inline void *ggml_ring_pop(struct ggml_ring *r) {
    ggml_spin_lock(&r->spin);
    if (r->len == 0) {
        ggml_spin_unlock(&r->spin);
        return NULL;
    }
    void *v = r->buf[r->head];
    GGML_ASSERT(v);
    ++r->head;
    --r->len;
    if (r->head == r->cap) {
        r->head = 0;
    }
    ggml_spin_unlock(&r->spin);
    return v;
}

static inline void ggml_spmc_pause(void) {
    if (ggml_thread_local_pause_counter % 2 == 0) {
        sched_yield();
    } else {
        ggml_mem_pause();
    }
    if (++ggml_thread_local_pause_counter == INT32_MAX) {
        ggml_thread_local_pause_counter = 0;
    }
}

// If `chan->blocking` is true, try reading until `chan` was closed or
// `blocking` was updated as false. Return NULL if: `chan` was closed or `ch` is
// empty, or switching from blocking to non-blocking. Producer always async
// read.
static void *ggml_spmc_read(ggml_spmc_t *chan) {
    GGML_ASSERT(chan);
    if (atomic_load(&chan->closing)) {
        return NULL;
    }

    if (!atomic_load(&chan->blocking) ||
        ggml_thread_local_id == GGML_SPMC_PRODUCER_ID) {
        if (atomic_load(&chan->closing)) {
            return NULL;
        }

        void *value = ggml_ring_pop(&chan->ring);
        return value;
    }

    // blocking read. NOTE: ch->blocking can be updated by producer on the fly.
    while (atomic_load(&chan->blocking) && !atomic_load(&chan->closing)) {
        void *value = ggml_ring_pop(&chan->ring);
        if (value != NULL) {
            return value;
        }

        atomic_fetch_add(&chan->n_wait, 1);

        pthread_mutex_lock(&chan->mutex);
        if (ggml_ring_empty(&chan->ring) && atomic_load(&chan->blocking) &&
            !atomic_load(&chan->closing)) {
            pthread_cond_wait(&chan->cond, &chan->mutex);
        }
        pthread_mutex_unlock(&chan->mutex);
        atomic_fetch_sub(&chan->n_wait, 1);
    }

    return NULL;
}

// Append `value` to `ch`. Must only be called by chan creator.
// Producer never block on pushing chan.
static void ggml_spmc_push(ggml_spmc_t *chan, void *value) {
    GGML_ASSERT(chan);
    GGML_ASSERT(!atomic_load(&chan->closing));
    GGML_ASSERT(value);
    GGML_ASSERT(ggml_thread_local_id == GGML_SPMC_PRODUCER_ID);
    GGML_ASSERT(chan->n_consumers > 0);

    while (ggml_ring_full(&chan->ring)) {
        ggml_spmc_pause();
        if (!chan->blocking) {
            continue;
        }
        if (atomic_load(&chan->n_wait) > 0) {
            // pthread_mutex_lock(&chan->mutex);
            pthread_cond_signal(&chan->cond);
            // pthread_mutex_unlock(&chan->mutex);
        }
    }

    ggml_ring_push(&chan->ring, value);

    if (atomic_load(&chan->n_wait) > 0) {
        // pthread_mutex_lock(&chan->mutex);
        pthread_cond_signal(&chan->cond);
        // pthread_mutex_unlock(&chan->mutex);
    }
}

static void ggml_spmc_wait_for(ggml_spmc_t *chan,
                               enum ggml_spmc_wait_for_type type,
                               int n_expected) {
    GGML_ASSERT(chan->n_consumers > 0);
    GGML_ASSERT(ggml_thread_local_id == GGML_SPMC_PRODUCER_ID);

    int notify_counter = 0;
    int64_t last_notify_time_us = 0;

    bool notify = false;
    int actual = 0;
    while (true) {
        if (type == GGML_WAIT_FOR_RUNNING_THREADS) {
            actual = atomic_load(&chan->thread_counter);
            notify = actual != n_expected;
        } else if (type == GGML_WAIT_FOR_INACTIVE_CONSUMERS) {
            actual = atomic_load(&chan->n_wait);
            notify = actual != n_expected;
        } else if (type == GGML_WAIT_FOR_FINISHED_TASKS) {
            actual = atomic_load(&chan->task_counter);
            notify =
                (actual != n_expected) && (atomic_load(&chan->n_wait) > 0 ||
                                           !ggml_ring_empty(&chan->ring));
        } else {
            GGML_ASSERT(false);
        }

        if (!notify) {
            break;
        }

        ggml_spmc_pause();

        // FIXME: don't notify frequently.
        if (atomic_load(&chan->n_wait) > 0) {
            int64_t t1 = ggml_time_us();
            if (last_notify_time_us == 0 ||
                (t1 - last_notify_time_us > MIN_NOTIFY_INTERVAL_US)) {
                // pthread_mutex_lock(&chan->mutex);
                pthread_cond_broadcast(&chan->cond);
                // pthread_mutex_unlock(&chan->mutex);

                last_notify_time_us = t1;
                ++notify_counter;

                if (notify_counter >= MAX_NOTIFY_COUNT) {
                    fprintf(stderr,
                            "%s: notified for %d times, possible deadlock!\n",
                            __func__, notify_counter);
                    abort();
                }

                PRINT_DEBUG(
                    "==== notified, blocking: %d, type: %d , actual: %d, "
                    "expected: %d ...\n",
                    chan->blocking, type, actual, n_expected);
            }
        }
    }
}

void ggml_spmc_close(ggml_spmc_t *chan) {
    GGML_ASSERT(chan);
    GGML_ASSERT(!chan->closing);
    GGML_ASSERT(ggml_thread_local_id == GGML_SPMC_PRODUCER_ID);

    chan->closing = true;
    ggml_ring_reset(&chan->ring);

    ggml_spmc_wait_for(chan, GGML_WAIT_FOR_RUNNING_THREADS, 0);
    if (chan->consumers) {
        free(chan->consumers);
    }
}

static void ggml_spmc_set_blocking(ggml_spmc_t *chan, bool blocking) {
    GGML_ASSERT(ggml_thread_local_id == GGML_SPMC_PRODUCER_ID);
    if (chan->n_consumers == 0 || chan->blocking == blocking) {
        return;
    }

    bool old_blocking = chan->blocking;
    int expected = old_blocking ? chan->n_consumers : 0;

    chan->blocking = blocking;
    ggml_spmc_wait_for(chan, GGML_WAIT_FOR_INACTIVE_CONSUMERS, expected);
}

ggml_thread_ret_t ggml_spmc_consumer_thread(void *arg) {
    struct ggml_spmc_consumer *consumer = (struct ggml_spmc_consumer *)arg;
    GGML_ASSERT(consumer);
    ggml_spmc_t *chan = consumer->chan;
    GGML_ASSERT(chan);
    GGML_ASSERT(consumer->handler);
    GGML_ASSERT(consumer->id > GGML_SPMC_PRODUCER_ID);

    ggml_thread_local_id = consumer->id;
    atomic_fetch_add(&chan->thread_counter, 1);

    while (!atomic_load(&chan->closing)) {
        void *v = ggml_spmc_read(chan);
        if (v == NULL) {
            if (atomic_load(&chan->closing)) {
                break;
            }
            ggml_spmc_pause();
            continue;
        }

        consumer->handler(v);
        atomic_fetch_add(&chan->task_counter, 1);
    }

    atomic_fetch_sub(&chan->thread_counter, 1);
    return 0;
}

ggml_spmc_t *ggml_spmc_new(int cap, bool blocking, int n_consumers,
                           ggml_spmc_handler chan_handler) {
    GGML_ASSERT(cap > 0 && cap <= GGML_SPMC_PRODUCER_ID);
    GGML_ASSERT(n_consumers > 0);
    GGML_ASSERT(chan_handler);

    ggml_spmc_t *chan = NULL;
    {
        size_t sz = sizeof(ggml_spmc_t);
        chan = malloc(sz);
        GGML_ASSERT(chan);
        memset(chan, 0, sz);
    }

    chan->closing = false;
    chan->blocking = blocking;

    chan->ring.cap = cap;
    atomic_flag_clear(&chan->ring.spin);

    ggml_spin_init(&chan->spin);
    GGML_ASSERT(pthread_mutex_init(&chan->mutex, NULL) == 0);
    GGML_ASSERT(pthread_cond_init(&chan->cond, NULL) == 0);

    ggml_thread_local_id = GGML_SPMC_PRODUCER_ID;

    chan->consumers = NULL;

    if (n_consumers > 0) {
        size_t sz = n_consumers * sizeof(struct ggml_spmc_consumer);
        struct ggml_spmc_consumer *consumers = malloc(sz);
        GGML_ASSERT(consumers);
        memset(consumers, 0, sz);

        for (int i = 0; i < n_consumers; i++) {
            consumers[i].id = GGML_SPMC_PRODUCER_ID + i + 1;
            consumers[i].handler = chan_handler;
            consumers[i].chan = chan;
            GGML_ASSERT(pthread_create(&consumers[i].thread_id, NULL,
                                       ggml_spmc_consumer_thread,
                                       &consumers[i]) == 0);
        }
        chan->n_consumers = n_consumers;
        chan->consumers = consumers;

        ggml_spmc_wait_for(chan, GGML_WAIT_FOR_RUNNING_THREADS, n_consumers);
    }

    return chan;
}

// implements `ggml_spmc_handler`.
static int demo_chan_task_handler(void *arg) {
    UNUSED(arg);

    int rnd = rand() % 25;
    int loops = 10000 * (100 + rnd); // 0-25%
    volatile int counter = 0;

    for (int i = 0; i < loops; i++) {
        ++counter;
    }

    UNUSED(counter);
    return 0;
}

static void test_chan(void) {
    printf("%s: enter\n", __func__);
    srand((unsigned)time(NULL));

    const int n_consumers = 6;

    // const int chan_cap = n_consumers + 1;
    const int chan_cap = GGML_SPMC_PRODUCER_ID;
    // const int n_tasks = chan_cap;
    const int n_tasks = 1024;

    bool blocking = false;

    ggml_spmc_t *chan =
        ggml_spmc_new(chan_cap, blocking, n_consumers, demo_chan_task_handler);

    printf("created channel, cap: %d, n_consumers: %d\n", chan_cap,
           n_consumers);

    if (false) { // non_blocking
        printf("ensure %d consumers waiting\n", n_consumers);
        int expected_inactive = blocking ? n_consumers : 0;
        ggml_spmc_wait_for(chan, GGML_WAIT_FOR_INACTIVE_CONSUMERS,
                           expected_inactive);

        int *tasks = alloca(sizeof(int) * n_tasks);

        for (int i = 0; i < n_tasks; i++) {
            tasks[i] = i + 1;
            ggml_spmc_push(chan, &tasks[i]);
        }
        printf("pushed %d tasks\n", n_tasks);

        // producer involves in computing.
        int n_producer_computed = 0;
        while (true) {
            void *value = ggml_spmc_read(chan);
            if (value == NULL) {
                break;
            }
            demo_chan_task_handler(value);
            ++n_producer_computed;
        }
        printf("producer computed %d of %d\n", n_producer_computed, n_tasks);
        ggml_spmc_wait_for(chan, GGML_WAIT_FOR_FINISHED_TASKS,
                           n_tasks - n_producer_computed);
        printf("all tasks done\n");
    }

    if (true) { // blocking
        ggml_spmc_set_blocking(chan, true);

        printf("updated blocking read mode to %d\n", chan->blocking);

        printf("pushing %d tasks ...\n", n_tasks);
        int *tasks = alloca(sizeof(int) * n_tasks);
        for (int i = 0; i < n_tasks; i++) {
            tasks[i] = i + 1;
            ggml_spmc_push(chan, &tasks[i]);
        }

        printf("pushed %d tasks\n", n_tasks);

        // producer involves in computing.
        int n_producer_computed = 0;
        while (true) {
            void *value = ggml_spmc_read(chan);
            if (value == NULL) {
                break;
            }
            demo_chan_task_handler(value);
            ++n_producer_computed;
        }
        printf("producer computed %d of %d\n", n_producer_computed, n_tasks);

        ggml_spmc_wait_for(chan, GGML_WAIT_FOR_FINISHED_TASKS,
                           n_tasks - n_producer_computed);
        printf("all tasks done\n");
    }

    { // close chan
        printf("closing task channel ...\n");
        ggml_spmc_close(chan);
        printf("all consumers exited\n");

        free(chan);
    }

    printf("%s: done\n", __func__);
}

// threading demo --------------------------------------------------------------

struct ggml_threading_context {
    ggml_spmc_t *chan;
};

static struct ggml_threading_context *
ggml_threading_start(int chan_cap, int n_consumers, bool blocking,
                     ggml_spmc_handler *task_handler) {
    size_t ctx_sz = sizeof(struct ggml_threading_context);
    struct ggml_threading_context *ctx = malloc(ctx_sz);
    memset(ctx, 0, ctx_sz);

    ctx->chan = ggml_spmc_new(chan_cap, blocking, n_consumers, task_handler);
    return ctx;
}

static void ggml_threading_suspend(struct ggml_threading_context *ctx) {
    ggml_spmc_set_blocking(ctx->chan, true);
}

static void ggml_threading_resume(struct ggml_threading_context *ctx) {
    ggml_spmc_set_blocking(ctx->chan, false);
    atomic_store(&ctx->chan->task_counter, 0);
}

static void ggml_threading_stop(struct ggml_threading_context *ctx) {
    ggml_spmc_close(ctx->chan);
    free(ctx->chan);
    free(ctx);
}

struct demo_params {
    int ith;
    int nth;
};

struct demo_task_stage {
    bool valid;
    bool parallel;
    bool wait; // when !parallel
};

struct demo_tensor {
    // src0
    // src1
    struct demo_task_stage task_stages[3];
    // ...
};

struct demo_compute_data {
    struct demo_params params;
    struct demo_tensor *tensor;
};

static int demo_threading_task_handler(void *arg) {
    GGML_ASSERT(arg);
    struct demo_compute_data *e = (struct demo_compute_data *)arg;
    GGML_ASSERT(e);

    int rnd = rand() % 25;
    int loops = 1000 * (100 + rnd); // 0-25%
    volatile int counter = 0;

    for (int i = 0; i < loops; i++) {
        ++counter;
    }

    UNUSED(counter);
    return 0;
}

static void ggml_threading_assign_task(struct ggml_threading_context *ctx,
                                       struct demo_compute_data *data) {
    ggml_spmc_push(ctx->chan, data);
}

static void ggml_threading_wait_for_tasks(struct ggml_threading_context *ctx,
                                          int n_tasks) {
    ggml_spmc_wait_for(ctx->chan, GGML_WAIT_FOR_FINISHED_TASKS, n_tasks);
}

static void test_demo_threading(void) {
    printf("%s: enter\n", __func__);
    srand((unsigned)time(NULL));

    int n_threads = 2;
    int chan_cap = 16;
    int n_consumers = n_threads - 1;
    bool blocking = false;

    struct demo_tensor *tensors = NULL;
    const int multiplier = 2;
    const int n_tensors = 3 * multiplier;

    size_t sz = sizeof(struct demo_tensor) * n_tensors;
    tensors = malloc(sz);
    GGML_ASSERT(tensors);
    memset(tensors, 0, sz);

    {
        int i = 0;
        for (int j = 0; j < multiplier; j++) {
            tensors[i].task_stages[1].valid = true;
            tensors[i].task_stages[1].parallel = true;
            ++i;

            tensors[i].task_stages[0].valid = true;
            tensors[i].task_stages[1].valid = true;
            tensors[i].task_stages[1].parallel = true;
            ++i;

            tensors[i].task_stages[1].valid = true;
            tensors[i].task_stages[1].wait = true;
            ++i;
        }
    }
    ggml_spmc_handler *task_runner = demo_threading_task_handler;

    struct demo_compute_data *data =
        malloc(n_threads * sizeof(struct demo_compute_data));
    GGML_ASSERT(data);

    struct ggml_threading_context *ctx =
        ggml_threading_start(chan_cap, n_consumers, blocking, task_runner);

    GGML_ASSERT(n_consumers + 1 == n_threads);

    for (int i = 0; i < n_tensors; i++) {
        struct demo_tensor *tensor = &tensors[i];

        for (int j = 0; j < 3; j++) {
            struct demo_task_stage *stage = &tensor->task_stages[j];
            if (!stage->valid) {
                continue;
            }

            if (stage->parallel) {
                for (int k = 0; k < n_consumers; k++) {
                    struct demo_compute_data *e = &data[k + 1];
                    e->tensor = tensor;
                    e->params.ith = k + 1;
                    e->params.nth = n_threads;
                    PRINT_DEBUG("main assign a task\n");
                    ggml_threading_assign_task(ctx, e);
                }
                ggml_threading_resume(ctx);
            } else {
                if (stage->wait) {
                    ggml_threading_suspend(ctx);
                }
            }

            data[0].tensor = tensor;
            data[0].params.ith = 0;
            data[0].params.nth = n_threads;
            task_runner(&data[0]);

            if (stage->parallel) {
                ggml_threading_wait_for_tasks(ctx, n_consumers);
            }
        }
    }

    PRINT_DEBUG("stopping threads ...\n");
    ggml_threading_stop(ctx);

    free(data);
    free(tensors);

    printf("%s: done.\n", __func__);
}

int main(void) {
    ggml_time_init();

    test_chan();

    printf("\n");

    test_demo_threading();
}
