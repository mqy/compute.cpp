#pragma once

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// build options and utilities.
//

// #define WORKER_WAIT
// #define SCHEDULER_WAIT
// #define SPIN_NOP
// #define SPIN_MEM_PAUSE
// #define SPIN_YIELD

#define UNUSED(x) (void)(x)
#define ASSERT(x)                                                              \
    do {                                                                       \
        if (!(x)) {                                                            \
            fprintf(stderr, "ASSERT FAILED: line %d: %s\n", __LINE__, #x);     \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Ref: https://github.com/google/marl/blob/main/src/scheduler.cpp
#ifdef SPIN_NOP
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
#define spin_nop(loops, break_if) {spin_nop_32_x_((loops)); if ((break_if)) break; }
#else // SPIN_NOP
#define spin_nop(loops, break_if)
#endif // SPIN_NOP

#ifdef SPIN_MEM_PAUSE

#if defined(__x86_64__)
#include <emmintrin.h>
#define mem_pause() _mm_pause()
#else // !__x86_64__
#define mem_pause()
#endif // __x86_64__
#define spin_mem_pause(break_if) { mem_pause(); if ((break_if)) break; }
#else // !SPIN_MEM_PAUSE
#define mem_pause()
#define spin_mem_pause(break_if)
#endif // SPIN_MEM_PAUSE

#ifdef SPIN_YIELD
#define spin_yield(break_if) { std::this_thread::yield(); if ((break_if)) break; }
#else
#define spin_yield(break_if)
#endif // SPIN_YIELD

static inline bool SCHEDULER_WAIT_enable() {
#ifdef SCHEDULER_WAIT
    return true;
#else
    return false;
#endif
}

static inline bool worker_wait_enable() {
#ifdef WORKER_WAIT
    return true;
#else
    return false;
#endif
}

static inline bool spin_nop_enable() {
#ifdef SPIN_NOP
    return true;
#else
    return false;
#endif
}

static inline bool spin_mem_pause_enable() {
#ifdef SPIN_MEM_PAUSE
    return true;
#else
    return false;
#endif
}

static inline bool spin_yield_enable() {
#ifdef SPIN_YIELD
    return true;
#else
    return false;
#endif
}

void print_build_options() {
    fprintf(stderr, "build options:\n");
    fprintf(stderr, "    WORKER_WAIT: %d\n",
        worker_wait_enable());
    fprintf(stderr, "    SCHEDULER_WAIT: %d\n",
        SCHEDULER_WAIT_enable());
    fprintf(stderr, "    SPIN_NOP: %d\n",
        spin_nop_enable());
    fprintf(stderr, "    SPIN_MEM_PAUSE: %d\n",
        spin_mem_pause_enable());
    fprintf(stderr, "    SPIN_YIELD: %d\n",
        spin_yield_enable());
}

#ifdef __cplusplus
}
#endif
