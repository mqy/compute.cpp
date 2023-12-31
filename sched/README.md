# Graph Computing Scheduler

Successive studies after [#1632](https://github.com/ggerganov/llama.cpp/pull/1632)
which shows working codes, but is hard to merge, lacking of feedbacks, having the following drawbacks:
- to support tuning, introduced too many updates.
- the threading implementation is ugly and full of tricks, not well-tested.
- hard to test for Windows and CL/CUDA due to limited personal devices.
- controversial design of task profiles: intrusive.
- hard to maintain and tends to become trouble maker.
- in favor of [ggml : get rid of BLAS and all it's variants](https://github.com/ggerganov/ggml/issues/293)

Technical reasons for the new design:
- Atomicity and threads are hard to make cross-platform correct and performant,
  especially with C APIs.
- C++11 has builtin atomic and threads, not perfect but better than never.
- Plenty of open-sourced general purpose modern solutions to evaluate.

## Design Goal

The scheduler must:

- be reliable (deadlock-free), performant, configurable, scalable.
- be well designed on top of interfaces, the reference implementation must be able to be easily replaced.

## References

- [C++ conditional variable](https://en.cppreference.com/w/cpp/thread/condition_variable)
- [spinlock mutexes are extremely dubious in practice](https://www.realworldtech.com/forum/?threadid=189711&curpostid=189723)
- [C++ marl](https://github.com/google/marl/)

## Current Status

Studying, prototyping, draft.
