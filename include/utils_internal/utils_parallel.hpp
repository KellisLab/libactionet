#ifndef ACTIONET_UTILS_PARALLEL_HPP
#define ACTIONET_UTILS_PARALLEL_HPP

#include "mini_thread/mini_thread.h"
#include <omp.h>
#include <thread>
// #include <cstddef>

// #define SYS_THREADS_DEF (std::thread::hardware_concurrency() - 2)

inline int get_max_threads() {
    int coresDet = std::thread::hardware_concurrency();
    if (coresDet >= 6) {
        coresDet -= 2;
    }
    return (coresDet);
}

const int SYS_THREADS_DEF = get_max_threads();

inline int get_num_threads(int max_threads, int thread_no = 0) {
    int threads_use;
    if (thread_no == 0) {
        threads_use = std::min(SYS_THREADS_DEF, max_threads);
    }
    else {
        threads_use = std::min(thread_no, SYS_THREADS_DEF + 2);
    }

    return (threads_use);
}

#endif //ACTIONET_UTILS_PARALLEL_HPP
