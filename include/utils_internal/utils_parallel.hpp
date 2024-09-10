#ifndef ACTIONET_UTILS_PARALLEL_HPP
#define ACTIONET_UTILS_PARALLEL_HPP

// #include <omp.h>
#include <thread>
// #include <cstddef>

inline int get_max_threads() {
    int coresDet = std::thread::hardware_concurrency();
    return (coresDet);
}

const int SYS_THREADS_DEF = get_max_threads();

inline int get_num_threads(int max_threads, int thread_no = 0) {
    int threads_use;

    if (thread_no <= 0) {
        threads_use = std::min(SYS_THREADS_DEF, max_threads);
    }
    else if (thread_no > 1) {
        threads_use = std::min(thread_no, SYS_THREADS_DEF);
    }
    else {
        threads_use = 1;
    }

    return (threads_use);
}

#endif //ACTIONET_UTILS_PARALLEL_HPP
