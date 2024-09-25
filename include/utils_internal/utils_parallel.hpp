#ifndef ACTIONET_UTILS_PARALLEL_HPP
#define ACTIONET_UTILS_PARALLEL_HPP

// #include <omp.h>
#include <thread>
// #include <cstddef>

inline unsigned int get_max_threads() {
    const unsigned int coresDet = std::thread::hardware_concurrency();
    return (coresDet);
}

const unsigned int SYS_THREADS_DEF = get_max_threads();

inline unsigned get_num_threads(unsigned int max_threads = 0, const unsigned int thread_no = 0) {

    max_threads = (max_threads > 0) ? std::min(max_threads, SYS_THREADS_DEF) : SYS_THREADS_DEF;

    unsigned int threads_use;

    if (thread_no <= 0) {
        threads_use = max_threads;
    }
    else if (thread_no > 1) {
        threads_use = std::min(thread_no, max_threads);
    }
    else {
        threads_use = 1;
    }

    return (threads_use);
}

#endif //ACTIONET_UTILS_PARALLEL_HPP
