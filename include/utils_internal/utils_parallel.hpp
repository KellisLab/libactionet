#ifndef ACTIONET_UTILS_PARALLEL_HPP
#define ACTIONET_UTILS_PARALLEL_HPP

#include "mini_thread/mini_thread.h"
#include <thread>
// #include <cstddef>

// #define SYS_THREADS_DEF (std::thread::hardware_concurrency() - 2)

inline int get_max_threads()
{
    int coresDet = std::thread::hardware_concurrency();
    if (coresDet >= 6)
    {
        coresDet -= 2;
    }
    return (coresDet);
}

const int SYS_THREADS_DEF = get_max_threads();

#endif //ACTIONET_UTILS_PARALLEL_HPP
