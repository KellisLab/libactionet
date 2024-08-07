#ifndef UTILS_PARALLEL_HPP
#define UTILS_PARALLEL_HPP

// #include "RcppPerpendicular.h"
#include "parallel/mini_thread/mini_thread.h"
#include <cstddef>

#define SYS_THREADS_DEF (std::thread::hardware_concurrency() - 2)

#endif
