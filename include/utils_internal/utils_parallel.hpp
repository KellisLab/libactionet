#ifndef ACTIONET_UTILS_PARALLEL_HPP
#define ACTIONET_UTILS_PARALLEL_HPP

#include "mini_thread/mini_thread.h"
#include <cstddef>

#define SYS_THREADS_DEF (std::thread::hardware_concurrency() - 2)

#endif //ACTIONET_UTILS_PARALLEL_HPP
