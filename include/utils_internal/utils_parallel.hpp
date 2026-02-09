#ifndef ACTIONET_UTILS_PARALLEL_HPP
#define ACTIONET_UTILS_PARALLEL_HPP

// #include <omp.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdio>
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

// Thread-safe progress monitor for parallel loops
// Uses standard printf (not R API) for thread safety
class ProgressMonitor {
private:
    std::atomic<int> counter;
    std::atomic<bool> done;
    std::thread monitor_thread;
    int total;
    int update_interval_ms;

    void monitor_loop() {
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(update_interval_ms));
            int completed = counter.load();
            if (completed > 0 && completed < total) {
                printf("\r\tProgress: %d/%d (%.1f%%)  ", completed, total,
                       100.0 * completed / total);
                fflush(stdout);
            }
        }
    }

public:
    ProgressMonitor(int total_items, int update_ms = 500)
        : counter(0), done(false), total(total_items), update_interval_ms(update_ms) {
        monitor_thread = std::thread(&ProgressMonitor::monitor_loop, this);
    }

    ~ProgressMonitor() {
        stop();
    }

    void increment() {
        ++counter;
    }

    void stop() {
        if (!done.load()) {
            done.store(true);
            if (monitor_thread.joinable()) {
                monitor_thread.join();
            }
        }
    }

    int get_count() const {
        return counter.load();
    }
};

#endif //ACTIONET_UTILS_PARALLEL_HPP
