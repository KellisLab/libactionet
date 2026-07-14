#ifndef ACTIONET_UTILS_PARALLEL_HPP
#define ACTIONET_UTILS_PARALLEL_HPP

#include <thread>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#endif

// Returns the number of CPUs available to this process.
// Detection priority:
//   1. OMP_NUM_THREADS env var (explicit user/scheduler intent, capped at hw)
//   2. sched_getaffinity (cgroup cpuset / taskset), only when it actively
//      restricts below hardware count — some schedulers (SGE) leave the
//      affinity mask at 1 core even when more slots are allocated
//   3. std::thread::hardware_concurrency (full node)
//
// The result is cached in a function-local static at first call: env-var and
// affinity are read once per process. This matches the OpenMP runtime's own
// behavior (OMP_NUM_THREADS is read at library init) and removes the syscall
// / getenv from every hot-path get_num_threads() invocation. When GPU work
// lands, this free function will be replaced by an ExecutionPolicy struct.
namespace detail {
inline unsigned int compute_max_threads_() {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;

    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    if (omp_env) {
        int omp_val = std::atoi(omp_env);
        if (omp_val > 0) {
            return std::min(static_cast<unsigned int>(omp_val), hw);
        }
    }

#if defined(__linux__) && !defined(__ANDROID__)
    cpu_set_t cpuset;
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
        unsigned int count = static_cast<unsigned int>(CPU_COUNT(&cpuset));
        if (count > 0 && count < hw) {
            return count;
        }
    }
#endif

    return hw;
}
} // namespace detail

inline unsigned int get_max_threads() {
    static const unsigned int cached = detail::compute_max_threads_();
    return cached;
}

namespace actionet {

namespace detail {
inline unsigned int& outer_parallel_depth_() {
    static thread_local unsigned int depth = 0;
    return depth;
}
} // namespace detail

// Marks work owned by a coarse-grained ACTION parallel loop.  The explicit
// marker is needed because some OpenMP runtimes report a one-thread team as
// inactive, which would otherwise allow a nested helper to start a new team.
class OuterParallelRegionScope {
public:
    OuterParallelRegionScope() { ++detail::outer_parallel_depth_(); }
    ~OuterParallelRegionScope() { --detail::outer_parallel_depth_(); }

    OuterParallelRegionScope(const OuterParallelRegionScope&) = delete;
    OuterParallelRegionScope& operator=(const OuterParallelRegionScope&) = delete;
};

inline unsigned get_num_threads(unsigned int max_threads = 0, const unsigned int thread_no = 0) {
    const unsigned int hw = get_max_threads();
    max_threads = (max_threads > 0) ? std::min(max_threads, hw) : hw;

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

// Nested-OMP guard: returns 1 at every nesting level below an OpenMP region,
// including a serialized num_threads(1) region.  Intel OpenMP may report
// omp_in_parallel() == false for that serialized case even though launching a
// new inner team would violate the outer ACTION thread limit.
inline unsigned get_num_threads_nested_safe(unsigned int max_threads = 0,
                                            const unsigned int thread_no = 0) {
#if defined(_OPENMP)
    if (omp_get_level() > 0 || detail::outer_parallel_depth_() > 0) return 1;
#else
    if (detail::outer_parallel_depth_() > 0) return 1;
#endif
    return get_num_threads(max_threads, thread_no);
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

} // namespace actionet

#endif //ACTIONET_UTILS_PARALLEL_HPP
