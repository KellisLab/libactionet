// This is adopted from RcppThread (https://github.com/tnagler/RcppThread)
#pragma once

#include <thread>
#include <future>

//! `mini_thread` functionality
namespace mini_thread {

//! @brief R-friendly version of `std::thread`.
//!
//! Instances of class `Thread` behave just like instances of `std::thread`,
//! see http://en.cppreference.com/w/cpp/thread/thread for methods and examples.
//! There is one difference exception: Whenever other threads are doing some
//! work, the main thread  periodically synchronizes with R. When the user
//! interrupts a threaded computation, any thread will stop as soon as it
//!
class Thread {
public:
    Thread() = default;
    Thread(Thread&) = delete;
    Thread(const Thread&) = delete;
    Thread(Thread&& other)
    {
        swap(other);
    }

    template<class Function, class... Args> explicit
    Thread(Function&& f, Args&&... args)
    {
        auto f0 = [=] () {
            f(args...);
        };
        auto task = std::packaged_task<void()>(f0);
        future_ = task.get_future();
        thread_ = std::thread(std::move(task));
    }

    ~Thread() noexcept
    {
        try {
            if (thread_.joinable())
                thread_.join();
        } catch (...) {}
    }

    Thread& operator=(const Thread&) = delete;
    Thread& operator=(Thread&& other)
    {
        if (thread_.joinable())
            std::terminate();
        swap(other);
        return *this;
    }

    void swap(Thread& other) noexcept
    {
        std::swap(thread_, other.thread_);
        std::swap(future_, other.future_);
    }

    bool joinable() const
    {
        return thread_.joinable();
    }

    //! checks for interruptions and messages every 0.25 seconds and after
    //! computations have finished.
    void join()
    {
        auto timeout = std::chrono::milliseconds(250);
        while (future_.wait_for(timeout) != std::future_status::ready) {
            std::this_thread::yield();
        }
        if (thread_.joinable())
            thread_.join();
    }

    void detach()
    {
        thread_.detach();
    }

    std::thread::id get_id() const
    {
        return thread_.get_id();
    }

    auto native_handle() -> decltype(std::thread().native_handle())
    {
        return thread_.native_handle();
    }

    static unsigned int hardware_concurrency()
    {
        return std::thread::hardware_concurrency();
    }

private:
    std::thread thread_;        //! underlying std::thread.
    std::future<void> future_;  //! future result of task passed to the thread.
};

}

// override std::thread to use mini_thread::Thread instead
#ifndef mini_thread_OVERRIDE_THREAD
    #define mini_thread_OVERRIDE_THREAD 0
#endif

#if mini_thread_OVERRIDE_THREAD
    #define thread mini_threadThread
    namespace std {
        using mini_threadThread = mini_thread::Thread;
    }
#endif
