#ifndef UTILS_PARALLEL_HPP
#define UTILS_PARALLEL_HPP

#include "parallel/RcppPerpendicular.h"
#include "parallel/mini_thread/mini_thread.h"
#include <cstddef>

#define SYS_THREADS_DEF (std::thread::hardware_concurrency() - 2)

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads,
                        Function fn)
{
  if (numThreads <= 0)
  {
    numThreads = SYS_THREADS_DEF;
  }

  if (numThreads == 1)
  {
    for (size_t id = start; id < end; id++)
    {
      fn(id, 0);
    }
  }
  else
  {
    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);

    // keep track of exceptions in threads
    // https://stackoverflow.com/a/32428427/1713196
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < numThreads; ++threadId)
    {
      threads.push_back(std::thread([&, threadId]
                                    {
                                      while (true)
                                      {
                                        size_t id = current.fetch_add(1);

                                        if ((id >= end))
                                        {
                                          break;
                                        }

                                        try
                                        {
                                          fn(id, threadId);
                                        }
                                        catch (...)
                                        {
                                          std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                                          lastException = std::current_exception();
                                          /*
             * This will work even when current is the largest value that
             * size_t can fit, because fetch_add returns the previous value
             * before the increment (what will result in overflow
             * and produce 0 instead of current + 1).
             */
                                          current = end;
                                          break;
                                        }
                                      } }));
    }
    for (auto &thread : threads)
    {
      thread.join();
    }
    if (lastException)
    {
      std::rethrow_exception(lastException);
    }
  }
}

#endif
