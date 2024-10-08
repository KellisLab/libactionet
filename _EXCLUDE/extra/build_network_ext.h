#ifndef ACTIONET_BUILD_NETWORK_EXT_H
#define ACTIONET_BUILD_NETWORK_EXT_H

#include "network.hpp"
#include "RcppPerpendicular.h"

// Functions (inline)
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads,
                        Function fn) {
    if (numThreads <= 0) {
        numThreads = SYS_THREADS_DEF;
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if ((id >= end)) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    }
                    catch (...) {
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
                }
            }));
        }
        for (auto &thread: threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

// Functions (source)

double computeJSD(const double *pVect1, const double *pVect2, const double *log_vec,
                  int N);

arma::mat computeFullSim(arma::mat &H, int thread_no);

arma::sp_mat buildNetwork_KstarNN_v2(arma::mat H_stacked, double density, int thread_no, double M,
                                     double ef_construction, double ef, bool mutual_edges_only,
                                     std::string distance_metric);

arma::sp_mat buildNetwork_bipartite(arma::mat H1, arma::mat H2, double density = 1.0, int thread_no = 0, double M = 16,
                                    double ef_construction = 200, double ef = 200, std::string distance_metric = "jsd");

#endif //ACTIONET_BUILD_NETWORK_EXT_H
