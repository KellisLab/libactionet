// SPDX-License-Identifier: MIT
//
// Copyright (c) 2026 MIT CompBio Group
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// -----------------------------------------------------------------------------
// Clean-room MIT reimplementation of the parallel-for surface required by
// uwot's optimize_layout() driver (see _UmapFactory.hpp, uwot/epoch.h,
// uwot/update.h).
//
// Replaces uwot/rparallel.h (GPLv3-or-later) and its RcppPerpendicular.h
// (GPLv2-or-later) dependency. libactionet already hard-requires OpenMP
// (context/DECISIONS.md), so the implementation is a direct OpenMP fan-out.
//
// Public types (matches the historical rparallel.h contract):
//   - RParallel(n_threads, grain_size)   .pfor(n_items,     worker)
//                                        .pfor(begin, end,  worker)
//   - RSerial                            .pfor(...) invokes worker(begin,end,0)
//
// worker must be a callable with signature
//     void worker(std::size_t begin, std::size_t end, std::size_t thread_id)
// This matches uwot::EdgeWorker::operator() and uwot::NodeWorker::operator().

#ifndef ACTIONET_UWOT_PARALLEL_HPP
#define ACTIONET_UWOT_PARALLEL_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct RParallel {
    std::size_t n_threads;
    std::size_t grain_size;

    RParallel(std::size_t n_threads_, std::size_t grain_size_)
        : n_threads(n_threads_), grain_size(grain_size_) {}

    template <typename Worker>
    void pfor(std::size_t n_items, Worker& worker) {
        pfor(0, n_items, worker);
    }

    template <typename Worker>
    void pfor(std::size_t begin, std::size_t end, Worker& worker) {
        if (end <= begin) {
            return;
        }

        const std::size_t length = end - begin;

        // Serial fast-path.
        std::size_t threads = n_threads == 0 ? 1 : n_threads;
        if (threads <= 1) {
            worker(begin, end, 0);
            return;
        }

        // Choose per-thread chunk size: at least grain_size, and never so small
        // that we spawn more chunks than there are items.
        std::size_t chunk = grain_size == 0 ? 1 : grain_size;
        const std::size_t max_chunks_by_length = length; // chunk >= 1
        const std::size_t desired_chunks = std::min(threads, max_chunks_by_length);
        std::size_t chunk_by_threads = (length + desired_chunks - 1) / desired_chunks;
        chunk = std::max(chunk, chunk_by_threads);
        if (chunk == 0) {
            chunk = 1;
        }

        // Materialize the (begin,end) partition ahead of the parallel region so
        // that thread_id maps deterministically to a specific chunk (matching
        // the historical semantics that per-worker RNG state indexes by
        // thread_id / chunk index).
        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        ranges.reserve(desired_chunks);
        for (std::size_t s = begin; s < end; s += chunk) {
            const std::size_t e = std::min(s + chunk, end);
            ranges.emplace_back(s, e);
        }

        const int n_ranges  = static_cast<int>(ranges.size());
        const int omp_threads = static_cast<int>(
            std::min<std::size_t>(threads, ranges.size()));

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(omp_threads)
        for (int i = 0; i < n_ranges; ++i) {
            const auto& r = ranges[static_cast<std::size_t>(i)];
            worker(r.first, r.second, static_cast<std::size_t>(i));
        }
#else
        for (int i = 0; i < n_ranges; ++i) {
            const auto& r = ranges[static_cast<std::size_t>(i)];
            worker(r.first, r.second, static_cast<std::size_t>(i));
        }
#endif
    }
};

struct RSerial {
    template <typename Worker>
    void pfor(std::size_t n_items, Worker& worker) {
        pfor(0, n_items, worker);
    }

    template <typename Worker>
    void pfor(std::size_t begin, std::size_t end, Worker& worker) {
        if (end <= begin) {
            return;
        }
        worker(begin, end, 0);
    }
};

#endif // ACTIONET_UWOT_PARALLEL_HPP
