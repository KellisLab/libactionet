//  UWOT -- An R package for dimensionality reduction using UMAP
//
//  Copyright (C) 2021 James Melville
//
//  This file is part of UWOT
//
//  UWOT is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  UWOT is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with UWOT.  If not, see <http://www.gnu.org/licenses

#ifndef UWOT_RPARALLEL_H
#define UWOT_RPARALLEL_H

#include <algorithm>
#include <vector>

#include "RcppPerpendicular.h"

#if defined(__linux__) && !defined(__ANDROID__) && defined(_OPENMP)
#include <omp.h>
#define UWOT_RPARALLEL_USE_OPENMP 1
#else
#define UWOT_RPARALLEL_USE_OPENMP 0
#endif

struct RParallel {
    std::size_t n_threads;
    std::size_t grain_size;

    RParallel(std::size_t n_threads, std::size_t grain_size)
        : n_threads(n_threads), grain_size(grain_size) {}

    template <typename Worker> void pfor(std::size_t n_items, Worker &worker) {
        pfor(0, n_items, worker);
    }

    template <typename Worker>
    void pfor(std::size_t begin, std::size_t end, Worker &worker) {
#if UWOT_RPARALLEL_USE_OPENMP
        pfor_openmp(begin, end, worker);
#else
        RcppPerpendicular::pfor(begin, end, worker, n_threads, grain_size);
#endif
    }

private:
#if UWOT_RPARALLEL_USE_OPENMP
    template <typename Worker>
    void pfor_openmp(std::size_t begin, std::size_t end, Worker &worker) const {
        if (end <= begin) {
            return;
        }

        std::size_t requested_threads = n_threads;
        if (requested_threads == 0) {
            requested_threads = RcppPerpendicular::available_concurrency();
        }

        if (requested_threads <= 1) {
            worker(begin, end, 0);
            return;
        }

        const RcppPerpendicular::IndexRange input_range(begin, end);
        std::vector<RcppPerpendicular::IndexRange> ranges =
            RcppPerpendicular::split_input_range(input_range, requested_threads, grain_size);

        if (ranges.empty()) {
            return;
        }

        // Deterministic task IDs and chunk boundaries for worker-local state.
        const std::size_t n_ranges = ranges.size();
        const int         omp_threads = static_cast<int>(std::min(requested_threads, n_ranges));

        #pragma omp parallel for schedule(static) num_threads(omp_threads)
        for (int range_idx = 0; range_idx < static_cast<int>(n_ranges); ++range_idx) {
            const auto &range = ranges[static_cast<std::size_t>(range_idx)];
            worker(range.first, range.second, static_cast<std::size_t>(range_idx));
        }
    }
#endif
};

struct RSerial {
    template <typename Worker> void pfor(std::size_t n_items, Worker &worker) {
        pfor(0, n_items, worker);
    }

    template <typename Worker>
    void pfor(std::size_t begin, std::size_t end, Worker &worker) {
        worker(begin, end, 0);
    }
};

#endif // UWOT_RPARALLEL_H
