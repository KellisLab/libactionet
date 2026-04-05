#include "action/action_decomp.hpp"
#include "action/spa.hpp"
#include "action/aa.hpp"
#include "utils_internal/utils_parallel.hpp"

namespace actionet {
    ResACTION
        decompACTION(const arma::mat& S_r, int k_min, int k_max, int max_it, double tol, int thread_no) {
        if (k_max == -1)
            k_max = (int)S_r.n_cols;

        k_min = std::max(k_min, 2);
        k_max = std::min(k_max, (int)S_r.n_cols);

        ResACTION trace;
        trace.selected_cols = arma::field<arma::uvec>(k_max + 1);

        int k_tot = k_max - k_min + 1;
        int threads_use = get_num_threads(k_tot, thread_no);

        // T = sum(k_min..k_max) — known analytically; allocate stacked buffers once.
        size_t T = (size_t)(k_tot) * (k_min + k_max) / 2;
        trace.C_stacked = arma::mat(S_r.n_cols, T);
        trace.H_stacked = arma::mat(T, S_r.n_cols);

        // Pre-compute per-k column offsets so each OMP thread can write its slice
        // without coordination. col_offset[k] is the first column index in the
        // stacked buffers for level k.
        std::vector<size_t> col_offset(k_max + 1, 0);
        for (int k = k_min; k <= k_max; k++)
            col_offset[k] = (k == k_min) ? 0 : col_offset[k - 1] + (size_t)(k - 1);

        stdout_printf("Running ACTION (%d threads):\n", threads_use);
        stdout_printf("\tIterating from k = %d ... %d\n", k_min, k_max);
        FLUSH;

        ProgressMonitor progress(k_tot);

        #pragma omp parallel for num_threads(threads_use)
        for (int k = k_min; k <= k_max; k++) {
            ResSPA SPA_res = runSPA(S_r, k);
            trace.selected_cols[k] = std::move(SPA_res.selected_cols);

            arma::mat W = S_r.cols(trace.selected_cols[k]);

            arma::field<arma::mat> AA_res = runAA(S_r, W, max_it, tol);

            size_t off = col_offset[k];
            trace.C_stacked.cols(off, off + k - 1) = AA_res(0);
            trace.H_stacked.rows(off, off + k - 1) = AA_res(1);

            progress.increment();
        }

        progress.stop();
        stdout_printf("\r\tCompleted: %d/%d (100.0%%)  \n", k_tot, k_tot);
        FLUSH;

        return trace;
    }
} // namespace actionet
