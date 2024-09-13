#include "action/action_decomp.hpp"
#include "action/spa.hpp"
#include "action/aa.hpp"
#include "tools/normalization.hpp"
#include "utils_internal/utils_parallel.hpp"

namespace actionet {
    ResACTION
        runACTION(arma::mat& S_r, int k_min, int k_max, int normalization, int max_it, double tol, int thread_no) {
        if (k_max == -1)
            k_max = (int)S_r.n_cols;

        k_min = std::max(k_min, 2);
        k_max = std::min(k_max, (int)S_r.n_cols);

        ResACTION trace;

        trace.H = arma::field<arma::mat>(k_max + 1);
        trace.C = arma::field<arma::mat>(k_max + 1);
        trace.selected_cols = arma::field<arma::uvec>(k_max + 1);

        // TODO: ???
        arma::mat X_r = normalizeMatrix(S_r, normalization, 0);

        int k_tot = k_max - k_min + 1;
        int threads_use = get_num_threads(k_tot, thread_no);

        stdout_printf("Running ACTION (%d threads):\n", threads_use);
        FLUSH;

        int k_curr = 0;
        char status_msg[50];

        snprintf(status_msg, sizeof(status_msg), "Iterating from k = %d ... %d:", k_min, k_max);
        stderr_printf("\t%s %d/%d finished", status_msg, k_curr, k_tot);
        FLUSH;

        #pragma omp parallel for num_threads(threads_use)
        for (int k = k_min; k <= k_max; k++) {
            ResSPA SPA_res = runSPA(X_r, k);
            trace.selected_cols[k] = SPA_res.selected_cols;

            arma::mat W = X_r.cols(trace.selected_cols[k]);

            arma::field<arma::mat> AA_res = runAA(X_r, W, max_it, tol);
            trace.C[k] = AA_res(0);
            trace.H[k] = AA_res(1);
            k_curr++;

            stderr_printf("\r\t%s %d/%d finished", status_msg, k_curr, k_tot);
            FLUSH;
        }

        stdout_printf("\r\t%s %d/%d finished\n", status_msg, k_curr, k_tot);
        FLUSH;

        return trace;
    }
} // namespace actionet
