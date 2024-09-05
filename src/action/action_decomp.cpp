#include "action/action_decomp.hpp"
#include "action/spa.hpp"
#include "action/aa.hpp"
#include "tools/normalization.hpp"
#include "utils_internal/utils_parallel.hpp"

namespace actionet {

    ResACTION
    run_ACTION(arma::mat &S_r, int k_min, int k_max, int normalization, int max_it, double tol, int thread_no) {
        // if (thread_no <= 0) {
        //     thread_no = SYS_THREADS_DEF;
        // }

        if (k_max == -1)
            k_max = (int) S_r.n_cols;

        k_min = std::max(k_min, 2);
        k_max = std::min(k_max, (int) S_r.n_cols);

        ResACTION trace;

        trace.H = arma::field<arma::mat>(k_max + 1);
        trace.C = arma::field<arma::mat>(k_max + 1);
        trace.selected_cols = arma::field<arma::uvec>(k_max + 1);

        // TODO: ???
        arma::mat X_r = normalize_matrix(S_r, normalization, 0);

        thread_no = (thread_no == 0) ? std::min(SYS_THREADS_DEF, k_max-k_min+1) : std::min(thread_no, SYS_THREADS_DEF + 2);
        stdout_printf("Running ACTION (%d threads):", thread_no);
        FLUSH;

        int current_k = 0;
        char status_msg[50];

        snprintf(status_msg, 50, "Iterating from k = %d ... %d:", k_min, k_max);
        stderr_printf("\n\t%s %d/%d finished", status_msg, current_k, (k_max - k_min + 1));
        FLUSH;

        #pragma omp parallel for num_threads(thread_no)
        for (int k = k_min; k <= k_max; k++) {
            ResSPA SPA_res = run_SPA(X_r, k);
            trace.selected_cols[k] = SPA_res.selected_cols;

            arma::mat W = X_r.cols(trace.selected_cols[k]);

            arma::field<arma::mat> AA_res = run_AA(X_r, W, max_it, tol);
            trace.C[k] = AA_res(0);
            trace.H[k] = AA_res(1);
            current_k++;

            stderr_printf("\r\t%s %d/%d finished", status_msg, current_k, (k_max - k_min + 1));
            FLUSH;
        }

        // mini_thread::parallelFor(
        //         k_min, k_max + 1,
        //         [&](size_t kk) {
        //             ResSPA SPA_res = run_SPA(X_r, kk);
        //             trace.selected_cols[kk] = SPA_res.selected_cols;
        //
        //             arma::mat W = X_r.cols(trace.selected_cols[kk]);
        //
        //             arma::field<arma::mat> AA_res = run_AA(X_r, W, max_it, tol);
        //             trace.C[kk] = AA_res(0);
        //             trace.H[kk] = AA_res(1);
        //             current_k++;
        //
        //             stderr_printf("\r\t%s %d/%d finished", status_msg, current_k, (k_max - k_min + 1));
        //             FLUSH;
        //         },
        //         thread_no);

        stdout_printf("\r\t%s %d/%d finished\n", status_msg, current_k, (k_max - k_min + 1));
        FLUSH;

        return trace;
    }

} // namespace actionet
