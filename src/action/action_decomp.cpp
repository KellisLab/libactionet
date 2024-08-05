#include "action/action_decomp.hpp"

// sp_mat &as_arma_sparse(cholmod_sparse *chol_A, sp_mat &A,
//                        cholmod_common *chol_c);

// void dsdmult(char transpose, int m, int n, const void *a, const double *b, double *c,
//              cholmod_common *chol_cp);

// cholmod_sparse *as_cholmod_sparse(const sp_mat &A, cholmod_sparse *chol_A,
//                                   cholmod_common *chol_c);

namespace ACTIONet
{

  ACTION_results run_ACTION(arma::mat &S_r, int k_min, int k_max, int thread_no,
                            int max_it, double min_delta,
                            int normalization)
  {
    if (thread_no <= 0)
    {
      thread_no = SYS_THREADS_DEF;
    }

    int feature_no = S_r.n_rows;

    stdout_printf("Running ACTION (%d threads):", thread_no);
    FLUSH;

    if (k_max == -1)
      k_max = (int)S_r.n_cols;

    k_min = std::max(k_min, 2);
    k_max = std::min(k_max, (int)S_r.n_cols);

    ACTION_results trace;

    trace.H = arma::field<arma::mat>(k_max + 1);
    trace.C = arma::field<arma::mat>(k_max + 1);
    trace.selected_cols = arma::field<arma::uvec>(k_max + 1);

    // ATTENTION!
    arma::mat X_r = normalize_mat(S_r, normalization, 0);

    int current_k = 0;
    char status_msg[50];

    sprintf(status_msg, "Iterating from k = %d ... %d:", k_min, k_max);
    stderr_printf("\n\t%s %d/%d finished", status_msg, current_k,
                  (k_max - k_min + 1));
    FLUSH;

    mini_thread::parallelFor(
        k_min, k_max + 1,
        [&](size_t kk)
        {
          SPA_results SPA_res = run_SPA(X_r, kk);
          trace.selected_cols[kk] = SPA_res.selected_columns;

          arma::mat W = X_r.cols(trace.selected_cols[kk]);

          arma::field<arma::mat> AA_res;

          AA_res = run_AA(X_r, W, max_it, min_delta);
          trace.C[kk] = AA_res(0);
          trace.H[kk] = AA_res(1);
          current_k++;

          stderr_printf("\r\t%s %d/%d finished", status_msg, current_k,
                        (k_max - k_min + 1));
          FLUSH;
        },
        thread_no);

    stdout_printf("\r\t%s %d/%d finished\n", status_msg, current_k,
                  (k_max - k_min + 1));
    FLUSH;

    return trace;
  }

} // namespace ACTIONet
