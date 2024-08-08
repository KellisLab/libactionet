#include "utils/utils_xmap.hpp"

arma::sp_mat smoothKNN(arma::sp_mat &D, int max_iter, double epsilon, double bandwidth, double local_connectivity,
                       double min_k_dist_scale, double min_sim, int thread_no) {

    int nV = D.n_cols;
    arma::sp_mat G = D;

    //#pragma omp parallel for num_threads(thread_no)
    for (int i = 0; i < nV; i++) {
        //  ParallelFor(0, nV, thread_no, [&](size_t i, size_t threadId) {
        arma::sp_mat v = D.col(i);
        arma::vec vals = nonzeros(v);
        if (vals.n_elem > local_connectivity) {
            double rho = min(vals);
            arma::vec negated_shifted_vals = -(vals - rho);
            arma::uvec deflate = arma::find(vals <= rho);
            negated_shifted_vals(deflate).zeros();

            double target = std::log2(vals.n_elem + 1);

            // Binary search to find optimal sigma
            double sigma = 1.0;
            double lo = 0.0;
            double hi = DBL_MAX;

            int j;
            for (j = 0; j < max_iter; j++) {
                double obj = sum(exp(negated_shifted_vals / sigma));

                if (abs(obj - target) < epsilon) {
                    break;
                }

                if (target < obj) {
                    hi = sigma;
                    sigma = 0.5 * (lo + hi);
                } else {
                    lo = sigma;
                    if (hi == DBL_MAX) {
                        sigma *= 2;
                    } else {
                        sigma = 0.5 * (lo + hi);
                    }
                }
            }

            // double obj = sum(exp(negated_shifted_vals / sigma));
            // TODO: This is obviously a bug. `mean_dist` does not exist.
            double mean_dist = arma::mean(mean_dist);
            sigma = std::max(min_k_dist_scale * mean_dist, sigma);

            for (arma::sp_mat::col_iterator it = G.begin_col(i); it != G.end_col(i); ++it) {
                *it = std::max(min_sim, std::exp(-std::max(0.0, (*it) - rho) / (sigma * bandwidth)));
            }
        } else {
            for (arma::sp_mat::col_iterator it = G.begin_col(i); it != G.end_col(i); ++it) {
                *it = 1.0;
            }
        }
    }

    return (G);
}
