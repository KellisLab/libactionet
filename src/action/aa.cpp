// Solves the standard Archetypal Analysis (AA) problem
#include "action/aa.hpp"
#include "action/simplex_regression.hpp"
#include "utils_internal/utils_small_dense.hpp"

namespace actionet {

    namespace small_dense = utils_internal::small_dense;

    // Squared-norm threshold below which the current archetype h-column is
    // considered singular and re-seeded from the residual argmax.
    // Note: the literal `10e-8` reads as `1e-7`, not `1e-8`; naming makes intent explicit.
    static constexpr double AA_SINGULAR_THRESHOLD = 1e-7;

    arma::field<arma::mat> runAA(const arma::mat &A, const arma::mat &W0, int max_it, double tol) {
        int sample_no = A.n_cols;
        int k = W0.n_cols; // AA components

        arma::mat C = arma::zeros(sample_no, k);
        arma::mat H = arma::zeros(k, sample_no);

        arma::mat W = W0;
        arma::vec c(sample_no);

        double old_RSS = 0;

        // A never changes inside the outer loop; compute the Frobenius norm once.
        const double A_norm = small_dense::frobenius_norm(A);
        const bool inline_kernel = small_dense::use_inline_kernel(A.n_rows, A.n_cols);

        for (int it = 0; it < max_it; it++) {
            H = actionet::runSimplexRegression(W, A, true);

            arma::mat R = small_dense::residual_product(A, W, H);
            arma::mat Ht = arma::trans(H);
            for (int i = 0; i < k; i++) {
                arma::vec w = W.col(i);
                arma::vec h = Ht.col(i);

                double norm_sq = small_dense::dot(
                    static_cast<int>(h.n_elem), h.memptr(), 1,
                    h.memptr(), 1, true);
                if (norm_sq < AA_SINGULAR_THRESHOLD) {
                    // singular
                    int max_res_idx = arma::index_max(arma::rowvec(arma::sum(arma::square(R), 0)));
                    W.col(i) = A.col(max_res_idx);
                    c.zeros();
                    c(max_res_idx) = 1;
                    C.col(i) = c;
                } else {
                    arma::vec b = w;
                    small_dense::gemv(
                        false, static_cast<int>(R.n_rows), static_cast<int>(R.n_cols),
                        1.0 / norm_sq, R.memptr(), static_cast<int>(R.n_rows),
                        h.memptr(), 1.0, b.memptr(), inline_kernel);

                    C.col(i) = actionet::runSimplexRegression(A, b, false);

                    arma::vec w_new(A.n_rows);
                    small_dense::gemv(
                        false, static_cast<int>(A.n_rows), static_cast<int>(A.n_cols),
                        1.0, A.memptr(), static_cast<int>(A.n_rows),
                        C.colptr(i), 0.0, w_new.memptr(), inline_kernel);
                    arma::vec delta = (w - w_new);

                    // Rank-1 update: R += delta*h
                    small_dense::rank_one_update(
                        static_cast<int>(R.n_rows), static_cast<int>(R.n_cols), 1.0,
                        delta.memptr(), h.memptr(), R.memptr(),
                        static_cast<int>(R.n_rows), inline_kernel);

                    W.col(i) = w_new;
                }
            }
            double RSS = small_dense::frobenius_norm(R);
            double delta_RSS = std::abs(RSS - old_RSS) / A_norm;
            old_RSS = RSS;

            if (delta_RSS < tol)
                break;
        }

        small_dense::clamp_and_normalize_columns(C);
        small_dense::clamp_and_normalize_columns(H);

        arma::field<arma::mat> decomposition(2, 1);
        decomposition(0) = C;
        decomposition(1) = H;

        return decomposition;
    }

} // namespace actionet
