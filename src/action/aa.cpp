// Solves the standard Archetypal Analysis (AA) problem
#include "action/aa.hpp"
#include "action/simplex_regression.hpp"

namespace actionet {

    arma::field<arma::mat> runAA(arma::mat &A, arma::mat &W0, int max_it, double tol) {
        int sample_no = A.n_cols;
        int k = W0.n_cols; // AA components

        arma::mat C = arma::zeros(sample_no, k);
        arma::mat H = arma::zeros(k, sample_no);

        arma::mat W = W0;
        arma::vec c(sample_no);

        double old_RSS = 0;

        for (int it = 0; it < max_it; it++) {
            arma::mat C_old = C;
            arma::mat H_old = H;
            double A_norm = arma::norm(A, "fro");
            H = actionet::runSimplexRegression(W, A, true);

            arma::mat R = A - W * H;
            arma::mat Ht = arma::trans(H);
            for (int i = 0; i < k; i++) {
                arma::vec w = W.col(i);
                arma::vec h = Ht.col(i);

                double norm_sq = arma::dot(h, h);
                if (norm_sq < double(10e-8)) {
                    // singular
                    int max_res_idx = arma::index_max(arma::rowvec(arma::sum(arma::square(R), 0)));
                    W.col(i) = A.col(max_res_idx);
                    c.zeros();
                    c(max_res_idx) = 1;
                    C.col(i) = c;
                } else {
                    arma::vec b = w;
                    cblas_dgemv(CblasColMajor, CblasNoTrans, R.n_rows, R.n_cols,
                                (1.0 / norm_sq), R.memptr(), R.n_rows, h.memptr(), 1, 1,
                                b.memptr(), 1);

                    C.col(i) = actionet::runSimplexRegression(A, b, false);

                    arma::vec w_new = A * C.col(i);
                    arma::vec delta = (w - w_new);

                    // Rank-1 update: R += delta*h
                    cblas_dger(CblasColMajor, R.n_rows, R.n_cols, 1.0, delta.memptr(), 1,
                               h.memptr(), 1, R.memptr(), R.n_rows);

                    W.col(i) = w_new;
                }
            }
            double RSS = arma::norm(R, "fro");
            double delta_RSS = std::abs(RSS - old_RSS) / A_norm;
            old_RSS = RSS;

            if (delta_RSS < tol)
                break;
        }

        C = arma::clamp(C, 0, 1);
        C = arma::normalise(C, 1);
        H = arma::clamp(H, 0, 1);
        H = arma::normalise(H, 1);

        arma::field<arma::mat> decomposition(2, 1);
        decomposition(0) = C;
        decomposition(1) = H;

        return decomposition;
    }

} // namespace actionet
