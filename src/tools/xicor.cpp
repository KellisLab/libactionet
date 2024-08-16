#include "tools/xicor.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_misc.hpp"
#include "aarand/aarand.hpp"

namespace ACTIONet {

    arma::vec xicor(arma::vec xvec, arma::vec yvec, bool compute_pval, int seed) {
        arma::vec out(2);

        std::mt19937_64 engine(seed);

        arma::vec idx = arma::regspace(0, xvec.n_elem - 1);
        aarand::shuffle(idx.memptr(), idx.n_elem, engine);
        arma::uvec perm = arma::conv_to<arma::uvec>::from(idx);

        xvec = xvec(perm);
        yvec = yvec(perm);

        int n = xvec.n_elem;

        arma::vec er = rank_vec(xvec);
        arma::vec fr = rank_vec(yvec, 1) / n;
        arma::vec gr = rank_vec(-yvec, 1) / n;

        arma::uvec ord = arma::sort_index(er);
        fr = fr(ord);

        // Calculate xi
        double A1 = arma::sum(arma::abs(fr(arma::span(0, n - 2)) - fr(arma::span(1, n - 1)))) / (2.0 * n);
        double CU = arma::mean(gr % (1.0 - gr));
        double xi = 1 - (A1 / CU);
        out(0) = xi;

        if (compute_pval == true) {
            // Calculate p-values
            arma::vec qfr = arma::sort(fr);
            arma::vec ind = arma::regspace(0, n - 1);
            arma::vec ind2 = 2 * n - 2 * ind + 1;
            double ai = arma::mean(ind2 % arma::square(qfr)) / n;
            double ci = arma::mean(ind2 % qfr) / n;
            arma::vec cq = arma::cumsum(qfr);
            arma::vec m = (cq + (n - ind) % qfr) / n;
            double b = arma::mean(arma::square(m));
            double v = (ai - 2 * b + (ci * ci)) / (CU * CU);

            double xi_sd = std::sqrt(v / n);
            double z = std::sqrt(n) * xi / std::sqrt(v);

            out(1) = z;
        } else {
            out(1) = 0;
        }

        return (out);
    }

    arma::field<arma::mat> XICOR(arma::mat &X, arma::mat &Y, bool compute_pval, int seed, int thread_no) {
        arma::field<arma::mat> out(2);

        // arma_rng::set_seed(seed);
        // uvec perm = randperm(X.n_rows);

        // X = X.rows(perm);
        // Y = Y.rows(perm);

        bool swapped = false;
        int n1 = X.n_cols, n2 = Y.n_cols;
        if (n1 < n2) {
            swapped = true;
            arma::mat Z = X;
            X = Y;
            Y = Z;
        }

        arma::mat XI = arma::zeros(X.n_cols, Y.n_cols);
        arma::mat XI_Z = arma::zeros(X.n_cols, Y.n_cols);

        // for(int k = 0; k < associations.n_cols; k++) {
        mini_thread::parallelFor(
                0, X.n_cols, [&](size_t i) {
                    arma::vec x = X.col(i);
                    for (int j = 0; j < Y.n_cols; j++) {
                        arma::vec y = Y.col(j);
                        arma::vec out = xicor(x, y, compute_pval, seed);
                        XI(i, j) = out(0);
                        XI_Z(i, j) = out(1);
                    }
                },
                thread_no);

        if (swapped) {
            XI = arma::trans(XI);
            XI_Z = arma::trans(XI_Z);
        }

        out(0) = XI;
        out(1) = XI_Z;

        return (out);
    }

} // namespace ACTIONet
