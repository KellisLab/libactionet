#include "tools/xicor.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "aarand/aarand.hpp"

namespace actionet {
    namespace {
    // Rank a numeric vector; ties get their average rank (method=0) or the upper-tie index (method=1).
    arma::vec rank_vec(arma::vec x, int method = 0) {
        int n = x.n_elem;
        arma::vec ranks(n);
        arma::uvec indx = arma::sort_index(x, "ascend");

        int ib = 0, i;
        double b = x[indx[0]];
        for (i = 1; i < n; ++i) {
            if (x[indx[i]] != b) {
                if (ib < i - 1) {
                    double rnk = method == 0 ? ((i - 1 + ib + 2) / 2.0) : (i - 1);
                    for (int j = ib; j <= i - 1; ++j)
                        ranks[indx[j]] = rnk;
                }
                else {
                    ranks[indx[ib]] = (double)(ib + 1);
                }
                b = x[indx[i]];
                ib = i;
            }
        }
        if (ib == i - 1)
            ranks[indx[ib]] = (double)i;
        else {
            double rnk = method == 0 ? ((i - 1 + ib + 2) / 2.0) : (i - 1);
            for (int j = ib; j <= i - 1; ++j)
                ranks[indx[j]] = rnk;
        }

        return ranks;
    }
    } // anonymous namespace

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

            double z = std::sqrt(n) * xi / std::sqrt(v);

            out(1) = z;
        }
        else {
            out(1) = 0;
        }

        return (out);
    }

    arma::field<arma::mat> XICOR(const arma::mat& X, const arma::mat& Y, bool compute_pval, int seed, int thread_no) {
        arma::field<arma::mat> out(2);

        arma::mat X_ = X;
        arma::mat Y_ = Y;
        bool swapped = false;
        if (X_.n_cols < Y_.n_cols) {
            swapped = true;
            std::swap(X_, Y_);
        }

        arma::mat XI = arma::zeros(X_.n_cols, Y_.n_cols);
        arma::mat XI_Z = arma::zeros(X_.n_cols, Y_.n_cols);

        int threads_use = get_num_threads(X_.n_cols, thread_no);
        #pragma omp parallel for num_threads(threads_use)
        for (int i = 0; i < X_.n_cols; i++) {
            arma::vec x = X_.col(i);
            for (int j = 0; j < Y_.n_cols; j++) {
                arma::vec y = Y_.col(j);
                arma::vec xi_out = xicor(x, y, compute_pval, seed);
                XI(i, j) = xi_out(0);
                XI_Z(i, j) = xi_out(1);
            }
        }

        if (swapped) {
            XI = arma::trans(XI);
            XI_Z = arma::trans(XI_Z);
        }

        out(0) = XI;
        out(1) = XI_Z;

        return (out);
    }
} // namespace actionet
