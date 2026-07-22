#include "tools/xicor.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "aarand/aarand.hpp"

namespace actionet {
    namespace {
    // Rank a numeric vector.
    //   method=0 (average) : ties get the average of their positions (1-based).
    //   method=1 (max)     : ties get the maximum position of their group (1-based).
    // Both conventions match R's rank() with ties.method="average" and "max" respectively.
    arma::vec rank_vec(const arma::vec& x, int method = 0) {
        const int n = x.n_elem;
        arma::vec ranks(n);
        // Stable sort so that equal-value elements retain their input order; the
        // caller controls tie-breaking of the *input* (e.g., by permuting `x`
        // beforehand) rather than relying on unspecified sort behavior.
        arma::uvec indx = arma::stable_sort_index(x, "ascend");

        int ib = 0, i;
        double b = x[indx[0]];
        for (i = 1; i < n; ++i) {
            if (x[indx[i]] != b) {
                if (ib < i - 1) {
                    // Tied group occupies 1-based positions [ib+1, i]. Its
                    // maximum position is `i`; its average is (ib+1 + i) / 2.
                    double rnk = (method == 0)
                        ? ((ib + 1 + i) / 2.0)
                        : (double)i;
                    for (int j = ib; j <= i - 1; ++j)
                        ranks[indx[j]] = rnk;
                } else {
                    ranks[indx[ib]] = (double)(ib + 1);
                }
                b = x[indx[i]];
                ib = i;
            }
        }
        // Final group (or final singleton) closes at position `n`.
        if (ib == i - 1) {
            ranks[indx[ib]] = (double)i;
        } else {
            double rnk = (method == 0)
                ? ((ib + 1 + i) / 2.0)
                : (double)i;
            for (int j = ib; j <= i - 1; ++j)
                ranks[indx[j]] = rnk;
        }

        return ranks;
    }
    } // anonymous namespace

    arma::vec xicor(arma::vec xvec, arma::vec yvec, bool compute_pval, int seed) {
        arma::vec out(2);

        const int n = xvec.n_elem;

        // Chatterjee 2020 breaks ties in X *randomly* to preserve the asymptotic
        // theory. We achieve this by permuting the (x, y) pair jointly before
        // ranking. When ranks of X are then ordered via a stable sort, tied X
        // values are visited in the (random) order set by this permutation.
        // Without this, xi would depend on the input order of tied X values.
        if (seed != 0) {
            std::mt19937_64 engine(seed);
            arma::vec idx = arma::regspace(0, n - 1);
            aarand::shuffle(idx.memptr(), idx.n_elem, engine);
            arma::uvec perm = arma::conv_to<arma::uvec>::from(idx);
            xvec = xvec(perm);
            yvec = yvec(perm);
        }

        // fr and gr correspond to Chatterjee's r_i and l_i divided by n:
        //   fr[i] = #{j : y_j <= y_i} / n   (rank(y, ties.method="max") / n)
        //   gr[i] = #{j : y_j >= y_i} / n   (rank(-y, ties.method="max") / n)
        arma::vec fr = rank_vec(yvec, 1) / (double)n;
        arma::vec gr = rank_vec(-yvec, 1) / (double)n;

        // Order indices by X (ties broken by the permutation applied above).
        arma::uvec ord = arma::stable_sort_index(xvec);
        fr = fr(ord);

        // xi coefficient:
        //   A1 = sum(|fr_{i+1} - fr_i|) / (2n)
        //   CU = mean(gr * (1 - gr))
        //   xi = 1 - A1 / CU
        double A1 = arma::sum(arma::abs(fr(arma::span(0, n - 2)) - fr(arma::span(1, n - 1)))) / (2.0 * n);
        double CU = arma::mean(gr % (1.0 - gr));
        double xi = 1.0 - (A1 / CU);
        out(0) = xi;

        if (compute_pval) {
            // Asymptotic variance under ties (Chatterjee & Holmes; XICOR::xicor).
            // ind = 1..n (1-based), matching the reference R implementation.
            arma::vec qfr = arma::sort(fr);
            arma::vec ind = arma::regspace(1, n);
            arma::vec ind2 = 2.0 * n - 2.0 * ind + 1.0;
            double ai = arma::mean(ind2 % arma::square(qfr)) / (double)n;
            double ci = arma::mean(ind2 % qfr) / (double)n;
            arma::vec cq = arma::cumsum(qfr);
            arma::vec m = (cq + ((double)n - ind) % qfr) / (double)n;
            double b = arma::mean(arma::square(m));
            double v = (ai - 2.0 * b + (ci * ci)) / (CU * CU);

            double z = std::sqrt((double)n) * xi / std::sqrt(v);
            out(1) = z;
        } else {
            out(1) = 0.0;
        }

        return out;
    }

    arma::field<arma::mat> XICOR(const arma::mat& X, const arma::mat& Y, bool compute_pval, int seed, int thread_no) {
        arma::field<arma::mat> out(2);

        // For each (i, j) column pair we need:
        //   fr_j = rank(Y_j, "max") / n     — depends only on Y_j
        //   gr_j = rank(-Y_j, "max") / n    — depends only on Y_j
        //   ord_i = stable_sort_index(X_i)  — depends only on X_i
        // Precompute per-column ranks/orders once and reuse across all pairs.
        // This turns the naive O(nX * nY) rank recomputations into O(nX + nY).
        //
        // Random tie-breaking on X: apply a single joint permutation to the
        // row space of X and Y before precomputation. This makes tied X values
        // appear in a random order inside stable_sort_index (matching the
        // scalar `xicor` path).
        const arma::uword n = X.n_rows;
        if (Y.n_rows != n) {
            throw std::invalid_argument(
                "XICOR: X and Y must have the same number of rows");
        }

        arma::uvec perm;
        bool have_perm = false;
        if (seed != 0) {
            std::mt19937_64 engine(seed);
            arma::vec idx = arma::regspace(0, (double)n - 1);
            aarand::shuffle(idx.memptr(), idx.n_elem, engine);
            perm = arma::conv_to<arma::uvec>::from(idx);
            have_perm = true;
        }

        auto permuted_col = [&](const arma::mat& M, arma::uword c) -> arma::vec {
            return have_perm ? arma::vec(M.col(c))(perm) : arma::vec(M.col(c));
        };

        // Precompute per-column fr, gr, and ord.
        std::vector<arma::vec> fr_cols(Y.n_cols);
        std::vector<arma::vec> gr_cols(Y.n_cols);
        std::vector<arma::uvec> ord_cols(X.n_cols);

        int threads_use = get_num_threads(std::max<arma::uword>(X.n_cols, Y.n_cols), thread_no);

        #pragma omp parallel for num_threads(threads_use)
        for (arma::uword j = 0; j < Y.n_cols; ++j) {
            arma::vec y = permuted_col(Y, j);
            fr_cols[j] = rank_vec(y, 1) / (double)n;
            gr_cols[j] = rank_vec(-y, 1) / (double)n;
        }

        #pragma omp parallel for num_threads(threads_use)
        for (arma::uword i = 0; i < X.n_cols; ++i) {
            arma::vec x = permuted_col(X, i);
            ord_cols[i] = arma::stable_sort_index(x);
        }

        arma::mat XI = arma::zeros(X.n_cols, Y.n_cols);
        arma::mat XI_Z = arma::zeros(X.n_cols, Y.n_cols);

        // Precompute the p-value ind/ind2 vectors once (reused for every pair).
        arma::vec ind, ind2;
        if (compute_pval) {
            ind = arma::regspace(1, n);
            ind2 = 2.0 * n - 2.0 * ind + 1.0;
        }

        #pragma omp parallel for collapse(2) num_threads(threads_use)
        for (arma::uword i = 0; i < X.n_cols; ++i) {
            for (arma::uword j = 0; j < Y.n_cols; ++j) {
                const arma::vec& fr_full = fr_cols[j];
                const arma::vec& gr = gr_cols[j];
                const arma::uvec& ord = ord_cols[i];

                arma::vec fr = fr_full(ord);

                double A1 = arma::sum(arma::abs(fr(arma::span(0, n - 2))
                                                - fr(arma::span(1, n - 1)))) / (2.0 * n);
                double CU = arma::mean(gr % (1.0 - gr));
                double xi = 1.0 - (A1 / CU);
                XI(i, j) = xi;

                if (compute_pval) {
                    arma::vec qfr = arma::sort(fr);
                    double ai = arma::mean(ind2 % arma::square(qfr)) / (double)n;
                    double ci = arma::mean(ind2 % qfr) / (double)n;
                    arma::vec cq = arma::cumsum(qfr);
                    arma::vec m = (cq + ((double)n - ind) % qfr) / (double)n;
                    double b = arma::mean(arma::square(m));
                    double v = (ai - 2.0 * b + (ci * ci)) / (CU * CU);
                    XI_Z(i, j) = std::sqrt((double)n) * xi / std::sqrt(v);
                }
            }
        }

        out(0) = XI;
        out(1) = XI_Z;

        return out;
    }
} // namespace actionet
