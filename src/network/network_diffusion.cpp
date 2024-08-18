// Network imputation using PageRank
#include "network/network_diffusion.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_matrix.hpp"
#include <cholmod.h>

namespace actionet {
    arma::mat compute_network_diffusion(arma::sp_mat &G, arma::sp_mat &X0, int thread_no, double alpha, int max_it) {
        thread_no = std::min(thread_no, (int) X0.n_cols);

        int N = G.n_rows;
        arma::vec z = arma::ones(N);
        arma::vec c = arma::vec(arma::trans(arma::sum(G, 0)));
        arma::uvec idx = arma::find(c);
        z(idx) = arma::ones(idx.n_elem) * (1.0 - alpha);
        z = z / N;

        arma::sp_mat P = alpha * arma::normalise(G, 1, 0);
        X0 = arma::normalise(X0, 1, 0);
        arma::mat X = arma::mat(X0);

        X0 *= N;
        arma::rowvec zt = trans(z);

        for (int it = 0; it < max_it; it++) {
            mini_thread::parallelFor(
                0, X.n_cols, [&](size_t i) {
                    X.col(i) = P * X.col(i) + X0.col(i) * (zt * X.col(i));
                },
                thread_no);
        }

        return (X);
    }

    arma::mat compute_network_diffusion_fast(arma::sp_mat &G, arma::sp_mat &X0, int thread_no, double alpha,
                                             int max_it) {
        thread_no = std::min(thread_no, (int) X0.n_cols);

        int n = G.n_rows;

        cholmod_common chol_c;
        cholmod_start(&chol_c);

        int *Ti, *Tj;
        double *Tx;

        arma::sp_mat P = alpha * arma::normalise(G, 1, 0);

        cholmod_triplet *T = cholmod_allocate_triplet(P.n_rows, P.n_cols, P.n_nonzero,
                                                      0, CHOLMOD_REAL, &chol_c);
        T->nnz = P.n_nonzero;
        Ti = static_cast<int *>(T->i);
        Tj = static_cast<int *>(T->j);
        Tx = static_cast<double *>(T->x);
        int idx = 0;
        for (arma::sp_mat::const_iterator it = P.begin(); it != P.end(); ++it) {
            Ti[idx] = it.row();
            Tj[idx] = it.col();
            Tx[idx] = (*it);
            idx++;
        }
        cholmod_sparse *AS = cholmod_triplet_to_sparse(T, P.n_nonzero, &chol_c);
        cholmod_free_triplet(&T, &chol_c);

        arma::vec z = arma::ones(n);
        arma::vec cs = arma::vec(arma::trans(arma::sum(G, 0)));
        arma::uvec nnz_idx = arma::find(cs > 0);
        z(nnz_idx) = arma::ones(nnz_idx.n_elem) * (1.0 - alpha);
        z = z / n;

        X0 = arma::normalise(X0, 1, 0);
        arma::mat X = arma::mat(X0);
        X0 *= n;
        arma::rowvec zt = arma::trans(z);

        for (int it = 0; it < max_it; it++) {
            arma::mat Y = X;
            mini_thread::parallelFor(
                0, X.n_cols, [&](size_t i) {
                    dsdmult('n', n, n, AS, X.colptr(i), Y.colptr(i), &chol_c);
                    X.col(i) = Y.col(i) + X0.col(i) * (zt * X.col(i));
                },
                thread_no);
        }

        // Free up matrices
        cholmod_free_triplet(&T, &chol_c);
        cholmod_free_sparse(&AS, &chol_c);
        cholmod_finish(&chol_c);

        return (X);
    }

    // P is already a stochastic (normalized) adjacency matrix
    arma::mat compute_network_diffusion_Chebyshev(arma::sp_mat &P, arma::mat &X0, int thread_no, double alpha,
                                                  int max_it, double res_threshold) {
        if (alpha == 1) {
            //            fprintf(stderr, "alpha should be in (0, 1). Value of %.2f was provided.\n", alpha);
            stderr_printf("alpha should be in (0, 1). Value of %.2f was provided.\n", alpha);
            FLUSH;
            return (X0);
        } else if (alpha == 0) {
            return (X0);
        }

        alpha = 1 - alpha; // Traditional definition is to have alpha as weight of prior.
        // Here, alpha is depth of diffusion

        arma::mat mPPreviousScore = X0; // zeros(size(X0));
        arma::mat mPreviousScore = (1 - alpha) * spmat_mat_product_parallel(P, mPPreviousScore, thread_no) + alpha * X0;
        double muPPrevious = 1.0, muPrevious = 1 / (1 - alpha);

        if (max_it <= 0)
            return (mPreviousScore);

        arma::mat mScore;
        for (int i = 0; i < max_it; i++) {
            double mu = 2.0 / (1.0 - alpha) * muPrevious - muPPrevious;

            mScore = 2 * (muPrevious / mu) * spmat_mat_product_parallel(P, mPreviousScore, thread_no) -
                     (muPPrevious / mu) * mPPreviousScore + (2 * muPrevious) / ((1 - alpha) * mu) * alpha * X0;

            double res = norm(mScore - mPreviousScore);
            if (res < res_threshold) {
                break;
            }

            // Change variables
            muPPrevious = muPrevious;
            muPrevious = mu;
            mPPreviousScore = mPreviousScore;
            mPreviousScore = mScore;
        }

        // Temporary fix. Sometimes diffusion values become small negative numbers
        double m0 = arma::min(arma::min(X0));
        if (0 <= m0) {
            mScore = arma::clamp(mScore, 0, arma::max(arma::max(mScore)));
        }

        return (mScore);
    }
} // namespace actionet
