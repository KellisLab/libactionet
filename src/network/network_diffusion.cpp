// Network imputation using PageRank
#include "network/network_diffusion.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_matrix.hpp"
#include <tools/matrix_transform.hpp>
#include <cholmod.h>

arma::mat computeDiffusion(arma::sp_mat& G, arma::sp_mat X0, int norm_type, double alpha, int max_it, int thread_no) {
    int n = G.n_rows;

    cholmod_common chol_c;
    cholmod_start(&chol_c);

    int *Ti, *Tj;
    double* Tx;

    arma::sp_mat P = alpha * actionet::normalizeGraph(G, norm_type);

    cholmod_triplet* T = cholmod_allocate_triplet(P.n_rows, P.n_cols, P.n_nonzero,
                                                  0, CHOLMOD_REAL, &chol_c);
    T->nnz = P.n_nonzero;
    Ti = static_cast<int*>(T->i);
    Tj = static_cast<int*>(T->j);
    Tx = static_cast<double*>(T->x);
    int idx = 0;
    for (arma::sp_mat::const_iterator it = P.begin(); it != P.end(); ++it) {
        Ti[idx] = it.row();
        Tj[idx] = it.col();
        Tx[idx] = (*it);
        idx++;
    }
    cholmod_sparse* AS = cholmod_triplet_to_sparse(T, P.n_nonzero, &chol_c);
    cholmod_free_triplet(&T, &chol_c);

    arma::vec z = arma::ones(n);
    arma::vec cs = arma::vec(arma::trans(arma::sum(G, 0)));
    arma::uvec nnz_idx = arma::find(cs > 0);
    z(nnz_idx) = arma::ones(nnz_idx.n_elem) * (1.0 - alpha);
    z = z / n;

    X0 = arma::normalise(X0, 1, 0);
    arma::mat X_out = arma::mat(X0);
    X0 *= n;
    arma::rowvec zt = arma::trans(z);

    int threads_use = get_num_threads(X_out.n_cols, thread_no);
    for (int it = 0; it < max_it; it++) {
        arma::mat Y = X_out;

        #pragma omp parallel for num_threads(threads_use)
        for (size_t i = 0; i < X_out.n_cols; i++) {
            dsdmult('n', n, n, AS, X_out.colptr(i), Y.colptr(i), &chol_c);
            X_out.col(i) = Y.col(i) + X0.col(i) * (zt * X_out.col(i));
        }
    }

    // Free up matrices
    cholmod_free_triplet(&T, &chol_c);
    cholmod_free_sparse(&AS, &chol_c);
    cholmod_finish(&chol_c);

    return (X_out);
}

// norm_type: 0 (pagerank), 2 (sym_pagerank)
arma::mat computeDiffusionChebyshev(arma::sp_mat& G, const arma::mat& X0, int norm_type, double alpha, int max_it, double tol,
                                    int thread_no) {
    // Traditional definition is to have alpha as weight of prior. Here, alpha is depth of diffusion
    alpha = 1 - alpha;

    arma::sp_mat P = actionet::normalizeGraph(G, norm_type);

    arma::mat mPPreviousScore = X0; // zeros(size(X0));
    arma::mat mPreviousScore = (1 - alpha) * spmat_mat_product_parallel(P, mPPreviousScore, thread_no) + alpha * X0;
    double muPPrevious = 1.0, muPrevious = 1 / (1 - alpha);

    if (max_it <= 0)
        return (mPreviousScore);

    arma::mat X_out;
    for (int i = 0; i < max_it; i++) {
        double mu = 2.0 / (1.0 - alpha) * muPrevious - muPPrevious;

        X_out = 2 * (muPrevious / mu) * spmat_mat_product_parallel(P, mPreviousScore, thread_no) -
            (muPPrevious / mu) * mPPreviousScore + (2 * muPrevious) / ((1 - alpha) * mu) * alpha * X0;

        double res = norm(X_out - mPreviousScore);
        if (res < tol) {
            break;
        }

        // Change variables
        muPPrevious = muPrevious;
        muPrevious = mu;
        mPPreviousScore = mPreviousScore;
        mPreviousScore = X_out;
    }

    // Temporary fix. Sometimes diffusion values become small negative numbers
    double m0 = arma::min(arma::min(X0));
    if (0 <= m0) {
        X_out = arma::clamp(X_out, 0, arma::max(arma::max(X_out)));
    }

    return (X_out);
}

namespace actionet {
    template <typename T>
    arma::mat computeNetworkDiffusion(arma::sp_mat& G, T& X0, double alpha, int max_it, int thread_no,
                                      bool approx, int norm_method, double tol) {
        if (alpha == 0) {
            return arma::mat(X0);
        }
        if (alpha <= 0 || alpha > 1) {
            throw std::invalid_argument("Invalid `alpha`");
        }

        arma::mat X_out(X0.n_rows, X0.n_cols);
        if (approx) { // Fast approximate PageRank
            X_out = computeDiffusionChebyshev(G, arma::mat(X0), norm_method, alpha, max_it, tol, thread_no);
        }
        else { // PageRank (using cholmod)
            X_out = computeDiffusion(G, arma::sp_mat(X0), norm_method, alpha, max_it, thread_no);
        }

        return (X_out);
    };

    template arma::mat computeNetworkDiffusion<arma::mat>(arma::sp_mat& G, arma::mat& X0, double alpha, int max_it,
                                                          int thread_no, bool approx, int norm_method, double tol);
    template arma::mat computeNetworkDiffusion<arma::sp_mat>(arma::sp_mat& G, arma::sp_mat& X0, double alpha,
                                                             int max_it, int thread_no, bool approx, int norm_method,
                                                             double tol);
} // namespace actionet
