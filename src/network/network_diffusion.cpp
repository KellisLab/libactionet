// Network imputation using PageRank
// Updated to use native Armadillo sparse operations
#include "network/network_diffusion.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_matrix.hpp"
#include <tools/matrix_transform.hpp>

// PageRank diffusion using native Armadillo sparse operations
arma::mat computeDiffusion(arma::sp_mat& G, arma::sp_mat X0, int norm_method, double alpha, int max_it, int thread_no) {
    size_t n = G.n_rows;

    arma::sp_mat P = alpha * actionet::normalizeGraph(G, norm_method);

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
            // Use native Armadillo sparse-dense multiply
            Y.col(i) = P * X_out.col(i);
            X_out.col(i) = Y.col(i) + X0.col(i) * (zt * X_out.col(i));
        }
    }

    return X_out;
}

// norm_method: 0 (pagerank), 2 (sym_pagerank)
arma::mat computeDiffusionChebyshev(arma::sp_mat& G, const arma::mat& X0, int norm_method, double alpha, int max_it,
                                    double tol, int thread_no) {
    // Traditional definition is to have alpha as weight of prior. Here, alpha is depth of diffusion
    alpha = 1 - alpha;

    arma::sp_mat P = actionet::normalizeGraph(G, norm_method);

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
            throw std::invalid_argument("'alpha' must be in (0,1)");
        }

        arma::mat X_out(X0.n_rows, X0.n_cols);
        if (approx) { // Fast approximate PageRank
            X_out = computeDiffusionChebyshev(G, arma::mat(X0), norm_method, alpha, max_it, tol, thread_no);
        }
        else { // PageRank (iterative)
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
