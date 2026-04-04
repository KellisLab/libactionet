// Network imputation using PageRank
#include "network/network_diffusion.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_matrix.hpp"
#include <tools/matrix_transform.hpp>

namespace {

// PageRank power-iteration diffusion.
// Operates directly on dense X0 (already L1-normalised column-wise by caller).
arma::mat diffusionPowerIter(arma::sp_mat& G, const arma::mat& X0_norm,
                              int norm_method, double alpha, int max_it, int thread_no) {
    const size_t n = G.n_rows;

    arma::vec cs = arma::vec(arma::trans(arma::sum(G, 0)));
    actionet::normalizeGraph(G, norm_method);
    G *= alpha;

    arma::vec z = arma::ones(n);
    z(arma::find(cs > 0)).fill(1.0 - alpha);
    z /= n;
    arma::rowvec zt = z.t();

    const double n_dbl = static_cast<double>(n);
    arma::mat X_out = X0_norm;  // start from normalised input

    int threads_use = actionet::get_num_threads(X_out.n_cols, thread_no);

    for (int it = 0; it < max_it; it++) {
        #pragma omp parallel for num_threads(threads_use) schedule(static)
        for (size_t i = 0; i < X_out.n_cols; i++) {
            arma::vec y = G * X_out.col(i);
            X_out.col(i) = y + (X0_norm.col(i) * n_dbl) * arma::as_scalar(zt * X_out.col(i));
        }
    }

    return X_out;
}

// Sparse-input variant: keeps X0 sparse for the initial sparse-dense multiply,
// then switches to dense for iteration.
arma::mat diffusionPowerIterSparse(arma::sp_mat& G, const arma::sp_mat& X0,
                                    int norm_method, double alpha, int max_it, int thread_no) {
    const size_t n = G.n_rows;

    arma::vec cs = arma::vec(arma::trans(arma::sum(G, 0)));
    actionet::normalizeGraph(G, norm_method);
    G *= alpha;

    arma::vec z = arma::ones(n);
    z(arma::find(cs > 0)).fill(1.0 - alpha);
    z /= n;
    arma::rowvec zt = z.t();

    arma::sp_mat X0_norm = arma::normalise(X0, 1, 0);
    arma::mat X_out(X0_norm);
    arma::sp_mat X0_scaled = X0_norm * static_cast<double>(n);

    int threads_use = actionet::get_num_threads(X_out.n_cols, thread_no);

    for (int it = 0; it < max_it; it++) {
        #pragma omp parallel for num_threads(threads_use) schedule(static)
        for (size_t i = 0; i < X_out.n_cols; i++) {
            arma::vec y = G * X_out.col(i);
            arma::vec x0_col(X0_scaled.col(i));
            X_out.col(i) = y + x0_col * arma::as_scalar(zt * X_out.col(i));
        }
    }

    return X_out;
}

// Chebyshev-accelerated approximate PageRank diffusion.
arma::mat diffusionChebyshev(arma::sp_mat& G, const arma::mat& X0, int norm_method,
                              double alpha, int max_it, double tol, int thread_no) {
    alpha = 1.0 - alpha;

    actionet::normalizeGraph(G, norm_method);

    arma::mat prev_prev = X0;
    arma::mat prev = (1.0 - alpha) * actionet::spmat_mat_product_parallel(G, prev_prev, thread_no) + alpha * X0;
    double mu_pp = 1.0, mu_p = 1.0 / (1.0 - alpha);

    if (max_it <= 0) return prev;

    arma::mat X_out;
    for (int i = 0; i < max_it; i++) {
        double mu = 2.0 / (1.0 - alpha) * mu_p - mu_pp;

        X_out = 2.0 * (mu_p / mu) * actionet::spmat_mat_product_parallel(G, prev, thread_no)
              - (mu_pp / mu) * prev_prev
              + (2.0 * mu_p) / ((1.0 - alpha) * mu) * alpha * X0;

        double res = arma::norm(X_out - prev, "fro");
        if (res < tol) break;

        mu_pp = mu_p;
        mu_p = mu;
        prev_prev = std::move(prev);
        prev = X_out;
    }

    double m0 = X0.min();
    if (m0 >= 0.0) {
        X_out = arma::clamp(X_out, 0.0, X_out.max());
    }

    return X_out;
}

} // anon namespace

namespace actionet {

    // Dense input specialisation: no sparse conversion needed.
    template <>
    arma::mat computeNetworkDiffusion<arma::mat>(
            arma::sp_mat& G, arma::mat& X0, double alpha, int max_it,
            int thread_no, bool approx, int norm_method, double tol) {
        if (alpha == 0.0) return X0;
        if (alpha <= 0.0 || alpha > 1.0)
            throw std::invalid_argument("'alpha' must be in (0,1)");

        if (approx) {
            return diffusionChebyshev(G, X0, norm_method, alpha, max_it, tol, thread_no);
        }
        arma::mat X0_norm = arma::normalise(X0, 1, 0);
        return diffusionPowerIter(G, X0_norm, norm_method, alpha, max_it, thread_no);
    }

    // Sparse input specialisation: stays sparse through normalisation.
    template <>
    arma::mat computeNetworkDiffusion<arma::sp_mat>(
            arma::sp_mat& G, arma::sp_mat& X0, double alpha, int max_it,
            int thread_no, bool approx, int norm_method, double tol) {
        if (alpha == 0.0) return arma::mat(X0);
        if (alpha <= 0.0 || alpha > 1.0)
            throw std::invalid_argument("'alpha' must be in (0,1)");

        if (approx) {
            return diffusionChebyshev(G, arma::mat(X0), norm_method, alpha, max_it, tol, thread_no);
        }
        return diffusionPowerIterSparse(G, X0, norm_method, alpha, max_it, thread_no);
    }

} // namespace actionet
