// Network imputation using PageRank
#include "network/network_diffusion.hpp"
#include "utils_internal/utils_parallel.hpp"
#include "utils_internal/utils_matrix.hpp"
#include <tools/matrix_transform.hpp>

namespace {

// Prepared graph state: normalized+scaled adjacency and teleportation vector.
// Created once by prepareGraph_(), consumed by the _prepared diffusion helpers.
struct PreparedGraph {
    arma::sp_mat Gn;       // normalized, then scaled by alpha
    arma::rowvec zt;       // teleportation vector (transposed)
};

PreparedGraph prepareGraph_(const arma::sp_mat& G, int norm_method, double alpha) {
    PreparedGraph pg;
    pg.Gn = G;
    arma::vec cs = arma::vec(arma::trans(arma::sum(pg.Gn, 0)));
    actionet::normalizeGraph(pg.Gn, norm_method);
    pg.Gn *= alpha;

    const size_t n = G.n_rows;
    arma::vec z = arma::ones(n);
    z(arma::find(cs > 0)).fill(1.0 - alpha);
    z /= n;
    pg.zt = z.t();
    return pg;
}

// ----------------------------------------------------------------
// _prepared variants: operate on a pre-prepared graph (no mutation)
// ----------------------------------------------------------------

arma::mat diffusionPowerIter_prepared(const PreparedGraph& pg,
                                      const arma::mat& X0_norm,
                                      int max_it, int thread_no) {
    const double n_dbl = static_cast<double>(pg.Gn.n_rows);
    arma::mat X_out = X0_norm;

    int threads_use = actionet::get_num_threads(X_out.n_cols, thread_no);

    for (int it = 0; it < max_it; it++) {
        #pragma omp parallel for num_threads(threads_use) schedule(static)
        for (size_t i = 0; i < X_out.n_cols; i++) {
            arma::vec y = pg.Gn * X_out.col(i);
            X_out.col(i) = y + (X0_norm.col(i) * n_dbl) * arma::as_scalar(pg.zt * X_out.col(i));
        }
    }

    return X_out;
}

arma::mat diffusionPowerIterSparse_prepared(const PreparedGraph& pg,
                                             const arma::sp_mat& X0,
                                             int max_it, int thread_no) {
    const double n_dbl = static_cast<double>(pg.Gn.n_rows);

    arma::sp_mat X0_norm = arma::normalise(X0, 1, 0);
    arma::mat X_out(X0_norm);
    arma::sp_mat X0_scaled = X0_norm * n_dbl;

    int threads_use = actionet::get_num_threads(X_out.n_cols, thread_no);

    for (int it = 0; it < max_it; it++) {
        #pragma omp parallel for num_threads(threads_use) schedule(static)
        for (size_t i = 0; i < X_out.n_cols; i++) {
            arma::vec y = pg.Gn * X_out.col(i);
            arma::vec x0_col(X0_scaled.col(i));
            X_out.col(i) = y + x0_col * arma::as_scalar(pg.zt * X_out.col(i));
        }
    }

    return X_out;
}

arma::mat diffusionChebyshev_prepared(const arma::sp_mat& Gn_unscaled,
                                       const arma::mat& X0,
                                       double alpha, int max_it,
                                       double tol, int thread_no) {
    // Chebyshev uses (1 - alpha) internally and only needs the
    // column-normalized graph (no alpha scaling), so the caller passes
    // a graph that has been normalizeGraph'd but NOT scaled by alpha.
    // alpha here is already flipped: caller passes (1 - original_alpha).

    arma::mat buf_a = X0;
    arma::mat buf_b = (1.0 - alpha) * actionet::spmat_mat_product_parallel(Gn_unscaled, buf_a, thread_no) + alpha * X0;
    double mu_pp = 1.0, mu_p = 1.0 / (1.0 - alpha);

    if (max_it <= 0) return buf_b;

    arma::mat buf_c(X0.n_rows, X0.n_cols);

    arma::mat* pp  = &buf_a;
    arma::mat* p   = &buf_b;
    arma::mat* cur = &buf_c;

    for (int i = 0; i < max_it; i++) {
        double mu = 2.0 / (1.0 - alpha) * mu_p - mu_pp;

        *cur = 2.0 * (mu_p / mu) * actionet::spmat_mat_product_parallel(Gn_unscaled, *p, thread_no)
             - (mu_pp / mu) * (*pp)
             + (2.0 * mu_p) / ((1.0 - alpha) * mu) * alpha * X0;

        double res = arma::norm(*cur - *p, "fro");

        mu_pp = mu_p;
        mu_p = mu;
        arma::mat* tmp = pp;
        pp  = p;
        p   = cur;
        cur = tmp;

        if (res < tol) break;
    }

    double m0 = X0.min();
    if (m0 >= 0.0) {
        *p = arma::clamp(*p, 0.0, p->max());
    }

    return std::move(*p);
}

} // anon namespace

namespace actionet {

    // Dense input specialisation.
    template <>
    arma::mat computeNetworkDiffusion<arma::mat>(
            const arma::sp_mat& G, arma::mat& X0, double alpha, int max_it,
            int thread_no, bool approx, int norm_method, double tol) {
        if (alpha == 0.0) return X0;
        if (alpha <= 0.0 || alpha > 1.0)
            throw std::invalid_argument("'alpha' must be in (0,1)");

        if (approx) {
            // Chebyshev needs normalized-only graph (no alpha scaling).
            arma::sp_mat Gn = G;
            normalizeGraph(Gn, norm_method);
            return diffusionChebyshev_prepared(Gn, X0, 1.0 - alpha, max_it, tol, thread_no);
        }
        PreparedGraph pg = prepareGraph_(G, norm_method, alpha);
        arma::mat X0_norm = arma::normalise(X0, 1, 0);
        return diffusionPowerIter_prepared(pg, X0_norm, max_it, thread_no);
    }

    // Sparse input specialisation.
    template <>
    arma::mat computeNetworkDiffusion<arma::sp_mat>(
            const arma::sp_mat& G, arma::sp_mat& X0, double alpha, int max_it,
            int thread_no, bool approx, int norm_method, double tol) {
        if (alpha == 0.0) return arma::mat(X0);
        if (alpha <= 0.0 || alpha > 1.0)
            throw std::invalid_argument("'alpha' must be in (0,1)");

        if (approx) {
            arma::sp_mat Gn = G;
            normalizeGraph(Gn, norm_method);
            return diffusionChebyshev_prepared(Gn, arma::mat(X0), 1.0 - alpha, max_it, tol, thread_no);
        }
        PreparedGraph pg = prepareGraph_(G, norm_method, alpha);
        return diffusionPowerIterSparse_prepared(pg, X0, max_it, thread_no);
    }

} // namespace actionet
