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
    // Fused: normalizeGraph reports the pre-normalization column sums from
    // its internal accumulation pass, eliminating the separate arma::sum(pg.Gn, 0)
    // pass that used to run before normalization.
    arma::vec cs;
    actionet::normalizeGraph(pg.Gn, norm_method, cs);
    pg.Gn *= alpha;

    const size_t n = G.n_rows;
    arma::vec z = arma::ones(n);
    z(arma::find(cs > 0)).fill(1.0 - alpha);
    z /= n;
    pg.zt = z.t();
    return pg;
}

// ----------------------------------------------------------------
// Templated power-iteration diffusion on a prepared graph.
// Handles dense and sparse X0 via a single templated function:
//   * arma::mat: caller-normalised (l1) X0, used directly.
//   * arma::sp_mat: normalise here (avoid dense copy), materialise X_out
//     from the normalised sparse matrix, cache a scaled dense copy of
//     X0 in the inner loop.
// ----------------------------------------------------------------
template <typename XT>
arma::mat diffusionPowerIter_prepared(const PreparedGraph& pg,
                                      const XT& X0_norm_or_raw,
                                      int max_it, int thread_no) {
    const double n_dbl = static_cast<double>(pg.Gn.n_rows);

    arma::mat X_out;
    // Precomputed reference term reused every outer iteration; equal to
    // n_dbl * (l1-normalised X0).
    arma::mat X0_scaled_dense;
    const bool sparse_input = arma::is_SpMat<XT>::value;

    if constexpr (arma::is_SpMat<XT>::value) {
        arma::sp_mat X0_norm = arma::normalise(X0_norm_or_raw, 1, 0);
        X_out = arma::mat(X0_norm);
        X0_scaled_dense = arma::mat(X0_norm * n_dbl);
    } else {
        X_out = X0_norm_or_raw;
    }

    int threads_use = actionet::get_num_threads(X_out.n_cols, thread_no);

    for (int it = 0; it < max_it; it++) {
        #pragma omp parallel for num_threads(threads_use) schedule(static)
        for (size_t i = 0; i < X_out.n_cols; i++) {
            arma::vec y = pg.Gn * X_out.col(i);
            if constexpr (arma::is_SpMat<XT>::value) {
                X_out.col(i) = y + X0_scaled_dense.col(i) *
                                        arma::as_scalar(pg.zt * X_out.col(i));
            } else {
                X_out.col(i) = y + (X0_norm_or_raw.col(i) * n_dbl) *
                                        arma::as_scalar(pg.zt * X_out.col(i));
            }
        }
    }

    (void)sparse_input;
    return X_out;
}

// Chebyshev-accelerated diffusion.  Takes @c one_minus_alpha explicitly so
// the caller does not have to encode the "prepared" semantics in the
// argument name.  Preludes the graph normalisation internally so the two
// call sites don't have to.
arma::mat diffusionChebyshev(const arma::sp_mat& G,
                             const arma::mat& X0,
                             double one_minus_alpha, int max_it,
                             double tol, int norm_method, int thread_no) {
    // Chebyshev uses (1 - alpha) internally and only needs the
    // column-normalized graph (no alpha scaling).
    arma::sp_mat Gn_unscaled = G;
    actionet::normalizeGraph(Gn_unscaled, norm_method);

    const double alpha = one_minus_alpha;
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
            return diffusionChebyshev(G, X0, 1.0 - alpha, max_it, tol, norm_method, thread_no);
        }
        PreparedGraph pg = prepareGraph_(G, norm_method, alpha);
        arma::mat X0_norm = arma::normalise(X0, 1, 0);
        return diffusionPowerIter_prepared<arma::mat>(pg, X0_norm, max_it, thread_no);
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
            return diffusionChebyshev(G, arma::mat(X0), 1.0 - alpha, max_it, tol, norm_method, thread_no);
        }
        PreparedGraph pg = prepareGraph_(G, norm_method, alpha);
        return diffusionPowerIter_prepared<arma::sp_mat>(pg, X0, max_it, thread_no);
    }

} // namespace actionet
