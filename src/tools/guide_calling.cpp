#include "tools/guide_calling.hpp"

#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"
#include "utils_internal/utils_parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace {
    constexpr double kLog10 = 2.3025850929940456840179914546844;
    constexpr double kSqrt2Pi = 2.506628274631000502415765284811;

    /// Random initialisation state for a single 2-component EM run.
    struct GuideInitState {
        double w0 = 0.5;
        double mu0 = 0.0;
        double mu1 = 1.0;
        double sigma = 1.0;
    };

    /// Output of a single EM run for one guide.  Fields default to NaN /
    /// -inf so that an unfilled state is distinguishable from a real result.
    struct GuideFitState {
        double w0 = std::numeric_limits<double>::quiet_NaN();
        double w1 = std::numeric_limits<double>::quiet_NaN();
        double mu0 = std::numeric_limits<double>::quiet_NaN();
        double mu1 = std::numeric_limits<double>::quiet_NaN();
        double sigma = std::numeric_limits<double>::quiet_NaN();
        double ll = -std::numeric_limits<double>::infinity();
        bool converged = false;
    };

    /// Clamp probability to (eps, 1-eps); non-finite inputs return NaN.
    double clamp_prob(const double p) {
        constexpr double eps = 1e-12;
        if (!std::isfinite(p)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::min(1.0 - eps, std::max(eps, p));
    }

    /// Peter J. Acklam's rational approximation to the inverse normal CDF.
    double inverse_normal_cdf(const double p_in) {
        const double p = clamp_prob(p_in);
        if (!std::isfinite(p)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Peter J. Acklam's rational approximation.
        static const double a1 = -3.969683028665376e+01;
        static const double a2 = 2.209460984245205e+02;
        static const double a3 = -2.759285104469687e+02;
        static const double a4 = 1.383577518672690e+02;
        static const double a5 = -3.066479806614716e+01;
        static const double a6 = 2.506628277459239e+00;

        static const double b1 = -5.447609879822406e+01;
        static const double b2 = 1.615858368580409e+02;
        static const double b3 = -1.556989798598866e+02;
        static const double b4 = 6.680131188771972e+01;
        static const double b5 = -1.328068155288572e+01;

        static const double c1 = -7.784894002430293e-03;
        static const double c2 = -3.223964580411365e-01;
        static const double c3 = -2.400758277161838e+00;
        static const double c4 = -2.549732539343734e+00;
        static const double c5 = 4.374664141464968e+00;
        static const double c6 = 2.938163982698783e+00;

        static const double d1 = 7.784695709041462e-03;
        static const double d2 = 3.224671290700398e-01;
        static const double d3 = 2.445134137142996e+00;
        static const double d4 = 3.754408661907416e+00;

        constexpr double p_low = 0.02425;
        constexpr double p_high = 1.0 - p_low;

        if (p < p_low) {
            const double q = std::sqrt(-2.0 * std::log(p));
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
        if (p <= p_high) {
            const double q = p - 0.5;
            const double r = q * q;
            return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
                   (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        }
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    /// Linear interpolation quantile on a pre-sorted vector.
    double quantile_sorted(const std::vector<double>& sorted, double q) {
        if (sorted.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        q = std::min(1.0, std::max(0.0, q));
        if (sorted.size() == 1) {
            return sorted[0];
        }
        const double idx = q * static_cast<double>(sorted.size() - 1);
        const auto lo = static_cast<size_t>(std::floor(idx));
        const auto hi = static_cast<size_t>(std::ceil(idx));
        if (lo == hi) {
            return sorted[lo];
        }
        const double t = idx - static_cast<double>(lo);
        return sorted[lo] * (1.0 - t) + sorted[hi] * t;
    }

    /// Deterministic seed composition: combines base seed, guide index, and
    /// init index into a single 64-bit hash for reproducible per-(guide, init)
    /// random states.
    uint64_t compose_seed(const int seed, const arma::uword guide_idx, const arma::uword init_idx) {
        uint64_t out = static_cast<uint64_t>(static_cast<uint32_t>(seed));
        out ^= 0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(guide_idx) + 1ULL);
        out ^= 0xBF58476D1CE4E5B9ULL * (static_cast<uint64_t>(init_idx) + 1ULL);
        out ^= (out >> 30);
        out *= 0xBF58476D1CE4E5B9ULL;
        out ^= (out >> 27);
        out *= 0x94D049BB133111EBULL;
        out ^= (out >> 31);
        return out;
    }

    /// Draw a random quantile-based split and derive initial means, weight,
    /// and shared sigma for the 2-component EM.
    GuideInitState initialize_fit_state(
        const std::vector<double>& sorted,
        const double stdev,
        const double sigma_floor,
        std::mt19937_64& rng
    ) {
        GuideInitState init;
        const double min_v = sorted.front();
        const double max_v = sorted.back();
        const double range = std::max(max_v - min_v, 1e-6);

        std::uniform_real_distribution<double> q_dist(0.2, 0.8);
        const double split_q = q_dist(rng);
        const double split = quantile_sorted(sorted, split_q);

        double sum0 = 0.0;
        double sum1 = 0.0;
        size_t n0 = 0;
        size_t n1 = 0;
        for (const double v : sorted) {
            if (v <= split) {
                sum0 += v;
                ++n0;
            } else {
                sum1 += v;
                ++n1;
            }
        }

        if (n0 > 0 && n1 > 0) {
            init.mu0 = sum0 / static_cast<double>(n0);
            init.mu1 = sum1 / static_cast<double>(n1);
            init.w0 = static_cast<double>(n0) / static_cast<double>(sorted.size());
        } else {
            init.mu0 = quantile_sorted(sorted, 0.25);
            init.mu1 = quantile_sorted(sorted, 0.75);
            init.w0 = 0.5;
        }

        if (!(init.mu1 > init.mu0)) {
            init.mu0 = quantile_sorted(sorted, 0.20);
            init.mu1 = quantile_sorted(sorted, 0.80);
            if (!(init.mu1 > init.mu0)) {
                init.mu1 = init.mu0 + 0.1 * range;
            }
        }

        init.w0 = std::min(0.9, std::max(0.1, init.w0));
        init.sigma = std::max(sigma_floor, std::max(stdev, 0.2 * range));
        return init;
    }

    /// Run EM for a 2-component shared-variance Gaussian mixture on 1-D data.
    ///
    /// Component 0 (background) always has the lower mean in the output.
    /// Returns false on numerical failure (degenerate responsibilities,
    /// non-finite variance, etc.).
    bool run_em_shared_variance_2comp(
        const std::vector<double>& x,
        const GuideInitState& init,
        const actionet::GuideGMMFitParams& params,
        GuideFitState& out
    ) {
        const size_t n = x.size();
        if (n < 2) {
            return false;
        }

        const double sigma_floor = std::max(params.variance_floor, 1e-8);
        double w0 = std::min(1.0 - 1e-8, std::max(1e-8, init.w0));
        double w1 = 1.0 - w0;
        double mu0 = init.mu0;
        double mu1 = init.mu1;
        double sigma = std::max(sigma_floor, init.sigma);

        std::vector<double> r0(n, 0.5);
        double prev_ll = -std::numeric_limits<double>::infinity();
        bool converged = false;

        for (arma::uword iter = 0; iter < std::max<arma::uword>(params.max_iter, 1); ++iter) {
            const double s2 = sigma * sigma;
            if (!std::isfinite(s2) || s2 <= 0.0) {
                return false;
            }
            const double log_const = -0.5 * std::log(2.0 * arma::datum::pi * s2);

            double N0 = 0.0;
            double sum0 = 0.0;
            double sum1 = 0.0;
            double ll = 0.0;

            const double log_w0 = std::log(std::max(w0, 1e-16));
            const double log_w1 = std::log(std::max(w1, 1e-16));

            for (size_t i = 0; i < n; ++i) {
                const double xi = x[i];
                const double d0 = xi - mu0;
                const double d1 = xi - mu1;
                const double lp0 = log_w0 + log_const - 0.5 * (d0 * d0) / s2;
                const double lp1 = log_w1 + log_const - 0.5 * (d1 * d1) / s2;

                const double m = std::max(lp0, lp1);
                const double e0 = std::exp(lp0 - m);
                const double e1 = std::exp(lp1 - m);
                const double denom = e0 + e1;
                if (!(denom > 0.0) || !std::isfinite(denom)) {
                    return false;
                }

                const double ri0 = e0 / denom;
                r0[i] = ri0;
                N0 += ri0;
                sum0 += ri0 * xi;
                sum1 += (1.0 - ri0) * xi;
                ll += m + std::log(denom);
            }

            const double N1 = static_cast<double>(n) - N0;
            if (!(N0 > 1e-8 && N1 > 1e-8) || !std::isfinite(ll)) {
                return false;
            }

            mu0 = sum0 / N0;
            mu1 = sum1 / N1;
            if (!(std::isfinite(mu0) && std::isfinite(mu1))) {
                return false;
            }

            double sse = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double xi = x[i];
                const double d0 = xi - mu0;
                const double d1 = xi - mu1;
                const double ri0 = r0[i];
                sse += ri0 * d0 * d0 + (1.0 - ri0) * d1 * d1;
            }

            const double new_sigma = std::sqrt(std::max(sse / static_cast<double>(n), sigma_floor * sigma_floor));
            sigma = std::max(sigma_floor, new_sigma);
            w0 = std::min(1.0 - 1e-8, std::max(1e-8, N0 / static_cast<double>(n)));
            w1 = 1.0 - w0;

            if (iter > 0) {
                const double rel = std::abs(ll - prev_ll) / (std::abs(prev_ll) + 1e-12);
                if (std::isfinite(rel) && rel <= params.tol) {
                    prev_ll = ll;
                    converged = true;
                    break;
                }
            }
            prev_ll = ll;
        }

        if (!std::isfinite(prev_ll) || !std::isfinite(sigma)) {
            return false;
        }

        if (mu0 > mu1) {
            std::swap(mu0, mu1);
            std::swap(w0, w1);
        }

        out.w0 = w0;
        out.w1 = w1;
        out.mu0 = mu0;
        out.mu1 = mu1;
        out.sigma = sigma;
        out.ll = prev_ll;
        out.converged = converged;
        return true;
    }

    /// Zero-initialise all arrays of a GuideGMMFitResult for n_guides rows.
    /// Sigma and thresholds default to NaN; status defaults to INSUFFICIENT_POINTS.
    void initialize_fit_result(actionet::GuideGMMFitResult& out, const arma::uword n_guides) {
        out.weights = arma::mat(n_guides, 2, arma::fill::zeros);
        out.means = arma::mat(n_guides, 2, arma::fill::zeros);
        out.sigma = arma::vec(n_guides, arma::fill::value(std::numeric_limits<double>::quiet_NaN()));
        out.log_likelihood = arma::vec(n_guides, arma::fill::value(-std::numeric_limits<double>::infinity()));
        out.n_points = arma::uvec(n_guides, arma::fill::zeros);
        out.status = arma::ivec(
            n_guides,
            arma::fill::value(static_cast<int>(actionet::GUIDE_GMM_INSUFFICIENT_POINTS))
        );
    }

    /// Return true if guide row i has status OK, finite positive sigma, and
    /// finite positive weights.
    bool fit_row_is_valid(const actionet::GuideGMMFitResult& fits, const arma::uword i) {
        if (i >= fits.weights.n_rows) {
            return false;
        }
        if (fits.status(i) != actionet::GUIDE_GMM_OK) {
            return false;
        }
        if (!std::isfinite(fits.sigma(i)) || fits.sigma(i) <= 0.0) {
            return false;
        }
        const double mu0 = fits.means(i, 0);
        const double mu1 = fits.means(i, 1);
        const double w0 = fits.weights(i, 0);
        const double w1 = fits.weights(i, 1);
        return std::isfinite(mu0) && std::isfinite(mu1) && std::isfinite(w0) && std::isfinite(w1) &&
               w0 > 0.0 && w1 > 0.0;
    }

    /// Single-threaded: scan stored nonzeros and collect (row, col) pairs for
    /// cells exceeding the per-guide background or foreground threshold.
    /// When global_col_indices is non-null, columns are remapped to global
    /// indices (used for backed chunked paths).
    void collect_threshold_triplets(
        const arma::sp_mat& X,
        const arma::vec& background_thresholds,
        const arma::vec& foreground_thresholds,
        const arma::uvec* global_col_indices,
        std::vector<arma::uword>& bg_rows,
        std::vector<arma::uword>& bg_cols,
        std::vector<arma::uword>& fg_rows,
        std::vector<arma::uword>& fg_cols
    ) {
        const arma::uword local_cols = X.n_cols;
        for (arma::uword j = 0; j < local_cols; ++j) {
            const arma::uword global_j = global_col_indices ? (*global_col_indices)(j) : j;
            const double bg_t = background_thresholds(global_j);
            const double fg_t = foreground_thresholds(global_j);
            const bool use_bg = std::isfinite(bg_t);
            const bool use_fg = std::isfinite(fg_t);

            if (!(use_bg || use_fg)) {
                continue;
            }

            for (arma::sp_mat::const_col_iterator it = X.begin_col(j); it != X.end_col(j); ++it) {
                const double v = *it;
                if (use_bg && v > bg_t) {
                    bg_rows.push_back(it.row());
                    bg_cols.push_back(global_j);
                }
                if (use_fg && v > fg_t) {
                    fg_rows.push_back(it.row());
                    fg_cols.push_back(global_j);
                }
            }
        }
    }

    /// Per-thread triplet accumulator for the parallel threshold collection.
    struct TripletBuf {
        std::vector<arma::uword> rows;
        std::vector<arma::uword> cols;
    };

    /// OpenMP-parallel version of collect_threshold_triplets.
    ///
    /// Each thread accumulates into a private TripletBuf, then results are
    /// concatenated sequentially.  Falls back to the serial version for
    /// single-threaded or single-column cases.
    void collect_threshold_triplets_parallel(
        const arma::sp_mat& X,
        const arma::vec& background_thresholds,
        const arma::vec& foreground_thresholds,
        std::vector<arma::uword>& bg_rows,
        std::vector<arma::uword>& bg_cols,
        std::vector<arma::uword>& fg_rows,
        std::vector<arma::uword>& fg_cols,
        const unsigned int n_threads
    ) {
        const arma::uword n_cols = X.n_cols;
        const unsigned int threads_use = actionet::get_num_threads(
            static_cast<unsigned int>(std::max<arma::uword>(1, n_cols)),
            static_cast<unsigned int>(std::max(static_cast<int>(n_threads), 0))
        );

        if (threads_use <= 1 || n_cols <= 1) {
            collect_threshold_triplets(X, background_thresholds, foreground_thresholds,
                                       nullptr, bg_rows, bg_cols, fg_rows, fg_cols);
            return;
        }

        std::vector<TripletBuf> bg_bufs(threads_use);
        std::vector<TripletBuf> fg_bufs(threads_use);

        #pragma omp parallel num_threads(threads_use)
        {
            const int tid = omp_get_thread_num();
            TripletBuf& bg_local = bg_bufs[static_cast<size_t>(tid)];
            TripletBuf& fg_local = fg_bufs[static_cast<size_t>(tid)];

            #pragma omp for schedule(static)
            for (long long jj = 0; jj < static_cast<long long>(n_cols); ++jj) {
                const arma::uword j = static_cast<arma::uword>(jj);
                const double bg_t = background_thresholds(j);
                const double fg_t = foreground_thresholds(j);
                const bool use_bg = std::isfinite(bg_t);
                const bool use_fg = std::isfinite(fg_t);
                if (!(use_bg || use_fg)) {
                    continue;
                }
                for (arma::sp_mat::const_col_iterator it = X.begin_col(j); it != X.end_col(j); ++it) {
                    const double v = *it;
                    if (use_bg && v > bg_t) {
                        bg_local.rows.push_back(it.row());
                        bg_local.cols.push_back(j);
                    }
                    if (use_fg && v > fg_t) {
                        fg_local.rows.push_back(it.row());
                        fg_local.cols.push_back(j);
                    }
                }
            }
        }

        size_t bg_total = 0, fg_total = 0;
        for (unsigned int t = 0; t < threads_use; ++t) {
            bg_total += bg_bufs[t].rows.size();
            fg_total += fg_bufs[t].rows.size();
        }
        bg_rows.reserve(bg_total);
        bg_cols.reserve(bg_total);
        fg_rows.reserve(fg_total);
        fg_cols.reserve(fg_total);
        for (unsigned int t = 0; t < threads_use; ++t) {
            bg_rows.insert(bg_rows.end(), bg_bufs[t].rows.begin(), bg_bufs[t].rows.end());
            bg_cols.insert(bg_cols.end(), bg_bufs[t].cols.begin(), bg_bufs[t].cols.end());
            fg_rows.insert(fg_rows.end(), fg_bufs[t].rows.begin(), fg_bufs[t].rows.end());
            fg_cols.insert(fg_cols.end(), fg_bufs[t].cols.begin(), fg_bufs[t].cols.end());
        }
    }

    /// Build a binary sparse matrix from (row, col) triplets using counting
    /// sort into CSC order.
    ///
    /// Steps:
    ///   1. Count entries per column → prefix sums (col_counts).
    ///   2. Scatter rows into column-ordered positions (write_pos).
    ///   3. Sort row indices within each column.
    ///   4. Feed pre-ordered locations to arma::sp_mat with sort_locations=false.
    arma::sp_mat triplets_to_sparse(
        const std::vector<arma::uword>& rows,
        const std::vector<arma::uword>& cols,
        const arma::uword n_rows,
        const arma::uword n_cols
    ) {
        if (rows.empty()) {
            return arma::sp_mat(n_rows, n_cols);
        }

        if (rows.size() != cols.size()) {
            throw std::runtime_error("triplets_to_sparse: rows/cols size mismatch");
        }

        const size_t nnz = rows.size();

        std::vector<arma::uword> col_counts(static_cast<size_t>(n_cols) + 1, 0);
        for (size_t k = 0; k < nnz; ++k) {
            col_counts[static_cast<size_t>(cols[k]) + 1]++;
        }
        for (size_t c = 1; c <= static_cast<size_t>(n_cols); ++c) {
            col_counts[c] += col_counts[c - 1];
        }

        std::vector<arma::uword> sorted_rows(nnz);
        std::vector<arma::uword> write_pos(col_counts.begin(), col_counts.end());
        for (size_t k = 0; k < nnz; ++k) {
            const size_t c = static_cast<size_t>(cols[k]);
            sorted_rows[write_pos[c]++] = rows[k];
        }

        for (size_t c = 0; c < static_cast<size_t>(n_cols); ++c) {
            const size_t start = col_counts[c];
            const size_t end = col_counts[c + 1];
            std::sort(sorted_rows.begin() + static_cast<ptrdiff_t>(start),
                      sorted_rows.begin() + static_cast<ptrdiff_t>(end));
        }

        arma::umat locations(2, nnz);
        arma::vec values(nnz, arma::fill::ones);
        size_t out_idx = 0;
        for (size_t c = 0; c < static_cast<size_t>(n_cols); ++c) {
            const size_t start = col_counts[c];
            const size_t end = col_counts[c + 1];
            for (size_t k = start; k < end; ++k) {
                locations(0, out_idx) = sorted_rows[k];
                locations(1, out_idx) = static_cast<arma::uword>(c);
                ++out_idx;
            }
        }

        return arma::sp_mat(locations, values, n_rows, n_cols, false, true);
    }

    /// Fit guides from a backed HDF5 operator by reading guide columns in
    /// chunks of size backed_chunk_guides, fitting each chunk via the in-memory
    /// sparse path, and merging results.
    template <typename OperatorT>
    actionet::GuideGMMFitResult fit_guides_backed_impl(
        OperatorT& op,
        const actionet::GuideGMMFitParams& params
    ) {
        actionet::GuideGMMFitResult out;
        const arma::uword n_guides = op.cols();
        initialize_fit_result(out, n_guides);

        const arma::uword chunk_guides = std::max<arma::uword>(1, params.backed_chunk_guides);
        for (arma::uword start = 0; start < n_guides; start += chunk_guides) {
            const arma::uword end = std::min(start + chunk_guides, n_guides);
            const arma::uvec col_indices = arma::regspace<arma::uvec>(start, end - 1);
            arma::sp_mat chunk = op.takeColumnsSparse(col_indices);
            actionet::GuideGMMFitResult local =
                actionet::fitGuidesSharedVarianceGMM(chunk, params, start);

            for (arma::uword j = 0; j < local.weights.n_rows; ++j) {
                const arma::uword g = start + j;
                out.weights.row(g) = local.weights.row(j);
                out.means.row(g) = local.means.row(j);
                out.sigma(g) = local.sigma(j);
                out.log_likelihood(g) = local.log_likelihood(j);
                out.n_points(g) = local.n_points(j);
                out.status(g) = local.status(j);
            }
        }

        return out;
    }

    /// Apply per-guide thresholds to a backed HDF5 operator by reading guide
    /// columns in chunks, collecting triplets, and assembling the final sparse
    /// indicator matrices.
    template <typename OperatorT>
    arma::field<arma::sp_mat> apply_thresholds_backed_impl(
        OperatorT& op,
        const arma::vec& background_thresholds,
        const arma::vec& foreground_thresholds,
        const arma::uword chunk_guides
    ) {
        const arma::uword n_rows = op.rows();
        const arma::uword n_cols = op.cols();

        if (background_thresholds.n_elem != n_cols || foreground_thresholds.n_elem != n_cols) {
            throw std::invalid_argument("Threshold vector lengths must match number of guide columns");
        }

        std::vector<arma::uword> bg_rows;
        std::vector<arma::uword> bg_cols;
        std::vector<arma::uword> fg_rows;
        std::vector<arma::uword> fg_cols;

        const arma::uword guides_per_chunk = std::max<arma::uword>(1, chunk_guides);
        for (arma::uword start = 0; start < n_cols; start += guides_per_chunk) {
            const arma::uword end = std::min(start + guides_per_chunk, n_cols);
            const arma::uvec col_indices = arma::regspace<arma::uvec>(start, end - 1);
            arma::sp_mat chunk = op.takeColumnsSparse(col_indices);
            collect_threshold_triplets(
                chunk,
                background_thresholds,
                foreground_thresholds,
                &col_indices,
                bg_rows,
                bg_cols,
                fg_rows,
                fg_cols
            );
        }

        arma::field<arma::sp_mat> out(2);
        out(0) = triplets_to_sparse(bg_rows, bg_cols, n_rows, n_cols);
        out(1) = triplets_to_sparse(fg_rows, fg_cols, n_rows, n_cols);
        return out;
    }

    /// Allocate a GuideThresholdResult with NaN-filled vectors.
    actionet::GuideThresholdResult init_threshold_result(const arma::uword n_guides) {
        actionet::GuideThresholdResult out;
        out.background = arma::vec(n_guides, arma::fill::value(std::numeric_limits<double>::quiet_NaN()));
        out.foreground = arma::vec(n_guides, arma::fill::value(std::numeric_limits<double>::quiet_NaN()));
        return out;
    }

    /// Evaluate the Gaussian PDF at x given mean mu and standard deviation sigma.
    double normal_pdf(const double x, const double mu, const double sigma) {
        const double z = (x - mu) / sigma;
        return std::exp(-0.5 * z * z) / (sigma * kSqrt2Pi);
    }
} // namespace

namespace actionet {
    GuideGMMFitResult fitGuidesSharedVarianceGMM(
        const arma::sp_mat& X,
        const GuideGMMFitParams& params,
        const arma::uword guide_index_offset
    ) {
        const arma::uword n_guides = X.n_cols;
        GuideGMMFitResult out;
        initialize_fit_result(out, n_guides);

        const unsigned int threads_use = get_num_threads(
            static_cast<unsigned int>(std::max<arma::uword>(1, n_guides)),
            params.n_threads
        );

        #pragma omp parallel for schedule(static) num_threads(threads_use) if(threads_use > 1 && n_guides > 1)
        for (long long jj = 0; jj < static_cast<long long>(n_guides); ++jj) {
            const arma::uword j = static_cast<arma::uword>(jj);
            std::vector<double> x;
            x.reserve(static_cast<size_t>(X.col_ptrs[j + 1] - X.col_ptrs[j]));

            for (arma::sp_mat::const_col_iterator it = X.begin_col(j); it != X.end_col(j); ++it) {
                double v = *it;
                if (!std::isfinite(v)) {
                    continue;
                }
                if (v < params.min_counts) {
                    continue;
                }
                if (params.apply_log10p1) {
                    if (v <= -1.0) {
                        continue;
                    }
                    v = std::log1p(v) / kLog10;
                }
                x.push_back(v);
            }

            out.n_points(j) = static_cast<arma::uword>(x.size());
            if (x.size() < static_cast<size_t>(std::max<arma::uword>(params.min_points, 2))) {
                out.status(j) = GUIDE_GMM_INSUFFICIENT_POINTS;
                continue;
            }

            const double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
            double var = 0.0;
            for (const double v : x) {
                const double d = v - mean;
                var += d * d;
            }
            var /= static_cast<double>(x.size());
            const double stdev = std::sqrt(std::max(0.0, var));

            if (!(std::isfinite(stdev)) || stdev <= 1e-12) {
                out.status(j) = GUIDE_GMM_DEGENERATE;
                continue;
            }

            std::vector<double> sorted = x;
            std::sort(sorted.begin(), sorted.end());

            GuideFitState best;
            bool found = false;
            const arma::uword n_init = std::max<arma::uword>(params.n_init, 1);
            for (arma::uword init_idx = 0; init_idx < n_init; ++init_idx) {
                std::mt19937_64 rng(compose_seed(
                    params.seed,
                    guide_index_offset + j,
                    init_idx
                ));
                GuideInitState init = initialize_fit_state(
                    sorted,
                    stdev,
                    std::max(params.variance_floor, 1e-8),
                    rng
                );

                GuideFitState fit;
                const bool ok = run_em_shared_variance_2comp(x, init, params, fit);
                if (!ok) {
                    continue;
                }
                if (!found || fit.ll > best.ll) {
                    best = fit;
                    found = true;
                }
            }

            if (!found) {
                out.status(j) = GUIDE_GMM_NUMERICAL_FAILURE;
                continue;
            }

            out.weights(j, 0) = best.w0;
            out.weights(j, 1) = best.w1;
            out.means(j, 0) = best.mu0;
            out.means(j, 1) = best.mu1;
            out.sigma(j) = best.sigma;
            out.log_likelihood(j) = best.ll;
            out.status(j) = GUIDE_GMM_OK;
        }

        return out;
    }

    GuideGMMFitResult fitGuidesSharedVarianceGMM(
        BackedSparseMatrixOperator& op,
        const GuideGMMFitParams& params
    ) {
        return fit_guides_backed_impl(op, params);
    }

    GuideGMMFitResult fitGuidesSharedVarianceGMM(
        BackedDenseMatrixOperator& op,
        const GuideGMMFitParams& params
    ) {
        return fit_guides_backed_impl(op, params);
    }

    GuideThresholdResult deriveGuideThresholdsQuantile(
        const GuideGMMFitResult& fits,
        const double bg_quantile,
        const double fg_quantile
    ) {
        if (!(bg_quantile > 0.0 && bg_quantile < 1.0)) {
            throw std::invalid_argument("bg_quantile must be in (0, 1)");
        }
        if (!(fg_quantile > 0.0 && fg_quantile < 1.0)) {
            throw std::invalid_argument("fg_quantile must be in (0, 1)");
        }

        const arma::uword n_guides = fits.weights.n_rows;
        GuideThresholdResult out = init_threshold_result(n_guides);
        const double z_bg = inverse_normal_cdf(bg_quantile);
        const double z_fg = inverse_normal_cdf(fg_quantile);

        for (arma::uword i = 0; i < n_guides; ++i) {
            if (!fit_row_is_valid(fits, i)) {
                continue;
            }
            const double sigma = fits.sigma(i);
            out.background(i) = fits.means(i, 0) + sigma * z_bg;
            out.foreground(i) = fits.means(i, 1) + sigma * z_fg;
        }

        return out;
    }

    GuideThresholdResult deriveGuideThresholdsEqualDensity(
        const GuideGMMFitResult& fits
    ) {
        const arma::uword n_guides = fits.weights.n_rows;
        GuideThresholdResult out = init_threshold_result(n_guides);

        for (arma::uword i = 0; i < n_guides; ++i) {
            if (!fit_row_is_valid(fits, i)) {
                continue;
            }

            const double mu0 = fits.means(i, 0);
            const double mu1 = fits.means(i, 1);
            const double sigma = fits.sigma(i);
            const double w0 = fits.weights(i, 0);
            const double w1 = fits.weights(i, 1);

            double thr;
            if (mu1 <= mu0 + 1e-12) {
                thr = 0.5 * (mu0 + mu1);
            } else {
                const double delta = mu1 - mu0;
                thr = 0.5 * (mu0 + mu1) + (sigma * sigma / delta) * std::log(w0 / w1);
                if (!std::isfinite(thr)) {
                    thr = 0.5 * (mu0 + mu1);
                }
                thr = std::min(mu1, std::max(mu0, thr));
            }

            out.background(i) = thr;
            out.foreground(i) = thr;
        }

        return out;
    }

    GuideThresholdResult deriveGuideThresholdsValley(
        const GuideGMMFitResult& fits,
        const arma::uword grid_size
    ) {
        const arma::uword n_guides = fits.weights.n_rows;
        GuideThresholdResult out = init_threshold_result(n_guides);
        const arma::uword n_grid = std::max<arma::uword>(grid_size, 3);

        for (arma::uword i = 0; i < n_guides; ++i) {
            if (!fit_row_is_valid(fits, i)) {
                continue;
            }

            const double mu0 = fits.means(i, 0);
            const double mu1 = fits.means(i, 1);
            const double sigma = fits.sigma(i);
            const double w0 = fits.weights(i, 0);
            const double w1 = fits.weights(i, 1);

            double thr;
            if (mu1 <= mu0 + 1e-12) {
                thr = 0.5 * (mu0 + mu1);
            } else {
                const double step = (mu1 - mu0) / static_cast<double>(n_grid - 1);
                double best_x = mu0;
                double best_d = std::numeric_limits<double>::infinity();

                for (arma::uword g = 0; g < n_grid; ++g) {
                    const double x = mu0 + static_cast<double>(g) * step;
                    const double d = w0 * normal_pdf(x, mu0, sigma) + w1 * normal_pdf(x, mu1, sigma);
                    if (d < best_d) {
                        best_d = d;
                        best_x = x;
                    }
                }
                thr = best_x;
            }

            out.background(i) = thr;
            out.foreground(i) = thr;
        }

        return out;
    }

    GuideThresholdSweepResult sweepGuideThresholdsQuantile(
        const GuideGMMFitResult& fits,
        const arma::vec& bg_quantiles,
        const arma::vec& fg_quantiles
    ) {
        for (arma::uword i = 0; i < bg_quantiles.n_elem; ++i) {
            const double q = bg_quantiles(i);
            if (!(q > 0.0 && q < 1.0)) {
                throw std::invalid_argument("All bg_quantiles must be in (0, 1)");
            }
        }
        for (arma::uword i = 0; i < fg_quantiles.n_elem; ++i) {
            const double q = fg_quantiles(i);
            if (!(q > 0.0 && q < 1.0)) {
                throw std::invalid_argument("All fg_quantiles must be in (0, 1)");
            }
        }

        const arma::uword n_guides = fits.weights.n_rows;
        GuideThresholdSweepResult out;
        out.background = arma::mat(
            n_guides,
            bg_quantiles.n_elem,
            arma::fill::value(std::numeric_limits<double>::quiet_NaN())
        );
        out.foreground = arma::mat(
            n_guides,
            fg_quantiles.n_elem,
            arma::fill::value(std::numeric_limits<double>::quiet_NaN())
        );

        arma::vec z_bg(bg_quantiles.n_elem, arma::fill::zeros);
        arma::vec z_fg(fg_quantiles.n_elem, arma::fill::zeros);
        for (arma::uword i = 0; i < bg_quantiles.n_elem; ++i) {
            z_bg(i) = inverse_normal_cdf(bg_quantiles(i));
        }
        for (arma::uword i = 0; i < fg_quantiles.n_elem; ++i) {
            z_fg(i) = inverse_normal_cdf(fg_quantiles(i));
        }

        const unsigned int threads_use = get_num_threads(
            static_cast<unsigned int>(std::max<arma::uword>(1, n_guides)),
            0
        );
        #pragma omp parallel for schedule(static) num_threads(threads_use) if(threads_use > 1 && n_guides > 1)
        for (long long ii = 0; ii < static_cast<long long>(n_guides); ++ii) {
            const arma::uword i = static_cast<arma::uword>(ii);
            if (!fit_row_is_valid(fits, i)) {
                continue;
            }
            const double sigma = fits.sigma(i);
            const double mu0 = fits.means(i, 0);
            const double mu1 = fits.means(i, 1);
            for (arma::uword b = 0; b < bg_quantiles.n_elem; ++b) {
                out.background(i, b) = mu0 + sigma * z_bg(b);
            }
            for (arma::uword f = 0; f < fg_quantiles.n_elem; ++f) {
                out.foreground(i, f) = mu1 + sigma * z_fg(f);
            }
        }

        return out;
    }

    arma::field<arma::sp_mat> applyGuideThresholds(
        const arma::sp_mat& X,
        const arma::vec& background_thresholds,
        const arma::vec& foreground_thresholds,
        const int n_threads
    ) {
        if (background_thresholds.n_elem != X.n_cols || foreground_thresholds.n_elem != X.n_cols) {
            throw std::invalid_argument("Threshold vector lengths must match number of guide columns");
        }

        std::vector<arma::uword> bg_rows;
        std::vector<arma::uword> bg_cols;
        std::vector<arma::uword> fg_rows;
        std::vector<arma::uword> fg_cols;

        bg_rows.reserve(static_cast<size_t>(X.n_nonzero / 4 + 1));
        bg_cols.reserve(static_cast<size_t>(X.n_nonzero / 4 + 1));
        fg_rows.reserve(static_cast<size_t>(X.n_nonzero / 8 + 1));
        fg_cols.reserve(static_cast<size_t>(X.n_nonzero / 8 + 1));

        collect_threshold_triplets_parallel(
            X,
            background_thresholds,
            foreground_thresholds,
            bg_rows,
            bg_cols,
            fg_rows,
            fg_cols,
            static_cast<unsigned int>(std::max(n_threads, 0))
        );

        arma::field<arma::sp_mat> out(2);
        out(0) = triplets_to_sparse(bg_rows, bg_cols, X.n_rows, X.n_cols);
        out(1) = triplets_to_sparse(fg_rows, fg_cols, X.n_rows, X.n_cols);
        return out;
    }

    arma::field<arma::sp_mat> applyGuideThresholds(
        BackedSparseMatrixOperator& op,
        const arma::vec& background_thresholds,
        const arma::vec& foreground_thresholds,
        const arma::uword chunk_guides
    ) {
        return apply_thresholds_backed_impl(op, background_thresholds, foreground_thresholds, chunk_guides);
    }

    arma::field<arma::sp_mat> applyGuideThresholds(
        BackedDenseMatrixOperator& op,
        const arma::vec& background_thresholds,
        const arma::vec& foreground_thresholds,
        const arma::uword chunk_guides
    ) {
        return apply_thresholds_backed_impl(op, background_thresholds, foreground_thresholds, chunk_guides);
    }
} // namespace actionet
