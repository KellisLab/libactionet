#include "annotation/marker_stats.hpp"
#include "action/aa.hpp"
#include "network/network_diffusion.hpp"
#include "io/backed_h5ad/backed_sparse_matrix_operator.hpp"
#include "io/backed_h5ad/backed_dense_matrix_operator.hpp"

double tol_approx = 1E-8;

namespace {
    // Shared standardization + diffusion logic for the VISION method.
    arma::mat vision_standardize_and_smooth_(
        arma::mat& stats,
        const arma::vec& mu,
        const arma::vec& sigma_sq,
        const arma::sp_mat& X,
        arma::sp_mat& G,
        int norm_method, double alpha, int max_it,
        bool approx, int thread_no) {

        arma::rowvec k1 = arma::rowvec(arma::sum(X));
        arma::rowvec k2 = arma::rowvec(arma::sum(arma::square(X)));

        arma::mat sampling_mu = mu * k1;
        arma::mat sampling_sigma_sq = sigma_sq * k2;
        arma::mat marker_stats = (stats - sampling_mu) / arma::sqrt(sampling_sigma_sq);
        marker_stats.replace(arma::datum::nan, 0);

        if (alpha != 0) {
            marker_stats = actionet::computeNetworkDiffusion(
                G, marker_stats, alpha, max_it, thread_no, approx,
                norm_method, tol_approx);
        }
        return marker_stats;
    }
} // namespace

namespace actionet {
    arma::mat computeFeatureStats(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& X, int norm_method,
                                  double alpha, int max_it, bool approx, int thread_no, bool ignore_baseline) {
        // S is cells x genes (obs x var, Plan 02 contract).
        // X is features(genes) x labels.
        // Result is cells x labels.
        arma::mat stats = arma::zeros(S.n_rows, X.n_cols);

        int n = G.n_rows;
        arma::sp_mat o = arma::sp_mat(arma::ones(n, 1));
        arma::vec pr = computeNetworkDiffusion(G, o, alpha, max_it, thread_no, approx).col(0);

        for (int i = 0; i < X.n_cols; i++) {
            // int marker_count = (int)sum(sum(spones(X.col(i))));
            int marker_count = arma::accu(arma::spones(X.col(i)));

            int idx = 0;
            arma::vec w = arma::zeros(marker_count);
            arma::vec baseline = arma::zeros(marker_count);
            arma::sp_mat raw_expression(S.n_rows, marker_count);
            for (arma::sp_mat::col_iterator it = X.begin_col(i);
                 it != X.end_col(i); ++it) {
                // S is cells x genes: gene it.row() is S.col(it.row())
                raw_expression.col(idx) = S.col(it.row());
                w(idx) = (*it);
                baseline(idx) = arma::accu(raw_expression.col(idx));
                idx++;
            }
            if (!ignore_baseline) {
                baseline = baseline / arma::sum(baseline);
                w = w % baseline;
            }
            w = w / std::sqrt(arma::sum(arma::square(w)));

            arma::mat imputed_expression = computeNetworkDiffusion(G, raw_expression, alpha, max_it, thread_no,
                                                                   approx, norm_method, tol_approx);

            for (int j = 0; j < imputed_expression.n_cols; j++) {
                arma::vec ppr = imputed_expression.col(j);
                arma::vec scores = arma::log2(ppr / pr);
                arma::uvec zero_idx = arma::find(ppr == 0);
                scores(zero_idx).zeros();
                scores = scores % ppr;

                stats.col(i) += w(j) * scores;
            }
        }

        return (stats);
    }

    arma::mat computeFeatureStatsVision(arma::sp_mat& G, arma::sp_mat& S, arma::sp_mat& X,
                                        int norm_method, double alpha, int max_it,
                                        bool approx, int thread_no) {
        if (S.n_cols != X.n_rows) {
            throw std::invalid_argument("Incompatible dimensions (S.n_cols != X.n_rows)");
        }
        if (S.n_rows != G.n_rows) {
            throw std::invalid_argument("Incompatible dimensions (S.n_rows != G.n_rows)");
        }

        arma::mat stats = arma::mat(S * X);

        arma::vec mu = arma::zeros(S.n_rows);
        arma::vec nnz = arma::zeros(S.n_rows);
        for (arma::sp_mat::const_iterator it = S.begin(); it != S.end(); ++it) {
            mu[it.row()] += (*it);
            nnz[it.row()]++;
        }
        mu /= S.n_cols;
        arma::vec p_nnz = nnz / S.n_cols;

        arma::vec sigma_sq = arma::zeros(S.n_rows);
        for (arma::sp_mat::const_iterator it = S.begin(); it != S.end(); ++it) {
            double delta = mu[it.row()] - (*it);
            sigma_sq[it.row()] += delta * delta;
        }
        sigma_sq += (S.n_cols * (1 - p_nnz)) % arma::square(mu);
        sigma_sq /= (S.n_cols - 1);

        return vision_standardize_and_smooth_(stats, mu, sigma_sq, X, G,
                                              norm_method, alpha, max_it, approx, thread_no);
    }

    // ------------------------------------------------------------------
    // Backed overloads of computeFeatureStatsVision
    // ------------------------------------------------------------------

    arma::mat computeFeatureStatsVision(BackedSparseMatrixOperator& op,
                                        arma::sp_mat& G, arma::sp_mat& X,
                                        int norm_method, double alpha,
                                        int max_it, bool approx,
                                        int thread_no) {
        const arma::uword n_obs = op.rows();
        const arma::uword n_var = op.cols();
        if (n_var != X.n_rows) {
            throw std::invalid_argument("Incompatible dimensions (op.cols != X.n_rows)");
        }
        if (n_obs != G.n_rows) {
            throw std::invalid_argument("Incompatible dimensions (op.rows != G.n_rows)");
        }

        // Pass 1: stats = S @ X via matmat
        arma::mat X_dense(X);
        arma::mat stats;
        op.matmat(X_dense, stats);

        // Pass 2: row_sum = S @ ones
        arma::vec ones_var(n_var, arma::fill::ones);
        arma::vec row_sum;
        op.matvec(ones_var, row_sum);

        // Pass 3: accumulate row_sum_sq and nnz via column-chunked extraction.
        // Three passes over NNZ is still much faster than the old Python path.
        arma::vec row_sum_sq(n_obs, arma::fill::zeros);
        arma::vec nnz_vec(n_obs, arma::fill::zeros);
        const arma::uword col_chunk = 1024;
        for (arma::uword c_start = 0; c_start < n_var; c_start += col_chunk) {
            const arma::uword c_end = std::min(n_var, c_start + col_chunk);
            const arma::uword n_cols = c_end - c_start;
            arma::uvec col_idx = arma::regspace<arma::uvec>(c_start, c_end - 1);
            arma::mat block = op.takeColumnsDense(col_idx);
            for (arma::uword r = 0; r < n_obs; ++r) {
                for (arma::uword c = 0; c < n_cols; ++c) {
                    double v = block(r, c);
                    if (v != 0.0) {
                        row_sum_sq(r) += v * v;
                        nnz_vec(r) += 1.0;
                    }
                }
            }
        }

        arma::vec mu = row_sum / static_cast<double>(n_var);
        arma::vec p_nnz = nnz_vec / static_cast<double>(n_var);

        // sigma_sq = (sum_sq_stored + (n - nnz)*mu^2 - 2*mu*sum_stored + nnz*mu^2) / (n-1)
        // But sum_stored = row_sum (since unstored = 0), so:
        // sigma_sq = (row_sum_sq - 2*mu*row_sum + n_var*mu^2) / (n_var - 1)
        arma::vec sigma_sq = (row_sum_sq - 2.0 * mu % row_sum +
                              static_cast<double>(n_var) * arma::square(mu)) /
                             static_cast<double>(n_var - 1);

        return vision_standardize_and_smooth_(stats, mu, sigma_sq, X, G,
                                              norm_method, alpha, max_it, approx, thread_no);
    }

    arma::mat computeFeatureStatsVision(BackedDenseMatrixOperator& op,
                                        arma::sp_mat& G, arma::sp_mat& X,
                                        int norm_method, double alpha,
                                        int max_it, bool approx,
                                        int thread_no) {
        const arma::uword n_obs = op.rows();
        const arma::uword n_var = op.cols();
        if (n_var != X.n_rows) {
            throw std::invalid_argument("Incompatible dimensions (op.cols != X.n_rows)");
        }
        if (n_obs != G.n_rows) {
            throw std::invalid_argument("Incompatible dimensions (op.rows != G.n_rows)");
        }

        // stats = S @ X via matmat
        arma::mat X_dense(X);
        arma::mat stats;
        op.matmat(X_dense, stats);

        // Per-cell row_sum via matvec.
        arma::vec ones_var(n_var, arma::fill::ones);
        arma::vec row_sum;
        op.matvec(ones_var, row_sum);

        // For dense-backed, accumulate row_sum_sq and nnz via slab reads.
        arma::vec row_sum_sq(n_obs, arma::fill::zeros);
        arma::vec nnz_vec(n_obs, arma::fill::zeros);
        arma::mat slab;
        const arma::uword chunk = op.effectiveChunkSize();
        for (arma::uword obs_start = 0; obs_start < n_obs; obs_start += chunk) {
            const arma::uword obs_end = std::min(n_obs, obs_start + chunk);
            const arma::uword obs_count = obs_end - obs_start;
            op.readSlab(obs_start, obs_count, slab);
            for (arma::uword r = 0; r < obs_count; ++r) {
                for (arma::uword c = 0; c < n_var; ++c) {
                    double v = slab(r, c);
                    if (v != 0.0) {
                        row_sum_sq(obs_start + r) += v * v;
                        nnz_vec(obs_start + r) += 1.0;
                    }
                }
            }
        }

        arma::vec mu = row_sum / static_cast<double>(n_var);

        arma::vec sigma_sq = (row_sum_sq - 2.0 * mu % row_sum +
                              static_cast<double>(n_var) * arma::square(mu)) /
                             static_cast<double>(n_var - 1);

        return vision_standardize_and_smooth_(stats, mu, sigma_sq, X, G,
                                              norm_method, alpha, max_it, approx, thread_no);
    }

    // ------------------------------------------------------------------
    // Pre-computed stats entry point
    // ------------------------------------------------------------------

    arma::mat computeFeatureStatsVisionFromStats(
        arma::sp_mat& G,
        arma::mat& stats,
        arma::vec& mu,
        arma::vec& sigma_sq,
        arma::sp_mat& X,
        int norm_method, double alpha, int max_it,
        bool approx, int thread_no) {

        if (stats.n_rows != G.n_rows) {
            throw std::invalid_argument("Incompatible dimensions (stats.n_rows != G.n_rows)");
        }
        if (stats.n_cols != X.n_cols) {
            throw std::invalid_argument("Incompatible dimensions (stats.n_cols != X.n_cols)");
        }
        if (mu.n_elem != stats.n_rows) {
            throw std::invalid_argument("Incompatible dimensions (mu.n_elem != stats.n_rows)");
        }
        if (sigma_sq.n_elem != stats.n_rows) {
            throw std::invalid_argument("Incompatible dimensions (sigma_sq.n_elem != stats.n_rows)");
        }

        return vision_standardize_and_smooth_(stats, mu, sigma_sq, X, G,
                                              norm_method, alpha, max_it, approx, thread_no);
    }

} // namespace actionet
