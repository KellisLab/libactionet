#ifndef ACTIONET_GUIDE_CALLING_HPP
#define ACTIONET_GUIDE_CALLING_HPP

#include "libactionet_config.hpp"

namespace actionet {

    class BackedSparseMatrixOperator;
    class BackedDenseMatrixOperator;

    /// Parameters controlling per-guide 2-component shared-variance GMM fitting.
    struct GuideGMMFitParams {
        arma::uword min_points = 5;
        double min_counts = 10.0;
        arma::uword n_init = 8;
        arma::uword max_iter = 200;
        double tol = 1e-6;
        double variance_floor = 1e-3; // Floor on shared sigma in transformed space.
        arma::uword backed_chunk_guides = 256;
        bool apply_log10p1 = true;
        int seed = 0;
        int n_threads = 0;
    };

    /// Status codes emitted per guide after GMM fitting.
    enum GuideGMMStatus : int {
        GUIDE_GMM_OK = 0,
        GUIDE_GMM_INSUFFICIENT_POINTS = 1,
        GUIDE_GMM_DEGENERATE = 2,
        GUIDE_GMM_NUMERICAL_FAILURE = 3
    };

    /// Per-guide fit results (one row per guide).
    struct GuideGMMFitResult {
        arma::mat weights;        // n_guides x 2 (ordered by mean ascending)
        arma::mat means;          // n_guides x 2
        arma::vec sigma;          // n_guides shared sigma
        arma::vec log_likelihood; // n_guides
        arma::uvec n_points;      // n_guides
        arma::ivec status;        // n_guides, values from GuideGMMStatus
    };

    /// Per-guide pair of (background, foreground) thresholds in transformed space.
    struct GuideThresholdResult {
        arma::vec background; // n_guides
        arma::vec foreground; // n_guides
    };

    /// Per-guide threshold matrices produced by a quantile sweep.
    struct GuideThresholdSweepResult {
        arma::mat background; // n_guides x n_bg_quantiles
        arma::mat foreground; // n_guides x n_fg_quantiles
    };

    /// @brief Fit per-guide 2-component shared-variance 1D GMMs on sparse counts.
    ///
    /// Matrix orientation is fixed to cells x guides (rows x columns).
    /// Fits are performed independently per guide over stored values passing
    /// @p params.min_counts.
    ///
    /// @param X Sparse count matrix (cells x guides).
    /// @param params Fit configuration.
    /// @param guide_index_offset Global guide offset for deterministic seed composition.
    /// @return Compact per-guide fit arrays.
    GuideGMMFitResult fitGuidesSharedVarianceGMM(
        const arma::sp_mat& X,
        const GuideGMMFitParams& params = {},
        arma::uword guide_index_offset = 0
    );

    /// @brief Backed sparse overload (chunked by guide columns).
    GuideGMMFitResult fitGuidesSharedVarianceGMM(
        BackedSparseMatrixOperator& op,
        const GuideGMMFitParams& params = {}
    );

    /// @brief Backed dense overload (chunked by guide columns, sparse extraction).
    GuideGMMFitResult fitGuidesSharedVarianceGMM(
        BackedDenseMatrixOperator& op,
        const GuideGMMFitParams& params = {}
    );

    /// @brief Derive per-guide thresholds from existing fits using component quantiles.
    ///
    /// background = mu_bg + sigma * qnorm(bg_quantile)
    /// foreground = mu_fg + sigma * qnorm(fg_quantile)
    ///
    /// Returned thresholds are in transformed fit space.
    GuideThresholdResult deriveGuideThresholdsQuantile(
        const GuideGMMFitResult& fits,
        double bg_quantile = 0.99,
        double fg_quantile = 0.01
    );

    /// @brief Derive equal-density intersection threshold(s) from existing fits.
    ///
    /// With shared variance this yields a closed-form crossing. Background and
    /// foreground thresholds are set to the same value.
    /// Returned thresholds are in transformed fit space.
    GuideThresholdResult deriveGuideThresholdsEqualDensity(
        const GuideGMMFitResult& fits
    );

    /// @brief Derive numeric valley thresholds by grid-searching mixture density
    ///        between component means.
    ///
    /// Background and foreground thresholds are set to the same valley point.
    /// Returned thresholds are in transformed fit space.
    GuideThresholdResult deriveGuideThresholdsValley(
        const GuideGMMFitResult& fits,
        arma::uword grid_size = 256
    );

    /// @brief Sweep quantile thresholds across parameter grids (no refit).
    ///
    /// Returned thresholds are in transformed fit space.
    GuideThresholdSweepResult sweepGuideThresholdsQuantile(
        const GuideGMMFitResult& fits,
        const arma::vec& bg_quantiles,
        const arma::vec& fg_quantiles
    );

    /// @brief Apply per-guide thresholds to sparse counts using stored nonzeros only.
    ///
    /// Returns a field of size 2:
    ///   out(0): background indicator matrix (cells x guides, 1 where count > bg threshold)
    ///   out(1): foreground indicator matrix (cells x guides, 1 where count > fg threshold)
    arma::field<arma::sp_mat> applyGuideThresholds(
        const arma::sp_mat& X,
        const arma::vec& background_thresholds,
        const arma::vec& foreground_thresholds,
        int n_threads = 0
    );

    /// @brief Apply per-guide thresholds to backed sparse counts in guide chunks.
    arma::field<arma::sp_mat> applyGuideThresholds(
        BackedSparseMatrixOperator& op,
        const arma::vec& background_thresholds,
        const arma::vec& foreground_thresholds,
        arma::uword chunk_guides = 256
    );

    /// @brief Apply per-guide thresholds to backed dense counts in guide chunks.
    arma::field<arma::sp_mat> applyGuideThresholds(
        BackedDenseMatrixOperator& op,
        const arma::vec& background_thresholds,
        const arma::vec& foreground_thresholds,
        arma::uword chunk_guides = 256
    );

} // namespace actionet

#endif // ACTIONET_GUIDE_CALLING_HPP
