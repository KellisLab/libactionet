# Guide Calling API (C++ Core)

This document describes the fit-first guide-calling API implemented in:

- `include/tools/guide_calling.hpp`
- `src/tools/guide_calling.cpp`

All symbols are under namespace `actionet` and exported via `include/libactionet.hpp`.

## Model and Data Contract

- Matrix orientation is fixed to **`cells x guides`**.
- Each guide is fit independently with a **1D, 2-component Gaussian mixture** with **shared variance**.
- Fits use stored values passing `GuideGMMFitParams::min_counts`.
- If `apply_log10p1=true` (default), fitting is performed in transformed space (`log10(1 + count)`).
- Threshold derivation and sweep are performed from compact fit parameters, without refitting.

## Core Types

### `GuideGMMFitParams`

Configuration for EM fitting and backed operator chunking.

Fields:

- `min_points` (default `5`)
- `min_counts` (default `10.0`)
- `n_init` (default `8`)
- `max_iter` (default `200`)
- `tol` (default `1e-6`)
- `variance_floor` (default `1e-3`)
- `backed_chunk_guides` (default `256`)
- `apply_log10p1` (default `true`)
- `seed` (default `0`)
- `n_threads` (default `0`, auto)

### `GuideGMMStatus`

Per-guide fit status codes:

- `GUIDE_GMM_OK = 0`
- `GUIDE_GMM_INSUFFICIENT_POINTS = 1`
- `GUIDE_GMM_DEGENERATE = 2`
- `GUIDE_GMM_NUMERICAL_FAILURE = 3`

### `GuideGMMFitResult`

Compact fit payload:

- `weights`: `n_guides x 2`
- `means`: `n_guides x 2` (ordered by ascending mean)
- `sigma`: `n_guides` shared variance scale
- `log_likelihood`: `n_guides`
- `n_points`: `n_guides`
- `status`: `n_guides` (`GuideGMMStatus`)

### `GuideThresholdResult`

- `background`: `n_guides`
- `foreground`: `n_guides`

### `GuideThresholdSweepResult`

- `background`: `n_guides x n_bg_quantiles`
- `foreground`: `n_guides x n_fg_quantiles`

## Fitting APIs

```cpp
GuideGMMFitResult fitGuidesSharedVarianceGMM(
    const arma::sp_mat& X,
    const GuideGMMFitParams& params = {},
    arma::uword guide_index_offset = 0
);

GuideGMMFitResult fitGuidesSharedVarianceGMM(
    BackedSparseMatrixOperator& op,
    const GuideGMMFitParams& params = {}
);

GuideGMMFitResult fitGuidesSharedVarianceGMM(
    BackedDenseMatrixOperator& op,
    const GuideGMMFitParams& params = {}
);
```

Notes:

- OpenMP parallelism is over guides.
- Deterministic seed composition is `seed + guide_idx + init_idx` (mixed via stable hash composition).
- Backed overloads iterate over guide chunks (`backed_chunk_guides`) and reuse the sparse in-memory fit kernel per chunk.

## Threshold Derivation APIs (No Refit)

```cpp
GuideThresholdResult deriveGuideThresholdsQuantile(
    const GuideGMMFitResult& fits,
    double bg_quantile = 0.99,
    double fg_quantile = 0.01
);

GuideThresholdResult deriveGuideThresholdsEqualDensity(
    const GuideGMMFitResult& fits
);

GuideThresholdResult deriveGuideThresholdsValley(
    const GuideGMMFitResult& fits,
    arma::uword grid_size = 256
);

GuideThresholdSweepResult sweepGuideThresholdsQuantile(
    const GuideGMMFitResult& fits,
    const arma::vec& bg_quantiles,
    const arma::vec& fg_quantiles
);
```

Semantics:

- `quantile`: uses component quantiles from fitted means/sigma.
- `equal_density`: shared-variance intersection (background and foreground thresholds equal).
- `valley`: numeric density minimum between component means (background and foreground thresholds equal).
- `sweep`: computes threshold matrices for quantile grids.

Threshold outputs from these APIs are in transformed fit space.

## Threshold Application APIs

```cpp
arma::field<arma::sp_mat> applyGuideThresholds(
    const arma::sp_mat& X,
    const arma::vec& background_thresholds,
    const arma::vec& foreground_thresholds,
    int n_threads = 0
);

arma::field<arma::sp_mat> applyGuideThresholds(
    BackedSparseMatrixOperator& op,
    const arma::vec& background_thresholds,
    const arma::vec& foreground_thresholds,
    arma::uword chunk_guides = 256
);

arma::field<arma::sp_mat> applyGuideThresholds(
    BackedDenseMatrixOperator& op,
    const arma::vec& background_thresholds,
    const arma::vec& foreground_thresholds,
    arma::uword chunk_guides = 256
);
```

Return value is an `arma::field<arma::sp_mat>` of size 2:

- `out(0)`: background indicator sparse matrix (`count > background_threshold`)
- `out(1)`: foreground indicator sparse matrix (`count > foreground_threshold`)

Application compares thresholds against stored nonzeros only and does not materialize a dense `cells x guides` matrix.

## Python Bindings (actionet-python)

Bindings are exposed in `src/actionet/wp_tools.cpp`:

- `fit_guides_gmm_sparse`
- `fit_guides_gmm_backed_operator`
- `derive_guide_thresholds_quantile`
- `derive_guide_thresholds_equal_density`
- `derive_guide_thresholds_valley`
- `sweep_guide_thresholds_quantile`
- `apply_guide_thresholds_sparse`
- `apply_guide_thresholds_backed_operator`

Higher-level Python convenience wrappers live in `src/actionet/guide_calling.py`.
