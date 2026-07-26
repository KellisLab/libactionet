## Primary
* Legacy arma::field vs typed structs
  * dual return system across frontends
  * inconsistent return types
* Document C++ interface
* GPU backend (see plans/GPU_BACKEND_PLAN.md)
* Patch `actionet-r` package: remove `algorithm=2` (Feng) and `algorithm=3` (PRIMME) bindings from `wr_decomposition.cpp` (Roxygen entries, `C_runSVDSparse`/`C_runSVDDense` guards). `libactionet` no longer provides those SVD algorithms. Note: the `wrappers_r/` files inside this submodule are reference-only copies and were intentionally left untouched during the PRIMME/Feng deletion.

## Secondary
* Compile to Windows x86
* Add formal test infrastructure (test/ directory)
  * `tests/` scaffold added (2026-07-21) with golden regression tests for `xicor`, `XICOR`, and `assess_enrichment`; opt-in via `-DLIBACTIONET_BUILD_TESTS=ON`. Extend as needed.

## Done
* Fix long-standing correctness bugs in `tools/xicor` and `tools/enrichment` (2026-07-21):
  * `rank_vec(method=1)` tie handling — now returns 1-based max-tie rank matching `R::rank(., ties.method="max")`.
  * `xicor` Z-score `ind` — switched to 1-based (`regspace(1, n)`) to match `XICOR::xicor` asymptotic. Prior 0-based version returned a systematically wrong Z on every input.
  * `xicor` seed — now controls random tie-breaking on X via joint permutation + `stable_sort_index`. Prior joint shuffle before non-stable sort was effectively non-deterministic; documented in header.
  * `XICOR` matrix path — removed the `swap(X, Y)` + transpose optimization, which was silently wrong for the asymmetric xi. Added per-column rank precomputation, yielding an O(min(nX, nY))× speedup.
  * `assess_enrichment` — no longer mutates its `associations` argument (was silently binarized in-place); signature now `const arma::sp_mat&`.
  * `assess_enrichment` output — renamed `thresholds` to `peak_rank_idx` (the return value is a rank position, not a score threshold); comment updated in header and R/Python bindings.
  * `assess_enrichment` — hoisted inner-loop scratch matrices, dropping the 4× per-iteration allocations.
  * Golden tests added under `tests/test_xicor_enrichment.cpp`.
* Automate link SuiteSparse on unix/macos
* Upgrade Armadillo
* Remove obsolete and broken igraph.
  * Fails to compile on ARM and newer x86.
* Remove obsolete libleidenalg
  * Fails to compile because of above. Unnecessarily complicated dependency.
* Compile to Unix x86
* Compile to Apple arm64 native
* Compile to Apple x86 via Rosetta2
* Remove Harmony
* Create "utils" module
* Duplicate JSD functions (wtf?) in "network_construction" and hnsw (space_js.h)
    * Moved to "network_construction_ext"
* Replace and remove inlined "ParallelFor" in build_network.
* Update StatsLib
* Fix duplicate PCG headers
* Fix threading (RcppThread, mini_thread, inline, OpenMP)
  * OpenMP is now the threading model.
* Rename .cc/.cpp and .h/.hpp
* Fix defaults (Source -> header)
* Fix namespace usage in headers
* Consistent header guards
* Rcpp wrappers
  * Condensed and modularized.
* Update hnswlib
  * Fixed redefinition bug
* Condense redundant SVD functions
  * New interface for SVD
* Automatically link R BLAS/LAPACK
* Removed packaged cblas.h
  * Automatically finds headers used by installed BLAS/LAPACK
* Completely automate build system.
  * Fully portable and cross-platform
  * Automatically detects if built by R or stand-alone
  * Uses R build system in R package mode (99% CRAN compliant)
* Select armadillo based on build mode
  * Compiles with RcppArmadillo in R build mode and packaged arma in stand-alone.
* Automatically find BLAS/LAPACK
  * System BLAS (Linux), OpenBLAS, Accelerate, MKL supported.
  * Prefers BLAS used by R if in R build mode.
* Restructured R wrappers 
  * Added config for wrappers in R build mode
  * Conversion from arma::vec to Rcpp::NumricVector now automatic
* Template abstraction of functions accepting both dense and sparse matrix input.
* Add hnsw parameters to interfaces
* Completely remade UMAP/uwot integration.
  * Updated uwot to 0.2.4
  * Modular graph optimization
* Stand-alone color mapping i.e. de novo colors.
* Consolidate and abstract network_diffusion
* Fix various normalizations 
  * Fast matrix normalization/manipulation interface for R/Python preprocessing
* Detect Rosetta build (x86) outside R "CMAKE_OSX_ARCHITECTURES
* Modular cmake
* Standardize 'norm_type' vs. 'norm_method'
* pybind 11 wrappers
* Consolidate svdIRLB()
* Backed SVD
  * Operator-based Halko and IRLB paths
  * HDF5-backed sparse and dense matrix operators
  * Chunked I/O with configurable byte budget
* Backed network construction
* Backed specificity and annotation
  * Parallel backed specificity with thread hint
  * Overloads for BackedSparseMatrixOperator and BackedDenseMatrixOperator
  * Precomputed marker stats path for faster Python workflows
* Speed up AA and SPA
  * Stacked C/H matrices during ACTION decomposition to reduce memory footprint
  * Release memory and move archetype matrices in runACTION
  * Conditionally parallel simplex regression
  * Optimized archetype merging
* Reduce memory footprint
  * Eliminated unnecessary matrix copies across decomposition and annotation
  * Chebyshev diffusion with pointer rotation to avoid matrix moves
  * Removed excess graph copies in diffusion
  * Memory corruption fixes
* Optimized kNN implementation
  * New graph builder with improved k*nn
* Batch orthogonalization (operator-backed variant)
* C++ified IRLB to fix memory leaks
* Optimized pseudobulk/aggregation functions
* Optimized specificity (analytical min-shift and single-pass label scatter)
* Approximate fastlog transform for backed operators
