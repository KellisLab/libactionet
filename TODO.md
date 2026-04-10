## Primary
* Legacy arma::field vs typed structs
  * dual return system across frontends
  * inconsistent return types
* Document C++ interface
* GPU backend (see context/GPU_BACKEND_PLAN.md)

## Secondary
* Compile to Windows x86
* Add formal test infrastructure (test/ directory)

## Done
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
  * Operator-based PRIMME, Halko, Feng, and IRLB paths
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
