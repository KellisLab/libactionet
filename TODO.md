## Primary
* pybind 11 wrappers
* Document C++ interface
* network_measures: Parallelize
* * network_measures: argument and return types (uvec/vec)
* Standardize 'norm_type' vs. 'norm_method'

## Secondary
* Fix compile warnings for svd.cpp
* Compile to Windows x86
* zscore is multithreaded???

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
  * <s>Now using mini_thread exclusively. Not great. Will probably change.</s>
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
  * Updated uwot
  * Modular graph optimization
* Stand-alone color mapping i.e. de novo colors.
* Consolidate and abstract network_diffusion
* Fix various normalizations 
  * Fast matrix normalization/manipulation interface for R/Python preprocessing
* Detect Rosetta build (x86) outside R "CMAKE_OSX_ARCHITECTURES
* Modular cmake