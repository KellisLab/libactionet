## Primary
* pybind 11 wrappers
* Automatically link R BLAS/LAPACK
* Eliminate external cholmod dependency
* Split colorspace from generate_layout.

## Secondary
* Cleanup unused code and comments
* Abstract `autocorrelation.cpp`
* Fix compile warnings for svd.cpp
* Compile to Windows x86
* Compile to macOS
  * Arm64 and x86
  * Figure out cblas dependency
* zscore is multithreaded???
* Remove OpenMP dependency?
  * May not even be there anymor
* Make interruptable from interface
  * Macros?
  
## Done
* Automate link SuiteSparse on unix/macos
* Upgrade Armadillo
* Remove obsolete igraph
* Remove obsolete libleidenalg
* Compile to Unix x86
* Compile to Apple arm64 native
* Compile to Apple x86 via Rosetta2
* Remove Harmony
* Create "utils" module
* Duplicate JSD functions (wtf?) in "network_construction" and hnsw (space_js.h)
    * Moved to "network_construction_ext"
* Replace and remove "ParallelFor" in build_network.
* Update StatsLib
* Fix duplicate PCG headers
* Fix threading (RcppThread, mini_thread, inline, OpenMP)
* Rename .cc/.cpp and .h/.hpp
* Fix defaults (Source -> header)
* Fix namespace usage in headers
* Consistent header guards
* Rcpp wrappers
* Update hnswlib
* Condense redundant SVD functions