## Primary
* Review defaults vs base.h and wrappers.
* Fix threading (RcppThread, mini_thread, inline, OpenMP)
* Condense redundant SVD functions
* Remove OpenMP dependency?
* Resolve missing eigen in n2 library
* Fix namespace usage in headers
* Fix duplicate PCG headers (run diff)?
* Rename .cc/.cpp and .h/.hpp
* Fix defaults (Source -> header)
* Replace and remove "ParallelFor" in build_network.

## Secondary
* Fix compile warnings for svd.cpp
* Compile to Windows x86
* Replace Armadillo with Eigen3?
* zscore is multithreaded???



## Done
* Automate link SuiteSparse on unix/macos
* Upgrade Armadillo
* Remove obselete igraph
* Remove obselete libleidenalg
* Compile to Unix x86
* Compile to Apple arm64 native
* Compile to Apple x86 via Rosetta2
* Remove Harmony
* Create "utils" module
* Duplicate JSD functions (wtf?) in "network_construction" and hnsw (space_js.h)
    * Moved to "network_construction_ext"
