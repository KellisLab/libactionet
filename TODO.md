## Primary
* Condense redundant SVD functions
* Remove OpenMP dependency?
* Resolve missing eigen in n2 library
* Fix namespace usage in headers
* Rename .cc/.cpp and .h/.hpp
* Fix defaults (Source -> header)
* Update hnswlib


## Secondary
* Fix compile warnings for svd.cpp
* Compile to Windows x86
* Replace Armadillo with Eigen3?
* zscore is multithreaded???
* Consistent header guards


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
* Fix duplicate PCG headers (run diff)?
* Fix threading (RcppThread, mini_thread, inline, OpenMP)