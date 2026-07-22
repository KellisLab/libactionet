## libactionet tests
##
## Build: cmake -B build -DLIBACTIONET_BUILD_TESTS=ON && cmake --build build --target test_xicor_enrichment
## Run:   ./build/tests/test_xicor_enrichment

if (NOT TARGET actionet)
    message(FATAL_ERROR "libactionet tests must be built alongside the actionet target")
endif ()

add_executable(test_xicor_enrichment test_xicor_enrichment.cpp)
target_link_libraries(test_xicor_enrichment PRIVATE actionet)
target_include_directories(test_xicor_enrichment
    PRIVATE
        "${actionet_SOURCE_DIR}/include"
        "${actionet_SOURCE_DIR}/include/extern/armadillo"
        "${actionet_SOURCE_DIR}/include/extern")
