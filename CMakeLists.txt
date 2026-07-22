cmake_minimum_required(VERSION 3.19)

project(actionet)
## Include third-party cmake modules
list(APPEND CMAKE_MODULE_PATH "${actionet_SOURCE_DIR}/cmake")
include(ConfigureApple)
include(ConfigureBLAS)
include(ConfigureR)
include(ConfigureOpenMP)

find_package(HDF5 REQUIRED COMPONENTS C)

# SKETCHY DEFENSIVE FIX. BUG MAY NOT EVEN EXIST.
# Use the modern imported target if available (avoids linking both system
# libhdf5_serial.so and a separately installed libhdf5.so simultaneously,
# which causes an HDF5 version mismatch abort at runtime when h5py is also
# loaded in the same process). The imported target resolves to a single lib.
if(TARGET hdf5::hdf5-shared)
    set(_HDF5_TARGET hdf5::hdf5-shared)
elseif(TARGET HDF5::HDF5)
    set(_HDF5_TARGET HDF5::HDF5)
else()
    set(_HDF5_TARGET "${HDF5_LIBRARIES}")
endif()

############# COMPILER OPTIONS
# Library uses C++17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

## Detect usable logical CPUs
cmake_host_system_information(RESULT NPROC QUERY NUMBER_OF_LOGICAL_CORES)
message(STATUS "Detecting CPU cores: ${NPROC}")
if (NPROC GREATER_EQUAL 6)
    math(EXPR NPROC "${NPROC} - 2" OUTPUT_FORMAT DECIMAL)
    message(STATUS "Using ${NPROC} cores")
endif ()

############# INITIALIZE LIBRARY
## Add all C/C++ code to sources for compilation
file(GLOB_RECURSE SOURCES_ACTION CONFIGURE_DEPENDS src/action/*.cpp)
file(GLOB_RECURSE SOURCES_ANNOTATION CONFIGURE_DEPENDS src/annotation/*.cpp)
file(GLOB_RECURSE SOURCES_DECOMPOSITION CONFIGURE_DEPENDS src/decomposition/*.cpp)
file(GLOB_RECURSE SOURCES_NETWORK CONFIGURE_DEPENDS src/network/*.cpp)
file(GLOB_RECURSE SOURCES_IO CONFIGURE_DEPENDS src/io/*.cpp)
file(GLOB_RECURSE SOURCES_TOOLS CONFIGURE_DEPENDS src/tools/*.cpp)
file(GLOB_RECURSE SOURCES_UTILS_INT CONFIGURE_DEPENDS src/utils_internal/*.cpp)
file(GLOB_RECURSE SOURCES_VISUALIZATION CONFIGURE_DEPENDS src/visualization/*.cpp)

add_library(actionet STATIC
        ${SOURCES_ACTION}
        ${SOURCES_ANNOTATION}
        ${SOURCES_DECOMPOSITION}
        ${SOURCES_NETWORK}
        ${SOURCES_IO}
        ${SOURCES_TOOLS}
        ${SOURCES_UTILS_INT}
        ${SOURCES_VISUALIZATION}
        src/extern/colorspace.cpp
)

############# FUNCTIONS AND MACROS
## Handle R build configuration FIRST (before CONFIGURE_APPLE) to extract architecture
if (LIBACTIONET_BUILD_R)
    # Extract R's target architecture before calling CONFIGURE_APPLE
    CONFIGURE_R_ARCHITECTURE()
endif()

## Set Apple CPU-specific compiler flags
if (APPLE)
    CONFIGURE_APPLE(actionet)
endif ()

############ BUILD MODES
if (LIBACTIONET_BUILD_R) ## OS-agnostic R library configurations
    CONFIGURE_R()
elseif (UNIX) # Build for Unix-alike configurations (including macOS)
    ## Set BLAS and LAPACK libraries
    CONFIGURE_BLAS_VENDOR_DEFAULT()
    CONFIGURE_BLAS(actionet)

    # Use packaged Armadillo
    target_include_directories(
            actionet
            PRIVATE "${actionet_SOURCE_DIR}/include/extern/armadillo"
    )
endif ()

############ CONFIGURE BUILD
# Link BLAS and LAPACK - properly handle both library paths and linker flags
if (BLAS_FOUND AND LAPACK_FOUND)
    # Modern CMake: use targets if available
    if (TARGET BLAS::BLAS)
        target_link_libraries(actionet PUBLIC BLAS::BLAS)
    else()
        target_link_libraries(actionet PUBLIC ${BLAS_LIBRARIES})
    endif()

    if (TARGET LAPACK::LAPACK)
        target_link_libraries(actionet PUBLIC LAPACK::LAPACK)
    else()
        target_link_libraries(actionet PUBLIC ${LAPACK_LIBRARIES})
    endif()

    # Handle linker flags separately (if provided)
    if (BLAS_LINKER_FLAGS)
        target_link_options(actionet PUBLIC ${BLAS_LINKER_FLAGS})
    endif ()
    if (LAPACK_LINKER_FLAGS)
        target_link_options(actionet PUBLIC ${LAPACK_LINKER_FLAGS})
    endif ()

    message(STATUS "Linked BLAS libraries: ${BLAS_LIBRARIES}")
    message(STATUS "Linked LAPACK libraries: ${LAPACK_LIBRARIES}")
else()
    message(WARNING "BLAS or LAPACK not found - build may fail")
endif()


target_include_directories(actionet PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(actionet PUBLIC ${_HDF5_TARGET})

## Add included headers
target_include_directories(
        actionet
        PRIVATE "${actionet_SOURCE_DIR}/include/extern/gcem"
        PRIVATE "${actionet_SOURCE_DIR}/include/extern"
        PRIVATE "${actionet_SOURCE_DIR}/include"
)


## Check for OpenMP support
CONFIGURE_OPENMP(actionet)

if (NOT OpenMP_FOUND)
    message(FATAL_ERROR
        "OpenMP is required but was not found.\n"
        "  Linux : install libgomp (e.g. apt install libgomp1, yum install libgomp)\n"
        "  macOS : brew install libomp\n"
        "  Conda : conda install -c conda-forge compilers\n"
        "To explicitly select a runtime, set LIBACTIONET_OPENMP_RUNTIME (GNU|INTEL|LLVM).")
endif ()

# Set OpenMP thread count in Armadillo configuration
target_compile_definitions(actionet PUBLIC ARMA_OPENMP_THREADS=${NPROC})
message(STATUS "OpenMP enabled with ${NPROC} threads")

## Additional (optional) compiler options
target_compile_options(actionet PRIVATE -Wno-psabi -Wno-deprecated-declarations)# -ferror-limit=10)#-fmax-errors=10)
#add_compile_options(-w)

## Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

############ TESTS
option(LIBACTIONET_BUILD_TESTS "Build libactionet regression tests" OFF)
if (LIBACTIONET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()
