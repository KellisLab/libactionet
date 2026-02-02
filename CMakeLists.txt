cmake_minimum_required(VERSION 3.19)

project(actionet)
## Include third-party cmake modules
list(APPEND CMAKE_MODULE_PATH "${actionet_SOURCE_DIR}/cmake")
include(ConfigureApple)
include(ConfigureBLAS)
include(ConfigureR)
include(ConfigureOpenMP)
include(ConfigureCHOLMOD)
include(BuildTest)

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
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(actionet STATIC ${SOURCES})

############# TEST MODES ### Must be set before any other macros
#TEST_BUILD_R(actionet) ## Uncomment to test R build mode.

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

## Configure CHOLMOD library
CONFIGURE_CHOLMOD(actionet)

## Build bundled PRIMME library (skip for R builds - R doesn't support >2^31 element matrices)
if (NOT LIBACTIONET_BUILD_R)
    add_subdirectory(src/extern/primme)
    target_link_libraries(actionet PUBLIC primme_bundled)
    message(STATUS "PRIMME_SVDS enabled for large matrix support")
else()
    message(STATUS "PRIMME_SVDS disabled for R build (R limited to 32-bit matrix indices)")
endif()

## Add included headers
target_include_directories(
        actionet
        PRIVATE "${actionet_SOURCE_DIR}/include/extern/gcem"
        PRIVATE "${actionet_SOURCE_DIR}/include/extern"
        PRIVATE "${actionet_SOURCE_DIR}/include"
        PRIVATE "${actionet_SOURCE_DIR}/include/extern/primme"
)

## Check for OpenMP support
CONFIGURE_OPENMP(actionet)

# Set OpenMP thread count in Armadillo configuration
if (OpenMP_FOUND)
    target_compile_definitions(actionet PUBLIC ARMA_OPENMP_THREADS=${NPROC})
    message(STATUS "OpenMP enabled with ${NPROC} threads")
else ()
    message(NOTICE "Building without OpenMP")
endif ()

## Additional (optional) compiler options
target_compile_options(actionet PUBLIC -Wno-psabi -Wno-deprecated-declarations)# -ferror-limit=10)#-fmax-errors=10)
#add_compile_options(-w)

## Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Build diagnostics (uncomment for debugging)
#get_directory_property(CPP_Defs COMPILE_DEFINITIONS)
#message(NOTICE "Debug info:")
#message(STATUS "COMPILE_DEFINITIONS = ${CPP_Defs}")
#message(STATUS "LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}")
#message(STATUS "LAPACK_LINKER_FLAGS: ${LAPACK_LINKER_FLAGS}")
#message(STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
#message(STATUS "BLAS_LINKER_FLAGS: ${BLAS_LINKER_FLAGS}")
#message(STATUS "OpenMP_C_INCLUDE_DIRS: ${OpenMP_C_INCLUDE_DIRS}")
#message(STATUS "OpenMP_CXX_INCLUDE_DIRS: ${OpenMP_CXX_INCLUDE_DIRS}")
