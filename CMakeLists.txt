cmake_minimum_required(VERSION 3.18)

# Specify macos arm64 compilers
#set(USE_GCC "/opt/homebrew/Cellar/gcc/14.1.0_1/bin")

## Specify macos x86 Rosetta2 compilers
# set(USE_GCC "/usr/local/Cellar/gcc/14.1.0_1/bin/")
# string(APPEND CMAKE_CXX_FLAGS_RELEASE " -arch x86_64")
# string(APPEND CMAKE_CC_FLAGS_RELEASE " -arch x86_64")

# Set compiler
#set(CMAKE_C_COMPILER "${USE_GCC}/gcc-14")
#set(CMAKE_CXX_COMPILER "${USE_GCC}/g++-14")

# Test Apple cross-compilation
#set(CMAKE_OSX_ARCHITECTURES x86_64)

project(actionet)
## Include third-part cmake modules
list(APPEND CMAKE_MODULE_PATH "${actionet_SOURCE_DIR}/cmake")
include(ConfigureBLAS)
############# COMPILER OPTIONS
# C++11 is not supported, as the code uses C++17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

## Suppress warning
add_compile_options(-w)
#add_compile_options(-O3)

## Detect usable logical CPUs
cmake_host_system_information(RESULT NPROC QUERY NUMBER_OF_LOGICAL_CORES)
message(STATUS "Detecting CPU cores: ${NPROC}")
if (NPROC GREATER_EQUAL 6)
    math(EXPR NPROC "${NPROC} - 2" OUTPUT_FORMAT DECIMAL)
    message(STATUS "Using ${NPROC} cores")
endif ()

############# FUNCTIONS AND MACROS
## Set BLAS vendor default
set(BLA_VENDOR All)

## Set Apple CPU-specific compiler flags
macro(CONFIGURE_APPLE arch_apple)
    message(NOTICE "Configuring cmake build for macOS")
    message(STATUS "Building for ${arch_apple}")

    if (${arch_apple} STREQUAL "arm64")
        add_compile_options(-flax-vector-conversions)
    endif ()
endmacro()

############# INITIALIZE BUILD
## Add all C/C++ code to sources for compilation
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(actionet STATIC ${SOURCES})

## Find and link CHOLMOD
find_package(SuiteSparse REQUIRED COMPONENTS CHOLMOD)
target_link_libraries(actionet PUBLIC SuiteSparse::CHOLMOD)

############# TEST MODES
## Test build for R interface
set(TEST_BUILD_R 0)
if (TEST_BUILD_R)
    set(LIBACTIONET_BUILD_R 1)
    execute_process(
            COMMAND sh -c "R RHOME"
            OUTPUT_VARIABLE R_HOME
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    ## Test Rcpp wrappers
    target_sources(actionet
            PRIVATE r_wrapper.cpp
            PRIVATE r_wrapper_experimental.cpp
    )
endif ()

############ BUILD MODES
if (LIBACTIONET_BUILD_R) ## OS-agnostic R library configurations

    if (APPLE) ## Detect R architecture on Apple systems
        execute_process(
                COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(R.version[[\"arch\"]])'"
                OUTPUT_VARIABLE arch
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        CONFIGURE_APPLE(${arch})
    endif ()

    message(NOTICE "Configuring for R library")
    if (NOT DEFINED R_HOME)
        message(FATAL_ERROR "R_HOME not defined")
    else ()
        message(NOTICE "Using R installation in ${R_HOME}")
    endif ()
    add_compile_definitions(LIBACTIONET_BUILD_R) # Set R build mode in config header

    ## Find R headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/R CMD config --cppflags | sed s/-I//g"
            OUTPUT_VARIABLE r_RInclude
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "R library headers: ${r_RInclude}")

    ## Find R CPPFLAGS
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/R CMD config CPPFLAGS | sed s/-I//g"
            OUTPUT_VARIABLE r_CppFlags
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "R CPPFLAGS: ${r_CppFlags}")

    ## Find Rcpp headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(system.file(\"include\", package=\"Rcpp\"))'"
            OUTPUT_VARIABLE r_RcppInclude
    )
    message(STATUS "Rcpp headers: ${r_RcppInclude}")

    ## Find RcppArmadillo headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(system.file(\"include\", package=\"RcppArmadillo\"))'"
            OUTPUT_VARIABLE r_RcppArmaInclude
    )
    message(STATUS "RcppArmadillo headers: ${r_RcppArmaInclude}")

    ## Set BLAS and LAPACK libraries
    if ((DEFINED BLA_VENDOR) AND (NOT ${BLA_VENDOR} STREQUAL "All")) ## User provided
        CONFIGURE_BLAS()
    else () ## Get BLAS/LAPACK from R
        message(NOTICE "Using BLAS/LAPACK from R")

        ## Find R BLAS_LIBS
        execute_process(
                COMMAND bash -c "${R_HOME}/bin/R CMD config BLAS_LIBS | sed s/-I//g"
                OUTPUT_VARIABLE BLAS_LIBRARIES
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        message(STATUS "R BLAS_LIBS: ${BLAS_LIBRARIES}")
        ## Find R LAPACK_LIBS
        execute_process(
                COMMAND bash -c "${R_HOME}/bin/R CMD config LAPACK_LIBS | sed s/-I//g"
                OUTPUT_VARIABLE LAPACK_LIBRARIES
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        message(STATUS "R LAPACK_LIBS: ${LAPACK_LIBRARIES}")

        CONFIGURE_BLAS_DEPENDS()
    endif ()

    target_include_directories(actionet
            PRIVATE "${r_RInclude}"
            PRIVATE "${r_CppFlags}"
            PRIVATE "${r_RcppInclude}"
            PRIVATE "${r_RcppArmaInclude}"
    )
elseif (UNIX) # Build for Unix-alike configurations (including macOS)

    if (APPLE) ## Detect CPU architecture for Apple systems
        execute_process(
                COMMAND bash -c "uname -m"
                OUTPUT_VARIABLE arch
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        CONFIGURE_APPLE(${arch})
    endif ()

    ## Set BLAS and LAPACK libraries
    CONFIGURE_BLAS()

    # Use packaged Armadillo
    target_include_directories(
            actionet
            PRIVATE "${actionet_SOURCE_DIR}/include/extern/armadillo"
    )
endif ()

############ CONFIGURE BUILD
if (LAPACK_LINKER_FLAGS)
    target_compile_options(actionet PUBLIC ${LAPACK_LINKER_FLAGS})
endif ()
if (BLAS_LINKER_FLAGS)
    target_compile_options(actionet PUBLIC ${BLAS_LINKER_FLAGS})
endif ()

## Link BLAS and LAPACK
target_link_libraries(
        actionet
        PUBLIC ${LAPACK_LIBRARIES}
        PUBLIC ${BLAS_LIBRARIES}
)

## Add included headers
target_include_directories(
        actionet
        PRIVATE "${actionet_SOURCE_DIR}/include/extern/gcem"
        PRIVATE "${actionet_SOURCE_DIR}/include/extern"
        PRIVATE "${actionet_SOURCE_DIR}/include"
)

## Check for OpenMP support
message(NOTICE "Detecting OpenMP capabilities")
find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS}")
    add_compile_definitions(ARMA_OPENMP_THREADS=${NPROC})
endif ()

## Build diagnostics and debugging
get_directory_property(CPP_Defs COMPILE_DEFINITIONS)
message(NOTICE "Debug info:")
message(STATUS "COMPILE_DEFINITIONS = ${CPP_Defs}")
message(STATUS "LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}")
message(STATUS "LAPACK_LINKER_FLAGS: ${LAPACK_LINKER_FLAGS}")
message(STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
message(STATUS "BLAS_LINKER_FLAGS: ${BLAS_LINKER_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
