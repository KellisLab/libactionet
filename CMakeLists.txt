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

project(actionet)
## Include third-part cmake modules
list(APPEND CMAKE_MODULE_PATH "${actionet_SOURCE_DIR}/cmake")

if (APPLE)
    ## Set default BLAS for macOS
    set(BLA_VENDOR Apple)
elseif (UNIX)
    #    set(BLA_VENDOR something)
endif ()

macro(CONFIGURE_ACCELERATE)
    ## Find and link Apple BLAS
    message("Using Apple Accelerate for BLAS and LAPACK")
    add_compile_definitions(LIBACTIONET_BLAS_ACCELERATE)

    set(ACCELERATE_HEADERS "${BLAS_LIBRARIES}/Frameworks/vecLib.framework/Headers")
    if (NOT EXISTS "${ACCELERATE_HEADERS}")
        message(FATAL_ERROR "Cannot locate Apple Accelerate headers")
    endif ()

    message("Accelerate headers: ${ACCELERATE_HEADERS}")
    target_include_directories(
            actionet
            PRIVATE "${ACCELERATE_HEADERS}"
    )
    # TODO: Detect Apple architecture. Currently assumes and requires arm64.
    add_compile_options(-flax-vector-conversions -framework Accelerate)
    #    TODO: Probably unneeded. Suppress deprecation warnings.
    add_compile_definitions(ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
endmacro()

# ########### Compiler options
# C++11 is not supported, as the code uses C++17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

## Detect usable logical CPUs
cmake_host_system_information(RESULT NPROC QUERY NUMBER_OF_LOGICAL_CORES)
message("Detecting CPU cores: ${NPROC}")
if (NPROC GREATER_EQUAL 6)
    math(EXPR NPROC "${NPROC} - 2" OUTPUT_FORMAT DECIMAL)
    message("Using ${NPROC} cores")
endif ()

## Suppress warning
add_compile_options(-w)
#add_compile_options(-O3)

## Add all C/C++ code to sources for compilation
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(actionet STATIC ${SOURCES})

## Find and link CHOLMOD
find_package(SuiteSparse REQUIRED COMPONENTS CHOLMOD)
target_link_libraries(actionet PUBLIC SuiteSparse::CHOLMOD)

############ Interface test
## Build mode for testing R interface
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

############ Build modes
if (LIBACTIONET_BUILD_R) ## OS-agnostic R library configurations
    message("cmake: Configuring for R library")
    if (NOT DEFINED R_HOME)
        message(FATAL_ERROR "R_HOME not defined")
    else ()
        message("Using R found at ${R_HOME}")
    endif ()
    add_compile_definitions(LIBACTIONET_BUILD_R) # Set R build mode in config header


    ## Find R headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/R CMD config --cppflags | sed s/-I//g"
            OUTPUT_VARIABLE rInclude
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    ## Find Rcpp headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(system.file(\"include\", package=\"Rcpp\"))'"
            OUTPUT_VARIABLE rcppInclude
    )

    ## Find RcppArmadillo headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(system.file(\"include\", package=\"RcppArmadillo\"))'"
            OUTPUT_VARIABLE rcppArmaInclude
    )

    ## Find R CPPFLAGS
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/R CMD config CPPFLAGS | sed s/-I//g"
            OUTPUT_VARIABLE rCppFLags
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (APPLE AND ("${BLA_VENDOR}" STREQUAL "Apple"))
        CONFIGURE_ACCELERATE()
    else ()
        ## Find R BLAS_LIBS
        execute_process(
                COMMAND bash -c "${R_HOME}/bin/R CMD config BLAS_LIBS | sed s/-I//g"
                OUTPUT_VARIABLE BLAS_LIBRARIES
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        ## Find R LAPACK_LIBS
        execute_process(
                COMMAND bash -c "${R_HOME}/bin/R CMD config LAPACK_LIBS | sed s/-I//g"
                OUTPUT_VARIABLE LAPACK_LIBRARIES
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif ()

    target_include_directories(actionet
            PRIVATE "${rInclude}"
            PRIVATE "${rCppFLags}"
            PRIVATE "${rcppInclude}"
            PRIVATE "${rcppArmaInclude}"
    )
elseif (UNIX) # Build for Unix-alike configurations (including macOS)

    ## Find BLAS
    find_package(BLAS REQUIRED)

    ## Find LAPACK
    find_package(LAPACK REQUIRED)

    if (APPLE) ## Mac-specific configurations
        message("cmake: Configuring for macOS")
        if ("${BLA_VENDOR}" STREQUAL "Apple")
            CONFIGURE_ACCELERATE()
        endif ()
    endif ()

    # Use packaged Armadillo
    target_include_directories(
            actionet
            PRIVATE "${actionet_SOURCE_DIR}/include/extern/armadillo"
    )

endif ()

## Configure Intel MKL if linked
foreach (lib ${BLAS_LIBRARIES})
    if (${lib} MATCHES "mkl")
        message("Using Intel MKL for BLAS and LAPACK")
        if (NOT DEFINED ENV{MKLROOT})
            message(FATAL_ERROR "Using Intel MKL but `MKLROOT` not defined")
        endif ()

        set(MKL_HEADERS "$ENV{MKLROOT}/include")
        if (EXISTS "${MKL_HEADERS}")
            message("MKL headers: ${MKL_HEADERS}")
            target_include_directories(
                    actionet
                    PRIVATE "${MKL_HEADERS}"
            )
            add_compile_definitions(LIBACTIONET_BLAS_MKL)
        else ()
            message("Not using MKL headers.")
        endif ()
        break()
    endif ()
endforeach ()

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
# TODO: Adapt for when building with Apple Clang. Does not support OpenMP.
find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS}")
    add_compile_definitions(ARMA_OPENMP_THREADS=${NPROC})
endif ()

## Build diagnostics and debugging
get_directory_property(CPP_Defs COMPILE_DEFINITIONS)
message("COMPILE_DEFINITIONS = ${CPP_Defs}")

message("LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}")
message("LAPACK_LINKER_FLAGS: ${LAPACK_LINKER_FLAGS}")
message("BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
message("BLAS_LINKER_FLAGS: ${BLAS_LINKER_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
