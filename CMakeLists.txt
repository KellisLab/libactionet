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
list(APPEND CMAKE_MODULE_PATH "${actionet_SOURCE_DIR}/cmake")

# list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/Cellar/)

# ########### Compiler options
# C++11 is not supported, as the code uses C++17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#MACRO(HEADER_DIRECTORIES return_list)
#    FILE(GLOB_RECURSE new_list *.h *.hpp)
#    SET(dir_list "")
#    FOREACH(file_path ${new_list})
#        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
#        SET(dir_list ${dir_list} ${dir_path})
#    ENDFOREACH()
#    LIST(REMOVE_DUPLICATES dir_list)
#    SET(${return_list} ${dir_list})
#ENDMACRO()

add_compile_options(-w)
#add_compile_options(-O3)

## Add all C/C++ code to sources for compilation
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(actionet
        STATIC ${SOURCES}
#         r_wrapper.cpp # Test Rcpp wrappers
)

## Check for BLAS. Platform dependent.
if (APPLE)
    message("cmake: Configuring for macOS.")
    add_definitions(-DLIBACTIONET_PLATFORM_APPLE)

    set(BLA_VENDOR Apple)
    find_package(BLAS REQUIRED)

    add_compile_options(-flax-vector-conversions -framework Accelerate)
    #    TODO: Probably unneeded. Suppress deprecation warnings.
    add_definitions(-DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64)

    target_include_directories(
            actionet
            PRIVATE "${BLAS_LIBRARIES}/Frameworks/vecLib.framework/Headers"
    )
elseif (UNIX)
    #    set(BLA_VENDOR Apple)
    find_package(BLAS REQUIRED)
endif ()

## Check for LAPACK
find_package(LAPACK REQUIRED)

## Check for CHOLMOD
find_package(SuiteSparse REQUIRED COMPONENTS CHOLMOD)

## Build mode for testing R interface
# set(LIBACTIONET_BUILD_R 1)

if (LIBACTIONET_BUILD_R)
    message("cmake: Configuring for R library.")
    if (NOT DEFINED R_HOME)
        message(FATAL_ERROR "R_HOME not defined.")
    else ()
        message("Using R found at ${R_HOME}")
    endif ()
    add_definitions(-DLIBACTIONET_BUILD_R) # Set R build mode in config headers

    # find R headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/R CMD config --cppflags | sed s/-I//g"
            OUTPUT_VARIABLE rInclude
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # find Rcpp headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(system.file(\"include\", package=\"Rcpp\"))'"
            OUTPUT_VARIABLE rcppInclude
    )

    # find RcppArmadillo headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(system.file(\"include\", package=\"RcppArmadillo\"))'"
            OUTPUT_VARIABLE rcppArmaInclude
    )

    target_include_directories(actionet
            PRIVATE "${rInclude}"
            PRIVATE "${rcppInclude}"
            PRIVATE "${rcppArmaInclude}")
else ()
    target_include_directories(
            actionet
            PRIVATE "${actionet_SOURCE_DIR}/include/extern/armadillo"
    )
endif ()
## Add include headers
target_include_directories(
        actionet
        PRIVATE "${actionet_SOURCE_DIR}/include/extern/gcem"
        PRIVATE "${actionet_SOURCE_DIR}/include/extern"
        PRIVATE "${actionet_SOURCE_DIR}/include"
)
#include_directories("${actionet_SOURCE_DIR}/../../inst/include")

## [Dirty] Add all header files in the directory for testing/
#HEADER_DIRECTORIES(header_list)
#include_directories(${header_list})

target_compile_options(actionet PUBLIC ${LAPACK_LINKER_FLAGS})
target_compile_options(actionet PUBLIC ${BLAS_LINKER_FLAGS})

target_link_libraries(
        actionet
        PUBLIC SuiteSparse::CHOLMOD
        PUBLIC ${LAPACK_LIBRARIES}
        PUBLIC ${BLAS_LIBRARIES}
)

#message("LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}")
#message("LAPACK_LINKER_FLAGS: ${LAPACK_LINKER_FLAGS}")
#message("BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
#message("BLAS_LINKER_FLAGS: ${BLAS_LINKER_FLAGS}")
