cmake_minimum_required(VERSION 3.12)


## Specify macos arm64 compilers
# set(USE_GCC "/opt/homebrew/Cellar/gcc/14.1.0_1/bin")

## Specify macos x86 Rosetta2 compilers
# set(USE_GCC "/usr/local/Cellar/gcc/14.1.0_1/bin/")
# string(APPEND CMAKE_CXX_FLAGS_RELEASE " -arch x86_64")
# string(APPEND CMAKE_CC_FLAGS_RELEASE " -arch x86_64")

## Set compiler
# set(CMAKE_C_COMPILER "${USE_GCC}/gcc-14")
# set(CMAKE_CXX_COMPILER "${USE_GCC}/g++-14")

project(libactionet)
list(APPEND CMAKE_MODULE_PATH "${libactionet_SOURCE_DIR}/cmake")

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

#add_compile_options(-w)
# add_compile_options(-O3)

# ########### Third-party dependencies
## Check for BLAS/LAPACK
find_package(BLAS REQUIRED)
# message(STATUS "BLAS_COMPILER_FLAGS: ${BLAS_COMPILER_FLAGS}")
#add_definitions("-DEIGEN_USE_BLAS") # Keep for future Eigen use
# list(APPEND EXTERNAL_LIBS "${BLAS_LIBRARIES}")

find_package(LAPACK REQUIRED)
# message(STATUS "BLAS_COMPILER_FLAGS: ${BLAS_COMPILER_FLAGS}")
# list(APPEND EXTERNAL_LIBS "${LAPACK_LIBRARIES}")

## Check for CHOLMOD
find_package(SuiteSparse REQUIRED COMPONENTS CHOLMOD)
# find_package(SuiteSparse_config REQUIRED NO_MODULE)
# include_directories(${SUITESPARSE_INCLUDE_DIRS})

## Add all C/C++ code to sources for compilation
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(libactionet STATIC ${SOURCES})

## Build mode for testing R/Python interface
set(LIBACTIONET_BUILD_R 1)

if (LIBACTIONET_BUILD_R)
    add_definitions(-DLIBACTIONET_BUILD_R) # Set R build mode in config headers

    # find R headers
    execute_process(
            COMMAND bash -c "R CMD config --cppflags | sed s/-I//g"
            OUTPUT_VARIABLE rInclude
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # find Rcpp headers
    execute_process(
            COMMAND bash -c "Rscript -e 'cat(system.file(\"include\", package=\"Rcpp\"))'"
            OUTPUT_VARIABLE rcppInclude
    )

    # find RcppArmadillo headers
    execute_process(
            COMMAND bash -c "Rscript -e 'cat(system.file(\"include\", package=\"RcppArmadillo\"))'"
            OUTPUT_VARIABLE rcppArmaInclude
    )

    target_include_directories(libactionet
            PRIVATE "${rInclude}"
            PRIVATE "${rcppInclude}"
            PRIVATE "${rcppArmaInclude}")
else ()
    target_include_directories(
            libactionet
            PRIVATE "${libactionet_SOURCE_DIR}/include/extern/armadillo"
    )
endif ()
## Add include headers
target_include_directories(
        libactionet
        PRIVATE "${libactionet_SOURCE_DIR}/include/extern/gcem"
        PRIVATE "${libactionet_SOURCE_DIR}/include/extern"
        PUBLIC "${libactionet_SOURCE_DIR}/include"
)
#include_directories("${libactionet_SOURCE_DIR}/../../inst/include")

## [Dirty] Add all header files in the directory for testing/
#HEADER_DIRECTORIES(header_list)
#include_directories(${header_list})

target_compile_options(libactionet PUBLIC ${LAPACK_LINKER_FLAGS})
target_compile_options(libactionet PUBLIC ${BLAS_LINKER_FLAGS})

target_link_libraries(
        libactionet PUBLIC
        SuiteSparse::CHOLMOD
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
)
