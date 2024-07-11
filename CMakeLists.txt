cmake_minimum_required(VERSION 3.12)

## Mac arm64 compilers
set(USE_GCC "/opt/homebrew/Cellar/gcc/14.1.0_1/bin")

## Mac x86 Rosetta2 compilers
set(USE_GCC "/usr/local/Cellar/gcc/14.1.0_1/bin/")

set(CMAKE_C_COMPILER "${USE_GCC}/gcc-14")
set(CMAKE_CXX_COMPILER "${USE_GCC}/g++-14")

# set(CMAKE_C_COMPILER /opt/homebrew/Cellar/gcc/14.1.0_1/bin/gcc-14)
# set(CMAKE_CXX_COMPILER /opt/homebrew/Cellar/gcc/14.1.0_1/bin/g++-14)


project(libactionet)
list(APPEND CMAKE_MODULE_PATH "${libactionet_SOURCE_DIR}/cmake")

# list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/Cellar/)

# ########### Compiler options
# C++11 is not supported, as the code uses C++17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

MACRO(HEADER_DIRECTORIES return_list)
    FILE(GLOB_RECURSE new_list *.h *.hpp)
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()

# add_compile_options(-w)
# add_compile_options(-O3)

# ########### Third-party dependencies
## Check for BLAS/LAPACK
find_package(BLAS REQUIRED)
message(STATUS "BLAS_COMPILER_FLAGS: ${BLAS_COMPILER_FLAGS}")
add_definitions("-DEIGEN_USE_BLAS") # Keep for future Eigen use
list(APPEND EXTERNAL_LIBS "${BLAS_LIBRARIES}")

## Check for CHOLMOD
find_package(SuiteSparse REQUIRED)
# find_package(SuiteSparse_config REQUIRED NO_MODULE)
# include_directories(${SUITESPARSE_INCLUDE_DIRS})

## Add all C/C++ code to sources for compilation
file(GLOB_RECURSE SOURCES *.cpp *.cc *.c)
add_library(actionet STATIC ${SOURCES})

## [Dirty] Add all header files in the directory for testing
HEADER_DIRECTORIES(header_list)
include_directories(${header_list})

## Add include headers
# set(HEADER_ROOT "${libactionet_SOURCE_DIR}/include")

# file(GLOB HEADER_DIRS ${libactionet_SOURCE_DIR}/include)
# include_directories(${HEADER_DIRS} )
# target_include_directories(libactionet PUBLIC ${libactionet_SOURCE_DIR}/include ${libactionet_SOURCE_DIR}/include/math/aarand )


target_link_libraries(
    actionet PRIVATE
    ${BLAS_LIBRARIES}
    SuiteSparse::CHOLMOD
    # ${SUITESPARSE_LIBRARIES}
    # ${HDF5_LIBRARIES}
)
