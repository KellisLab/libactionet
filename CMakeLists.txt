cmake_minimum_required(VERSION 3.19)

project(actionet)
## Include third-part cmake modules
list(APPEND CMAKE_MODULE_PATH "${actionet_SOURCE_DIR}/cmake")
include(ConfigureApple)
include(ConfigureBLAS)
include(ConfigureR)
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
## Set BLAS vendor default
set(BLA_VENDOR All)

## Set Apple CPU-specific compiler flags
if (APPLE)
    CONFIGURE_APPLE(actionet)
endif ()

############ BUILD MODES
if (LIBACTIONET_BUILD_R) ## OS-agnostic R library configurations
    CONFIGURE_R()
elseif (UNIX) # Build for Unix-alike configurations (including macOS)
    ## Set BLAS and LAPACK libraries
    CONFIGURE_BLAS(actionet)

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

message(NOTICE "Searching for SuiteSparse installation")
## Find and link CHOLMOD
find_package(SuiteSparse REQUIRED COMPONENTS CHOLMOD)
target_link_libraries(actionet PUBLIC SuiteSparse::CHOLMOD)

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
    target_link_libraries(actionet
            PRIVATE OpenMP::OpenMP_C
            PRIVATE OpenMP::OpenMP_CXX
    )
    target_compile_definitions(actionet PUBLIC ARMA_OPENMP_THREADS=${NPROC})
else ()
    message(NOTICE "OpenMP not found")
    message(STATUS "Building without OpenMP")
endif ()

## Additional (optional) compiler options
target_compile_options(actionet PUBLIC -Wno-psabi -Wno-deprecated-declarations)# -ferror-limit=10)#-fmax-errors=10)
#add_compile_options(-w)

## Build diagnostics and debugging
#get_directory_property(CPP_Defs COMPILE_DEFINITIONS)
#message(NOTICE "Debug info:")
#message(STATUS "COMPILE_DEFINITIONS = ${CPP_Defs}")
#message(STATUS "LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}")
#message(STATUS "LAPACK_LINKER_FLAGS: ${LAPACK_LINKER_FLAGS}")
#message(STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
#message(STATUS "BLAS_LINKER_FLAGS: ${BLAS_LINKER_FLAGS}")
#message(STATUS "OpenMP_C_INCLUDE_DIRS: ${OpenMP_C_INCLUDE_DIRS}")
#message(STATUS "OpenMP_CXX_INCLUDE_DIRS: ${OpenMP_CXX_INCLUDE_DIRS}")
#
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
