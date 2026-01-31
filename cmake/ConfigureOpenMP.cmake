#[=============================================================================[
ConfigureOpenMP.cmake
================================================================================
Detect and configure OpenMP support for libactionet.

This module handles cross-platform OpenMP detection with special handling for
macOS Homebrew installations, which are keg-only and not auto-discoverable.

The following variables control the behaviour of this module:

``LIBACTIONET_OPENMP_RUNTIME``
    Select OpenMP runtime: AUTO, GNU, INTEL, LLVM, OFF.

``LIBACTIONET_OPENMP_CXXFLAGS``
    OpenMP CXX flags (used for R builds when provided).

``LIBACTIONET_OPENMP_LDFLAGS``
    OpenMP linker flags (used for R builds when provided).

``TARGET_ARCHITECTURE``
    Target CPU architecture (arm64, x86_64, aarch64, etc.). Used to determine
    correct Homebrew installation paths.

]=============================================================================]

set(LIBACTIONET_OPENMP_RUNTIME "AUTO" CACHE STRING "OpenMP runtime: AUTO, GNU, INTEL, LLVM, OFF")
set_property(CACHE LIBACTIONET_OPENMP_RUNTIME PROPERTY STRINGS AUTO GNU INTEL LLVM OFF)
set(LIBACTIONET_OPENMP_CXXFLAGS "" CACHE STRING "OpenMP CXX flags (R builds)")
set(LIBACTIONET_OPENMP_LDFLAGS "" CACHE STRING "OpenMP linker flags (R builds)")

macro(CONFIGURE_OPENMP libtarget)
    message(NOTICE "Configuring OpenMP support")

    set(_openmp_handled FALSE)

    if (LIBACTIONET_BUILD_R)
        string(STRIP "${LIBACTIONET_OPENMP_CXXFLAGS}" _r_openmp_cxxflags)
        string(STRIP "${LIBACTIONET_OPENMP_LDFLAGS}" _r_openmp_ldflags)

        if (NOT _r_openmp_cxxflags STREQUAL "" OR NOT _r_openmp_ldflags STREQUAL "")
            set(_openmp_handled TRUE)
            set(OpenMP_FOUND TRUE)

            if (NOT _r_openmp_cxxflags STREQUAL "")
                if (_r_openmp_cxxflags MATCHES ";")
                    set(_r_openmp_cxxflags_list ${_r_openmp_cxxflags})
                else()
                    separate_arguments(_r_openmp_cxxflags_list NATIVE_COMMAND "${_r_openmp_cxxflags}")
                endif()
                target_compile_options(${libtarget} PRIVATE ${_r_openmp_cxxflags_list})
                set(OpenMP_CXX_FLAGS "${_r_openmp_cxxflags}")
            endif()

            if (NOT _r_openmp_ldflags STREQUAL "")
                if (_r_openmp_ldflags MATCHES ";")
                    set(_r_openmp_ldflags_list ${_r_openmp_ldflags})
                else()
                    separate_arguments(_r_openmp_ldflags_list NATIVE_COMMAND "${_r_openmp_ldflags}")
                endif()
                target_link_options(${libtarget} PRIVATE ${_r_openmp_ldflags_list})
            endif()

            message(STATUS "OpenMP configured from R SHLIB_OPENMP flags")
        else()
            # R SHLIB_OPENMP flags are empty - fall back to standard OpenMP detection
            # instead of disabling OpenMP entirely
            message(STATUS "R build: SHLIB_OPENMP flags empty, attempting standard OpenMP detection")
        endif()
    endif()

    if (NOT _openmp_handled AND APPLE)
        # On macOS, libomp from Homebrew is installed as keg-only, so we need special handling
        message(STATUS "Detecting OpenMP on macOS (target arch: ${TARGET_ARCHITECTURE})")

        # First, try the standard find_package approach
        find_package(OpenMP QUIET)

        # If not found, try to locate libomp from Homebrew
        if (NOT OpenMP_FOUND)
            message(STATUS "OpenMP not found via find_package, checking Homebrew libomp...")

            # Determine Homebrew prefix based on architecture
            # On Apple Silicon (arm64), Homebrew defaults to /opt/homebrew
            # On Intel (x86_64), Homebrew defaults to /usr/local
            set(HOMEBREW_LIBOMP_PATHS)

            if ("${TARGET_ARCHITECTURE}" MATCHES "arm64" OR "${TARGET_ARCHITECTURE}" MATCHES "aarch64")
                list(APPEND HOMEBREW_LIBOMP_PATHS "/opt/homebrew/opt/libomp")
                message(VERBOSE "Searching Homebrew ARM64 paths")
            elseif ("${TARGET_ARCHITECTURE}" MATCHES "x86_64")
                list(APPEND HOMEBREW_LIBOMP_PATHS "/usr/local/opt/libomp")
                message(VERBOSE "Searching Homebrew Intel x86_64 paths")
            else()
                # Unknown architecture, try both
                message(WARNING "Unknown target architecture: ${TARGET_ARCHITECTURE}. Trying common Homebrew paths.")
                list(APPEND HOMEBREW_LIBOMP_PATHS "/opt/homebrew/opt/libomp")
                list(APPEND HOMEBREW_LIBOMP_PATHS "/usr/local/opt/libomp")
            endif()

            # Iterate through possible Homebrew paths
            foreach(HOMEBREW_PATH ${HOMEBREW_LIBOMP_PATHS})
                if (EXISTS "${HOMEBREW_PATH}")
                    message(STATUS "Found Homebrew libomp at: ${HOMEBREW_PATH}")

                    # Set OpenMP library path
                    set(OpenMP_libomp_LIBRARY "${HOMEBREW_PATH}/lib/libomp.dylib")

                    # Verify library exists
                    if (NOT EXISTS "${OpenMP_libomp_LIBRARY}")
                        message(WARNING "libomp.dylib not found at ${OpenMP_libomp_LIBRARY}")
                        continue()
                    endif()

                    # Add include and library directories
                    target_include_directories(${libtarget} PRIVATE "${HOMEBREW_PATH}/include")
                    target_link_libraries(${libtarget} PRIVATE "${OpenMP_libomp_LIBRARY}")

                    # Add OpenMP compiler flags (separate -Xclang and -fopenmp)
                    target_compile_options(${libtarget} PRIVATE -Xclang -fopenmp)
                    target_link_options(${libtarget} PRIVATE -Xclang -fopenmp)

                    set(OpenMP_FOUND TRUE)
                    break()
                endif()
            endforeach()
        endif()
    endif()

    # Standard find_package for non-Apple systems or if not yet found
    if (NOT _openmp_handled AND NOT APPLE)
        string(TOUPPER "${LIBACTIONET_OPENMP_RUNTIME}" _omp_runtime)
        string(TOUPPER "${CMAKE_CXX_COMPILER_ID}" _omp_compiler_id)

        if (_omp_runtime STREQUAL "OFF")
            set(OpenMP_FOUND FALSE)
            set(_openmp_handled TRUE)
            message(STATUS "OpenMP disabled by LIBACTIONET_OPENMP_RUNTIME=OFF")
        endif()

        set(_omp_search_paths "")
        if (DEFINED ENV{CONDA_PREFIX})
            list(APPEND _omp_search_paths "$ENV{CONDA_PREFIX}/lib")
        endif()
        if (DEFINED ENV{MKLROOT})
            list(APPEND _omp_search_paths "$ENV{MKLROOT}/lib" "$ENV{MKLROOT}/lib/intel64")
        endif()

        # Detect whether MKL is in use (from BLAS vendor, MKLROOT, or linked libs)
        set(_mkl_detected FALSE)
        if (DEFINED BLA_VENDOR AND BLA_VENDOR MATCHES "Intel")
            set(_mkl_detected TRUE)
        endif()
        if (DEFINED ENV{MKLROOT})
            set(_mkl_detected TRUE)
        endif()
        foreach (_blas_lib IN LISTS BLAS_LIBRARIES)
            if (_blas_lib MATCHES "mkl")
                set(_mkl_detected TRUE)
                break()
            endif()
        endforeach()

        # Auto runtime selection:
        # - If MKL is present and libiomp5 is available, pick Intel OMP.
        # - If MKL is present but libiomp5 is missing, fall back to GNU OMP and force
        #   MKL_THREADING_LAYER=GNU to avoid mixed runtimes.
        # - Otherwise prefer GNU (or Intel if using Intel compilers).
        if (_omp_runtime STREQUAL "AUTO")
            if (_mkl_detected)
                if (NOT OPENMP_IOMP5_LIBRARY)
                    find_library(OPENMP_IOMP5_LIBRARY NAMES iomp5 libiomp5 HINTS ${_omp_search_paths})
                endif()
                if (OPENMP_IOMP5_LIBRARY)
                    set(_omp_runtime "INTEL")
                    message(STATUS "AUTO OpenMP: MKL detected with libiomp5 available; selecting INTEL runtime.")
                else()
                    set(_omp_runtime "GNU")
                    set(_force_mkl_threading_layer_gnu TRUE)
                    message(STATUS "AUTO OpenMP: MKL detected without libiomp5; selecting GNU runtime and setting MKL_THREADING_LAYER=GNU.")
                endif()
            elseif (_omp_compiler_id MATCHES "INTEL")
                set(_omp_runtime "INTEL")
                message(STATUS "AUTO OpenMP: Intel compiler detected; selecting INTEL runtime.")
            else()
                set(_omp_runtime "GNU")
                message(STATUS "AUTO OpenMP: no MKL detected; selecting GNU runtime.")
            endif()
        endif()

        if (_omp_runtime STREQUAL "GNU")
            find_library(OPENMP_GOMP_LIBRARY NAMES gomp HINTS ${_omp_search_paths})
            if (OPENMP_GOMP_LIBRARY)
                set(OpenMP_C_FLAGS "-fopenmp")
                set(OpenMP_CXX_FLAGS "-fopenmp")
                set(OpenMP_C_LIB_NAMES "gomp")
                set(OpenMP_CXX_LIB_NAMES "gomp")
                set(OpenMP_gomp_LIBRARY "${OPENMP_GOMP_LIBRARY}")
                set(OpenMP_FOUND TRUE)

                target_compile_options(${libtarget} PRIVATE -fopenmp)
                target_link_options(${libtarget} PRIVATE -fopenmp)
                target_link_libraries(${libtarget} PRIVATE "${OPENMP_GOMP_LIBRARY}")

                if (_mkl_detected)
                    # Force MKL to use GNU threading to avoid runtime mismatch.
                    set(ENV{MKL_THREADING_LAYER} "GNU")
                    message(STATUS "Configured MKL_THREADING_LAYER=GNU for MKL + GNU OpenMP build.")
                endif()
            else()
                message(WARNING "GNU OpenMP (libgomp) not found; falling back to default OpenMP detection.")
                find_package(OpenMP QUIET)
            endif()
        elseif (_omp_runtime STREQUAL "INTEL")
            find_library(OPENMP_IOMP5_LIBRARY NAMES iomp5 libiomp5 HINTS ${_omp_search_paths})
            if (OPENMP_IOMP5_LIBRARY)
                if (_omp_compiler_id MATCHES "INTEL")
                    set(_omp_compile_flag "-qopenmp")
                    set(_omp_link_flag "-qopenmp")
                else()
                    set(_omp_compile_flag "-fopenmp")
                    set(_omp_link_flag "")
                    message(WARNING "Intel OpenMP runtime selected with non-Intel compiler; ensure toolchain compatibility.")
                endif()

                set(OpenMP_C_FLAGS "${_omp_compile_flag}")
                set(OpenMP_CXX_FLAGS "${_omp_compile_flag}")
                set(OpenMP_C_LIB_NAMES "iomp5")
                set(OpenMP_CXX_LIB_NAMES "iomp5")
                set(OpenMP_iomp5_LIBRARY "${OPENMP_IOMP5_LIBRARY}")
                set(OpenMP_FOUND TRUE)

                target_compile_options(${libtarget} PRIVATE ${_omp_compile_flag})
                if (_omp_link_flag)
                    target_link_options(${libtarget} PRIVATE ${_omp_link_flag})
                endif()
                target_link_libraries(${libtarget} PRIVATE "${OPENMP_IOMP5_LIBRARY}")
            else()
                message(WARNING "Intel OpenMP runtime requested but libiomp5 not found; falling back to default OpenMP detection.")
                find_package(OpenMP QUIET)
            endif()
        elseif (_omp_runtime STREQUAL "LLVM")
            find_library(OPENMP_OMP_LIBRARY NAMES omp HINTS ${_omp_search_paths})
            if (OPENMP_OMP_LIBRARY)
                set(OpenMP_C_FLAGS "-fopenmp")
                set(OpenMP_CXX_FLAGS "-fopenmp")
                set(OpenMP_C_LIB_NAMES "omp")
                set(OpenMP_CXX_LIB_NAMES "omp")
                set(OpenMP_omp_LIBRARY "${OPENMP_OMP_LIBRARY}")
                set(OpenMP_FOUND TRUE)

                target_compile_options(${libtarget} PRIVATE -fopenmp)
                target_link_options(${libtarget} PRIVATE -fopenmp)
                target_link_libraries(${libtarget} PRIVATE "${OPENMP_OMP_LIBRARY}")
            else()
                message(WARNING "LLVM OpenMP runtime requested but libomp not found; falling back to default OpenMP detection.")
                find_package(OpenMP QUIET)
            endif()
        endif()

        if (OpenMP_FOUND)
            if (_mkl_detected AND NOT _omp_runtime STREQUAL "INTEL")
                message(WARNING "MKL detected with ${_omp_runtime} OpenMP runtime. Set MKL_THREADING_LAYER=GNU or select LIBACTIONET_OPENMP_RUNTIME=INTEL to avoid mixed runtimes.")
            endif()
        endif()
    endif()

    if (OpenMP_FOUND)
        # If we didn't already link (Homebrew case), do it here for standard CMake OpenMP targets
        if ((NOT APPLE OR NOT TARGET OpenMP::OpenMP_CXX) AND NOT OPENMP_GOMP_LIBRARY AND NOT OPENMP_IOMP5_LIBRARY AND NOT OPENMP_OMP_LIBRARY AND NOT _openmp_handled)
            if (TARGET OpenMP::OpenMP_C)
                target_link_libraries(${libtarget} PRIVATE OpenMP::OpenMP_C)
            endif()
            if (TARGET OpenMP::OpenMP_CXX)
                target_link_libraries(${libtarget} PRIVATE OpenMP::OpenMP_CXX)
            endif()
        endif()

        message(STATUS "OpenMP successfully configured")
        message(VERBOSE "OpenMP details:")
        message(VERBOSE "  C_FLAGS: ${OpenMP_C_FLAGS}")
        message(VERBOSE "  CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
    else()
        message(WARNING "OpenMP not found. Building without OpenMP support.")
    endif()
endmacro()
