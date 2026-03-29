#[=============================================================================[
ConfigureCHOLMOD.cmake
================================================================================
DEPRECATED: This module is kept for reference but is no longer used.

As of the 2026 refactoring, libactionet uses native Armadillo sparse operations
instead of CHOLMOD for sparse-dense matrix multiplications. This provides:
- Better thread-safety (no shared CHOLMOD context)
- Support for matrices with >2^31 non-zero elements
- ~50% memory reduction (no matrix duplication)
- Simpler codebase without external SuiteSparse dependency

This file is retained for potential future use or reference.
================================================================================

Configure CHOLMOD (SuiteSparse) library for libactionet.

This module handles detection of the CHOLMOD library from SuiteSparse with
architecture-aware search paths for Homebrew packages.

The following variables are set:

``CHOLMOD_INCLUDE_DIR``
    Path to cholmod.h header file

``CHOLMOD_LIBRARY``
    Path to libcholmod library file

The module respects the following variables:

``TARGET_ARCHITECTURE``
    Target CPU architecture (arm64, x86_64, etc.). Used to determine
    correct Homebrew installation paths.

]=============================================================================]

macro(CONFIGURE_CHOLMOD libtarget)
    message(NOTICE "Searching for CHOLMOD library")

    ## CHOLMOD is typically installed via package managers (brew, apt, etc.)
    ## Architecture-aware Homebrew paths:
    ##   - ARM64 (Apple Silicon): /opt/homebrew
    ##   - x86_64 (Intel): /usr/local
    ##   - Generic/unknown: try both

    set(CHOLMOD_SEARCH_INCLUDE_PATHS
        /usr/local/include/suitesparse
        /usr/include/suitesparse
        /opt/local/include/suitesparse
        /usr/local/include
        /usr/include
    )

    set(CHOLMOD_SEARCH_LIB_PATHS
        /usr/local/lib
        /usr/lib
        /opt/local/lib
    )

    # Conda / custom prefixes (rootless installs)
    set(CHOLMOD_PREFIX_CANDIDATES "")
    if (DEFINED ENV{CONDA_PREFIX})
        list(APPEND CHOLMOD_PREFIX_CANDIDATES "$ENV{CONDA_PREFIX}")
    endif()
    if (DEFINED ENV{CHOLMOD_ROOT})
        list(APPEND CHOLMOD_PREFIX_CANDIDATES "$ENV{CHOLMOD_ROOT}")
    endif()
    if (DEFINED ENV{SUITESPARSE_ROOT})
        list(APPEND CHOLMOD_PREFIX_CANDIDATES "$ENV{SUITESPARSE_ROOT}")
    endif()
    if (DEFINED ENV{SuiteSparse_DIR})
        list(APPEND CHOLMOD_PREFIX_CANDIDATES "$ENV{SuiteSparse_DIR}")
    endif()

    foreach(prefix IN LISTS CHOLMOD_PREFIX_CANDIDATES)
        list(INSERT CHOLMOD_SEARCH_INCLUDE_PATHS 0 "${prefix}/include/suitesparse")
        list(INSERT CHOLMOD_SEARCH_INCLUDE_PATHS 0 "${prefix}/include")
        list(INSERT CHOLMOD_SEARCH_LIB_PATHS 0 "${prefix}/lib")
    endforeach()

    # Add architecture-specific Homebrew paths if on macOS
    if (APPLE)
        if (DEFINED TARGET_ARCHITECTURE)
            if ("${TARGET_ARCHITECTURE}" MATCHES "arm64" OR "${TARGET_ARCHITECTURE}" MATCHES "aarch64")
                list(INSERT CHOLMOD_SEARCH_INCLUDE_PATHS 0 /opt/homebrew/include/suitesparse)
                list(INSERT CHOLMOD_SEARCH_LIB_PATHS 0 /opt/homebrew/lib)
                message(STATUS "Using ARM64 Homebrew paths for CHOLMOD search")
            elseif ("${TARGET_ARCHITECTURE}" MATCHES "x86_64")
                list(INSERT CHOLMOD_SEARCH_INCLUDE_PATHS 0 /usr/local/opt/suite-sparse/include/suitesparse)
                list(INSERT CHOLMOD_SEARCH_INCLUDE_PATHS 0 /usr/local/include/suitesparse)
                list(INSERT CHOLMOD_SEARCH_LIB_PATHS 0 /usr/local/opt/suite-sparse/lib)
                list(INSERT CHOLMOD_SEARCH_LIB_PATHS 0 /usr/local/lib)
                message(STATUS "Using x86_64 Homebrew paths for CHOLMOD search")
            endif()
        else()
            # Fallback: add both Homebrew paths
            list(APPEND CHOLMOD_SEARCH_INCLUDE_PATHS
                /opt/homebrew/include/suitesparse
                /opt/homebrew/opt/suite-sparse/include/suitesparse)
            list(APPEND CHOLMOD_SEARCH_LIB_PATHS
                /opt/homebrew/lib
                /opt/homebrew/opt/suite-sparse/lib)
            message(VERBOSE "Added both ARM64 and x86_64 Homebrew paths for CHOLMOD")
        endif()

        ## NO_DEFAULT_PATH ensures we do not always pick up arm64 libraries for the wrong architecture under Rosetta
        # Find cholmod.h header
        find_path(CHOLMOD_INCLUDE_DIR
                NAMES cholmod.h
                PATHS ${CHOLMOD_SEARCH_INCLUDE_PATHS}
                DOC "CHOLMOD include directory"
                NO_DEFAULT_PATH ## Prevent finding default arm64 paths on x86
        )

        # Find libcholmod library
        find_library(CHOLMOD_LIBRARY
                NAMES cholmod
                PATHS ${CHOLMOD_SEARCH_LIB_PATHS}
                DOC "CHOLMOD library"
                NO_DEFAULT_PATH ## Prevent finding default arm64 paths on x86
        )
    else ()
        # Find cholmod.h header
        find_path(CHOLMOD_INCLUDE_DIR
                NAMES cholmod.h
                PATHS ${CHOLMOD_SEARCH_INCLUDE_PATHS}
                DOC "CHOLMOD include directory"
        )

        # Find libcholmod library
        find_library(CHOLMOD_LIBRARY
                NAMES cholmod
                PATHS ${CHOLMOD_SEARCH_LIB_PATHS}
                DOC "CHOLMOD library"
        )
    endif()

    # Check if found
    if (CHOLMOD_INCLUDE_DIR AND CHOLMOD_LIBRARY)
        message(STATUS "Found CHOLMOD:")
        message(STATUS "  Include: ${CHOLMOD_INCLUDE_DIR}")
        message(STATUS "  Library: ${CHOLMOD_LIBRARY}")

        # Add include directory
        target_include_directories(${libtarget} PRIVATE ${CHOLMOD_INCLUDE_DIR})

        # Link library
        target_link_libraries(${libtarget} PUBLIC ${CHOLMOD_LIBRARY})

        # Note: Code uses cholmod_l_* functions (64-bit API) directly, no compile definition needed
    else()
        if (NOT CHOLMOD_INCLUDE_DIR)
            message(FATAL_ERROR "Could not find cholmod.h. Please install SuiteSparse (brew install suite-sparse or apt-get install libsuitesparse-dev)")
        endif()
        if (NOT CHOLMOD_LIBRARY)
            message(FATAL_ERROR "Could not find libcholmod. Please install SuiteSparse (brew install suite-sparse or apt-get install libsuitesparse-dev)")
        endif()
    endif()
endmacro()
