#[=============================================================================[
PreloadBLAS.cmake
================================================================================
Preload BLAS/LAPACK variables to prevent FindBLAS/FindLAPACK from searching.

This module must be included BEFORE any find_package(BLAS) or find_package(LAPACK) calls.
It sets all the variables that FindBLAS and FindLAPACK would set, effectively
short-circuiting their search logic.

This is necessary when using R's BLAS/LAPACK libraries, which are in non-standard
locations that CMake's FindBLAS cannot discover on its own.

]=============================================================================]

# Check if BLAS/LAPACK are already set (from R configuration)
if (DEFINED PRELOAD_BLAS_LIBRARIES AND DEFINED PRELOAD_LAPACK_LIBRARIES)
    message(STATUS "Preloading BLAS/LAPACK configuration to prevent redundant searches")

    # Set all variables that FindBLAS.cmake would set
    # Using CACHE STRING (not INTERNAL) so they appear in CMakeCache.txt
    # Using FORCE to override any previous values
    set(BLAS_FOUND TRUE CACHE STRING "BLAS library status" FORCE)
    set(BLAS_LIBRARIES "${PRELOAD_BLAS_LIBRARIES}" CACHE STRING "BLAS libraries" FORCE)
    set(BLAS_LINKER_FLAGS "" CACHE STRING "BLAS linker flags" FORCE)
    set(BLAS95_FOUND FALSE CACHE STRING "BLAS95 status" FORCE)
    set(BLAS95_LIBRARIES "" CACHE STRING "BLAS95 libraries" FORCE)

    set(LAPACK_FOUND TRUE CACHE STRING "LAPACK library status" FORCE)
    set(LAPACK_LIBRARIES "${PRELOAD_LAPACK_LIBRARIES}" CACHE STRING "LAPACK libraries" FORCE)
    set(LAPACK_LINKER_FLAGS "" CACHE STRING "LAPACK linker flags" FORCE)
    set(LAPACK95_FOUND FALSE CACHE STRING "LAPACK95 status" FORCE)
    set(LAPACK95_LIBRARIES "" CACHE STRING "LAPACK95 libraries" FORCE)

    # Mark these variables as advanced to hide from normal users
    mark_as_advanced(BLAS_FOUND BLAS_LIBRARIES BLAS_LINKER_FLAGS BLAS95_FOUND BLAS95_LIBRARIES)
    mark_as_advanced(LAPACK_FOUND LAPACK_LIBRARIES LAPACK_LINKER_FLAGS LAPACK95_FOUND LAPACK95_LIBRARIES)

    message(STATUS "Preloaded BLAS: ${BLAS_LIBRARIES}")
    message(STATUS "Preloaded LAPACK: ${LAPACK_LIBRARIES}")

    # Create IMPORTED targets if they don't exist
    if (NOT TARGET BLAS::BLAS)
        add_library(BLAS::BLAS INTERFACE IMPORTED)
        set_target_properties(BLAS::BLAS PROPERTIES
            INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}"
        )
    endif()

    if (NOT TARGET LAPACK::LAPACK)
        add_library(LAPACK::LAPACK INTERFACE IMPORTED)
        set_target_properties(LAPACK::LAPACK PROPERTIES
            INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}"
        )
    endif()
endif()
