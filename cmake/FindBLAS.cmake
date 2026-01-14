# FindBLAS.cmake wrapper for libactionet
#
# This module wraps CMake's standard FindBLAS module to handle the special case
# where BLAS libraries are provided by R and should not be re-searched.
#
# When building as an R package, R provides its own BLAS libraries via BLAS_LIBS.
# However, FindSuiteSparse.cmake calls find_package(BLAS), which would normally
# invoke CMake's FindBLAS module and fail because R's BLAS is in a non-standard location.
#
# This wrapper detects if BLAS has already been configured (via PRELOAD_BLAS_LIBRARIES)
# and if so, sets all the necessary variables without invoking the standard FindBLAS search.

# Check if BLAS/LAPACK are already configured (from R or previous configuration)
if (DEFINED PRELOAD_BLAS_LIBRARIES AND NOT PRELOAD_BLAS_LIBRARIES STREQUAL "")
    message(STATUS "FindBLAS wrapper: Using pre-configured BLAS from R")

    # Set all the variables that FindBLAS would set
    set(BLAS_FOUND TRUE)
    set(BLAS_LIBRARIES "${PRELOAD_BLAS_LIBRARIES}")
    set(BLAS_LINKER_FLAGS "")
    set(BLAS95_FOUND FALSE)
    set(BLAS95_LIBRARIES "")

    # Create IMPORTED target if it doesn't exist
    if (NOT TARGET BLAS::BLAS)
        add_library(BLAS::BLAS INTERFACE IMPORTED)
        set_target_properties(BLAS::BLAS PROPERTIES
            INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}"
        )
    endif()

    # Mark the find operation as successful
    set(BLAS_FIND_QUIETLY TRUE)
    set(BLAS_FIND_REQUIRED FALSE)

    message(STATUS "FindBLAS wrapper: BLAS_LIBRARIES = ${BLAS_LIBRARIES}")
else()
    # No pre-configured BLAS, fall back to CMake's standard FindBLAS module
    message(STATUS "FindBLAS wrapper: No pre-configured BLAS, using standard FindBLAS")

    # Save our module path
    set(_SAVED_CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}")

    # Remove our cmake/ directory from CMAKE_MODULE_PATH to avoid recursion
    list(REMOVE_ITEM CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

    # Call the standard FindBLAS module
    include(FindBLAS)

    # Restore module path
    set(CMAKE_MODULE_PATH "${_SAVED_CMAKE_MODULE_PATH}")
endif()
