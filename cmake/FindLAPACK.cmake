# FindLAPACK.cmake wrapper for libactionet
#
# This module wraps CMake's standard FindLAPACK module to handle the special case
# where LAPACK libraries are provided by R and should not be re-searched.
#
# When building as an R package, R provides its own LAPACK libraries via LAPACK_LIBS.
# However, FindSuiteSparse.cmake calls find_package(LAPACK), which would normally
# invoke CMake's FindLAPACK module and fail because R's LAPACK is in a non-standard location.
#
# This wrapper detects if LAPACK has already been configured (via PRELOAD_LAPACK_LIBRARIES)
# and if so, sets all the necessary variables without invoking the standard FindLAPACK search.

# Check if LAPACK is already configured (from R or previous configuration)
if (DEFINED PRELOAD_LAPACK_LIBRARIES AND NOT PRELOAD_LAPACK_LIBRARIES STREQUAL "")
    message(STATUS "FindLAPACK wrapper: Using pre-configured LAPACK from R")

    # Set all the variables that FindLAPACK would set
    set(LAPACK_FOUND TRUE)
    set(LAPACK_LIBRARIES "${PRELOAD_LAPACK_LIBRARIES}")
    set(LAPACK_LINKER_FLAGS "")
    set(LAPACK95_FOUND FALSE)
    set(LAPACK95_LIBRARIES "")

    # Create IMPORTED target if it doesn't exist
    if (NOT TARGET LAPACK::LAPACK)
        add_library(LAPACK::LAPACK INTERFACE IMPORTED)
        set_target_properties(LAPACK::LAPACK PROPERTIES
            INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}"
        )
    endif()

    # Mark the find operation as successful
    set(LAPACK_FIND_QUIETLY TRUE)
    set(LAPACK_FIND_REQUIRED FALSE)

    message(STATUS "FindLAPACK wrapper: LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}")
else()
    # No pre-configured LAPACK, fall back to CMake's standard FindLAPACK module
    message(STATUS "FindLAPACK wrapper: No pre-configured LAPACK, using standard FindLAPACK")

    # Save our module path
    set(_SAVED_CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}")

    # Remove our cmake/ directory from CMAKE_MODULE_PATH to avoid recursion
    list(REMOVE_ITEM CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

    # Call the standard FindLAPACK module
    include(FindLAPACK)

    # Restore module path
    set(CMAKE_MODULE_PATH "${_SAVED_CMAKE_MODULE_PATH}")
endif()
