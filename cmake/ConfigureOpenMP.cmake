#[=============================================================================[
ConfigureOpenMP.cmake
================================================================================
Detect and configure OpenMP support for libactionet.

This module handles cross-platform OpenMP detection with special handling for
macOS Homebrew installations, which are keg-only and not auto-discoverable.

The following variables control the behaviour of this module:

``TARGET_ARCHITECTURE``
    Target CPU architecture (arm64, x86_64, aarch64, etc.). Used to determine
    correct Homebrew installation paths.

]=============================================================================]

macro(CONFIGURE_OPENMP libtarget)
    message(NOTICE "Configuring OpenMP support")

    if (APPLE)
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
    if (NOT APPLE AND NOT OpenMP_FOUND)
        find_package(OpenMP QUIET)
    endif()

    if (OpenMP_FOUND)
        # If we didn't already link (Homebrew case), do it here for standard CMake OpenMP targets
        if (NOT APPLE OR NOT TARGET OpenMP::OpenMP_CXX)
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

