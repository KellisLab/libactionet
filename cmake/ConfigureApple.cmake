#[=============================================================================[
ConfigureApple.cmake
================================================================================
Set Mac target architecture and Apple CPU-specific compiler flags.

This module handles architecture detection for both native and cross-compilation
scenarios, including Rosetta emulation. It sets the TARGET_ARCHITECTURE variable
used by other configuration modules.

The following variables are defined:

``TARGET_ARCHITECTURE``
    Canonical target architecture string (arm64, x86_64, etc.) used throughout
    the build system for conditional configuration.

``CMAKE_OSX_ARCHITECTURES`` (set by this module)
    CMake-native macOS architecture specification.

The following variables control the behaviour of this module:

``LIBACTIONET_BUILD_R``
    If building library for R, set target CPU and flags using configuration
    defined by R installation. For R, CMAKE_OSX_ARCHITECTURES must be provided.

``CMAKE_OSX_ARCHITECTURES`` (optional)
    User-defined target CPU architecture. Overrides automatic detection.

]=============================================================================]

macro(CONFIGURE_APPLE libtarget)
    message(NOTICE "Configuring cmake build for macOS")

    # Detect target architecture
    # For R builds, CMAKE_OSX_ARCHITECTURES must be set by the configure script
    if (LIBACTIONET_BUILD_R)
        if (NOT CMAKE_OSX_ARCHITECTURES)
            message(FATAL_ERROR "CMAKE_OSX_ARCHITECTURES must be set when building for R. Please pass it via cmake command line.")
        endif()
        set(TARGET_ARCHITECTURE "${CMAKE_OSX_ARCHITECTURES}")
        message(STATUS "Building for R with architecture: ${TARGET_ARCHITECTURE}")
    # Check if CMAKE_OSX_ARCHITECTURES was set (e.g., from command line or cache)
    elseif (CMAKE_OSX_ARCHITECTURES)
        set(TARGET_ARCHITECTURE "${CMAKE_OSX_ARCHITECTURES}")
        message(STATUS "Using CMAKE_OSX_ARCHITECTURES: ${TARGET_ARCHITECTURE}")
    else ()
        ## Default to current system architecture
        execute_process(
                COMMAND bash -c "uname -m"
                OUTPUT_VARIABLE TARGET_ARCHITECTURE
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(CMAKE_OSX_ARCHITECTURES "${TARGET_ARCHITECTURE}" CACHE STRING "Target architecture for macOS" FORCE)
        message(STATUS "Building for current system architecture: ${TARGET_ARCHITECTURE}")
    endif ()

    # Ensure TARGET_ARCHITECTURE is set
    if (NOT DEFINED TARGET_ARCHITECTURE OR "${TARGET_ARCHITECTURE}" STREQUAL "")
        message(FATAL_ERROR "Architecture not detected. Please set CMAKE_OSX_ARCHITECTURES explicitly.")
    endif()

    # Make TARGET_ARCHITECTURE available to parent scope and other modules
    set(TARGET_ARCHITECTURE "${TARGET_ARCHITECTURE}" CACHE STRING "Target architecture (arm64, x86_64, etc.)" FORCE)

    # Apply architecture-specific compiler flags
    if ("${TARGET_ARCHITECTURE}" MATCHES "arm64" OR "${TARGET_ARCHITECTURE}" MATCHES "aarch64")
        target_compile_options(${libtarget} PUBLIC -flax-vector-conversions)
        message(STATUS "Applied ARM64-specific compiler flags")
    elseif ("${TARGET_ARCHITECTURE}" MATCHES "x86_64")
        message(STATUS "Applied x86_64-specific compiler flags")
    endif ()

    # Set minimum macOS deployment target if not already set
    if (NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target" FORCE)
        message(STATUS "Set CMAKE_OSX_DEPLOYMENT_TARGET to ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
endmacro()
