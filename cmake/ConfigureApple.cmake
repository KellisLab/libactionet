#[=============================================================================[
ConfigureApple.cmake
================================================================================
Set Mac target architecture and Apple CPU-specific compiler flags.

The following variables control the behaviour of this module:

``LIBACTIONET_BUILD_R``
    If building library for R, set target CPU and flags using configuration defined by R installation.

``CMAKE_OSX_ARCHITECTURES`` (optional)
    User-defined target CPU architecture.

]=============================================================================]

macro(CONFIGURE_APPLE libtarget)
    message(NOTICE "Configuring cmake build for macOS")

    # For R builds, CMAKE_OSX_ARCHITECTURES must be set by the configure script
    if (LIBACTIONET_BUILD_R)
        if (NOT CMAKE_OSX_ARCHITECTURES)
            message(FATAL_ERROR "CMAKE_OSX_ARCHITECTURES must be set when building for R. Please pass it via cmake command line.")
        endif()
        set(arch "${CMAKE_OSX_ARCHITECTURES}")
        message(STATUS "Building for R with architecture: ${arch}")
    # Check if CMAKE_OSX_ARCHITECTURES was set (e.g., from command line or cache)
    elseif (CMAKE_OSX_ARCHITECTURES)
        set(arch "${CMAKE_OSX_ARCHITECTURES}")
        message(STATUS "Using CMAKE_OSX_ARCHITECTURES: ${arch}")
    else () ## Default to current system architecture
        execute_process(
                COMMAND bash -c "uname -m"
                OUTPUT_VARIABLE arch
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(CMAKE_OSX_ARCHITECTURES "${arch}" CACHE STRING "Target architecture for macOS" FORCE)
        message(STATUS "Building for current system architecture: ${arch}")
    endif ()

    # Ensure arch is set
    if (NOT DEFINED arch OR "${arch}" STREQUAL "")
        message(FATAL_ERROR "Architecture not detected. Please set CMAKE_OSX_ARCHITECTURES explicitly.")
    endif()

    # Apply architecture-specific compiler flags
    if ("${arch}" MATCHES "arm64" OR "${arch}" MATCHES "aarch64")
        target_compile_options(${libtarget} PUBLIC -flax-vector-conversions)
        message(STATUS "Applied ARM64-specific compiler flags")
    elseif ("${arch}" MATCHES "x86_64")
        message(STATUS "Applied x86_64-specific compiler flags")
    endif ()

    # Set minimum macOS deployment target if not already set
    if (NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target" FORCE)
        message(STATUS "Set CMAKE_OSX_DEPLOYMENT_TARGET to ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
endmacro()
