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
    if (LIBACTIONET_BUILD_R) ## Detect R architecture on Apple systems
        execute_process(
                COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(R.version[[\"arch\"]])'"
                OUTPUT_VARIABLE arch
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        message(STATUS "Building for architecture of R installation: ${arch}")
        # Set CMAKE_OSX_ARCHITECTURES if not already set
        if (NOT DEFINED CMAKE_OSX_ARCHITECTURES OR CMAKE_OSX_ARCHITECTURES STREQUAL "")
            set(CMAKE_OSX_ARCHITECTURES "${arch}" CACHE STRING "Target architecture for macOS" FORCE)
        endif()
    elseif ((DEFINED CMAKE_OSX_ARCHITECTURES) AND (NOT ${CMAKE_OSX_ARCHITECTURES} STREQUAL "")) ## User specific compilation target
        message(STATUS "CMAKE_OSX_ARCHITECTURES set: ${CMAKE_OSX_ARCHITECTURES}")
        set(arch "${CMAKE_OSX_ARCHITECTURES}")
    else () ## Default to current system architecture
        execute_process(
                COMMAND bash -c "uname -m"
                OUTPUT_VARIABLE arch
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        message(STATUS "Building for current system architecture: ${arch}")
        set(CMAKE_OSX_ARCHITECTURES "${arch}" CACHE STRING "Target architecture for macOS" FORCE)
    endif ()

    # Apply architecture-specific compiler flags
    if (${arch} MATCHES "arm64")
        target_compile_options(${libtarget} PUBLIC -flax-vector-conversions)
        message(STATUS "Applied ARM64-specific compiler flags")
    elseif (${arch} MATCHES "x86_64")
        message(STATUS "Applied x86_64-specific compiler flags")
    endif ()

    # Set minimum macOS deployment target if not already set
    if (NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target" FORCE)
        message(STATUS "Set CMAKE_OSX_DEPLOYMENT_TARGET to ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
endmacro()
