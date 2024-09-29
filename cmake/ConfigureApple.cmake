#[=============================================================================[
ConfigureApple.cmake
================================================================================
Set Mac target architecture and Apple CPU=specific compiler flags.

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
    elseif ((DEFINED CMAKE_OSX_ARCHITECTURES) AND (NOT ${CMAKE_OSX_ARCHITECTURES} STREQUAL "")) ## User specific compilation target
        message(STATUS "CMAKE_OSX_ARCHITECTURES set: ${CMAKE_OSX_ARCHITECTURES}")
        set(arch "${CMAKE_OSX_ARCHITECTURES}")
    else () ## Default to current system architecture
        execute_process(
                COMMAND bash -c "uname -m"
                OUTPUT_VARIABLE arch
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        message(STATUS "Building for ${arch}")
    endif ()

    if (${arch} MATCHES "arm64")
        target_compile_options(${libtarget} PUBLIC -flax-vector-conversions)
    endif ()
endmacro()