#[=============================================================================[
ConfigureR.cmake
================================================================================
Configure library to use as R package backend.

This module assumes the following modules have been included:

    * ConfigureBLAS.cmake
    * ConfigureApple.cmake

This module conditionally defines the following:

``r_RInclude``
    Required C/C++ header libraries from R installation.

``r_RcppInclude``
    Header libraries used by Rcpp

``r_RcppArmaInclude``
    Header libraries used by RcppArmadillo

``r_CPPFLAGS``
    CPPFLAGS use by R installation and passed to compiler.

``BLAS_LIBARARIES``
    BLAS library location and flags. See FindBLAS cmake documentation

``LAPACK_LIBARARIES``
    LAPACK library location and flags.  See FindBLAS cmake documentation

The following variables control the behaviour of this module:

``LIBACTIONET_BUILD_R``
    Activates module. If defined and true, ``CONFIGURE_R()`` macro.

``R_HOME``
    Directory of system R installation. Environment variable defined by calling configuration script or user.

``BLA_VENDOR``
    Specify vendor for BLAS and LAPACK. Passed to ``CONFIGURE_BLAS()``

]=============================================================================]

macro(CONFIGURE_R_ARCHITECTURE)
    # This is called early to set CMAKE_OSX_ARCHITECTURES before CONFIGURE_APPLE
    message(STATUS "Extracting R target architecture for CMake configuration")
    if (NOT DEFINED R_HOME)
        message(FATAL_ERROR "R_HOME not defined")
    else ()
        message(STATUS "Using R installation: ${R_HOME}")
    endif ()

    ## Extract target architecture from R if on macOS
    if (APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
        message(STATUS "Detecting R target architecture on macOS")

        # Primary method: Detect from R binary using file command
        execute_process(
                COMMAND bash -c "file '${R_HOME}/bin/R' 2>/dev/null | grep -oE '(arm64|aarch64|x86_64|i386)' | head -1"
                OUTPUT_VARIABLE R_BINARY_ARCH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE FILE_CMD_RESULT
        )

        if (R_BINARY_ARCH AND FILE_CMD_RESULT EQUAL 0)
            # Normalize architecture names
            if ("${R_BINARY_ARCH}" MATCHES "aarch64")
                set(R_BINARY_ARCH "arm64")
            endif()
            message(STATUS "R target architecture detected from binary: ${R_BINARY_ARCH}")
            set(CMAKE_OSX_ARCHITECTURES "${R_BINARY_ARCH}" CACHE STRING "Target architecture from R binary" FORCE)
        else()
            # Fallback: Try to detect from R config variables
            execute_process(
                    COMMAND bash -c "${R_HOME}/bin/R CMD config SIZEOF_LONG_DOUBLE 2>/dev/null | head -c 1"
                    OUTPUT_VARIABLE R_SIZEOF_RESULT
                    OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            # Also try OBJECT_MODE which might indicate 64-bit
            execute_process(
                    COMMAND bash -c "${R_HOME}/bin/R CMD config SHLIB_CXXLD 2>/dev/null | grep -oE '(arm64|aarch64|x86_64|i386)' | head -1"
                    OUTPUT_VARIABLE R_COMPILER_ARCH
                    OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            if (R_COMPILER_ARCH)
                if ("${R_COMPILER_ARCH}" MATCHES "aarch64")
                    set(R_COMPILER_ARCH "arm64")
                endif()
                message(STATUS "R target architecture detected from compiler settings: ${R_COMPILER_ARCH}")
                set(CMAKE_OSX_ARCHITECTURES "${R_COMPILER_ARCH}" CACHE STRING "Target architecture from R compiler" FORCE)
            else()
                message(WARNING "Could not automatically detect R target architecture. Attempting system default.")
                # Use current system architecture as fallback
                execute_process(
                        COMMAND bash -c "uname -m"
                        OUTPUT_VARIABLE SYSTEM_ARCH
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                )
                if (SYSTEM_ARCH)
                    message(STATUS "Using system architecture as fallback: ${SYSTEM_ARCH}")
                    set(CMAKE_OSX_ARCHITECTURES "${SYSTEM_ARCH}" CACHE STRING "Target architecture (system default)" FORCE)
                else()
                    message(FATAL_ERROR "Could not detect R target architecture. Please set CMAKE_OSX_ARCHITECTURES explicitly.")
                endif()
            endif()
        endif()
    endif()
endmacro()

macro(CONFIGURE_R)
    message(NOTICE "Configuring for R library")
    if (NOT DEFINED R_HOME)
        message(FATAL_ERROR "R_HOME not defined")
    else ()
        message(STATUS "Using R installation: ${R_HOME}")
    endif ()
    add_compile_definitions(LIBACTIONET_BUILD_R) # Set R build mode in config header

    ## Find R headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/R CMD config --cppflags | sed 's/-I//g'"
            OUTPUT_VARIABLE r_RInclude
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "R library headers: ${r_RInclude}")

    ## Find Rcpp headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(system.file(\"include\", package=\"Rcpp\"))'"
            OUTPUT_VARIABLE r_RcppInclude
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Rcpp headers: ${r_RcppInclude}")

    ## Find RcppArmadillo headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(system.file(\"include\", package=\"RcppArmadillo\"))'"
            OUTPUT_VARIABLE r_RcppArmaInclude
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "RcppArmadillo headers: ${r_RcppArmaInclude}")

    ## Get R CPPFLAGS
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/R CMD config CPPFLAGS"
            OUTPUT_VARIABLE r_CPPFLAGS
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    separate_arguments(r_CPPFLAGS NATIVE_COMMAND ${r_CPPFLAGS})
    message(STATUS "R CPPFLAGS: ${r_CPPFLAGS}")

    ## Set BLAS and LAPACK libraries
    if ((DEFINED BLA_VENDOR) AND (NOT ${BLA_VENDOR} STREQUAL "All")) ## User provided
        message(STATUS "Using user-specified BLA_VENDOR: ${BLA_VENDOR}")
        CONFIGURE_BLAS(actionet)
    else () ## Get BLAS/LAPACK from R
        message(NOTICE "Using BLAS/LAPACK from R")
        ## Find R BLAS_LIBS
        execute_process(
                COMMAND bash -c "${R_HOME}/bin/R CMD config BLAS_LIBS"
                OUTPUT_VARIABLE BLAS_LIBRARIES_RAW
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
#        message(STATUS "R BLAS_LIBS (raw): ${BLAS_LIBRARIES_RAW}")

        ## Find R LAPACK_LIBS
        execute_process(
                COMMAND bash -c "${R_HOME}/bin/R CMD config LAPACK_LIBS"
                OUTPUT_VARIABLE LAPACK_LIBRARIES_RAW
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
#        message(STATUS "R LAPACK_LIBS (raw): ${LAPACK_LIBRARIES_RAW}")

        # Check if R provides valid BLAS/LAPACK libraries
        if (BLAS_LIBRARIES_RAW STREQUAL "" OR LAPACK_LIBRARIES_RAW STREQUAL "")
            message(WARNING "R BLAS/LAPACK libraries are empty. Falling back to system BLAS/LAPACK detection.")
            CONFIGURE_BLAS(actionet)
        else()
            # Parse the R BLAS/LAPACK flags
            separate_arguments(BLAS_LIBRARIES NATIVE_COMMAND ${BLAS_LIBRARIES_RAW})
            separate_arguments(LAPACK_LIBRARIES NATIVE_COMMAND ${LAPACK_LIBRARIES_RAW})

            message(STATUS "R BLAS_LIBS: ${BLAS_LIBRARIES}")
            message(STATUS "R LAPACK_LIBS: ${LAPACK_LIBRARIES}")

            # Mark BLAS/LAPACK as found to prevent redundant searches
            set(BLAS_FOUND TRUE)
            set(LAPACK_FOUND TRUE)

            ## Find BLAS dependencies (headers, etc.)
            CONFIGURE_BLAS_DEPENDS(actionet)
        endif()
    endif ()

    target_include_directories(actionet
            PRIVATE "${r_RInclude}"
            PRIVATE "${r_RcppInclude}"
            PRIVATE "${r_RcppArmaInclude}"
    )
    target_compile_options(actionet PRIVATE ${r_CPPFLAGS})
endmacro()
