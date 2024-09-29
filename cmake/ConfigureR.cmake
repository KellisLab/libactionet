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
            COMMAND bash -c "${R_HOME}/bin/R CMD config --cppflags | sed s/-I//g"
            OUTPUT_VARIABLE r_RInclude
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "R library headers: ${r_RInclude}")

    ## Find Rcpp headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(system.file(\"include\", package=\"Rcpp\"))'"
            OUTPUT_VARIABLE r_RcppInclude
    )
    message(STATUS "Rcpp headers: ${r_RcppInclude}")

    ## Find RcppArmadillo headers
    execute_process(
            COMMAND bash -c "${R_HOME}/bin/Rscript -e 'cat(system.file(\"include\", package=\"RcppArmadillo\"))'"
            OUTPUT_VARIABLE r_RcppArmaInclude
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
        CONFIGURE_BLAS()
    else () ## Get BLAS/LAPACK from R
        message(NOTICE "Using BLAS/LAPACK from R")
        ## Find R BLAS_LIBS
        execute_process(
                COMMAND bash -c "${R_HOME}/bin/R CMD config BLAS_LIBS | sed s/-I//g"
                OUTPUT_VARIABLE BLAS_LIBRARIES
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        separate_arguments(BLAS_LIBRARIES NATIVE_COMMAND ${BLAS_LIBRARIES})
        message(STATUS "R BLAS_LIBS: ${BLAS_LIBRARIES}")
        ## Find R LAPACK_LIBS
        execute_process(
                COMMAND bash -c "${R_HOME}/bin/R CMD config LAPACK_LIBS | sed s/-I//g"
                OUTPUT_VARIABLE LAPACK_LIBRARIES
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        separate_arguments(LAPACK_LIBRARIES NATIVE_COMMAND ${LAPACK_LIBRARIES})
        message(STATUS "R LAPACK_LIBS: ${LAPACK_LIBRARIES}")
        ## Find BLAS dependencies
        CONFIGURE_BLAS_DEPENDS(actionet)
    endif ()

    target_include_directories(actionet
            PRIVATE "${r_RInclude}"
            PRIVATE "${r_RcppInclude}"
            PRIVATE "${r_RcppArmaInclude}"
    )
    target_compile_options(actionet PRIVATE ${r_CPPFLAGS})
endmacro()