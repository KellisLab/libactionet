#[=======================================================================[.rst:
ConfigureBLAS.cmake
===============

Configure BLAS and LAPACK for libactionet

This module wraps `FindBLAS` defines all its variables.

This module conditionally defines the following:

``LIBACTIONET_BLAS_MKL``
    ``TRUE`` iff SuiteSparse and all dependencies have been found.

``LIBACTIONET_BLAS_ACCELERATE``
    Extracted from ``SuiteSparse_config.h`` (>= v4).

``BLAS_HEADERS_USE``
Equal to 4 if ``SuiteSparse_VERSION`` = 4.2.1


The following variables control the behaviour of this module:

``BLA_VENDOR``
    Do not attempt to use the native SuiteSparse CMake package configuration.

]=======================================================================]

macro(CONFIGURE_BLAS_MKL)
    message(STATUS "Using Intel MKL for BLAS and LAPACK")

    ## Check if MKL is active
    if (NOT DEFINED ENV{MKLROOT})
        message(FATAL_ERROR "Using Intel MKL but `MKLROOT` not defined")
    endif ()

    ## Find optional MKL headers
    set(BLAS_HEADERS_USE "$ENV{MKLROOT}/include")
    if (EXISTS "${BLAS_HEADERS_USE}")
        message(STATUS "MKL headers: ${BLAS_HEADERS_USE}")
        add_compile_definitions(LIBACTIONET_BLAS_MKL)
    else ()
        message("Not using MKL headers")
    endif ()
endmacro()

macro(CONFIGURE_BLAS_ACCELERATE)
    if (NOT APPLE)
        message(FATAL_ERROR "Accelerate can only be used on macOS")
    endif ()

    ## Find and link Apple BLAS
    message(STATUS "Using Apple Accelerate for BLAS and LAPACK")
    add_compile_definitions(LIBACTIONET_BLAS_ACCELERATE)

    ## Find required Accelerate headers
    set(BLAS_HEADERS_USE "${BLAS_LIBRARIES}/Frameworks/vecLib.framework/Headers")
    if (NOT EXISTS "${BLAS_HEADERS_USE}")
        message(FATAL_ERROR "Cannot locate Apple Accelerate headers")
    endif ()
    message(STATUS "Accelerate headers: ${BLAS_HEADERS_USE}")

    ## Set required compiler options
    add_compile_options(-framework Accelerate)
    #    TODO: Probably unneeded. Suppress deprecation warnings.
    add_compile_definitions(ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
endmacro()

macro(CONFIGURE_BLAS_DEPENDS)
    if ((DEFINED BLA_VENDOR) AND (NOT ${BLA_VENDOR} STREQUAL "All"))
        message(STATUS "Using provided BLA_VENDOR")
        if ("${BLA_VENDOR}" MATCHES "Intel")
            CONFIGURE_BLAS_MKL()
        elseif ("${BLA_VENDOR}" STREQUAL "Apple")
            CONFIGURE_BLAS_ACCELERATE()
        endif ()
    else ()
        message(STATUS "Detecting BLAS implementation from detected BLAS_LIBRARIES")
        foreach (lib ${BLAS_LIBRARIES})
            if (${lib} MATCHES "mkl")
                CONFIGURE_BLAS_MKL()
                break()
            elseif (${lib} MATCHES "Accelerate\.framework")
                CONFIGURE_BLAS_ACCELERATE()
                break()
            endif ()
        endforeach ()
    endif ()

    if (DEFINED BLAS_HEADERS_USE)
        target_include_directories(
                actionet
                PRIVATE "${BLAS_HEADERS_USE}"
        )
    endif ()

endmacro()

## Configure BLAS/LAPACK
macro(CONFIGURE_BLAS)
    message(NOTICE "Configuring BLAS/LAPACK")
    find_package(BLAS REQUIRED) ## Find BLAS
    find_package(LAPACK REQUIRED) ## Find LAPACK

    CONFIGURE_BLAS_DEPENDS()
endmacro()