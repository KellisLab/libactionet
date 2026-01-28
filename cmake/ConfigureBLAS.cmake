#[=============================================================================[
ConfigureBLAS.cmake
================================================================================
Configure BLAS and LAPACK for libactionet.

This module wraps the `FindBLAS` module and defines all its variables.

This module conditionally defines the following:

``LIBACTIONET_BLAS_MKL``
    Preprocessor flag passed to `libactionet_config.hpp` to use Intel MKL.

``LIBACTIONET_BLAS_ACCELERATE``
    Preprocessor flag passed to `libactionet_config.hpp` to use Apple Accelerate.

``BLAS_HEADERS_USE``
    Location of required CBLAS headers for Intel MKL and Apple Accelerate. Unset if other BLAS is used.

The following variables control the behaviour of this module:

``BLA_VENDOR`` (optional)
    Specify vendor for BLAS and LAPACK

]=============================================================================]

## Configure Intel MKL
macro(CONFIGURE_BLAS_MKL libtarget)
    message(STATUS "Using Intel MKL for BLAS and LAPACK")

    ## Check if MKL is active
    if (NOT DEFINED ENV{MKLROOT})
        message(NOTICE "Using Intel MKL but `MKLROOT` not defined")
    endif ()

    ## Find optional MKL headers
    set(BLAS_HEADERS_USE "$ENV{MKLROOT}/include")
    if (EXISTS "${BLAS_HEADERS_USE}")
        message(STATUS "MKL headers: ${BLAS_HEADERS_USE}")
        ## Enable MKL in libactionet_config.hpp
        target_compile_definitions(${libtarget} PUBLIC LIBACTIONET_BLAS_MKL)
    else ()
        message(WARNING "MKL headers not found at ${BLAS_HEADERS_USE}")
        unset(BLAS_HEADERS_USE)
    endif ()
endmacro()

## Configure Apple Accelerate
macro(CONFIGURE_BLAS_ACCELERATE libtarget)
    if (NOT APPLE)
        message(FATAL_ERROR "Accelerate can only be used on macOS")
    endif ()

    message(STATUS "Using Apple Accelerate for BLAS and LAPACK")
    ## Enable Accelerate in libactionet_config.hpp
    target_compile_definitions(${libtarget} PUBLIC LIBACTIONET_BLAS_ACCELERATE)

    ## Find required Accelerate headers
    # Try multiple possible locations for Accelerate headers
    set(ACCELERATE_HEADER_SEARCH_PATHS
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers"
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers"
        "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers"
    )

    # Also check for SDK paths dynamically
    if (CMAKE_OSX_SYSROOT)
        list(APPEND ACCELERATE_HEADER_SEARCH_PATHS
            "${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers")
    endif()

    unset(BLAS_HEADERS_USE)
    foreach(header_path ${ACCELERATE_HEADER_SEARCH_PATHS})
        if (EXISTS "${header_path}/cblas.h")
            set(BLAS_HEADERS_USE "${header_path}")
            break()
        endif()
    endforeach()

    if (NOT BLAS_HEADERS_USE)
        message(FATAL_ERROR "Cannot locate Apple Accelerate cblas.h header. Searched paths: ${ACCELERATE_HEADER_SEARCH_PATHS}")
    endif ()
    message(STATUS "Accelerate headers: ${BLAS_HEADERS_USE}")

    ## Set required compiler/linker options
    target_link_options(${libtarget} PUBLIC "-framework" "Accelerate")
endmacro()

## Find dependencies for MKL and Accelerate
macro(CONFIGURE_BLAS_DEPENDS libtarget)
    if ((DEFINED BLA_VENDOR) AND (NOT ${BLA_VENDOR} STREQUAL "All"))
        ## Find dependencies for user-specified BLAS
        message(STATUS "Using provided BLA_VENDOR: ${BLA_VENDOR}")
        if ("${BLA_VENDOR}" MATCHES "Intel")
            CONFIGURE_BLAS_MKL(${libtarget})
        elseif ("${BLA_VENDOR}" STREQUAL "Apple")
            CONFIGURE_BLAS_ACCELERATE(${libtarget})
        endif ()
    else ()
        ## Find dependencies based on BLAS link line pattern
        message(STATUS "Detecting BLAS implementation from BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
        set(BLAS_VENDOR_DETECTED FALSE)
        foreach (lib ${BLAS_LIBRARIES})
            if (${lib} MATCHES "mkl")
                CONFIGURE_BLAS_MKL(${libtarget})
                set(BLAS_VENDOR_DETECTED TRUE)
                break()
            elseif ((${lib} MATCHES "Accelerate") OR (${lib} MATCHES "vecLib"))
                CONFIGURE_BLAS_ACCELERATE(${libtarget})
                set(BLAS_VENDOR_DETECTED TRUE)
                break()
            endif ()
        endforeach ()

        if (NOT BLAS_VENDOR_DETECTED)
            message(STATUS "Using generic BLAS/LAPACK (no vendor-specific optimizations)")
            # For generic BLAS, try to find cblas.h in common locations
            set(GENERIC_CBLAS_SEARCH_PATHS
                /usr/include
                /usr/local/include
                /usr/include/openblas
                /usr/local/opt/openblas/include
                /opt/local/include
            )

            # Conda / custom prefixes (rootless installs)
            set(BLAS_PREFIX_CANDIDATES "")
            if (DEFINED ENV{CONDA_PREFIX})
                list(APPEND BLAS_PREFIX_CANDIDATES "$ENV{CONDA_PREFIX}")
            endif()
            if (DEFINED ENV{OPENBLAS_ROOT})
                list(APPEND BLAS_PREFIX_CANDIDATES "$ENV{OPENBLAS_ROOT}")
            endif()
            if (DEFINED ENV{BLAS_ROOT})
                list(APPEND BLAS_PREFIX_CANDIDATES "$ENV{BLAS_ROOT}")
            endif()
            if (DEFINED ENV{LAPACK_ROOT})
                list(APPEND BLAS_PREFIX_CANDIDATES "$ENV{LAPACK_ROOT}")
            endif()

            foreach(prefix IN LISTS BLAS_PREFIX_CANDIDATES)
                list(INSERT GENERIC_CBLAS_SEARCH_PATHS 0 "${prefix}/include/openblas")
                list(INSERT GENERIC_CBLAS_SEARCH_PATHS 0 "${prefix}/include")
                list(INSERT GENERIC_CBLAS_SEARCH_PATHS 0 "${prefix}/include/cblas")
            endforeach()

            # Add architecture-specific Homebrew paths if on macOS
            if (DEFINED TARGET_ARCHITECTURE)
                if ("${TARGET_ARCHITECTURE}" MATCHES "arm64" OR "${TARGET_ARCHITECTURE}" MATCHES "aarch64")
                    list(INSERT GENERIC_CBLAS_SEARCH_PATHS 0 /opt/homebrew/opt/openblas/include)
                elseif ("${TARGET_ARCHITECTURE}" MATCHES "x86_64")
                    list(INSERT GENERIC_CBLAS_SEARCH_PATHS 0 /usr/local/opt/openblas/include)
                endif()
            else()
                # Fallback: add both Homebrew paths
                list(APPEND GENERIC_CBLAS_SEARCH_PATHS
                    /opt/homebrew/opt/openblas/include)
            endif()

            find_path(GENERIC_CBLAS_INCLUDE_DIR
                NAMES cblas.h
                PATHS ${GENERIC_CBLAS_SEARCH_PATHS}
                PATH_SUFFIXES cblas
            )
            if (GENERIC_CBLAS_INCLUDE_DIR)
                set(BLAS_HEADERS_USE "${GENERIC_CBLAS_INCLUDE_DIR}")
                message(STATUS "Found generic cblas.h at: ${BLAS_HEADERS_USE}")
            else()
                message(WARNING "Could not find cblas.h header. Build may fail if code uses CBLAS directly.")
            endif()
        endif()
    endif ()

    ## Include BLAS headers if found
    if (DEFINED BLAS_HEADERS_USE AND BLAS_HEADERS_USE)
        target_include_directories(
                ${libtarget}
                PRIVATE "${BLAS_HEADERS_USE}"
        )
        message(STATUS "Added BLAS header include directory: ${BLAS_HEADERS_USE}")
    endif ()
endmacro()

## Configure BLAS/LAPACK
macro(CONFIGURE_BLAS libtarget)
    message(NOTICE "Configuring BLAS/LAPACK")

    # Store the BLA_VENDOR for potential retry
    set(_SAVED_BLA_VENDOR "${BLA_VENDOR}")

    find_package(BLAS QUIET) ## Find BLAS
    find_package(LAPACK QUIET) ## Find LAPACK

    if (NOT BLAS_FOUND OR NOT LAPACK_FOUND)
        message(WARNING "Initial BLAS/LAPACK detection failed. Attempting with BLA_VENDOR=All")
        set(BLA_VENDOR "All")
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
    else()
        message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
        message(STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
    endif()

    # Restore BLA_VENDOR
    set(BLA_VENDOR "${_SAVED_BLA_VENDOR}")

    CONFIGURE_BLAS_DEPENDS(${libtarget})
endmacro()
