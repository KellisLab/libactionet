#[=============================================================================[
ConfigurePRIMME.cmake
================================================================================
Configure PRIMME (PReconditioned Iterative MultiMethod Eigensolver) library
for libactionet.

PRIMME provides SVD capabilities for large sparse matrices with 64-bit integer
support, handling matrices with >2^31 non-zero elements.

Note: PRIMME is disabled for R builds because R is limited to 32-bit matrix
indices (max 2^31-1 elements). R users cannot create matrices large enough
to require PRIMME.

The following variables are used:

``LIBACTIONET_BUILD_R``
    If set to 1, PRIMME is excluded from the build

]=============================================================================]

macro(CONFIGURE_PRIMME libtarget)
    ## Add PRIMME sources directly to target (skip for R builds)
    if (NOT LIBACTIONET_BUILD_R)
        message(STATUS "Configuring PRIMME_SVDS for large sparse matrix support")

        # Collect PRIMME source files
        file(GLOB PRIMME_EIGS_SOURCES ${actionet_SOURCE_DIR}/src/extern/primme/eigs/*.c)
        file(GLOB PRIMME_SVDS_SOURCES ${actionet_SOURCE_DIR}/src/extern/primme/svds/*.c)
        file(GLOB PRIMME_LINALG_SOURCES ${actionet_SOURCE_DIR}/src/extern/primme/linalg/*.c)

        # Add PRIMME sources directly to the target
        target_sources(${libtarget} PRIVATE
            ${PRIMME_EIGS_SOURCES}
            ${PRIMME_SVDS_SOURCES}
            ${PRIMME_LINALG_SOURCES}
        )

        # Add PRIMME include directory (contains both public API and internal headers)
        target_include_directories(${libtarget} PRIVATE
            "${actionet_SOURCE_DIR}/src/extern/primme/include"
        )

        # PRIMME compile options: suppress warnings and define macros
        set_source_files_properties(
            ${PRIMME_EIGS_SOURCES} ${PRIMME_SVDS_SOURCES} ${PRIMME_LINALG_SOURCES}
            PROPERTIES
            COMPILE_OPTIONS "-Wno-unused-parameter;-Wno-unused-function;-Wno-macro-redefined"
            COMPILE_DEFINITIONS "USE_DOUBLE;USE_LONGLONG"
        )

        message(STATUS "PRIMME_SVDS enabled: sources added to ${libtarget}")
        message(STATUS "  EIGS sources: ${PRIMME_EIGS_SOURCES}")
        message(STATUS "  SVDS sources: ${PRIMME_SVDS_SOURCES}")
        message(STATUS "  LINALG sources: ${PRIMME_LINALG_SOURCES}")
    else()
        message(STATUS "PRIMME_SVDS disabled for R build")
        message(STATUS "  Reason: R is limited to 32-bit matrix indices (max 2^31-1 elements)")
    endif()
endmacro()
