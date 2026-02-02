#===============================================================================
# CMakeLists.txt for PRIMME (bundled version)
#===============================================================================
# PRIMME: PReconditioned Iterative MultiMethod Eigensolver
# Version: 3.2.3
# License: 3-clause BSD (see COPYING.txt)
# URL: https://github.com/primme/primme
#===============================================================================

# Collect all PRIMME source files
file(GLOB PRIMME_EIGS_SOURCES "eigs/*.c")
file(GLOB PRIMME_SVDS_SOURCES "svds/*.c")
file(GLOB PRIMME_LINALG_SOURCES "linalg/*.c")

# Create static library
add_library(primme_bundled STATIC
    ${PRIMME_EIGS_SOURCES}
    ${PRIMME_SVDS_SOURCES}
    ${PRIMME_LINALG_SOURCES}
)

# Include directories
target_include_directories(primme_bundled PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/extern/primme
)

target_include_directories(primme_bundled PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/extern/primme
)

# Link BLAS/LAPACK (required by PRIMME)
if(BLAS_LIBRARIES)
    target_link_libraries(primme_bundled PUBLIC ${BLAS_LIBRARIES})
endif()

if(LAPACK_LIBRARIES)
    target_link_libraries(primme_bundled PUBLIC ${LAPACK_LIBRARIES})
endif()

# Enable OpenMP if available (PRIMME benefits from it)
if(OpenMP_FOUND OR OpenMP_C_FOUND)
    target_link_libraries(primme_bundled PUBLIC OpenMP::OpenMP_C)
endif()

# Compiler flags for PRIMME
target_compile_options(primme_bundled PRIVATE
    $<$<C_COMPILER_ID:GNU>:-Wno-unused-parameter -Wno-unused-function -Wno-macro-redefined>
    $<$<C_COMPILER_ID:Clang>:-Wno-unused-parameter -Wno-unused-function -Wno-macro-redefined>
    $<$<C_COMPILER_ID:AppleClang>:-Wno-unused-parameter -Wno-unused-function -Wno-macro-redefined>
)

# PRIMME configuration
# Note: PRIMME's internal headers define these macros, so we suppress redefinition warnings above
target_compile_definitions(primme_bundled PRIVATE
    USE_DOUBLE     # Use double precision (PRIMME defines this internally too)
    USE_LONGLONG   # Use 64-bit integers for large matrices
)

# Set properties
set_target_properties(primme_bundled PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
)

# Install headers (optional)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../include/extern/primme/
    DESTINATION include/libactionet/extern/primme
    FILES_MATCHING PATTERN "*.h"
)
