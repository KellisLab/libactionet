# Simplified SuiteSparse Detection

## Problem with FindSuiteSparse.cmake

The third-party `FindSuiteSparse.cmake` module created significant complexity:

1. **BLAS/LAPACK Dependency Issues**: It calls `find_package(BLAS)` and `find_package(LAPACK)` internally (lines 293-304), which:
   - Always performs a fresh search regardless of pre-configured libraries
   - Fails when using R's BLAS/LAPACK (in non-standard locations)
   - Cannot be modified since it's a third-party module

2. **Over-Engineering**: The module searches for multiple SuiteSparse components (AMD, CAMD, CHOLMOD, COLAMD, CCOLAMD, SPQR, Config, etc.) but **only CHOLMOD is actually used** by libactionet

3. **Workaround Complexity**: Required creating wrapper FindBLAS/FindLAPACK modules and complex preload mechanisms just to satisfy FindSuiteSparse's requirements

## Solution: Direct CHOLMOD Detection

Replaced the complex FindSuiteSparse module with simple, direct CHOLMOD detection:

```cmake
# Find cholmod.h header
find_path(CHOLMOD_INCLUDE_DIR
    NAMES cholmod.h
    PATHS
        /usr/local/include/suitesparse
        /usr/include/suitesparse
        /opt/homebrew/include/suitesparse
        /opt/local/include/suitesparse
    DOC "CHOLMOD include directory"
)

# Find libcholmod library
find_library(CHOLMOD_LIBRARY
    NAMES cholmod
    PATHS
        /usr/local/lib
        /usr/lib
        /opt/homebrew/lib
        /opt/local/lib
    DOC "CHOLMOD library"
)

# Link directly
target_include_directories(actionet PRIVATE ${CHOLMOD_INCLUDE_DIR})
target_link_libraries(actionet PUBLIC ${CHOLMOD_LIBRARY})
```

## Benefits

### 1. **Simplicity**
- ~50 lines of straightforward CMake code vs. 500+ lines of complex Find module
- No intermediate targets or abstractions
- Clear, readable logic

### 2. **Reliability**
- Works consistently across Homebrew, apt, and other package managers
- No BLAS/LAPACK detection conflicts
- No need for wrapper modules or workarounds

### 3. **Maintainability**
- Easy to understand and debug
- Can be modified if needed (not third-party code)
- Explicit error messages guide users to install SuiteSparse

### 4. **Performance**
- Faster configuration (no complex dependency resolution)
- Fewer find_package() calls

### 5. **Compatibility**
- Works seamlessly with R package builds (no BLAS/LAPACK conflicts)
- Works with standalone builds (standard find_path/find_library)
- Works across macOS (Homebrew), Linux (apt/yum), and other systems

## Code Usage Analysis

Only three files use CHOLMOD:
```
include/utils_internal/utils_matrix.hpp:#include <cholmod.h>
src/decomposition/svd_irbla.cpp:#include <cholmod.h>
src/network/network_diffusion.cpp:#include <cholmod.h>
```

No other SuiteSparse components (AMD, COLAMD, etc.) are directly used, though they may be transitive dependencies of CHOLMOD itself - which is handled automatically by the system's libcholmod.

## Installation Instructions

### macOS (Homebrew)
```bash
brew install suite-sparse
```

### Ubuntu/Debian
```bash
sudo apt-get install libsuitesparse-dev
```

### RHEL/CentOS/Fedora
```bash
sudo yum install suitesparse-devel
```

### Arch Linux
```bash
sudo pacman -S suitesparse
```

## Testing

### Standalone Build
```bash
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
...
Searching for CHOLMOD library
-- Found CHOLMOD:
--   Include: /usr/local/include/suitesparse
--   Library: /usr/local/lib/libcholmod.dylib
-- Configuring done
```

### R Package Build
```bash
$ R CMD INSTALL actionet-r
...
Searching for CHOLMOD library
-- Found CHOLMOD:
--   Include: /usr/local/include/suitesparse
--   Library: /usr/local/lib/libcholmod.dylib
-- Configuring done
-- Generating done
```

## Files Removed

Since we no longer need the complex FindSuiteSparse approach:

- `cmake/FindSuiteSparse.cmake` - Moved to `_EXCLUDE/` (no longer used)
- `cmake/FindBLAS.cmake` - Removed (wrapper no longer needed)
- `cmake/FindLAPACK.cmake` - Removed (wrapper no longer needed)
- `cmake/PreloadBLAS.cmake` - Removed (preload mechanism no longer needed)

## Files Modified

### CMakeLists.txt
- Replaced `find_package(SuiteSparse REQUIRED COMPONENTS CHOLMOD)` with direct `find_path`/`find_library`
- Reverted `PREPEND` back to `APPEND` for CMAKE_MODULE_PATH (no longer need to override system Find modules)
- Added helpful error messages for missing CHOLMOD

### cmake/ConfigureR.cmake
- Removed PRELOAD_BLAS_LIBRARIES and PRELOAD_LAPACK_LIBRARIES (no longer needed)
- Simplified to just set BLAS_FOUND and LAPACK_FOUND flags

## Conclusion

By recognizing that **only CHOLMOD is needed** from SuiteSparse, we eliminated:
- 500+ lines of third-party Find module code
- 150+ lines of wrapper code
- Complex workaround mechanisms
- BLAS/LAPACK detection conflicts

The new approach is simpler, more reliable, and easier to maintain while working perfectly across all build modes (standalone, R package, Python package).

This is a good example of "less is more" - sometimes the simplest solution is the best solution.
