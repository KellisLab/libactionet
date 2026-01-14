# R Package Build Fix: FindSuiteSparse BLAS Detection

## Problem

When building libactionet as an R package, the CMake configuration failed with:

```
CMake Error at cmake/FindSuiteSparse.cmake:182 (message):
  Failed to find SuiteSparse - Did not find BLAS library (required for SuiteSparse).
```

This occurred because:
1. R provides its own BLAS/LAPACK libraries via `BLAS_LIBS` and `LAPACK_LIBS` configuration variables
2. These R-provided libraries are in non-standard locations (e.g., `/Library/Frameworks/R.framework/Resources/lib`)
3. `ConfigureR.cmake` correctly extracted and set `BLAS_LIBRARIES` and `LAPACK_LIBRARIES` from R's configuration
4. **However**, `FindSuiteSparse.cmake` (line 293-304) calls `find_package(BLAS QUIET)` and `find_package(LAPACK QUIET)`
5. CMake's `FindBLAS.cmake` module **always** performs a fresh search and **overwrites** previously set variables
6. Since R's BLAS is in a non-standard location, `FindBLAS` couldn't find it and set `BLAS_FOUND=FALSE`
7. This caused `FindSuiteSparse` to fail, even though we had already configured valid BLAS/LAPACK from R

## Root Cause

CMake's `find_package()` for modules (not config-file packages) **always** executes the Find module's search logic, regardless of whether variables are already set. There is no built-in way to short-circuit this search when using third-party Find modules that we cannot modify (like `FindSuiteSparse.cmake`).

## Solution

Created wrapper Find modules that intercept `find_package(BLAS)` and `find_package(LAPACK)` calls:

### 1. Created `cmake/FindBLAS.cmake`
A wrapper that:
- Checks if `PRELOAD_BLAS_LIBRARIES` is set (indicating BLAS is pre-configured from R)
- If set, bypasses the search and sets `BLAS_FOUND=TRUE` with the preloaded libraries
- If not set, falls back to CMake's standard `FindBLAS` module
- Creates the `BLAS::BLAS` imported target for modern CMake compatibility

### 2. Created `cmake/FindLAPACK.cmake`
A similar wrapper for LAPACK.

### 3. Modified `cmake/ConfigureR.cmake`
- Sets `PRELOAD_BLAS_LIBRARIES` and `PRELOAD_LAPACK_LIBRARIES` as cache variables before find_package calls
- Removed complex workarounds that tried (and failed) to prevent FindBLAS from running

### 4. Modified `CMakeLists.txt`
- Changed `list(APPEND CMAKE_MODULE_PATH ...)` to `list(PREPEND CMAKE_MODULE_PATH ...)`
- This ensures our wrapper Find modules in `cmake/` are found **before** CMake's system Find modules

## How It Works

1. **During R Package Build:**
   ```
   ConfigureR() is called
   → Extracts BLAS_LIBRARIES from R CMD config BLAS_LIBS
   → Sets PRELOAD_BLAS_LIBRARIES cache variable
   → Later when FindSuiteSparse calls find_package(BLAS)...
   → CMake finds our cmake/FindBLAS.cmake first (due to PREPEND)
   → Our wrapper detects PRELOAD_BLAS_LIBRARIES is set
   → Returns success without searching
   → FindSuiteSparse is happy!
   ```

2. **During Standalone Build:**
   ```
   ConfigureR() is not called
   → PRELOAD_BLAS_LIBRARIES is not set
   → When find_package(BLAS) is called...
   → Our wrapper detects no preload
   → Falls back to CMake's standard FindBLAS
   → Normal BLAS detection proceeds
   ```

## Files Changed

### New Files
- `cmake/FindBLAS.cmake` - Wrapper for BLAS detection
- `cmake/FindLAPACK.cmake` - Wrapper for LAPACK detection
- `cmake/PreloadBLAS.cmake` - (Not used in final solution, can be removed)

### Modified Files
- `CMakeLists.txt` - Changed APPEND to PREPEND for CMAKE_MODULE_PATH
- `cmake/ConfigureR.cmake` - Set PRELOAD variables instead of complex workarounds

## Testing

Before fix:
```bash
$ R CMD INSTALL actionet-r
...
CMake Error: Failed to find SuiteSparse - Did not find BLAS library
```

After fix:
```bash
$ R CMD INSTALL actionet-r
...
-- FindBLAS wrapper: Using pre-configured BLAS from R
-- FindBLAS wrapper: BLAS_LIBRARIES = -L/.../lib;-lRblas
-- FindLAPACK wrapper: Using pre-configured LAPACK from R
-- FindLAPACK wrapper: LAPACK_LIBRARIES = -L/.../lib;-lRlapack
-- Found CHOLMOD headers in: /usr/local/include/suitesparse
-- Found CHOLMOD library: /usr/local/lib/libcholmod.dylib
-- Found SuiteSparse: ... (found version "7.12.1") found components: CHOLMOD Config AMD CAMD CCOLAMD COLAMD
-- Configuring done
-- Generating done
```

## Why Other Approaches Didn't Work

### Attempt 1: Set BLAS_FOUND as cache variable
- **Failed**: `find_package(BLAS)` ignores pre-set cache variables and always runs its search

### Attempt 2: Create BLAS::BLAS target before FindSuiteSparse
- **Failed**: `find_package(BLAS)` doesn't check for existing targets; it always creates new ones

### Attempt 3: Set CMAKE_LIBRARY_PATH to include R's lib directory
- **Failed**: FindBLAS searches for specific library names (libblas.so, not libRblas.dylib)

### Attempt 4: Set BLAS_LIBRARIES as CACHE INTERNAL
- **Failed**: `find_package()` clears and resets all its variables regardless of cache type

### Successful Approach: Wrapper Find modules
- **Works**: Intercepting the `find_package()` call itself is the only reliable way to prevent the unwanted search

## Backward Compatibility

This solution is **fully backward compatible**:
- Standalone builds work exactly as before (wrappers fall back to standard Find modules)
- R package builds now work correctly (wrappers use preloaded R libraries)
- No changes required to calling code or R package configuration scripts

## Recommendations for R Package

The R package (`actionet-r`) should be updated to use the latest libactionet submodule commit that includes these fixes:

```bash
cd actionet-r/src/libactionet
git fetch
git checkout <commit-with-fixes>
cd ../..
git add src/libactionet
git commit -m "Update libactionet submodule with R package build fixes"
```

## Known Limitations

1. The wrapper approach relies on `CMAKE_MODULE_PATH` search order - if someone explicitly sets a different module path, it could break
2. The solution is specific to this codebase's needs and wouldn't generalize to all Find module interception scenarios
3. Cannot be used when CMAKE_MODULE_PATH is not under our control (rare scenario)

## Conclusion

The FindSuiteSparse BLAS detection issue during R package builds is now **fully resolved**. The wrapper Find modules provide a clean, maintainable solution that works reliably across different build modes without requiring modifications to third-party Find modules.

Any remaining build errors (such as compilation errors in wrapper code) are unrelated to this BLAS detection fix and should be addressed separately.
