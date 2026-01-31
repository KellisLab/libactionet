# libactionet

libactionet is the C++ backend for ACTIONet, providing high-performance kernels for graph-based dimensionality reduction, decomposition, and annotation. It is used by both the Python (`actionet-python`) and R (`actionet`) frontends.

## System Requirements
- macOS 11+ (arm64 or x86_64) or Linux (manylinux2014+ / glibc ≥ 2.17 on x86_64).
- CMake ≥ 3.19.
- C++17 compiler:
  - macOS: Apple Clang or LLVM Clang; supports Accelerate by default.
  - Linux: GCC or Clang; manylinux-compatible flags are used by default.
- BLAS/LAPACK:
  - MKL preferred when available; otherwise generic/system BLAS (can be OpenBLAS/Accelerate/etc.).
  - SuiteSparse/CHOLMOD (development headers and libs).
- OpenMP runtime:
  - GNU (`libgomp`) is the default on non-Apple.
  - Apple builds use Homebrew `libomp` if available.
  - INTEL or SEQUENTIAL threading can be requested explicitly.

## Build Configuration (CMake)
Key CMake options (with defaults):
- `BLA_VENDOR` (`All`): BLAS/LAPACK vendor. If unset and MKL is detected, it defaults to `Intel10_64lp` (non-R builds).
- `MKL_THREADING` (unset): When set to `GNU|INTEL|SEQUENTIAL` and MKL libs are found, libactionet links explicitly to the chosen threading variant (`mkl_gnu_thread`, `mkl_intel_thread`, or `mkl_sequential`) along with `mkl_intel_lp64` and `mkl_core`. This prevents accidental linkage to `mkl_intel_thread` when GNU OMP is desired.
- `LIBACTIONET_OPENMP_RUNTIME` (`AUTO`): `AUTO|GNU|INTEL|LLVM|OFF`. AUTO defaults to GNU unless using Intel compilers. R builds ignore this and use R’s OpenMP flags.
- `LIBACTIONET_OPENMP_CXXFLAGS` / `LIBACTIONET_OPENMP_LDFLAGS` (empty): Used by the R build to pass `SHLIB_OPENMP_*` flags into the static library build.
- `LIBACTIONET_BUILD_R` (`OFF`): Enable R integration mode (set by the R wrapper).
- `TARGET_ARCHITECTURE` (macOS/R builds): Set by the R configure script to match R’s target arch; used for Apple-specific paths.

OpenMP behavior:
- Non-R builds: `AUTO` ⇒ GNU (if found) → LLVM/Clang fallback via `find_package(OpenMP)`. `INTEL` searches `libiomp5`; `OFF` disables OpenMP.
- R builds: The R package passes `SHLIB_OPENMP_*` into these cache vars. If empty, OpenMP is disabled for the static library.
- MKL mixing: If MKL is detected and a non-Intel OpenMP runtime is selected, a warning is emitted. Use `MKL_THREADING_LAYER=GNU` with `mkl_rt` or `MKL_THREADING=GNU` to keep a single runtime.

BLAS behavior:
- MKL is preferred when detectable (`MKLROOT`, `libmkl_rt`). With `MKL_THREADING` set, an explicit MKL link line is used to avoid unintended `mkl_intel_thread`.
- Apple: Accelerate is supported and detected.
- Generic/system BLAS is used otherwise; `cblas.h` is searched in common system/conda/Homebrew paths.

## Building Standalone (C++)
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..    # add -DBLA_VENDOR=Intel10_64lp -DMKL_THREADING=GNU if desired
cmake --build . -j$(nproc)
```

## R Package Build Notes
- The R wrapper’s `configure` script passes R’s `SHLIB_OPENMP_*` into `LIBACTIONET_OPENMP_*`. If R reports no OpenMP flags, libactionet is built without OpenMP.
- BLAS/LAPACK come from R (`R CMD config BLAS_LIBS/LAPACK_LIBS`) unless the user overrides `BLA_VENDOR`.
- On macOS, Accelerate is the default via R; OpenMP requires an OpenMP-enabled R toolchain (e.g., LLVM + libomp).

## Python Package Build Notes
- `actionet-python` includes libactionet as a submodule and is automatically installed.
- A typical install with custom build options:
  ```bash
  pip install .                        # defaults
  # or, to force MKL GNU threading:
  MKL_THREADING_LAYER=GNU pip install . -C cmake.define.MKL_THREADING=GNU
  # or to switch OpenMP runtime:
  pip install . -C cmake.define.LIBACTIONET_OPENMP_RUNTIME=INTEL
  ```
- Ensure `libcholmod` is available (system or conda).

## Conda Environments
- If MKL is present, it will be preferred. To avoid `libiomp5` when using GNU OpenMP, set `MKL_THREADING_LAYER=GNU` and/or `MKL_THREADING=GNU` so the link line uses `mkl_gnu_thread`.
- Ensure `mkl-include` is installed if building against MKL.

## Troubleshooting
- Mixed OpenMP runtimes (GNU vs Intel) can cause crashes in rare cases.
  - Align runtimes: use `MKL_THREADING=GNU` or `LIBACTIONET_OPENMP_RUNTIME=INTEL` consistently.
- Missing `cblas.h`: install BLAS dev headers (MKL, OpenBLAS, Accelerate SDK), if not in standard paths.
- Missing CHOLMOD: install SuiteSparse development packages.
