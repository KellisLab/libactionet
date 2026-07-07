# libactionet

libactionet is the C++ backend for ACTIONet, providing high-performance kernels for graph-based dimensionality reduction, decomposition, and annotation. It is used by both the Python (`actionet-python`) and R (`actionet-r`) frontends.

## Architecture
libactionet uses native [Armadillo](http://arma.sourceforge.net/) sparse matrix operations for all sparse-dense computations, providing:
- **Thread-safe parallelism**: No shared context issues, scales efficiently to 30+ threads
- **Large matrix support**: Handles matrices with >2³¹ non-zero elements (tested with 5B+ nnz)
- **Memory efficiency**: ~50% lower peak memory vs. previous CHOLMOD-based implementation
- **Simplified dependencies**: No external SuiteSparse/CHOLMOD requirement

For large sparse matrix SVD, [PRIMME](https://github.com/primme/primme) is used, which also leverages native Armadillo operations.

HDF5-backed (out-of-core) operators allow SVD, kernel reduction, specificity, and annotation on datasets that exceed available memory, reading from `.h5ad` files via chunked I/O.

## System Requirements
- macOS 11+ (arm64 or x86_64) or Linux (manylinux2014+ / glibc ≥ 2.17 on x86_64).
- CMake ≥ 3.19.
- C++17 compiler:
  - macOS: Apple Clang or LLVM Clang; supports Accelerate by default.
  - Linux: GCC or Clang; manylinux-compatible flags are used by default.
- BLAS/LAPACK:
  - MKL preferred when available; otherwise generic/system BLAS (can be OpenBLAS/Accelerate/etc.).
- HDF5 (C library, required):
  - System or conda HDF5 is detected via `find_package(HDF5)`.
- OpenMP runtime (required):
  - GNU (`libgomp`) is the default on non-Apple.
  - Apple builds use Homebrew `libomp` if available.
  - INTEL runtime can be requested explicitly.

## Repository Layout

```
include/             Public C++ headers (API surface consumed by bindings)
  action/            Archetypal analysis (AA, SPA, simplex regression, kernel reduction)
  annotation/        Feature specificity and marker statistics
  decomposition/     SVD algorithms, matrix operators, batch orthogonalization
  io/backed_h5ad/    HDF5-backed out-of-core matrix operators
  network/           Network construction (HNSW kNN), diffusion, measures
  tools/             Matrix transforms, aggregation, autocorrelation, enrichment, MWM, XICOR
  visualization/     UMAP/layout optimization, node coloring
  utils_internal/    Internal helpers (not part of public API)
  extern/            Third-party headers (drop-in, do not modify)
src/                 Core implementations (mirrors include/ structure)
  extern/            Third-party source files
cmake/               CMake modules (Apple, BLAS, OpenMP, PRIMME, R configuration)
docs/                Algorithm and API documentation
context/             Agent context and project decision records
wrappers_r/          Reference copy of Rcpp wrapper code (primary R package lives separately)
_EXCLUDE/            Deprecated/archived code (not compiled)
```

## Build Configuration (CMake)
Key CMake options (with defaults):
- `BLA_VENDOR` (`All`): BLAS/LAPACK vendor. If unset and MKL is detected, it defaults to `Intel10_64lp` (non-R builds).
- `MKL_THREADING` (unset): When set to `GNU|INTEL|SEQUENTIAL` and MKL libs are found, libactionet links explicitly to the chosen threading variant (`mkl_gnu_thread`, `mkl_intel_thread`, or `mkl_sequential`) along with `mkl_intel_lp64` and `mkl_core`. This prevents accidental linkage to `mkl_intel_thread` when GNU OMP is desired.
- `LIBACTIONET_OPENMP_RUNTIME` (`AUTO`): `AUTO|GNU|INTEL|LLVM`. AUTO defaults to GNU unless using Intel compilers. R builds ignore this and use R's OpenMP flags.
- `LIBACTIONET_OPENMP_CXXFLAGS` / `LIBACTIONET_OPENMP_LDFLAGS` (empty): Used by the R build to pass `SHLIB_OPENMP_*` flags into the static library build.
- `LIBACTIONET_BUILD_R` (`OFF`): Enable R integration mode (set by the R wrapper).
- `TARGET_ARCHITECTURE` (macOS/R builds): Set by the R configure script to match R's target arch; used for Apple-specific paths.
- `CMAKE_INTERPROCEDURAL_OPTIMIZATION` (`OFF`): Enable LTO/IPO if your toolchain supports it.

OpenMP behavior:
- Non-R builds: `AUTO` ⇒ GNU (if found) → LLVM/Clang fallback via `find_package(OpenMP)`. `INTEL` searches `libiomp5`. The build fails if no OpenMP runtime is found.
- R builds: The R package passes `SHLIB_OPENMP_*` into these cache vars. If empty, standard OpenMP detection is attempted; the build fails if OpenMP cannot be found.
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
- The R wrapper's `configure` script passes R's `SHLIB_OPENMP_*` into `LIBACTIONET_OPENMP_*`. If R reports no OpenMP flags, standard OpenMP detection is attempted; the build will fail if no OpenMP runtime is found.
- BLAS/LAPACK come from R (`R CMD config BLAS_LIBS/LAPACK_LIBS`) unless the user overrides `BLA_VENDOR`.
- On macOS, Accelerate is the default via R; OpenMP requires an OpenMP-enabled R toolchain (e.g., LLVM + libomp).
- PRIMME and HDF5-backed operators are excluded from R builds (R does not support >2^31 element matrices).

## Python Package Build Notes
- `actionet-python` includes libactionet as a submodule and is automatically installed.
- A typical install with custom build options:
  ```bash
  pip install .                        # defaults
  # or, to force MKL GNU threading:
  MKL_THREADING_LAYER=GNU pip install . -C cmake.define.MKL_THREADING=GNU
  # or to switch OpenMP runtime:
  pip install . -C cmake.define.LIBACTIONET_OPENMP_RUNTIME=INTEL
  # to mirror R's native-tuned builds (when portability is not required):
  pip install . -C cmake.define.MKL_THREADING=INTEL \
                   -C cmake.define.LIBACTIONET_OPENMP_RUNTIME=INTEL \
                   -C cmake.define.CMAKE_CXX_FLAGS="-march=native -mtune=native -O3 -ffp-contract=fast -funroll-loops -fomit-frame-pointer -fno-strict-aliasing" \
                   -C cmake.define.CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
  ```

## Conda Environments
- If MKL is present, it will be preferred. To avoid `libiomp5` when using GNU OpenMP, set `MKL_THREADING_LAYER=GNU` and/or `MKL_THREADING=GNU` so the link line uses `mkl_gnu_thread`.
- Ensure `mkl-include` is installed if building against MKL.

## Troubleshooting
- Mixed OpenMP runtimes (GNU vs Intel) can cause crashes in rare cases.
  - Align runtimes: use `MKL_THREADING=GNU` or `LIBACTIONET_OPENMP_RUNTIME=INTEL` consistently.
- Missing `cblas.h`: install BLAS dev headers (MKL, OpenBLAS, Accelerate SDK), if not in standard paths.
- HDF5 not found: install `hdf5` dev package (system, conda, or Homebrew). CMake requires the C component.

## Public API (C++)
All public symbols are exposed under the `actionet` namespace via `include/libactionet.hpp`.

- **Decomposition / SVD**: `runSVD`, `runSVD_Operator`, `runSVD_PRIMME_Operator`, `perturbedSVD`,
  plus result structs (`SVDResult`, `PerturbedSVDResult`) and the `MatrixOperator` interface (`DenseMatrixOperator`, `SparseMatrixOperator`)
- **Batch orthogonalization**: `orthogonalizeBatchEffect`, `orthogonalizeBasal` (in-memory and operator-backed variants)
- **Kernel reduction**: `reduceKernel`, `reduceKernel_Operator`, `reduceKernelFromSVD`, `reduceKernelFromSVD_Operator`, `reduceKernelFromSVD_InMemory`, `applyKernelPostSVD`, plus `KernelReductionResult`
- **ACTION decomposition**: `runACTION`, `decompACTION`, `collectArchetypes`, `mergeArchetypes`, `runAA`, `runSPA`, `runSimplexRegression`
- **Backed I/O**: `createBackedOperator` (auto-detects sparse/dense from HDF5), `BackedSparseMatrixOperator`, `BackedDenseMatrixOperator`
- **Network**: `buildNetwork`, `computeNetworkDiffusion`, `runLPA`, `computeCoreness`, `computeArchetypeCentrality`
- **Annotation / specificity**: `computeFeatureSpecificity`, `computeFeatureStats`, `computeFeatureStatsVision`, `computeFeatureStatsVisionFromStats` (with backed-operator overloads)
- **Visualization**: `layoutNetwork`, `computeNodeColors`
- **Tools**: `normalizeMatrix`, `scaleMatrix`, `normalizeGraph`, `normalize_scores`, `computeGroupedSums/Means/Vars`, `autocorrelation_Moran/Moran_parametric/Geary`, `assess_enrichment`, `computeGraphLabelEnrichment`, `MWM_hungarian`, `MWM_rank1`, `xicor`, `XICOR`, `fitGuidesSharedVarianceGMM`, `deriveGuideThresholdsQuantile`, `deriveGuideThresholdsEqualDensity`, `deriveGuideThresholdsValley`, `sweepGuideThresholdsQuantile`, `applyGuideThresholds`

Guide-calling API details: `docs/guide_calling_api.md`

## License
GNU Affero General Public License v3 (AGPL-3.0). See [LICENSE](LICENSE).
