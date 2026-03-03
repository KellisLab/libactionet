# OOM Backed H5AD v1: Code Review Handoff

## Metadata
- Prepared: March 3, 2026
- Scope repos:
  - `/Users/sebastian/Documents/git_projects/libactionet`
  - `/Users/sebastian/Documents/git_projects/actionet-python`
  - `/Users/sebastian/Documents/git_projects/actionet-r`
- Feature branch (all repos): `codex/oom-backed-h5ad-v1`
- Base branch (verified ancestor in all repos): `codex/oom-backed-extension`

## Objective
Implement the Python-first, modular backed H5AD optimization plan for 10M+ cell feasibility, with intentional API breaks and strict isolation of disk-backed code from non-Python consumers.

## Constraints Requested by User (Status)
1. API/backward-compatibility breaks allowed: implemented.
2. No `assume_nonnegative` flag; no full negative pre-scan pass: implemented.
3. Keep source/header organization consistent: implemented (`include/io/backed_h5ad`, `src/io/backed_h5ad`).
4. HDF5 required unconditionally: implemented (`find_package(HDF5 REQUIRED ...)`).

## Branch and Sync Status

### Repository branches
- `libactionet`: `codex/oom-backed-h5ad-v1`
- `actionet-python`: `codex/oom-backed-h5ad-v1`
- `actionet-r`: `codex/oom-backed-h5ad-v1`
- `actionet-python/src/libactionet` submodule: `codex/oom-backed-h5ad-v1`

### Core sync check
- Standalone `libactionet` and `actionet-python/src/libactionet` have matching content for all touched core files (verified with file-by-file content comparison).

### actionet-r
- Branch created and synced from base.
- No code changes in `actionet-r` for this iteration.

## Implementation Summary

### A) Core C++ (`libactionet`)

#### 1. New modular backed I/O operator module
- Added:
  - `include/io/backed_h5ad/backed_sparse_matrix_operator.hpp`
  - `src/io/backed_h5ad/backed_sparse_matrix_operator.cpp`
- Implements `BackedSparseMatrixOperator` (AnnData sparse group reader):
  - reads sparse groups (`data`, `indices`, `indptr`, `shape`, encoding attrs)
  - supports CSR and CSC
  - supports runtime transforms (`row_scale_factors`, `apply_log1p`)
  - implements `matvec`, `rmatvec`, `matmat`, `rmatmat`

#### 2. Matrix operator interface extension
- Updated `include/decomposition/matrix_operator.hpp`:
  - added `matmat` and `rmatmat` virtual methods
  - default fallback loops through `matvec`/`rmatvec`
  - dense/sparse in-memory adapters override block paths directly

#### 3. Operator-backed SVD algorithms
- Added operator overloads for Halko/Feng:
  - `include/decomposition/svd_halko.hpp`, `src/decomposition/svd_halko.cpp`
  - `include/decomposition/svd_feng.hpp`, `src/decomposition/svd_feng.cpp`
- Added generic operator dispatcher:
  - `runSVD_Operator(...)`
  - `runSVD_Halko_Operator(...)`
  - `runSVD_Feng_Operator(...)`
  - in `include/decomposition/svd_main.hpp`, `src/decomposition/svd_main.cpp`

#### 4. Kernel reduction operator path now algorithm-driven
- Updated `include/action/reduce_kernel.hpp`, `src/action/reduce_kernel.cpp`:
  - `reduceKernel_Operator` now accepts `svd_alg`
  - dispatches via `runSVD_Operator(...)` instead of PRIMME-only entry

#### 5. PRIMME callback optimization
- Updated `src/decomposition/svd_primme.cpp`:
  - block callback path uses operator `matmat`/`rmatmat` when `blockSize > 1`
  - retains column fallback otherwise
  - `max_it=0` now uses bounded default (`maxMatvecs = 1000 * k_eff`)

#### 6. Build system changes
- Updated `CMakeLists.txt`:
  - `find_package(HDF5 REQUIRED COMPONENTS C)`
  - includes `src/io/*.cpp` in core source set
  - links HDF5 to `actionet`
  - exposes `HDF5_INCLUDE_DIRS` as `PUBLIC` include dirs (needed because backed header includes `hdf5.h`)

#### 7. Modularity guard
- `include/libactionet.hpp` does **not** re-export backed H5AD header.
- This prevents forcing the new Python-backed I/O surface onto all consumers via umbrella include.

---

### B) Python frontend (`actionet-python`)

#### 1. New dedicated IO binding unit
- Added `src/actionet/wp_io.cpp`
- Registered in:
  - `src/actionet/_core.cpp` (`init_io`)
  - `CMakeLists.txt` (wrapper source list)
- New bindings:
  - `create_backed_operator(...)`
  - `run_svd_backed_operator(...)`
  - `reduce_kernel_backed_operator(...)`
  - `reduce_kernel_from_svd_backed_operator(...)`

#### 2. Existing wrappers updated for algorithm-aware operator dispatch
- `src/actionet/wp_decomposition.cpp`: `run_svd_operator` now routes to `runSVD_Operator(...)`
- `src/actionet/wp_action.cpp`: `reduce_kernel_operator` now takes and forwards `svd_alg`

#### 3. High-level API transition to string-only algorithms
- Updated `src/actionet/core.py`:
  - accepted names: `"auto"`, `"halko"`, `"feng"`, `"irlb"`, `"primme"`
  - backed `"auto"` now selects Halko
  - backed allows only `"auto"|"halko"|"feng"|"primme"`
  - integer algorithm-style handling removed in high-level entrypoints

#### 4. Backed operator path replaces `_TransposeMatrixOperator` hot path
- Backed `reduce_kernel` and `run_svd` now construct and use C++ `BackedSparseMatrixOperator`.
- Old Python callback transpose operator path is no longer the default backed execution path.

#### 5. Compression policy changes
- Added auto-decompression helper for backed AnnData path when feasible.
- If disk is insufficient:
  - continues in compressed mode
  - emits warning
- Explicit `allow_compressed` opt-out path retained.

#### 6. Preprocessing/backed normalization updates
- Updated `src/actionet/preprocessing.py`:
  - backed normalization default output dtype is `float32`
  - removed global negative pre-scan pass
  - sparse subsetting writer converted to single-pass extensible dataset write

---

### C) Tests/benchmarks updated for new API

Updated scripts to use string `svd_algorithm` values instead of integer IDs:
- `tests/test_svd_methods.py`
- `tests/test_svd_sparse_vs_dense.py`
- `tests/benchmark_svd_algorithms.py`
- `tests/test_svd_backed_vs_inmemory.py`
- `tests/parity_test_small.py`
- `tests/benchmark_backed_extension.py`
- `tests/backed/test_backed_extension.py` (already adjusted earlier in the branch work)

## File Change Map

### `libactionet` changed files
- `CMakeLists.txt`
- `include/action/reduce_kernel.hpp`
- `include/decomposition/matrix_operator.hpp`
- `include/decomposition/svd_feng.hpp`
- `include/decomposition/svd_halko.hpp`
- `include/decomposition/svd_main.hpp`
- `src/action/reduce_kernel.cpp`
- `src/decomposition/svd_feng.cpp`
- `src/decomposition/svd_halko.cpp`
- `src/decomposition/svd_main.cpp`
- `src/decomposition/svd_primme.cpp`
- `include/io/backed_h5ad/backed_sparse_matrix_operator.hpp` (new)
- `src/io/backed_h5ad/backed_sparse_matrix_operator.cpp` (new)

### `actionet-python` changed files
- `CMakeLists.txt`
- `src/actionet/_core.cpp`
- `src/actionet/core.py`
- `src/actionet/preprocessing.py`
- `src/actionet/wp_action.cpp`
- `src/actionet/wp_decomposition.cpp`
- `src/actionet/wp_io.cpp` (new)
- `tests/backed/test_backed_extension.py`
- `tests/benchmark_backed_extension.py`
- `tests/benchmark_svd_algorithms.py`
- `tests/parity_test_small.py`
- `tests/test_svd_backed_vs_inmemory.py`
- `tests/test_svd_methods.py`
- `tests/test_svd_sparse_vs_dense.py`
- `src/libactionet` submodule pointer (dirty pointer only; no commit yet)

### `actionet-r` changed files
- None in this iteration (branch sync only).

## Code Review Focus Areas (Priority-Ordered)

### P0: Correctness / numerical behavior
1. `BackedSparseMatrixOperator` dimension conventions and transpose semantics:
   - `rows() = n_var`, `cols() = n_obs`
   - CSR/CSC kernels for `matvec/rmatvec/matmat/rmatmat`
2. Operator Halko/Feng correctness and truncation logic under tall vs wide matrices.
3. PRIMME block callback path:
   - packed/unpacked memory handling (`ldx`, `ldy`)
   - `blockSize > 1` dispatch correctness
4. `reduceKernel_Operator(..., svd_alg, ...)` dispatch behavior and parity with prior code paths.

### P1: API and behavior contracts
1. Python string algorithm normalization/validation paths.
2. Backed `auto -> halko` semantics.
3. Decompression workflow:
   - temp file lifecycle and cleanup
   - insufficient disk fallback warning path
4. Removal of negative pre-scan in normalization (expected by design).

### P2: Build/package boundary
1. Unconditional HDF5 requirement in core CMake.
2. `PUBLIC` include propagation for HDF5 headers.
3. Modularity isolation:
   - backed code kept under `io/backed_h5ad`
   - no umbrella export from `libactionet.hpp`.

## Deviations / Gaps vs Requested Plan
1. HighFive vendoring (`include/extern/highfive`) was **not** added.
   - Implementation uses HDF5 C API directly.
2. `actionet-r` has no integration updates in this pass (branch synchronized only).
3. Full compile/test validation is blocked by local dependency gaps (below).

## Validation Performed

### Completed
1. Python syntax parse (`ast.parse`) for modified Python files: PASS.
2. Cross-repo branch ancestry checks to `codex/oom-backed-extension`: PASS.
3. Standalone `libactionet` and python submodule core content parity check: PASS.

### Blocked (environment dependency)
1. `libactionet` configure:
   - `cmake -S . -B cmake-build-codex -DCMAKE_BUILD_TYPE=Release`
   - FAIL: missing HDF5 dev package (`HDF5_LIBRARIES`, `HDF5_INCLUDE_DIRS`)
2. `actionet-python` configure:
   - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
   - FAIL: missing `pybind11Config.cmake`
3. `actionet-python/src/libactionet` configure:
   - same HDF5 missing failure as standalone core.

## Suggested Reviewer Workflow

### 1) Core architecture and API diffs
- Review C++ API surface first:
  - `include/decomposition/matrix_operator.hpp`
  - `include/decomposition/svd_main.hpp`
  - `include/action/reduce_kernel.hpp`
- Then implementation:
  - `src/decomposition/svd_halko.cpp`
  - `src/decomposition/svd_feng.cpp`
  - `src/decomposition/svd_primme.cpp`
  - `src/action/reduce_kernel.cpp`
  - `src/io/backed_h5ad/backed_sparse_matrix_operator.cpp`

### 2) Python API behavior and compatibility break review
- `src/actionet/core.py`
- `src/actionet/preprocessing.py`
- `src/actionet/wp_io.cpp`
- `src/actionet/wp_decomposition.cpp`
- `src/actionet/wp_action.cpp`

### 3) Test adaptation review
- Ensure string-only algorithm usage is consistent across:
  - `tests/test_svd_methods.py`
  - `tests/test_svd_sparse_vs_dense.py`
  - `tests/benchmark_svd_algorithms.py`
  - `tests/test_svd_backed_vs_inmemory.py`
  - `tests/parity_test_small.py`
  - `tests/benchmark_backed_extension.py`

## Post-Review Integration Tasks
1. Install/configure HDF5 + pybind11 CMake dependencies and run full build/tests.
2. Decide whether HighFive vendoring is still required.
3. Commit core changes in standalone `libactionet`.
4. Commit python changes and update `src/libactionet` submodule pointer.
5. If needed, update `actionet-r` to pin new submodule commit after core commits exist.

## Current Risk Register
1. Build portability risk from unconditional HDF5 requirement (expected but must be communicated).
2. Backed operator I/O performance sensitivity to HDF5 dataset chunking/layout in external h5ad files.
3. Potential numerical drift between Halko/Feng/PRIMME operator-backed and in-memory paths (expected but should be benchmarked/characterized on real large datasets).

