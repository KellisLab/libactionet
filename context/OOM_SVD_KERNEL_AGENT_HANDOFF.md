# OOM SVD + Kernel Reduction: Agent Handoff

## Metadata
- Created: 2026-02-14
- Scope repos:
  - `/Users/sebastian/Documents/git_projects/libactionet`
  - `/Users/sebastian/Documents/git_projects/actionet-python`
- Feature branches:
  - `libactionet`: `codex/oom-backed-svd-kernel`
  - `actionet-python`: `codex/oom-backed-svd-kernel`

## Problem Statement
- Need out-of-memory (OOM) matrix support for very large datasets (especially AnnData backed sparse `.X`) for kernel reduction.
- Current Python path materializes matrices into Armadillo memory (`scipy_to_arma_sparse`), which does not scale for very large backed data.
- Need v1 OOM support starting with SVD and kernel reduction entrypoint (`reduce_kernel()`), with common HPC constraints:
  - Avoid heavy new dependencies.
  - Preserve existing builds/platform requirements.
  - Keep existing in-memory workflows working.
- Need modularization so kernel reduction can reuse a precomputed SVD to avoid repeated expensive decompositions.

## Agreed Implementation Plan (Decision Summary)
- OOM backend architecture: Python callback operator (`matvec` / `rmatvec`) consumed by C++.
- OOM SVD scope in v1: PRIMME only.
- Matrix scope in v1: backed sparse first.
- Delivery scope: both `libactionet` and `actionet-python`.
- API direction:
  - Add typed C++ result structs.
  - Add operator-driven C++ APIs.
  - Keep legacy in-memory APIs as wrappers for now.
  - Add Python bindings + high-level API for operator path and precomputed SVD reuse.

## Execution Details

### 1) libactionet changes
- New matrix operator abstraction:
  - Added `/Users/sebastian/Documents/git_projects/libactionet/include/decomposition/matrix_operator.hpp`
  - Includes:
    - `actionet::MatrixOperator` interface
    - `DenseMatrixOperator`
    - `SparseMatrixOperator`
- SVD API expansion:
  - Updated `/Users/sebastian/Documents/git_projects/libactionet/include/decomposition/svd_main.hpp`
  - Added:
    - `struct SVDResult { U, sigma, V }`
    - `runSVD_PRIMME_Operator(...)`
    - conversion helpers between result struct and legacy `arma::field<arma::mat>`
- Kernel API modularization:
  - Updated `/Users/sebastian/Documents/git_projects/libactionet/include/action/reduce_kernel.hpp`
  - Added:
    - `struct KernelReductionResult { S_r, sigma, V, A, B }`
    - `computeKernelPerturbationTerms(...)`
    - `applyKernelPostSVD(...)`
    - `reduceKernelFromSVD_Operator(...)`
    - `reduceKernel_Operator(...)`
    - `reduceKernelFromSVD(...)`
- Core implementation updates:
  - Reworked `/Users/sebastian/Documents/git_projects/libactionet/src/decomposition/svd_primme.cpp`
    - Added PRIMME operator callback context.
    - Added shared PRIMME core routine with safer dimension/buffer handling.
    - Added `runSVD_PRIMME_Operator(...)` implementation.
  - Reworked `/Users/sebastian/Documents/git_projects/libactionet/src/action/reduce_kernel.cpp`
    - Split kernel math into reusable components.
    - Added operator-based reduction + from-precomputed-SVD path.
    - Kept legacy in-memory `reduceKernel(T&)` behavior.
  - Updated `/Users/sebastian/Documents/git_projects/libactionet/include/libactionet.hpp` to export new operator header.
- Documentation update:
  - Updated `/Users/sebastian/Documents/git_projects/libactionet/README.md` public API list.

### 2) actionet-python changes
- Pybind utility extension:
  - Updated `/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/wp_utils.h`
  - Updated `/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/wp_utils.cpp`
  - Added `PythonMatrixOperator` wrapper class mapping Python operator object to C++ `MatrixOperator`.
- New pybind endpoints:
  - Updated `/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/wp_decomposition.cpp`
    - Added `_core.run_svd_operator(...)`
  - Updated `/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/wp_action.cpp`
    - Added `_core.reduce_kernel_operator(...)`
    - Added `_core.reduce_kernel_from_svd_operator(...)`
- High-level Python API updates:
  - Updated `/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/core.py`
  - Added:
    - backed sparse detection (`_is_backed_sparse_matrix`)
    - operator adapter for transposed matrix view (`_TransposeMatrixOperator`)
    - `reduce_kernel(..., precomputed_svd=..., backed_chunk_size=...)`
    - `reduce_kernel_from_svd(...)`
    - `run_svd(..., return_operator_compatible=True, backed_chunk_size=...)` with operator path support
  - Updated `/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/__init__.py`
    - exported `reduce_kernel_from_svd`
- User docs update:
  - Updated `/Users/sebastian/Documents/git_projects/actionet-python/README.md` with OOM v1 notes and precomputed SVD usage.

### 3) Branch/submodule coordination
- `libactionet` branch created and pushed: `codex/oom-backed-svd-kernel`
- `actionet-python` branch created and pushed: `codex/oom-backed-svd-kernel`
- `actionet-python/src/libactionet` submodule switched to and pinned at:
  - branch: `codex/oom-backed-svd-kernel`
  - commit: `99b2ea2c69845d2c9da7d99b03d0eb29af2696c0`

### 4) Commit log
- `libactionet`
  - `7580f7f` Add operator-based PRIMME SVD and modular kernel reduction APIs
  - `99b2ea2` Document operator-based SVD and kernel reduction APIs
- `actionet-python`
  - `6311f67` Add backed sparse OOM operator path and precomputed SVD APIs

## Validation Status
- `libactionet` build: PASS
  - Command used:
    - `cmake --build /Users/sebastian/Documents/git_projects/libactionet/cmake-build-llvm-arm64 -j4`
- `actionet-python` build: BLOCKED in current environment
  - Configure failed due missing `pybind11Config.cmake`.
  - Error class:
    - `find_package(pybind11 CONFIG REQUIRED)` failed.

## Open Items / Follow-up Steps
- Build validation in `actionet-python` environment with pybind11 available.
- Add/expand tests:
  - `libactionet`
    - operator PRIMME vs in-memory PRIMME consistency tests on small sparse fixtures
    - `reduceKernelFromSVD` parity with legacy path
    - large-dimension safety tests for index/buffer boundaries
  - `actionet-python`
    - backed sparse `.X` route executes operator path
    - in-memory dense/sparse paths unchanged
    - `precomputed_svd` avoids recomputation and returns expected shapes
- Add benchmarks:
  - compare peak memory and runtime for large sparse data between legacy and operator path
- API hardening:
  - finalize expected shape conventions for `u/d/v` inputs in precomputed-SVD calls
  - decide whether to deprecate any legacy interfaces with warnings
- PR cleanup:
  - separate mechanical/API/docs deltas if reviewers request smaller review slices

## Known Constraints and Notes
- OOM v1 is intentionally PRIMME-only for operator path.
- Dense OOM path is out of scope in this implementation.
- Existing unrelated file in `actionet-python` remains modified and uncommitted:
  - `/Users/sebastian/Documents/git_projects/actionet-python/TODO.md`

## Quick Resume Commands
- libactionet:
  - `cd /Users/sebastian/Documents/git_projects/libactionet`
  - `git checkout codex/oom-backed-svd-kernel`
- actionet-python:
  - `cd /Users/sebastian/Documents/git_projects/actionet-python`
  - `git checkout codex/oom-backed-svd-kernel`
  - `git submodule update --init --recursive`
  - `cd src/libactionet && git checkout codex/oom-backed-svd-kernel`
