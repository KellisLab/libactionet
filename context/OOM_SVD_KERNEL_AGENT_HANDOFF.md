# OOM SVD + Kernel Reduction: Agent Handoff

## Metadata
- Created: 2026-02-14
- Last updated: 2026-02-15
- Scope repos:
  - `/Users/sebastian/Documents/git_projects/libactionet`
  - `/Users/sebastian/Documents/git_projects/actionet-python`
  - `/Users/sebastian/Documents/git_projects/actionet-r`
- Feature branches:
  - `libactionet`: `codex/oom-backed-svd-kernel`
  - `actionet-python`: `codex/oom-backed-svd-kernel`
  - `actionet-r`: `OOM-SVD-API-update`

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
- Delivery scope: `libactionet`, `actionet-python`, and `actionet-r`.
- API direction:
  - Add typed C++ result structs (`SVDResult`, `PerturbedSVDResult`, `KernelReductionResult`).
  - Add operator-driven C++ APIs.
  - Keep legacy in-memory APIs as wrappers for now.
  - Add Python bindings + high-level API for operator path and precomputed SVD reuse.

## Execution Details

### Phase 1: Initial OOM Implementation (completed)

#### 1) libactionet changes
- New matrix operator abstraction:
  - Added `include/decomposition/matrix_operator.hpp`
  - Includes: `actionet::MatrixOperator` interface, `DenseMatrixOperator`, `SparseMatrixOperator`
  - Documented OOM design, threading/GIL constraints, single-threaded requirement
- SVD API expansion:
  - Updated `include/decomposition/svd_main.hpp`
  - Added: `struct SVDResult { U, sigma, V }`, `struct PerturbedSVDResult { U, sigma, V, A, B }`
  - Added `runSVD_PRIMME_Operator(...)`, struct-based `perturbedSVD()` as primary API
  - Conversion helpers `svdResultFromField`/`svdFieldFromResult` retained (needed by `orient_SVD`, `runSVD`, legacy wrapper)
- Kernel API modularization:
  - Updated `include/action/reduce_kernel.hpp`
  - Added: `struct KernelReductionResult { S_r, sigma, U, A, B }`
  - Added: `computeKernelPerturbationTerms(...)`, `applyKernelPostSVD(...)`, `reduceKernelFromSVD_Operator(...)`, `reduceKernel_Operator(...)`, `reduceKernelFromSVD_InMemory<T>(...)`
- Core implementation:
  - `src/decomposition/svd_primme.cpp`: Consolidated 3 redundant matvec callbacks into single `primmeMatvec` with `PrimmeCallbackCtx` dispatch. Fixed PRIMME seed constraints (`iseed` values in `[0, 4095]`, `iseed[3]` odd).
  - `src/action/reduce_kernel.cpp`: `applyKernelPostSVD` now uses struct-based `perturbedSVD` directly. Added `validateSVDAndPerturbation()`. Added `reduceKernelFromSVD_InMemory` with template instantiations for `arma::mat` and `arma::sp_mat`. Documented 0.01 rounding constant and R-build limitation.
  - `src/decomposition/orthogonalization.cpp`: `deflateReduction` updated to take `const arma::mat&` for A/B, creates local augmented copies.

#### 2) actionet-python changes
- Pybind utility extension:
  - Updated `wp_utils.h` / `wp_utils.cpp`: Added `PythonMatrixOperator` with GIL/threading documentation. Optimized with zero-copy NumPy views for input and `std::memcpy` for output.
- New pybind endpoints:
  - `wp_decomposition.cpp`: Added `_core.run_svd_operator(...)`
  - `wp_action.cpp`: Added `_core.reduce_kernel_operator(...)`, `_core.reduce_kernel_from_svd_operator(...)`, `_core.reduce_kernel_from_svd_sparse(...)`, `_core.reduce_kernel_from_svd_dense(...)`
- High-level Python API updates (`core.py`):
  - Replaced heuristic `_is_backed_sparse_matrix` with AnnData native detection (`isbacked` + `group` fallback)
  - Fixed dead code in `_select_svd_algorithm` (moved backed-sparse guard before explicit algorithm check)
  - In-memory precomputed SVD now uses efficient C++ path instead of operator path
  - Removed unused `from statsmodels.stats.rates import norm`
  - Added `_TransposeMatrixOperator` with comprehensive performance docs
  - Updated `reduce_kernel(..., precomputed_svd=..., backed_chunk_size=...)`
  - Added `reduce_kernel_from_svd(...)` and `run_svd(...)` with operator path support

### Phase 2: Review Fixes, Naming Convention, and Tests (completed)

#### 3) V → U naming convention fix (all three repos)
The `KernelReductionResult` field for left singular vectors was incorrectly named `V`. Renamed to `U` throughout the entire stack:

**libactionet:**
- `include/action/reduce_kernel.hpp`: `KernelReductionResult::V` → `::U`
- `wrappers_r/wr_action.cpp`: `res["V"]` → `res["U"]` in `C_reduceKernelSparse` and `C_reduceKernelDense`
- `wrappers_r/wr_decomposition.cpp`: `old_V` → `old_U` params, `SVD_results(0) = old_V` → `= old_U`, `res["V"]` → `res["U"]` in all 4 orthogonalization functions

**actionet-python:**
- `core.py`: `varm[f"{key}_V"]` → `varm[f"{key}_U"]`, result dict key `"V"` → `"U"`
- `batch_correction.py`: `old_V` → `old_U`, AnnData keys `{key}_V` → `{key}_U` in both `correct_batch_effect` and `correct_basal_expression`
- `imputation.py`: AnnData key `{key}_V` → `{key}_U`, fixed misleading local variable names (`V` → `U_left`, `U` → `V_right`)
- `wp_decomposition.cpp`: All 4 orthogonalization functions: `old_V` → `old_U` params, `V_mat` → `U_mat`, dict key `"V"` → `"U"`, `py::arg("old_V")` → `py::arg("old_U")`
- `wp_action.cpp`: result dict key `"V"` → `"U"` in all kernel result paths

**actionet-r:**
- `R/r_action.R`: `out$V` → `out$U`, map slot `_V` → `_U`, column prefix `"V"` → `"U"`
- `R/batch_correct.R`: `correctBatchEffectFastMNN` and `correctBatchEffect`: map slots `_V` → `_U`, `old_V` → `old_U`, `out$V` → `out$U`, column names `"V"` → `"U"`
- `R/utils_internal_main.R`: `vars$V` → `vars$U`, map slot `_V` → `_U`, local vars `V` → `U`, `U` → `U_right`
- `R/RcppExports.R`: All 4 orthogonalization wrappers: `old_V` → `old_U`

#### 4) Test suite (actionet-python)
New and updated tests in `tests/`:

- **`test_svd_methods.py`** (updated):
  - Added `validate_reduction_keys()` helper verifying `_U` naming, shape, NaN/Inf, stale `_V` detection
  - Added PRIMME backed-mode test (`test_primme_backed_mode`) creating temp h5ad, running operator path, validating output
  - Updated `main()` to run backed test and compare backed vs in-memory

- **`test_svd_sparse_vs_dense.py`** (updated):
  - Fixed `varm['..._V']` → `varm['..._U']` key references
  - Added `validate_reduction_keys()` with full output validation
  - Added `test_primme_backed_mode()` and `test_primme_backed_vs_inmemory()`
  - Updated `main()` to run backed tests and report consistency

- **`test_svd_backed_vs_inmemory.py`** (new):
  - Dedicated comparison of PRIMME backed (operator) vs in-memory modes
  - Two comparison pairs: sparse in-memory vs backed, dense in-memory vs backed
  - Per-component absolute correlation, flattened correlation, sigma relative diff
  - Three output figures:
    - `svd_backed_vs_inmemory_comparison.png` (scatter facets)
    - `svd_backed_vs_inmemory_summary.png` (correlation/error bars)
    - `svd_backed_vs_inmemory_sigma.png` (singular value overlay)

- **`benchmark_svd_algorithms.py`** (updated):
  - Added `benchmark_primme_backed()` for operator-mode benchmarking
  - Visualization dynamically handles all input types (sparse, dense, backed)
  - Backed results plot as green bars alongside blue (sparse) and coral (dense)
  - `--include-backed` / `--backed-chunk-size` CLI arguments

- **`run_all_svd_tests.sh`** (updated): Added `test_svd_backed_vs_inmemory.py`, listed new output figures
- **`run_benchmark.sh`** (updated): Added `-b`/`--include-backed` flag

#### 5) Validation status
- `libactionet` build: **PASS** (cmake --build, exit 0)
- `actionet-python` tests (manual run by user): **PASS** — all SVD methods, sparse/dense, backed/in-memory
- `actionet-python` benchmark with `--include-backed`: **PASS** — plots render correctly for all input types
- `actionet-r` V→U rename: committed and clean

### Branch/Submodule Coordination
- `libactionet` branch: `codex/oom-backed-svd-kernel`
- `actionet-python` branch: `codex/oom-backed-svd-kernel`
  - `src/libactionet` submodule pinned to `codex/oom-backed-svd-kernel`
- `actionet-r` branch: `OOM-SVD-API-update`

### Commit Log
- `libactionet`:
  - `7580f7f` Add operator-based PRIMME SVD and modular kernel reduction APIs
  - `99b2ea2` Document operator-based SVD and kernel reduction APIs
  - `069e1f0` Fix bugs and document (review fixes, V→U rename, const-correctness, validation, PRIMME seed fix, matvec consolidation)
- `actionet-python`:
  - `6311f67` Add backed sparse OOM operator path and precomputed SVD APIs
  - *(uncommitted)* Review fixes, V→U rename, zero-copy optimization, backed detection, tests, benchmarks
- `actionet-r`:
  - `64f4bc2` Working (V→U rename across R wrappers and exports)

## Uncommitted Changes

### actionet-python (17 files, +1 new)
All changes staged against `codex/oom-backed-svd-kernel` HEAD (`6311f67`):

| File | Summary |
|------|---------|
| `src/actionet/core.py` | Backed detection, dead code fix, in-memory precomputed SVD path, V→U keys, docstrings |
| `src/actionet/batch_correction.py` | V→U rename in AnnData keys and variable names |
| `src/actionet/imputation.py` | V→U rename, fixed misleading variable names |
| `src/actionet/wp_decomposition.cpp` | V→U in params, locals, dict keys, py::arg |
| `src/actionet/wp_action.cpp` | `reduce_kernel_from_svd_sparse/dense` bindings, V→U keys |
| `src/actionet/wp_utils.cpp` | Zero-copy NumPy input, memcpy output |
| `src/actionet/wp_utils.h` | GIL/threading documentation |
| `src/libactionet` | Submodule pointer update |
| `tests/test_svd_methods.py` | Key validation, PRIMME backed test |
| `tests/test_svd_sparse_vs_dense.py` | V→U fix, key validation, backed tests |
| `tests/test_svd_backed_vs_inmemory.py` | **New** — dedicated backed vs in-memory comparison |
| `tests/benchmark_svd_algorithms.py` | Backed benchmark, dynamic visualization |
| `tests/run_all_svd_tests.sh` | Added backed test |
| `tests/run_benchmark.sh` | Added --include-backed flag |
| `R/imputation.R` | Minor (unrelated to feature) |
| `TODO.md` | Pre-existing modification |
| `tests/test_batchcorr.ipynb` | Notebook updates |
| `tests/test_fast.ipynb` | Notebook updates |

## Open Items / Follow-up Steps
- **Commit** uncommitted `actionet-python` changes.
- **PR cleanup**: Separate mechanical/API/docs deltas if reviewers request smaller review slices.
- **API hardening**:
  - Finalize expected shape conventions for `u/d/v` inputs in precomputed-SVD calls.
  - Decide whether to deprecate any legacy interfaces with warnings.
- **Additional tests**:
  - `libactionet`: large-dimension safety tests for index/buffer boundaries.
  - `actionet-python`: `precomputed_svd` avoids recomputation test (shape parity).
- **Additional benchmarks**:
  - Compare peak memory for large sparse data (operator vs legacy path) — backed_chunk_size sensitivity.
- **R package build**: Verify `actionet-r` builds cleanly against updated `libactionet` submodule after `Rcpp::compileAttributes()` regeneration.

## Known Constraints and Notes
- OOM v1 is intentionally PRIMME-only for operator path.
- Dense OOM path is out of scope in this implementation.
- Backed vs in-memory PRIMME results may differ slightly due to iterative solver tolerance and chunked I/O; test thresholds are intentionally looser than sparse-vs-dense (|corr| > 0.99 vs 0.9999).
- `pseudobulk_DGE.R` references to `with_V` / `slot_V` are unrelated (variance, not singular vectors) — intentionally not renamed.

## Quick Resume Commands
- libactionet:
  - `cd /Users/sebastian/Documents/git_projects/libactionet`
  - `git checkout codex/oom-backed-svd-kernel`
- actionet-python:
  - `cd /Users/sebastian/Documents/git_projects/actionet-python`
  - `git checkout codex/oom-backed-svd-kernel`
  - `git submodule update --init --recursive`
  - `cd src/libactionet && git checkout codex/oom-backed-svd-kernel`
- actionet-r:
  - `cd /Users/sebastian/Documents/git_projects/actionet-r`
  - `git checkout OOM-SVD-API-update`
- Run all tests:
  - `cd /Users/sebastian/Documents/git_projects/actionet-python/tests`
  - `bash run_all_svd_tests.sh`
- Run benchmarks (with backed):
  - `bash run_benchmark.sh --include-backed`
