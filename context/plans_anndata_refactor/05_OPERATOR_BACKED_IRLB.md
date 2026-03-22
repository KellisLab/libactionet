# Plan 05 — Operator-Backed IRLB

## Position in Sequence

```
   00 Parity Baseline            [DONE]
   01 R Network Cleanup          [optional; currently pending]
   02 C++ Core Contract Flip     [required; currently pending]
   03 R Frontend Adaptation      [parallel follow-up; currently pending]
   04 Python Frontend Adaptation [parallel follow-up; currently pending]
>> 05 Operator-Backed IRLB <<
   06 Unified Specificity
   07 Final Cross-Language Parity Validation
```

**Dependencies**: Plan 02 (backed operators expose new orientation).
**Blocks**: None strictly, but enables backed `reduceKernel` in R.
**Can run in parallel with**: Plans 03 and 04.

## Contract Notice

This plan is part of a coordinated cross-repo AnnData orientation unification.
Public API breakage across `libactionet`, `actionet-r`, and `actionet-python` is
explicitly permitted and expected until the full sequence completes.

See `context/ANNDATA_UNIFICATION_HANDOFF.md` for the complete rationale.

## Agent Execution Note

Implementing agents may create repo-local virtual environments or temporary
environments under `.venv` or `/tmp` and install Python, R, or build
dependencies as needed to run the validation steps in this plan.

Prefer isolated environments over modifying unrelated global environments, and
record any nontrivial setup commands in the handoff.

## Objective

Add `MatrixOperator` support to the IRLB SVD implementation, then remove the
compile-time gate that blocks operator-backed `reduceKernel` in R builds.

### Why this matters

- IRLB is the default SVD algorithm in both R and Python
- `runSVD_Operator` currently throws a runtime error for IRLB
- `reduceKernel_Operator` is hard-blocked in R builds (`#if LIBACTIONET_BUILD_R`)
  because PRIMME (the only operator-capable default) is excluded from R
- Without this fix, R cannot use backed H5AD operators for `reduceKernel`,
  forcing full expression matrix materialization even for on-disk datasets
- Python can use Halko/Feng operator paths, but gains IRLB as a familiar
  default

## Repo

`libactionet` only.

## Current State

### IRLB implementation (`src/decomposition/svd_irbla.cpp`)

Two overloads exist:

```cpp
arma::field<arma::mat> svdIRLB(arma::sp_mat& A, int dim, int iters, int seed, bool verbose);
arma::field<arma::mat> svdIRLB(arma::mat& A, int dim, int iters, int seed, bool verbose);
```

Both implement Golub-Kahan-Lanczos bidiagonalization with implicit restart.
The matrix `A` is accessed only through two operations:

1. **Forward product** (`y = A * x`):
   - Sparse: `sparse_matvec('n', A, x, W + j*m)` → `A * x_vec`
   - Dense: `y = A * v`

2. **Transpose product** (`y = A' * x`):
   - Sparse: `sparse_matvec('t', A, W + j*m, F)` → `A.t() * x_vec`
   - Dense: `y = arma::trans(arma::trans(v) * A)`

Everything else (Lanczos iteration, bidiagonal B matrix, convergence tests,
`cblas_dgemm` for restarting, output construction) is matrix-independent.

### SVD dispatch (`src/decomposition/svd_main.cpp`)

```cpp
// Operator dispatch
SVDResult runSVD_Operator(const MatrixOperator& op, ...) {
    switch (algorithm) {
        case ALG_HALKO:  return runSVD_Halko_Operator(op, ...);
        case ALG_FENG:   return runSVD_Feng_Operator(op, ...);
        case ALG_PRIMME: return runSVD_PRIMME_Operator(op, ...);  // gated for R
        case ALG_IRLB:
        default:
            throw std::runtime_error("Operator-backed IRLB is unsupported");
    }
}
```

### R operator gate (`src/action/reduce_kernel.cpp`)

```cpp
KernelReductionResult reduceKernel_Operator(const MatrixOperator& S, ...) {
#if defined(LIBACTIONET_BUILD_R) && LIBACTIONET_BUILD_R == 1
    throw std::runtime_error("reduceKernel_Operator is unavailable in R build mode");
#else
    // ... normal operator path ...
#endif
}
```

## Detailed Changes

### Stage A: Add IRLB MatrixOperator Overload

**Files**: `include/decomposition/svd_irbla.hpp`, `src/decomposition/svd_irbla.cpp`

#### A1: Add header declaration

```cpp
// svd_irbla.hpp — add third overload
arma::field<arma::mat> svdIRLB(const MatrixOperator& A, int dim,
                                int iters = 1000, int seed = 0,
                                bool verbose = true);
```

#### A2: Implement the operator overload

The implementation is a mechanical adaptation of the existing sparse overload.
The structure is identical — only the matrix-vector product calls change.

Create a helper function or adapt the `sparse_matvec` pattern:

```cpp
static void operator_matvec(char transpose, const MatrixOperator& A,
                             const double* x, double* out) {
    if (transpose == 'n') {
        arma::vec x_vec(const_cast<double*>(x), A.cols(), false, true);
        arma::vec result(out, A.rows(), false, true);
        A.matvec(x_vec, result);
    } else {
        arma::vec x_vec(const_cast<double*>(x), A.rows(), false, true);
        arma::vec result(out, A.cols(), false, true);
        A.rmatvec(x_vec, result);
    }
}
```

Then duplicate the sparse IRLB function body, replacing:
- `sparse_matvec('n', A, ...)` → `operator_matvec('n', A, ...)`
- `sparse_matvec('t', A, ...)` → `operator_matvec('t', A, ...)`
- `m = A.n_rows` → `m = A.rows()`
- `n = A.n_cols` → `n = A.cols()`

All other code (workspace allocation, Lanczos iteration, bidiagonal SVD via
`cblas_*` / LAPACK, convergence check, output assembly, `orient_SVD` call)
remains identical.

#### A3: Alternative — refactor to shared template

Instead of duplicating the function body, refactor IRLB to use a strategy
pattern for matrix-vector products:

```cpp
template <typename MatVecFn, typename RMatVecFn>
arma::field<arma::mat> svdIRLB_impl(arma::uword m, arma::uword n,
                                     MatVecFn matvec, RMatVecFn rmatvec,
                                     int dim, int iters, int seed, bool verbose);
```

Then the three public overloads become thin wrappers:

```cpp
arma::field<arma::mat> svdIRLB(arma::sp_mat& A, ...) {
    return svdIRLB_impl(A.n_rows, A.n_cols,
        [&](const double* x, double* y) { /* A*x */ },
        [&](const double* x, double* y) { /* A'*x */ },
        dim, iters, seed, verbose);
}

arma::field<arma::mat> svdIRLB(const MatrixOperator& A, ...) {
    return svdIRLB_impl(A.rows(), A.cols(),
        [&](const double* x, double* y) { A.matvec(...); },
        [&](const double* x, double* y) { A.rmatvec(...); },
        dim, iters, seed, verbose);
}
```

This is cleaner but requires more refactoring. Choose based on confidence
in the IRLB internals — if the code is stable and well-understood, the
template approach is preferred. If it is fragile, duplicate-and-adapt is
safer.

### Stage B: Update SVD Dispatch

**File**: `src/decomposition/svd_main.cpp`

Replace the `throw` with a dispatch:

```cpp
case ALG_IRLB:
default: {
    arma::field<arma::mat> result = svdIRLB(op, k, max_it, seed, verbose);
    return svdResultFromField(result);
}
```

The `svdResultFromField` helper already exists and converts the
`arma::field<arma::mat>` return format to the `SVDResult` struct.

### Stage C: Remove the R Operator Gate

**File**: `src/action/reduce_kernel.cpp`

Remove the `#if defined(LIBACTIONET_BUILD_R)` guard:

```cpp
// Old:
KernelReductionResult reduceKernel_Operator(const MatrixOperator& S, ...) {
#if defined(LIBACTIONET_BUILD_R) && LIBACTIONET_BUILD_R == 1
    throw std::runtime_error("reduceKernel_Operator is unavailable in R build mode");
#else
    SVDResult svd = runSVD_Operator(S, k, max_it, seed, svd_alg, verbose);
    return reduceKernelFromSVD_Operator(S, svd, verbose);
#endif
}

// New:
KernelReductionResult reduceKernel_Operator(const MatrixOperator& S, ...) {
    SVDResult svd = runSVD_Operator(S, k, max_it, seed, svd_alg, verbose);
    return reduceKernelFromSVD_Operator(S, svd, verbose);
}
```

The PRIMME case in `runSVD_Operator` is still gated by `LIBACTIONET_BUILD_R`,
so R builds will use IRLB (the default) instead of PRIMME.

### Stage D: Update R Wrapper Gate (if present)

**File**: `wrappers_r/wr_decomposition.cpp` (and `actionet-r/src/wr_decomposition.cpp`)

If the R wrapper has an `Rcpp::stop()` guard that mirrors the C++ gate,
remove it as well.

## Validation

### Build — both modes

```bash
# Standard build (PRIMME available)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# R build (PRIMME excluded)
cmake .. -DCMAKE_BUILD_TYPE=Release -DLIBACTIONET_BUILD_R=1
make -j$(nproc)
```

Both must compile without errors.

### Numerical parity — IRLB operator vs template

Write a C++ test that:

1. Creates a small dense matrix (e.g., 200 x 100)
2. Wraps it in a `DenseMatrixOperator`
3. Runs `svdIRLB(dense_mat, k=10, seed=42)`
4. Runs `svdIRLB(operator, k=10, seed=42)`
5. Compares U, sigma, V — should be identical (same random seed, same
   algorithm, same matrix-vector products)

```cpp
arma::mat A = arma::randn(200, 100);
DenseMatrixOperator op(A);

auto result_template = svdIRLB(A, 10, 1000, 42, false);
auto result_operator = svdIRLB(op, 10, 1000, 42, false);

// Compare sigma
assert(arma::approx_equal(result_template(1), result_operator(1), "absdiff", 1e-10));
// Compare U, V (with sign canonicalization)
```

### Numerical parity — operator reduceKernel

Using the parity fixture:

1. Load expression matrix as dense (cells x genes after Plan 02)
2. Run `reduceKernel(sparse_mat, k=20, seed=42)` (template path)
3. Wrap the same matrix in a `DenseMatrixOperator`
4. Run `reduceKernel_Operator(op, k=20, seed=42)` (operator path)
5. Compare S_r, sigma, U — should match within tolerance

### Integration — backed H5AD in R

After Plans 03 and 05 are both done, test the full backed path in R:

```r
# Open H5AD without loading into memory
# (requires R backed operator support — may need additional R wrapper work)
adata <- anndataR::read_h5ad("path/to/parity_fixture.h5ad", backed = TRUE)
reduceKernel(adata, seed = 42L)
```

This verifies the complete chain: backed operator → IRLB → reduceKernel →
AnnData storage.

## Files Modified (Summary)

| File | Changes |
|------|---------|
| `include/decomposition/svd_irbla.hpp` | Add operator overload declaration |
| `src/decomposition/svd_irbla.cpp` | Add operator overload implementation |
| `src/decomposition/svd_main.cpp` | Enable IRLB in operator dispatch |
| `src/action/reduce_kernel.cpp` | Remove R build gate |

## Files NOT Modified

| File | Reason |
|------|--------|
| `src/decomposition/svd_halko.cpp` | Already has operator support |
| `src/decomposition/svd_feng.cpp` | Already has operator support |
| `src/decomposition/svd_primme.cpp` | Remains R-excluded, no change |
| `cmake/ConfigurePRIMME.cmake` | PRIMME exclusion unchanged |

## Completion Criteria

- `svdIRLB(const MatrixOperator&, ...)` compiles and produces correct results
- `runSVD_Operator` dispatches IRLB without throwing
- `reduceKernel_Operator` compiles in both standard and R build modes
- Operator IRLB produces bit-compatible results with template IRLB on same input
- R build compiles without errors
