# Plan 04 — Python Frontend Adaptation + Boundary Optimization

## Position in Sequence

```
   00 Parity Baseline            [DONE]
   01 R Network Cleanup          [optional; currently pending]
   02 C++ Core Contract Flip     [DONE]
   02A Orthogonalization Repair  [DONE]
   03 R Frontend Adaptation      [parallel follow-up; currently pending]
>> 04 Python Frontend Adaptation + Boundary Optimization <<
   05 Operator-Backed IRLB
   06 Unified Specificity
   07 Final Cross-Language Parity Validation
```

**Dependencies**: Plans 02 and 02A (`libactionet` core updated, including the
repaired orthogonalization reduction contract).
**Blocks**: Plan 07 (final parity). Can run in parallel with Plan 03.

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

1. Remove all transpose shims from the Python frontend
2. Optimize the pybind11 boundary to exploit orientation alignment
3. Eliminate unnecessary materialization copies at the Python-C++ boundary

This plan covers both the functional adaptation (removing transposes) and the
performance optimization (reducing copies), because they are tightly coupled —
the copy reduction is only possible once the orientation matches.

## Repo

`actionet-python` only. Assumes `libactionet` has been updated per Plan 02.

## Current State Summary

Python currently performs 8 `anndata_to_matrix(transpose=True)` calls and
numerous `.T` operations at the C++ boundary. The pybind11 layer copies every
matrix element-by-element due to row-major/column-major mismatch.

Key overhead points (for a 30k x 100k sparse matrix at 5% density):

| Point | Operation | Peak cost |
|-------|-----------|-----------|
| `anndata_to_matrix(transpose=True)` | `X.T.tocsr()` | ~1.8 GB |
| `scipy_to_arma_sparse()` | tocsc + COO + sp_mat | ~8 GB transient |
| `numpy_to_arma_mat()` | element-by-element copy | ~24 GB for dense |
| Result `.T` operations | `S_r.T`, `H_stacked.T` | small |

## Detailed Changes

### Stage A: Remove Transpose Shims

#### A1: `anndata_to_matrix()` — the central transpose

**File**: `src/actionet/anndata_utils.py`

Current:

```python
def anndata_to_matrix(adata, layer=None, transpose=False):
    X = adata.X if layer is None else adata.layers[layer]
    if transpose:
        if sp.issparse(X):
            X = X.T.tocsr()
        else:
            X = X.T
    return X
```

The `transpose` parameter can be deprecated or removed. All current callers
that pass `transpose=True` should be updated to pass `transpose=False` (or
no argument). The function simplifies to just extracting the matrix.

Alternatively, keep the parameter for backward compatibility but change the
semantics so callers stop using it:

```python
def anndata_to_matrix(adata, layer=None, transpose=False):
    X = adata.X if layer is None else adata.layers[layer]
    if transpose:
        warnings.warn("transpose=True is deprecated; C++ now accepts native orientation",
                       DeprecationWarning)
        X = X.T.tocsr() if sp.issparse(X) else X.T
    return X
```

#### A2: `reduce_kernel()` — expression input and output

**File**: `src/actionet/core.py`

Current (approximately line 279):

```python
S = anndata_to_matrix(adata, layer=layer, transpose=True)  # cells x genes → genes x cells
```

Change to:

```python
S = anndata_to_matrix(adata, layer=layer)  # cells x genes, pass directly
```

Current output storage (approximately lines 309-317):

```python
obsm={
    key_added: result["S_r"].T,           # k x cells → cells x k
    f"{key_added}_B": result["B"],        # cells x p, direct
},
varm={
    f"{key_added}_U": result["U"],        # genes x k, direct
    f"{key_added}_A": result["A"],        # genes x p, direct
},
```

After Plan 02, C++ returns `S_r` as `cells x k`:

```python
obsm={
    key_added: result["S_r"],             # cells x k, direct — no .T
    f"{key_added}_B": result["B"],        # unchanged
},
varm={
    f"{key_added}_U": result["U"],        # unchanged
    f"{key_added}_A": result["A"],        # unchanged
},
```

#### A3: `run_action()` — S_r input and H output

**File**: `src/actionet/core.py`

Current (approximately lines 430-452):

```python
S_r = adata.obsm[reduction_key].T        # cells x k → k x cells
S_r = np.ascontiguousarray(S_r)
# ...
result = _core.run_action(S_r, ...)
# ...
obsm={
    "H_stacked": result["H_stacked"].T,  # archetypes x cells → cells x archetypes
    "H_merged": result["H_merged"].T,
    "C_stacked": result["C_stacked"],
    "C_merged": result["C_merged"],
},
```

After Plan 02:

```python
S_r = np.ascontiguousarray(adata.obsm[reduction_key])  # cells x k, direct
result = _core.run_action(S_r, ...)
obsm={
    "H_stacked": result["H_stacked"],    # cells x archetypes, direct
    "H_merged": result["H_merged"],      # direct
    "C_stacked": result["C_stacked"],    # unchanged
    "C_merged": result["C_merged"],      # unchanged
},
```

#### A4: `run_svd()` — expression input

**File**: `src/actionet/core.py`

Current (approximately lines 1250-1257):

```python
result = _core.run_svd_sparse(matrix.T, ...)
result = _core.run_svd_dense(matrix.T, ...)
```

After:

```python
result = _core.run_svd_sparse(matrix, ...)  # cells x genes, direct
result = _core.run_svd_dense(matrix, ...)
```

Note: the SVD result U/V roles swap. With old orientation (genes x cells),
U = genes x k, V = cells x k. With new (cells x genes), U = cells x k,
V = genes x k. Callers that use U as gene loadings now need to read V.
Review all `run_svd` callers and update accordingly.

#### A5: `correct_batch_effect()` / `correct_basal_expression()`

**File**: `src/actionet/batch_correction.py`

Remove `anndata_to_matrix(transpose=True)` calls. Remove `result["S_r"].T`
post-transposes.

Current:

```python
S = anndata_to_matrix(adata, layer=layer, transpose=True)  # genes x cells
# ...
obsm={corrected_key: result["S_r"].T}                      # k x cells → cells x k
```

After:

```python
S = anndata_to_matrix(adata, layer=layer)                   # cells x genes
# ...
obsm={corrected_key: result["S_r"]}                         # cells x k, direct
```

#### A5b: `wp_decomposition.cpp` — operator orthogonalization wrappers

After Plan 02A, `libactionet` no longer expects a manually reconstructed
`SVDResult` in the orthogonalization operator path. The pybind wrapper should
pass the public reduction contract directly:

```cpp
actionet::KernelReductionResult reduction;
reduction.S_r = S_r_mat;      // cells x k
reduction.sigma = sigma_vec;  // k
reduction.U = U_mat;          // genes x k
reduction.A = A_mat;          // genes x p
reduction.B = B_mat;          // cells x p

actionet::KernelReductionResult result =
    actionet::orthogonalizeBatchEffect_Operator(*op, reduction, design_mat);
```

Apply the same pattern to `orthogonalizeBasal_Operator(...)`.

Remove the old wrapper-side reconstruction:

```cpp
// Old legacy reconstruction — remove this
svd.U = U_mat;
svd.sigma = sigma_vec;
svd.V = S_r_mat;
for (size_t i = 0; i < sigma_vec.n_elem; i++) {
    svd.V.col(i) /= sigma_vec(i);
}
```

and return the corrected reduction directly:

```cpp
out["S_r"] = arma_mat_to_numpy(result.S_r);  // cells x k
out["U"] = arma_mat_to_numpy(result.U);      // genes x k
out["A"] = arma_mat_to_numpy(result.A);
out["B"] = arma_mat_to_numpy(result.B);
out["sigma"] = arma_vec_to_numpy(result.sigma);
```

#### A6: `compute_feature_specificity()` / `compute_archetype_feature_specificity()`

**File**: `src/actionet/core.py`

Remove `anndata_to_matrix(transpose=True)` for expression matrix. Remove
`H.T` for archetype specificity (C++ now accepts `cells x k`).

Current:

```python
S = anndata_to_matrix(adata, layer=layer, transpose=True)   # genes x cells
H_t = np.ascontiguousarray(H.T, dtype=np.float64)           # cells x k → k x cells
result = _core.archetype_feature_specificity_sparse(S, H_t, ...)
```

After:

```python
S = anndata_to_matrix(adata, layer=layer)                    # cells x genes
H = np.ascontiguousarray(adata.obsm[...], dtype=np.float64)  # cells x k
result = _core.archetype_feature_specificity_sparse(S, H, ...)
```

#### A7: `annotate_cells()`

**File**: `src/actionet/annotation.py`

Current:

```python
S = source.matrix
S = csr_matrix(S).T   # cells x genes → genes x cells
```

After:

```python
S = csr_matrix(source.matrix)  # cells x genes, direct
```

Same for backed path — remove `.T.tocsr()`.

#### A8: `impute_features()`

**File**: `src/actionet/imputation.py`

Remove `anndata_to_matrix(transpose=True)`. Remove the double-transpose
pattern in feature extraction and diffusion.

#### A9: Full inventory of `.T` removals

Every `.T` operation listed in the investigation must be evaluated:

| File | Line (approx) | Expression | Action |
|------|--------------|-----------|--------|
| `anndata_utils.py` | 38-40 | `X.T.tocsr()` / `X.T` | Remove (inside anndata_to_matrix) |
| `core.py` | 279 | `anndata_to_matrix(transpose=True)` | Remove transpose |
| `core.py` | 311 | `result["S_r"].T` | Remove .T |
| `core.py` | 430 | `adata.obsm[...].T` | Remove .T |
| `core.py` | 446-447 | `result["H_*"].T` | Remove .T |
| `core.py` | 1021 | `H.T` | Remove .T |
| `core.py` | 1041 | `H.T` | Remove .T |
| `core.py` | 1254 | `matrix.T` | Remove .T |
| `core.py` | 1257 | `matrix.T` | Remove .T |
| `batch_correction.py` | 116 | `anndata_to_matrix(transpose=True)` | Remove transpose |
| `batch_correction.py` | 131 | `result["S_r"].T` | Remove .T |
| `batch_correction.py` | 204 | `anndata_to_matrix(transpose=True)` | Remove transpose |
| `batch_correction.py` | 218 | `result["S_r"].T` | Remove .T |
| `annotation.py` | 390 | `S_cells.T.tocsr()` | Remove .T.tocsr() |
| `annotation.py` | 395 | `S.T` | Remove .T |
| `imputation.py` | 104 | `X0_cells.T` | Remove .T |
| `imputation.py` | 106 | `anndata_to_matrix(transpose=True)` | Remove transpose |

Some `.T` operations are for genuinely Python-internal math (e.g.,
`annotation.py:447` for graph normalization, `annotation.py:680` for
enrichment reshaping). These should be kept. Evaluate each individually.

### Stage B: Pybind11 Boundary Optimization

After Stage A, the orientations match: Python sends `cells x genes`
(C-contiguous, row-major) and C++ expects `cells x genes` (Armadillo stores
column-major). This enables significant optimization.

#### B1: Optimize `scipy_to_arma_sparse()` — direct CSC handoff

**File**: `src/actionet/wp_utils.cpp`

Current path: input CSR → `.tocsc()` → extract arrays → build COO → construct
`arma::sp_mat` from COO.

AnnData typically stores `X` as CSC (`scipy.sparse.csc_matrix`). After
removing the transpose, the input to `scipy_to_arma_sparse` is already CSC.

New implementation:

```cpp
arma::sp_mat scipy_csc_to_arma(py::object scipy_csc) {
    // Ensure CSC
    py::object csc = scipy_csc.attr("tocsc")();  // no-op if already CSC
    py::array_t<double> data = csc.attr("data").cast<py::array_t<double>>();
    py::array_t<int> indices = csc.attr("indices").cast<py::array_t<int>>();
    py::array_t<int> indptr = csc.attr("indptr").cast<py::array_t<int>>();
    auto shape = csc.attr("shape").cast<py::tuple>();
    arma::uword n_rows = shape[0].cast<arma::uword>();
    arma::uword n_cols = shape[1].cast<arma::uword>();
    arma::uword nnz = data.size();

    // Construct arma::sp_mat directly from CSC arrays
    // Armadillo sp_mat constructor: sp_mat(rowind, colptr, values, n_rows, n_cols)
    arma::uvec rowind(nnz);
    arma::uvec colptr(n_cols + 1);
    arma::vec values(nnz);

    auto ind_ptr = indices.unchecked<1>();
    auto ptr_ptr = indptr.unchecked<1>();
    auto dat_ptr = data.unchecked<1>();

    for (arma::uword i = 0; i < nnz; ++i) {
        rowind(i) = static_cast<arma::uword>(ind_ptr(i));
        values(i) = dat_ptr(i);
    }
    for (arma::uword i = 0; i <= n_cols; ++i) {
        colptr(i) = static_cast<arma::uword>(ptr_ptr(i));
    }

    return arma::sp_mat(rowind, colptr, values, n_rows, n_cols);
}
```

This eliminates:
- The COO intermediary (no `locations` matrix)
- The internal sort (CSC arrays are already sorted)
- The tocsc() conversion (input is already CSC)

If Armadillo does not have a direct CSC constructor with separate rowind and
colptr, use the batch constructor or the internal `init_from_csc()` if
available. Alternatively, construct via the existing `sp_mat(locations, values)`
but pre-build the locations from CSC arrays more efficiently than the current
code.

Research Armadillo's API for the most efficient CSC construction path.

#### B2: Optimize `numpy_to_arma_mat()` — reduce copy overhead

**File**: `src/actionet/wp_utils.cpp`

Option 1 — Fortran-order zero-copy:

```cpp
arma::mat numpy_to_arma_mat(py::array_t<double, py::array::f_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    auto ptr = static_cast<double*>(buf.ptr);
    // Armadillo is column-major = Fortran-order → zero-copy possible
    return arma::mat(ptr, buf.shape[0], buf.shape[1], /*copy_aux_mem=*/false, /*strict=*/true);
}
```

This requires the Python side to provide Fortran-order arrays.  Add
`np.asfortranarray()` calls at the Python boundary for matrices going to C++.
For small matrices (S_r, H, design matrices), this is negligible. For large
matrices (expression), the Fortran conversion is still O(n*m) but avoids the
element-by-element Python→C++ copy.

Option 2 — Fast memcpy with layout flip:

```cpp
arma::mat numpy_to_arma_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();
    auto ptr = static_cast<double*>(buf.ptr);
    arma::uword n_rows = buf.shape[0];
    arma::uword n_cols = buf.shape[1];
    // C-contiguous (n_rows, n_cols) row-major = Fortran-contiguous (n_cols, n_rows)
    // Construct transposed, then transpose — uses optimized Armadillo transpose
    arma::mat tmp(ptr, n_cols, n_rows, /*copy=*/false, /*strict=*/true);
    return tmp.t();  // Armadillo in-place transpose for square, or fast copy for non-square
}
```

This avoids the element-by-element loop but still copies for non-square
matrices. For the common case of narrow matrices (cells x k where k << cells),
Armadillo's `mat.t()` is efficient.

Choose Option 1 for large matrices (expression), Option 2 as fallback.

#### B3: Optimize `arma_mat_to_numpy()` — return path

**File**: `src/actionet/wp_utils.cpp`

Similar optimization in reverse. Return a Fortran-order NumPy array to
avoid the element-by-element copy:

```cpp
py::array_t<double> arma_mat_to_numpy(const arma::mat& mat) {
    // Return Fortran-order array — column-major matches Armadillo
    std::vector<ssize_t> shape = {(ssize_t)mat.n_rows, (ssize_t)mat.n_cols};
    std::vector<ssize_t> strides = {(ssize_t)sizeof(double),
                                     (ssize_t)(mat.n_rows * sizeof(double))};
    auto arr = py::array_t<double>(shape, strides);
    std::memcpy(arr.mutable_data(), mat.memptr(), mat.n_elem * sizeof(double));
    return arr;
}
```

This replaces the O(n*m) element-by-element loop with a single `memcpy`.
The returned array is Fortran-ordered, which NumPy handles natively.

#### B4: Optimize `arma_sparse_to_scipy()` — return path

**File**: `src/actionet/wp_utils.cpp`

Armadillo `sp_mat` is internally CSC. Return a CSC matrix directly instead
of converting to CSR:

```cpp
py::object arma_sparse_to_scipy_csc(const arma::sp_mat& sp_mat) {
    // Build CSC arrays directly from Armadillo internals
    // sp_mat.col_ptrs, sp_mat.row_indices, sp_mat.values
    // Construct scipy.sparse.csc_matrix((data, indices, indptr), shape=...)
}
```

This avoids the double-iteration CSC→CSR conversion.

### Stage C: Backed/Streaming Path

#### C1: `_matrix_source.py` — streamed matvec

**File**: `src/actionet/_matrix_source.py`

If the streamed matvec implementation uses `.T` to handle orientation,
update it to use the native orientation. The operator-backed path should
require no transpose after the backed operator flip in Plan 02 Stage E.

#### C2: Backed specificity path

The backed sparse specificity path passes `H.T` to C++. After Plan 02,
C++ accepts `H` as `cells x k`:

```python
# Old:
H_t = np.ascontiguousarray(H.T, dtype=np.float64)
result = _run_specificity_backed_sparse(adata, ..., H=H_t, ...)

# New:
H = np.ascontiguousarray(adata.obsm[...], dtype=np.float64)
result = _run_specificity_backed_sparse(adata, ..., H=H, ...)
```

## Validation

### Build

```bash
cd actionet-python
pip install -e .
# or:
python -m pip install -e . --no-build-isolation
```

Must build without errors.

### Functional regression test

```python
import actionet
import anndata

adata = anndata.read_h5ad("path/to/parity_fixture.h5ad")
actionet.reduce_kernel(adata, seed=42)
actionet.run_action(adata, seed=42)
actionet.build_network(adata)
actionet.compute_feature_specificity(adata, ...)
actionet.compute_archetype_feature_specificity(adata, ...)
actionet.correct_batch_effect(adata, ...)
actionet.layout_network(adata, seed=42)
```

### Parity checks

1. **Shape check**: All obsm slots `(n_obs, k)`, varm slots `(n_var, k)`,
   obsp slots `(n_obs, n_obs)`.

2. **Value check**: Compare against Python baseline from Plan 00. Values
   must match within tolerance. The code path changed (no transposes) but
   the numerical result must be identical.

3. **Performance check**: For a medium dataset (~50k cells), measure:
   - Peak memory of `reduce_kernel` (should drop significantly)
   - Wall time of `reduce_kernel` (should drop due to eliminated copies)
   - Compare before/after using `/usr/bin/time -v` or `tracemalloc`

4. **Cross-language check**: Compare against R outputs from Plan 03.

## Files Modified (Summary)

| File | Changes |
|------|---------|
| `src/actionet/anndata_utils.py` | Deprecate/remove transpose parameter |
| `src/actionet/core.py` | Remove all `.T` operations at C++ boundary |
| `src/actionet/batch_correction.py` | Remove transpose in/out |
| `src/actionet/annotation.py` | Remove `.T` on expression matrix |
| `src/actionet/imputation.py` | Remove transpose in/out |
| `src/actionet/_matrix_source.py` | Update streamed matvec orientation |
| `src/actionet/wp_utils.cpp` | Optimize sparse/dense transport |
| `src/actionet/wp_utils.h` | Update function signatures if needed |

## Completion Criteria

- Zero `transpose=True` calls to `anndata_to_matrix` on the C++ path
- Zero `.T` operations on matrices crossing the C++ boundary (except
  genuinely Python-internal math)
- Sparse transport avoids COO intermediary
- Dense transport uses memcpy or zero-copy instead of element-by-element
- Full pipeline produces identical values to baseline
- Measurable memory reduction on a ~50k cell dataset
