# Plan 03 — R Frontend Adaptation

## Position in Sequence

```
   00 Parity Baseline            [DONE]
   01 R Network Cleanup          [DONE]
   02 C++ Core Contract Flip     [DONE — libactionet headers/src updated]
>> 03 R Frontend Adaptation <<
   04 Python Frontend Adaptation + Boundary Optimization
   05 Operator-Backed IRLB
   06 Unified Specificity
   07 Final Cross-Language Parity Validation
```

**Dependencies**: Plan 02 (C++ core now expects cells x genes / cells x k).
**Blocks**: Plan 07 (final parity). Can run in parallel with Plan 04.

## Contract Notice

This plan is part of a coordinated cross-repo AnnData orientation unification.
Public API breakage across `libactionet`, `actionet-r`, and `actionet-python` is
explicitly permitted and expected until the full sequence completes.

See `context/ANNDATA_UNIFICATION_HANDOFF.md` for the complete rationale.

## Objective

Update `actionet-r` so that:

1. Expression matrices are passed to C++ in native AnnData orientation
   (cells x genes) — no `Matrix::t()` before calling C++.
2. Reduced representations (S_r, H) are passed to C++ as cells x k — no
   `Matrix::t()` before calling C++.
3. C++ results are stored directly into AnnData `obsm`/`varm`/`obsp` — no
   `Matrix::t()` after receiving from C++.
4. The AnnData adapter layer is simplified to reflect that the internal
   convention now matches AnnData orientation.

## Repo

`actionet-r` only. Assumes `libactionet` has been updated per Plan 02.

## Current State Summary

The R frontend has a centralized adapter layer:

- `.get_layer_matrix(adata, transpose = TRUE)` — default transposes
  AnnData `X` from `cells x genes` to `genes x cells`
- `.validate_assay()` calls `.get_layer_matrix(transpose = TRUE)`
- Every C++ call receives `genes x cells` and returns results that R
  transposes back before storing in `obsm`/`varm`

After Plan 02, C++ expects `cells x genes` and returns `cells x k` / etc.
The entire transpose dance must be removed.

## Detailed Changes

### Stage A: Core Adapter Layer

**File**: `R/utils_anndata_adapter.R`

#### A1: Change `.get_layer_matrix()` default

```r
# Old:
.get_layer_matrix <- function(adata, layer = NULL, transpose = TRUE, ...)

# New:
.get_layer_matrix <- function(adata, layer = NULL, transpose = FALSE, ...)
```

This single change propagates to every caller that relies on the default.
Search for all explicit `transpose = TRUE` calls and evaluate each:

- If the caller passes the matrix to C++: change to `transpose = FALSE`
  (or rely on the new default)
- If the caller genuinely needs a transposed matrix for R-internal math:
  keep `transpose = TRUE` but add a comment explaining why

#### A2: Update `.set_layer_matrix()`

Current code always transposes before writing back:

```r
.set_layer_matrix <- function(adata, layer = NULL, value) {
  value <- if (.is_sparse_matrix(value)) Matrix::t(value) else t(as.matrix(value))
  # ...
}
```

After the flip, C++ returns matrices in AnnData orientation. If
`.set_layer_matrix()` is used for C++ outputs, the transpose must be removed.
If it is also used for R-internal matrices that are still in legacy
orientation, a `transpose` parameter may be needed:

```r
.set_layer_matrix <- function(adata, layer = NULL, value, transpose = FALSE) {
  if (transpose) {
    value <- if (.is_sparse_matrix(value)) Matrix::t(value) else t(as.matrix(value))
  }
  # ...
}
```

#### A3: Update `.actionet_nrow` / `.actionet_ncol`

These currently swap dimensions to present AnnData objects in legacy
`features x cells` convention:

```r
# Old:
.actionet_nrow <- function(obj) {
  if (.is_anndata(obj)) return(.n_vars(obj))  # genes
  nrow(obj)
}
```

After the flip, the internal convention matches AnnData. Remove the swap:

```r
# New:
.actionet_nrow <- function(obj) {
  if (.is_anndata(obj)) return(.n_obs(obj))   # cells
  nrow(obj)
}
.actionet_ncol <- function(obj) {
  if (.is_anndata(obj)) return(.n_vars(obj))  # genes
  ncol(obj)
}
```

Similarly update `.actionet_rownames` / `.actionet_colnames` — these currently
swap obs/var names. After the flip, they should return names in native
AnnData order.

#### A4: Simplify `colMaps` / `rowMaps` accessors

These currently map to `obsm`/`varm` using "col"/"row" terminology from the
legacy convention (cells = columns). The mapping itself is correct (colMaps →
obsm, rowMaps → varm), but review any transpose logic inside these accessors.

If colMaps/rowMaps contain transpose operations, remove them. The naming
is legacy but the functionality should be pass-through.

### Stage B: Reduction Path

**Files**: `R/r_action.R`, `R/r_decomposition.R`

#### B1: `reduceKernel()` — expression input

Current:

```r
X <- .ace_or_assay(obj = adata, assay_name = layer, ...)
# X is genes x cells (from .validate_assay → .get_layer_matrix(transpose=TRUE))
out <- C_reduceKernelSparse(X, ...)
```

After Stage A1 (default changed), X is now `cells x genes` automatically.
No code change needed here if `.ace_or_assay` / `.validate_assay` calls
`.get_layer_matrix()` with the new default.

#### B2: `reduceKernel()` — output storage

Current:

```r
colMaps(adata)[[reduction_slot]] <- Matrix::t(S_r)  # k x cells → cells x k
rowMaps(adata)[[paste0(reduction_slot, "_U")]] <- U  # genes x k, stored as-is
```

After Plan 02, C++ returns `S_r` as `cells x k` directly:

```r
colMaps(adata)[[reduction_slot]] <- S_r              # cells x k, store directly
rowMaps(adata)[[paste0(reduction_slot, "_U")]] <- U  # genes x k, unchanged
```

Remove the `Matrix::t(S_r)` call.

#### B3: `runSVD()` — if exposed

If `runSVD()` is exposed as an R function, update its wrapper to pass
the matrix in the new orientation and document U/V roles.

### Stage C: ACTION Decomposition

**File**: `R/main.R`, `R/utils_internal_main.R`

#### C1: `runACTION()` — S_r input

Current:

```r
S_r <- colMaps(adata)[[reduction_key]]  # cells x k (from obsm)
out <- C_runACTION(S_r = Matrix::t(S_r), ...)  # transpose to k x cells for C++
```

After Plan 02, C++ accepts `cells x k`:

```r
S_r <- colMaps(adata)[[reduction_key]]  # cells x k
out <- C_runACTION(S_r = S_r, ...)      # pass directly
```

Remove the `Matrix::t(S_r)` call.

#### C2: `runACTION()` — output storage

Current:

```r
colMaps(adata)[["H_stacked"]] <- Matrix::t(as(out$H_stacked, "sparseMatrix"))
# H_stacked from C++: archetypes x cells → cells x archetypes
colMaps(adata)[["H_merged"]] <- Matrix::t(as(out$H_merged, "sparseMatrix"))
# same pattern
colMaps(adata)[["C_stacked"]] <- as(out$C_stacked, "sparseMatrix")
# C_stacked: cells x archetypes, stored as-is
```

After Plan 02, C++ returns H_stacked as `cells x archetypes`:

```r
colMaps(adata)[["H_stacked"]] <- as(out$H_stacked, "sparseMatrix")  # direct
colMaps(adata)[["H_merged"]] <- as(out$H_merged, "sparseMatrix")    # direct
colMaps(adata)[["C_stacked"]] <- as(out$C_stacked, "sparseMatrix")  # unchanged
```

Remove all `Matrix::t()` calls on H outputs.

#### C3: Internal helpers

Update `.run.collectArchetypes()` and `.run.mergeArchetypes()` in
`R/utils_internal_main.R`:

- Remove `Matrix::t(S_r)` before passing to C++
- Remove `Matrix::t(H_stacked)` before passing to C++
- Remove `Matrix::t(out$H_*)` after receiving from C++

### Stage D: Specificity

**File**: `R/r_specificity.R`

#### D1: `computeFeatureSpecificity()` — expression input

The expression matrix already comes from `.ace_or_assay()`, which now returns
`cells x genes` after Stage A1. No change needed.

#### D2: `archetypeFeatureSpecificity()` — H input

Current:

```r
H <- .validate_map(ace = adata, map_slot = map_slot, ...)
H <- Matrix::t(H)  # cells x archetypes → archetypes x cells
out <- C_archetypeFeatureSpecificitySparse(X, H = H, ...)
```

After Plan 02, C++ accepts `H` as `cells x k`:

```r
H <- .validate_map(ace = adata, map_slot = map_slot, ...)
out <- C_archetypeFeatureSpecificitySparse(X, H = H, ...)  # pass directly
```

Remove the `Matrix::t(H)` call.

### Stage E: Batch Correction

**File**: `R/batch_correct.R`

#### E1: Expression and S_r inputs

Current:

```r
X <- .ace_or_assay(...)       # genes x cells (from .get_layer_matrix(transpose=TRUE))
S_r <- colMaps(adata)[[...]]  # cells x k (from obsm)
```

After Stage A1, X is `cells x genes`. S_r is already `cells x k`. Pass both
directly to C++.

#### E2: Output storage

Current:

```r
colMaps(adata)[[name_Sr]] <- Matrix::t(S_r_out)  # k x cells → cells x k
```

After Plan 02, C++ returns S_r as `cells x k`:

```r
colMaps(adata)[[name_Sr]] <- S_r_out  # direct
```

### Stage F: Rcpp Wrappers

**File**: `src/wr_action.cpp`, `src/wr_decomposition.cpp`,
`src/wr_annotation.cpp`

These wrapper files call into `libactionet` C++ functions. After Plan 02,
the C++ functions expect the new orientation. The wrappers should:

1. **Not transpose** matrices received from R (R now passes native orientation)
2. **Not transpose** results before returning to R (C++ now returns native orientation)

#### F1: `wr_decomposition.cpp` — orthogonalization wrappers

Current pattern (appears 4+ times):

```cpp
// Reconstruct V from S_r
SVD_results(2) = S_r_mat;  // was: k x cells interpreted as V
for (size_t i = 0; i < sigma_vec.n_elem; i++) {
    SVD_results(2).col(i) /= sigma_vec(i);
}
// ... orthogonalize ...
// Return S_r as V.t()
res["S_r"] = arma::trans(V);  // k x cells
```

After the flip, S_r arrives as `cells x k` and should be returned as
`cells x k`:

```cpp
// S_r_mat is cells x k (from R, new orientation)
// V = S_r / sigma = cells x k (each column divided by sigma)
SVD_results(2) = S_r_mat;
for (size_t i = 0; i < sigma_vec.n_elem; i++) {
    SVD_results(2).col(i) /= sigma_vec(i);
}
// ... orthogonalize ...
// Return S_r directly (cells x k)
res["S_r"] = V_scaled;  // no arma::trans()
```

Trace through the exact SVD field indices to ensure U/V are assigned
correctly for the new orientation.

#### F2: `wr_action.cpp` — reduceKernel wrappers

Verify that `C_reduceKernelSparse` and `C_reduceKernelDense` pass S to
`reduceKernel()` without transposing. The wrapper should be a simple
pass-through.

#### F3: `wr_annotation.cpp` — specificity wrappers

Verify that S and H are passed through without transposing. Output matrices
(genes x k) are returned as-is.

### Stage G: Imputation

**File**: `R/imputation.R`

#### G1: `imputeFeatures()` — actionet algorithm

Current:

```r
out <- networkDiffusion(adata = adata, scores = Matrix::t(X0), ...)
# X0 is features x selected_cells → transpose to cells x features for diffusion
out <- Matrix::t(out)  # cells x features → features x cells
```

Review whether the transpose is still needed. `networkDiffusion` takes
`scores` as `cells x features` (this was already correct for the network
subsystem). The issue is that `X0` may have been extracted in legacy
orientation. After Stage A1, `.get_layer_matrix()` returns `cells x genes`,
so `X0` subsetting should yield `cells x selected_genes` directly.

Update the extraction and remove the double-transpose.

#### G2: `imputeFeatures()` — pca algorithm

```r
out <- W %*% Matrix::t(H)  # W: features x k, H: cells x k → H': k x cells
# Result: features x cells
```

This R-internal math may still need the transpose. Evaluate case by case.

### Stage H: Visualization

**File**: `R/r_visualization.R`

`layoutNetwork()` and `computeNodeColors()` already pass `cells x n_components`
directly — no transpose involved. Verify no changes are needed.

## Validation

### Build

```bash
cd actionet-r
R CMD INSTALL .
```

Must build without errors or warnings related to dimension mismatches.

### Functional regression test

Run the full pipeline on the parity fixture:

```r
library(actionet)
adata <- anndataR::read_h5ad("path/to/parity_fixture.h5ad")
reduceKernel(adata, seed = 42L)
runACTION(adata, seed = 42L)
buildNetwork(adata)
computeFeatureSpecificity(adata, ...)
archetypeFeatureSpecificity(adata, ...)
correctBatchEffect(adata, ...)
layoutNetwork(adata, seed = 42L)
```

### Parity checks

1. **Shape check**: Verify all obsm slots are `(n_obs, k)`, all varm slots
   are `(n_var, k)`, all obsp slots are `(n_obs, n_obs)`.

2. **Value check**: Compare each slot against the R baseline from Plan 00.
   Values must match within tolerance (atol=1e-6). The key difference: in the
   baseline, slots were stored via `Matrix::t()`. Now they are stored directly.
   The *values* should be identical; only the code path to store them changed.

3. **Cross-language check**: Run the comparison script from Plan 00 against
   the Python baseline (from Plan 04 or the original baseline). Document any
   pre-existing cross-language differences.

## Files Modified (Summary)

| File | Changes |
|------|---------|
| `R/utils_anndata_adapter.R` | Default transpose, dimension helpers, name helpers |
| `R/utils_validation.R` | Remove explicit `transpose = TRUE` in `.validate_assay()` |
| `R/r_action.R` | Remove `Matrix::t()` on S_r output, pass X directly |
| `R/main.R` | Remove `Matrix::t()` on S_r input and H outputs in `runACTION()` |
| `R/r_specificity.R` | Remove `Matrix::t(H)` in archetype specificity |
| `R/batch_correct.R` | Remove `Matrix::t()` on S_r output |
| `R/network_tools.R` | Already done in Plan 01 |
| `R/imputation.R` | Remove double-transpose in `imputeFeatures()` |
| `R/utils_internal_main.R` | Remove transposes in archetype collect/merge helpers |
| `src/wr_action.cpp` | Verify pass-through, remove any transposes |
| `src/wr_decomposition.cpp` | Remove `arma::trans(V)` return pattern |
| `src/wr_annotation.cpp` | Verify pass-through |

## Completion Criteria

- No `Matrix::t()` calls remain on the C++ call path (except for genuinely
  R-internal math that requires it)
- No `transpose = TRUE` in `.get_layer_matrix()` calls on the C++ path
- All AnnData slots are stored in native orientation without post-transpose
- Full pipeline runs without error
- Numerical values match the Plan 00 baseline within tolerance
- R package builds and installs cleanly
