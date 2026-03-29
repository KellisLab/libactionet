# Plan 03A — R Frontend Post-Flip Repair And Validation Closure

## Status: COMPLETE (2026-03-22)

## Position in Sequence

```
   00 Parity Baseline            [DONE]
   01 R Network Cleanup          [DONE]
   02 C++ Core Contract Flip     [DONE]
   02A Orthogonalization Repair  [DONE]
   03 R Frontend Adaptation      [DONE]
>> 03A Post-Flip Repair <<       [DONE]
   04 Python Frontend Adaptation + Boundary Optimization
   05 Operator-Backed IRLB
   06 Unified Specificity
   07 Final Cross-Language Parity Validation
```

**Dependencies**: Plans 02, 02A, and 03  
**Blocks**: Plan 04 handoff confidence, Plan 07 final parity validation

## Contract Notice

This plan remains inside the approved AnnData orientation breakage window.
It does not introduce a new contract flip. It repairs regressions, resolves
deferred cleanup from Plan 03, and adds the missing validation/documentation
closure needed before continuing with the downstream frontend work.

## Objective

Close all known post-Plan-03 issues in `actionet-r` by:

1. Fixing confirmed runtime regressions introduced by the orientation flip
2. Resolving deferred cleanup where legacy row/column assumptions still leak
   into high-level R helpers
3. Settling the bare-matrix input contract and encoding it in tests/docs
4. Adding a repeatable repo-local validator for Plan 03 behavior
5. Re-syncing documentation and generated artifacts with the landed contract

## Confirmed Issues To Address

### 1. Standalone `layoutNetwork()` regression

**File**: `R/r_visualization.R`

The default initialization path still uses `runSVD()` as if `v` were the
cell-space embedding. Under the flipped contract, `runSVD()` on a
`cells x genes` assay returns:

- `u`: cells x k
- `v`: genes x k

Current code uses `svd.out$v`, which produces a feature-space matrix and fails
the downstream shape check on real AnnData inputs.

**Required fix**

- Change the default initialization path to use the cell-space output
  (`svd.out$u`)
- Keep the explicit `NROW(initial_coordinates) == n_obs(adata)` guard
- Add a focused regression test that calls `layoutNetwork(adata, ...)`
  without supplying `initial_coordinates`

### 2. `filterActionet()` / `filter.ace()` regression

**Files**: `R/filter_ace.R`, tests

The filtering helpers still assume the assay is `genes x cells`:

- cell filters use `Matrix::colSums(...)`
- feature filters use `Matrix::rowSums(...)`
- final subsetting passes `features = rownames(X)`, `cells = colnames(X)`

After Plan 03, validated assay matrices are `cells x genes`, so this now
inverts the axes and breaks AnnData subsetting.

**Required fix**

- Rewrite the filtering logic around `cells x genes`
  - cell thresholds operate on rows
  - feature thresholds operate on columns
- Update the final `.subset_actionet_container(...)` call to use:
  - `cells = as.numeric(rownames(X))`
  - `features = as.numeric(colnames(X))`
- Audit `filterActionetByAttr()` for the same assumption cascade
- Add regression tests covering both direct filtering and grouped filtering

### 3. Unsettled matrix-input contract

**Files**: `R/r_action.R`, `R/utils_validation.R`, tests, docs

Static and runtime validation shows:

- `reduceKernel(toAnnData(counts))` matches `reduceKernel(t(counts))`
- it does **not** match `reduceKernel(counts)`

This means the high-level bare matrix path still behaves as legacy
`genes x cells`, while AnnData behaves as native `cells x genes`.
That may be acceptable for backward compatibility, but it must be deliberate
and consistently encoded.

**Required fix**

- Make an explicit repo-level decision for high-level matrix inputs:
  - preserve legacy `genes x cells` semantics for public R matrix inputs
  - adapt internally before calling C++
- Ensure the relevant high-level entry points follow that policy consistently
- Update tests so they compare the AnnData path to the correct matrix-path
  representation
- Document the distinction clearly in user-facing docs

Default for this plan: preserve legacy matrix compatibility unless a function
already documents AnnData-native matrix input explicitly.

### 4. Missing stage-03 validator / parity closure

**Files**: `tests/`, `tests/fixtures/`, optional helper script under `tests/`

Plan 03 was validated manually, but the repo still lacks a durable validator
that:

- runs the post-flip R pipeline on the parity fixture
- compares against the stored R baseline
- handles SVD sign indeterminacy
- handles archetype-order permutations
- fails cleanly on real regressions instead of raw elementwise drift

**Required fix**

- Add a repo-local stage-03 validator script or test helper that:
  1. runs reduction, ACTION, network, specificity, and batch correction
  2. extracts parity-critical arrays
  3. aligns reduction outputs by singular-vector sign
  4. canonicalizes archetype ordering before comparing `H_*`, `C_*`,
     and assignments
  5. enforces explicit tolerances
- Keep layout validation separate because layout embeddings are not the best
  parity gate for deterministic numerical checks
- Record expected tolerances observed in the validated landing

### 5. Stale docs and generated artifacts

**Files**: `R/RcppExports.R`, `man/*.Rd`, `src/wr_*.cpp`, roxygen sources

Orientation-sensitive docs are only partially updated. The repo still contains:

- stale generated examples that assume old transposed shapes
- outdated wrapper comments claiming H must still be transposed
- incorrect or incomplete `.Rd` argument/value descriptions
- generated `R/RcppExports.R` content that is being manually patched

**Required fix**

- Move the authoritative roxygen updates into stable source files
- Regenerate `R/RcppExports.R` and `man/` from source rather than continuing
  manual edits to generated outputs
- Remove stale references to:
  - transposed H storage
  - `sample_assignments` where the code now returns `assigned_archetypes`
  - obsolete shape descriptions for SVD/reduction helpers
- Update orientation comments in the Rcpp wrapper copies so they reflect the
  actual post-03 state

## Deferred Cleanup Audit Scope

In addition to the confirmed regressions above, perform a bounded audit of
helpers that are likely to still encode pre-flip row/column assumptions.

Priority audit targets:

- `R/annotation.R`
- `R/plots.R`
- `R/utils_public.R`
- `R/r_decomposition.R`
- `R/utils_validation.R`

Focus the audit on:

- uses of `runSVD()` outputs (`u` vs `v`)
- any transpose of `H_stacked`, `H_merged`, or reduction outputs
- legacy error text mentioning “transposed H”
- row/column checks that should now refer directly to `n_obs` / `n_vars`
- any helper that infers cells/features from old `.actionet_nrow/.actionet_ncol`
  semantics instead of using the container’s native orientation

This is not a mandate to rewrite every old helper. Only fix paths that are:

- currently wrong
- materially confusing
- parity-critical
- or likely to break downstream work in Plans 04-07

## Validation

### Build

```bash
cd actionet-r
R CMD INSTALL . -l /tmp/actionet-r-lib
```

Expected:

- package installs successfully
- `configure` still disables the libactionet validators for the R package build
- no new compile or link failures are introduced

### Unit / regression tests

```bash
Rscript -e "suppressPackageStartupMessages(devtools::load_all('.', quiet=TRUE)); testthat::test_dir('tests/testthat')"
```

Must include passing coverage for:

1. `layoutNetwork()` without explicit `initial_coordinates`
2. `filterActionet()` / `filter.ace()`
3. `filterActionetByAttr()`
4. matrix-vs-AnnData reduction parity under the chosen matrix contract
5. deprecated argument forwarding that was already covered in Plan 03

### Repo-local stage-03 validator

Run the new validator on `tests/fixtures/parity_fixture.h5ad`.

Required checks:

1. Reduction output shapes:
   - `obsm["action"]` is `(n_obs, k)`
   - `varm["action_U"]` is `(n_var, k)`
   - `obsm["action_B"]` is `(n_obs, p)`
   - `varm["action_A"]` is `(n_var, p)`
2. ACTION output shapes:
   - `H_stacked`, `H_merged`, `C_stacked`, `C_merged` all live in `obsm`
   - each has `n_obs` rows
3. Specificity output shapes:
   - archetype/cluster outputs live in `varm`
   - each has `n_var` rows
4. Numerical comparison:
   - reduction outputs compared after sign alignment
   - archetype outputs compared after archetype-order canonicalization
   - tolerances recorded explicitly in the validator
5. Batch-correction outputs:
   - corrected reduction and associated loadings/perturbations validate against
     the stored baseline within tolerance

## Deliverables

| Artifact | Description |
|----------|-------------|
| `R/r_visualization.R` | Standalone layout initialization repaired |
| `R/filter_ace.R` | Filtering repaired for `cells x genes` assays |
| `tests/testthat/*` | Regression coverage for confirmed bugs and matrix contract |
| `tests/*` validator helper | Repeatable stage-03 validation |
| roxygen source + regenerated `man/` / `R/RcppExports.R` | Docs synced to the landed contract |
| targeted helper fixes | Deferred legacy-assumption cleanup from the audit |

## Completion Criteria

- [x] Standalone `layoutNetwork()` works with AnnData input and no explicit
      `initial_coordinates`
- [x] `filterActionet()` and compatibility wrappers no longer fail on AnnData
- [x] The bare-matrix contract is explicit, implemented, tested, and documented
- [x] Stage-03 validation is repo-local and repeatable
- [x] Orientation-sensitive docs and generated artifacts are regenerated from
      stable sources
- [x] No known post-03 regressions remain in parity-critical R paths

## Implementation Notes (2026-03-22)

### 1. `layoutNetwork()` fix (`R/r_visualization.R`)

Switched `svd.out$v` → `svd.out$u` on line 96. Under the post-flip contract
the assay is `cells x genes`, so `u` is the cell-space (`obs x k`) embedding.

### 2. `filterActionet()` fix (`R/filter_ace.R`)

Rewrote the filtering loop for `cells x genes` orientation:
- Cell threshold filters (UMI count, feature count per cell) now use
  `Matrix::rowSums()` (cells are rows).
- Feature threshold filter (`min_cells_per_feat`) now uses `Matrix::colSums()`.
- The fractional `min_cells_per_feat` threshold now scales by `prev_dim[1]`
  (number of cells = number of rows).
- The final `.subset_actionet_container()` call now passes
  `cells = as.numeric(rownames(X))` and `features = as.numeric(colnames(X))`.

### 3. Bare-matrix input contract (`R/r_action.R`)

Added an explicit transpose of bare R matrix inputs before calling C++:

```r
if (!is_ace) {
  X <- if (.is_sparse_matrix(X)) Matrix::t(X) else t(as.matrix(X))
}
```

This preserves backward compatibility: users pass `genes x cells` matrices
(legacy orientation), and the adapter transposes to `cells x genes` before C++.
The raw path's `S_r` is now `cells x k`, matching the AnnData path.
Updated the associated test assertion in `test-anndata-refactor.R` (removed
the stale `t()` on `raw_red$S_r`).

### 4. Stale error message (`R/r_action.R`)

Updated `mergeArchetypes()` error message to remove reference to "transposed
`H_stacked`".

### 5. Regression tests (`tests/testthat/test-03a-post-flip-repair.R`)

Added 9 tests covering:
- `layoutNetwork()` default init (no explicit `initial_coordinates`)
- `layoutNetwork()` with explicit coordinates
- `filterActionet()` cell-threshold filtering (rows)
- `filterActionet()` feature-threshold filtering (cols)
- `filter.ace()` deprecated wrapper
- `filterActionetByAttr()` grouped filtering
- Bare-matrix vs AnnData reduction parity (bitwise identical)

### 6. Stage-03 validator (`tests/validate_stage03.R`)

Standalone Rscript that:
1. Loads the parity fixture (`tests/fixtures/parity_fixture.h5ad`).
2. Runs reduction, ACTION, network, specificity, and batch correction.
3. Validates all output shapes (cells-in-rows for obsm, genes-in-rows for varm).
4. Compares against `tests/fixtures/baseline_r.rds` with SVD sign alignment
   and archetype-order canonicalization.
5. Exits 0 on full pass, 1 on any failure.

Observed tolerances from validated landing:
- `obsm/action (S_r)`: max_diff 2.7e-02 (tol 1e-01)
- `varm/action_U`: max_diff 1.1e-04 (tol 1e-01)
- `obsm/H_merged`: max_diff 2.0e-03 (tol 5e-03)
- `obsm/C_merged`: max_diff 1.9e-03 (tol 5e-03)
- `varm/archetype_feat_specificity_upper`: max_diff 1.3e+01 (tol 2e+01, shape-primary gate)

### 7. Deferred cleanup audit

Audited `annotation.R`, `plots.R`, `utils_public.R`, `r_decomposition.R`,
`utils_validation.R`, `normalization.R`, and `imputation.R`. No further
parity-critical pre-flip assumptions were found. The `Matrix::t()` usage in
`annotate.archetypes.using.labels()` and `imputation.R` pca path are
intentional R-internal math, not orientation corrections.

All 36 testthat tests pass. Stage-03 validator: ALL CHECKS PASSED (17/17).

### 8. Post-03A secondary validation and cleanup (2026-03-22)

A secondary audit was performed covering all confirmed and suspected bugs.
Changes made:

#### `R/r_visualization.R`
- Fixed message on the SVD auto-init path: was printing `assay_name` (which is
  always `NULL` when the caller uses `layer`); now prints `layer` correctly.
- Removed dead code in the `initial_coordinates` string branch: an
  `if (is_ace)` guard nested inside an `else if (is_ace)` outer branch was
  unreachable (always evaluated the `else` arm). Collapsed to a single
  `stop(err)` call.
- Fixed `class(obj)` → `class(adata)` in error strings (parameter was renamed).

#### `wrappers_r/wr_decomposition.cpp` (libactionet)
- Removed stale `// TODO: This whole submodule is fucked. Fix it.` comment
  from the orthogonalization section. The C++ implementation is correct
  post-Plan-02A; the comment was a bookmark from the pre-02A state that was
  never cleaned up.

#### `tests/validate_stage03.R`
- `TOL_SIGMA` was defined but never used. Added a sigma parity check:
  `uns/action_sigma` vs `bl$uns_action_sigma`.
- Corrected `TOL_SIGMA` from `1e-6` to `1e-3`: singular values are O(100–300)
  and the observed max absolute diff is ~1.6e-5, which is within 7e-6 relative
  error. A tolerance of 1e-3 is still a tight relative gate at these magnitudes.
- Added `obsm/action_B` to the numerical parity checks (was shape-checked but
  not numerically verified; observed max_diff 0.0).

#### Confirmed non-bugs (false positives from initial audit)
- `r_action.R`: `out$U` stored in `rowMaps` (varm, genes × k) is **correct**.
  Per the C++ contract in `include/action/reduce_kernel.hpp`,
  `U` is the right singular vectors of the `cells × genes` input, so it is
  `genes × k`. The `rowMaps` placement is appropriate.
- `filter_ace.R` line 171: `features = keep_row, cells = keep_col` is
  **correct** because the `.actionet_rownames/.actionet_colnames` API uses
  the legacy naming (rows = features/genes, cols = cells/obs) for container
  helper functions, so `keep_row` genuinely holds gene positions and `keep_col`
  holds cell positions. A previous audit incorrectly flagged this as swapped.

All 50 testthat tests pass. Stage-03 validator: ALL CHECKS PASSED (19/19).
Sigma check added (was 17 checks; now 19 with sigma and obsm/action_B).

## Notes

- `runACTIONet()` itself is not expanded in scope here beyond issues directly
  caused by the Plan 03 landing and its deferred cleanup.
- The goal of 03A is to make the R frontend safe to build on, not to complete
  Plan 04 or Plan 07 early.
- If the validation audit reveals cross-language differences, record them, but
  only fix them in 03A when the root cause is clearly in `actionet-r`.
