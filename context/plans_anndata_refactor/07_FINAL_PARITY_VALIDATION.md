# Plan 07 — Final Cross-Language Parity Validation

## Position in Sequence

```
   00 Parity Baseline            [DONE]
   01 R Network Cleanup          [DONE]
   02 C++ Core Contract Flip     [DONE]
   03 R Frontend Adaptation      [DONE]
   04 Python Frontend Adaptation [DONE]
   05 Operator-Backed IRLB       [DONE]
   06 Unified Specificity        [DONE]
>> 07 Final Cross-Language Parity Validation <<
```

**Dependencies**: All prior plans (00–06).
**Blocks**: Nothing — this is the final validation step.

## Contract Notice

This plan concludes the AnnData orientation unification. After this plan
passes, the contract breakage period is over. The new orientation contract
is the permanent API.

## Objective

Confirm that all three repos produce identical (within tolerance) outputs
for the same input, across all algorithms and storage modes. Document any
remaining differences. Establish the parity test suite as a permanent CI
artifact.

## Scope

This plan touches all three repos but only adds test infrastructure — no
functional code changes.

## Validation Matrix

### Cross-Language Parity (R vs Python)

Run the full pipeline on the parity fixture in both languages and compare
every output slot.

| Slot | R source | Python source | Comparison |
|------|----------|--------------|------------|
| `obsm["action"]` (cells x k) | `colMaps(adata)[["action"]]` | `adata.obsm["action"]` | SVD sign canonicalization + allclose |
| `uns sigma` (k,) | `adata$uns[["action_params"]][["sigma"]]` | `adata.uns["action_params"]["sigma"]` | allclose |
| `varm["action_U"]` (genes x k) | `rowMaps(adata)[["action_U"]]` | `adata.varm["action_U"]` | SVD sign canonicalization + allclose |
| `varm["action_A"]` (genes x p) | `rowMaps(adata)[["action_A"]]` | `adata.varm["action_A"]` | allclose |
| `obsm["action_B"]` (cells x p) | `colMaps(adata)[["action_B"]]` | `adata.obsm["action_B"]` | allclose |
| `obsm["H_stacked"]` (cells x archs) | `colMaps(adata)[["H_stacked"]]` | `adata.obsm["H_stacked"]` | Archetype ordering + allclose |
| `obsm["H_merged"]` (cells x archs) | `colMaps(adata)[["H_merged"]]` | `adata.obsm["H_merged"]` | Archetype ordering + allclose |
| `obsm["C_stacked"]` | `colMaps(adata)[["C_stacked"]]` | `adata.obsm["C_stacked"]` | Archetype ordering + allclose |
| `obsm["C_merged"]` | `colMaps(adata)[["C_merged"]]` | `adata.obsm["C_merged"]` | Archetype ordering + allclose |
| `obs["assigned_archetype"]` | `adata$obs[["assigned_archetype"]]` | `adata.obs["assigned_archetype"]` | Archetype ordering + exact match |
| `obsp["actionet"]` (cells x cells) | `colNets(adata)[["actionet"]]` | `adata.obsp["actionet"]` | CSR normalization + allclose |
| Cluster specificity (genes x k) | `rowMaps(adata)[["specificity_*"]]` | `adata.varm["specificity_*"]` | allclose |
| Archetype specificity (genes x k) | `rowMaps(adata)[["arch_specificity_*"]]` | `adata.varm["arch_specificity_*"]` | Archetype ordering + allclose |
| Corrected reduction (cells x k) | `colMaps(adata)[["corrected"]]` | `adata.obsm["corrected"]` | SVD sign canonicalization + allclose |

### Intra-Language Regression (vs Plan 00 baseline)

Each language's output must match its own Plan 00 baseline within tolerance.
The values are the same; only the storage path changed (no more transposes
at the boundary).

| Check | Expected result |
|-------|----------------|
| Python new vs Python baseline | Values identical, shapes identical |
| R new vs R baseline | Values identical, shapes identical |
| Python new vs R new | Values match within tolerance (cross-language) |

### Intra-Language Storage Parity

Within each language, compare outputs from different storage modes:

| Mode A | Mode B | Expected |
|--------|--------|----------|
| In-memory sparse | In-memory dense | Identical |
| In-memory sparse | Backed sparse (operator) | Identical |
| In-memory dense | Backed dense (operator) | Identical |

This confirms that the operator paths produce the same results as the
in-memory paths.

## Canonicalization Rules

### SVD Sign Canonicalization

For each column of U (or V, or S_r), flip the sign so that the element with
the largest absolute value is positive. Apply the same flip to the
corresponding column in the paired matrix (if U is flipped, flip V's
corresponding column too).

```python
def canonicalize_svd_signs(U, V):
    for j in range(U.shape[1]):
        max_idx = np.argmax(np.abs(U[:, j]))
        if U[max_idx, j] < 0:
            U[:, j] *= -1
            V[:, j] *= -1
    return U, V
```

### Archetype Ordering Canonicalization

Archetypes may be discovered in different orders. Sort columns of H by
descending L2 norm, then apply the same permutation to C, assigned labels, etc.

```python
def canonicalize_archetype_order(H, C):
    norms = np.linalg.norm(H, axis=0)
    order = np.argsort(-norms)
    return H[:, order], C[:, order]
```

### Sparse Graph Normalization

Convert to CSR, sort indices within each row, then compare.

```python
def normalize_sparse(G):
    G = G.tocsr()
    G.sort_indices()
    return G
```

## Comparison Tolerances

| Data type | Tolerance |
|-----------|-----------|
| Dense matrices (reduction, H, C) | `atol=1e-6, rtol=1e-4` |
| Sparse graph values | `atol=1e-6, rtol=1e-4` |
| Sparse graph structure (nnz, shape) | Exact match |
| Sigma (singular values) | `atol=1e-8, rtol=1e-6` |
| Integer labels (assigned_archetype) | Exact match (after canonicalization) |
| Specificity p-values | `atol=1e-4, rtol=1e-3` (less tight due to tail bounds) |

## Deliverables

### 1. Updated comparison script

Extend `libactionet/test/compare_baselines.py` (from Plan 00) to handle
all the slots and canonicalization rules above. Make it accept two H5AD files
as arguments:

```bash
python test/compare_baselines.py \
    path/to/python_output.h5ad \
    path/to/r_output.h5ad
```

Output: per-slot pass/fail, maximum deviation, overall status.

### 2. Python end-to-end test script

`actionet-python/tests/test_parity.py`:

```python
def test_full_pipeline_parity():
    """Run full pipeline, compare against baseline."""
    adata = anndata.read_h5ad("path/to/parity_fixture.h5ad")
    # Run pipeline...
    # Load baseline...
    # Compare slot by slot...
```

### 3. R end-to-end test script

`actionet-r/tests/test_parity.R`:

```r
test_that("full pipeline matches baseline", {
    adata <- anndataR::read_h5ad("path/to/parity_fixture.h5ad")
    # Run pipeline...
    # Load baseline...
    # Compare slot by slot...
})
```

### 4. Storage mode parity tests

Python test that runs the pipeline in multiple storage modes and compares:

```python
def test_storage_mode_parity():
    """In-memory sparse vs backed sparse produce identical outputs."""
    adata_mem = anndata.read_h5ad("fixture.h5ad")
    adata_backed = anndata.read_h5ad("fixture.h5ad", backed="r")
    # Run pipeline on both...
    # Compare...
```

### 5. Documentation

Write a brief summary documenting:
- The new AnnData-native orientation contract
- Which slots are stored where and in what shape
- Any known cross-language differences and their causes
- How to run the parity tests

Place in `libactionet/context/` or `libactionet/docs/`.

## Failure Handling

If parity checks fail:

1. **Identify the divergence point**: Run each algorithm step individually
   and compare. Find the first step that diverges.

2. **Check canonicalization**: Many apparent failures are sign/ordering
   differences. Apply canonicalization before concluding there is a real
   divergence.

3. **Check floating-point accumulation order**: Sparse matrix operations
   may accumulate in different orders depending on storage format. This
   can cause O(machine-epsilon * nnz) differences. These are acceptable.

4. **Check algorithm defaults**: R and Python may use slightly different
   default parameters (e.g., max iterations, convergence tolerance).
   Align these before comparing.

5. **Document genuine differences**: If a difference persists after all
   checks, document it with the maximum deviation and the algorithmic
   cause. File a follow-up issue if the difference exceeds tolerance.

## Shape Verification Checklist

After all plans are applied, verify these shapes in both languages:

| Slot | Shape | Axis 0 | Axis 1 |
|------|-------|--------|--------|
| `adata.X` | (n_obs, n_var) | cells | genes |
| `obsm["action"]` | (n_obs, k) | cells | components |
| `obsm["action_B"]` | (n_obs, p) | cells | perturbation |
| `varm["action_U"]` | (n_var, k) | genes | components |
| `varm["action_A"]` | (n_var, p) | genes | perturbation |
| `obsm["H_stacked"]` | (n_obs, n_arch) | cells | archetypes |
| `obsm["H_merged"]` | (n_obs, n_arch) | cells | archetypes |
| `obsm["C_stacked"]` | (n_obs, n_arch) | cells | archetypes |
| `obsm["C_merged"]` | (n_obs, n_arch) | cells | archetypes |
| `obsp["actionet"]` | (n_obs, n_obs) | cells | cells |
| `varm["specificity_*"]` | (n_var, k) | genes | groups |

All `obsm` slots have `n_obs` rows. All `varm` slots have `n_var` rows.
All `obsp` slots are square `n_obs x n_obs`. No exceptions.

## Completion Criteria

- All cross-language comparisons pass within tolerance
- All intra-language regression checks pass (new vs baseline)
- All storage mode comparisons pass (in-memory vs backed)
- Shape verification passes in both languages
- Parity test scripts are committed and documented
- Any remaining differences are documented with causes
- The orientation unification is declared complete
