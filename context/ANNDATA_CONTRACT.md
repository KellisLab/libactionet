# AnnData Orientation Contract

**Status**: Active — orientation unification complete (Plans 00–07 passed)  
**Date**: 2026-03-22  
**Scope**: libactionet C++ core, actionet-python, actionet-r

---

## Summary

The ACTIONet ecosystem stores all matrices in **cells × genes** (obs × var) orientation,
matching the AnnData native convention. There are no transpose shims at any language
boundary. All C++ functions receive and return matrices in the AnnData-native orientation.

---

## Contract: Matrix Storage Layout

### AnnData slot conventions

| Slot | Shape | Axis 0 | Axis 1 | Description |
|------|-------|--------|--------|-------------|
| `X` | (n_obs, n_var) | cells | genes | Expression matrix |
| `obsm["action"]` | (n_obs, k) | cells | latent components | Kernel reduction (S_r) |
| `obsm["action_B"]` | (n_obs, p) | cells | perturbation dims | Batch-group footprint |
| `varm["action_U"]` | (n_var, k) | genes | latent components | Left singular vectors |
| `varm["action_A"]` | (n_var, p) | genes | perturbation dims | Batch-group gene loading |
| `uns["action_params"]["sigma"]` | (k,) | — | — | Singular values |
| `obsm["H_stacked"]` | (n_obs, n_arch_raw) | cells | archetypes | Raw per-k archetype weights |
| `obsm["H_merged"]` | (n_obs, n_arch) | cells | archetypes | Merged archetype weights |
| `obsm["C_stacked"]` | (n_obs, n_arch_raw) | cells | archetypes | Raw per-k archetype membership |
| `obsm["C_merged"]` | (n_obs, n_arch) | cells | archetypes | Merged archetype membership |
| `obs["assigned_archetype"]` | (n_obs,) | cells | — | Dominant archetype index |
| `obsp["actionet"]` | (n_obs, n_obs) | cells | cells | ACTIONet sparse graph |
| `varm["specificity_upper"]` | (n_var, k) | genes | clusters | Upper specificity scores |
| `varm["specificity_lower"]` | (n_var, k) | genes | clusters | Lower specificity scores |
| `varm["specificity_profile"]` | (n_var, k) | genes | clusters | Average expression profile |
| `varm["archetype_feat_profile"]` | (n_var, n_arch) | genes | archetypes | Archetype feature profile |
| `varm["archetype_feat_specificity_upper"]` | (n_var, n_arch) | genes | archetypes | Archetype upper specificity |
| `varm["archetype_feat_specificity_lower"]` | (n_var, n_arch) | genes | archetypes | Archetype lower specificity |
| `obsm["action_corrected"]` | (n_obs, k) | cells | components | Batch-corrected reduction |
| `varm["action_corrected_U"]` | (n_var, k) | genes | components | Batch-corrected left SV |
| `varm["action_corrected_A"]` | (n_var, p) | genes | perturbation | Batch-corrected gene loading |

**Rules**:
- All `obsm` slots have `n_obs` rows (no exceptions).
- All `varm` slots have `n_var` rows (no exceptions).
- All `obsp` slots are square `(n_obs, n_obs)`.
- No transposition is applied at any language boundary.

### R-specific slot names

The R frontend uses slightly different key names in some slots:

| Canonical (Python/C++) | R slot name (in h5ad) | R accessor |
|------------------------|----------------------|------------|
| `obsm["action_corrected"]` | `obsm["action_orth"]` | `colMaps(adata)[["action_orth"]]` |
| `varm["action_corrected_U"]` | `varm["action_U_orth"]` | `rowMaps(adata)[["action_U_orth"]]` |
| `varm["action_corrected_A"]` | `varm["action_A_orth"]` | `rowMaps(adata)[["action_A_orth"]]` |
| `varm["specificity_upper"]` | `varm["cluster_upper"]` | `rowMaps(adata)[["cluster_upper"]]` |
| `varm["specificity_lower"]` | `varm["cluster_lower"]` | `rowMaps(adata)[["cluster_lower"]]` |

Python uses `adata.obsm`, `adata.varm`, `adata.obsp` directly.  
R uses `colMaps(adata)`, `rowMaps(adata)`, `colNets(adata)` wrapper functions.

---

## C++ Core Contract

All public C++ functions in `libactionet` accept and return matrices in
**cells × genes (obs × var)** orientation:

- `arma::sp_mat S` — sparse expression: rows = cells, cols = genes
- `arma::mat S` — dense expression: rows = cells, cols = genes
- `arma::mat S_r` — kernel reduction (S_r): rows = cells (obs), cols = k
- `arma::mat U` — left singular vectors: rows = genes (var), cols = k
- `arma::mat H` — archetype footprint: rows = cells (obs), cols = archetypes

There are no internal transposes at the pybind11 boundary (Python) or the
Rcpp boundary (R). Sparse matrices cross the boundary as direct CSC → arma
conversion (without creating a CSR transpose intermediate).

---

## Known Cross-Language Differences

The following differences between R and Python are **expected and documented**,
not regressions:

### 1. `varm/specificity_profile` (Python-only)

The Python `compute_feature_specificity` stores the average feature profile
in `varm["specificity_profile"]`. The R `computeFeatureSpecificity` function
returns this value but does not store it in the AnnData object by default.

This is a known API asymmetry, not a regression.

### 2. 0-vs-1 indexed archetype assignment

Python `obs["assigned_archetype"]` is 0-indexed.  
R `adata$obs[["assigned_archetype"]]` is 1-indexed.  
Difference of 1 for all cells is expected and correct.

---

## Resolved Parity Issue

On 2026-03-22 we traced the previously documented archetype-count divergence to
the Python frontend, not the C++ pruning core. `actionet-python` normalized the
reduced matrix `obsm["action"]` along columns before calling `run_action()`,
while the R frontend correctly normalized rows/cells. After changing Python to
row-normalize `S_r`, the full parity fixture matches the R baseline across
ACTION outputs, network, specificity, and batch-correction slots.

---

## How to Run Parity Tests

### Python end-to-end parity test

```bash
cd actionet-python
.venv/bin/python3 tests/test_parity.py
```

Runs the full pipeline, checks shape verification, intra-language regression
(vs Plan 00 baseline), and cross-language parity (vs R baseline).

### Python storage mode parity test

```bash
cd actionet-python
.venv/bin/python3 tests/test_storage_parity.py
```

Confirms backed operator paths agree with in-memory paths.

### R end-to-end parity test

```bash
cd actionet-r
Rscript tests/test_parity.R
```

Runs the full pipeline, checks shape verification, intra-language regression,
and cross-language parity (vs Python baseline).

### Cross-language comparison tool

```bash
# Compare two h5ad output files directly:
python libactionet/test/compare_baselines.py \
    path/to/python_output.h5ad \
    path/to/r_output.h5ad

# Or compare pre-built .npz baselines:
python libactionet/test/compare_baselines.py \
    --python-npz actionet-python/tests/fixtures/baseline_python.npz
```

---

## Validation Results (Plan 07)

All parity checks were run with the parity fixture (500 cells × 2000 genes,
seed=42) on 2026-03-22.

### Python `test_parity.py`
- **44 PASS, 0 FAIL**
- **0 WARN**

### R `test_parity.R`
- **49 PASS, 0 FAIL**
- **0 WARN**

### Python `test_storage_parity.py`
- **17 PASS, 0 FAIL**

### Reduction slots (cross-language, exact match after sign canonicalization)

| Slot | Python vs R max deviation | Status |
|------|--------------------------|--------|
| `obsm["action"]` | 0.000e+00 | PASS |
| `obsm["action_B"]` | 0.000e+00 | PASS |
| `varm["action_U"]` | 0.000e+00 | PASS |
| `varm["action_A"]` | 0.000e+00 | PASS |
| `uns/sigma` | 0.000e+00 | PASS |
| `obsm["action_corrected"]` | 0.000e+00 | PASS |
| `varm["action_corrected_U"]` | 0.000e+00 | PASS |
| `varm["action_corrected_A"]` | 0.000e+00 | PASS |

---

## Orientation Unification History

| Plan | Description | Status |
|------|-------------|--------|
| 00 | Parity baseline infrastructure | Complete |
| 01 | R network cleanup | Complete |
| 02 | C++ core contract flip (genes×cells → cells×genes) | Complete |
| 02A | Orthogonalization contract repair | Complete |
| 03 | R frontend adaptation (remove all transpose shims) | Complete |
| 03A | R frontend post-flip regression repair | Complete |
| 04 | Python frontend adaptation + boundary optimization | Complete |
| 05 | Operator-backed IRLB | Complete |
| 06 | Unified specificity (cells×genes throughout) | Complete |
| 07 | Final cross-language parity validation | **Complete** |
