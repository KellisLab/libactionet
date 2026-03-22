# Plan 00 — Parity Baseline Infrastructure

## Position in Sequence

```
>> 00 Parity Baseline <<
   01 R Network Cleanup
   02 C++ Core Contract Flip
   03 R Frontend Adaptation
   04 Python Frontend Adaptation + Boundary Optimization
   05 Operator-Backed IRLB
   06 Unified Specificity
   07 Final Cross-Language Parity Validation
```

**Dependencies**: None. This plan executes first.
**Blocks**: Plans 01–06 (each plan validates against the baseline produced here).

## Contract Notice

This plan is part of a coordinated cross-repo AnnData orientation unification.
Public API breakage across `libactionet`, `actionet-r`, and `actionet-python` is
explicitly permitted and expected until the full sequence completes.

See `context/ANNDATA_UNIFICATION_HANDOFF.md` for the complete rationale.

## Objective

Capture golden reference outputs from the current `dev-backed` branch of all
three repos **before any orientation changes begin**. Every subsequent plan
validates its changes against this baseline to detect regressions and confirm
that new outputs are either identical or differ only in documented, expected
ways (e.g., transposed shape stored in AnnData slots).

## Repos and Branches

| Repo | Local path (assumed) | Branch |
|------|---------------------|--------|
| `libactionet` | `../libactionet` | `dev-backed` |
| `actionet-r` | `../actionet-r` | `dev-backed` |
| `actionet-python` | `../actionet-python` | `dev-backed` |

All three repos must be checked out to `dev-backed` when generating baseline
outputs. Confirm branch with `git rev-parse --abbrev-ref HEAD` in each repo
before proceeding.

## Fixture Dataset

Create or identify a small `.h5ad` fixture suitable for deterministic testing:

- ~500 cells, ~2000 genes
- Sparse expression matrix (CSC on disk)
- At least 3 distinct cell populations (for specificity and archetype tests)
- Known batch labels in `obs["batch"]` (at least 2 batches, for batch correction)
- Fixed random seed: use `42` everywhere

If a suitable fixture does not already exist, generate one using `scanpy` or
equivalent:

```python
import scanpy as sc
import numpy as np

np.random.seed(42)
adata = sc.datasets.pbmc3k_processed()  # or similar small dataset
# Subsample to ~500 cells
sc.pp.subsample(adata, n_obs=500, random_state=42)
# Ensure sparse CSC
import scipy.sparse as sp
if not sp.issparse(adata.X):
    adata.X = sp.csc_matrix(adata.X)
# Add batch labels if missing
adata.obs["batch"] = np.random.choice(["A", "B"], size=adata.n_obs)
adata.write("test_fixture.h5ad")
```

Place the fixture at a shared location readable by both R and Python test
scripts, e.g., `libactionet/test/fixtures/parity_fixture.h5ad`.

## Baseline Capture — Python

Write a script `actionet-python/tests/generate_baseline.py` that:

1. Loads the fixture
2. Runs the full canonical pipeline with fixed seed=42:
   - `reduce_kernel(adata, seed=42)`
   - `run_action(adata, seed=42)`
   - `build_network(adata)`
   - `compute_feature_specificity(adata, ...)`  (cluster labels)
   - `compute_archetype_feature_specificity(adata, ...)`
   - `correct_batch_effect(adata, ...)`
   - `layout_network(adata, seed=42)`
3. Extracts and saves all parity-critical slots:

| Slot | Expected shape | Key |
|------|---------------|-----|
| `obsm["action"]` | (cells, k) | reduction |
| `varm["action_U"]` | (genes, k) | reduction |
| `varm["action_A"]` | (genes, p) | reduction |
| `obsm["action_B"]` | (cells, p) | reduction |
| `uns["action_params"]["sigma"]` | (k,) | reduction |
| `obsm["H_stacked"]` | (cells, archetypes) | ACTION |
| `obsm["H_merged"]` | (cells, archetypes) | ACTION |
| `obsm["C_stacked"]` | (k, archetypes) | ACTION |
| `obsm["C_merged"]` | (k, archetypes) | ACTION |
| `obs["assigned_archetype"]` | (cells,) | ACTION |
| `obsp["actionet"]` | (cells, cells) sparse | network |
| specificity outputs | (genes, k) | specificity |
| corrected reduction | (cells, k) | batch correction |

4. Saves to `test/fixtures/baseline_python.h5ad` (full AnnData) and
   `test/fixtures/baseline_python.npz` (individual arrays for easy comparison).

## Baseline Capture — R

Write a script `actionet-r/tests/generate_baseline.R` that:

1. Loads the same fixture using `anndataR::read_h5ad()`
2. Runs the identical pipeline with `seed = 42L`:
   - `reduceKernel(adata, seed = 42L)`
   - `runACTION(adata, seed = 42L)`
   - `buildNetwork(adata)`
   - `computeFeatureSpecificity(adata, ...)`
   - `archetypeFeatureSpecificity(adata, ...)`
   - `correctBatchEffect(adata, ...)`
   - `layoutNetwork(adata, seed = 42L)`
3. Extracts the same slots from colMaps/rowMaps/colNets
4. Saves to `test/fixtures/baseline_r.h5ad` and `test/fixtures/baseline_r.rds`

## Comparison Script

Write `libactionet/test/compare_baselines.py` that:

1. Loads both baseline outputs
2. For each slot, applies canonicalization:
   - **SVD sign**: for each column, flip sign so that the element with largest
     absolute value is positive
   - **Archetype ordering**: sort archetypes by descending column norm of H
   - **Sparse graphs**: convert to CSR, sort indices within each row
3. Compares with `np.allclose(a, b, atol=1e-6, rtol=1e-4)` for dense arrays
4. For sparse matrices: compare sorted CSR structure and values
5. Reports per-slot pass/fail and maximum deviation

This script becomes the reusable parity checker for all subsequent plans.

## Intra-Language Regression Check

Each subsequent plan must also verify that its changes do not regress
within-language outputs. The pattern is:

1. Before making changes, run the pipeline on the fixture and save outputs
   (or rely on the baseline from this plan).
2. After making changes, run the pipeline again and compare against the saved
   baseline.
3. Expected differences are documented explicitly (e.g., "shape of
   `obsm['action']` changes from `(cells, k)` stored via `.T` to `(cells, k)`
   stored directly — values identical").

## Deliverables

| Artifact | Location |
|----------|----------|
| Fixture `.h5ad` | `libactionet/test/fixtures/parity_fixture.h5ad` |
| Python baseline script | `actionet-python/tests/generate_baseline.py` |
| Python baseline outputs | `actionet-python/tests/fixtures/baseline_python.{h5ad,npz}` |
| R baseline script | `actionet-r/tests/generate_baseline.R` |
| R baseline outputs | `actionet-r/tests/fixtures/baseline_r.{h5ad,rds}` |
| Comparison script | `libactionet/test/compare_baselines.py` |

## Completion Criteria

- Both baseline scripts run without error on `dev-backed`
- Comparison script runs and reports per-slot parity status
- Any pre-existing cross-language differences are documented (these represent
  the current state of the system, not regressions)
- Fixture and baselines are committed to their respective repos on a feature
  branch (e.g., `feature/orientation-unification`)
