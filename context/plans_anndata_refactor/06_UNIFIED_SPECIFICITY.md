# Plan 06 — Unified Specificity

## Position in Sequence

```
   00 Parity Baseline            [DONE]
   01 R Network Cleanup          [optional; currently pending]
   02 C++ Core Contract Flip     [DONE]
   03 R Frontend Adaptation      [DONE]
   04 Python Frontend Adaptation [DONE]
>> 06 Unified Specificity <<
   05 Operator-Backed IRLB       [optional / parallel; currently pending]
   07 Final Cross-Language Parity Validation
```

**Dependencies**: Plans 02, 03, 04 (all three repos use new orientation).
**Blocks**: Plan 07 (final parity).

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

## Problem

Feature specificity is currently implemented three separate ways:

| Path | Location | Input mutation | Orientation | Notes |
|------|----------|---------------|-------------|-------|
| In-memory C++ | `src/annotation/specificity.cpp` (template overloads) | **Mutates** input S (normalizes in place) | cells x genes (post Plan 02) | Used by both R and Python for in-memory data |
| Backed sparse C++ | `src/annotation/specificity.cpp` (operator overloads) | Non-mutating | cells x genes (via operator) | Used by Python for backed sparse H5AD |
| Python dense-backed | `src/actionet/core.py` (streaming Python) | Non-mutating | Python-internal | Fallback when C++ backed path unavailable |

Problems:
1. The in-memory path mutates the input matrix — destructive and surprising
2. The backed path has separate accumulator logic that may diverge
3. The Python fallback duplicates normalization math in Python
4. Three implementations of the same algorithm increase parity risk

## Objective

Replace all three with a single non-mutating C++ implementation that handles:
- In-memory dense
- In-memory sparse
- Backed dense (via operator)
- Backed sparse (via operator)

With identical normalization, tail-bound, and significance computation across
all storage types.

## Repos

- `libactionet`: core implementation changes
- `actionet-python`: remove Python fallback, update wrappers
- `actionet-r`: verify wrapper compatibility (likely no changes beyond Plan 03)

## Current Specificity Math

The specificity pipeline has three phases:

### Phase 1: Observed statistics (`getProbsObs`)

Compute weighted observation matrix `Obs = f(S, H)` where:
- `S` is the expression matrix (cells x genes after Plan 02)
- `H` is group membership (cells x k after Plan 02)
- `Obs(g, k)` = weighted sum of gene g's expression across cells in group k

Current in-memory path (post Plan 02 orientation):

```
Obs = S.t() * H_norm     // (genes x cells)(cells x k) = genes x k
row_p = colsum(S_norm)    // gene-length: sum of normalized expression per gene
col_p = rowsum(S_norm)    // cell-length: sum of normalized expression per cell
```

The normalization step (which currently mutates S) computes:
- Normalize S so elements sum to 1
- Extract row/column marginals

### Phase 2: Expected statistics (`getProbsExp`)

Compute expected observation matrix under independence assumption:

```
Exp(g, k) = row_p(g) * col_p_per_group(k)
```

### Phase 3: Significance (`getSignificance`)

Compute z-scores and p-values using a Poisson or binomial tail bound,
comparing Obs vs Exp.

## Detailed Changes

### Stage A: Non-Mutating In-Memory Path

**File**: `src/annotation/specificity.cpp`

#### A1: Remove input mutation

The current code normalizes `S` in place before computing products. Replace
with a non-mutating approach:

Instead of:
```cpp
S /= accu(S);  // normalize S in place (DESTRUCTIVE)
arma::vec row_p = sum(S, 1);
arma::rowvec col_p = sum(S, 0);
arma::mat Obs = S * Ht;
```

Use (post Plan 02 orientation, S is cells x genes):
```cpp
double total = accu(S);
// Compute marginals without modifying S
arma::rowvec row_p_raw = sum(S, 0);       // gene sums (sum down cell axis)
arma::vec col_p_raw = sum(S, 1);          // cell sums (sum across gene axis)
arma::vec row_p = row_p_raw.t() / total;  // normalized gene marginals
arma::vec col_p = col_p_raw / total;      // normalized cell marginals

// Compute Obs using normalized H but UN-normalized S, then scale
arma::mat H_norm = H;  // cells x k
// Normalize H columns to sum to 1 (group membership normalization)
for (arma::uword j = 0; j < H_norm.n_cols; ++j) {
    double hs = accu(H_norm.col(j));
    if (hs > 0) H_norm.col(j) /= hs;
}
arma::mat Obs = S.t() * H_norm / total;   // (genes x cells)(cells x k) / total = genes x k
```

This produces the same `Obs`, `row_p`, `col_p` as the mutating version but
without modifying `S`.

#### A2: Sparse specialization

For sparse `S`, the same approach works:
- `accu(S)` sums all nonzeros
- `arma::sum(S, 0)` and `arma::sum(S, 1)` are efficient on CSC
- `S.t() * H_norm` is efficient (CSC transpose-multiply)

No special handling needed beyond using the sparse Armadillo operations.

### Stage B: Unify Backed and In-Memory Accumulator

**File**: `src/annotation/specificity.cpp`

The backed path currently has separate chunked accumulation logic that
iterates the H5AD file and accumulates `Obs` chunk by chunk. The math
is the same, but the code is separate.

#### B1: Extract shared accumulator

Create a shared function for the normalization/significance phases:

```cpp
// Shared: given Obs (genes x k), row_p, col_p, total → compute significance
arma::field<arma::mat> computeSignificance(
    const arma::mat& Obs,
    const arma::vec& row_p,      // gene marginals (genes,)
    const arma::vec& col_p,      // cell marginals (cells,) — only used for group sums
    const arma::mat& H_col_sums, // group sizes (k,) or group marginals
    double total,
    int thread_no
);
```

Both in-memory and backed paths compute `Obs`, `row_p`, `col_p`, then call
this shared function for Phase 2 (expected) and Phase 3 (significance).

#### B2: Backed accumulation produces same intermediates

The backed path already accumulates `Obs`, `row_p`, `col_p` via chunked
iteration. Ensure the final values are normalized identically to the in-memory
path before passing to the shared significance function.

### Stage C: Labels Overload

The labels-based overload (takes `arma::uvec& labels` instead of `arma::mat& H`)
internally constructs a binary H matrix. Update it to:

1. Use the new orientation (S is cells x genes)
2. Construct H as cells x k (one-hot encoding of labels)
3. Call the same non-mutating path

### Stage D: Remove Python Fallback

**File**: `actionet-python/src/actionet/core.py`

#### D1: Delete the streaming Python specificity

Current code has a `_compute_specificity_streamed` function (or similar) that
implements specificity in Python for the dense-backed case. Delete it.

#### D2: Route all paths through C++

After the backed C++ path handles all storage types, the Python code simplifies
to:

```python
def compute_feature_specificity(adata, ...):
    S = anndata_to_matrix(adata, layer=layer)  # cells x genes, no transpose
    labels = ...
    result = _core.compute_feature_specificity_sparse(S, labels, n_threads)
    # or _dense, depending on type
    return result
```

For backed data, the operator path handles it automatically if the Python
wrapper creates a backed operator.

#### D3: Archetype specificity simplification

```python
def compute_archetype_feature_specificity(adata, ...):
    S = anndata_to_matrix(adata, layer=layer)  # cells x genes
    H = np.ascontiguousarray(adata.obsm[...])  # cells x k
    result = _core.archetype_feature_specificity_sparse(S, H, n_threads)
    return result
```

No `.T` anywhere.

### Stage E: Update R Specificity Wrappers

**File**: `actionet-r/R/r_specificity.R`

Likely no changes needed beyond Plan 03 (which already removed the transposes).
Verify that:
- Expression matrix arrives as cells x genes
- H arrives as cells x k
- Output (genes x k) is stored directly in varm

### Stage F: Pybind11 Wrapper Updates

**File**: `actionet-python/src/actionet/wp_annotation.cpp`

Verify that the C++ specificity functions are called with the correct
orientation. The pybind wrapper should be a simple pass-through after Plan 04.

If backed operator specificity is exposed through a separate pybind function,
ensure it accepts a `MatrixOperator` (or a Python `LinearOperator` wrapper)
and routes to the unified C++ implementation.

### Stage G: Update `computeFeatureStats` / `computeFeatureStatsVision` orientation

**Files**: `libactionet/src/annotation/marker_stats.cpp`,
`libactionet/include/annotation/marker_stats.hpp`,
`actionet-python/src/actionet/annotation.py`

**Context (deferred from Plan 04):**

`computeFeatureStats` and `computeFeatureStatsVision` were **not** updated in
Plan 02. Their headers still document `S` as `features × cells`. As a result,
`actionet-python/src/actionet/annotation.py` (`annotate_cells`) was left with
the following transpose shims (deliberately preserved in Plan 04):

```python
# backed path (annotation.py ~line 390):
S = S_cells.T.tocsr()   # cells x features → features x cells  ← keep until here

# in-memory path (annotation.py ~line 395):
S = S.T                  # cells x genes → genes x cells  ← keep until here
```

These shims must be removed as part of this plan, in conjunction with
flipping the C++ functions to accept `cells × genes`:

#### G1: Flip `computeFeatureStats` and `computeFeatureStatsVision`

Update both functions to accept `S` as `cells × genes` (obs × var), consistent
with the Plan 02 contract. Internally, wherever `S.row(i)` (gene-indexed rows)
is used for cell-column access, switch to `S.col(i)` (gene-indexed columns),
or use `S.t()` only on the narrow access point rather than materializing a full
transpose.

#### G2: Remove transpose shims in `annotation.py`

After G1, remove the two residual transpose operations in `annotate_cells`:

```python
# backed path — after G1:
if not issparse(S_cells):
    S_cells = csr_matrix(np.asarray(S_cells))
S = S_cells  # cells x features, direct (was S_cells.T.tocsr())

# in-memory path — after G1:
S = source.matrix
if not issparse(S):
    S = csr_matrix(S)
# S is cells x genes, no .T needed
```

#### G3: Update marker_stats.hpp documentation

Update the `@param S` documentation in `marker_stats.hpp` to reflect the new
`cells × genes` (obs × var) orientation.

## Validation

### Numerical parity — in-memory, non-mutating

1. Load the parity fixture
2. Run specificity with the OLD (mutating) code on `dev-backed`, capture output
3. Run specificity with the NEW (non-mutating) code, capture output
4. Compare: must be identical (the math is the same, only the mutation is removed)

### Mutation check

Verify that S is not modified after calling specificity:

```python
import numpy as np
S_before = adata.X.copy()
compute_feature_specificity(adata, ...)
S_after = adata.X
assert np.array_equal(S_before.toarray(), S_after.toarray()), "S was mutated!"
```

### Cross-storage parity

On the same fixture, compare:
1. In-memory sparse specificity
2. In-memory dense specificity (convert to dense first)
3. Backed sparse specificity (open H5AD as backed)
4. Backed dense specificity (if applicable)

All four must produce identical results.

### Cross-language parity

Compare R and Python specificity outputs on the parity fixture. Use the
comparison script from Plan 00.

### Python fallback removal

Verify that no Python-side specificity computation code remains. The only
specificity code in Python should be the wrapper that calls C++.

## Files Modified (Summary)

| Repo | File | Changes |
|------|------|---------|
| `libactionet` | `src/annotation/specificity.cpp` | Non-mutating, shared accumulator |
| `libactionet` | `include/annotation/specificity.hpp` | Updated docs (const correctness) |
| `libactionet` | `src/annotation/marker_stats.cpp` | Flip S to cells × genes (Stage G) |
| `libactionet` | `include/annotation/marker_stats.hpp` | Update @param S docs (Stage G) |
| `actionet-python` | `src/actionet/core.py` | Delete Python fallback |
| `actionet-python` | `src/actionet/wp_annotation.cpp` | Verify pass-through |
| `actionet-python` | `src/actionet/annotation.py` | Remove `.T`/`.T.tocsr()` shims in `annotate_cells` (Stage G) |
| `actionet-r` | `R/r_specificity.R` | Verify compatibility |

## Completion Criteria

- Single non-mutating C++ specificity implementation
- Works for: in-memory dense, in-memory sparse, backed dense, backed sparse
- Python streaming fallback deleted
- All four storage paths produce identical results
- Cross-language parity confirmed
- Input matrix is never mutated
- `computeFeatureStats` / `computeFeatureStatsVision` accept `cells × genes`
- Zero `.T` operations on expression matrix in `annotation.py`
