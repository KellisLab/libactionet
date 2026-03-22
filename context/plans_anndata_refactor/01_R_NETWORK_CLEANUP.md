# Plan 01 — R Network Cleanup

## Position in Sequence

```
   00 Parity Baseline            [DONE]
>> 01 R Network Cleanup          [DONE]
   02 C++ Core Contract Flip
   03 R Frontend Adaptation
   04 Python Frontend Adaptation + Boundary Optimization
   05 Operator-Backed IRLB
   06 Unified Specificity
   07 Final Cross-Language Parity Validation
```

**Status**: **Complete** (2026-03-21)

**Validation result**: `C_buildNetwork` with `cells x k` H matrix produces
bit-identical output (max abs diff = 0) to the Plan 00 baseline.
R package builds cleanly with `devtools::load_all()`.

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

`buildNetwork()` in R performs a pointless double-transpose:

1. AnnData stores `H_stacked` as `cells x archetypes` in `obsm`
2. R reads it via `.ace_or_map(..., transpose_map = TRUE)` → flips to
   `archetypes x cells`
3. The C++ wrapper `C_buildNetwork` receives `archetypes x cells` (column-major),
   then iterates columns to pack a row-major `cells x archetypes` float32 buffer
   for HNSW

Step 2 materializes a full matrix transpose that step 3 immediately undoes.
Python does not have this problem — it passes `cells x k` directly to
`buildNetworkCore`.

## Scope

| Repo | Files to modify |
|------|----------------|
| `actionet-r` | `R/network_tools.R`, `src/wr_network.cpp` |
| `libactionet` | `wrappers_r/wr_network.cpp` (reference copy), optionally `src/network/build_network.cpp` |
| `actionet-python` | None — already correct |

## Detailed Changes

### 1. `actionet-r/R/network_tools.R` — `buildNetwork()`

Current code (approximately lines 20–27):

```r
H <- .ace_or_map(
  obj = adata,
  map_slot = map_slot,
  matrix_type = "dense",
  force_type = TRUE,
  transpose_map = TRUE,   # <--- flips cells x k → k x cells
  return_elem = TRUE
)
```

Change to:

```r
H <- .ace_or_map(
  obj = adata,
  map_slot = map_slot,
  matrix_type = "dense",
  force_type = TRUE,
  transpose_map = FALSE,  # <--- pass cells x k directly
  return_elem = TRUE
)
```

### 2. `actionet-r/src/wr_network.cpp` — `C_buildNetwork()`

Current code:

```cpp
// Assumes H is k x cells (column-major). Each column is one cell.
const std::size_t dim      = static_cast<std::size_t>(H.nrow());  // k
const std::size_t n_points = static_cast<std::size_t>(H.ncol());  // cells
// ... iterates H.colptr(col) to pack row-major float buffer
```

Change to accept `cells x k` (column-major from R):

```cpp
// H is now cells x k (column-major from R). Each row is one cell.
const std::size_t n_points = static_cast<std::size_t>(H.nrow());  // cells
const std::size_t dim      = static_cast<std::size_t>(H.ncol());  // k

std::vector<float> data(n_points * dim);
for (std::size_t i = 0; i < n_points; ++i) {
    for (std::size_t j = 0; j < dim; ++j) {
        data[i * dim + j] = static_cast<float>(H(i, j));
    }
}
```

This iterates rows of the column-major R matrix. Each row becomes one
contiguous row in the float buffer — matching what `buildNetworkCore` expects.

Note: accessing `H(i, j)` on a column-major matrix is already efficient for
the inner loop over `j` (stride-1 access down a column). If profiling shows
concern, the loop can be reordered to iterate columns in the outer loop and
scatter into the row-major buffer, but for the dimensions involved (k is
typically 10–100), this is irrelevant.

### 3. `libactionet/wrappers_r/wr_network.cpp` — Reference copy

Update the reference copy in `libactionet` to match the change in step 2.
This file is documented as "may not be synced" in the playbook, but keeping it
aligned avoids confusion.

### 4. Optional: `libactionet/src/network/build_network.cpp` — Legacy `buildNetwork(arma::mat& H)`

The legacy Armadillo wrapper `buildNetwork(arma::mat& H)` assumes `H` is
`k x cells`. Two options:

**Option A (preferred)**: Update it to accept `cells x k`:

```cpp
CSRGraph buildNetwork(const arma::mat& H, NetworkParams params) {
    const std::size_t n_points = H.n_rows;  // cells (was n_cols)
    const std::size_t dim      = H.n_cols;  // k     (was n_rows)
    // Pack row-major buffer: iterate rows
    std::vector<float> data(n_points * dim);
    for (std::size_t i = 0; i < n_points; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            data[i * dim + j] = static_cast<float>(H(i, j));
        }
    }
    return buildNetworkCore(data.data(), n_points, dim, params);
}
```

**Option B**: Deprecate the Armadillo wrapper entirely. Python already calls
`buildNetworkCore` directly. The Armadillo wrapper exists only for the R path.
After the R wrapper is updated in step 2, the Armadillo wrapper is only used
by the `wrappers_r` code. If we update `wr_network.cpp` to call
`buildNetworkCore` directly (bypassing the Armadillo wrapper), the legacy
wrapper can be removed.

Choose Option A unless the legacy wrapper has other callers.

## Validation

### Build

```bash
# R package
cd actionet-r
R CMD INSTALL .
# or devtools::install() from R
```

### Functional test

Run `buildNetwork()` on the parity fixture from Plan 00:

```r
library(actionet)
adata <- anndataR::read_h5ad("test/fixtures/parity_fixture.h5ad")
# Run reduction + ACTION first (produces H_stacked in obsm)
reduceKernel(adata, seed = 42L)
runACTION(adata, seed = 42L)
# Now test the network build
buildNetwork(adata)
G_new <- colNets(adata)[["actionet"]]
```

### Parity check

Compare `G_new` against the R baseline from Plan 00:

```r
baseline <- readRDS("tests/fixtures/baseline_r.rds")
G_baseline <- baseline$obsp_actionet

# Convert both to dgCMatrix, sort, compare
# Symmetric graph → compare upper triangle
stopifnot(all.equal(
  as(G_new, "CsparseMatrix"),
  as(G_baseline, "CsparseMatrix"),
  tolerance = 1e-6
))
```

The graph must be **identical** (not just close) because the only change is
removing a no-op double transpose — the actual float values reaching HNSW
should be bit-identical.

### Python parity

Python is unchanged, but verify that R and Python still produce the same
network graph (within tolerance) by running the comparison script from Plan 00.

## Deliverables

| Artifact | Description |
|----------|-------------|
| Modified `R/network_tools.R` | `transpose_map = FALSE` |
| Modified `src/wr_network.cpp` | Accepts cells x k |
| Updated `wrappers_r/wr_network.cpp` | Reference copy synced |
| Parity test script or output | Confirms graph identity |

## Completion Criteria

- `buildNetwork()` produces identical output to baseline
- No `Matrix::t()` or `transpose_map = TRUE` in the network path
- R package builds and installs cleanly
- Python network output unchanged
