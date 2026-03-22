# AnnData Orientation Unification Handoff

## Purpose

This note captures the current investigation into orientation mismatches across:

- `libactionet`
- `actionet-python`
- `actionet-r`

The immediate goal is to make all three branches coherent around the AnnData
container contract, remove avoidable transpose/materialization overhead, reduce
front-end shim code, and preserve output parity across R and Python.

This document is intended as a handoff for implementation and follow-up
analysis.

## Decision Direction

Use **AnnData-native orientation as the only public cross-language contract**:

- expression matrices: `cells x genes`
- reduced representations: `cells x k`
- archetype maps in `obsm`: `cells x k`
- feature maps in `varm`: `genes x k`
- graphs in `obsp`: `cells x cells`

Do **not** preserve the current `libactionet` public `genes x cells` contract
with more front-end compatibility shims.

The user explicitly approved breaking the operator/public API block if that
drastically simplifies the system. That should be taken literally here: the
cleanest solution is to flip the C++ contract instead of continuing to emulate
the old one in R and Python.

## Main Findings

### 1. `libactionet` still assumes logical `features x cells` almost everywhere

Important clarification: the real mismatch is mostly **logical orientation**
(`cells x genes` vs `genes x cells`), not Armadillo’s physical column-major
storage.

Relevant C++ entry points and docs still encode `features x cells`:

- `include/action/reduce_kernel.hpp`
- `src/action/reduce_kernel.cpp`
- `include/action/action_main.hpp`
- `src/action/action_main.cpp`
- `include/annotation/specificity.hpp`
- `src/annotation/specificity.cpp`
- `include/decomposition/orthogonalization.hpp`
- `src/decomposition/orthogonalization.cpp`

This is the root reason both front ends currently transpose expression data
before calling into C++.

### 2. `buildNetworkCore` is already aligned with AnnData orientation

The network core is the clearest precedent for the target architecture:

- `include/network/build_network_core.hpp`
- `src/network/build_network.cpp`

`buildNetworkCore` already accepts row-major `float32` input with one point per
row, i.e. `cells x k`. Python already uses this path directly. R currently
transposes into the old shape and then converts back into the row-oriented core.

This is a low-risk early cleanup target.

### 3. `MatrixOperator` already exists and partially decouples storage from orientation

Relevant files:

- `include/decomposition/matrix_operator.hpp`
- `include/io/backed_h5ad/backed_sparse_matrix_operator.hpp`
- `include/io/backed_h5ad/backed_dense_matrix_operator.hpp`
- `src/io/backed_h5ad/create_backed_operator.cpp`

Backed H5AD operators already read native AnnData `obs x var` storage and expose
it to the current core as a logical transpose. This proves the codebase already
accepts the idea of:

- native storage remaining `cells x genes`
- core math consuming an abstracted view

That existing abstraction should be extended rather than duplicated in wrapper
code.

### 4. Python has centralized transpose shims, but still duplicates them widely

Key files:

- `actionet-python/src/actionet/core.py`
- `actionet-python/src/actionet/anndata_utils.py`
- `actionet-python/src/actionet/batch_correction.py`
- `actionet-python/src/actionet/annotation.py`
- `actionet-python/src/actionet/wp_utils.cpp`

Current issues:

- `anndata_to_matrix(..., transpose=True)` is used broadly for expression-bound
  algorithms.
- reduced outputs are often transposed again before storing into `obsm`.
- pybind conversion copies row-major NumPy into new Armadillo matrices.
- dense-backed specificity still falls back to a Python streaming path.

The Python network path is already correct and should be treated as the target
shape for the rest of the stack.

### 5. R now uses AnnData natively but preserves legacy semantics through adapters

Key files:

- `actionet-r/R/utils_anndata_adapter.R`
- `actionet-r/R/utils_validation.R`
- `actionet-r/R/r_action.R`
- `actionet-r/R/main.R`
- `actionet-r/R/network_tools.R`
- `actionet-r/R/r_specificity.R`
- `actionet-r/src/wr_network.cpp`

Current issues:

- generic helpers transpose AnnData expression back to `features x cells`
- map validators still expose legacy shapes
- `buildNetwork()` loads `H_stacked` as `k x cells` and the C++ wrapper converts
  it back into `cells x k`

This adapter layer is now the main source of extra complexity on the R side.

### 6. The current operator path is artificially blocked in R

Relevant files:

- `src/action/reduce_kernel.cpp`
- `src/decomposition/svd_main.cpp`
- `cmake/ConfigurePRIMME.cmake`
- `wrappers_r/wr_decomposition.cpp`

Current state:

- `MatrixOperator` SVD paths exist for Halko, Feng, and PRIMME.
- `runSVD_Operator` does **not** support IRLB today.
- `reduceKernel_Operator` is hard-blocked in R builds even though operator SVD
  support exists in principle.
- PRIMME is excluded from R builds, but that does not justify blocking all
  operator-backed execution.

Conclusion:

- If we want no-transpose/native-orientation execution in both languages,
  `libactionet` needs either:
  - operator-backed IRLB, or
  - a deliberate algorithm change for native-orientation paths

### 7. Specificity is split into too many implementations

Current state:

- in-memory C++ specificity mutates the input matrix
- backed sparse C++ specificity is non-mutating and streaming
- Python dense-backed specificity uses a separate Python fallback

This is unnecessary divergence for a parity-sensitive algorithm. A shared
non-mutating C++ implementation should replace the current split.

## Recommended Contract

Treat the following as the canonical API contract across repos:

- expression input to core/front ends: `cells x genes`
- reduced kernel returned to front ends: `cells x k`
- ACTION decomposition input: `cells x k`
- `H_stacked`, `H_merged`: `cells x archetypes`
- `C_stacked`, `C_merged`: keep current simplex/regression shapes unless and
  until there is a clear reason to change them
- feature specificity outputs: `genes x k`

## Recommended Technical Direction

### Phase 1: Remove obvious reduced-space shims first

These changes are low risk and should land early:

- make R `buildNetwork()` read `obsm` as `cells x k`
- remove the `transpose_map = TRUE` network path in R
- make the R wrapper accept native `cells x k` directly
- keep Python network unchanged

Also normalize `runACTION` wrapper semantics so front ends always pass
`cells x k`, even if `libactionet` internally transposes the reduced matrix in
v1.

### Phase 2: Flip `libactionet` public orientation to AnnData-native

Change public headers and wrapper expectations so the C++ boundary is native:

- `reduceKernel` consumes `cells x genes`
- `runACTION` consumes `cells x k`
- specificity consumes `cells x genes`
- batch and basal orthogonalization consume `cells x genes`
- network consumes `cells x k`

This is the actual simplification step. Without it, front-end transpose shims
will keep reappearing.

### Phase 3: Move orientation adaptation inside `libactionet`

For v1, it is acceptable to keep some internal orientation adaptation if it is:

- inside `libactionet` only
- not materializing full expression transposes
- not exposed to R/Python

In particular:

- expression-bound algorithms should use operator/view-based access
- reduced-space `runACTION` may internally transpose the much smaller
  `cells x k` matrix if rewriting AA/SPA immediately is not worth the cost

This keeps the major memory/performance win while avoiding a large AA rewrite on
the critical path.

### Phase 4: Add operator-backed IRLB

If preserving current algorithm defaults matters, implement operator-backed IRLB
on `MatrixOperator` instead of forcing all native-orientation paths onto
Halko/Feng/PRIMME.

Why this matters:

- current default behavior in R and Python leans heavily on IRLB
- `runSVD_Operator` currently throws for IRLB
- R currently blocks operator `reduceKernel` entirely because of this gap

Recommended outcome:

- `runSVD_Operator` supports IRLB, Halko, Feng
- PRIMME remains optional and Python-first
- R build no longer blocks operator-native reduction solely because PRIMME is
  missing

### Phase 5: Unify specificity in one shared C++ implementation

Replace the current split with one non-mutating implementation that works for:

- in-memory dense
- in-memory sparse
- backed dense
- backed sparse

Target properties:

- native `cells x genes` input
- no mutation of source matrix
- shared accumulator math
- identical normalization/tail-bound logic across all storage types

This should eliminate the dense-backed Python fallback and reduce parity risk.

## Concrete Opportunities

### Opportunity A: R network cleanup

Current path:

1. AnnData stores `H_stacked` as `cells x k`
2. R helper transposes to `k x cells`
3. R wrapper converts it back into row-oriented `cells x k`

Fix this first.

### Opportunity B: eliminate front-end expression transposes

Current path in both languages:

- materialize `genes x cells`
- convert/copy into Armadillo
- often transpose outputs again for storage

Target:

- pass native `cells x genes`
- let C++ consume views/operators
- avoid full expression transpose materialization entirely

### Opportunity C: consolidate Python orientation helpers

After the C++ contract flip:

- delete `transpose=True` dependence in high-level paths
- remove duplicated `.T` logic in `core.py`, `batch_correction.py`, and
  specificity code
- keep one small boundary layer for AnnData persistence only

### Opportunity D: simplify R AnnData adapters

After the C++ contract flip:

- `.get_layer_matrix()` should return native AnnData orientation by default
- generic validators should stop emulating legacy `features x cells` semantics
- `ACTIONetExperiment` conversion should be compatibility-only, not the core
  execution model

## Output Parity Strategy

Current testing is fragmented. There are intra-language parity checks, but not a
real R-vs-Python golden suite over the same AnnData fixture.

Add one shared small `.h5ad` fixture and compare the same canonical outputs in
both languages:

- reduction:
  - `obsm["action"]`
  - `varm["action_U"]`
  - `varm["action_A"]`
  - `obsm["action_B"]`
  - `uns["action_params"]["sigma"]`
- ACTION:
  - `obsm["H_stacked"]`
  - `obsm["H_merged"]`
  - `obsm["C_stacked"]`
  - `obsm["C_merged"]`
  - `obs["assigned_archetype"]`
- network:
  - `obsp["actionet"]`
- specificity:
  - cluster specificity outputs
  - archetype specificity outputs
- batch correction:
  - corrected reduction and associated parameters

Comparison rules:

- canonicalize SVD sign and ordering where needed
- compare sparse graphs in normalized CSR form
- use fixed seeds throughout
- require dense/sparse and in-memory/backed parity inside each language as a
  prerequisite for cross-language parity

## Suggested Implementation Sequence

1. R network cleanup
2. public contract update in `libactionet` headers and wrapper layers
3. native-orientation `reduceKernel` path without materialized expression
   transpose
4. native-orientation batch/basal orthogonalization
5. shared native-orientation specificity implementation
6. operator-backed IRLB
7. parity/golden test suite spanning R and Python
8. optional later cleanup of AA/SPA internals if full native-orientation math is
   still desirable beyond the wrapper boundary

## Risks

- Breaking `libactionet` public orientation is invasive and requires coordinated
  wrapper changes in both front ends.
- If operator-backed IRLB is skipped, native-orientation execution may change
  algorithm defaults and numerical behavior.
- Specificity has enough implementation drift today that parity failures are
  likely during consolidation unless the golden suite lands early.
- R build assumptions around PRIMME and large-index support need careful
  re-evaluation so operator support is not accidentally blocked again.

## Working Assumptions

- Breaking the public `libactionet` orientation contract is acceptable.
- PRIMME remains optional.
- The priority is removing full expression-matrix transpose/materialization
  overhead, not guaranteeing zero internal transpose operations on reduced
  matrices.
- A small internal transpose of `cells x k` inside `runACTION` is acceptable in
  v1 if it avoids delaying the broader unification effort.

## Useful File Map

`libactionet`

- `include/action/reduce_kernel.hpp`
- `src/action/reduce_kernel.cpp`
- `include/action/action_main.hpp`
- `src/action/action_main.cpp`
- `include/annotation/specificity.hpp`
- `src/annotation/specificity.cpp`
- `include/decomposition/matrix_operator.hpp`
- `include/decomposition/orthogonalization.hpp`
- `src/decomposition/orthogonalization.cpp`
- `include/network/build_network_core.hpp`
- `src/network/build_network.cpp`

`actionet-r`

- `R/utils_anndata_adapter.R`
- `R/utils_validation.R`
- `R/r_action.R`
- `R/main.R`
- `R/network_tools.R`
- `R/r_specificity.R`
- `src/wr_network.cpp`

`actionet-python`

- `src/actionet/core.py`
- `src/actionet/anndata_utils.py`
- `src/actionet/batch_correction.py`
- `src/actionet/annotation.py`
- `src/actionet/_matrix_source.py`
- `src/actionet/wp_utils.cpp`
- `src/actionet/wp_io.cpp`
- `src/actionet/wp_decomposition.cpp`
- `src/actionet/wp_annotation.cpp`

