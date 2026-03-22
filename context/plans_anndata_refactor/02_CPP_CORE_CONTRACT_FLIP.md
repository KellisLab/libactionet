# Plan 02 — C++ Core Contract Flip

## Position in Sequence

```
   00 Parity Baseline            [DONE]
   01 R Network Cleanup          [DONE]
>> 02 C++ Core Contract Flip <<
   03 R Frontend Adaptation
   04 Python Frontend Adaptation + Boundary Optimization
   05 Operator-Backed IRLB
   06 Unified Specificity
   07 Final Cross-Language Parity Validation
```

**Dependencies**: Plan 00 (baseline exists). Plan 01 is desirable but not
strictly required.
**Blocks**: Plans 03, 04 (frontend adaptations depend on this).

## Contract Notice

This plan is part of a coordinated cross-repo AnnData orientation unification.
Public API breakage across `libactionet`, `actionet-r`, and `actionet-python` is
explicitly permitted and expected until the full sequence completes.

See `context/ANNDATA_UNIFICATION_HANDOFF.md` for the complete rationale.

**This plan intentionally breaks the `libactionet` public header contract.**
After this plan lands, both `actionet-r` and `actionet-python` will fail to
produce correct results until Plans 03 and 04 are applied. This is expected.

## Objective

Change the `libactionet` public C++ API so that all external-facing functions
accept and return matrices in AnnData-native orientation:

| Data | Old contract | New contract |
|------|-------------|-------------|
| Expression matrix S | genes x cells | **cells x genes** |
| Reduced kernel S_r | k x cells | **cells x k** |
| Archetype weights H | k x cells | **cells x k** |
| Gene loadings U | genes x k | genes x k (unchanged) |
| Feature specificity | genes x k | genes x k (unchanged) |
| Network graph G | cells x cells | cells x cells (unchanged) |
| Perturbation A | genes x p | genes x p (unchanged) |
| Perturbation B | cells x p | cells x p (unchanged) |

## Repo

`libactionet` only. All changes in this plan are confined to `libactionet`.

## Strategy

The key insight: most internal algorithms (SPA, AA, simplex regression) are
deeply column-oriented and impractical to rewrite. The strategy is:

1. **Flip all public function signatures** to accept the new orientation.
2. **Inside each function**, apply small transposes on narrow matrices
   (k-wide, not gene-wide) where the inner algorithm requires the old layout.
3. **Never materialize a full expression-matrix transpose** inside C++.
   Expression-bound operations use transposed products (`S.t() * x` instead
   of `S * x`) or operator `rmatmat`/`rmatvec` calls.

This keeps the inner algorithm code untouched while moving all orientation
adaptation inside the C++ boundary.

## Detailed Changes by Subsystem

### Stage A: `reduceKernel` — The SVD Reduction Path

**Files**: `include/action/reduce_kernel.hpp`, `src/action/reduce_kernel.cpp`

#### A1: Update `KernelReductionResult` struct

Current:

```cpp
struct KernelReductionResult {
    arma::mat S_r;    // k x cells
    arma::vec sigma;  // k
    arma::mat U;      // genes x k
    arma::mat A;      // genes x p (perturbation left)
    arma::mat B;      // cells x p (perturbation right)
};
```

New:

```cpp
struct KernelReductionResult {
    arma::mat S_r;    // cells x k       (was k x cells)
    arma::vec sigma;  // k               (unchanged)
    arma::mat U;      // genes x k       (unchanged — gene loadings)
    arma::mat A;      // genes x p       (unchanged)
    arma::mat B;      // cells x p       (unchanged)
};
```

Update all doc comments to reflect the new shapes.

#### A2: Update `computeKernelPerturbationTermsInMemory`

This function computes mean-centering perturbation terms from the raw
expression matrix. Current logic (S is genes x cells):

```cpp
arma::vec mu = arma::mean(S, 1);          // row means → gene-length
arma::vec a1 = mu / arma::norm(mu);       // gene-length unit vector
arma::vec b1 = -S.t() * a1;              // cell-length projection
arma::rowvec c = arma::mean(S, 0);        // col means → cell-length
arma::vec a2 = ...;                        // gene-length
arma::vec b2 = ...;                        // cell-length
A = join_rows(a1, a2);                     // genes x 2
B = join_rows(b1, b2);                     // cells x 2
```

After flip (S is cells x genes):

```cpp
arma::vec mu = arma::mean(S, 0).t();      // col means → gene-length (was row means)
arma::vec a1 = mu / arma::norm(mu);       // gene-length unit vector
arma::vec b1 = -(S * a1);                 // cell-length = (cells x genes)(genes x 1)
                                           // was: -S.t() * a1 = -(genes x cells)' * genes
arma::vec c = arma::mean(S, 1);           // row means → cell-length (was col means)
// a2, b2 adjust similarly — the algebra is symmetric, just swap dim 0 ↔ 1
A = join_rows(a1, a2);                     // genes x 2 (unchanged shape)
B = join_rows(b1, b2);                     // cells x 2 (unchanged shape)
```

The critical property: `A` remains genes x p and `B` remains cells x p.
The perturbation identity `S_centered ≈ S - A * B'` still holds because the
centering math is symmetric — we are just reading means from the transposed
axes.

#### A3: Update `computeKernelPerturbationTerms` (operator path)

Current:

```cpp
S.matvec(ones_n, mu_sum);    // forward: (genes x cells)(cells x 1) → genes
S.rmatvec(a1, b1_tmp);       // reverse: (genes x cells)'(genes x 1) → cells
```

After flip (operator is cells x genes):

```cpp
S.rmatvec(ones_m, mu_sum);   // reverse: (cells x genes)'(cells x 1) → genes
S.matvec(a1, b1_tmp);        // forward: (cells x genes)(genes x 1) → cells
```

Swap every `matvec` ↔ `rmatvec` call. The operator dimensions also change:
`S.rows()` = cells, `S.cols()` = genes.

#### A4: Update `applyKernelPostSVD`

The SVD of `S` (cells x genes) yields:
- `U` = cells x k (left singular vectors)
- `sigma` = k
- `V` = genes x k (right singular vectors)

This is the natural swap from the old convention. The perturbed SVD adjusts
U and V using perturbation terms A and B.

Current code:

```cpp
out.S_r = V_rounded.t();  // V (cells x k) → S_r (k x cells)
out.U = perturbed.U;      // genes x k
```

New code:

```cpp
out.S_r = U_rounded;      // U (cells x k) → S_r (cells x k) — no transpose!
out.U = perturbed.V;      // V (genes x k) — gene loadings now come from V side
```

Wait — careful. The perturbation terms also swap sides. In the perturbed SVD:
- Left perturbation A corresponds to the row space of S (now cells)
- Right perturbation B corresponds to the column space of S (now genes)

But we defined A = genes x p and B = cells x p in the perturbation computation
(unchanged shapes). So the perturbed SVD call needs:

```cpp
SVDResult perturbed = perturbedSVD(svd, B, A);  // swap A ↔ B in the call
// Because: perturbedSVD expects (Apert for left/row-space, Bpert for right/col-space)
// Old: row-space = genes → A (genes x p). Col-space = cells → B (cells x p).
// New: row-space = cells → B (cells x p). Col-space = genes → A (genes x p).
```

Then:

```cpp
out.S_r = perturbed.U_scaled;  // cells x k (row-space side, scaled by sigma)
out.U   = perturbed.V;         // genes x k (col-space side, gene loadings)
```

Verify by checking `perturbedSVD` in `svd_main.cpp` to confirm parameter
order. The key identity: `S ≈ U * diag(sigma) * V'` and
`S_perturbed ≈ (U + Apert * ...) * diag(sigma) * (V + Bpert * ...)'`.

#### A5: Update template `reduceKernel<T>` and `reduceKernel_Operator`

The public signatures stay the same (they take S by reference), but the doc
comments and any internal dimension assertions change. The call to `runSVD`
is orientation-agnostic (SVD decomposes whatever matrix it receives).

Update the `#if LIBACTIONET_BUILD_R` gate comment in `reduceKernel_Operator`
to note that the operator now represents `cells x genes` (not `genes x cells`).

### Stage B: `runACTION` — The ACTION Decomposition Path

**Files**: `include/action/action_main.hpp`, `src/action/action_main.cpp`,
`src/action/action_decomp.cpp`, `src/action/action_post.cpp`

#### B1: Boundary transpose in `runACTION`

The SPA/AA/simplex internals are deeply column-oriented and must not be
rewritten. Instead, transpose at the boundary:

```cpp
// action_main.cpp
ResACTION runACTION(const arma::mat& S_r_in, int k_min, int k_max,
                    int max_it, double tol, int thread_no) {
    // Public contract: S_r_in is cells x k
    // Internal pipeline needs k x cells
    arma::mat S_r = S_r_in.t();

    // ... existing pipeline unchanged ...
    ResACTION result = runACTION_internal(S_r, k_min, k_max, max_it, tol, thread_no);

    // Transpose outputs back to cells-first orientation
    // H_stacked: was k_total x cells → cells x k_total
    // C_stacked: was cells x k_total → unchanged? Check actual shapes.
    // ... transpose H fields, keep C fields ...

    return result;
}
```

The exact output transpositions depend on the current shapes of
`ResACTION` fields. Current shapes:

| Field | Current shape | New shape |
|-------|--------------|-----------|
| `H_stacked` | archetypes x cells | **cells x archetypes** |
| `H_merged` | archetypes x cells | **cells x archetypes** |
| `C_stacked` | cells x archetypes | cells x archetypes (unchanged) |
| `C_merged` | cells x archetypes | cells x archetypes (unchanged) |
| `assigned_archetypes` | (cells,) | (cells,) (unchanged) |

So only H matrices need transposition on output.

#### B2: Update `ResACTION` struct documentation

Update all doc comments to reflect new shapes.

#### B3: Do NOT modify inner files

The following files must remain untouched:
- `src/action/spa.cpp`
- `src/action/aa.cpp`
- `src/action/simplex_regression.cpp`
- `src/action/action_decomp.cpp` (internal, receives k x cells)
- `src/action/action_post.cpp` (internal, receives k x cells)

The boundary transpose in B1 shields these from the contract change.

#### B4: Cost analysis

For a 100k-cell dataset with k=50:
- S_r is 100k x 50 = 5M elements = 40 MB
- Transpose cost: one O(n*k) memcpy ≈ microseconds
- This is negligible compared to AA iteration (seconds to minutes)

### Stage C: Specificity — Expression-Bound Path

**Files**: `include/annotation/specificity.hpp`, `src/annotation/specificity.cpp`

#### C1: In-memory overloads

Current (S is genes x cells, H is k x cells):

```cpp
arma::mat Ht = H.t();           // cells x k
arma::mat Obs = S * Ht;         // (genes x cells)(cells x k) = genes x k
```

After flip (S is cells x genes, H is cells x k):

```cpp
arma::mat Obs = S.t() * H;      // (cells x genes)'(cells x k) = genes x k
```

For sparse S, `S.t() * H` is efficient — Armadillo iterates the CSC structure
of S (columns = genes) and accumulates into Obs.

Row/column sum adjustments:

```cpp
// Old:
arma::vec row_p = arma::sum(Sb, 1);    // gene-length (sum across cells)
arma::rowvec col_p = arma::sum(Sb, 0); // cell-length (sum across genes)

// New (S is cells x genes):
arma::rowvec row_p_r = arma::sum(Sb, 0); // gene-length (sum down cell axis)
arma::vec row_p = row_p_r.t();            // keep as column vec for downstream
arma::vec col_p_v = arma::sum(Sb, 1);    // cell-length (sum across gene axis)
```

The normalization math (`getProbsObs`, `getProbsExp`, `getSignificance`)
operates on gene x k matrices and gene-length / cell-length vectors, and does
not depend on the orientation of S directly — only on the derived quantities
`Obs`, `row_p`, `col_p`. These are all the same shape regardless of S
orientation, so the downstream normalization code is unchanged.

#### C2: Labels overload

Current: builds `H(max_labels, S.n_cols)` where `S.n_cols` = cells.

After flip: `S.n_cols` = genes, `S.n_rows` = cells. Fix:

```cpp
arma::mat H(max_labels, S.n_rows);  // k x cells (for internal math)
// Or build as cells x k and transpose — either way the internal accumulation
// math stays the same once H has the right shape.
```

#### C3: Backed sparse overloads

These already use the operator interface. After the backed operator flip
(Stage E), `op.rows()` = cells, `op.cols()` = genes. Adjust the CSR/CSC
scan loops accordingly — the chunk iteration logic needs to swap the role
of "outer" and "inner" indices.

### Stage D: Orthogonalization

**Files**: `include/decomposition/orthogonalization.hpp`,
`src/decomposition/orthogonalization.cpp`

#### D1: In-memory batch effect path

Current (S is genes x cells, design is cells x covariates):

```cpp
arma::mat Z = S * design;          // (genes x cells)(cells x q) = genes x q
arma::mat B_raw = -(Z.t() * S).t();  // (genes x q)'(genes x cells) → (q x cells) → (cells x q)
```

After flip (S is cells x genes):

```cpp
arma::mat Z = S.t() * design;     // (cells x genes)'(cells x q) = genes x q
arma::mat B_raw = -(S * Z);       // (cells x genes)(genes x q) = cells x q — direct!
```

The output `Z` (genes x q) and `B_raw` (cells x q) have the same shapes as
before — they feed into `perturbedSVD` with the same roles.

#### D2: Operator batch effect path

Swap `matmat` ↔ `rmatmat`:

```cpp
// Old: S.matmat(design, Z)    → (genes x cells)(cells x q) = genes x q
// New: S.rmatmat(design, Z)   → (cells x genes)'(cells x q) = genes x q

// Old: S.rmatmat(Z, B_raw)   → (genes x cells)'(genes x q) = cells x q
// New: S.matmat(Z, B_raw)    → (cells x genes)(genes x q) = cells x q
```

#### D3: Basal orthogonalization

Same pattern — `basal_state` is genes x q (unchanged), and the products
with S swap from forward to reverse (or vice versa).

#### D4: Post-orthogonalization S_r assembly

The orthogonalization functions currently return S_r as `V.t()` (k x cells)
via `arma::trans(V)`. After the flip:

```cpp
// Old: out.S_r = V_scaled.t();  // cells x k → k x cells
// New: out.S_r = U_scaled;      // cells x k (direct, no transpose)
```

Wait — the orthogonalization uses `perturbedSVD` on the already-reduced SVD
components, not on the raw expression matrix. The S_r assembly here mirrors
the `reduceKernel` post-SVD logic. The U/V swap depends on whether the SVD
was computed on the old or new orientation. Since these functions take a
pre-computed SVD (from `reduceKernel`), and `reduceKernel` now returns
`S_r` as cells x k (with U = cells x k, V = genes x k), the orthogonalization
functions receive:

- `old_S_r` = cells x k
- `old_U` = genes x k
- `old_sigma` = k

They reconstruct V = S_r / diag(sigma) = cells x k, then U = old_U = genes x k.
After perturbation and re-SVD, the new S_r = new_V_scaled = cells x k.

This needs careful verification — trace through the exact math with the new
orientations to ensure the perturbed SVD is called with the correct argument
order.

### Stage E: Backed H5AD Operators

**Files**: `include/io/backed_h5ad/backed_sparse_matrix_operator.hpp`,
`include/io/backed_h5ad/backed_dense_matrix_operator.hpp`,
`src/io/backed_h5ad/create_backed_operator.cpp`

#### E1: Remove the internal transpose

Current design: on-disk `obs x var` → operator exposes `var x obs`:

```cpp
arma::uword rows() const override { return n_var_; }  // genes
arma::uword cols() const override { return n_obs_; }  // cells
```

New design: operator exposes **native** `obs x var`:

```cpp
arma::uword rows() const override { return n_obs_; }  // cells
arma::uword cols() const override { return n_var_; }  // genes
```

#### E2: Simplify matvec/matmat implementations

The current implementations internally transpose the file read operations to
present `genes x cells` semantics. After the flip, they present `cells x genes`
semantics, which matches the on-disk layout directly. This simplifies the
CSR/CSC iteration logic — the "outer" index in a CSR scan is now the obs
(cell) axis, which is the row axis of the operator, matching directly.

Review each of `matvec`, `rmatvec`, `matmat`, `rmatmat` and remove the
internal swaps.

### Stage F: R Wrapper Reference Copies

**Files**: `wrappers_r/wr_action.cpp`, `wrappers_r/wr_decomposition.cpp`,
`wrappers_r/wr_annotation.cpp`, `wrappers_r/wr_network.cpp`

Update these reference copies to document the new expected input/output
orientations. These are not build-critical (the actual R wrappers live in
`actionet-r/src/`), but keeping them aligned prevents confusion.

**Do not invest significant time here** — the real wrapper updates happen in
Plans 03 and 04.

## Validation

### Unit tests

If `libactionet/test/` has existing tests, update them to pass matrices in
the new orientation. If tests are sparse, add minimal round-trip tests:

1. Create a small synthetic `cells x genes` matrix
2. Run `reduceKernel` → verify S_r is `cells x k`
3. Run `runACTION` on S_r → verify H_stacked is `cells x archetypes`
4. Run `computeFeatureSpecificity` → verify output is `genes x k`

### Numerical parity

The most important check: given the same input data (just transposed to match
the new contract), do the outputs contain the same numerical values (possibly
transposed)?

Write a C++ test or a Python/R script that:

1. Loads the parity fixture
2. Runs the pipeline with the OLD code (dev-backed), captures outputs
3. Runs the pipeline with the NEW code, captures outputs
4. Compares values (accounting for the shape transposition)

For `reduceKernel`:
- Old `S_r` (k x cells) should equal new `S_r.t()` (or equivalently, new
  `S_r` (cells x k) transposed should match old `S_r`)
- Old `U` (genes x k) should equal new `U` (genes x k)
- `sigma` should be identical

For `runACTION`:
- Old `H_stacked` (archetypes x cells) should equal new `H_stacked.t()`

### Build

```bash
cd libactionet
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Also test R build mode:
cmake .. -DCMAKE_BUILD_TYPE=Release -DLIBACTIONET_BUILD_R=1
make -j$(nproc)
```

Both build modes must compile without error.

## Files Modified (Summary)

| File | Change type |
|------|------------|
| `include/action/reduce_kernel.hpp` | Struct docs, function docs |
| `src/action/reduce_kernel.cpp` | Perturbation math, post-SVD assembly, operator path |
| `include/action/action_main.hpp` | Struct docs, function docs |
| `src/action/action_main.cpp` | Boundary transpose in/out |
| `include/annotation/specificity.hpp` | Function docs |
| `src/annotation/specificity.cpp` | Product orientation, sum axes, labels overload |
| `include/decomposition/orthogonalization.hpp` | Function docs |
| `src/decomposition/orthogonalization.cpp` | Product orientation, operator swap |
| `include/io/backed_h5ad/backed_sparse_matrix_operator.hpp` | rows/cols swap, simplify |
| `include/io/backed_h5ad/backed_dense_matrix_operator.hpp` | rows/cols swap, simplify |
| `src/io/backed_h5ad/create_backed_operator.cpp` | Possibly no change (factory) |
| `wrappers_r/wr_*.cpp` | Doc updates only |

## Files NOT Modified

| File | Reason |
|------|--------|
| `src/action/spa.cpp` | Shielded by boundary transpose |
| `src/action/aa.cpp` | Shielded by boundary transpose |
| `src/action/simplex_regression.cpp` | Shielded by boundary transpose |
| `src/action/action_decomp.cpp` | Internal; receives old orientation from boundary |
| `src/action/action_post.cpp` | Internal; receives old orientation from boundary |
| `src/decomposition/svd_*.cpp` | SVD is orientation-agnostic |
| `src/network/build_network*.cpp` | Already handled in Plan 01, or Python-only |

## Completion Criteria

- All public headers document the new orientation contract
- `reduceKernel`, `runACTION`, `computeFeatureSpecificity`,
  `orthogonalizeBatchEffect` accept and return matrices in AnnData-native
  orientation
- Backed operators expose native `obs x var` without internal transpose
- No full expression-matrix transpose is materialized inside C++
- Both build modes (standard + R) compile without error
- Numerical parity with baseline confirmed (values match, shapes transposed)
