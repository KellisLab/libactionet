# Plan 02A — Orthogonalization Contract Repair

## Status: COMPLETE

Implemented on `feature/orientation-unification` on 2026-03-22.

This follow-up patch closed the last Plan 02 contract gap before starting
Plan 03: orthogonalization now consumes and returns the same public reduction
layout as `reduceKernel()` in both the in-memory and operator-backed APIs.

Repo-local validation now passes in both standard and R build modes via the
`validate_plan02_core` executable.

## Position in Sequence

```
   00 Parity Baseline            [DONE]
   01 R Network Cleanup          [DONE]
   02 C++ Core Contract Flip     [DONE]
>> 02A Orthogonalization Repair  [DONE] <<
   03 R Frontend Adaptation
   04 Python Frontend Adaptation + Boundary Optimization
   05 Operator-Backed IRLB
   06 Unified Specificity
   07 Final Cross-Language Parity Validation
```

**Dependencies**: Plan 02  
**Blocks**: Plans 03 and 04

## Contract Notice

This plan remains inside the approved AnnData orientation breakage window.
It does not introduce a new orientation change; it repairs the remaining
orthogonalization APIs so the public `libactionet` contract is internally
consistent before any frontend adaptation starts.

## Problem

The first Plan 02 landing left one critical inconsistency:

1. `reduceKernel()` exported the public reduction contract
   `{S_r, sigma, U, A, B}` in AnnData-native orientation.
2. The field-based orthogonalization path still forwarded those fields into the
   legacy `perturbedSVD(field, A, B)` adapter, which interprets fields as
   `{U, sigma, V, A, B}`.
3. The operator-backed orthogonalization API still exposed raw `SVDResult`
   semantics, which no longer matched the public reduction contract used by the
   rest of Plan 02.

This meant the orthogonalization surface was not safe for either C++ callers or
for Plans 03/04 to target directly.

## Implemented Changes

### 1. Public reduction contract is now the only orthogonalization contract

Updated `include/decomposition/orthogonalization.hpp` and
`src/decomposition/orthogonalization.cpp` so that:

- `orthogonalizeBatchEffect(T&, field, design)` expects `field` in public
  reduction layout `{S_r, sigma, U, A, B}`
- `orthogonalizeBasal(T&, field, basal)` expects the same layout
- `orthogonalizeBatchEffect_Operator(...)` now takes
  `KernelReductionResult`, not raw `SVDResult`
- `orthogonalizeBasal_Operator(...)` now takes `KernelReductionResult`

### 2. Added explicit conversion between public reduction state and mathematical SVD state

Inside `orthogonalization.cpp`, the implementation now:

- reconstructs the mathematical left singular vectors from `S_r / sigma`
- uses public `U` as the right singular vector matrix (gene loadings)
- maps public perturbation history back to mathematical left/right spaces:
  - public `B` → left / row-space (cells)
  - public `A` → right / col-space (genes)
- converts the perturbed result back into the public reduction layout by:
  - scaling left singular vectors into `S_r`
  - storing right singular vectors as public `U`
  - swapping perturbation history back to public `A/B`

### 3. Fixed the orthogonalization deflation call order

The deflation helper now computes:

- gene-space perturbation `A` (genes × q)
- cell-space perturbation `B` (cells × q)

and calls:

```cpp
perturbedSVD(svd, B_aug, A_aug, prior)
```

so the left/right perturbation arguments match the cells×genes SVD orientation.

### 4. Repaired Plan 02 runtime axis conversions

While adding the validator, two latent runtime bugs from the original Plan 02
landing were fixed:

- `src/action/reduce_kernel.cpp` now converts row means to `arma::vec`
  correctly for both dense and sparse matrices
- `src/annotation/specificity.cpp` now converts gene-axis row/column sums to
  `arma::vec` without invalid rowvec→vec construction

### 5. Synced the R wrapper reference copy

Updated `wrappers_r/wr_decomposition.cpp` so the reference copy now passes the
public reduction contract directly instead of reconstructing legacy `V` state
and transposing `S_r` on the way out.

## Validation

### Build

Both library build modes compile successfully:

```bash
cmake -S . -B /tmp/libactionet-review -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/libactionet-review -j4

cmake -S . -B /tmp/libactionet-review-r \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBACTIONET_BUILD_R=1 \
  -DR_HOME="$(R RHOME)"
cmake --build /tmp/libactionet-review-r -j4
```

### Repo-local core validator

Added `test/validate_plan02_core.cpp` and a `validate_plan02_core` build
target. The validator:

1. Builds a non-square synthetic matrix (7 cells × 4 genes)
2. Runs `reduceKernel()` for dense and sparse inputs
3. Validates the public reduction shapes
4. Validates in-memory `orthogonalizeBatchEffect()` / `orthogonalizeBasal()`
5. Validates operator-backed `orthogonalizeBatchEffect_Operator()` /
   `orthogonalizeBasal_Operator()`
6. Compares each result against a direct reference `perturbedSVD(...)`
   computation using the repaired public-contract conversions

Run:

```bash
/tmp/libactionet-review/validate_plan02_core
```

Expected output ends with:

```text
validate_plan02_core: PASS
```

## Deliverables

| Artifact | Description |
|----------|-------------|
| `include/decomposition/orthogonalization.hpp` | Public contract repaired |
| `src/decomposition/orthogonalization.cpp` | Conversion + deflation fix |
| `test/validate_plan02_core.cpp` | Repo-local core validator |
| `CMakeLists.txt` | Validator target |
| `wrappers_r/wr_decomposition.cpp` | Reference wrapper copy synced |
| `src/action/reduce_kernel.cpp` | Runtime axis-conversion fix |
| `src/annotation/specificity.cpp` | Runtime axis-conversion fix |

## Completion Criteria

- [x] In-memory orthogonalization accepts the public reduction contract directly
- [x] Operator-backed orthogonalization accepts the public reduction contract directly
- [x] Left/right perturbation spaces are mapped correctly under cells×genes orientation
- [x] Repo-local core validator passes
- [x] Both standard and R build modes compile
- [x] Plans 03 and 04 updated to target the repaired API
