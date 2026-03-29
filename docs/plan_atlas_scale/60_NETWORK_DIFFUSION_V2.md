# 60 - Network Diffusion v2

## Objective

Eliminate diffusion bottlenecks by replacing repeated column-wise sparse matvec loops with blocked sparse-dense kernels and sparse-aware approximation modes.

## Scope

- Blocked SpMM-based diffusion kernels.
- Approximate push-style diffusion mode for sparse seeds.
- Optional GraphBLAS backend for CPU sparse algebra.
- Numerical and invariant checks.

## Proposed Interfaces

- `DiffusionOptionsV2`
  - `method`: `chebyshev|power|push`.
  - `block_cols`: integer block width.
  - `approx_tol`, `max_it`, `alpha`, `norm_method`.
  - `backend`: `native|graphblas` (optional).

## Work Packages

### WP1 - Blocked native kernels

- Replace per-column SpMV loop with blocked sparse-dense multiply path.
- Reuse dense work buffers across iterations.

### WP2 - Sparse push approximation

- Add push-based approximate diffusion path for sparse inputs.
- Tune residual thresholds for quality/performance tradeoff.

### WP3 - Optional GraphBLAS backend

- Integrate CPU GraphBLAS path behind feature flag.
- Maintain identical API and fallback behavior.

### WP4 - Invariant safety layer

- Preserve non-negativity clamping and normalization invariants where required.
- Add method-specific convergence diagnostics.

## Performance / Memory / I/O Estimate

| Metric | Estimate | Notes |
|---|---:|---|
| Runtime | 2x-6x faster on diffusion-heavy workloads | Strongest gains for multi-feature diffusion matrices. |
| Peak RAM | 10%-30% lower | Reused blocked buffers and fewer transient vectors. |
| I/O | Neutral | Compute-dominant stage. |
| Confidence | High | Current hotspot is clearly column-wise kernel structure. |

## Dependencies

- Architecture split from `10_REPO_LAYOUT_ABI_V2.md`.
- Python parameterization via `70_PYTHON_FRONTEND_V2.md`.

## Test Plan

- Numeric parity against v1 for default methods.
- Convergence behavior tests across graph sizes and sparsity levels.
- Approximation quality checks for push mode.

## Acceptance Criteria

- Diffusion invariants preserved.
- Benchmark gates for runtime gains pass.
- Fallback behavior is deterministic across methods/backends.
