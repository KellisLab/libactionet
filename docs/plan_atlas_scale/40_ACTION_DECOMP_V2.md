# 40 - ACTION Decomposition v2

## Objective

Reduce runtime and memory in ACTION decomposition by eliminating redundant per-k work and accelerating simplex-heavy AA loops.

## Scope

- `decompACTION` k-path execution strategy.
- AA solver internals and simplex regression backend.
- Early stopping/pruning policy.
- Numerical parity and stability controls.

## Proposed Interfaces

- `ACTIONOptionsV2`
  - `warm_start_k_path`: bool.
  - `simplex_solver`: `active_set|pgd|fista|auto`.
  - `early_stop_enabled`: bool.
  - `objective_delta_tol`, `stability_window`, `max_it`.

## Work Packages

### WP1 - k-path warm starts

- Reuse decomposition state from `k` to `k+1`.
- Avoid cold-start SPA/AA at each `k` when prior state is valid.

### WP2 - Batched simplex solver path

- Implement projected-gradient/FISTA simplex path for batch updates.
- Retain active-set fallback for edge-conditioned problems.
- Add deterministic convergence and fallback thresholds.

### WP3 - Adaptive pruning

- Stop exploring `k` values with diminishing objective return.
- Keep override flags for exhaustive mode.

### WP4 - Threading and memory controls

- Bounded scratch allocations per thread.
- Prevent over-parallelization memory spikes on large k-ranges.

## Performance / Memory / I/O Estimate

| Metric | Estimate | Notes |
|---|---:|---|
| Runtime | 1.4x-3.0x faster in ACTION stage | Gains depend on k-range width and dataset geometry. |
| Peak RAM | 20%-40% lower | Primarily from bounded scratch and fewer duplicated states. |
| I/O | Neutral | Stage is compute-dominant once reductions are in memory. |
| Confidence | Medium | High upside with moderate algorithmic risk. |

## Dependencies

- Uses reduced representations from `30_SVD_KERNEL_V2.md`.
- Python parameter surface through `70_PYTHON_FRONTEND_V2.md`.

## Test Plan

- Archetype assignment stability checks against v1 baseline.
- Objective trajectory parity/tolerance checks.
- Edge-case tests for low-rank/noisy matrices.

## Acceptance Criteria

- Stable outputs within agreed tolerance bands.
- Lower runtime and peak memory at target scales.
- Fallback logic verified on adversarial cases.
