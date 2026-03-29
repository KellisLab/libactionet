# 30 - SVD and Kernel Reduction v2

## Objective

Cut SVD and kernel-reduction costs for large matrices by reducing pass count, improving numerical throughput, and lowering peak memory.

## Scope

- New v2 SVD backend set and selection policy.
- Block Krylov implementation for operator-backed mode.
- Mixed precision strategy.
- Pass budgeting and observability.
- Integration with kernel reduction (`reduceKernel` v2 path).

## Proposed Interfaces

- `SVDOptionsV2`
  - `algorithm`: `auto|halko|block_krylov|primme`.
  - `dtype`: `float32|float64|mixed`.
  - `pass_budget`: integer (hard cap on matrix passes or equivalent).
  - `max_it`, `seed`, `verbose`.
  - `telemetry_enabled`: bool.

- `SVDTelemetryV2`
  - matvec count, block-matvec count, pass estimate.
  - cumulative bytes read and decompression time (from operator telemetry).
  - convergence diagnostics and residual summary.

## Work Packages

### WP1 - Block Krylov backend

- Implement randomized block Krylov method for matrix operators.
- Reuse block `matmat/rmatmat` calls to reduce callback overhead.
- Add robust orthogonalization checks and fallback handling.

### WP2 - Mixed precision execution path

- Compute dominant products in float32 where safe.
- Keep numerically sensitive operations in float64 (orthogonalization/residual checks).
- Add deterministic fallback to full float64 on instability triggers.

### WP3 - Pass-budgeted scheduler

- Enforce user or auto-calibrated pass budget.
- Support graceful fallback (`block_krylov` -> `halko` -> `primme`) if convergence risk detected.

### WP4 - Kernel reduction integration

- Integrate new SVD outputs into v2 kernel reduction.
- Minimize additional passes after SVD (fuse perturbation/reconstruction steps where practical).

## Performance / Memory / I/O Estimate

| Metric | Estimate | Notes |
|---|---:|---|
| Runtime | 1.3x-2.4x faster in SVD-dominant stages | Highest gain on backed matrices with efficient block products. |
| Peak RAM | 30%-50% lower | From mixed precision and bounded intermediate allocations. |
| I/O | 20%-45% improvement | Mostly from reduced or better-utilized passes. |
| Confidence | Medium | Strong basis, but data-dependent convergence behavior. |

## Dependencies

- Requires `20_BACKED_IO_V2.md` operator telemetry and block ops.
- Exposes options through `70_PYTHON_FRONTEND_V2.md`.

## Test Plan

- Numerical parity tests for singular vectors/values under tolerance envelopes.
- Stress tests on ill-conditioned matrices.
- Regression checks on deterministic seeds.
- Backed vs in-memory consistency checks.

## Acceptance Criteria

- No correctness regressions beyond accepted tolerances.
- Runtime and memory improvements at benchmark scales.
- Pass-budget logic behaves predictably and logs reasoned fallback decisions.
