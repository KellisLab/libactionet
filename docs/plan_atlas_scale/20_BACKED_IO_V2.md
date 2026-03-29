# 20 - Backed I/O v2

## Objective

Reduce backed-mode bottlenecks by minimizing disk passes, decompression overhead, and temporary allocations in operator kernels.

## Scope

- v2 backed operator API and configuration.
- Sparse and dense backed kernel rewrites.
- HDF5 cache tuning controls.
- Optional staged uncompressed sidecar for multi-pass algorithms.
- I/O telemetry for ROI decisions.

## Proposed Interfaces

- `MatrixOperatorV2`
  - Required block ops: `matmat`, `rmatmat`.
  - Declared scalar type (`float32` or `float64`).
  - Access hint flags: `sequential_scan`, `transpose_scan`, `random_probe`.
- `BackedOperatorConfigV2`
  - `row_chunk_size`, `slab_byte_budget`.
  - `rdcc_nbytes`, `rdcc_nslots`, `rdcc_w0`.
  - `staging_mode`: `off|auto|force`.
  - `compression_policy`: `native|decompress_sidecar`.

## Work Packages

### WP1 - Sparse kernel rewrite

- Remove per-row `arma::rowvec` allocations in tight loops.
- Use thread-local reusable accumulators.
- Fuse transform + accumulate loops.
- Preserve row/column ordering semantics.

### WP2 - Dense slab path optimization

- Reduce row-major->column-major conversion overhead.
- Reuse slab buffers across iterations.
- Introduce blocked transform application and SIMD-friendly loops.

### WP3 - HDF5 cache control

- Plumb dataset cache properties per operator instance.
- Add safe defaults by dataset/chunk shape heuristics.
- Expose user overrides in Python v2 config.

### WP4 - Staged sidecar mode

- For multi-pass kernels (SVD/ACTION), optionally stage matrix to local uncompressed sidecar.
- Reuse sidecar by deterministic fingerprint.
- Auto-disable when free disk is insufficient.

### WP5 - I/O telemetry

- Emit stats: bytes read, decompression time, cache hit/miss proxy counters, pass count.
- Persist benchmark artifacts in structured JSON.

## Performance / Memory / I/O Estimate

| Metric | Estimate | Notes |
|---|---:|---|
| Runtime | 1.4x-3.5x faster in backed-heavy stages | Highly dependent on compression and chunk locality. |
| Peak RAM | 15%-35% lower | Mostly from reduced temporary allocations and controlled slabs. |
| I/O | 30%-70% better effective throughput | Driven by fewer redundant reads and decompression events. |
| Confidence | High | Hotspot locations and optimization levers are well identified. |

## Dependencies

- `10_REPO_LAYOUT_ABI_V2.md` must land first.
- Feeds directly into `30_SVD_KERNEL_V2.md`, `40_ACTION_DECOMP_V2.md`, and `70_PYTHON_FRONTEND_V2.md`.

## Test Plan

- Unit tests for operator correctness by sparse/dense and CSR/CSC variants.
- Backed parity tests against v1 outputs.
- Compression vs uncompressed benchmark A/B runs.

## Acceptance Criteria

- Functional parity within tolerances for operator products.
- Measurable reduction in allocation count and read/decompress wall time.
- Sidecar fallback behavior is deterministic and safe under low-disk conditions.
