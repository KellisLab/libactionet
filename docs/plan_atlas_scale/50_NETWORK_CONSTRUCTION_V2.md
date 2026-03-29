# 50 - Network Construction v2

## Objective

Scale network construction with better ANN backends, lower index memory, and policy controls that avoid pathological growth at very large N.

## Scope

- HNSW path upgrades.
- Optional FAISS backend with compressed index modes.
- Large-N policy controls (`k*nn` vs bounded `knn`).
- Quality controls (recall/graph topology invariants).

## Proposed Interfaces

- `NetworkOptionsV2`
  - `ann_backend`: `hnsw|faiss`.
  - `index_codec`: backend-specific (`flat|sq8|pq`).
  - `algorithm`: `knn|k*nn`.
  - `large_n_policy`: `auto|prefer_knn|allow_kstar`.
  - existing controls (`M`, `ef`, `ef_construction`, `k`, `mutual_edges_only`).

## Work Packages

### WP1 - HNSW modernization

- Upgrade bundled hnswlib and expose relevant tuning controls.
- Re-validate memory safety/alignment assumptions in current integration.

### WP2 - FAISS backend integration (optional)

- Add CPU FAISS path with configurable index types.
- Provide capability detection and graceful fallback to HNSW.

### WP3 - Large-N policy tuning

- Add default policy to avoid unbounded expensive neighbor fanout for huge N.
- Keep explicit opt-in for `k*nn` when desired.

### WP4 - Graph quality/recall guardrails

- Benchmark recall and downstream quality impact for compressed modes.
- Enforce minimum quality gates before enabling new defaults.

## Performance / Memory / I/O Estimate

| Metric | Estimate | Notes |
|---|---:|---|
| Runtime | 1.2x-2.0x faster typical | Can be larger if avoiding high-cost `k*nn` at extreme N. |
| Peak RAM | 30%-80% lower index memory | Depends on backend/index codec selection. |
| I/O | Neutral | Mostly in-memory index operations for this stage. |
| Confidence | Medium | Backend-dependent and quality-threshold-sensitive. |

## Dependencies

- Requires v2 architecture from `10_REPO_LAYOUT_ABI_V2.md`.
- Config surfaced in `70_PYTHON_FRONTEND_V2.md`.

## Test Plan

- Recall benchmarks vs baseline nearest-neighbor quality.
- Graph structural invariants (symmetry rules, degree distribution sanity).
- Regression tests for metric modes (`jsd`, `l2`, `ip`).

## Acceptance Criteria

- Quality metrics meet defined minima.
- Runtime/memory gains demonstrated on benchmark tiers.
- Optional backend fallback works automatically when unavailable.
