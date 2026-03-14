# Adaptive `k*nn` Rewrite With Future Backed-Mode Compatibility

## Summary

- Current `k*nn` scalability is blocked inside [build_network.cpp](/Users/sebastian/Documents/git_projects/libactionet/src/network/build_network.cpp), not in the Python or R wrappers. The hot path still allocates global `N x (kNN+1)` neighbor-id, distance, and lambda buffers, then duplicates edge state again during symmetrization.
- The Python and R front-ends are already thin enough for this work:
  - Python [build_network()`](/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/core.py#L450) passes row-major `float32` into `buildNetworkCore`.
  - Python [run_actionet()`](/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/pipeline.py#L183) still hardcodes `obsm_key="H_stacked"`, which is a separate scaling constraint and remains out of scope for this item.
  - R currently converts `NumericMatrix` to row-major `float32` and calls `buildNetworkCore`; no public API change is needed.
- The bundled HNSW is v0.8.0 (the current upstream release) and exposes `BaseSearchStopCondition` and `searchStopConditionClosest`. There is no `searchKnnCloserFirst` function in any hnswlib release — ascending-distance order is obtained by draining the `searchKnn` max-heap into a local vector and iterating in reverse. Do not mix a vendor refresh into this implementation. Design the rewrite so a later HNSW upgrade or stop-condition experiment can be evaluated independently.
- **Note:** Python preflight guardrails (parent plan item 1 — refusing or warning on obviously unsafe k*nn sizes) remain pending. This rewrite improves memory behavior materially but does not substitute for those guardrails. Item 1 should be implemented after this pass.

## Core Design

- Keep all public APIs unchanged:
  - `actionet::buildNetwork(...)`
  - `actionet::buildNetworkCore(const float*, n, dim, params)`
  - Python `build_network(...)`
  - R `buildNetwork(...)`
- Refactor the internals around a private row-access abstraction. Use **static polymorphism via a template parameter** — not a virtual base class — so the hot path inlines fully and no vtable overhead is introduced. Because `buildNetworkCore_from_reader` is instantiated only inside `build_network.cpp`, it can remain a private function template in that translation unit and does not need to appear in any public header.

```cpp
// Private to build_network.cpp.
// Reader must satisfy the PointReader concept (see below).
template<typename Reader>
static actionet::CSRGraph
buildNetworkCore_from_reader(const Reader& reader, const actionet::BuildNetworkParams& params);
```

The `PointReader` concept (informal, duck-typed):

```cpp
struct PointReader {
    std::size_t n_points() const;
    std::size_t dim() const;
    // Returns a pointer to the row's float32 data.
    // For in-memory readers: may return a pointer into the source buffer directly.
    // For JSD/backed readers: fills scratch and returns scratch.data().
    // The returned pointer is valid until the next call on the same scratch buffer.
    // Each thread must use its own scratch buffer (see AdaptiveScratch::row_buf).
    const float* load_row(std::size_t i, std::vector<float>& scratch) const;
};
```

- Implement one concrete reader in this pass:
  - `ContiguousFloat32Reader` over the existing row-major `const float* X`.
  - For `l2`/`ip`, `load_row()` returns `X + i * dim` directly; `scratch` is unused.
  - For `jsd`, `load_row()` clamps and normalizes row `i` into `scratch`, then returns `scratch.data()`.
- Do not keep the current full `X_norm_buf` copy for JSD.
  - Normalize rows on demand during index insertion and query.
  - This is the main backed-compatibility requirement for the input side: the core algorithm must consume rows, not a globally materialized normalized matrix.
- `buildNetworkCore(const float*, ...)` becomes a thin wrapper that constructs `ContiguousFloat32Reader` and forwards into `buildNetworkCore_from_reader`.
- Keep the adaptive logic exact-semantic at the graph level:
  - same `kNN = min(n-1, 5 * round(sqrt(n)))`
  - same `LC = 1 / density`
  - same metric handling and distance-to-similarity rules
  - same mutual and non-mutual symmetrization semantics
- Do not use `searchStopConditionClosest()` in the initial rewrite.
  - Reason: the first pass should be a parity-preserving memory rewrite.
  - Structure the query helper so a future implementation can swap from `searchKnnCloserFirst(kNN+1)` to a custom stop-condition search without changing wrappers, row readers, or CSR assembly.

## `ef` and `ef_construction` Behavior in k*NN

The current `buildNetworkCore_KstarNN` silently overrides both `ef_construction` and `ef` with `kNN`, discarding `params.ef_construction` and `params.ef`:

```cpp
const double ef_val = static_cast<double>(kNN);
auto idx_kstar = makeHnswIndex(..., p.M, ef_val);  // ef_construction = kNN
idx_kstar.hnsw->setEf(ef_val);                     // ef = kNN
```

This diverges from the `knn` path, which respects both parameters from `BuildNetworkParams`. The rationale is that `kNN` for adaptive search can be very large (e.g., 6,520 at N=1.7M), and using a smaller user-supplied `ef` or `ef_construction` would degrade recall at the adaptive query sizes.

**Decision for this pass:** Preserve this override behavior, but make it explicit and documented:

- Use `std::max(p.ef_construction, static_cast<double>(kNN))` for index construction.
- Use `std::max(p.ef, static_cast<double>(kNN))` for query.
- Add a comment in the code and a note in `BuildNetworkParams` documenting that for `k*nn`, effective `ef` and `ef_construction` are floored at `kNN` because the adaptive search radius requires it.

This is a behavior-preserving change (current code is equivalent to this floor for users who supply values below `kNN`) that makes the semantics explicit and avoids silently discarding user-supplied values that are larger than `kNN`.

## Adaptive Query Rewrite

- Replace the current global-buffer adaptive path with a row-local per-thread path.
- Add a private scratch struct, one instance per worker thread. The `row_buf` field serves as `load_row()` scratch and must not be shared between threads:

```cpp
struct AdaptiveScratch {
    std::vector<float> row_buf;                  // JSD row normalization or future backed rows; one per thread
    std::vector<std::pair<float, hnswlib::labeltype>> knn;
    std::vector<VertexIndex> local_srcs;
    std::vector<VertexIndex> local_dsts;
    std::vector<float> local_dists;
};
```

- Build HNSW by iterating rows through `Reader::load_row(i, scratch.row_buf)`:
  - For `jsd`, rows are normalized into `scratch.row_buf` before `addPoint()`.
  - For `l2`/`ip`, rows are passed directly from the source buffer.
- Query each row with `searchKnnCloserFirst(kNN + 1)`.
  - `searchKnnCloserFirst` returns results in ascending distance order (closest first).
  - **Self-exclusion:** exclude the query point by label, not by position. Do not assume self occupies any particular slot. This is correct regardless of result ordering, and handles duplicated rows (where a different point at distance 0 may appear before or at the same position as self). After filtering the self-label, apply the adaptive cutoff to the remaining non-self neighbor list in the order returned.
- Compute the adaptive cutoff incrementally without allocating a lambda array:
  - maintain `beta_sum` and `beta_sq_sum`
  - compute `lambda` inline for each successive non-self neighbor
  - as soon as `lambda < beta` for neighbor at position `k` (1-indexed within the non-self list), stop scanning; do **not** emit neighbor `k` (it is the first failing neighbor, not the last accepted one)
  - emit neighbors `1` through `k-1` (the neighbors that passed the cutoff check) — this matches the current code's semantics exactly: `neighbor_no` is set to `k` at the first failure, and neighbors `1..neighbor_no-1` are emitted
- Emit directed edges immediately into the worker-local arrays:
  - No global `idx_flat`
  - No global `dist_flat`
  - No global `lambda_flat`
  - No `#pragma omp critical` append inside the hot loop
- After the parallel region:
  - compute total emitted edge count from worker-local arrays
  - allocate final flat arrays once
  - concatenate worker-local arrays in one sequential pass

### `kNN = 0` Edge Case

`compute_kstar_knn(n)` returns 0 for `n < 2`. For `n = 1`, the early-return guard produces an empty graph before any HNSW operations. For `n = 2`, `kNN = 1`; `searchKnnCloserFirst(2)` returns 2 candidates; after self-exclusion, at most 1 neighbor remains, and the adaptive cutoff loop runs at most one iteration — producing a valid (possibly non-empty) graph. No special handling is needed beyond the existing `n < 2` guard.

## Direct CSR Symmetrization

- Replace the current `symmetrize_to_csr()` implementation with a direct two-pass CSR builder shared by both `k*nn` and `knn`.
- Input remains three directed-edge arrays: `srcs`, `dsts`, `dists`.

### Pass 1: Sort, Aggregate, and Count

1. Convert directed distances to directed similarities using the current metric-specific rules:
   - `jsd`: `sim = max(epsilon, 1.0 - dist)`
   - `l2`/`ip`: `sim = max(epsilon, max_d[dst] - dist)` where `max_d[dst]` is computed over all directed edges pointing to `dst`
2. Sort directed edges by unordered pair key `(min(src,dst), max(src,dst), src, dst)`.
   - The tie-break by directed `(src, dst)` makes deduplication of exact duplicate directed edges stable.
3. Aggregate (sum) consecutive directed edges with identical `(src, dst)` — these arise only from degenerate inputs but must be handled.
4. Walk the sorted, deduplicated list in unordered-pair groups. For each unordered pair `(lo, hi)`:
   - Collect `w_lo_hi` (sum of similarities for directed `lo→hi`) and `w_hi_lo` (sum of similarities for directed `hi→lo`).
   - Apply symmetrization rule and skip diagonal entries (`lo == hi`).
   - If the resulting symmetric weight `w_sym > 0`, increment `degree[lo]` and `degree[hi]` each by 1.
5. At the end of Pass 1: `degree[i]` holds the number of symmetric neighbors for row `i`.

Intermediate data at Pass 1 end:
- `flat_pairs`: sorted, aggregated unordered-pair list with computed `w_sym` (can be stored as `struct { VertexIndex lo, hi; float w_sym; }`).
- `degree[n]`: row degree array.

### Pass 2: Write CSR

1. Prefix-sum `degree` into `indptr` (length `n+1`, `indptr[0] = 0`).
2. Allocate `indices` and `data` with `total_nnz = indptr[n]` entries.
3. Walk `flat_pairs` in order. For each pair `(lo, hi, w_sym)`:
   - Write `hi` into `indices[write_pos[lo]]` and `w_sym` into `data[write_pos[lo]]`; advance `write_pos[lo]`.
   - Write `lo` into `indices[write_pos[hi]]` and `w_sym` into `data[write_pos[hi]]`; advance `write_pos[hi]`.
   - (`write_pos` is initialized to a copy of `indptr[0..n-1]` before this step.)

### Row-Sort Guarantee

The two-pass approach produces sorted row indices without an explicit per-row sort. Proof:
- For row `lo`: pairs in `flat_pairs` are sorted by `(lo, hi)`. For a fixed `lo`, `hi` is non-decreasing across the pairs where this row is the lower endpoint. Therefore entries written into row `lo` arrive with non-decreasing column index.
- For row `hi`: the lower endpoint `lo` increases in the global sort order. For a fixed `hi`, entries written into row `hi` arrive with non-decreasing `lo`, which is non-decreasing column index for that row.

An explicit post-fill per-row sort is not needed. Regression tests should confirm sorted indices on seeded inputs. If a regression test ever reveals unsorted rows, add the sort rather than patching the reasoning above.

### Output

- Keep the final output as `CSRGraph`.
- This is important for later backed mode: the same symmetrizer can eventually feed either an in-memory `CSRGraph` or a streamed `/obsp` writer.

## `hnsw_imp.hpp` Cleanup: Remove `getApproximationAlgo`

The legacy `getApproximationAlgo` function in `hnsw_imp.hpp` has a memory leak: it allocates a `SpaceInterface*` internally and returns only the `HierarchicalNSW<float>*`. The caller cannot delete the space, because it is not returned. The comment acknowledges this.

As of the previous pass, `getApproximationAlgo` has no callers in `build_network.cpp` or any other file that uses the new `makeHnswIndex` path. It should be removed in this pass:

- Delete `getApproximationAlgo` from `hnsw_imp.hpp`.
- Verify no other files in `libactionet`, `actionet-python`, or `actionet-r` call it.
- If any caller is found, replace the call with `makeHnswIndex(...)` which owns both objects via `HnswIndex` RAII.

## Backed-Mode Compatibility Requirements

- Treat this rewrite as stage 1 of a shared core for both in-memory and future disk-backed network construction.
- Hard requirements for this implementation:
  - all point access goes through the `Reader` template parameter (satisfies the `PointReader` concept)
  - all JSD preprocessing happens at row-load time inside `load_row()`, not as a whole-matrix copy
  - adaptive query code only depends on the `Reader`, HNSW, and per-thread scratch
  - CSR symmetrization only depends on directed-edge arrays and graph-size metadata
  - no Armadillo dense matrices in the adaptive hot path
  - no Python/R-specific logic in the adaptive hot path
- Explicitly avoid implementation choices that would block item `#6` later:
  - do not reintroduce a requirement for contiguous globally normalized `X`
  - do not make the adaptive path depend on `numpy` layout assumptions beyond the existing wrapper boundary
  - do not entangle CSR assembly with SciPy or `arma::sp_mat`
- Later backed mode should be able to add a `BackedObsmReader` that satisfies the `PointReader` concept and reuse:
  - HNSW build/query logic
  - adaptive cutoff logic
  - directed-edge emission
  - direct CSR symmetrization
- This rewrite does **not** make adaptive `k*nn` the recommended backed large-N path.
  - It only ensures the work does not have to be undone when a backed `knn` or experimental backed `k*nn` path is added.

## Estimated Memory Impact

### Removed Scratch Terms

**Old global scratch (eliminated):**

| Buffer | Formula | N=1.7M, kNN=6520 |
| --- | --- | --- |
| `idx_flat` (`hnswlib::labeltype`) | `8 * N * (kNN+1)` bytes | 88.7 GB |
| `dist_flat` (`float`) | `4 * N * (kNN+1)` bytes | 44.3 GB |
| `lambda_flat` (`float`) | `4 * N * (kNN+1)` bytes | 44.3 GB |
| **Total global scratch** | `16 * N * (kNN+1)` bytes | **177.4 GB** |

**New per-thread scratch (replaces above):**

Assumptions: `T = 16` threads, `k_bar = 32` average retained adaptive degree.

| Buffer | Formula | N=1.7M, T=16 |
| --- | --- | --- |
| Per-thread `knn` result (`pair<float,labeltype>`) | `12 * T * (kNN+1)` bytes | ~1.25 MB |
| Per-thread `row_buf` (JSD scratch, `float`) | `4 * T * dim` bytes | ~30 MB at d=464 |
| Per-thread `local_srcs/dsts/dists` edge vectors | `~12 * (N/T) * k_bar` bytes per thread | ~51 MB/thread, ~820 MB total |
| **Total new scratch** | — | **~851 MB** |

The dominant new scratch term is the per-thread local edge vectors, not the query buffers. At T=16 and k_bar=32 these are approximately 51 MB per thread before concatenation. This is still a reduction of over 170x compared to the global scratch at N=1.7M.

### HNSW Index Memory Estimate

The HNSW index memory (for `M = 16`) is estimated as:

```
HNSW_bytes ≈ N * (M * 2 * sizeof(std::size_t)   // layer-0 adjacency
              + sizeof(void*)                      // data pointer
              + sizeof(std::mutex)                 // per-element lock
              + level_overhead)
```

For `N = 1.7M`, `M = 16`, 64-bit platform:
- Layer-0 adjacency: `1.7M * 32 * 8 ≈ 435 MB`
- Per-element pointers + locks: ~`1.7M * 24 ≈ 41 MB`
- Higher-level graph (exponentially smaller, typically <10% of layer-0): ~`50 MB`
- Total: approximately `530 MB – 600 MB`

The `3.39 GB` figure in the previous table was inflated by inclusion of the eliminated global scratch and the `X_norm_buf` copy. With those removed, the HNSW index is the second-largest allocation after the caller-supplied `float32` input.

### Process-Peak Estimates After the Rewrite

Assumptions:
- in-memory Python/R path
- worst-case `H_stacked` width `d = 464`
- `M = 16`
- `distance_metric = "jsd"`
- `T = 16` threads
- average retained adaptive degree `k_bar = 32`
- estimates exclude allocator overhead and OS effects

| N | Caller `float32` input | HNSW index | Per-thread edge scratch | Final CSR | Peak estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.7M | 3.16 GB | ~0.57 GB | ~0.85 GB | ~1.09 GB | **~5.7 GB** |
| 2.0M | 3.71 GB | ~0.67 GB | ~1.00 GB | ~1.28 GB | **~6.7 GB** |

If `k_bar = 64`, add approximately:
- `+0.85 GB` at `1.7M`
- `+1.00 GB` at `2.0M`

Compared with the previous implementation, these estimates are lower by:
- the eliminated global adaptive scratch (~177 GB at N=1.7M)
- the eliminated full `X_norm_buf` copy for JSD (~3.16 GB at N=1.7M)

## Estimated Runtime Impact

- The rewrite does **not** change the dominant asymptotic cost of adaptive search.
  - `kNN` is still `O(sqrt(N))`
  - adaptive `k*nn` remains much slower than fixed-`k` `knn` at very large `N`
- Expected wall-time changes:
  - `l2`/`ip`, moderate datasets that already fit comfortably: roughly `1.2x` to `1.8x` faster
  - `jsd`, moderate datasets: roughly `0.9x` to `1.5x` of current wall time
  - large runs near the current memory cliff: practical speedup can be much larger because the old path enters allocator pressure, paging, or OOM
- Reason for the `jsd` range:
  - row-local normalization removes a full-matrix copy but computes normalization during both insert and query
  - on adaptive workloads, the removed `N * kNN` scratch traffic and post-processing usually dominates that extra `N * d` normalization work
- End-to-end expectation:
  - primary gain is memory collapse removal
  - secondary gain is moderate runtime improvement from:
    - no global neighbor/lambda passes
    - no `omp critical` edge merges
    - cheaper direct CSR finalization

## Repo-Specific Changes

- `libactionet`
  - implement the reader-based core and row-local adaptive rewrite in [build_network.cpp](/Users/sebastian/Documents/git_projects/libactionet/src/network/build_network.cpp)
  - implement the two-pass direct CSR builder as a drop-in replacement for `symmetrize_to_csr`, shared by both `k*nn` and `knn`
  - keep `BuildNetworkParams` and `CSRGraph` public contracts unchanged
  - add a comment to `BuildNetworkParams` documenting that `ef` and `ef_construction` are floored at `kNN` for the `k*nn` algorithm
  - remove `getApproximationAlgo` from `hnsw_imp.hpp` (memory leak, no remaining callers)
  - do not modify vendored HNSW headers in this pass
- `actionet-python`
  - no API change in [build_network()`](/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/core.py#L450)
  - sync vendored `src/libactionet` mirror after core implementation
  - no change to SciPy CSR conversion path
  - no change to [run_actionet()`](/Users/sebastian/Documents/git_projects/actionet-python/src/actionet/pipeline.py#L183) in this item
- `actionet-r`
  - no API change
  - sync vendored `src/libactionet` mirror after core implementation
  - keep current wrapper conversion and `armaSpMatFromCSR` path unchanged

## Validation and Benchmarks

### Parity Tests (`n_threads=1`)

- `algorithm="k*nn"`
- `distance_metric in {"jsd","l2","ip"}`
- `mutual_edges_only in {true, false}`
- compare old vs new graph topology and weights on seeded synthetic datasets

### Edge Cases

- `n < 2`
- `n = 2` (kNN=1, minimal adaptive graph)
- duplicated rows (two identical points; self-exclusion by label must still work)
- zero-sum JSD rows (normalization guard must produce a valid, not NaN, row)
- clipped JSD rows with values outside `[0,1]` (clamp must apply before normalization)
- low positive `density`
- very small `n` where `kNN < 2`

### Python Validation

- existing CSR/smoke tests remain unchanged
- add one adaptive-path parity/smoke test via Python binding

### R Validation

- small seeded smoke/parity run through the R wrapper
- if automated R coverage is not readily available, record this as manual validation

### Performance Benchmark Matrix

- `N in {50k, 100k, 250k}`
- `d in {30, 464}`
- `metric in {"jsd","l2"}`
- `n_threads in {1, 8}`

### Memory Regression Test

The primary acceptance criterion — that peak RSS no longer scales as `16 * N * (kNN+1)` — must be verified with an explicit measurement. Use one of the following methods depending on platform availability:

- **Linux:** `/usr/bin/time -v ./benchmark_binary 2>&1 | grep "Maximum resident"` or `valgrind --tool=massif --pages-as-heap=yes`
- **macOS:** `/usr/bin/time -l ./benchmark_binary 2>&1 | grep "maximum resident"` or `leaks --atExit -- ./benchmark_binary`
- **Python wrapper:** `import resource; resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` before and after `build_network()`

The benchmark should test at `N = 250k` (where the old scratch would be ~7 GB) and confirm that peak RSS is consistent with the new estimate (caller input + HNSW + per-thread edge scratch + CSR, no global scratch term).

### Acceptance Criteria

- peak RSS no longer grows according to `16 * N * (kNN+1)` adaptive scratch — verified by the memory measurement above
- no correctness regression versus current output on parity cases
- no clear runtime regression on medium `l2`/`ip` benchmarks
- small `jsd` regressions are acceptable only if memory behavior matches the new design and medium/large cases improve materially
- `getApproximationAlgo` is absent from `hnsw_imp.hpp` (leak removed, no callers remain)

## Assumptions and Follow-Up

- This pass preserves current adaptive graph semantics; it is not a scientific-method change.
- This pass is in-memory-first but backed-compatible by construction.
- This pass is not the place to expose `network_obsm_key`; that remains the follow-up needed for package-wide scaling in Python (parent plan item 2).
- This pass does **not** implement Python preflight guardrails for unsafe k*nn sizes; that remains pending as parent plan item 1 and should be implemented after this pass.
- This pass is not the place to vendor-refresh HNSW.
  - If desired, benchmark an upstream HNSW refresh afterward as a separate change.
  - Candidate follow-up: test whether `searchStopConditionClosest()` can reduce adaptive query runtime without changing graph outputs materially.
