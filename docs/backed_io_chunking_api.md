# Backed Sparse I/O Chunking API (C++/pybind)

This note documents the backed sparse read-chunk controls exposed by the C++
API and low-level pybind bindings.

## Where the API lives

- C++ constructor:
  - `actionet::BackedSparseMatrixOperator(...)`
  - declared in `src/libactionet/include/io/backed_h5ad/backed_sparse_matrix_operator.hpp`
- C++ factory:
  - `actionet::createBackedOperator(...)`
  - declared in `src/libactionet/include/io/backed_h5ad/create_backed_operator.hpp`
- pybind low-level bindings:
  - `_core.BackedSparseMatrixOperator(...)`
  - `_core.create_backed_operator(...)`
  - implemented in `src/actionet/wp_io.cpp`

These parameters are currently exposed at the low-level `_core` interface. The
high-level Python frontend functions are intentionally not wired to the new
fraction knob yet.

Thread control wiring:

- High-level `run_svd(...)` and `reduce_kernel(...)` pass
  `backed_n_threads -> _core.create_backed_operator(..., n_threads=...)`.
- In-memory dense/sparse paths are unchanged and continue to rely on
  BLAS/library threading behavior.

## Parameters

For sparse-backed operators, the new controls are:

- `io_target_chunk_bytes` (`size_t`, default `0`)
- `io_target_chunk_fraction_of_cap` (`double`, default `0.5`)
- `n_threads` (`int`, default `0`; `0` = auto, `1` = serial)

Behavior:

- If `io_target_chunk_bytes > 0`, that explicit byte budget is used.
- If `io_target_chunk_bytes == 0`, an automatic target is computed from
  `chunk_size`, sparse structure, and `io_target_chunk_fraction_of_cap`.

## Mathematical relationship

Let:

- `bytes_per_nnz = sizeof(double) + sizeof(uint64_t) = 16`
- `axis_len = n_obs` for CSR, `n_var` for CSC
- `mean_nnz_axis = total_nnz / axis_len`
- `estimated_cap_nnz = chunk_size * mean_nnz_axis`

Then auto-targeting uses:

- `target_chunk_nnz = ceil(io_target_chunk_fraction_of_cap * estimated_cap_nnz)`
- `target_chunk_bytes = ceil(target_chunk_nnz * bytes_per_nnz)`

Equivalent MB form (sparse cap estimate):

- `estimated_cap_mb ~= chunk_size * mean_nnz_axis * 16 / 2^20`
- `auto_target_mb ~= io_target_chunk_fraction_of_cap * estimated_cap_mb`

So for a fixed dataset structure, auto target is approximately linear in
`chunk_size`.

## Why default fraction is 0.5

A coarse sweep on
`data/adata_agg_Hm_STR_MSN_1000plus_only_processed.h5ad` showed:

- `0.75 * cap` reduced memory, but left substantial avoidable RSS.
- `0.5 * cap` preserved practical backed SVD throughput while reducing peak RSS
  much more aggressively.

Integrated benchmark summary artifact:

- `tests/_tmp_chunk_target_integration_summary.json`

On that dataset, relative to prior backed defaults:

- Halko: about `+4%` wall-time, about `-61%` peak RSS
- IRLB: about `+4%` wall-time, about `-61%` peak RSS

Given the memory-first objective for backed atlas-scale workloads, `0.5` is the
default auto-target fraction.
