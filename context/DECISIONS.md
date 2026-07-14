# Decisions (ADR-lite)

This document records **deliberate architectural and operational decisions** for the ACTIONet ecosystem. These decisions are considered settled unless explicitly revised.

---

## Software architecture

### Multi-repo structure

**Decision:** Maintain separate repositories for:

- `libactionet` (C++ core)
- `actionet-r` (R front-end)
- `actionet-python` (Python front-end)
- `ACTIONetExperiment` (Deprecated: Backwards-compatibility, AnnData symmetric data container for `actionet-r`)

**Rationale:**

- Clear separation of concerns
- Independent packaging and release cycles (C++ / CRAN-style / PyPI-style)
- Avoids monorepo friction while preserving coordination via shared specs

---

## Language bindings

### C++ core + wrappers

**Decision:**

- C++ core built with **CMake**
- R bindings via **Rcpp**
- Python bindings via **pybind11**

**Rationale:**

- Mature, stable tooling
- Explicit control over ABI and performance
- Good compatibility with HPC environments

---

## Front-end prioritization

### Python vs R

**Decision:**

- Python front-end is the **performance-first and pipeline-critical interface**
- R front-end remains supported and is more feature-complete.

**Rationale:**

- R performance and ecosystem limitations at scale
- Python integration with pipeline and HPC workflows
- Preserve backward compatibility for existing R users

---

<!-- ## Reproducibility and stability

### Output contracts
**Decision:**
- Output formats, directory structures, and file naming are treated as **contracts**
- Changes require explicit documentation and migration plans

### Versioning
**Decision:**
- Critical dependencies (especially `actionet-python`) must be version-pinned and logged in pipeline runs

--- -->

## SVD algorithm strategy

### Public Python SVD surface: IRLB, Halko (Feng and PRIMME removed)

**Decision:**

- The public Python SVD API exposes only `"irlb"` and `"halko"` plus `"auto"`.
- `"auto"` selects IRLB for sparse in-memory inputs, Halko for dense in-memory inputs, and Halko for backed operator inputs.
- The core SVD entry points expose two algorithm codes at the C++ level (`ALG_IRLB` = 0, `ALG_HALKO` = 1). The Python pybind layer rejects any other raw algorithm ID.
- The backed `IRLB -> PRIMME` fast path in `runSVD_Operator` has been removed. Backed operators requesting `ALG_IRLB` now use the honest `svdIRLB(MatrixOperator&, ...)` overload unconditionally.
- The `MatrixOperator::prefer_block_solver_for_irlb()` virtual hint and its overrides in `BackedSparseMatrixOperator` / `BackedDenseMatrixOperator` have been deleted.
- `svd_primme.{cpp,hpp}`, `runSVD_PRIMME_Operator`, the vendored `src/extern/primme/` tree, `cmake/ConfigurePRIMME.cmake`, `svd_feng.{cpp,hpp}`, `ALG_FENG`/`ALG_PRIMME`, and the corresponding C++ dispatch cases have all been deleted.
- The standalone `actionet-r` package still exposes `algorithm=2` (Feng) and `algorithm=3` (PRIMME) bindings and needs a matching cleanup patch; see `TODO.md`. The `wrappers_r/` files inside this repository are reference-only copies and were intentionally left untouched.

**Rationale:**

- Sparse `nnz > 2^31 - 1` is 64-bit clean under the force-defined `ARMA_64BIT_WORD` in `libactionet_config.hpp`; PRIMME is no longer needed for realistic omics matrices.
- The hidden backed dispatch violated the algorithm contract callers had a right to expect.
- Feng did not win any auto-selection tier on the benchmark set and duplicated Halko's randomized category; retaining it added no distinct capability.
- PRIMME also caused persistent ODR/LTO warnings against Armadillo's BLAS/LAPACK symbols; the deletion obsoletes that defect class.

**Related:**

- Python-front-end decision record: `../../../context/DECISIONS.md`.
- GPU-backed SVD launchpad: `../../../plans/GPU_BACKED_SVD_AGENT_LAUNCHPAD.md`.
- C++/build GPU roadmap: `../plans/GPU_BACKEND_PLAN.md`.

---

## BLAS policy for ACTION

### Internal kernels for small and skinny AA operations

**Decision:**

- Keep ACTION's existing OpenMP decomposition over archetype count `k` as the
  owner of coarse-grained parallelism.
- Route dense operations whose smaller matrix dimension is at most 128 through
  the private column-major kernel layer in
  `include/utils_internal/utils_small_dense.hpp`.
- Retain CBLAS/Armadillo for larger general-purpose matrices.
- Do not add BLAS-vendor detection, process-global or per-scope BLAS thread
  mutation, environment requirements, or public diagnostics.
- Preserve active-set structure, iteration limits, regularization,
  convergence behavior, output orientation, and public C++ interfaces.

**Rationale:**

- On identical source and input, MKL completed `run_action` in 9.40 s while
  OpenBLAS-OpenMP required 79.36 s. The regression begins in AA, not SPA, and
  is concentrated in repeated tiny/skinny BLAS calls.
- Runtime OpenBLAS setters changed the reported thread count without reliably
  changing the already-initialized execution path, so the rejected thread
  guards did not address the measured bottleneck.
- The shape threshold covers default reduced ACTION workloads while preserving
  optimized BLAS throughput for genuinely large dense products.

**Correctness contract:**

- Thread/backend comparisons require identical assignments and numerically
  equivalent C/H matrices at `rtol=1e-8`, `atol=1e-10`.

**Deferred:**

- Batched active-set solves, blocked/fused AA updates, workspace reuse, and
  convergence-policy evaluation are a separate redesign and are not part of
  this semantics-preserving fix.

**Related:**

- `../../../plans/openblas_threading_and_odr_findings.md`
- `../../../tests/benchmark_action_blas_backends.py`

---

## GPU backend scope and platform

### NVIDIA CUDA backend: optional, Python-first, SVD-first

**Decision:**

- GPU support is optional and disabled by default.
- The first supported front-end is Python. R-facing GPU support is deferred.
- The first algorithmic target is Halko-style randomized SVD over product
  backends; GPU support is not a separate public SVD algorithm.
- Disk-backed data is a first-class GPU target. Do not treat CPU
  `MatrixOperator::matmat` as the GPU product boundary.
- Native CUDA toolkit primitives are the default implementation direction.
  RAFT/RAPIDS may be evaluated only as an optional spike after the native
  product/streaming boundary exists.
- PRIMME and Feng have been deleted; they are not GPU implementation routes.

**Platform contract:**

- Linux x86_64 with NVIDIA GPUs is the production/runtime target.
- Linux x86_64 without GPUs must build like the CPU-only baseline.
- Windows 11 + WSL2 with NVIDIA GPUs is the developer validation target.
- CUDA 12.2 is the minimum toolkit target. CUDA 11.x is out of scope.
- Supported hardware starts at SM 8.0 / Ampere. Default architecture lists
  should cover Ampere, Ada, and Hopper (`80;86;89;90`) unless narrowed for a
  specific deployment.
- macOS remains CPU-only. Native Windows is out of scope.

**Rationale:**

- The failed PRIMME/cuBLAS attempt exposed the need for explicit host/device
  ownership, real GPU canaries, and validation on actual target hardware.
- The default dependency profile must remain compatible with de-containerized
  conda/HPC builds.

**Related:**

- `../plans/GPU_BACKEND_PLAN.md`
- `../../../plans/GPU_BACKED_SVD_AGENT_LAUNCHPAD.md`
- `../../../plans/GPU_INTEGRATION.md`

---

## Change management

### Backward compatibility

**Decision:**

- Breaking changes are allowed if justified.
- Such changes must substantially improve:
  - Performance
  - Resource usage
  - User ease-of-use
  - Reproducibility

### Agent behavior

**Decision:**

- LLM/coding agents should not re-litigate decisions recorded in this document
- Deviations require explicit human approval

---
