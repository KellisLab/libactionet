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

### Public Python SVD surface: IRLB, Halko (Feng and PRIMME quarantined)

**Decision:**

- The public Python SVD API exposes only `"irlb"` and `"halko"` plus `"auto"`.
- `"auto"` selects IRLB for sparse in-memory inputs, Halko for dense in-memory inputs, and Halko for backed operator inputs.
- The core SVD entry points still expose four algorithm codes at the C++ level (`ALG_IRLB`, `ALG_HALKO`, `ALG_FENG`, `ALG_PRIMME`), but Feng and PRIMME are quarantined from Python. The Python pybind layer rejects raw algorithm IDs other than `ALG_IRLB` and `ALG_HALKO`.
- The backed `IRLB -> PRIMME` fast path in `runSVD_Operator` has been removed. Backed operators requesting `ALG_IRLB` now use the honest `svdIRLB(MatrixOperator&, ...)` overload unconditionally.
- The `MatrixOperator::prefer_block_solver_for_irlb()` virtual hint and its overrides in `BackedSparseMatrixOperator` / `BackedDenseMatrixOperator` have been deleted.
- `svd_primme.{cpp,hpp}`, `runSVD_PRIMME_Operator`, the vendored `src/extern/primme/` tree, `svd_feng.{cpp,hpp}`, and the corresponding C++ dispatch cases remain in the tree for one release window. Full deletion is tracked in `TODO.md` and gated on the follow-up SVD/GPU work stabilizing.

**Rationale:**

- Sparse `nnz > 2^31 - 1` is 64-bit clean under the force-defined `ARMA_64BIT_WORD` in `libactionet_config.hpp`; PRIMME is no longer needed for realistic omics matrices.
- The hidden backed dispatch violated the algorithm contract callers had a right to expect.
- Retaining the quarantined Feng/PRIMME sources for one release preserves a simple revert path without expanding the Python surface.

**Related:**

- Python-front-end decision record: `../../../context/DECISIONS.md`.
- GPU-backed SVD launchpad: `../../../plans/GPU_BACKED_SVD_AGENT_LAUNCHPAD.md`.
- C++/build GPU roadmap: `../plans/GPU_BACKEND_PLAN.md`.

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
- PRIMME and Feng are quarantined legacy code, not GPU implementation routes.

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
