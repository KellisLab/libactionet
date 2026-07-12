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

- Python-front-end decision record: `../../context/DECISIONS.md`.
- GPU-backed SVD launchpad: `../../plans/GPU_BACKED_SVD_AGENT_LAUNCHPAD.md`.
- C++/build GPU roadmap: `../plans/GPU_BACKEND_PLAN.md`.

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
