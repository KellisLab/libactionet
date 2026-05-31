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

## UMAP / uwot disconnected-vertex repair

**Decision:**

- `optimize_layout_uwot` (in `src/visualization/uwot_actionet.cpp`) invokes
  `uwot::connected_components_undirected` on the post-pruned graph and applies
  the canonical umap-learn `simplicial_set_embedding` initialization fix:
  per-component centroid subtraction + random per-component offset
  (`N(0, 10)`) for every connected component beyond the largest, plus a small
  per-coordinate Gaussian jitter (`N(0, 1e-4)`) on the entire embedding.
- The behavior is gated by `UwotArgs::repair_disconnected = true` (default).
- The R-side (Rcpp) layout wrapper is intentionally not yet updated to expose
  this knob; that is tracked as a deferred follow-up. Callers of the R wrapper
  get the safeguard implicitly via the C++ default.

**Rationale:**

- After `H.clean(w_max / n_epochs)` prunes weak edges, vertices whose only
  edges fall below the threshold end up with zero entries in `positive_ptr`.
  In batch mode (the default), `NodeWorker::operator()` iterates
  `positive_ptr[p] .. positive_ptr[p+1]`, so such vertices receive zero
  attractive AND zero repulsive updates - they are frozen at their seed
  coordinate for the entire run.
- With `run_actionet`-style seeds (`scale(archetype_footprint)[:, :n_components]`),
  a frozen vertex whose archetype membership is "average" lands exactly on
  one of the coordinate axes (`X = 0` or `Y = 0`), producing the cross-shaped
  artifact whose severity grows with N.
- The fix matches `umap-learn`'s reference behavior verbatim and adds linear-time
  overhead (`O(nnz + n_vertices)`), well below the cost of one optimization
  epoch.

**Reference:**

- `umap-learn` (lmcinnes/umap), `umap/umap_.py::simplicial_set_embedding`:
  `connected_components_undirected` -> `noisy_scale_coords` + per-component
  recentering loop.
- Vendored components helper: `include/extern/uwot/connected_components.h`.

---
