# Agent Playbook — libactionet (C++ core)

## Purpose of this repository

This repository contains the **C++ core backend** for the ACTIONet ecosystem. It implements performance‑critical algorithms and data structures that are exposed to users via language bindings.

Downstream consumers:

- **R front‑end** (`actionet-r`) via Rcpp
- **Python front‑end** (`actionet-python`) via pybind11

Any change here may impact both wrappers and downstream pipelines.

---

## Repository layout (high level)

- `include/` — public C++ headers (API surface consumed by bindings)
  - `action/`, `annotation/`, `decomposition/`, `io/`, `network/`, `tools/`, `visualization/` — module headers
  - `utils_internal/` — internal helpers (not part of public API contract)
- `src/` — core implementations (mirrors `include/` structure)
- `cmake/` — CMake modules (`ConfigureApple`, `ConfigureBLAS`, `ConfigureOpenMP`, `ConfigurePRIMME`, `ConfigureR`)
- `CMakeLists.txt` — root build file
- `docs/` — algorithm and API documentation
- `context/` — agent context files and decision records
- `wrappers_r/` — reference copy of Rcpp wrapper code (primary R package lives in `actionet-r`)
- `_EXCLUDE/` — deprecated/archived code (not compiled)
- `**/extern/` — third-party code. Drop-in architecture. Must not be modified

---

## What success looks like

- Correct, well‑tested core algorithms
- Stable and clearly defined public APIs
- Performance improvements that do **not** silently change semantics
- Clean separation between internal implementation and exported interfaces
- Documentation that explains:
  - intended usage of public APIs
  - performance characteristics and assumptions
  - expected input/output behavior

---

## Hard guardrails (must follow)

- **Do not break public headers** without explicitly calling out the change and coordinating wrapper updates.
- Assume **both Rcpp and pybind11 bindings** consume exported APIs.
- Avoid introducing heavy dependencies or build steps that complicate HPC usage. Current external requirements: BLAS/LAPACK, HDF5 (C library), OpenMP.
- Avoid global state unless explicitly justified and documented.

---

## How to work safely in this repo

### When modifying existing functionality

1. Identify affected public headers in `include/`.
2. Check whether R and/or Python wrappers rely on these symbols.
3. Update or add tests that cover the modified behavior.
4. Document behavior changes in `README.md` or `docs/`.

### When adding new functionality

1. Define the C++ API first (header + documentation comment).
2. Implement in `src/` with clear ownership and lifetime semantics.
3. Add focused tests with minimal input sizes.
4. Flag whether wrappers should expose the new feature (do not assume).

---

## Performance work checklist

Before implementing optimizations, confirm the goal:

- Reduce runtime?
- Reduce memory usage?
- Improve scalability?

Preferred approaches:

- Algorithmic improvements over micro‑optimizations
- Reduce allocations and copies
- Favor contiguous memory and cache‑friendly layouts

Be cautious about:

- Changing numeric stability
- Changing ordering or determinism
- Exposing performance‑sensitive templates through the ABI

If profiling data is not available, **ask for it**.

---

## Interaction with language bindings

- Treat exported APIs as a **contract**.
- Minimize complex ownership semantics across the C++/binding boundary.
- Prefer simple POD‑like structures or clearly documented lifetimes.

If an API change is required:

- Propose the C++ change first
- Explicitly note required R and Python wrapper updates
- Avoid staggered breakage (core updated but wrappers lagging)

---

## Common pitfalls

- Adding features that assume Python‑only or R-only usage
- Breaking R behavior unintentionally when optimizing core code
- Breaking R package build when modifying core code or cmake build system
- Introducing build options that diverge between platforms

---

## When blocked, ask for

- Minimal example inputs that trigger the issue
- Expected outputs (golden or reference results)
- Which wrapper(s) currently fail (R, Python, or both)
- Build environment details (compiler, platform, flags)
