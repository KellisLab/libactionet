# AnnData Orientation Unification — Implementation Plans

## Overview

This directory contains the phased implementation plans for unifying the
matrix orientation contract across the ACTIONet ecosystem:

- `libactionet` (C++ core)
- `actionet-r` (R frontend)
- `actionet-python` (Python frontend)

The goal is to move from the current `genes x cells` C++ contract to the
AnnData-native `cells x genes` contract, eliminating all frontend transpose
shims and reducing memory/compute overhead.

## Contract Breakage Notice

**All plans in this sequence intentionally break the existing public API
contract.** During the implementation period, intermediate states will exist
where one or more repos are incompatible. This is expected and approved.

The contract breakage period ends when Plan 07 passes all parity checks.

## Current Execution State

- As of 2026-03-21, Plans 00 and 01 are complete and verified.
- As of 2026-03-22, Plans 02 and 02A are complete.
- Plan 02A repaired the remaining orthogonalization contract mismatch and added
  a repo-local core validator (`validate_plan02_core`) that passes in both
  build modes.
- As of 2026-03-22, Plan 03 is complete.
- Plan 03 removed all C++-path transpose shims from `actionet-r`, fixed
  downstream dimension-helper cascade effects, and verified numerical parity
  with the Plan 00 baseline up to SVD sign conventions.
- As of 2026-03-22, Plan 03A is complete.
- Plan 03A fixed three confirmed regressions (layoutNetwork SVD-space bug,
  filterActionet axis inversion, bare-matrix input contract), added 9 new
  regression tests, and added a repeatable stage-03 validator script
  (`tests/validate_stage03.R`) that passes all 17 shape and numerical checks
  against the Plan 00 baseline.
- As of 2026-03-22, Plan 04 is complete.
- Plan 04 removed all C++-path transpose shims from `actionet-python`, updated
  the pybind11 boundary to use direct CSC construction (scipy→arma) and
  memcpy-based dense transport (arma→numpy, Fortran-order), fixed the
  orthogonalization field layout in `wp_decomposition.cpp` to the Plan 02
  public contract, updated operator orthogonalization wrappers to use the typed
  `KernelReductionResult` API, and added a repeatable stage-04 validator script
  (`tests/validate_stage04.py`) that passes all 20 shape and consistency checks.
|- As of 2026-03-22, Plan 05 is complete.
|- Plan 05 added a `MatrixOperator` overload to IRLB, removed the `ALG_IRLB` throw
  in `runSVD_Operator`, and removed the `LIBACTIONET_BUILD_R` gate from
  `reduceKernel_Operator`. Both standard and R builds compile cleanly. Operator IRLB
  agrees with in-memory IRLB to machine precision (`< 1e-13` in sigma).
- Plans 06-07 remain pending.
|- As of 2026-03-22, Plan 06 is complete (dense-backed C++ path addendum added).
|- Plan 06 made `computeFeatureSpecificity` non-mutating (shifts an internal copy instead of
|  modifying the caller's matrix), flipped `computeFeatureStats` and `computeFeatureStatsVision`
|  to accept `cells × genes` (removing the last `features × cells` survivors), removed the
|  two `.T`/`.T.tocsr()` transpose shims from `annotate_cells` in `annotation.py`, and updated
|  header documentation for both `specificity.hpp` and `marker_stats.hpp`.
|  A subsequent addendum added a full C++ dense-backed specificity path
|  (`BackedDenseMatrixOperator` overloads + pybind11 wrappers); `_compute_specificity_streamed`
|  is no longer called and is deprecated. All 21 Python checks and 12 R checks pass;
|  stage-03 R regression tests are unaffected.
|  A post-completion audit applied six additional fixes: `const` correctness on all
|  `computeFeatureSpecificity` signatures, unified `> 0` binarization gating across
|  dense/sparse/backed-sparse `getProbsObs` paths, `float` → `double` precision fix in
|  `computeFeatureStatsVision`, upgraded Python validator backed sections from SKIP to FAIL,
|  added sparse-vs-dense cross-storage parity to the R validator, and added `average_profile`
|  cross-storage consistency checks to both validators.
|- Plan 07 remains pending.

## Agent Environment Guidance

Implementing agents may create repo-local virtual environments or temporary
environments under `.venv` or `/tmp` and install Python, R, or build
dependencies as needed to execute the validation steps in these plans.

Prefer isolated environments over modifying unrelated global environments, and
record any nontrivial setup commands in the implementation handoff.

## Plan Index

| Plan | Title | Primary Repo(s) | Risk | Status |
|------|-------|-----------------|------|--------|
| [00](00_PARITY_BASELINE.md) | Parity Baseline Infrastructure | all | Low | **Complete** |
| [01](01_R_NETWORK_CLEANUP.md) | R Network Cleanup | actionet-r, libactionet | Low | **Complete** |
| [02](02_CPP_CORE_CONTRACT_FLIP.md) | C++ Core Contract Flip | libactionet | High | **Complete** |
| [02A](02A_ORTHOGONALIZATION_CONTRACT_REPAIR.md) | Orthogonalization Contract Repair | libactionet | High | **Complete** |
| [03](03_R_FRONTEND_ADAPTATION.md) | R Frontend Adaptation | actionet-r | Medium | **Complete** |
| [03A](03A_R_FRONTEND_POST_FLIP_REPAIR.md) | R Frontend Post-Flip Repair | actionet-r | Medium | **Complete** |
| [04](04_PYTHON_FRONTEND_ADAPTATION.md) | Python Frontend Adaptation + Boundary Optimization | actionet-python | Medium | **Complete** |
| [05](05_OPERATOR_BACKED_IRLB.md) | Operator-Backed IRLB | libactionet | Medium | **Complete** |
| [06](06_UNIFIED_SPECIFICITY.md) | Unified Specificity | libactionet, actionet-python | Medium | **Complete** |
| [07](07_FINAL_PARITY_VALIDATION.md) | Final Cross-Language Parity Validation | all | Low | Pending |

## Dependency Graph

```
00 Parity Baseline
│
├──► 01 R Network Cleanup
│
├──► 02 C++ Core Contract Flip
│    │
│    ├──► 02A Orthogonalization Repair
│    │    │
│    │    ├──► 03 R Frontend Adaptation ─────┐
│    │    │                                   │
│    │    └──► 04 Python Frontend Adaptation ─┤
│    │                                        │
│    └──► 05 Operator-Backed IRLB             │
│                                            ▼
│                                   06 Unified Specificity
│                                            │
└────────────────────────────────────────────►│
                                             ▼
                                    07 Final Parity Validation
```

### Parallelism

- Plans 03 and 04 can execute in parallel after Plan 02A
- Plan 05 can execute in parallel with Plans 03 and 04
- Plan 06 requires Plans 02A, 03, and 04 to be complete
- Plan 01 is independent and can land at any time after Plan 00

## Estimated Scope

| Plan | Files modified | Estimated effort |
|------|---------------|-----------------|
| 00 | ~3 new scripts + fixture | Small |
| 01 | ~3 files across 2 repos | Small |
| 02 | ~12 files in libactionet | Large |
| 02A | ~8 files in libactionet + docs | Small-Medium |
| 03 | ~12 files in actionet-r | Medium |
| 04 | ~8 files in actionet-python | Medium-Large |
| 05 | ~4 files in libactionet | Medium |
| 06 | ~5 files across 3 repos | Medium |
| 07 | ~4 new test scripts | Small |

## Performance Impact Summary

### Memory reduction (100k cells, 30k genes, 5% density sparse)

| Overhead source | Before | After |
|----------------|--------|-------|
| Python sparse expression transpose | ~1.8 GB | 0 |
| Python pybind sparse transport | ~8 GB transient | ~0.5 GB |
| Python pybind dense transport | ~24 GB (if dense) | ~0 (zero-copy) |
| R Matrix::t() expression transpose | ~2-4 GB | 0 |
| R Rcpp sparse transport | ~2 GB | ~0.5 GB |
| S_r internal transpose in runACTION | 0 | ~40 MB (negligible) |

### Runtime reduction

- Expression transpose elimination: saves O(nnz) per C++ call
- Boundary copy reduction: saves O(nnz) to O(n*m) per C++ call
- Network build double-transpose: saves O(n*k) in R
- Backed operator simplification: marginal improvement

## How to Use These Plans

Each plan is self-contained with:

1. **Position in sequence** — shows what must be done before and after
2. **Contract notice** — confirms that API breakage is permitted
3. **Detailed changes** — file-by-file, function-by-function instructions
4. **Validation steps** — how to confirm correctness after implementation
5. **Completion criteria** — what "done" looks like

An implementing agent should:

1. Read the plan fully before starting
2. Check out the appropriate branch
3. Implement changes stage by stage
4. Run validation after each stage
5. Run final validation at the end
6. Mark the plan as complete

## Reference Documents

- [ANNDATA_UNIFICATION_HANDOFF.md](../ANNDATA_UNIFICATION_HANDOFF.md) — original investigation and rationale
- [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) — project overview
- [DECISIONS.md](../DECISIONS.md) — architectural decisions
- [AGENT_PLAYBOOK.md](../AGENT_PLAYBOOK.md) — agent operating guidelines
