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

## Plan Index

| Plan | Title | Primary Repo(s) | Risk | Status |
|------|-------|-----------------|------|--------|
| [00](00_PARITY_BASELINE.md) | Parity Baseline Infrastructure | all | Low | Pending |
| [01](01_R_NETWORK_CLEANUP.md) | R Network Cleanup | actionet-r, libactionet | Low | Pending |
| [02](02_CPP_CORE_CONTRACT_FLIP.md) | C++ Core Contract Flip | libactionet | High | Pending |
| [03](03_R_FRONTEND_ADAPTATION.md) | R Frontend Adaptation | actionet-r | Medium | Pending |
| [04](04_PYTHON_FRONTEND_ADAPTATION.md) | Python Frontend Adaptation + Boundary Optimization | actionet-python | Medium | Pending |
| [05](05_OPERATOR_BACKED_IRLB.md) | Operator-Backed IRLB | libactionet | Medium | Pending |
| [06](06_UNIFIED_SPECIFICITY.md) | Unified Specificity | libactionet, actionet-python | Medium | Pending |
| [07](07_FINAL_PARITY_VALIDATION.md) | Final Cross-Language Parity Validation | all | Low | Pending |

## Dependency Graph

```
00 Parity Baseline
│
├──► 01 R Network Cleanup
│
├──► 02 C++ Core Contract Flip
│    │
│    ├──► 03 R Frontend Adaptation ──────────┐
│    │                                       │
│    ├──► 04 Python Frontend Adaptation ─────┤
│    │                                       │
│    └──► 05 Operator-Backed IRLB            │
│                                            ▼
│                                   06 Unified Specificity
│                                            │
└────────────────────────────────────────────►│
                                             ▼
                                    07 Final Parity Validation
```

### Parallelism

- Plans 03 and 04 can execute in parallel (they modify different repos)
- Plan 05 can execute in parallel with Plans 03 and 04
- Plan 06 requires Plans 02, 03, and 04 to be complete
- Plan 01 is independent and can land at any time after Plan 00

## Estimated Scope

| Plan | Files modified | Estimated effort |
|------|---------------|-----------------|
| 00 | ~3 new scripts + fixture | Small |
| 01 | ~3 files across 2 repos | Small |
| 02 | ~12 files in libactionet | Large |
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
