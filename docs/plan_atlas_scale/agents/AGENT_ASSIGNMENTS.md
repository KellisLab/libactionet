# Agent Assignments - ACTIONet V2 Program

This file defines delegation ownership, write scopes, and handoff contracts for parallel execution.

## Global Rules

- Every agent owns a disjoint primary write scope to minimize merge conflicts.
- Agents do not revert or overwrite changes outside their scope.
- Any cross-scope change requires explicit handoff note and integration-owner approval.
- If a change may affect R behavior, mark `R_IMPACT=PATCH_REQUIRED` and notify Agent H.

## Agent Map

### Agent A - Repo Split and ABI Scaffolding

- Scope: architecture and build split.
- Primary files/modules:
  - `src/libactionet/include/v2/...`
  - `src/libactionet/src/v2/...` (bootstrap)
  - CMake target wiring for `actionet_v2` and `_core_v2`.
- Depends on: none.
- Deliverables: compiles with v1 and v2 side by side.

### Agent B - Backed I/O

- Scope: operator and backed I/O kernels.
- Primary modules:
  - `src/libactionet/src/v2/io/...`
  - `src/libactionet/include/v2/io/...`
- Depends on: Agent A.
- Deliverables: backed I/O telemetry + kernel improvements.

### Agent C - SVD

- Scope: v2 SVD backends and kernel-reduction integration.
- Primary modules:
  - `src/libactionet/src/v2/decomposition/...`
  - `src/libactionet/src/v2/action/...` (kernel-reduction interface points)
- Depends on: Agent A, Agent B.

### Agent D - ACTION Decomposition

- Scope: AA/simplex and k-path optimizations.
- Primary modules:
  - `src/libactionet/src/v2/action/...`
- Depends on: Agent A, Agent C.

### Agent E - Network Construction

- Scope: ANN backend integration and graph builder policies.
- Primary modules:
  - `src/libactionet/src/v2/network/build_*`
  - backend integration submodules.
- Depends on: Agent A.

### Agent F - Diffusion

- Scope: diffusion kernels and optional backend integration.
- Primary modules:
  - `src/libactionet/src/v2/network/diffusion_*`
- Depends on: Agent A.

### Agent G - Python Frontend

- Scope: `_core_v2` bindings and `actionet.v2` package.
- Primary modules:
  - `src/actionet/_core_v2.cpp`
  - `src/actionet/wp_*_v2.cpp`
  - `src/actionet/v2/...`
- Depends on: Agents B-F for stable interfaces.

### Agent H - R API Patching (Blocking)

- Scope: R compatibility adapters and R regression checks.
- Primary modules:
  - [r_decomposition.R](/Users/sebastian/Documents/git_projects/actionet-python/R/r_decomposition.R)
  - [r_action.R](/Users/sebastian/Documents/git_projects/actionet-python/R/r_action.R)
  - [r_specificity.R](/Users/sebastian/Documents/git_projects/actionet-python/R/r_specificity.R)
  - [wr_decomposition.cpp](/Users/sebastian/Documents/git_projects/actionet-python/R/wr_decomposition.cpp)
  - [wr_action.cpp](/Users/sebastian/Documents/git_projects/actionet-python/R/wr_action.cpp)
  - [wr_network.cpp](/Users/sebastian/Documents/git_projects/actionet-python/R/wr_network.cpp)
- Depends on: any agent introducing R-impacting changes.
- Gate authority: merge block if compatibility status is not GREEN.

### Agent I - Benchmark/Gates Integration

- Scope: benchmark harness, metrics, gate reporting.
- Primary modules:
  - benchmark scripts and CI gate wiring.
  - run-report artifacts and threshold enforcement.
- Depends on: all technical agents.

## Dependency Graph

- Phase 1: A
- Phase 2 (parallel): B, E, F
- Phase 3: C (after B), D (after C), G (after B-F)
- Phase 4 (continuous + blocking): H
- Phase 5: I

## Handoff Contract (Required in each agent PR)

1. What changed.
2. File list within owned scope.
3. Public/interface impacts.
4. `R_IMPACT=NONE|PATCH_REQUIRED`.
5. Tests executed.
6. Known risks and follow-up items.

## Merge Readiness Checklist

- Scope ownership respected.
- Interface changes documented.
- Benchmark status attached (if performance-affecting).
- R compatibility status GREEN for impacted changes.
