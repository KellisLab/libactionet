# ACTIONet V2 Plan Pack

This directory is the authoritative implementation plan pack for the ACTIONet v2 optimization program.

## Program Goals

1. Deliver materially better scalability for 1M-10M+ cells across runtime, memory footprint, and backed-mode I/O.
2. Keep v2 physically separated from v1 codepaths to maximize modularity and simplify v1 deprecation.
3. Preserve the R user-facing API: if a v2 change breaks R behavior, patch the R layer immediately.

## Hard Constraints

- **Physical separation rule**
  - C++ v2 code resides in `src/libactionet/include/v2/...` and `src/libactionet/src/v2/...`.
  - v1 remains in current paths.
  - Shared code between v1 and v2 is limited to explicit utility/shared modules.
- **R API preservation rule**
  - Internal breakage is allowed.
  - Public R API signatures and expected behavior must remain preserved via compatibility patches.
  - Any impacted v2 merge is blocked until R compatibility status is green.

## File Map

- `00_DECISION_LOG.md` - Program-level decisions and change control.
- `10_REPO_LAYOUT_ABI_V2.md` - Physical separation and ABI v2 architecture.
- `20_BACKED_IO_V2.md` - Backed mode and disk I/O optimization plan.
- `30_SVD_KERNEL_V2.md` - SVD and kernel-reduction optimization plan.
- `40_ACTION_DECOMP_V2.md` - ACTION decomposition optimization plan.
- `50_NETWORK_CONSTRUCTION_V2.md` - Network build and ANN backend plan.
- `60_NETWORK_DIFFUSION_V2.md` - Diffusion kernel optimization plan.
- `70_PYTHON_FRONTEND_V2.md` - Python API/binding split and routing plan.
- `75_R_API_PATCHING.md` - Mandatory R API compatibility workstream.
- `80_BENCHMARK_GATES.md` - Performance and compatibility gate definitions.
- `90_ROLLOUT_DEPRECATION.md` - Rollout, opt-in->default migration, and deprecation policy.
- `agents/AGENT_ASSIGNMENTS.md` - Delegation map, ownership, and handoff contracts.

## Execution Order

1. Establish architecture and invariants: `00`, `10`.
2. Build performance foundations: `20`, `30`, `60`.
3. Optimize decomposition and graph stages: `40`, `50`.
4. Surface and control behavior in user APIs: `70`, `75`.
5. Enforce objective gates and rollout controls: `80`, `90`.

## Definition of Done (Program)

- v2 paths exist and are physically isolated from v1.
- v2 passes functional correctness tests against accepted tolerances.
- Benchmark gates are green for approved workstreams.
- R API compatibility gates are green for all impacted functions.
- Rollout flags and fallback controls are documented and implemented.
