# Decision Log - ACTIONet V2

This log records binding decisions for the v2 program. Decisions here are normative unless explicitly superseded.

## Active Decisions

| ID | Decision | Status | Rationale | Impact |
|---|---|---|---|---|
| D-001 | Implement v2 with physical source separation from v1 wherever possible. | Approved | Reduce coupling, simplify migration/deprecation, avoid accidental regressions. | Build/layout changes; separate targets and modules. |
| D-002 | ABI-breaking changes are allowed only in v2 versioned paths. | Approved | Preserve v1 stability while enabling deep optimization work. | New symbols/interfaces in `actionet::v2`. |
| D-003 | R user-facing API must be preserved; breakages require immediate R patching. | Approved | R API continuity is mandatory. | Blocking compatibility gates for impacted merges. |
| D-004 | CPU + NVMe single-node is the primary optimization target. | Approved | Matches dominant scaling bottlenecks and deployment assumptions. | Prioritize I/O, sparse kernels, and RAM efficiency. |
| D-005 | Dependency posture is aggressive, but new backends must be optional/guarded. | Approved | Keep adoption flexibility while allowing high-payoff integrations. | Feature flags and fallback paths required. |
| D-006 | Not all workstreams must ship; implementation is priority-driven by measured ROI. | Approved | Practical delivery and risk control. | Gate-driven go/no-go by section. |
| D-007 | v2 remains opt-in until benchmark and R compatibility gates are green. | Approved | Safe rollout and reversibility. | Rollout flags and staged migration policy. |

## Change Control

A decision can be changed only when all conditions are met:

1. A replacement decision is proposed with explicit rationale.
2. Impact analysis includes runtime, memory, I/O, and compatibility implications.
3. Affected plan files are updated in the same change set.
4. For decisions affecting R behavior, `75_R_API_PATCHING.md` is updated.

## Non-Negotiable Program Invariants

- v1 and v2 can be built and tested independently.
- Each v2 merge declares whether R compatibility is impacted.
- If impacted, R compatibility checks are required before merge.

## Open Policy Items

- Exact deprecation date for v1 defaults is deferred until milestone gates in `80_BENCHMARK_GATES.md` are satisfied.
