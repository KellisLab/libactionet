# 90 - Rollout and Deprecation Policy

## Objective

Ship v2 safely with staged activation, clear fallback controls, and reversible default changes.

## Rollout Stages

### Stage A - Internal alpha (opt-in only)

- v2 available behind explicit flags.
- v1 remains sole default path.
- Focus: correctness, telemetry validation, R compatibility patch cycle.

### Stage B - Public beta (opt-in recommended for large workloads)

- Documented v2 opt-in APIs.
- Gate reporting published for major stages.
- Fallback-to-v1 controls mandatory for all v2-exposed endpoints.

### Stage C - Default-on for selected stages

- Promote only stages with green benchmark and compatibility gates.
- Keep emergency override to force v1 paths.

### Stage D - v1 deprecation preparation

- Announce planned deprecation windows.
- Freeze non-critical feature additions in v1.
- Continue security/bugfix support during overlap period.

## Feature Flags and Controls

- `ACTIONET_V2_ENABLE=0|1`
- Stage-specific toggles (SVD, ACTION, network, diffusion)
- `ACTIONET_FORCE_V1=1` emergency fallback

## R API Deprecation Interaction

- v1 backend internals may be retired progressively.
- R public API remains preserved through compatibility adapters until an explicit major-version policy change is published.

## Exit Criteria per Stage

- Stage A -> B: functional parity and core smoke benchmarks pass.
- Stage B -> C: stage-level performance gates pass and R compatibility gates stay green.
- Stage C -> D: sustained stability window with no unresolved high-severity regressions.

## Communication Requirements

- Release notes must state active defaults and fallback controls.
- Known deviations and unsupported v2 combinations must be documented.
- R users must receive explicit compatibility status in release notes.
