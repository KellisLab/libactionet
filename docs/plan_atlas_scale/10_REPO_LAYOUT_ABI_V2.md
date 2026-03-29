# 10 - Repository Layout and ABI v2

## Objective

Create a physically separated v2 implementation architecture that allows aggressive optimization without destabilizing v1.

## Scope

- C++ library layout and namespace strategy.
- Build target split and linkage policy.
- Compatibility shim policy and controlled sharing.
- Python binding module separation.

## Proposed Layout

```text
src/libactionet/
  include/
    ... existing v1 headers
    v2/
      api/
      decomposition/
      io/
      network/
      action/
      annotation/
      visualization/
      tools/
    v1_compat/
      adapters/
      dispatch/
  src/
    ... existing v1 sources
    v2/
      decomposition/
      io/
      network/
      action/
      annotation/
      visualization/
      tools/
    v1_compat/
      adapters/
      dispatch/
```

Python split:

```text
src/actionet/
  ... existing v1 python modules
  v2/
    __init__.py
    core.py
    pipeline.py
    config.py
  _core.cpp        # v1 binding module
  _core_v2.cpp     # v2 binding module
  wp_*_v2.cpp      # v2 wrappers
```

## ABI and Symbol Strategy

- New public symbols under `actionet::v2`.
- v1 symbols remain unchanged.
- No in-place signature mutations for exported v1 symbols.
- Compatibility adapters allowed only in `v1_compat` directories.

## Build Targets

- `actionet_v1` (existing behavior)
- `actionet_v2` (new behavior)
- `actionet_shared_utils` (small, explicit, dependency-minimal shared internals)

Recommended CMake controls:

- `LIBACTIONET_ENABLE_V1=ON`
- `LIBACTIONET_ENABLE_V2=ON`
- `ACTIONET_PY_ENABLE_V2=ON`
- `ACTIONET_V2_EXPERIMENTAL_BACKENDS=ON|OFF`

## Implementation Steps

1. Create v2 directory skeleton and module headers.
2. Add new `actionet_v2` target with minimal bootstrap functionality.
3. Introduce `_core_v2` pybind module with smoke-test callable.
4. Add `v1_compat` adapter layer boundaries.
5. Prohibit direct include-crossing from v1 to v2 except through adapters.

## Performance / Memory / I/O Estimate

| Metric | Estimate | Notes |
|---|---:|---|
| Runtime | Neutral | Structural refactor; no intended runtime gain by itself. |
| Peak RAM | Neutral | No direct memory optimization in this section. |
| I/O | Neutral | No direct I/O optimization in this section. |
| Confidence | High | Architecture-only change with low measurement uncertainty. |

## Risks and Mitigation

- Risk: accidental duplication and drift between v1 and v2.
  - Mitigation: explicit shared-utility policy and static include checks.
- Risk: build complexity growth.
  - Mitigation: feature flags and narrow target boundaries.

## Acceptance Criteria

- v1 and v2 targets compile independently.
- v1 behavior unchanged for existing tests.
- v2 smoke tests execute.
- No unapproved path overlap between v1 and v2 modules.
