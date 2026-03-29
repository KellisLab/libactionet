# 70 - Python Frontend and Binding v2

## Objective

Expose v2 functionality through a separate Python module stack while preserving v1 behavior and minimizing Python-side copy overhead.

## Scope

- Separate pybind extension module (`_core_v2`).
- `actionet.v2` Python package surface.
- v2 option objects and defaults.
- Compatibility and fallback behavior.

## Proposed Structure

```text
src/actionet/
  _core.cpp        # existing v1 bindings
  _core_v2.cpp     # new v2 bindings
  wp_*_v2.cpp      # v2 C++ wrapper entrypoints
  v2/
    __init__.py
    config.py      # dataclasses / typed options
    core.py        # stage APIs
    pipeline.py    # v2 orchestration defaults
```

## API Surface (v2)

- `actionet.v2.core.run_svd(...)`
- `actionet.v2.core.reduce_kernel(...)`
- `actionet.v2.core.run_action(...)`
- `actionet.v2.core.build_network(...)`
- `actionet.v2.core.compute_network_diffusion(...)`

Each API accepts explicit `*OptionsV2` or normalized keyword equivalents.

## Work Packages

### WP1 - Binding split

- Add `_core_v2` target and bootstrap functions.
- Register v2 enums/options structs and telemetry objects.

### WP2 - Copy elision and dtype policy

- Centralize `np.ascontiguousarray` usage only where required.
- Keep float32 fast paths where numerically acceptable.
- Prevent accidental extra conversions between Python and C++ boundaries.

### WP3 - Compatibility routing

- Keep v1 calls untouched by default.
- Add explicit opt-in toggles for v2 in public API and pipeline wrappers.

### WP4 - User observability

- Expose telemetry and gate-friendly metrics at Python level.
- Standardize structured benchmark output objects.

## Performance / Memory / I/O Estimate

| Metric | Estimate | Notes |
|---|---:|---|
| Runtime | 1.05x-1.3x faster | Mainly from copy reduction and cleaner dispatch. |
| Peak RAM | 10%-35% lower transient overhead | Depends on matrix sizes and conversion frequency. |
| I/O | 5%-15% lower incidental overhead | By reducing unnecessary materialization. |
| Confidence | Medium-High | Clear opportunities, modest but reliable gains. |

## Dependencies

- Depends on all technical workstreams (`20`-`60`).
- Must coordinate with `75_R_API_PATCHING.md` where shared semantics change.

## Test Plan

- API parity tests for v1 unchanged behavior.
- v2 argument validation and fallback tests.
- End-to-end pipeline smoke tests with v2 opt-in.

## Acceptance Criteria

- v1 remains default and unchanged.
- v2 APIs are usable and fully documented.
- Python-layer overhead is reduced in benchmark traces.
