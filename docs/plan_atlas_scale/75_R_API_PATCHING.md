# 75 - Mandatory R API Patching and Compatibility Gates

## Objective

Preserve the public R API while v2 internals evolve. Any v2 change that affects R behavior must include R-side compatibility patches before merge.

## Hard Rule

- If a v2 C++ change impacts R-facing behavior, the corresponding R patch is **blocking**.
- No impacted change merges with R compatibility status `RED`.

## Scope

- R wrapper and bridge updates.
- Semantic adapters for orientation, dtype, and shape differences.
- R regression tests and check gates.
- Compatibility matrix maintenance.

## Compatibility Matrix (Mandatory Artifact)

Maintain this table for every merged v2 change:

| Change ID | C++ Symbol / Behavior Change | R Functions Impacted | Surface Impact | Required Adapter | Test Added | Status |
|---|---|---|---|---|---|---|
| EXAMPLE | `actionet::v2::runSVD(...)` output orientation | `run_svd`, `reduce_kernel`, downstream plot helpers | Potential orientation mismatch | transpose adapter in R bridge | yes | GREEN |

Status values:

- `GREEN`: fully patched and tested.
- `YELLOW`: patch in progress, merge blocked.
- `RED`: no patch or failing compatibility tests, merge blocked.

## Expected R Touchpoints

Primary files likely to be impacted by v2 integration:

- [r_decomposition.R](/Users/sebastian/Documents/git_projects/actionet-python/R/r_decomposition.R)
- [r_action.R](/Users/sebastian/Documents/git_projects/actionet-python/R/r_action.R)
- [r_specificity.R](/Users/sebastian/Documents/git_projects/actionet-python/R/r_specificity.R)
- [wr_decomposition.cpp](/Users/sebastian/Documents/git_projects/actionet-python/R/wr_decomposition.cpp)
- [wr_action.cpp](/Users/sebastian/Documents/git_projects/actionet-python/R/wr_action.cpp)
- [wr_network.cpp](/Users/sebastian/Documents/git_projects/actionet-python/R/wr_network.cpp)

## Work Packages

### WP1 - Impact declaration contract

Every v2 PR must declare one of:

- `R_IMPACT=NONE`
- `R_IMPACT=PATCH_REQUIRED`

If `PATCH_REQUIRED`, include matrix row in this file and linked patch PR/commit.

### WP2 - Adapter layer implementation

- Add explicit conversion/adapters where v2 semantics differ from R expectations.
- Preserve existing argument names, return object structure, and default behavior.

### WP3 - Regression suite expansion

- Add tests for each patched function path.
- Cover shape/orientation invariants, numeric tolerance parity, and edge cases.

### WP4 - Continuous compatibility gate

- Run `R CMD check` and targeted R regression suite in CI for impacted changes.
- Gate merge on green status.

## Performance / Memory / I/O Estimate

| Metric | Estimate | Notes |
|---|---:|---|
| Runtime | Neutral to slight overhead (0%-10%) | Adapter work may add minor conversion overhead. |
| Peak RAM | Neutral | Small temporary buffers possible depending on adapter path. |
| I/O | Neutral | No direct I/O heavy operations expected. |
| Confidence | High | Compatibility process is procedural and test-driven. |

## Acceptance Criteria

- R API signatures remain stable for public functions.
- Existing R examples and workflows execute successfully.
- `R CMD check` is clean on supported platforms.
- Numeric parity to agreed tolerance bands for impacted operations.
