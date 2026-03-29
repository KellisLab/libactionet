# 80 - Benchmark and Compatibility Gates

## Objective

Provide objective go/no-go criteria for each workstream and prevent regressions during rollout.

## Benchmark Matrix

### Dataset scales

- Tier 1: ~1M cells
- Tier 2: ~5M cells
- Tier 3: ~10M+ cells

### Storage modes

- Sparse backed (compressed)
- Sparse backed (uncompressed)
- Dense backed
- In-memory reference slices

### Stages

- SVD / kernel reduction
- ACTION decomposition
- Network construction
- Network diffusion
- End-to-end pipeline

## Required Metrics

- Stage wall time (seconds)
- Peak RSS (GB)
- Read bytes and effective throughput (where relevant)
- Decompression time (backed compressed)
- Quality metrics per stage:
  - SVD residual and reconstruction error
  - ACTION stability/objective deltas
  - ANN recall and graph quality invariants
  - Diffusion output parity/tolerance

## Gate Thresholds

| Workstream | Gate |
|---|---|
| Backed I/O (`20`) | >=30% reduction in read/decompress wall time on backed benchmarks OR equivalent wall-time gain with parity maintained. |
| SVD (`30`) | >=30% wall-time reduction OR >=35% peak-memory reduction at accepted numeric tolerance. |
| ACTION (`40`) | >=25% wall-time reduction and >=20% peak-memory reduction with assignment stability in tolerance band. |
| Network (`50`) | >=25% index-memory reduction at or above quality/recall threshold. |
| Diffusion (`60`) | >=40% wall-time reduction on diffusion-heavy workloads with invariant parity. |
| Python frontend (`70`) | measurable copy-overhead reduction and no v1 behavior regression. |
| R compatibility (`75`) | all impacted checks GREEN (`R CMD check` clean, regression tests pass). |

## Reporting Format

Each benchmark run must produce:

1. Machine profile and build flags.
2. Dataset metadata and storage mode.
3. Raw metrics and normalized comparison vs baseline.
4. Pass/fail per gate.
5. Regression notes and follow-up actions.

## Failure Policy

- Any failed functional or compatibility gate blocks default migration.
- Failed performance gates do not block correctness merges, but block promotion to default paths.

## Confidence and Priority

This gate framework is **P0 program infrastructure**: no default switch without gate evidence.
