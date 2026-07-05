
## GPU backend

See [`GPU_BACKEND_PLAN.md`](GPU_BACKEND_PLAN.md) for the full roadmap
and the post-mortem of the scrapped first attempt (`dev-gpu-v2`
branch, July 2026). The decisions below are the settled meta-decisions
that survive the reset and constrain the re-attempt.

### CUDA toolkit floor

**Decision:** CUDA 12.2 minimum. No 11.x fallback path.

**Rationale:** Intersection of (a) the oldest CUDA available on common
HPC sites that is Ampere-compatible, and (b) the minimum cuVS / RAPIDS
requirement. Phase 2 hard-requires cuVS >= 12.2. Supporting 11.x would
be effort spent on hardware that cannot run Phase 2 anyway.

### Compute capability floor

**Decision:** SM 8.0 (Ampere). Hard-coded, no CMake override, no
environment variable, no runtime flag.

**Rationale:** Phase 2 cuVS work makes Ampere a hard requirement, not
a soft preference. Ada (SM 8.9) and Hopper (SM 9.0) are the current
inclusion targets; earlier architectures are permanently out of scope.

### Platform support

**Decision:**

- Linux x86_64 with NVIDIA GPU: primary target, must build and run.
- Linux x86_64 without GPU: default configuration, must build identically to a no-GPU baseline (build option defaults `OFF`).
- Windows 11 + WSL2 with NVIDIA GPU: primary test environment for the developer, manual sign-off required before each push to a GPU branch.
- macOS (Apple Silicon or Intel): CPU-only forever. Primary regression guard via existing macOS CI.
- Windows native (no WSL2): out of scope, not supported, not tested.

### R + GPU combination

**Decision:** `LIBACTIONET_BUILD_R=ON` combined with
`LIBACTIONET_ENABLE_NVIDIA_GPU=ON` is a hard `FATAL_ERROR` in CMake
for v1. Deferred to Phase 4.

**Rationale:** R bindings currently can't benefit from PRIMME's
large-matrix path (>2^31 elements) without upstream `R_xlen_t` work,
and no R-side GPU rollout is scoped for v1. Failing early keeps the
constraint visible.

### Operator-backed SVD GPU support

**Decision:** SVD on `MatrixOperator`-backed inputs (HDF5-streamed,
LazyTransform, etc.) stays CPU-only permanently. Not a Phase-1 defer;
a permanent architectural decision.

**Rationale:** The matvec callbacks are CPU-side (Armadillo on host
memory). Any GPU dispatch would force per-iteration H2D/D2H round
trips that dominate any cuBLAS speedup. Users who need GPU SVD on
out-of-memory data must materialize a dense or sparse view first and
then call the in-memory GPU path.

### Multi-GPU / multi-process

**Decision:** Out of scope through Phase 3. Per-call `device_id` is
sufficient for the foreseeable HPC workflow.

---
