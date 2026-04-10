## NVIDIA GPU Backend v1 for `libactionet` (3 Phases)

### Summary
Implement an **optional, NVIDIA-only GPU backend** for SVD, network construction, and ACTION (`AA` + `SPA`) with these locked constraints:
- CUDA toolkit/runtime requirement: **>= 12.2**.
- Hardware floor: **Ampere+ only** for all GPU features.
- CPU remains default behavior; GPU is opt-in per call.
- R scope now: C++ hooks only; no R-facing GPU rollout in this effort.
- Performance gates: SVD >=1.5x, Network >=1.5x, ACTION >=1.3x with parity tolerances.

### Implementation Changes
1. Cross-cutting runtime/build foundation (start in Phase 1, reused in all phases)
- Add build flag `LIBACTIONET_ENABLE_NVIDIA_GPU` (default `OFF`).
- When enabled, require CUDA >=12.2 and link required GPU libs (cuBLAS/cuSOLVER + cuVS for network phase).
- Add a shared per-call execution policy in public C++ API:
  - `ComputeBackend = {cpu, gpu, auto}`
  - `device_id` (default `0`)
  - `allow_cpu_fallback` (default `true`)
- Keep existing function signatures; add overloads that accept execution policy.
- Runtime dispatch policy:
  - `cpu`: always CPU.
  - `auto`: use GPU only if build + runtime capability checks pass.
  - `gpu`: use GPU; if unsupported and `allow_cpu_fallback=true`, fallback to CPU; otherwise throw clear error.
- Add explicit runtime checks for CUDA runtime presence and compute capability >= 8.0.

2. Phase 1: SVD GPU (PRIMME-first, in-memory first)
- Scope:
  - GPU-enable **in-memory** PRIMME SVD path first (dense/sparse entry points).
  - Keep operator-backed `MatrixOperator` SVD on CPU in v1 with explicit fallback reason.
- Backend:
  - Use PRIMME GPU entrypoints (cublas path) behind execution policy.
  - Keep existing IRLB/Halko/Feng CPU implementations unchanged.
- API:
  - Add policy-enabled overloads for SVD calls used by kernel reduction path.
- Behavior:
  - If `algorithm=PRIMME` and backend is GPU-capable, use GPU path.
  - Otherwise preserve current CPU behavior.
- Acceptance gate:
  - SVD benchmark speedup >=1.5x on supported GPUs with parity tolerances for singular values/vectors.

3. Phase 2: Network GPU (cuVS + JSD semantic compatibility)
- Scope:
  - Add GPU backend for `knn` and `k*nn`.
  - Support `l2`, `ip`, and `jsd` in v1 GPU path.
- `l2`/`ip`:
  - Use cuVS ANN for candidate generation/query.
- `jsd`:
  - Match current CPU semantics (row clamp+normalize, current JSD behavior, and current graph weighting/symmetrization logic).
  - Implement two-stage GPU path:
    - candidate generation on GPU ANN backend,
    - exact JSD rerank kernel matching CPU semantics as closely as possible.
  - If quality gates fail for a run, auto-fallback to CPU path.
- API:
  - Extend network parameter surface with backend/policy fields and GPU tuning knobs needed for candidate pool/rerank.
- Acceptance gate:
  - End-to-end network build >=1.5x with graph-quality and parity gates.

4. Phase 3: ACTION GPU (`AA` primary, `SPA` completed by phase end)
- Scope/order inside phase:
  - Implement GPU `AA` first, then GPU `SPA`, both delivered before phase close.
- `AA`:
  - Use a GPU-native simplex regression solver with parity gates (not strict active-set port).
  - Keep CPU active-set as fallback/reference.
  - Use cuBLAS/cuSOLVER primitives for dense updates where applicable.
- `SPA`:
  - Implement GPU primitives for column norm/max selection and orthogonalization steps.
  - Preserve CPU tie-break/selection semantics.
- Orchestration:
  - `runACTION` gets policy-enabled overload; CPU path remains default/unchanged.
- Acceptance gate:
  - End-to-end `runACTION` >=1.3x with output parity checks.

### Test Plan
- Add backend-dispatch tests for CPU/GPU/auto/fallback/error behavior.
- Add numerical parity tests:
  - SVD: singular values + subspace agreement.
  - Network: degree/nnz invariants, recall/quality checks, and `jsd` compatibility checks.
  - ACTION: `C/H` output tolerances and archetype assignment stability checks.
- Add failure-mode tests:
  - No CUDA runtime, wrong driver/runtime, unsupported GPU capability, forced GPU without fallback.
- Add benchmark harness for phase gates on representative datasets and publish pass/fail criteria.
- Add HPC smoke profile (CUDA 12.2 + H100 class) and local Ampere/Ada smoke profile.

### Assumptions and Defaults
- NVIDIA-only in v1; Apple Silicon deferred.
- Ampere+ is required for all GPU features in this plan.
- CPU remains default in all existing front-end flows.
- R-facing GPU enablement is deferred; this plan only adds C++ hooks for future bindings.
- Bounded custom CUDA is allowed only where library APIs cannot preserve required behavior (`jsd` rerank, selective SPA primitives).
- Existing third-party drop-ins remain untouched as source; integration is via build configuration and adapter layers.
