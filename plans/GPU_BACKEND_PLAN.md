# NVIDIA GPU Backend for `libactionet` (roadmap)

## Status

**Not started.** A prior attempt (`dev-gpu-v2` branch) implemented Phase 1
plumbing (`ExecutionPolicy`, `ComputeBackend` enum, cuBLAS-backed PRIMME
SVD dispatch, CMake `LIBACTIONET_ENABLE_NVIDIA_GPU` option) without
access to real hardware. When the branch was finally exercised on
WSL2 + RTX 4000 Ada + CUDA 13.2 in July 2026 it segfaulted immediately
on every `compute_backend="gpu"` call. The failure was structural, not
a small fix (see section 6 "Post-mortem"), and the branch was scrapped
in favor of a clean re-attempt on hardware.

| Phase | Scope | Status |
|---|---|---|
| 1 | SVD on GPU (in-memory dense + sparse via cuBLAS-backed PRIMME) | Not started |
| 1-hardening | Error taxonomy, benchmark harness, CI commitments | Not started |
| 2 | Network construction on GPU (cuVS L2/IP + bespoke JSD rerank) | Planned |
| 3 | ACTION (`AA` + `SPA`) on GPU | Planned |
| 4 | R-facing GPU surface | Deferred indefinitely |

This document is the durable roadmap. The settled meta-decisions (CUDA
floor, compute-capability floor, platform contract, R deferral,
operator-mode CPU-only) survive the reset and are enumerated in
[`DECISIONS.md`](DECISIONS.md).

## 1. Platform support contract

These constraints are non-negotiable.

| Platform | GPU support | Build status | CI guarantee |
|---|---|---|---|
| Linux x86_64 (HPC, conda, no Docker) | Yes — primary target | `LIBACTIONET_ENABLE_NVIDIA_GPU=ON` with CUDA >= 12.2 + Ampere+ device | Smoke-build job against CUDA 12.2 stub libs (link-only, no test execution) |
| Linux x86_64 (no GPU) | N/A | `LIBACTIONET_ENABLE_NVIDIA_GPU=OFF` (default) builds and runs identically to `dev` | CPU-only build + full test suite, required check |
| Windows 11 + WSL2 (CUDA 13.2 dev box) | Yes — primary test environment | Same as Linux GPU, with WSL2-specific gotchas documented (section 4) | Manual run on the dev WSL2 machine before each push to the GPU branch |
| macOS (Apple Silicon, Intel) | No, ever | `LIBACTIONET_ENABLE_NVIDIA_GPU=OFF` enforced; option silently ignored if accidentally set | Existing macOS CI must keep passing throughout the GPU rollout — primary regression guard |
| Windows native (no WSL2) | Out of scope | Not supported, not tested | None |

### CUDA version policy

- **Floor: CUDA 12.2.** Intersection of (a) oldest CUDA available on common HPC sites that is also Ampere-compatible, and (b) minimum cuVS / RAPIDS requirement.
- **No fallback to 11.x.** cuVS / RAPIDS hard-require >= 12.2; the "fall back to CUDA 11.8" route is explicitly rejected.
- **Develop on 13.2, target 12.2.** The dev WSL2 box runs CUDA 13.2. All new GPU code must compile against the 12.2 API surface; no use of 13.x-only symbols. The smoke-build CI must point at a CUDA 12.2 stub layout to enforce this.

### Compute capability policy

- **Floor: SM 8.0 (Ampere).** Hard-coded. No CMake override, no environment variable, no runtime flag.
- Phase 2 cuVS work makes Ampere a hard requirement, not a soft preference.

### macOS regression guard

Every commit on the future GPU branch must keep `pip install .` working
on macOS without CUDA, with `LIBACTIONET_ENABLE_NVIDIA_GPU=OFF` (the
default). The existing macOS CI job is the primary regression guard.
The CMake macro structure introduced on the old `dev-gpu` branch
(commit `abc3e24`: the `if/else/return` shape) is what made this
possible and should be reused in any re-implementation.

## 2. Build configuration (target)

```cmake
option(LIBACTIONET_ENABLE_NVIDIA_GPU "Enable NVIDIA GPU backend (CUDA >= 12.2, Ampere+)" OFF)
```

When `OFF` (default), nothing CUDA-related is included, linked, or
referenced. The `actionet` target builds identically to `dev`.

When `ON`, a `cmake/ConfigureNvidiaGPU.cmake` module should:

1. Call `find_package(CUDAToolkit 12.2 REQUIRED)`.
2. Enable the `CUDA` language and set `CMAKE_CUDA_ARCHITECTURES` to `80;86;89;90` (Ampere + Ada + Hopper). *(This was documented as a target on `dev-gpu-v2` but not actually implemented — plan-vs-code drift; fix on re-attempt.)*
3. Add `LIBACTIONET_ENABLE_NVIDIA_GPU` (and, if PRIMME is used, `PRIMME_WITH_CUBLAS`) compile definitions to the `actionet` target.
4. Link `CUDA::cudart`, `CUDA::cublas`, `CUDA::cusolver`, plus any algorithm-specific libraries as they are introduced (cuVS in Phase 2).
5. **Refuse** simultaneous `LIBACTIONET_BUILD_R + LIBACTIONET_ENABLE_NVIDIA_GPU` with a `FATAL_ERROR` (R deferral).

The macro `CONFIGURE_NVIDIA_GPU(target_name)` should be invoked from
the top-level `CMakeLists.txt` after the target is created, and must
`return()` early when `LIBACTIONET_ENABLE_NVIDIA_GPU=OFF` while the
surrounding `if/else` block in the top-level CMakeLists routes around
any GPU-only steps. Both paths must exit cleanly — that is what keeps
macOS green.

## 3. Runtime API: `ExecutionPolicy` (target)

```cpp
namespace actionet {

enum class ComputeBackend : int { cpu = 0, gpu = 1, automatic = 2 };

struct ExecutionPolicy {
    ComputeBackend backend = ComputeBackend::automatic;
    int device_id = 0;
    bool allow_cpu_fallback = true;
};

bool isNvidiaGpuBackendCompiled() noexcept;
std::pair<bool, std::string> isCudaRuntimeAvailable() noexcept;
std::pair<bool, std::string> isCudaDeviceSupported(int device_id) noexcept;
std::pair<bool, std::string> canUseNvidiaGpu(int device_id) noexcept;

}  // namespace actionet
```

Intended home: `include/decomposition/compute_backend.hpp` /
`src/decomposition/compute_backend.cpp`; re-exported via the umbrella
`include/libactionet.hpp`.

### Dispatch rules

| Policy backend | Build has GPU | Runtime/device usable | Action |
|---|---|---|---|
| `cpu` | any | any | run CPU path unchanged |
| `automatic` | no | n/a | run CPU path |
| `automatic` | yes | no | run CPU path |
| `automatic` | yes | yes | run GPU path |
| `gpu` | no | n/a | `allow_cpu_fallback?` CPU path : throw |
| `gpu` | yes | no | `allow_cpu_fallback?` CPU path : throw |
| `gpu` | yes | yes | run GPU path |

Additive, ABI-compatible policy-aware overloads on every entry point
that may eventually support GPU execution. Existing zero-policy
overloads forward to `ExecutionPolicy{}` (i.e. `automatic` / device 0 /
fallback on).

### Error taxonomy (target)

Add `include/util/gpu_error.hpp`:

```cpp
namespace actionet {
class GpuError : public std::runtime_error { using std::runtime_error::runtime_error; };
class GpuUnavailableError : public GpuError { using GpuError::GpuError; };
class GpuRuntimeError    : public GpuError { using GpuError::GpuError; };
}  // namespace actionet
```

`GpuUnavailableError` for "GPU not usable on this build/host, retrying
with `cpu` will succeed"; `GpuRuntimeError` for "an operation was
attempted on the GPU and failed at runtime, e.g. cuBLAS / cuSOLVER /
OOM / kernel error". Pybind11 maps them to `RuntimeError` subclasses
on the Python side.

## 4. WSL2 quirks (must read before benchmarking)

- **Pinned-host memory:** WSL2's GPU paravirtualization changes the semantics of `cudaHostAlloc(cudaHostAllocPortable | cudaHostAllocMapped)`. Test explicitly on the dev box; if performance regresses vs. plain `cudaMalloc` + `cudaMemcpyAsync`, document and prefer the latter.
- **VRAM ceiling:** The WSL2 driver imposes its own VRAM ceiling that may be lower than the physical card's. **Never trust `nvidia-smi` for available memory inside WSL2.** Always call `cudaMemGetInfo` at runtime.
- **Driver clock policy:** WSL2 driver may report different clock policies than native Linux; benchmarks comparing CPU vs GPU should be run on the same OS, not Linux-vs-WSL2.

## 5. Phase specifics (targets)

### Phase 1: SVD

- GPU-enable **in-memory** PRIMME SVD path first (dense + sparse entry points) via PRIMME's vendored `cublas_dprimme_svds` if PRIMME remains the algorithm choice — **but see section 6.** The vendored cuBLAS PRIMME path has a device-pointer contract that the CPU-style call sites in `svd_primme.cpp` cannot satisfy without a substantial rewrite. The re-attempt must decide, on hardware, between: (a) doing that rewrite (device-resident svals/svecs/rnorms + device matvec), (b) picking a different SVD algorithm on GPU (cuSOLVER-based Halko, for example), or (c) walking away from GPU SVD.
- Keep operator-backed `MatrixOperator` SVD on CPU. CPU-side matvec callbacks force per-iteration H2D/D2H round trips that dominate any cuBLAS speedup. This is a settled decision, not a Phase-1 defer.
- Acceptance gate: SVD benchmark speedup >= 1.5x on supported GPUs with parity tolerances for singular values / subspace agreement.

### Phase 1 hardening

Must precede Phase 2:

- **Benchmark harness** (`tests/benchmark_gpu_svd.py` in the Python front-end) reporting wall time, peak RSS, GPU memory peak (via `cudaMemGetInfo`), singular-value correlation vs CPU, reconstruction error, on a fixed small/medium/large dense+sparse+operator-backed matrix. Without this the `>= 1.5x` gate is unfalsifiable.
- **CI commitments:** macOS no-GPU (required), Linux CPU-only (required), Linux GPU smoke-build against CUDA 12.2 stubs (link-only, required), plus manual WSL2 checklist before each push.

### Phase 2: Network construction

- `l2` / `ip`: cuVS ANN for candidate generation / query.
- `jsd`: two-stage path — Stage 1 cuVS L2/IP top-K to get a candidate pool of size `k * pool_multiplier` (default `4`, exposed as tuning knob); Stage 2 exact JSD rerank kernel matching the CPU `compute_jsd` semantics.
- The JSD kernel must be written as a numerical-parity unit test against the CPU path **before** integrating it into `build_network`.
- If quality gates fail at runtime, auto-fallback to CPU per the policy contract.
- Acceptance gate: end-to-end network build >= 1.5x with graph-quality and parity gates.

### Phase 3: ACTION (`AA` + `SPA`)

- **`AA`:** projected-gradient with Frank-Wolfe corrections is the algorithmic choice for v1. Well-suited to GPU dense BLAS, no active-set bookkeeping to port. The CPU active-set solver remains the reference; a parity-tolerance test compares the two. The `>= 1.3x` speedup gate is conditional on this algorithm.
- **`SPA`:** GPU primitives for column norm / max selection and orthogonalization, preserving CPU tie-break / selection semantics.
- Acceptance gate: end-to-end `runACTION` >= 1.3x with `C`/`H` output tolerances and archetype assignment stability checks.

### Memory budget

Every Phase 2/3 GPU entry point accepts an explicit
`gpu_workspace_bytes` parameter:

- Default: 80% of free VRAM at call entry, queried via `cudaMemGetInfo` (required because WSL2 reports a different ceiling than `nvidia-smi`).
- On exceeding budget: fall back per `ExecutionPolicy.allow_cpu_fallback`.
- Pinned-host vs unified memory: prefer pinned-host on native Linux; **verify on WSL2 per section 4 before committing to it there.**

### R deferral

Keep the `FATAL_ERROR` on `LIBACTIONET_BUILD_R +
LIBACTIONET_ENABLE_NVIDIA_GPU` for v1. Soften to a `STATUS` skip only
after the unicore `R_xlen_t` work lands and someone explicitly asks
for R-side GPU access. Phase 4 item, not Phase 1.

### Multi-GPU / multi-process

Out of scope through Phase 3. Per-call `device_id` is sufficient for
the foreseeable HPC workflow.

## 6. Post-mortem: why the first attempt was scrapped

The `dev-gpu-v2` implementation of Phase 1 (`libactionet` `0a1e938`,
`actionet-python` `0b455a0`) was written without hardware access. It
compiled and passed the runtime probes on the WSL2 dev box, but every
`compute_backend="gpu"` PRIMME SVD call segfaulted immediately, before
the matvec callback was ever reached.

**Structural cause: device-vs-host pointer contract violation** in the
call to PRIMME's `cublas_dprimme_svds`. The `USE_DOUBLE_CUBLAS`
template variant of PRIMME defines every `SCALAR*` as a CUDA **device**
pointer — see PRIMME's own header comment at
`src/extern/primme/include/template.h` lines 44-51:

> When SCALAR is a GPU type, the pointers SCALAR* are supposed to point
> out memory allocated on GPUs [...] Use HSCALAR and HREAL as the
> non-GPU, also called host, versions of SCALAR and REAL.

The wrapper in `src/decomposition/svd_primme.cpp` violated this in
three places simultaneously:

1. `svals` / `svecs` / `rnorms` were passed as host `std::vector<double>`. PRIMME called `Num_matrix_astype_Sprimme` which short-circuits when the input/output types match (`PRIMME_OP_SCALAR` in both) and simply aliases the pointer, then ran device-side cuBLAS `dgemm` / `dgemv` on host memory.
2. `primme_svds.matrixMatvec_type` was never set in `runPrimmeCore`. It defaulted to `primme_op_default` -> `PRIMME_OP_SCALAR` -> device pointers -> the matvec callback received device pointers, wrapped them in `arma::vec`, and did host CPU matmul.
3. `runPrimmeCore` had no `cudaMemcpy` back to host after the call because there was no device buffer to copy from in the first place.

Additionally, on any GPU-visible machine `compute_backend="gpu"` with
`allow_cpu_fallback=True` **did not** fall back, because the runtime
probes succeeded and the dispatch entered the broken path anyway.
`compute_backend="auto"` on the same box resolved to the same broken
path.

**Lessons for the re-attempt:**

- Do not build GPU dispatch on top of a vendored library's
  device-pointer entry point unless the wrapper actually allocates and
  passes device memory end to end. Prototype the smallest possible
  end-to-end call on hardware before writing the rest.
- The `allow_cpu_fallback` semantics must include "the GPU path itself
  crashed" as a fallback trigger, not only "probes said the GPU is
  unusable". Consider a runtime canary (small dummy solve at startup)
  or a signal-safe wrapper.
- Any GPU-marked test that is not gated by `@requires_gpu` will crash
  on any real GPU box, not just under `-m gpu`. The three
  `_fallback`-suffixed tests in the deleted
  `actionet-python tests/test_gpu_backend_policy.py` had this problem.
  Gate strictly.
- Plan-vs-code drift crept in fast without hardware feedback: the
  section-2 target of "enable CUDA language, set
  CMAKE_CUDA_ARCHITECTURES" was documented but never implemented in
  `cmake/ConfigureNvidiaGPU.cmake`. It was harmless only because no
  `.cu` sources were introduced. Re-attempt should either enforce the
  target or delete it from the plan.

## 7. Decisions log

Settled. Re-opening any of them requires updating `DECISIONS.md` with
rationale.

- CUDA floor is fixed at 12.2 (no 11.x fallback).
- Compute capability floor is fixed at 8.0 (Ampere, no override).
- Apple Silicon GPU support is permanently out of scope.
- R-facing GPU is deferred, hard `FATAL_ERROR` while combined with R build.
- Operator-backed PRIMME stays CPU-only (CPU-side matvec callbacks dominate).
- Multi-GPU / multi-process is out of scope through Phase 3.
- Phase ordering: SVD -> Network -> ACTION; hardening between Phase 1 and Phase 2.
- First implementation attempt (`dev-gpu-v2`) was scrapped on hardware sign-off. The re-attempt starts from `dev`, with all lessons in section 6 folded into the initial design.
