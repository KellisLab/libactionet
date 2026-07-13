# NVIDIA GPU Backend for `libactionet` (v2 roadmap)

Status: active roadmap, reviewed 2026-07-12.

This document supersedes the earlier PRIMME-centered GPU roadmap. That older
plan was too implementation-specific and now contradicts the current SVD
strategy. This version records only durable goals, constraints, and design
direction. Concrete implementation details should live in focused spike notes
or implementation plans once they have been validated on real hardware.

## Background

A previous GPU attempt targeted cuBLAS-backed PRIMME SVD. It compiled and
passed basic runtime probes, but failed immediately on real GPU hardware. The
failure was structural: the wrapper design did not satisfy the device-memory
contract expected by PRIMME's GPU entry points.

The lesson is not merely "fix PRIMME." The larger lesson is that GPU support
must be designed around explicit CPU/device ownership, streaming boundaries,
fallback behavior, and hardware validation from the start.

PRIMME is no longer the intended GPU foundation.

## Primary Goals

- Add NVIDIA GPU acceleration without changing CPU behavior by default.
- Treat disk-backed data as a first-class GPU target from the beginning.
- Use one coherent SVD strategy across in-memory and backed data rather than
  adding a separate method for every storage/backend combination.
- Preserve realistic 64-bit sparse CPU support without relying on PRIMME.
- Keep the build usable in HPC-style conda environments without containers.
- Keep macOS and CPU-only Linux builds clean when GPU support is disabled.
- Prefer CUDA toolkit primitives and minimal dependencies unless a higher-level
  library clearly pays for its build and portability cost.

## Platform Constraints

These constraints are part of the current implementation contract:

- Linux x86_64 with NVIDIA GPUs is the production/runtime target.
- Linux x86_64 without a GPU must build and behave like the CPU-only baseline.
- Windows 11 + WSL2 with an NVIDIA GPU is the required developer test
  environment for GPU branches.
- macOS remains CPU-only and is the primary non-CUDA regression target.
- Native Windows is out of scope.
- CUDA 12.2 is the minimum CUDA toolkit target. Do not add a CUDA 11.x fallback.
- Supported GPUs are SM 8.0 (Ampere) or newer.
- Default CUDA architecture lists should cover Ampere, Ada, and Hopper
  (`80;86;89;90`) unless a build intentionally narrows the list for a specific
  deployment. Such narrowing does not lower the supported hardware floor.
- GPU support is optional at build time and disabled by default.
- R-facing GPU support is deferred; Python is the first supported front-end.

CUDA versions newer than 12.2 may be used by developers, but they are not a
project recommendation or support floor until validated on the target
Linux/WSL2 hardware and recorded here.

## Runtime Contract

GPU-capable entry points should expose a small backend policy:

- requested backend: automatic, CPU, or CUDA.
- device selection: CUDA device ordinal when relevant.
- fallback policy: whether unavailable or failed GPU execution may fall back
  to CPU.

The exact C++ names and wrapper details may change, but the contract should
remain:

- CPU execution is the baseline and must remain available.
- Automatic mode may choose GPU only when build, runtime, device, and a real
  execution canary pass.
- Forced GPU mode should either run on GPU or fail clearly unless fallback is
  explicitly allowed.
- Results should record the resolved backend so benchmarks and user reports
  are auditable.

## SVD Direction

SVD is the first GPU target, but the GPU SVD design should not be limited to
in-memory matrices. Disk-backed SVD is a primary use case.

Current direction:

- Use Halko-style randomized SVD as the shared scalable SVD family.
- Keep in-memory and disk-backed inputs on the same algorithmic path wherever
  possible.
- Implement CPU/GPU differences at the matrix-product and data-streaming
  layer, not as separate public SVD methods.
- Do not use PRIMME as the GPU vehicle.
- Keep PRIMME and Feng out of the Python SVD surface. Their C++ code has been
  fully deleted and must not be reintroduced.

Open choices:

- The first production implementation should start from CUDA toolkit primitives
  and the package-native product/streaming boundary. RAFT/RAPIDS may be
  evaluated later as an optional spike or reference backend, but it is not a
  default dependency.
- How much of the backed-data transform pipeline should run on CPU versus GPU.

See the Python-front-end launchpad at
`../../../plans/GPU_BACKED_SVD_AGENT_LAUNCHPAD.md` for the current SVD discussion.

## Disk-backed GPU Principle

The existing CPU `MatrixOperator` interface is a useful CPU abstraction, but it
must not become the GPU performance boundary. A GPU-backed path should avoid
running CPU matrix products and then copying their results to the GPU.

Instead, disk-backed GPU support should be planned around an explicit streaming
boundary:

- HDF5-backed data is read in bounded chunks.
- Existing lazy-transform semantics are preserved.
- Chunks are transferred to device memory in a controlled way.
- Matrix products accumulate into GPU-resident sketch/workspace buffers when
  possible.
- The same high-level randomized SVD driver can use either CPU products or GPU
  products.

The precise chunk representation, buffering strategy, and transform placement
are implementation details to be decided by spikes and benchmarks.

## Dependency Policy

Dependencies should be added only when they materially reduce complexity or
risk.

Preference order:

1. CUDA toolkit libraries that are commonly available in target HPC
   environments.
2. Optional higher-level NVIDIA/RAPIDS libraries behind explicit build flags.
3. Heavy framework dependencies only if a spike proves they are worth the
   portability and maintenance cost.

Any dependency that requires raising the C++ or CUDA language standard must be
treated as an architectural decision, not an incidental implementation detail.

## Validation Requirements

GPU work is not considered viable until it is validated on real hardware.

Required validation themes:

- CPU-only builds remain unchanged.
- macOS builds remain CPU-only and green.
- GPU-enabled builds fail clearly when CUDA support is unavailable or
  unsupported.
- Runtime canaries exercise an actual small GPU computation, not just device
  discovery.
- Numerical parity is checked against CPU baselines.
- Disk-backed benchmarks report wall time, host memory, GPU memory, and data
  transfer behavior.
- Fallback behavior is tested for unavailable GPUs and GPU-path runtime
  failures.

Benchmarks should include both in-memory and disk-backed datasets. Disk-backed
performance is a primary acceptance criterion, not a later optimization.

## Phase Outline

This is intentionally high-level. Detailed phase plans should be written only
when the relevant design choices are settled.

1. Clean up SVD strategy.
   - Completed for Python: public SVD algorithms are `auto`, `irlb`, and
     `halko`; PRIMME/Feng have been fully deleted from the C++ core and no
     hidden backed IRLB dispatch remains.

2. Define the shared SVD/product abstraction.
   - Keep the randomized SVD algorithm independent of storage and execution
     backend.
   - Preserve existing CPU behavior while introducing the abstraction.

3. Establish GPU build/runtime policy.
   - Add optional GPU build plumbing.
   - Add runtime capability checks, canaries, errors, and fallback behavior.
   - Keep CPU-only builds clean by default.

4. Prototype GPU SVD product backends.
   - Cover in-memory and disk-backed data in the same design effort.
   - Implement the native CUDA path first; evaluate optional higher-level
     libraries only after the product and chunk-stream boundaries exist.
   - Validate on real hardware before committing to public API behavior.

5. Harden and expose.
   - Add parity tests, benchmarks, and CI/smoke coverage.
   - Record resolved backend metadata in user-visible results.
   - Document supported platforms, limitations, and fallback semantics.

## Out of Scope for v1

- Apple Silicon GPU support.
- Native Windows GPU support.
- Multi-GPU scheduling.
- Distributed GPU execution.
- R-facing GPU API.
- PRIMME GPU integration.
- Committing to RAPIDS/RAFT as a required dependency before a validated spike.

## Change Control

This roadmap should remain concise. Avoid adding low-level implementation
details until they have been validated and are unlikely to mislead future
agents. If a spike settles a design choice, record the decision in the relevant
decision log and update this document only at the level of stable direction.
