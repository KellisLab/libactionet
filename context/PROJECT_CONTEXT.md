# Project Context

## Overview

This work aims to develop a multi-language computational biology data analysis toolkit:

### ACTIONet software toolchain

- C++ core library: `KellisLab/libactionet`
- R front-end: `KellisLab/actionet-r` (Rcpp)
- Python front-end: `KellisLab/actionet-python` (pybind11)

## Dependency Graph

- `libactionet` is a core dependency for both `actionet-r` and `actionet-python`.
- Both front-end packages are typically used for interactive and iterative data processing and analysis,
- `actionet-python` is often used in non-permissive and headless HPC environments (e.g. conda, SGE, slurm) for data processing pipelines.

## Current Status

- All branches are usable but still in active development.
- libactionet dev branch is 88 commits ahead of main; merge imminent.
- Remaining work:
  - libactionet: GPU backend (planned, not started). Continue optimizations.
  - actionet-python: Optimize and port remaining core R functions.
  - actionet-r: Modernize and consolidate codebase. Improve build system. Fix bugs.
  - all: Document.

## Build/Binding Stack

- C++: CMake (≥ 3.19, C++17)
- R bindings: Rcpp
- Python bindings: pybind11
- Runtime dependencies: BLAS/LAPACK, HDF5 (C), OpenMP
- No containerization

## Key Principles (for humans + agents)

- Changes must preserve correctness and (where applicable) parity between R and Python front-ends.
- R front-end is currently more feature complete. Python version is higher performance.
- Prefer spec/contract-driven interfaces across repos (I/O schemas, parameter names, output formats).
- Assume all repos are present and readable on local machine.
