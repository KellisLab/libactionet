"""
Plan 02 orientation validation script.

Verifies that the C++ core outputs have the correct AnnData-native shapes
after the contract flip:
  S   : cells x genes  (input)
  S_r : cells x k      (output of reduceKernel)
  H   : cells x archetypes (output of runACTION)
  C   : cells x archetypes (output, unchanged)
  Obs : genes x k      (output of computeFeatureSpecificity)

This script requires actionet-python (with Plan 04 applied) or the R bindings
(with Plan 03 applied) to be installed and callable.

Usage (Python path, after Plan 04):
    python test/validate_plan02_orientation.py --mode python

Usage (R path, after Plan 03):
    python test/validate_plan02_orientation.py --mode r

NOTE: Until Plans 03 and 04 are applied, calling this script will fail because
the language bindings still use the old orientation. That is expected.

For pre-03 core validation inside `libactionet`, use the repo-local
`validate_plan02_core` executable added by Plan 02A. This script is only a
post-frontend wrapper smoke test.

The fixture at test/fixtures/parity_fixture.h5ad contains the synthetic
parity dataset (100 cells, 200 genes, sparse).
"""

from __future__ import annotations
import argparse
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_H5AD = os.path.join(_HERE, "fixtures", "parity_fixture.h5ad")


def validate_python():
    """Validate Python binding shapes after Plan 04."""
    try:
        import anndata as ad
        import numpy as np
        import actionet
    except ImportError as e:
        print(f"SKIP: required package not available: {e}")
        return True

    print("Loading fixture:", FIXTURE_H5AD)
    adata = ad.read_h5ad(FIXTURE_H5AD)

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    k = 10

    print(f"  n_cells={n_cells}  n_genes={n_genes}  k={k}")
    print()

    # Stage 1: reduceKernel
    # S is cells x genes (AnnData X matrix in scipy sparse CSR)
    S = adata.X  # (cells x genes)
    print("Running reduceKernel...")
    reduction = actionet.reduce_kernel(adata, dim=k)

    S_r = reduction["S_r"]
    assert S_r.shape == (n_cells, k), (
        f"FAIL: S_r shape {S_r.shape} != expected ({n_cells}, {k})"
    )
    print(f"  PASS: S_r shape = {S_r.shape}  (cells x k)")

    U = reduction["U"]
    assert U.shape == (n_genes, k), (
        f"FAIL: U shape {U.shape} != expected ({n_genes}, {k})"
    )
    print(f"  PASS: U shape = {U.shape}  (genes x k)")

    # Stage 2: runACTION
    print("Running runACTION...")
    action_out = actionet.run_action(adata, k_min=2, k_max=k)

    H_stacked = action_out["H_stacked"]
    assert H_stacked.shape[0] == n_cells, (
        f"FAIL: H_stacked.shape[0]={H_stacked.shape[0]} != n_cells={n_cells}"
    )
    print(f"  PASS: H_stacked shape = {H_stacked.shape}  (cells x archetypes)")

    C_stacked = action_out["C_stacked"]
    assert C_stacked.shape[0] == n_cells, (
        f"FAIL: C_stacked.shape[0]={C_stacked.shape[0]} != n_cells={n_cells}"
    )
    print(f"  PASS: C_stacked shape = {C_stacked.shape}  (cells x archetypes)")

    # Stage 3: computeFeatureSpecificity
    print("Running computeFeatureSpecificity...")
    H = action_out["H_merged"]
    spec = actionet.compute_feature_specificity(adata, H)

    obs_profile = spec[0]
    assert obs_profile.shape[0] == n_genes, (
        f"FAIL: specificity profile shape[0]={obs_profile.shape[0]} != n_genes={n_genes}"
    )
    print(f"  PASS: specificity profile shape = {obs_profile.shape}  (genes x k)")

    print()
    print("All shape checks PASS (Python).")
    return True


def validate_r():
    """Validate R binding shapes after Plan 03 via rpy2."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
    except ImportError as e:
        print(f"SKIP: rpy2 not available: {e}")
        return True

    r = ro.r
    r("library(devtools)")
    actionet_r_path = os.path.join(_HERE, "..", "..", "actionet-r")
    if not os.path.isdir(actionet_r_path):
        print(f"SKIP: actionet-r not found at {actionet_r_path}")
        return True

    print(f"Loading actionet-r from {actionet_r_path}")
    r(f'devtools::load_all("{actionet_r_path}", quiet=TRUE)')
    r(f'fixture <- ACTIONetExperiment::readH5AD("{FIXTURE_H5AD}")')
    r("ace <- reduce(fixture, k=10)")

    S_r = r("reducedDim(ace, 'ACTION')")  # Should be cells x k after Plan 03
    import numpy as np
    S_r_arr = np.array(S_r)
    n_cells = int(r("nrow(ace)")[0])
    assert S_r_arr.shape[0] == n_cells, (
        f"FAIL: R reducedDim rows {S_r_arr.shape[0]} != n_cells {n_cells}"
    )
    print(f"  S_r shape from R: {S_r_arr.shape}  (expected cells x k)")

    print()
    print("All shape checks PASS (R).")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["python", "r", "both"], default="both")
    args = parser.parse_args(argv)

    ok = True
    if args.mode in ("python", "both"):
        ok = validate_python() and ok
    if args.mode in ("r", "both"):
        ok = validate_r() and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
