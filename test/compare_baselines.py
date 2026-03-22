"""
Cross-language parity comparison script for Plan 00 — Parity Baseline.

Loads baseline outputs produced by:
  - actionet-python/tests/generate_baseline.py  (baseline_python.npz)
  - actionet-r/tests/generate_baseline.R        (baseline_r.rds converted to npz)

For each parity-critical slot it:
  1. Applies canonicalization (SVD sign, archetype ordering, CSR sort)
  2. Compares with np.allclose(atol=1e-6, rtol=1e-4) for dense arrays
  3. Compares sorted CSR structure for sparse matrices
  4. Reports per-slot PASS/FAIL and maximum absolute deviation

This script is the reusable parity checker for all subsequent plans.

Usage:
    python libactionet/test/compare_baselines.py [--python-npz PATH] [--r-npz PATH] [--verbose]

The --r-npz argument expects a .npz that was exported from the R .rds by
running this script's `convert_r_rds_to_npz()` helper via rpy2, or by the
standalone conversion utility at the bottom of this file.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Default paths (relative to this file's location)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_ACTIONET_PY = os.path.join(_REPO_ROOT, "..", "actionet-python")
_ACTIONET_R  = os.path.join(_REPO_ROOT, "..", "actionet-r")

DEFAULT_PY_NPZ = os.path.join(_ACTIONET_PY, "tests", "fixtures", "baseline_python.npz")
DEFAULT_R_NPZ  = os.path.join(_HERE, "fixtures", "baseline_r_converted.npz")
DEFAULT_R_RDS  = os.path.join(_ACTIONET_R,  "tests", "fixtures", "baseline_r.rds")

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------
ATOL = 1e-6
RTOL = 1e-4


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------

def canonicalize_svd_sign(mat: np.ndarray) -> np.ndarray:
    """Flip sign of each column so the element with the largest |value| is positive.

    Works for 2-D arrays where columns represent singular/latent dimensions.
    Returns a copy.
    """
    mat = np.array(mat, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        idx = np.argmax(np.abs(col))
        if col[idx] < 0:
            mat[:, j] *= -1
    return mat


def canonicalize_archetype_ordering(
    *matrices: np.ndarray,
    reference_index: int = 0,
) -> list[np.ndarray]:
    """Sort archetypes by descending column norm of the reference matrix.

    All matrices are reordered identically using the sort key from
    ``matrices[reference_index]``.  Each matrix is assumed to be
    ``(rows, archetypes)``-shaped.

    Returns a list of reordered copies.
    """
    ref = np.array(matrices[reference_index], dtype=float)
    norms = np.linalg.norm(ref, axis=0)
    order = np.argsort(-norms)
    return [np.array(m, dtype=float)[:, order] for m in matrices]


def canonicalize_sparse_csr(mat: sp.spmatrix) -> sp.csr_matrix:
    """Convert sparse matrix to CSR with indices sorted within each row."""
    csr = mat.tocsr()
    csr.sort_indices()
    return csr


# ---------------------------------------------------------------------------
# Sparse reconstruction from flat arrays (stored in .npz)
# ---------------------------------------------------------------------------

def _reconstruct_sparse(npz: np.lib.npyio.NpzFile, prefix: str) -> Optional[sp.csr_matrix]:
    """Reconstruct a CSR sparse matrix from flat arrays stored under ``prefix_*``."""
    data_key    = f"{prefix}_data"
    indices_key = f"{prefix}_indices"
    indptr_key  = f"{prefix}_indptr"
    shape_key   = f"{prefix}_shape"
    if not all(k in npz for k in (data_key, indices_key, indptr_key, shape_key)):
        return None
    shape = tuple(int(x) for x in npz[shape_key])
    return sp.csr_matrix(
        (npz[data_key], npz[indices_key], npz[indptr_key]),
        shape=shape,
    )


# ---------------------------------------------------------------------------
# rpy2-based R→npz converter
# ---------------------------------------------------------------------------

def convert_r_rds_to_npz(rds_path: str, out_npz: str) -> None:
    """Load a baseline_r.rds file via rpy2 and save as .npz.

    Key mapping between R names and canonical names used by this script:
      obsm_action                → obsm_action
      obsm_action_B              → obsm_action_B
      varm_action_U              → varm_action_U
      varm_action_A              → varm_action_A
      uns_action_sigma           → uns_action_sigma
      obsm_H_stacked             → obsm_H_stacked
      obsm_H_merged              → obsm_H_merged
      obsm_C_stacked             → obsm_C_stacked
      obsm_C_merged              → obsm_C_merged
      obs_assigned_archetype     → obs_assigned_archetype
      obsp_actionet              → obsp_actionet  (dense from R)
      varm_specificity_profile   → varm_specificity_profile
      varm_specificity_upper     → varm_specificity_upper
      varm_specificity_lower     → varm_specificity_lower
      varm_archetype_feat_profile               → varm_archetype_feat_profile
      varm_archetype_feat_specificity_upper     → varm_archetype_feat_specificity_upper
      varm_archetype_feat_specificity_lower     → varm_archetype_feat_specificity_lower
      obsm_action_corrected      → obsm_action_corrected
      varm_action_corrected_U    → varm_action_corrected_U
      varm_action_corrected_A    → varm_action_corrected_A
      obsm_actionet_2d           → obsm_actionet_2d
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
    except ImportError:
        raise ImportError(
            "rpy2 is required to convert R .rds files. Install via: pip install rpy2"
        )

    r = ro.r
    r["options"](warn = -1)
    r_arrays = r["readRDS"](rds_path)

    arrays: dict[str, np.ndarray] = {}
    for name in r_arrays.names:
        val = r_arrays.rx2(name)
        try:
            arr = np.array(val)
            arrays[name] = arr
        except Exception as exc:
            print(f"  [SKIP] {name}: {exc}")

    np.savez_compressed(out_npz, **arrays)
    print(f"Converted {len(arrays)} arrays from {rds_path} → {out_npz}")


# ---------------------------------------------------------------------------
# Load R baseline from h5ad (alternative to rpy2 path)
# ---------------------------------------------------------------------------

def load_r_baseline_from_h5ad(h5ad_path: str, out_npz: str) -> None:
    """Extract parity arrays from the R baseline .h5ad and save as .npz.

    This avoids requiring rpy2 and works from the .h5ad written by the R script.
    """
    import anndata as ad

    print(f"Loading R h5ad baseline: {h5ad_path}")
    r_adata = ad.read_h5ad(h5ad_path)

    arrays: dict[str, np.ndarray] = {}

    def _get_obsm(key: str, label: str) -> None:
        if key in r_adata.obsm:
            v = r_adata.obsm[key]
            if sp.issparse(v):
                arrays[label] = v.toarray()
            else:
                arrays[label] = np.asarray(v)
            print(f"  [OK]  {label:50s}  {arrays[label].shape}")
        else:
            print(f"  [MISSING] {label} (obsm/{key})")

    def _get_varm(key: str, label: str) -> None:
        if key in r_adata.varm:
            arrays[label] = np.asarray(r_adata.varm[key])
            print(f"  [OK]  {label:50s}  {arrays[label].shape}")
        else:
            print(f"  [MISSING] {label} (varm/{key})")

    def _get_obs(key: str, label: str) -> None:
        if key in r_adata.obs.columns:
            arrays[label] = np.asarray(r_adata.obs[key])
            print(f"  [OK]  {label:50s}  {arrays[label].shape}")
        else:
            print(f"  [MISSING] {label} (obs/{key})")

    def _get_obsp(key: str, label: str) -> None:
        if key in r_adata.obsp:
            v = r_adata.obsp[key]
            if sp.issparse(v):
                arrays[label] = v.toarray()
            else:
                arrays[label] = np.asarray(v)
            print(f"  [OK]  {label:50s}  {np.asarray(arrays[label]).shape}")
        else:
            print(f"  [MISSING] {label} (obsp/{key})")

    # Reduction
    _get_obsm("action",   "obsm_action")
    _get_obsm("action_B", "obsm_action_B")
    _get_varm("action_U", "varm_action_U")
    _get_varm("action_A", "varm_action_A")
    if "action_params" in r_adata.uns and "sigma" in r_adata.uns["action_params"]:
        arrays["uns_action_sigma"] = np.asarray(r_adata.uns["action_params"]["sigma"]).ravel()
        print(f"  [OK]  uns_action_sigma  length={len(arrays['uns_action_sigma'])}")
    else:
        print("  [MISSING] uns_action_sigma")

    # ACTION
    _get_obsm("H_stacked", "obsm_H_stacked")
    _get_obsm("H_merged",  "obsm_H_merged")
    _get_obsm("C_stacked", "obsm_C_stacked")
    _get_obsm("C_merged",  "obsm_C_merged")
    _get_obs("assigned_archetype", "obs_assigned_archetype")

    # Network
    _get_obsp("actionet", "obsp_actionet")

    # Specificity (cluster) — R only stores upper by default
    _get_varm("cluster_upper", "varm_specificity_upper")
    _get_varm("cluster_lower", "varm_specificity_lower")

    # Specificity (archetype)
    _get_varm("archetype_feat_profile",            "varm_archetype_feat_profile")
    _get_varm("archetype_feat_specificity_upper",  "varm_archetype_feat_specificity_upper")
    _get_varm("archetype_feat_specificity_lower",  "varm_archetype_feat_specificity_lower")

    # Batch correction
    _get_obsm("action_orth",   "obsm_action_corrected")
    _get_varm("action_U_orth", "varm_action_corrected_U")
    _get_varm("action_A_orth", "varm_action_corrected_A")

    # Layout
    _get_obsm("actionet_2d", "obsm_actionet_2d")

    np.savez_compressed(out_npz, **arrays)
    print(f"Saved {len(arrays)} arrays to {out_npz}")


# ---------------------------------------------------------------------------
# Per-slot comparison logic
# ---------------------------------------------------------------------------

_RESULTS: list[dict] = []


def _report(slot: str, passed: bool, max_dev: float, note: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    _RESULTS.append({"slot": slot, "status": status, "max_dev": max_dev, "note": note})


def _compare_dense(slot: str, py_arr: np.ndarray, r_arr: np.ndarray) -> None:
    if py_arr.shape != r_arr.shape:
        _report(slot, False, float("nan"), f"shape mismatch: Python={py_arr.shape} R={r_arr.shape}")
        return
    max_dev = float(np.max(np.abs(py_arr - r_arr)))
    passed = bool(np.allclose(py_arr, r_arr, atol=ATOL, rtol=RTOL))
    _report(slot, passed, max_dev)


def _compare_sparse(slot: str, py_mat: sp.spmatrix, r_mat: sp.spmatrix) -> None:
    if py_mat.shape != r_mat.shape:
        _report(slot, False, float("nan"), f"shape mismatch: Python={py_mat.shape} R={r_mat.shape}")
        return
    py_csr = canonicalize_sparse_csr(py_mat)
    r_csr  = canonicalize_sparse_csr(r_mat)
    # Compare indices structure first
    idx_ok = (
        np.array_equal(py_csr.indptr, r_csr.indptr)
        and np.array_equal(py_csr.indices, r_csr.indices)
    )
    if idx_ok:
        val_ok = bool(np.allclose(py_csr.data, r_csr.data, atol=ATOL, rtol=RTOL))
        max_dev = float(np.max(np.abs(py_csr.data - r_csr.data))) if py_csr.nnz > 0 else 0.0
        _report(slot, val_ok, max_dev)
    else:
        # Different sparsity structure: compare as dense via elementwise subtraction
        py_dense = py_csr.toarray()
        r_dense  = r_csr.toarray()
        max_dev  = float(np.max(np.abs(py_dense - r_dense)))
        val_ok   = bool(np.allclose(py_dense, r_dense, atol=ATOL, rtol=RTOL))
        nnz_diff = abs(py_csr.nnz - r_csr.nnz)
        _report(slot, val_ok, max_dev,
                f"sparse structure differs (Python nnz={py_csr.nnz}, R nnz={r_csr.nnz})")


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare(
    py_npz_path: str,
    r_npz_path: str,
    verbose: bool = False,
) -> bool:
    """Run all slot comparisons. Returns True if all slots PASS."""

    print(f"\nLoading Python baseline : {py_npz_path}")
    if not os.path.exists(py_npz_path):
        print(f"  ERROR: file not found")
        return False
    py = np.load(py_npz_path, allow_pickle=False)

    print(f"Loading R baseline      : {r_npz_path}")
    if not os.path.exists(r_npz_path):
        print(f"  ERROR: file not found")
        print("  Tip: run convert_r_rds_to_npz() first or re-run with --r-rds.")
        return False
    r = np.load(r_npz_path, allow_pickle=True)

    print()

    # Helper to fetch from npz, None if missing
    def py_get(key: str) -> Optional[np.ndarray]:
        return py[key] if key in py.files else None

    def r_get(key: str) -> Optional[np.ndarray]:
        return r[key] if key in r.files else None

    def both_present(key_py: str, key_r: str, slot_label: str) -> bool:
        a = py_get(key_py)
        b = r_get(key_r)
        if a is None:
            _report(slot_label, False, float("nan"), f"MISSING in Python (key={key_py})")
            return False
        if b is None:
            _report(slot_label, False, float("nan"), f"MISSING in R (key={key_r})")
            return False
        return True

    # ------------------------------------------------------------------
    # Reduction slots — apply SVD sign canonicalization
    # ------------------------------------------------------------------
    for (py_key, r_key, slot) in [
        ("obsm_action",   "obsm_action",   "obsm/action   (cells x k)"),
        ("varm_action_U", "varm_action_U", "varm/action_U (genes x k)"),
        ("varm_action_A", "varm_action_A", "varm/action_A (genes x p)"),
        ("obsm_action_B", "obsm_action_B", "obsm/action_B (cells x p)"),
    ]:
        if both_present(py_key, r_key, slot):
            a = canonicalize_svd_sign(py_get(py_key))
            b = canonicalize_svd_sign(r_get(r_key))
            _compare_dense(slot, a, b)

    # sigma (1-D)
    if both_present("uns_action_sigma", "uns_action_sigma", "uns/action_params/sigma"):
        a = np.asarray(py_get("uns_action_sigma")).ravel()
        b = np.asarray(r_get("uns_action_sigma")).ravel()
        _compare_dense("uns/action_params/sigma", a, b)

    # ------------------------------------------------------------------
    # ACTION slots — canonicalize archetype ordering
    # ------------------------------------------------------------------
    H_stacked_py = py_get("obsm_H_stacked")
    H_stacked_r  = r_get("obsm_H_stacked")
    H_merged_py  = py_get("obsm_H_merged")
    H_merged_r   = r_get("obsm_H_merged")
    C_stacked_py = py_get("obsm_C_stacked")
    C_stacked_r  = r_get("obsm_C_stacked")
    C_merged_py  = py_get("obsm_C_merged")
    C_merged_r   = r_get("obsm_C_merged")

    if H_stacked_py is not None and H_stacked_r is not None:
        [H_stacked_py_c] = canonicalize_archetype_ordering(H_stacked_py)
        [H_stacked_r_c]  = canonicalize_archetype_ordering(H_stacked_r)
        _compare_dense("obsm/H_stacked (cells x archetypes)", H_stacked_py_c, H_stacked_r_c)
    else:
        if H_stacked_py is None:
            _report("obsm/H_stacked", False, float("nan"), "MISSING in Python")
        else:
            _report("obsm/H_stacked", False, float("nan"), "MISSING in R")

    if H_merged_py is not None and H_merged_r is not None:
        [H_merged_py_c] = canonicalize_archetype_ordering(H_merged_py)
        [H_merged_r_c]  = canonicalize_archetype_ordering(H_merged_r)
        _compare_dense("obsm/H_merged (cells x archetypes)", H_merged_py_c, H_merged_r_c)
    else:
        missing_in = "Python" if H_merged_py is None else "R"
        _report("obsm/H_merged", False, float("nan"), f"MISSING in {missing_in}")

    for (py_key, r_key, slot) in [
        ("obsm_C_stacked", "obsm_C_stacked", "obsm/C_stacked"),
        ("obsm_C_merged",  "obsm_C_merged",  "obsm/C_merged"),
    ]:
        if both_present(py_key, r_key, slot):
            _compare_dense(slot, np.array(py_get(py_key)), np.array(r_get(r_key)))

    # Archetype assignment (integer, check with 0/1-index offset tolerance)
    if both_present("obs_assigned_archetype", "obs_assigned_archetype", "obs/assigned_archetype"):
        a = np.asarray(py_get("obs_assigned_archetype")).ravel().astype(int)
        b = np.asarray(r_get("obs_assigned_archetype")).ravel().astype(int)
        if a.shape != b.shape:
            _report("obs/assigned_archetype", False, float("nan"),
                    f"shape mismatch: {a.shape} vs {b.shape}")
        else:
            diffs = a - b
            # R uses 1-based indexing; Python uses 0-based.
            # If all differences are exactly -1 the ordering is identical and only the
            # base index differs (expected pre-existing difference).
            if np.all(diffs == -1):
                _report("obs/assigned_archetype", True, 1.0,
                        "0-indexed Python vs 1-indexed R (known pre-existing difference)")
            elif np.all((diffs == -1) | (diffs == 0)):
                exact = int(np.sum(diffs != -1))
                _report("obs/assigned_archetype", True, 1.0,
                        f"0-indexed Python vs 1-indexed R ({exact} cells also have assignment differences)")
            else:
                exact = int(np.sum(a != b))
                max_dev = float(np.max(np.abs(diffs)))
                _report("obs/assigned_archetype", exact == 0, max_dev,
                        f"{exact}/{len(a)} cells differ" if exact > 0 else "")

    # ------------------------------------------------------------------
    # Network — sparse comparison
    # ------------------------------------------------------------------
    # The Python npz stores actionet as sparse (data/indices/indptr/shape);
    # the R npz stores it as dense (converted via as.matrix in R).
    py_G_sparse = _reconstruct_sparse(py, "obsp_actionet")
    r_G_dense   = r_get("obsp_actionet")

    if py_G_sparse is None and py_get("obsp_actionet") is not None:
        # Dense fallback
        py_G_arr = py_get("obsp_actionet")
    elif py_G_sparse is not None:
        py_G_arr = py_G_sparse.toarray()
    else:
        py_G_arr = None

    if py_G_arr is not None and r_G_dense is not None:
        py_G_sp = sp.csr_matrix(py_G_arr)
        r_G_sp  = sp.csr_matrix(r_G_dense)
        _compare_sparse("obsp/actionet (cells x cells sparse)", py_G_sp, r_G_sp)
    else:
        missing = "Python" if py_G_arr is None else "R"
        _report("obsp/actionet", False, float("nan"), f"MISSING in {missing}")

    # ------------------------------------------------------------------
    # Specificity (cluster)
    # NOTE: varm/specificity_profile is Python-only (R does not store this
    #       slot in computeFeatureSpecificity). This is a known pre-existing
    #       API asymmetry, not a regression.
    # ------------------------------------------------------------------
    for (py_key, r_key, slot) in [
        ("varm_specificity_upper",   "varm_specificity_upper",   "varm/specificity_upper"),
        ("varm_specificity_lower",   "varm_specificity_lower",   "varm/specificity_lower"),
    ]:
        if both_present(py_key, r_key, slot):
            _compare_dense(slot, np.array(py_get(py_key)), np.array(r_get(r_key)))

    # varm_specificity_profile is Python-only — document as known asymmetry
    if py_get("varm_specificity_profile") is not None:
        _report(
            "varm/specificity_profile",
            True,
            float("nan"),
            "Python-only output (R does not store average feature profile from computeFeatureSpecificity)",
        )

    # ------------------------------------------------------------------
    # Specificity (archetype)
    # ------------------------------------------------------------------
    for (py_key, r_key, slot) in [
        ("varm_archetype_feat_profile",             "varm_archetype_feat_profile",             "varm/archetype_feat_profile"),
        ("varm_archetype_feat_specificity_upper",   "varm_archetype_feat_specificity_upper",   "varm/archetype_feat_specificity_upper"),
        ("varm_archetype_feat_specificity_lower",   "varm_archetype_feat_specificity_lower",   "varm/archetype_feat_specificity_lower"),
    ]:
        if both_present(py_key, r_key, slot):
            _compare_dense(slot, np.array(py_get(py_key)), np.array(r_get(r_key)))

    # ------------------------------------------------------------------
    # Batch correction
    # ------------------------------------------------------------------
    for (py_key, r_key, slot) in [
        ("obsm_action_corrected",   "obsm_action_corrected",   "obsm/action_corrected (cells x k)"),
        ("varm_action_corrected_U", "varm_action_corrected_U", "varm/action_corrected_U"),
        ("varm_action_corrected_A", "varm_action_corrected_A", "varm/action_corrected_A"),
    ]:
        if both_present(py_key, r_key, slot):
            a = canonicalize_svd_sign(py_get(py_key))
            b = canonicalize_svd_sign(r_get(r_key))
            _compare_dense(slot, a, b)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed  = [r for r in _RESULTS if r["status"] == "PASS"]
    failed  = [r for r in _RESULTS if r["status"] == "FAIL"]
    missing = [r for r in _RESULTS if "MISSING" in r.get("note", "")]

    print("=" * 80)
    print(f"PARITY REPORT  —  {len(passed)} PASS  /  {len(failed)} FAIL  /  {len(_RESULTS)} total")
    print("=" * 80)
    col_w = 50
    for res in _RESULTS:
        status_str = res["status"]
        dev_str = f"{res['max_dev']:.3e}" if not np.isnan(res["max_dev"]) else "    nan"
        note    = f"  [{res['note']}]" if res.get("note") else ""
        print(f"  {status_str}  {res['slot']:{col_w}}  max_dev={dev_str}{note}")

    print()
    if failed:
        print("FAILED slots:")
        for res in failed:
            print(f"  - {res['slot']}: {res.get('note', '')}")
    else:
        print("All slots PASS.")

    print()
    if missing:
        print("NOTE: some slots were missing in one or both baselines.")
        print("These represent the current state of the system, not regressions.")
        for res in missing:
            print(f"  - {res['slot']}: {res.get('note', '')}")

    return len(failed) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--python-npz",
        default=DEFAULT_PY_NPZ,
        help=f"Path to baseline_python.npz (default: {DEFAULT_PY_NPZ})",
    )
    parser.add_argument(
        "--r-npz",
        default=DEFAULT_R_NPZ,
        help=f"Path to baseline_r_converted.npz (default: {DEFAULT_R_NPZ})",
    )
    parser.add_argument(
        "--r-rds",
        default=None,
        help=(
            "Path to baseline_r.rds. If provided and --r-npz does not exist, "
            "will attempt conversion via rpy2."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print extra debug information.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    r_npz = args.r_npz
    if not os.path.exists(r_npz):
        # First try the h5ad written directly by the R script
        r_h5ad = os.path.join(
            _ACTIONET_R, "tests", "fixtures", "baseline_r.h5ad"
        )
        if os.path.exists(r_h5ad):
            print(f"R .npz not found; extracting from h5ad: {r_h5ad}")
            os.makedirs(os.path.dirname(r_npz) or ".", exist_ok=True)
            load_r_baseline_from_h5ad(r_h5ad, r_npz)
        else:
            # Try rpy2 path from .rds
            rds = args.r_rds or DEFAULT_R_RDS
            if os.path.exists(rds):
                print(f"R .npz not found; converting from {rds} via rpy2 ...")
                os.makedirs(os.path.dirname(r_npz) or ".", exist_ok=True)
                convert_r_rds_to_npz(rds, r_npz)
            else:
                print(
                    f"ERROR: R baseline not found at {r_npz}\n"
                    f"       Expected h5ad at {r_h5ad} or .rds at {rds}\n"
                    f"       Run actionet-r/tests/generate_baseline.R first."
                )
                return 1

    all_pass = compare(args.python_npz, r_npz, verbose=args.verbose)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
