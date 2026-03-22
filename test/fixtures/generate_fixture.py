"""
Generate the shared parity fixture for the AnnData orientation unification baseline.

Produces: parity_fixture.h5ad
  - ~500 cells, ~2000 genes
  - Sparse CSC expression matrix
  - At least 3 distinct cell populations
  - batch labels in obs["batch"] (2 batches: "A" and "B")
  - Fixed random seed: 42

Run from any directory:
    python libactionet/test/fixtures/generate_fixture.py
"""

import os
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "parity_fixture.h5ad")


def main():
    rng = np.random.default_rng(42)

    n_cells = 500
    n_genes = 2000
    n_populations = 4  # at least 3

    pop_sizes = [130, 120, 130, 120]  # sum = 500
    assert sum(pop_sizes) == n_cells

    # Gene name / cell name scaffolding
    gene_names = [f"Gene{i:04d}" for i in range(n_genes)]
    cell_names = [f"Cell{i:04d}" for i in range(n_cells)]

    # -----------------------------------------------------------------------
    # Simulate a sparse expression matrix with population structure
    # Each population expresses a distinct subset of ~200 "marker" genes at
    # elevated levels; background expression is very sparse.
    # -----------------------------------------------------------------------
    base_rate = 0.02          # average non-zero probability in background
    marker_rate = 0.40        # probability of expression in marker genes
    marker_scale = 5.0        # fold-enrichment for marker genes

    rows, cols, vals = [], [], []

    pop_assignments = np.repeat(np.arange(n_populations), pop_sizes)
    markers_per_pop = n_genes // n_populations  # ~500 genes each

    cell_offset = 0
    for pop_idx, pop_size in enumerate(pop_sizes):
        marker_start = pop_idx * markers_per_pop
        marker_end = (pop_idx + 1) * markers_per_pop

        for local_cell in range(pop_size):
            cell_idx = cell_offset + local_cell
            # Background expression
            bg_mask = rng.random(n_genes) < base_rate
            bg_vals = rng.exponential(1.0, size=n_genes) * bg_mask

            # Marker expression (elevated)
            marker_mask = np.zeros(n_genes, dtype=bool)
            marker_mask[marker_start:marker_end] = rng.random(markers_per_pop) < marker_rate
            marker_vals = rng.exponential(marker_scale, size=n_genes) * marker_mask

            expr = bg_vals + marker_vals
            nz = np.where(expr > 0)[0]
            rows.extend([cell_idx] * len(nz))
            cols.extend(nz.tolist())
            vals.extend(expr[nz].tolist())

        cell_offset += pop_size

    X_csr = sp.csr_matrix(
        (np.array(vals, dtype=np.float32), (rows, cols)),
        shape=(n_cells, n_genes),
    )
    # Store as CSC on disk (AnnData-native for .h5ad)
    X_csc = X_csr.tocsc()

    # -----------------------------------------------------------------------
    # Observations metadata
    # -----------------------------------------------------------------------
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(
                rng.choice(["A", "B"], size=n_cells).tolist()
            ),
            "population": pd.Categorical(
                [f"pop{p}" for p in pop_assignments]
            ),
        },
        index=cell_names,
    )

    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=X_csc, obs=obs, var=var)

    # -----------------------------------------------------------------------
    # Add a log-normalised layer (expected by R pipeline as "logcounts")
    # -----------------------------------------------------------------------
    # library-size normalise to 10k counts then log1p
    cell_counts = np.asarray(X_csr.sum(axis=1)).ravel()
    cell_counts = np.where(cell_counts == 0, 1.0, cell_counts)
    scale = 1e4 / cell_counts
    X_norm = X_csr.multiply(scale[:, None]).tocsr()
    X_log = X_norm.copy()
    X_log.data = np.log1p(X_log.data)
    adata.layers["logcounts"] = X_log.tocsc()

    adata.write_h5ad(OUTPUT_PATH, compression="gzip")
    print(f"Fixture written to: {OUTPUT_PATH}")
    print(f"  Shape : {adata.shape}  (cells x genes)")
    print(f"  Density: {X_csc.nnz / (n_cells * n_genes):.3%}")
    print(f"  Populations: {obs['population'].value_counts().to_dict()}")
    print(f"  Batches    : {obs['batch'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
