// Kernel reduction for the ACTION algorithm.
//
// Implements the reduction pipeline described in reduce_kernel.hpp:
//   SVD (cells x genes) → perturbation terms → perturbedSVD (B/A swapped) → rounding/scaling → S_r (cells x k)
//
// AnnData-native orientation (Plan 02):
//   S is cells x genes.  Gene-space (col-space) perturbation → A (genes x p).
//   Cell-space (row-space) perturbation → B (cells x p).
//   S_r is assembled from the left singular vectors U (cells x k), not V.

#include "action/reduce_kernel.hpp"
#include "decomposition/svd_main.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace actionet {

    // ---- Internal helpers (in-memory perturbation computation) ----------------------------

    namespace {
        /// @brief Compute centering perturbation terms from a fully materialised matrix.
        ///
        /// S is cells × genes (obs × var).
        ///
        /// The perturbation encodes a rank-2 correction that centers the matrix:
        ///   a1 = mu / ||mu||    (normalised gene-mean direction, genes-length)
        ///   b1 = -(S * a1)      (cell-length projection: (cells x genes)(genes x 1))
        ///   a2 = ones            (gene-length all-ones vector)
        ///   b2 = -(mean(a1) * b1 + row_means_of_S)
        ///
        /// A = [a1, a2] (genes × 2), B = [b1, b2] (cells × 2).
        template <typename T>
        void computeKernelPerturbationTermsInMemory(const T& S, arma::mat& A, arma::mat& B) {
            // Column means (gene means) — S is cells × genes, so mean across rows (dim=0)
            arma::vec mu(arma::trans(arma::mean(S, 0)));
            double mu_norm = arma::norm(mu, 2);
            if (mu_norm <= std::numeric_limits<double>::epsilon()) {
                throw std::runtime_error("reduceKernel: mean vector has zero norm");
            }

            arma::vec a1 = mu / mu_norm;                  // genes-length
            arma::vec b1 = -(S * a1);                     // cells-length: (cells x genes)(genes x 1)

            // Row means (cell means) — mean across columns (dim=1)
            arma::vec c = arma::vec(arma::mean(S, 1));    // cells-length
            double a1_mean = arma::mean(a1);
            arma::vec a2 = arma::ones(S.n_cols);           // genes-length
            arma::vec b2 = -(a1_mean * b1 + c);

            A = arma::join_rows(a1, a2);   // genes × 2
            B = arma::join_rows(b1, b2);   // cells × 2
        }

        /// @brief Validate that SVD dimensions are self-consistent and compatible with
        ///        perturbation matrices A and B.
        ///
        /// With the AnnData-native orientation (S is cells × genes):
        ///   svd.U is (cells × k), svd.V is (genes × k).
        ///   A (genes × p) corresponds to the col-space (V side).
        ///   B (cells × p) corresponds to the row-space (U side).
        void validateSVDAndPerturbation(const SVDResult& svd, const arma::mat& A, const arma::mat& B) {
            if (svd.U.n_cols != svd.sigma.n_elem) {
                throw std::runtime_error("applyKernelPostSVD: U.n_cols != sigma.n_elem");
            }
            if (svd.V.n_cols != svd.sigma.n_elem) {
                throw std::runtime_error("applyKernelPostSVD: V.n_cols != sigma.n_elem");
            }
            // A is genes × p → must match V.n_rows (genes)
            if (A.n_rows != svd.V.n_rows) {
                throw std::runtime_error("applyKernelPostSVD: A.n_rows != V.n_rows (genes)");
            }
            // B is cells × p → must match U.n_rows (cells)
            if (B.n_rows != svd.U.n_rows) {
                throw std::runtime_error("applyKernelPostSVD: B.n_rows != U.n_rows (cells)");
            }
            if (A.n_cols != B.n_cols) {
                throw std::runtime_error("applyKernelPostSVD: A.n_cols != B.n_cols");
            }
        }
    } // namespace

    // ---- Operator-backed perturbation computation ----------------------------------------

    void computeKernelPerturbationTerms(const MatrixOperator& S, arma::mat& A, arma::mat& B) {
        arma::uword m = S.rows();  // cells (obs)
        arma::uword n = S.cols();  // genes (var)
        if (m == 0 || n == 0) {
            throw std::runtime_error("computeKernelPerturbationTerms: empty matrix");
        }

        // Gene means (column means of cells × genes) via rmatvec with all-ones over cells.
        // rmatvec: S' * ones_m = (cells × genes)' * (cells × 1) → genes-length
        arma::vec ones_m = arma::ones<arma::vec>(m);
        arma::vec mu_sum(n);
        S.rmatvec(ones_m, mu_sum);
        arma::vec mu = mu_sum / static_cast<double>(m);   // gene means, length = n

        double mu_norm = arma::norm(mu, 2);
        if (mu_norm <= std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error("computeKernelPerturbationTerms: mean vector has zero norm");
        }

        arma::vec a1 = mu / mu_norm;                       // genes-length
        arma::vec b1_tmp(m);
        S.matvec(a1, b1_tmp);                              // S * a1: (cells × genes)(genes × 1) → cells-length
        arma::vec b1 = -b1_tmp;

        // Row means (cell means) via matvec with all-ones over genes.
        // matvec: S * ones_n = (cells × genes)(genes × 1) → cells-length
        arma::vec ones_n = arma::ones<arma::vec>(n);
        arma::vec c_sum(m);
        S.matvec(ones_n, c_sum);
        arma::vec c = c_sum / static_cast<double>(n);      // cell means, length = m

        double a1_mean = arma::mean(a1);
        arma::vec a2 = arma::ones<arma::vec>(n);           // genes-length
        arma::vec b2 = -(a1_mean * b1 + c);

        A = arma::join_rows(a1, a2);   // genes × 2
        B = arma::join_rows(b1, b2);   // cells × 2
    }

    // ---- Core post-SVD kernel assembly ---------------------------------------------------

    KernelReductionResult applyKernelPostSVD(const SVDResult& svd, const arma::mat& A, const arma::mat& B) {
        validateSVDAndPerturbation(svd, A, B);

        // S (cells × genes): row-space = cells (B), col-space = genes (A).
        // perturbedSVD(svd, Apert, Bpert) expects:
        //   Apert for the row-space side (left/U side) → B (cells × p)
        //   Bpert for the col-space side (right/V side) → A (genes × p)
        PerturbedSVDResult perturbed = perturbedSVD(svd, B, A);

        KernelReductionResult out;
        out.sigma = perturbed.sigma;

        // Discretisation step: round the LEFT singular vectors (cells × k) to a grid
        // determined by epsilon = 0.01 / sqrt(n_cells).  This suppresses small numerical
        // noise in the cell coordinates before scaling by the singular values to form S_r.
        // The constant 0.01 was empirically chosen to balance noise suppression against
        // loss of discriminating detail in downstream archetypal analysis.
        double epsilon = 0.01 / std::sqrt(static_cast<double>(perturbed.U.n_rows));
        arma::mat U_rounded = arma::round(perturbed.U / epsilon) * epsilon;
        for (arma::uword i = 0; i < U_rounded.n_cols; i++) {
            U_rounded.col(i) *= out.sigma(i);
        }

        out.S_r = U_rounded;        // cells × k  (left singular vectors, scaled)
        out.U   = perturbed.V;      // genes × k  (right singular vectors = gene loadings)
        out.A   = perturbed.B;      // genes × p  (accumulated col-space perturbation, stored as A for compat)
        out.B   = perturbed.A;      // cells × p  (accumulated row-space perturbation, stored as B for compat)
        return out;
    }

    // ---- Operator-backed entry points ----------------------------------------------------

    KernelReductionResult reduceKernelFromSVD_Operator(const MatrixOperator& S, const SVDResult& svd, bool verbose) {
        if (verbose) {
            stdout_printf("Computing reduced ACTION kernel from precomputed SVD (operator):\n");
            FLUSH;
        }

        arma::mat A, B;
        computeKernelPerturbationTerms(S, A, B);
        KernelReductionResult out = applyKernelPostSVD(svd, A, B);

        if (verbose) {
            stdout_printf("Kernel computed successfully.\n");
            FLUSH;
        }
        return out;
    }

    KernelReductionResult reduceKernel_Operator(const MatrixOperator& S, int k, int svd_alg, int max_it,
                                                int seed, bool verbose) {
        // NOTE: The operator / PRIMME path is unavailable in R builds because the R build
        // system does not link PRIMME.  If R support for operator-backed SVD is needed in
        // the future, the recommended approach is to:
        //   1. Add PRIMME as an optional CMake dependency gated on LIBACTIONET_BUILD_R.
        //   2. Provide a pure-R fallback using irlba::irlba with custom matvec.
        //
        // The operator now represents cells × genes (obs × var) natively.
#if defined(LIBACTIONET_BUILD_R) && LIBACTIONET_BUILD_R == 1
        (void)S;
        (void)k;
        (void)svd_alg;
        (void)max_it;
        (void)seed;
        (void)verbose;
        throw std::runtime_error("reduceKernel_Operator is unavailable in R build mode "
                                 "(PRIMME is not linked). Use the in-memory reduceKernel() path.");
#else
        if (verbose) {
            stdout_printf("Computing reduced ACTION kernel (operator):\n");
            FLUSH;
        }

        SVDResult svd = runSVD_Operator(S, k, max_it, seed, svd_alg, verbose);
        return reduceKernelFromSVD_Operator(S, svd, verbose);
#endif
    }

    // ---- In-memory precomputed SVD entry points ------------------------------------------

    KernelReductionResult reduceKernelFromSVD(const SVDResult& svd, const arma::mat& A, const arma::mat& B) {
        return applyKernelPostSVD(svd, A, B);
    }

    template <typename T>
    KernelReductionResult reduceKernelFromSVD_InMemory(const T& S, const SVDResult& svd, bool verbose) {
        if (verbose) {
            stdout_printf("Computing reduced ACTION kernel from precomputed SVD (in-memory):\n");
            FLUSH;
        }

        arma::mat A, B;
        computeKernelPerturbationTermsInMemory(S, A, B);
        KernelReductionResult out = applyKernelPostSVD(svd, A, B);

        if (verbose) {
            stdout_printf("Kernel computed successfully.\n");
            FLUSH;
        }
        return out;
    }

    // Explicit instantiations for dense and sparse.
    template KernelReductionResult reduceKernelFromSVD_InMemory<arma::mat>(
        const arma::mat& S, const SVDResult& svd, bool verbose);
    template KernelReductionResult reduceKernelFromSVD_InMemory<arma::sp_mat>(
        const arma::sp_mat& S, const SVDResult& svd, bool verbose);

    // ---- Legacy in-memory entry point ----------------------------------------------------

    template <typename T>
    arma::field<arma::mat> reduceKernel(T& S, int k, int svd_alg, int max_it, int seed, bool verbose) {
        if (verbose) {
            stdout_printf("Computing reduced ACTION kernel:\n");
            FLUSH;
        }

        arma::field<arma::mat> svd_out = runSVD(S, k, max_it, seed, svd_alg, verbose);
        SVDResult svd = svdResultFromField(svd_out);

        arma::mat A, B;
        computeKernelPerturbationTermsInMemory(S, A, B);

        KernelReductionResult reduction = applyKernelPostSVD(svd, A, B);

        if (verbose) {
            stdout_printf("Kernel computed successfully.\n");
            FLUSH;
        }
        return kernelFieldFromResult(reduction);
    }

    template arma::field<arma::mat> reduceKernel<arma::mat>(arma::mat& S, int k, int svd_alg,
                                                            int iter, int seed, bool verbose);

    template arma::field<arma::mat> reduceKernel<arma::sp_mat>(arma::sp_mat& S, int k, int svd_alg,
                                                               int iter, int seed, bool verbose);
} // namespace actionet
