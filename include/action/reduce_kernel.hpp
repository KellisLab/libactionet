// Main interface for reduction/decomposition algorithms
#ifndef ACTIONET_REDUCE_KERNEL_HPP
#define ACTIONET_REDUCE_KERNEL_HPP

#include "libactionet_config.hpp"
#include "decomposition/svd_main.hpp"

// Exported
namespace actionet {
    struct KernelReductionResult {
        arma::mat S_r;
        arma::vec sigma;
        arma::mat V;
        arma::mat A;
        arma::mat B;
    };

    inline KernelReductionResult kernelResultFromField(const arma::field<arma::mat>& reduction) {
        KernelReductionResult out;
        out.S_r = reduction(0);
        out.sigma = arma::vec(reduction(1));
        out.V = reduction(2);
        out.A = reduction(3);
        out.B = reduction(4);
        return out;
    }

    inline arma::field<arma::mat> kernelFieldFromResult(const KernelReductionResult& reduction) {
        arma::field<arma::mat> out(5);
        out(0) = reduction.S_r;
        out(1) = reduction.sigma;
        out(2) = reduction.V;
        out(3) = reduction.A;
        out(4) = reduction.B;
        return out;
    }

    /// @brief Compute perturbation matrices A and B from an operator-backed matrix.
    void computeKernelPerturbationTerms(const MatrixOperator& S, arma::mat& A, arma::mat& B);

    /// @brief Apply ACTION kernel post-processing from a precomputed SVD and perturbation terms.
    KernelReductionResult applyKernelPostSVD(const SVDResult& svd, const arma::mat& A, const arma::mat& B);

    /// @brief Compute reduced ACTION kernel from precomputed SVD and operator-backed matrix.
    KernelReductionResult reduceKernelFromSVD_Operator(const MatrixOperator& S, const SVDResult& svd,
                                                       bool verbose = true);

    /// @brief Compute reduced ACTION kernel from operator-backed matrix using PRIMME SVD.
    KernelReductionResult reduceKernel_Operator(const MatrixOperator& S, int k, int max_it = 0,
                                                int seed = 0, bool verbose = true);

    /// @brief Compute reduced ACTION kernel from precomputed SVD and in-memory perturbation terms.
    KernelReductionResult reduceKernelFromSVD(const SVDResult& svd, const arma::mat& A, const arma::mat& B);

    /// @brief Compute a reduced kernel matrix using truncated SVD.
    ///
    /// @tparam T Dense or sparse Armadillo matrix type.
    /// @param S Input matrix (<em>vars</em> x <em>obs</em>).
    /// @param k Number of singular vectors to estimate.
    /// @param svd_alg SVD algorithm (see <code>runSVD()</code>).
    /// @param max_it Maximum number of SVD iterations.
    /// @param seed Random seed.
    /// @param verbose Print status messages.
    ///
    /// @return Field with 5 elements: {S_r, sigma, U, A, B}.
    template <typename T>
    arma::field<arma::mat> reduceKernel(T& S, int k, int svd_alg = 0, int max_it = 0,
                                        int seed = 0, bool verbose = true);
}

#endif //ACTIONET_REDUCE_KERNEL_HPP
