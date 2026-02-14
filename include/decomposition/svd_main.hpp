// Interface for value decomposition (SVD) algorithms
#ifndef ACTIONET_SVD_MAIN_HPP
#define ACTIONET_SVD_MAIN_HPP

#include "libactionet_config.hpp"
#include "decomposition/matrix_operator.hpp"

// SVD algorithm options
#define ALG_IRLB 0
#define ALG_HALKO 1
#define ALG_FENG 2
#define ALG_PRIMME 3  // PRIMME_SVDS (recommended for large sparse matrices >2^31 elements)

// Exported
namespace actionet {
    struct SVDResult {
        arma::mat U;
        arma::vec sigma;
        arma::mat V;
    };

    inline SVDResult svdResultFromField(const arma::field<arma::mat>& svd) {
        SVDResult out;
        out.U = svd(0);
        out.sigma = arma::vec(svd(1));
        out.V = svd(2);
        return out;
    }

    inline arma::field<arma::mat> svdFieldFromResult(const SVDResult& svd) {
        arma::field<arma::mat> out(3);
        out(0) = svd.U;
        out(1) = svd.sigma;
        out(2) = svd.V;
        return out;
    }

    /// @brief Compute truncated SVD using the selected algorithm.
    ///
    /// @tparam T Dense or sparse matrix type.
    /// @param A Input matrix.
    /// @param k Number of singular vectors/values.
    /// @param max_it Maximum iterations (0 = auto).
    /// @param seed Random seed.
    /// @param algorithm SVD algorithm code.
    /// @param verbose Print progress messages.
    ///
    /// @return Field containing {U, S, V} (or algorithm-specific outputs).
    template <typename T>
    arma::field<arma::mat> runSVD(T& A, int k, int max_it = 0, int seed = 0, int algorithm = ALG_IRLB, bool verbose = true);

    /// @brief Compute truncated SVD using PRIMME via matrix operator callbacks.
    ///
    /// @param op Input matrix operator.
    /// @param k Number of singular vectors/values.
    /// @param max_it Maximum iterations (0 = auto).
    /// @param seed Random seed.
    /// @param verbose Print progress messages.
    ///
    /// @return Structured SVD result.
    SVDResult runSVD_PRIMME_Operator(const MatrixOperator& op, int k, int max_it = 0,
                                     int seed = 0, bool verbose = true);

    /// @brief Apply perturbation correction to an SVD decomposition.
    ///
    /// @param SVD_results Field containing SVD outputs.
    /// @param A Perturbation matrix A.
    /// @param B Perturbation matrix B.
    ///
    /// @return Field containing corrected SVD outputs.
    arma::field<arma::mat> perturbedSVD(arma::field<arma::mat>& SVD_results, arma::mat& A, arma::mat& B);
} // namespace actionet

#endif //ACTIONET_SVD_MAIN_HPP
