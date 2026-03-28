// Interface for singular value decomposition (SVD) algorithms
//
// This header defines SVD result types, algorithm selection constants, and
// the primary decomposition entry points consumed by language bindings and
// downstream modules (kernel reduction, orthogonalization, etc.).

#ifndef ACTIONET_SVD_MAIN_HPP
#define ACTIONET_SVD_MAIN_HPP

#include "libactionet_config.hpp"
#include "decomposition/matrix_operator.hpp"

// SVD algorithm options
namespace actionet {
    constexpr int ALG_IRLB   = 0;
    constexpr int ALG_HALKO  = 1;
    constexpr int ALG_FENG   = 2;
    constexpr int ALG_PRIMME = 3;  // PRIMME_SVDS (recommended for large sparse matrices >2^31 elements)
}

// Exported
namespace actionet {

    /// @brief Structured result of a truncated SVD: A ≈ U * diag(sigma) * V'.
    struct SVDResult {
        arma::mat U;      ///< Left singular vectors  (m × k)
        arma::vec sigma;  ///< Singular values         (k)
        arma::mat V;      ///< Right singular vectors  (n × k)
    };

    /// @brief Result of perturbedSVD: corrected SVD plus accumulated perturbation terms.
    struct PerturbedSVDResult {
        arma::mat U;      ///< Updated left singular vectors  (m × k)
        arma::vec sigma;  ///< Updated singular values         (k)
        arma::mat V;      ///< Updated right singular vectors  (n × k)
        arma::mat A;      ///< Accumulated left perturbation  (m × p)
        arma::mat B;      ///< Accumulated right perturbation (n × p)
    };

    // ---- Legacy field ↔ struct conversion helpers ----------------------------------------
    // These exist solely to bridge code that still uses arma::field<arma::mat> (e.g. the
    // templated runSVD, orient_SVD, orthogonalization paths).  New code should prefer the
    // typed structs directly.

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
    /// @param algorithm SVD algorithm code (ALG_IRLB, ALG_HALKO, ALG_FENG, ALG_PRIMME).
    /// @param verbose Print progress messages.
    ///
    /// @return Field containing {U, sigma, V}.
    template <typename T>
    arma::field<arma::mat> runSVD(const T& A, int k, int max_it = 0, int seed = 0, int algorithm = ALG_IRLB, bool verbose = true);

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

    /// @brief Compute truncated SVD with a matrix operator and explicit algorithm.
    SVDResult runSVD_Operator(const MatrixOperator& op, int k, int max_it = 0, int seed = 0,
                              int algorithm = ALG_HALKO, bool verbose = true);

    /// @brief Compute truncated SVD using Halko with matrix operator callbacks.
    SVDResult runSVD_Halko_Operator(const MatrixOperator& op, int k, int iters = 5,
                                    int seed = 0, bool verbose = true);

    /// @brief Compute truncated SVD using Feng with matrix operator callbacks.
    SVDResult runSVD_Feng_Operator(const MatrixOperator& op, int k, int max_it = 5,
                                   int seed = 0, bool verbose = true);

    /// @brief Apply perturbation correction to an SVD decomposition (struct API).
    ///
    /// Implements the Brand (2006) perturbation update:  given A ≈ U Σ V' and low-rank
    /// perturbation terms (Apert, Bpert) such that the corrected matrix is
    /// A + Apert * Bpert', compute an updated truncated SVD of the corrected matrix.
    ///
    /// If @p prior contains previously accumulated perturbation terms (non-empty A/B),
    /// the new terms are horizontally concatenated onto them.
    ///
    /// @param svd        Input truncated SVD.
    /// @param Apert      Left perturbation matrix  (m × p).
    /// @param Bpert      Right perturbation matrix (n × p).
    /// @param prior      Optional previously accumulated perturbation (may be nullptr).
    ///
    /// @return Updated SVD with accumulated perturbation terms.
    PerturbedSVDResult perturbedSVD(const SVDResult& svd,
                                    const arma::mat& Apert,
                                    const arma::mat& Bpert,
                                    const PerturbedSVDResult* prior = nullptr);

    /// @brief Legacy overload: perturbedSVD operating on arma::field containers.
    ///
    /// Provided for backward compatibility with orthogonalization and R wrapper code.
    /// New callers should prefer the struct-based overload above.
    ///
    /// @param SVD_results Field containing SVD outputs (3 or 5 elements).
    /// @param A Perturbation matrix A.
    /// @param B Perturbation matrix B.
    ///
    /// @return Field containing corrected SVD outputs {U', sigma', V', A, B}.
    arma::field<arma::mat> perturbedSVD(const arma::field<arma::mat>& SVD_results,
                                        const arma::mat& A, const arma::mat& B);

} // namespace actionet

#endif //ACTIONET_SVD_MAIN_HPP
