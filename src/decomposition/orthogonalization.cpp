#include "decomposition/orthogonalization.hpp"
#include "decomposition/svd_main.hpp"
#include "utils_internal/utils_decomp.hpp"

arma::field<arma::mat> deflateReduction(arma::field<arma::mat>& SVD_results, arma::mat& A, arma::mat& B) {
    stdout_printf("\tDeflating reduction ... ");
    FLUSH;

    arma::vec mu_A = arma::vec(arma::trans(arma::mean(A, 0)));
    arma::vec mu = B * mu_A;

    A = arma::join_rows(arma::ones(A.n_rows), A);
    B = arma::join_rows(-mu, B);
    stdout_printf("done\n");
    FLUSH;

    arma::field<arma::mat> perturbed_SVD = actionet::perturbedSVD(SVD_results, A, B);
    return (perturbed_SVD);
}

namespace actionet {
    template <typename T>
    arma::field<arma::mat> orthogonalizeBatchEffect(T& S, arma::field<arma::mat>& SVD_results, arma::mat& design) {
        stdout_printf("Orthogonalizing batch effect:\n");
        FLUSH;

        arma::mat Z = arma::mat(S * design);
        gram_schmidt(Z);

        arma::mat A = Z;
        arma::mat B = -arma::mat(arma::trans(arma::trans(Z) * S));

        arma::field<arma::mat> perturbed_SVD = deflateReduction(SVD_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }

    template arma::field<arma::mat>
        orthogonalizeBatchEffect<arma::mat>(arma::mat& S, arma::field<arma::mat>& SVD_results, arma::mat& design);

    template arma::field<arma::mat>
        orthogonalizeBatchEffect<arma::sp_mat>(arma::sp_mat& S, arma::field<arma::mat>& SVD_results,
                                               arma::mat& design);

    template <typename T>
    arma::field<arma::mat> orthogonalizeBasal(T& S, arma::field<arma::mat>& SVD_results, arma::mat& basal_state) {
        stdout_printf("Orthogonalizing basal:\n");
        FLUSH;

        arma::mat Z = basal_state;
        gram_schmidt(Z);

        arma::mat A = Z;
        arma::mat B = -arma::mat(arma::trans(arma::trans(Z) * S));

        arma::field<arma::mat> perturbed_SVD = deflateReduction(SVD_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }

    template arma::field<arma::mat>
        orthogonalizeBasal<arma::mat>(arma::mat& S, arma::field<arma::mat>& SVD_results, arma::mat& basal_state);

    template arma::field<arma::mat>
        orthogonalizeBasal<arma::sp_mat>(arma::sp_mat& S, arma::field<arma::mat>& SVD_results, arma::mat& basal_state);
} // namespace actionet
