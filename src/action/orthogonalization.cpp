#include "action/orthogonalization.hpp"
#include "action/svd.hpp"
#include "action/reduction.hpp"

namespace ACTIONet
{

    arma::field<arma::mat> orthogonalize_batch_effect(arma::sp_mat &S, arma::field<arma::mat> SVD_results,
                                                      arma::mat &design)
    {
        stdout_printf("Orthogonalizing batch effect (sparse):\n");
        FLUSH;

        arma::mat Z = arma::mat(S * design);
        gram_schmidt(Z);

        arma::mat A = Z;
        arma::mat B = -arma::mat(arma::trans(arma::trans(Z) * S));

        arma::field<arma::mat> perturbed_SVD = deflate_reduction(SVD_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }

    arma::field<arma::mat> orthogonalize_batch_effect(arma::mat &S, arma::field<arma::mat> SVD_results,
                                                      arma::mat &design)
    {
        stdout_printf("Orthogonalizing batch effect: (dense):\n");
        FLUSH;

        arma::mat Z = arma::mat(S * design);
        gram_schmidt(Z);

        arma::mat A = Z;
        arma::mat B = -arma::mat(arma::trans(arma::trans(Z) * S));

        arma::field<arma::mat> perturbed_SVD = deflate_reduction(SVD_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }

    arma::field<arma::mat> orthogonalize_basal(arma::sp_mat &S, arma::field<arma::mat> SVD_results,
                                               arma::mat &basal_state)
    {
        stdout_printf("Orthogonalizing basal (sparse):\n");
        FLUSH;

        arma::mat Z = basal_state;
        gram_schmidt(Z);

        arma::mat A = Z;
        arma::mat B = -arma::mat(arma::trans(arma::trans(Z) * S));

        arma::field<arma::mat> perturbed_SVD = deflate_reduction(SVD_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }

    arma::field<arma::mat> orthogonalize_basal(arma::mat &S, arma::field<arma::mat> SVD_results,
                                               arma::mat &basal_state)
    {
        stdout_printf("Orthogonalizing basal (dense):\n");
        FLUSH;

        arma::mat Z = basal_state;
        gram_schmidt(Z);

        arma::mat A = Z;
        arma::mat B = -arma::mat(arma::trans(arma::trans(Z) * S));

        arma::field<arma::mat> perturbed_SVD = deflate_reduction(SVD_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }
} // namespace ACTIONet
