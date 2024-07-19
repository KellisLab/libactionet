#include "orthogonalization.hpp"
#include "svd.hpp"

using namespace arma;

namespace ACTIONet
{

    field<mat> orthogonalize_batch_effect(sp_mat &S, field<mat> SVD_results,
                                          mat &design)
    {
        stdout_printf("Orthogonalizing batch effect (sparse):\n");
        FLUSH;

        mat Z = mat(S * design);
        gram_schmidt(Z);

        mat A = Z;
        mat B = -mat(trans(trans(Z) * S));

        field<mat> perturbed_SVD = deflate_reduction(SVD_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }

    field<mat> orthogonalize_batch_effect(mat &S, field<mat> SVD_results,
                                          mat &design)
    {
        stdout_printf("Orthogonalizing batch effect: (dense):\n");
        FLUSH;

        mat Z = mat(S * design);
        gram_schmidt(Z);

        mat A = Z;
        mat B = -mat(trans(trans(Z) * S));

        field<mat> perturbed_SVD = deflate_reduction(SVD_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }

    field<mat> orthogonalize_basal(sp_mat &S, field<mat> SVD_results,
                                   mat &basal_state)
    {
        stdout_printf("Orthogonalizing basal (sparse):\n");
        FLUSH;

        mat Z = basal_state;
        gram_schmidt(Z);

        mat A = Z;
        mat B = -mat(trans(trans(Z) * S));

        field<mat> perturbed_SVD = deflate_reduction(SVD_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }

    field<mat> orthogonalize_basal(mat &S, field<mat> SVD_results,
                                   mat &basal_state)
    {
        stdout_printf("Orthogonalizing basal (dense):\n");
        FLUSH;

        mat Z = basal_state;
        gram_schmidt(Z);

        mat A = Z;
        mat B = -mat(trans(trans(Z) * S));

        field<mat> perturbed_SVD = deflate_reduction(SVD_results, A, B);
        FLUSH;
        return (perturbed_SVD);
    }
} // namespace ACTIONet
