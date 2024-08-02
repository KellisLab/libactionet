#include "reduction.hpp"
#include "svd.hpp"

using namespace arma;

field<mat> deflate_reduction(field<mat> SVD_results, mat &A, mat &B)
{
  stdout_printf("\tDeflating reduction ... ");
  FLUSH;

  vec mu_A = vec(trans(mean(A, 0)));
  vec mu = B * mu_A;

  A = join_rows(ones(A.n_rows), A);
  B = join_rows(-mu, B);
  stdout_printf("done\n");
  FLUSH;

  field<mat> perturbed_SVD = ACTIONet::perturbedSVD(SVD_results, A, B);
  return (perturbed_SVD);
}

namespace ACTIONet
{
  field<mat> PCA2SVD(sp_mat &S, field<mat> PCA_results)
  {
    int n = S.n_rows;

    stdout_printf("PCA => SVD (sparse)\n");
    FLUSH;
    mat U = PCA_results(0);
    vec s = PCA_results(1);
    mat V = PCA_results(2);

    int dim = U.n_cols;

    mat A = ones(S.n_rows, 1);
    mat B = mat(trans(mean(S, 0)));

    field<mat> perturbed_SVD = perturbedSVD(PCA_results, A, B);

    return perturbed_SVD;
  }

  field<mat> PCA2SVD(mat &S, field<mat> PCA_results)
  {
    int n = S.n_rows;

    stdout_printf("PCA => SVD (dense)\n");
    FLUSH;
    mat U = PCA_results(0);
    vec s = PCA_results(1);
    mat V = PCA_results(2);

    int dim = U.n_cols;

    mat A = ones(S.n_rows, 1);
    mat B = mat(trans(mean(S, 0)));

    field<mat> perturbed_SVD = perturbedSVD(PCA_results, A, B);

    return perturbed_SVD;
  }

  field<mat> SVD2PCA(sp_mat &S, field<mat> SVD_results)
  {
    int n = S.n_rows;

    stdout_printf("SVD => PCA (sparse)\n");
    FLUSH;
    mat U = SVD_results(0);
    vec s = SVD_results(1);
    mat V = SVD_results(2);

    int dim = U.n_cols;

    mat A = ones(S.n_rows, 1);
    mat B = -mat(trans(mean(S, 0)));

    field<mat> perturbed_SVD = perturbedSVD(SVD_results, A, B);

    return perturbed_SVD;
  }

  field<mat> SVD2PCA(mat &S, field<mat> SVD_results)
  {
    int n = S.n_rows;

    stdout_printf("SVD => PCA (dense)\n");
    FLUSH;
    mat U = SVD_results(0);
    vec s = SVD_results(1);
    mat V = SVD_results(2);

    int dim = U.n_cols;

    mat A = ones(S.n_rows, 1);
    mat B = -mat(trans(mean(S, 0)));

    field<mat> perturbed_SVD = perturbedSVD(SVD_results, A, B);

    return perturbed_SVD;
  }

  field<mat> reduce_kernel(sp_mat &S, int dim, int iter = 5, int seed = 0,
                           int SVD_algorithm = HALKO_ALG,
                           bool prenormalize = false,
                           int verbose = 1)
  {
    int n = S.n_rows;

    if (prenormalize)
      S = normalise(S, 2);

    stdout_printf("Computing reduced ACTION kernel (sparse):\n");
    FLUSH;

    stdout_printf("\tPerforming SVD on original matrix: ");
    FLUSH;
    vec s;
    mat U, V;
    field<mat> SVD_results(3);

    switch (SVD_algorithm)
    {
    case FULL_SVD:
      svd_econ(U, s, V, mat(S));
      SVD_results(0) = U;
      SVD_results(1) = s;
      SVD_results(2) = V;
      break;
    case IRLB_ALG:
      SVD_results = IRLB_SVD(S, dim, iter, seed, verbose);
      break;
    case HALKO_ALG:
      SVD_results = HalkoSVD(S, dim, iter, seed, verbose);
      break;
    case FENG_ALG:
      SVD_results = FengSVD(S, dim, iter, seed, verbose);
      break;
    default:
      stderr_printf("Unknown SVD algorithm chosen (%d). Switching to Halko.\n",
                    SVD_algorithm);
      FLUSH;
      SVD_results = HalkoSVD(S, dim, iter, seed, verbose);
      break;
    }

    // Update 1: Orthogonalize columns w.r.t. background (mean)
    vec mu = vec(mean(S, 1));
    vec v = mu / norm(mu, 2);
    vec a1 = v;
    vec b1 = -trans(S) * v;

    // Update 2: Center columns of orthogonalized matrix before performing SVD
    vec c = vec(trans(mean(S, 0)));
    double a1_mean = mean(a1);
    vec a2 = ones(S.n_rows);
    vec b2 = -(a1_mean * b1 + c);

    mat A = join_rows(a1, a2);
    mat B = join_rows(b1, b2);

    field<mat> perturbed_SVD = perturbedSVD(SVD_results, A, B);

    return perturbed_SVD;
  }

  field<mat> reduce_kernel(mat &S, int dim, int iter = 5, int seed = 0,
                           int SVD_algorithm = HALKO_ALG,
                           bool prenormalize = false,
                           int verbose = 1)
  {
    int n = S.n_rows;

    if (prenormalize)
      S = normalise(S, 2);

    stdout_printf("Computing reduced ACTION kernel (dense):\n");
    FLUSH;
    stdout_printf("\tPerforming SVD on original matrix: ");
    FLUSH;

    vec s;
    mat U, V;
    field<mat> SVD_results(3);
    switch (SVD_algorithm)
    {
    case FULL_SVD:
      svd_econ(U, s, V, S);
      SVD_results(0) = U;
      SVD_results(1) = s;
      SVD_results(2) = V;
      break;
    case IRLB_ALG:
      SVD_results = IRLB_SVD(S, dim, iter, seed, verbose);
      break;
    case HALKO_ALG:
      SVD_results = HalkoSVD(S, dim, iter, seed, verbose);
      break;
    case FENG_ALG:
      SVD_results = FengSVD(S, dim, iter, seed, verbose);
      break;
    default:
      stderr_printf("Unknown SVD algorithm chosen (%d). Switching to Halko.\n",
                    SVD_algorithm);
      FLUSH;
      SVD_results = HalkoSVD(S, dim, iter, seed, verbose);
      break;
    }

    // Update 1: Orthogonalize columns w.r.t. background (mean)
    vec mu = vec(mean(S, 1));
    vec v = mu / norm(mu, 2);
    vec a1 = v;
    vec b1 = -trans(S) * v;

    // Update 2: Center columns of orthogonalized matrix before performing SVD
    vec c = vec(trans(mean(S, 0)));
    double a1_mean = mean(a1);
    vec a2 = ones(S.n_rows);
    vec b2 = -(a1_mean * b1 + c);

    mat A = join_rows(a1, a2);
    mat B = join_rows(b1, b2);

    field<mat> perturbed_SVD = perturbedSVD(SVD_results, A, B);

    return perturbed_SVD;
  }

} // namespace ACTIONet
