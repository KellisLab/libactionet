#include "utils_internal/utils_active_set.hpp"
#include "utils_internal/utils_action_numeric_policy.hpp"
#include "utils_internal/utils_small_dense.hpp"
#include <cassert>

// min(|| AX - B ||) s.t. simplex constraint

namespace actionet {

namespace small_dense = utils_internal::small_dense;
namespace numeric_policy = utils_internal::action_numeric_policy;

/* **************************
 * Active-Set Method with direct inversion, with update(matrix inversion lemma)
 * **************************/
arma::vec activeSet_arma(const arma::mat &M, const arma::vec &b, double lambda2, double epsilon) {
    int m = M.n_rows;
    int p = M.n_cols;
    int L = std::min(m, p) + 1;
    const bool inline_kernel = small_dense::use_inline_kernel(M.n_rows, M.n_cols);

    arma::vec c(M.n_cols);
    small_dense::gemv(true, m, p, -1.0, M.memptr(), m, b.memptr(),
                      0.0, c.memptr(), inline_kernel);

    double lam2sq = lambda2 * lambda2;

    arma::vec x(p);
    const double *pr_M = M.memptr();
    // constraint matrix
    arma::vec A = arma::ones(L);
    double *pr_A = A.memptr();

    // Non-Active Constraints Set
    arma::ivec NASet(L);
    NASet.ones();
    NASet = -NASet;
    arma::ivec NAMask(p);
    NAMask.zeros();

    int na;
    arma::vec xRed(L);
    arma::vec cRed(L);
    arma::mat MRed(m, L);
    double *pr_MRed = MRed.memptr();
    arma::mat GRed(L, L);
    // double *pr_GRed = GRed.memptr();
    arma::mat GRedinv(L, L);
    double *pr_GRedinv = GRedinv.memptr();

    // cold-start
    x.zeros();
    x[0] = double(1.0);
    // Non-Active Constraints Set
    NASet[0] = 0;
    NAMask[0] = 1;
    na = 1;
    xRed[0] = x[0];
    cRed[0] = c[0];
    small_dense::copy(m, pr_M, 1, pr_MRed, 1, inline_kernel);

    // BLAS GRed = MRedT * MRed + lam2sq (na = 1 for now)
    double coeff = small_dense::dot(m, pr_MRed, 1, pr_MRed, 1, inline_kernel)
                 + lam2sq;
    GRed(0, 0) = coeff;
    GRedinv(0, 0) = double(1.0) / GRed(0, 0);

    arma::vec Mx(M.n_rows);
    arma::vec Gplus(size(c));
    small_dense::gemv(false, m, p, 1.0, M.memptr(), m, x.memptr(),
                      0.0, Mx.memptr(), inline_kernel);
    small_dense::gemv(true, m, p, 1.0, M.memptr(), m, Mx.memptr(),
                      0.0, Gplus.memptr(), inline_kernel);
    Gplus += (lam2sq * x + c);

    double *pr_Gplus = Gplus.memptr();

    arma::vec gRed(L);
    double *pr_gRed = gRed.memptr();
    arma::vec MRedxRed(m);

    arma::vec GinvA(L);
    double *pr_GinvA = GinvA.memptr();
    arma::vec Ginvg(L);
    double *pr_Ginvg = Ginvg.memptr();
    arma::vec PRed(L);
    double *pr_PRed = PRed.memptr();
    arma::vec MRedPRed(m);
    double *pr_MRedPRed = MRedPRed.memptr();
    arma::vec UB(L);
    double *pr_UB = UB.memptr();
    arma::vec UAiB(L);
    double *pr_UAiB = UAiB.memptr();

    // main loop active set
    int iter = 0;
    while (iter <= 100 * p) {
        ++iter;
        // update of na, NASet, NAMask, xRed, cRed, Gplus, GRedinv, already done
        // now update MRed, gRed, GinvA, Ginvg  (no need to update GRed)
        // MRed
        for (int i = 0; i < na; ++i) {
            // BLAS copy first columns of M to MRed
            small_dense::copy(m, pr_M + m * NASet[i], 1,
                              pr_MRed + m * i, 1, inline_kernel);
        }
        // gRed
        for (int i = 0; i < na; ++i) {
            gRed[i] = Gplus[NASet[i]];
        }

        // GinvA
        // BLAS GinvA = GRedinv * A (ARed == A)
        small_dense::symv_upper(na, 1.0, pr_GRedinv, L, pr_A, 0.0,
                                pr_GinvA, inline_kernel);

        // Ginvg
        // BLAS Ginvg = GRedinv * gRed
        small_dense::symv_upper(na, 1.0, pr_GRedinv, L, pr_gRed, 0.0,
                                pr_Ginvg, inline_kernel);
        double sGinvg = double();
        double sGinvA = double();
        for (int i = 0; i < na; ++i) {
            sGinvg += Ginvg[i];
            sGinvA += GinvA[i];
        }
        double lambdaS = sGinvg / sGinvA;
        // BLAS PRed = GinvA * lambdaS - Ginvg
        small_dense::copy(na, pr_GinvA, 1, pr_PRed, 1, inline_kernel);
        small_dense::scale(na, lambdaS, pr_PRed, 1, inline_kernel);
        small_dense::axpy(na, -1.0, pr_Ginvg, 1, pr_PRed, 1, inline_kernel);

        double maxPRed = std::abs(PRed[0]);
        for (int i = 0; i < na; ++i) {
            if (std::abs(PRed[i]) > maxPRed)
                maxPRed = std::abs(PRed[i]);
        }
        if (maxPRed < numeric_policy::active_set_zero_step_tolerance) {
            // P = 0, no advance possible
            bool isOpt = true;
            double lamMin = -epsilon;
            int indexMin = -1;
            for (int i = 0; i < p; ++i) {
                if (!NAMask[i] && Gplus[i] - lambdaS < lamMin) {
                    isOpt = false;
                    lamMin = Gplus[i] - lambdaS;
                    indexMin = i;
                }
            }

            if (isOpt) {
                // Got the optimal, STOP!
                return (x);
            } else {
                // Add one constraint
                NAMask[indexMin] = 1;
                NASet[na] = indexMin;
                xRed[na] = x[indexMin];
                cRed[na] = c[indexMin];
                // Gplus inchange

                // update GRedinv
                // BLAS UB = MRed.double * M[:, indexMin]
                small_dense::gemv(true, m, na, 1.0, pr_MRed, m,
                                  pr_M + indexMin * m, 0.0, pr_UB,
                                  inline_kernel);
                // BLAS UC = M[:,indexMin].double* M[:, indexMin]
                double UC = small_dense::dot(m, pr_M + indexMin * m, 1,
                                             pr_M + indexMin * m, 1,
                                             inline_kernel) + lam2sq;
                // BLAS UAiB = GRedinv * UB
                small_dense::symv_upper(na, 1.0, pr_GRedinv, L, pr_UB,
                                        0.0, pr_UAiB, inline_kernel);
                double USi = 1.0 / (UC - small_dense::dot(
                    na, pr_UB, 1, pr_UAiB, 1, inline_kernel));
                // GRedinv (restricted) += USi * UAiB*UAiB
                // replace cblas_syr(CblasColMajor,CblasUpper,na,USi, pr_UAiB, 1,
                // pr_GRedinv, L);
                small_dense::rank_one_update(na, na, USi, pr_UAiB, pr_UAiB,
                                             pr_GRedinv, L, inline_kernel);
                // copy -UAiB*USi, -UAiB.double*USi, USi to GRedinv
                small_dense::copy(na, pr_UAiB, 1, pr_GRedinv + na * L, 1,
                                  inline_kernel);
                small_dense::scale(na, -USi, pr_GRedinv + na * L, 1,
                                   inline_kernel);
                small_dense::copy(na, pr_UAiB, 1, pr_GRedinv + na, L,
                                  inline_kernel);
                small_dense::scale(na, -USi, pr_GRedinv + na, L,
                                   inline_kernel);
                GRedinv(na, na) = USi;

                na += 1;
                assert(na <= L);
            }
        } else {
            // P != 0, can advance
            int indexMin = -1;
            auto alphaMin = double(1.0);
            for (int i = 0; i < na; ++i) {
                if (PRed[i] < 0 && -xRed[i] / PRed[i] < alphaMin) {
                    indexMin = i;
                    alphaMin = -xRed[i] / PRed[i];
                }
            }
            // update x and Gplus
            small_dense::scale(na, std::min(1.0, alphaMin), pr_PRed, 1,
                               inline_kernel);
            for (int i = 0; i < na; ++i) {
                x[NASet[i]] += PRed[i];
                xRed[i] = x[NASet[i]];
                // BLAS Gplus += M.double * M[:, NASet[i]] * cAdv
                // cblas_dgemv(CblasColMajor,CblasTrans,m,p,cAdv,pr_M,m,
                // pr_M+NASet[i]*m,1,double(1.0),pr_Gplus,1);
                // BLAS Gplus
                Gplus[NASet[i]] += PRed[i] * lam2sq;
            }
            // Gplus += M.double * MRed * (scaled PRed)
            small_dense::gemv(false, m, na, 1.0, pr_MRed, m, pr_PRed,
                              0.0, pr_MRedPRed, inline_kernel);
            small_dense::gemv(true, m, p, 1.0, pr_M, m, pr_MRedPRed,
                              1.0, pr_Gplus, inline_kernel);

            // delete one constraint or not?
            if (indexMin != -1) {
                // give true 0
                // x[NASet[indexMin]] = double();
                // delete one constraint
                NAMask[NASet[indexMin]] = 0;
                // downdate remove this -1;
                na -= 1;
                for (int i = indexMin; i < na; ++i) {
                    NASet[i] = NASet[i + 1];
                    xRed[i] = xRed[i + 1];
                    cRed[i] = cRed[i + 1];
                }
                NASet[na] = -1;
                xRed[na] = double();
                cRed[na] = double();
                // PRed also
                PRed[na] = double();

                // downdate GRedinv
                double UCi = double(1.0) / GRedinv(indexMin, indexMin);
                // BLAS UB = GRedinv[ALL\indexMin,indexMin]
                small_dense::copy(na + 1, pr_GRedinv + indexMin * L, 1,
                                  pr_UB, 1, inline_kernel);
                for (int i = indexMin; i < na; ++i)
                    UB[i] = UB[i + 1];
                UB[na] = double();
                // get (GRedinv translated)
                // column first
                for (int i = indexMin; i < na; ++i)
                    small_dense::copy(na + 1, pr_GRedinv + (i + 1) * L, 1,
                                      pr_GRedinv + i * L, 1, inline_kernel);
                // row then
                for (int i = indexMin; i < na; ++i)
                    small_dense::copy(na + 1, pr_GRedinv + i + 1, L,
                                      pr_GRedinv + i, L, inline_kernel);

                // BLAS GRedinv = (GRedinv translated) - UB*UB.double*UCi
                // replace cblas_syr(CblasColMajor,CblasUpper,na,-UCi, pr_UB, 1, pr_GRedinv, L);
                small_dense::rank_one_update(na, na, -UCi, pr_UB, pr_UB,
                                             pr_GRedinv, L, inline_kernel);
            }
        }
    }
    return (x);
}

// Active-Set Method with direct inversion, with update(matrix inversion lemma)
// Memorize M.double* M + lam2sq = G
arma::vec activeSetS_arma(const arma::mat &M, const arma::vec &b, const arma::mat &G, double lambda2, double epsilon) {
    int m = M.n_rows;
    int p = M.n_cols;
    int L = std::min(m, p) + 1;
    double lam2sq = lambda2 * lambda2;
    const bool inline_kernel = small_dense::use_inline_kernel(M.n_rows, M.n_cols);
    const double *pr_G = G.memptr();

    /*
    mat Mt = trans(M);
    arma::vec c0 = -Mt*b;
    */
    arma::vec c(M.n_cols);
    small_dense::gemv(true, m, p, -1.0, M.memptr(), m, b.memptr(),
                      0.0, c.memptr(), inline_kernel);

    arma::vec x(p);
    const double *pr_M = M.memptr();
    // constraint matrix
    arma::vec A = arma::ones(L);
    double *pr_A = A.memptr();

    // Non-Active Constraints Set
    arma::ivec NASet(L);
    NASet.ones();
    NASet = -NASet;
    arma::ivec NAMask(p);
    NAMask.zeros();

    int na;
    arma::vec xRed(L);
    arma::vec cRed(L);
    arma::mat MRed(m, L);
    double *pr_MRed = MRed.memptr();
    arma::mat GRed(L, L);
    arma::mat GRedinv(L, L);
    double *pr_GRedinv = GRedinv.memptr();
    arma::mat MTMRed(p, L);
    double *pr_MTMRed = MTMRed.memptr();

    x.zeros();
    x[0] = double(1.0);
    // Non-Active Constraints Set
    NASet[0] = 0;
    NAMask[0] = 1;
    na = 1;
    xRed[0] = x[0];
    cRed[0] = c[0];
    small_dense::copy(m, pr_M, 1, pr_MRed, 1, inline_kernel);

    // BLAS GRed = MRedT * MRed + lam2sq (na = 1 for now)
    double coeff = small_dense::dot(m, pr_MRed, 1, pr_MRed, 1, inline_kernel)
                 + lam2sq;
    GRed(0, 0) = coeff;
    GRedinv(0, 0) = double(1.0) / GRed(0, 0);
    small_dense::copy(p, pr_G, 1, pr_MTMRed, 1, inline_kernel);

    // arma::vec Gplus = G*x + c;
    arma::vec Gplus = c;
    small_dense::gemv(false, p, p, 1.0, G.memptr(), p, x.memptr(),
                      1.0, Gplus.memptr(), inline_kernel);

    double *pr_Gplus = Gplus.memptr();

    arma::vec gRed(L);
    double *pr_gRed = gRed.memptr();
    arma::vec MRedxRed(m);

    arma::vec GinvA(L);
    double *pr_GinvA = GinvA.memptr();
    arma::vec Ginvg(L);
    double *pr_Ginvg = Ginvg.memptr();
    arma::vec PRed(L);
    double *pr_PRed = PRed.memptr();
    arma::vec UB(L);
    double *pr_UB = UB.memptr();
    arma::vec UAiB(L);
    double *pr_UAiB = UAiB.memptr();
    // main loop active set
    int iter = 0;
    while (iter <= 100 * p) {
        ++iter;
        // update of na, NASet, NAMask, xRed, cRed, Gplus, GRedinv, already done
        // now update gRed, GinvA, Ginvg  (no need to update GRed)

        // gRed
        // gRed = Gplus[NASet]
        for (int i = 0; i < na; ++i) {
            gRed[i] = Gplus[NASet[i]];
        }

        // GinvA
        // BLAS GinvA = GRedinv * A (ARed == A)
        small_dense::symv_upper(na, 1.0, pr_GRedinv, L, pr_A, 0.0,
                                pr_GinvA, inline_kernel);
        // Ginvg
        // BLAS Ginvg = GRedinv * gRed
        small_dense::symv_upper(na, 1.0, pr_GRedinv, L, pr_gRed, 0.0,
                                pr_Ginvg, inline_kernel);
        double sGinvg = double();
        double sGinvA = double();
        for (int i = 0; i < na; ++i) {
            sGinvg += Ginvg[i];
            sGinvA += GinvA[i];
        }
        double lambdaS = sGinvg / sGinvA;
        // BLAS PRed = GinvA * lambdaS - Ginvg
        small_dense::copy(na, pr_GinvA, 1, pr_PRed, 1, inline_kernel);
        small_dense::scale(na, lambdaS, pr_PRed, 1, inline_kernel);
        small_dense::axpy(na, -1.0, pr_Ginvg, 1, pr_PRed, 1, inline_kernel);

        double maxPRed = std::abs(PRed[0]);
        for (int i = 0; i < na; ++i) {
            if (std::abs(PRed[i]) > maxPRed)
                maxPRed = std::abs(PRed[i]);
        }
        if (maxPRed < numeric_policy::active_set_zero_step_tolerance) {
            // P = 0, no advance possible
            bool isOpt = true;
            double lamMin = -epsilon;
            int indexMin = -1;
            for (int i = 0; i < p; ++i) {
                if (!NAMask[i] && Gplus[i] - lambdaS < lamMin) {
                    isOpt = false;
                    lamMin = Gplus[i] - lambdaS;
                    indexMin = i;
                }
            }

            if (isOpt) {
                // Got the optimal, STOP!
                return (x);
            } else {
                // Add one constraint
                NAMask[indexMin] = 1;
                NASet[na] = indexMin;
                xRed[na] = x[indexMin];
                cRed[na] = c[indexMin];
                // Gplus inchange
                // update MTMRed
                small_dense::copy(p, pr_G + indexMin * p, 1,
                                  pr_MTMRed + na * p, 1, inline_kernel);

                // update GRedinv
                // BLAS UB = MRed.double * M[:, indexMin]
                // Use G instead here
                for (int i = 0; i < na; ++i) {
                    UB[i] = G(NASet[i], indexMin);
                }
                // BLAS UC = M[:,indexMin].double* M[:, indexMin]
                double UC = G(indexMin, indexMin);
                // BLAS UAiB = GRedinv * UB
                small_dense::symv_upper(na, 1.0, pr_GRedinv, L, pr_UB,
                                        0.0, pr_UAiB, inline_kernel);
                double USi = 1.0 / (UC - small_dense::dot(
                    na, pr_UB, 1, pr_UAiB, 1, inline_kernel));
                // GRedinv (restricted) += USi * UAiB*UAiB
                // replace cblas_syr(CblasColMajor,CblasUpper,na,USi, pr_UAiB, 1,
                // pr_GRedinv, L);
                small_dense::rank_one_update(na, na, USi, pr_UAiB, pr_UAiB,
                                             pr_GRedinv, L, inline_kernel);
                // copy -UAiB*USi, -UAiB.double*USi, USi to GRedinv
                small_dense::copy(na, pr_UAiB, 1, pr_GRedinv + na * L, 1,
                                  inline_kernel);
                small_dense::scale(na, -USi, pr_GRedinv + na * L, 1,
                                   inline_kernel);
                small_dense::copy(na, pr_UAiB, 1, pr_GRedinv + na, L,
                                  inline_kernel);
                small_dense::scale(na, -USi, pr_GRedinv + na, L,
                                   inline_kernel);
                GRedinv(na, na) = USi;

                na += 1;
                assert(na <= L);
            }
        } else {
            // P != 0, can advance
            int indexMin = -1;
            auto alphaMin = double(1.0);
            for (int i = 0; i < na; ++i) {
                if (PRed[i] < 0 && -xRed[i] / PRed[i] < alphaMin) {
                    indexMin = i;
                    alphaMin = -xRed[i] / PRed[i];
                }
            }
            // update x and Gplus
            small_dense::scale(na, std::min(1.0, alphaMin), pr_PRed, 1,
                               inline_kernel);
            for (int i = 0; i < na; ++i) {
                x[NASet[i]] += PRed[i];
                xRed[i] = x[NASet[i]];
            }
            // Gplus += MTMRed * (scaled PRed)
            small_dense::gemv(false, p, na, 1.0, pr_MTMRed, p, pr_PRed,
                              1.0, pr_Gplus, inline_kernel);

            // delete one constraint or not?
            if (indexMin != -1) {
                // give true 0
                // x[NASet[indexMin]] = double();
                // delete one constraint
                NAMask[NASet[indexMin]] = 0;
                // downdate remove this -1;
                na -= 1;
                for (int i = indexMin; i < na; ++i) {
                    NASet[i] = NASet[i + 1];
                    xRed[i] = xRed[i + 1];
                    cRed[i] = cRed[i + 1];
                }
                NASet[na] = -1;
                xRed[na] = double();
                cRed[na] = double();
                // PRed also
                PRed[na] = double();

                // downdate MTMRed
                for (int i = indexMin; i < na; ++i)
                    small_dense::copy(p, pr_MTMRed + (i + 1) * p, 1,
                                      pr_MTMRed + i * p, 1, inline_kernel);

                // downdate GRedinv
                double UCi = double(1.0) / GRedinv(indexMin, indexMin);
                // BLAS UB = GRedinv[ALL\indexMin,indexMin]
                small_dense::copy(na + 1, pr_GRedinv + indexMin * L, 1,
                                  pr_UB, 1, inline_kernel);
                for (int i = indexMin; i < na; ++i)
                    UB[i] = UB[i + 1];
                UB[na] = double();
                // get (GRedinv translated)
                // column first
                for (int i = indexMin; i < na; ++i)
                    small_dense::copy(na + 1, pr_GRedinv + (i + 1) * L, 1,
                                      pr_GRedinv + i * L, 1, inline_kernel);
                // row then
                for (int i = indexMin; i < na; ++i)
                    small_dense::copy(na + 1, pr_GRedinv + i + 1, L,
                                      pr_GRedinv + i, L, inline_kernel);

                // BLAS GRedinv = (GRedinv translated) - UB*UB.double*UCi
                // replace cblas_syr(CblasColMajor,CblasUpper,na,-UCi, pr_UB, 1,
                // pr_GRedinv, L);
                small_dense::rank_one_update(na, na, -UCi, pr_UB, pr_UB,
                                             pr_GRedinv, L, inline_kernel);
            }
        }
    }
    return (x);
}

} // namespace actionet
