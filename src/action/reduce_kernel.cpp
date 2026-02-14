#include "action/reduce_kernel.hpp"
#include "decomposition/svd_main.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace actionet {
    namespace {
        template <typename T>
        void computeKernelPerturbationTermsInMemory(const T& S, arma::mat& A, arma::mat& B) {
            arma::vec mu = arma::vec(arma::mean(S, 1));
            double mu_norm = arma::norm(mu, 2);
            if (mu_norm <= std::numeric_limits<double>::epsilon()) {
                throw std::runtime_error("reduceKernel: mean vector has zero norm");
            }

            arma::vec a1 = mu / mu_norm;
            arma::vec b1 = -arma::trans(S) * a1;

            arma::vec c = arma::vec(arma::trans(arma::mean(S, 0)));
            double a1_mean = arma::mean(a1);
            arma::vec a2 = arma::ones(S.n_rows);
            arma::vec b2 = -(a1_mean * b1 + c);

            A = arma::join_rows(a1, a2);
            B = arma::join_rows(b1, b2);
        }
    } // namespace

    void computeKernelPerturbationTerms(const MatrixOperator& S, arma::mat& A, arma::mat& B) {
        arma::uword m = S.rows();
        arma::uword n = S.cols();
        if (m == 0 || n == 0) {
            throw std::runtime_error("reduceKernel_Operator: empty matrix");
        }

        arma::vec ones_n = arma::ones<arma::vec>(n);
        arma::vec mu_sum(m);
        S.matvec(ones_n, mu_sum);
        arma::vec mu = mu_sum / static_cast<double>(n);

        double mu_norm = arma::norm(mu, 2);
        if (mu_norm <= std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error("reduceKernel_Operator: mean vector has zero norm");
        }

        arma::vec a1 = mu / mu_norm;
        arma::vec b1_tmp(n);
        S.rmatvec(a1, b1_tmp);
        arma::vec b1 = -b1_tmp;

        arma::vec ones_m = arma::ones<arma::vec>(m);
        arma::vec c_sum(n);
        S.rmatvec(ones_m, c_sum);
        arma::vec c = c_sum / static_cast<double>(m);

        double a1_mean = arma::mean(a1);
        arma::vec a2 = arma::ones<arma::vec>(m);
        arma::vec b2 = -(a1_mean * b1 + c);

        A = arma::join_rows(a1, a2);
        B = arma::join_rows(b1, b2);
    }

    KernelReductionResult applyKernelPostSVD(const SVDResult& svd, const arma::mat& A, const arma::mat& B) {
        arma::field<arma::mat> svd_field = svdFieldFromResult(svd);
        arma::mat A_copy = A;
        arma::mat B_copy = B;
        arma::field<arma::mat> reduction = perturbedSVD(svd_field, A_copy, B_copy);

        KernelReductionResult out;
        out.sigma = arma::vec(reduction(1));

        double epsilon = 0.01 / std::sqrt(reduction(2).n_rows);
        arma::mat V = arma::round(reduction(2) / epsilon) * epsilon;
        for (arma::uword i = 0; i < V.n_cols; i++) {
            V.col(i) *= out.sigma(i);
        }

        out.S_r = V.t();
        out.V = reduction(0);
        out.A = reduction(3);
        out.B = reduction(4);
        return out;
    }

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

    KernelReductionResult reduceKernel_Operator(const MatrixOperator& S, int k, int max_it, int seed, bool verbose) {
#if defined(LIBACTIONET_BUILD_R) && LIBACTIONET_BUILD_R == 1
        (void)S;
        (void)k;
        (void)max_it;
        (void)seed;
        (void)verbose;
        throw std::runtime_error("reduceKernel_Operator is unavailable in R build mode");
#else
        if (verbose) {
            stdout_printf("Computing reduced ACTION kernel (operator/PRIMME):\n");
            FLUSH;
        }

        SVDResult svd = runSVD_PRIMME_Operator(S, k, max_it, seed, verbose);
        return reduceKernelFromSVD_Operator(S, svd, verbose);
#endif
    }

    KernelReductionResult reduceKernelFromSVD(const SVDResult& svd, const arma::mat& A, const arma::mat& B) {
        return applyKernelPostSVD(svd, A, B);
    }

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
