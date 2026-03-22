#include "action/reduce_kernel.hpp"
#include "decomposition/matrix_operator.hpp"
#include "decomposition/orthogonalization.hpp"
#include "utils_internal/utils_decomp.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(LIBACTIONET_BUILD_R)
#include <Rembedded.h>
#endif

namespace {

constexpr double kTol = 1e-8;

void require_true(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string shape_string(const arma::mat& x) {
    std::ostringstream oss;
    oss << x.n_rows << "x" << x.n_cols;
    return oss.str();
}

std::string shape_string(const arma::vec& x) {
    std::ostringstream oss;
    oss << x.n_elem;
    return oss.str();
}

void expect_mat_close(const std::string& name, const arma::mat& actual, const arma::mat& expected) {
    require_true(actual.n_rows == expected.n_rows && actual.n_cols == expected.n_cols,
                 name + ": shape mismatch (" + shape_string(actual) + " vs " + shape_string(expected) + ")");

    if (!arma::approx_equal(actual, expected, "absdiff", kTol)) {
        const double max_abs = arma::abs(actual - expected).max();
        std::ostringstream oss;
        oss << name << ": value mismatch (max abs diff = " << max_abs << ")";
        throw std::runtime_error(oss.str());
    }
}

void expect_vec_close(const std::string& name, const arma::vec& actual, const arma::vec& expected) {
    require_true(actual.n_elem == expected.n_elem,
                 name + ": length mismatch (" + shape_string(actual) + " vs " + shape_string(expected) + ")");

    if (!arma::approx_equal(actual, expected, "absdiff", kTol)) {
        const double max_abs = arma::abs(actual - expected).max();
        std::ostringstream oss;
        oss << name << ": value mismatch (max abs diff = " << max_abs << ")";
        throw std::runtime_error(oss.str());
    }
}

arma::mat make_expression_matrix() {
    arma::mat S(7, 4);
    S(0, 0) = 1.0; S(0, 1) = 0.0; S(0, 2) = 2.0; S(0, 3) = 0.0;
    S(1, 0) = 0.0; S(1, 1) = 3.0; S(1, 2) = 0.0; S(1, 3) = 1.0;
    S(2, 0) = 4.0; S(2, 1) = 0.0; S(2, 2) = 5.0; S(2, 3) = 0.0;
    S(3, 0) = 0.0; S(3, 1) = 2.0; S(3, 2) = 0.0; S(3, 3) = 6.0;
    S(4, 0) = 1.0; S(4, 1) = 1.0; S(4, 2) = 0.0; S(4, 3) = 0.0;
    S(5, 0) = 0.0; S(5, 1) = 0.0; S(5, 2) = 3.0; S(5, 3) = 2.0;
    S(6, 0) = 2.0; S(6, 1) = 1.0; S(6, 2) = 1.0; S(6, 3) = 0.0;
    return S;
}

arma::mat make_design_matrix() {
    arma::mat design(7, 2);
    design(0, 0) = 1.0; design(0, 1) = 0.0;
    design(1, 0) = 1.0; design(1, 1) = 0.0;
    design(2, 0) = 1.0; design(2, 1) = 0.0;
    design(3, 0) = 0.0; design(3, 1) = 1.0;
    design(4, 0) = 0.0; design(4, 1) = 1.0;
    design(5, 0) = 0.0; design(5, 1) = 1.0;
    design(6, 0) = 0.5; design(6, 1) = 0.5;
    return design;
}

arma::mat make_basal_state() {
    arma::mat basal(4, 2);
    basal(0, 0) = 1.0; basal(0, 1) = 0.0;
    basal(1, 0) = 0.0; basal(1, 1) = 1.0;
    basal(2, 0) = 1.0; basal(2, 1) = 1.0;
    basal(3, 0) = 0.5; basal(3, 1) = 2.0;
    return basal;
}

actionet::SVDResult svd_from_reduction(const actionet::KernelReductionResult& reduction) {
    actionet::SVDResult svd;
    svd.U = reduction.S_r;
    svd.sigma = reduction.sigma;
    svd.V = reduction.U;

    for (arma::uword i = 0; i < svd.sigma.n_elem; ++i) {
        if (std::abs(svd.sigma(i)) <= std::numeric_limits<double>::epsilon()) {
            svd.U.col(i).zeros();
        } else {
            svd.U.col(i) /= svd.sigma(i);
        }
    }

    return svd;
}

std::optional<actionet::PerturbedSVDResult> prior_from_reduction(const actionet::KernelReductionResult& reduction) {
    if (reduction.A.n_elem == 0 && reduction.B.n_elem == 0) {
        return std::nullopt;
    }

    actionet::PerturbedSVDResult prior;
    prior.A = reduction.B;  // cells
    prior.B = reduction.A;  // genes
    return prior;
}

actionet::KernelReductionResult reduction_from_perturbed(const actionet::PerturbedSVDResult& perturbed) {
    actionet::KernelReductionResult out;
    out.sigma = perturbed.sigma;
    out.S_r = perturbed.U;
    for (arma::uword i = 0; i < out.S_r.n_cols; ++i) {
        out.S_r.col(i) *= out.sigma(i);
    }
    out.U = perturbed.V;
    out.A = perturbed.B;
    out.B = perturbed.A;
    return out;
}

actionet::KernelReductionResult reference_batch(const arma::mat& S,
                                                const actionet::KernelReductionResult& reduction,
                                                const arma::mat& design) {
    actionet::SVDResult svd = svd_from_reduction(reduction);
    std::optional<actionet::PerturbedSVDResult> prior = prior_from_reduction(reduction);
    const actionet::PerturbedSVDResult* prior_ptr = prior ? &(*prior) : nullptr;

    arma::mat Z = arma::mat(S.t() * design);
    gram_schmidt(Z);
    arma::mat B = -arma::mat(S * Z);

    arma::vec mu_A = arma::vec(arma::trans(arma::mean(Z, 0)));
    arma::vec mu = B * mu_A;
    arma::mat A_aug = arma::join_rows(arma::ones(Z.n_rows), Z);  // genes
    arma::mat B_aug = arma::join_rows(-mu, B);                   // cells

    actionet::PerturbedSVDResult perturbed = actionet::perturbedSVD(svd, B_aug, A_aug, prior_ptr);
    return reduction_from_perturbed(perturbed);
}

actionet::KernelReductionResult reference_basal(const arma::mat& S,
                                                const actionet::KernelReductionResult& reduction,
                                                const arma::mat& basal_state) {
    actionet::SVDResult svd = svd_from_reduction(reduction);
    std::optional<actionet::PerturbedSVDResult> prior = prior_from_reduction(reduction);
    const actionet::PerturbedSVDResult* prior_ptr = prior ? &(*prior) : nullptr;

    arma::mat Z = basal_state;
    gram_schmidt(Z);
    arma::mat B = -arma::mat(S * Z);

    arma::vec mu_A = arma::vec(arma::trans(arma::mean(Z, 0)));
    arma::vec mu = B * mu_A;
    arma::mat A_aug = arma::join_rows(arma::ones(Z.n_rows), Z);  // genes
    arma::mat B_aug = arma::join_rows(-mu, B);                   // cells

    actionet::PerturbedSVDResult perturbed = actionet::perturbedSVD(svd, B_aug, A_aug, prior_ptr);
    return reduction_from_perturbed(perturbed);
}

void expect_public_shapes(const std::string& name,
                          const actionet::KernelReductionResult& reduction,
                          arma::uword n_cells,
                          arma::uword n_genes,
                          arma::uword k,
                          arma::uword p) {
    require_true(reduction.S_r.n_rows == n_cells && reduction.S_r.n_cols == k,
                 name + ": S_r shape mismatch");
    require_true(reduction.U.n_rows == n_genes && reduction.U.n_cols == k,
                 name + ": U shape mismatch");
    require_true(reduction.sigma.n_elem == k, name + ": sigma length mismatch");
    require_true(reduction.A.n_rows == n_genes && reduction.A.n_cols == p,
                 name + ": A shape mismatch");
    require_true(reduction.B.n_rows == n_cells && reduction.B.n_cols == p,
                 name + ": B shape mismatch");
}

void compare_reductions(const std::string& name,
                        const actionet::KernelReductionResult& actual,
                        const actionet::KernelReductionResult& expected) {
    expect_mat_close(name + " S_r", actual.S_r, expected.S_r);
    expect_vec_close(name + " sigma", actual.sigma, expected.sigma);
    expect_mat_close(name + " U", actual.U, expected.U);
    expect_mat_close(name + " A", actual.A, expected.A);
    expect_mat_close(name + " B", actual.B, expected.B);
}

template <typename MatrixT, typename OperatorT>
void run_case(const std::string& label, MatrixT& S, OperatorT& op,
              const arma::mat& S_dense, const arma::mat& design, const arma::mat& basal_state) {
    constexpr int k = 2;
    std::cout << label << ": reduceKernel" << std::endl;
    arma::field<arma::mat> reduction_field = actionet::reduceKernel(S, k, ALG_IRLB, 1000, 42, false);
    actionet::KernelReductionResult reduction = actionet::kernelResultFromField(reduction_field);

    expect_public_shapes(label + " reduceKernel", reduction, S_dense.n_rows, S_dense.n_cols, k, 2);

    std::cout << label << ": batch in-memory" << std::endl;
    arma::mat design_copy = design;
    arma::field<arma::mat> batch_field = actionet::orthogonalizeBatchEffect(S, reduction_field, design_copy);
    actionet::KernelReductionResult batch_in_memory = actionet::kernelResultFromField(batch_field);
    std::cout << label << ": batch reference" << std::endl;
    actionet::KernelReductionResult batch_reference = reference_batch(S_dense, reduction, design);
    expect_public_shapes(label + " batch in-memory", batch_in_memory, S_dense.n_rows, S_dense.n_cols, k, 5);
    compare_reductions(label + " batch in-memory", batch_in_memory, batch_reference);

    std::cout << label << ": batch operator" << std::endl;
    actionet::KernelReductionResult batch_operator = actionet::orthogonalizeBatchEffect_Operator(op, reduction, design);
    expect_public_shapes(label + " batch operator", batch_operator, S_dense.n_rows, S_dense.n_cols, k, 5);
    compare_reductions(label + " batch operator", batch_operator, batch_reference);

    std::cout << label << ": basal in-memory" << std::endl;
    arma::mat basal_copy = basal_state;
    arma::field<arma::mat> basal_field = actionet::orthogonalizeBasal(S, reduction_field, basal_copy);
    actionet::KernelReductionResult basal_in_memory = actionet::kernelResultFromField(basal_field);
    std::cout << label << ": basal reference" << std::endl;
    actionet::KernelReductionResult basal_reference = reference_basal(S_dense, reduction, basal_state);
    expect_public_shapes(label + " basal in-memory", basal_in_memory, S_dense.n_rows, S_dense.n_cols, k, 5);
    compare_reductions(label + " basal in-memory", basal_in_memory, basal_reference);

    std::cout << label << ": basal operator" << std::endl;
    actionet::KernelReductionResult basal_operator = actionet::orthogonalizeBasal_Operator(op, reduction, basal_state);
    expect_public_shapes(label + " basal operator", basal_operator, S_dense.n_rows, S_dense.n_cols, k, 5);
    compare_reductions(label + " basal operator", basal_operator, basal_reference);
}

} // namespace

int main() {
    bool r_initialized = false;

#if defined(LIBACTIONET_BUILD_R)
    setenv("R_HOME", VALIDATE_PLAN02_R_HOME, 0);
    char arg0[] = "validate_plan02_core";
    char arg1[] = "--vanilla";
    char* r_argv[] = {arg0, arg1};
    Rf_initEmbeddedR(2, r_argv);
    r_initialized = true;
#endif

    try {
        arma::mat S_dense = make_expression_matrix();
        arma::sp_mat S_sparse(S_dense);
        arma::mat design = make_design_matrix();
        arma::mat basal_state = make_basal_state();

        actionet::DenseMatrixOperator dense_op(S_dense);
        run_case("dense", S_dense, dense_op, S_dense, design, basal_state);

        actionet::SparseMatrixOperator sparse_op(S_sparse);
        run_case("sparse", S_sparse, sparse_op, S_dense, design, basal_state);

        std::cout << "validate_plan02_core: PASS" << std::endl;
#if defined(LIBACTIONET_BUILD_R)
        if (r_initialized) {
            Rf_endEmbeddedR(0);
        }
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "validate_plan02_core: FAIL: " << e.what() << std::endl;
#if defined(LIBACTIONET_BUILD_R)
        if (r_initialized) {
            Rf_endEmbeddedR(1);
        }
#endif
        return 1;
    }
}
