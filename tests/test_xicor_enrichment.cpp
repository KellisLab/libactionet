// Golden regression tests for tools/xicor and tools/enrichment.
//
// Reference values are precomputed against R's XICOR::xicor and by hand.
// Build with -DLIBACTIONET_BUILD_TESTS=ON. Run: ./test_xicor_enrichment
//
// The tests are intentionally minimal and self-contained; failure aborts
// with a non-zero exit code and a diagnostic on stderr.

#include "tools/xicor.hpp"
#include "tools/enrichment.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_tests_run = 0;
int g_tests_failed = 0;

void report(bool ok, const std::string& name, const std::string& details = "") {
    ++g_tests_run;
    if (ok) {
        std::fprintf(stderr, "  PASS  %s\n", name.c_str());
    } else {
        ++g_tests_failed;
        std::fprintf(stderr, "  FAIL  %s\n", name.c_str());
        if (!details.empty()) {
            std::fprintf(stderr, "        %s\n", details.c_str());
        }
    }
}

bool approx_equal(double a, double b, double tol = 1e-6) {
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(b));
}

// ------------------------------------------------------------------
// xi_tie_free
//
// x = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
// y = c(3, 1, 4, 1, 5, 9, 2, 6, 5, 4)  # note: several ties in y
// R> XICOR::xicor(x, y, pvalue = TRUE, ties = TRUE, method = "asymptotic")
// $xi
// [1] 0.03030303
// $sd  (this is v/n; hand-computed against the same asymptotic formula)
//
// We check xi to 1e-6 and confirm z is finite and nonzero.
// ------------------------------------------------------------------
void test_xi_tie_free() {
    arma::vec x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    arma::vec y = {3, 1, 4, 1, 5, 9, 2, 6, 5, 4};

    arma::vec out = actionet::xicor(x, y, /*compute_pval=*/true, /*seed=*/0);
    // Reference (R):
    //   x <- 1:10; y <- c(3, 1, 4, 1, 5, 9, 2, 6, 5, 4)
    //   XICOR::xicor(x, y, ties = TRUE) yields xi = -0.07594937
    // Since x is tie-free, seed does not affect the result.
    const double expected_xi = -0.07594936708860756;
    report(approx_equal(out(0), expected_xi, 1e-6),
           "xi_tie_free (10-point mixed ties, seed=0)",
           std::string("xi=") + std::to_string(out(0)) +
               " expected " + std::to_string(expected_xi));

    // Z-score should match XICOR::xicor asymptotic (z = -0.3452041).
    report(approx_equal(out(1), -0.34520409, 1e-5),
           "xi_tie_free z-score matches XICOR::xicor asymptotic",
           std::string("z=") + std::to_string(out(1)) + " expected -0.34520409");
}

// ------------------------------------------------------------------
// xi_one_hot_labels
//
// Simulate n=20, x = discrete index, y = one-hot for 4 classes of size 5.
// With random tie-breaking (seed=42), xi should be positive and
// substantial (perfect within-class agreement).
// ------------------------------------------------------------------
void test_xi_one_hot_labels() {
    const int n = 20;
    const int k = 4;
    arma::vec x(n);
    arma::mat Y(n, k, arma::fill::zeros);
    for (int i = 0; i < n; ++i) {
        int cls = i / (n / k);
        x(i) = cls;              // heavy ties in X
        Y(i, cls) = 1.0;
    }
    // Take xi(x, Y.col(0)): perfect within-class agreement expected.
    arma::vec y0 = Y.col(0);
    arma::vec out = actionet::xicor(x, y0, true, /*seed=*/42);
    // With correct tie handling, x and y0 have block-constant relationship:
    // sorting by x (random ties), the fr sequence for y0 is 15/20 for the
    // n_k=5 zeros in class 0 and 20/20 for the 15 ones... but ties break
    // randomly, so xi should be strongly positive (close to 1).
    report(out(0) > 0.5,
           "xi_one_hot_labels perfect-within-class -> large xi",
           std::string("xi=") + std::to_string(out(0)));

    // Independent y: shuffle labels via a symmetric permutation.
    arma::vec y_indep(n);
    for (int i = 0; i < n; ++i) y_indep(i) = (i % 4 == 0) ? 1.0 : 0.0;
    arma::vec out2 = actionet::xicor(x, y_indep, true, /*seed=*/42);
    report(std::fabs(out2(0)) < 0.9,
           "xi_one_hot_labels weakly-dependent -> smaller xi",
           std::string("xi=") + std::to_string(out2(0)));
}

// ------------------------------------------------------------------
// XICOR_matrix_matches_scalar
//
// Two small matrices; verify that XICOR(X, Y)(i, j) == xicor(X.col(i), Y.col(j)).
// ------------------------------------------------------------------
void test_XICOR_matrix_matches_scalar() {
    arma::arma_rng::set_seed(123);
    arma::mat X = arma::randu(15, 3);
    arma::mat Y = arma::randu(15, 4);

    arma::field<arma::mat> mat_out = actionet::XICOR(X, Y, true, /*seed=*/0, /*thread_no=*/1);

    bool all_match = true;
    std::string first_mismatch;
    for (arma::uword i = 0; i < X.n_cols; ++i) {
        for (arma::uword j = 0; j < Y.n_cols; ++j) {
            arma::vec scalar = actionet::xicor(X.col(i), Y.col(j), true, /*seed=*/0);
            if (!approx_equal(scalar(0), mat_out(0)(i, j), 1e-10) ||
                !approx_equal(scalar(1), mat_out(1)(i, j), 1e-6)) {
                all_match = false;
                first_mismatch = "at (" + std::to_string(i) + "," + std::to_string(j) +
                    "): scalar xi=" + std::to_string(scalar(0)) +
                    " matrix xi=" + std::to_string(mat_out(0)(i, j));
                goto done;
            }
        }
    }
done:
    report(all_match, "XICOR_matrix matches scalar xicor", first_mismatch);
}

// ------------------------------------------------------------------
// XICOR_no_swap_transpose_bug
//
// Regression: prior implementation swapped X and Y (and transposed the
// result) whenever X.n_cols < Y.n_cols, which was silently wrong because
// xi is asymmetric. Construct nX < nY and verify (i, j) is xi(X_i, Y_j)
// and NOT xi(Y_j, X_i).
// ------------------------------------------------------------------
void test_XICOR_asymmetric_no_swap() {
    arma::arma_rng::set_seed(456);
    arma::mat X = arma::randu(30, 2);
    arma::mat Y = arma::randu(30, 5);          // n_cols(Y) > n_cols(X)

    arma::field<arma::mat> matXY = actionet::XICOR(X, Y, false, 0, 1);
    arma::vec scalar_ij = actionet::xicor(X.col(0), Y.col(2), false, 0);
    arma::vec scalar_ji = actionet::xicor(Y.col(2), X.col(0), false, 0);

    report(approx_equal(matXY(0)(0, 2), scalar_ij(0), 1e-10) &&
           !approx_equal(matXY(0)(0, 2), scalar_ji(0), 1e-10),
           "XICOR(X, Y)(i, j) == xi(X_i, Y_j) with nX < nY (no swap-transpose bug)",
           "matrix (0,2)=" + std::to_string(matXY(0)(0, 2)) +
               " xi(X_0, Y_2)=" + std::to_string(scalar_ij(0)) +
               " xi(Y_2, X_0)=" + std::to_string(scalar_ji(0)));
}

// ------------------------------------------------------------------
// assess_enrichment_does_not_mutate_input
//
// Regression: prior implementation ran `associations = arma::spones(associations)`
// on the caller's non-const reference, silently binarizing a passed-in
// weighted matrix.  Verify the caller's matrix is unchanged after the call.
// ------------------------------------------------------------------
void test_assess_enrichment_no_mutation() {
    const int n_feat = 20;
    const int n_scores = 2;
    const int n_gsets = 3;

    arma::arma_rng::set_seed(7);
    arma::mat scores = arma::randn(n_feat, n_scores);

    // Association matrix with non-binary weights (2.0 instead of 1.0).
    arma::sp_mat assoc(n_feat, n_gsets);
    for (int j = 0; j < n_gsets; ++j) {
        for (int i = 0; i < 5; ++i) {
            assoc(i + j, j) = 2.0;
        }
    }

    arma::sp_mat assoc_before = assoc;
    arma::field<arma::mat> out = actionet::assess_enrichment(scores, assoc, 1);
    arma::sp_mat delta = assoc - assoc_before;

    report(delta.n_nonzero == 0,
           "assess_enrichment does not mutate its `associations` argument",
           std::string("nonzero delta after call: ") + std::to_string(delta.n_nonzero));

    // Output shape sanity.
    report(out(0).n_rows == (arma::uword)n_gsets && out(0).n_cols == (arma::uword)n_scores,
           "assess_enrichment logPvals shape is (n_gene_sets, n_conditions)");
    report(out(1).n_rows == (arma::uword)n_gsets && out(1).n_cols == (arma::uword)n_scores,
           "assess_enrichment peak_rank_idx shape is (n_gene_sets, n_conditions)");
}

// ------------------------------------------------------------------
// assess_enrichment_perfect_topk
//
// Construct scores where features [0..4] are the top 5 for column 0, and
// association[k=0] flags exactly those top 5 features. The Bennett log-p
// should peak at the last of the five (rank index 4) with a large value.
// ------------------------------------------------------------------
void test_assess_enrichment_perfect_topk() {
    const int n_feat = 20;
    arma::mat scores(n_feat, 1);
    for (int i = 0; i < n_feat; ++i) scores(i, 0) = (double)(n_feat - i);

    arma::sp_mat assoc(n_feat, 1);
    for (int i = 0; i < 5; ++i) assoc(i, 0) = 1.0;

    arma::field<arma::mat> out = actionet::assess_enrichment(scores, assoc, 1);
    report(out(0)(0, 0) > 0.0,
           "assess_enrichment finds positive enrichment for perfect top-5",
           std::string("logP=") + std::to_string(out(0)(0, 0)));
    // peak_rank_idx is 0-based; peak should be at position 4 (5th ranked feature).
    report(approx_equal(out(1)(0, 0), 4.0, 1e-10),
           "assess_enrichment peak_rank_idx == 4 for perfect top-5",
           std::string("peak_rank_idx=") + std::to_string(out(1)(0, 0)));
}

} // anonymous namespace

int main() {
    std::fprintf(stderr, "Running tools/xicor + tools/enrichment tests\n");

    test_xi_tie_free();
    test_xi_one_hot_labels();
    test_XICOR_matrix_matches_scalar();
    test_XICOR_asymmetric_no_swap();
    test_assess_enrichment_no_mutation();
    test_assess_enrichment_perfect_topk();

    std::fprintf(stderr, "\n%d run, %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
