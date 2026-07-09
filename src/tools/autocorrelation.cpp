#include "tools/autocorrelation.hpp"
#include "tools/matrix_transform.hpp"
#include "utils_internal/utils_parallel.hpp"
#include <random>
#include <numeric>

// Thread-safe permutation using per-call seeded RNG
static arma::uvec thread_safe_randperm(int n, unsigned int seed) {
    std::vector<arma::uword> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    return arma::uvec(idx.data(), n);
}

namespace {
    // Shared permutation-based null estimator for autocorrelation
    // statistics.  Computes @p stat = diag(scores' * op * scores) exactly,
    // then permutes rows of @p normalized_scores independently for each of
    // @p perm_no draws and re-evaluates the same quadratic form.  Returns
    // stat, mu, sigma, z of the null distribution.
    //
    // @param op                 sparse operator (G for Moran, L for Geary)
    // @param normalized_scores  cells x scores_no
    // @param perm_no            number of permutations (>= 0)
    // @param thread_no          OpenMP thread hint
    // @param seed_bump          per-call seed offset (distinguishes callers)
    void run_permutation_stat(const arma::sp_mat& op,
                              const arma::mat& normalized_scores,
                              int perm_no, int thread_no,
                              unsigned int seed_bump,
                              arma::vec& stat,
                              arma::vec& mu,
                              arma::vec& sigma,
                              arma::vec& z) {
        const int nV = op.n_rows;
        const int scores_no = normalized_scores.n_cols;

        stat.zeros(scores_no);
        int threads_use = actionet::get_num_threads(scores_no, thread_no);
        #pragma omp parallel for num_threads(threads_use)
        for (unsigned int i = 0; i < static_cast<unsigned int>(scores_no); ++i) {
            arma::vec x = normalized_scores.col(i);
            stat(i) = arma::dot(x, op * x);
        }

        mu.zeros(scores_no);
        sigma.zeros(scores_no);
        z.zeros(scores_no);
        if (perm_no <= 0) return;

        arma::mat rand_stats(scores_no, perm_no, arma::fill::zeros);
        threads_use = actionet::get_num_threads(perm_no, thread_no);
        #pragma omp parallel for num_threads(threads_use)
        for (unsigned int j = 0; j < static_cast<unsigned int>(perm_no); ++j) {
            arma::uvec perm = thread_safe_randperm(nV, j * 1000003u + seed_bump);
            arma::mat score_permuted = normalized_scores.rows(perm);

            arma::vec v(scores_no);
            for (int i = 0; i < scores_no; ++i) {
                arma::vec rand_x = score_permuted.col(i);
                v(i) = arma::dot(rand_x, op * rand_x);
            }
            rand_stats.col(j) = v;
        }

        mu = arma::mean(rand_stats, 1);
        sigma = arma::stddev(rand_stats, 0, 1);
        z = (stat - mu) / sigma;
        z.replace(arma::datum::nan, 0);
    }
} // namespace

namespace actionet {
    arma::field<arma::vec>
        autocorrelation_Moran_parametric(const arma::sp_mat& G, const arma::mat& scores, int normalization_method,
                                         int thread_no) {
        double nV = G.n_rows;
        int scores_no = scores.n_cols;
        stdout_printf("Normalizing scores (method=%d) ... ", normalization_method);
        arma::mat normalized_scores = normalize_scores(scores, normalization_method, thread_no);
        stdout_printf("done\n");
        FLUSH;

        stdout_printf("Computing auto-correlation over network ... ");

        double W = arma::accu(G);
        double Wsq = W * W;

        arma::vec norm_sq = arma::vec(arma::trans(arma::sum(arma::square(normalized_scores))));
        arma::vec norm_factors = nV / (W * norm_sq);
        norm_factors.replace(arma::datum::nan, 0); // replace each NaN with 0

        arma::vec stat = arma::zeros(scores_no);
        int threads_use = get_num_threads(scores_no, thread_no);
        #pragma omp parallel for num_threads(threads_use)
        for (unsigned int i = 0; i < scores_no; i++) {
            arma::vec x = normalized_scores.col(i);
            double y = dot(x, G * x);
            stat(i) = y;
        }

        stat = stat % norm_factors;

        stdout_printf("done\n");
        FLUSH;

        arma::vec mu = -arma::ones(scores_no) / (nV - 1);

        arma::sp_mat Gsym = (G + arma::trans(G));
        double S1 = 0.5 * arma::accu(arma::square(Gsym));

        arma::vec rs = arma::vec(arma::sum(G, 1));
        arma::vec cs = arma::vec(arma::trans(arma::sum(G, 0)));
        arma::vec sg = rs + cs;
        double S2 = arma::sum(arma::square(sg));

        arma::mat normalized_scores_sq = arma::square(normalized_scores);
        arma::vec S3_vec = arma::trans((arma::sum(arma::square(normalized_scores_sq), 0) / nV) /
            (arma::square(arma::sum(normalized_scores_sq, 0) / nV)));
        double S4 = (nV * (nV - 3) + 3) * S1 - nV * S2 + 3 * Wsq;
        double S5 = (nV * (nV - 1)) * S1 - 2 * nV * S2 + 6 * Wsq;

        double k1 = (nV * S4) / ((nV - 1) * (nV - 2) * (nV - 3) * Wsq);
        double k2 = S5 / ((nV - 1) * (nV - 2) * (nV - 3) * Wsq);

        arma::vec sigma_sq = k1 - k2 * S3_vec - arma::square(mu);
        arma::vec sigma = arma::sqrt(sigma_sq);

        arma::vec zscores = (stat - mu) / sigma;

        // Summary stats
        arma::field<arma::vec> results(4);
        results(0) = stat;
        results(1) = zscores;
        results(2) = mu;
        results(3) = sigma;

        return (results);
    }

    arma::field<arma::vec>
        autocorrelation_Moran(const arma::sp_mat& G, const arma::mat& scores, int normalization_method, int perm_no,
                              int thread_no) {
        int nV = G.n_rows;

        arma::mat normalized_scores = normalize_scores(scores, normalization_method, thread_no);

        stdout_printf("Computing auto-correlation over network ... ");
        double W = arma::accu(G);
        arma::vec norm_sq = arma::vec(arma::trans(arma::sum(arma::square(normalized_scores))));
        arma::vec norm_factors = nV / (W * norm_sq);
        norm_factors.replace(arma::datum::nan, 0); // replace each NaN with 0

        arma::vec stat, mu, sigma, z;
        run_permutation_stat(G, normalized_scores, perm_no, thread_no,
                             /*seed_bump=*/42u, stat, mu, sigma, z);
        stdout_printf("done\n");
        FLUSH;

        arma::field<arma::vec> results(4);
        results(0) = stat % norm_factors;
        results(1) = z;
        results(2) = mu;
        results(3) = sigma;

        return (results);
    }

    arma::field<arma::vec>
        autocorrelation_Geary(const arma::sp_mat& G, const arma::mat& scores, int normalization_method, int perm_no,
                              int thread_no) {
        int nV = G.n_rows;
        arma::mat normalized_scores = normalize_scores(scores, normalization_method, thread_no);

        stdout_printf("Computing auto-correlation over network ... ");
        double W = arma::accu(G);
        arma::vec norm_sq = arma::vec(arma::trans(arma::sum(arma::square(normalized_scores))));
        arma::vec norm_factors = (nV - 1) / ((2 * W) * norm_sq);
        norm_factors.replace(arma::datum::nan, 0); // replace each NaN with 0

        // Compute graph Laplacian
        arma::vec d = arma::vec(arma::trans(arma::sum(G)));
        arma::sp_mat L(-G);
        L.diag() = d;

        arma::vec stat, mu, sigma, z;
        run_permutation_stat(L, normalized_scores, perm_no, thread_no,
                             /*seed_bump=*/137u, stat, mu, sigma, z);
        stdout_printf("done\n");
        FLUSH;

        arma::field<arma::vec> results(4);
        results(0) = stat % norm_factors;
        results(1) = -z;   // Geary convention: sign-flipped z
        results(2) = mu;
        results(3) = sigma;

        return (results);
    }
} // namespace actionet
