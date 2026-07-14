#include "action/action_post.hpp"
#include "action/spa.hpp"
#include "action/simplex_regression.hpp"
#include "utils_internal/utils_action_numeric_policy.hpp"
#include "utils_internal/utils_stats.hpp"

namespace actionet {
    namespace numeric_policy = utils_internal::action_numeric_policy;

    ResCollectArch collectArchetypes(arma::mat& C_stacked, arma::mat& H_stacked,
                                     double spec_th, int min_obs) {
        size_t total_archs = H_stacked.n_rows;

        ResCollectArch results;

        stdout_printf("Pruning %d archetypes:\n", (int)total_archs);
        FLUSH;

        arma::mat backbone = arma::cor(arma::trans(H_stacked));
        backbone.diag().zeros();
        backbone.transform([](double val) { return (val < 0 ? 0 : val); });

        arma::vec pruned = arma::zeros(total_archs);

        // Barrat weighted transitivity via matrix operations.
        // For each node k: T(k) = sum_{i,j} [(w_ki + w_kj)/2] * [a_ik * a_kj * a_ji > 0] / [s(k)*(d(k)-1)]
        // where a = (backbone > 0) is the adjacency matrix.
        // The triangle indicator for node k over (i,j) is (A * A)(i,j) restricted to
        // pairs where a_ik and a_kj are nonzero. The full sum equals:
        //   sum_{i,j} w_ki * a_kj * t_ij  +  sum_{i,j} a_ki * w_kj * t_ij
        // where t_ij = a_ij (triangle closing edge), divided by 2.
        // This simplifies to: for each k, dot(backbone.row(k), (A * A).row(k)) * 2 / 2
        // which is backbone.row(k) * A * A.row(k)^T ... but we need to be more careful.
        //
        // Expanding: for fixed k, sum_{i,j} (w_ki + w_kj)/2 * a_ik * a_kj * a_ji
        //   = (1/2) * sum_{i,j} w_ki * a_ik * a_kj * a_ji  +  (1/2) * sum_{i,j} w_kj * a_ik * a_kj * a_ji
        //   = (1/2) * sum_i [w_ki * a_ik * sum_j(a_kj * a_ji)]  +  (1/2) * sum_j [w_kj * a_kj * sum_i(a_ik * a_ji)]
        //   = (1/2) * sum_i [w_ki * a_ik * (A * A^T)_{k,i}]      +  (1/2) * sum_j [w_kj * a_kj * (A^T * A)_{j,k}]
        //  Since A is symmetric (backbone is symmetric): A = A^T, so (A*A)_{k,i} = (A*A^T)_{k,i}
        //   = (1/2) * sum_i w_ki * a_ik * (A²)_{k,i}  +  (1/2) * sum_j w_kj * a_kj * (A²)_{j,k}
        //  And since backbone is symmetric: w_ki = w_ik, a_ik = a_ki, (A²)_{k,i} = (A²)_{i,k}
        //  Both terms are identical. So total = sum_i w_ki * a_ik * (A²)_{k,i}
        //  = sum_i backbone(k,i) * (A²)(k,i)  [since a_ik = (backbone(k,i)>0), and backbone already has the weight]
        //  But we need a_ik factor too. Since backbone(k,i) > 0 implies a_ik=1, and backbone(k,i)=0 implies the term vanishes:
        //  = sum_i backbone(k,i) * (A²)(k,i)  =  dot(backbone.row(k), (A²).row(k))

        arma::mat A_bin = arma::conv_to<arma::mat>::from(backbone > 0);
        arma::mat A2 = A_bin * A_bin;

        arma::vec s = arma::sum(backbone, 1);
        arma::vec d = arma::sum(A_bin, 1);
        arma::vec transitivity = arma::zeros(total_archs);
        for (size_t k = 0; k < total_archs; k++) {
            double denom = s(k) * (d(k) - 1);
            if (denom > 0)
                transitivity(k) = arma::dot(backbone.row(k), A2.row(k)) / denom;
        }

        arma::vec transitivity_z = zscore(transitivity);
        arma::uvec nonspecific_idx = arma::find(transitivity_z < spec_th);
        pruned(nonspecific_idx).ones();
        stdout_printf("\tNon-specific archetypes: %d\n", (int)nonspecific_idx.n_elem);
        FLUSH;
        // Release O(T^2) buffers before per-archetype checks to reduce peak RSS.
        A2.reset();
        A_bin.reset();
        backbone.reset();
        s.reset();
        d.reset();
        transitivity.reset();
        transitivity_z.reset();

        // Find landmark cells
        // i.e., closest cells to each multi-level archetype (its projection on to the cell space)
        int bad_archs = 0;
        for (size_t i = 0; i < total_archs; i++) {
            const arma::subview_row<double> h = H_stacked.row(i);
            const arma::subview_col<double> c = C_stacked.col(i);
            double h_max = h.max();

            arma::uvec h_landmarks = arma::find(
                (h_max - h) < numeric_policy::landmark_proximity_tolerance);
            arma::uvec c_landmarks = arma::find(
                c > numeric_policy::simplex_coefficient_support_tolerance);
            arma::uvec common_landmarks = arma::intersect(h_landmarks, c_landmarks);

            if (0 < common_landmarks.n_elem) { // At least one supported cell agrees.
                continue;
            }
            else { // Potentially noisy archetype
                pruned(i) = 1;
                bad_archs++;
            }
        }

        stdout_printf("\tUnreproducible archetypes: %d\n", bad_archs);
        FLUSH;

        arma::urowvec membership_counts =
            arma::sum(
                arma::conv_to<arma::umat>::from(
                    C_stacked > numeric_policy::simplex_coefficient_support_tolerance),
                0);
        arma::uvec trivial_idx = arma::find(membership_counts < (arma::uword)min_obs);
        pruned(trivial_idx).ones();

        stdout_printf("\tTrivial archetypes: %d\n", (int)trivial_idx.n_elem);
        FLUSH;

        arma::uvec selected_archs = arma::find(pruned == 0);
        results.selected_archs = selected_archs;
        results.C_stacked = C_stacked.cols(selected_archs);
        results.H_stacked = H_stacked.rows(selected_archs);

        return (results);
    }

    ResMergeArch
        mergeArchetypes(const arma::mat& S_r, const arma::mat& C_stacked, arma::mat& H_stacked, int thread_no) {
        stdout_printf("Merging %d archetypes:\n", (int)C_stacked.n_cols);
        FLUSH;

        ResMergeArch output;

        arma::rowvec col_sums = arma::sum(H_stacked, 0);
        col_sums.transform([](double val) { return (val > 0.0) ? val : 1.0; });
        H_stacked.each_row() /= col_sums;
        // H_stacked is uniformly dense after column-wise L1 normalisation (every
        // column sums to 1 with all non-negative values). Converting to sp_mat here
        // would allocate an equally-sized sparse copy while the dense original stays
        // live — doubling peak RSS for no algorithmic benefit. Use a plain dense matmul.
        arma::mat H_arch = H_stacked * C_stacked;
        H_arch.replace(arma::datum::nan, 0); // replace each NaN with 0

        ResSPA SPA_out = runSPA(H_arch, (int)H_arch.n_cols);
        H_arch.reset(); // H_arch no longer needed; free before simplex regression
        arma::uvec candidates = SPA_out.selected_cols;
        arma::vec scores = SPA_out.column_norms;
        double x1 = arma::sum(scores);
        double x2 = arma::sum(arma::square(scores));
        int arch_no = std::round((x1 * x1) / x2);
        candidates = candidates(arma::span(0, arch_no - 1));

        stdout_printf("Archetypes in merged set: %d\n", arch_no);
        FLUSH;

        arma::mat C_merged = C_stacked.cols(candidates);
        arma::mat W_r_merged = S_r * C_merged;
        arma::mat H_merged = runSimplexRegression(W_r_merged, S_r, false);
        arma::uvec assigned_archetypes = arma::trans(arma::index_max(H_merged, 0));

        output.selected_archetypes = std::move(candidates);
        output.C_merged = std::move(C_merged);
        output.H_merged = std::move(H_merged);
        output.assigned_archetypes = std::move(assigned_archetypes);

        return (output);
    }
} // namespace actionet
