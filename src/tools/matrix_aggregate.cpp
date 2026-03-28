#include "tools/matrix_aggregate.hpp"

namespace actionet {

namespace {

struct GroupIndex {
    int n_groups;
    std::vector<arma::uvec> members;
    std::vector<int> group_of;  // 0-based group id for each sample
};

// Precompute group membership once. Labels in sample_assignments are assumed
// to be 1-based contiguous integers (as produced by the R/Python frontends).
GroupIndex build_group_index(const arma::vec& sample_assignments) {
    int n_groups = static_cast<int>(arma::max(sample_assignments));
    GroupIndex gi;
    gi.n_groups = n_groups;
    gi.members.resize(n_groups);
    gi.group_of.resize(sample_assignments.n_elem);

    std::vector<std::vector<arma::uword>> tmp(n_groups);
    for (arma::uword i = 0; i < sample_assignments.n_elem; ++i) {
        int g = static_cast<int>(sample_assignments[i]) - 1;
        gi.group_of[i] = g;
        tmp[g].push_back(i);
    }
    for (int g = 0; g < n_groups; ++g) {
        gi.members[g] = arma::uvec(tmp[g].data(), tmp[g].size());
    }
    return gi;
}

} // anon namespace

// ---- Sums ----

// Sparse input: iterate over nonzeros (works for both mat and sp_mat output)
template <typename OutputT>
static OutputT grouped_sums_sparse(arma::sp_mat& S, const GroupIndex& gi, int axis) {
    if (axis == 0) {
        OutputT pb(S.n_rows, gi.n_groups);
        pb.zeros();
        for (auto it = S.begin(); it != S.end(); ++it) {
            int g = gi.group_of[it.col()];
            pb(it.row(), g) += (*it);
        }
        return pb;
    } else {
        OutputT pb(gi.n_groups, S.n_cols);
        pb.zeros();
        for (auto it = S.begin(); it != S.end(); ++it) {
            int g = gi.group_of[it.row()];
            pb(g, it.col()) += (*it);
        }
        return pb;
    }
}

// Dense input → dense output: use Armadillo column/row slicing
static arma::mat grouped_sums_dense(arma::mat& S, const GroupIndex& gi, int axis) {
    if (axis == 0) {
        arma::mat pb = arma::zeros(S.n_rows, gi.n_groups);
        for (int g = 0; g < gi.n_groups; ++g) {
            const arma::uvec& idx = gi.members[g];
            if (idx.n_elem == 0) continue;
            if (idx.n_elem == 1) {
                pb.col(g) = S.col(idx(0));
            } else {
                pb.col(g) = arma::sum(S.cols(idx), 1);
            }
        }
        return pb;
    } else {
        arma::mat pb = arma::zeros(gi.n_groups, S.n_cols);
        for (int g = 0; g < gi.n_groups; ++g) {
            const arma::uvec& idx = gi.members[g];
            if (idx.n_elem == 0) continue;
            if (idx.n_elem == 1) {
                pb.row(g) = S.row(idx(0));
            } else {
                pb.row(g) = arma::sum(S.rows(idx), 0);
            }
        }
        return pb;
    }
}

// Template instantiations for Sums
template <>
arma::mat computeGroupedSums<arma::sp_mat, arma::mat>(const arma::sp_mat& S, const arma::vec& sa, int axis) {
    if ((axis == 0 && sa.n_elem != S.n_cols) || (axis == 1 && sa.n_elem != S.n_rows))
        throw std::invalid_argument("Length of 'sample_assignments' must match the grouped dimension of S.");
    GroupIndex gi = build_group_index(sa);
    return grouped_sums_sparse<arma::mat>(S, gi, axis);
}

template <>
arma::sp_mat computeGroupedSums<arma::sp_mat, arma::sp_mat>(const arma::sp_mat& S, const arma::vec& sa, int axis) {
    if ((axis == 0 && sa.n_elem != S.n_cols) || (axis == 1 && sa.n_elem != S.n_rows))
        throw std::invalid_argument("Length of 'sample_assignments' must match the grouped dimension of S.");
    GroupIndex gi = build_group_index(sa);
    return grouped_sums_sparse<arma::sp_mat>(S, gi, axis);
}

template <>
arma::mat computeGroupedSums<arma::mat, arma::mat>(const arma::mat& S, const arma::vec& sa, int axis) {
    if ((axis == 0 && sa.n_elem != S.n_cols) || (axis == 1 && sa.n_elem != S.n_rows))
        throw std::invalid_argument("Length of 'sample_assignments' must match the grouped dimension of S.");
    GroupIndex gi = build_group_index(sa);
    return grouped_sums_dense(S, gi, axis);
}

// ---- Means ----

template <>
arma::mat computeGroupedMeans<arma::sp_mat, arma::mat>(const arma::sp_mat& S, const arma::vec& sa, int axis) {
    if ((axis == 0 && sa.n_elem != S.n_cols) || (axis == 1 && sa.n_elem != S.n_rows))
        throw std::invalid_argument("Length of 'sample_assignments' must match the grouped dimension of S.");
    GroupIndex gi = build_group_index(sa);
    arma::mat pb = grouped_sums_sparse<arma::mat>(S, gi, axis);
    for (int g = 0; g < gi.n_groups; ++g) {
        double denom = std::max<arma::uword>(1, gi.members[g].n_elem);
        if (axis == 0) pb.col(g) /= denom;
        else           pb.row(g) /= denom;
    }
    return pb;
}

template <>
arma::sp_mat computeGroupedMeans<arma::sp_mat, arma::sp_mat>(const arma::sp_mat& S, const arma::vec& sa, int axis) {
    if ((axis == 0 && sa.n_elem != S.n_cols) || (axis == 1 && sa.n_elem != S.n_rows))
        throw std::invalid_argument("Length of 'sample_assignments' must match the grouped dimension of S.");
    GroupIndex gi = build_group_index(sa);
    arma::sp_mat pb = grouped_sums_sparse<arma::sp_mat>(S, gi, axis);
    // Divide each nonzero by its group size (preserves sparsity since sum==0 iff all inputs zero)
    for (auto it = pb.begin(); it != pb.end(); ++it) {
        int g = (axis == 0) ? static_cast<int>(it.col()) : static_cast<int>(it.row());
        (*it) /= std::max<arma::uword>(1, gi.members[g].n_elem);
    }
    return pb;
}

template <>
arma::mat computeGroupedMeans<arma::mat, arma::mat>(const arma::mat& S, const arma::vec& sa, int axis) {
    if ((axis == 0 && sa.n_elem != S.n_cols) || (axis == 1 && sa.n_elem != S.n_rows))
        throw std::invalid_argument("Length of 'sample_assignments' must match the grouped dimension of S.");
    GroupIndex gi = build_group_index(sa);
    arma::mat pb = grouped_sums_dense(S, gi, axis);
    for (int g = 0; g < gi.n_groups; ++g) {
        double denom = std::max<arma::uword>(1, gi.members[g].n_elem);
        if (axis == 0) pb.col(g) /= denom;
        else           pb.row(g) /= denom;
    }
    return pb;
}

// ---- Vars ----

// Sparse input: single-pass Welford-style using precomputed means
template <typename OutputT>
static OutputT grouped_vars_sparse(const arma::sp_mat& S, const GroupIndex& gi, int axis) {
    // First compute means (as dense, always needed for variance calc)
    arma::mat pb_mu = grouped_sums_sparse<arma::mat>(S, gi, axis);
    for (int g = 0; g < gi.n_groups; ++g) {
        double denom = std::max<arma::uword>(1, gi.members[g].n_elem);
        if (axis == 0) pb_mu.col(g) /= denom;
        else           pb_mu.row(g) /= denom;
    }

    arma::uword out_rows, out_cols;
    if (axis == 0) { out_rows = S.n_rows; out_cols = gi.n_groups; }
    else           { out_rows = gi.n_groups; out_cols = S.n_cols; }

    arma::mat pb  = arma::zeros(out_rows, out_cols);
    arma::mat pbz = arma::zeros(out_rows, out_cols);

    for (auto it = S.begin(); it != S.end(); ++it) {
        int r, c;
        if (axis == 0) { r = it.row(); c = gi.group_of[it.col()]; }
        else           { r = gi.group_of[it.row()]; c = it.col(); }
        double diff = (*it) - pb_mu(r, c);
        pb(r, c) += diff * diff;
        pbz(r, c) += 1;
    }

    // Account for structural zeros: they contribute (0 - mu)^2 each
    for (int g = 0; g < gi.n_groups; ++g) {
        int n_total = static_cast<int>(gi.members[g].n_elem);
        if (axis == 0) {
            for (arma::uword i = 0; i < out_rows; ++i) {
                int nz = n_total - static_cast<int>(pbz(i, g));
                pb(i, g) += nz * pb_mu(i, g) * pb_mu(i, g);
            }
            pb.col(g) /= std::max(1, n_total - 1);
        } else {
            for (arma::uword j = 0; j < out_cols; ++j) {
                int nz = n_total - static_cast<int>(pbz(g, j));
                pb(g, j) += nz * pb_mu(g, j) * pb_mu(g, j);
            }
            pb.row(g) /= std::max(1, n_total - 1);
        }
    }

    if constexpr (std::is_same_v<OutputT, arma::sp_mat>) {
        return arma::sp_mat(pb);
    } else {
        return pb;
    }
}

// Dense input → dense output
static arma::mat grouped_vars_dense(const arma::mat& S, const GroupIndex& gi, int axis) {
    if (axis == 0) {
        arma::mat pb = arma::zeros(S.n_rows, gi.n_groups);
        for (int g = 0; g < gi.n_groups; ++g) {
            const arma::uvec& idx = gi.members[g];
            if (idx.n_elem <= 1) continue;
            pb.col(g) = arma::var(S.cols(idx), 0, 1);
        }
        return pb;
    } else {
        arma::mat pb = arma::zeros(gi.n_groups, S.n_cols);
        for (int g = 0; g < gi.n_groups; ++g) {
            const arma::uvec& idx = gi.members[g];
            if (idx.n_elem <= 1) continue;
            pb.row(g) = arma::var(S.rows(idx), 0, 0);
        }
        return pb;
    }
}

template <>
arma::mat computeGroupedVars<arma::sp_mat, arma::mat>(const arma::sp_mat& S, const arma::vec& sa, int axis) {
    if ((axis == 0 && sa.n_elem != S.n_cols) || (axis == 1 && sa.n_elem != S.n_rows))
        throw std::invalid_argument("Length of 'sample_assignments' must match the grouped dimension of S.");
    GroupIndex gi = build_group_index(sa);
    return grouped_vars_sparse<arma::mat>(S, gi, axis);
}

template <>
arma::sp_mat computeGroupedVars<arma::sp_mat, arma::sp_mat>(const arma::sp_mat& S, const arma::vec& sa, int axis) {
    if ((axis == 0 && sa.n_elem != S.n_cols) || (axis == 1 && sa.n_elem != S.n_rows))
        throw std::invalid_argument("Length of 'sample_assignments' must match the grouped dimension of S.");
    GroupIndex gi = build_group_index(sa);
    return grouped_vars_sparse<arma::sp_mat>(S, gi, axis);
}

template <>
arma::mat computeGroupedVars<arma::mat, arma::mat>(const arma::mat& S, const arma::vec& sa, int axis) {
    if ((axis == 0 && sa.n_elem != S.n_cols) || (axis == 1 && sa.n_elem != S.n_rows))
        throw std::invalid_argument("Length of 'sample_assignments' must match the grouped dimension of S.");
    GroupIndex gi = build_group_index(sa);
    return grouped_vars_dense(S, gi, axis);
}

} // namespace actionet
