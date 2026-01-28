#include "network/build_network.hpp"
#include "network/hnsw_imp.hpp"
#include "utils_internal/utils_parallel.hpp"
#include <set> // CLion says you don't need this, but it's lying.

// Argument options
// valid distance metrics for hnsw
std::set<std::string> distance_metrics = {"jsd", "l2", "ip"};

// valid nearest-neighbor methods
std::set<std::string> nn_approaches = {"k*nn", "knn"};

// k^{*}-Nearest Neighbors: From Global to Local (NIPS 2016)
arma::sp_mat
    buildNetwork_KstarNN(arma::mat H, double density = 1.0, int thread_no = 0, double M = 16,
                         bool mutual_edges_only = true,
                         const std::string& distance_metric = "jsd") {
    double LC = 1.0 / density;
    // verify that a support distance metric has been specified
    //  the following distance metrics are supported in hnswlib: https://github.com/hnswlib/hnswlib#supported-distances
    if (distance_metrics.find(distance_metric) == distance_metrics.end()) {
        // invalid distance metric was provided; exit
        throw distMetException;
    }

    stdout_printf("Building adaptive network (density = %.2f)\n", density);
    FLUSH;

    if (distance_metric == "jsd") {
        H = arma::clamp(H, 0, 1);
        H = arma::normalise(H, 1, 0);
    }

    double kappa = 5.0;
    int sample_no = H.n_cols;
    // start with uniform k=sqrt(N) ["Pattern Classification" book by Duda et al.]
    int kNN = std::min(sample_no - 1, (int)(kappa * round(std::sqrt(sample_no))));

    double ef_construction;
    double ef;
    ef_construction = ef = kNN;

    hnswlib::HierarchicalNSW<float>* appr_alg = getApproximationAlgo(distance_metric, H, M, ef_construction);
    appr_alg->setEf(ef);

    int max_elements = H.n_cols;
    stdout_printf("\tBuilding index ... ");
    arma::fmat X = arma::conv_to<arma::fmat>::from(H);

    int threads_use = get_num_threads(max_elements, thread_no);
    #pragma omp parallel for num_threads(threads_use)
    for (size_t j = 0; j < max_elements; j++) {
        appr_alg->addPoint(X.colptr(j), j);
    }

    stdout_printf("done\n");
    FLUSH;

    stdout_printf("\tIdentifying nearest neighbors ... ");

    arma::mat idx = arma::zeros(sample_no, kNN + 1);
    arma::mat dist = arma::zeros(sample_no, kNN + 1);

    threads_use = get_num_threads(sample_no, thread_no);
    #pragma omp parallel for num_threads(threads_use)
    for (size_t i = 0; i < sample_no; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = appr_alg->searchKnn(X.colptr(i), kNN + 1);

        for (size_t j = 0; j <= kNN; j++) {
            auto& result_tuple = result.top();
            dist(i, kNN - j) = result_tuple.first;
            idx(i, kNN - j) = result_tuple.second;

            result.pop();
        }
    }

    delete (appr_alg);

    stdout_printf("done\n");
    FLUSH;

    stdout_printf("\tConstructing adaptive-nearest neighbor graph ... ");
    if (distance_metric == "jsd") {
        dist = arma::clamp(dist, 0.0, 1.0);
    }

    arma::mat Delta;
    arma::mat beta = LC * dist;
    arma::vec beta_sum = arma::zeros(sample_no);
    arma::vec beta_sq_sum = arma::zeros(sample_no);
    arma::mat lambda = arma::zeros(arma::size(beta));

    int k;
    for (k = 1; k <= kNN; k++) {
        beta_sum += beta.col(k);
        beta_sq_sum += arma::square(beta.col(k));

        lambda.col(k) = (1.0 / (double)k) * (beta_sum + arma::sqrt(k + arma::square(beta_sum) - k * beta_sq_sum));
    }
    lambda.replace(arma::datum::nan, 0);

    lambda = arma::trans(lambda);
    arma::vec node_lambda = arma::zeros(sample_no);
    beta = arma::trans(beta);

    Delta = lambda - beta;
    Delta.shed_row(0);

    // Build triplet lists in parallel to avoid race conditions on sparse matrix
    size_t estimated_edges = sample_no * kNN;
    std::vector<arma::uword> rows_vec, cols_vec;
    std::vector<double> vals_vec;
    rows_vec.reserve(estimated_edges);
    cols_vec.reserve(estimated_edges);
    vals_vec.reserve(estimated_edges);

    #pragma omp parallel
    {
        std::vector<arma::uword> local_rows, local_cols;
        std::vector<double> local_vals;

        #pragma omp for nowait
        for (size_t v = 0; v < sample_no; v++) {
            arma::vec delta = Delta.col(v);

            // uvec rows = find(delta > 0, 1, "last");
            arma::uvec rows = arma::find(delta < 0, 1, "first");
            int neighbor_no = rows.n_elem == 0 ? kNN : (rows(0));

            int dst = v;
            arma::rowvec v_dist = dist.row(v);
            arma::rowvec v_idx = idx.row(v);

            for (int i = 1; i < neighbor_no; i++) {
                int src = v_idx(i);
                local_rows.push_back(src);
                local_cols.push_back(dst);
                local_vals.push_back(v_dist(i));
            }
        }

        #pragma omp critical
        {
            rows_vec.insert(rows_vec.end(), local_rows.begin(), local_rows.end());
            cols_vec.insert(cols_vec.end(), local_cols.begin(), local_cols.end());
            vals_vec.insert(vals_vec.end(), local_vals.begin(), local_vals.end());
        }
    }

    // Construct sparse matrix from triplets
    arma::umat locations(2, rows_vec.size());
    locations.row(0) = arma::conv_to<arma::urowvec>::from(rows_vec);
    locations.row(1) = arma::conv_to<arma::urowvec>::from(cols_vec);
    arma::vec values = arma::conv_to<arma::vec>::from(vals_vec);
    arma::sp_mat G(locations, values, sample_no, sample_no);

    stdout_printf("done\n");
    FLUSH;

    G.replace(arma::datum::nan, 0); // replace each NaN with 0

    double epsilon = 1e-7;
    if (distance_metric == "jsd") {
        // Optimized path for JSD: no need to compute max_dist
        arma::sp_mat::iterator it = G.begin();
        arma::sp_mat::const_iterator it_end = G.end();
        for (; it != it_end; ++it) {
            *it = std::max(epsilon, 1.0 - (*it));
        }
    } else {
        // For other metrics, compute max_dist per column
        arma::vec max_dist = arma::vec(arma::trans(arma::max(G)));
        arma::sp_mat::iterator it = G.begin();
        arma::sp_mat::const_iterator it_end = G.end();
        for (; it != it_end; ++it) {
            *it = std::max(epsilon, max_dist(it.col()) - (*it));
        }
    }

    stdout_printf("\tFinalizing network ... ");
    arma::sp_mat Gt = arma::trans(G);

    arma::sp_mat G_sym;
    if (mutual_edges_only == false) {
        G_sym = (G + Gt);
        G_sym.for_each([](arma::sp_mat::elem_type& val) { val /= 2.0; });
    }
    else {
        // Default to MNN
        G_sym = arma::sqrt(G % Gt);
    }
    stdout_printf("done\n");
    FLUSH;

    G_sym.diag().zeros();

    return (G_sym);
}

arma::sp_mat buildNetwork_KNN(arma::mat H, int k, int thread_no = 0, double M = 16, double ef_construction = 200,
                              double ef = 200, bool mutual_edges_only = true,
                              const std::string& distance_metric = "jsd") {
    // verify that a support distance metric has been specified
    //  the following distance metrics are supported in hnswlib: https://github.com/hnswlib/hnswlib#supported-distances
    if (distance_metrics.find(distance_metric) == distance_metrics.end()) {
        // invalid distance metric was provided; exit
        throw distMetException;
    }

    stdout_printf("Building fixed-degree network (k = %d)\n", (int)k);
    FLUSH;

    if (distance_metric == "jsd") {
        H = arma::clamp(H, 0, 1);
        H = arma::normalise(H, 1, 0);
    }

    int sample_no = H.n_cols;

    hnswlib::HierarchicalNSW<float>* appr_alg = getApproximationAlgo(distance_metric, H, M, ef_construction);
    appr_alg->setEf(ef);

    stdout_printf("\tBuilding index ... ");
    int max_elements = H.n_cols;
    arma::fmat X = arma::conv_to<arma::fmat>::from(H);

    int threads_use = get_num_threads(max_elements, thread_no);
    #pragma omp parallel for num_threads(threads_use)
    for (size_t j = 0; j < max_elements; j++) {
        appr_alg->addPoint(X.colptr(j), j);
    }

    stdout_printf("done\n");
    FLUSH;

    stdout_printf("\tConstructing k*-NN ... ");

    // Build triplet lists in parallel to avoid race conditions on sparse matrix
    size_t estimated_edges = sample_no * k;
    std::vector<arma::uword> rows_vec, cols_vec;
    std::vector<double> vals_vec;
    rows_vec.reserve(estimated_edges);
    cols_vec.reserve(estimated_edges);
    vals_vec.reserve(estimated_edges);

    threads_use = get_num_threads(sample_no, thread_no);
    #pragma omp parallel
    {
        std::vector<arma::uword> local_rows, local_cols;
        std::vector<double> local_vals;

        #pragma omp for nowait
        for (size_t i = 0; i < sample_no; i++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
                appr_alg->searchKnn(X.colptr(i), k);

            for (size_t j = 0; j < result.size(); j++) {
                auto& result_tuple = result.top();

                local_rows.push_back(i);
                local_cols.push_back(result_tuple.second);
                local_vals.push_back(result_tuple.first);
                result.pop();
            }
        }

        #pragma omp critical
        {
            rows_vec.insert(rows_vec.end(), local_rows.begin(), local_rows.end());
            cols_vec.insert(cols_vec.end(), local_cols.begin(), local_cols.end());
            vals_vec.insert(vals_vec.end(), local_vals.begin(), local_vals.end());
        }
    }

    // Construct sparse matrix from triplets
    arma::umat locations(2, rows_vec.size());
    locations.row(0) = arma::conv_to<arma::urowvec>::from(rows_vec);
    locations.row(1) = arma::conv_to<arma::urowvec>::from(cols_vec);
    arma::vec values = arma::conv_to<arma::vec>::from(vals_vec);
    arma::sp_mat G(locations, values, sample_no, sample_no);

    stdout_printf("done\n");
    FLUSH;

    delete (appr_alg);

    G.replace(arma::datum::nan, 0); // replace each NaN with 0

    double epsilon = 1e-7;
    if (distance_metric == "jsd") {
        // Optimized path for JSD: no need to compute max_dist
        arma::sp_mat::iterator it = G.begin();
        arma::sp_mat::const_iterator it_end = G.end();
        for (; it != it_end; ++it) {
            *it = std::max(epsilon, 1.0 - (*it));
        }
    } else {
        // For other metrics, compute max_dist per column
        arma::vec max_dist = arma::vec(arma::trans(arma::max(G)));
        arma::sp_mat::iterator it = G.begin();
        arma::sp_mat::const_iterator it_end = G.end();
        for (; it != it_end; ++it) {
            *it = std::max(epsilon, max_dist(it.col()) - (*it));
        }
    }

    stdout_printf("\tFinalizing network ... ");

    arma::sp_mat Gt = arma::trans(G);

    arma::sp_mat G_sym;
    if (mutual_edges_only == false) {
        G_sym = (G + Gt);
        G_sym.for_each([](arma::sp_mat::elem_type& val) { val /= 2.0; });
    }
    else {
        // Default to MNN
        stdout_printf("\n\t\tKeeping mutual nearest-neighbors only ... ");
        G_sym = arma::sqrt(G % Gt);
    }
    stdout_printf("done\n");
    FLUSH;

    G_sym.diag().zeros();

    return (G_sym);
}

namespace actionet {
    arma::sp_mat
        buildNetwork(const arma::mat& H, std::string algorithm, std::string distance_metric, double density,
                     int thread_no,
                     double M, double ef_construction, double ef, bool mutual_edges_only, int k) {
        // Verify that valid distance metric has been specified
        if (distance_metrics.find(distance_metric) == distance_metrics.end()) {
            // Invalid distance metric was provided; exit
            throw distMetException;
        }

        // Verify that valid nn approach has been specified
        if (nn_approaches.find(algorithm) == nn_approaches.end()) {
            // Invalid nn approach was provided; exit
            throw nnApproachException;
        }

        /// Build ACTIONet with k*nn or fixed k knn, based on passed parameter
        arma::sp_mat G;
        if (algorithm == "k*nn") {
            G = buildNetwork_KstarNN(H, density, thread_no, M, mutual_edges_only, distance_metric);
        }
        else {
            G = buildNetwork_KNN(H, k, thread_no, M, ef_construction, ef, mutual_edges_only, distance_metric);
        }
        return (G);
    }
} // namespace actionet
