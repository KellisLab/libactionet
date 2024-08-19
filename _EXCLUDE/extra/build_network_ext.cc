#include "build_network_ext.h"

// Compute Jensen-Shannon distance (JSD)
double computeJSD(const double *pVect1, const double *pVect2, const double *log_vec,
                  int N) {
    double half = 0.5;

    double sum1 = 0, sum2 = 0;
    for (size_t i = 0; i < N; i++) {
        double p = pVect1[i];
        double q = pVect2[i];
        double m = (p + q) * half;

        int p_idx = (int) std::floor(p * 1000000.0);
        int q_idx = (int) std::floor(q * 1000000.0);
        int m_idx = (int) std::floor(m * 1000000.0);

        double lg_p = log_vec[p_idx];
        double lg_q = log_vec[q_idx];
        double lg_m = log_vec[m_idx];

        sum1 += (p * lg_p) + (q * lg_q);
        sum2 += m * lg_m;
    }

    double JS = std::max(half * sum1 - sum2, 0.0);

    return (double) (1.0 - std::sqrt(JS));
}

arma::mat computeFullSim(arma::mat &H, int thread_no) {
    double log_vec[1000001];
    for (int i = 0; i <= 1000000; i++) {
        log_vec[i] = (double) std::log2((double) i / 1000000.0);
    }
    log_vec[0] = 0;

    H = arma::clamp(H, 0, 1);
    H = arma::normalise(H, 1, 0); // make the norm (sum) of each column 1

    int sample_no = H.n_cols;
    int dim = H.n_rows;

    arma::mat G = arma::zeros(sample_no, sample_no);
    mini_thread::parallelFor(
            0, sample_no, [&](size_t i) {
                for (int j = 0; j < sample_no; j++) {
                    //use the JSD distance function here
                    G(i, j) = computeJSD(H.colptr(i), H.colptr(j), log_vec, dim);
                }
            },
            thread_no);

    G = arma::clamp(G, 0.0, 1.0);

    return (G);
}

arma::sp_mat buildNetwork_KstarNN_v2(arma::mat H, double density = 1.0, int thread_no = 0, double M = 16,
                                     double ef_construction = 200,
                                     double ef = 200, bool mutual_edges_only = true,
                                     std::string distance_metric = "jsd") {

    ef_construction = ef = 5 * std::sqrt(H.n_cols);

    double LC = 1.0 / density;
    // verify that a support distance metric has been specified
    //  the following distance metrics are supported in hnswlib: https://github.com/hnswlib/hnswlib#supported-distances
    if (distance_metrics.find(distance_metric) == distance_metrics.end()) {
        // invalid distance metric was provided; exit
        throw distMetException;
    }

    if (thread_no <= 0) {
        thread_no = SYS_THREADS_DEF; // std::thread::hardware_concurrency();
    }

    stdout_printf("Building adaptive network (%d threads):\n", thread_no);
    stdout_printf("\tParameters: metric = %s, density = %0.2f, mutual_edges_only = %s\n",
                  distance_metric.c_str(), density, mutual_edges_only ? "TRUE" : "FALSE");
    FLUSH;

    if (distance_metric == "jsd") {
        H = arma::clamp(H, 0, 1);
        H = arma::normalise(H, 1, 0);
    }

    double kappa = 5.0;
    int sample_no = H.n_cols;
    // int kNN = min(sample_no-1, (int)(kappa*round(sqrt(sample_no)))); // start
    // with uniform k=sqrt(N) ["Pattern Classification" book by Duda et al.]

    // TODO, add cosine -- which is InnerProductSpace but with normalization; see here:
    //  https://github.com/hnswlib/hnswlib/blob/master/python_bindings/bindings.cpp#L97
    hnswlib::HierarchicalNSW<float> *appr_alg = getApproximationAlgo(distance_metric, H, M, ef_construction);
    appr_alg->setEf(ef);

    stdout_printf("\tBuilding index ... (updated) ");
    arma::fmat X = arma::conv_to<arma::fmat>::from(H);
    int max_elements = X.n_cols;

    /*
        appr_alg->addPoint(X.colptr(0), (size_t) 0 ); // Critical for reproducibility!! :: https://github.com/nmslib/hnswlib/blob/master/sift_1b.cpp
        ParallelFor(1, max_elements, thread_no, [&](size_t j, size_t threadId) {
             //const std::lock_guard<std::mutex> lock(hnsw_mutex);
             appr_alg->addPoint(X.colptr(j), j);
        });
    */
    AddWorker worker(appr_alg, X);
    RcppPerpendicular::parallel_for(0, max_elements, worker, thread_no, 1);

    //    int j2 = 1;
    /*    parallelFor(1, max_elements, [&] (size_t j)
                    { const std::lock_guard<std::mutex> lock(hnsw_mutex); appr_alg->addPoint(X.colptr(j), static_cast<size_t>(j)); }, thread_no);
    */
    stdout_printf("done\n");
    FLUSH;

    stdout_printf("\tConstructing k*-NN ... ");

    std::vector<std::vector<int>> ii(thread_no);
    std::vector<std::vector<int>> jj(thread_no);
    std::vector<std::vector<double>> vv(thread_no);

    ParallelFor(0, sample_no, thread_no, [&](size_t i, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> results =
                appr_alg->searchKStarnn(X.colptr(i), LC);

        while (!results.empty()) {
            auto &res = results.top();
            int j = res.second;


            double v = res.first;
            ii[threadId].push_back(i);
            jj[threadId].push_back(j);
            vv[threadId].push_back(v);

            results.pop();
        }
    });

    arma::vec values;
    arma::umat locations;
    for (int threadId = 0; threadId < thread_no; threadId++) {
        if (threadId == 0) {
            values = arma::conv_to<arma::vec>::from(vv[threadId]);

            arma::uvec iv = arma::conv_to<arma::uvec>::from(ii[threadId]);
            arma::uvec jv = arma::conv_to<arma::uvec>::from(jj[threadId]);
            locations = arma::trans(arma::join_rows(iv, jv));
        } else {
            values = arma::join_vert(values, arma::conv_to<arma::vec>::from(vv[threadId]));

            arma::uvec iv = arma::conv_to<arma::uvec>::from(ii[threadId]);
            arma::uvec jv = arma::conv_to<arma::uvec>::from(jj[threadId]);
            locations = arma::join_rows(locations, arma::trans(arma::join_rows(iv, jv)));
        }
    }
    arma::sp_mat G(locations, values, sample_no, sample_no);

    G.replace(arma::datum::nan, 0); // replace each NaN with 0
    arma::vec max_dist = arma::vec(arma::trans(arma::max(G)));
    arma::sp_mat::iterator it = G.begin();
    arma::sp_mat::const_iterator it_end = G.end();

    double epsilon = 1e-7;
    for (; it != it_end; ++it) {
        double upper_bound = (distance_metric == "jsd") ? 1.0 : max_dist(it.col());
        *it = std::max(epsilon, upper_bound - (*it));
    }

    stdout_printf("done\n");
    FLUSH;

    delete (appr_alg);

    stdout_printf("\tFinalizing network ... ");
    arma::sp_mat Gt = arma::trans(G);

    arma::sp_mat G_sym;
    if (mutual_edges_only == false) {
        G_sym = (G + Gt);
        G_sym.for_each([](arma::sp_mat::elem_type &val) { val /= 2.0; });
    } else { // Default to MNN
        // stdout_printf("\n\t\tKeeping mutual nearest-neighbors only ... ");
        G_sym = arma::sqrt(G % Gt);
    }
    stdout_printf("done\n");
    FLUSH;

    G_sym.diag().zeros();

    return (G_sym);
}

arma::sp_mat
buildNetwork_bipartite(arma::mat H1, arma::mat H2, double density, int thread_no, double M, double ef_construction,
                       double ef, std::string distance_metric) {

    double LC = 1.0 / density;
    // verify that a support distance metric has been specified
    //  the following distance metrics are supported in hnswlib: https://github.com/hnswlib/hnswlib#supported-distances
    if (distance_metrics.find(distance_metric) == distance_metrics.end()) {
        // invalid distance metric was provided; exit
        throw distMetException;
    }

    if (thread_no <= 0) {
        thread_no = SYS_THREADS_DEF; // std::thread::hardware_concurrency();
    }

    stdout_printf("Building adaptive bipartite network (%d threads):\n", thread_no);
    stdout_printf("\tParameters: metric = %s, density = %0.2f\n",
                  distance_metric.c_str(), density);
    FLUSH;

    if (distance_metric == "jsd") {
        H1 = arma::clamp(H1, 0, 1);
        H1 = arma::normalise(H1, 1, 0);

        H2 = arma::clamp(H2, 0, 1);
        H2 = arma::normalise(H2, 1, 0);
    }

    double kappa = 5.0;
    // TODO, add cosine -- which is InnerProductSpace but with normalization; see here:
    //  https://github.com/hnswlib/hnswlib/blob/master/python_bindings/bindings.cpp#L97
    hnswlib::HierarchicalNSW<float> *appr_alg = getApproximationAlgo(distance_metric, H1, M, ef_construction);
    appr_alg->setEf(ef);

    stdout_printf("\tBuilding index ... (updated) ");
    arma::fmat X1 = arma::conv_to<arma::fmat>::from(H1);
    arma::fmat X2 = arma::conv_to<arma::fmat>::from(H2);

    int max_elements = X1.n_cols;
    AddWorker worker(appr_alg, X1);
    RcppPerpendicular::parallel_for(0, max_elements, worker, thread_no, 1);

    //    int j2 = 1;
    /*    parallelFor(1, max_elements, [&] (size_t j)
                    { const std::lock_guard<std::mutex> lock(hnsw_mutex); appr_alg->addPoint(X.colptr(j), static_cast<size_t>(j)); }, thread_no);
    */
    stdout_printf("done\n");
    FLUSH;

    stdout_printf("\tConstructing k*-NN ... ");

    std::vector<std::vector<int>> ii(thread_no);
    std::vector<std::vector<int>> jj(thread_no);
    std::vector<std::vector<double>> vv(thread_no);

    ParallelFor(0, X2.n_cols, thread_no, [&](size_t j, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> results =
                appr_alg->searchKStarnn(X2.colptr(j), LC);

        while (!results.empty()) {
            auto &res = results.top();
            int i = res.second;
            double v = res.first;
            ii[threadId].push_back(i);
            jj[threadId].push_back(j);
            vv[threadId].push_back(v);

            results.pop();
        }
    });

    arma::vec values;
    arma::umat locations;
    for (int threadId = 0; threadId < thread_no; threadId++) {
        if (threadId == 0) {
            values = arma::conv_to<arma::vec>::from(vv[threadId]);

            arma::uvec iv = arma::conv_to<arma::uvec>::from(ii[threadId]);
            arma::uvec jv = arma::conv_to<arma::uvec>::from(jj[threadId]);
            locations = arma::trans(arma::join_rows(iv, jv));
        } else {
            values = arma::join_vert(values, arma::conv_to<arma::vec>::from(vv[threadId]));

            arma::uvec iv = arma::conv_to<arma::uvec>::from(ii[threadId]);
            arma::uvec jv = arma::conv_to<arma::uvec>::from(jj[threadId]);
            locations = arma::join_rows(locations, arma::trans(arma::join_rows(iv, jv)));
        }
    }
    delete (appr_alg);

    arma::sp_mat G(locations, values, H1.n_cols, H2.n_cols);

    G.replace(arma::datum::nan, 0); // replace each NaN with 0
    arma::vec max_dist = arma::vec(arma::trans(arma::max(G)));
    arma::sp_mat::iterator it = G.begin();
    arma::sp_mat::const_iterator it_end = G.end();

    double epsilon = 1e-7;
    for (; it != it_end; ++it) {
        double upper_bound = (distance_metric == "jsd") ? 1.0 : max_dist(it.col());
        *it = std::max(epsilon, upper_bound - (*it));
    }

    stdout_printf("done\n");
    FLUSH;

    return (G);
}
