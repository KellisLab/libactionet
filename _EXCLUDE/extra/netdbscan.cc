#include "netdbscan.h"

arma::vec NetDBSCAN(arma::sp_mat &G, int minPts, double eps, double alpha_val) {

    int n = G.n_rows;
    G.diag().zeros();  // Remove self-loops

    // Process vertices in a particular order to give higher priority to the
    // inner, the most well-connected vertices
    stdout_printf("Ordering vertices ... ");
    arma::vec cn = arma::conv_to<arma::vec>::from(actionet::computeCoreness(G));
    arma::sp_mat X0(G.n_rows, 1);
    X0.col(0) = cn;
    arma::mat pr = actionet::computeNetworkDiffusion(G, X0, 1, alpha_val, 3);
    arma::uvec perm = arma::sort_index(pr.col(0), "descend");
    stdout_printf("done\n");

    // Main body of the Net-DBSCAN
    stdout_printf("Clustering nodes\n");
    unsigned int C = 0;
    arma::vec clusters = UNDEFINED * arma::ones(n);
    arma::vec visited = arma::zeros(n);

    std::vector<int> N;
    N.reserve(n);  // Allocate memory for the maximum size
    for (int i = 0; i < n; i++) {
        int v = perm[i];           // Visit vertices in order of connectivity
        if (visited[v]) continue;  // Skip previously visited vertices
        visited[v] = 1;            // Mark vertex as visited

        N.clear();  // Flush previous content
        int counts = 0;
        for (arma::sp_mat::col_iterator col_it = G.begin_col(v); col_it != G.end_col(v);
             ++col_it) {
            counts++;
            int u = col_it.row();
            if ((eps < (*col_it)) &&
                ((clusters[u] == UNDEFINED) || (clusters[u] == NOISE)))
                N.push_back(u);
        }

        if (N.size() < minPts) {  // Mark it as "noise"
            clusters[v] = NOISE;
            continue;
        }

        clusters[v] = ++C;

        // Mark all neighbors as visited and add them to the seed "queue"
        std::vector<int> S;
        while (!N.empty()) {
            int u = N.back();
            if ((clusters[u] == UNDEFINED) || (clusters[u] == NOISE)) {
                visited[u] = 1;
                S.push_back(u);
            }
            N.pop_back();
        }
        while (!S.empty()) {
            int u = S.back();
            S.pop_back();

            visited[u] = 1;

            if (clusters[u] == NOISE) {
                clusters[u] = C;
                continue;
            }
            clusters[u] = C;

            N.clear();  // Flush previous content
            for (arma::sp_mat::col_iterator col_it = G.begin_col(u); col_it != G.end_col(u);
                 ++col_it) {
                int w = col_it.row();
                if ((eps < (*col_it)) &&
                    ((clusters[w] == UNDEFINED) || (clusters[w] == NOISE)))
                    N.push_back(w);
            }
            if (N.size() >= minPts) {
                while (!N.empty()) {
                    int w = N.back();
                    if ((clusters[w] == UNDEFINED) || (clusters[w] == NOISE)) {
                        visited[w] = 1;
                        S.push_back(w);
                    }
                    N.pop_back();
                }
            }
        }
    }

    return (clusters);
}
