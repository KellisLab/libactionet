#include "tools/graph_measures.hpp"

namespace ACTIONet {

    arma::uvec compute_core_number(arma::sp_mat &G) {
        unsigned int i, j = 0;
        unsigned int no_of_nodes = G.n_rows;

        // Construct node neighborhood sets
        std::vector<std::vector<unsigned int>> N(no_of_nodes);
        arma::sp_mat::const_iterator it = G.begin();
        arma::sp_mat::const_iterator it_end = G.end();

        arma::uvec cores(no_of_nodes);
        cores.zeros();
        for (; it != it_end; ++it) {
            N[it.row()].push_back((unsigned int) it.col());
            cores[it.row()]++;
        }
        unsigned int maxdeg = arma::max(cores);

        /* degree histogram */
        arma::uvec bin(maxdeg + 1);
        bin.zeros();
        for (i = 0; i < no_of_nodes; i++) {
            bin[(unsigned int) cores[i]]++;
        }

        /* start pointers */
        j = 0;
        for (i = 0; i <= maxdeg; i++) {
            unsigned int k = bin[i];
            bin[i] = j;
            j += k;
        }

        /* sort in vert (and corrupt bin) */
        arma::uvec pos(no_of_nodes);
        pos.zeros();
        arma::uvec vert(no_of_nodes);
        vert.zeros();
        for (i = 0; i < no_of_nodes; i++) {
            pos[i] = bin[(unsigned int) cores[i]];
            vert[pos[i]] = i;
            bin[(unsigned int) cores[i]]++;
        }

        /* correct bin */
        for (i = maxdeg; i > 0; i--) {
            bin[i] = bin[i - 1];
        }
        bin[0] = 0;

        /* this is the main algorithm */
        for (i = 0; i < no_of_nodes; i++) {
            unsigned int v = vert[i];

            for (j = 0; j < N[v].size(); j++) {
                unsigned int u = (N[v])[j];

                if (cores[u] > cores[v]) {
                    unsigned int du = (unsigned int) cores[u];
                    unsigned int pu = pos[u];
                    unsigned int pw = bin[du];
                    unsigned int w = vert[pw];
                    if (u != w) {
                        pos[u] = pw;
                        pos[w] = pu;
                        vert[pu] = w;
                        vert[pw] = u;
                    }
                    bin[du]++;
                    cores[u]--;
                }
            }
        }

        return (cores);
    }

    arma::uvec compute_induced_core_number(arma::sp_mat &G, arma::uvec mask) {
        unsigned int i, j = 0;
        unsigned int no_of_nodes = G.n_rows;

        // Construct node neighborhood sets
        std::vector<std::vector<unsigned int>> N(no_of_nodes);
        arma::sp_mat::const_iterator it = G.begin();
        arma::sp_mat::const_iterator it_end = G.end();

        arma::uvec cores(no_of_nodes);
        cores.zeros();
        for (; it != it_end; ++it) {
            if (mask[it.row()] && mask[it.col()]) {
                N[it.row()].push_back((unsigned int) it.col());
                cores[it.row()]++;
            }
        }
        unsigned int maxdeg = arma::max(cores);

        /* degree histogram */
        arma::uvec bin(maxdeg + 1);
        bin.zeros();
        for (i = 0; i < no_of_nodes; i++) {
            bin[(unsigned int) cores[i]]++;
        }

        /* start pointers */
        j = 0;
        for (i = 0; i <= maxdeg; i++) {
            unsigned int k = bin[i];
            bin[i] = j;
            j += k;
        }

        /* sort in vert (and corrupt bin) */
        arma::uvec pos(no_of_nodes);
        pos.zeros();
        arma::uvec vert(no_of_nodes);
        vert.zeros();
        for (i = 0; i < no_of_nodes; i++) {
            pos[i] = bin[(unsigned int) cores[i]];
            vert[pos[i]] = i;
            bin[(unsigned int) cores[i]]++;
        }

        /* correct bin */
        for (i = maxdeg; i > 0; i--) {
            bin[i] = bin[i - 1];
        }
        bin[0] = 0;

        /* this is the main algorithm */
        for (i = 0; i < no_of_nodes; i++) {
            unsigned int v = vert[i];

            for (j = 0; j < N[v].size(); j++) {
                unsigned int u = (N[v])[j];

                if (cores[u] > cores[v]) {
                    unsigned int du = (unsigned int) cores[u];
                    unsigned int pu = pos[u];
                    unsigned int pw = bin[du];
                    unsigned int w = vert[pw];
                    if (u != w) {
                        pos[u] = pw;
                        pos[w] = pu;
                        vert[pu] = w;
                        vert[pw] = u;
                    }
                    bin[du]++;
                    cores[u]--;
                }
            }
        }

        return (cores);
    }

    arma::vec compute_archetype_core_centrality(arma::sp_mat &G, arma::uvec sample_assignments) {

        arma::vec conn = arma::zeros(G.n_cols);

        for (int i = 0; i <= arma::max(sample_assignments); i++) {
            arma::uvec mask = arma::conv_to<arma::uvec>::from(sample_assignments == i);

            if (sum(mask) == 0) {
                continue;
            }
            arma::uvec induced_coreness = compute_induced_core_number(G, mask);

            arma::uvec idx = arma::find(mask > 0);
            arma::vec v = arma::conv_to<arma::vec>::from(induced_coreness(idx));
            double sigma = arma::stddev(v);
            arma::vec z = v / (sigma == 0 ? 1 : sigma);

            conn(idx) = z;
        }

        return (conn);
    }

}  // namespace ACTIONet
