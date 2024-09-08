#include "visualization/generate_layout_umappp.hpp"
// #include "utils_internal/utils_parallel.hpp"
#include "visualization/initialize_umappp.hpp"

namespace actionet {
    arma::mat layoutNetwork_umappp(arma::sp_mat& G, arma::mat& initial_embedding, int thread_no) {
        // int threads_use = get_num_threads(SYS_THREADS_DEF, thread_no);

        stdout_printf("Initialize NeighborList...");
        std::vector<std::vector<std::pair<int, double>>> x;
        stdout_printf("sdone\n");
        // knncolle::NeighborList<int, double> x;

        // arma::sp_mat::const_iterator it = G.begin();
        // arma::sp_mat::const_iterator it_end = G.end();
        // for (; it != it_end; ++it) {
        //     // "val" (*it)
        //     // "row" it.row()
        //     // "col" it.col()
        // }
        stdout_printf("Generating NeighborList...");
        for (size_t i = 0; i < G.n_cols; i++) {
            arma::sp_mat::const_col_iterator it = G.begin_col(i);
            arma::sp_mat::const_col_iterator it_end = G.end_col(i);
            std::vector<std::pair<int, double>> r;
            for (; it != it_end; ++it) {
                r.emplace_back((int)it.row(), (double)(*it));
            }
            x.push_back(r);
        }
        stdout_printf("done\n");
        stdout_printf("NeighborList length: %d\n", (int)x.size());

        size_t out_dim = initial_embedding.n_cols;

        stdout_printf("Embedding dim: %d\n", (int)out_dim);

        stdout_printf("Initialize embedding...");
        std::vector<double> embedding = arma::conv_to<std::vector<double>>::from(initial_embedding);
        stdout_printf("done\n");

        // size_t out_dim = 2;
        // std::vector<double> embedding(G.n_cols * out_dim);
        stdout_printf("Initialize Options...");
        umappp::Options opt;
        stdout_printf("done\n");

        stdout_printf("Initialize umappp...");
        auto status = umappp::initialize_custom(x, out_dim, embedding.data(), opt);
        stdout_printf("done\n");

        stdout_printf("Running umappp...");
        status.run();
        stdout_printf("done\n");

        stdout_printf("Extracting embedding...");
        arma::mat embedding_out(status.embedding(), out_dim, G.n_rows);
        // arma::mat coordinates_2D = arma::trans(arma::conv_to<arma::mat>::from(coordinates_float));
        stdout_printf("done\n");

        stdout_printf("Embedding dims: %d x %d\n", (int)embedding_out.n_rows, (int)embedding_out.n_cols);

        return (embedding_out);
    }
}
