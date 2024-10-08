#include "generate_layout_umappp.hpp"
#include "initialize_umappp.hpp"

namespace actionet {
    arma::mat layoutNetwork_umappp(arma::sp_mat& G, arma::mat& initial_embedding, int thread_no) {
        // int threads_use = get_num_threads(SYS_THREADS_DEF, thread_no);

        stdout_printf("Initialize Options...");
        umappp::Options opt;
        opt.local_connectivity = 1;
        opt.bandwidth = 1;
        opt.mix_ratio = 1;
        opt.spread = 1;
        opt.min_dist = 1;
        opt.repulsion_strength = 1;
        opt.num_epochs = 50;
        opt.learning_rate = 1;
        opt.negative_sample_rate = NEGATIVE_SAMPLE_RATE;
        opt.seed = 0;
        opt.num_threads = 4;
        opt.parallel_optimization = true;
        stdout_printf("done\n");

        stdout_printf("Finding a/b...");
        auto found = umappp::internal::find_ab(opt.spread, opt.min_dist);
        opt.a = found.first;
        opt.b = found.second;
        stdout_printf("done\n");

        stdout_printf("Using: a = %0.3f, b = %0.3f\n", opt.a, opt.b);

        stdout_printf("Initialize NeighborList...");
        std::vector<std::vector<std::pair<int, double>>> x;
        stdout_printf("done\n");
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
        // std::vector<double> embedding = arma::conv_to<std::vector<double>>::from(initial_embedding);
        std::vector<double> embedding(initial_embedding.begin(), initial_embedding.end());

        stdout_printf("done\n");

        // size_t out_dim = 2;
        // std::vector<double> embedding(G.n_cols * out_dim);

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
