#include "hdbscan.h"

arma::field<arma::vec> run_HDBSCAN(arma::mat &X, int minPoints, int minClusterSize) {
    Hdbscan hdbscan(X);
    hdbscan.execute(minPoints, minClusterSize, "Euclidean");

    arma::vec labels(X.n_rows);
    arma::vec membershipProbabilities(X.n_rows);
    arma::vec outlierScores(X.n_rows);

    for (int i = 0; i < X.n_rows; i++) {
        labels[i] = hdbscan.labels_[i];
        membershipProbabilities[i] = hdbscan.membershipProbabilities_[i];
        outlierScores[hdbscan.outlierScores_[i].id] =
                hdbscan.outlierScores_[i].score;
    }

    arma::field<arma::vec> out(3);
    out(0) = labels;
    out(1) = membershipProbabilities;
    out(2) = outlierScores;

    return (out);
}
