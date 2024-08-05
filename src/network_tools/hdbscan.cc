#include "action_post.hpp"

namespace ACTIONet
{

  field<vec> run_HDBSCAN(mat &X, int minPoints = 5, int minClusterSize = 5)
  {
    Hdbscan hdbscan(X);
    hdbscan.execute(minPoints, minClusterSize, "Euclidean");

    vec labels(X.n_rows);
    vec membershipProbabilities(X.n_rows);
    vec outlierScores(X.n_rows);

    for (int i = 0; i < X.n_rows; i++)
    {
      labels[i] = hdbscan.labels_[i];
      membershipProbabilities[i] = hdbscan.membershipProbabilities_[i];
      outlierScores[hdbscan.outlierScores_[i].id] =
          hdbscan.outlierScores_[i].score;
    }

    field<vec> out(3);
    out(0) = labels;
    out(1) = membershipProbabilities;
    out(2) = outlierScores;

    return (out);
  }

} // namespace ACTIONet
