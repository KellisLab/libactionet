#ifndef SIMPLEX_REGRESSION_FW_H
#define SIMPLEX_REGRESSION_FW_H

#include "action.hpp"
#include <chrono>

arma::mat run_simplex_regression_FW_base(arma::mat &A, arma::mat &B, int max_iter, double min_diff);

arma::mat run_simplex_regression_FW_test1(arma::mat &A, arma::mat &B, int max_iter, double min_diff);

arma::mat run_simplex_regression_FW_working(arma::mat &A, arma::mat &B, int max_iter, double min_diff);

arma::mat run_simplex_regression_FW(arma::mat &A, arma::mat &B, int max_iter, double min_diff);

#endif //SIMPLEX_REGRESSION_FW_H
