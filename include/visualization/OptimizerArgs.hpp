// #ifndef ACTIONET_OPTIMIZERARGS_HPP
// #define ACTIONET_OPTIMIZERARGS_HPP
//
// // Constants
// constexpr int METHOD_ADAM = 1;
// constexpr int METHOD_SGD = 2;
// constexpr float OPT_ALPHA = 1.0; /*same as learning_rate*/
// constexpr float ADAM_BETA1 = 0.5; /* between 0 and 1*/
// constexpr float ADAM_BETA2 = 0.9; /* between 0 and 1*/
// constexpr float ADAM_EPS = 1e-7; /* between 1e-8 and 1e-3*/
//
// class OptimizerArgs {
// public:
//     int opt_method = METHOD_ADAM;
//     float alpha = OPT_ALPHA;
//     float beta1 = ADAM_BETA1;
//     float beta2 = ADAM_BETA2;
//     float eps = ADAM_EPS;
//     // Default constructor
//
//     OptimizerArgs() = default;
//
//     // Minimal constructor
//     explicit OptimizerArgs(const float alpha) : alpha(alpha) {}
//
//     // Full constructor
//     OptimizerArgs(const std::string& method, const float alpha, const float beta1, const float beta2,
//                   const float eps)
//         : alpha(alpha), beta1(beta1), beta2(beta2), eps(eps) {
//         this->opt_method = (method == "sgd") ? METHOD_SGD : METHOD_ADAM;
//     }
// };
//
// #endif //ACTIONET_OPTIMIZERARGS_HPP
