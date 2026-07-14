#ifndef ACTIONET_UTILS_ACTION_NUMERIC_POLICY_HPP
#define ACTIONET_UTILS_ACTION_NUMERIC_POLICY_HPP

namespace actionet::utils_internal::action_numeric_policy {

// These constants define ACTION's internal numerical decision policy. Keep
// them backend-independent so CPU and future accelerator implementations make
// the same discrete decisions even when their floating-point reductions are
// not bitwise identical.
inline constexpr double spa_relative_tie_tolerance = 1e-16;
inline constexpr double aa_singular_squared_norm = 1e-7;

inline constexpr double simplex_l2_regularization = 1e-5;
inline constexpr double simplex_optimality_tolerance = 1e-5;
inline constexpr double active_set_zero_step_tolerance = 1e-10;

inline constexpr double landmark_proximity_tolerance = 1e-3;

// Simplex coefficients are dimensionless and constrained to [0, 1]. Values
// at or below this tolerance are numerical residue, not meaningful support.
// The comparison is intentionally strict: coefficient > tolerance.
inline constexpr double simplex_coefficient_support_tolerance = 1e-6;

} // namespace actionet::utils_internal::action_numeric_policy

#endif // ACTIONET_UTILS_ACTION_NUMERIC_POLICY_HPP
