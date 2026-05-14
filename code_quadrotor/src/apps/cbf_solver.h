#ifndef CBF_SOLVER_H
#define CBF_SOLVER_H

#include <array>
#include "cbf_calc.h" // for the definition of CbfSoftminResults

#include <algorithm> // for std::max
#include <cmath>     // for math functions

#include <iostream>
#include <cstdlib>

/**
 * @brief Computes a single-exponential control input that satisfies the CBF constraint.
 *
 * Given the reference control (u_ref) and the parameters computed via the softmin approach
 * (contained in softminResults), this function evaluates
 *
 *   u = u_ref + max(beta - (alpha dot u_ref), 0) * alpha / (||alpha||^2),
 *
 * where alpha = softminResults.combined_Lg,
 *       beta  = - softminResults.combined_Lf - p1 * softminResults.combined_psi.
 *
 * If ||alpha||^2 is almost zero, u_ref is returned.
 *
 * @param u_ref Reference control input as a 3D vector.
 * @param softminResults Structure containing combined_Lg, combined_Lf, and combined_psi.
 * @param p1 The scalar parameter p1.
 * @return A 3-dimensional control input vector.
 */
std::array<float, 3> cbf_solver(const std::array<float, 3>& u_ref,
                                                     const CbfSoftminResults& softminResults,
                                                     float p1);

int cbf_solver_tests();

#endif // CBF_SOLVER_H