#ifndef ROTATION_UTIL_HPP
#define ROTATION_UTIL_HPP

#include <array>

// Rotates a 3D vector by the given 3x3 matrix.
// Computes: result[i] = sum_{j=0}^{2} matrix[i][j] * vec[j]
inline std::array<float, 3> rotate(const float matrix[3][3], const std::array<float, 3>& vec) {
    std::array<float, 3> result = { 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

// Rotates a 3D vector by the transpose of the given 3x3 matrix.
// Computes: result[i] = sum_{j=0}^{2} matrix[j][i] * vec[j]
inline std::array<float, 3> rotateTransposed(const float matrix[3][3], const std::array<float, 3>& vec) {
    std::array<float, 3> result = { 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i] += matrix[j][i] * vec[j];
        }
    }
    return result;
}

#endif // ROTATION_UTIL_HPP