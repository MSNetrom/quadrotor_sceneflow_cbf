#include "cbf_solver.h"

std::array<float, 3> cbf_solver(const std::array<float, 3>& u_ref,
                                                     const CbfSoftminResults& softminResults,
                                                     float p1) {
    // alpha corresponds to Lg_psi in the Python code.
    std::array<float, 3> alpha = { softminResults.combined_Lg[0],
                                   softminResults.combined_Lg[1],
                                   softminResults.combined_Lg[2] };

    // Compute beta = -Lf_psi - p1 * psi, where Lf_psi and psi correspond to combined_Lf and combined_psi.
    float beta = -softminResults.combined_Lf - p1 * softminResults.combined_psi;
    
    // Compute the squared length of alpha.
    float a_len_squared = alpha[0] * alpha[0] + alpha[1] * alpha[1] + alpha[2] * alpha[2];
    
    // If the squared length is (almost) zero, return the reference control.
    if (a_len_squared < 1e-6f) {
        return u_ref;
    }
    
    // Compute the dot product (alpha • u_ref).
    float dot_product = alpha[0] * u_ref[0] + alpha[1] * u_ref[1] + alpha[2] * u_ref[2];
    
    // Calculate the scaling factor: max(beta - (alpha • u_ref), 0)
    float factor = std::max(beta - dot_product, 0.0f);
    
    // Compute the final result: u = u_ref + (factor * alpha) / a_len_squared.
    std::array<float, 3> result;
    result[0] = u_ref[0] + factor * alpha[0] / a_len_squared;
    result[1] = u_ref[1] + factor * alpha[1] / a_len_squared;
    result[2] = u_ref[2] + factor * alpha[2] / a_len_squared;
    
    return result;
}

// Helper function to compare two floats.
bool is_close(float a, float b, float tol = 1e-6f) {
    return std::fabs(a - b) < tol;
}

// Helper function to print a 3-dimensional vector.
void print_vector(const std::array<float, 3>& vec) {
    std::cout << "[" << vec[0] << ", " << vec[1] << ", " << vec[2] << "]";
}

// Test when alpha is zero so that u_ref is immediately returned.
void testReturnsReferenceWhenAlphaIsZero() {
    std::cout << "Running testReturnsReferenceWhenAlphaIsZero..." << std::endl;
    std::array<float, 3> u_ref = {1.0f, 2.0f, 3.0f};

    CbfSoftminResults softminResults;
    // Set each element of combined_Lg to 0.0f.
    softminResults.combined_Lg[0] = 0.0f;
    softminResults.combined_Lg[1] = 0.0f;
    softminResults.combined_Lg[2] = 0.0f;
    // Other parameters are arbitrary.
    softminResults.combined_Lf = 5.0f;
    softminResults.combined_psi = 10.0f;

    float p1 = 1.0f;

    std::array<float, 3> result = cbf_solver(u_ref, softminResults, p1);

    std::cout << "Output u: ";
    print_vector(result);
    std::cout << std::endl;

    if (!(is_close(result[0], u_ref[0]) &&
          is_close(result[1], u_ref[1]) &&
          is_close(result[2], u_ref[2]))) {
        std::cerr << "testReturnsReferenceWhenAlphaIsZero failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Test when the computed scaling factor is positive.
void testPositiveFactorUpdatesControl() {
    std::cout << "Running testPositiveFactorUpdatesControl..." << std::endl;
    std::array<float, 3> u_ref = {1.0f, 2.0f, 3.0f};

    CbfSoftminResults softminResults;
    // Set combined_Lg to a non-zero vector: {0, 1, 0}.
    softminResults.combined_Lg[0] = 0.0f;
    softminResults.combined_Lg[1] = 1.0f;
    softminResults.combined_Lg[2] = 0.0f;
    // Choose combined_Lf and combined_psi so that:
    //   beta = -combined_Lf - p1 * combined_psi = 6.
    // With p1 = 1, setting combined_Lf = -1 and combined_psi = -5 yields beta = 1 + 5 = 6.
    softminResults.combined_Lf = -1.0f;
    softminResults.combined_psi = -5.0f;

    float p1 = 1.0f;

    // Expected:
    // dot_product = (0*1 + 1*2 + 0*3) = 2.
    // factor = max(6 - 2, 0) = 4.
    // Since ||alpha||² = 1, u is updated as u = u_ref + (4 * alpha) = {1, 6, 3}.
    std::array<float, 3> expected = {1.0f, 6.0f, 3.0f};

    std::array<float, 3> result = cbf_solver(u_ref, softminResults, p1);
    
    std::cout << "Output u: ";
    print_vector(result);
    std::cout << std::endl;
    
    if (!(is_close(result[0], expected[0]) &&
          is_close(result[1], expected[1]) &&
          is_close(result[2], expected[2]))) {
        std::cerr << "testPositiveFactorUpdatesControl failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Test when beta - (alpha dot u_ref) is negative so that factor is clamped to zero.
void testNegativeFactorReturnsReference() {
    std::cout << "Running testNegativeFactorReturnsReference..." << std::endl;
    std::array<float, 3> u_ref = {2.0f, 2.0f, 2.0f};

    CbfSoftminResults softminResults;
    // Set combined_Lg to a non-zero vector: {1, 0, 0}.
    softminResults.combined_Lg[0] = 1.0f;
    softminResults.combined_Lg[1] = 0.0f;
    softminResults.combined_Lg[2] = 0.0f;
    // Here, dot_product = 2.
    // Choose beta = 1 so that factor = max(1 - 2, 0) = 0.
    // For p1 = 1, one option is: combined_Lf = -2 and combined_psi = 1.
    softminResults.combined_Lf = -2.0f;
    softminResults.combined_psi = 1.0f;

    float p1 = 1.0f;

    std::array<float, 3> result = cbf_solver(u_ref, softminResults, p1);
    
    std::cout << "Output u: ";
    print_vector(result);
    std::cout << std::endl;

    if (!(is_close(result[0], u_ref[0]) &&
          is_close(result[1], u_ref[1]) &&
          is_close(result[2], u_ref[2]))) {
        std::cerr << "testNegativeFactorReturnsReference failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Test when alpha is nearly zero (but not exactly) to verify threshold handling.
void testNearlyZeroAlphaReturnsReference() {
    std::cout << "Running testNearlyZeroAlphaReturnsReference..." << std::endl;
    std::array<float, 3> u_ref = {3.0f, -1.0f, 0.5f};

    CbfSoftminResults softminResults;
    // Set each element of combined_Lg to a very small value.
    softminResults.combined_Lg[0] = 1e-4f;
    softminResults.combined_Lg[1] = 1e-4f;
    softminResults.combined_Lg[2] = 1e-4f;
    softminResults.combined_Lf = -5.0f;
    softminResults.combined_psi = 10.0f;

    float p1 = 1.0f;

    std::array<float, 3> result = cbf_solver(u_ref, softminResults, p1);
    
    std::cout << "Output u: ";
    print_vector(result);
    std::cout << std::endl;
    
    if (!(is_close(result[0], u_ref[0]) &&
          is_close(result[1], u_ref[1]) &&
          is_close(result[2], u_ref[2]))) {
        std::cerr << "testNearlyZeroAlphaReturnsReference failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Test when beta exactly equals the dot product so that factor is zero.
void testZeroFactorReturnsReference() {
    std::cout << "Running testZeroFactorReturnsReference..." << std::endl;
    std::array<float, 3> u_ref = {5.0f, 5.0f, 5.0f};

    CbfSoftminResults softminResults;
    // Set combined_Lg to a non-zero vector: {1, 2, 3}.
    softminResults.combined_Lg[0] = 1.0f;
    softminResults.combined_Lg[1] = 2.0f;
    softminResults.combined_Lg[2] = 3.0f;
    // For u_ref = {5, 5, 5}, the dot product equals (5*1 + 5*2 + 5*3) = 30.
    // To get beta = 30 when p1 = 1, choose combined_Lf = -10 and combined_psi = -20.
    softminResults.combined_Lf = -10.0f;
    softminResults.combined_psi = -20.0f;

    float p1 = 1.0f;

    std::array<float, 3> result = cbf_solver(u_ref, softminResults, p1);
    
    std::cout << "Output u: ";
    print_vector(result);
    std::cout << std::endl;
    
    if (!(is_close(result[0], u_ref[0]) &&
          is_close(result[1], u_ref[1]) &&
          is_close(result[2], u_ref[2]))) {
        std::cerr << "testZeroFactorReturnsReference failed." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int cbf_solver_tests() {
    testReturnsReferenceWhenAlphaIsZero();
    testPositiveFactorUpdatesControl();
    testNegativeFactorReturnsReference();
    testNearlyZeroAlphaReturnsReference();
    testZeroFactorReturnsReference();

    std::cout << "All tests passed successfully." << std::endl;
    return EXIT_SUCCESS;
}