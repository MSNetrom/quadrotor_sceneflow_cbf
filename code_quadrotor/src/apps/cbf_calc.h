#ifndef CBF_CALC_H
#define CBF_CALC_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "../src/cud/cuda_logsumexp.h"
#include <opencv2/opencv.hpp>

/*
 * Structure to hold outputs computed on the GPU.
 *   - d_Lg_h_x: x-component of the Lie derivative Lg_h = 2·rₓ.
 *   - d_Lg_h_y: y-component of the Lie derivative Lg_h = 2·rᵧ.
 *   - d_Lg_h_z: z-component of the Lie derivative Lg_h = 2·r_z.
 *   - d_Lf_h:   Computed as 2·||ṙ||² + 2·α₁·(rᵀ·ṙ) for each point.
 *   - d_h:      Computed as 2·(rᵀ·ṙ) + α₁·(||r||² - R²) for each point.
 *   - d_psi:    [numPoints] - Add this line to store psi values
 */
struct CbfCalcOutputs {
    float *d_Lg_h_x; // [numPoints]
    float *d_Lg_h_y; // [numPoints]
    float *d_Lg_h_z; // [numPoints]
    float *d_Lf_h;   // [numPoints]
    float *d_h;      // [numPoints]
    float *d_psi;    // [numPoints]
    int numPoints;
};

/*
 * Structure to hold the softmin-combined results.
 *   - combined_Lg: Combined Lie derivative (x, y, z components).
 *   - combined_Lf: Combined Lf value.
 *   - combined_psi: Combined psi value.
 *   - min_point: x, y, z coordinates of the closest point
 *   - min_distance: Distance to the closest point
 */
struct CbfSoftminResults {
    float combined_Lg[3];
    float combined_Lf;
    float combined_psi;
};

class CbfCalc {
public:
    /*
     * Constructor accepts the safety radius, class‑K function parameter, and the maximum number
     * of points for which the output memory is preallocated.
     */
    CbfCalc(float radius, float alpha1, int maxPoints);

    // Destructor: Free preallocated GPU memory.
    ~CbfCalc();

    /*
     * Compute the outputs given separate input arrays for the x, y, and z positions and their
     * corresponding velocity components.
     */
    CbfCalcOutputs get_Lg_Lf_and_h(const float* d_r_x, const float* d_r_y, const float* d_r_z,
                                   const float* d_r_dot_x, const float* d_r_dot_y, const float* d_r_dot_z,
                                   cudaStream_t stream = 0);

    /*
     * Compute a softmin combination of the computed outputs.
     *
     * The method computes:
     *   ins[i] = -k * psi[i]    (here, psi[i] is assumed to be stored in d_h)
     *   LSE = log(sum(exp(ins)))
     *   coeff[i] = exp(ins[i]-LSE)
     *
     *   combined_psi = -LSE/k,
     *   combined_Lf = sum( coeff[i] * d_Lf_h[i] )
     *   combined_Lg = ( sum(coeff[i] * d_Lg_h_x[i]),
     *                   sum(coeff[i] * d_Lg_h_y[i]),
     *                   sum(coeff[i] * d_Lg_h_z[i]) )
     */
    CbfSoftminResults getSoftminCombined(float k, cudaStream_t stream = 0);

    // Declare the unit test as a friend so it can access the private members.
    friend int getSoftminCombined_unit_test(void);

    // Add method to get the psi values as a CV matrix (similar to how you'd get h values)
    cv::Mat getPsiMatrix(int rows, int cols) const;
    cv::Mat getHMatrix(int rows, int cols) const;

private:
    float radius;
    float alpha1;
    int maxPoints;

    // Preallocated GPU buffers for outputs.
    float *d_Lg_h_x;
    float *d_Lg_h_y;
    float *d_Lg_h_z;
    float *d_Lf_h;
    float *d_h;
    float *d_psi;  // Add this to store psi values

    // For point tracking
    float *d_min_point_x;
    float *d_min_point_y;
    float *d_min_point_z;
    float *d_min_distance;
    float min_point[3];
    float min_distance;
    float min_psi_value;
    int numPoints;
};

int getSoftminCombined_unit_test(void);

#endif // CBF_CALC_H



