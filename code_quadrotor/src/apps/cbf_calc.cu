#include "cbf_calc.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <float.h>

//-----------------------------------------------------------------------------
// Helper function: Copy a device array to host and print the first few entries and its min/max values.
static void debugPrintDeviceArray(const float* d_array, int numElements, const char* name) {
    // Print at most 5 elements.
    int printCount = (numElements < 5) ? numElements : 5;
    float* h_array = new float[numElements];
    cudaError_t err = cudaMemcpy(h_array, d_array, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[Debug] Error copying device array %s to host: %s\n", name, cudaGetErrorString(err));
    } else {
        // Print the first few elements.
        for (int i = 0; i < printCount; i++) {
            printf("[Debug] %s[%d] = %f\n", name, i, h_array[i]);
        }
        // Compute min and max values.
        float minVal = h_array[0];
        float maxVal = h_array[0];
        for (int i = 1; i < numElements; i++) {
            if (h_array[i] < minVal)
                minVal = h_array[i];
            if (h_array[i] > maxVal)
                maxVal = h_array[i];
        }
        printf("[Debug] %s min = %f, max = %f\n", name, minVal, maxVal);
    }
    delete[] h_array;
}

//-----------------------------------------------------------------------------
// Kernel to compute outputs for each point.
// d_r_x, d_r_y, d_r_z: positions
// d_r_dot_x, d_r_dot_y, d_r_dot_z: velocities
// Computes:
//   ψ = rₓ² + rᵧ² + r_z² - R²,
//   dot = rₓ·ṙₓ + rᵧ·ṙᵧ + r_z·ṙ_z,
//   h = 2·(dot) + α₁ · ψ,
//   Lg_h = 2·r on each component,
//   Lf_h = 2·||ṙ||² + 2·α₁·(dot).
__global__ void get_LgLfAndHKernel(const float* d_r_x, const float* d_r_y, const float* d_r_z,
                                   const float* d_r_dot_x, const float* d_r_dot_y, const float* d_r_dot_z,
                                   int numPoints,
                                   float alpha1, float radius_squared,
                                   float* d_Lg_h_x, float* d_Lg_h_y, float* d_Lg_h_z,
                                   float* d_Lf_h, float* d_h, float* d_psi)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numPoints) {
        float r_x = d_r_x[i];
        float r_y = d_r_y[i];
        float r_z = d_r_z[i];
        float rdot_x = d_r_dot_x[i]; 
        float rdot_y = d_r_dot_y[i]; 
        float rdot_z = d_r_dot_z[i]; 

        // Compute psi = rₓ² + rᵧ² + r_z² - radius_squared.
        float psi = (r_x * r_x + r_y * r_y + r_z * r_z) - radius_squared;
        
        // Store psi value
        d_psi[i] = psi;
        
        // Debug: Print psi for the first few elements.
        //if (i < 5) {
        //    printf("[Kernel: get_LgLfAndHKernel] Index %d: r=(%f, %f, %f), psi=%f\\n", i, r_x, r_y, r_z, psi);
        //}

        // Compute the dot product: rₓ·ṙₓ + rᵧ·ṙᵧ + r_z·ṙ_z.
        float dot = r_x * rdot_x + r_y * rdot_y + r_z * rdot_z;
        
        // Compute h = 2·dot + α₁ · psi.
        float h_value = 2.0f * dot + alpha1 * psi;
        
        // Debug: Print dot and h_value for the first few elements.
        //if (i < 5) {
        //    printf("[Kernel: get_LgLfAndHKernel] Index %d: dot=%f, h_value=%f\\n", i, dot, h_value);
        //}

        // Lg_h = -2·r on each component.
        d_Lg_h_x[i] = -2.0f * r_x;
        d_Lg_h_y[i] = -2.0f * r_y;
        d_Lg_h_z[i] = -2.0f * r_z;

        // Compute Lf_h = 2·||ṙ||² + 2·α₁·(dot).
        float rdot_norm_sq = rdot_x * rdot_x + rdot_y * rdot_y + rdot_z * rdot_z;
        float Lf_h_val = 2.0f * rdot_norm_sq + 2.0f * alpha1 * dot;

        d_Lf_h[i] = Lf_h_val;
        d_h[i]    = h_value;
        
        
        //if (i < 5) {
        //    printf("[Kernel: get_LgLfAndHKernel] Index %d: Lg_h=(%f, %f, %f), Lf_h=%f\\n", 
        //           i, d_Lg_h_x[i], d_Lg_h_y[i], d_Lg_h_z[i], Lf_h_val);
        //}
    }
}

//-----------------------------------------------------------------------------
// Constructor: Preallocate the device memory for outputs.
CbfCalc::CbfCalc(float radius, float alpha1, int maxPoints)
    : radius(radius), alpha1(alpha1), maxPoints(maxPoints),
      d_Lg_h_x(nullptr), d_Lg_h_y(nullptr), d_Lg_h_z(nullptr),
      d_Lf_h(nullptr), d_h(nullptr), numPoints(0)
{
    cudaError_t cudaStat;
    
    cudaStat = cudaMalloc((void**)&d_Lg_h_x, maxPoints * sizeof(float));
    if (cudaStat != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_Lg_h_x\n");
    
    cudaStat = cudaMalloc((void**)&d_Lg_h_y, maxPoints * sizeof(float));
    if (cudaStat != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_Lg_h_y\n");
    
    cudaStat = cudaMalloc((void**)&d_Lg_h_z, maxPoints * sizeof(float));
    if (cudaStat != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_Lg_h_z\n");

    cudaStat = cudaMalloc((void**)&d_Lf_h, maxPoints * sizeof(float));
    if (cudaStat != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_Lf_h\n");

    cudaStat = cudaMalloc((void**)&d_h, maxPoints * sizeof(float));
    if (cudaStat != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_h\n");
        
    // Allocate memory for tracking minimum point
    cudaStat = cudaMalloc((void**)&d_min_point_x, sizeof(float));
    if (cudaStat != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_min_point_x\n");
        
    cudaStat = cudaMalloc((void**)&d_min_point_y, sizeof(float));
    if (cudaStat != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_min_point_y\n");
        
    cudaStat = cudaMalloc((void**)&d_min_point_z, sizeof(float));
    if (cudaStat != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_min_point_z\n");
        
    cudaStat = cudaMalloc((void**)&d_min_distance, sizeof(float));
    if (cudaStat != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_min_distance\n");
        
    // Allocate memory for d_psi
    cudaMalloc(&d_psi, maxPoints * sizeof(float));
}

//-----------------------------------------------------------------------------
// Destructor: Free the preallocated device memory.
CbfCalc::~CbfCalc()
{
    cudaFree(d_Lg_h_x);
    cudaFree(d_Lg_h_y);
    cudaFree(d_Lg_h_z);
    cudaFree(d_Lf_h);
    cudaFree(d_h);
    cudaFree(d_psi);
}

//-----------------------------------------------------------------------------
// Use preallocated buffers to compute outputs with the separate input arrays.
CbfCalcOutputs CbfCalc::get_Lg_Lf_and_h(const float* h_r_x, const float* h_r_y, const float* h_r_z,
                                         const float* h_r_dot_x, const float* h_r_dot_y, const float* h_r_dot_z,
                                         cudaStream_t stream)
{
    // Save numPoints for later use
    numPoints = maxPoints;
    
    // These input pointers now refer to host (CPU) memory.
    // We need to allocate device (GPU) memory for each array and copy the data over.

    float* d_r_x_dev = nullptr;
    float* d_r_y_dev = nullptr;
    float* d_r_z_dev = nullptr;
    float* d_r_dot_x_dev = nullptr;
    float* d_r_dot_y_dev = nullptr;
    float* d_r_dot_z_dev = nullptr;
    size_t size = maxPoints * sizeof(float);
    cudaError_t err;

    // Allocate GPU memory for each input array.
    err = cudaMalloc((void**)&d_r_x_dev, size);
    if(err != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_r_x_dev: %s\n", cudaGetErrorString(err));

    err = cudaMalloc((void**)&d_r_y_dev, size);
    if(err != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_r_y_dev: %s\n", cudaGetErrorString(err));

    err = cudaMalloc((void**)&d_r_z_dev, size);
    if(err != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_r_z_dev: %s\n", cudaGetErrorString(err));

    err = cudaMalloc((void**)&d_r_dot_x_dev, size);
    if(err != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_r_dot_x_dev: %s\n", cudaGetErrorString(err));

    err = cudaMalloc((void**)&d_r_dot_y_dev, size);
    if(err != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_r_dot_y_dev: %s\n", cudaGetErrorString(err));

    err = cudaMalloc((void**)&d_r_dot_z_dev, size);
    if(err != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed for d_r_dot_z_dev: %s\n", cudaGetErrorString(err));

    // Copy the data from host to device asynchronously.
    cudaMemcpyAsync(d_r_x_dev, h_r_x, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_r_y_dev, h_r_y, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_r_z_dev, h_r_z, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_r_dot_x_dev, h_r_dot_x, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_r_dot_y_dev, h_r_dot_y, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_r_dot_z_dev, h_r_dot_z, size, cudaMemcpyHostToDevice, stream);

    // Ensure the copies are finished before proceeding.
    cudaStreamSynchronize(stream);

    // Prepare kernel launch parameters.
    float radius_squared = radius * radius;
    int threadsPerBlock = 256;
    int blocksPerGrid = (maxPoints + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the kernel with our newly copied device arrays.
    get_LgLfAndHKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_r_x_dev, d_r_y_dev, d_r_z_dev,
        d_r_dot_x_dev, d_r_dot_y_dev, d_r_dot_z_dev,
        maxPoints,
        alpha1,
        radius_squared,
        d_Lg_h_x, d_Lg_h_y, d_Lg_h_z,  // Already allocated in the constructor.
        d_Lf_h, d_h, d_psi
    );

    // Check for any kernel launch or execution errors.
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) {
        fprintf(stderr, "[Host: get_Lg_Lf_and_h] Kernel launch error: %s\n", cudaGetErrorString(errSync));
    }
    if (errAsync != cudaSuccess) {
        fprintf(stderr, "[Host: get_Lg_Lf_and_h] Kernel execution error: %s\n", cudaGetErrorString(errAsync));
    }
    
    // Free the temporary device memory allocated for the input arrays.
    cudaFree(d_r_x_dev);
    cudaFree(d_r_y_dev);
    cudaFree(d_r_z_dev);
    cudaFree(d_r_dot_x_dev);
    cudaFree(d_r_dot_y_dev);
    cudaFree(d_r_dot_z_dev);
    
    // Return the outputs (note that the output pointers remain device pointers).
    CbfCalcOutputs outputs;
    outputs.d_Lg_h_x = d_Lg_h_x;
    outputs.d_Lg_h_y = d_Lg_h_y;
    outputs.d_Lg_h_z = d_Lg_h_z;
    outputs.d_Lf_h   = d_Lf_h;
    outputs.d_h      = d_h;
    outputs.d_psi    = d_psi;
    outputs.numPoints = maxPoints;

    return outputs;
}

//-----------------------------------------------------------------------------
// Kernel to compute: d_ins[i] = -k * d_psi[i].
// Here we assume that d_h holds the psi values.
__global__ void computeInsKernel(const float* d_psi, float* d_ins, float k, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        d_ins[i] = -k * d_psi[i];
        //if (i < 5) {
        //    printf("[Kernel: computeInsKernel] Index %d: psi=%f, d_ins=%f\n", i, d_psi[i], d_ins[i]);
        //}
    }
}

//-----------------------------------------------------------------------------
// Kernel to compute coefficients: d_coeff[i] = exp( d_ins[i] - logsumexp ).
__global__ void computeCoeffKernel(const float* d_ins, float* d_coeff, float logsumexp, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        d_coeff[i] = expf(d_ins[i] - logsumexp);
        //if (i < 5) {
        //    printf("[Kernel: computeCoeffKernel] Index %d: d_ins=%f, logsumexp=%f, coeff=%f\n", i, d_ins[i], logsumexp, d_coeff[i]);
        //}
    }
}

//-----------------------------------------------------------------------------
// Compute softmin combination using the CUDA logsumexp functions and cuBLAS dot products.
CbfSoftminResults CbfCalc::getSoftminCombined(float k, cudaStream_t stream) {
    int N = maxPoints;
    float *d_ins = nullptr, *d_coeff = nullptr;
    cudaMalloc((void**)&d_ins, N * sizeof(float));
    cudaMalloc((void**)&d_coeff, N * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    //printf("[Host: getSoftminCombined] Launching computeInsKernel with %d blocks of %d threads\n", blocks, threads);
    // Compute d_ins = -k * d_h.
    computeInsKernel<<<blocks, threads, 0, stream>>>(d_h, d_ins, k, N);
    cudaDeviceSynchronize();

    // Allocate memory for logsumexp result.
    float *d_logsumexp = nullptr;
    cudaMalloc((void**)&d_logsumexp, sizeof(float));

    // Compute logsumexp over d_ins.
    //printf("[Host: getSoftminCombined] Calling cudaLogSumExp\n");
    cudaLogSumExp(d_ins, d_logsumexp, N);
    cudaDeviceSynchronize();

    float h_logsumexp = 0.0f;
    cudaMemcpy(&h_logsumexp, d_logsumexp, sizeof(float), cudaMemcpyDeviceToHost);
    //printf("[Host: getSoftminCombined] h_logsumexp = %f\n", h_logsumexp);

    // Compute coefficients: d_coeff = exp( d_ins - h_logsumexp ).
    //printf("[Host: getSoftminCombined] Launching computeCoeffKernel with %d blocks of %d threads\n", blocks, threads);
    computeCoeffKernel<<<blocks, threads, 0, stream>>>(d_ins, d_coeff, h_logsumexp, N);
    cudaDeviceSynchronize();

    // Use cuBLAS to compute weighted sums.
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    float combined_Lf = 0.0f;
    cublasSdot(handle, N, d_coeff, 1, d_Lf_h, 1, &combined_Lf);
    //printf("[Host: getSoftminCombined] combined_Lf = %f\n", combined_Lf);

    float combined_Lg_x = 0.0f, combined_Lg_y = 0.0f, combined_Lg_z = 0.0f;
    cublasSdot(handle, N, d_coeff, 1, d_Lg_h_x, 1, &combined_Lg_x);
    cublasSdot(handle, N, d_coeff, 1, d_Lg_h_y, 1, &combined_Lg_y);
    cublasSdot(handle, N, d_coeff, 1, d_Lg_h_z, 1, &combined_Lg_z);
    //printf("[Host: getSoftminCombined] combined_Lg = (%f, %f, %f)\n", combined_Lg_x, combined_Lg_y, combined_Lg_z);

    cublasDestroy(handle);

    float combined_psi = -h_logsumexp / k;
    //printf("[Host: getSoftminCombined] combined_psi = %f\n", combined_psi);

    cudaFree(d_ins);
    cudaFree(d_coeff);
    cudaFree(d_logsumexp);

    CbfSoftminResults result;
    result.combined_Lg[0] = combined_Lg_x;
    result.combined_Lg[1] = combined_Lg_y;
    result.combined_Lg[2] = combined_Lg_z;
    result.combined_Lf   = combined_Lf;
    result.combined_psi  = combined_psi;
    
    return result;
}

int getSoftminCombined_unit_test(void)
{
    // Use 2 test points
    const int n = 3;

    // Define test values on the host.
    // We assume here that d_h will hold psi values.
    float h_d_h[3]     = { 2.0f, 4.0f, 6.0f };        // psi values
    float h_d_Lf_h[3]  = { 1.0f, 3.0f, 5.0f };          // Lf_h values
    float h_d_Lg_h_x[3] = { 1.0f, 1.0f, 1.0f };         // Lg_h x-components
    float h_d_Lg_h_y[3] = { 0.5f, 0.5f, 0.9f };         // Lg_h y-components
    float h_d_Lg_h_z[3] = { 0.0f, 2.0f, 2.0f };          // Lg_h z-components

    // Instantiate the calculator.
    // The radius and alpha1 values are not used in getSoftminCombined,
    // so we can set them arbitrarily.
    CbfCalc calc(1.0f, 1.0f, n);

    // Copy the test values from host to the GPU memory preallocated inside the CbfCalc object.
    // Note: For unit testing, you may need to make these members public or add setter methods.
    cudaMemcpy(calc.d_h, h_d_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(calc.d_Lf_h, h_d_Lf_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(calc.d_Lg_h_x, h_d_Lg_h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(calc.d_Lg_h_y, h_d_Lg_h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(calc.d_Lg_h_z, h_d_Lg_h_z, n * sizeof(float), cudaMemcpyHostToDevice);

    // Set the softmin parameter k.
    float k = 1.0f;

    // Call getSoftminCombined.
    CbfSoftminResults res = calc.getSoftminCombined(k);

    // Print out the results.
    printf("Unit Test Results for getSoftminCombined:\n");
    printf("Combined psi: %f\n", res.combined_psi);
    printf("Combined Lf: %f\n", res.combined_Lf);
    printf("Combined Lg: (%f, %f, %f)\n", res.combined_Lg[0],
                                           res.combined_Lg[1],
                                           res.combined_Lg[2]);

    // Expected (approximate) results:
    // combined_psi ~ 1.87307
    // combined_Lf  ~ 1.237
    // combined_Lg  ~ (1.0, 0.38, 0.24)

    return 0;
}

cv::Mat CbfCalc::getPsiMatrix(int rows, int cols) const {
    // Create a new matrix to hold the psi values
    cv::Mat psiMat(rows, cols, CV_32FC1);
    
    // Copy data from device to host
    cudaMemcpy(psiMat.data, d_psi, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    return psiMat;
}

cv::Mat CbfCalc::getHMatrix(int rows, int cols) const {
    // Create a new matrix to hold the h values
    cv::Mat hMat(rows, cols, CV_32FC1);
    
    // Copy data from device to host
    cudaMemcpy(hMat.data, d_h, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    return hMat;
}





