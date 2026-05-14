// cuda_logsumexp.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// choose a block size for the reductions and elementwise kernels
#define BLOCK_SIZE 256

// -----------------------------------------------------------------------------
// Reduction kernel to compute the maximum of an array
// -----------------------------------------------------------------------------
__global__ void reduce_max_kernel(const float *d_in, float *d_out, int N)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    // Each block processes two elements per thread if available
    int i = blockIdx.x * (BLOCK_SIZE * 2) + tid;
    float max_val = -FLT_MAX;
    if (i < N) {
        max_val = d_in[i];
        if (i + BLOCK_SIZE < N) {
            float temp = d_in[i + BLOCK_SIZE];
            if (temp > max_val)
                max_val = temp;
        }
    }
    sdata[tid] = max_val;
    __syncthreads();

    // do reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

// -----------------------------------------------------------------------------
// Reduction kernel to compute the sum of an array.
// -----------------------------------------------------------------------------
__global__ void reduce_sum_kernel(const float *d_in, float *d_out, int N)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * (BLOCK_SIZE * 2) + tid;
    float sum = 0.0f;
    if (i < N) {
        sum = d_in[i];
        if (i + BLOCK_SIZE < N)
            sum += d_in[i + BLOCK_SIZE];
    }
    sdata[tid] = sum;
    __syncthreads();

    // do reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

// -----------------------------------------------------------------------------
// Kernel: apply exp(x - max_val)
// -----------------------------------------------------------------------------
__global__ void exp_shift_kernel(const float *d_in, float *d_out, float max_val, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        d_out[i] = expf(d_in[i] - max_val);
}

// -----------------------------------------------------------------------------
// Kernel: compute softmax output (divide by sum)
// -----------------------------------------------------------------------------
__global__ void softmax_kernel(const float *d_exp, float *d_out, float sum_exp, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        d_out[i] = d_exp[i] / sum_exp;
}

// -----------------------------------------------------------------------------
// Kernel: compute log softmax output: out = x - logsumexp
// -----------------------------------------------------------------------------
__global__ void log_softmax_kernel(const float *d_in, float *d_out, float lse, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        d_out[i] = d_in[i] - lse;
}

// -----------------------------------------------------------------------------
// Host helper: iteratively reduce an array using reduce_max_kernel
// Returns the maximum value.
// -----------------------------------------------------------------------------
float reduce_max_host(const float *d_in, int N)
{
    float *d_current = (float *)d_in;
    int currentN = N;
    float result;
    float *d_temp = NULL;

    // We iteratively launch the reduction kernel until we have one value.
    while (currentN > 1) {
        int blocks = (currentN + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
        cudaMalloc(&d_temp, blocks * sizeof(float));
        reduce_max_kernel<<<blocks, BLOCK_SIZE>>>(d_current, d_temp, currentN);
        cudaDeviceSynchronize();
        if (d_current != d_in)
            cudaFree(d_current);
        d_current = d_temp;
        currentN = blocks;
        d_temp = NULL;
    }
    cudaMemcpy(&result, d_current, sizeof(float), cudaMemcpyDeviceToHost);
    if (d_current != d_in)
        cudaFree(d_current);
    return result;
}

// -----------------------------------------------------------------------------
// Host helper: iteratively reduce an array using reduce_sum_kernel
// Returns the sum.
// -----------------------------------------------------------------------------
float reduce_sum_host(const float *d_in, int N)
{
    float *d_current = (float *)d_in;
    int currentN = N;
    float result;
    float *d_temp = NULL;

    while (currentN > 1) {
        int blocks = (currentN + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
        cudaMalloc(&d_temp, blocks * sizeof(float));
        reduce_sum_kernel<<<blocks, BLOCK_SIZE>>>(d_current, d_temp, currentN);
        cudaDeviceSynchronize();
        if (d_current != d_in)
            cudaFree(d_current);
        d_current = d_temp;
        currentN = blocks;
        d_temp = NULL;
    }
    cudaMemcpy(&result, d_current, sizeof(float), cudaMemcpyDeviceToHost);
    if (d_current != d_in)
        cudaFree(d_current);
    return result;
}

// -----------------------------------------------------------------------------
// GPU function: Compute logsumexp for a 1D array
//
// d_in  : pointer to device input (array of float)
// d_out : pointer to device output (a single float, result of logsumexp)
// N     : number of elements in d_in
// -----------------------------------------------------------------------------
extern "C" void cudaLogSumExp(const float *d_in, float *d_out, int N)
{
    // 1. Compute maximum value (for stability)
    float max_val = reduce_max_host(d_in, N);

    // 2. Compute exp(x - max) for each element.
    float *d_exp;
    cudaMalloc(&d_exp, N * sizeof(float));
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    exp_shift_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_exp, max_val, N);
    cudaDeviceSynchronize();

    // 3. Compute sum(exp(x - max))
    float sum_exp = reduce_sum_host(d_exp, N);
    cudaFree(d_exp);

    // 4. Compute logsumexp = log(sum_exp) + max_val
    float lse = logf(sum_exp) + max_val;
    cudaMemcpy(d_out, &lse, sizeof(float), cudaMemcpyHostToDevice);
}

// -----------------------------------------------------------------------------
// GPU function: Compute softmax for a 1D array
//
// d_in  : pointer to device input array (float)
// d_out : pointer to device output array (float), same length as d_in
// N     : number of elements in d_in/d_out
// -----------------------------------------------------------------------------
extern "C" void cudaSoftmax(const float *d_in, float *d_out, int N)
{
    // Compute maximum value for numerical stability.
    float max_val = reduce_max_host(d_in, N);

    // Compute exp(x - max).
    float *d_exp;
    cudaMalloc(&d_exp, N * sizeof(float));
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    exp_shift_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_exp, max_val, N);
    cudaDeviceSynchronize();

    // Sum of exponents.
    float sum_exp = reduce_sum_host(d_exp, N);

    // Divide: softmax = exp(x - max) / sum(exp(x - max))
    softmax_kernel<<<blocks, BLOCK_SIZE>>>(d_exp, d_out, sum_exp, N);
    cudaDeviceSynchronize();
    cudaFree(d_exp);
}

// -----------------------------------------------------------------------------
// GPU function: Compute log-softmax for a 1D array
//
// d_in  : pointer to device input array (float)
// d_out : pointer to device output array (float), same length as d_in
// N     : number of elements in d_in/d_out
// -----------------------------------------------------------------------------
extern "C" void cudaLogSoftmax(const float *d_in, float *d_out, int N)
{
    // Compute maximum value.
    float max_val = reduce_max_host(d_in, N);

    // Compute shifted exponentials.
    float *d_exp;
    cudaMalloc(&d_exp, N * sizeof(float));
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    exp_shift_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_exp, max_val, N);
    cudaDeviceSynchronize();

    // Compute sum and then logsumexp.
    float sum_exp = reduce_sum_host(d_exp, N);
    cudaFree(d_exp);
    float lse = logf(sum_exp) + max_val;

    // log_softmax = x - logsumexp.
    log_softmax_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_out, lse, N);
    cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------
// Unit test function for CUDA logsumexp, softmax, and logsoftmax.
// This test uses 5 different input arrays (each of size 5) that you can manually verify.
// -----------------------------------------------------------------------------
extern "C" void cudaLogSumExpUnitTest(void)
{
    // Define 5 test cases (each with 5 elements)
    #define TEST_SIZE 5
    const int numTests = 5;
    float testInputs[numTests][TEST_SIZE] = {
        {  1.0f,   2.0f,   3.0f,   4.0f,   5.0f},    // ascending
        {  5.0f,   4.0f,   3.0f,   2.0f,   1.0f},    // descending
        { -1.0f,   0.0f,   1.0f,   2.0f,  -2.0f},    // mixed signs
        {  3.0f,   3.0f,   3.0f,   3.0f,   3.0f},    // all identical
        {-100.0f, -100.0f,  100.0f, -100.0f, -100.0f}   // one very large value
    };

    for (int t = 0; t < numTests; t++) {
        printf("========================================\n");
        printf("[Test %d] Input array: ", t + 1);
        for (int i = 0; i < TEST_SIZE; i++) {
            printf("%.2f ", testInputs[t][i]);
        }
        printf("\n");

        // Allocate GPU memory for the input
        float *d_in = nullptr;
        cudaMalloc(&d_in, TEST_SIZE * sizeof(float));
        cudaMemcpy(d_in, testInputs[t], TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        // Test cudaLogSumExp
        float *d_out = nullptr;
        cudaMalloc(&d_out, sizeof(float));
        cudaLogSumExp(d_in, d_out, TEST_SIZE);
        float lse;
        cudaMemcpy(&lse, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        printf("  LogSumExp  = %f\n", lse);
        cudaFree(d_out);

        // Test cudaSoftmax and cudaLogSoftmax
        float *d_softmax = nullptr, *d_logsoftmax = nullptr;
        cudaMalloc(&d_softmax, TEST_SIZE * sizeof(float));
        cudaMalloc(&d_logsoftmax, TEST_SIZE * sizeof(float));

        cudaSoftmax(d_in, d_softmax, TEST_SIZE);
        cudaLogSoftmax(d_in, d_logsoftmax, TEST_SIZE);

        float h_softmax[TEST_SIZE];
        float h_logsoftmax[TEST_SIZE];
        cudaMemcpy(h_softmax, d_softmax, TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_logsoftmax, d_logsoftmax, TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        printf("  Softmax    = ");
        for (int i = 0; i < TEST_SIZE; i++)
            printf("%f ", h_softmax[i]);
        printf("\n");

        printf("  LogSoftmax = ");
        for (int i = 0; i < TEST_SIZE; i++)
            printf("%f ", h_logsoftmax[i]);
        printf("\n");

        // Free allocated GPU memory for this test case
        cudaFree(d_softmax);
        cudaFree(d_logsoftmax);
        cudaFree(d_in);
        printf("========================================\n\n");
    }
}

#ifdef TEST_CUDA_LOGSUMEXP
// (Optional) Debug/test main()
int main(void)
{
    cudaLogSumExpUnitTest();
    return 0;
}
#endif