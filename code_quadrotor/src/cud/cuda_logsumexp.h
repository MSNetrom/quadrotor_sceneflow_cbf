#ifndef CUDA_LOGSUMEXP_H
#define CUDA_LOGSUMEXP_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute the logsumexp of a 1D array on the GPU.
 *
 * @param d_in  Pointer to the device input array (float).
 * @param d_out Pointer to the device output (a single float).
 * @param N     Number of elements in d_in.
 */
void cudaLogSumExp(const float *d_in, float *d_out, int N);

/**
 * Compute the softmax of a 1D array on the GPU.
 *
 * @param d_in  Pointer to the device input array (float).
 * @param d_out Pointer to the device output array (float) of length N.
 * @param N     Number of elements in the arrays.
 */
void cudaSoftmax(const float *d_in, float *d_out, int N);

/**
 * Compute the log softmax of a 1D array on the GPU.
 *
 * @param d_in  Pointer to the device input array (float).
 * @param d_out Pointer to the device output array (float) of length N.
 * @param N     Number of elements in the arrays.
 */
void cudaLogSoftmax(const float *d_in, float *d_out, int N);

/**
 * Unit test function for CUDA logsumexp, softmax, and logsoftmax.
 * This test uses 5 different input arrays (each of size 5) that you can manually verify.
 */
void cudaLogSumExpUnitTest(void);

#ifdef __cplusplus
}
#endif

#endif // CUDA_LOGSUMEXP_H