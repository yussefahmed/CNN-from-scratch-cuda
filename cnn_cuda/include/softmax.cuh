#ifndef CNN_CUDA_SOFTMAX_CUH
#define CNN_CUDA_SOFTMAX_CUH

#include "tensor.cuh"

__global__ void cuda_softmax_test_load_shared_kernel(
    CudaTensor input,
    CudaTensor output);

/**
 * Applies numerically stable Softmax independently to each tensor batch.
 *
 * All non-batch dimensions are treated as one flat logits vector. Results
 * are written in place using FP32 storage and arithmetic.
 *
 * @param tensor Initialized NHWC tensor backed by device memory.
 *
 * @return true when the Softmax kernel is launched successfully; false
 *         otherwise.
 */
__host__ bool cuda_softmax_forward(CudaTensor *tensor);

#endif
