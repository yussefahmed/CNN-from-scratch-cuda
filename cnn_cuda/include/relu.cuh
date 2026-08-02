#ifndef CNN_CUDA_RELU_CUH
#define CNN_CUDA_RELU_CUH

#include "tensor.cuh"

/**
 * Applies ReLU in place to every element of an FP16 CUDA tensor.
 *
 * @param tensor Initialized tensor backed by device memory.
 *
 * @return true when the kernel is launched successfully; false otherwise.
 */
__host__ bool cuda_relu_forward(CudaTensor *tensor);

#endif
