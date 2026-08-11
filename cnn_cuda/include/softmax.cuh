#ifndef CNN_CUDA_SOFTMAX_CUH
#define CNN_CUDA_SOFTMAX_CUH

#include "tensor.cuh"

/**
 * Applies numerically stable Softmax to a tensor.
 *
 * The input and output tensors must already be initialized with matching
 * element counts. All elements are treated as one flat logits vector.
 *
 * @param input Initialized tensor containing logits in device memory.
 * @param output Initialized tensor receiving probabilities in device memory.
 */
__host__ void cuda_softmax_forward(CudaTensor *input, CudaTensor *output);

#endif
