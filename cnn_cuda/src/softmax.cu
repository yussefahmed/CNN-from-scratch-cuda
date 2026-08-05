#include "softmax.cuh"

#include <float.h>


__device__ void cuda_block_load_shared_values(){}

__device__ float cuda_block_find_largest_value(){}

__device__ void cuda_block_exp(){}

__device__ float cuda_block_sum(){}

__device__ void cuda_block_divide (){}


__global__ void cuda_softmax_partial_max_kernel(){}

__global__ void cuda_softmax_partial_sum_kernel(){}


__host__ bool cuda_softmax_forward(){}
