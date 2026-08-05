#include "softmax.cuh"

#include <float.h>

__device__ void cuda_block_load_shared_values(
    const float *global_values,
    size_t value_count,
    float *shared_values)
{
    size_t thread_idx = threadIdx.x;
    size_t global_thread_idx =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (global_thread_idx < value_count){
        
        shared_values[thread_idx] = global_values[global_thread_idx];

    }
    else {
        shared_values[thread_idx] = -FLT_MAX;
    }
    __syncthreads();
}

__device__ float cuda_block_find_largest_value(float *shared_values)
{
    
    size_t thread_idx = threadIdx.x;
    

    for (size_t stride = blockDim.x / 2 ; stride > 0 ; stride /= 2){
        if (thread_idx < stride){
        shared_values[thread_idx] = fmaxf(shared_values[thread_idx] , shared_values[thread_idx + stride]);
        }

        __syncthreads();
        }
    return shared_values[0];
}

__device__ void cuda_block_exp(float largest_value , float *shared_values){

    size_t local_thread_idx = threadIdx.x;

    shared_values[local_thread_idx]= expf(shared_values[local_thread_idx] - largest_value);

}

__device__ float cuda_block_sum(float *shared_values){

    size_t thread_idx = threadIdx.x;

    float sum = 0.0f;
    for ( size_t stride = blockDim.x/2 ; stride > 0 ; stride /= 2){

        if(thread_idx < stride){

            shared_values[thread_idx] = shared_values[thread_idx]+shared_values[thread_idx + stride];
        }

    __syncthreads();

    }

    return shared_values[0];


}

__device__ void cuda_block_divide (float sum , float *shared_values){

    size_t thread_idx = threadIdx.x;
    shared_values[thread_idx] = shared_values[thread_idx] / sum;


}


__global__ void cuda_softmax_partial_max_kernel(CudaTensor input,float *partial_max_values)
{
    extern __shared__ float shared_values[];

    cuda_block_load_shared_values(
        input.device_pointer,
        input.count,
        shared_values);

    float block_maximum =
        cuda_block_find_largest_value(shared_values);

    if (threadIdx.x == 0) {
        partial_max_values[blockIdx.x] = block_maximum;
    }
}

__global__ void cuda_softmax_partial_sum_kernel(float *values){

  



}


__host__ bool cuda_softmax_forward(CudaTensor *tensor)
{
    /*. */

    return false;
}
