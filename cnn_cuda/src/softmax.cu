#include "softmax.cuh"

#include <float.h>






/**/
__device__ void cuda_softmax_block_load_shared_values(const CudaTensor input,
                                                      float *shared_values) {
    
    size_t local_thread_idx = threadIdx.x;
    size_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_thread_idx < input.count){
    shared_values[local_thread_idx] = input.device_pointer[global_thread_idx];
    }
    else{
        shared_values[local_thread_idx] = -FLT_MAX;
    }
    __syncthreads();
}





__device__ void cuda_softmax_find_partial_max(const float *shared_values,
                                              float *partial_max) {
    __shared__ float temp_store[128];

    size_t local_thread = threadIdx.x;
    size_t stride = blockDim.x / 2;

    if (stride == 0) {
        if (local_thread == 0) {
            partial_max[blockIdx.x] = shared_values[0];
        }
        return;
    }

    if (local_thread < stride) {
        temp_store[local_thread] =
            fmaxf(shared_values[local_thread],
                  shared_values[local_thread + stride]);
    }

    __syncthreads();

    for (stride /= 2; stride > 0; stride /= 2) {
        if (local_thread < stride) {
            temp_store[local_thread] =
                fmaxf(temp_store[local_thread],
                      temp_store[local_thread + stride]);
        }

        __syncthreads();
    }

    if (local_thread == 0) {
        partial_max[blockIdx.x] = temp_store[0];
    }
}

__device__ void cuda_softmax_find_final_max(const float *partial_max,
                                            size_t partial_count,
                                            float *final_max) {
    __shared__ float temp_store[128];

    size_t local_thread_idx = threadIdx.x;
    size_t stride = partial_count / 2;

    if (partial_count == 0) {
        return;
    }

    if (stride == 0) {
        if (local_thread_idx == 0) {
            final_max[0] = partial_max[0];
        }
        return;
    }

    if (local_thread_idx < stride) {
        temp_store[local_thread_idx] =
            fmaxf(partial_max[local_thread_idx],
                  partial_max[local_thread_idx + stride]);
    }

    __syncthreads();

    for (stride /= 2; stride > 0; stride /= 2) {
        if (local_thread_idx < stride) {
            temp_store[local_thread_idx] =
                fmaxf(temp_store[local_thread_idx],
                      temp_store[local_thread_idx + stride]);
        }

        __syncthreads();
    }

    if (local_thread_idx == 0) {
        final_max[0] = temp_store[0];
    }
}

__device__ void cuda_softmax_do_exp(CudaTensor input,
                                    float *shared_values,
                                    float max_value) {

    size_t local_thread_idx = threadIdx.x;
    size_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x ;
    if (local_thread_idx < input.count){
    input.device_pointer[global_thread_idx] = expf(shared_values[local_thread_idx] - max_value);
    }
    
}

__device__ void cuda_softmax_find_partial_sum(const float *shared_values,
                                              float *partial_sum) {
    __shared__ float temp_store[128];

    size_t local_thread_idx = threadIdx.x;
    size_t stride = blockDim.x / 2;

    if (stride == 0) {
        if (local_thread_idx == 0) {
            partial_sum[blockIdx.x] = shared_values[0];
        }
        return;
    }

    if (local_thread_idx < stride) {
        temp_store[local_thread_idx] =
            shared_values[local_thread_idx] +
            shared_values[local_thread_idx + stride];
    }

    __syncthreads();

    for (stride /= 2; stride > 0; stride /= 2) {
        if (local_thread_idx < stride) {
            temp_store[local_thread_idx] +=
                temp_store[local_thread_idx + stride];
        }

        __syncthreads();
    }

    if (local_thread_idx == 0) {
        partial_sum[blockIdx.x] = temp_store[0];
    }
}

__device__ void cuda_softmax_find_final_sum(const float *partial_sum,
                                            size_t partial_count,
                                            float *final_sum) {
    __shared__ float temp_store[128];

    size_t local_thread_idx = threadIdx.x;
    size_t stride = partial_count / 2;

    if (partial_count == 0) {
        return;
    }

    if (stride == 0) {
        if (local_thread_idx == 0) {
            final_sum[0] = partial_sum[0];
        }
        return;
    }

    if (local_thread_idx < stride) {
        temp_store[local_thread_idx] =
            partial_sum[local_thread_idx] +
            partial_sum[local_thread_idx + stride];
    }

    __syncthreads();

    for (stride /= 2; stride > 0; stride /= 2) {
        if (local_thread_idx < stride) {
            temp_store[local_thread_idx] +=
                temp_store[local_thread_idx + stride];
        }

        __syncthreads();
    }

    if (local_thread_idx == 0) {
        final_sum[0] = temp_store[0];
    }
}

__device__ void cuda_softmax_do_division(CudaTensor input,
                                         const float *final_sum) {
    size_t global_thread_idx =
        (size_t)blockDim.x * blockIdx.x + threadIdx.x;

    if (global_thread_idx < input.count) {
        input.device_pointer[global_thread_idx] /= final_sum[0];
    }
}

__global__ void cuda_global_softmax_find_partial_max(CudaTensor input,
                                      float *partial_max) {
    /* TODO(human): Load this block's values and reduce its partial maximum. */

    extern __shared__ float shared_values[];
    cuda_softmax_block_load_shared_values(input , shared_values);
    cuda_softmax_find_partial_max(shared_values, partial_max);
    
}

__global__ void cuda_global_softmax_find_final_max(float *partial_max ,size_t count , float *final_max){

    cuda_softmax_find_final_max(partial_max , count , final_max);

}


/*


__device__ void cuda_softmax_block_exp(){}


__global__ void cuda_softmax_partial_sum_kernel(){}


__host__ bool cuda_softmax_forward(){}

*/
