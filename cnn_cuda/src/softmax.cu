#include "softmax.cuh"
#include <float.h>




__global__ void cuda_softmax_max_reduce0(float *input, float *output,
                                         size_t input_count) {

    extern __shared__ float data_s[];

    // eah thread loads one elem from global to shared memory
    size_t local_thread_idx = threadIdx.x;
    size_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (global_thread_idx < input_count) {
        data_s[local_thread_idx] = input[global_thread_idx];
    } else {
        data_s[local_thread_idx] = -FLT_MAX;
    }
    __syncthreads();

    //do reduction in shared memory
    for (size_t active_count = blockDim.x; active_count > 1;) {
        size_t next_count = (active_count + 1) / 2;

        if (local_thread_idx < active_count / 2) {
            data_s[local_thread_idx] =
                fmaxf(data_s[local_thread_idx],
                      data_s[local_thread_idx + next_count]);
        }
        __syncthreads();

        active_count = next_count;
    }

    if (local_thread_idx == 0){
        output[blockIdx.x] = data_s[0];
    }

}

__global__ void cuda_softmax_max(float *input, float *output) {
    extern __shared__ float data_s[];
    size_t local_thread_idx = threadIdx.x;
    data_s[local_thread_idx] = input[local_thread_idx];
    __syncthreads();

    for (size_t active_count = blockDim.x; active_count > 1;) {
        size_t next_count = (active_count + 1) / 2;

        if (local_thread_idx < active_count / 2) {
            data_s[local_thread_idx] =
                fmaxf(data_s[local_thread_idx],
                      data_s[local_thread_idx + next_count]);
        }
        __syncthreads();

        active_count = next_count;
    }

    if (local_thread_idx == 0){
        output[0] = data_s[0];
    }


}

__global__ void cuda_softmax_exp(float *input, float *output,
                                 const float *max_value,
                                 size_t input_count) {
    size_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (global_thread_idx < input_count) {
        output[global_thread_idx] =
            expf(input[global_thread_idx] - max_value[0]);
    }

}


__global__ void cuda_softmax_sum_reduce0(float *input, float *output,
                                         size_t input_count) {

    extern __shared__ float data_s[];
    size_t local_thread_idx = threadIdx.x;
    size_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_thread_idx < input_count) {
        data_s[local_thread_idx] = input[global_thread_idx];
    } else {
        data_s[local_thread_idx] = 0.0f;
    }
    __syncthreads();

    for (size_t active_count = blockDim.x; active_count > 1;) {
        size_t next_count = (active_count + 1) / 2;

        if (local_thread_idx < active_count / 2) {
            data_s[local_thread_idx] +=
                data_s[local_thread_idx + next_count];
        }
        __syncthreads();

        active_count = next_count;
    }

    if (local_thread_idx == 0){
        output[blockIdx.x] = data_s[0];
    }

}

__global__ void cuda_softmax_sum(float *input , float *output){

    extern __shared__ float data_s[];
    size_t local_thread_idx = threadIdx.x;
    data_s[local_thread_idx] = input[local_thread_idx];
    __syncthreads();

    for (size_t active_count = blockDim.x; active_count > 1;) {
        size_t next_count = (active_count + 1) / 2;

        if (local_thread_idx < active_count / 2) {
            data_s[local_thread_idx] +=
                data_s[local_thread_idx + next_count];
        }
        __syncthreads();

        active_count = next_count;
    }

    if (local_thread_idx == 0){
        output[0]= data_s[0];
    }

}


__global__ void cuda_softmax_div(float *input, float *output,
                                 const float *sum,
                                 size_t input_count) {
    size_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (global_thread_idx < input_count) {
        output[global_thread_idx] = input[global_thread_idx] / sum[0];
    }
}


__host__ void cuda_softmax_forward(CudaTensor *input , CudaTensor *output){

    size_t number_of_threads = 256;
    size_t number_of_blocks =
        (input->count + number_of_threads - 1) / number_of_threads;
    size_t shared_memory_size = number_of_threads * sizeof(float);
    size_t final_reduction_shared_memory_size =
        number_of_blocks * sizeof(float);

    float *softmax_max_reduce0_output;
    float *softmax_max_output;
    float *softmax_sum_reduce0_output;
    float *softmax_sum_output;

    cudaMalloc((void **)&softmax_max_reduce0_output,
               number_of_blocks * sizeof(float));
    cudaMalloc((void **)&softmax_max_output, sizeof(float));
    cudaMalloc((void **)&softmax_sum_reduce0_output,
               number_of_blocks * sizeof(float));
    cudaMalloc((void **)&softmax_sum_output, sizeof(float));

    cuda_softmax_max_reduce0<<<number_of_blocks,
                               number_of_threads,
                               shared_memory_size>>>(
        input->device_pointer,
        softmax_max_reduce0_output,
        input->count);

    cuda_softmax_max<<<1,
                       number_of_blocks,
                       final_reduction_shared_memory_size>>>(
        softmax_max_reduce0_output,
        softmax_max_output);

    cuda_softmax_exp<<<number_of_blocks, number_of_threads>>>(
        input->device_pointer,
        output->device_pointer,
        softmax_max_output,
        input->count);

    cuda_softmax_sum_reduce0<<<number_of_blocks,
                               number_of_threads,
                               shared_memory_size>>>(
        output->device_pointer,
        softmax_sum_reduce0_output,
        input->count);

    cuda_softmax_sum<<<1,
                       number_of_blocks,
                       final_reduction_shared_memory_size>>>(
        softmax_sum_reduce0_output,
        softmax_sum_output);

    cuda_softmax_div<<<number_of_blocks, number_of_threads>>>(
        output->device_pointer,
        output->device_pointer,
        softmax_sum_output,
        input->count);

    cudaDeviceSynchronize();

    cudaFree(softmax_max_reduce0_output);
    cudaFree(softmax_max_output);
    cudaFree(softmax_sum_reduce0_output);
    cudaFree(softmax_sum_output);
}
