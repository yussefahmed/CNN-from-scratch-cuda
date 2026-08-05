#include "relu.cuh"

__global__ void cuda_relu_kernel(CudaTensor tensor){

    size_t thread =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (thread >= tensor.count){
        return;
    }

    float zero = 0.0f;
    if (tensor.device_pointer[thread] < zero){
        tensor.device_pointer[thread] = zero;
    }
}

__host__ bool cuda_relu_forward(CudaTensor *tensor){

    
    size_t threads = tensor->count;
    size_t blocks = (threads + 256 - 1 )  / 256;
    cuda_relu_kernel<<<blocks,256>>>(*tensor);
    return true;

}
