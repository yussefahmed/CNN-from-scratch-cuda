#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// save it as NHWC instead of NCHW because the first makes channels contigious 
typedef struct {
    size_t batch_size;
    size_t height;
    size_t width;
    size_t channels;
    size_t count; 
    __half *device_pointer; // gpu data pointer  (we store fp16 so we can use tensor core and acc at fp32)
} CudaTensor;

__host__ bool cuda_tensor_init(CudaTensor *tensor, size_t batch_size , size_t height , size_t width , size_t channels ){

    if (batch_size == 0 || height == 0 || width == 0 || channels == 0 || tensor == NULL) {
        return false;
    }

    
    // check if count can be represented (pretty much redundant?)
    if (height > SIZE_MAX / width){
        return false;
    }
    size_t count = height * width;

    if(channels > SIZE_MAX/ count){
        return false;
    }
    count = count * channels;

    if (batch_size > SIZE_MAX / count){
        return false;
    }
    count = count * batch_size;

    // check if the amount of mem we want can be presented in size_t 
    if (count > SIZE_MAX / sizeof(__half) ){
        return false;
    }

    size_t requested_device_memory = count * sizeof(__half);

    __half *device_data = NULL;


    cudaError_t error = cudaMalloc((void **) &device_data , requested_device_memory);
    if (error != cudaSuccess){
        return false;

    }
    tensor->batch_size = batch_size;
    tensor->height = height;
    tensor->width = width;
    tensor->channels = channels;
    tensor->count = count;
    tensor->device_pointer = device_data;

    return true; 
}

__host__ bool cuda_tensor_destroy(CudaTensor *tensor){
    if (tensor == NULL){
        return true;
    }


    if (tensor->device_pointer != NULL){

        cudaError_t error = cudaFree(tensor->device_pointer);
        if (error !=cudaSuccess){
            return false;
        }
        
        
    }

    tensor->device_pointer = NULL;
    tensor->batch_size = 0;
    tensor->channels = 0;
    tensor->height = 0;
    tensor->width = 0;
    tensor->count = 0; 

    return true;
}

__host__ bool cuda_tensor_copy_from_host(const CudaTensor *tensor ,  const __half *host_data){
     
    if (tensor == NULL || tensor->device_pointer == NULL || tensor->count == 0  || host_data == NULL){
        return false;
    }
    

    cudaError_t error = cudaMemcpy(tensor->device_pointer, host_data , tensor->count * sizeof(__half) , cudaMemcpyHostToDevice );
    if (error != cudaSuccess){
        return false; 
    }
    return true; 



}

__host__ bool cuda_tensor_copy_to_host(const CudaTensor *tensor , __half *host_data){
    if (tensor == NULL || host_data == NULL || tensor->count == 0 || tensor->device_pointer == NULL){
        return false;
    }

    cudaError_t error = cudaMemcpy(host_data , tensor->device_pointer , sizeof(__half) * tensor->count , cudaMemcpyDeviceToHost);
    if (error != cudaSuccess){
        return false;
    }
    return true;
}









    

    

   



