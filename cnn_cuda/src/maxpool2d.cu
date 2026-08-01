#include "maxpool2d.cuh"

__host__ bool cuda_maxpool2d_init(CudaMaxPool2DLayer *layer,
                                  size_t kernel_height,
                                  size_t kernel_width,
                                  size_t stride)
{
    if (layer == NULL || kernel_height == 0 || kernel_width == 0 || stride == 0){
        return false;
    }
    layer->kernel_height = kernel_height;
    layer->kernel_width = kernel_width;
    layer->stride = stride;
    return true; 
}

__host__ bool cuda_maxpool2d_output_shape(
    const CudaMaxPool2DLayer *layer,
    const CudaTensor *input,
    size_t *output_height,
    size_t *output_width)
{
    if (layer == NULL || input == NULL){
        return false;
    }
    if (layer->kernel_width == 0 || layer->kernel_height == 0 || layer->stride == 0){
        return false;
    }

    if (input->batch_size == 0 || input->channels == 0 || input->height == 0 || input->width == 0){
        return false; 
    }
    if (output_height == NULL || output_width == NULL){
        return false;
    }

    if (layer->kernel_height > input->height || layer->kernel_width > input->width){
       
      return false;
    }

    *output_height = (input->height - layer->kernel_height) / layer->stride + 1;

    *output_width = (input->width - layer->kernel_width) / layer->stride + 1;



    
    return true;
}

__global__ void maxpool2d_kernel(CudaMaxPool2DLayer layer,
                                 CudaTensor input,
                                 CudaTensor output)
{
    /* 
    
    GENERAL 
    FAILURE CHECKING
    
    */
   size_t thread_idx = (size_t) blockDim.x * blockIdx.x + threadIdx.x ; 
   if (thread_idx >= output.count){
    return;
   }
   size_t remaining = thread_idx;
   size_t channel = remaining % output.channels;
   remaining /= output.channels;

   size_t output_col = remaining % output.width;
   remaining /= output.width;

   size_t output_row = remaining % output.height;
   remaining /= output.height;

   size_t batch = remaining;

   

    

   

   
}
