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

__global__ void cuda_maxpool2d_kernel(CudaMaxPool2DLayer layer,
                                 CudaTensor input,
                                 CudaTensor output)
{
    size_t output_index =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (output_index >= output.count) {
        return;
    }

    /* Decode the flat output index as [batch][row][col][channel]. */
    /* i don't understand the indexing yet but it works */
    size_t remaining = output_index;

    size_t channel = remaining % output.channels;
    remaining /= output.channels;

    size_t output_col = remaining % output.width;
    remaining /= output.width;

    size_t output_row = remaining % output.height;
    remaining /= output.height;

    size_t batch = remaining;

    size_t input_row = output_row * layer.stride;
    size_t input_col = output_col * layer.stride;

    size_t first_input_index =
        (((batch * input.height + input_row) * input.width + input_col)
         * input.channels) + channel;

    float max_value = input.device_pointer[first_input_index];

    for (size_t kernel_row = 0;
         kernel_row < layer.kernel_height;
         ++kernel_row) {
        for (size_t kernel_col = 0;
             kernel_col < layer.kernel_width;
             ++kernel_col) {
            size_t row = input_row + kernel_row;
            size_t col = input_col + kernel_col;

            size_t input_index =
                (((batch * input.height + row) * input.width + col)
                 * input.channels) + channel;

            float value = input.device_pointer[input_index];
            if (value > max_value) {
                max_value = value;
            }
        }
    }

    output.device_pointer[output_index] = max_value;
}


__host__ bool cuda_maxpool2d_forward(const CudaMaxPool2DLayer *layer , const CudaTensor *input , CudaTensor *output){

    size_t output_width;
    size_t output_height;
    if (!cuda_maxpool2d_output_shape(layer,input,&output_height,&output_width)){
        return false;
    }
    
    if (!cuda_tensor_init(output , input->batch_size , output_height ,output_width ,input->channels)){
        return false;
    }
    // total number of threads is the number of outputs  
    size_t threads = output->count;
    size_t blocks =(threads + 256 - 1) / 256; 

    cuda_maxpool2d_kernel<<<blocks, 256>>>(*layer,*input ,*output);
    return true; 
}
