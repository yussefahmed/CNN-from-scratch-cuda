#ifndef CNN_CUDA_MAXPOOL2D_CUH
#define CNN_CUDA_MAXPOOL2D_CUH

#include "tensor.cuh"

#include <stddef.h>

typedef struct {
    size_t kernel_height;
    size_t kernel_width;
    size_t stride;
} CudaMaxPool2DLayer;

__host__ bool cuda_maxpool2d_init(CudaMaxPool2DLayer *layer,
                                  size_t kernel_height,
                                  size_t kernel_width,
                                  size_t stride);

__host__ bool cuda_maxpool2d_output_shape(
    const CudaMaxPool2DLayer *layer,
    const CudaTensor *input,
    size_t *output_height,
    size_t *output_width);

#endif
