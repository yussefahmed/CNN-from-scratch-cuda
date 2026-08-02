#ifndef CNN_CUDA_MAXPOOL2D_CUH
#define CNN_CUDA_MAXPOOL2D_CUH

#include "tensor.cuh"

#include <stddef.h>

/**
 * Configuration for a two-dimensional max-pooling layer.
 *
 * Pooling is applied independently to every channel and uses no padding.
 */
typedef struct {
    size_t kernel_height;
    size_t kernel_width;
    size_t stride;
} CudaMaxPool2DLayer;

/**
 * Initializes a CUDA MaxPool2D layer configuration.
 *
 * @param layer         Layer configuration to initialize.
 * @param kernel_height Number of rows in each pooling window.
 * @param kernel_width  Number of columns in each pooling window.
 * @param stride        Number of spatial positions between windows.
 *
 * @return true when all arguments are valid; false otherwise.
 */
__host__ bool cuda_maxpool2d_init(CudaMaxPool2DLayer *layer,
                                  size_t kernel_height,
                                  size_t kernel_width,
                                  size_t stride);

/**
 * Calculates the spatial output dimensions for an NHWC input tensor.
 *
 * Batch size and channel count remain unchanged. This function does not
 * allocate device memory.
 *
 * @param layer         Initialized MaxPool2D layer.
 * @param input         Input tensor metadata.
 * @param output_height Receives the calculated output height.
 * @param output_width  Receives the calculated output width.
 *
 * @return true when the configuration and input shape are valid; false
 *         otherwise.
 */
__host__ bool cuda_maxpool2d_output_shape(
    const CudaMaxPool2DLayer *layer,
    const CudaTensor *input,
    size_t *output_height,
    size_t *output_width);

/**
 * Allocates an NHWC output tensor and launches MaxPool2D on the GPU.
 *
 * @param layer  Initialized MaxPool2D layer.
 * @param input  Initialized input tensor backed by device memory.
 * @param output Fresh tensor that receives the allocated pooling result.
 *
 * @return true when the output is allocated and the kernel is launched;
 *         false otherwise.
 *
 * @post On success, the caller owns output and must release it with
 *       cuda_tensor_destroy().
 */
__host__ bool cuda_maxpool2d_forward(const CudaMaxPool2DLayer *layer,
                                     const CudaTensor *input,
                                     CudaTensor *output);

#endif
