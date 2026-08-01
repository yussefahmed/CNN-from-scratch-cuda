#ifndef CNN_CUDA_TENSOR_CUH
#define CNN_CUDA_TENSOR_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>

/* Contiguous NHWC FP16 activation tensor backed by GPU memory. */
typedef struct {
    size_t batch_size;
    size_t height;
    size_t width;
    size_t channels;
    size_t count;
    __half *device_pointer;
} CudaTensor;

__host__ bool cuda_tensor_init(CudaTensor *tensor, size_t batch_size,
                               size_t height, size_t width,
                               size_t channels);

__host__ bool cuda_tensor_destroy(CudaTensor *tensor);

__host__ bool cuda_tensor_copy_from_host(const CudaTensor *tensor,
                                         const __half *host_data);

__host__ bool cuda_tensor_copy_to_host(const CudaTensor *tensor,
                                       __half *host_data);

#endif
