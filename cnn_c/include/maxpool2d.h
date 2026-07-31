#ifndef CNN_MAXPOOL2D_H
#define CNN_MAXPOOL2D_H

#include "tensor.h"

#include <stdbool.h>
#include <stddef.h>

/**
 * A two-dimensional max-pooling layer with no padding.
 *
 * MaxPool2D processes each input channel independently and does not contain
 * trainable weights or biases.
 */
typedef struct {
    size_t kernel_height;
    size_t kernel_width;
    size_t stride;
} MaxPool2DLayer;

/**
 * Initializes a MaxPool2D layer.
 *
 * @param layer         Layer to initialize.
 * @param kernel_height Number of rows in the pooling window.
 * @param kernel_width  Number of columns in the pooling window.
 * @param stride        Number of positions moved between pooling windows.
 *
 * @return true if the layer and parameters are valid; false otherwise.
 *
 * @post On failure, a non-NULL layer is left in an empty state.
 */
bool maxpool2d_init(MaxPool2DLayer *layer, size_t kernel_height,
                    size_t kernel_width, size_t stride);

/**
 * Calculates the CHW output shape produced by a MaxPool2D layer.
 *
 * The output channel count is equal to the input channel count. This function
 * reads only tensor dimensions and does not allocate an output tensor.
 *
 * @param layer           Initialized pooling layer.
 * @param input           Input tensor whose dimensions define the operation.
 * @param output_channels Receives the number of output channels.
 * @param output_height   Receives the calculated output height.
 * @param output_width    Receives the calculated output width.
 *
 * @return true if all pointers and dimensions are valid and the pooling
 *         window fits the input; false otherwise.
 *
 * @post On failure, the three output variables are not modified.
 */
bool maxpool2d_output_shape(const MaxPool2DLayer *layer,
                            const Tensor3D *input,
                            size_t *output_channels,
                            size_t *output_height,
                            size_t *output_width);

/**
 * Performs naive, single-threaded max pooling with no padding.
 *
 * @param layer  Initialized pooling layer.
 * @param input  Initialized CHW input tensor.
 * @param output Fresh or previously destroyed tensor that receives the result.
 *
 * @return true if the inputs are valid and the output allocation succeeds;
 *         false otherwise.
 *
 * @pre output must not alias input.
 * @pre output must not already own an allocated data buffer.
 * @post On success, output owns a CHW buffer that the caller must release
 *       with tensor3d_destroy().
 */
bool maxpool2d_forward(const MaxPool2DLayer *layer, const Tensor3D *input,
                       Tensor3D *output);

#endif
