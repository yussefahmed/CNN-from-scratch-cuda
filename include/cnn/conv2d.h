#ifndef CNN_CONV2D_H
#define CNN_CONV2D_H

#include "cnn/tensor.h"

#include <stdbool.h>
#include <stddef.h>

/**
 * A two-dimensional convolution layer with weights stored in OIHW order.
 *
 * The weight dimensions are:
 * [output_channel][input_channel][kernel_row][kernel_column].
 * Each output channel also owns one scalar bias.
 */
typedef struct {
    size_t input_channels;
    size_t output_channels;
    size_t kernel_height;
    size_t kernel_width;
    float *weights;
    size_t weight_count;
    float *biases;
    size_t stride;
    size_t padding;
} Conv2DLayer;

/**
 * Initializes a Conv2D layer and allocates its weights and biases.
 *
 * All weights and biases are initially set to zero.
 *
 * @param layer           Layer to initialize.
 * @param input_channels  Number of channels expected in an input tensor.
 * @param output_channels Number of convolution filters and output channels.
 * @param kernel_height   Number of rows in each 2D kernel.
 * @param kernel_width    Number of columns in each 2D kernel.
 * @param stride          Number of positions moved between kernel placements.
 * @param padding         Number of conceptual zero cells added on every side.
 *
 * @return true if the parameters are valid and both allocations succeed;
 *         false otherwise.
 *
 * @pre layer must be fresh or previously passed to conv2d_destroy().
 * @post On success, layer owns contiguous weight and bias buffers.
 * @post On failure, a non-NULL layer is left in an empty state.
 */
bool conv2d_init(Conv2DLayer *layer, size_t input_channels,
                 size_t output_channels, size_t kernel_height,
                 size_t kernel_width, size_t stride, size_t padding);

/**
 * Releases a Conv2D layer's weights and biases and resets its fields.
 *
 * @param layer Layer to destroy. May be NULL.
 *
 * @post If layer is non-NULL, its pointers are NULL and its dimensions,
 *       element count, stride, and padding are zero.
 *
 * @note This function releases only buffers owned by the layer, not the
 *       Conv2DLayer structure itself.
 * @note Calling this function repeatedly on the same valid layer is safe.
 */
void conv2d_destroy(Conv2DLayer *layer);

/**
 * Stores one convolution weight.
 *
 * @param layer          Layer whose weight will be modified.
 * @param output_channel Output filter coordinate.
 * @param input_channel  Input channel coordinate.
 * @param kernel_row     Row coordinate inside the 2D kernel.
 * @param kernel_col     Column coordinate inside the 2D kernel.
 * @param value          Weight value to store.
 *
 * @pre layer and layer->weights must not be NULL.
 * @pre Every coordinate must be within the layer's bounds.
 */
void conv2d_set_weight(Conv2DLayer *layer, size_t output_channel,
                       size_t input_channel, size_t kernel_row,
                       size_t kernel_col, float value);

/**
 * Retrieves one convolution weight.
 *
 * @param layer          Layer whose weight will be read.
 * @param output_channel Output filter coordinate.
 * @param input_channel  Input channel coordinate.
 * @param kernel_row     Row coordinate inside the 2D kernel.
 * @param kernel_col     Column coordinate inside the 2D kernel.
 *
 * @return The weight stored at the specified coordinate.
 *
 * @pre layer and layer->weights must not be NULL.
 * @pre Every coordinate must be within the layer's bounds.
 */
float conv2d_get_weight(const Conv2DLayer *layer, size_t output_channel,
                        size_t input_channel, size_t kernel_row,
                        size_t kernel_col);

/**
 * Stores the scalar bias for one output channel.
 *
 * @param layer          Layer whose bias will be modified.
 * @param output_channel Output channel coordinate.
 * @param value          Bias value to store.
 *
 * @pre layer and layer->biases must not be NULL.
 * @pre output_channel must be less than layer->output_channels.
 */
void conv2d_set_bias(Conv2DLayer *layer, size_t output_channel, float value);

/**
 * Retrieves the scalar bias for one output channel.
 *
 * @param layer          Layer whose bias will be read.
 * @param output_channel Output channel coordinate.
 *
 * @return The bias stored for the specified output channel.
 *
 * @pre layer and layer->biases must not be NULL.
 * @pre output_channel must be less than layer->output_channels.
 */
float conv2d_get_bias(const Conv2DLayer *layer, size_t output_channel);

/**
 * Calculates the CHW shape produced by a Conv2D layer.
 *
 * This function uses only the input tensor's dimensions. It does not read
 * input data and does not allocate an output tensor.
 *
 * @param layer           Initialized convolution layer.
 * @param input           Input tensor whose dimensions define the operation.
 * @param output_channels Receives the number of output channels.
 * @param output_height   Receives the calculated output height.
 * @param output_width    Receives the calculated output width.
 *
 * @return true if all pointers and dimensions are valid and the kernel fits
 *         the padded input; false otherwise.
 *
 * @post On failure, the three output variables are not modified.
 */
bool conv2d_output_shape(const Conv2DLayer *layer, const Tensor3D *input,
                         size_t *output_channels, size_t *output_height,
                         size_t *output_width);

/**
 * Performs a naive, single-threaded forward convolution.
 *
 * For each output position, the function multiplies the input region by one
 * complete filter, sums across all input channels, and adds that filter's
 * bias. Positions inside the conceptual padding border contribute zero.
 *
 * @param layer  Initialized convolution layer containing weights and biases.
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
bool conv2d_forward(const Conv2DLayer *layer, const Tensor3D *input,
                    Tensor3D *output);

#endif
