#ifndef CNN_DENSE_H
#define CNN_DENSE_H

#include "cnn/tensor.h"

#include <stdbool.h>
#include <stddef.h>

/**
 * A fully connected layer with weights stored in output-input order.
 *
 * The weight dimensions are [output_size][input_size]. Each output neuron
 * also owns one scalar bias.
 */
typedef struct {
    size_t input_size;
    size_t output_size;
    float *weights;
    size_t weight_count;
    float *biases;
} DenseLayer;

/**
 * Initializes a Dense layer and allocates its weights and biases.
 *
 * All weights and biases are initially set to zero.
 *
 * @param layer       Layer to initialize.
 * @param input_size  Number of flattened input values expected.
 * @param output_size Number of output neurons.
 *
 * @return true if the sizes are valid and both allocations succeed;
 *         false otherwise.
 *
 * @pre layer must be fresh or previously passed to dense_destroy().
 * @post On success, layer owns contiguous weight and bias buffers.
 * @post On failure, a non-NULL layer is left in an empty state.
 */
bool dense_init(DenseLayer *layer, size_t input_size, size_t output_size);

/**
 * Releases a Dense layer's weights and biases and resets its fields.
 *
 * @param layer Layer to destroy. May be NULL.
 *
 * @note This function releases only buffers owned by the layer, not the
 *       DenseLayer structure itself.
 * @note Calling this function repeatedly on the same valid layer is safe.
 */
void dense_destroy(DenseLayer *layer);

/**
 * Stores one Dense weight.
 *
 * @param layer        Layer whose weight will be modified.
 * @param output_index Output neuron coordinate.
 * @param input_index  Flattened input coordinate.
 * @param value        Weight value to store.
 *
 * @pre layer and layer->weights must not be NULL.
 * @pre Both indices must be within the layer's bounds.
 */
void dense_set_weight(DenseLayer *layer, size_t output_index,
                      size_t input_index, float value);

/**
 * Retrieves one Dense weight.
 *
 * @param layer        Layer whose weight will be read.
 * @param output_index Output neuron coordinate.
 * @param input_index  Flattened input coordinate.
 *
 * @return The weight stored at the specified coordinate.
 *
 * @pre layer and layer->weights must not be NULL.
 * @pre Both indices must be within the layer's bounds.
 */
float dense_get_weight(const DenseLayer *layer, size_t output_index,
                       size_t input_index);

/**
 * Stores the scalar bias for one output neuron.
 *
 * @param layer        Layer whose bias will be modified.
 * @param output_index Output neuron coordinate.
 * @param value        Bias value to store.
 *
 * @pre layer and layer->biases must not be NULL.
 * @pre output_index must be less than layer->output_size.
 */
void dense_set_bias(DenseLayer *layer, size_t output_index, float value);

/**
 * Retrieves the scalar bias for one output neuron.
 *
 * @param layer        Layer whose bias will be read.
 * @param output_index Output neuron coordinate.
 *
 * @return The bias stored for the specified output neuron.
 *
 * @pre layer and layer->biases must not be NULL.
 * @pre output_index must be less than layer->output_size.
 */
float dense_get_bias(const DenseLayer *layer, size_t output_index);

/**
 * Performs a naive, single-threaded Dense forward pass.
 *
 * The input tensor's contiguous CHW data is treated as a flat vector. The
 * output tensor is allocated with shape [output_size][1][1].
 *
 * @param layer  Initialized Dense layer containing weights and biases.
 * @param input  Initialized Tensor3D whose data is treated as a flat vector.
 * @param output Fresh or previously destroyed tensor that receives the result.
 *
 * @return true if the inputs are valid and the output allocation succeeds;
 *         false otherwise.
 *
 * @pre input->count must equal layer->input_size.
 * @pre output must not alias input.
 * @pre output must not already own an allocated data buffer.
 * @post On success, output owns a buffer that the caller must release with
 *       tensor3d_destroy().
 */
bool dense_forward(const DenseLayer *layer, const Tensor3D *input,
                   Tensor3D *output);

#endif
