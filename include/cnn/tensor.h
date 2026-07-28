#ifndef CNN_TENSOR_H
#define CNN_TENSOR_H

#include <stdbool.h>
#include <stddef.h>

/**
 * A three-dimensional tensor stored contiguously in CHW order.
 */
typedef struct {
    size_t channels;
    size_t height;
    size_t width;
    float *data;
    size_t count;
} Tensor3D;

/**
 * Initializes an empty Tensor3D and allocates its contiguous data buffer.
 *
 * @param tensor   Tensor to initialize.
 * @param channels Number of channels in the tensor.
 * @param height   Number of rows in each channel.
 * @param width    Number of columns in each row.
 *
 * @return true if the dimensions are valid and allocation succeeds;
 *         false otherwise.
 *
 * @pre tensor must not already own an allocated data buffer.
 * @post On success, tensor owns a buffer containing
 *       channels * height * width floats.
 * @post On failure, a non-NULL tensor is left in an empty state.
 */
bool tensor3d_init(Tensor3D *tensor, size_t channels, size_t height, size_t width);

/**
 * Releases a Tensor3D's data buffer and resets it to an empty state.
 *
 * @param tensor Tensor to destroy. May be NULL.
 *
 * @post If tensor is non-NULL, its data pointer is NULL and all dimensions
 *       and its element count are zero.
 *
 * @note This function releases only the tensor's data buffer, not the
 *       Tensor3D structure itself.
 * @note Calling this function repeatedly on the same valid Tensor3D is safe.
 */
void tensor3d_destroy(Tensor3D *tensor);

/**
 * Stores a value at the specified coordinate in a Tensor3D.
 *
 * @param tensor  Tensor whose data will be modified.
 * @param channel Channel coordinate.
 * @param row     Row coordinate.
 * @param col     Column coordinate.
 * @param value   Value to store.
 *
 * @pre tensor must not be NULL.
 * @pre tensor->data must not be NULL.
 * @pre channel, row, and col must be within the tensor's bounds.
 */
void tensor3d_set(Tensor3D *tensor, size_t channel, size_t row, size_t col, float value);

/**
 * Retrieves the value at the specified coordinate in a Tensor3D.
 *
 * @param tensor  Tensor whose data will be read.
 * @param channel Channel coordinate.
 * @param row     Row coordinate.
 * @param col     Column coordinate.
 *
 * @return The value stored at the specified coordinate.
 *
 * @pre tensor must not be NULL.
 * @pre tensor->data must not be NULL.
 * @pre channel, row, and col must be within the tensor's bounds.
 */
float tensor3d_get(const Tensor3D *tensor, size_t channel, size_t row, size_t col);

/**
 * Converts a Tensor3D coordinate into a flat array index using CHW layout.
 *
 * @param tensor  Tensor whose dimensions define the memory layout.
 * @param channel Channel coordinate.
 * @param row     Row coordinate.
 * @param col     Column coordinate.
 *
 * @return The corresponding index in the tensor's contiguous data array.
 *
 * @pre tensor must not be NULL.
 * @pre channel must be less than tensor->channels.
 * @pre row must be less than tensor->height.
 * @pre col must be less than tensor->width.
 */
size_t tensor3d_index(const Tensor3D *tensor, size_t channel, size_t row, size_t col);

#endif
