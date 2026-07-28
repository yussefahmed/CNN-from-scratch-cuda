#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


typedef struct {
    size_t channels;
    size_t height;
    size_t width;
    float *data;
    size_t count;
} Tensor3D;



/**
 * Converts a Tensor3D coordinate into a flat array index using CHW layout.
 *
 * @param channel Channel coordinate.
 * @param row     Row coordinate.
 * @param col     Column coordinate.
 * @param tensor  Tensor whose dimensions define the memory layout.
 *
 * @return The corresponding index in the tensor's contiguous data array.
 *
 * @pre tensor must not be NULL.
 * @pre channel must be less than tensor->channels.
 * @pre row must be less than tensor->height.
 * @pre col must be less than tensor->width.
 */
size_t tensor3d_index(size_t channel, size_t row , size_t col , const Tensor3D* tensor){
    assert(tensor != NULL);
    assert(channel < tensor->channels);
    assert(row < tensor->height);
    assert(col < tensor->width);

    size_t index =  (channel * tensor->width * tensor->height ) + (row * tensor->width + col);
    return index;

}

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
bool tensor3d_init(Tensor3D *tensor, size_t channels, size_t height, size_t width){
    if (tensor == NULL) {
        return false;
    }

    tensor->channels = 0;
    tensor->height = 0;
    tensor->width = 0;
    tensor->count = 0;
    tensor->data = NULL;

    if (channels == 0 || height == 0 || width == 0) {
        return false;
    }

    if (height > SIZE_MAX / width) {
        return false;
    }
    size_t channel_size = height * width;

    if (channels > SIZE_MAX / channel_size) {
        return false;
    }
    size_t count = channels * channel_size;

    if (count > SIZE_MAX / sizeof(float)) {
        return false;
    }

    float *data = calloc(count, sizeof(float));
    if (data == NULL) {
        return false;
    }

    tensor->channels = channels;
    tensor->height = height;
    tensor->width = width;
    tensor->count = count;
    tensor->data = data;

    return true;
}

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
void tensor3d_destroy(Tensor3D *tensor){
    if (tensor == NULL){
        return;
    }

    free(tensor->data);
    tensor->data = NULL;
    tensor->channels = 0;
    tensor->height = 0;
    tensor->width = 0;
    tensor->count = 0;

}


int 
main(){





    return 0; 
}
