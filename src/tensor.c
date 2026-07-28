#include "cnn/tensor.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

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


void tensor3d_set(Tensor3D *tensor, size_t channel, size_t row, size_t col, float value){
    assert(tensor != NULL);
    assert(tensor->data != NULL);
    size_t index = tensor3d_index(tensor, channel, row, col);
    tensor->data[index] = value;
}

float tensor3d_get(const Tensor3D *tensor, size_t channel, size_t row, size_t col){
    assert(tensor != NULL);
    assert(tensor->data != NULL);
    size_t index = tensor3d_index(tensor, channel, row, col);
    return tensor->data[index];

}

size_t tensor3d_index(const Tensor3D *tensor, size_t channel, size_t row, size_t col){
    assert(tensor != NULL);
    assert(channel < tensor->channels);
    assert(row < tensor->height);
    assert(col < tensor->width);

    size_t index = (channel * tensor->width * tensor->height) + (row * tensor->width + col);
    return index;
}
