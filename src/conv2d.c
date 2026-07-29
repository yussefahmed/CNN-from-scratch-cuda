#include "cnn/conv2d.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

static size_t conv2d_weight_index(const Conv2DLayer *layer,
                                  size_t output_channel,
                                  size_t input_channel,
                                  size_t kernel_row,
                                  size_t kernel_col){
    assert(layer != NULL);
    assert(output_channel < layer->output_channels);
    assert(input_channel < layer->input_channels);
    assert(kernel_row < layer->kernel_height);
    assert(kernel_col < layer->kernel_width);

    size_t kernel_area = layer->kernel_height * layer->kernel_width;
    size_t filter_size = layer->input_channels * kernel_area;
    size_t index = output_channel * filter_size
                 + input_channel * kernel_area
                 + kernel_row * layer->kernel_width
                 + kernel_col;
    return index;
}


bool conv2d_init(Conv2DLayer *layer, size_t input_channels,
                 size_t output_channels, size_t kernel_height,
                 size_t kernel_width, size_t stride, size_t padding){
    if (layer == NULL) {
        return false;
    }

    layer->input_channels = 0;
    layer->output_channels = 0;
    layer->kernel_height = 0;
    layer->kernel_width = 0;
    layer->weights = NULL;
    layer->weight_count = 0;
    layer->biases = NULL;
    layer->stride = 0;
    layer->padding = 0;

    if (output_channels == 0 || input_channels == 0 ||
        kernel_width == 0 || kernel_height == 0 || stride == 0) {
        return false;
    }

    if (kernel_height > SIZE_MAX / kernel_width) {
        return false;
    }
    size_t kernel_area = kernel_height * kernel_width;

    if (input_channels > SIZE_MAX / kernel_area) {
        return false;
    }
    size_t filter_size = input_channels * kernel_area;

    if (output_channels > SIZE_MAX / filter_size) {
        return false;
    }
    size_t weight_count = output_channels * filter_size;

    if (weight_count > SIZE_MAX / sizeof(float) ||
        output_channels > SIZE_MAX / sizeof(float)) {
        return false;
    }

    float *weights = calloc(weight_count, sizeof(float));
    if (weights == NULL) {
        return false;
    }

    float *biases = calloc(output_channels, sizeof(float));
    if (biases == NULL) {
        free(weights);
        return false;
    }

    layer->input_channels = input_channels;
    layer->output_channels = output_channels;
    layer->kernel_height = kernel_height;
    layer->kernel_width = kernel_width;
    layer->padding = padding;
    layer->stride = stride;
    layer->weight_count = weight_count;
    layer->weights = weights;
    layer->biases = biases;

    return true;
}

void conv2d_destroy(Conv2DLayer *layer){
    if (layer == NULL){
        return;
    }
    layer->input_channels = 0;
    layer->output_channels = 0;
    layer->kernel_height = 0;
    layer->kernel_width = 0;
    free(layer->weights);
    layer->weights = NULL;
    layer->weight_count = 0;
    free(layer->biases);
    layer->biases = NULL;
    layer->stride = 0;
    layer->padding = 0;

}

void conv2d_set_weight(Conv2DLayer *layer, size_t output_channel,
                       size_t input_channel, size_t kernel_row,
                       size_t kernel_col, float value){
    assert(layer != NULL);
    assert(layer->weights != NULL);

    size_t index = conv2d_weight_index(layer, output_channel, input_channel,
                                       kernel_row, kernel_col);
    layer->weights[index] = value;
}

float conv2d_get_weight(const Conv2DLayer *layer, size_t output_channel,
                        size_t input_channel, size_t kernel_row,
                        size_t kernel_col){
    assert(layer != NULL);
    assert(layer->weights != NULL);

    size_t index = conv2d_weight_index(layer, output_channel, input_channel,
                                       kernel_row, kernel_col);
    return layer->weights[index];
}

void conv2d_set_bias(Conv2DLayer *layer, size_t output_channel, float value){
    assert(layer != NULL);
    assert(layer->biases != NULL);
    assert(output_channel < layer->output_channels);

    layer->biases[output_channel] = value;
}

float conv2d_get_bias(const Conv2DLayer *layer, size_t output_channel){
    assert(layer != NULL);
    assert(layer->biases != NULL);
    assert(output_channel < layer->output_channels);

    return layer->biases[output_channel];
}

bool conv2d_output_shape(const Conv2DLayer *layer, const Tensor3D *input,
                         size_t *output_channels, size_t *output_height,
                         size_t *output_width){
    if (layer == NULL || input == NULL ||
        output_channels == NULL || output_height == NULL ||
        output_width == NULL) {
        return false;
    }

    if (layer->input_channels == 0 || layer->output_channels == 0 ||
        layer->kernel_height == 0 || layer->kernel_width == 0 ||
        layer->stride == 0 || input->height == 0 || input->width == 0 ||
        input->channels != layer->input_channels) {
        return false;
    }

    if (layer->padding > (SIZE_MAX - input->height) / 2 ||
        layer->padding > (SIZE_MAX - input->width) / 2) {
        return false;
    }

    size_t padded_height = input->height + 2 * layer->padding;
    size_t padded_width = input->width + 2 * layer->padding;

    if (layer->kernel_height > padded_height ||
        layer->kernel_width > padded_width) {
        return false;
    }

    size_t calculated_height =
        ((padded_height - layer->kernel_height) / layer->stride) + 1;
    size_t calculated_width =
        ((padded_width - layer->kernel_width) / layer->stride) + 1;

    *output_channels = layer->output_channels;
    *output_height = calculated_height;
    *output_width = calculated_width;
    return true;
}

bool conv2d_forward(const Conv2DLayer *layer, const Tensor3D *input,
                    Tensor3D *output){
    if (layer == NULL || input == NULL || output == NULL ||
        output == input || layer->weights == NULL ||
        layer->biases == NULL || input->data == NULL) {
        return false;
    }

    size_t output_channels;
    size_t output_height;
    size_t output_width;

    if (!conv2d_output_shape(layer, input, &output_channels, &output_height,
                             &output_width)) {
        return false;
    }

    if (!tensor3d_init(output, output_channels, output_height, output_width)) {
        return false;
    }

    for (size_t output_channel = 0;
         output_channel < output->channels;
         ++output_channel) {
        for (size_t output_row = 0;
             output_row < output->height;
             ++output_row) {
            for (size_t output_col = 0;
                 output_col < output->width;
                 ++output_col) {
                float sum = conv2d_get_bias(layer, output_channel);

                for (size_t input_channel = 0;
                     input_channel < input->channels;
                     ++input_channel) {
                    for (size_t kernel_row = 0;
                         kernel_row < layer->kernel_height;
                         ++kernel_row) {
                        size_t padded_row =
                            output_row * layer->stride + kernel_row;

                        if (padded_row < layer->padding) {
                            continue;
                        }

                        size_t input_row = padded_row - layer->padding;

                        if (input_row >= input->height) {
                            continue;
                        }

                        for (size_t kernel_col = 0;
                             kernel_col < layer->kernel_width;
                             ++kernel_col) {
                            size_t padded_col =
                                output_col * layer->stride + kernel_col;

                            if (padded_col < layer->padding) {
                                continue;
                            }

                            size_t input_col = padded_col - layer->padding;

                            if (input_col >= input->width) {
                                continue;
                            }

                            float input_value = tensor3d_get(
                                input, input_channel, input_row, input_col);
                            float weight_value = conv2d_get_weight(
                                layer, output_channel, input_channel,
                                kernel_row, kernel_col);

                            sum += input_value * weight_value;
                        }
                    }
                }

                tensor3d_set(output, output_channel, output_row, output_col,
                             sum);
            }
        }
    }

    return true;
}
