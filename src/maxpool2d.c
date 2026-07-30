#include "cnn/maxpool2d.h"

bool maxpool2d_init(MaxPool2DLayer *layer, size_t kernel_height,
                    size_t kernel_width, size_t stride){
    if (layer == NULL) {
        return false;
    }

    layer->kernel_height = 0;
    layer->kernel_width = 0;
    layer->stride = 0;

    if (kernel_height == 0 || kernel_width == 0 || stride == 0) {
        return false;
    }

    layer->kernel_height = kernel_height;
    layer->kernel_width = kernel_width;
    layer->stride = stride;
    return true;
}

bool maxpool2d_output_shape(const MaxPool2DLayer *layer,
                            const Tensor3D *input,
                            size_t *output_channels,
                            size_t *output_height,
                            size_t *output_width){
    if (layer == NULL || input == NULL || output_channels == NULL ||
        output_height == NULL || output_width == NULL) {
        return false;
    }

    if (layer->kernel_height == 0 || layer->kernel_width == 0 ||
        layer->stride == 0 || input->channels == 0 ||
        input->height == 0 || input->width == 0) {
        return false;
    }

    if (layer->kernel_height > input->height ||
        layer->kernel_width > input->width) {
        return false;
    }

    size_t calculated_height =
        ((input->height - layer->kernel_height) / layer->stride) + 1;
    size_t calculated_width =
        ((input->width - layer->kernel_width) / layer->stride) + 1;

    *output_channels = input->channels;
    *output_height = calculated_height;
    *output_width = calculated_width;
    return true;
}

bool maxpool2d_forward(const MaxPool2DLayer *layer, const Tensor3D *input,
                       Tensor3D *output){
    if (layer == NULL || input == NULL || output == NULL ||
        output == input || input->data == NULL) {
        return false;
    }

    size_t output_channels;
    size_t output_height;
    size_t output_width;

    if (!maxpool2d_output_shape(layer, input, &output_channels,
                                &output_height, &output_width)) {
        return false;
    }

    if (!tensor3d_init(output, output_channels, output_height, output_width)) {
        return false;
    }

    size_t output_channel_size = output->height * output->width;

    for (size_t output_channel = 0;
         output_channel < output->channels;
         ++output_channel) {
        for (size_t output_index = 0;
             output_index < output_channel_size;
             ++output_index) {
            size_t row = output_index / output->width;
            size_t col = output_index % output->width;

            size_t input_row = row * layer->stride;
            size_t input_col = col * layer->stride;

            float max_value = tensor3d_get(
                input, output_channel, input_row, input_col);

            for (size_t kernel_relative_row_idx = 0;
                 kernel_relative_row_idx < layer->kernel_height;
                 ++kernel_relative_row_idx) {
                for (size_t kernel_relative_col_idx = 0;
                     kernel_relative_col_idx < layer->kernel_width;
                     ++kernel_relative_col_idx) {
                    size_t real_row_idx =
                        input_row + kernel_relative_row_idx;
                    size_t real_col_idx =
                        input_col + kernel_relative_col_idx;

                    float value = tensor3d_get(
                        input, output_channel, real_row_idx, real_col_idx);

                    if (value > max_value) {
                        max_value = value;
                    }
                }
            }

            tensor3d_set(output, output_channel, row, col, max_value);
        }
    }

    return true;
}
