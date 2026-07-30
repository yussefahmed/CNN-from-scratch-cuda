#include "cnn/dense.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

static size_t dense_weight_index(const DenseLayer *layer,
                                 size_t output_index,
                                 size_t input_index){
    assert(layer != NULL);
    assert(output_index < layer->output_size);
    assert(input_index < layer->input_size);

    return output_index * layer->input_size + input_index;
}

bool dense_init(DenseLayer *layer, size_t input_size, size_t output_size){
    if (layer == NULL) {
        return false;
    }

    layer->input_size = 0;
    layer->output_size = 0;
    layer->weights = NULL;
    layer->weight_count = 0;
    layer->biases = NULL;

    if (input_size == 0 || output_size == 0) {
        return false;
    }

    if (output_size > SIZE_MAX / input_size) {
        return false;
    }
    size_t weight_count = output_size * input_size;

    if (weight_count > SIZE_MAX / sizeof(float) ||
        output_size > SIZE_MAX / sizeof(float)) {
        return false;
    }

    float *weights = calloc(weight_count, sizeof(float));
    if (weights == NULL) {
        return false;
    }

    float *biases = calloc(output_size, sizeof(float));
    if (biases == NULL) {
        free(weights);
        return false;
    }

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = weights;
    layer->weight_count = weight_count;
    layer->biases = biases;
    return true;
}

void dense_destroy(DenseLayer *layer){
    if (layer == NULL) {
        return;
    }

    free(layer->weights);
    layer->weights = NULL;
    free(layer->biases);
    layer->biases = NULL;
    layer->input_size = 0;
    layer->output_size = 0;
    layer->weight_count = 0;
}

void dense_set_weight(DenseLayer *layer, size_t output_index,
                      size_t input_index, float value){
    assert(layer != NULL);
    assert(layer->weights != NULL);

    size_t index = dense_weight_index(layer, output_index, input_index);
    layer->weights[index] = value;
}

float dense_get_weight(const DenseLayer *layer, size_t output_index,
                       size_t input_index){
    assert(layer != NULL);
    assert(layer->weights != NULL);

    size_t index = dense_weight_index(layer, output_index, input_index);
    return layer->weights[index];
}

void dense_set_bias(DenseLayer *layer, size_t output_index, float value){
    assert(layer != NULL);
    assert(layer->biases != NULL);
    assert(output_index < layer->output_size);

    layer->biases[output_index] = value;
}

float dense_get_bias(const DenseLayer *layer, size_t output_index){
    assert(layer != NULL);
    assert(layer->biases != NULL);
    assert(output_index < layer->output_size);

    return layer->biases[output_index];
}

bool dense_forward(const DenseLayer *layer, const Tensor3D *input,
                   Tensor3D *output){
    if (layer == NULL || input == NULL || output == NULL ||
        output == input || layer->weights == NULL ||
        layer->biases == NULL || input->data == NULL ||
        input->count != layer->input_size) {
        return false;
    }

    if (!tensor3d_init(output, layer->output_size, 1, 1)) {
        return false;
    }

    for (size_t output_idx = 0; output_idx < layer->output_size ; ++output_idx){

        float sum = layer->biases[output_idx];

        for (size_t input_idx = 0 ; input_idx < input->count ; ++input_idx){


            sum+= input->data[input_idx] * dense_get_weight(layer, output_idx, input_idx);


        }
        output->data[output_idx] = sum;


    }
    return true;
}
