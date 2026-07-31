#include "softmax.h"

#include <math.h>
#include <stddef.h>

bool softmax_forward(Tensor3D *tensor){
    if (tensor == NULL || tensor->data == NULL || tensor->count == 0) {
        return false;
    }

    float largest_value = tensor->data[0];
    for (size_t tensor_idx = 1 ; tensor_idx < tensor->count ; ++tensor_idx){

        if (tensor->data[tensor_idx] > largest_value){
            largest_value = tensor->data[tensor_idx];
        }


    }
    float sum = 0.0f;
    for (size_t tensor_idx = 0; tensor_idx < tensor->count ; ++tensor_idx){

        float exponential = expf(tensor->data[tensor_idx] - largest_value);
        tensor->data[tensor_idx] = exponential;
        sum+=exponential;
    }

    for (size_t tensor_idx = 0; tensor_idx < tensor->count ; ++tensor_idx){
        tensor->data[tensor_idx] = tensor->data[tensor_idx] / sum;
    }

    return true;
}
