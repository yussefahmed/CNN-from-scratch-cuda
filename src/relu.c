#include "relu.h"
#include "stdlib.h"
#include "stdio.h"
#include "stdbool.h"

bool relu_forward(Tensor3D *tensor){
    if (tensor->data == NULL || tensor->count == 0){
        return false;
    }

    for (size_t count = 0; count < tensor->count ; ++count){
        if (tensor->data[count] < 0){
            tensor->data[count] = 0.0f;
        }
    }
    return true;
}
