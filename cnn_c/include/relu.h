#ifndef CNN_RELU_H
#define CNN_RELU_H

#include "tensor.h"

#include <stdbool.h>

/**
 * Applies the ReLU activation to every element of a Tensor3D in place.
 *
 * Negative values are replaced with zero. Zero and positive values are left
 * unchanged. The tensor's dimensions and element count do not change.
 *
 * @param tensor Initialized tensor whose data will be modified.
 *
 * @return true if the tensor and its data buffer are valid; false otherwise.
 */
bool relu_forward(Tensor3D *tensor);

#endif
