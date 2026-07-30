#ifndef CNN_SOFTMAX_H
#define CNN_SOFTMAX_H

#include "tensor.h"

#include <stdbool.h>

/**
 * Applies Softmax to every element of a Tensor3D in place.
 *
 * The tensor is treated as one flat vector of logits. Each logit is converted
 * into a probability, and the resulting probabilities sum to approximately
 * one. The tensor's dimensions and element count do not change.
 *
 * @param tensor Initialized tensor whose data will be modified.
 *
 * @return true if Softmax is applied successfully; false if the tensor or its
 *         data is invalid.
 */
bool softmax_forward(Tensor3D *tensor);

#endif
