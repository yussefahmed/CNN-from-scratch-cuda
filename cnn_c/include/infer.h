#ifndef CNN_INFER_H
#define CNN_INFER_H

#include "conv2d.h"
#include "dense.h"
#include "maxpool2d.h"
#include "tensor.h"
#include "relu.h"
#include "softmax.h"
#include <stdbool.h>

/**
 * Runs a complete forward inference pass through the fixed CNN pipeline.
 *
 * The pipeline is:
 * Conv2D -> ReLU -> MaxPool2D -> Conv2D -> ReLU -> Dense -> Softmax.
 *
 * The inference function does not own or modify the layer structures. It
 * allocates temporary activation tensors internally and releases them before
 * returning.
 *
 * @param conv1  Initialized first convolution layer.
 * @param pool1  Initialized max-pooling layer.
 * @param conv2  Initialized second convolution layer.
 * @param dense  Initialized Dense output layer.
 * @param input  Initialized input tensor.
 * @param output Fresh or previously destroyed tensor that receives the class
 *               probabilities with shape [dense->output_size][1][1].
 *
 * @return true if every stage succeeds; false otherwise.
 *
 * @pre The output shape of each layer must match the input expected by the
 *      following layer.
 * @pre output must not alias input.
 * @pre output must not already own an allocated data buffer.
 * @post On success, output must be released with tensor3d_destroy().
 */
bool cnn_infer(const Conv2DLayer *conv1, const MaxPool2DLayer *pool1,
               const Conv2DLayer *conv2, const DenseLayer *dense,
               const Tensor3D *input, Tensor3D *output);

    
               
  

#endif
