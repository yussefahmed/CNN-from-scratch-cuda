#include "infer.h"

bool cnn_infer(const Conv2DLayer *conv1, const MaxPool2DLayer *pool1,
               const Conv2DLayer *conv2, const DenseLayer *dense,
               const Tensor3D *input, Tensor3D *output){
    if (conv1 == NULL || pool1 == NULL || conv2 == NULL || dense == NULL ||
        input == NULL || output == NULL) {
        return false;
    }

    if (input->data == NULL || input->count == 0) {
        return false;
    }

    if (output == input || output->data != NULL) {
        return false;
    }

    /* TODO(human): Implement the fixed CNN inference pipeline. */

    
    Tensor3D conv1_output = {0};
    Tensor3D conv2_output = {0};
    Tensor3D maxpool2d_output = {0};

    
    

    // it calculates the shape itself 
    bool flag = conv2d_forward(conv1 , input , &conv1_output);
    if (flag == false){
        return false;
    }

    flag = relu_forward(&conv1_output);
    if(flag == false){
        return false;
    }


    
    flag = maxpool2d_forward(pool1,&conv1_output, &maxpool2d_output);
    if (flag == false){
        return false;
    }
    

    flag = conv2d_forward(conv2,&maxpool2d_output,&conv2_output);
    if (flag == false){
        return false;
    }

    flag = relu_forward(&conv2_output);
    if (flag == false){
        return false;
    }


    

    flag = dense_forward(dense , &conv2_output , output );
    if (flag == false){
        return false;
    }
    flag = softmax_forward(output);
    if (flag == false){
        return false; 
    }

    tensor3d_destroy(&conv1_output);
    tensor3d_destroy(&maxpool2d_output);
    tensor3d_destroy(&conv2_output);

    return true; 

    
}
