#include "infer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void){
    Tensor3D input = {0};
    Tensor3D output = {0};
    Conv2DLayer conv1 = {0};
    MaxPool2DLayer pool1 = {0};
    Conv2DLayer conv2 = {0};
    DenseLayer dense = {0};
    bool passed = false;

    if (!tensor3d_init(&input, 1, 4, 4) ||
        !conv2d_init(&conv1, 1, 1, 2, 2, 1, 0) ||
        !maxpool2d_init(&pool1, 2, 2, 1) ||
        !conv2d_init(&conv2, 1, 1, 2, 2, 1, 0) ||
        !dense_init(&dense, 1, 2)) {
        fprintf(stderr, "Failed to initialize the example CNN.\n");
        goto cleanup;
    }

    for (size_t index = 0; index < input.count; ++index) {
        input.data[index] = (float)(index + 1);
    }

    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 2; ++col) {
            conv2d_set_weight(&conv1, 0, 0, row, col, 1.0f);
            conv2d_set_weight(&conv2, 0, 0, row, col, 1.0f);
        }
    }

    conv2d_set_bias(&conv1, 0, -20.0f);
    dense_set_weight(&dense, 0, 0, 0.01f);
    dense_set_weight(&dense, 1, 0, 0.02f);

    if (!cnn_infer(&conv1, &pool1, &conv2, &dense, &input, &output)) {
        fprintf(stderr, "CNN inference failed.\n");
        goto cleanup;
    }

    printf("Output shape: %zux%zux%zu\n",
           output.channels, output.height, output.width);
    printf("Class 0 probability: %.6f\n", output.data[0]);
    printf("Class 1 probability: %.6f\n", output.data[1]);

    const float expected_class_0 = 0.276878f;
    const float expected_class_1 = 0.723122f;
    const float tolerance = 0.00001f;

    if (output.channels != 2 || output.height != 1 || output.width != 1 ||
        fabsf(output.data[0] - expected_class_0) > tolerance ||
        fabsf(output.data[1] - expected_class_1) > tolerance ||
        fabsf(output.data[0] + output.data[1] - 1.0f) > tolerance) {
        fprintf(stderr, "Unexpected inference result.\n");
        goto cleanup;
    }

    puts("End-to-end inference test passed.");
    passed = true;

cleanup:
    tensor3d_destroy(&output);
    tensor3d_destroy(&input);
    dense_destroy(&dense);
    conv2d_destroy(&conv2);
    conv2d_destroy(&conv1);
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
