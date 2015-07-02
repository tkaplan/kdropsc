#include "./QuantTensor.cu"

__device__ Tensor wt;

extern "C"
__global__ void TestQuantTensor (
    // Parse weights into weight tensor
    float *weights,

    // Our initial layer
    float *l0,

    // Dimmensions of layers
    int height,
    int width,
    int size,

    // Number of layers
    int layers,

    // Number of softmax outs
    int outs,

    // Get the results of our softmax
    float *results
) {
    // First we want to parse our
    // linear weights array into weight tensors
    // Build tensor blocks
    BuildWeightTensor(width, height, layers, size, weights, &wt);

    // Now that we have our weight tensors we can apply massively parrallel
    // techniques using all blocks of threads

    // Clear all memory
    clean();
}