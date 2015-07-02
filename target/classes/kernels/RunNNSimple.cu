#include "./BlockSync.cu"
#include "./PreSum.cu"
#include "./QuantTensor.cu"

__device__ Tensor wt;
__device__ Matrix output;
__device__ int semaphore;

__device__ void LoadSharedLayer(float *dest, float *src, int size, int block_size, int index) {
    for (int i = index; i < size; i += block_size) {
        dest[i] = src[i];
    }
}

__device__ void SumLayer(Matrix* weights) {

}

extern "C"
__global__ void RunNNSimple(
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

    int grid_size = gridDim.x * gridDim.y;
    int block_size = blockDim.x * blockDim.y;
    int block_index = blockIdx.x + blockIdx.y * gridDim.x;
    int local_index = threadIdx.x + threadIdx.y * blockDim.x;
    int global_index = block_index * block_size + local_index;
    int matrix_size = width * height;

    InitBlockSync();

    __shared__ float intermediate_layer[784];

    // Allocate our output layer
    if (global_index == 0) {
        semaphore = 0;

        output.height = height;
        output.width = width;
        output.elements = (float *)malloc(sizeof(float) * matrix_size);
    }

    LoadSharedLayer(intermediate_layer, l0, matrix_size, block_size, local_index);
    __syncthreads();

    // Build out our weight tensors
    BuildWeightTensor(width, height, layers, size, weights, &wt);

    // Now that we have our weight tensor we need to iterate
    // through our layers and compute the PreSum.
    for (int i = 0; i < layers; i ++) {
        // Retrieve our layer
        Tensor *layer = wt.tensor[i];

        // Retrieve weight matrix and assign it to a block
        for (int j = block_index; j < matrix_size; j += grid_size) {
            Matrix *matrix = layer->matrix[j];
            //SumLayer(matrix, )
            //PreSum(matrix->elements, l0, &result);

            // We now have the result, now we want to
            // apply our transfer function to the result
            // and assign our element to our intermediate
            // layer.

            //intermediate_layer[j] = tanhf(result);
        }
        // Wait until we are finished computing our layers
        __syncthreads();
    }
}