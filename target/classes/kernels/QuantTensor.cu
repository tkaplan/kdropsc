#include "./BlockSync.cu"
#ifndef QUANTTENSOR
#define QUANTTENSOR

    typedef struct Matrix{
        int width;
        int height;
        float *elements;
    } Matrix;

    // A tensor will point to itself
    // if rank > 2
    typedef struct Tensor {
        int width;
        int height;
        int size;
        int layers;
        int rank;
        Tensor **tensor;
        Matrix **matrix;
    } Tensor;

    __device__ void BuildWeightTensor(int width, int height, int layers, int size, float* elements, Tensor* tensor) {
        InitBlockSync();
        // Lets get our thread index
        int grid_size = gridDim.x * gridDim.y;
        int block_size = blockDim.x * blockDim.y;
        int block_index = blockIdx.x + blockIdx.y * gridDim.x;
        int local_index = threadIdx.x + threadIdx.y * blockDim.x;
        int global_index = block_index * block_size + local_index;
        int matrix_size = width * height;

        // First we want to create tensor structures to hold
        // our weight tensors for each layer. Avoid race condition
        // in if block.
        if (global_index == 0) {
            tensor->width = width;
            tensor->height = height;
            tensor->size = size;
            tensor->layers = layers;

            // Rank 4 Tensor: tensors -> (tensors -> matrices)
            // Rank 3 Tensor: tensors -> matrices
            // Rank 2 Tensor: holds matrices
            // Rank 1 Tensor: holds vectors
            // Rank 0 Tensor: holds scalars
            tensor->rank = 4;

            tensor->tensor = (Tensor **)malloc(sizeof(Tensor *) * layers);
        }

        BlockSync();

        // Avoid race condition by allowing only one thread
        // to modify tensor metadata
        if (global_index == 0) {
            // Now we iterate through our tensor list and allocate memory
            // for each tensor structure
            for (int i = 0; i < layers; i ++) {
                tensor->tensor[i] = (Tensor *)malloc(sizeof(Tensor));
                tensor->tensor[i]->width = width;
                tensor->tensor[i]->height = height;
                tensor->tensor[i]->rank = 3;
                tensor->tensor[i]->matrix = (Matrix **)malloc(sizeof(Matrix *) * matrix_size);
            }
        }

        BlockSync();

        // Matrix double pointers our allocated, we have alot of allocation
        // operations to do now so we will parrallelize that now
        Matrix* matrix;
        int weight_index;
        for (int i = 0; i < layers; i ++) {
            Tensor* tensor_e = tensor->tensor[i];

            for (int j = global_index; j < matrix_size; j += block_size * grid_size) {
                // Allocate our matrix
                tensor_e->matrix[j] = (Matrix *)malloc(sizeof(Matrix));

                matrix = tensor_e->matrix[j];

                matrix->width = width;
                matrix->height = height;

                matrix->elements = (float *)malloc(sizeof(float) * matrix_size);
                for (int k = 0; k < matrix_size; k++) {
                    weight_index = k + j * matrix_size + i * matrix_size * matrix_size;
                    matrix->elements[k] = elements[weight_index];
                }
            }
        }

        BlockSync();
    }

    __device__ void clean() {

    }

#endif