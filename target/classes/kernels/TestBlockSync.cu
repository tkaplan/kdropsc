#include "./BlockSync.cu"

extern "C"
__global__ void TestBlockSync(unsigned int *output) {
    int block_size = blockDim.x * blockDim.y;
    int block_index = blockIdx.x + blockIdx.y * gridDim.x;
    int local_index = threadIdx.x + threadIdx.y * blockDim.x;
    int global_index = block_index * block_size + local_index;

    if (global_index == 0)
        atomicCAS((unsigned int *)&QUANT_SEMAPHORE, 0, QUANT_SEMAPHORE);

    // Sync all threads
    while(QUANT_SEMAPHORE > 0);

    __syncthreads();

    BlockSyncTest(output);
}