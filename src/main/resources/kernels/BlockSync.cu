#ifndef QAUNT_BLOCKSYNC
#define QAUNT_BLOCKSYNC
    volatile __device__ unsigned int QUANT_SEMAPHORE;

    __device__ void BlockSyncTest(unsigned int *test) {
        int grid_size = gridDim.x * gridDim.y;
        int block_size = blockDim.x * blockDim.y;
        int block_index = blockIdx.x + blockIdx.y * gridDim.x;
        int local_index = threadIdx.x + threadIdx.y * blockDim.x;
        int global_index = block_index * block_size + local_index;

        if (local_index == 0) {
            atomicAdd((unsigned int *)&QUANT_SEMAPHORE, 1);

            // Sync the first local_index of threads
            while (QUANT_SEMAPHORE < grid_size);

            if (global_index == 0)
                atomicSub((unsigned int*)&QUANT_SEMAPHORE, grid_size);

            // Wait until thread zero releases
            while(QUANT_SEMAPHORE != 0);
        }

        __syncthreads();
    }

    __device__ void BlockSync() {
        int grid_size = gridDim.x * gridDim.y;
        int block_size = blockDim.x * blockDim.y;
        int block_index = blockIdx.x + blockIdx.y * gridDim.x;
        int local_index = threadIdx.x + threadIdx.y * blockDim.x;
        int global_index = block_index * block_size + local_index;

        if (local_index == 0) {
            atomicAdd((unsigned int *)&QUANT_SEMAPHORE, 1);

            // Sync the first local_index of threads
            while (QUANT_SEMAPHORE < grid_size);

            if (global_index == 0)
                atomicSub((unsigned int*)&QUANT_SEMAPHORE, grid_size);

            // Wait until thread zero releases
            while(QUANT_SEMAPHORE != 0);
        }

        __syncthreads();
    }

    __device__ void InitBlockSync() {
        int block_size = blockDim.x * blockDim.y;
        int block_index = blockIdx.x + blockIdx.y * gridDim.x;
        int local_index = threadIdx.x + threadIdx.y * blockDim.x;
        int global_index = block_index * block_size + local_index;

        if (global_index == 0)
            atomicCAS((unsigned int *)&QUANT_SEMAPHORE, 0, QUANT_SEMAPHORE);

        // Sync all threads
        while(QUANT_SEMAPHORE != 0);
        BlockSync();
    }
#endif