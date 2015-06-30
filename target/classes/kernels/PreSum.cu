extern "C"
__global__ void PreSum(
    float *weights,
    float *layer_matrix,
    int height,
    int width,
    int size,
    float *accumulated
    ) {
    // Declare shared memory
    __shared__ float matrix[256];
    __shared__ int indexer;

    // Get thread relative position
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Get thread height and width
    int threadIndex = tx + blockDim.x * ty;
    int thread_num = blockDim.y * blockDim.x;

    // We need to find out how many cells each of our
    // matrix our cell block gets
    int group_size = (float)ceil((double)size / (double)thread_num);

    // Get the initial and indexed position in our array
    int index = (tx + ty * blockDim.x) * group_size;

    // Start acc to 0
    float acc = 0;
    int upper_bound = index + group_size < size ? index + group_size : size;

    // Accumulate our values
    for (; index <  upper_bound; index ++)
        acc += layer_matrix[index] * weights[index];

    // Reduce our arbitrary sized array to 16 x 16
    matrix[threadIndex] = acc;

    if (threadIndex == thread_num - 1)
        indexer = 1;

    __syncthreads();

    while ((threadIndex + 1) % indexer == 0) {
        // If we are the right in the next term
        // lets do the addition
        if ((threadIndex + 1) % (indexer * 2) == 0) {
            // The left most node will be accumulated into the right
            // most node
            matrix[threadIndex] += matrix[threadIndex - indexer];
        }

        // Shifting indexer to the left
        // effectively multiplies it by two
        if (threadIndex == thread_num - 1) {
            indexer *= 2;
        }
        __syncthreads();
    }

    if (threadIndex == thread_num - 1) {
        *accumulated = matrix[threadIndex];
    }
}