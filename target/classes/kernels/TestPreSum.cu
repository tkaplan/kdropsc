#ifndef TEST_PRESUM
#define TEST_PRESUM
    #include "./PreSum.cu"

    extern "C"
    __global__ void TestPreSum(
        float *weights,
        float *layer_matrix,
        int height,
        int width,
        float *accumulated
        ) {

        // Declare shared memory
        __shared__ float matrix[4][256];
        __shared__ int indexer;

        PreSum(
            weights,
            layer_matrix,
            height,
            width,
            accumulated,

            matrix[0],
            &indexer
        );
    }
#endif