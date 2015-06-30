#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    int stride;
    float *elements;
} Matrix;

__device__ float getElement(const Matrix C, int row, int col) {
    return C.elements[row * C.stride + col];
}

__device__ void setElement(Matrix C, int row, int col, float value) {
    C.elements[row * C.stride + col] = value;
}

__device__ Matrix getSubMatrix(Matrix C, int row, int col) {
    Matrix Csub;
    Csub.width = BLOCK_SIZE;
    Csub.height = BLOCK_SIZE;
    Csub.stride = C.stride;
    Csub.elements = &C.elements[C.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];

    return Csub;
}

__device__ Matrix buildMatrix(int width, int height, float *elements) {
    Matrix matrix;
    matrix.width = width;
    matrix.height = height;
    matrix.stride = width;
    matrix.elements = elements;
    return matrix;
}

extern "C"
__global__ void JCudaMatrixSharedMemKernel(
    int widthA,
    int heightA,
    float *elementsA,
    int widthB,
    int heightB,
    float * elementsB,
    float * elementsC
    ) {

    // Build matrices
    Matrix A = buildMatrix(widthA, heightA, elementsA);
    Matrix B = buildMatrix(widthB, heightB, elementsB);
    Matrix C = buildMatrix(widthB, heightA, elementsC);

    // Block row and col
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = getSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and col within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m <= (A.width / BLOCK_SIZE); m++) {
        // Get sub-matrix Asub of A
        Matrix Asub = getSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = getSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matirx
        As[row][col] = getElement(Asub, row, col);
        Bs[row][col] = getElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        // Multipy Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            if(
                (A.width < BLOCK_SIZE * blockCol + e) ||
                (A.height < BLOCK_SIZE * m + e) ||
                (B.width < BLOCK_SIZE * m + e) ||
                (B.height < BLOCK_SIZE * blockRow + e)
            ) {}
            else {
            Cvalue += As[row][e] * Bs[e][col];
            }

        }

        // Synchronize to make sure that the preceding
        // compuation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    setElement(Csub, row, col, Cvalue);
}