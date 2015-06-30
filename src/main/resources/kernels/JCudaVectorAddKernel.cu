extern "C"
__global__ void JCudaVectorAddKernel(int n, int *a, int *b, int *sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        sum[i] = a[i] + b[i];
    }
}