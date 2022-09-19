#include <stdio.h>
#define N (1024 * 1024)

__global__ void kernel (float * data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // current thread number
    float x = 2.0f * 3.1415926f * (float) idx / (float) N; // argument value
    data[idx] = sinf(sqrtf(x)); // find a value and write it down to an array
}

// - allocate a private thread to each array element (total N)
// - this private threads calculate required values
// - each thread has unique id

int main (int argc, char * argv[])
{
    float * a = new float [N]; // allocate host memory (dynamic array)
    float * dev = NULL; // allocate device memory

    // allocate GPU memory for N elements
    cudaMalloc((void**) & dev, N * sizeof(float));

    // launch N blocks by 512 threads
    // @variable: kernel - performed function per thread
    // @variable: dev - data array
    kernel <<< dim3((N / 512), 1), dim3(512, 1) >>> (dev);

    // copy the results from GPU (DRAM) to CPU (N elements)
    cudaMemcpy(a, dev, N * sizeof(float), cudaMemcpyDeviceToHost);

    // free up memory
    cudaFree(dev);

    // print results
    for (int idx = 0; idx < N; idx++) 
    {
        printf("a[%d] = %.5f\n", idx, a[idx]);
    }

    return 0;
}