//  Implement a user-driven program in CUDA C for Addition of two Vectors. Take size of vector (number of elements) as input. Compare the time required for the same task on CPU.
// vector_addition.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

__global__ void vectorAdd(int *A, int *B, int *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(int argc, char **argv)
{
    int N;
    printf("Enter vector size: ");
    N = atoi(argv[1]);
    

    int size = N * sizeof(int);

    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C_cpu = (int *)malloc(size);
    int *h_C_gpu = (int *)malloc(size);

    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // ---------------- CPU computation ----------------
    clock_t cpu_start = clock();
    for (int i = 0; i < N; i++)
        h_C_cpu[i] = h_A[i] + h_B[i];
    clock_t cpu_end = clock();

    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // ---------------- GPU computation ----------------
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // ---------------- Output ----------------
    printf("\nCPU Time: %f seconds", cpu_time);
    printf("\nGPU Time: %f milliseconds\n", gpu_time);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
