// Implement a user-driven program in CUDA C for computing sum of all elements of a vector. Also determine MAX and MIN element from the vector. Compare the time required for the same. task on CPU.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <limits.h>

__global__ void computeSumMaxMin(int *arr, int *sum, int *max, int *min, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        atomicAdd(sum, arr[idx]);
        atomicMax(max, arr[idx]);
        atomicMin(min, arr[idx]);
    }
}

int main(int argc ,char**argv)
{
    int N;
    printf("Enter vector size: ");
    N = atoi(argv[1]);
    int size = N * sizeof(int);

    int *h_arr = (int *)malloc(size);
    for (int i = 0; i < N; i++)
        h_arr[i] = i + 1;

    // ---------------- CPU computation ----------------
    clock_t cpu_start = clock();
    int cpu_sum = 0, cpu_max = INT_MIN, cpu_min = INT_MAX;

    for (int i = 0; i < N; i++)
    {
        cpu_sum += h_arr[i];
        if (h_arr[i] > cpu_max) cpu_max = h_arr[i];
        if (h_arr[i] < cpu_min) cpu_min = h_arr[i];
    }

    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // ---------------- GPU computation ----------------
    int *d_arr, *d_sum, *d_max, *d_min;
    cudaMalloc((void **)&d_arr, size);
    cudaMalloc((void **)&d_sum, sizeof(int));
    cudaMalloc((void **)&d_max, sizeof(int));
    cudaMalloc((void **)&d_min, sizeof(int));

    int zero = 0, min_init = INT_MAX, max_init = INT_MIN;

    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &max_init, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, &min_init, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    computeSumMaxMin<<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_sum, d_max, d_min, N);
    cudaEventRecord(stop);

    int gpu_sum, gpu_max, gpu_min;
    cudaMemcpy(&gpu_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gpu_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gpu_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // ---------------- Output ----------------
    printf("\nCPU -> Sum: %d, Max: %d, Min: %d", cpu_sum, cpu_max, cpu_min);
    printf("\nGPU -> Sum: %d, Max: %d, Min: %d", gpu_sum, gpu_max, gpu_min);
    printf("\nCPU Time: %f seconds", cpu_time);
    printf("\nGPU Time: %f milliseconds\n", gpu_time);

    cudaFree(d_arr);
    cudaFree(d_sum);
    cudaFree(d_max);
    cudaFree(d_min);
    free(h_arr);

    return 0;
}
