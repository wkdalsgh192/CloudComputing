#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define TILE_SIZE 16

__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N)
{

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Each thread computes one element of the output matrix
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // global row index
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // global col index

    float sum = 0.0f;
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
         // load elements into shared memory if within bounds
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_SIZE + threadIdx.y < N && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // wait for all threads to load tiles

        // multiply the two tiles
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads(); // wait for all threads before next tile
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void matrixMultiplyCPU(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 1024; // allow matrix size as input
    size_t size = N * N * sizeof(float);
    
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_cpu = (float *)malloc(size);

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // define block and grid dimemsions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // launch kernel with proper timing
    cudaEventRecord(start);  // start recording before launch

    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);

    // check for any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // stop event after all GPU work finishes
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // measure elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU execution time (N=%d): %.6f seconds\n", N, milliseconds / 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
     // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}