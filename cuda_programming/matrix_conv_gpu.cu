#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16

__global__ void convolution2D(float *image, float *filter, float *output,
                                    int width, int height, int N)
{
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2]; // pad 1 pixel border (for N=3)

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int pad = N / 2;

    // Load each thread’s pixel into shared memory (with 1-pixel padding for N=3)
    int gx = x - pad;
    int gy = y - pad;
    if (gx >= 0 && gx < width && gy >= 0 && gy < height)
        tile[ty][tx] = image[gy * width + gx];
    else
        tile[ty][tx] = 0.0f;

    __syncthreads();

    // Compute only for valid threads (exclude halo threads)
    if (tx >= pad && ty >= pad && tx < BLOCK_SIZE - pad && ty < BLOCK_SIZE - pad &&
        x < width && y < height)
    {
        float sum = 0.0f;
        for (int fy = 0; fy < N; fy++)
        {
            for (int fx = 0; fx < N; fx++)
            {
                sum += tile[ty - pad + fy][tx - pad + fx] * filter[fy * N + fx];
            }
        }

        // Clamp
        if (sum < 0) sum = 0;
        if (sum > 255) sum = 255;

        output[y * width + x] = sum;
    }
}

extern "C" void gpu_convolution2D(float *image, float *filter, float *output,
                                        int width, int height, int N) {
    float *d_image, *d_filter, *d_output;
    int size = width * height * sizeof(float);
    cudaMalloc(&d_image, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_filter, N * N * sizeof(float));

    cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE + 2, BLOCK_SIZE + 2);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convolution2D<<<grid, block>>>(d_image, d_filter, d_output, width, height, N);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_image); cudaFree(d_filter); cudaFree(d_output);
}

int main(int argc, char **argv) {

    const char *inputFile = argv[1];

    int width, height, channels;
    unsigned char *img = stbi_load(inputFile, &width, &height, &channels, 1);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", inputFile);
        return 1;
    }

    printf("Loaded image: %d x %d (grayscale)\n", width, height);

    int size = width * height * sizeof(float);
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    for (int i=0; i < width * height; i++) {
        h_input[i] = (float)img[i];
    }

    int Ns[3] = {3, 5, 7};
    for (int test = 0; test < 3; test++) {
        int N = Ns[test];
        float *h_filter = (float *) malloc(N * N * sizeof(float));
    
        for (int i = 0; i < N * N; i++)
            h_filter[i] = 1.0f / (N * N);

        
        printf("\nRunning convolution with N=%d ...\n", N);
        
        float *d_input, *d_output, *d_filter;
        cudaMalloc(&d_input, size);
        cudaMalloc(&d_output, size);
        cudaMalloc(&d_filter, N * N * sizeof(float));

        cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, h_filter, N * N * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        convolution2D<<<grid, block>>>(d_input, d_filter, d_output, width, height, N);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("GPU Execution time: %.6f seconds\n", ms / 1000.0f);

        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

        free(h_filter);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_filter);
    }

    free(h_input);
    free(h_output);
    stbi_image_free(img);

    return 0;
}