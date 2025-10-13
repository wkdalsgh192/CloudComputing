#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void convolution2D(float *image, float *filter, float *output,
                   int width, int height, int N)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pad = N / 2;

    float sum = 0.0f;

    for (int fy = 0; fy < N; fy++)
    {
        for (int fx = 0; fx < N; fx++)
        {
            int iy = y + fy - pad;
            int ix = x + fx - pad;

            if (iy >= 0 && iy < height && ix >= 0 && ix < width)
            {
                sum += image[iy * width + ix] * filter[fy * N + fx];
            }
        }
    }

    // Clamp result to [0,255]
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;

    output[y * width + x] = sum;
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

    int Ns[4] = {1, 3, 5, 7};
    for (int test = 0; test < 4; test++) {
        int N = Ns[test];
        float *h_filter = (float *) malloc(N * N * sizeof(float));
    
        for (int i = 0; i < N * N; i++)
            h_filter[i] = 1.0f / (N * N);

        if (N > 1)
            printf("\nRunning convolution with N=%d ...\n", N);
        
        float *d_input, *d_output, *d_filter;
        cudaMalloc(&d_input, size);
        cudaMalloc(&d_output, size);
        cudaMalloc(&d_filter, N * N * sizeof(float));

        cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, h_filter, N * N * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
        
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
        if (N > 1)
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