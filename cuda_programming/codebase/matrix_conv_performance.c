#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void convolution2D(float *image, float *filter, float *output,
                   int width, int height, int N)
{
    int pad = N / 2;

    for (int y = 0; y < height; y++)            // row index
    {
        for (int x = 0; x < width; x++)         // column index
        {
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
    }
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

    float *image = malloc(width * height * sizeof(float));
    float *output = malloc(width * height * sizeof(float));
    for (int i=0; i < width * height; i++) {
        image[i] = (float)img[i];
    }

    int Ns[4] = {1, 3, 5, 7};
    for (int test = 0; test < 4; test++) {
        int N = Ns[test];
        float *kernel = malloc(N * N * sizeof(float));
    
        for (int i = 0; i < N * N; i++)
            kernel[i] = 1.0f / (N * N);

        if (N > 1)
            printf("\nRunning convolution with N=%d ...\n", N);
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        convolution2D(image, kernel, output, width, height, N);
    
        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / 1e9;


        if (N > 1)
            printf("Execution time (N=%d): %.4f seconds\n", N, elapsed);

        free(kernel);
    }

    stbi_image_free(img);
    free(image);
    free(output);

    return 0;
}