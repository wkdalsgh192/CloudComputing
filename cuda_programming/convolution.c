#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void convolution2D(float *image, float *filter, float *output, int M, int N) {
    
    int pad = N / 2;
    for( int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            float sum = 0.0f;
            for (int fi = 0; fi < N; fi++) {
                for (int fj = 0; fj < N; fj++) {
                    int img_i = i + fi - pad;
                    int img_j = j + fj - pad;

                    if (img_i >= 0 && img_i < M && img_j >= 0 && img_j < M) {
                        sum += image[img_i * M + img_j] * filter[fi * N + fj];
                    }
                }
            }

            output[i * M + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    int M = 5; // image size (M x M)
    int N = 3; // filter size (N x N)

    float image[25] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25};

    float filter[9] = {
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0}; // sharpening filter

    float output[25] = {0};

    convolution2D(image, filter, output, M, N);

    printf("Output matrix after convolution:\n");
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("%6.2f ", output[i * M + j]);
        }
        printf("\n");
    }

    return 0;
}