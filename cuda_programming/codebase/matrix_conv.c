#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void convolution2D(float *image, float *filter, float *output, int width, int height, int N) {
    
    int pad = N / 2;
    for( int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            float sum = 0.0f;
            for (int fi = 0; fi < N; fi++) {
                for (int fj = 0; fj < N; fj++) {
                    int img_i = i + fi - pad;
                    int img_j = j + fj - pad;

                    if (img_i >= 0 && img_i < height && img_j >= 0 && img_j < width) {
                        sum += image[img_i * width + img_j] * filter[fi * N + fj];
                    }
                }
            }

            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[i * width+ j] = sum;
        }
    }
}

int main(int argc, char **argv) {

    const char *inputFile = argv[1];
    const char *outputFile = argv[2];
    const char *filter_name = argv[3];

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

    int N = 3;
    float kernel[N * N];

    if (strcmp(filter_name, "blur") == 0) {
        float k[9] = {
            1, 1, 1,
            1, 1, 1,
            1, 1, 1
        };
        for (int i = 0; i < 9; i++) kernel[i] = k[i] / 9.0f;
    }
    else if (strcmp(filter_name, "sharpen") == 0) {
        float k[9] = {
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0
        };
        memcpy(kernel, k, sizeof(k));
    }
    else if (strcmp(filter_name, "edge") == 0) {
        float k[9] = {
            -1, -1, -1,
            -1, 8, -1,
            -1, -1, -1
        };
        memcpy(kernel, k, sizeof(k));
    }
    else if (strcmp(filter_name, "emboss") == 0) {
        float k[9] = {
            -2, -1, 0,
            -1, 1, 1,
            0, 1, 2
        };
        memcpy(kernel, k, sizeof(k));
    }
    else if (strcmp(filter_name, "identity") == 0) {
        float k[9] = {
            0, 0, 0,
            0, 1, 0,
            0, 0, 0};
        memcpy(kernel, k, sizeof(k));
    }
    else
    {
        fprintf(stderr, "Unknown filter '%s'\n", filter_name);
        free(image);
        free(output);
        return 1;
    }

    convolution2D(image, kernel, output, width, height, N);

    unsigned char *out_img = malloc(width * height);
    for (int i=0; i < width * height; i++) {
        out_img[i] = (unsigned char)output[i];
    }

    if (!stbi_write_png(outputFile, width, height, 1, out_img, width))
        fprintf(stderr, "Failed to save image: %s\n", outputFile);
    else
        printf("Saved processed image: %s\n", outputFile);

    stbi_image_free(img);
    free(image);
    free(output);
    free(out_img);

    return 0;
}