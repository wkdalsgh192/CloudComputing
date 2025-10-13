import ctypes
import numpy as np
import time
import sys
import cv2

input_path = sys.argv[1]
# Load shared library
lib = ctypes.cdll.LoadLibrary("./libconv.so")
# Define argument types
lib.gpu_convolution2D.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),  # image
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),  # filter
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),  # output
    ctypes.c_int,  # width
    ctypes.c_int,  # height
    ctypes.c_int   # N (filter size)
]
lib.gpu_convolution2D.restype = None  # void


image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Failed to load image: {input_path}")
    sys.exit(1)

height, width = image.shape
print(f"Loaded image: {width}x{height} (grayscale)")

img_f = image.astype(np.float32)

Ns = [1, 3, 5, 7]
for N in Ns:
    if N > 1:
        print(f"\nRunning GPU convolution with N={N}...")

    # Create simple blur kernel dynamically
    kernel = np.ones((N, N), dtype=np.float32) / (N * N)
    output = np.zeros_like(img_f, dtype=np.float32)

    start = time.time()
    lib.gpu_convolution2D(img_f.ravel(), kernel.ravel(), output.ravel(),
                          width, height, N)
    end = time.time()

    # print(f"N={N} completed in {end - start:.4f} seconds")

    if N > 1:
        print(f"Python call to CUDA library completed in {end - start:.4f} seconds")