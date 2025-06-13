#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel
__global__ void minimalKernel() {
    // This kernel doesn't do much, just to test compilation
    // You could add a simple printf from the device if needed for more verbose testing
    // printf("Hello from GPU thread %d block %d\n", threadIdx.x, blockIdx.x);
}

// Host function
int main() {
    printf("Attempting to launch minimal CUDA kernel...\n");

    // Launch the kernel with 1 block and 1 thread
    minimalKernel<<<1, 1>>>();

    // Check for errors from kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Synchronize to ensure the kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during device synchronization: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Minimal CUDA kernel launched and synchronized successfully.\n");
    return 0;
}