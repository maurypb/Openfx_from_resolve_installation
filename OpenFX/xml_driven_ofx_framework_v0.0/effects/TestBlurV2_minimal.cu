// TestBlurV2.cu - Debug version without texture sampling
#include <cuda_runtime.h>
#include <cmath>

// Simple debug kernel - just fill with a test color
__global__ void TestBlurV2Kernel(
    int width,
    int height,
    cudaTextureObject_t SourceTex,
    cudaTextureObject_t maskTex,
    bool maskPresent,
    cudaTextureObject_t selectiveTex,
    bool selectivePresent,
    float* output,
    float brightness,
    float radius,
    int quality,
    float maskStrength
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds checking
    if (x >= width || y >= height) return;
    
    // Calculate output array index
    const int index = (y * width + x) * 4;
    
    // Simple test: fill with red color to verify kernel is working
    output[index + 0] = brightness;  // Red channel
    output[index + 1] = 0.0f;        // Green channel
    output[index + 2] = 0.0f;        // Blue channel  
    output[index + 3] = 1.0f;        // Alpha channel
}

// Bridge function - same as before
extern "C" void call_testblurv2_kernel(
    void* stream, int width, int height,
    cudaTextureObject_t SourceTex,
    cudaTextureObject_t maskTex,
    bool maskPresent,
    cudaTextureObject_t selectiveTex,
    bool selectivePresent,
    float* output,
    float brightness,
    float radius,
    int quality,
    float maskStrength
) {
    if (!output || width <= 0 || height <= 0) return;
    
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);

    dim3 threads(16, 16, 1);
    dim3 blocks(((width + threads.x - 1) / threads.x), ((height + threads.y - 1) / threads.y), 1);

    TestBlurV2Kernel<<<blocks, threads, 0, cudaStream>>>(
        width, height,
        SourceTex, maskTex, maskPresent,
        selectiveTex, selectivePresent,
        output,
        brightness, radius, quality, maskStrength
    );
    
    cudaError_t err = cudaGetLastError();
    // Can't log the error, but this will help isolate the issue
}