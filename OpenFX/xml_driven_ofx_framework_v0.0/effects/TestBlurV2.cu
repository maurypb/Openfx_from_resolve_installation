// TestBlurV2.cu - Generated skeleton
// Fill in the image processing logic in the kernel function

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>  // For uintptr_t

// Image processing kernel - implement your algorithm here
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
    float maskStrength,
    float redness
)
{
    // Standard CUDA coordinate calculation
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < width) && (y < height)) {
        // Normalize coordinates to [0,1] range for texture sampling
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;
        
        // Calculate output array index
        const int index = ((y * width) + x) * 4;


        // Read mask value if available
        float maskValue = 1.0f;  // Default to full blur 
        
        // Sample directly from source image
        float4 srcColor = tex2D<float4>(SourceTex, u, v);
        
        if (maskPresent) {
            if (maskStrength >= 0.0f) {
                // Sample mask and apply (includes 0.0f case)
                float4 maskColor = tex2D<float4>(maskTex, u, v);
                maskValue = maskStrength * maskColor.w;
            } else {
                // Negative mask strength = no blur at all
                maskValue = 0.0f;
            }
        }

        // Calculate effective blur radius based on mask
        float effectiveRadius = radius * maskValue;
        
        // // Early exit if no blur needed (either radius is 0 or mask is 0)
        // if (effectiveRadius <= 0.0f) {
        //     // Just copy the source pixel - no blur applied
        //     output[index + 0] = srcColor.x*brightness+redness;
        //     output[index + 1] = srcColor.y*brightness;
        //     output[index + 2] = srcColor.z*brightness;
        //     output[index + 3] = srcColor.w;
        //     return;
        // }
        
        // Gaussian blur implementation (lousy)
        float4 sum = srcColor;  // Start with the source pixel
        float weightSum = 1.0f;  // Center pixel has weight 1
        
        // Perform sampling in a circle
        for (int i = 0; i < quality; ++i) {
            // Calculate sample angle
            float angle = (2.0f * 3.14159f * i) / quality;
            
            // Calculate sample positions at different distances
            for (float distance = 1.0f; distance <= effectiveRadius; distance += 1.0f) {
                float sampleX = x + cos(angle) * distance;
                float sampleY = y + sin(angle) * distance;
                
                // Normalize coordinates to [0,1] range
                float sample_u = (sampleX + 0.5f) / width;
                float sample_v = (sampleY + 0.5f) / height;
                
                // Sample using texture
                float4 color = tex2D<float4>(SourceTex, sample_u, sample_v);
                
                // Calculate weight (simplified for now)
                float weight = 1.0f;
                
                // Accumulate weighted color
                sum.x += color.x * weight;
                sum.y += color.y * weight;
                sum.z += color.z * weight;
                sum.w += color.w * weight;
                weightSum += weight;
            }
        }
        
        // Normalize by total weight
        if (weightSum > 0.0f) {
            sum.x /= weightSum;
            sum.y /= weightSum;
            sum.z /= weightSum;
            sum.w /= weightSum;
        }
        sum.x = sum.x * brightness+redness;
        sum.y=sum.y * brightness;
        sum.z=sum.z*brightness;
 
        // Write to output
        output[index + 0] = sum.x;
        output[index + 1] = sum.y;
        output[index + 2] = sum.z;
        output[index + 3] = sum.w;

    }
}

// Standardized wrapper function - framework calls this
extern "C" void call_testblurv2_kernel(
    void* stream,
    int width,
    int height,
    void** textures,
    int textureCount,
    bool* presenceFlags,
    float* output,
    float* floatParams,
    int* intParams,
    bool* boolParams
) {
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);

    // Unpack texture objects
    cudaTextureObject_t SourceTex = (cudaTextureObject_t)(uintptr_t)textures[0];
    cudaTextureObject_t maskTex = (cudaTextureObject_t)(uintptr_t)textures[1];
    cudaTextureObject_t selectiveTex = (cudaTextureObject_t)(uintptr_t)textures[2];

    // Unpack parameters
    float brightness = floatParams[0];
    float radius = floatParams[1];
    int quality = intParams[0];
    float maskStrength = floatParams[2];
    float redness = floatParams[3];

    // Launch configuration
    dim3 threads(16, 16, 1);
    dim3 blocks(((width + threads.x - 1) / threads.x), ((height + threads.y - 1) / threads.y), 1);

    // Launch the kernel
    TestBlurV2Kernel<<<blocks, threads, 0, cudaStream>>>(
        width,
        height,
        SourceTex,
        maskTex,
        presenceFlags[0],
        selectiveTex,
        presenceFlags[1],
        output,
        brightness,
        radius,
        quality,
        maskStrength,
        redness
    );
}