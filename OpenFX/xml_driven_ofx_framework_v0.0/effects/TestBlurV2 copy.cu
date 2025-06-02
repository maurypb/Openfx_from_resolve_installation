// TestBlurV2.cu - Generated skeleton
// Fill in the image processing logic in the kernel function

#include <cuda_runtime.h>
#include <cmath>

// Image processing kernel - implement your algorithm here
__global__ void TestBlurV2Kernel(
    int width,
    int height,
    cudaTextureObject_t inputTex,
    cudaTextureObject_t maskTex,
    float* output,
    bool maskPresent,
    // Auto-generated from XML parameters:
    float brightness,  // from <parameter name="brightness" type="double">
    float radius,  // from <parameter name="radius" type="double">
    int quality,  // from <parameter name="quality" type="int">
    float maskStrength  // from <parameter name="maskStrength" type="double">
)
{
    // Standard CUDA coordinate calculation
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
   // Normalize coordinates to [0,1] range for texture fetch
   float u = (x + 0.5f) / width;
   float v = (y + 0.5f) / height;
 
    if ((x < width) && (y < height))
    {
        // Output index
        const int index = ((y * width) + x) * 4;
        
        // Read mask value if available
        float maskValue = 1.0f;  // Default to full blur 
        
        // Sample directly from source image
        float4 srcColor = tex2D<float4>(inputTex, u, v);
        
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
        
        // Early exit if no blur needed (either radius is 0 or mask is 0)
        if (effectiveRadius <= 0.0f) {
            // Just copy the source pixel - no blur applied
            output[index + 0] = srcColor.x;
            output[index + 1] = srcColor.y;
            output[index + 2] = srcColor.z;
            output[index + 3] = srcColor.w;
            return;
        }
        
        // Gaussian blur implementation
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float weightSum = 0.0f;
        
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
                float4 color = tex2D<float4>(inputTex, sample_u, sample_v);
                
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
        
        // Write to output
        output[index + 0] = sum.x;
        output[index + 1] = sum.y;
        output[index + 2] = sum.z;
        output[index + 3] = sum.w;
    }
}

// Bridge function - connects framework to your kernel
extern "C" void call_testblurv2_kernel(
    void* stream, int width, int height,
    float brightness,
    float radius,
    int quality,
    float maskStrength,
    cudaTextureObject_t inputTex, cudaTextureObject_t maskTex,
    float* output, bool maskPresent
) {
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);

    // Launch configuration
    dim3 threads(16, 16, 1);
    dim3 blocks(((width + threads.x - 1) / threads.x), ((height + threads.y - 1) / threads.y), 1);

    // Launch the kernel
    TestBlurV2Kernel<<<blocks, threads, 0, cudaStream>>>(
        width, height, inputTex, maskTex, output, maskPresent,
        brightness,
        radius,
        quality,
        maskStrength
    );
}