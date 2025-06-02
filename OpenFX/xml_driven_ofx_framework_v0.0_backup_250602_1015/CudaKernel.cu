// Simplified CudaKernel.cu - Contains ONLY image processing code
// All setup/teardown moved to KernelWrappers.cpp

#include <cuda_runtime.h>
#include <cmath>

// Pure image processing kernel - this is what effect authors write
__global__ void GaussianBlurKernel(int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                                  cudaTextureObject_t tex_Input, cudaTextureObject_t tex_Mask, float* p_Output, bool p_MaskPresent)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   
   // Normalize coordinates to [0,1] range for texture fetch
   float u = (x + 0.5f) / p_Width;
   float v = (y + 0.5f) / p_Height;

   if ((x < p_Width) && (y < p_Height))
   {
       // Output index
       const int index = ((y * p_Width) + x) * 4;
       
       // Read mask value if available
       float maskValue = 1.0f;  // Default to full blur 
       
       // Sample directly from source image
       float4 srcColor = tex2D<float4>(tex_Input, u, v);
       
       if (p_MaskPresent) {
           if (p_MaskStrength >= 0.0f) {
               // Sample mask and apply (includes 0.0f case)
               float4 maskColor = tex2D<float4>(tex_Mask, u, v);
               maskValue = p_MaskStrength * maskColor.w;
           } else {
               // Negative mask strength = no blur at all
               maskValue = 0.0f;
           }
       }

       // Calculate effective blur radius based on mask
       float effectiveRadius = p_Radius * maskValue;
       
       // Early exit if no blur needed (either radius is 0 or mask is 0)
       if (effectiveRadius <= 0.0f) {
           // Just copy the source pixel - no blur applied
           p_Output[index + 0] = srcColor.x;
           p_Output[index + 1] = srcColor.y;
           p_Output[index + 2] = srcColor.z;
           p_Output[index + 3] = srcColor.w;
           return;
       }
       
       // Gaussian blur implementation
       float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
       float weightSum = 0.0f;
       
       // Perform sampling in a circle
       for (int i = 0; i < p_Quality; ++i) {
           // Calculate sample angle
           float angle = (2.0f * 3.14159f * i) / p_Quality;
           
           // Calculate sample positions at different distances
           for (float distance = 1.0f; distance <= effectiveRadius; distance += 1.0f) {
               float sampleX = x + cos(angle) * distance;
               float sampleY = y + sin(angle) * distance;
               
               // Normalize coordinates to [0,1] range
               float sample_u = (sampleX + 0.5f) / p_Width;
               float sample_v = (sampleY + 0.5f) / p_Height;
               
               // Sample using texture
               float4 color = tex2D<float4>(tex_Input, sample_u, sample_v);
               
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
       p_Output[index + 0] = sum.x;
       p_Output[index + 1] = sum.y;
       p_Output[index + 2] = sum.z;
       p_Output[index + 3] = sum.w;
   }
}

// Simple bridge function to launch the kernel from C++ code
// This is the only "plumbing" left in the kernel file
extern "C" void call_gaussian_blur_kernel(
    void* stream, int width, int height,
    float radius, int quality, float maskStrength,
    cudaTextureObject_t inputTex, cudaTextureObject_t maskTex, 
    float* output, bool maskPresent
) {
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    
    // Launch configuration
    dim3 threads(16, 16, 1);
    dim3 blocks(((width + threads.x - 1) / threads.x), ((height + threads.y - 1) / threads.y), 1);
    
    // Launch the actual image processing kernel
    GaussianBlurKernel<<<blocks, threads, 0, cudaStream>>>(
        width, height, radius, quality, maskStrength,
        inputTex, maskTex, output, maskPresent
    );
}