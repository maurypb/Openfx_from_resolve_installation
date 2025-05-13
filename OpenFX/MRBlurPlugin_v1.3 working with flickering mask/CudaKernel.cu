// Using modern CUDA texture object API instead of texture references
#include <cuda_runtime.h>
#include <cstring>  // For memset
#include <stdio.h>
#include <cmath>    // For isnan and isinf
#include <algorithm> // For std::min


// the next part is the image processing part;
// __global__ is a CUDA-specific keyword marking this as a kernel function 
//that runs on the GPU and is callable from the host (CPU).
/*

The parameters are laid out in the call to this image processing function, towards the end of this file, at the end of the 
"RunCudaKernel()" function, which sets things up for this to run.
Specifically:
p_Width, p_Height is the frame size in pixels, float p_Radius is the blur radius as set by the user, p_Quality p_MaskStrength are parameters set by the user;

cudaTextureObject_t objects for input and mask images (similar to sampler2D in GLSL)

float* p_Output is the output buffer (in GLSL you'd write to gl_FragColor or a framebuffer)

*/

__global__ void GaussianBlurKernel(int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                                  cudaTextureObject_t tex_Input, cudaTextureObject_t tex_Mask, float* p_Output)
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
       float maskValue = 0.0f;  // Default to NO blur 
       
       // Sample directly from source image
       float4 srcColor = tex2D<float4>(tex_Input, u, v);
       
       // Only apply blur if mask strength is positive
       if (p_MaskStrength > 0.0f) {
           // Sample the mask texture with explicit bounds checking
           if (tex_Mask != 0) {
               float4 maskColor = tex2D<float4>(tex_Mask, u, v);
               // Use alpha channel for mask
               maskValue = maskColor.w;
               
               // Safety check for invalid values
               if (isnan(maskValue) || isinf(maskValue)) {
                   maskValue = 0.0f;  // Default to no blur if mask is corrupted
               }
               
               // Apply mask strength
               maskValue = p_MaskStrength * maskValue;
           }
       }
   
       // Calculate effective blur radius based on mask
       float effectiveRadius = p_Radius * maskValue;
       
       // Early exit if no blur needed (either radius is 0 or mask is 0)
       if (effectiveRadius <= 0.001f) {
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
               
               // Calculate Gaussian weight
               float weight = 1.0f;
               // Accumulate weighted color
               sum.x += color.x * weight;
               sum.y += color.y * weight;
               sum.z += color.z * weight;
               sum.w += color.w * weight;
               weightSum += weight;
           }
       }
       
       // Add center pixel with highest weight
       float centerWeight = 2.0f;  // Give center pixel more weight
       sum.x += srcColor.x * centerWeight;
       sum.y += srcColor.y * centerWeight;
       sum.z += srcColor.z * centerWeight;
       sum.w += srcColor.w * centerWeight;
       weightSum += centerWeight;
       
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

void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                  const float* p_Input, const float* p_Mask, float* p_Output)
{
    cudaStream_t stream = static_cast<cudaStream_t>(p_Stream);
    
    // Create CUDA channel description for RGBA float format
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    
    // Allocate CUDA array for input texture
    cudaArray_t inputArray = NULL; 
    cudaError_t err = cudaMallocArray(&inputArray, &channelDesc, p_Width, p_Height);
    if (err != cudaSuccess) {
        // Handle error - just return without processing
        return;
    }

    // Copy input data to CUDA array
    err = cudaMemcpy2DToArray(inputArray, 0, 0, p_Input, p_Width * sizeof(float4), 
                        p_Width * sizeof(float4), p_Height, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFreeArray(inputArray);
        return;
    }
    
    // Create input texture object
    cudaResourceDesc inputResDesc;
    memset(&inputResDesc, 0, sizeof(cudaResourceDesc));
    inputResDesc.resType = cudaResourceTypeArray;
    inputResDesc.res.array.array = inputArray;
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    
    cudaTextureObject_t inputTex = 0;
    err = cudaCreateTextureObject(&inputTex, &inputResDesc, &texDesc, NULL);
    if (err != cudaSuccess) {
        cudaFreeArray(inputArray);
        return;
    }
    
    // Create mask texture object if mask is provided
    cudaArray_t maskArray = NULL;
    cudaTextureObject_t maskTex = 0;
    
    if (p_Mask) {
        err = cudaMallocArray(&maskArray, &channelDesc, p_Width, p_Height);
        if (err != cudaSuccess) {
            cudaDestroyTextureObject(inputTex);
            cudaFreeArray(inputArray);
            return;
        }
        
        // Force memory synchronization before copying
        cudaDeviceSynchronize();
        
        // Copy mask data to CUDA array with explicit error checking
        err = cudaMemcpy2DToArray(maskArray, 0, 0, p_Mask, p_Width * sizeof(float4), 
                          p_Width * sizeof(float4), p_Height, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaDestroyTextureObject(inputTex);
            cudaFreeArray(inputArray);
            cudaFreeArray(maskArray);
            return;
        }
        
        // Force synchronization to ensure data is copied
        cudaDeviceSynchronize();
                          
        cudaResourceDesc maskResDesc;
        memset(&maskResDesc, 0, sizeof(cudaResourceDesc));
        maskResDesc.resType = cudaResourceTypeArray;
        maskResDesc.res.array.array = maskArray;
        
        err = cudaCreateTextureObject(&maskTex, &maskResDesc, &texDesc, NULL);
        if (err != cudaSuccess) {
            cudaDestroyTextureObject(inputTex);
            cudaFreeArray(inputArray);
            cudaFreeArray(maskArray);
            return;
        }
        
        // Force synchronization again
        cudaDeviceSynchronize();
    } else {
        // If no mask provided, create a dummy texture with zeros
        // This ensures the kernel always has a valid texture to sample from
        err = cudaMallocArray(&maskArray, &channelDesc, p_Width, p_Height);
        if (err != cudaSuccess) {
            cudaDestroyTextureObject(inputTex);
            cudaFreeArray(inputArray);
            return;
        }
        
        // Create a zero-filled dummy mask
        float* dummyMask = new float[p_Width * p_Height * 4]();  // Zero-initialized
        err = cudaMemcpy2DToArray(maskArray, 0, 0, dummyMask, p_Width * sizeof(float4), 
                          p_Width * sizeof(float4), p_Height, cudaMemcpyHostToDevice);
        delete[] dummyMask;
        
        if (err != cudaSuccess) {
            cudaDestroyTextureObject(inputTex);
            cudaFreeArray(inputArray);
            cudaFreeArray(maskArray);
            return;
        }
        
        cudaResourceDesc maskResDesc;
        memset(&maskResDesc, 0, sizeof(cudaResourceDesc));
        maskResDesc.resType = cudaResourceTypeArray;
        maskResDesc.res.array.array = maskArray;
        
        err = cudaCreateTextureObject(&maskTex, &maskResDesc, &texDesc, NULL);
        if (err != cudaSuccess) {
            cudaDestroyTextureObject(inputTex);
            cudaFreeArray(inputArray);
            cudaFreeArray(maskArray);
            return;
        }
    }
    
    // Launch kernel
    dim3 threads(16, 16, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), ((p_Height + threads.y - 1) / threads.y), 1);
    
    GaussianBlurKernel<<<blocks, threads, 0, stream>>>(p_Width, p_Height, p_Radius, p_Quality, p_MaskStrength, inputTex, maskTex, p_Output);
    
    // Wait for kernel to finish before cleaning up resources
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaDestroyTextureObject(inputTex);
    cudaDestroyTextureObject(maskTex);
    cudaFreeArray(inputArray);
    cudaFreeArray(maskArray);
}