// Using modern CUDA texture object API instead of texture references
#include <cuda_runtime.h>
#include <cstring>  // For memset
#include <stdio.h>


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
    /*
    each time this is run, it's in a separate "thread".  Unlike glsl, you don't get gl_FragCoord automatically, with the x,y of the current pixel;
    Instead, you get blockIdx, blockDim and threadIdx, which you can combine as shown to get the x,y of the current pixel.
    
    */

   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   // Normalize coordinates to [0,1] range for texture fetch
   float u = (x + 0.5f) / p_Width;
   float v = (y + 0.5f) / p_Height;
// below, we wrap everything in a range check, which glsl does automatically, but cuda does not;
//note that this is needed because cuda processes things in blocks of threads, and it's possible 
//for the block to have extra threads (for instance, if the image's width isn't divisible by the block size, 
// there will be extra threads at the end of each row for pixels that don't exist)
// and we don't want to write to memory outside of our buffers.  
//this is unlike sampling, where the cudaArray_t deals with boundary mirroring, clamping etc


   if ((x < p_Width) && (y < p_Height))
   {
       // Read mask value if available
       float maskValue = 1.0f;
       if (p_MaskStrength > 0.0f) {
           // Normalize coordinates to [0,1] range for texture fetch
           //float u = (x + 0.5f) / p_Width;
           //float v = (y + 0.5f) / p_Height;
           float4 maskColor = tex2D<float4>(tex_Mask, u, v);
           // Use alpha channel for mask
           maskValue = maskColor.w;
           // Apply mask strength
           //maskValue = 1.0f - (p_MaskStrength * maskValue); //this is inverted
           maskValue =  (p_MaskStrength * maskValue);
       }

       // Calculate effective blur radius based on mask
       float effectiveRadius = p_Radius * maskValue;
       
       // Output index
       //this is quite important!  the output frame buffer is essentially a linear array, with one float per channel, per pixel.
       //So, to find the position in the output buffer;
       /*
       first, consider the position of the first pixel in the current row, which is 
       the current row number (y) * the number of pixels in the row (p_width).
       then add the x position of this pixel, then multiply that whole thing by the number of channels per pixel.
       int index = ((y*p_Width)+x) *4;
       
       */
       const int index = ((y * p_Width) + x) * 4;
       
       // Early exit if no blur needed
       if (effectiveRadius == 0.0f) {
           // Just copy the source pixel
        //    float u = (x + 0.5f) / p_Width;
        //    float v = (y + 0.5f) / p_Height;
           float4 srcColor = tex2D<float4>(tex_Input, u, v);
           
           // Write to output
           // remember, we write to a linear array, one component per memory location.
           // so the index we calculated above is where the r channel would go, the next position is where green goes, next blue and next alpha.
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
               //float weight = exp(-(distance * distance) / (2.0f * effectiveRadius * effectiveRadius));
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
    //    float u = (x + 0.5f) / p_Width;
    //    float v = (y + 0.5f) / p_Height;
    //    float4 centerColor = tex2D<float4>(tex_Input, u, v);
    //    float centerWeight = 1.0f;
    //    sum.x += centerColor.x * centerWeight;
    //    sum.y += centerColor.y * centerWeight;
    //    sum.z += centerColor.z * centerWeight;
    //    sum.w += centerColor.w * centerWeight;
    //    weightSum += centerWeight;
       
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
    // so channelDesc specifies that there are 4 floats per pixel, eg 4 bytes per channel * 4 channels, or 16 bytes per channel.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    
    // Allocate CUDA array for input texture
    // "cudaArray_t" reads as "cuda Array Type".  the "_t" means "type".  This type of array is designed for texture memory,
    //unlike other cuda stuff which is all linear memory addressing, directly.  with a cudaArray_t buffer,  you need to use tex2d()
    //to access the bufffer, and it provides hardware filtering, normalized addressing (eg [0,1] uv space), and border handling (like wrap, clamp, mirror)
    cudaArray_t inputArray; 
    cudaMallocArray(&inputArray, &channelDesc, p_Width, p_Height);
    // we just allocated our frame buffer for the input image.

    
    // next, Copy input data to CUDA array
    /*
    
    parametersL
    inputArray - the destination array on the gpu we just allocated
    0,0 any offsets from the top left corner
    p_Input - the input image
    p_Width*sizeof(float4) this appears twice - the first time is the size of each row, and that is just the number of pixels across times the sizeof(float4)
    The second time it appears, it is the "pitch" of the source data.  The pitch is different than the width, in situations where there is extra horizontal padding after each line.channelDesc
    In this case, they will always be the same, as there's no padding.
    pHeight is the number of lines, and the last parameter (cudaMemcpyHostToDevice) is the "direction" of transfer, in
    this case, from the host (cpu memory) to the device (gpu memory).
    
    */
    cudaMemcpy2DToArray(inputArray, 0, 0, p_Input, p_Width * sizeof(float4), 
                        p_Width * sizeof(float4), p_Height, cudaMemcpyHostToDevice);
    
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
    cudaCreateTextureObject(&inputTex, &inputResDesc, &texDesc, NULL);
    
    // Create mask texture object if mask is provided
    cudaArray_t maskArray = NULL;
    cudaTextureObject_t maskTex = 0;
    
    if (p_Mask) {
        cudaMallocArray(&maskArray, &channelDesc, p_Width, p_Height);
        cudaMemcpy2DToArray(maskArray, 0, 0, p_Mask, p_Width * sizeof(float4), 
                          p_Width * sizeof(float4), p_Height, cudaMemcpyHostToDevice);
                          
        cudaResourceDesc maskResDesc;
        memset(&maskResDesc, 0, sizeof(cudaResourceDesc));
        maskResDesc.resType = cudaResourceTypeArray;
        maskResDesc.res.array.array = maskArray;
        
        cudaCreateTextureObject(&maskTex, &maskResDesc, &texDesc, NULL);
    } else {
        // Create a dummy mask with all zeros if no mask provided
        cudaMallocArray(&maskArray, &channelDesc, p_Width, p_Height);
        float* dummyMask = new float[p_Width * p_Height * 4]();
        cudaMemcpy2DToArray(maskArray, 0, 0, dummyMask, p_Width * sizeof(float4), 
                          p_Width * sizeof(float4), p_Height, cudaMemcpyHostToDevice);
        delete[] dummyMask;
        
        cudaResourceDesc maskResDesc;
        memset(&maskResDesc, 0, sizeof(cudaResourceDesc));
        maskResDesc.resType = cudaResourceTypeArray;
        maskResDesc.res.array.array = maskArray;
        
        cudaCreateTextureObject(&maskTex, &maskResDesc, &texDesc, NULL);
    }
    
    // Launch kernel
    dim3 threads(16, 16, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), ((p_Height + threads.y - 1) / threads.y), 1);
    
    GaussianBlurKernel<<<blocks, threads, 0, stream>>>(p_Width, p_Height, p_Radius, p_Quality, p_MaskStrength, inputTex, maskTex, p_Output);
    
    // Add synchronization to ensure kernel completion
    cudaDeviceSynchronize();
    
    // Check for any errors that occurred during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        FILE* logFile = fopen("/tmp/cudablur_log.txt", "a");
        if (logFile) {
            fprintf(logFile, "CUDA kernel error: %s\n", cudaGetErrorString(err));
            fclose(logFile);
        }
    }





    // Cleanup
    cudaDestroyTextureObject(inputTex);
    cudaDestroyTextureObject(maskTex);
    cudaFreeArray(inputArray);
    cudaFreeArray(maskArray);
}