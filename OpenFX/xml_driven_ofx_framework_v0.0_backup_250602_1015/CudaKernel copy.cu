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
       

        // if (p_MaskPresent)
        // {
        //     // Only apply blur if mask strength is positive
        //     if (p_MaskStrength > 0.0f) {
        //         // Sample the mask texture with explicit bounds checking
        //         if (tex_Mask != 0) {
        //             float4 maskColor = tex2D<float4>(tex_Mask, u, v);
        //             // Use alpha channel for mask
        //             maskValue = maskColor.w;
                    
        //             // Safety check for invalid values
        //             if (isnan(maskValue) || isinf(maskValue)) {
        //                 maskValue = 0.0f;  // Default to no blur if mask is corrupted
        //             }
                    
        //             // Apply mask strength
        //             maskValue = p_MaskStrength * maskValue;
        //         }
        //     }
        // }


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
       
    //    // Add center pixel with highest weight
    //    float centerWeight = 2.0f;  // Give center pixel more weight
    //    sum.x += srcColor.x * centerWeight;
    //    sum.y += srcColor.y * centerWeight;
    //    sum.z += srcColor.z * centerWeight;
    //    sum.w += srcColor.w * centerWeight;
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
    cudaStream_t stream = static_cast<cudaStream_t>(p_Stream); //housekeeping - convert
    //generic pointer to a cudaStream_t pointer.
    
    // Create CUDA channel description for RGBA float format
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>(); //hard-coded to float4 datatype (eg rgba)
    // will be changed to query the input image as to what it consists of (eg pixel format).
    // perhaps Get format from the actual image (not hard-coded):
    // pixelFormat = getPixelFormatFromImage(_srcImg);
    
    // Allocate CUDA array for input texture
    cudaArray_t inputArray = NULL; // declaring a pointer to cuda array, but it's null for now, as we've not yet actually allocated that memory.
    cudaMallocArray(&inputArray, &channelDesc, p_Width, p_Height); //here we actually allocate, and assign the starting address to inputArray (which is a pointer)

    // Copy input data to CUDA array
    cudaMemcpy2DToArray(inputArray, 0, 0, p_Input, p_Width * sizeof(float4), 
                       p_Width * sizeof(float4), p_Height, cudaMemcpyHostToDevice);


    // let's go over these parameters of cudaMemcpyToArray;
    /*

    inputArray: the pointer to where, on the device, the memory starts.
    0,0: any offsets from this
    p_Input: the pointer to the pixel array on the cpu
    p_Width*sizeof(float4) this is the memory size of one row.  float4 indicates 4 channels, each channel takes one float.
        sizeof(float4) is then the size of 1 pixel's data. p_Width*sizeof(float4) is the size of one row of pixels.
    p_Width*sizeof(float4) appears again! This is the size of one row, on the device, so you can have (for instance) padding.channelDesc
    So the first one is the source array's row size, and the second is the destination's row size.
    p_height is the number of rows (I guess there's no vertical padding)
    That final variable is a constant indicating the direction of the transfer, in this case host (cpu) to device (gpu).


    */


    
    // Create input resource description, and texture description. 
    cudaResourceDesc inputResDesc; //declare a var (object - struct) called inputResDesc that is a cudaResourceDesc;
    // it will house the description (not the content) of the input clip.
    memset(&inputResDesc, 0, sizeof(cudaResourceDesc)); //this "zeros out" all of the elements in inputResDesc
    inputResDesc.resType = cudaResourceTypeArray;// set the resource type to "array type"
    inputResDesc.res.array.array = inputArray;//this is the actual pointer to the beginning of the image array, that we just allocated above.
    
    cudaTextureDesc texDesc;  //declare a var (struct) called texDesc, that is a cudaTextureDesc object. Note that we didn't call it inputTexDesc, because we will use this for both the input and mask.
    //This will house metadata about how the resource above should be handled, as a "texture"
    memset(&texDesc, 0, sizeof(cudaTextureDesc)); //first zero out all of the values
    texDesc.addressMode[0] = cudaAddressModeClamp; //set the edge handling in the horizontal? to clamped
    texDesc.addressMode[1] = cudaAddressModeClamp;//then the vertical
    texDesc.filterMode = cudaFilterModeLinear; //set filtering
    texDesc.readMode = cudaReadModeElementType; //??
    texDesc.normalizedCoords = 1; // not sure... allows addressing data in the texture by normalized (0-1) uv coords?
    
    cudaTextureObject_t inputTex = 0;  //inputTex isn't a pointer, it's an "index" that cuda will use to reference the input texture
    cudaCreateTextureObject(&inputTex, &inputResDesc, &texDesc, NULL);// we create a "textureObject" that contains the information from the 
    //inputResDesc and texDesc we defined above;  We passed stuff by ref, but we don't modify anything except inputTex, which we assign the index that this function has created
    //we passed inputResDesc and texDesc by reference, solely to avoid the overhead of making local copies within the function.
    
    //  **** input setup complete, now the mask, very similar.





    // Create mask texture object if mask is provided
    cudaArray_t maskArray = NULL; // create a pointer to a cudaArray, but for now, points to nothing.
    cudaTextureObject_t maskTex = 0; //this is the variable that will hold that texture index, like above.
    
    if (p_Mask) {
        cudaMallocArray(&maskArray, &channelDesc, p_Width, p_Height);// cuda allocates memory on GPU;
        //maskArray points to the first element of the actual mask data; channelDesc is how many elements per pixel;
        //multiplying channelDesc by width and height give us the size of the memory needed.
        
        // Force synchronization before copying mask data **** this looks like where the flickering came from, add this back
        cudaDeviceSynchronize();
        
        cudaMemcpy2DToArray(maskArray, 0, 0, p_Mask, p_Width * sizeof(float4), 
                         p_Width * sizeof(float4), p_Height, cudaMemcpyHostToDevice);
        
   
                          
        cudaResourceDesc maskResDesc;
        memset(&maskResDesc, 0, sizeof(cudaResourceDesc));
        maskResDesc.resType = cudaResourceTypeArray;
        maskResDesc.res.array.array = maskArray;
        
        cudaCreateTextureObject(&maskTex, &maskResDesc, &texDesc, NULL); //notice that we've reused the texDesc from the input
        
        // Force synchronization after creating texture
        //cudaDeviceSynchronize();
    } 
    // else 
    // {
    //     // Create a dummy mask with all zeros
    //     cudaMallocArray(&maskArray, &channelDesc, p_Width, p_Height);
    //     cudaDeviceSynchronize();

    //     float* dummyMask = new float[p_Width * p_Height * 4]();  // Zero-initialized
    //     cudaMemcpy2DToArray(maskArray, 0, 0, dummyMask, p_Width * sizeof(float4), 
    //                       p_Width * sizeof(float4), p_Height, cudaMemcpyHostToDevice);
    //     delete[] dummyMask;
        
    //     cudaResourceDesc maskResDesc;
    //     memset(&maskResDesc, 0, sizeof(cudaResourceDesc));
    //     maskResDesc.resType = cudaResourceTypeArray;
    //     maskResDesc.res.array.array = maskArray;
        
    //     cudaCreateTextureObject(&maskTex, &maskResDesc, &texDesc, NULL);
    // }
    
    // Launch kernel
    dim3 threads(16, 16, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), ((p_Height + threads.y - 1) / threads.y), 1);
    
    GaussianBlurKernel<<<blocks, threads, 0, stream>>>(p_Width, p_Height, p_Radius, p_Quality, p_MaskStrength, 
        inputTex, maskTex, p_Output, p_Mask != nullptr);
    
    // Wait for kernel to finish
    //cudaDeviceSynchronize();
    //instead, wait for this stream to finish?
    cudaStreamSynchronize(stream);
    // Cleanup
    cudaDestroyTextureObject(inputTex);
    cudaDestroyTextureObject(maskTex);
    cudaFreeArray(inputArray);
    cudaFreeArray(maskArray);
}