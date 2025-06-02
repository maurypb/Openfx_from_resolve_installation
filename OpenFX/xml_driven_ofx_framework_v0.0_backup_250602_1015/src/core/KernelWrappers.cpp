#include "KernelWrappers.h"
#include "../Logger.h"
#include "ParameterValue.h"
#include <map>
#include <string>

#ifndef __APPLE__
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#endif

// Forward declaration for the actual __global__ kernel function
// This will be loaded dynamically from the .cu file specified in XML
#ifndef __APPLE__
extern "C" void call_gaussian_blur_kernel(
    void* stream, int width, int height,
    float radius, int quality, float maskStrength,
    cudaTextureObject_t inputTex, cudaTextureObject_t maskTex, 
    float* output, bool maskPresent
);
#endif

void RunGenericCudaKernel(
    void* stream, 
    int width, 
    int height,
    const std::map<std::string, ParameterValue>& params,
    const std::map<std::string, float*>& images,
    const std::map<std::string, std::string>& borderModes
) {
#ifndef __APPLE__
    Logger::getInstance().logMessage("RunGenericCudaKernel called");
    
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    
    // Extract parameters (specific to GaussianBlur for now - will be generalized)
    float radius = params.count("radius") ? params.at("radius").asFloat() : 5.0f;
    int quality = params.count("quality") ? params.at("quality").asInt() : 8;
    float maskStrength = params.count("maskStrength") ? params.at("maskStrength").asFloat() : 1.0f;
    
    Logger::getInstance().logMessage("  Extracted parameters: radius=%.2f, quality=%d, maskStrength=%.2f", 
                                   radius, quality, maskStrength);
    
    // Extract images
    const float* input = nullptr;
    const float* mask = nullptr;
    float* output = nullptr;
    
    if (images.count("Source")) {
        input = images.at("Source");
        Logger::getInstance().logMessage("  Found Source input: %p", input);
    }
    
    if (images.count("mask")) {
        mask = images.at("mask");
        Logger::getInstance().logMessage("  Found mask input: %p", mask);
    }
    
    if (images.count("output")) {
        output = images.at("output");
        Logger::getInstance().logMessage("  Found output: %p", output);
    }
    
    if (!input || !output) {
        Logger::getInstance().logMessage("ERROR: Missing required images - input: %p, output: %p", input, output);
        return;
    }
    
    // ALL THE CUDA SETUP CODE MOVED FROM CudaKernel.cu:
    
    // Create CUDA channel description for RGBA float format
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    
    // Allocate and setup input texture
    cudaArray_t inputArray = NULL;
    cudaMallocArray(&inputArray, &channelDesc, width, height);
    cudaMemcpy2DToArray(inputArray, 0, 0, input, width * sizeof(float4), 
                       width * sizeof(float4), height, cudaMemcpyHostToDevice);
    
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
    
    // Setup mask texture if available
    cudaArray_t maskArray = NULL;
    cudaTextureObject_t maskTex = 0;
    bool maskPresent = (mask != nullptr);
    
    if (maskPresent) {
        cudaMallocArray(&maskArray, &channelDesc, width, height);
        cudaDeviceSynchronize(); // Prevent mask flickering
        
        cudaMemcpy2DToArray(maskArray, 0, 0, mask, width * sizeof(float4), 
                         width * sizeof(float4), height, cudaMemcpyHostToDevice);
        
        cudaResourceDesc maskResDesc;
        memset(&maskResDesc, 0, sizeof(cudaResourceDesc));
        maskResDesc.resType = cudaResourceTypeArray;
        maskResDesc.res.array.array = maskArray;
        
        cudaCreateTextureObject(&maskTex, &maskResDesc, &texDesc, NULL);
    }
    
    // Launch the kernel
    Logger::getInstance().logMessage("  Launching CUDA kernel...");
    dim3 threads(16, 16, 1);
    dim3 blocks(((width + threads.x - 1) / threads.x), ((height + threads.y - 1) / threads.y), 1);
    
    // Call the specific kernel function (this will be generalized to load from XML)
    call_gaussian_blur_kernel(
        cudaStream, width, height,
        radius, quality, maskStrength,
        inputTex, maskTex, output, maskPresent
    );
    
    // Wait for completion
    cudaStreamSynchronize(cudaStream);
    
    // Cleanup
    cudaDestroyTextureObject(inputTex);
    if (maskPresent) {
        cudaDestroyTextureObject(maskTex);
    }
    cudaFreeArray(inputArray);
    if (maskArray) {
        cudaFreeArray(maskArray);
    }
    
    Logger::getInstance().logMessage("  CUDA kernel completed successfully");
#endif
}

void RunGenericOpenCLKernel(
    void* cmdQueue,
    int width,
    int height,
    const std::map<std::string, ParameterValue>& params,
    const std::map<std::string, float*>& images,
    const std::map<std::string, std::string>& borderModes
) {
    Logger::getInstance().logMessage("RunGenericOpenCLKernel called");
    
    // Extract parameters for GaussianBlur kernel
    float radius = params.count("radius") ? params.at("radius").asFloat() : 5.0f;
    int quality = params.count("quality") ? params.at("quality").asInt() : 8;
    float maskStrength = params.count("maskStrength") ? params.at("maskStrength").asFloat() : 1.0f;
    
    // Extract images
    const float* input = images.count("Source") ? images.at("Source") : nullptr;
    const float* mask = images.count("mask") ? images.at("mask") : nullptr;
    float* output = images.count("output") ? images.at("output") : nullptr;
    
    if (!input || !output) {
        Logger::getInstance().logMessage("ERROR: Missing required images");
        return;
    }
    
    // Call existing OpenCL function for now
    // TODO: Move OpenCL setup code here like we did for CUDA
    extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                               const float* p_Input, const float* p_Mask, float* p_Output);
    
    RunOpenCLKernel(cmdQueue, width, height, radius, quality, maskStrength, input, mask, output);
    Logger::getInstance().logMessage("  OpenCL kernel completed");
}

void RunGenericMetalKernel(
    void* cmdQueue,
    int width,
    int height,
    const std::map<std::string, ParameterValue>& params,
    const std::map<std::string, float*>& images,
    const std::map<std::string, std::string>& borderModes
) {
#ifdef __APPLE__
    Logger::getInstance().logMessage("RunGenericMetalKernel called");
    
    // Extract parameters for GaussianBlur kernel
    float radius = params.count("radius") ? params.at("radius").asFloat() : 5.0f;
    int quality = params.count("quality") ? params.at("quality").asInt() : 8;
    float maskStrength = params.count("maskStrength") ? params.at("maskStrength").asFloat() : 1.0f;
    
    // Extract images
    const float* input = images.count("Source") ? images.at("Source") : nullptr;
    const float* mask = images.count("mask") ? images.at("mask") : nullptr;
    float* output = images.count("output") ? images.at("output") : nullptr;
    
    if (!input || !output) {
        Logger::getInstance().logMessage("ERROR: Missing required images");
        return;
    }
    
    // Call existing Metal function for now  
    // TODO: Move Metal setup code here like we did for CUDA
    extern void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                              const float* p_Input, const float* p_Mask, float* p_Output);
    
    RunMetalKernel(cmdQueue, width, height, radius, quality, maskStrength, input, mask, output);
    Logger::getInstance().logMessage("  Metal kernel completed");
#endif
}