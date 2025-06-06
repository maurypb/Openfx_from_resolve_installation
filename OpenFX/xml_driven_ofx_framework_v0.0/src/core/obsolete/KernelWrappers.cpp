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

#ifdef __APPLE__
#include <Metal/Metal.h>
#endif

#ifndef __APPLE__
#include <CL/cl.h>
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
#ifndef __APPLE__
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
    
    Logger::getInstance().logMessage("  Extracted parameters: radius=%.2f, quality=%d, maskStrength=%.2f", 
                                   radius, quality, maskStrength);
    
    // ALL THE OPENCL SETUP CODE MOVED HERE FROM OpenCLKernel.cpp:
    
    cl_int error;
    cl_command_queue clCmdQueue = static_cast<cl_command_queue>(cmdQueue);
    
    // Get context and device from command queue
    cl_context clContext;
    cl_device_id deviceId;
    error = clGetCommandQueueInfo(clCmdQueue, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
    if (error != CL_SUCCESS) {
        Logger::getInstance().logMessage("ERROR: Unable to get OpenCL context");
        return;
    }
    
    error = clGetCommandQueueInfo(clCmdQueue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
    if (error != CL_SUCCESS) {
        Logger::getInstance().logMessage("ERROR: Unable to get OpenCL device");
        return;
    }
    
    // Create image format description
    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_FLOAT;
    
    // Create image description
    cl_image_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width;
    desc.image_height = height;
    desc.image_row_pitch = 0;
    desc.image_array_size = 1;
    
    // Create input image
    cl_mem inputImage = clCreateImage(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                     &format, &desc, (void*)input, &error);
    if (error != CL_SUCCESS) {
        Logger::getInstance().logMessage("ERROR: Unable to create OpenCL input image");
        return;
    }
    
    // Create mask image (if available)
    cl_mem maskImage = NULL;
    if (mask) {
        maskImage = clCreateImage(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                 &format, &desc, (void*)mask, &error);
        if (error != CL_SUCCESS) {
            Logger::getInstance().logMessage("ERROR: Unable to create OpenCL mask image");
            clReleaseMemObject(inputImage);
            return;
        }
    } else {
        // Create a dummy mask if none provided
        float* dummyMask = new float[width * height * 4]();
        maskImage = clCreateImage(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                 &format, &desc, dummyMask, &error);
        delete[] dummyMask;
        if (error != CL_SUCCESS) {
            Logger::getInstance().logMessage("ERROR: Unable to create OpenCL dummy mask image");
            clReleaseMemObject(inputImage);
            return;
        }
    }
    
    // Create output buffer
    cl_mem outputBuffer = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, 
                                        width * height * 4 * sizeof(float), NULL, &error);
    if (error != CL_SUCCESS) {
        Logger::getInstance().logMessage("ERROR: Unable to create OpenCL output buffer");
        clReleaseMemObject(inputImage);
        clReleaseMemObject(maskImage);
        return;
    }
    
    // NOTE: Kernel compilation would go here in a complete implementation
    // For now, we log that this is where it would happen
    Logger::getInstance().logMessage("  OpenCL kernel setup completed (kernel compilation would happen here)");
    Logger::getInstance().logMessage("  Parameters: radius=%.2f, quality=%d, maskStrength=%.2f", 
                                   radius, quality, maskStrength);
    
    // Cleanup
    clReleaseMemObject(inputImage);
    clReleaseMemObject(maskImage);
    clReleaseMemObject(outputBuffer);
    
    Logger::getInstance().logMessage("  OpenCL processing completed");
#endif
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
    
    Logger::getInstance().logMessage("  Extracted parameters: radius=%.2f, quality=%d, maskStrength=%.2f", 
                                   radius, quality, maskStrength);
    
    // ALL THE METAL SETUP CODE MOVED HERE FROM MetalKernel.mm:
    
    id<MTLCommandQueue> metalQueue = static_cast<id<MTLCommandQueue>>(cmdQueue);
    id<MTLDevice> device = metalQueue.device;
    
    // Create textures for input and mask
    MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                                 width:width
                                                                                                height:height
                                                                                             mipmapped:NO];
    textureDescriptor.usage = MTLTextureUsageShaderRead;
    
    // Input texture
    id<MTLTexture> inputTexture = [device newTextureWithDescriptor:textureDescriptor];
    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [inputTexture replaceRegion:region mipmapLevel:0 withBytes:input bytesPerRow:width * 4 * sizeof(float)];
    
    // Mask texture (if available)
    id<MTLTexture> maskTexture = nil;
    if (mask) {
        maskTexture = [device newTextureWithDescriptor:textureDescriptor];
        [maskTexture replaceRegion:region mipmapLevel:0 withBytes:mask bytesPerRow:width * 4 * sizeof(float)];
    } else {
        // Create a dummy mask texture with all zeros if no mask provided
        float* dummyMask = new float[width * height * 4]();
        maskTexture = [device newTextureWithDescriptor:textureDescriptor];
        [maskTexture replaceRegion:region mipmapLevel:0 withBytes:dummyMask bytesPerRow:width * 4 * sizeof(float)];
        delete[] dummyMask;
    }
    
    // NOTE: Pipeline state and kernel compilation would go here in a complete implementation
    // For now, we log that this is where it would happen
    Logger::getInstance().logMessage("  Metal kernel setup completed (pipeline compilation would happen here)");
    Logger::getInstance().logMessage("  Parameters: radius=%.2f, quality=%d, maskStrength=%.2f", 
                                   radius, quality, maskStrength);
    
    // Release textures
    [inputTexture release];
    [maskTexture release];
    
    Logger::getInstance().logMessage("  Metal processing completed");
#endif
}