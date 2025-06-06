#include "KernelWrappers.h"
#include "../Logger.h"
#include "ParameterValue.h"
#include <map>
#include <string>
#include <vector>
#include "KernelRegistry.h"

#ifndef __APPLE__
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#endif

void cleanupCudaResources(const std::vector<cudaTextureObject_t> &textures,
                          const std::vector<cudaArray_t> &arrays)
{
    Logger::getInstance().logMessage("  Cleaning up GPU resources...");

    // Destroy texture objects
    for (cudaTextureObject_t tex : textures)
    {
        if (tex != 0)
        {
            cudaDestroyTextureObject(tex);
        }
    }

    // Free CUDA arrays
    for (cudaArray_t array : arrays)
    {
        if (array != NULL)
        {
            cudaFreeArray(array);
        }
    }

    Logger::getInstance().logMessage("  ✓ Cleaned up %d textures and %d arrays",
                                     (int)textures.size(), (int)arrays.size());
}

void RunGenericCudaKernel(
    void *stream,
    int width,
    int height,
    const std::map<std::string, ParameterValue> &params,
    const std::map<std::string, float *> &images,
    const std::map<std::string, std::string> &borderModes,
    const XMLEffectDefinition &xmlDef)
{
#ifndef __APPLE__
    Logger::getInstance().logMessage("RunGenericCudaKernel called for effect: %s", xmlDef.getName().c_str());

    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);

    // Storage for cleanup - track arrays and texture objects
    std::vector<cudaArray_t> allocatedArrays;
    std::vector<cudaTextureObject_t> createdTextures;

    // Build kernel arguments in XML-defined order
    std::vector<void *> kernelArgs;
    std::vector<std::unique_ptr<void, void (*)(void *)>> argStorage;

    // Helper lambda to store arguments safely
    auto storeArg = [&](auto value) -> void *
    {
        using T = decltype(value);
        T *ptr = new T(value);
        argStorage.emplace_back(ptr, [](void *p)
                                { delete static_cast<T *>(p); });
        return ptr;
    };

    // 1. Fixed parameters (always first)
    kernelArgs.push_back(storeArg(width));
    kernelArgs.push_back(storeArg(height));

    Logger::getInstance().logMessage("  Added fixed parameters: width=%d, height=%d", width, height);

    // 2. XML-defined inputs (in XML order)
    for (const auto &inputDef : xmlDef.getInputs())
    {
        const std::string &inputName = inputDef.name;

        // Create texture object for this input
        cudaTextureObject_t inputTex = 0;
        bool inputPresent = false;

        if (images.count(inputName) && images.at(inputName) != nullptr)
        {
            Logger::getInstance().logMessage("  Setting up texture for input: %s", inputName.c_str());

            // Create CUDA texture object
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
            cudaArray_t inputArray = NULL;

            cudaError_t err = cudaMallocArray(&inputArray, &channelDesc, width, height);
            if (err == cudaSuccess)
            {
                allocatedArrays.push_back(inputArray); // Track for cleanup

                err = cudaMemcpy2DToArray(inputArray, 0, 0, images.at(inputName), width * sizeof(float4),
                                          width * sizeof(float4), height, cudaMemcpyHostToDevice);

                if (err == cudaSuccess)
                {
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

                    err = cudaCreateTextureObject(&inputTex, &inputResDesc, &texDesc, NULL);
                    if (err == cudaSuccess)
                    {
                        createdTextures.push_back(inputTex); // Track for cleanup
                        inputPresent = true;
                        Logger::getInstance().logMessage("    ✓ Texture created for %s", inputName.c_str());
                    }
                }
            }

            if (err != cudaSuccess)
            {
                Logger::getInstance().logMessage("    ERROR: Texture creation failed for %s: %s",
                                                 inputName.c_str(), cudaGetErrorString(err));
            }
        }
        else
        {
            Logger::getInstance().logMessage("    - Input %s not available", inputName.c_str());
        }

        // Add texture object to kernel arguments
        kernelArgs.push_back(storeArg(inputTex));

        // Add presence boolean for optional inputs
        if (inputDef.optional)
        {
            kernelArgs.push_back(storeArg(inputPresent));
            Logger::getInstance().logMessage("    Added presence flag for optional input %s: %s",
                                             inputName.c_str(), inputPresent ? "true" : "false");
        }
    }
// we are having a "flickering" issue, where the matte seems to go away for a frame here and there.
//this cudaDeviceSynchronize is a band-aid, not a fix, as it is too agressive (makes the whole device finish, not the stream)
    // Synchronize immediately after all texture creation to ensure uploads complete
    cudaDeviceSynchronize(); 

    // 3. Output buffer
    float *output = images.count("output") ? images.at("output") : nullptr;
    if (!output)
    {
        Logger::getInstance().logMessage("ERROR: No output buffer found");
        cleanupCudaResources(createdTextures, allocatedArrays);
        return;
    }
    kernelArgs.push_back(storeArg(output));
    Logger::getInstance().logMessage("  Added output buffer: %p", output);

    // 4. XML-defined parameters (in XML order)
    for (const auto &paramDef : xmlDef.getParameters())
    {
        const std::string &paramName = paramDef.name;

        if (params.count(paramName))
        {
            const ParameterValue &value = params.at(paramName);

            if (paramDef.type == "double" || paramDef.type == "float")
            {
                kernelArgs.push_back(storeArg(value.asFloat()));
                Logger::getInstance().logMessage("    Added float parameter %s = %.3f", paramName.c_str(), value.asFloat());
            }
            else if (paramDef.type == "int" || paramDef.type == "choice")
            {
                kernelArgs.push_back(storeArg(value.asInt()));
                Logger::getInstance().logMessage("    Added int parameter %s = %d", paramName.c_str(), value.asInt());
            }
            else if (paramDef.type == "bool")
            {
                kernelArgs.push_back(storeArg(value.asBool()));
                Logger::getInstance().logMessage("    Added bool parameter %s = %s", paramName.c_str(), value.asBool() ? "true" : "false");
            }
            else
            {
                Logger::getInstance().logMessage("    WARNING: Unsupported parameter type %s for %s", paramDef.type.c_str(), paramName.c_str());
                kernelArgs.push_back(storeArg(0.0f));
            }
        }
        else
        {
            Logger::getInstance().logMessage("    WARNING: Parameter %s not found, using default", paramName.c_str());
            if (paramDef.type == "double" || paramDef.type == "float")
            {
                kernelArgs.push_back(storeArg(static_cast<float>(paramDef.defaultValue)));
            }
            else if (paramDef.type == "int" || paramDef.type == "choice")
            {
                kernelArgs.push_back(storeArg(static_cast<int>(paramDef.defaultValue)));
            }
            else if (paramDef.type == "bool")
            {
                kernelArgs.push_back(storeArg(paramDef.defaultBool));
            }
            else
            {
                kernelArgs.push_back(storeArg(0.0f));
            }
        }
    }

    Logger::getInstance().logMessage("  Built kernel argument list with %d arguments", (int)kernelArgs.size());

    // 5. Extract arguments and call kernel through registry
    void **args = kernelArgs.data();
    int argIndex = 0;

    int w = *static_cast<int *>(args[argIndex++]);
    int h = *static_cast<int *>(args[argIndex++]);

    std::vector<cudaTextureObject_t> textures;
    std::vector<bool> presenceFlags;

    for (const auto &inputDef : xmlDef.getInputs())
    {
        cudaTextureObject_t tex = *static_cast<cudaTextureObject_t *>(args[argIndex++]);
        textures.push_back(tex);

        if (inputDef.optional)
        {
            bool present = *static_cast<bool *>(args[argIndex++]);
            presenceFlags.push_back(present);
        }
    }

    float *out = *static_cast<float **>(args[argIndex++]);

    std::vector<float> floatParams;
    std::vector<int> intParams;

    for (const auto &paramDef : xmlDef.getParameters())
    {
        if (paramDef.type == "double" || paramDef.type == "float")
        {
            float val = *static_cast<float *>(args[argIndex++]);
            floatParams.push_back(val);
        }
        else if (paramDef.type == "int" || paramDef.type == "choice")
        {
            int val = *static_cast<int *>(args[argIndex++]);
            intParams.push_back(val);
        }
        else if (paramDef.type == "bool")
        {
            bool val = *static_cast<bool *>(args[argIndex++]);
            // Handle bool params if needed
        }
    }

    // Synchronize to ensure all texture uploads are complete before kernel execution
    // cudaStreamSynchronize(cudaStream);

    // Call kernel through registry
    std::string effectName = xmlDef.getName();
    Logger::getInstance().logMessage("  Looking up kernel function in registry for: %s", effectName.c_str());

    KernelFunction kernelFunc = getKernelFunction(effectName);
    if (kernelFunc)
    {
        Logger::getInstance().logMessage("  Found kernel function in registry");
        Logger::getInstance().logMessage("  Float params size: %d, Int params size: %d", (int)floatParams.size(), (int)intParams.size());

        kernelFunc(stream, w, h,
                   (void *)(uintptr_t)textures[0], (void *)(uintptr_t)textures[1], presenceFlags.empty() ? false : presenceFlags[0],
                   (void *)(uintptr_t)textures[2], presenceFlags.size() > 1 ? presenceFlags[1] : false,
                   out,
                   floatParams[0], floatParams[1], intParams[0], floatParams[2]);

        Logger::getInstance().logMessage("  ✓ Registry kernel call completed");
    }
    else
    {
        Logger::getInstance().logMessage("  ERROR: No kernel function found in registry for '%s'", effectName.c_str());
    }

    // Clean up GPU resources
    cleanupCudaResources(createdTextures, allocatedArrays);

    Logger::getInstance().logMessage("RunGenericCudaKernel completed");
#endif
}

void RunGenericOpenCLKernel(
    void *cmdQueue,
    int width,
    int height,
    const std::map<std::string, ParameterValue> &params,
    const std::map<std::string, float *> &images,
    const std::map<std::string, std::string> &borderModes,
    const XMLEffectDefinition &xmlDef)
{
    Logger::getInstance().logMessage("RunGenericOpenCLKernel called for effect: %s", xmlDef.getName().c_str());

    // For now, extract parameters for known effects (like TestBlurV2) and call original functions
    // TODO: Implement full XML-driven OpenCL dispatch like CUDA

    std::string effectName = xmlDef.getName();

    if (effectName == "TestBlurV2")
    {
        // Extract parameters for TestBlurV2 specifically
        float radius = params.count("radius") ? params.at("radius").asFloat() : 5.0f;
        int quality = params.count("quality") ? params.at("quality").asInt() : 8;
        float maskStrength = params.count("maskStrength") ? params.at("maskStrength").asFloat() : 1.0f;

        // Extract images
        const float *input = images.count("Source") ? images.at("Source") : nullptr;
        const float *mask = images.count("mask") ? images.at("mask") : nullptr;
        float *output = images.count("output") ? images.at("output") : nullptr;

        if (!input || !output)
        {
            Logger::getInstance().logMessage("ERROR: Missing required images for OpenCL");
            return;
        }

        // Call existing OpenCL function for now
        extern void RunOpenCLKernel(void *p_CmdQ, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength,
                                    const float *p_Input, const float *p_Mask, float *p_Output);

        RunOpenCLKernel(cmdQueue, width, height, radius, quality, maskStrength, input, mask, output);
        Logger::getInstance().logMessage("  ✓ OpenCL TestBlurV2 kernel completed");
    }
    else
    {
        Logger::getInstance().logMessage("  ERROR: OpenCL dispatch for '%s' not implemented yet", effectName.c_str());
    }

    Logger::getInstance().logMessage("RunGenericOpenCLKernel completed");
}

void RunGenericMetalKernel(
    void *cmdQueue,
    int width,
    int height,
    const std::map<std::string, ParameterValue> &params,
    const std::map<std::string, float *> &images,
    const std::map<std::string, std::string> &borderModes,
    const XMLEffectDefinition &xmlDef)
{
#ifdef __APPLE__
    Logger::getInstance().logMessage("RunGenericMetalKernel called for effect: %s", xmlDef.getName().c_str());

    // For now, extract parameters for known effects (like TestBlurV2) and call original functions
    // TODO: Implement full XML-driven Metal dispatch like CUDA

    std::string effectName = xmlDef.getName();

    if (effectName == "TestBlurV2")
    {
        // Extract parameters for TestBlurV2 specifically
        float radius = params.count("radius") ? params.at("radius").asFloat() : 5.0f;
        int quality = params.count("quality") ? params.at("quality").asInt() : 8;
        float maskStrength = params.count("maskStrength") ? params.at("maskStrength").asFloat() : 1.0f;

        // Extract images
        const float *input = images.count("Source") ? images.at("Source") : nullptr;
        const float *mask = images.count("mask") ? images.at("mask") : nullptr;
        float *output = images.count("output") ? images.at("output") : nullptr;

        if (!input || !output)
        {
            Logger::getInstance().logMessage("ERROR: Missing required images for Metal");
            return;
        }

        // Call existing Metal function for now
        extern void RunMetalKernel(void *p_CmdQ, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength,
                                   const float *p_Input, const float *p_Mask, float *p_Output);

        RunMetalKernel(cmdQueue, width, height, radius, quality, maskStrength, input, mask, output);
        Logger::getInstance().logMessage("  ✓ Metal TestBlurV2 kernel completed");
    }
    else
    {
        Logger::getInstance().logMessage("  ERROR: Metal dispatch for '%s' not implemented yet", effectName.c_str());
    }

    Logger::getInstance().logMessage("RunGenericMetalKernel completed");
#endif
}