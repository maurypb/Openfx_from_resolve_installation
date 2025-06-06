#ifndef KERNEL_WRAPPERS_H
#define KERNEL_WRAPPERS_H

#include "ParameterValue.h"
#include <map>
#include <string>

/**
 * @brief Generic CUDA kernel wrapper
 * @param stream CUDA stream
 * @param width Image width
 * @param height Image height
 * @param params All parameters from XML as ParameterValue map
 * @param images All images as float* map (keyed by XML input names)
 * @param borderModes Border mode strings for each input (from XML)
 */
void RunGenericCudaKernel(
    void* stream,
    int width,
    int height,
    const std::map<std::string, ParameterValue>& params,
    const std::map<std::string, float*>& images,
    const std::map<std::string, std::string>& borderModes
);

/**
 * @brief Generic OpenCL kernel wrapper
 * @param cmdQueue OpenCL command queue
 * @param width Image width
 * @param height Image height
 * @param params All parameters from XML as ParameterValue map
 * @param images All images as float* map (keyed by XML input names)
 * @param borderModes Border mode strings for each input (from XML)
 */
void RunGenericOpenCLKernel(
    void* cmdQueue,
    int width,
    int height,
    const std::map<std::string, ParameterValue>& params,
    const std::map<std::string, float*>& images,
    const std::map<std::string, std::string>& borderModes
);

/**
 * @brief Generic Metal kernel wrapper
 * @param cmdQueue Metal command queue
 * @param width Image width
 * @param height Image height
 * @param params All parameters from XML as ParameterValue map
 * @param images All images as float* map (keyed by XML input names)
 * @param borderModes Border mode strings for each input (from XML)
 */
void RunGenericMetalKernel(
    void* cmdQueue,
    int width,
    int height,
    const std::map<std::string, ParameterValue>& params,
    const std::map<std::string, float*>& images,
    const std::map<std::string, std::string>& borderModes
);

#endif // KERNEL_WRAPPERS_H