#ifndef KERNEL_WRAPPERS_H
#define KERNEL_WRAPPERS_H

#include "ParameterValue.h"
#include "XMLEffectDefinition.h"
#include <map>
#include <string>

/**
 * @brief Generic CUDA kernel wrapper - XML-driven
 * @param stream CUDA stream
 * @param width Image width
 * @param height Image height
 * @param params All parameters from XML as ParameterValue map
 * @param images All images as float* map (keyed by XML input names)
 * @param borderModes Border mode strings for each input (from XML)
 * @param xmlDef XML effect definition for dynamic dispatch
 */
void RunGenericCudaKernel(
    void* stream,
    int width,
    int height,
    const std::map<std::string, ParameterValue>& params,
    const std::map<std::string, float*>& images,
    const std::map<std::string, std::string>& borderModes,
    const XMLEffectDefinition& xmlDef
);

/**
 * @brief Generic OpenCL kernel wrapper - XML-driven
 * @param cmdQueue OpenCL command queue
 * @param width Image width
 * @param height Image height
 * @param params All parameters from XML as ParameterValue map
 * @param images All images as float* map (keyed by XML input names)
 * @param borderModes Border mode strings for each input (from XML)
 * @param xmlDef XML effect definition for dynamic dispatch
 */
void RunGenericOpenCLKernel(
    void* cmdQueue,
    int width,
    int height,
    const std::map<std::string, ParameterValue>& params,
    const std::map<std::string, float*>& images,
    const std::map<std::string, std::string>& borderModes,
    const XMLEffectDefinition& xmlDef
);

/**
 * @brief Generic Metal kernel wrapper - XML-driven
 * @param cmdQueue Metal command queue
 * @param width Image width
 * @param height Image height
 * @param params All parameters from XML as ParameterValue map
 * @param images All images as float* map (keyed by XML input names)
 * @param borderModes Border mode strings for each input (from XML)
 * @param xmlDef XML effect definition for dynamic dispatch
 */
void RunGenericMetalKernel(
    void* cmdQueue,
    int width,
    int height,
    const std::map<std::string, ParameterValue>& params,
    const std::map<std::string, float*>& images,
    const std::map<std::string, std::string>& borderModes,
    const XMLEffectDefinition& xmlDef
);

#endif // KERNEL_WRAPPERS_H