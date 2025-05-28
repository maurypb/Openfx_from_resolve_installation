#include "GenericProcessor.h"
#include "../../Logger.h"
#include <cstring>
#include "ParameterValue.h"

// External kernel function declarations (same pattern as BlurPlugin)
#ifndef __APPLE__
extern void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                         const float* p_Input, const float* p_Mask, float* p_Output);
#endif

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                          const float* p_Input, const float* p_Mask, float* p_Output);
#endif

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                           const float* p_Input, const float* p_Mask, float* p_Output);

GenericProcessor::GenericProcessor(OFX::ImageEffect& effect, const XMLEffectDefinition& xmlDef)
    : OFX::ImageProcessor(effect), m_xmlDef(xmlDef) {
    
    Logger::getInstance().logMessage("GenericProcessor created for effect: %s", xmlDef.getName().c_str());
}

void GenericProcessor::setImages(const std::map<std::string, OFX::Image*>& images) {
    m_images = images;
    
    // Set the required _dstImg for base class (same pattern as ImageBlurrer)
    auto outputIt = images.find("output");
    if (outputIt != images.end()) {
        _dstImg = outputIt->second;
    }
    
    Logger::getInstance().logMessage("GenericProcessor: Set %d images", (int)images.size());
}

void GenericProcessor::setParameters(const std::map<std::string, ParameterValue>& params) {
    m_paramValues = params;
    Logger::getInstance().logMessage("GenericProcessor: Set %d parameters", (int)params.size());
}

void GenericProcessor::processImagesCUDA() {
#ifndef __APPLE__
    Logger::getInstance().logMessage("GenericProcessor::processImagesCUDA called");
    callDynamicKernel("cuda");
#endif
}

void GenericProcessor::processImagesOpenCL() {
    Logger::getInstance().logMessage("GenericProcessor::processImagesOpenCL called");
    callDynamicKernel("opencl");
}

void GenericProcessor::processImagesMetal() {
#ifdef __APPLE__
    Logger::getInstance().logMessage("GenericProcessor::processImagesMetal called");
    callDynamicKernel("metal");
#endif
}

void GenericProcessor::multiThreadProcessImages(OfxRectI p_ProcWindow) {
    Logger::getInstance().logMessage("GenericProcessor::multiThreadProcessImages called (CPU fallback)");
    
    // Simple CPU fallback - just copy source to output
    OFX::Image* srcImg = nullptr;
    auto srcIt = m_images.find("source");
    if (srcIt != m_images.end()) {
        srcImg = srcIt->second;
    }
    
    if (srcImg && _dstImg) {
        // Copy source to destination (same pattern as ImageBlurrer)
        for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y) {
            if (_effect.abort()) break;

            float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
            float* srcPix = static_cast<float*>(srcImg->getPixelAddress(p_ProcWindow.x1, y));

            if (srcPix && dstPix) {
                for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x) {
                    for (int c = 0; c < 4; ++c) {
                        dstPix[c] = srcPix[c];  // Simple copy
                    }
                    dstPix += 4;
                    srcPix += 4;
                }
            }
        }
        
        Logger::getInstance().logMessage("CPU fallback processing completed");
    } else {
        Logger::getInstance().logMessage("ERROR: Missing source or destination image for CPU processing");
    }
}

void GenericProcessor::callDynamicKernel(const std::string& platform) {
    Logger::getInstance().logMessage("GenericProcessor::callDynamicKernel called for platform: %s", platform.c_str());
    
    // Get image dimensions from output image (same pattern as ImageBlurrer)
    if (!_dstImg) {
        Logger::getInstance().logMessage("ERROR: No destination image");
        return;
    }
    
    const OfxRectI& bounds = _dstImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    
    Logger::getInstance().logMessage("Processing %dx%d image", width, height);
    
    // Get kernels for this platform from XML
    auto kernels = m_xmlDef.getKernelsForPlatform(platform);
    if (kernels.empty()) {
        Logger::getInstance().logMessage("No %s kernel defined in XML", platform.c_str());
        return;
    }
    
    Logger::getInstance().logMessage("Found %s kernel: %s", platform.c_str(), kernels[0].file.c_str());
    
    // // Build parameter map dynamically from XML (no hardcoded names!)
    // std::map<std::string, ParameterValue> finalParams;
    
    // for (const auto& paramDef : m_xmlDef.getParameters()) {
    //     if (m_paramValues.count(paramDef.name)) {
    //         Logger::getInstance().logMessage("  - Using actual value");
    //         // Use actual parameter value
    //         finalParams[paramDef.name] = m_paramValues.at(paramDef.name);
    //     } else {
    //         Logger::getInstance().logMessage("  - Using XML default: %.2f", paramDef.defaultValue);
    //         if (paramDef.type == "double" || paramDef.type == "float") {
    //             Logger::getInstance().logMessage("  - About to create ParameterValue...");
                
    //             // Test if the defaultValue is valid
    //             double testValue = paramDef.defaultValue;
    //             Logger::getInstance().logMessage("  - Copied defaultValue: %.2f", testValue);
                
    //             // Test creating ParameterValue with a known good value first
    //             ParameterValue goodValue(5.0);
    //             Logger::getInstance().logMessage("  - Created ParameterValue with literal 5.0");
                
    //             // Now try with the XML value
    //             ParameterValue xmlValue(testValue);
    //             Logger::getInstance().logMessage("  - Created ParameterValue with XML value");
    //         } else if (paramDef.type == "int") {
    //             finalParams[paramDef.name] = ParameterValue((int)paramDef.defaultValue);
    //             Logger::getInstance().logMessage("  - Created int ParameterValue successfully");
    //         } else if (paramDef.type == "bool") {
    //             finalParams[paramDef.name] = ParameterValue(paramDef.defaultBool);
    //             Logger::getInstance().logMessage("  - Created bool ParameterValue successfully");
    //         }
    //     }
        
    //     Logger::getInstance().logMessage("Parameter %s (%s) = %s", 
    //                                    paramDef.name.c_str(), 
    //                                    paramDef.type.c_str(),
    //                                    finalParams[paramDef.name].asString().c_str());
    // }
    
    // // Build image map dynamically from XML (no hardcoded names!)
    // std::map<std::string, float*> imagePointers;
    
    // for (const auto& inputDef : m_xmlDef.getInputs()) {
    //     if (m_images.count(inputDef.name)) {
    //         imagePointers[inputDef.name] = static_cast<float*>(m_images.at(inputDef.name)->getPixelData());
    //         Logger::getInstance().logMessage("Found XML input: %s", inputDef.name.c_str());
    //     } else if (!inputDef.optional) {
    //         Logger::getInstance().logMessage("ERROR: Required input %s not found", inputDef.name.c_str());
    //         return;
    //     }
    // }
    
    // // Add output image
    // imagePointers["output"] = static_cast<float*>(_dstImg->getPixelData());
    
    // // For now, we still need to extract specific parameters for RunCudaKernel signature
    // // TODO: Phase 4 will make this completely dynamic
    // Logger::getInstance().logMessage("Extracting parameters for kernel call...");
    // Logger::getInstance().logMessage("Available parameters: %d", (int)finalParams.size());
    
    // float radius = finalParams.count("radius") ? finalParams["radius"].asFloat() : 5.0f;
    // int quality = finalParams.count("quality") ? finalParams["quality"].asInt() : 8;  
    // float maskStrength = finalParams.count("maskStrength") ? finalParams["maskStrength"].asFloat() : 1.0f;
    
    // Logger::getInstance().logMessage("Using defaults: radius=%.2f, quality=%d, maskStrength=%.2f", radius, quality, maskStrength);
    
    // float* input = imagePointers.count("source") ? imagePointers["source"] : nullptr;
    // float* mask = imagePointers.count("mask") ? imagePointers["mask"] : nullptr;
    // float* output = imagePointers["output"];
    
    // if (!input || !output) {
    //     Logger::getInstance().logMessage("ERROR: Missing input or output image data");
    //     return;
    // }
    
    Logger::getInstance().logMessage("Skipping parameter map building, using hardcoded defaults");

    // Get image pointers directly
    float* input = nullptr;
    float* mask = nullptr;
    float* output = static_cast<float*>(_dstImg->getPixelData());
    
    if (m_images.count("source")) {
        input = static_cast<float*>(m_images.at("source")->getPixelData());
        Logger::getInstance().logMessage("Found source image");
    }
    
    if (m_images.count("mask")) {
        mask = static_cast<float*>(m_images.at("mask")->getPixelData());
        Logger::getInstance().logMessage("Found mask image");
    }
    
    if (!input || !output) {
        Logger::getInstance().logMessage("ERROR: Missing input or output image data");
        return;
    }
    
    // Use hardcoded parameters
    float radius = 10.0f;
    int quality = 8;
    float maskStrength = 1.0f;
    
    Logger::getInstance().logMessage("Using hardcoded: radius=%.2f, quality=%d, maskStrength=%.2f", radius, quality, maskStrength);
    






    // Call the appropriate kernel (still hardcoded for now - Phase 4 will make this dynamic)
    if (platform == "cuda") {
#ifndef __APPLE__
        RunCudaKernel(_pCudaStream, width, height, radius, quality, maskStrength, input, mask, output);
#endif
    }
    else if (platform == "opencl") {
        RunOpenCLKernel(_pOpenCLCmdQ, width, height, radius, quality, maskStrength, input, mask, output);
    }
    else if (platform == "metal") {
#ifdef __APPLE__
        RunMetalKernel(_pMetalCmdQ, width, height, radius, quality, maskStrength, input, mask, output);
#endif
    }
    
    Logger::getInstance().logMessage("Kernel execution completed");
}