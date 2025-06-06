#include "GenericProcessor.h"
#include "KernelWrappers.h"
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
    
    // Log the actual parameter values we received
    for (const auto& param : params) {
        Logger::getInstance().logMessage("  Parameter %s = %s (type: %s)", 
                                       param.first.c_str(), 
                                       param.second.asString().c_str(),
                                       param.second.getType().c_str());
    }
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
    
    // Get image dimensions from output image
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
    
    // Build complete parameter map from XML
    Logger::getInstance().logMessage("Building parameter map from XML:");
    std::map<std::string, ParameterValue> allParams;
    for (const auto& paramDef : m_xmlDef.getParameters()) {
        if (m_paramValues.count(paramDef.name)) {
            allParams[paramDef.name] = m_paramValues.at(paramDef.name);
            Logger::getInstance().logMessage("  - %s = %s", paramDef.name.c_str(), 
                                           allParams[paramDef.name].asString().c_str());
        } else {
            // Use XML default if parameter not provided
            if (paramDef.type == "double" || paramDef.type == "float") {
                allParams[paramDef.name] = ParameterValue(paramDef.defaultValue);
            } else if (paramDef.type == "int") {
                allParams[paramDef.name] = ParameterValue((int)paramDef.defaultValue);
            } else if (paramDef.type == "bool") {
                allParams[paramDef.name] = ParameterValue(paramDef.defaultBool);
            }
            Logger::getInstance().logMessage("  - %s = %s (XML default)", paramDef.name.c_str(), 
                                           allParams[paramDef.name].asString().c_str());
        }
    }
    
    // Build complete image map from XML
    Logger::getInstance().logMessage("Building image map from XML:");
    std::map<std::string, float*> allImages;
    for (const auto& inputDef : m_xmlDef.getInputs()) {
        if (m_images.count(inputDef.name)) {
            allImages[inputDef.name] = static_cast<float*>(m_images.at(inputDef.name)->getPixelData());
            Logger::getInstance().logMessage("  - %s: %p", inputDef.name.c_str(), allImages[inputDef.name]);
        } else {
            if (!inputDef.optional) {
                Logger::getInstance().logMessage("ERROR: Required input %s not found", inputDef.name.c_str());
                return;
            }
            allImages[inputDef.name] = nullptr;
            Logger::getInstance().logMessage("  - %s: null (optional)", inputDef.name.c_str());
        }
    }
    
    // Add output image
    float* output = static_cast<float*>(_dstImg->getPixelData());
    allImages["output"] = output;
    Logger::getInstance().logMessage("  - output: %p", output);
    
    // Build border mode map from XML
    Logger::getInstance().logMessage("Building border mode map from XML:");
    std::map<std::string, std::string> borderModes;
    for (const auto& inputDef : m_xmlDef.getInputs()) {
        borderModes[inputDef.name] = inputDef.borderMode;
        Logger::getInstance().logMessage("  - %s: %s", inputDef.name.c_str(), inputDef.borderMode.c_str());
    }
    
    // Call the generalized kernel wrapper for each platform
    Logger::getInstance().logMessage("Calling generalized kernel wrapper:");
    Logger::getInstance().logMessage("  Parameters: %d items", (int)allParams.size());
    Logger::getInstance().logMessage("  Images: %d items", (int)allImages.size());
    Logger::getInstance().logMessage("  Border modes: %d items", (int)borderModes.size());
    
    if (platform == "cuda") {
#ifndef __APPLE__
        RunGenericCudaKernel(_pCudaStream, width, height, allParams, allImages, borderModes);
#endif
    }
    else if (platform == "opencl") {
        RunGenericOpenCLKernel(_pOpenCLCmdQ, width, height, allParams, allImages, borderModes);
    }
    else if (platform == "metal") {
#ifdef __APPLE__
        RunGenericMetalKernel(_pMetalCmdQ, width, height, allParams, allImages, borderModes);
#endif
    }
    
    Logger::getInstance().logMessage("Generalized kernel execution completed");
}