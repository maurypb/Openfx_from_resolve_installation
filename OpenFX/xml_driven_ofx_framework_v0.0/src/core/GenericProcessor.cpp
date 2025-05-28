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
    
    // Extract parameters from the actual parameter values (not hardcoded!)
    float radius = m_paramValues.count("radius") ? m_paramValues.at("radius").asFloat() : 5.0f;
    int quality = m_paramValues.count("quality") ? m_paramValues.at("quality").asInt() : 8;  
    float maskStrength = m_paramValues.count("maskStrength") ? m_paramValues.at("maskStrength").asFloat() : 1.0f;
    
    Logger::getInstance().logMessage("Using REAL parameters: radius=%.2f, quality=%d, maskStrength=%.2f", 
                                   radius, quality, maskStrength);
    
    // Get image pointers DYNAMICALLY from XML inputs
    float* input = nullptr;
    float* mask = nullptr;
    float* output = static_cast<float*>(_dstImg->getPixelData());
    
    Logger::getInstance().logMessage("Available images in processor:");
    for (const auto& imagePair : m_images) {
        Logger::getInstance().logMessage("  - '%s': %p", imagePair.first.c_str(), imagePair.second);
    }
    
    // Find the first non-optional input as the main source
    for (const auto& inputDef : m_xmlDef.getInputs()) {
        if (!inputDef.optional && m_images.count(inputDef.name)) {
            input = static_cast<float*>(m_images.at(inputDef.name)->getPixelData());
            Logger::getInstance().logMessage("Found main source image: %s", inputDef.name.c_str());
            break;
        }
    }
    
    // Find mask input (look for inputs with "mask" in the name)
    for (const auto& inputDef : m_xmlDef.getInputs()) {
        std::string lowerName = inputDef.name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
        
        if (lowerName.find("mask") != std::string::npos && m_images.count(inputDef.name)) {
            mask = static_cast<float*>(m_images.at(inputDef.name)->getPixelData());
            Logger::getInstance().logMessage("Found mask image: %s", inputDef.name.c_str());
            break;
        }
    }
    
    if (!input || !output) {
        Logger::getInstance().logMessage("ERROR: Missing input (%p) or output (%p) image data", input, output);
        return;
    }
    
    // Call the appropriate kernel with REAL parameters
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