#ifndef GENERIC_PROCESSOR_H
#define GENERIC_PROCESSOR_H

#include "ofxsImageEffect.h"
#include "ofxsProcessing.h"
#include "XMLEffectDefinition.h"
#include "ParameterValue.h"
#include <string>
#include <map>
#include <memory>

/**
 * @class GenericProcessor
 * @brief Dynamic image processor that replaces fixed processors like ImageBlurrer
 * 
 * This processor can handle ANY XML effect definition by dynamically accepting
 * parameters and images, then calling appropriate GPU kernels.
 */
class GenericProcessor : public OFX::ImageProcessor {
private:
    const XMLEffectDefinition& m_xmlDef;
    
    // Dynamic storage for images and parameters
    std::map<std::string, OFX::Image*> m_images;           // Raw pointers (borrowed)
    std::map<std::string, ParameterValue> m_paramValues;   // Extracted values

public:
    /**
     * @brief Constructor
     * @param effect The parent effect instance
     * @param xmlDef The XML effect definition
     */
    GenericProcessor(OFX::ImageEffect& effect, const XMLEffectDefinition& xmlDef);

    /**
     * @brief Set images for processing (borrowed pointers)
     * @param images Map of image name to image pointer
     */
    void setImages(const std::map<std::string, OFX::Image*>& images);

    /**
     * @brief Set parameter values for processing
     * @param params Map of parameter name to value
     */
    void setParameters(const std::map<std::string, ParameterValue>& params);

    // GPU processing methods (override from ImageProcessor)
    virtual void processImagesCUDA() override;
    virtual void processImagesOpenCL() override;
    virtual void processImagesMetal() override;
    
    // CPU fallback
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow) override;

private:
    /**
     * @brief Call dynamic kernel for specific platform
     * @param platform Platform name ("cuda", "opencl", "metal")
     */
    void callDynamicKernel(const std::string& platform);
};

#endif // GENERIC_PROCESSOR_H