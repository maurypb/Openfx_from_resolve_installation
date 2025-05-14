#pragma once

#include "GenericImageEffect.h"
#include "ofxsImageEffect.h"
#include "BlurPluginParameters.h"

// Forward declaration
class ImageBlurrer;

class BlurPlugin : public GenericImageEffect<BlurPluginParameters, ImageBlurrer> {
public:
    explicit BlurPlugin(OfxImageEffectHandle p_Handle);
    virtual ~BlurPlugin();
    
    // Override required methods from GenericImageEffect
    virtual void setupProcessor(ImageBlurrer& processor, const OFX::RenderArguments& args) override;
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime) override;
    
    // Static info methods for the generic factory
    static const char* getName() { return "GaussianBlur"; }
    static const char* getGrouping() { return "Filter"; }
    static const char* getDescription() { return "Apply Gaussian blur with optional mask control"; }
    static bool supportsTiles() { return false; }
    static bool supportsMultiResolution() { return false; }
    static bool supportsMultipleClipPARs() { return false; }
    static bool usesMask() { return true; }
    static bool hasAdvancedParameters() { return false; }
    static bool supportsOpenCL() { return true; }
    static bool supportsCuda() { return true; }
    static bool supportsCudaStream() { return true; }
    static bool supportsMetal() { return true; }

private:
    // Parameters
    OFX::DoubleParam* m_Radius;
    OFX::IntParam* m_Quality;
    OFX::DoubleParam* m_MaskStrength;
};

// Image processor specific to blur effect
class ImageBlurrer : public GenericImageProcessor {
public:
    explicit ImageBlurrer(OFX::ImageEffect& p_Instance);

    // Override GPU processing methods
    virtual void processImagesCUDA() override;
    virtual void processImagesOpenCL() override;
    virtual void processImagesMetal() override;
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow) override;
    
    // Override support methods
    virtual bool supportsOpenCL() const override;
    virtual bool supportsCUDA() const override;
    virtual bool supportsMetal() const override;

    // Set blur-specific parameters
    void setParams(float p_Radius, int p_Quality, float p_MaskStrength);

private:
    float _radius;
    int _quality;
    float _maskStrength;
};