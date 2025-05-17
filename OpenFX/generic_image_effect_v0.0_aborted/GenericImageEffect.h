#pragma once

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "Logger.h"

// Forward declarations
namespace OFX {
    class Clip;
    class Image;
}

// Generic base class for image processors
class GenericImageProcessor {
public:
    explicit GenericImageProcessor(OFX::ImageEffect& p_Instance);
    virtual ~GenericImageProcessor() = default;

    // Virtual GPU processing methods that derived classes can override
    virtual void processImagesCUDA() {}
    virtual void processImagesOpenCL() {}
    virtual void processImagesMetal() {}
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow) {}
    
    // Method to check if the processor supports specific GPU APIs
    virtual bool supportsOpenCL() const;
    virtual bool supportsCUDA() const;
    virtual bool supportsMetal() const;

    // Common setup methods
    void setDstImg(OFX::Image* p_DstImg) { _dstImg = p_DstImg; }
    void setSrcImg(OFX::Image* p_SrcImg) { _srcImg = p_SrcImg; }
    void setMaskImg(OFX::Image* p_MaskImg) { _maskImg = p_MaskImg; }
    void setRenderWindow(OfxRectI p_RenderWindow) { _renderWindow = p_RenderWindow; }
    void setGPURenderArgs(const OFX::RenderArguments& p_Args);
    void process();
    
    // Helper method for CPU processing
    void processCPU();

protected:
    OFX::ImageEffect& _effect;
    OFX::Image* _dstImg;
    OFX::Image* _srcImg;
    OFX::Image* _maskImg;
    OfxRectI _renderWindow;

    // GPU render data
    void* _pOpenCLCmdQ;    // OpenCL command queue
    void* _pCudaStream;    // CUDA stream
    void* _pMetalCmdQ;     // Metal command queue
};

// Generic base class for OFX image effect plugins
template <typename ParameterClass, typename ProcessorClass>
class GenericImageEffect : public OFX::ImageEffect {
public:
    explicit GenericImageEffect(OfxImageEffectHandle p_Handle);
    virtual ~GenericImageEffect();

    // Virtual methods that must be implemented by derived classes
    virtual void setupProcessor(ProcessorClass& processor, const OFX::RenderArguments& args) = 0;
    
    // Core OFX override methods with default implementations
    virtual void render(const OFX::RenderArguments& p_Args);
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName) {}
    virtual void changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName) {}

protected:
    // Utility method for setting up and running a processor
    void setupAndProcess(ProcessorClass& p_Processor, const OFX::RenderArguments& p_Args);

    // Common clips all effects have
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;
    OFX::Clip* m_MaskClip;
};

// Template implementation

template <typename ParameterClass, typename ProcessorClass>
GenericImageEffect<ParameterClass, ProcessorClass>::GenericImageEffect(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    // Fetch standard clips
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    
    // Mask clip may be optional
    try {
        m_MaskClip = fetchClip("Mask");
    } catch (...) {
        m_MaskClip = nullptr;
    }

    // Log instance creation
    Logger::getInstance().logMessage("GenericImageEffect instance created");
}

template <typename ParameterClass, typename ProcessorClass>
GenericImageEffect<ParameterClass, ProcessorClass>::~GenericImageEffect()
{
    // Log instance destruction
    Logger::getInstance().logMessage("GenericImageEffect instance destroyed");
}

template <typename ParameterClass, typename ProcessorClass>
void GenericImageEffect<ParameterClass, ProcessorClass>::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && 
        (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ProcessorClass processor(*this);
        setupAndProcess(processor, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

template <typename ParameterClass, typename ProcessorClass>
bool GenericImageEffect<ParameterClass, ProcessorClass>::isIdentity(
    const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    // Default implementation - derived classes should override if they support identity
    return false;
}

template <typename ParameterClass, typename ProcessorClass>
void GenericImageEffect<ParameterClass, ProcessorClass>::setupAndProcess(
    ProcessorClass& p_Processor, const OFX::RenderArguments& p_Args)
{
    // Get the dst image
    std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    // Get the src image
    std::unique_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    // Check to see if the bit depth and number of components are the same
    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }

    // Get the mask image if it exists
    std::unique_ptr<OFX::Image> mask;
    if (m_MaskClip && m_MaskClip->isConnected())
    {
        mask.reset(m_MaskClip->fetchImage(p_Args.time));
    }

    // Set the images
    p_Processor.setDstImg(dst.get());
    p_Processor.setSrcImg(src.get());
    p_Processor.setMaskImg(mask.get());

    // Setup OpenCL and CUDA Render arguments
    p_Processor.setGPURenderArgs(p_Args);

    // Set the render window
    p_Processor.setRenderWindow(p_Args.renderWindow);

    // Set effect-specific parameters
    setupProcessor(p_Processor, p_Args);

    // Call the base class process member
    p_Processor.process();
}