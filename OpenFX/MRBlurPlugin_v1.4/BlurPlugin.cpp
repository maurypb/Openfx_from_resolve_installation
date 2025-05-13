#include "BlurPlugin.h"

#include <stdio.h>

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"
#include "ofxDrawSuite.h"
#include "ofxsSupportPrivate.h"

#define kPluginName "GaussianBlur"
#define kPluginGrouping "Filter"
#define kPluginDescription "Apply Gaussian blur with optional mask control"
#define kPluginIdentifier "com.Maury.GaussianBlur"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 2

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

class ImageBlurrer : public OFX::ImageProcessor
{
public:
    explicit ImageBlurrer(OFX::ImageEffect& p_Instance);

    virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void processImagesMetal();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setMaskImg(OFX::Image* p_MaskImg);
    void setParams(float p_Radius, int p_Quality, float p_MaskStrength);

private:
    OFX::Image* _srcImg;
    OFX::Image* _maskImg;
    float _radius;
    int _quality;
    float _maskStrength;
};

ImageBlurrer::ImageBlurrer(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
    _maskImg = nullptr;
    _radius = 5.0f;
    _quality = 8;
    _maskStrength = 1.0f;
}

#ifndef __APPLE__
extern void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                         const float* p_Input, const float* p_Mask, float* p_Output);
#endif

void ImageBlurrer::processImagesCUDA()
{
#ifndef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());
    
    // Get mask data if available
    float* mask = nullptr;
    if (_maskImg) {
        mask = static_cast<float*>(_maskImg->getPixelData());
    }

    RunCudaKernel(_pCudaStream, width, height, _radius, _quality, _maskStrength, input, mask, output);
#endif
}

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                          const float* p_Input, const float* p_Mask, float* p_Output);
#endif

void ImageBlurrer::processImagesMetal()
{
#ifdef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());
    
    // Get mask data if available
    float* mask = nullptr;
    if (_maskImg) {
        mask = static_cast<float*>(_maskImg->getPixelData());
    }

    RunMetalKernel(_pMetalCmdQ, width, height, _radius, _quality, _maskStrength, input, mask, output);
#endif
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                           const float* p_Input, const float* p_Mask, float* p_Output);

void ImageBlurrer::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());
    
    // Get mask data if available
    float* mask = nullptr;
    if (_maskImg) {
        mask = static_cast<float*>(_maskImg->getPixelData());
    }

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _radius, _quality, _maskStrength, input, mask, output);
}

void ImageBlurrer::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    // This is a simplified fallback for when GPU processing isn't available
    // A proper Gaussian blur implementation would be much more complex
    
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            // Just copy source to destination as a fallback
            if (srcPix)
            {
                for(int c = 0; c < 4; ++c)
                {
                    dstPix[c] = srcPix[c];  // Simple copy instead of blur
                }
            }
            else
            {
                // no src pixel here, be black and transparent
                for (int c = 0; c < 4; ++c)
                {
                    dstPix[c] = 0;
                }
            }

            // increment the dst pixel
            dstPix += 4;
        }
    }
    
    // Note: A proper CPU-based Gaussian blur would need to:
    // 1. Consider the blur radius parameter
    // 2. Sample multiple pixels in a radius around each target pixel
    // 3. Apply Gaussian weighting to the samples
    // 4. Handle the mask if present
}

void ImageBlurrer::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void ImageBlurrer::setMaskImg(OFX::Image* p_MaskImg)
{
    _maskImg = p_MaskImg;
}

void ImageBlurrer::setParams(float p_Radius, int p_Quality, float p_MaskStrength)
{
    _radius = p_Radius;
    _quality = p_Quality;
    _maskStrength = p_MaskStrength;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class BlurPlugin : public OFX::ImageEffect
{
public:
    explicit BlurPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Override changed clip */
    virtual void changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName);

    /* Set up and run a processor */
    void setupAndProcess(ImageBlurrer &p_ImageBlurrer, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;
    OFX::Clip* m_MaskClip;

    OFX::DoubleParam* m_Radius;
    OFX::IntParam* m_Quality;
    OFX::DoubleParam* m_MaskStrength;
};

BlurPlugin::BlurPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    m_MaskClip = fetchClip("Mask");

    m_Radius = fetchDoubleParam("radius");
    m_Quality = fetchIntParam("quality");
    m_MaskStrength = fetchDoubleParam("maskStrength");
}

void BlurPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ImageBlurrer imageBlurrer(*this);
        setupAndProcess(imageBlurrer, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool BlurPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    double radius = m_Radius->getValueAtTime(p_Args.time);

    if (radius <= 0.0)
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void BlurPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    // Handle any parameter changes here if needed
}

void BlurPlugin::changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName)
{
    // Handle any clip changes here if needed
}

void BlurPlugin::setupAndProcess(ImageBlurrer &p_ImageBlurrer, const OFX::RenderArguments& p_Args)
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

    // Get blur parameters
    double radius = m_Radius->getValueAtTime(p_Args.time);
    int quality = m_Quality->getValueAtTime(p_Args.time);
    double maskStrength = m_MaskStrength->getValueAtTime(p_Args.time);

    // Set the images
    p_ImageBlurrer.setDstImg(dst.get());
    p_ImageBlurrer.setSrcImg(src.get());
    p_ImageBlurrer.setMaskImg(mask.get());

    // Setup OpenCL and CUDA Render arguments
    p_ImageBlurrer.setGPURenderArgs(p_Args);

    // Set the render window
    p_ImageBlurrer.setRenderWindow(p_Args.renderWindow);

    // Set the parameters
    p_ImageBlurrer.setParams(radius, quality, maskStrength);

    // Call the base class process member, this will call the derived templated process code
    p_ImageBlurrer.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

BlurPluginFactory::BlurPluginFactory()
    : OFX::PluginFactoryHelper<BlurPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void BlurPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    // Basic labels
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    // Add the supported contexts, only filter at the moment
    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);

    // Add supported pixel depths
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    // Set a few flags
    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    // Setup OpenCL render capability flags
    p_Desc.setSupportsOpenCLRender(true);

    // Setup CUDA render capability flags on non-Apple system
#ifndef __APPLE__
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(true);
#endif

    // Setup Metal render capability flags only on Apple system
#ifdef __APPLE__
    p_Desc.setSupportsMetalRender(true);
#endif

    // Indicates that the plugin output does not depend on location or neighbours of a given pixel.
    // Therefore, this plugin could be executed during LUT generation.
    // MR:  this should be false
    p_Desc.setNoSpatialAwareness(false);

}

void BlurPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Source clip only in the filter context
    // Create the mandated source clip
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Create the optional mask clip
    ClipDescriptor* maskClip = p_Desc.defineClip("Mask");
    maskClip->addSupportedComponent(ePixelComponentRGBA);
    maskClip->addSupportedComponent(ePixelComponentAlpha);
    maskClip->setTemporalClipAccess(false);
    maskClip->setSupportsTiles(kSupportsTiles);
    maskClip->setOptional(true);
    maskClip->setIsMask(true);

    // Create the mandated output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->setSupportsTiles(kSupportsTiles);

    // Make some pages and to things in
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // Blur radius parameter
    DoubleParamDescriptor* radiusParam = p_Desc.defineDoubleParam("radius");
    radiusParam->setLabels("Radius", "Radius", "Radius");
    radiusParam->setHint("Blur radius in pixels");
    radiusParam->setRange(0.0, 100.0);
    radiusParam->setDisplayRange(0.0, 50.0);
    radiusParam->setDefault(5.0);
    page->addChild(*radiusParam);

    // Quality parameter
    IntParamDescriptor* qualityParam = p_Desc.defineIntParam("quality");
    qualityParam->setLabels("Quality", "Quality", "Quality");
    qualityParam->setHint("Number of samples for the blur");
    qualityParam->setRange(1, 32);
    qualityParam->setDisplayRange(1, 16);
    qualityParam->setDefault(8);
    page->addChild(*qualityParam);

    // Mask strength parameter
    DoubleParamDescriptor* maskStrengthParam = p_Desc.defineDoubleParam("maskStrength");
    maskStrengthParam->setLabels("Mask Strength", "Mask Str", "Mask");
    maskStrengthParam->setHint("How strongly the mask affects the blur radius");
    maskStrengthParam->setRange(0.0, 1.0);
    maskStrengthParam->setDisplayRange(0.0, 1.0);
    maskStrengthParam->setDefault(1.0);
    page->addChild(*maskStrengthParam);
}

ImageEffect* BlurPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new BlurPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static BlurPluginFactory blurPlugin;
    p_FactoryArray.push_back(&blurPlugin);
}