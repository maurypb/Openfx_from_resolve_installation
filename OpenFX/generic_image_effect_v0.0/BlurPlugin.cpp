#include "BlurPlugin.h"
#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "Logger.h"

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

// ImageBlurrer implementation
ImageBlurrer::ImageBlurrer(OFX::ImageEffect& p_Instance)
    : GenericImageProcessor(p_Instance)
{
    _radius = 5.0f;
    _quality = 8;
    _maskStrength = 1.0f;
}

// Override support methods
bool ImageBlurrer::supportsOpenCL() const
{
    return true;
}

bool ImageBlurrer::supportsCUDA() const
{
    return true;
}

bool ImageBlurrer::supportsMetal() const
{
    return true;
}

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
}

void ImageBlurrer::setParams(float p_Radius, int p_Quality, float p_MaskStrength)
{
    _radius = p_Radius;
    _quality = p_Quality;
    _maskStrength = p_MaskStrength;
}

// BlurPlugin implementation
BlurPlugin::BlurPlugin(OfxImageEffectHandle p_Handle)
    : GenericImageEffect(p_Handle)
{
    // Fetch the parameters
    m_Radius = fetchDoubleParam(BlurPluginParameters::PARAM_RADIUS);
    m_Quality = fetchIntParam(BlurPluginParameters::PARAM_QUALITY);
    m_MaskStrength = fetchDoubleParam(BlurPluginParameters::PARAM_MASK_STRENGTH);

    Logger::getInstance().logMessage("BlurPlugin instance created");
}

BlurPlugin::~BlurPlugin()
{
    Logger::getInstance().logMessage("BlurPlugin instance destroyed");
}

void BlurPlugin::setupProcessor(ImageBlurrer &p_ImageBlurrer, const OFX::RenderArguments& p_Args)
{
    // Get blur parameters
    double radius = m_Radius->getValueAtTime(p_Args.time);
    int quality = m_Quality->getValueAtTime(p_Args.time);
    double maskStrength = m_MaskStrength->getValueAtTime(p_Args.time);

    // Set the parameters
    p_ImageBlurrer.setParams(radius, quality, maskStrength);
    
    Logger::getInstance().logMessage("Blur parameters set: radius=%.2f, quality=%d, maskStrength=%.2f",
                                     radius, quality, maskStrength);
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