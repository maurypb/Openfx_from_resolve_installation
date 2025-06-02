#include "BlurPlugin.h"
#include "Logger.h"
#include <stdio.h>
#include <stdarg.h>

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"
#include "ofxDrawSuite.h"
#include "ofxsSupportPrivate.h"

#include "PluginClips.h"
#include "PluginParameters.h"
#include "BlurPluginParameters.h"
#include <fstream>
#include "src/core/GenericEffectFactory.h"

#include "src/core/GenericEffect.h"

#include "include/pugixml/pugixml.hpp"
#include "src/core/XMLEffectDefinition.h"
#include "src/core/XMLInputManager.h"
#include "src/core/XMLParameterManager.h"


#define kPluginName "GaussianBlur"
#define kPluginGrouping "Filter"
#define kPluginDescription "Apply Gaussian blur with optional mask control"
#define kPluginIdentifier "com.Maury.GaussianBlur"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 5

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

void testGenericEffectFactory() {
    Logger::getInstance().logMessage("=== Testing GenericEffectFactory ===");
    
    try {
        std::string xmlPath = "/mnt/tank/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/xml_driven_ofx_framework_v0.0/TestBlurV2.xml";
        
        Logger::getInstance().logMessage("Creating GenericEffectFactory...");
        GenericEffectFactory factory(xmlPath);
        Logger::getInstance().logMessage("✓ GenericEffectFactory created successfully");
        
        // Test XML loading
        const XMLEffectDefinition& xmlDef = factory.getXMLDefinition();
        Logger::getInstance().logMessage("✓ XML definition retrieved");
        Logger::getInstance().logMessage("  Effect name: %s", xmlDef.getName().c_str());
        Logger::getInstance().logMessage("  Effect category: %s", xmlDef.getCategory().c_str());
        Logger::getInstance().logMessage("  Plugin identifier: %s", factory.getPluginIdentifier().c_str());
        
        // Test parameter parsing
        auto params = xmlDef.getParameters();
        Logger::getInstance().logMessage("  Parameter count: %d", (int)params.size());
        for (const auto& param : params) {
            Logger::getInstance().logMessage("    - %s (%s): default=%.2f", 
                                           param.name.c_str(), param.type.c_str(), param.defaultValue);
        }
        
        // Test input parsing
        auto inputs = xmlDef.getInputs();
        Logger::getInstance().logMessage("  Input count: %d", (int)inputs.size());
        for (const auto& input : inputs) {
            Logger::getInstance().logMessage("    - %s (optional: %s, border: %s)", 
                                           input.name.c_str(), 
                                           input.optional ? "true" : "false",
                                           input.borderMode.c_str());
        }
        
        // Test kernel parsing
        auto kernels = xmlDef.getKernels();
        Logger::getInstance().logMessage("  Kernel count: %d", (int)kernels.size());
        for (const auto& kernel : kernels) {
            Logger::getInstance().logMessage("    - %s: %s", kernel.platform.c_str(), kernel.file.c_str());
        }
        
        Logger::getInstance().logMessage("✓ GenericEffectFactory test completed successfully");
        
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("✗ GenericEffectFactory test failed: %s", e.what());
    }
    
    Logger::getInstance().logMessage("=== GenericEffectFactory Test Complete ===");
}


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
    // FILE* logFile = fopen("/tmp/blur_plugin_log.txt", "a");
    // if (logFile) {
    //     fprintf(logFile, "processImagesCUDA called: _radius=%.2f, _maskStrength=%.2f\n", _radius, _maskStrength);
    //     fprintf(logFile, "  Mask image pointer: %p\n", _maskImg);
    //     fclose(logFile);
    // }
    
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());
    
    // Get mask data if available
    float* mask = nullptr;
    if (_maskImg) {
        mask = static_cast<float*>(_maskImg->getPixelData());
        
        // logFile = fopen("/tmp/blur_plugin_log.txt", "a");
        // if (logFile) {
        //     fprintf(logFile, "  Mask data pointer: %p\n", mask);
        //     fclose(logFile);
        // }
    }

    // logFile = fopen("/tmp/blur_plugin_log.txt", "a");
    // if (logFile) {
    //     fprintf(logFile, "  Calling RunCudaKernel\n");
    //     fclose(logFile);
    // }
    
    RunCudaKernel(_pCudaStream, width, height, _radius, _quality, _maskStrength, input, mask, output);
    
    // logFile = fopen("/tmp/blur_plugin_log.txt", "a");
    // if (logFile) {
    //     fprintf(logFile, "  RunCudaKernel completed\n");
    //     fclose(logFile);
    // }
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
    virtual ~BlurPlugin();
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
    //void logMessage(const char* format, ...);
};


//original blurPlugin constructor

// BlurPlugin::BlurPlugin(OfxImageEffectHandle p_Handle)
//     : ImageEffect(p_Handle)
// {

//     m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
//     m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
//     m_MaskClip = fetchClip("Mask");

//     m_Radius = fetchDoubleParam("radius");
//     m_Quality = fetchIntParam("quality");
//     m_MaskStrength = fetchDoubleParam("maskStrength");


    
//     Logger::getInstance().logMessage("BlurPlugin instance created");

// }

BlurPlugin::BlurPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    // Fetch clips
    m_DstClip = fetchClip(PluginClips::CLIP_OUTPUT);
    m_SrcClip = fetchClip(PluginClips::CLIP_SOURCE);
    m_MaskClip = fetchClip(PluginClips::CLIP_MASK);

    // Fetch parameters
    m_Radius = fetchDoubleParam(BlurPluginParameters::PARAM_RADIUS);
    m_Quality = fetchIntParam(BlurPluginParameters::PARAM_QUALITY);
    m_MaskStrength = fetchDoubleParam(BlurPluginParameters::PARAM_MASK_STRENGTH);

    // Log instance creation
    Logger::getInstance().logMessage("BlurPlugin instance created");
}





BlurPlugin::~BlurPlugin()
{
    // destructor
    Logger::getInstance().logMessage("BlurPlugin instance destroyed");
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
    //logMessage("setupAndProcess: Time = %f", p_Args.time);

    // Get the dst image
    std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    // Get the src image
    std::unique_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    // // Log source image details
    // logMessage("  Src image: %p, dimensions: %d x %d", 
    //           src.get(), 
    //           src->getBounds().x2 - src->getBounds().x1,
    //           src->getBounds().y2 - src->getBounds().y1);

    // Check to see if the bit depth and number of components are the same
    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }

    // Get the mask image if it exists
    std::unique_ptr<OFX::Image> mask;
    if (m_MaskClip && m_MaskClip->isConnected())
    {
        //logMessage("  Mask clip is connected");
        mask.reset(m_MaskClip->fetchImage(p_Args.time));
        //logMessage("  Mask image fetched: %p", mask.get());
        
    }
    else
    {
        //logMessage("  No mask connected");
    }

    // Get blur parameters
    double radius = m_Radius->getValueAtTime(p_Args.time);
    int quality = m_Quality->getValueAtTime(p_Args.time);
    double maskStrength = m_MaskStrength->getValueAtTime(p_Args.time);

    // logMessage("  Parameters: radius=%.2f, quality=%d, maskStrength=%.2f",
    //          radius, quality, maskStrength);

    // Set the images
    p_ImageBlurrer.setDstImg(dst.get());
    p_ImageBlurrer.setSrcImg(src.get());
    p_ImageBlurrer.setMaskImg(mask.get());

    // Setup OpenCL and CUDA Render arguments
    p_ImageBlurrer.setGPURenderArgs(p_Args);  // right here is where the information needed for cuda is passed to ImageBlurrer object.
    //this includes the cudaStream.  This object is maintained as it is passed along, so when "processImagesCUDA() is called, that information is already present"

    // Set the render window
    p_ImageBlurrer.setRenderWindow(p_Args.renderWindow);

    // Set the parameters
    p_ImageBlurrer.setParams(radius, quality, maskStrength);

    //logMessage("  About to process image");

    // Call the base class process member, this will call the derived templated process code
    p_ImageBlurrer.process();  //our ImageBlurrer object doesn't have a process function - it's in the parent class.
    // this function (of the parent class) decides which these functions to call (all of which we have defined (overridden - because "virtual") in this ImageBlurrer class): 


// note that we passed raw pointers (dst.get()) to ImageBlurrer, which will actually process the images.
// as we created the Image objects dst,src and mask with unique_pointers, they will get "cleaned up" (eg deleted)
//here, as those pointers will go out of scope... but they will only get deleted AFTER ImageBlurrer.process()
//returns!  Or, after the image is processed, generated, and moved back to the cpu memory.





    // virtual void processImagesCUDA();
    // virtual void processImagesOpenCL();
    // virtual void processImagesMetal();
    // virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);




    //logMessage("  Image processing completed");
}


////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

BlurPluginFactory::BlurPluginFactory()
    : OFX::PluginFactoryHelper<BlurPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
    // Add this line to test the factory
    testGenericEffectFactory();
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
    // Define base clips (source and output)
    PluginClips::defineBaseClips(p_Desc, kSupportsTiles);
    
    // Define mask clip
    PluginClips::defineMaskClip(p_Desc, kSupportsTiles);

    // Create main parameters page
    OFX::PageParamDescriptor* mainPage = PluginParameters::definePage(p_Desc, PluginParameters::PAGE_MAIN);
    
    // Define blur parameters on the main page
    BlurPluginParameters::defineParameters(p_Desc, mainPage);
    
    // Optionally, create an advanced page with additional parameters
    // OFX::PageParamDescriptor* advancedPage = PluginParameters::definePage(p_Desc, PluginParameters::PAGE_ADVANCED);
    // BlurPluginParameters::defineAdvancedParameters(p_Desc, advancedPage);
}







ImageEffect* BlurPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new BlurPlugin(p_Handle);

    // std::string xmlPath = "/mnt/tank/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/xml_driven_ofx_framework_v0.0/TestBlurV2.xml";
    // return new GenericEffect(p_Handle, xmlPath);


}


// Add this BEFORE the getPluginIDs function
class TestFactory : public OFX::PluginFactoryHelper<TestFactory> {
    public:
        TestFactory() : OFX::PluginFactoryHelper<TestFactory>("com.test.minimal", 1, 0) {}
        
        virtual void describe(OFX::ImageEffectDescriptor& desc) override {
            desc.setLabels("TestMinimal", "TestMinimal", "TestMinimal");
            desc.setPluginGrouping("Test");
            desc.addSupportedContext(OFX::eContextFilter);
            desc.addSupportedBitDepth(OFX::eBitDepthFloat);
        }


        virtual void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum /*p_Context*/) override {
            try {
                Logger::getInstance().logMessage("Testing parameter with page...");
                
                // Create a page first
                OFX::PageParamDescriptor* page = desc.definePageParam("Controls");
                page->setLabels("Controls", "Controls", "Controls");
                
                // Create parameter
                OFX::DoubleParamDescriptor* testParam = desc.defineDoubleParam("testParam");
                testParam->setLabels("Test Param", "Test Param", "Test Param");
                testParam->setDefault(5.0);
                testParam->setRange(0.0, 100.0);
                
                // Add parameter to page
                page->addChild(*testParam);
                
                Logger::getInstance().logMessage("Parameter with page created successfully");


                // Add basic clips - Resolve needs these to show parameters
                OFX::ClipDescriptor* srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
                srcClip->addSupportedComponent(OFX::ePixelComponentRGBA);
                srcClip->setSupportsTiles(false);

                OFX::ClipDescriptor* dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
                dstClip->addSupportedComponent(OFX::ePixelComponentRGBA);
                dstClip->setSupportsTiles(false);

                Logger::getInstance().logMessage("Basic clips created successfully");
                
            } catch (const std::exception& e) {
                Logger::getInstance().logMessage("ERROR in parameter with page: %s", e.what());
            }
        }

        virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum /*p_Context*/) override {
            return new BlurPlugin(p_Handle);
        }
    }; 





void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{

    // static BlurPluginFactory blurPlugin;
    // p_FactoryArray.push_back(&blurPlugin);

    // static TestFactory testPlugin;
    // p_FactoryArray.push_back(&testPlugin);

    std::string xmlPath = "/mnt/tank/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/xml_driven_ofx_framework_v0.0/TestBlurV2.xml";
    static GenericEffectFactory genericPlugin(xmlPath);
    p_FactoryArray.push_back(&genericPlugin);




}