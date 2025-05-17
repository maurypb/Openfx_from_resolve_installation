#pragma once

#include "ofxsImageEffect.h"

// Template-based generic plugin factory
template<class EffectClass, class ParametersClass>
class GenericPluginFactory : public OFX::PluginFactoryHelper<GenericPluginFactory<EffectClass, ParametersClass>> {
public:
    GenericPluginFactory(
        const std::string& identifier,
        unsigned int versionMajor,
        unsigned int versionMinor
    ) : OFX::PluginFactoryHelper<GenericPluginFactory<EffectClass, ParametersClass>>(
            identifier, versionMajor, versionMinor) {}
    
    virtual void load() {}
    virtual void unload() {}
    
    virtual void describe(OFX::ImageEffectDescriptor& p_Desc);
    virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context);
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum p_Context);
};

// Implementation of template methods
template<class EffectClass, class ParametersClass>
void GenericPluginFactory<EffectClass, ParametersClass>::describe(OFX::ImageEffectDescriptor& p_Desc) {
    // Basic labels come from static methods in the EffectClass
    p_Desc.setLabels(EffectClass::getName(), EffectClass::getName(), EffectClass::getName());
    p_Desc.setPluginGrouping(EffectClass::getGrouping());
    p_Desc.setPluginDescription(EffectClass::getDescription());

    // Add the supported contexts
    p_Desc.addSupportedContext(OFX::eContextFilter);
    p_Desc.addSupportedContext(OFX::eContextGeneral);

    // Add supported pixel depths
    p_Desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    // Set a few flags - these could also come from the EffectClass
    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(EffectClass::supportsMultiResolution());
    p_Desc.setSupportsTiles(EffectClass::supportsTiles());
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(EffectClass::supportsMultipleClipPARs());

    // GPU support flags
    p_Desc.setSupportsOpenCLRender(EffectClass::supportsOpenCL());
    
#ifndef __APPLE__
    p_Desc.setSupportsCudaRender(EffectClass::supportsCuda());
    p_Desc.setSupportsCudaStream(EffectClass::supportsCudaStream());
#endif

#ifdef __APPLE__
    p_Desc.setSupportsMetalRender(EffectClass::supportsMetal());
#endif

    p_Desc.setNoSpatialAwareness(false);
}

template<class EffectClass, class ParametersClass>
void GenericPluginFactory<EffectClass, ParametersClass>::describeInContext(
    OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context) {
    
    // Define base clips (source and output)
    PluginClips::defineBaseClips(p_Desc, EffectClass::supportsTiles());
    
    // Define mask clip if needed
    if (EffectClass::usesMask()) {
        PluginClips::defineMaskClip(p_Desc, EffectClass::supportsTiles());
    }
    
    // Create main parameters page
    OFX::PageParamDescriptor* mainPage = PluginParameters::definePage(p_Desc, PluginParameters::PAGE_MAIN);
    
    // Define effect-specific parameters
    ParametersClass::defineParameters(p_Desc, mainPage);
    
    // Define advanced parameters if applicable
    if (EffectClass::hasAdvancedParameters()) {
        OFX::PageParamDescriptor* advancedPage = 
            PluginParameters::definePage(p_Desc, PluginParameters::PAGE_ADVANCED);
        ParametersClass::defineAdvancedParameters(p_Desc, advancedPage);
    }
}

template<class EffectClass, class ParametersClass>
OFX::ImageEffect* GenericPluginFactory<EffectClass, ParametersClass>::createInstance(
    OfxImageEffectHandle p_Handle, OFX::ContextEnum p_Context) {
    
    return new EffectClass(p_Handle);
}