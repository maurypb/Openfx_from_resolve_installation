#include "BlurPluginParameters.h"

// Define constants
const char* const BlurPluginParameters::PARAM_RADIUS = "radius";
const char* const BlurPluginParameters::PARAM_QUALITY = "quality";
const char* const BlurPluginParameters::PARAM_MASK_STRENGTH = "maskStrength";

void BlurPluginParameters::defineParameters(OFX::ImageEffectDescriptor& p_Desc, OFX::PageParamDescriptor* page) {
    // If page is not provided, create one
    OFX::PageParamDescriptor* localPage = page;
    if (!localPage) {
        localPage = PluginParameters::definePage(p_Desc, PluginParameters::PAGE_MAIN);
    }
    
    // Blur radius parameter
    OFX::DoubleParamDescriptor* radiusParam = PluginParameters::defineDoubleParam(
        p_Desc,
        PARAM_RADIUS,
        "Radius",
        "Blur radius in pixels",
        5.0,  // default
        0.0,  // min
        100.0, // max
        0.0,  // display min
        50.0  // display max
    );
    PluginParameters::addParamToPage(localPage, *radiusParam);

    // Quality parameter
    OFX::IntParamDescriptor* qualityParam = PluginParameters::defineIntParam(
        p_Desc,
        PARAM_QUALITY,
        "Quality",
        "Number of samples for the blur",
        8,   // default
        1,   // min
        32,  // max
        1,   // display min
        16   // display max
    );
    PluginParameters::addParamToPage(localPage, *qualityParam);

    // Mask strength parameter
    OFX::DoubleParamDescriptor* maskStrengthParam = PluginParameters::defineDoubleParam(
        p_Desc,
        PARAM_MASK_STRENGTH,
        "Mask Strength",
        "How strongly the mask affects the blur radius",
        1.0,  // default
        0.0,  // min
        1.0,  // max
        0.0,  // display min
        1.0   // display max
    );
    PluginParameters::addParamToPage(localPage, *maskStrengthParam);
}

void BlurPluginParameters::defineAdvancedParameters(OFX::ImageEffectDescriptor& p_Desc, OFX::PageParamDescriptor* page) {
    // If page is not provided, create the advanced page
    OFX::PageParamDescriptor* localPage = page;
    if (!localPage) {
        localPage = PluginParameters::definePage(p_Desc, PluginParameters::PAGE_ADVANCED);
    }
    
    // Here you could add advanced parameters for the blur plugin
    // For now this is just a placeholder for future extensions
}