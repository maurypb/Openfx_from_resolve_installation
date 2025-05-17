#pragma once

#include "PluginParameters.h"
#include "ofxsImageEffect.h"

class BlurPluginParameters {
public:
    // Define parameter names as constants
    static const char* const PARAM_RADIUS;
    static const char* const PARAM_QUALITY;
    static const char* const PARAM_MASK_STRENGTH;
    
    // Method to define all blur-specific parameters
    static void defineParameters(OFX::ImageEffectDescriptor& p_Desc, OFX::PageParamDescriptor* page = nullptr);
    
    // Method to define advanced blur parameters (if needed in the future)
    static void defineAdvancedParameters(OFX::ImageEffectDescriptor& p_Desc, OFX::PageParamDescriptor* page = nullptr);
};