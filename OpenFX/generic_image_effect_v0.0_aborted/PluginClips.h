#pragma once

#include "ofxsImageEffect.h"

class PluginClips {
public:
    // Common clip names as constants
    static const char* const CLIP_SOURCE;      // Main source input
    static const char* const CLIP_OUTPUT;      // Output clip
    static const char* const CLIP_MASK;        // Optional mask clip
    
    // Define base clips method (Source, Output, and optional Mask)
    static void defineBaseClips(OFX::ImageEffectDescriptor& p_Desc, bool p_SupportsTiles);
    
    // Helper method to define an optional mask clip
    static void defineMaskClip(OFX::ImageEffectDescriptor& p_Desc, bool p_SupportsTiles);
    
    // Helper method to define a custom input clip
    static OFX::ClipDescriptor* defineCustomClip(
        OFX::ImageEffectDescriptor& p_Desc, 
        const char* clipName, 
        bool p_SupportsTiles, 
        bool isOptional = false,
        bool isMask = false
    );
};