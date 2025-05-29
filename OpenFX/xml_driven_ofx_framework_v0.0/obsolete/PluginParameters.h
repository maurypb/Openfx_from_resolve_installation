#pragma once

#include "ofxsImageEffect.h"
#include <string>

class PluginParameters {
public:
    // Define common parameter page names
    static const char* const PAGE_MAIN;
    static const char* const PAGE_ADVANCED;
    static const char* const PAGE_COLOR;
    
    // Helper methods for creating parameter pages
    static OFX::PageParamDescriptor* definePage(OFX::ImageEffectDescriptor& p_Desc, const char* pageName);
    
    // Double parameter with range and default
    static OFX::DoubleParamDescriptor* defineDoubleParam(
        OFX::ImageEffectDescriptor& p_Desc,
        const char* name,
        const char* label,
        const char* hint,
        double defaultValue,
        double minValue,
        double maxValue,
        double displayMin,
        double displayMax
    );
    
    // Integer parameter with range and default
    static OFX::IntParamDescriptor* defineIntParam(
        OFX::ImageEffectDescriptor& p_Desc,
        const char* name,
        const char* label,
        const char* hint,
        int defaultValue,
        int minValue,
        int maxValue,
        int displayMin,
        int displayMax
    );
    
    // Boolean parameter with default
    static OFX::BooleanParamDescriptor* defineBoolParam(
        OFX::ImageEffectDescriptor& p_Desc,
        const char* name,
        const char* label,
        const char* hint,
        bool defaultValue
    );
    
    // Choice parameter
    static OFX::ChoiceParamDescriptor* defineChoiceParam(
        OFX::ImageEffectDescriptor& p_Desc,
        const char* name,
        const char* label,
        const char* hint,
        int defaultOption
    );
    
    // Add choice option to a choice parameter
    static void addChoiceOption(
        OFX::ChoiceParamDescriptor* param,
        const char* optionName,
        const char* optionLabel
    );
    
    // Add parameter to a page
    static void addParamToPage(OFX::PageParamDescriptor* page, const OFX::ParamDescriptor& param);
};