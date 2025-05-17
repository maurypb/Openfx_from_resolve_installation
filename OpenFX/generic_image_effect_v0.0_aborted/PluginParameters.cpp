#include "PluginParameters.h"

// Define constants
const char* const PluginParameters::PAGE_MAIN = "Controls";
const char* const PluginParameters::PAGE_ADVANCED = "Advanced";
const char* const PluginParameters::PAGE_COLOR = "Color";

OFX::PageParamDescriptor* PluginParameters::definePage(OFX::ImageEffectDescriptor& p_Desc, const char* pageName) {
    return p_Desc.definePageParam(pageName);
}

OFX::DoubleParamDescriptor* PluginParameters::defineDoubleParam(
    OFX::ImageEffectDescriptor& p_Desc,
    const char* name,
    const char* label,
    const char* hint,
    double defaultValue,
    double minValue,
    double maxValue,
    double displayMin,
    double displayMax
) {
    OFX::DoubleParamDescriptor* param = p_Desc.defineDoubleParam(name);
    param->setLabels(label, label, label);
    param->setHint(hint);
    param->setRange(minValue, maxValue);
    param->setDisplayRange(displayMin, displayMax);
    param->setDefault(defaultValue);
    return param;
}

OFX::IntParamDescriptor* PluginParameters::defineIntParam(
    OFX::ImageEffectDescriptor& p_Desc,
    const char* name,
    const char* label,
    const char* hint,
    int defaultValue,
    int minValue,
    int maxValue,
    int displayMin,
    int displayMax
) {
    OFX::IntParamDescriptor* param = p_Desc.defineIntParam(name);
    param->setLabels(label, label, label);
    param->setHint(hint);
    param->setRange(minValue, maxValue);
    param->setDisplayRange(displayMin, displayMax);
    param->setDefault(defaultValue);
    return param;
}

OFX::BooleanParamDescriptor* PluginParameters::defineBoolParam(
    OFX::ImageEffectDescriptor& p_Desc,
    const char* name,
    const char* label,
    const char* hint,
    bool defaultValue
) {
    OFX::BooleanParamDescriptor* param = p_Desc.defineBooleanParam(name);
    param->setLabels(label, label, label);
    param->setHint(hint);
    param->setDefault(defaultValue);
    return param;
}

OFX::ChoiceParamDescriptor* PluginParameters::defineChoiceParam(
    OFX::ImageEffectDescriptor& p_Desc,
    const char* name,
    const char* label,
    const char* hint,
    int defaultOption
) {
    OFX::ChoiceParamDescriptor* param = p_Desc.defineChoiceParam(name);
    param->setLabels(label, label, label);
    param->setHint(hint);
    param->setDefault(defaultOption);
    return param;
}

void PluginParameters::addChoiceOption(
    OFX::ChoiceParamDescriptor* param,
    const char* optionName,
    const char* optionLabel
) {
    param->appendOption(optionName, optionLabel);
}

void PluginParameters::addParamToPage(OFX::PageParamDescriptor* page, const OFX::ParamDescriptor& param) {
    page->addChild(param);
}