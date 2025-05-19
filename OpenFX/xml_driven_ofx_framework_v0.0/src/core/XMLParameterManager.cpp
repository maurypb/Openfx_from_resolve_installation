#include "XMLParameterManager.h"
#include <iostream>
#include <stdexcept>

XMLParameterManager::XMLParameterManager() {
}

XMLParameterManager::~XMLParameterManager() {
}

bool XMLParameterManager::createParameters(
    const XMLEffectDefinition& xmlDef,
    OFX::ImageEffectDescriptor& desc,
    std::map<std::string, OFX::PageParamDescriptor*>& pages
) {
    try {
        // Create a default page if none exists yet
        if (pages.empty()) {
            pages["Main"] = createPage("Main", desc);
        }
        
        // Create all parameters
        for (const auto& paramDef : xmlDef.getParameters()) {
            if (paramDef.type == "double" || paramDef.type == "float") {
                createDoubleParam(paramDef, desc);
            }
            else if (paramDef.type == "int") {
                createIntParam(paramDef, desc);
            }
            else if (paramDef.type == "bool") {
                createBooleanParam(paramDef, desc);
            }
            else if (paramDef.type == "choice") {
                createChoiceParam(paramDef, desc);
            }
            else if (paramDef.type == "color") {
                createColorParam(paramDef, desc);
            }
            else if (paramDef.type == "vec2") {
                createVec2Param(paramDef, desc);
            }
            else if (paramDef.type == "string") {
                createStringParam(paramDef, desc);
            }
            else if (paramDef.type == "curve") {
                createCurveParam(paramDef, desc);
            }
            else {
                std::cerr << "Warning: Unsupported parameter type: " << paramDef.type << std::endl;
            }
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error creating parameters: " << e.what() << std::endl;
        return false;
    }
}

bool XMLParameterManager::organizeUI(
    const XMLEffectDefinition& xmlDef,
    OFX::ImageEffectDescriptor& desc,
    std::map<std::string, OFX::PageParamDescriptor*>& pages
) {
    try {
        // Create pages
        for (const auto& pageDef : xmlDef.getUIPages()) {
            // Create or get page
            OFX::PageParamDescriptor* page;
            auto it = pages.find(pageDef.name);
            if (it == pages.end()) {
                page = createPage(pageDef.name, desc);
                pages[pageDef.name] = page;
            } else {
                page = it->second;
            }
            
            // Add parameters to page
            for (const auto& columnDef : pageDef.columns) {
                for (const auto& paramDef : columnDef.parameters) {
                    OFX::ParamDescriptor* param = desc.getParamDescriptor(paramDef.name.c_str());
                    if (param) {
                        page->addChild(*param);
                    } else {
                        std::cerr << "Warning: Parameter not found: " << paramDef.name << std::endl;
                    }
                }
            }
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error organizing UI: " << e.what() << std::endl;
        return false;
    }
}

OFX::DoubleParamDescriptor* XMLParameterManager::createDoubleParam(
    const XMLEffectDefinition::ParameterDef& paramDef,
    OFX::ImageEffectDescriptor& desc
) {
    OFX::DoubleParamDescriptor* param = desc.defineDoubleParam(paramDef.name.c_str());
    
    // Basic properties
    param->setLabels(paramDef.label.c_str(), paramDef.label.c_str(), paramDef.label.c_str());
    param->setHint(paramDef.hint.c_str());
    
    // Range
    param->setRange(paramDef.minValue, paramDef.maxValue);
    param->setDisplayRange(paramDef.displayMin, paramDef.displayMax);
    param->setDefault(paramDef.defaultValue);
    
    // Increment (not directly supported in OFX, but stored for UI)
    param->setIncrement(paramDef.inc);
    
    // Resolution dependency
    applyResolutionDependency(*param, paramDef.resDependent, paramDef.hint);
    
    return param;
}

OFX::IntParamDescriptor* XMLParameterManager::createIntParam(
    const XMLEffectDefinition::ParameterDef& paramDef,
    OFX::ImageEffectDescriptor& desc
) {
    OFX::IntParamDescriptor* param = desc.defineIntParam(paramDef.name.c_str());
    
    // Basic properties
    param->setLabels(paramDef.label.c_str(), paramDef.label.c_str(), paramDef.label.c_str());
    param->setHint(paramDef.hint.c_str());
    
    // Range
    param->setRange(static_cast<int>(paramDef.minValue), static_cast<int>(paramDef.maxValue));
    param->setDisplayRange(static_cast<int>(paramDef.displayMin), static_cast<int>(paramDef.displayMax));
    param->setDefault(static_cast<int>(paramDef.defaultValue));
    
    // Resolution dependency
    applyResolutionDependency(*param, paramDef.resDependent, paramDef.hint);
    
    return param;
}

OFX::BooleanParamDescriptor* XMLParameterManager::createBooleanParam(
    const XMLEffectDefinition::ParameterDef& paramDef,
    OFX::ImageEffectDescriptor& desc
) {
    OFX::BooleanParamDescriptor* param = desc.defineBooleanParam(paramDef.name.c_str());
    
    // Basic properties
    param->setLabels(paramDef.label.c_str(), paramDef.label.c_str(), paramDef.label.c_str());
    param->setHint(paramDef.hint.c_str());
    
    // Default
    param->setDefault(paramDef.defaultBool);
    
    return param;
}

OFX::ChoiceParamDescriptor* XMLParameterManager::createChoiceParam(
    const XMLEffectDefinition::ParameterDef& paramDef,
    OFX::ImageEffectDescriptor& desc
) {
    OFX::ChoiceParamDescriptor* param = desc.defineChoiceParam(paramDef.name.c_str());
    
    // Basic properties
    param->setLabels(paramDef.label.c_str(), paramDef.label.c_str(), paramDef.label.c_str());
    param->setHint(paramDef.hint.c_str());
    
    // Add options
    for (const auto& option : paramDef.options) {
        param->appendOption(option.label.c_str(), option.label.c_str());
    }
    
    // Default
    param->setDefault(static_cast<int>(paramDef.defaultValue));
    
    return param;
}

OFX::RGBParamDescriptor* XMLParameterManager::createColorParam(
    const XMLEffectDefinition::ParameterDef& paramDef,
    OFX::ImageEffectDescriptor& desc
) {
    OFX::RGBParamDescriptor* param = desc.defineRGBParam(paramDef.name.c_str());
    
    // Basic properties
    param->setLabels(paramDef.label.c_str(), paramDef.label.c_str(), paramDef.label.c_str());
    param->setHint(paramDef.hint.c_str());
    
    // Find component defaults
    double r = 0.0, g = 0.0, b = 0.0;
    for (const auto& comp : paramDef.components) {
        if (comp.name == "r") r = comp.defaultValue;
        else if (comp.name == "g") g = comp.defaultValue;
        else if (comp.name == "b") b = comp.defaultValue;
    }
    
    // Default
    param->setDefault(r, g, b);
    
    return param;
}

OFX::Double2DParamDescriptor* XMLParameterManager::createVec2Param(
    const XMLEffectDefinition::ParameterDef& paramDef,
    OFX::ImageEffectDescriptor& desc
) {
    OFX::Double2DParamDescriptor* param = desc.defineDouble2DParam(paramDef.name.c_str());
    
    // Basic properties
    param->setLabels(paramDef.label.c_str(), paramDef.label.c_str(), paramDef.label.c_str());
    param->setHint(paramDef.hint.c_str());
    
    // Find component defaults
    double x = 0.0, y = 0.0;
    double minX = 0.0, minY = 0.0;
    double maxX = 1.0, maxY = 1.0;
    double displayMinX = 0.0, displayMinY = 0.0;
    double displayMaxX = 1.0, displayMaxY = 1.0;
    
    for (const auto& comp : paramDef.components) {
        if (comp.name == "x") {
            x = comp.defaultValue;
            minX = comp.minValue;
            maxX = comp.maxValue;
            displayMinX = comp.minValue;
            displayMaxX = comp.maxValue;
        }
        else if (comp.name == "y") {
            y = comp.defaultValue;
            minY = comp.minValue;
            maxY = comp.maxValue;
            displayMinY = comp.minValue;
            displayMaxY = comp.maxValue;
        }
    }
    
    // Range
    param->setRange(minX, minY, maxX, maxY);
    param->setDisplayRange(displayMinX, displayMinY, displayMaxX, displayMaxY);
    
    // Default
    param->setDefault(x, y);
    
    // Resolution dependency
    applyResolutionDependency(*param, paramDef.resDependent, paramDef.hint);
    
    return param;
}

OFX::StringParamDescriptor* XMLParameterManager::createStringParam(
    const XMLEffectDefinition::ParameterDef& paramDef,
    OFX::ImageEffectDescriptor& desc
) {
    OFX::StringParamDescriptor* param = desc.defineStringParam(paramDef.name.c_str());
    
    // Basic properties
    param->setLabels(paramDef.label.c_str(), paramDef.label.c_str(), paramDef.label.c_str());
    param->setHint(paramDef.hint.c_str());
    
    // Default
    param->setDefault(paramDef.defaultString.c_str());
    
    return param;
}

OFX::ParametricParamDescriptor* XMLParameterManager::createCurveParam(
    const XMLEffectDefinition::ParameterDef& paramDef,
    OFX::ImageEffectDescriptor& desc
) {
    OFX::ParametricParamDescriptor* param = desc.defineParametricParam(paramDef.name.c_str());
    
    // Basic properties
    param->setLabels(paramDef.label.c_str(), paramDef.label.c_str(), paramDef.label.c_str());
    param->setHint(paramDef.hint.c_str());
    
    // Set up curve - in OFX, we need to set min/max range
    param->setRange(0.0, 1.0);  // x range from 0 to 1
    param->setDimension(1);
    
    // Define a constant time for all control points
    // OFX requires a time parameter for animation support
    const double time = 0.0;  // Use frame 0 as the default time
    const bool addKey = false;  // We're not adding a keyframe, just a point
    
    // Handle default shape
    if (paramDef.defaultShape == "linear") {
        // Linear curve: y = x
        param->addControlPoint(0, time, 0.0, 0.0, addKey);
        param->addControlPoint(0, time, 1.0, 1.0, addKey);
    }
    else if (paramDef.defaultShape == "ease_in") {
        // Ease-in curve: y = x^2
        param->addControlPoint(0, time, 0.0, 0.0, addKey);
        param->addControlPoint(0, time, 0.5, 0.25, addKey);
        param->addControlPoint(0, time, 1.0, 1.0, addKey);
    }
    else if (paramDef.defaultShape == "ease_out") {
        // Ease-out curve: y = sqrt(x)
        param->addControlPoint(0, time, 0.0, 0.0, addKey);
        param->addControlPoint(0, time, 0.5, 0.75, addKey);
        param->addControlPoint(0, time, 1.0, 1.0, addKey);
    }
    else if (paramDef.defaultShape == "ease_in_out") {
        // Ease-in-out curve: smoothstep
        param->addControlPoint(0, time, 0.0, 0.0, addKey);
        param->addControlPoint(0, time, 0.25, 0.1, addKey);
        param->addControlPoint(0, time, 0.5, 0.5, addKey);
        param->addControlPoint(0, time, 0.75, 0.9, addKey);
        param->addControlPoint(0, time, 1.0, 1.0, addKey);
    }
    else {
        // Default to linear
        param->addControlPoint(0, time, 0.0, 0.0, addKey);
        param->addControlPoint(0, time, 1.0, 1.0, addKey);
    }
    
    return param;
}

OFX::PageParamDescriptor* XMLParameterManager::createPage(
    const std::string& name,
    OFX::ImageEffectDescriptor& desc
) {
    OFX::PageParamDescriptor* page = desc.definePageParam(name.c_str());
    page->setLabels(name.c_str(), name.c_str(), name.c_str());
    return page;
}

void XMLParameterManager::applyResolutionDependency(
    OFX::ParamDescriptor& param,
    const std::string& resDependent,
    const std::string& currentHint
) {
    // Note: OFX API may vary across versions
    // This is a simplified implementation that might need to be adapted
    
    // Some OFX implementations use doubleType, others use different methods
    // If your OFX version doesn't support setDoubleType, this will need to be modified
    
    // Try to set the parameter's hint about how it relates to dimensions
    if (resDependent == "width") {
        // For width-dependent parameters, hint that it's an X coordinate
        param.setHint((currentHint + " (width-dependent)").c_str());
    }
    else if (resDependent == "height") {
        // For height-dependent parameters, hint that it's a Y coordinate
        param.setHint((currentHint + " (height-dependent)").c_str());
    }
    else if (resDependent == "xy") {
        // For both width and height, hint both
        param.setHint((currentHint + " (dimension-dependent)").c_str());
    }
    else {
        // If not resolution dependent, just set the original hint
        param.setHint(currentHint.c_str());
    }
    
    // For your specific OFX version, you may need to use a different approach
    // such as specific parameter properties or custom properties
}