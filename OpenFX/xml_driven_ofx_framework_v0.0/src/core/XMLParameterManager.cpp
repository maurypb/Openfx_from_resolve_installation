#include "XMLParameterManager.h"
#include <iostream>
#include <stdexcept>
#include "Logger.h"

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
        Logger::getInstance().logMessage("XMLParameterManager::organizeUI starting");
        
        // Create a default page if none exists yet
        OFX::PageParamDescriptor* mainPage;
        if (pages.empty()) {
            mainPage = createPage("Main", desc);
            pages["Main"] = mainPage;
        } else {
            mainPage = pages.begin()->second;  // Use first available page
        }
        
        // Create groups for each "group" in XML (these become expandable sections)
        for (const auto& groupDef : xmlDef.getUIGroups()) {
            Logger::getInstance().logMessage("Creating group for: %s", groupDef.name.c_str());
            
            // Create group parameter (this creates the expandable section)
            OFX::GroupParamDescriptor* group = desc.defineGroupParam(groupDef.name.c_str());
            group->setLabels(groupDef.name.c_str(), groupDef.name.c_str(), groupDef.name.c_str());
            group->setOpen(true);  // Start expanded
            if (!groupDef.tooltip.empty()) {
                group->setHint(groupDef.tooltip.c_str());
            }
            
            // Add group to the main page
            mainPage->addChild(*group);
            
            // Handle OLD FORMAT: page -> columns -> parameters
            int paramCount = 0;
            for (const auto& columnDef : groupDef.columns) {
                for (const auto& paramDef : columnDef.parameters) {
                    OFX::ParamDescriptor* param = desc.getParamDescriptor(paramDef.name.c_str());
                    if (param) {
                        param->setParent(*group);  // Set the group as parent
                        paramCount++;
                        Logger::getInstance().logMessage("  Added parameter %s to group %s", 
                                                       paramDef.name.c_str(), groupDef.name.c_str());
                    } else {
                        Logger::getInstance().logMessage("  WARNING: Parameter not found: %s", paramDef.name.c_str());
                    }
                }
            }
            
            // Handle NEW FORMAT: page -> parameters directly
            for (const auto& paramDef : groupDef.parameters) {
                OFX::ParamDescriptor* param = desc.getParamDescriptor(paramDef.name.c_str());
                if (param) {
                    param->setParent(*group);  // Set the group as parent
                    paramCount++;
                    Logger::getInstance().logMessage("  Added parameter %s to group %s", 
                                                   paramDef.name.c_str(), groupDef.name.c_str());
                } else {
                    Logger::getInstance().logMessage("  WARNING: Parameter not found: %s", paramDef.name.c_str());
                }
            }
            
            Logger::getInstance().logMessage("Group %s created with %d parameters", groupDef.name.c_str(), paramCount);
        }
        
        Logger::getInstance().logMessage("XMLParameterManager::organizeUI completed successfully");
        return true;
    }
    catch (const std::exception& e) {
        Logger::getInstance().logMessage("ERROR in organizeUI: %s", e.what());
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
    // NOTE: Don't call setHint here - let applyResolutionDependency handle it
    
    // Range
    param->setRange(paramDef.minValue, paramDef.maxValue);
    param->setDisplayRange(paramDef.displayMin, paramDef.displayMax);
    param->setDefault(paramDef.defaultValue);
    
    // Increment is supported in OFX 1.4
    param->setIncrement(paramDef.inc);
    
    // Resolution dependency - this will set the hint
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
    // NOTE: Don't call setHint here - let applyResolutionDependency handle it
    
    // Range
    param->setRange(static_cast<int>(paramDef.minValue), static_cast<int>(paramDef.maxValue));
    param->setDisplayRange(static_cast<int>(paramDef.displayMin), static_cast<int>(paramDef.displayMax));
    param->setDefault(static_cast<int>(paramDef.defaultValue));
    
    // Resolution dependency - this will set the hint
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
    // NOTE: Don't call setHint here - let applyResolutionDependency handle it
    
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
    
    // Resolution dependency - this will set the hint
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
    // Build the final hint string without calling setHint multiple times
    std::string finalHint = currentHint;
    
    if (resDependent == "width") {
        finalHint += " (width-dependent)";
    }
    else if (resDependent == "height") {
        finalHint += " (height-dependent)";
    }
    else if (resDependent == "xy") {
        finalHint += " (dimension-dependent)";
    }
    // If resDependent is "none" or empty, just use the original hint
    
    // Set the hint only once with the final string
    param.setHint(finalHint.c_str());
}