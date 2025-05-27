#include "GenericEffect.h"
#include "GenericProcessor.h"
#include "../../Logger.h"
#include <stdexcept>
#include <cstring>

GenericEffect::GenericEffect(OfxImageEffectHandle p_Handle, const std::string& xmlFile)
    : OFX::ImageEffect(p_Handle), m_xmlDef(xmlFile), m_xmlFilePath(xmlFile) {
    
    try {
        // Fetch all parameters that the factory created
        Logger::getInstance().logMessage("GenericEffect: Fetching parameters from XML");
        for (const auto& paramDef : m_xmlDef.getParameters()) {
            OFX::Param* param = nullptr;
            
            // Use type-specific fetch methods
            if (paramDef.type == "double" || paramDef.type == "float") {
                param = fetchDoubleParam(paramDef.name);
            }
            else if (paramDef.type == "int") {
                param = fetchIntParam(paramDef.name);
            }
            else if (paramDef.type == "bool") {
                param = fetchBooleanParam(paramDef.name);
            }
            else if (paramDef.type == "string") {
                param = fetchStringParam(paramDef.name);
            }
            else if (paramDef.type == "choice") {
                param = fetchChoiceParam(paramDef.name);
            }
            else if (paramDef.type == "color") {
                param = fetchRGBParam(paramDef.name);
            }
            else if (paramDef.type == "vec2") {
                param = fetchDouble2DParam(paramDef.name);
            }
            else {
                Logger::getInstance().logMessage("  - WARNING: Unsupported parameter type for fetch: %s", paramDef.type.c_str());
                continue;
            }
            
            if (param) {
                m_dynamicParams[paramDef.name] = param;
                Logger::getInstance().logMessage("  - Fetched parameter: %s (%s)", paramDef.name.c_str(), paramDef.type.c_str());
            } else {
                Logger::getInstance().logMessage("  - WARNING: Parameter not found: %s", paramDef.name.c_str());
            }
        }
        
        // Fetch all clips that the factory created
        Logger::getInstance().logMessage("GenericEffect: Fetching clips from XML");
        for (const auto& inputDef : m_xmlDef.getInputs()) {
            OFX::Clip* clip = nullptr;
            
            // Map XML clip names to OFX standard names
            if (inputDef.name == "source") {
                clip = fetchClip(kOfxImageEffectSimpleSourceClipName);  // "Source"
            } else {
                clip = fetchClip(inputDef.name.c_str());  // Use XML name directly
            }
            
            if (clip) {
                m_dynamicClips[inputDef.name] = clip;  // Store with XML name
                Logger::getInstance().logMessage("  - Fetched clip: %s", inputDef.name.c_str());
            } else {
                Logger::getInstance().logMessage("  - WARNING: Clip not found: %s", inputDef.name.c_str());
            }
        }
        
        // Fetch output clip
        m_dynamicClips["output"] = fetchClip(kOfxImageEffectOutputClipName);
        Logger::getInstance().logMessage("  - Fetched output clip");
        
        Logger::getInstance().logMessage("GenericEffect created successfully with %d parameters and %d clips", 
                                       (int)m_dynamicParams.size(), (int)m_dynamicClips.size());
        
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("GenericEffect constructor failed: %s", e.what());
        throw;
    }
}

GenericEffect::~GenericEffect() {
    Logger::getInstance().logMessage("GenericEffect destroyed");
}

void GenericEffect::render(const OFX::RenderArguments& p_Args) {
    Logger::getInstance().logMessage("GenericEffect::render called");
    
    // Get output clip
    OFX::Clip* dstClip = m_dynamicClips["output"];
    if (!dstClip) {
        Logger::getInstance().logMessage("ERROR: No output clip found");
        OFX::throwSuiteStatusException(kOfxStatErrValue);
        return;
    }
    
    // Check format support (same pattern as BlurPlugin and Resolve examples)
    if ((dstClip->getPixelDepth() == OFX::eBitDepthFloat) && 
        (dstClip->getPixelComponents() == OFX::ePixelComponentRGBA)) {
        
        Logger::getInstance().logMessage("Format supported, calling setupAndProcess");
        setupAndProcess(p_Args);
    }
    else {
        Logger::getInstance().logMessage("ERROR: Unsupported pixel format");
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool GenericEffect::isIdentity(const OFX::IsIdentityArguments& p_Args, 
                              OFX::Clip*& p_IdentityClip, double& p_IdentityTime) {
    
    Logger::getInstance().logMessage("GenericEffect::isIdentity called");
    
    // Evaluate identity conditions from XML
    for (const auto& condition : m_xmlDef.getIdentityConditions()) {
        if (evaluateIdentityCondition(condition, p_Args.time)) {
            // Pass through first input
            if (!m_xmlDef.getInputs().empty()) {
                std::string firstInputName = m_xmlDef.getInputs()[0].name;
                p_IdentityClip = m_dynamicClips[firstInputName];
                p_IdentityTime = p_Args.time;
                Logger::getInstance().logMessage("Identity condition met, passing through %s", firstInputName.c_str());
                return true;
            }
        }
    }
    
    return false;
}

ParameterValue GenericEffect::getParameterValue(const std::string& paramName, double time) {
    // Check if parameter exists
    auto paramIt = m_dynamicParams.find(paramName);
    if (paramIt == m_dynamicParams.end()) {
        Logger::getInstance().logMessage("WARNING: Parameter %s not found", paramName.c_str());
        return ParameterValue(); // Default value
    }
    
    OFX::Param* param = paramIt->second;
    const auto& paramDef = m_xmlDef.getParameter(paramName);
    
    // Extract value based on type
    if (paramDef.type == "double" || paramDef.type == "float") {
        OFX::DoubleParam* doubleParam = static_cast<OFX::DoubleParam*>(param);
        double value = doubleParam->getValueAtTime(time);
        return ParameterValue(value);
    }
    else if (paramDef.type == "int") {
        OFX::IntParam* intParam = static_cast<OFX::IntParam*>(param);
        int value = intParam->getValueAtTime(time);
        return ParameterValue(value);
    }
    else if (paramDef.type == "bool") {
        OFX::BooleanParam* boolParam = static_cast<OFX::BooleanParam*>(param);
        bool value = boolParam->getValueAtTime(time);
        return ParameterValue(value);
    }
    else if (paramDef.type == "string") {
        OFX::StringParam* stringParam = static_cast<OFX::StringParam*>(param);
        std::string value;
        stringParam->getValueAtTime(time, value);
        return ParameterValue(value);
    }
    else {
        Logger::getInstance().logMessage("WARNING: Unsupported parameter type: %s", paramDef.type.c_str());
        return ParameterValue(); // Default value
    }
}

bool GenericEffect::evaluateIdentityCondition(const XMLEffectDefinition::IdentityConditionDef& condition, double time) {
    ParameterValue value = getParameterValue(condition.paramName, time);
    
    if (condition.op == "lessEqual") {
        return value.asDouble() <= condition.value;
    }
    else if (condition.op == "equal") {
        return value.asDouble() == condition.value;
    }
    else if (condition.op == "lessThan") {
        return value.asDouble() < condition.value;
    }
    else if (condition.op == "greaterThan") {
        return value.asDouble() > condition.value;
    }
    else if (condition.op == "greaterEqual") {
        return value.asDouble() >= condition.value;
    }
    else if (condition.op == "notEqual") {
        return value.asDouble() != condition.value;
    }
    else {
        Logger::getInstance().logMessage("WARNING: Unknown identity operator: %s", condition.op.c_str());
        return false;
    }
}

void GenericEffect::setupAndProcess(const OFX::RenderArguments& p_Args) {
    Logger::getInstance().logMessage("GenericEffect::setupAndProcess called");
    
    try {
        // Create processor
        GenericProcessor processor(*this, m_xmlDef);
        
        // Get images
        std::unique_ptr<OFX::Image> dst(m_dynamicClips["output"]->fetchImage(p_Args.time));
        std::unique_ptr<OFX::Image> src(m_dynamicClips["source"]->fetchImage(p_Args.time));
        
        // Set up images map
        std::map<std::string, OFX::Image*> images;
        images["source"] = src.get();
        images["output"] = dst.get();
        
        processor.setImages(images);
        processor.setGPURenderArgs(p_Args);
        processor.setRenderWindow(p_Args.renderWindow);
        processor.process();
        
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("ERROR: %s", e.what());
    }
}