#include "GenericEffect.h"
#include "GenericProcessor.h"
#include "../../Logger.h"
#include "ParameterValue.h"
#include <stdexcept>
#include <cstring>



GenericEffect::GenericEffect(OfxImageEffectHandle p_Handle, const std::string& xmlFile)
    : OFX::ImageEffect(p_Handle), m_xmlDef(xmlFile), m_xmlFilePath(xmlFile) {
    
    Logger::getInstance().logMessage("GenericEffect: Constructor called with XML file: %s", xmlFile.c_str());
    
    try {
        Logger::getInstance().logMessage("GenericEffect: XMLEffectDefinition created successfully");
        
        // Just fetch clips (these are available immediately)
        Logger::getInstance().logMessage("GenericEffect: Fetching clips from XML");
        for (const auto& inputDef : m_xmlDef.getInputs()) {
            OFX::Clip* clip = nullptr;
            
            // Map XML clip names to OFX standard names
            if (inputDef.name == "source") {
                try {
                    clip = fetchClip(kOfxImageEffectSimpleSourceClipName);  // "Source"
                    Logger::getInstance().logMessage("  - Fetched clip: %s", inputDef.name.c_str());
                } catch (const std::exception& e) {
                    Logger::getInstance().logMessage("  - Failed to fetch clip %s: %s", inputDef.name.c_str(), e.what());
                }
            } else {
                try {
                    clip = fetchClip(inputDef.name.c_str());  // Use XML name directly
                    Logger::getInstance().logMessage("  - Fetched clip: %s", inputDef.name.c_str());
                } catch (const std::exception& e) {
                    Logger::getInstance().logMessage("  - Failed to fetch clip %s: %s", inputDef.name.c_str(), e.what());
                }
            }
            
            if (clip) {
                m_dynamicClips[inputDef.name] = clip;  // Store with XML name
            }
        }
        
        // Fetch output clip
        try {
            m_dynamicClips["output"] = fetchClip(kOfxImageEffectOutputClipName);
            Logger::getInstance().logMessage("  - Fetched output clip");
        } catch (const std::exception& e) {
            Logger::getInstance().logMessage("  - Failed to fetch output clip: %s", e.what());
        }
        
        Logger::getInstance().logMessage("GenericEffect created successfully with 0 parameters (will fetch on first render) and %d clips", 
                                       (int)m_dynamicClips.size());
        
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("GenericEffect constructor failed: %s", e.what());
        // Don't re-throw - let the constructor succeed even if some things failed
        Logger::getInstance().logMessage("Continuing with partial initialization...");
    }
}

GenericEffect::~GenericEffect() {
    Logger::getInstance().logMessage("GenericEffect destroyed");
}

void GenericEffect::fetchParametersLazily() {
    Logger::getInstance().logMessage("GenericEffect: Fetching parameters lazily on first render");
    

    Logger::getInstance().logMessage("=== DIAGNOSTIC: Checking if any parameters exist ===");

    // Try to get parameter count or list from OFX
    try {
        // Test if we can access the parameter set at all
        OFX::Param* testParam = getParam("nonexistent");
        Logger::getInstance().logMessage("getParam() call succeeded (unexpected)");
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("getParam(\"nonexistent\") failed as expected: %s", e.what());
    }
    
    // Check if the effect instance has been properly initialized
    Logger::getInstance().logMessage("Effect handle: %p", getHandle());
    Logger::getInstance().logMessage("Effect context: %d", getContext());
    
    Logger::getInstance().logMessage("=== END DIAGNOSTIC ===");




    // TEST: Try hardcoded parameter names first
    Logger::getInstance().logMessage("=== TESTING HARDCODED PARAMETER NAMES ===");
    try {
        OFX::Param* testParam = fetchDoubleParam("radius");
        Logger::getInstance().logMessage("✓ Hardcoded fetchDoubleParam(\"radius\") SUCCEEDED: %p", testParam);
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("✗ Hardcoded fetchDoubleParam(\"radius\") FAILED: %s", e.what());
    }
    
    try {
        OFX::Param* testParam = fetchIntParam("quality");
        Logger::getInstance().logMessage("✓ Hardcoded fetchIntParam(\"quality\") SUCCEEDED: %p", testParam);
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("✗ Hardcoded fetchIntParam(\"quality\") FAILED: %s", e.what());
    }
    Logger::getInstance().logMessage("=== END HARDCODED TEST ===");






    for (const auto& paramDef : m_xmlDef.getParameters()) {
        Logger::getInstance().logMessage("  - Trying to fetch parameter: %s (%s)", paramDef.name.c_str(), paramDef.type.c_str());
        
        OFX::Param* param = nullptr;
        
        // Use type-specific fetch methods
        if (paramDef.type == "double" || paramDef.type == "float") {
            Logger::getInstance().logMessage("    - Calling fetchDoubleParam...");
            try {
                param = fetchDoubleParam(paramDef.name);
                Logger::getInstance().logMessage("    - fetchDoubleParam succeeded");
            } catch (const std::exception& e) {
                Logger::getInstance().logMessage("    - fetchDoubleParam threw exception: %s", e.what());
                param = nullptr;
            }
        }
        else if (paramDef.type == "int") {
            Logger::getInstance().logMessage("    - Calling fetchIntParam...");
            try {
                param = fetchIntParam(paramDef.name);
                Logger::getInstance().logMessage("    - fetchIntParam succeeded");
            } catch (const std::exception& e) {
                Logger::getInstance().logMessage("    - fetchIntParam threw exception: %s", e.what());
                param = nullptr;
            }
        }
        else if (paramDef.type == "bool") {
            Logger::getInstance().logMessage("    - Calling fetchBooleanParam...");
            try {
                param = fetchBooleanParam(paramDef.name);
                Logger::getInstance().logMessage("    - fetchBooleanParam succeeded");
            } catch (const std::exception& e) {
                Logger::getInstance().logMessage("    - fetchBooleanParam threw exception: %s", e.what());
                param = nullptr;
            }
        }
        else if (paramDef.type == "string") {
            Logger::getInstance().logMessage("    - Calling fetchStringParam...");
            try {
                param = fetchStringParam(paramDef.name);
                Logger::getInstance().logMessage("    - fetchStringParam succeeded");
            } catch (const std::exception& e) {
                Logger::getInstance().logMessage("    - fetchStringParam threw exception: %s", e.what());
                param = nullptr;
            }
        }
        else if (paramDef.type == "choice") {
            Logger::getInstance().logMessage("    - Calling fetchChoiceParam...");
            try {
                param = fetchChoiceParam(paramDef.name);
                Logger::getInstance().logMessage("    - fetchChoiceParam succeeded");
            } catch (const std::exception& e) {
                Logger::getInstance().logMessage("    - fetchChoiceParam threw exception: %s", e.what());
                param = nullptr;
            }
        }
        else if (paramDef.type == "color") {
            Logger::getInstance().logMessage("    - Calling fetchRGBParam...");
            try {
                param = fetchRGBParam(paramDef.name);
                Logger::getInstance().logMessage("    - fetchRGBParam succeeded");
            } catch (const std::exception& e) {
                Logger::getInstance().logMessage("    - fetchRGBParam threw exception: %s", e.what());
                param = nullptr;
            }
        }
        else if (paramDef.type == "vec2") {
            Logger::getInstance().logMessage("    - Calling fetchDouble2DParam...");
            try {
                param = fetchDouble2DParam(paramDef.name);
                Logger::getInstance().logMessage("    - fetchDouble2DParam succeeded");
            } catch (const std::exception& e) {
                Logger::getInstance().logMessage("    - fetchDouble2DParam threw exception: %s", e.what());
                param = nullptr;
            }
        }
        else {
            Logger::getInstance().logMessage("  - WARNING: Unsupported parameter type for fetch: %s", paramDef.type.c_str());
            continue;
        }
        
        if (param) {
            m_dynamicParams[paramDef.name] = param;
            Logger::getInstance().logMessage("    - ✓ Success!");
        } else {
            Logger::getInstance().logMessage("    - ✗ Failed to fetch, will use XML defaults");
        }
    }
    
    Logger::getInstance().logMessage("Parameter fetching completed: %d parameters available", (int)m_dynamicParams.size());
}




void GenericEffect::render(const OFX::RenderArguments& p_Args) {
    Logger::getInstance().logMessage("GenericEffect::render called");
    
    // Lazy parameter fetching on first render
    if (m_dynamicParams.empty()) {
        Logger::getInstance().logMessage("Parameters not fetched yet, fetching now...");
        fetchParametersLazily();
    }
    
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