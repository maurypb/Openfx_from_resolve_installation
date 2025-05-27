#ifndef XML_PARAMETER_MANAGER_H
#define XML_PARAMETER_MANAGER_H

#include "XMLEffectDefinition.h"
#include "ofxsImageEffect.h"
#include <string>
#include <map>
#include <memory>

/**
 * @class XMLParameterManager
 * @brief Maps XML parameter definitions to OFX parameters
 * 
 * This class is responsible for creating OFX parameters based on
 * XML parameter definitions and organizing them into pages and columns.
 */
class XMLParameterManager {
public:
    /**
     * @brief Constructor
     */
    XMLParameterManager();
    
    /**
     * @brief Destructor
     */
    ~XMLParameterManager();
    
    /**
     * @brief Create OFX parameters from XML definitions
     * @param xmlDef The XML effect definition
     * @param desc The OFX image effect descriptor
     * @param pages Map of page names to page descriptors (output)
     * @return True if successful, false otherwise
     */
    bool createParameters(
        const XMLEffectDefinition& xmlDef,
        OFX::ImageEffectDescriptor& desc,
        std::map<std::string, OFX::PageParamDescriptor*>& pages
    );
    
    /**
     * @brief Organize parameters into UI pages and columns
     * @param xmlDef The XML effect definition
     * @param desc The OFX image effect descriptor
     * @param pages Map of page names to page descriptors
     * @return True if successful, false otherwise
     */
    bool organizeUI(
        const XMLEffectDefinition& xmlDef,
        OFX::ImageEffectDescriptor& desc,
        std::map<std::string, OFX::PageParamDescriptor*>& pages
    );
    
private:
    // Helper methods for different parameter types
    OFX::DoubleParamDescriptor* createDoubleParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    OFX::IntParamDescriptor* createIntParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    OFX::BooleanParamDescriptor* createBooleanParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    OFX::ChoiceParamDescriptor* createChoiceParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    OFX::RGBParamDescriptor* createColorParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    OFX::Double2DParamDescriptor* createVec2Param(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    OFX::StringParamDescriptor* createStringParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    OFX::ParametricParamDescriptor* createCurveParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    // Create a page
    OFX::PageParamDescriptor* createPage(
        const std::string& name,
        OFX::ImageEffectDescriptor& desc
    );
    
    // Handle resolution dependency
    void applyResolutionDependency(
        OFX::ParamDescriptor& param,
        const std::string& resDependent,
        const std::string& currentHint
    );
};

#endif // XML_PARAMETER_MANAGER_H