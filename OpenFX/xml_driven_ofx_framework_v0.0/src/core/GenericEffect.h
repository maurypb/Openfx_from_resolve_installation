#ifndef GENERIC_EFFECT_H
#define GENERIC_EFFECT_H

#include "ofxsImageEffect.h"
#include "XMLEffectDefinition.h"
#include "ParameterValue.h"
#include <string>
#include <map>
#include <memory>

/**
 * @class GenericEffect
 * @brief Dynamic effect instance that replaces fixed plugins like BlurPlugin
 * 
 * This class can handle ANY XML effect definition by dynamically fetching
 * parameters and clips by name, then passing them to appropriate processors.
 * It replaces the need to write individual plugin classes for each effect.
 */
class GenericEffect : public OFX::ImageEffect {
private:
    XMLEffectDefinition m_xmlDef;
    std::string m_xmlFilePath;
    
    // Dynamic storage - fetched by name from XML
    std::map<std::string, OFX::Param*> m_dynamicParams;
    std::map<std::string, OFX::Clip*> m_dynamicClips;

public:
    /**
     * @brief Constructor - fetches parameters and clips created by GenericEffectFactory
     * @param p_Handle OFX effect handle
     * @param xmlFile Path to XML effect definition
     */
    GenericEffect(OfxImageEffectHandle p_Handle, const std::string& xmlFile);
    
    /**
     * @brief Destructor
     */
    virtual ~GenericEffect();

    /**
     * @brief OFX render method - processes images using XML-defined parameters
     * @param p_Args Render arguments from host
     */
    virtual void render(const OFX::RenderArguments& p_Args) override;

    /**
     * @brief OFX identity check - uses XML identity conditions
     * @param p_Args Identity arguments
     * @param p_IdentityClip Output clip for pass-through
     * @param p_IdentityTime Output time for pass-through
     * @return True if effect should pass through input unchanged
     */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, 
                           OFX::Clip*& p_IdentityClip, double& p_IdentityTime) override;

    /**
     * @brief Get XML effect definition
     * @return Reference to loaded XML definition
     */
    const XMLEffectDefinition& getXMLDefinition() const { return m_xmlDef; }

private:
    /**
     * @brief Get parameter value at specific time with type-safe conversion
     * @param paramName Parameter name from XML
     * @param time Time to get value at
     * @return ParameterValue with type-safe conversion
     */
    ParameterValue getParameterValue(const std::string& paramName, double time);
    
    /**
     * @brief Evaluate identity condition from XML
     * @param condition Identity condition to evaluate
     * @param time Time to evaluate at
     * @return True if condition is met
     */
    bool evaluateIdentityCondition(const XMLEffectDefinition::IdentityConditionDef& condition, double time);
    
    /**
     * @brief Set up and run processor (to be implemented in Step 3.5)
     * @param p_Args Render arguments
     */
    void setupAndProcess(const OFX::RenderArguments& p_Args);
};

#endif // GENERIC_EFFECT_H