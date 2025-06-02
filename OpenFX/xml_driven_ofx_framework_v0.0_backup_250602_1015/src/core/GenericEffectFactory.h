#ifndef GENERIC_EFFECT_FACTORY_H
#define GENERIC_EFFECT_FACTORY_H

#include "ofxsImageEffect.h"
#include "XMLEffectDefinition.h"
#include <string>

/**
 * @class GenericEffectFactory
 * @brief XML-driven factory that creates OFX plugins from XML definitions
 * 
 * This factory replaces the pattern used by BlurPluginFactory but can load
 * ANY XML effect definition instead of being hard-coded for one specific effect.
 * It handles the OFX describe/describeInContext phases using XML data and creates
 * GenericEffect instances.
 */
class GenericEffectFactory : public OFX::PluginFactoryHelper<GenericEffectFactory> {
private:
    XMLEffectDefinition m_xmlDef;
    std::string m_xmlFilePath;
    std::string m_pluginIdentifier;

public:
    /**
     * @brief Constructor that loads XML effect definition
     * @param xmlFile Path to the XML effect definition file
     * @throws std::runtime_error if XML file cannot be loaded or is invalid
     */
    GenericEffectFactory(const std::string& xmlFile);
    
    /**
     * @brief Destructor
     */
    virtual ~GenericEffectFactory();

    /**
     * @brief OFX describe phase - tells host about basic effect properties
     * @param p_Desc The image effect descriptor to fill
     */
    virtual void describe(OFX::ImageEffectDescriptor& p_Desc) override;

    /**
     * @brief OFX describe in context phase - creates parameters and clips from XML
     * @param p_Desc The image effect descriptor to fill
     * @param p_Context The OFX context (filter, general, etc.)
     */
    virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, 
                                  OFX::ContextEnum p_Context) override;

    /**
     * @brief OFX create instance phase - creates a GenericEffect instance
     * @param p_Handle The OFX effect handle
     * @param p_Context The OFX context
     * @return New GenericEffect instance
     */
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle p_Handle, 
                                           OFX::ContextEnum p_Context) override;

    /**
     * @brief Get the XML effect definition
     * @return Reference to the loaded XML definition
     */
    const XMLEffectDefinition& getXMLDefinition() const { return m_xmlDef; }

    /**
     * @brief Get the plugin identifier
     * @return The plugin identifier string
     */
    const std::string& getPluginIdentifier() const { return m_pluginIdentifier; }

private:
    /**
     * @brief Generate unique plugin identifier from XML file path
     * @param xmlFile Path to the XML file
     * @return Generated plugin identifier
     */
    static std::string generatePluginIdentifier(const std::string& xmlFile);
    
    /**
     * @brief Generate unique plugin identifier from XML (instance method)
     * @return Generated plugin identifier
     */
    std::string generatePluginIdentifier();
    
    /**
     * @brief Set up basic OFX properties
     * @param p_Desc The descriptor to configure
     */
    void setupBasicProperties(OFX::ImageEffectDescriptor& p_Desc);
    
    /**
     * @brief Set up GPU support flags
     * @param p_Desc The descriptor to configure
     */
    void setupGPUSupport(OFX::ImageEffectDescriptor& p_Desc);
};

#endif // GENERIC_EFFECT_FACTORY_H