#ifndef XML_INPUT_MANAGER_H
#define XML_INPUT_MANAGER_H

#include "core/XMLEffectDefinition.h"
#include "ofxsImageEffect.h"
#include <string>
#include <map>
#include <algorithm> // For std::transform

/**
 * @class XMLInputManager
 * @brief Maps XML input definitions to OFX clips
 * 
 * This class is responsible for creating OFX clips based on
 * XML input definitions and storing border modes for each input.
 */
class XMLInputManager {
public:
    /**
     * @brief Constructor
     */
    XMLInputManager();
    
    /**
     * @brief Destructor
     */
    ~XMLInputManager();
    
    /**
     * @brief Create OFX clips from XML definitions
     * @param xmlDef The XML effect definition
     * @param desc The OFX image effect descriptor
     * @param clipBorderModes Map of clip names to border modes (output)
     * @return True if successful, false otherwise
     */
    bool createInputs(
        const XMLEffectDefinition& xmlDef,
        OFX::ImageEffectDescriptor& desc,
        std::map<std::string, std::string>& clipBorderModes
    );
    
private:
    // Helper method to create a clip
    OFX::ClipDescriptor* createClip(
        const XMLEffectDefinition::InputDef& inputDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    // Set border mode props using OFX 1.4 properties
    void setBorderModeProps(
        OFX::ClipDescriptor* clip,
        const std::string& borderMode
    );
};

#endif // XML_INPUT_MANAGER_H