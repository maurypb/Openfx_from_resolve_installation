#include "XMLInputManager.h"
#include "../../Logger.h"
#include <iostream>
#include <stdexcept>
#include <algorithm> // For std::transform
#include <cctype>    // For ::tolower

XMLInputManager::XMLInputManager() {
    Logger::getInstance().logMessage("XMLInputManager::constructor");
}

XMLInputManager::~XMLInputManager() {
}

bool XMLInputManager::createInputs(
    const XMLEffectDefinition& xmlDef,
    OFX::ImageEffectDescriptor& desc,
    std::map<std::string, std::string>& clipBorderModes
) {
    Logger::getInstance().logMessage("XMLInputManager::starting createInputs");
    
    try {
        Logger::getInstance().logMessage("XMLInputManager: Getting inputs from XML...");
        auto inputs = xmlDef.getInputs();
        Logger::getInstance().logMessage("XMLInputManager: Found %d inputs in XML", (int)inputs.size());
        
        // Create all input clips
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& inputDef = inputs[i];
            Logger::getInstance().logMessage("XMLInputManager: Processing input %d: name='%s', optional=%s", 
                                           (int)i, inputDef.name.c_str(), inputDef.optional ? "true" : "false");
            
            Logger::getInstance().logMessage("XMLInputManager: About to call createClip...");
            OFX::ClipDescriptor* clip = createClip(inputDef, desc);
            Logger::getInstance().logMessage("XMLInputManager: createClip returned: %p", clip);
            
            if (clip) {
                // Store border mode for each clip
                clipBorderModes[inputDef.name] = inputDef.borderMode;
                Logger::getInstance().logMessage("XMLInputManager: Stored border mode '%s' for clip '%s'", 
                                               inputDef.borderMode.c_str(), inputDef.name.c_str());
            } else {
                Logger::getInstance().logMessage("XMLInputManager: ERROR - createClip returned null for '%s'", 
                                               inputDef.name.c_str());
            }
        }
        
        Logger::getInstance().logMessage("XMLInputManager: All input clips processed, creating output clip...");
        
        // Create output clip
        Logger::getInstance().logMessage("XMLInputManager: About to create output clip...");
        OFX::ClipDescriptor* dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
        Logger::getInstance().logMessage("XMLInputManager: defineClip for output returned: %p", dstClip);
        
        if (dstClip) {
            Logger::getInstance().logMessage("XMLInputManager: Setting output clip properties...");
            dstClip->addSupportedComponent(OFX::ePixelComponentRGBA);
            Logger::getInstance().logMessage("XMLInputManager: addSupportedComponent done");
            dstClip->setSupportsTiles(true);
            Logger::getInstance().logMessage("XMLInputManager: setSupportsTiles done");
        } else {
            Logger::getInstance().logMessage("XMLInputManager: ERROR - output clip creation returned null");
            return false;
        }
        
        Logger::getInstance().logMessage("XMLInputManager: createInputs completed successfully");
        return true;
        
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("XMLInputManager: EXCEPTION in createInputs: %s", e.what());
        return false;
    } catch (...) {
        Logger::getInstance().logMessage("XMLInputManager: UNKNOWN EXCEPTION in createInputs");
        return false;
    }
}

OFX::ClipDescriptor* XMLInputManager::createClip(
    const XMLEffectDefinition::InputDef& inputDef,
    OFX::ImageEffectDescriptor& desc
) {
    Logger::getInstance().logMessage("XMLInputManager::createClip called for '%s'", inputDef.name.c_str());
    
    try {
        // Create clip
        Logger::getInstance().logMessage("XMLInputManager: About to call defineClip with name '%s'", inputDef.name.c_str());
        OFX::ClipDescriptor* clip = desc.defineClip(inputDef.name.c_str());
        Logger::getInstance().logMessage("XMLInputManager: defineClip returned: %p", clip);
        
        if (!clip) {
            Logger::getInstance().logMessage("XMLInputManager: ERROR - defineClip returned null");
            return nullptr;
        }
        
        // Set basic properties
        Logger::getInstance().logMessage("XMLInputManager: Setting basic clip properties...");
        clip->addSupportedComponent(OFX::ePixelComponentRGBA);
        Logger::getInstance().logMessage("XMLInputManager: addSupportedComponent done");
        
        clip->setTemporalClipAccess(false);
        Logger::getInstance().logMessage("XMLInputManager: setTemporalClipAccess done");
        
        clip->setSupportsTiles(true);
        Logger::getInstance().logMessage("XMLInputManager: setSupportsTiles done");
        
        clip->setOptional(inputDef.optional);
        Logger::getInstance().logMessage("XMLInputManager: setOptional(%s) done", inputDef.optional ? "true" : "false");
        
        // Set border mode properties using OFX 1.4 approach
        Logger::getInstance().logMessage("XMLInputManager: About to call setBorderModeProps...");
        setBorderModeProps(clip, inputDef.borderMode);
        Logger::getInstance().logMessage("XMLInputManager: setBorderModeProps completed");
        
        // Handle mask clips - if name contains "mask" or "matte" (case insensitive)
        std::string lowerName = inputDef.name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), 
                      [](unsigned char c) { return std::tolower(c); });
                      
        if (lowerName.find("mask") != std::string::npos || 
            lowerName.find("matte") != std::string::npos) {
            Logger::getInstance().logMessage("XMLInputManager: Setting clip as mask for '%s'", inputDef.name.c_str());
            clip->setIsMask(true);
            Logger::getInstance().logMessage("XMLInputManager: setIsMask done");
        }
        
        Logger::getInstance().logMessage("XMLInputManager: createClip completed successfully for '%s'", inputDef.name.c_str());
        return clip;
        
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("XMLInputManager: EXCEPTION in createClip for '%s': %s", inputDef.name.c_str(), e.what());
        return nullptr;
    } catch (...) {
        Logger::getInstance().logMessage("XMLInputManager: UNKNOWN EXCEPTION in createClip for '%s'", inputDef.name.c_str());
        return nullptr;
    }
}

void XMLInputManager::setBorderModeProps(
    OFX::ClipDescriptor* clip,
    const std::string& borderMode
) {
    Logger::getInstance().logMessage("XMLInputManager::setBorderModeProps called with mode '%s'", borderMode.c_str());
    
    try {
        // COMMENT OUT THE PROPERTY SETTING FOR NOW - MIGHT BE CAUSING CRASH
        Logger::getInstance().logMessage("XMLInputManager: Skipping property setting (commented out for debugging)");
        
        /*
        // In OFX 1.4, we can set properties directly using PropertySet
        
        if (borderMode == "repeat") {
            Logger::getInstance().logMessage("XMLInputManager: Setting repeat mode");
            clip->getPropertySet().propSetInt("OfxImageClipPropTileMode", 1, 0);
        }
        else if (borderMode == "mirror") {
            Logger::getInstance().logMessage("XMLInputManager: Setting mirror mode");
            clip->getPropertySet().propSetInt("OfxImageClipPropTileMode", 2, 0);
        }
        else if (borderMode == "black") {
            Logger::getInstance().logMessage("XMLInputManager: Setting black mode");
            clip->getPropertySet().propSetInt("OfxImageClipPropTileMode", 0, 0);
        }
        else if (borderMode == "clamp") {
            Logger::getInstance().logMessage("XMLInputManager: Setting clamp mode");
            clip->getPropertySet().propSetInt("OfxImageClipPropTileMode", 3, 0);
        }
        else {
            Logger::getInstance().logMessage("XMLInputManager: Unknown border mode, defaulting to clamp");
            clip->getPropertySet().propSetInt("OfxImageClipPropTileMode", 3, 0);
        }
        */
        
        Logger::getInstance().logMessage("XMLInputManager: setBorderModeProps completed");
        
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("XMLInputManager: EXCEPTION in setBorderModeProps: %s", e.what());
        throw;
    } catch (...) {
        Logger::getInstance().logMessage("XMLInputManager: UNKNOWN EXCEPTION in setBorderModeProps");
        throw;
    }
}