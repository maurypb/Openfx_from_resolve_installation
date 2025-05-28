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
        // Create all input clips
        for (const auto& inputDef : xmlDef.getInputs()) {
            createClip(inputDef, desc);
            
            // Store border mode for each clip
            clipBorderModes[inputDef.name] = inputDef.borderMode;
        }
        Logger::getInstance().logMessage("XMLInputManager::creating the output clip...");
        // Create output clip
        OFX::ClipDescriptor* dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
        dstClip->addSupportedComponent(OFX::ePixelComponentRGBA);
        dstClip->setSupportsTiles(true);
        
        return true;
    }
    catch (const std::exception& e) {
        Logger::getInstance().logMessage("XMLInputManager::error creating inputs");
        std::cerr << "Error creating inputs: " << e.what() << std::endl;
        return false;
    }
}

OFX::ClipDescriptor* XMLInputManager::createClip(
    const XMLEffectDefinition::InputDef& inputDef,
    OFX::ImageEffectDescriptor& desc
) {
    // Create clip
    OFX::ClipDescriptor* clip = desc.defineClip(inputDef.name.c_str());
    
    // Set basic properties
    clip->addSupportedComponent(OFX::ePixelComponentRGBA);
    clip->setTemporalClipAccess(false);
    clip->setSupportsTiles(true);
    clip->setOptional(inputDef.optional);
    
    // Set border mode properties using OFX 1.4 approach
    setBorderModeProps(clip, inputDef.borderMode);
    
    // Handle mask clips - if name contains "mask" or "matte" (case insensitive)
    std::string lowerName = inputDef.name;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), 
                  [](unsigned char c) { return std::tolower(c); });
                  
    if (lowerName.find("mask") != std::string::npos || 
        lowerName.find("matte") != std::string::npos) {
        clip->setIsMask(true);
    }
    
    return clip;
}

void XMLInputManager::setBorderModeProps(
    OFX::ClipDescriptor* clip,
    const std::string& borderMode
) {
    // In OFX 1.4, we can set properties directly using PropertySet
    
    // The approach below uses OFX 1.4's property system 
    // These are common OFX properties for clip extent handling
    
    // Using this approach instead of the enum since different hosts 
    // might implement these properties differently
    
    if (borderMode == "repeat") {
        // Tell host we want the image repeated at boundaries
        clip->getPropertySet().propSetInt("OfxImageClipPropTileMode", 1, 0); // Set to repeat mode
    }
    else if (borderMode == "mirror") {
        // Tell host we want the image mirrored at boundaries  
        clip->getPropertySet().propSetInt("OfxImageClipPropTileMode", 2, 0); // Set to mirror mode
    }
    else if (borderMode == "black") {
        // Tell host we want black/transparent outside boundaries
        clip->getPropertySet().propSetInt("OfxImageClipPropTileMode", 0, 0); // Set to black mode
    }
    else if (borderMode == "clamp") {
        // Tell host we want edge pixels repeated
        clip->getPropertySet().propSetInt("OfxImageClipPropTileMode", 3, 0); // Set to clamp mode
    }
    else {
        // Default to clamp if unrecognized
        clip->getPropertySet().propSetInt("OfxImageClipPropTileMode", 3, 0); // Set to clamp mode
    }
}