#include "GenericEffectFactory.h"
#include "XMLParameterManager.h"
#include "XMLInputManager.h"
#include "GenericEffect.h"
#include "../../Logger.h"
#include <stdexcept>
#include <sstream>

GenericEffectFactory::GenericEffectFactory(const std::string& xmlFile) 
    : OFX::PluginFactoryHelper<GenericEffectFactory>(generatePluginIdentifier(xmlFile), 1, 0),
      m_xmlDef(xmlFile), m_xmlFilePath(xmlFile) {
    
    // Store the generated identifier
    m_pluginIdentifier = generatePluginIdentifier(xmlFile);
}

GenericEffectFactory::~GenericEffectFactory() {
    // Destructor - cleanup if needed
}

void GenericEffectFactory::describe(OFX::ImageEffectDescriptor& p_Desc) {
    Logger::getInstance().logMessage("GenericEffectFactory::describe called");
    Logger::getInstance().logMessage("  Plugin identifier: %s", getPluginIdentifier().c_str());
    Logger::getInstance().logMessage("  Effect name from XML: %s", m_xmlDef.getName().c_str());

    // Set basic effect info from XML
    p_Desc.setLabels(m_xmlDef.getName().c_str(), 
                    m_xmlDef.getName().c_str(), 
                    m_xmlDef.getName().c_str());
    p_Desc.setPluginGrouping(m_xmlDef.getCategory().c_str());
    p_Desc.setPluginDescription(m_xmlDef.getDescription().c_str());
    
    // Set up basic OFX properties
    setupBasicProperties(p_Desc);
    
    // Set up GPU support
    setupGPUSupport(p_Desc);
}

void GenericEffectFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, 
                                            OFX::ContextEnum /*p_Context*/) {
    try {
        Logger::getInstance().logMessage("GenericEffectFactory::describeInContext called - TESTING XML CLIPS");
        
        // TRY XML CLIPS WITH DEBUG LOGGING
        std::map<std::string, std::string> clipBorderModes;
        XMLInputManager inputManager;
        
        if (!inputManager.createInputs(m_xmlDef, p_Desc, clipBorderModes)) {
            Logger::getInstance().logMessage("XML clips failed, falling back to manual clips");
            
            // FALLBACK: Create clips manually if XML fails
            OFX::ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
            srcClip->addSupportedComponent(OFX::ePixelComponentRGBA);
            srcClip->setSupportsTiles(false);

            OFX::ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
            dstClip->addSupportedComponent(OFX::ePixelComponentRGBA);
            dstClip->setSupportsTiles(false);
            
            OFX::ClipDescriptor* maskClip = p_Desc.defineClip("mask");
            maskClip->addSupportedComponent(OFX::ePixelComponentRGBA);
            maskClip->setSupportsTiles(false);
            maskClip->setOptional(true);
            maskClip->setIsMask(true);
            
            Logger::getInstance().logMessage("✓ Fallback manual clips created");
        } else {
            Logger::getInstance().logMessage("✓ XML clips created successfully!");
        }

        // CREATE PARAMETERS FROM XML (this works!)
        std::map<std::string, OFX::PageParamDescriptor*> pages;
        XMLParameterManager paramManager;
        
        if (!paramManager.createParameters(m_xmlDef, p_Desc, pages)) {
            throw std::runtime_error("Failed to create parameters from XML");
        }
        
        if (!paramManager.organizeUI(m_xmlDef, p_Desc, pages)) {
            throw std::runtime_error("Failed to organize UI from XML");
        }

        Logger::getInstance().logMessage("✓ XML parameters created successfully");
        
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("✗ GenericEffectFactory::describeInContext failed: %s", e.what());
        throw;
    }
}

OFX::ImageEffect* GenericEffectFactory::createInstance(OfxImageEffectHandle p_Handle, 
                                                      OFX::ContextEnum /*p_Context*/) {
    Logger::getInstance().logMessage("GenericEffectFactory::createInstance called");
    Logger::getInstance().logMessage("  Factory XML file: %s", m_xmlFilePath.c_str());
    
    return new GenericEffect(p_Handle, m_xmlFilePath);
}

std::string GenericEffectFactory::generatePluginIdentifier(const std::string& xmlFile) {
    // Create identifier from filename
    std::string basename = xmlFile;
    size_t lastSlash = basename.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        basename = basename.substr(lastSlash + 1);
    }
    size_t lastDot = basename.find_last_of(".");
    if (lastDot != std::string::npos) {
        basename = basename.substr(0, lastDot);
    }
    
    std::ostringstream identifier;
    identifier << "com.xmlframework." << basename;
    return identifier.str();
}

std::string GenericEffectFactory::generatePluginIdentifier() {
    // Generate unique identifier from effect name and category
    std::string category = m_xmlDef.getCategory();
    std::string name = m_xmlDef.getName();
    
    // Convert to lowercase and replace spaces with dots
    std::transform(category.begin(), category.end(), category.begin(), ::tolower);
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    
    std::replace(category.begin(), category.end(), ' ', '.');
    std::replace(name.begin(), name.end(), ' ', '.');
    
    std::ostringstream identifier;
    identifier << "com.xmlframework." << category << "." << name;
    
    return identifier.str();
}

void GenericEffectFactory::setupBasicProperties(OFX::ImageEffectDescriptor& p_Desc) {
    // Add supported contexts
    p_Desc.addSupportedContext(OFX::eContextFilter);
    p_Desc.addSupportedContext(OFX::eContextGeneral);
    
    // Add supported pixel formats
    p_Desc.addSupportedBitDepth(OFX::eBitDepthFloat);
    
    // Set basic flags
    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(false);
    p_Desc.setSupportsTiles(false);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(false);
    
    // Indicates that plugin output depends on pixel location
    p_Desc.setNoSpatialAwareness(false);
}

void GenericEffectFactory::setupGPUSupport(OFX::ImageEffectDescriptor& p_Desc) {
    // Check what kernels are available in XML and set appropriate flags
    auto kernels = m_xmlDef.getKernels();
    
    bool hasCuda = false;
    bool hasOpenCL = false;
    bool hasMetal = false;
    
    for (const auto& kernel : kernels) {
        if (kernel.platform == "cuda") {
            hasCuda = true;
        } else if (kernel.platform == "opencl") {
            hasOpenCL = true;
        } else if (kernel.platform == "metal") {
            hasMetal = true;
        }
    }
    
    // Set support flags based on available kernels
    if (hasOpenCL) {
        p_Desc.setSupportsOpenCLRender(true);
    }
    
#ifndef __APPLE__
    if (hasCuda) {
        p_Desc.setSupportsCudaRender(true);
        p_Desc.setSupportsCudaStream(true);
    }
#endif

#ifdef __APPLE__
    if (hasMetal) {
        p_Desc.setSupportsMetalRender(true);
    }
#endif
}