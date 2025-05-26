#include "GenericEffectFactory.h"
#include "XMLParameterManager.h"
#include "XMLInputManager.h"
// Note: GenericEffect.h will be included when we implement it
// #include "GenericEffect.h"
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
        // Use existing XMLInputManager to create clips from XML
        XMLInputManager inputManager;
        std::map<std::string, std::string> clipBorderModes;
        if (!inputManager.createInputs(m_xmlDef, p_Desc, clipBorderModes)) {
            throw std::runtime_error("Failed to create input clips from XML");
        }
        
        // Use existing XMLParameterManager to create parameters from XML
        XMLParameterManager paramManager;
        std::map<std::string, OFX::PageParamDescriptor*> pages;
        if (!paramManager.createParameters(m_xmlDef, p_Desc, pages)) {
            throw std::runtime_error("Failed to create parameters from XML");
        }
        
        // Organize UI from XML
        if (!paramManager.organizeUI(m_xmlDef, p_Desc, pages)) {
            throw std::runtime_error("Failed to organize UI from XML");
        }
    }
    catch (const std::exception& e) {
        // Log error and re-throw
        // TODO: Add proper logging when Logger is available
        throw std::runtime_error(std::string("GenericEffectFactory::describeInContext failed: ") + e.what());
    }
}

OFX::ImageEffect* GenericEffectFactory::createInstance(OfxImageEffectHandle p_Handle, 
                                                      OFX::ContextEnum /*p_Context*/) {
    // TODO: This will create GenericEffect when we implement it
    // return new GenericEffect(p_Handle, m_xmlFilePath);
    
    // For now, return nullptr to avoid compilation errors
    // This will be implemented in Step 3.3
    throw std::runtime_error("GenericEffect not yet implemented - Step 3.3");
    return nullptr;
}

std::string GenericEffectFactory::generatePluginIdentifier(const std::string& xmlFile) {
    // For now, create a simple identifier from filename
    // We'll improve this when we can actually load the XML
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
    // Format: com.xmlframework.category.name
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
    p_Desc.setSupportsMultiResolution(false);  // May be configurable from XML in future
    p_Desc.setSupportsTiles(false);            // May be configurable from XML in future
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(false);
    
    // Indicates that plugin output depends on pixel location
    // (most image effects do depend on spatial information)
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