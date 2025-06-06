#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"
#include "ofxDrawSuite.h"
#include "ofxsSupportPrivate.h"
#include "src/core/GenericEffectFactory.h"
#include "include/pugixml/pugixml.hpp"
#include "Logger.h"

void testGenericEffectFactory() {
    Logger::getInstance().logMessage("=== Testing GenericEffectFactory ===");
    
    try {
        // TODO auto-discover xml files in effects/ folder
        std::string xmlPath = "/mnt/tank/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/xml_driven_ofx_framework_v0.0/effects/TestBlurV2.xml";
        
        Logger::getInstance().logMessage("Creating GenericEffectFactory...");
        GenericEffectFactory factory(xmlPath);
        Logger::getInstance().logMessage("✓ GenericEffectFactory created successfully");
        
        // Test XML loading
        const XMLEffectDefinition& xmlDef = factory.getXMLDefinition();
        Logger::getInstance().logMessage("✓ XML definition retrieved");
        Logger::getInstance().logMessage("  Effect name: %s", xmlDef.getName().c_str());
        Logger::getInstance().logMessage("  Effect category: %s", xmlDef.getCategory().c_str());
        Logger::getInstance().logMessage("  Plugin identifier: %s", factory.getPluginIdentifier().c_str());
        
        // Test parameter parsing
        auto params = xmlDef.getParameters();
        Logger::getInstance().logMessage("  Parameter count: %d", (int)params.size());
        for (const auto& param : params) {
            Logger::getInstance().logMessage("    - %s (%s): default=%.2f", 
                                           param.name.c_str(), param.type.c_str(), param.defaultValue);
        }
        
        // Test input parsing
        auto inputs = xmlDef.getInputs();
        Logger::getInstance().logMessage("  Input count: %d", (int)inputs.size());
        for (const auto& input : inputs) {
            Logger::getInstance().logMessage("    - %s (optional: %s, border: %s)", 
                                           input.name.c_str(), 
                                           input.optional ? "true" : "false",
                                           input.borderMode.c_str());
        }
        
        // Test kernel parsing
        auto kernels = xmlDef.getKernels();
        Logger::getInstance().logMessage("  Kernel count: %d", (int)kernels.size());
        for (const auto& kernel : kernels) {
            Logger::getInstance().logMessage("    - %s: %s", kernel.platform.c_str(), kernel.file.c_str());
        }
        
        Logger::getInstance().logMessage("✓ GenericEffectFactory test completed successfully");
        
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("✗ GenericEffectFactory test failed: %s", e.what());
    }
    
    Logger::getInstance().logMessage("=== GenericEffectFactory Test Complete ===");
}

// Plugin factory for the XML-driven framework
class GenericPluginFactory : public OFX::PluginFactoryHelper<GenericPluginFactory> {
public:
    GenericPluginFactory() : OFX::PluginFactoryHelper<GenericPluginFactory>("com.xmlframework.GenericPlugin", 1, 0) {
        // Test the framework on startup
        testGenericEffectFactory();
    }
    
    virtual void describe(OFX::ImageEffectDescriptor& p_Desc) override {
        // This factory is just for testing - the real plugins come from GenericEffectFactory
        p_Desc.setLabels("XML Framework Test", "XML Framework Test", "XML Framework Test");
        p_Desc.setPluginGrouping("Developer");
        p_Desc.setPluginDescription("Test plugin for XML framework - real effects come from GenericEffectFactory");
        
        p_Desc.addSupportedContext(OFX::eContextFilter);
        p_Desc.addSupportedBitDepth(OFX::eBitDepthFloat);
        p_Desc.setSingleInstance(false);
        p_Desc.setHostFrameThreading(false);
        p_Desc.setSupportsMultiResolution(false);
        p_Desc.setSupportsTiles(false);
        p_Desc.setSupportsOpenCLRender(true);
#ifndef __APPLE__
        p_Desc.setSupportsCudaRender(true);
        p_Desc.setSupportsCudaStream(true);
#endif
#ifdef __APPLE__
        p_Desc.setSupportsMetalRender(true);
#endif
    }
    
    virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/) override {
        // Minimal implementation - this is just for testing
        OFX::ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
        srcClip->addSupportedComponent(OFX::ePixelComponentRGBA);
        
        OFX::ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
        dstClip->addSupportedComponent(OFX::ePixelComponentRGBA);
    }
    
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle /*p_Handle*/, OFX::ContextEnum /*p_Context*/) override {
        // This test factory doesn't create real instances
        return nullptr;
    }
};


// void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& p_FactoryArray) {
//     // Just register the test factory - no XML loading
//     static GenericPluginFactory testFactory;
//     p_FactoryArray.push_back(&testFactory);
    
//     // Skip the XML plugin for now - just test if basic loading works
// }




void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& p_FactoryArray) {

    Logger::getInstance().logMessage("=== getPluginIDs called ===");

    // Register the test factory (optional - just for framework testing)
    //static GenericPluginFactory testFactory;
    //Logger::getInstance().logMessage("declared testFactory");
    //p_FactoryArray.push_back(&testFactory);
    //Logger::getInstance().logMessage("✓ Test factory registered");
    // Register the real XML-driven plugin
    Logger::getInstance().logMessage("About to try XML plugin registration...");
    std::string xmlPath = "/mnt/tank/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/xml_driven_ofx_framework_v0.0/effects/TestBlurV2.xml";
    
    try {
        Logger::getInstance().logMessage("creating GenericEffectFactory with the xmlPath");
        //static GenericEffectFactory* xmlFactory = new GenericEffectFactory(xmlPath);
        static GenericEffectFactory xmlFactory(xmlPath);
        Logger::getInstance().logMessage("created GenericEffectFactory successfully, going to push_back");
        //p_FactoryArray.push_back(xmlFactory);
        p_FactoryArray.push_back(&xmlFactory);
        Logger::getInstance().logMessage("✓ XML-driven plugin registered successfully");
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("✗ Failed to register XML plugin: %s", e.what());
    }
}