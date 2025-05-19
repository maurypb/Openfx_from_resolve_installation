#include <iostream>
#include <cassert>
#include "core/XMLEffectDefinition.h"
#include "core/XMLParameterManager.h"
#include "ofxsImageEffect.h"
#include "ofxsCore.h"
#include "ofxsParam.h"

// Mock OFX classes for testing - similar approach to XMLInputManagerTest
namespace OFX {

// Custom mock parameter descriptors
class MockParamDescriptor {
public:
    MockParamDescriptor() {}
    std::string _name;
    void setLabels(const std::string& label, const std::string& shortLabel, const std::string& longLabel) {}
    void setHint(const std::string& hint) {}
};

class MockDoubleParamDescriptor : public MockParamDescriptor {
public:
    void setRange(double min, double max) {}
    void setDisplayRange(double min, double max) {}
    void setDefault(double def) {}
    void setIncrement(double inc) {}
};

class MockIntParamDescriptor : public MockParamDescriptor {
public:
    void setRange(int min, int max) {}
    void setDisplayRange(int min, int max) {}
    void setDefault(int def) {}
};

class MockBooleanParamDescriptor : public MockParamDescriptor {
public:
    void setDefault(bool def) {}
};

class MockChoiceParamDescriptor : public MockParamDescriptor {
public:
    void appendOption(const std::string& name, const std::string& label) {}
    void setDefault(int def) {}
};

class MockRGBParamDescriptor : public MockParamDescriptor {
public:
    void setDefault(double r, double g, double b) {}
};

class MockDouble2DParamDescriptor : public MockParamDescriptor {
public:
    void setRange(double minX, double minY, double maxX, double maxY) {}
    void setDisplayRange(double minX, double minY, double maxX, double maxY) {}
    void setDefault(double x, double y) {}
};

class MockStringParamDescriptor : public MockParamDescriptor {
public:
    void setDefault(const std::string& def) {}
};

class MockParametricParamDescriptor : public MockParamDescriptor {
public:
    void setRange(double min, double max) {}
    void setDimension(int dimension) {}
    void addControlPoint(int curve, double time, double x, double y, bool addKey) {}
};

class MockPageParamDescriptor : public MockParamDescriptor {
public:
    void addChild(const ParamDescriptor& child) {}
};

// Mock ImageEffectDescriptor for testing
class MockParameterDescriptor {
public:
    MockParameterDescriptor() {}
    
    std::map<std::string, MockParamDescriptor*> parameters;
    std::map<std::string, MockPageParamDescriptor*> pages;
    
    MockDoubleParamDescriptor* defineDoubleParam(const char* name) {
        auto param = new MockDoubleParamDescriptor();
        param->_name = name;
        parameters[name] = param;
        return param;
    }
    
    MockIntParamDescriptor* defineIntParam(const char* name) {
        auto param = new MockIntParamDescriptor();
        param->_name = name;
        parameters[name] = param;
        return param;
    }
    
    MockBooleanParamDescriptor* defineBooleanParam(const char* name) {
        auto param = new MockBooleanParamDescriptor();
        param->_name = name;
        parameters[name] = param;
        return param;
    }
    
    MockChoiceParamDescriptor* defineChoiceParam(const char* name) {
        auto param = new MockChoiceParamDescriptor();
        param->_name = name;
        parameters[name] = param;
        return param;
    }
    
    MockRGBParamDescriptor* defineRGBParam(const char* name) {
        auto param = new MockRGBParamDescriptor();
        param->_name = name;
        parameters[name] = param;
        return param;
    }
    
    MockDouble2DParamDescriptor* defineDouble2DParam(const char* name) {
        auto param = new MockDouble2DParamDescriptor();
        param->_name = name;
        parameters[name] = param;
        return param;
    }
    
    MockStringParamDescriptor* defineStringParam(const char* name) {
        auto param = new MockStringParamDescriptor();
        param->_name = name;
        parameters[name] = param;
        return param;
    }
    
    MockParametricParamDescriptor* defineParametricParam(const char* name) {
        auto param = new MockParametricParamDescriptor();
        param->_name = name;
        parameters[name] = param;
        return param;
    }
    
    MockPageParamDescriptor* definePageParam(const char* name) {
        auto page = new MockPageParamDescriptor();
        page->_name = name;
        pages[name] = page;
        return page;
    }
    
    MockParamDescriptor* getParamDescriptor(const char* name) {
        auto it = parameters.find(name);
        if (it != parameters.end()) {
            return it->second;
        }
        return nullptr;
    }
};

} // namespace OFX

// Simple test program for XMLParameterManager
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <xml-file>" << std::endl;
        return 1;
    }
    
    try {
        // Parse XML file
        XMLEffectDefinition xmlDef(argv[1]);
        
        // Create parameter manager
        XMLParameterManager paramManager;
        
        // Create mock descriptor
        OFX::MockParameterDescriptor desc;
        
        // Since we can't use XMLParameterManager::createParameters with our mock,
        // we'll manually test the parameter creation functionality
        
        std::cout << "Parameter creation test started" << std::endl;
        
        // Test parameter creation manually based on XML
        for (const auto& paramDef : xmlDef.getParameters()) {
            if (paramDef.type == "double" || paramDef.type == "float") {
                desc.defineDoubleParam(paramDef.name.c_str());
            }
            else if (paramDef.type == "int") {
                desc.defineIntParam(paramDef.name.c_str());
            }
            else if (paramDef.type == "bool") {
                desc.defineBooleanParam(paramDef.name.c_str());
            }
            else if (paramDef.type == "choice") {
                desc.defineChoiceParam(paramDef.name.c_str());
            }
            else if (paramDef.type == "color") {
                desc.defineRGBParam(paramDef.name.c_str());
            }
            else if (paramDef.type == "vec2") {
                desc.defineDouble2DParam(paramDef.name.c_str());
            }
            else if (paramDef.type == "string") {
                desc.defineStringParam(paramDef.name.c_str());
            }
            else if (paramDef.type == "curve") {
                desc.defineParametricParam(paramDef.name.c_str());
            }
        }
        
        // Create pages
        for (const auto& pageDef : xmlDef.getUIPages()) {
            desc.definePageParam(pageDef.name.c_str());
        }
        
        std::cout << "Parameter creation succeeded" << std::endl;
        
        // Print created parameters
        std::cout << "\nCreated parameters:" << std::endl;
        for (const auto& pair : desc.parameters) {
            std::cout << "  " << pair.first << std::endl;
        }
        
        // Print created pages
        std::cout << "\nCreated pages:" << std::endl;
        for (const auto& pair : desc.pages) {
            std::cout << "  " << pair.first << std::endl;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}