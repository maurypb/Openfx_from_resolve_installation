# XML-based OFX Framework Implementation Plan

## Overview

This document outlines a step-by-step implementation plan for transforming the current OFX Gaussian Blur plugin into a flexible XML-based framework. The goal is to create a system where image processing artists can create new OFX plugins by simply writing GPU kernels and XML parameter definitions, without needing to understand OFX C++ infrastructure.

Each implementation stage is designed to be independently testable while maintaining a functional plugin throughout the process. The plan follows a gradual approach that minimizes risk and allows for course correction at each step.

## Prerequisites

- Existing OFX Gaussian Blur plugin with refactored parameter and clip handling
- Working build environment for OFX plugins
- Basic XML parsing library (e.g., TinyXML, RapidXML, or pugixml)

## Stage 1: XML Parameter Definition Parser

**Goal**: Create XML parsing infrastructure without changing existing functionality.

**Steps**:
1. Design XML schema for effect definitions
2. Implement `XMLEffectDefinition` class
3. Create a sample GaussianBlur.xml file that mirrors current parameters

**Example XML Schema**:
```xml
<effect name="GaussianBlur" category="Filter">
  <description>Apply Gaussian blur with optional mask control</description>
  
  <parameters>
    <parameter name="radius" type="double" default="5.0" min="0.0" max="100.0" 
               displayMin="0.0" displayMax="50.0">
      <label>Radius</label>
      <hint>Blur radius in pixels</hint>
    </parameter>
    <!-- Other parameters -->
  </parameters>
</effect>
```

**Implementation**:
```cpp
class XMLEffectDefinition {
public:
    XMLEffectDefinition(const std::string& filename);
    
    struct ParameterDef {
        std::string name;
        std::string type;
        std::string label;
        std::string hint;
        double defaultValue;
        double minValue;
        double maxValue;
        double displayMin;
        double displayMax;
    };
    
    std::string getName() const;
    std::string getCategory() const;
    std::string getDescription() const;
    std::vector<ParameterDef> getParameters() const;
    
private:
    std::string _name;
    std::string _category;
    std::string _description;
    std::vector<ParameterDef> _parameters;
    
    void parseXML(const std::string& filename);
};
```

**Test**: 
- Create a simple test program that loads the XML and prints out parameters
- Verify parsing correctness without modifying the OFX plugin

## Stage 2: XML-to-OFX Parameter Mapper

**Goal**: Create a class to convert XML definitions to OFX parameters without modifying existing code.

**Steps**:
1. Create `XMLParameterMapper` class
2. Implement non-destructive testing in `BlurPluginFactory`
3. Add logging to compare XML-created parameters with manually-created ones

**Implementation**:
```cpp
class XMLParameterMapper {
public:
    bool mapParametersToOFX(
        const XMLEffectDefinition& xmlDef,
        OFX::ImageEffectDescriptor& desc,
        OFX::PageParamDescriptor* page
    );
    
private:
    void mapDoubleParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc,
        OFX::PageParamDescriptor* page
    );
    
    // Similar methods for other parameter types
};
```

**Integration in BlurPluginFactory**:
```cpp
void BlurPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context) {
    // Original implementation (keep as fallback)
    PluginClips::defineBaseClips(p_Desc, kSupportsTiles);
    PluginClips::defineMaskClip(p_Desc, kSupportsTiles);
    OFX::PageParamDescriptor* mainPage = PluginParameters::definePage(p_Desc, PluginParameters::PAGE_MAIN);
    
    // Try XML approach (non-destructively)
    bool xmlSuccess = false;
    try {
        XMLEffectDefinition xmlDef("GaussianBlur.xml");
        XMLParameterMapper mapper;
        xmlSuccess = mapper.mapParametersToOFX(xmlDef, p_Desc, mainPage);
        Logger::getInstance().logMessage("XML parameter mapping %s", xmlSuccess ? "succeeded" : "failed");
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("XML mapping error: %s", e.what());
        xmlSuccess = false;
    }
    
    // If XML approach failed, use original implementation
    if (!xmlSuccess) {
        BlurPluginParameters::defineParameters(p_Desc, mainPage);
    }
}
```

**Test**:
- The plugin should work exactly as before
- Logs should show if XML mapping succeeded or failed
- Compare parameter settings between both approaches

## Stage 3: Clip Definitions from XML

**Goal**: Add support for defining clips from XML.

**Steps**:
1. Extend XML schema to include clip definitions
2. Implement clip mapping in `XMLParameterMapper`
3. Update test function to include clip mapping

**XML Schema Extension**:
```xml
<effect name="GaussianBlur" category="Filter">
  <!-- Previous content -->
  
  <clips>
    <clip name="Source" optional="false" />
    <clip name="Mask" optional="true" isMask="true" />
  </clips>
  
  <!-- Parameters section -->
</effect>
```

**Implementation**:
```cpp
// Add to XMLEffectDefinition
struct ClipDef {
    std::string name;
    bool optional;
    bool isMask;
};

std::vector<ClipDef> getClips() const;

// Add to XMLParameterMapper
bool mapClipsToOFX(
    const XMLEffectDefinition& xmlDef,
    OFX::ImageEffectDescriptor& desc
);
```

**Integration**:
```cpp
// In BlurPluginFactory::describeInContext
// First try XML for clips
bool xmlSuccess = false;
try {
    XMLEffectDefinition xmlDef("GaussianBlur.xml");
    XMLParameterMapper mapper;
    
    // Try to map clips from XML
    xmlSuccess = mapper.mapClipsToOFX(xmlDef, p_Desc);
    Logger::getInstance().logMessage("XML clip mapping %s", xmlSuccess ? "succeeded" : "failed");
    
    // If clips worked, try parameters too
    if (xmlSuccess) {
        OFX::PageParamDescriptor* mainPage = PluginParameters::definePage(p_Desc, PluginParameters::PAGE_MAIN);
        xmlSuccess = mapper.mapParametersToOFX(xmlDef, p_Desc, mainPage);
        Logger::getInstance().logMessage("XML parameter mapping %s", xmlSuccess ? "succeeded" : "failed");
    }
} catch (const std::exception& e) {
    Logger::getInstance().logMessage("XML mapping error: %s", e.what());
    xmlSuccess = false;
}

// If XML approach failed, use original implementation
if (!xmlSuccess) {
    PluginClips::defineBaseClips(p_Desc, kSupportsTiles);
    PluginClips::defineMaskClip(p_Desc, kSupportsTiles);
    OFX::PageParamDescriptor* mainPage = PluginParameters::definePage(p_Desc, PluginParameters::PAGE_MAIN);
    BlurPluginParameters::defineParameters(p_Desc, mainPage);
}
```

**Test**:
- Plugin should work with XML-defined clips
- Verify that the Mask clip is properly set as optional and as a mask

## Stage 4: Generic Effect Parameter Storage

**Goal**: Replace hard-coded parameters in BlurPlugin with dynamic parameter storage.

**Steps**:
1. Create `GenericEffect` base class extending OFX::ImageEffect
2. Add dynamic parameter storage
3. Create `XMLBasedBlurPlugin` that uses this approach
4. Add configurable plugin creation in the factory

**Implementation**:
```cpp
class GenericEffect : public OFX::ImageEffect {
protected:
    // Maps to store parameters of various types
    std::map<std::string, OFX::DoubleParam*> _doubleParams;
    std::map<std::string, OFX::IntParam*> _intParams;
    std::map<std::string, OFX::BooleanParam*> _boolParams;
    // etc.
    
    // Clips
    OFX::Clip* _dstClip;
    std::map<std::string, OFX::Clip*> _srcClips;
    
    // XML definition
    XMLEffectDefinition _xmlDef;
    
public:
    GenericEffect(OfxImageEffectHandle handle, const std::string& xmlFile);
    
    // Load all parameters from XML definition
    void loadParameters();
    
    // OFX override methods
    virtual void render(const OFX::RenderArguments& args) override;
    virtual bool isIdentity(const OFX::IsIdentityArguments& args, 
                          OFX::Clip*& identityClip, double& identityTime) override;
};

// Specific implementation for Blur
class XMLBasedBlurPlugin : public GenericEffect {
public:
    XMLBasedBlurPlugin(OfxImageEffectHandle handle)
        : GenericEffect(handle, "GaussianBlur.xml") {}
        
    // Any blur-specific overrides if needed
};
```

**Factory Integration**:
```cpp
// In BlurPluginFactory
ImageEffect* BlurPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/) {
    // Check if we should use XML-based approach
    bool useXML = true;  // Eventually make this configurable
    
    if (useXML) {
        try {
            return new XMLBasedBlurPlugin(p_Handle);
        } catch (const std::exception& e) {
            Logger::getInstance().logMessage("Failed to create XML-based plugin: %s", e.what());
            // Fall back to original implementation
        }
    }
    
    // Original or fallback
    return new BlurPlugin(p_Handle);
}
```

**Test**:
- Add a preprocessor flag to enable/disable XML-based approach
- Test plugin functionality with both approaches
- Verify parameter values are correctly loaded from XML

## Stage 5: Dynamic Parameter Handling in render/isIdentity

**Goal**: Replace hard-coded parameter access with dynamic handling.

**Steps**:
1. Implement virtual methods in GenericEffect
2. Create XML schema for identity conditions
3. Implement parameter value collection for render

**XML Schema Extension**:
```xml
<effect name="GaussianBlur" category="Filter">
  <!-- Previous content -->
  
  <identity_conditions>
    <condition>
      <parameter name="radius" operator="lessEqual" value="0.0" />
    </condition>
  </identity_conditions>
  
  <!-- Clips and parameters -->
</effect>
```

**Implementation**:
```cpp
// Add to XMLEffectDefinition
struct IdentityCondition {
    std::string paramName;
    std::string op;  // "equal", "lessEqual", "greaterEqual", etc.
    double value;
};

std::vector<IdentityCondition> getIdentityConditions() const;

// GenericEffect implementation
bool GenericEffect::isIdentity(const OFX::IsIdentityArguments& args, 
                             OFX::Clip*& identityClip, double& identityTime) {
    // Check conditions defined in XML
    for (const auto& condition : _xmlDef.getIdentityConditions()) {
        bool conditionMet = false;
        
        // Get the parameter value at this time
        if (_doubleParams.find(condition.paramName) != _doubleParams.end()) {
            double value = _doubleParams[condition.paramName]->getValueAtTime(args.time);
            
            // Check the condition
            if (condition.op == "equal" && value == condition.value) conditionMet = true;
            else if (condition.op == "lessEqual" && value <= condition.value) conditionMet = true;
            else if (condition.op == "greaterEqual" && value >= condition.value) conditionMet = true;
            // etc. for other operators
        }
        
        if (conditionMet) {
            identityClip = _srcClips["Source"];
            identityTime = args.time;
            return true;
        }
    }
    return false;
}

// For render, collect parameters
void GenericEffect::render(const OFX::RenderArguments& args) {
    // Create a processor
    GenericProcessor processor(*this, _xmlDef);
    
    // Collect all parameter values
    std::map<std::string, OFX::ParamValue> paramValues;
    
    // Collect double parameters
    for (const auto& param : _doubleParams) {
        paramValues[param.first] = param.second->getValueAtTime(args.time);
    }
    
    // Same for other parameter types
    
    // Set up and process
    setupAndProcess(processor, args, paramValues);
}
```

**Test**:
- Test isIdentity with a simple condition that mimics current behavior
- Verify parameter collection works correctly in render

## Stage 6: Generic Processor and Kernel Management

**Goal**: Create a processor that dynamically calls kernels based on XML definitions.

**Steps**:
1. Extend XML schema to include kernel definitions
2. Create GenericProcessor class
3. Implement basic KernelManager for CUDA

**XML Schema Extension**:
```xml
<effect name="GaussianBlur" category="Filter">
  <!-- Previous content -->
  
  <kernels>
    <kernel type="cuda" file="GaussianBlur.cu" function="GaussianBlurKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="radius" mapTo="radius" />
        <param name="quality" mapTo="quality" />
        <param name="maskStrength" mapTo="maskStrength" />
        <param name="input" type="image" role="input" />
        <param name="mask" type="image" role="mask" optional="true" />
        <param name="output" type="image" role="output" />
      </params>
    </kernel>
    
    <!-- Similar entries for OpenCL and Metal -->
  </kernels>
</effect>
```

**Implementation**:
```cpp
// GenericProcessor class
class GenericProcessor : public OFX::ImageProcessor {
private:
    GenericEffect& _effect;
    XMLEffectDefinition& _xmlDef;
    std::map<std::string, OFX::ParamValue> _paramValues;
    
public:
    GenericProcessor(GenericEffect& effect, XMLEffectDefinition& xmlDef)
        : OFX::ImageProcessor(effect), _effect(effect), _xmlDef(xmlDef) {}
    
    void setParamValues(const std::map<std::string, OFX::ParamValue>& values) {
        _paramValues = values;
    }
    
    virtual void processImagesCUDA() override {
        const auto& kernelDef = _xmlDef.getCUDAKernel();
        if (kernelDef.file.empty() || kernelDef.function.empty()) {
            // No CUDA kernel defined, fall back to CPU
            multiThreadProcessImages(renderWindow);
            return;
        }
        
        // Get kernel parameters in the right order
        std::vector<void*> args;
        
        // Frame dimensions first
        const OfxRectI& bounds = _srcImg->getBounds();
        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        args.push_back(&width);
        args.push_back(&height);
        
        // Add effect parameters
        for (const auto& paramDef : kernelDef.params) {
            if (paramDef.mapTo.empty()) continue;
            
            // Get parameter value
            if (_paramValues.find(paramDef.mapTo) != _paramValues.end()) {
                args.push_back(&_paramValues[paramDef.mapTo].value);
            }
        }
        
        // Add image pointers
        float* input = static_cast<float*>(_srcImg->getPixelData());
        float* mask = _maskImg ? static_cast<float*>(_maskImg->getPixelData()) : nullptr;
        float* output = static_cast<float*>(_dstImg->getPixelData());
        
        args.push_back(input);
        args.push_back(mask);
        args.push_back(output);
        
        // Call kernel through KernelManager
        KernelManager::invokeCUDAKernel(
            kernelDef.function, 
            _pCudaStream,
            width, height,
            args
        );
    }
    
    // Similar methods for OpenCL and Metal
};

// Kernel manager class (simplified)
class KernelManager {
public:
    static void invokeCUDAKernel(
        const std::string& function,
        void* stream,
        int width, int height,
        const std::vector<void*>& args
    );
};
```

**Test**:
- Start with a hard-coded implementation in KernelManager
- Compare results with original implementation
- Add detailed logging for parameter passing

## Stage 7: Complete Integration

**Goal**: Fully integrate XML-based approach while maintaining compatibility.

**Steps**:
1. Refine previous components
2. Create unified factory
3. Update build system
4. Add error handling throughout

**Implementation**:
```cpp
// XMLEffectFactory
class XMLEffectFactory : public OFX::PluginFactoryHelper<XMLEffectFactory> {
private:
    XMLEffectDefinition _xmlDef;
    
public:
    XMLEffectFactory(const std::string& xmlFile);
    
    virtual void describe(OFX::ImageEffectDescriptor& desc) override;
    virtual void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context) override;
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum context) override;
};

// Plugin entry point
void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray) {
    // Look for XML effect definitions in plugin directory
    std::vector<std::string> xmlFiles = findXMLEffectDefinitions();
    
    for (const auto& xmlFile : xmlFiles) {
        try {
            // Create a factory for each XML definition
            XMLEffectFactory* factory = new XMLEffectFactory(xmlFile);
            p_FactoryArray.push_back(factory);
        } catch (const std::exception& e) {
            Logger::getInstance().logMessage("Failed to load effect from %s: %s", 
                                           xmlFile.c_str(), e.what());
        }
    }
    
    // Add original hardcoded plugin as fallback
    static BlurPluginFactory blurPlugin;
    p_FactoryArray.push_back(&blurPlugin);
}
```

**Build System Updates**:
- Add rules to compile kernel files specified in XML
- Add XML files to installation package
- Add XML schema validation

**Test**:
- Test with both XML-defined plugins and original implementation
- Verify that parameters and clips work correctly
- Test error handling with invalid XML files

## Stage 8: Multiple Effect Support

**Goal**: Support multiple different effects from XML definitions.

**Steps**:
1. Create additional XML files for different effects
2. Test loading multiple effects
3. Create templates for common effects

**Example Effects**:
- Sharpen.xml
- ColorCorrection.xml
- EdgeDetect.xml

**Test**:
- Verify that multiple effects load correctly
- Test different parameter types
- Test different kernel implementations

## Stage 9: Artist-Friendly Tools

**Goal**: Create tools to make the framework accessible to artists.

**Steps**:
1. Create GUI tool for editing XML files
2. Create templates for common effects
3. Add documentation and examples

**Tools**:
- XML Editor with parameter preview
- Kernel template generator
- Documentation with examples

**Test**:
- Have artists create simple effects without C++ knowledge
- Gather feedback and improve tools

## Conclusion

This implementation plan provides a gradual, testable path to transform the current OFX plugin into a flexible XML-based framework. By focusing on one step at a time and maintaining backward compatibility, the risk is minimized while steadily progressing toward the goal of artist-friendly OFX plugin creation.

The end result will be a system where image processing artists can create new OFX plugins by writing GPU kernels and XML parameter definitions, without needing to understand OFX C++ infrastructure.
