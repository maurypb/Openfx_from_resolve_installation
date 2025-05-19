# XML-Based OFX Framework Implementation Plan

## Introduction

This document outlines a detailed implementation plan for the XML-based OFX image processing framework. The plan is divided into incremental stages, each with clearly defined goals and testable outcomes. This approach minimizes risk by ensuring a working product at each stage while progressively building toward the complete system.

The implementation is separated into two major versions:
- **Version 1**: Single-kernel processing with XML-defined parameters and inputs
- **Version 2**: Multi-kernel processing with sequential kernels and iterative execution

This plan focuses primarily on Version 1 implementation, with forward-compatible design choices that will facilitate Version 2 development in the future.

## Version 1 Implementation Plan

### Stage 1: XML Schema and Parser Development

**Goal**: Create a robust XML schema and parser for effect definitions without modifying existing code.

**Steps**:
1. Define comprehensive XML schema for effect definitions
2. Implement `XMLEffectDefinition` class for parsing and validation
3. Create a sample XML for the existing Gaussian Blur effect
4. Write unit tests for XML parsing and validation

**Implementation**:
```cpp
class XMLEffectDefinition {
public:
    XMLEffectDefinition(const std::string& filename);
    
    // Accessors for effect metadata
    std::string getName() const;
    std::string getCategory() const;
    std::string getDescription() const;
    
    // Input definition structure
    struct InputDef {
        std::string name;
        std::string label;
        bool optional;
    };
    
    // Parameter definition structure
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
    
    // Kernel definition structure
    struct KernelDef {
        std::string name;
        std::string file;
        std::string label;
        int executions;
    };
    
    // Implementation definition structure
    struct ImplementationDef {
        std::string function;
        
        struct ParamMapping {
            std::string name;
            std::string type;
            std::string role;
            bool optional;
        };
        
        std::vector<ParamMapping> params;
    };
    
    // Identity condition structure
    struct IdentityCondition {
        std::string paramName;
        std::string op;
        double value;
    };
    
    // UI organization structures
    struct UIParameter {
        std::string name;
    };
    
    struct UIColumn {
        std::string name;
        std::vector<UIParameter> parameters;
    };
    
    struct UIPage {
        std::string name;
        std::vector<UIColumn> columns;
    };
    
    // Accessors for definitions
    std::vector<InputDef> getInputs() const;
    std::vector<ParameterDef> getParameters() const;
    std::vector<KernelDef> getKernels() const;
    ImplementationDef getCUDAImplementation(const std::string& kernelName) const;
    ImplementationDef getOpenCLImplementation(const std::string& kernelName) const;
    ImplementationDef getMetalImplementation(const std::string& kernelName) const;
    std::vector<IdentityCondition> getIdentityConditions() const;
    std::vector<UIPage> getUIPages() const;
    
private:
    std::string _name;
    std::string _category;
    std::string _description;
    std::vector<InputDef> _inputs;
    std::vector<ParameterDef> _parameters;
    std::vector<KernelDef> _kernels;
    std::map<std::string, ImplementationDef> _cudaImplementations;
    std::map<std::string, ImplementationDef> _openclImplementations;
    std::map<std::string, ImplementationDef> _metalImplementations;
    std::vector<IdentityCondition> _identityConditions;
    std::vector<UIPage> _uiPages;
    
    void parseXML(const std::string& filename);
};
```

**XML Example**:
```xml
<effect name="GaussianBlur" category="Filter">
  <description>Apply Gaussian blur with optional mask control</description>
  
  <inputs>
    <source name="source" label="Input Image" />
    <source name="matte" label="Mask" optional="true" />
  </inputs>
  
  <parameters>
    <parameter name="radius" type="double" default="5.0" min="0.0" max="100.0" 
               displayMin="0.0" displayMax="50.0">
      <label>Radius</label>
      <hint>Blur radius in pixels</hint>
    </parameter>
    <parameter name="quality" type="int" default="8" min="1" max="32" 
               displayMin="1" displayMax="16">
      <label>Quality</label>
      <hint>Number of samples for the blur</hint>
    </parameter>
  </parameters>
  
  <ui>
    <page name="Main">
      <column name="Parameters">
        <parameter>radius</parameter>
        <parameter>quality</parameter>
      </column>
    </page>
  </ui>
  
  <!-- Forward-compatible structure for Version 2 -->
  <kernel name="GaussianBlur" file="GaussianBlur.cu" label="Gaussian Blur" executions="1">
  </kernel>
  
  <identity_conditions>
    <condition>
      <parameter name="radius" operator="lessEqual" value="0.0" />
    </condition>
  </identity_conditions>
  
  <implementations>
    <cuda function="GaussianBlurKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" />
        <param name="totalExecutions" type="int" />
        <param name="radius" type="float" />
        <param name="quality" type="int" />
        <param name="source" type="image" role="input" />
        <param name="matte" type="image" role="mask" optional="true" />
        <param name="output" type="image" role="output" />
      </params>
    </cuda>
  </implementations>
</effect>
```

**Test Criteria**:
- XML parser correctly reads all elements and attributes
- Validation catches missing required elements
- Sample XML for Gaussian Blur parses without errors
- All accessor methods return expected values

### Stage 2: Parameter Mapping Implementation

**Goal**: Create a system to map XML parameter definitions to OFX parameters.

**Steps**:
1. Implement `XMLParameterMapper` class to create OFX parameters from XML
2. Add non-destructive testing in the existing plugin factory
3. Log comparison between XML-created and manually-created parameters
4. Add support for different parameter types (double, int, boolean, choice, curve)

**Implementation**:
```cpp
class XMLParameterMapper {
public:
    // Map XML parameters to OFX
    bool mapParametersToOFX(
        const XMLEffectDefinition& xmlDef,
        OFX::ImageEffectDescriptor& desc,
        OFX::PageParamDescriptor* page
    );
    
    // Map UI organization
    bool mapUIToOFX(
        const XMLEffectDefinition& xmlDef,
        OFX::ImageEffectDescriptor& desc
    );
    
private:
    // Helper methods for different parameter types
    OFX::DoubleParamDescriptor* createDoubleParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    OFX::IntParamDescriptor* createIntParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    OFX::BooleanParamDescriptor* createBooleanParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    OFX::ChoiceParamDescriptor* createChoiceParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    OFX::ParametricParamDescriptor* createCurveParam(
        const XMLEffectDefinition::ParameterDef& paramDef,
        OFX::ImageEffectDescriptor& desc
    );
    
    // Create UI pages and columns
    OFX::PageParamDescriptor* createPage(
        const XMLEffectDefinition::UIPage& pageDef,
        OFX::ImageEffectDescriptor& desc
    );
};
```

**Integration**:
```cpp
void BlurPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context) {
    // Try XML approach (non-destructively)
    bool xmlSuccess = false;
    try {
        XMLEffectDefinition xmlDef("GaussianBlur.xml");
        XMLParameterMapper mapper;
        
        // Map UI structure and parameters
        xmlSuccess = mapper.mapUIToOFX(xmlDef, p_Desc);
        
        if (xmlSuccess) {
            Logger::getInstance().logMessage("XML UI mapping succeeded");
        } else {
            Logger::getInstance().logMessage("XML UI mapping failed");
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
}
```

**Test Criteria**:
- Parameters created from XML match those created manually
- Parameter properties (default, range, labels) are correctly set
- UI organization is correctly applied
- Error handling gracefully falls back to manual implementation
- Log comparison shows parameter equivalence

### Stage 3: Input Mapping Implementation

**Goal**: Add support for defining input sources from XML definitions.

**Steps**:
1. Implement input mapping in `XMLParameterMapper`
2. Update test integration to try XML input definitions
3. Add support for optional inputs
4. Test with the existing GaussianBlur effect

**Implementation**:
```cpp
// Add to XMLParameterMapper
bool XMLParameterMapper::mapInputsToOFX(
    const XMLEffectDefinition& xmlDef,
    OFX::ImageEffectDescriptor& desc
) {
    try {
        // Get input definitions from XML
        const auto& inputDefs = xmlDef.getInputs();
        
        // Create each input
        for (const auto& inputDef : inputDefs) {
            OFX::ClipDescriptor* clip = desc.defineClip(inputDef.name.c_str());
            clip->addSupportedComponent(OFX::ePixelComponentRGBA);
            clip->setTemporalClipAccess(false);
            clip->setSupportsTiles(kSupportsTiles);
            clip->setOptional(inputDef.optional);
            
            // Set mask flag if name contains "mask" or "matte" (case insensitive)
            std::string lowerName = inputDef.name;
            std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
            if (lowerName.find("mask") != std::string::npos || 
                lowerName.find("matte") != std::string::npos) {
                clip->setIsMask(true);
            } else {
                clip->setIsMask(false);
            }
        }
        
        // Define output clip
        OFX::ClipDescriptor* dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
        dstClip->addSupportedComponent(OFX::ePixelComponentRGBA);
        dstClip->setSupportsTiles(kSupportsTiles);
        
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("Error mapping inputs: %s", e.what());
        return false;
    }
}
```

**Integration**:
```cpp
// In BlurPluginFactory::describeInContext
bool xmlSuccess = false;
try {
    XMLEffectDefinition xmlDef("GaussianBlur.xml");
    XMLParameterMapper mapper;
    
    // Try to map inputs from XML
    xmlSuccess = mapper.mapInputsToOFX(xmlDef, p_Desc);
    Logger::getInstance().logMessage("XML input mapping %s", xmlSuccess ? "succeeded" : "failed");
    
    // If inputs worked, try UI and parameters
    if (xmlSuccess) {
        xmlSuccess = mapper.mapUIToOFX(xmlDef, p_Desc);
        Logger::getInstance().logMessage("XML UI and parameter mapping %s", xmlSuccess ? "succeeded" : "failed");
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

**Test Criteria**:
- Inputs defined from XML match those created manually
- Optional and mask flags are correctly set
- Error handling gracefully falls back to manual implementation
- Plugin loads and functions in host applications

### Stage 4: Generic Effect Base Class

**Goal**: Create a generic base class for XML-defined effects.

**Steps**:
1. Implement `GenericEffect` base class extending OFX::ImageEffect
2. Add dynamic parameter storage for various parameter types
3. Create `XMLBasedBlurPlugin` derived from GenericEffect
4. Add configurable plugin creation in the factory

**Implementation**:
```cpp
class GenericEffect : public OFX::ImageEffect {
protected:
    // XML definition
    XMLEffectDefinition _xmlDef;
    
    // Output clip
    OFX::Clip* _dstClip;
    
    // Input source clips
    std::map<std::string, OFX::Clip*> _srcClips;
    
    // Parameter storage by type
    std::map<std::string, OFX::DoubleParam*> _doubleParams;
    std::map<std::string, OFX::IntParam*> _intParams;
    std::map<std::string, OFX::BooleanParam*> _boolParams;
    std::map<std::string, OFX::ChoiceParam*> _choiceParams;
    std::map<std::string, OFX::RGBParam*> _rgbParams;
    std::map<std::string, OFX::StringParam*> _stringParams;
    std::map<std::string, OFX::ParametricParam*> _curveParams;
    
public:
    GenericEffect(OfxImageEffectHandle handle, const std::string& xmlFile);
    virtual ~GenericEffect();
    
    // Load parameters and clips from XML
    void loadParameters();
    void loadInputs();
    
    // OFX override methods
    virtual void render(const OFX::RenderArguments& args) override;
    virtual bool isIdentity(const OFX::IsIdentityArguments& args, 
                          OFX::Clip*& identityClip, double& identityTime) override;
    virtual void changedParam(const OFX::InstanceChangedArgs& args, 
                            const std::string& paramName) override;
    
protected:
    // Version 1: Single kernel processing
    virtual void processSingleKernel(const OFX::RenderArguments& args);
    
    // Version 2: Will override this for multi-kernel
    virtual void process(const OFX::RenderArguments& args) {
        // Default implementation just calls single kernel
        processSingleKernel(args);
    }
    
    // Collect parameter values at current time
    std::map<std::string, OFX::ParamValue> collectParameterValues(double time);
};

// Specific implementation for Blur
class XMLBasedBlurPlugin : public GenericEffect {
public:
    XMLBasedBlurPlugin(OfxImageEffectHandle handle)
        : GenericEffect(handle, "GaussianBlur.xml") {}
};
```

**Implementation of GenericEffect constructor**:
```cpp
GenericEffect::GenericEffect(OfxImageEffectHandle handle, const std::string& xmlFile)
    : OFX::ImageEffect(handle), _xmlDef(xmlFile)
{
    try {
        // Load clips and parameters
        loadInputs();
        loadParameters();
        
        Logger::getInstance().logMessage("GenericEffect created successfully from %s", xmlFile.c_str());
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("Error creating GenericEffect: %s", e.what());
        throw;
    }
}

void GenericEffect::loadInputs() {
    // Get output clip
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    
    // Load source clips
    for (const auto& inputDef : _xmlDef.getInputs()) {
        _srcClips[inputDef.name] = fetchClip(inputDef.name.c_str());
    }
}

void GenericEffect::loadParameters() {
    // Load parameters by type
    for (const auto& paramDef : _xmlDef.getParameters()) {
        if (paramDef.type == "double") {
            _doubleParams[paramDef.name] = fetchDoubleParam(paramDef.name.c_str());
        } else if (paramDef.type == "int") {
            _intParams[paramDef.name] = fetchIntParam(paramDef.name.c_str());
        } else if (paramDef.type == "bool") {
            _boolParams[paramDef.name] = fetchBooleanParam(paramDef.name.c_str());
        } else if (paramDef.type == "choice") {
            _choiceParams[paramDef.name] = fetchChoiceParam(paramDef.name.c_str());
        } else if (paramDef.type == "rgb") {
            _rgbParams[paramDef.name] = fetchRGBParam(paramDef.name.c_str());
        } else if (paramDef.type == "string") {
            _stringParams[paramDef.name] = fetchStringParam(paramDef.name.c_str());
        } else if (paramDef.type == "curve") {
            _curveParams[paramDef.name] = fetchParametricParam(paramDef.name.c_str());
        }
    }
}
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

**Test Criteria**:
- GenericEffect successfully loads parameters from XML
- Parameters can be accessed and used in the effect
- Plugin works with both XML and hard-coded approaches
- Error handling properly falls back to original implementation

### Stage 5: Dynamic Identity Condition Handling

**Goal**: Implement dynamic identity condition checking based on XML.

**Steps**:
1. Implement `isIdentity` method in GenericEffect
2. Process identity conditions from XML definition
3. Test with conditions that match the existing blur behavior
4. Add support for multiple conditions and operators

**Implementation**:
```cpp
bool GenericEffect::isIdentity(const OFX::IsIdentityArguments& args, 
                             OFX::Clip*& identityClip, double& identityTime) {
    // Check identity conditions defined in XML
    for (const auto& condition : _xmlDef.getIdentityConditions()) {
        bool conditionMet = false;
        
        // Check parameter type and get value
        if (_doubleParams.find(condition.paramName) != _doubleParams.end()) {
            double value = _doubleParams[condition.paramName]->getValueAtTime(args.time);
            
            // Check condition with appropriate operator
            if (condition.op == "equal" && value == condition.value) {
                conditionMet = true;
            } else if (condition.op == "notEqual" && value != condition.value) {
                conditionMet = true;
            } else if (condition.op == "lessThan" && value < condition.value) {
                conditionMet = true;
            } else if (condition.op == "lessEqual" && value <= condition.value) {
                conditionMet = true;
            } else if (condition.op == "greaterThan" && value > condition.value) {
                conditionMet = true;
            } else if (condition.op == "greaterEqual" && value >= condition.value) {
                conditionMet = true;
            }
        } else if (_intParams.find(condition.paramName) != _intParams.end()) {
            int value = _intParams[condition.paramName]->getValueAtTime(args.time);
            int intValue = static_cast<int>(condition.value);
            
            // Similar condition checks for int parameters
            if (condition.op == "equal" && value == intValue) {
                conditionMet = true;
            }
            // Other operators...
        } else if (_boolParams.find(condition.paramName) != _boolParams.end()) {
            bool value = _boolParams[condition.paramName]->getValueAtTime(args.time);
            bool boolValue = condition.value != 0.0;
            
            if (condition.op == "equal" && value == boolValue) {
                conditionMet = true;
            }
            // Other operators...
        }
        
        // If condition is met, use source clip as identity
        if (conditionMet) {
            // Find the main source clip
            std::string sourceClipName = "source";
            for (const auto& input : _xmlDef.getInputs()) {
                if (!input.optional && input.name != "matte" && input.name != "mask") {
                    sourceClipName = input.name;
                    break;
                }
            }
            
            if (_srcClips.find(sourceClipName) != _srcClips.end()) {
                identityClip = _srcClips[sourceClipName];
                identityTime = args.time;
                return true;
            }
        }
    }
    
    return false;
}
```

**Test Criteria**:
- Identity conditions from XML work correctly
- Operators (equal, lessEqual, etc.) function as expected
- Parameters of different types can be used in conditions
- Behavior matches the original hard-coded implementation

### Stage 6: Parameter Collection for Render

**Goal**: Implement dynamic parameter collection for the render method.

**Steps**:
1. Implement parameter value collection in GenericEffect
2. Create render method that passes parameters to processor
3. Test parameter collection with the existing blur effect
4. Add support for collecting different parameter types

**Implementation**:
```cpp
// Collect all parameter values at the given time
std::map<std::string, OFX::ParamValue> GenericEffect::collectParameterValues(double time) {
    std::map<std::string, OFX::ParamValue> values;
    
    // Collect double parameters
    for (const auto& param : _doubleParams) {
        double value = param.second->getValueAtTime(time);
        values[param.first] = OFX::ParamValue(value);
    }
    
    // Collect integer parameters
    for (const auto& param : _intParams) {
        int value = param.second->getValueAtTime(time);
        values[param.first] = OFX::ParamValue(value);
    }
    
    // Collect boolean parameters
    for (const auto& param : _boolParams) {
        bool value = param.second->getValueAtTime(time);
        values[param.first] = OFX::ParamValue(value);
    }
    
    // Collect choice parameters
    for (const auto& param : _choiceParams) {
        int value = param.second->getValueAtTime(time);
        values[param.first] = OFX::ParamValue(value);
    }
    
    // Collect RGB parameters
    for (const auto& param : _rgbParams) {
        double r, g, b;
        param.second->getValueAtTime(time, r, g, b);
        values[param.first + "_r"] = OFX::ParamValue(r);
        values[param.first + "_g"] = OFX::ParamValue(g);
        values[param.first + "_b"] = OFX::ParamValue(b);
    }
    
    // Collect string parameters
    for (const auto& param : _stringParams) {
        std::string value = param.second->getValueAtTime(time);
        values[param.first] = OFX::ParamValue(value);
    }
    
    // Curve parameters would require special handling
    // ...
    
    return values;
}

// Render method
void GenericEffect::render(const OFX::RenderArguments& args) {
    // Use process method (supports both Version 1 and future Version 2)
    process(args);
}

// Single-kernel processing (Version 1)
void GenericEffect::processSingleKernel(const OFX::RenderArguments& args) {
    // Create processor
    GenericProcessor processor(*this, _xmlDef);
    
    // Set up processor with the destination clip's pixel data
    processor.setDstImg(this->_dstClip->fetchImage(args.time));
    
    // Set up source images
    for (const auto& input : _xmlDef.getInputs()) {
        const std::string& inputName = input.name;
        if (_srcClips.find(inputName) != _srcClips.end() && _srcClips[inputName]->isConnected()) {
            processor.setSrcImg(inputName, _srcClips[inputName]->fetchImage(args.time));
        }
    }
    
    // Collect parameter values
    std::map<std::string, OFX::ParamValue> paramValues = collectParameterValues(args.time);
    
    // Set up render window
    processor.setRenderWindow(args.renderWindow);
    
    // Set GPU render arguments
    processor.setGPURenderArgs(args);
    
    // Set parameters and process
    processor.setParamValues(paramValues);
    processor.process();
}
```

**Test Criteria**:
- Parameter collection gathers all parameter values correctly
- Parameter values are passed to the processor
- The effect renders correctly with XML-defined parameters
- Render output matches the original hard-coded implementation

### Stage 7: Generic Processor Implementation

**Goal**: Create a processor that can dynamically call GPU kernels based on XML.

**Steps**:
1. Implement `GenericProcessor` class extending OFX::ImageProcessor
2. Create `KernelManager` for handling GPU kernels
3. Implement dynamic parameter mapping to kernel arguments
4. Test with CUDA as initial GPU backend

**Implementation**:
```cpp
class GenericProcessor : public OFX::ImageProcessor {
private:
    GenericEffect& _effect;
    XMLEffectDefinition& _xmlDef;
    std::map<std::string, OFX::ParamValue> _paramValues;
    
    // Input images
    std::map<std::string, OFX::Image*> _srcImgs;
    
public:
    GenericProcessor(GenericEffect& effect, XMLEffectDefinition& xmlDef)
        : OFX::ImageProcessor(effect), _effect(effect), _xmlDef(xmlDef) {}
    
    void setSrcImg(const std::string& name, OFX::Image* srcImg) { 
        _srcImgs[name] = srcImg; 
    }
    
    void setParamValues(const std::map<std::string, OFX::ParamValue>& values) {
        _paramValues = values;
    }
    
    // GPU processing methods
    virtual void processImagesCUDA() override;
    virtual void processImagesOpenCL() override;
    virtual void processImagesMetal() override;
    
    // CPU fallback
    virtual void multiThreadProcessImages(OfxRectI procWindow) override;
};

// CUDA processing implementation
void GenericProcessor::processImagesCUDA() {
    // Get kernel implementation
    const auto& kernelDef = _xmlDef.getKernels().front(); // Version 1: single kernel
    const auto& implDef = _xmlDef.getCUDAImplementation(kernelDef.name);
    
    if (implDef.function.empty()) {
        // No CUDA implementation defined, fall back to CPU
        multiThreadProcessImages(renderWindow);
        return;
    }
    
    // Get frame dimensions
    const OfxRectI& bounds = _dstImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    
    // Prepare kernel parameters in the order defined in XML
    std::vector<void*> kernelParams;
    std::vector<void*> kernelValues; // Hold actual values for parameters
    
    for (const auto& param : implDef.params) {
        if (param.type == "int" && param.name == "width") {
            kernelValues.push_back(new int(width));
            kernelParams.push_back(kernelValues.back());
        } else if (param.type == "int" && param.name == "height") {
            kernelValues.push_back(new int(height));
            kernelParams.push_back(kernelValues.back());
        } else if (param.type == "int" && param.name == "executionNumber") {
            // For Version 1, always 0
            kernelValues.push_back(new int(0));
            kernelParams.push_back(kernelValues.back());
        } else if (param.type == "int" && param.name == "totalExecutions") {
            // For Version 1, always 1
            kernelValues.push_back(new int(1));
            kernelParams.push_back(kernelValues.back());
        } else if (_paramValues.find(param.name) != _paramValues.end()) {
            // Map to parameter value
            if (param.type == "double" || param.type == "float") {
                double value = _paramValues[param.name].as<double>();
                kernelValues.push_back(new float(static_cast<float>(value)));
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
