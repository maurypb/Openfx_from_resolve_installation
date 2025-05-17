# XML-Based OFX Framework Implementation Plan - Version 1

## Introduction

This document outlines a detailed step-by-step implementation plan for Version 1 of the XML-based OFX image processing framework. The plan is structured into small, testable increments to ensure stable progress and minimize risks.

> **Key Principle**: The user should only need to modify XML effect definitions and kernel code files, never the framework code itself.

## Phase 1: Core XML Parsing and Validation

### Step 1.1: Basic XML Schema Design (1-2 days)

**Goal**: Create a well-defined XML schema for effect definitions.

**Tasks**:
1. Design XML schema with inputs, parameters, UI, and kernel sections
2. Include attribute-based parameters with label/hint as attributes
3. Add border_mode attributes for source inputs
4. Create sample GaussianBlur.xml based on schema

**Test Criteria**:
- XML schema is complete and documented
- Sample XML is valid against schema

### Step 1.2: XMLEffectDefinition Class Implementation (2-3 days)

**Goal**: Create a robust class to parse and validate XML effect definitions.

**Tasks**:
1. Implement basic XMLEffectDefinition class with constructors
2. Add parsing for effect metadata (name, category, description)
3. Add parsing for input sources including border_mode attributes
4. Add parsing for parameters with all attributes
5. Add parsing for UI organization
6. Add parsing for identity conditions
7. Add parsing for kernel definitions

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
        std::string borderMode; // "clamp", "repeat", "mirror", "black"
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
        // Additional fields for special parameter types (choice, curve, etc.)
    };
    
    // Kernel definition structure
    struct KernelDef {
        std::string platform; // "cuda", "opencl", "metal"
        std::string file;
        int executions;
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
    std::vector<IdentityCondition> getIdentityConditions() const;
    std::vector<UIPage> getUIPages() const;
    
private:
    std::string _name;
    std::string _category;
    std::string _description;
    std::vector<InputDef> _inputs;
    std::vector<ParameterDef> _parameters;
    std::vector<KernelDef> _kernels;
    std::vector<IdentityCondition> _identityConditions;
    std::vector<UIPage> _uiPages;
    
    void parseXML(const std::string& filename);
};
```

**Test Criteria**:
- XML parser correctly loads all elements and attributes
- Error handling for invalid XML files works correctly
- Accessors return correct values

### Step 1.3: Unit Tests for XML Parsing (1-2 days)

**Goal**: Create comprehensive tests for XML parsing.

**Tasks**:
1. Create test suite for XMLEffectDefinition
2. Test all getter methods with various XML inputs
3. Test error handling with malformed XML

**Test Criteria**:
- All tests pass with valid XML
- Invalid XML is rejected with appropriate error messages
- Edge cases (optional attributes, missing sections) are handled correctly

## Phase 2: OFX Parameter Creation

### Step 2.1: XMLParameterManager Class (2-3 days)

**Goal**: Create a class to map XML parameter definitions to OFX parameters.

**Tasks**:
1. Implement XMLParameterManager class
2. Add support for creating Double, Int, Boolean parameters
3. Add support for creating Choice and Curve parameters
4. Add UI organization (pages, columns)

**Implementation**:
```cpp
class XMLParameterManager {
public:
    // Map XML parameters to OFX
    bool createParameters(
        const XMLEffectDefinition& xmlDef,
        OFX::ImageEffectDescriptor& desc,
        std::map<std::string, OFX::PageParamDescriptor*>& pages
    );
    
    // Map UI organization
    bool organizeUI(
        const XMLEffectDefinition& xmlDef,
        OFX::ImageEffectDescriptor& desc,
        std::map<std::string, OFX::PageParamDescriptor*>& pages
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
        const std::string& name,
        OFX::ImageEffectDescriptor& desc
    );
};
```

**Test Criteria**:
- Parameters created match XML definitions
- Parameter properties (default, range, labels) are correctly set
- UI organization is applied correctly

### Step 2.2: XMLInputManager Class (1-2 days)

**Goal**: Create a class to map XML input definitions to OFX clips.

**Tasks**:
1. Implement XMLInputManager class
2. Add support for creating source clips with proper labels
3. Add support for optional clips
4. Store border mode information for each clip

**Implementation**:
```cpp
class XMLInputManager {
public:
    // Map XML inputs to OFX clips
    bool createInputs(
        const XMLEffectDefinition& xmlDef,
        OFX::ImageEffectDescriptor& desc,
        std::map<std::string, std::string>& clipBorderModes
    );
    
private:
    // Create a single clip
    OFX::ClipDescriptor* createClip(
        const XMLEffectDefinition::InputDef& inputDef,
        OFX::ImageEffectDescriptor& desc
    );
};
```

**Test Criteria**:
- Clips created match XML definitions
- Optional clips are properly flagged
- Border modes are correctly stored for each clip

### Step 2.3: Integration with BlurPluginFactory (1-2 days)

**Goal**: Create non-destructive integration test with existing plugin.

**Tasks**:
1. Create a test harness in BlurPluginFactory
2. Add XML-based parameter and clip creation alongside existing code
3. Add logging to compare XML vs. manual results

**Implementation**:
```cpp
void BlurPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context) {
    // Try XML approach (non-destructively)
    bool xmlSuccess = false;
    try {
        XMLEffectDefinition xmlDef("GaussianBlur.xml");
        
        // Create clips
        std::map<std::string, std::string> clipBorderModes;
        XMLInputManager inputManager;
        xmlSuccess = inputManager.createInputs(xmlDef, p_Desc, clipBorderModes);
        Logger::getInstance().logMessage("XML input creation %s", xmlSuccess ? "succeeded" : "failed");
        
        // Create parameters and UI organization
        if (xmlSuccess) {
            std::map<std::string, OFX::PageParamDescriptor*> pages;
            XMLParameterManager paramManager;
            xmlSuccess = paramManager.createParameters(xmlDef, p_Desc, pages);
            Logger::getInstance().logMessage("XML parameter creation %s", xmlSuccess ? "succeeded" : "failed");
            
            if (xmlSuccess) {
                xmlSuccess = paramManager.organizeUI(xmlDef, p_Desc, pages);
                Logger::getInstance().logMessage("XML UI organization %s", xmlSuccess ? "succeeded" : "failed");
            }
        }
    } catch (const std::exception& e) {
        Logger::getInstance().logMessage("XML error: %s", e.what());
        xmlSuccess = false;
    }
    
    // If XML approach failed, use original implementation
    if (!xmlSuccess) {
        // Original implementation here...
    }
}
```

**Test Criteria**:
- XML-based clips and parameters match manually-created ones
- Log comparison shows equivalence
- Fallback to original works correctly

## Phase 3: Dynamic Effect Base Class

### Step 3.1: GenericEffect Base Class (2-3 days)

**Goal**: Create a base class for XML-defined effects.

**Tasks**:
1. Implement GenericEffect extending OFX::ImageEffect
2. Add dynamic parameter storage for various types
3. Add dynamic clip storage including border modes
4. Add methods to load from XMLEffectDefinition

**Implementation**:
```cpp
class GenericEffect : public OFX::ImageEffect {
protected:
    // XML definition
    XMLEffectDefinition _xmlDef;
    
    // Output clip
    OFX::Clip* _dstClip;
    
    // Input source clips with their border modes
    struct InputClip {
        OFX::Clip* clip;
        std::string borderMode;
    };
    std::map<std::string, InputClip> _srcClips;
    
    // Parameter storage by type
    std::map<std::string, OFX::DoubleParam*> _doubleParams;
    std::map<std::string, OFX::IntParam*> _intParams;
    std::map<std::string, OFX::BooleanParam*> _boolParams;
    std::map<std::string, OFX::ChoiceParam*> _choiceParams;
    std::map<std::string, OFX::RGBParam*> _colorParams;
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
    // Collect parameter values at current time
    std::map<std::string, double> collectDoubleParameters(double time);
    std::map<std::string, int> collectIntParameters(double time);
    std::map<std::string, bool> collectBoolParameters(double time);
    // Similar methods for other parameter types
};
```

**Test Criteria**:
- GenericEffect successfully loads parameters from XML
- Parameters can be accessed and used in the effect
- Clips are properly loaded with border modes

### Step 3.2: Identity Condition Implementation (1 day)

**Goal**: Implement XML-driven identity condition checking.

**Tasks**:
1. Implement isIdentity method in GenericEffect
2. Process identity conditions from XML definition

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
            } // Other operators...
        } 
        // Similar checks for other parameter types...
        
        // If condition is met, use source clip as identity
        if (conditionMet) {
            // Find the main source clip
            for (const auto& entry : _srcClips) {
                if (entry.first == "source") {
                    identityClip = entry.second.clip;
                    identityTime = args.time;
                    return true;
                }
            }
        }
    }
    
    return false;
}
```

**Test Criteria**:
- Identity conditions from XML work correctly
- Different operators function as expected
- Identity behavior matches original plugin

## Phase 4: Image Processing and Kernel Management

### Step 4.1: GenericProcessor Class (2-3 days)

**Goal**: Create a processor that handles dynamic parameter passing.

**Tasks**:
1. Implement GenericProcessor extending OFX::ImageProcessor
2. Add parameter collection from GenericEffect
3. Add methods for processing GPU kernels
4. Handle source border modes

**Implementation**:
```cpp
class GenericProcessor : public OFX::ImageProcessor {
private:
    GenericEffect& _effect;
    
    // Parameter values collected from the effect
    std::map<std::string, double> _doubleParams;
    std::map<std::string, int> _intParams;
    std::map<std::string, bool> _boolParams;
    // Other parameter types...
    
    // Source images with border modes
    struct SourceImage {
        OFX::Image* image;
        std::string borderMode;
    };
    std::map<std::string, SourceImage> _srcImages;
    
public:
    explicit GenericProcessor(GenericEffect& p_Effect);
    
    // Set source images
    void setSrcImg(const std::string& name, OFX::Image* p_SrcImg, const std::string& borderMode);
    
    // Set parameters
    void setParameters(
        const std::map<std::string, double>& doubleParams,
        const std::map<std::string, int>& intParams,
        const std::map<std::string, bool>& boolParams
        // Other parameter types...
    );
    
    // GPU processing methods
    virtual void processImagesCUDA() override;
    virtual void processImagesOpenCL() override;
    virtual void processImagesMetal() override;
    
    // CPU fallback
    virtual void multiThreadProcessImages(OfxRectI procWindow) override;
};
```

**Test Criteria**:
- Processor correctly handles dynamic parameters
- Source images with border modes are properly set up
- Processing methods pass parameters to kernels

### Step 4.2: KernelManager Implementation (2-3 days)

**Goal**: Create a system to manage GPU kernel execution.

**Tasks**:
1. Implement KernelManager class
2. Add support for loading CUDA kernels
3. Add support for loading OpenCL kernels
4. Add support for loading Metal kernels

**Implementation**:
```cpp
class KernelManager {
public:
    // CUDA kernel execution
    static void executeCUDAKernel(
        const std::string& kernelFile,
        void* stream,
        int width, int height,
        int executionNumber, int totalExecutions,
        const std::map<std::string, float*>& sourceBuffers,
        const std::map<std::string, int>& borderModes,
        float* outputBuffer,
        const std::map<std::string, double>& doubleParams,
        const std::map<std::string, int>& intParams,
        const std::map<std::string, bool>& boolParams
    );
    
    // Similar methods for OpenCL and Metal
    
private:
    // Helper methods for kernel loading
    static void* loadCUDAKernel(const std::string& kernelFile);
    static void* loadOpenCLKernel(const std::string& kernelFile);
    static void* loadMetalKernel(const std::string& kernelFile);
};
```

**Test Criteria**:
- Kernels are correctly loaded from files
- Parameters are passed to kernels in the right order
- Border modes are correctly handled

### Step 4.3: Render Method Implementation (1-2 days)

**Goal**: Implement the render method in GenericEffect.

**Tasks**:
1. Implement render method in GenericEffect
2. Set up processor with parameters and images
3. Call processor to execute the kernel

**Implementation**:
```cpp
void GenericEffect::render(const OFX::RenderArguments& args) {
    // Create processor
    GenericProcessor processor(*this);
    
    // Get the destination image
    std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
    processor.setDstImg(dst.get());
    
    // Set source images with their border modes
    for (const auto& entry : _srcClips) {
        const std::string& clipName = entry.first;
        const InputClip& inputClip = entry.second;
        
        if (inputClip.clip && inputClip.clip->isConnected()) {
            std::unique_ptr<OFX::Image> src(inputClip.clip->fetchImage(args.time));
            processor.setSrcImg(clipName, src.get(), inputClip.borderMode);
        }
    }
    
    // Collect parameters
    std::map<std::string, double> doubleParams = collectDoubleParameters(args.time);
    std::map<std::string, int> intParams = collectIntParameters(args.time);
    std::map<std::string, bool> boolParams = collectBoolParameters(args.time);
    // Collect other parameter types...
    
    // Set parameters on processor
    processor.setParameters(doubleParams, intParams, boolParams);
    
    // Set render window and arguments
    processor.setRenderWindow(args.renderWindow);
    processor.setGPURenderArgs(args);
    
    // Process the image
    processor.process();
}
```

**Test Criteria**:
- Render correctly sets up the processor
- All parameters are collected and passed
- Source images with border modes are correctly handled

### Step 4.4: CUDA Kernel Implementation (2-3 days)

**Goal**: Create sample CUDA kernel with standard entry point.

**Tasks**:
1. Implement GaussianBlur.cu with process() entry point
2. Add border mode handling
3. Implement parameter processing

**Implementation**:
```cpp
// In GaussianBlur.cu

// Helper functions for border handling
__device__ float4 sampleWithBorderMode(float* buffer, int width, int height, float x, float y, int borderMode) {
    // Border mode implementation
    // ...
}

// Standard entry point - must match this signature
__global__ void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    // Source buffers with border modes
    float* sourceBuffer, int sourceBorderMode,
    float* matteBuffer, int matteBorderMode,
    // Output buffer
    float* outputBuffer,
    // All effect parameters follow
    float radius,
    int quality
    // Other parameters will be passed in order defined in XML
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Use each source with its appropriate border mode
    // ...
    
    // Implement Gaussian blur using parameters
    // ...
}
```

**Test Criteria**:
- Kernel follows standard entry point convention
- Border modes are correctly used
- Parameters are correctly used in the processing

## Phase 5: Plugin Factory and Testing

### Step 5.1: XMLEffectFactory Implementation (2-3 days)

**Goal**: Create a factory that generates plugins from XML.

**Tasks**:
1. Implement XMLEffectFactory extending PluginFactoryHelper
2. Add describe and describeInContext methods
3. Add createInstance method
4. Set appropriate OFX metadata

**Implementation**:
```cpp
class XMLEffectFactory : public OFX::PluginFactoryHelper<XMLEffectFactory> {
private:
    std::string _xmlFile;
    XMLEffectDefinition _xmlDef;
    
public:
    XMLEffectFactory(const std::string& xmlFile)
        : OFX::PluginFactoryHelper<XMLEffectFactory>(
            /* Extract plugin ID, version, etc. from XML */),
          _xmlFile(xmlFile),
          _xmlDef(xmlFile)
    {
    }
    
    virtual void describe(OFX::ImageEffectDescriptor& desc) override {
        // Set labels from XML
        desc.setLabels(
            _xmlDef.getName().c_str(),
            _xmlDef.getName().c_str(),
            _xmlDef.getName().c_str()
        );
        desc.setPluginGrouping(_xmlDef.getCategory().c_str());
        desc.setPluginDescription(_xmlDef.getDescription().c_str());
        
        // Set basic properties
        desc.addSupportedContext(OFX::eContextFilter);
        desc.addSupportedContext(OFX::eContextGeneral);
        desc.addSupportedBitDepth(OFX::eBitDepthFloat);
        
        // Set GPU support
        desc.setSupportsOpenCLRender(true);
        #ifndef __APPLE__
        desc.setSupportsCudaRender(true);
        #endif
        #ifdef __APPLE__
        desc.setSupportsMetalRender(true);
        #endif
    }
    
    virtual void describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum context) override {
        // Create inputs
        std::map<std::string, std::string> clipBorderModes;
        XMLInputManager inputManager;
        inputManager.createInputs(_xmlDef, desc, clipBorderModes);
        
        // Create parameters and UI
        std::map<std::string, OFX::PageParamDescriptor*> pages;
        XMLParameterManager paramManager;
        paramManager.createParameters(_xmlDef, desc, pages);
        paramManager.organizeUI(_xmlDef, desc, pages);
    }
    
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle handle, OFX::ContextEnum context) override {
        return new GenericEffect(handle, _xmlFile);
    }
};
```

**Test Criteria**:
- Factory creates plugins with the correct metadata
- Parameters and clips are correctly defined
- Plugin instances are correctly created

### Step 5.2: Plugin Registration System (1-2 days)

**Goal**: Create a system to automatically register XML-defined plugins.

**Tasks**:
1. Implement plugin registration in getPluginIDs
2. Add XML file discovery in plugin directory
3. Add error handling for XML loading

**Implementation**:
```cpp
void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& p_FactoryArray) {
    // Find XML effect definitions in plugin directory
    std::vector<std::string> xmlFiles = findXMLFiles();
    
    // Create a factory for each valid XML file
    for (const auto& xmlFile : xmlFiles) {
        try {
            XMLEffectFactory* factory = new XMLEffectFactory(xmlFile);
            p_FactoryArray.push_back(factory);
        } catch (const std::exception& e) {
            Logger::getInstance().logMessage("Failed to load XML effect: %s - %s", 
                                           xmlFile.c_str(), e.what());
        }
    }
}
```

**Test Criteria**:
- XML-defined plugins are automatically registered
- Invalid XML files are handled gracefully
- Multiple plugins can be loaded from one directory

### Step 5.3: Integration Testing (2-3 days)

**Goal**: Test the framework with a complete example.

**Tasks**:
1. Create complete GaussianBlur.xml
2. Create matching CUDA, OpenCL, and Metal kernels
3. Test in various OFX hosts
4. Compare results with original plugin

**Test Criteria**:
- Plugin loads correctly in OFX hosts
- Parameters appear with correct labels and ranges
- Processing produces correct results
- Performance is comparable to original plugin

## Phase 6: Documentation and Packaging

### Step 6.1: Developer Documentation (1-2 days)

**Goal**: Create documentation for framework developers.

**Tasks**:
1. Document XML schema with examples
2. Document class architecture
3. Document build process
4. Create developer guide

### Step 6.2: User Documentation (1-2 days)

**Goal**: Create documentation for effect authors.

**Tasks**:
1. Document XML format with examples
2. Document kernel entry point requirements
3. Create step-by-step tutorial
4. Create troubleshooting guide

### Step 6.3: Example Effects (1-2 days)

**Goal**: Create example effects to demonstrate the framework.

**Tasks**:
1. Create Blur example (single kernel)
2. Create Sharpen example (single kernel)
3. Create Color Correction example (single kernel)
4. Create templates for new effects

## Timeline Summary

**Phase 1: Core XML Parsing and Validation** (4-7 days)
- XML Schema Design
- XMLEffectDefinition Class
- Unit Tests

**Phase 2: OFX Parameter Creation** (4-7 days)
- XMLParameterManager Class
- XMLInputManager Class
- Integration with BlurPluginFactory

**Phase 3: Dynamic Effect Base Class** (3-4 days)
- GenericEffect Base Class
- Identity Condition Implementation

**Phase 4: Image Processing and Kernel Management** (7-11 days)
- GenericProcessor Class
- KernelManager Implementation
- Render Method Implementation
- CUDA Kernel Implementation

**Phase 5: Plugin Factory and Testing** (5-8 days)
- XMLEffectFactory Implementation
- Plugin Registration System
- Integration Testing

**Phase 6: Documentation and Packaging** (3-6 days)
- Developer Documentation
- User Documentation
- Example Effects

**Total Estimated Time**: 26-43 days

## Conclusion

This implementation plan provides a structured approach to building Version 1 of the XML-based OFX framework. By breaking the work into small, testable increments, we ensure that progress is steady and verifiable.

The plan emphasizes:
1. Robust XML parsing and validation
2. Parameter and clip management
3. Simplified kernel interface with standard entry point
4. Automatic parameter passing to kernels
5. Border mode handling for sources
6. Comprehensive testing at each step

Following this plan will result in a framework that allows artists to create new OFX plugins by simply writing XML definitions and kernel code, without needing to understand the underlying OFX C++ infrastructure.
