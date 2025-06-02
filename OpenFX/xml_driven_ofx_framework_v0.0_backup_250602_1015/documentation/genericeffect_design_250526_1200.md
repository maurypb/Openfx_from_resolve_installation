# GenericEffect Architecture Design
*Detailed design for Phase 3 of the XML-driven OFX framework*

## Overview

This document details the architecture for GenericEffect and related classes that replace BlurPlugin's fixed structure with a dynamic, XML-driven approach. This design addresses the core limitations identified in BlurPlugin.cpp analysis.

**Implementation Status**: Phase 3 is being implemented incrementally with thorough validation at each step:
- ✅ **ParameterValue**: Type-safe dynamic parameter storage (Step 3.1)
- ✅ **GenericEffectFactory**: XML-driven plugin factory (Step 3.2)  
- 🔄 **GenericEffect**: Dynamic effect instance (Step 3.3 - Next)
- 📋 **GenericProcessor**: Dynamic rendering (Step 3.5)

## Class Architecture

### ParameterValue (✅ Implemented)
Type-safe parameter value storage for dynamic parameter passing between OFX and GPU kernels.

```cpp
class ParameterValue {
private:
    enum Type { DOUBLE, INT, BOOL, STRING } m_type;
    union {
        double m_double;
        int m_int;
        bool m_bool;
    };
    std::string m_string;

public:
    ParameterValue();
    ParameterValue(double value);
    ParameterValue(int value);
    ParameterValue(bool value);
    ParameterValue(const std::string& value);
    ParameterValue(const char* value);  // Prevents bool conversion ambiguity
    
    // Type-safe accessors
    double asDouble() const;
    int asInt() const;
    bool asBool() const;
    float asFloat() const;
    std::string asString() const;
    std::string getType() const;
    
    // Copy/assignment
    ParameterValue(const ParameterValue& other);
    ParameterValue& operator=(const ParameterValue& other);
};
```

**Key Features**:
- **Union storage** for memory efficiency
- **Type-safe conversions** between all supported types
- **String parsing** for numeric and boolean values
- **Explicit const char* constructor** to avoid accidental bool conversion
- **Comprehensive error handling** (invalid conversions return sensible defaults)

**Testing**: Complete unit test suite validates all type conversions, edge cases, and memory management.

### GenericEffectFactory (✅ Implemented)
Replaces BlurPluginFactory with XML-driven plugin creation. Handles the OFX describe and describeInContext phases.

```cpp
class GenericEffectFactory : public OFX::PluginFactoryHelper<GenericEffectFactory> {
private:
    XMLEffectDefinition m_xmlDef;
    std::string m_xmlFilePath;

public:
    // Constructor loads the XML file and generates plugin identifier
    GenericEffectFactory(const std::string& xmlFile) 
        : OFX::PluginFactoryHelper<GenericEffectFactory>(generatePluginIdentifier(xmlFile), 1, 0),
          m_xmlDef(xmlFile), m_xmlFilePath(xmlFile) {
        m_pluginIdentifier = generatePluginIdentifier(xmlFile);
    }

    // Describe phase - tell host about basic effect properties
    virtual void describe(OFX::ImageEffectDescriptor& p_Desc) override {
        // Set basic effect info from XML
        p_Desc.setLabels(m_xmlDef.getName().c_str(), 
                        m_xmlDef.getName().c_str(), 
                        m_xmlDef.getName().c_str());
        p_Desc.setPluginGrouping(m_xmlDef.getCategory().c_str());
        p_Desc.setPluginDescription(m_xmlDef.getDescription().c_str());
        
        // Standard OFX setup
        p_Desc.addSupportedContext(eContextFilter);
        p_Desc.addSupportedBitDepth(eBitDepthFloat);
        p_Desc.setSingleInstance(false);
        p_Desc.setHostFrameThreading(false);
        p_Desc.setSupportsMultiResolution(false);
        p_Desc.setSupportsTiles(false);
        
        // GPU support flags
        p_Desc.setSupportsOpenCLRender(true);
#ifndef __APPLE__
        p_Desc.setSupportsCudaRender(true);
        p_Desc.setSupportsCudaStream(true);
#endif
#ifdef __APPLE__
        p_Desc.setSupportsMetalRender(true);
#endif
    }

    // Describe in context - create parameters and clips from XML
    virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, 
                                  OFX::ContextEnum /*p_Context*/) override {
        // Use existing XMLParameterManager to create parameters from XML
        XMLParameterManager paramManager;
        std::map<std::string, OFX::PageParamDescriptor*> pages;
        paramManager.createParameters(m_xmlDef, p_Desc, pages);
        paramManager.organizeUI(m_xmlDef, p_Desc, pages);

        // Use existing XMLInputManager to create clips from XML
        XMLInputManager inputManager;
        std::map<std::string, std::string> clipBorderModes;
        inputManager.createInputs(m_xmlDef, p_Desc, clipBorderModes);
    }

    // Create instance - make a GenericEffect that uses this XML
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle p_Handle, 
                                           OFX::ContextEnum /*p_Context*/) override {
        return new GenericEffect(p_Handle, m_xmlFilePath);
    }

private:
    // Generate plugin identifier from XML file path (static for constructor use)
    static std::string generatePluginIdentifier(const std::string& xmlFile);
    
    // Generate plugin identifier from loaded XML data (instance method)
    std::string generatePluginIdentifier();
};
```

**Implementation Status**: ✅ Complete and integration tested
- XML loading and parsing validated with test XML file (TestEffect.xml)
- Parameter extraction tested (3 parameters: radius, quality, maskStrength)
- Input parsing tested (source + optional mask with border modes)
- Kernel detection tested (CUDA, OpenCL, Metal variants)
- Plugin identifier generation working (com.xmlframework.TestEffect)
- OFX API integration challenges resolved (PluginFactoryHelper constructor)

### GenericEffect (🔄 Next Implementation Priority)
Replaces BlurPlugin with dynamic parameter/clip handling. Handles the OFX instance and render phases.

```cpp
class GenericEffect : public OFX::ImageEffect {
private:
    XMLEffectDefinition m_xmlDef;
    std::map<std::string, OFX::Param*> m_dynamicParams;
    std::map<std::string, OFX::Clip*> m_dynamicClips;

public:
    GenericEffect(OfxImageEffectHandle p_Handle, const std::string& xmlFile) 
        : OFX::ImageEffect(p_Handle), m_xmlDef(xmlFile) {
        
        // Fetch all parameters that the factory created
        for (const auto& paramDef : m_xmlDef.getParameters()) {
            OFX::Param* param = fetchParam(paramDef.name.c_str());
            m_dynamicParams[paramDef.name] = param;
        }
        
        // Fetch all clips that the factory created
        for (const auto& inputDef : m_xmlDef.getInputs()) {
            OFX::Clip* clip = fetchClip(inputDef.name.c_str());
            m_dynamicClips[inputDef.name] = clip;
        }
        
        // Output clip
        m_dynamicClips["output"] = fetchClip(kOfxImageEffectOutputClipName);
    }

    // Override render method
    virtual void render(const OFX::RenderArguments& p_Args) override {
        // Get output clip
        OFX::Clip* dstClip = m_dynamicClips["output"];
        
        // Check format support (same pattern as BlurPlugin)
        if ((dstClip->getPixelDepth() == OFX::eBitDepthFloat) && 
            (dstClip->getPixelComponents() == OFX::ePixelComponentRGBA)) {
            
            // Create dynamic processor instead of fixed ImageBlurrer
            GenericProcessor processor(*this, m_xmlDef);
            setupAndProcess(processor, p_Args);
        }
        else {
            OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
        }
    }

    // Override identity check - use XML conditions instead of hard-coded
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, 
                           OFX::Clip*& p_IdentityClip, double& p_IdentityTime) override {
        // Evaluate identity conditions from XML
        for (const auto& condition : m_xmlDef.getIdentityConditions()) {
            if (evaluateIdentityCondition(condition, p_Args.time)) {
                p_IdentityClip = m_dynamicClips[m_xmlDef.getInputs()[0].name]; // First input
                p_IdentityTime = p_Args.time;
                return true;
            }
        }
        return false;
    }

private:
    void setupAndProcess(GenericProcessor& processor, const OFX::RenderArguments& p_Args) {
        // Fetch all images dynamically
        std::map<std::string, std::unique_ptr<OFX::Image>> images;
        
        for (const auto& inputDef : m_xmlDef.getInputs()) {
            OFX::Clip* clip = m_dynamicClips[inputDef.name];
            if (clip && clip->isConnected()) {
                images[inputDef.name] = std::unique_ptr<OFX::Image>(clip->fetchImage(p_Args.time));
            }
        }
        
        // Get output image
        images["output"] = std::unique_ptr<OFX::Image>(m_dynamicClips["output"]->fetchImage(p_Args.time));
        
        // Get all parameter values dynamically
        std::map<std::string, ParameterValue> paramValues;
        for (const auto& paramDef : m_xmlDef.getParameters()) {
            paramValues[paramDef.name] = getParameterValue(paramDef.name, p_Args.time);
        }
        
        // Pass everything to the dynamic processor
        processor.setImages(images);
        processor.setParameters(paramValues);
        processor.setRenderWindow(p_Args.renderWindow);
        processor.setGPURenderArgs(p_Args);
        processor.process();
    }

    ParameterValue getParameterValue(const std::string& paramName, double time) {
        OFX::Param* param = m_dynamicParams[paramName];
        const auto& paramDef = m_xmlDef.getParameter(paramName);
        
        if (paramDef.type == "double" || paramDef.type == "float") {
            OFX::DoubleParam* doubleParam = static_cast<OFX::DoubleParam*>(param);
            return ParameterValue(doubleParam->getValueAtTime(time));
        }
        else if (paramDef.type == "int") {
            OFX::IntParam* intParam = static_cast<OFX::IntParam*>(param);
            return ParameterValue(intParam->getValueAtTime(time));
        }
        else if (paramDef.type == "bool") {
            OFX::BooleanParam* boolParam = static_cast<OFX::BooleanParam*>(param);
            return ParameterValue(boolParam->getValueAtTime(time));
        }
        // Add other parameter types as needed
        
        return ParameterValue(); // Default/invalid value
    }

    bool evaluateIdentityCondition(const XMLEffectDefinition::IdentityConditionDef& condition, double time) {
        ParameterValue value = getParameterValue(condition.paramName, time);
        
        if (condition.op == "lessEqual") {
            return value.asDouble() <= condition.value;
        }
        else if (condition.op == "equal") {
            return value.asDouble() == condition.value;
        }
        // Add other operators as needed
        
        return false;
    }
};
```

### GenericProcessor (📋 Future Implementation)
Replaces ImageBlurrer with dynamic parameter/image handling.

```cpp
class GenericProcessor : public OFX::ImageProcessor {
private:
    XMLEffectDefinition m_xmlDef;
    std::map<std::string, OFX::Image*> m_images;
    std::map<std::string, ParameterValue> m_paramValues;

public:
    GenericProcessor(OFX::ImageEffect& effect, const XMLEffectDefinition& xmlDef) 
        : OFX::ImageProcessor(effect), m_xmlDef(xmlDef) {
    }

    void setImages(const std::map<std::string, std::unique_ptr<OFX::Image>>& images) {
        // Store raw pointers (processor doesn't own the images)
        for (const auto& pair : images) {
            m_images[pair.first] = pair.second.get();
        }
    }

    void setParameters(const std::map<std::string, ParameterValue>& params) {
        m_paramValues = params;
    }

    // Override the platform-specific methods
    virtual void processImagesCUDA() override {
        callDynamicKernel("cuda");
    }

    virtual void processImagesOpenCL() override {
        callDynamicKernel("opencl"); 
    }

    virtual void processImagesMetal() override {
        callDynamicKernel("metal");
    }

    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow) override {
        // CPU fallback - would need CPU implementation of effects
        // For now, just copy source to output
        OFX::Image* srcImg = m_images.count("source") ? m_images["source"] : nullptr;
        
        if (srcImg) {
            // Simple copy as fallback
            for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y) {
                if (_effect.abort()) break;
                
                float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
                float* srcPix = static_cast<float*>(srcImg->getPixelAddress(p_ProcWindow.x1, y));
                
                for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x) {
                    for (int c = 0; c < 4; ++c) {
                        dstPix[c] = srcPix[c];
                    }
                    dstPix += 4;
                    srcPix += 4;
                }
            }
        }
    }

private:
    void callDynamicKernel(const std::string& platform) {
        // Get image dimensions from output image
        OFX::Image* outputImg = m_images["output"];
        const OfxRectI& bounds = outputImg->getBounds();
        const int width = bounds.x2 - bounds.x1;
        const int height = bounds.y2 - bounds.y1;
        
        // Get kernel definition for this platform from XML
        auto kernels = m_xmlDef.getKernelsForPlatform(platform);
        if (kernels.empty()) {
            // No kernel for this platform - could fallback to CPU
            return;
        }
        
        // Use the first kernel (Version 1 - single kernel per platform)
        const auto& kernel = kernels[0];
        
        // Call the appropriate dynamic kernel wrapper
        if (platform == "cuda") {
            callCudaKernel(kernel.file, width, height, m_images, m_paramValues);
        }
        else if (platform == "opencl") {
            callOpenCLKernel(kernel.file, width, height, m_images, m_paramValues);
        }
        else if (platform == "metal") {
            callMetalKernel(kernel.file, width, height, m_images, m_paramValues);
        }
    }

    void callCudaKernel(const std::string& kernelFile, int width, int height,
                       const std::map<std::string, OFX::Image*>& images,
                       const std::map<std::string, ParameterValue>& params) {
        // This is where we call the effect-specific kernel wrapper
        // The wrapper name is derived from the effect name
        std::string effectName = m_xmlDef.getName();
        
        if (effectName == "GaussianBlur") {
            RunGaussianBlurKernel(_pCudaStream, width, height, images, params);
        }
        else if (effectName == "ColorCorrect") {
            RunColorCorrectKernel(_pCudaStream, width, height, images, params);
        }
        // Add more effects as they're implemented
        
        // TODO: In future, this could be made more dynamic using
        // dynamic loading or function pointers
    }
    
    // Similar methods for OpenCL and Metal...
};
```

## Supporting Classes

### ParameterValue (✅ Implemented and Tested)
Type-safe parameter value storage for dynamic parameter passing.

**Key Design Decisions**:
- **Union storage** for memory efficiency while supporting multiple types
- **Explicit const char* constructor** to prevent ambiguous bool conversion
- **Comprehensive type conversion** with sensible fallbacks for invalid data
- **String parsing support** for numeric values from configuration files

**Memory Management**: Value semantics with proper copy constructor and assignment operator. String member handled automatically by std::string.

**Testing Coverage**: Complete unit test suite validates:
- All type conversions between double, int, bool, string
- Edge cases (zero values, invalid string conversions)
- Copy/assignment operations
- Memory management (no leaks detected)

**Usage Pattern**:
```cpp
// Create from various types
ParameterValue radius(5.7);        // double
ParameterValue quality(8);         // int
ParameterValue enable(true);       // bool
ParameterValue name("blur");       // string

// Type-safe extraction
float kernelRadius = radius.asFloat();
int kernelQuality = quality.asInt();
bool kernelEnable = enable.asBool();
```

```cpp
class ParameterValue {
private:
    enum Type { DOUBLE, INT, BOOL, STRING } m_type;
    union {
        double m_double;
        int m_int;
        bool m_bool;
    };
    std::string m_string;

public:
    ParameterValue() : m_type(DOUBLE), m_double(0.0) {}
    ParameterValue(double value) : m_type(DOUBLE), m_double(value) {}
    ParameterValue(int value) : m_type(INT), m_int(value) {}
    ParameterValue(bool value) : m_type(BOOL), m_bool(value) {}
    ParameterValue(const std::string& value) : m_type(STRING), m_string(value) {}
    
    double asDouble() const {
        switch (m_type) {
            case DOUBLE: return m_double;
            case INT: return static_cast<double>(m_int);
            case BOOL: return m_bool ? 1.0 : 0.0;
            default: return 0.0;
        }
    }
    
    int asInt() const {
        switch (m_type) {
            case INT: return m_int;
            case DOUBLE: return static_cast<int>(m_double);
            case BOOL: return m_bool ? 1 : 0;
            default: return 0;
        }
    }
    
    bool asBool() const {
        switch (m_type) {
            case BOOL: return m_bool;
            case DOUBLE: return m_double != 0.0;
            case INT: return m_int != 0;
            default: return false;
        }
    }
    
    float asFloat() const {
        return static_cast<float>(asDouble());
    }
    
    std::string asString() const {
        return m_string;
    }
};
```

## Dynamic Kernel Interface

### Kernel Wrapper Pattern
Each effect needs a CPU-side wrapper function that extracts parameters and calls the actual GPU kernel:

```cpp
// Example: RunGaussianBlurKernel wrapper
void RunGaussianBlurKernel(void* stream, int width, int height,
                          const std::map<std::string, OFX::Image*>& images,
                          const std::map<std::string, ParameterValue>& params) {
    
    // Extract parameters for this specific effect
    float radius = params.at("radius").asFloat();
    int quality = params.at("quality").asInt();
    float maskStrength = params.count("maskStrength") ? 
                        params.at("maskStrength").asFloat() : 1.0f;
    
    // Setup input textures
    cudaTextureObject_t sourceTexture = setupCudaTexture(images.at("source"));
    cudaTextureObject_t maskTexture = images.count("mask") ? 
                                     setupCudaTexture(images.at("mask")) : 0;
    
    // Get output buffer
    float* output = static_cast<float*>(images.at("output")->getPixelData());
    
    // Call the actual GPU kernel with explicit parameters
    dim3 threads(16, 16, 1);
    dim3 blocks(((width + threads.x - 1) / threads.x), 
                ((height + threads.y - 1) / threads.y), 1);
    
    GaussianBlurKernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        width, height, radius, quality, maskStrength,
        sourceTexture, maskTexture, output);
    
    // Cleanup textures
    cudaDestroyTextureObject(sourceTexture);
    if (maskTexture) cudaDestroyTextureObject(maskTexture);
}
```

### Kernel Signature Generation
To help effect authors create kernels with correct signatures, a script generates signature templates:

```bash
python generate_signature.py GaussianBlur.xml
```

Creates `GaussianBlur_signature.txt`:
```cpp
__global__ void GaussianBlurKernel(
    int width, int height,
    float radius,                    // From <parameter name="radius" type="double" ...>
    int quality,                     // From <parameter name="quality" type="int" ...>
    float maskStrength,              // From <parameter name="maskStrength" type="double" ...>
    cudaTextureObject_t source,      // From <source name="source" ...>
    cudaTextureObject_t mask,        // From <source name="mask" optional="true" ...>
    float* output
)
```

Effect authors copy this signature into their kernel implementation.

## Key Advantages Over BlurPlugin

### Dynamic vs Fixed Structure
- **BlurPlugin**: Exactly 3 parameters, source + optional mask
- **GenericEffect**: Any number of parameters and inputs from XML

### Parameter Handling
- **BlurPlugin**: `m_Radius->getValueAtTime()`, `m_Quality->getValueAtTime()` (manual for each)
- **GenericEffect**: Loop through XML parameters, automatic extraction

### Kernel Interface
- **BlurPlugin**: `RunCudaKernel(stream, width, height, radius, quality, maskStrength, input, mask, output)`
- **GenericEffect**: `RunGenericKernel(stream, width, height, paramMap, imageMap)`

### UI Creation
- **BlurPlugin**: Manual parameter creation in `describeInContext()`
- **GenericEffect**: Automatic from XML using existing XMLParameterManager

### Format Handling
- **BlurPlugin**: Hard-coded float RGBA assumption
- **GenericEffect**: Can be extended to detect and handle multiple formats

## Implementation Notes

### Incremental Implementation Strategy (✅ Proven Effective)
The framework is being implemented in small, thoroughly tested increments:

1. **ParameterValue** (✅ Complete): Pure type-safe storage with no dependencies
2. **GenericEffectFactory** (✅ Complete): OFX factory integration with XML loading
3. **GenericEffect** (🔄 Next): Effect instance with dynamic parameter/clip handling
4. **GenericProcessor** (📋 Future): Dynamic rendering with kernel dispatch
5. **Integration Testing** (📋 Future): End-to-end validation with real kernels

**Benefits of This Approach**:
- **Early Issue Detection**: OFX API integration problems caught and solved early
- **Build System Validation**: Makefile compilation issues resolved incrementally  
- **Foundation Verification**: XML→OFX pipeline proven working before complex rendering
- **Risk Reduction**: Each step builds on validated components

### Build System Integration (✅ Lessons Learned)
**Critical Discovery**: Makefile compilation rules must appear **before** the main target that uses them.

**Issue**: Missing object files (`GenericEffectFactory.o`, `XMLEffectDefinition.o`, `pugixml.o`) weren't being built
**Root Cause**: Compilation rules appeared after the main target in Makefile
**Solution**: Move all compilation rules before `BlurPlugin.ofx:` target
**Result**: All framework components now build correctly with main plugin

**Correct Makefile Pattern**:
```makefile
# Variable definitions
PLUGIN_MODULES = Logger.o PluginClips.o ... GenericEffectFactory.o XMLEffectDefinition.o pugixml.o

# ALL COMPILATION RULES FIRST
GenericEffectFactory.o: src/core/GenericEffectFactory.cpp
	$(CXX) -c $< $(CXXFLAGS) -I./src -I./include/pugixml

XMLEffectDefinition.o: src/core/XMLEffectDefinition.cpp  
	$(CXX) -c $< $(CXXFLAGS) -I./include/pugixml

# MAIN TARGET AFTER ALL RULES
BlurPlugin.ofx: $(PLUGIN_OBJS) $(OFX_SUPPORT_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)
```

### OFX API Integration (✅ Lessons Learned)
**Challenge**: `PluginFactoryHelper` requires plugin identifier in constructor, but identifier generation needs XML parsing.

**Solution**: Static helper method pattern
```cpp
// Static method can be called before XML loading
static std::string generatePluginIdentifier(const std::string& xmlFile);

// Constructor uses static method in initialization list
GenericEffectFactory(const std::string& xmlFile) 
    : OFX::PluginFactoryHelper<GenericEffectFactory>(generatePluginIdentifier(xmlFile), 1, 0),
      m_xmlDef(xmlFile), m_xmlFilePath(xmlFile)
```

**Key Insight**: OFX base classes have specific initialization requirements that must be satisfied in the initialization list, before member initialization.

### Phase 3 Implementation Order
1. **ParameterValue class** ✅ - Type-safe parameter storage (completed and tested)
2. **GenericEffectFactory** ✅ - XML-driven factory using existing managers (completed and tested)
3. **GenericEffect constructor** 🔄 - Dynamic parameter/clip fetching (next priority)
4. **GenericProcessor skeleton** 📋 - Platform method stubs
5. **Dynamic kernel calling** 📋 - Effect name dispatch pattern
6. **Signature generation script** 📋 - Help authors create correct kernel signatures

### Integration with Existing Framework
This design leverages the existing XMLEffectDefinition, XMLParameterManager, and XMLInputManager components from Phase 1-2. The implementation follows a proven incremental approach:

**Phase 1-2 Foundation** (✅ Complete):
- XMLEffectDefinition: XML parsing and validation
- XMLParameterManager: OFX parameter creation from XML
- XMLInputManager: OFX clip creation from XML  
- All helper classes: PluginParameters, PluginClips, BlurPluginParameters

**Phase 3 Dynamic Infrastructure** (🔄 In Progress):
- ParameterValue: ✅ Type-safe dynamic parameter storage
- GenericEffectFactory: ✅ XML-driven OFX factory integration
- GenericEffect: 🔄 Dynamic effect instance (next)
- GenericProcessor: 📋 Dynamic rendering (future)

**Integration Points Validated**:
- ✅ XML parsing → OFX parameter creation (XMLParameterManager tested)
- ✅ XML parsing → OFX clip creation (XMLInputManager tested)  
- ✅ OFX factory integration (GenericEffectFactory working)
- ✅ Build system integration (all components compile together)

The GenericEffect classes provide the "bridge" between the XML framework and the OFX plugin system, with each component thoroughly tested before proceeding to the next.

### Forward Compatibility
The design supports Version 2 multi-kernel features:
- `m_xmlDef.hasPipeline()` check for multi-kernel effects
- `callDynamicKernel()` can be extended to handle pipeline steps
- Parameter passing system works for any number of kernels

This architecture transforms OFX plugin development from complex C++ programming to straightforward XML configuration + kernel implementation, achieving the core goal of the framework.
