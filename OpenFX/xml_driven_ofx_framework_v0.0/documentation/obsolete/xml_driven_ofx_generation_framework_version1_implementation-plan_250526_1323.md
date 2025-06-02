# XML-Based OFX Framework Implementation Plan - Version 1

## Introduction

This document outlines a detailed step-by-step implementation plan for Version 1 of the XML-based OFX image processing framework. The plan is structured into small, testable increments to ensure stable progress and minimize risks.

> **Key Principle**: The user should only need to modify XML effect definitions and kernel code files, never the framework code itself.

## Phase 1: Core XML Parsing and Validation (completed)

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

## Phase 2: OFX Parameter Creation (completed)

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

## Phase 3: Dynamic Effect Base Class - UPDATED STRUCTURE (may 26,2025)

**Status**: 🔄 In Progress (2/7 steps complete)  
**Design Reference**: See [GenericEffect Architecture Design](GenericEffect_Architecture_Design.md) for complete technical details.

### Step 3.1: ParameterValue Support Class ✅ COMPLETED
**Goal**: Create type-safe parameter value storage for dynamic parameter passing.  
**Status**: ✅ Complete - Full implementation with comprehensive testing

**What Was Accomplished**:
- Type-safe storage with union for double, int, bool, string values
- Proper const char* constructor to avoid bool conversion ambiguity  
- Comprehensive type conversion methods (asDouble, asInt, asBool, asFloat, asString)
- Copy constructor and assignment operator working correctly
- Complete unit test suite with edge case coverage

**Files Created**:
- `src/core/ParameterValue.h`
- `src/core/ParameterValue.cpp`
- `src/tools/ParameterValueTest.cpp`

**Test Results**: All type conversions, edge cases, and memory management verified working correctly.

**Validation**: ✅ Complete standalone testing via `make -f Makefile.xml test_paramvalue`

---

### Step 3.2: GenericEffectFactory Implementation ✅ COMPLETED  
**Goal**: Create XML-driven factory that can load ANY XML effect definition and prepare it for OFX registration.  
**Status**: ✅ Complete - Full implementation with integration testing

**What Was Accomplished**:
- XML-driven plugin creation from any XML effect definition
- Automatic plugin identifier generation from XML file names
- Reuses existing XMLParameterManager and XMLInputManager for OFX integration
- Smart GPU support detection based on kernels defined in XML
- Proper OFX PluginFactoryHelper inheritance with correct constructor parameters

**Files Created**:
- `src/core/GenericEffectFactory.h`
- `src/core/GenericEffectFactory.cpp`
- `TestEffect.xml` (test XML file)

**Build System Fixes**:
- Added missing compilation rules to main Makefile
- Resolved compilation rule ordering issues
- All object files now build correctly

**Integration Test Results** (via BlurPlugin.cpp test):
- ✅ XML loading and parsing successful (TestBlur effect)
- ✅ Parameter extraction (3 parameters: radius, quality, maskStrength with correct types/defaults)
- ✅ Input parsing (source + optional mask with border modes)
- ✅ Kernel detection (CUDA, OpenCL, Metal variants)
- ✅ Plugin identifier generation (com.xmlframework.TestEffect)

**Validation**: ✅ Complete integration testing - factory can load and parse any XML effect definition

---

### Step 3.3: GenericEffect Base Class ⏳ NEXT STEP
**Goal**: Create the main effect instance class that replaces BlurPlugin's fixed structure with XML-driven dynamic behavior.  
**Status**: 🔲 Ready for Implementation

**Tasks**:
1. Implement GenericEffect extending OFX::ImageEffect
2. Add dynamic parameter storage using ParameterValue maps
3. Add dynamic clip storage with names from XML
4. Implement constructor that fetches parameters/clips created by GenericEffectFactory
5. Add getParameterValue() helper with type-specific extraction
6. Create basic render() method structure (detailed implementation in Step 3.5)

**Implementation Approach**:
```cpp
class GenericEffect : public OFX::ImageEffect {
    XMLEffectDefinition m_xmlDef;
    std::map<std::string, OFX::Param*> m_dynamicParams;     // Fetched from factory-created params
    std::map<std::string, OFX::Clip*> m_dynamicClips;       // Fetched from factory-created clips
    
    // Constructor fetches existing params/clips by name from XML
    // render() method delegates to GenericProcessor (Step 3.5)
};
```

**Test Plan**:
- Update GenericEffectFactory::createInstance() to return GenericEffect
- Test parameter fetching by name matches XML definitions
- Test clip fetching by name matches XML definitions  
- Verify GenericEffect constructor doesn't crash during plugin loading

**Success Criteria**:
- GenericEffect can be instantiated from any XML definition
- Dynamic parameter fetching works for all XML parameter types
- Dynamic clip fetching works for all XML input configurations
- Plugin loads successfully in OFX host without render testing

---

### Step 3.4: Identity Condition Implementation
**Goal**: Implement XML-driven identity condition checking for pass-through behavior.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement isIdentity() method in GenericEffect
2. Add evaluateIdentityCondition() helper method
3. Process identity conditions from XML definition (lessEqual, equal, etc.)
4. Add parameter value comparison logic
5. Return appropriate identity clip when conditions are met

**Implementation**:
```cpp
virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, 
                       OFX::Clip*& p_IdentityClip, double& p_IdentityTime) override {
    for (const auto& condition : m_xmlDef.getIdentityConditions()) {
        if (evaluateIdentityCondition(condition, p_Args.time)) {
            p_IdentityClip = m_dynamicClips[m_xmlDef.getInputs()[0].name]; 
            p_IdentityTime = p_Args.time;
            return true;
        }
    }
    return false;
}
```

**Test Plan**:
- Create test XML with identity conditions (e.g., radius <= 0.0)
- Test that effect returns identity when conditions are met
- Test that effect processes normally when conditions are not met
- Verify identity behavior matches BlurPlugin pattern

**Success Criteria**:
- Identity conditions from XML work correctly
- Different operators (lessEqual, equal, etc.) function as expected
- Performance equivalent to BlurPlugin identity checking

---

### Step 3.5: GenericProcessor Implementation
**Goal**: Create processor that handles dynamic parameter passing and replaces ImageBlurrer's fixed structure.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement GenericProcessor extending OFX::ImageProcessor
2. Add dynamic image and parameter storage from GenericEffect
3. Implement setImages() and setParameters() methods
4. Add platform-specific process methods (CUDA, OpenCL, Metal) 
5. Implement callDynamicKernel() with effect name dispatch pattern

**Implementation Approach**:
```cpp
class GenericProcessor : public OFX::ImageProcessor {
    XMLEffectDefinition m_xmlDef;
    std::map<std::string, OFX::Image*> m_images;           // Raw pointers (borrowed)
    std::map<std::string, ParameterValue> m_paramValues;   // Extracted values
    
    // Platform methods call callDynamicKernel() with platform name
    // callDynamicKernel() dispatches to effect-specific wrappers
};
```

**Test Plan**:
- Test parameter value extraction for all XML parameter types
- Test image handling for arbitrary input configurations
- Test platform method selection (CUDA/OpenCL/Metal)
- Verify memory ownership pattern (processor borrows, doesn't own)

**Success Criteria**:
- Can handle any parameter/image configuration from XML
- Platform selection works correctly
- Memory ownership pattern is safe (no crashes/leaks)
- Performance equivalent to ImageBlurrer approach

---

### Step 3.6: Dynamic Render Implementation  
**Goal**: Implement GenericEffect::render() that orchestrates dynamic processing.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement render() method in GenericEffect
2. Add setupAndProcess() helper that dynamically fetches images/parameters  
3. Create dynamic parameter value extraction loop using ParameterValue
4. Integrate with GenericProcessor from Step 3.5
5. Add error handling for missing images/parameters

**Implementation Pattern**:
```cpp
void render(const OFX::RenderArguments& p_Args) override {
    // Create dynamic processor
    GenericProcessor processor(*this, m_xmlDef);
    
    // Fetch all images dynamically by name from XML
    std::map<std::string, std::unique_ptr<OFX::Image>> images;
    for (const auto& inputDef : m_xmlDef.getInputs()) {
        // Fetch images by XML input names
    }
    
    // Extract all parameter values dynamically by name from XML
    std::map<std::string, ParameterValue> paramValues;
    for (const auto& paramDef : m_xmlDef.getParameters()) {
        paramValues[paramDef.name] = getParameterValue(paramDef.name, p_Args.time);
    }
    
    // Pass to processor and render
    processor.setImages(images);
    processor.setParameters(paramValues);
    processor.process();
}
```

**Test Plan**:
- Test with XML configurations of varying complexity
- Test parameter extraction handles all parameter types correctly
- Test image fetching works for any number of inputs  
- Test error handling for disconnected optional inputs

**Success Criteria**:
- Works with any XML configuration
- Parameter/image extraction is robust and efficient
- Error handling provides clear feedback
- Performance equivalent to BlurPlugin::render()

---

### Step 3.7: Integration Testing with Real Kernel Call
**Goal**: Test complete GenericEffect pipeline with actual kernel execution.  
**Status**: 🔲 Not Started

**Tasks**:
1. Create simple test kernel wrapper (RunTestBlurKernel)
2. Implement basic parameter extraction pattern  
3. Test end-to-end XML → GenericEffect → Kernel execution
4. Compare results with equivalent BlurPlugin execution
5. Performance benchmarking

**Test Setup**:
- Use TestEffect.xml with simple blur parameters
- Create RunTestBlurKernel wrapper that extracts radius, quality, maskStrength
- Call existing GaussianBlurKernel with extracted parameters
- Verify identical results to BlurPlugin

**Success Criteria**:
- End-to-end XML-defined effect produces correct visual results
- Parameter passing from XML to kernel works correctly
- Performance is equivalent to BlurPlugin approach
- Memory usage is reasonable

---

## Phase 3 Integration and Validation

### End-to-End Framework Test
**Goal**: Validate complete GenericEffect can replace BlurPlugin functionality.

**Test Matrix**:
- [ ] Parameter types: double, int, bool (from ParameterValue)
- [ ] Input configurations: single source, source + mask, multiple inputs
- [ ] Identity conditions: various operators and parameter types
- [ ] Platform support: CUDA, OpenCL, Metal kernel dispatch
- [ ] UI organization: pages, columns, parameter grouping
- [ ] Memory management: no leaks, proper cleanup

### Success Metrics for Phase 3
- [ ] GenericEffect can load any XML effect definition
- [ ] Dynamic parameter system supports all XML parameter types  
- [ ] Dynamic clip system supports arbitrary input configurations
- [ ] Identity conditions work correctly from XML definitions
- [ ] Kernel dispatch works for all platforms
- [ ] Memory management is leak-free and performant
- [ ] Performance equivalent to BlurPlugin approach

# Implementation Lessons Learned - Phase 3 Progress

## Critical Build System Discovery

### Makefile Rule Ordering (CRITICAL)
**Problem**: Missing object files (`GenericEffectFactory.o`, `XMLEffectDefinition.o`, `pugixml.o`) weren't building
**Root Cause**: Compilation rules appeared AFTER the main target in Makefile
**Solution**: Move ALL compilation rules BEFORE main target
**Impact**: Framework components now compile correctly

```makefile
# WRONG - Rules after target
BlurPlugin.ofx: $(PLUGIN_OBJS) $(OFX_SUPPORT_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

GenericEffectFactory.o: src/core/GenericEffectFactory.cpp
	$(CXX) -c $< $(CXXFLAGS)

# CORRECT - Rules before target  
GenericEffectFactory.o: src/core/GenericEffectFactory.cpp
	$(CXX) -c $< $(CXXFLAGS)

BlurPlugin.ofx: $(PLUGIN_OBJS) $(OFX_SUPPORT_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)
```

**Key Learning**: Make processes rules in order - dependencies must be visible when target is processed.

## OFX API Integration Challenges

### PluginFactoryHelper Constructor Requirements
**Challenge**: Constructor needs plugin identifier, but identifier generation requires XML parsing
**Failed Approach**: Try to parse XML in initialization list
**Working Solution**: Static helper method pattern

```cpp
// This works
GenericEffectFactory(const std::string& xmlFile) 
    : OFX::PluginFactoryHelper<GenericEffectFactory>(generatePluginIdentifier(xmlFile), 1, 0),
      m_xmlDef(xmlFile)  // XML loaded after base class construction
```

**Lesson**: OFX base classes have specific initialization requirements that must be satisfied in initialization list.

## Type Safety Discoveries

### ParameterValue const char* Ambiguity
**Problem**: `ParameterValue param("test")` could be interpreted as bool constructor
**Root Cause**: C++ string literal is `const char*`, bool constructor preferred in overload resolution
**Solution**: Explicit `const char*` constructor prevents ambiguity

```cpp
ParameterValue(const char* value);  // Explicit constructor
ParameterValue(bool value);         // Now unambiguous
```

**Lesson**: Always provide explicit constructors for string literals when bool constructors exist.

## Testing Strategy Success

### Incremental Component Validation
**Pattern Used**: Component → Unit Test → Integration Test → Next Component
**Benefits Realized**:
- OFX API integration problems caught early
- Build system issues resolved incrementally  
- Foundation verification before complex rendering
- Risk reduction through validated components

**Specific Example**: ParameterValue unit tests caught type conversion edge cases before integration.

## Memory Management Patterns

### Framework Ownership Model
**Discovery**: Clear ownership pattern essential for GPU processing
**Pattern**: 
- GenericEffect owns `unique_ptr<OFX::Image>` (RAII cleanup)
- GenericProcessor receives raw pointers (borrowed references)
- Automatic cleanup when GenericEffect scope ends

```cpp
// GenericEffect::render()
std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
processor.setDstImg(dst.get());  // Borrow pointer
processor.process();             // Use image
// dst automatically cleaned up here
```

## Integration Testing Insights

### XML → OFX Validation Success
**Tested Scenarios**:
- ✅ XML loading and parsing (TestEffect.xml)  
- ✅ Parameter extraction (3 parameters with correct types/defaults)
- ✅ Input parsing (source + optional mask with border modes)
- ✅ Kernel detection (CUDA, OpenCL, Metal variants)
- ✅ Plugin identifier generation (com.xmlframework.TestEffect)

**Key Insight**: GenericEffectFactory can successfully load and prepare ANY XML effect definition for OFX registration.

## Code Organization Lessons

### Separation of Concerns Success
**Factory vs Effect Pattern**:
- GenericEffectFactory: Handles OFX describe/describeInContext (static info from XML)
- GenericEffect: Handles instance creation and rendering (dynamic processing)
- Clean lifecycle separation simplifies debugging

### PIMPL Pattern Benefits  
**XMLEffectDefinition Design**: 
- Header exposes clean API
- Implementation details hidden (pugixml dependency)
- Compilation time reduced
- Easy to change XML library in future

## Error Handling Discoveries

### Exception Safety in OFX Context
**Challenge**: OFX expects graceful failure, not exceptions
**Pattern**: Catch exceptions in factory methods, return appropriate OFX errors
```cpp
try {
    XMLEffectDefinition xmlDef(xmlFile);
    // Process XML...
} catch (const std::exception& e) {
    // Log error, return OFX error code
    return nullptr; // or appropriate OFX status
}
```

## Performance Considerations

### Dynamic Parameter Overhead
**Measurement Needed**: Map lookups vs direct member access
**Current Approach**: Accept slight overhead for massive flexibility gain
**Future Optimization**: Parameter value caching if needed

### XML Parsing Cost
**Current**: Parse XML once in factory constructor
**Optimization**: Cache parsed definitions for repeated plugin creation
**Impact**: Negligible for typical usage patterns

## Next Phase Readiness

### Step 3.3 Prerequisites Met
- ✅ ParameterValue type-safe storage working
- ✅ GenericEffectFactory XML integration proven  
- ✅ Build system compiling all components
- ✅ OFX API integration patterns established

### Identified Implementation Path
1. GenericEffect constructor: Fetch parameters/clips by name from XML
2. Dynamic parameter value extraction using ParameterValue
3. Generic processor creation and image handling
4. End-to-end XML → GenericEffect → Kernel execution

## Framework Architecture Validation

### Design Decisions Proven Correct
- **Union storage in ParameterValue**: Memory efficient, type safe
- **Factory pattern separation**: Clean OFX lifecycle handling
- **XML-driven approach**: Successfully replaces fixed plugin structure
- **Incremental testing**: Catches integration issues early

### Areas for Future Enhancement
- **Error reporting**: More detailed XML validation messages
- **Performance profiling**: Measure dynamic vs static parameter access
- **Documentation**: Code examples for effect authors
- **Tool support**: Script generation for kernel signatures

## Key Success Metrics
- ✅ Framework compiles and links correctly
- ✅ Can load any XML effect definition  
- ✅ Type-safe parameter handling working
- ✅ OFX integration proven functional
- 🔄 End-to-end processing pipeline (next milestone)
## Next Phase Preview

**Phase 4**: With GenericEffect providing complete dynamic effect infrastructure, Phase 4 will focus on:
- Signature generation script for kernel authors
- Dynamic kernel wrapper pattern implementation  
- Unified kernel management system
- Multi-platform kernel deployment

The framework architecture supports both single-kernel (Version 1) and multi-kernel pipeline (Version 2) effects without architectural changes.

---
*Updated: May 26, 2025 - After successful ParameterValue and GenericEffectFactory implementation and testing*


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
