 XML-Based OFX Framework Implementation Plan - Version 1 (updated v2.3 and forward on may 25 2025)

## Introduction

This document outlines a detailed step-by-step implementation plan for Version 1 of the XML-based OFX image processing framework. The plan is structured into small, testable increments to ensure stable progress and minimize risks, with priorities informed by analysis of existing OFX plugin code.

> **Key Principle**: The user should only need to modify XML effect definitions and kernel code files, never the framework code itself.

## Lessons from Legacy Code Analysis

Analysis of BlurPlugin.cpp revealed specific technical challenges that guide our implementation priorities:

### Current Pain Points in OFX Development:
- **Fixed parameter passing**: `RunCudaKernel(stream, width, height, radius, quality, maskStrength, input, mask, output)`
- **Hard-coded formats**: `cudaCreateChannelDesc<float4>()` assumes RGBA float
- **Platform duplication**: Separate `processImagesCUDA()`, `processImagesOpenCL()`, `processImagesMetal()`
- **Poor naming**: `p_Args` provides no indication of contents
- **Manual memory management**: Explicit GPU buffer allocation and copying
- **Fixed structure**: Hard-coded source + optional mask, exactly 3 parameters

### Framework Solutions to Implement:
- **Dynamic parameter maps**: `std::map<std::string, ParameterValue> params`
- **Automatic format detection**: Query image metadata instead of assuming
- **Unified kernel interface**: Single entry point across platforms
- **Descriptive API naming**: `RenderContext` instead of `p_Args`
- **Automated memory handling**: Framework manages GPU buffer lifecycle
- **Flexible structure**: Any number of parameters and inputs from XML

## Phase 1: Core XML Parsing and Validation ✅ COMPLETED (May 17, 2025)

### Step 1.1: Basic XML Schema Design ✅ COMPLETED
**Goal**: Create a well-defined XML schema for effect definitions addressing legacy limitations.  
**Status**: ✅ Complete - XML schema defined and documented

**Tasks**:
1. Design XML schema with inputs, parameters, UI, and kernel sections ✅
2. Include attribute-based parameters with label/hint as attributes ✅
3. Add border_mode attributes for source inputs ✅
4. Create sample GaussianBlur.xml based on schema ✅
5. Support unlimited parameters (vs BlurPlugin's fixed 3) ✅
6. Support arbitrary input configurations (vs fixed source + mask) ✅

**Test Criteria**:
- XML schema is complete and documented ✅
- Sample XML is valid against schema ✅
- Schema supports more parameters than BlurPlugin ✅

**Notes**: Schema supports comprehensive parameter types (double, int, bool, choice, color, vec2, string, curve), UI organization, border modes, and both single-kernel and multi-kernel (pipeline) effects. Addresses fixed parameter limitations found in BlurPlugin.cpp.

### Step 1.2: XMLEffectDefinition Class Implementation ✅ COMPLETED
**Goal**: Create a robust class to parse and validate XML effect definitions.  
**Status**: ✅ Complete - Full implementation with comprehensive testing

**Tasks**:
1. Implement basic XMLEffectDefinition class with constructors ✅
2. Add parsing for effect metadata (name, category, description) ✅
3. Add parsing for input sources including border_mode attributes ✅
4. Add parsing for parameters with all attributes ✅
5. Add parsing for UI organization ✅
6. Add parsing for identity conditions ✅
7. Add parsing for kernel definitions ✅

**Implementation Details**:
- Uses PIMPL pattern for clean API
- Supports all parameter types with component-level control
- Comprehensive error handling and validation
- Forward-compatible with Version 2 pipeline support
- Handles unlimited parameters and inputs dynamically (vs hard-coded structure)

**Test Criteria**:
- XML parser correctly loads all elements and attributes ✅
- Error handling for invalid XML files works correctly ✅
- Accessors return correct values ✅
- Supports arbitrary parameter count (tested with more than BlurPlugin's 3) ✅

**Notes**: Implementation successfully handles complex XML with nested components, validates references, and provides comprehensive accessor methods.

### Step 1.3: Unit Tests for XML Parsing ✅ COMPLETED
**Goal**: Create comprehensive tests for XML parsing.  
**Status**: ✅ Complete - Comprehensive test coverage

**Tasks**:
1. Create test suite for XMLEffectDefinition ✅
2. Test all getter methods with various XML inputs ✅
3. Test error handling with malformed XML ✅
4. Test flexible parameter and input handling ✅

**Test Criteria**:
- All tests pass with valid XML ✅
- Invalid XML is rejected with appropriate error messages ✅
- Edge cases (optional attributes, missing sections) are handled correctly ✅
- Dynamic parsing works correctly with varying parameter counts ✅

**Notes**: Test program provides detailed output showing all parsed elements, confirming comprehensive XML processing capability that surpasses fixed parameter limitations.

## Phase 2: OFX Parameter Creation ✅ COMPLETED (May 17, 2025)

### Step 2.1: XMLParameterManager Class ✅ COMPLETED
**Goal**: Create a class to map XML parameter definitions to OFX parameters.  
**Status**: ✅ Complete - Full implementation with OFX integration

**Tasks**:
1. Implement XMLParameterManager class ✅
2. Add support for creating Double, Int, Boolean parameters ✅
3. Add support for creating Choice and Curve parameters ✅
4. Add support for creating Color, Vec2, String parameters ✅
5. Add UI organization (pages, columns) ✅

**Legacy Improvement Demonstrated**:
```cpp
// Old approach (BlurPlugin.cpp):
m_Radius = fetchDoubleParam("radius");          // Hard-coded parameters
m_Quality = fetchIntParam("quality");
m_MaskStrength = fetchDoubleParam("maskStrength");

// New approach (XMLParameterManager):
for (const auto& paramDef : xmlDef.getParameters()) {    // Dynamic creation
    if (paramDef.type == "double") {
        doubleParams[paramDef.name] = createDoubleParam(paramDef);
    }
    // Handles any parameter type automatically
}
```

**Implementation Details**:
- Comprehensive parameter type support
- Resolution dependency handling
- Component-level control for vector/color parameters
- Proper OFX API integration
- Dynamic parameter creation (vs fixed parameter set)

**Test Criteria**:
- Parameters created match XML definitions ✅
- Parameter properties (default, range, labels) are correctly set ✅
- UI organization is applied correctly ✅
- Can create more parameters than BlurPlugin's fixed set ✅

**Notes**: Successfully creates all parameter types with mock OFX objects. Ready for integration with real OFX plugins.

### Step 2.2: XMLInputManager Class ✅ COMPLETED
**Goal**: Create a class to map XML input definitions to OFX clips.  
**Status**: ✅ Complete - Full implementation with border mode support

**Tasks**:
1. Implement XMLInputManager class ✅
2. Add support for creating source clips with proper labels ✅
3. Add support for optional clips ✅
4. Store border mode information for each clip ✅

**Legacy Improvement Demonstrated**:
```cpp
// Old approach (BlurPlugin.cpp):
m_SrcClip = fetchClip("Source");     // Fixed input names
m_MaskClip = fetchClip("Mask");      // Hard-coded structure

// New approach (XMLInputManager):
for (const auto& inputDef : xmlDef.getInputs()) {       // Any inputs
    clips[inputDef.name] = createClip(inputDef);        // Dynamic creation
    borderModes[inputDef.name] = inputDef.borderMode;   // Per-input settings
}
```

**Implementation Details**:
- Automatic mask detection based on naming conventions
- Border mode property setting via OFX PropertySet API
- Optional clip handling
- Comprehensive error handling
- Support for arbitrary input configurations

**Test Criteria**:
- Clips created match XML definitions ✅
- Optional clips are properly flagged ✅
- Border modes are correctly stored for each clip ✅
- Can create more/different inputs than BlurPlugin's fixed structure ✅

**Notes**: Successfully handles all clip types including masks. Border modes properly set via OFX properties for use in kernel processing.

### Step 2.3: Integration with BlurPluginFactory ⏳ NEXT STEP
**Goal**: Create non-destructive integration test with existing plugin.  
**Status**: 🔲 Pending

**Tasks**:
1. Create a test harness in BlurPluginFactory
2. Add XML-based parameter and clip creation alongside existing code
3. Add logging to compare XML vs. manual results
4. Validate that XMLParameterManager produces equivalent parameters
5. Validate that XMLInputManager produces equivalent clips

**Test Criteria**:
- XML-based clips and parameters match manually-created ones
- Log comparison shows equivalence
- Fallback to original works correctly
- No performance degradation

**Notes**: This bridges Phase 2 and Phase 3, validating our XML managers work with real OFX infrastructure before building GenericEffect on top of them.

## Phase 3: Dynamic Effect Base Class ✅ DESIGN COMPLETE

**Status**: 🔲 Ready for Implementation  
**Design Reference**: See [GenericEffect Architecture Design](GenericEffect_Architecture_Design.md) for complete technical details.

### Step 3.1: ParameterValue Support Class
**Goal**: Create type-safe parameter value storage for dynamic parameter passing.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement ParameterValue class with union storage
2. Add type-safe accessors (asDouble, asInt, asBool, asFloat, asString)
3. Add constructors for all supported types
4. Create unit tests for type conversion

**Implementation**: Simple class with union storage and type enum, following design in GenericEffect Architecture Design document.

**Test Criteria**:
- Type conversions work correctly
- No memory leaks with string parameters
- Thread-safe for concurrent access

### Step 3.2: GenericEffectFactory Implementation ⏳ NEXT PRIORITY
**Goal**: Create XML-driven factory that replaces BlurPluginFactory pattern.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement GenericEffectFactory extending PluginFactoryHelper
2. Add constructor that loads XMLEffectDefinition from file
3. Implement describe() method using XML metadata
4. Implement describeInContext() using existing XMLParameterManager and XMLInputManager
5. Implement createInstance() returning GenericEffect

**Key Innovation**: Factory loads ANY XML file and creates appropriate plugin, vs. BlurPluginFactory's hard-coded structure.

**Test Criteria**:
- Factory loads XML files correctly
- OFX metadata matches XML definitions
- Parameters and clips created match XML specifications
- Multiple different XML effects can be loaded

### Step 3.3: GenericEffect Instance Class
**Goal**: Create effect instance that replaces BlurPlugin's fixed parameter/clip structure.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement GenericEffect extending OFX::ImageEffect
2. Add dynamic parameter and clip storage maps
3. Implement constructor with XML loading and parameter/clip fetching
4. Add getParameterValue() helper with type-specific extraction
5. Implement isIdentity() using XML identity conditions

**Legacy vs Framework Comparison**:
```cpp
// BlurPlugin (fixed):
OFX::DoubleParam* m_Radius;
OFX::IntParam* m_Quality;
OFX::Clip* m_SrcClip;

// GenericEffect (dynamic):
std::map<std::string, OFX::Param*> m_dynamicParams;
std::map<std::string, OFX::Clip*> m_dynamicClips;
```

**Test Criteria**:
- Can load any XML effect definition
- Dynamic parameter fetching works for all parameter types
- Identity conditions from XML function correctly
- Memory management is correct (no leaks)

### Step 3.4: GenericProcessor Implementation  
**Goal**: Create processor that replaces ImageBlurrer's fixed parameter pattern.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement GenericProcessor extending OFX::ImageProcessor
2. Add dynamic image and parameter storage
3. Implement setImages() and setParameters() methods
4. Add platform-specific process methods (CUDA, OpenCL, Metal)
5. Implement callDynamicKernel() with effect name dispatch

**Key Challenge**: Replace fixed `RunCudaKernel(stream, width, height, radius, quality, maskStrength, input, mask, output)` with dynamic `RunGenericKernel(stream, width, height, paramMap, imageMap)`.

**Test Criteria**:
- Can handle any parameter/image configuration from XML
- Platform selection works correctly
- Memory ownership pattern is safe (raw pointers, no ownership transfer)
- Performance is equivalent to fixed approach

### Step 3.5: Dynamic Render Implementation
**Goal**: Implement GenericEffect::render() that orchestrates dynamic processing.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement render() method in GenericEffect
2. Add setupAndProcess() helper that dynamically fetches images/parameters
3. Create dynamic parameter value extraction loop
4. Integrate with GenericProcessor
5. Add error handling for missing images/parameters

**Legacy vs Framework Flow**:
```cpp
// BlurPlugin setupAndProcess:
// Fixed: _srcImg, _maskImg, _radius, _quality, _maskStrength

// GenericEffect setupAndProcess:
for (auto& inputDef : m_xmlDef.getInputs()) {
    images[inputDef.name] = fetchImage(inputDef.name, time);
}
for (auto& paramDef : m_xmlDef.getParameters()) {
    paramValues[paramDef.name] = getParameterValue(paramDef.name, time);
}
```

**Test Criteria**:
- Works with any XML configuration
- Parameter extraction handles all types correctly
- Image fetching works for any number of inputs
- Error handling for disconnected optional inputs

### Step 3.6: Signature Generation Script Implementation
**Goal**: Create script to generate kernel signatures from XML definitions.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement `generate_signature.py` script
2. Add XML parsing to extract parameter names, types, and order
3. Add input source parsing for texture parameters
4. Generate CUDA, OpenCL, and Metal signature variants
5. Create template signatures in separate files for copy-paste

**Script Usage**:
```bash
python generate_signature.py GaussianBlur.xml
# Creates: GaussianBlur_cuda_signature.txt
#         GaussianBlur_opencl_signature.txt  
#         GaussianBlur_metal_signature.txt
```

**Test Criteria**:
- Generated signatures match XML parameter order exactly
- All parameter types are correctly converted to GPU types
- Input sources become appropriate texture parameters
- Script handles optional inputs correctly

### Step 3.7: Dynamic Kernel Wrapper Pattern Implementation
**Goal**: Create pattern for effect-specific kernel wrappers that convert maps to explicit parameters.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement example `RunGaussianBlurKernel` wrapper function
2. Create parameter extraction pattern from ParameterValue maps
3. Add image-to-texture conversion utilities
4. Implement error handling for missing parameters/inputs
5. Document wrapper pattern for other effects

**Example Implementation**:
```cpp
void RunGaussianBlurKernel(void* stream, int width, int height,
                          const std::map<std::string, OFX::Image*>& images,
                          const std::map<std::string, ParameterValue>& params) {
    // Extract parameters
    float radius = params.at("radius").asFloat();
    int quality = params.at("quality").asInt();
    
    // Setup textures
    cudaTextureObject_t source = setupCudaTexture(images.at("source"));
    
    // Call kernel
    GaussianBlurKernel<<<blocks, threads>>>(width, height, radius, quality, source, output);
}
```

**Test Criteria**:
- Parameter extraction works for all XML-defined parameters
- Texture setup works for all XML-defined inputs
- Error handling for missing required parameters/inputs
- Memory management is correct (texture cleanup)

### Step 3.8: Format Detection System Implementation
**Goal**: Replace BlurPlugin's hard-coded float RGBA assumption with dynamic format detection.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement pixel format detection in GenericProcessor
2. Add format validation before kernel calling
3. Create format conversion utilities if needed
4. Add unsupported format error handling
5. Document supported formats for effect authors

**Legacy Issue Addressed**:
```cpp
// BlurPlugin assumption:
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>(); // Hard-coded!

// GenericEffect approach:
OFX::BitDepthEnum bitDepth = srcImg->getPixelDepth();
OFX::PixelComponentEnum components = srcImg->getPixelComponents();
if (bitDepth != OFX::eBitDepthFloat || components != OFX::ePixelComponentRGBA) {
    // Handle other formats or report error
}
```

**Test Criteria**:
- Format detection works correctly for all supported formats
- Clear error messages for unsupported formats
- Framework can be extended to support additional formats
- Performance impact is minimal

## Phase 3 Integration Tests

### Step 3.9: End-to-End XML Effect Test
**Goal**: Validate complete GenericEffect pipeline with real XML file.  
**Status**: 🔲 Not Started

**Tasks**:
1. Create test XML file with representative parameters/inputs
2. Test GenericEffectFactory creation and registration
3. Test GenericEffect instantiation in OFX host environment
4. Validate parameter UI creation matches XML
5. Test render flow without actual kernel execution

**Test XML Example**: Use enhanced GaussianBlur.xml from existing test files.

**Success Metrics**:
- XML effect loads and registers with OFX host
- UI controls match XML parameter definitions
- Render flow executes without crashes
- Memory usage is reasonable

### Step 3.10: Comparative Testing vs BlurPlugin
**Goal**: Ensure GenericEffect produces equivalent results to legacy approach.  
**Status**: 🔲 Not Started

**Tasks**:
1. Create equivalent XML definition for existing BlurPlugin
2. Side-by-side testing of parameter creation
3. Compare memory usage patterns
4. Validate UI organization equivalence
5. Performance benchmarking

**Validation Criteria**:
- Parameter count and types match exactly
- UI layout is equivalent or better
- Memory usage is similar or improved
- Performance overhead is minimal

## Integration with Existing Framework

### Leverages Completed Components (Phase 1-2)
- ✅ XMLEffectDefinition for XML parsing
- ✅ XMLParameterManager for parameter creation  
- ✅ XMLInputManager for clip creation
- ✅ All existing helper classes (PluginParameters, PluginClips, etc.)

### Prepares for Phase 4 (Kernel Management)
- Dynamic parameter passing maps ready for kernel interface
- Effect name dispatch structure ready for dynamic kernel loading
- Platform abstraction ready for unified kernel management

## Technical Decisions Made

### OFX Lifecycle Clarification
**Issue**: Original plan didn't clearly separate describe vs. instance phases.  
**Solution**: GenericEffectFactory (describe) + GenericEffect (instance) pattern.

### Parameter Passing Strategy  
**Issue**: GPU kernels can't receive arbitrary dictionaries.  
**Solution**: CPU wrapper functions extract parameters and call kernels with explicit signatures.

### Kernel Signature Management
**Issue**: How do kernel authors know what signature to use?  
**Solution**: Signature generation script creates template signatures from XML.

### Memory Ownership Pattern
**Issue**: Complex ownership with unique_ptr and raw pointers.  
**Solution**: "Doomed from birth" pattern - unique_ptr owns, processor borrows raw pointers during synchronous processing.

## Success Metrics for Phase 3

### Technical Achievements
- [ ] GenericEffect can load any XML effect definition
- [ ] Dynamic parameter system supports all XML parameter types
- [ ] Dynamic clip system supports arbitrary input configurations  
- [ ] Memory management is leak-free and performant
- [ ] Integration with existing XMLParameterManager/XMLInputManager works

### Developer Experience
- [ ] Creating new effect requires only XML + kernel files (no C++ knowledge)
- [ ] Parameter changes require only XML editing (no recompilation)
- [ ] Signature generation script eliminates kernel signature guessing
- [ ] Error messages are clear and actionable

### Performance Goals
- [ ] Memory usage equivalent to BlurPlugin approach
- [ ] Parameter extraction overhead is minimal
- [ ] Dynamic dispatch performance is acceptable
- [ ] No memory leaks in dynamic parameter system

## Next Phase Preview

**Phase 4**: With GenericEffect providing the dynamic effect infrastructure, Phase 4 will focus on the dynamic kernel interface - creating the wrapper functions that convert parameter maps to specific kernel calls, implementing the signature generation system, and creating the unified kernel management system.

The GenericEffect architecture is designed to support both single-kernel (Version 1) and multi-kernel pipeline (Version 2) effects, ensuring the framework can evolve without architectural changes.

---
*Phase 3 design completed based on BlurPlugin.cpp analysis and GenericEffect architecture sessions*