# XML-Based OFX Framework Implementation Plan - Version 1 (Updated)

## Introduction

This document outlines a detailed step-by-step implementation plan for Version 1 of the XML-based OFX image processing framework. The plan is structured into small, testable increments to ensure stable progress and minimize risks.

> **Key Principle**: The user should only need to modify XML effect definitions and kernel code files, never the framework code itself.

## Phase 1: Core XML Parsing and Validation ✅ COMPLETED (May 17, 2025)

### Step 1.1: Basic XML Schema Design ✅ COMPLETED
**Goal**: Create a well-defined XML schema for effect definitions.  
**Status**: ✅ Complete - XML schema defined and documented

**Tasks**:
1. Design XML schema with inputs, parameters, UI, and kernel sections
2. Include attribute-based parameters with label/hint as attributes
3. Add border_mode attributes for source inputs
4. Create sample GaussianBlur.xml based on schema

**Test Criteria**:
- XML schema is complete and documented ✅
- Sample XML is valid against schema ✅

**Notes**: Schema supports comprehensive parameter types (double, int, bool, choice, color, vec2, string, curve), UI organization, border modes, and both single-kernel and multi-kernel (pipeline) effects.

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

**Test Criteria**:
- XML parser correctly loads all elements and attributes ✅
- Error handling for invalid XML files works correctly ✅
- Accessors return correct values ✅

**Notes**: Implementation successfully handles complex XML with nested components, validates references, and provides comprehensive accessor methods.

### Step 1.3: Unit Tests for XML Parsing ✅ COMPLETED
**Goal**: Create comprehensive tests for XML parsing.  
**Status**: ✅ Complete - Comprehensive test coverage

**Tasks**:
1. Create test suite for XMLEffectDefinition ✅
2. Test all getter methods with various XML inputs ✅
3. Test error handling with malformed XML ✅

**Test Criteria**:
- All tests pass with valid XML ✅
- Invalid XML is rejected with appropriate error messages ✅
- Edge cases (optional attributes, missing sections) are handled correctly ✅

**Notes**: Test program provides detailed output showing all parsed elements, confirming comprehensive XML processing capability.

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

**Implementation Details**:
- Comprehensive parameter type support
- Resolution dependency handling
- Component-level control for vector/color parameters
- Proper OFX API integration

**Test Criteria**:
- Parameters created match XML definitions ✅
- Parameter properties (default, range, labels) are correctly set ✅
- UI organization is applied correctly ✅

**Notes**: Successfully creates all parameter types with mock OFX objects. Ready for integration with real OFX plugins.

### Step 2.2: XMLInputManager Class ✅ COMPLETED
**Goal**: Create a class to map XML input definitions to OFX clips.  
**Status**: ✅ Complete - Full implementation with border mode support

**Tasks**:
1. Implement XMLInputManager class ✅
2. Add support for creating source clips with proper labels ✅
3. Add support for optional clips ✅
4. Store border mode information for each clip ✅

**Implementation Details**:
- Automatic mask detection based on naming conventions
- Border mode property setting via OFX PropertySet API
- Optional clip handling
- Comprehensive error handling

**Test Criteria**:
- Clips created match XML definitions ✅
- Optional clips are properly flagged ✅
- Border modes are correctly stored for each clip ✅

**Notes**: Successfully handles all clip types including masks. Border modes properly set via OFX properties for use in kernel processing.

### Step 2.3: Integration with BlurPluginFactory ⏳ NEXT STEP
**Goal**: Create non-destructive integration test with existing plugin.  
**Status**: 🔲 Pending

**Tasks**:
1. Create a test harness in BlurPluginFactory
2. Add XML-based parameter and clip creation alongside existing code
3. Add logging to compare XML vs. manual results

**Test Criteria**:
- XML-based clips and parameters match manually-created ones
- Log comparison shows equivalence
- Fallback to original works correctly

**Notes**: Ready to begin integration testing with actual OFX plugin.

## Phase 3: Dynamic Effect Base Class

### Step 3.1: GenericEffect Base Class
**Goal**: Create a base class for XML-defined effects.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement GenericEffect extending OFX::ImageEffect
2. Add dynamic parameter storage for various types
3. Add dynamic clip storage including border modes
4. Add methods to load from XMLEffectDefinition

**Test Criteria**:
- GenericEffect successfully loads parameters from XML
- Parameters can be accessed and used in the effect
- Clips are properly loaded with border modes

### Step 3.2: Identity Condition Implementation
**Goal**: Implement XML-driven identity condition checking.  
**Status**: 🔲 Not Started

**Tasks**:
1. Implement isIdentity method in GenericEffect
2. Process identity conditions from XML definition

**Test Criteria**:
- Identity conditions from XML work correctly
- Different operators function as expected
- Identity behavior matches original plugin

## Phase 4: Image Processing and Kernel Management

### Step 4.1: GenericProcessor Class
**Goal**: Create a processor that handles dynamic parameter passing.  
**Status**: 🔲 Not Started

### Step 4.2: KernelManager Implementation
**Goal**: Create a system to manage GPU kernel execution.  
**Status**: 🔲 Not Started

### Step 4.3: Render Method Implementation
**Goal**: Implement the render method in GenericEffect.  
**Status**: 🔲 Not Started

### Step 4.4: CUDA Kernel Implementation
**Goal**: Create sample CUDA kernel with standard entry point.  
**Status**: 🔲 Not Started

## Phase 5: Plugin Factory and Testing

### Step 5.1: XMLEffectFactory Implementation
**Goal**: Create a factory that generates plugins from XML.  
**Status**: 🔲 Not Started

### Step 5.2: Plugin Registration System
**Goal**: Create a system to automatically register XML-defined plugins.  
**Status**: 🔲 Not Started

### Step 5.3: Integration Testing
**Goal**: Test the framework with a complete example.  
**Status**: 🔲 Not Started

## Phase 6: Documentation and Packaging

### Step 6.1: Developer Documentation
**Goal**: Create documentation for framework developers.  
**Status**: 🔲 Not Started

### Step 6.2: User Documentation
**Goal**: Create documentation for effect authors.  
**Status**: 🔲 Not Started

### Step 6.3: Example Effects
**Goal**: Create example effects to demonstrate the framework.  
**Status**: 🔲 Not Started

## Technical Notes and Lessons Learned

### Build System Challenges (Resolved)
**Challenge**: OFX Support library compilation issues with ofxsImageEffect.cpp
**Solution**: 
- Identified that original Makefile successfully compiles OFX Support library
- Used same compilation flags and approach for XML test framework
- Created minimal plugin stub to satisfy OFX::Plugin::getPluginIDs requirement
- Properly separated dependencies (XML-only vs XML+OFX components)

**Key Insight**: The existing build system works correctly; new components must follow the same patterns for OFX integration.

### Mock Testing Strategy
**Challenge**: OFX descriptor classes have protected constructors, making direct instantiation impossible
**Solution**: Created comprehensive mock classes that simulate OFX behavior without inheriting from OFX classes
**Result**: Enables thorough testing of XML managers without requiring full OFX host environment

### XML Schema Evolution
**Current**: Single-kernel effects with comprehensive parameter support
**Future**: Multi-kernel pipeline support already designed into schema
**Status**: Schema is forward-compatible with Version 2 multi-kernel features

### Environment Compatibility
**Platform**: Rocky Linux 8.10, g++ compiler, OFX 1.4
**Status**: ✅ Fully compatible and tested
**OFX Location**: ../OpenFX-1.4/include, ../Support/include

## Timeline Summary (Updated)

**Phase 1: Core XML Parsing and Validation** ✅ 4 days (May 17, 2025)
- XML Schema Design
- XMLEffectDefinition Class  
- Unit Tests

**Phase 2: OFX Parameter Creation** ✅ 4 days (May 17, 2025)
- XMLParameterManager Class
- XMLInputManager Class
- Build system integration and testing

**Phase 3: Dynamic Effect Base Class** 🔲 3-4 days (Estimated)
- GenericEffect Base Class
- Identity Condition Implementation

**Phase 4: Image Processing and Kernel Management** 🔲 7-11 days (Estimated)
- GenericProcessor Class
- KernelManager Implementation
- Render Method Implementation
- CUDA Kernel Implementation

**Phase 5: Plugin Factory and Testing** 🔲 5-8 days (Estimated)
- XMLEffectFactory Implementation
- Plugin Registration System
- Integration Testing

**Phase 6: Documentation and Packaging** 🔲 3-6 days (Estimated)
- Developer Documentation
- User Documentation
- Example Effects

**Progress**: 8/43 days completed (19% complete)
**Next Milestone**: Step 2.3 - Integration with BlurPluginFactory

## Conclusion

The XML-based OFX framework foundation is now complete and thoroughly tested. All core XML parsing and OFX integration components are working correctly with the actual OFX Support library. The framework successfully demonstrates the ability to create OFX parameters and clips from XML definitions, providing a solid foundation for the complete system.

Key achievements:
1. ✅ Robust XML parsing with comprehensive error handling
2. ✅ Complete OFX parameter creation from XML
3. ✅ Full OFX clip management with border modes
4. ✅ Successful integration with OFX Support library
5. ✅ Comprehensive test coverage with mock objects

The framework is ready to proceed with GenericEffect implementation and real plugin integration, maintaining the goal of allowing image processing artists to create effects by only modifying XML definitions and kernel code files.