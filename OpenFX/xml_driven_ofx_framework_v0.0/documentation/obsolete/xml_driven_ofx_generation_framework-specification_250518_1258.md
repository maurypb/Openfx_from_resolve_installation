# XML-Based OFX Image Processing Framework Specification (Updated)

> **IMPORTANT NOTE**: This framework is designed so that image processing artists only need to modify two types of files:
> 1. XML effect definition files - defining parameters, inputs, and processing steps
> 2. Kernel code files (.cu/.cl/.metal) - containing the image processing algorithms
>
> No C++ knowledge or modification of the framework code is required to create new effects.

## 1. Overview

This document describes the architecture for an XML-based OpenFX (OFX) image processing framework. The system simplifies the creation of new OFX plugins by allowing image processing artists to focus on writing GPU kernels and parameter definitions without needing to understand the OFX C++ infrastructure.

### 1.1 Lessons from Legacy Code Analysis

Analysis of existing OFX plugins (particularly BlurPlugin.cpp) revealed several limitations in traditional OFX development that this framework addresses:

#### Current Limitations:
- **Fixed Parameter Signatures**: Adding new parameters requires modifying C++ function signatures
  ```cpp
  // Current rigid approach:
  RunCudaKernel(stream, width, height, radius, quality, maskStrength, input, mask, output);
  ```
- **Hard-coded Pixel Formats**: Effects assume specific formats (e.g., RGBA float) rather than detecting dynamically
- **Manual GPU Platform Handling**: Separate implementations for CUDA/OpenCL/Metal with duplicate code
- **Poor API Naming**: Generic names like `p_Args` provide no indication of contents
- **Fixed Input Structure**: Effects assume specific input patterns (source + optional mask)

#### Framework Solutions:
- **Dynamic Parameter Maps**: Parameters defined in XML, automatically passed as key-value pairs
  ```cpp
  // New flexible approach:
  std::map<std::string, ParameterValue> params = getParamsFromXML();
  RunGenericKernel(stream, width, height, params, inputBuffers, outputBuffer);
  ```
- **Automatic Format Detection**: Framework detects pixel format from image metadata
- **Unified GPU Interface**: Single kernel interface across all platforms
- **Descriptive Naming**: APIs clearly indicate purpose (e.g., `RenderContext` vs `p_Args`)
- **Arbitrary Input Support**: Any number of inputs with any names from XML

The framework will be implemented in two phases:
- **Version 1**: A single-kernel system supporting GPU-accelerated image effects with XML-defined parameters and clips
- **Version 2**: A multi-kernel system supporting sequential kernels, iterative processing, and complex effects

## 2. System Components

### 2.1 Core Components

The framework consists of these primary components:

1. **XML Effect Definitions**: XML files that describe effect parameters, inputs, and processing logic
2. **XML Parser**: System to read and validate XML effect definitions
3. **Parameter Manager**: Handles parameter creation, UI, and dynamic value passing to kernels
4. **Input Manager**: Manages arbitrary input/output connections between the plugin and host
5. **Kernel Manager**: Loads and executes GPU kernels with unified interface across platforms
6. **Buffer Manager**: Manages image buffers and automatic format detection
7. **Generic Effect**: Base class for dynamically-generated OFX plugins
8. **Factory System**: Creates OFX plugins from XML definitions

### 2.2 Improved Architecture Based on Legacy Analysis

Based on analysis of traditional OFX plugin structure, the framework implements these improvements:

#### Parameter Handling Evolution:
```cpp
// Legacy approach (BlurPlugin.cpp):
class BlurPlugin {
    OFX::DoubleParam* m_Radius;      // Fixed parameter set
    OFX::IntParam* m_Quality;        // Hard-coded in constructor
    OFX::DoubleParam* m_MaskStrength;
    
    void render() {
        double radius = m_Radius->getValueAtTime(time);  // Manual retrieval
        // Fixed function call:
        RunCudaKernel(stream, w, h, radius, quality, strength, input, mask, output);
    }
};

// Framework approach:
class GenericEffect {
    std::map<std::string, OFX::Param*> m_DynamicParams;  // Any parameters from XML
    
    void render() {
        auto params = getParameterValues(time);  // Automatic from XML
        // Dynamic function call:
        runKernelFromXML(platformContext, dimensions, params, inputs, outputs);
    }
};
```

#### Input Management Evolution:
```cpp
// Legacy approach:
OFX::Clip* m_SrcClip = fetchClip("Source");  // Hard-coded clip names
OFX::Clip* m_MaskClip = fetchClip("Mask");   // Fixed input structure

// Framework approach:
std::map<std::string, OFX::Clip*> m_DynamicClips;  // From XML definition
for (auto& inputDef : xmlDef.getInputs()) {        // Arbitrary inputs
    m_DynamicClips[inputDef.name] = fetchClip(inputDef.name);
}
```

### 2.3 XML Schema

The XML schema defines the structure of effect definitions, addressing legacy limitations:

```xml
<effect name="EffectName" category="Category">
  <description>Effect description text</description>
  
  <!-- Arbitrary number of inputs with any names (vs fixed source/mask) -->
  <inputs>
    <source name="foreground" label="Foreground Image" border_mode="clamp" />
    <source name="background" label="Background Image" border_mode="repeat" />
    <source name="displacement_map" label="Displacement" border_mode="black" />
    <source name="control_matte" label="Control Matte" optional="true" />
  </inputs>
  
  <!-- Unlimited parameters with automatic type handling -->
  <parameters>
    <parameter name="blur_radius" type="double" default="5.0" min="0.0" max="100.0" 
               res_dependent="width" label="Blur Radius" hint="Blur radius in pixels" />
    
    <parameter name="edge_threshold" type="double" default="0.2" min="0.0" max="1.0"
               label="Edge Threshold" hint="Threshold for edge detection" />
               
    <parameter name="blend_mode" type="choice" default="0"
               label="Blend Mode" hint="How to combine layers">
      <option value="0" label="Normal" />
      <option value="1" label="Add" />
      <option value="2" label="Multiply" />
    </parameter>
    
    <parameter name="enable_glow" type="bool" default="false"
               label="Enable Glow" hint="Add glow effect" />
               
    <!-- Any number of parameters with any names -->
  </parameters>
  
  <!-- UI organization with clear naming -->
  <ui>
    <page name="Main Controls">
      <column name="Blur Settings">
        <parameter>blur_radius</parameter>
        <parameter>edge_threshold</parameter>
      </column>
      <column name="Advanced">
        <parameter>blend_mode</parameter>
        <parameter>enable_glow</parameter>
      </column>
    </page>
  </ui>
  
  <!-- Dynamic identity conditions -->
  <identity_conditions>
    <condition>
      <parameter name="blur_radius" operator="lessEqual" value="0.0" />
    </condition>
  </identity_conditions>
  
  <!-- Version 1: Single kernel with dynamic parameters -->
  <kernels>
    <cuda file="AdvancedEffect.cu" />
    <opencl file="AdvancedEffect.cl" />
    <metal file="AdvancedEffect.metal" />
  </kernels>
</effect>
```

### 2.4 Enhanced Class Architecture

The framework's class architecture addresses legacy code limitations:

```
- XMLEffectDefinition
  - Parses effect metadata, unlimited parameters/inputs from XML
  - Supports automatic parameter and input validation

- XMLParameterManager  
  - Maps any XML parameter to appropriate OFX parameter type
  - Handles automatic UI generation with clear naming
  - Supports resolution-dependent parameters

- XMLInputManager
  - Creates clips for any number of inputs with any names
  - Handles border modes per input as specified in XML
  - Automatic mask detection based on naming patterns

- GenericEffect (inherits from OFX::ImageEffect)
  - Dynamically creates parameters and clips from XML
  - Replaces fixed parameter handling with map-based approach
  - Unified rendering across GPU platforms

- UnifiedKernelManager
  - Single interface for CUDA/OpenCL/Metal kernels
  - Automatic parameter passing from XML to kernels
  - Dynamic format detection and buffer management
  - Eliminates platform-specific code duplication

- XMLEffectFactory (inherits from OFX::PluginFactoryHelper)
  - Creates plugin instances from XML definitions
  - Improved naming and documentation vs legacy factories
```

## 3. Kernel Interface Improvements

### 3.1 Legacy Limitations Addressed

Analysis of BlurPlugin.cpp revealed these kernel interface issues:

#### Fixed Parameter Passing:
```cpp
// Legacy: Must modify signature for each new parameter
__global__ void GaussianBlurKernel(int width, int height, 
                                  float radius, int quality, float maskStrength,
                                  /* adding new param requires signature change */);
```

#### Hard-coded Pixel Format:
```cpp
// Legacy: Assumes RGBA float
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();  // Hard-coded!
```

#### Platform-specific Implementations:
```cpp
// Legacy: Separate functions for each platform
void processImagesCUDA();    // CUDA-specific implementation
void processImagesOpenCL();  // OpenCL-specific implementation  
void processImagesMetal();   // Metal-specific implementation
```

### 3.2 Framework Solutions

#### Dynamic Parameter Interface:
```cpp
// New approach: Universal kernel signature
__global__ void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    ParameterBlock* params,     // All XML parameters automatically provided
    InputBuffers* inputs,       // All XML inputs automatically provided
    OutputBuffers* outputs      // Output buffers
);

// Example usage in kernel:
float radius = params->getFloat("blur_radius");
int quality = params->getInt("sample_quality");
bool enableGlow = params->getBool("enable_glow");
```

#### Automatic Format Detection:
```cpp
// Framework automatically detects and handles format:
PixelFormat format = detectPixelFormat(inputImage);
KernelConfig config = createKernelConfig(format, dimensions);
```

#### Unified Platform Interface:
```cpp
// Single interface across all platforms:
class UnifiedKernel {
    void execute(PlatformContext& ctx, KernelParameters& params);
    // Automatically dispatches to appropriate platform implementation
};
```

### 3.3 Standard Kernel Entry Point

All kernel files provide a standardized `process` function with automatic parameter access:

```cpp
// CUDA example with framework improvements:
__global__ void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    // Framework automatically provides all XML-defined parameters:
    ParameterBlock* params,
    // Framework automatically provides all XML-defined inputs:
    InputBuffers* inputs,
    // Framework provides output buffer:
    OutputBuffers* outputs
) {
    // Get parameters by name from XML:
    float blurRadius = params->getFloat("blur_radius");
    float edgeThreshold = params->getFloat("edge_threshold");
    int blendMode = params->getInt("blend_mode");
    
    // Access inputs by name from XML:
    float4 foreground = inputs->sample("foreground", texCoord);
    float4 background = inputs->sample("background", texCoord);
    float displacementMap = inputs->sample("displacement_map", texCoord);
    
    // Process pixels...
    float4 result = processEffect(foreground, background, blurRadius, edgeThreshold);
    
    // Write result:
    outputs->write("output", result);
}
```

## 4. Implementation Priorities Based on Legacy Analysis

### 4.1 High Priority Fixes
1. **Replace Fixed Parameter Signatures**: Implement dynamic parameter system
2. **Eliminate Hard-coded Pixel Formats**: Add automatic format detection
3. **Unify GPU Platform Handling**: Create single kernel interface
4. **Improve API Naming**: Replace generic names with descriptive ones

### 4.2 Medium Priority Improvements  
1. **Automate Memory Management**: Reduce boilerplate buffer allocation code
2. **Enhanced Error Handling**: Better debugging for dynamic parameters
3. **Performance Optimization**: Maintain efficiency with flexible systems

### 4.3 Future Enhancements
1. **Advanced Parameter Types**: Support for curves, gradients, custom UI
2. **Multi-kernel Optimization**: Version 2 pipeline features
3. **Real-time Parameter Updates**: Live preview during adjustment

## 5. Developer Experience Improvements

### 5.1 Simplified Workflow

**Legacy approach** (what BlurPlugin.cpp demonstrates):
1. Create OFX plugin class with fixed parameters
2. Manually implement parameter fetching and UI creation
3. Write separate GPU kernel functions for each platform
4. Handle memory management and format conversion manually
5. Implement separate render paths for each GPU platform
6. Debug platform-specific issues separately

**Framework approach**:
1. Write XML file defining parameters and inputs
2. Write single kernel file with standard `process` entry point
3. Framework automatically handles all OFX infrastructure
4. Automatic parameter passing and memory management
5. Unified debugging across platforms

### 5.2 Key Principles Derived from Analysis

1. **Eliminate Boilerplate**: Framework handles OFX infrastructure automatically
2. **Clear Intent**: Descriptive naming replaces generic technical terms
3. **Flexibility**: Support any number of parameters/inputs vs fixed assumptions
4. **Platform Agnostic**: Write once, run on all supported GPUs
5. **Type Safety**: Automatic parameter validation from XML definitions

## 6. Conclusion

Analysis of legacy OFX plugin code reveals significant opportunities for simplification and improvement. This XML-based framework addresses concrete limitations found in traditional development:

- **Dynamic vs Fixed**: Moving from rigid parameter lists to flexible XML-driven configuration
- **Generic vs Hard-coded**: Supporting detected formats rather than assumptions
- **Unified vs Platform-specific**: Single development path instead of multiple implementations
- **Clear vs Cryptic**: Self-documenting APIs instead of generic technical names

The framework transforms OFX plugin development from a complex C++ programming task to a straightforward XML configuration and kernel development process, enabling image processing artists to focus on creative algorithms rather than infrastructure concerns.

---
*Updated based on detailed analysis of BlurPlugin.cpp - May 2025*
