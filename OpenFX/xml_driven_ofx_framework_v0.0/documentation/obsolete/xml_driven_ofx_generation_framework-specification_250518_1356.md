# XML-Based OFX Image Processing Framework Specification

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

The XML schema defines the structure of effect definitions, accommodating both Version 1 and Version 2 features:

```xml
<effect name="EffectName" category="Category">
  <description>Effect description text</description>
  
  <!-- Define arbitrary input sources (vs fixed source/mask pattern) -->
  <inputs>
    <source name="source" label="Main Image" border_mode="clamp" />
    <source name="matte" label="Matte Input" optional="true" border_mode="black" />
    <source name="foreground" label="Foreground Layer" border_mode="repeat" />
    <source name="displacement" label="Displacement Map" border_mode="mirror" />
    <!-- Any number of sources with any names and individual border modes -->
  </inputs>
  
  <!-- Parameters define UI controls and processing values -->
  <parameters>
    <parameter name="radius" type="double" default="5.0" min="0.0" max="100.0" 
               displayMin="0.0" displayMax="50.0" res_dependent="width">
      <label>Radius</label>
      <hint>Blur radius in pixels</hint>
    </parameter>
    
    <parameter name="quality" type="int" default="8" min="1" max="32" 
               displayMin="1" displayMax="16">
      <label>Quality</label>
      <hint>Number of samples for the blur</hint>
    </parameter>
    
    <parameter name="border_mode" type="choice" default="0">
      <label>Border Mode</label>
      <hint>How to handle pixels at the edge of the image</hint>
      <option value="0" label="Clamp" />
      <option value="1" label="Repeat" />
      <option value="2" label="Mirror" />
      <option value="3" label="Black" />
    </parameter>
    
    <parameter name="alpha_fade" type="curve" default_shape="ease_out">
      <label>Alpha Fade</label>
      <hint>Controls alpha falloff from inside to outside</hint>
    </parameter>
    
    <parameter name="enable_glow" type="bool" default="false">
      <label>Enable Glow</label>
      <hint>Add glow effect to the result</hint>
    </parameter>
    
    <parameter name="blend_color" type="rgba" default="1.0,1.0,1.0,1.0">
      <label>Blend Color</label>
      <hint>Color to blend with the effect</hint>
    </parameter>
    
    <!-- Unlimited parameters with any names and types -->
  </parameters>
  
  <!-- UI organization with descriptive naming -->
  <ui>
    <page name="Main Controls">
      <column name="Blur Settings">
        <parameter>radius</parameter>
        <parameter>quality</parameter>
      </column>
      <column name="Advanced Options">
        <parameter>alpha_fade</parameter>
        <parameter>border_mode</parameter>
      </column>
    </page>
    <page name="Color & Effects">
      <column name="Color Controls">
        <parameter>blend_color</parameter>
      </column>
      <column name="Effects">
        <parameter>enable_glow</parameter>
      </column>
    </page>
  </ui>
  
  <!-- Identity conditions define when the effect is a pass-through -->
  <identity_conditions>
    <condition>
      <parameter name="radius" operator="lessEqual" value="0.0" />
    </condition>
    <condition>
      <parameter name="enable_glow" operator="equals" value="false" />
      <parameter name="blend_color" operator="equals" value="1.0,1.0,1.0,1.0" />
    </condition>
    <!-- Multiple conditions for complex identity logic -->
  </identity_conditions>
  
  <!-- Version 1: Single kernel processing with dynamic parameters -->
  <kernels>
    <cuda file="EffectKernel.cu" executions="1" />
    <opencl file="EffectKernel.cl" executions="1" />
    <metal file="EffectKernel.metal" executions="1" />
  </kernels>
  
  <!-- Version 2: Multi-kernel pipeline (replaces the kernels section) -->
  <!-- <pipeline>
    <step name="EdgeDetect" executions="1">
      <kernels>
        <cuda file="EdgeDetect.cu" />
        <opencl file="EdgeDetect.cl" />
        <metal file="EdgeDetect.metal" />
      </kernels>
    </step>
    
    <step name="Blur" executions="3">
      <kernels>
        <cuda file="GaussianBlur.cu" />
        <opencl file="GaussianBlur.cl" />
        <metal file="GaussianBlur.metal" />
      </kernels>
    </step>
    
    <step name="Composite" executions="1">
      <kernels>
        <cuda file="Composite.cu" />
        <opencl file="Composite.cl" />
        <metal file="Composite.metal" />
      </kernels>
    </step>
  </pipeline> -->
</effect>
```

### 2.4 Enhanced Class Architecture

The framework's class architecture addresses legacy limitations while supporting both Version 1 and Version 2:

```
- XMLEffectDefinition
  - Parses and stores effect metadata, unlimited parameters, inputs, kernels
  - Supports future expansion to multi-kernel effects
  - Validates parameter types and constraints from XML

- XMLParameterManager
  - Maps any XML parameter definitions to OFX parameters dynamically
  - Creates UI controls based on XML definitions with descriptive naming
  - Handles resolution-dependent parameters and animation curves
  - Replaces fixed parameter handling with flexible map-based approach

- XMLInputManager
  - Creates clips for arbitrary number of inputs with any names
  - Handles individual border modes per input as specified in XML
  - Supports optional inputs and automatic mask detection

- GenericEffect (inherits from OFX::ImageEffect)
  - Base class for all XML-defined effects with dynamic creation
  - Manages parameter storage and access via descriptive APIs
  - Handles rendering and processing with unified interface
  - Extensible for multi-kernel processing in Version 2

- BufferManager
  - Manages image buffers for processing with automatic format detection
  - Handles buffer allocation, access, and cleanup efficiently
  - Supports future expansion to multi-kernel with intermediate buffers
  - Eliminates hard-coded pixel format assumptions

- UnifiedKernelManager
  - Loads and executes GPU kernels across CUDA, OpenCL, and Metal
  - Handles dynamic parameter passing to kernel functions
  - Uses standardized entry point (process) for all kernels
  - Replaces platform-specific implementations with unified interface

- XMLEffectFactory (inherits from OFX::PluginFactoryHelper)
  - Creates plugin instances from XML definitions
  - Handles OFX lifecycle and metadata with clear naming
  - Supports dynamic plugin registration from XML files
```

## 3. Two-Phase Implementation Plan

### 3.1 Version 1 Features

Version 1 implements a single-kernel image processing framework with:

1. **XML-Defined Parameters**: Unlimited effect parameters defined in XML with any names/types
2. **XML-Defined Inputs**: Arbitrary input sources defined in XML with individual border modes
3. **GPU Acceleration**: Support for CUDA, OpenCL, and Metal kernels with unified interface
4. **UI Organization**: Parameters organized into pages and columns with descriptive naming
5. **Dynamic Parameter Handling**: All parameters available to kernels via map-based access
6. **Standard Kernel Interface**: One function per file with standardized entry point
7. **Forward-Compatible Architecture**: Designed to support Version 2 features seamlessly
8. **Automatic Format Detection**: Support for multiple pixel formats without hard-coding
9. **Border Handling**: Per-source border modes (clamp, repeat, mirror, black)

### 3.2 Version 2 Features

Version 2 extends the framework with:

1. **Multi-Kernel Processing**: Sequential execution of multiple kernels in pipeline
2. **Multi-Execution Kernels**: Support for kernels that execute multiple times
3. **Intermediate Buffers**: Management of buffers between kernels automatically
4. **Automatic Data Flow**: Previous outputs automatically available to subsequent kernels
5. **Advanced Parameter Types**: Curves, gradients, custom UI elements
6. **Performance Optimization**: Buffer reuse and memory management improvements

Enhanced features:
- Access to results from previous kernels by name
- Access to previous iterations within multi-execution kernels
- Buffer swapping for efficient iterative processing
- Configurable execution counts for each kernel step

## 4. Kernel Interface Evolution

### 4.1 Legacy Limitations Addressed

Analysis of BlurPlugin.cpp revealed these kernel interface issues:

#### Fixed Parameter Passing:
```cpp
// Legacy: Must modify signature for each new parameter
__global__ void GaussianBlurKernel(int width, int height, 
                                  float radius, int quality, float maskStrength);
// Adding new parameter requires changing signature everywhere
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

### 4.2 Framework Solutions

#### Dynamic Parameter Interface:
```cpp
// New approach: Universal kernel signature with parameter access helpers
__global__ void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    ParameterBlock* params,     // All XML parameters automatically provided
    InputBuffers* inputs,       // All XML inputs automatically provided
    OutputBuffers* outputs      // Output buffers with automatic format handling
);

// Example usage in kernel:
float radius = params->getFloat("radius");
int quality = params->getInt("quality");
bool enableGlow = params->getBool("enable_glow");
float4 blendColor = params->getRGBA("blend_color");
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

### 4.3 Standard Kernel Entry Point

All kernel files must provide a standardized entry point function named `process`. This function receives all parameters defined in the XML, along with standard arguments for image dimensions and buffers.

```cpp
// CUDA kernel example with standard entry point
__global__ void process(
    int width, int height,         // Image dimensions
    int executionNumber,           // Current execution number (for multi-execution)
    int totalExecutions,           // Total executions for this kernel
    // All XML-defined inputs automatically available:
    float* sourceBuffer, int sourceBorderMode,      // Main source with border mode
    float* matteBuffer, int matteBorderMode,        // Optional matte input
    float* foregroundBuffer, int foregroundBorderMode,  // Additional layers
    float* outputBuffer,           // Output image
    // All effect parameters defined in XML follow automatically:
    float radius, 
    int quality,
    bool enableGlow,
    float4 blendColor
    // Any number of parameters with any names/types
);

// OpenCL kernel example
__kernel void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    __read_only image2d_t sourceBuffer, int sourceBorderMode,
    __read_only image2d_t matteBuffer, int matteBorderMode,
    __write_only image2d_t outputBuffer,
    float radius,
    int quality,
    bool enableGlow,
    float4 blendColor
    // Unlimited parameters from XML
);

// Metal kernel example
kernel void process(
    uint2 gid [[thread_position_in_grid]],
    constant int& width [[buffer(0)]],
    constant int& height [[buffer(1)]],
    constant int& executionNumber [[buffer(2)]],
    constant int& totalExecutions [[buffer(3)]],
    texture2d<float, access::read> sourceBuffer [[texture(0)]],
    constant int& sourceBorderMode [[buffer(4)]],
    texture2d<float, access::read> matteBuffer [[texture(1)]],
    constant int& matteBorderMode [[buffer(5)]],
    texture2d<float, access::write> outputBuffer [[texture(2)]],
    constant float& radius [[buffer(6)]],
    constant int& quality [[buffer(7)]],
    constant bool& enableGlow [[buffer(8)]],
    constant float4& blendColor [[buffer(9)]
    // Framework automatically passes all XML parameters
);
```

### 4.4 Version 2 Multi-Kernel Interface

In Version 2, kernels receive additional inputs for previous processing steps. All buffers from previous steps are automatically available, named according to the step that produced them.

```cpp
// CUDA kernel example for multi-kernel effect (Composite step)
__global__ void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    float* sourceBuffer,           // Original source image
    float* EdgeDetectBuffer,       // Output from EdgeDetect step
    float* BlurBuffer,             // Output from Blur step (after 3 executions)
    float* outputBuffer,           // Output for this step
    // All parameters from XML still available:
    float edgeThreshold,
    float blurRadius,
    float sharpAmount
    // Unlimited parameters supported
);
```

## 5. User Workflow

### 5.1 Artist Workflow (Version 1)

1. Create an XML file defining the effect parameters and inputs (unlimited count/names)
2. Write GPU kernels for CUDA, OpenCL, and/or Metal with the standard entry point
3. Place XML and kernel files in the plugin directory
4. Framework automatically loads and registers the effect with dynamic parameter handling

### 5.2 Artist Workflow (Version 2)

1. Create an XML file defining multiple processing steps with any parameters
2. Write GPU kernels for each processing step with the standard entry point
3. Place XML and kernel files in the plugin directory
4. Framework handles multi-kernel execution and buffer management automatically

### 5.3 Example: Enhanced Blur Effect (Version 1)

**EnhancedBlur.xml**:
```xml
<effect name="EnhancedBlur" category="Blur">
  <description>Advanced blur with edge detection and glow effects</description>
  
  <!-- Multiple inputs with individual border modes -->
  <inputs>
    <source name="source" label="Main Image" border_mode="clamp" />
    <source name="edge_mask" label="Edge Mask" optional="true" border_mode="black" />
    <source name="glow_source" label="Glow Source" optional="true" border_mode="repeat" />
  </inputs>
  
  <!-- Unlimited parameters with descriptive names -->
  <parameters>
    <parameter name="blur_radius" type="double" default="5.0" min="0.0" max="100.0" 
               res_dependent="width">
      <label>Blur Radius</label>
      <hint>Blur radius in pixels</hint>
    </parameter>
    
    <parameter name="quality_samples" type="int" default="8" min="1" max="32">
      <label>Quality Samples</label>
      <hint>Number of samples for the blur</hint>
    </parameter>
    
    <parameter name="edge_preservation" type="double" default="0.5" min="0.0" max="1.0">
      <label>Edge Preservation</label>
      <hint>How much to preserve edges during blur</hint>
    </parameter>
    
    <parameter name="glow_intensity" type="double" default="0.0" min="0.0" max="2.0">
      <label>Glow Intensity</label>
      <hint>Strength of the glow effect</hint>
    </parameter>
    
    <parameter name="glow_color" type="rgba" default="1.0,1.0,1.0,1.0">
      <label>Glow Color</label>
      <hint>Color tint for the glow effect</hint>
    </parameter>
    
    <parameter name="border_handling" type="choice" default="0">
      <label>Global Border Mode</label>
      <hint>How to handle pixels at image edges</hint>
      <option value="0" label="Clamp" />
      <option value="1" label="Repeat" />
      <option value="2" label="Mirror" />
      <option value="3" label="Black" />
    </parameter>
    
    <parameter name="enable_advanced" type="bool" default="false">
      <label>Enable Advanced Features</label>
      <hint>Enable edge detection and glow</hint>
    </parameter>
  </parameters>
  
  <!-- Organized UI with clear structure -->
  <ui>
    <page name="Basic Blur">
      <column name="Blur Settings">
        <parameter>blur_radius</parameter>
        <parameter>quality_samples</parameter>
        <parameter>border_handling</parameter>
      </column>
    </page>
    <page name="Advanced Effects">
      <column name="Edge Control">
        <parameter>enable_advanced</parameter>
        <parameter>edge_preservation</parameter>
      </column>
      <column name="Glow Effects">
        <parameter>glow_intensity</parameter>
        <parameter>glow_color</parameter>
      </column>
    </page>
  </ui>
  
  <!-- Multiple identity conditions -->
  <identity_conditions>
    <condition>
      <parameter name="blur_radius" operator="lessEqual" value="0.0" />
    </condition>
    <condition>
      <parameter name="enable_advanced" operator="equals" value="false" />
      <parameter name="glow_intensity" operator="lessEqual" value="0.0" />
    </condition>
  </identity_conditions>
  
  <!-- Version 1: Single kernel with all features -->
  <kernels>
    <cuda file="EnhancedBlur.cu" executions="1" />
    <opencl file="EnhancedBlur.cl" executions="1" />
    <metal file="EnhancedBlur.metal" executions="1" />
  </kernels>
</effect>
```

**EnhancedBlur.cu**:
```cpp
#include <cuda_runtime.h>

// Helper device function
__device__ float gaussian(float x, float sigma) {
    return exp(-(x*x) / (2.0f * sigma * sigma));
}

// Enhanced border handling function
__device__ float4 sampleWithBorderMode(
    float* sourceBuffer,
    int width, int height,
    float x, float y,
    int borderMode
) {
    int ix = (int)x;
    int iy = (int)y;
    
    // Apply border mode
    switch(borderMode) {
        case 0: // Clamp
            ix = max(0, min(width-1, ix));
            iy = max(0, min(height-1, iy));
            break;
            
        case 1: // Repeat
            ix = ix % width;
            if (ix < 0) ix += width;
            iy = iy % height;
            if (iy < 0) iy += height;
            break;
            
        case 2: // Mirror
            if (ix < 0) ix = -ix;
            if (ix >= width) ix = 2*width - ix - 1;
            if (iy < 0) iy = -iy;
            if (iy >= height) iy = 2*height - iy - 1;
            break;
            
        case 3: // Black
            if (ix < 0 || ix >= width || iy < 0 || iy >= height) {
                return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            break;
    }
    
    // Sample the pixel
    int index = (iy * width + ix) * 4;
    return make_float4(
        sourceBuffer[index],
        sourceBuffer[index+1],
        sourceBuffer[index+2],
        sourceBuffer[index+3]
    );
}

// Standard entry point - framework automatically provides all XML parameters
__global__ void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    // All inputs from XML with their border modes
    float* sourceBuffer, int sourceBorderMode,
    float* edgeMaskBuffer, int edgeMaskBorderMode,        // Optional
    float* glowSourceBuffer, int glowSourceBorderMode,    // Optional
    float* outputBuffer,
    // All parameters from XML automatically provided
    float blurRadius,
    int qualitySamples,
    float edgePreservation,
    float glowIntensity,
    float4 glowColor,
    int borderHandling,
    bool enableAdvanced
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Sample original pixel
    float4 originalColor = sampleWithBorderMode(sourceBuffer, width, height, x, y, sourceBorderMode);
    
    // Early exit for identity conditions
    if (blurRadius <= 0.0f && (!enableAdvanced || glowIntensity <= 0.0f)) {
        int index = (y * width + x) * 4;
        outputBuffer[index] = originalColor.x;
        outputBuffer[index+1] = originalColor.y;
        outputBuffer[index+2] = originalColor.z;
        outputBuffer[index+3] = originalColor.w;
        return;
    }
    
    float4 blurredColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;
    
    // Gaussian blur implementation with edge preservation
    if (blurRadius > 0.0f) {
        for (int i = -qualitySamples; i <= qualitySamples; i++) {
            for (int j = -qualitySamples; j <= qualitySamples; j++) {
                float dx = (float)i * (blurRadius / qualitySamples);
                float dy = (float)j * (blurRadius / qualitySamples);
                float weight = gaussian(dx, blurRadius) * gaussian(dy, blurRadius);
                
                float4 sampleColor = sampleWithBorderMode(sourceBuffer, width, height, 
                                                        x + dx, y + dy, borderHandling);
                
                // Edge preservation logic
                if (enableAdvanced && edgeMaskBuffer && edgePreservation > 0.0f) {
                    float4 edgeMask = sampleWithBorderMode(edgeMaskBuffer, width, height, 
                                                         x + dx, y + dy, edgeMaskBorderMode);
                    float edgeStrength = (edgeMask.x + edgeMask.y + edgeMask.z) / 3.0f;
                    weight *= (1.0f - edgeStrength * edgePreservation);
                }
                
                blurredColor.x += sampleColor.x * weight;
                blurredColor.y += sampleColor.y * weight;
                blurredColor.z += sampleColor.z * weight;
                blurredColor.w += sampleColor.w * weight;
                totalWeight += weight;
            }
        }
        
        if (totalWeight > 0.0f) {
            blurredColor.x /= totalWeight;
            blurredColor.y /= totalWeight;
            blurredColor.z /= totalWeight;
            blurredColor.w /= totalWeight;
        }
    } else {
        blurredColor = originalColor;
    }
    
    // Add glow effect
    if (enableAdvanced && glowIntensity > 0.0f && glowSourceBuffer) {
        float4 glowSource = sampleWithBorderMode(glowSourceBuffer, width, height, 
                                                x, y, glowSourceBorderMode);
        
        float4 glow = make_float4(
            glowSource.x * glowColor.x * glowIntensity,
            glowSource.y * glowColor.y * glowIntensity,
            glowSource.z * glowColor.z * glowIntensity,
            glowSource.w * glowColor.w
        );
        
        // Additive blend for glow
        blurredColor.x = min(1.0f, blurredColor.x + glow.x);
        blurredColor.y = min(1.0f, blurredColor.y + glow.y);
        blurredColor.z = min(1.0f, blurredColor.z + glow.z);
        blurredColor.w = min(1.0f, blurredColor.w + glow.w);
    }
    
    // Write result
    int index = (y * width + x) * 4;
    outputBuffer[index] = blurredColor.x;
    outputBuffer[index+1] = blurredColor.y;
    outputBuffer[index+2] = blurredColor.z;
    outputBuffer[index+3] = blurredColor.w;
}
```

### 5.4 Example: Complex Multi-Kernel Effect (Version 2)

**EdgeEnhancement.xml**:
```xml
<effect name="EdgeEnhancement" category="Filter">
  <description>Advanced edge enhancement with multi-stage processing</description>
  
  <inputs>
    <source name="source" label="Input Image" border_mode="clamp" />
    <source name="control_mask" label="Control Mask" optional="true" border_mode="black" />
  </inputs>
  
  <parameters>
    <parameter name="edge_threshold" type="double" default="0.2" min="0.0" max="1.0">
      <label>Edge Threshold</label>
      <hint>Threshold for edge detection</hint>
    </parameter>
    <parameter name="blur_radius" type="double" default="3.0" min="0.0" max="10.0">
      <label>Blur Radius</label>
      <hint>Radius for edge smoothing</hint>
    </parameter>
    <parameter name="sharpen_amount" type="double" default="2.0" min="0.0" max="10.0">
      <label>Sharpen Amount</label>
      <hint>Strength of edge enhancement</hint>
    </parameter>
    
    <parameter name="blend_mix" type="double" default="0.8" min="0.0" max="1.0">
      <label>Blend Mix</label>
      <hint>Mix between original and enhanced</hint>
    </parameter>
    
    <parameter name="color_enhance" type="bool" default="true">
      <label>Color Enhancement</label>
      <hint>Apply color enhancement to edges</hint>
    </parameter>
  </parameters>
  
  <ui>
    <page name="Enhancement">
      <column name="Edge Control">
        <parameter>edge_threshold</parameter>
        <parameter>sharpen_amount</parameter>
      </column>
      <column name="Processing">
        <parameter>blur_radius</parameter>
        <parameter>blend_mix</parameter>
        <parameter>color_enhance</parameter>
      </column>
    </page>
  </ui>
  
  <identity_conditions>
    <condition>
      <parameter name="sharpen_amount" operator="lessEqual" value="0.0" />
      <parameter name="edge_threshold" operator="lessEqual" value="0.0" />
    </condition>
  </identity_conditions>
  
  <!-- Version 2: Multi-kernel pipeline -->
  <pipeline>
    <step name="EdgeDetect" executions="1">
      <kernels>
        <cuda file="EdgeDetect.cu" />
        <opencl file="EdgeDetect.cl" />
        <metal file="EdgeDetect.metal" />
      </kernels>
    </step>
    
    <step name="GaussianBlur" executions="3">
      <kernels>
        <cuda file="GaussianBlur.cu" />
        <opencl file="GaussianBlur.cl" />
        <metal file="GaussianBlur.metal" />
      </kernels>
    </step>
    
    <step name="EdgeEnhance" executions="1">
      <kernels>
        <cuda file="EdgeEnhance.cu" />
        <opencl file="EdgeEnhance.cl" />
        <metal file="EdgeEnhance.metal" />
      </kernels>
    </step>
    
    <step name="FinalComposite" executions="1">
      <kernels>
        <cuda file="FinalComposite.cu" />
        <opencl file="FinalComposite.cl" />
        <metal file="FinalComposite.metal" />
      </kernels>
    </step>
  </pipeline>
</effect>
```

**EdgeDetect.cu** (first step in the pipeline):
```cpp
#include <cuda_runtime.h>

// Standard entry point for edge detection
__global__ void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    // Input from XML
    float* sourceBuffer, int sourceBorderMode,
    float* controlMaskBuffer, int controlMaskBorderMode,  // Optional
    float* outputBuffer,
    // All parameters from XML
    float edgeThreshold,
    float blurRadius,
    float sharpenAmount,
    float blendMix,
    bool colorEnhance
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Sobel edge detection
    float3 sobel_x = make_float3(0.0f, 0.0f, 0.0f);
    float3 sobel_y = make_float3(0.0f, 0.0f, 0.0f);
    
    // Sobel kernels
    int sobel_kernel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sobel_kernel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    for (int i = 0; i < 9; i++) {
        int offsetX = (i % 3) - 1;
        int offsetY = (i / 3) - 1;
        
        // Sample with border mode
        int sampleX = x + offsetX;
        int sampleY = y + offsetY;
        
        // Apply border handling
        if (sourceBorderMode == 0) { // Clamp
            sampleX = max(0, min(width-1, sampleX));
            sampleY = max(0, min(height-1, sampleY));
        }
        // Add other border modes as needed
        
        int sampleIndex = (sampleY * width + sampleX) * 4;
        float3 sampleColor = make_float3(sourceBuffer[sampleIndex], 
                                       sourceBuffer[sampleIndex+1], 
                                       sourceBuffer[sampleIndex+2]);
        
        sobel_x.x += sampleColor.x * sobel_kernel_x[i];
        sobel_x.y += sampleColor.y * sobel_kernel_x[i];
        sobel_x.z += sampleColor.z * sobel_kernel_x[i];
        
        sobel_y.x += sampleColor.x * sobel_kernel_y[i];
        sobel_y.y += sampleColor.y * sobel_kernel_y[i];
        sobel_y.z += sampleColor.z * sobel_kernel_y[i];
    }
    
    // Calculate edge magnitude
    float edgeStrength = sqrt(sobel_x.x*sobel_x.x + sobel_y.x*sobel_y.x +
                             sobel_x.y*sobel_x.y + sobel_y.y*sobel_y.y +
                             sobel_x.z*sobel_x.z + sobel_y.z*sobel_y.z) / 3.0f;
    
    // Apply threshold
    edgeStrength = (edgeStrength > edgeThreshold) ? edgeStrength : 0.0f;
    
    // Apply control mask if available
    if (controlMaskBuffer) {
        int maskIndex = (y * width + x) * 4;
        float maskValue = controlMaskBuffer[maskIndex];
        edgeStrength *= maskValue;
    }
    
    // Write edge map
    int index = (y * width + x) * 4;
    outputBuffer[index] = edgeStrength;
    outputBuffer[index+1] = edgeStrength;
    outputBuffer[index+2] = edgeStrength;
    outputBuffer[index+3] = 1.0f;
}
```

**FinalComposite.cu** (final step - accesses all previous buffers):
```cpp
#include <cuda_runtime.h>

// Final composite step - automatically receives all previous outputs
__global__ void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    // Original inputs still available
    float* sourceBuffer, int sourceBorderMode,
    float* controlMaskBuffer, int controlMaskBorderMode,
    // Previous step outputs automatically available
    float* EdgeDetectBuffer,      // Output from EdgeDetect step
    float* GaussianBlurBuffer,    // Output from GaussianBlur step (after 3 executions)
    float* EdgeEnhanceBuffer,     // Output from EdgeEnhance step
    float* outputBuffer,          // Final output
    // All parameters still available
    float edgeThreshold,
    float blurRadius,
    float sharpenAmount,
    float blendMix,
    bool colorEnhance
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int index = (y * width + x) * 4;
    
    // Get original color
    float4 originalColor = make_float4(sourceBuffer[index], sourceBuffer[index+1], 
                                     sourceBuffer[index+2], sourceBuffer[index+3]);
    
    // Get enhanced edges
    float4 enhancedColor = make_float4(EdgeEnhanceBuffer[index], EdgeEnhanceBuffer[index+1],
                                     EdgeEnhanceBuffer[index+2], EdgeEnhanceBuffer[index+3]);
    
    // Get edge strength
    float edgeStrength = EdgeDetectBuffer[index];
    
    // Blend based on mix parameter
    float4 finalColor = make_float4(
        originalColor.x * (1.0f - blendMix) + enhancedColor.x * blendMix,
        originalColor.y * (1.0f - blendMix) + enhancedColor.y * blendMix,
        originalColor.z * (1.0f - blendMix) + enhancedColor.z * blendMix,
        originalColor.w
    );
    
    // Apply color enhancement if enabled
    if (colorEnhance && edgeStrength > 0.1f) {
        // Boost saturation on edges
        float luminance = 0.299f * finalColor.x + 0.587f * finalColor.y + 0.114f * finalColor.z;
        finalColor.x = luminance + (finalColor.x - luminance) * (1.0f + edgeStrength);
        finalColor.y = luminance + (finalColor.y - luminance) * (1.0f + edgeStrength);
        finalColor.z = luminance + (finalColor.z - luminance) * (1.0f + edgeStrength);
        
        // Clamp values
        finalColor.x = min(1.0f, max(0.0f, finalColor.x));
        finalColor.y = min(1.0f, max(0.0f, finalColor.y));
        finalColor.z = min(1.0f, max(0.0f, finalColor.z));
    }
    
    // Write final result
    outputBuffer[index] = finalColor.x;
    outputBuffer[index+1] = finalColor.y;
    outputBuffer[index+2] = finalColor.z;
    outputBuffer[index+3] = finalColor.w;
}
```

## 6. Technical Considerations

### 6.1 API Design and Naming

Following analysis of legacy code, the framework emphasizes clear, descriptive naming:

**Legacy Pattern**:
```cpp
void processImages(void* p_Args);  // Generic, unclear what p_Args contains
```

**Framework Pattern**:
```cpp
void processFrameWithContext(FrameRenderingContext& context);  // Clear purpose and content
```

**Key Principles**:
- Replace generic names (`p_Args`, `Args`) with descriptive ones (`RenderContext`, `FrameParameters`)
- Use meaningful parameter names that indicate their purpose
- APIs should be self-documenting through clear naming

### 6.2 Standardized Kernel Entry Point

All kernel files must provide a function named `process` as the entry point. This standardization:

1. **Simplifies Framework Implementation**: No need to parse function names from XML
2. **Provides Clear Convention**: Kernel authors always know the expected entry point
3. **Eliminates Configuration**: No function attribute needed in XML
4. **Enables Automatic Parameter Passing**: Framework knows which function to call with dynamic parameters

Kernel authors can still define helper functions and utilities, as long as the standard entry point is provided.

### 6.3 Automatic Parameter and Buffer Availability

The framework provides automatic access to:

1. **All XML Parameters**: Every parameter defined in XML is automatically passed to kernels
2. **All Input Buffers**: Every input source with its border mode is available
3. **Previous Step Outputs**: In multi-kernel effects, all previous outputs are accessible by step name
4. **Execution Context**: Current execution number and total executions for iterative kernels

Benefits:
- **No Manual Mapping**: No need to explicitly map XML parameters to kernel arguments
- **Future-Proof**: Adding new parameters doesn't require signature changes
- **Consistent Interface**: Same pattern across all kernels and platforms

### 6.4 Performance Considerations

1. **Memory Management**: 
   - Efficient buffer allocation and reuse across kernel executions
   - Automatic cleanup of intermediate buffers between pipeline steps
   - Smart caching for frequently accessed parameters

2. **Parallelization**: 
   - GPU acceleration across CUDA, OpenCL, and Metal platforms
   - Optimal thread block sizing based on kernel complexity
   - Automatic workgroup optimization per platform

3. **Resource Lifecycle**: 
   - Proper cleanup of allocated GPU resources
   - Reference counting for shared buffers in multi-kernel pipelines
   - Automatic memory pool management for frequent allocations

4. **Intermediate Storage**: 
   - Minimize copying between pipeline steps
   - In-place processing where possible
   - Efficient buffer swapping for iterative kernels

5. **Parameter Optimization**:
   - Constant buffer usage for frequently accessed parameters
   - Automatic parameter change detection to avoid unnecessary GPU updates
   - Batch parameter updates to reduce GPU state changes

### 6.5 OFX Compatibility

1. **Host Compliance**: 
   - Follow OFX API specifications exactly
   - Support all standard OFX parameter types and behaviors
   - Proper implementation of OFX lifecycle management

2. **Parameter Types**: 
   - Map XML parameter types to appropriate OFX parameter types
   - Support parameter animation and keyframe interpolation
   - Handle parameter groups and UI organization

3. **Input Support**: 
   - Handle various pixel formats (8-bit, 16-bit, float) automatically
   - Support different color spaces and bit depths
   - Proper handling of optional inputs and clips

4. **Threading Model**: 
   - Support host-provided threading models
   - Thread-safe parameter access during rendering
   - Proper synchronization for multi-kernel effects

5. **Resource Management**: 
   - Proper allocation and cleanup following OFX guidelines
   - Host memory vs plugin memory management
   - Error handling and graceful fallbacks

### 6.6 Platform Support

1. **CUDA Support**: 
   - NVIDIA GPUs with compute capability 3.0+
   - Automatic device capability detection
   - Optimized memory access patterns for CUDA architecture

2. **OpenCL Support**: 
   - Cross-platform GPU acceleration (NVIDIA, AMD, Intel)
   - Automatic platform and device selection
   - Support for different OpenCL versions and extensions

3. **Metal Support**: 
   - Apple Silicon and Intel Macs with Metal-capable GPUs
   - iOS and tvOS support for mobile applications
   - Optimized for Apple's unified memory architecture

4. **CPU Fallback**: 
   - Software implementation when GPU unavailable
   - Multi-threaded CPU processing using available cores
   - Automatic fallback detection and switching

5. **Platform Detection**:
   - Automatic detection of available GPU platforms
   - Performance-based platform selection when multiple options available
   - User override capability for platform preferences

### 6.7 Border Handling

Border handling is critical for effects that sample outside image bounds. The framework provides comprehensive border mode support:

#### Per-Source Border Modes

Each input source can specify its own border handling:

1. **Clamp** (`border_mode="clamp"`): 
   - Repeat edge pixels infinitely
   - Best for effects that should respect image boundaries
   - Default for most image processing operations

2. **Repeat** (`border_mode="repeat"`): 
   - Wrap around to opposite edge (tile the image)
   - Useful for seamless texture effects
   - Good for displacement maps and patterns

3. **Mirror** (`border_mode="mirror"`): 
   - Reflect pixels at the boundary
   - Maintains continuity across boundaries
   - Ideal for blur effects that need smooth transitions

4. **Black** (`border_mode="black"`): 
   - Treat outside pixels as transparent/black
   - Useful for alpha-aware effects
   - Default for optional inputs like masks

#### Implementation Details

```cpp
// Border mode constants passed to kernels
#define BORDER_MODE_CLAMP  0
#define BORDER_MODE_REPEAT 1
#define BORDER_MODE_MIRROR 2
#define BORDER_MODE_BLACK  3

// Each input buffer includes its border mode
__global__ void process(
    // ...
    float* sourceBuffer, int sourceBorderMode,    // Individual border mode per input
    float* maskBuffer, int maskBorderMode,        // Each input has its own mode
    // ...
);
```

#### Border Handling Helpers

The framework provides consistent border handling across platforms:

```cpp
// CUDA helper function example
__device__ float4 sampleWithBorderMode(
    float* buffer, int width, int height,
    float x, float y, int borderMode
) {
    int ix = (int)x;
    int iy = (int)y;
    
    switch(borderMode) {
        case BORDER_MODE_CLAMP:
            ix = max(0, min(width-1, ix));
            iy = max(0, min(height-1, iy));
            break;
        case BORDER_MODE_REPEAT:
            ix = ((ix % width) + width) % width;
            iy = ((iy % height) + height) % height;
            break;
        case BORDER_MODE_MIRROR:
            if (ix < 0) ix = -ix;
            if (ix >= width) ix = 2*width - ix - 1;
            if (iy < 0) iy = -iy;
            if (iy >= height) iy = 2*height - iy - 1;
            break;
        case BORDER_MODE_BLACK:
            if (ix < 0 || ix >= width || iy < 0 || iy >= height) {
                return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            break;
    }
    
    int index = (iy * width + ix) * 4;
    return make_float4(buffer[index], buffer[index+1], 
                      buffer[index+2], buffer[index+3]);
}
```

#### Benefits

1. **Consistency**: All GPU platforms implement border handling identically
2. **Flexibility**: Each input can have its own border behavior
3. **Performance**: Border handling optimized for each GPU architecture
4. **Ease of Use**: Helper functions simplify kernel development

## 7. Implementation Priorities Based on Legacy Analysis

### 7.1 High Priority Fixes

1. **Replace Fixed Parameter Signatures**: 
   - Implement dynamic parameter system using maps instead of fixed function signatures
   - Allow unlimited parameters with any names defined in XML
   - Automatic parameter validation and type conversion

2. **Eliminate Hard-coded Pixel Formats**: 
   - Add automatic format detection from image metadata
   - Support multiple pixel formats (8-bit, 16-bit, float) dynamically
   - Remove assumptions about RGBA float format

3. **Unify GPU Platform Handling**: 
   - Create single kernel interface that works across CUDA/OpenCL/Metal
   - Eliminate platform-specific code duplication
   - Automatic platform selection and fallback

4. **Improve API Naming**: 
   - Replace generic names (`p_Args`) with descriptive ones (`RenderContext`)
   - Use self-documenting parameter and function names
   - Clear indication of API purpose and contents

### 7.2 Medium Priority Improvements

1. **Automate Memory Management**: 
   - Reduce boilerplate buffer allocation code
   - Automatic cleanup and resource management
   - Smart caching for frequently accessed data

2. **Enhanced Error Handling**: 
   - Better debugging for dynamic parameter systems
   - Clear error messages for XML validation failures
   - Graceful fallbacks for unsupported features

3. **Performance Optimization**: 
   - Maintain efficiency with flexible parameter systems
   - Optimize parameter passing for GPU kernels
   - Minimize overhead of dynamic dispatch

### 7.3 Future Enhancements

1. **Advanced Parameter Types**: 
   - Support for curves, gradients, and custom UI elements
   - Resolution-dependent parameters that scale with image size
   - Complex parameter relationships and constraints

2. **Multi-kernel Optimization**: 
   - Version 2 pipeline features with automatic buffer management
   - Optimal scheduling of kernel execution
   - Inter-kernel data flow optimization

3. **Real-time Parameter Updates**: 
   - Live preview during parameter adjustment
   - Incremental processing for parameter changes
   - Efficient parameter change detection

## 8. Developer Experience Improvements

### 8.1 Simplified Workflow Comparison

**Legacy OFX Development Process**:
1. Create OFX plugin class with fixed parameter set
2. Manually implement parameter fetching and UI creation in C++
3. Write separate GPU kernel functions for each platform (CUDA/OpenCL/Metal)
4. Handle memory management and format conversion manually
5. Implement separate render paths for each GPU platform
6. Debug platform-specific issues separately
7. Modify C++ code for every parameter change

**Framework Development Process**:
1. Write XML file defining parameters and inputs (any names/types)
2. Write single kernel file with standard `process` entry point
3. Framework automatically handles all OFX infrastructure
4. Automatic parameter passing and memory management
5. Unified debugging across platforms
6. Add parameters by editing XML only

### 8.2 Key Principles from Legacy Analysis

1. **Eliminate Boilerplate**: 
   - Framework handles OFX infrastructure automatically
   - No manual parameter creation or UI generation
   - Automatic memory management and cleanup

2. **Clear Intent**: 
   - Descriptive naming replaces generic technical terms
   - Self-documenting APIs and parameter names
   - Obvious purpose for each system component

3. **Maximum Flexibility**: 
   - Support unlimited parameters/inputs vs fixed assumptions
   - Any parameter names and types from XML
   - Dynamic behavior based on XML definitions

4. **Platform Agnostic**: 
   - Write once, run on all supported GPU platforms
   - Unified development and debugging experience
   - Automatic platform selection and optimization

5. **Type Safety**: 
   - Automatic parameter validation from XML definitions
   - Compile-time checking where possible
   - Clear error messages for mismatched types

### 8.3 Development Tools and Support

1. **XML Validation**: 
   - Schema validation for effect definitions
   - Real-time XML editing with error checking
   - Templates for common effect patterns

2. **Kernel Development**: 
   - Code completion for parameter names from XML
   - Cross-platform kernel testing tools
   - Performance profiling across GPU platforms

3. **Debugging Support**: 
   - Unified debugging interface for all platforms
   - Parameter inspection during kernel execution
   - Visual debugging of intermediate buffers

## 9. Migration Path from Legacy Code

### 9.1 Automated Migration Tools

For existing OFX plugins, the framework can provide migration assistance:

1. **Parameter Extraction**: 
   - Analyze existing C++ code to identify parameters
   - Generate XML parameter definitions automatically
   - Preserve existing parameter names and ranges

2. **Kernel Conversion**: 
   - Convert existing GPU kernels to use standard entry point
   - Update parameter access to use framework helpers
   - Maintain existing algorithm logic

3. **Testing Framework**: 
   - Validate that migrated effects produce identical results
   - Performance comparison between old and new implementations
   - Automated regression testing

### 9.2 Hybrid Development

During transition, effects can use:

1. **Gradual Migration**: 
   - Keep existing C++ infrastructure while adding XML definitions
   - Migrate parameters one at a time
   - Test each migration step independently

2. **Coexistence**: 
   - Framework effects alongside traditional plugins
   - Shared utility functions between old and new systems
   - Common debugging and profiling tools

## 10. Conclusion

This enhanced XML-based OFX framework addresses all major limitations identified in legacy OFX plugin development while maintaining backward compatibility and providing a clear migration path. By combining the comprehensive technical specifications from the original document with the practical insights gained from analyzing real-world OFX plugins, the framework offers:

### Key Innovations

1. **Dynamic Parameter System**: Eliminates fixed parameter signatures, allowing unlimited parameters with any names defined in XML
2. **Automatic Format Detection**: Removes hard-coded pixel format assumptions
3. **Unified GPU Interface**: Single kernel development path across CUDA, OpenCL, and Metal
4. **Descriptive APIs**: Self-documenting interfaces that clearly indicate purpose
5. **Comprehensive Border Handling**: Per-source border modes with consistent cross-platform implementation
6. **Multi-kernel Pipeline Support**: Sequential processing with automatic buffer management
7. **Forward-Compatible Architecture**: Version 1 foundation that seamlessly extends to Version 2

### Benefits for Developers

- **Simplified Development**: Focus on algorithms rather than OFX infrastructure
- **Reduced Complexity**: Single kernel file instead of platform-specific implementations
- **Enhanced Flexibility**: Any number of parameters and inputs with arbitrary names
- **Better Maintainability**: Descriptive naming and clear API design
- **Performance Optimization**: Framework handles GPU-specific optimizations automatically

### Benefits for Artists

- **Faster Iteration**: Modify effects by editing XML and kernel files only
- **No C++ Knowledge Required**: Focus on creative algorithms and parameters
- **Consistent UI**: Automatic UI generation from XML definitions
- **Better Organization**: Logical parameter grouping and clear labeling

The framework transforms OFX plugin development from a complex, platform-specific C++ programming task to a straightforward XML configuration and kernel development process. This enables image processing artists to focus on creating innovative visual effects rather than wrestling with infrastructure concerns, while maintaining the performance and flexibility that professional applications demand.

By learning from the limitations of legacy code and implementing proven solutions, this framework establishes a new standard for GPU-accelerated image processing plugin development that will serve the industry's evolving needs for years to come.