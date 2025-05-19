# XML-Based OFX Image Processing Framework Specification

> **IMPORTANT NOTE**: This framework is designed so that image processing artists only need to modify two types of files:
> 1. XML effect definition files - defining parameters, inputs, and processing steps
> 2. Kernel code files (.cu/.cl/.metal) - containing the image processing algorithms
>
> No C++ knowledge or modification of the framework code is required to create new effects.

## 1. Overview

This document describes the architecture for an XML-based OpenFX (OFX) image processing framework. The system simplifies the creation of new OFX plugins by allowing image processing artists to focus on writing GPU kernels and parameter definitions without needing to understand the OFX C++ infrastructure.

The framework will be implemented in two phases:
- **Version 1**: A single-kernel system supporting GPU-accelerated image effects with XML-defined parameters and clips
- **Version 2**: A multi-kernel system supporting sequential kernels, iterative processing, and complex effects

## 2. System Components

### 2.1 Core Components

The framework consists of these primary components:

1. **XML Effect Definitions**: XML files that describe effect parameters, inputs, and processing logic
2. **XML Parser**: System to read and validate XML effect definitions
3. **Parameter Manager**: Handles parameter creation, UI, and value passing to kernels
4. **Input Manager**: Manages input/output connections between the plugin and host
5. **Kernel Manager**: Loads and executes GPU kernels (CUDA, OpenCL, Metal)
6. **Buffer Manager**: Manages image buffers for processing
7. **Generic Effect**: Base class for dynamically-generated OFX plugins
8. **Factory System**: Creates OFX plugins from XML definitions

### 2.2 XML Schema

The XML schema defines the structure of effect definitions. Here's the schema that accommodates both Version 1 and Version 2 features:

```xml
<effect name="EffectName" category="Category">
  <description>Effect description text</description>
  
  <!-- Define input sources -->
  <inputs>
    <source name="source" label="Main Image" border_mode="clamp" />
    <source name="matte" label="Matte Input" optional="true" border_mode="black" />
    <!-- Additional source inputs can be defined here -->
  </inputs>
  
  <!-- Parameters define UI controls and processing values -->
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
    
    <!-- Additional parameters... -->
  </parameters>
  
  <!-- UI organization -->
  <ui>
    <page name="Main">
      <column name="Basic">
        <parameter>radius</parameter>
        <parameter>quality</parameter>
      </column>
      <column name="Advanced">
        <parameter>alpha_fade</parameter>
      </column>
    </page>
    <page name="Color">
      <!-- More parameters -->
    </page>
  </ui>
  
  <!-- Identity conditions define when the effect is a pass-through -->
  <identity_conditions>
    <condition>
      <parameter name="radius" operator="lessEqual" value="0.0" />
    </condition>
    <!-- Additional conditions... -->
  </identity_conditions>
  
  <!-- Version 1: Single kernel processing -->
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

### 2.3 Class Architecture

The framework's class architecture is designed to be extensible, supporting both Version 1 and Version 2 features:

```
- XMLEffectDefinition
  - Parses and stores effect metadata, parameters, inputs, kernels
  - Supports future expansion to multi-kernel effects

- XMLParameterManager
  - Maps XML parameter definitions to OFX parameters
  - Creates UI controls based on XML definitions

- GenericEffect (inherits from OFX::ImageEffect)
  - Base class for all XML-defined effects
  - Manages parameter storage and access
  - Handles rendering and processing
  - Extensible for multi-kernel processing

- BufferManager
  - Manages image buffers for processing
  - Handles buffer allocation, access, and cleanup
  - Supports future expansion to multi-kernel with intermediate buffers

- KernelManager
  - Loads and executes GPU kernels
  - Supports CUDA, OpenCL, and Metal
  - Handles parameter passing to kernel functions
  - Uses standardized entry point (process) for all kernels

- XMLEffectFactory (inherits from OFX::PluginFactoryHelper)
  - Creates plugin instances from XML definitions
  - Handles OFX lifecycle and metadata
```

## 3. Two-Phase Implementation Plan

### 3.1 Version 1 Features

Version 1 implements a single-kernel image processing framework with:

1. **XML-Defined Parameters**: Effect parameters defined in XML
2. **XML-Defined Inputs**: Input sources defined in XML
3. **GPU Acceleration**: Support for CUDA, OpenCL, and Metal kernels
4. **UI Organization**: Parameters organized into pages and columns
5. **Dynamic Parameter Handling**: All parameters available to kernels
6. **Standard Kernel Interface**: One function per file with standard entry point
7. **Forward-Compatible Architecture**: Designed to support Version 2 features

### 3.2 Version 2 Features

Version 2 extends the framework with:

1. **Multi-Kernel Processing**: Sequential execution of multiple kernels
2. **Multi-Execution Kernels**: Support for kernels that execute multiple times
3. **Intermediate Buffers**: Management of buffers between kernels
4. **Automatic Data Flow**: Previous outputs automatically available to subsequent kernels

Enhanced features:
- Access to results from previous kernels
- Access to previous iterations within multi-execution kernels
- Buffer swapping for efficient iterative processing
- Configurable execution counts for each kernel

## 4. Kernel Interface

### 4.1 Standard Kernel Entry Point

All kernel files must provide a standardized entry point function named `process`. This function will receive all parameters defined in the XML, along with standard arguments for image dimensions and buffers.

```cpp
// CUDA kernel example with standard entry point
__global__ void process(
    int width, int height,         // Image dimensions
    int executionNumber,           // Current execution number (for multi-execution)
    int totalExecutions,           // Total executions for this kernel
    float* sourceBuffer,           // Main source image
    float* outputBuffer,           // Output image
    // All effect parameters defined in XML follow automatically
    float radius, 
    int quality,
    // etc.
);

// OpenCL kernel example
__kernel void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    __read_only image2d_t sourceBuffer,
    __write_only image2d_t outputBuffer,
    float radius,
    int quality,
    // etc.
);

// Metal kernel example
kernel void process(
    uint2 gid [[thread_position_in_grid]],
    constant int& width [[buffer(0)]],
    constant int& height [[buffer(1)]],
    constant int& executionNumber [[buffer(2)]],
    constant int& totalExecutions [[buffer(3)]],
    texture2d<float, access::read> sourceBuffer [[texture(0)]],
    texture2d<float, access::write> outputBuffer [[texture(1)]],
    constant float& radius [[buffer(4)]],
    constant int& quality [[buffer(5)]],
    // etc.
);
```

### 4.2 Version 2 Multi-Kernel Interface

In Version 2, kernels will receive additional inputs for previous processing steps. All buffers from previous steps are automatically available, named according to the step that produced them.

```cpp
// CUDA kernel example for multi-kernel effect (Composite step)
__global__ void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    float* sourceBuffer,           // Original source image
    float* EdgeDetectBuffer,       // Output from EdgeDetect step
    float* BlurBuffer,             // Output from Blur step
    float* outputBuffer,           // Output for this step
    float blendAmount,             // Parameter from XML
    // etc.
);
```

## 5. User Workflow

### 5.1 Artist Workflow (Version 1)

1. Create an XML file defining the effect parameters and inputs
2. Write GPU kernels for CUDA, OpenCL, and/or Metal with the standard entry point
3. Place XML and kernel files in the plugin directory
4. Framework automatically loads and registers the effect

### 5.2 Artist Workflow (Version 2)

1. Create an XML file defining multiple processing steps
2. Write GPU kernels for each processing step with the standard entry point
3. Place XML and kernel files in the plugin directory
4. Framework handles multi-kernel execution and buffer management

### 5.3 Example: Simple Blur Effect (Version 1)

**GaussianBlur.xml**:
```xml
<effect name="GaussianBlur" category="Filter">
  <description>Apply Gaussian blur with optional mask control</description>
  
  <inputs>
    <source name="source" label="Input Image" border_mode="clamp" />
    <source name="matte" label="Mask" optional="true" border_mode="black" />
  </inputs>
  
  <parameters>
    <parameter name="radius" type="double" default="5.0" min="0.0" max="100.0" 
               displayMin="0.0" displayMax="50.0" label="Radius" hint="Blur radius in pixels" />
    <parameter name="quality" type="int" default="8" min="1" max="32" 
               displayMin="1" displayMax="16" label="Quality" hint="Number of samples for the blur" />
  </parameters>
  
  <ui>
    <page name="Main">
      <column name="Parameters">
        <parameter>radius</parameter>
        <parameter>quality</parameter>
      </column>
    </page>
  </ui>
  
  <kernels>
    <cuda file="GaussianBlur.cu" executions="1" />
    <opencl file="GaussianBlur.cl" executions="1" />
    <metal file="GaussianBlur.metal" executions="1" />
  </kernels>
  
  <identity_conditions>
    <condition>
      <parameter name="radius" operator="lessEqual" value="0.0" />
    </condition>
  </identity_conditions>
</effect>
```

**GaussianBlur.cu**:
```cpp
**GaussianBlur.cu**:
```cpp
#include <cuda_runtime.h>

// Helper device function
__device__ float gaussian(float x, float sigma) {
    return exp(-(x*x) / (2.0f * sigma * sigma));
}

// Helper device function for border handling
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

// Standard entry point for all kernels
__global__ void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    // Source buffers with their border modes
    float* sourceBuffer, int sourceBorderMode,
    float* matteBuffer, int matteBorderMode,      // Optional matte buffer
    float* outputBuffer,
    // Parameters from XML
    float radius,
    int quality
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Use appropriate border modes for each source
    float4 sourceColor = sampleWithBorderMode(sourceBuffer, width, height, x-radius, y, sourceBorderMode);
    float4 matteColor = matteBuffer ? 
                       sampleWithBorderMode(matteBuffer, width, height, x, y, matteBorderMode) : 
                       make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    
    // Blur implementation using border-aware sampling
    // ...
}
```
```

### 5.4 Example: Complex Effect (Version 2)

**EdgeEnhance.xml**:
```xml
<effect name="EdgeEnhance" category="Filter">
  <description>Advanced edge enhancement with blur and sharpening</description>
  
  <inputs>
    <source name="source" label="Input Image" />
  </inputs>
  
  <parameters>
    <parameter name="edgeThreshold" type="double" default="0.2" min="0.0" max="1.0">
      <label>Edge Threshold</label>
      <hint>Threshold for edge detection</hint>
    </parameter>
    <parameter name="blurRadius" type="double" default="3.0" min="0.0" max="10.0">
      <label>Blur Radius</label>
      <hint>Radius for edge smoothing</hint>
    </parameter>
    <parameter name="sharpAmount" type="double" default="2.0" min="0.0" max="10.0">
      <label>Sharpening Amount</label>
      <hint>Strength of edge enhancement</hint>
    </parameter>
  </parameters>
  
  <ui>
    <page name="Main">
      <column name="Edges">
        <parameter>edgeThreshold</parameter>
      </column>
      <column name="Enhancement">
        <parameter>blurRadius</parameter>
        <parameter>sharpAmount</parameter>
      </column>
    </page>
  </ui>
  
  <pipeline>
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
  </pipeline>
</effect>
```

**Composite.cu** (third step in the pipeline):
```cpp
#include <cuda_runtime.h>

// Standard entry point - all buffers automatically available
__global__ void process(
    int width, int height,
    int executionNumber, int totalExecutions,
    float* sourceBuffer,         // Original image
    float* EdgeDetectBuffer,     // Output from EdgeDetect step
    float* BlurBuffer,           // Output from Blur step
    float* outputBuffer,         // Output for this step
    float edgeThreshold,         // From XML
    float blurRadius,            // From XML
    float sharpAmount            // From XML
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Composite processing code here
    // ...
}
```

## 6. Technical Considerations

### 6.1 Standardized Kernel Entry Point

All kernel files must provide a function named `process` as the entry point. This standardization:

1. Simplifies the framework implementation
2. Makes it clear which function to call
3. Eliminates the need for a function attribute in the XML
4. Provides a consistent pattern for kernel authors

Kernel authors are still free to define helper functions and include utility files, as long as the standard entry point is provided.

### 6.2 Automatic Parameter and Buffer Availability

1. All parameters defined in the XML are automatically passed to all kernels
2. In multi-kernel pipelines, all previous outputs are automatically available
3. Kernels only need to declare the parameters and buffers they intend to use
4. No explicit mapping is required in the XML

### 6.3 Performance Considerations

1. **Memory Management**: Efficient buffer allocation and reuse
2. **Parallelization**: GPU acceleration across platforms
3. **Resource Lifecycle**: Proper cleanup of allocated resources
4. **Intermediate Storage**: Minimize copying between buffers
5. **Error Handling**: Graceful failure and fallback mechanisms

### 6.4 OFX Compatibility

1. **Host Compliance**: Follow OFX API specifications
2. **Parameter Types**: Support standard OFX parameter types
3. **Input Support**: Handle various pixel formats and bit depths
4. **Threading Model**: Support host-provided threading
5. **Resource Management**: Proper resource allocation and cleanup

### 6.5 Platform Support

1. **CUDA Support**: NVIDIA GPUs
2. **OpenCL Support**: Cross-platform GPU acceleration
3. **Metal Support**: Apple platforms
4. **CPU Fallback**: Software implementation when GPU unavailable

### 6.6 Border Handling

Border handling is critical for image processing operations, especially for effects that sample outside the image bounds (blur, distortion, etc.). The framework provides:

1. **Per-Source Border Modes**: Each source input can specify its own border handling mode:
   - `clamp` (repeat edge pixels)
   - `repeat` (wrap around to opposite edge)
   - `mirror` (reflect pixels at the boundary)
   - `black` (treat outside pixels as transparent/black)

2. **Default Behavior**: Each source can specify its default border mode in the XML

3. **Implementation Consistency**: All GPU implementations (CUDA, OpenCL, Metal) implement border handling consistently

4. **Helper Functions**: The framework provides helper functions for border handling to simplify kernel development

The border mode for each source is passed to the kernel along with the source data, allowing consistent sampling behavior regardless of GPU platform.

## 7. Conclusion

This XML-based OFX framework provides a flexible and extensible system for creating image processing effects. By separating the effect definition (parameters, inputs, processing flow) from the implementation details (OFX infrastructure), the framework allows image processing artists to focus on what they do best – creating visual effects – without getting bogged down in OFX C++ infrastructure.

The two-phase implementation plan ensures that a solid foundation is established in Version 1, which can then be extended to support more complex effects in Version 2, all while maintaining backward compatibility and minimizing refactoring.

The key innovations in this framework are:
1. XML-driven effect definitions
2. Standardized kernel entry points
3. Automatic parameter and buffer availability
4. Multi-kernel pipeline support
5. Integration with industry-standard GPU platforms
