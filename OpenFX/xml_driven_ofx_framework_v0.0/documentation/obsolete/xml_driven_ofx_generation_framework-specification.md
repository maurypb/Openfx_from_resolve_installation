# XML-Based OFX Image Processing Framework Specification

## 1. Overview

This document describes the architecture and roadmap for an XML-based OpenFX (OFX) image processing framework. The system aims to simplify the creation of new OFX plugins by allowing image processing artists to focus on writing GPU kernels and parameter definitions without needing to understand the OFX C++ infrastructure.

The framework will be implemented in two phases:
- **Version 1**: A single-kernel system supporting GPU-accelerated image effects with XML-defined parameters and clips
- **Version 2**: A multi-kernel system supporting sequential kernels, iterative processing, and complex effects

## 2. System Components

### 2.1 Core Components

The framework consists of these primary components:

1. **XML Effect Definitions**: XML files that describe effect parameters, inputs, kernels, and processing logic
2. **XML Parser**: System to read and validate XML effect definitions
3. **Parameter Manager**: Handles parameter creation, UI, and value passing to kernels
4. **Input Manager**: Manages input/output connections between the plugin and host
5. **Kernel Manager**: Loads and executes GPU kernels (CUDA, OpenCL, Metal)
6. **Buffer Manager**: Manages image buffers for processing
7. **Generic Effect**: Base class for dynamically-generated OFX plugins
8. **Factory System**: Creates OFX plugins from XML definitions

### 2.2 XML Schema

The XML schema defines the structure of effect definitions. Here's a comprehensive schema that accommodates both Version 1 and Version 2 features (with Version 2 components commented):

```xml
<effect name="EffectName" category="Category">
  <description>Effect description text</description>
  
  <!-- Define input sources -->
  <inputs>
    <source name="source" label="Main Image" />
    <source name="matte" label="Matte Input" optional="true" />
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
  
  <!-- Kernels define processing steps (Version 1: single kernel only) -->
  <kernel name="MainProcess" file="EffectKernel.cu" label="Main Process" executions="1">
    <!-- Kernel implementation details -->
  </kernel>
  
  <!-- Multiple kernels (Version 2) -->
  <!-- <kernel name="EdgeDetect" file="EdgeDetect.cu" label="Edge Detection" executions="1">
  </kernel>
  
  <kernel name="Blur" file="GaussianBlur.cu" label="Blur Pass" executions="3">
  </kernel>
  
  <kernel name="Composite" file="Composite.cu" label="Final Composite" executions="1">
  </kernel> -->
  
  <!-- Identity conditions define when the effect is a pass-through -->
  <identity_conditions>
    <condition>
      <parameter name="radius" operator="lessEqual" value="0.0" />
    </condition>
    <!-- Additional conditions... -->
  </identity_conditions>
  
  <!-- Kernel definitions for different GPU platforms -->
  <implementations>
    <cuda function="EffectKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" /> <!-- For multi-execution kernels -->
        <param name="totalExecutions" type="int" /> <!-- For multi-execution kernels -->
        <!-- Other parameters are automatically available -->
        <param name="input" type="image" role="source" />
        <param name="mask" type="image" role="matte" optional="true" />
        <param name="output" type="image" role="output" />
      </params>
    </cuda>
    
    <opencl function="EffectKernel">
      <!-- Similar structure to CUDA implementation -->
    </opencl>
    
    <metal function="EffectKernel">
      <!-- Similar structure to CUDA implementation -->
    </metal>
  </implementations>
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
  - Handles kernel parameter mapping
  - Extensible for multi-execution kernels

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
6. **Forward-Compatible Architecture**: Designed to support Version 2 features

Limitations:
- Only single-kernel effects are supported (with potential for multiple executions)
- No intermediate buffers or multi-kernel processing

### 3.2 Version 2 Features

Version 2 extends the framework with:

1. **Multi-Kernel Processing**: Sequential execution of multiple kernels
2. **Multi-Execution Kernels**: Support for kernels that execute multiple times
3. **Intermediate Buffers**: Management of buffers between kernels
4. **Complex Execution Flows**: Support for iterative and convergent algorithms

Enhanced features:
- Access to results from previous kernels
- Access to previous iterations within multi-execution kernels
- Buffer swapping for efficient iterative processing
- Configurable execution counts for each kernel

### 3.3 Compatibility Strategy

To ensure smooth transition between versions:

1. **Forward-Compatible XML**: Version 1 XML schema includes placeholders for Version 2 features
2. **Extensible Architecture**: Base classes designed with extension points for Version 2
3. **Minimal Refactoring**: Version 2 builds on Version 1 rather than replacing it
4. **Clear Documentation**: Version 2 features documented from the start

## 4. Kernel Interface

### 4.1 Version 1 Kernel Interface

GPU kernels in Version 1 follow this general signature:

```cpp
// CUDA kernel example
__global__ void EffectKernel(
    int width, int height,
    int executionNumber, int totalExecutions, // For multi-execution support
    float parameterA, int parameterB, // Effect parameters (all available)
    cudaTextureObject_t sourceTexture, // Main source image
    cudaTextureObject_t matteTexture,  // Optional matte input
    float* outputBuffer               // Output image
);

// Function to invoke the kernel
void RunCudaKernel(
    void* stream,
    int width, int height,
    int executionNumber, int totalExecutions,
    float parameterA, int parameterB, // All effect parameters
    const float* sourceBuffer,
    const float* matteBuffer,
    float* outputBuffer
);
```

### 4.2 Version 2 Kernel Interface

Version 2 extends the kernel interface to support multi-kernel processing:

```cpp
// CUDA kernel example for multi-kernel effect
__global__ void EdgeDetectKernel(
    int width, int height,
    int executionNumber, int totalExecutions,
    float threshold, // All effect parameters available
    cudaTextureObject_t sourceTexture,   // Original source image
    float* outputBuffer                  // Output for this kernel
);

__global__ void BlurKernel(
    int width, int height,
    int executionNumber, int totalExecutions,
    float radius, int quality, // All effect parameters available
    cudaTextureObject_t sourceTexture,   // Original source image
    cudaTextureObject_t edgeTexture,     // Output from EdgeDetect kernel
    cudaTextureObject_t previousIterationTexture, // Previous iteration result
    float* outputBuffer                  // Output for this kernel
);

// Function to invoke the kernels
void RunEdgeDetectKernel(
    void* stream,
    int width, int height,
    int executionNumber, int totalExecutions,
    float threshold,
    const float* sourceBuffer,
    float* outputBuffer
);

void RunBlurKernel(
    void* stream,
    int width, int height,
    int executionNumber, int totalExecutions,
    float radius, int quality,
    const float* sourceBuffer,
    const float* edgeBuffer,
    const float* previousIterationBuffer,
    float* outputBuffer
);
```

## 5. User Workflow

### 5.1 Artist Workflow (Version 1)

1. Create an XML file defining the effect parameters and inputs
2. Write GPU kernels for CUDA, OpenCL, and/or Metal
3. Place XML and kernel files in the plugin directory
4. Framework automatically loads and registers the effect

### 5.2 Artist Workflow (Version 2)

1. Create an XML file defining multiple kernels, parameters, and execution counts
2. Write GPU kernels for each processing step
3. Place XML and kernel files in the plugin directory
4. Framework handles multi-kernel execution and buffer management

### 5.3 Example: Simple Blur Effect (Version 1)

**GaussianBlur.xml**:
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
  
  <kernel name="EdgeDetect" file="EdgeDetect.cu" label="Edge Detection" executions="1">
  </kernel>
  
  <kernel name="Blur" file="GaussianBlur.cu" label="Edge Blur" executions="3">
    <!-- Output is automatically named "Blur" -->
  </kernel>
  
  <kernel name="Enhance" file="EdgeEnhance.cu" label="Edge Enhancement" executions="1">
    <!-- Has automatic access to EdgeDetect and Blur outputs -->
  </kernel>
  
  <implementations>
    <cuda function="EdgeDetectKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" />
        <param name="totalExecutions" type="int" />
        <param name="threshold" type="float" />
        <param name="source" type="image" role="input" />
        <param name="output" type="image" role="output" />
      </params>
    </cuda>
    
    <cuda function="GaussianBlurKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" />
        <param name="totalExecutions" type="int" />
        <param name="radius" type="float" />
        <param name="quality" type="int" />
        <param name="source" type="image" role="input" />
        <param name="EdgeDetect" type="image" role="input" />
        <param name="self_previous" type="image" role="input" optional="true" />
        <param name="output" type="image" role="output" />
      </params>
    </cuda>
    
    <cuda function="EdgeEnhanceKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" />
        <param name="totalExecutions" type="int" />
        <param name="amount" type="float" />
        <param name="source" type="image" role="input" />
        <param name="EdgeDetect" type="image" role="input" />
        <param name="Blur" type="image" role="input" />
        <param name="output" type="image" role="output" />
      </params>
    </cuda>
  </implementations>
</effect>
```

## 6. Technical Considerations

### 6.1 Performance Considerations

1. **Memory Management**: Efficient buffer allocation and reuse
2. **Parallelization**: GPU acceleration across platforms
3. **Resource Lifecycle**: Proper cleanup of allocated resources
4. **Intermediate Storage**: Minimize copying between buffers
5. **Error Handling**: Graceful failure and fallback mechanisms

### 6.2 OFX Compatibility

1. **Host Compliance**: Follow OFX API specifications
2. **Parameter Types**: Support standard OFX parameter types
3. **Input Support**: Handle various pixel formats and bit depths
4. **Threading Model**: Support host-provided threading
5. **Resource Management**: Proper resource allocation and cleanup

### 6.3 Platform Support

1. **CUDA Support**: NVIDIA GPUs
2. **OpenCL Support**: Cross-platform GPU acceleration
3. **Metal Support**: Apple platforms
4. **CPU Fallback**: Software implementation when GPU unavailable

## 7. Conclusion

This XML-based OFX framework provides a flexible and extensible system for creating image processing effects. By separating the effect definition (parameters, inputs, processing flow) from the implementation details (OFX infrastructure), the framework allows image processing artists to focus on what they do best – creating visual effects – without getting bogged down in OFX C++ infrastructure.

The two-phase implementation plan ensures that a solid foundation is established in Version 1, which can then be extended to support more complex effects in Version 2, all while maintaining backward compatibility and minimizing refactoring.
    <parameter name="blurRadius" type="double" default="3.0" min="0.0" max="10.0">
      <label>Blur Radius</label>
      <hint>Radius for edge smoothing</hint>
    </parameter>
    <parameter name="sharpAmount" type="double" default="2.0" min="0.0" max="10.0">
      <label>Sharpening Amount</label>
      <hint>Strength of edge enhancement</hint>
    </parameter>
  </parameters>
  
  <passes>
    <pass name="EdgeDetect" kernel="EdgeDetect.cu" executions="1">
      <inputs>
        <input name="source" source="source" />
      </inputs>
    </pass>
    
    <pass name="Blur" kernel="GaussianBlur.cu" executions="3">
      <inputs>
        <input name="source" source="source" />
        <input name="edges" source="EdgeDetect.output" />
        <input name="previous" source="self.previous" />
      </inputs>
      <parameters>
        <parameter name="radius" mapTo="blurRadius" />
      </parameters>
    </pass>
    
    <pass name="Enhance" kernel="EdgeEnhance.cu" executions="1">
      <inputs>
        <input name="source" source="source" />
        <input name="edges" source="EdgeDetect.output" />
        <input name="blurred" source="Blur.output" />
      </inputs>
      <parameters>
        <parameter name="amount" mapTo="sharpAmount" />
      </parameters>
    </pass>
  </passes>
</effect>
```

## 6. Technical Considerations

### 6.1 Performance Considerations

1. **Memory Management**: Efficient buffer allocation and reuse
2. **Parallelization**: GPU acceleration across platforms
3. **Resource Lifecycle**: Proper cleanup of allocated resources
4. **Intermediate Storage**: Minimize copying between buffers
5. **Error Handling**: Graceful failure and fallback mechanisms

### 6.2 OFX Compatibility

1. **Host Compliance**: Follow OFX API specifications
2. **Parameter Types**: Support standard OFX parameter types
3. **Clip Support**: Handle various pixel formats and bit depths
4. **Threading Model**: Support host-provided threading
5. **Resource Management**: Proper resource allocation and cleanup

### 6.3 Platform Support

1. **CUDA Support**: NVIDIA GPUs
2. **OpenCL Support**: Cross-platform GPU acceleration
3. **Metal Support**: Apple platforms
4. **CPU Fallback**: Software implementation when GPU unavailable

## 7. Conclusion

This XML-based OFX framework provides a flexible and extensible system for creating image processing effects. By separating the effect definition (parameters, clips, processing flow) from the implementation details (OFX infrastructure), the framework allows image processing artists to focus on what they do best – creating visual effects – without getting bogged down in OFX C++ infrastructure.

The two-phase implementation plan ensures that a solid foundation is established in Version 1, which can then be extended to support more complex effects in Version 2, all while maintaining backward compatibility and minimizing refactoring.
