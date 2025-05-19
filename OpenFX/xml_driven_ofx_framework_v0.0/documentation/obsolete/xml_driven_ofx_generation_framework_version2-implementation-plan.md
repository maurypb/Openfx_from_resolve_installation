# Version 2 Implementation Plan: Multi-Kernel OFX Framework

## Overview

This document outlines the implementation plan for Version 2 of the XML-based OFX framework, which extends the single-kernel system to support multi-kernel and multi-execution processing. This plan assumes that Version 1 has been successfully implemented and tested, providing a solid foundation of XML parsing, parameter handling, and basic kernel execution.

The key enhancements in Version 2 are:
1. Supporting multiple sequential kernels
2. Supporting multiple executions of the same kernel
3. Managing intermediate buffers between kernels
4. Providing access to results from previous kernels and iterations

## Prerequisites

- Complete Version 1 implementation with XML parsing and parameter handling
- Working GenericEffect and GenericProcessor base classes
- Basic kernel management for CUDA, OpenCL, and Metal
- Forward-compatible design choices in Version 1

## Implementation Stages

### Stage 1: Enhanced XML Schema for Multi-Kernel Definitions

**Goal**: Extend the XML schema to fully support multi-kernel and multi-execution definitions.

**Steps**:
1. Update XMLEffectDefinition to parse multiple kernels
2. Add support for execution count settings
3. Implement kernel labeling for documentation

**Implementation Example**:
```cpp
// Add to XMLEffectDefinition class
struct KernelDef {
    std::string name;
    std::string file;
    std::string label;
    int executions;
};

std::vector<KernelDef> _kernels;
std::vector<KernelDef> getKernels() const;
```

**XML Example**:
```xml
<kernel name="EdgeDetect" file="EdgeDetect.cu" label="Edge Detection" executions="1">
</kernel>

<kernel name="Blur" file="GaussianBlur.cu" label="Edge Blur" executions="3">
</kernel>

<kernel name="Enhance" file="EdgeEnhance.cu" label="Edge Enhancement" executions="1">
</kernel>
```

**Test Criteria**:
- XML parser correctly loads multi-kernel definitions
- Kernel counts, execution counts, and labels are properly stored
- Backward compatibility with Version 1 XML is maintained

### Stage 2: Buffer Manager Implementation

**Goal**: Create a buffer management system that handles multiple intermediate buffers for multi-kernel processing.

**Steps**:
1. Implement BufferManager class
2. Add support for named input/output/intermediate buffers
3. Implement buffer swapping for multi-execution kernels
4. Add memory optimization for buffer reuse

**Implementation**:
```cpp
class BufferManager {
public:
    BufferManager(int width, int height);
    ~BufferManager();
    
    // Create and get buffers
    float* createBuffer(const std::string& name);
    float* getBuffer(const std::string& name) const;
    bool hasBuffer(const std::string& name) const;
    
    // Buffer operations
    void copyBuffer(const std::string& source, const std::string& destination);
    void swapBuffers(const std::string& buffer1, const std::string& buffer2);
    void clearBuffer(const std::string& name);
    
    // Common buffers
    float* getSourceBuffer(const std::string& sourceName = "source") const;
    float* getOutputBuffer() const { return getBuffer("output"); }
    
    // Special buffer for multi-execution (previous iteration)
    float* getPreviousIterationBuffer() const { return getBuffer("self_previous"); }
    
    // Buffer info
    int getWidth() const { return _width; }
    int getHeight() const { return _height; }
    
private:
    int _width, _height;
    std::map<std::string, float*> _buffers;
    
    void freeBuffer(const std::string& name);
};
```

**Test Criteria**:
- Buffer creation and management works correctly
- Buffer swapping efficiently handles multi-execution kernels
- Memory allocation and cleanup is properly handled
- Buffers are correctly sized for the frame dimensions

### Stage 3: Multi-Kernel Processing in GenericEffect

**Goal**: Extend the GenericEffect class to support sequential multi-kernel processing.

**Steps**:
1. Update process() method to handle multiple kernels
2. Implement buffer routing between kernels
3. Create execution plan based on XML definition

**Implementation**:
```cpp
// Update GenericEffect class
void GenericEffect::process(const OFX::RenderArguments& args) {
    // For single-kernel effects, just use the original method
    if (_xmlDef.getKernels().size() <= 1) {
        processSingleKernel(args);
        return;
    }
    
    // Create buffer manager for the frame
    const OfxRectI& bounds = _dstClip->getPixelData()->getBounds();
    int width = bounds.x2 - bounds.x1;
    int height = bounds.y2 - bounds.y1;
    BufferManager bufferManager(width, height);
    
    // Set up source buffers for each defined input
    for (const auto& source : _xmlDef.getInputs()) {
        std::string clipName = source.name;
        if (_srcClips.find(clipName) != _srcClips.end() && _srcClips[clipName]->isConnected()) {
            OFX::Image* srcImg = _srcClips[clipName]->fetchImage(args.time);
            float* sourceData = static_cast<float*>(srcImg->getPixelData());
            
            bufferManager.createBuffer(clipName);
            float* sourceBuffer = bufferManager.getBuffer(clipName);
            memcpy(sourceBuffer, sourceData, width * height * 4 * sizeof(float));
        }
    }
    
    // Set up output buffer
    OFX::Image* dstImg = _dstClip->fetchImage(args.time);
    bufferManager.createBuffer("output");
    float* outputBuffer = bufferManager.getOutputBuffer();
    
    // Collect parameter values
    std::map<std::string, OFX::ParamValue> paramValues = collectParameterValues(args.time);
    
    // Process each kernel in sequence
    const auto& kernels = _xmlDef.getKernels();
    for (const auto& kernel : kernels) {
        // Create kernel output buffer
        std::string kernelOutputName = kernel.name;
        if (!bufferManager.hasBuffer(kernelOutputName)) {
            bufferManager.createBuffer(kernelOutputName);
        }
        
        // Create temp buffer for multi-execution if needed
        if (kernel.executions > 1) {
            if (!bufferManager.hasBuffer("self_previous")) {
                bufferManager.createBuffer("self_previous");
            }
        }
        
        // Execute the kernel (possibly multiple times)
        for (int execution = 0; execution < kernel.executions; execution++) {
            // Set up buffer routing for this execution
            std::string currentOutput = (execution == kernel.executions - 1) ? 
                                      kernelOutputName : "temp";
            
            std::string previousOutput = (execution == 0) ? 
                                       kernelOutputName : "self_previous";
            
            // Execute the kernel
            executeKernel(
                kernel.name,
                kernel.file,
                execution,
                kernel.executions,
                paramValues,
                previousOutput,
                currentOutput,
                bufferManager,
                args
            );
            
            // Swap buffers for next iteration if needed
            if (execution < kernel.executions - 1) {
                bufferManager.copyBuffer(currentOutput, "self_previous");
            }
        }
    }
    
    // Copy the last kernel result to the output
    const std::string finalKernelName = kernels.back().name;
    float* finalOutput = bufferManager.getBuffer(finalKernelName);
    float* dstData = static_cast<float*>(dstImg->getPixelData());
    
    memcpy(dstData, finalOutput, width * height * 4 * sizeof(float));
}

// Helper method to execute a single kernel
void GenericEffect::executeKernel(
    const std::string& kernelName,
    const std::string& kernelFile,
    int executionNumber,
    int totalExecutions,
    const std::map<std::string, OFX::ParamValue>& params,
    const std::string& previousOutputName,
    const std::string& currentOutputName,
    BufferManager& bufferManager,
    const OFX::RenderArguments& args
) {
    // Create processor
    GenericProcessor processor(*this, _xmlDef, kernelName);
    
    // Set up processor
    processor.setDstImg(_dstClip->fetchImage(args.time));
    processor.setRenderWindow(args.renderWindow);
    processor.setGPURenderArgs(args);
    
    // Set up buffer manager and execution info
    processor.setBufferManager(bufferManager);
    processor.setOutputBuffer(currentOutputName);
    processor.setExecutionInfo(executionNumber, totalExecutions);
    
    // Set parameters and process
    processor.setParamValues(params);
    processor.process();
}
```

**Test Criteria**:
- Multiple kernels execute in the correct sequence
- Buffers are correctly routed between kernels
- Multi-execution kernels work with the right iteration count
- Complex effects with multiple kernels render correctly

### Stage 4: Enhanced GenericProcessor for Multi-Kernel Support

**Goal**: Extend the GenericProcessor to handle multi-kernel and multi-execution processing.

**Steps**:
1. Update processor to work with the BufferManager
2. Add support for execution counting
3. Update GPU kernel calls to support multi-kernel

**Implementation**:
```cpp
// Update GenericProcessor class
class GenericProcessor : public OFX::ImageProcessor {
private:
    GenericEffect& _effect;
    XMLEffectDefinition& _xmlDef;
    std::string _kernelName;
    std::map<std::string, OFX::ParamValue> _paramValues;
    
    // Multi-kernel additions
    BufferManager* _bufferManager;
    std::string _outputBuffer;
    
    int _executionNumber;
    int _totalExecutions;
    
public:
    // Constructor for specific kernel
    GenericProcessor(GenericEffect& effect, XMLEffectDefinition& xmlDef, const std::string& kernelName)
        : OFX::ImageProcessor(effect), _effect(effect), _xmlDef(xmlDef), 
          _kernelName(kernelName), _bufferManager(nullptr), 
          _executionNumber(0), _totalExecutions(1) {}
    
    // Buffer manager
    void setBufferManager(BufferManager& bufferManager) { 
        _bufferManager = &bufferManager; 
    }
    
    // Output buffer
    void setOutputBuffer(const std::string& name) { 
        _outputBuffer = name; 
    }
    
    // Execution info
    void setExecutionInfo(int executionNumber, int totalExecutions) {
        _executionNumber = executionNumber;
        _totalExecutions = totalExecutions;
    }
    
    // Parameter values
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

// CUDA implementation for multi-kernel
void GenericProcessor::processImagesCUDA() {
    // Validate requirements
    if (!_bufferManager) {
        Logger::getInstance().logMessage("Error: BufferManager not set");
        return;
    }
    
    // Get kernel implementation details
    const auto& implDef = _xmlDef.getCUDAImplementation(_kernelName);
    if (implDef.function.empty()) {
        // No CUDA implementation defined, fall back to CPU
        multiThreadProcessImages(renderWindow);
        return;
    }
    
    // Get frame dimensions
    int width = _bufferManager->getWidth();
    int height = _bufferManager->getHeight();
    
    // Get output buffer
    float* outputBuffer = _bufferManager->getBuffer(_outputBuffer);
    
    // Prepare kernel parameters
    std::vector<void*> kernelParams;
    std::vector<void*> kernelValues; // Hold actual values for parameters
    
    // Add standard parameters
    kernelValues.push_back(new int(width));
    kernelParams.push_back(kernelValues.back());
    
    kernelValues.push_back(new int(height));
    kernelParams.push_back(kernelValues.back());
    
    // Add execution parameters
    kernelValues.push_back(new int(_executionNumber));
    kernelParams.push_back(kernelValues.back());
    
    kernelValues.push_back(new int(_totalExecutions));
    kernelParams.push_back(kernelValues.back());
    
    // Add effect parameters based on mapping
    for (const auto& param : implDef.params) {
        if (param.type == "float" || param.type == "double") {
            if (_paramValues.find(param.name) != _paramValues.end()) {
                double value = _paramValues[param.name].as<double>();
                kernelValues.push_back(new float(static_cast<float>(value)));
                kernelParams.push_back(kernelValues.back());
            }
        } else if (param.type == "int") {
            if (_paramValues.find(param.name) != _paramValues.end()) {
                int value = _paramValues[param.name].as<int>();
                kernelValues.push_back(new int(value));
                kernelParams.push_back(kernelValues.back());
            }
        } else if (param.type == "bool") {
            if (_paramValues.find(param.name) != _paramValues.end()) {
                bool value = _paramValues[param.name].as<bool>();
                kernelValues.push_back(new bool(value));
                kernelParams.push_back(kernelValues.back());
            }
        } else if (param.type == "image") {
            // Handle image parameters
            if (param.role == "source" || param.role == "input") {
                float* buffer = _bufferManager->getSourceBuffer(param.name);
                kernelParams.push_back(&buffer);
            } else if (param.role == "self_previous") {
                float* buffer = _bufferManager->getPreviousIterationBuffer();
                kernelParams.push_back(&buffer);
            } else if (_bufferManager->hasBuffer(param.name)) {
                // Get buffer for previous kernel output
                float* buffer = _bufferManager->getBuffer(param.name);
                kernelParams.push_back(&buffer);
            } else if (!param.optional) {
                Logger::getInstance().logMessage("Error: Required input %s not available", param.name.c_str());
                return;
            } else {
                kernelParams.push_back(nullptr);
            }
        }
    }
    
    // Add output buffer
    kernelParams.push_back(&outputBuffer);
    
    // Call kernel through KernelManager
    KernelManager::invokeCUDAKernel(
        implDef.function,
        _pCudaStream,
        kernelParams
    );
    
    // Clean up allocated parameter values
    for (auto valuePtr : kernelValues) {
        delete valuePtr;
    }
}

// Similar implementations for OpenCL and Metal
// ...
```

**Test Criteria**:
- Processor correctly handles multi-kernel configuration
- Execution numbering works for multi-execution kernels
- Buffer management properly handles intermediate results
- Parameters are correctly passed to each kernel

### Stage 5: Enhanced Kernel Management

**Goal**: Extend KernelManager to support multi-kernel execution with execution counting.

**Steps**:
1. Update KernelManager to support execution counters
2. Add support for accessing multiple input buffers
3. Enhance parameter passing for multi-kernel processing
4. Implement kernel function calling with correct parameter order

**Implementation**:
```cpp
// Update KernelManager class
class KernelManager {
public:
    // CUDA kernel invocation with multi-kernel support
    static void invokeCUDAKernel(
        const std::string& function,
        void* stream,
        const std::vector<void*>& args
    ) {
        // Extract standard parameters
        int width = *static_cast<int*>(args[0]);
        int height = *static_cast<int*>(args[1]);
        int executionNumber = *static_cast<int*>(args[2]);
        int totalExecutions = *static_cast<int*>(args[3]);
        
        // Effect-specific parameters start at index 4
        int paramStart = 4;
        
        // For now, map to specific kernel functions
        // In a more advanced implementation, this would use dynamic loading
        if (function == "GaussianBlurKernel") {
            float radius = *static_cast<float*>(args[paramStart]);
            int quality = *static_cast<int*>(args[paramStart + 1]);
            
            float* source = *static_cast<float**>(args[paramStart + 2]);
            float* mask = args[paramStart + 3] ? *static_cast<float**>(args[paramStart + 3]) : nullptr;
            float* selfPrevious = args[paramStart + 4] ? *static_cast<float**>(args[paramStart + 4]) : nullptr;
            float* output = *static_cast<float**>(args[paramStart + 5]);
            
            // Call multi-execution aware kernel
            RunMultiExecBlurKernel(
                stream, 
                width, height,
                executionNumber, totalExecutions,
                radius, quality,
                source, mask, selfPrevious, output
            );
        } else if (function == "EdgeDetectKernel") {
            float threshold = *static_cast<float*>(args[paramStart]);
            
            float* source = *static_cast<float**>(args[paramStart + 1]);
            float* output = *static_cast<float**>(args[paramStart + 2]);
            
            // Call edge detect kernel
            RunEdgeDetectKernel(
                stream,
                width, height,
                executionNumber, totalExecutions,
                threshold,
                source, output
            );
        } else if (function == "CompositeKernel") {
            float blendAmount = *static_cast<float*>(args[paramStart]);
            
            float* source = *static_cast<float**>(args[paramStart + 1]);
            float* edges = *static_cast<float**>(args[paramStart + 2]);
            float* blur = *static_cast<float**>(args[paramStart + 3]);
            float* output = *static_cast<float**>(args[paramStart + 4]);
            
            // Call composite kernel
            RunCompositeKernel(
                stream,
                width, height,
                executionNumber, totalExecutions,
                blendAmount,
                source, edges, blur, output
            );
        } else {
            Logger::getInstance().logMessage("Unknown CUDA kernel function: %s", function.c_str());
        }
    }
    
    // Similar methods for OpenCL and Metal
    // ...
};
```

**External Kernel Function Declarations**:
```cpp
// Multi-execution aware kernel functions
extern void RunMultiExecBlurKernel(
    void* stream,
    int width, int height,
    int executionNumber, int totalExecutions,
    float radius, int quality,
    const float* source, const float* mask, const float* previousIteration, float* output
);

extern void RunEdgeDetectKernel(
    void* stream,
    int width, int height,
    int executionNumber, int totalExecutions,
    float threshold,
    const float* source, float* output
);

extern void RunCompositeKernel(
    void* stream,
    int width, int height,
    int executionNumber, int totalExecutions,
    float blendAmount,
    const float* source, const float* edges, const float* blur, float* output
);
```

**Test Criteria**:
- KernelManager correctly invokes kernel functions with execution info
- Parameters are passed in the right order to each kernel
- Multiple input buffers are correctly handled
- Error handling gracefully manages missing or invalid inputs

### Stage 6: Sample Multi-Kernel Effects

**Goal**: Create sample effects that demonstrate multi-kernel and multi-execution capabilities.

**Steps**:
1. Create EdgeEnhance effect with edge detection and blur kernels
2. Implement Bloom effect with threshold, blur, and composite kernels
3. Create iterative blur effect with multi-execution kernel
4. Test effects in different OFX hosts

**Sample Effect: EdgeEnhance.xml**:
```xml
<effect name="EdgeEnhance" category="Filter">
  <description>Edge detection and enhancement</description>
  
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
    <parameter name="blendAmount" type="double" default="0.5" min="0.0" max="1.0">
      <label>Blend Amount</label>
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
        <parameter>blendAmount</parameter>
      </column>
    </page>
  </ui>
  
  <kernel name="EdgeDetect" file="EdgeDetect.cu" label="Edge Detection" executions="1">
  </kernel>
  
  <kernel name="Blur" file="GaussianBlur.cu" label="Edge Blur" executions="1">
  </kernel>
  
  <kernel name="Composite" file="Composite.cu" label="Edge Enhancement" executions="1">
  </kernel>
  
  <implementations>
    <cuda function="EdgeDetectKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" />
        <param name="totalExecutions" type="int" />
        <param name="edgeThreshold" type="float" />
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
        <param name="blurRadius" type="float" />
        <param name="quality" type="int" value="8" />
        <param name="source" type="image" role="input" />
        <param name="EdgeDetect" type="image" role="input" />
        <param name="self_previous" type="image" role="input" optional="true" />
        <param name="output" type="image" role="output" />
      </params>
    </cuda>
    
    <cuda function="CompositeKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" />
        <param name="totalExecutions" type="int" />
        <param name="blendAmount" type="float" />
        <param name="source" type="image" role="input" />
        <param name="EdgeDetect" type="image" role="input" />
        <param name="Blur" type="image" role="input" />
        <param name="output" type="image" role="output" />
      </params>
    </cuda>
  </implementations>
</effect>
```

**Sample Effect: IterativeBlur.xml**:
```xml
<effect name="IterativeBlur" category="Filter">
  <description>Progressive blur with multiple iterations</description>
  
  <inputs>
    <source name="source" label="Input Image" />
  </inputs>
  
  <parameters>
    <parameter name="radius" type="double" default="1.0" min="0.1" max="5.0">
      <label>Radius Per Iteration</label>
      <hint>Blur radius applied in each iteration</hint>
    </parameter>
    <parameter name="iterations" type="int" default="5" min="1" max="10">
      <label>Iterations</label>
      <hint>Number of blur iterations</hint>
    </parameter>
  </parameters>
  
  <ui>
    <page name="Main">
      <column name="Parameters">
        <parameter>radius</parameter>
        <parameter>iterations</parameter>
      </column>
    </page>
  </ui>
  
  <kernel name="Blur" file="GaussianBlur.cu" label="Iterative Blur" executions="5">
  </kernel>
  
  <implementations>
    <cuda function="IterativeBlurKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" />
        <param name="totalExecutions" type="int" />
        <param name="radius" type="float" />
        <param name="source" type="image" role="input" />
        <param name="self_previous" type="image" role="input" optional="true" />
        <param name="output" type="image" role="output" />
      </params>
    </cuda>
  </implementations>
</effect>
```

**Test Criteria**:
- Sample effects demonstrate multi-kernel processing
- Effects render correctly with expected visual results
- Parameters correctly influence all relevant kernels
- Multi-execution kernels work with different iteration counts

### Stage 7: Documentation and Tools

**Goal**: Create documentation and tools to help artists create multi-kernel effects.

**Steps**:
1. Create comprehensive documentation for multi-kernel XML schema
2. Implement visual tools for designing effect graphs
3. Add debugging capabilities for multi-kernel effects
4. Create templates for common multi-kernel effect patterns

**Documentation Topics**:
- Multi-kernel XML schema reference
- Buffer management between kernels
- Multi-execution kernel design
- Parameter sharing across kernels
- Performance optimization for multi-kernel effects

**Visual Tools**:
- Effect graph designer
- Parameter UI editor
- XML validator with visual feedback
- Performance analyzer for multi-kernel effects

**Templates**:
- Edge enhancement
- Bloom effect
- Diffusion effects
- Multi-stage noise reduction
- Image processing pipelines

**Test Criteria**:
- Documentation is clear and comprehensive
- Tools help artists create complex effects more easily
- Templates work correctly and provide good starting points
- Validation helps identify and fix common issues

## Conclusion

This implementation plan provides a roadmap for extending the XML-based OFX framework to support multi-kernel and multi-execution processing. By building on the solid foundation of Version 1, Version 2 will enable much more sophisticated image processing effects while maintaining the artist-friendly approach of the framework.

The key benefits of Version 2 will be:
1. Support for complex effects that require multiple processing steps
2. Ability to create iterative algorithms with multiple execution passes
3. Flexible buffer routing between processing stages
4. Reuse of processing building blocks across different effects

Following this plan will result in a powerful framework that allows image processing artists to create sophisticated OFX plugins with minimal C++ knowledge, similar to how Matchbox shaders work in Autodesk Flame.(const std::string& name);
    
    // Common buffers
    float* getSourceBuffer() const { return getBuffer("source"); }
    float* getMaskBuffer() const { return getBuffer("mask"); }
    float* getOutputBuffer() const { return getBuffer("output"); }
    
    // Special buffer for multi-execution (previous iteration)
    float* getPreviousIterationBuffer() const { return getBuffer("temp"); }
    
    // Buffer info
    int getWidth() const { return _width; }
    int getHeight() const { return _height; }
    
private:
    int _width, _height;
    std::map<std::string, float*> _buffers;
    
    void freeBuffer(const std::string& name);
};
```

**Test Criteria**:
- Buffer creation and management works correctly
- Buffer swapping efficiently handles multi-execution kernels
- Memory allocation and cleanup is properly handled
- Buffers are correctly sized for the frame dimensions

### Stage 3: Multi-Pass Processing in GenericEffect

**Goal**: Extend the GenericEffect class to support sequential multi-pass processing.

**Steps**:
1. Update process() method to handle multiple passes
2. Implement buffer routing between passes
3. Add support for processing pass-specific parameters
4. Create execution plan based on XML definition

**Implementation**:
```cpp
// Update GenericEffect class
void GenericEffect::process(const OFX::RenderArguments& args) {
    // For single-pass effects, just use the original method
    if (_xmlDef.getPasses().size() <= 1) {
        processSinglePass(args);
        return;
    }
    
    // Create buffer manager for the frame
    const OfxRectI& bounds = _dstClip->getPixelData()->getBounds();
    int width = bounds.x2 - bounds.x1;
    int height = bounds.y2 - bounds.y1;
    BufferManager bufferManager(width, height);
    
    // Set up source and output buffers
    OFX::Image* dstImg = _dstClip->fetchImage(args.time);
    bufferManager.createBuffer("output");
    float* outputBuffer = bufferManager.getOutputBuffer();
    
    // Load source image into buffer
    std::string sourceClipName = findMainSourceClip();
    if (_srcClips[sourceClipName]) {
        OFX::Image* srcImg = _srcClips[sourceClipName]->fetchImage(args.time);
        float* sourceData = static_cast<float*>(srcImg->getPixelData());
        
        bufferManager.createBuffer("source");
        float* sourceBuffer = bufferManager.getSourceBuffer();
        memcpy(sourceBuffer, sourceData, width * height * 4 * sizeof(float));
    }
    
    // Load mask if available
    std::string maskClipName = findMaskClip();
    if (!maskClipName.empty() && _srcClips[maskClipName] && _srcClips[maskClipName]->isConnected()) {
        OFX::Image* maskImg = _srcClips[maskClipName]->fetchImage(args.time);
        float* maskData = static_cast<float*>(maskImg->getPixelData());
        
        bufferManager.createBuffer("mask");
        float* maskBuffer = bufferManager.getMaskBuffer();
        memcpy(maskBuffer, maskData, width * height * 4 * sizeof(float));
    }
    
    // Collect global parameter values
    std::map<std::string, OFX::ParamValue> globalParams = collectParameterValues(args.time);
    
    // Process each pass in sequence
    const auto& passes = _xmlDef.getPasses();
    for (const auto& pass : passes) {
        // Create pass output buffer if it doesn't exist
        std::string passOutputName = "pass:" + pass.name;
        if (!bufferManager.hasBuffer(passOutputName)) {
            bufferManager.createBuffer(passOutputName);
        }
        
        // Create temp buffer for multi-execution if needed
        if (pass.executions > 1 && !bufferManager.hasBuffer("temp")) {
            bufferManager.createBuffer("temp");
        }
        
        // Create input mapping for this pass
        std::map<std::string, std::string> inputMapping;
        for (const auto& input : pass.inputs) {
            inputMapping[input.name] = input.source;
        }
        
        // Merge global and pass-specific parameters
        std::map<std::string, OFX::ParamValue> passParams = globalParams;
        for (const auto& param : pass.parameters) {
            // Overwrite with pass-specific parameter
            if (_doubleParams.find(param.name) != _doubleParams.end()) {
                passParams[param.name] = _doubleParams[param.name]->getValueAtTime(args.time);
            }
            // Handle other parameter types...
        }
        
        // Execute the pass (possibly multiple times)
        for (int execution = 0; execution < pass.executions; execution++) {
            // Determine current and previous buffers
            std::string currentOutput = (execution == pass.executions - 1) ? 
                                      passOutputName : "temp";
            
            std::string previousOutput = (execution == 0) ? 
                                      passOutputName : // First execution, no previous
                                      (execution % 2 == 1) ? passOutputName : "temp";
            
            // Execute the kernel for this pass
            executePassKernel(
                pass.kernel,
                execution,
                pass.executions,
                inputMapping,
                previousOutput,
                currentOutput,
                passParams,
                bufferManager,
                args
            );
            
            // If not the last execution, swap buffers for next iteration
            if (execution < pass.executions - 1 && execution % 2 == 0) {
                bufferManager.swapBuffers(passOutputName, "temp");
            }
        }
    }
    
    // Copy the last pass result to the output
    const std::string finalPassName = passes.back().name;
    const std::string finalOutputName = "pass:" + finalPassName;
    
    float* finalOutput = bufferManager.getBuffer(finalOutputName);
    float* dstData = static_cast<float*>(dstImg->getPixelData());
    
    memcpy(dstData, finalOutput, width * height * 4 * sizeof(float));
}

// Helper method to execute a single pass kernel
void GenericEffect::executePassKernel(
    const std::string& kernelName,
    int executionNumber,
    int totalExecutions,
    const std::map<std::string, std::string>& inputMapping,
    const std::string& previousOutputName,
    const std::string& currentOutputName,
    const std::map<std::string, OFX::ParamValue>& params,
    BufferManager& bufferManager,
    const OFX::RenderArguments& args
) {
    // Create processor
    GenericProcessor processor(*this, _xmlDef);
    
    // Set up processor
    processor.setDstImg(_dstClip->fetchImage(args.time));
    processor.setRenderWindow(args.renderWindow);
    processor.setGPURenderArgs(args);
    
    // Set up buffer mapping
    processor.setBufferManager(bufferManager);
    processor.setInputMapping(inputMapping);
    processor.setPreviousOutput(previousOutputName);
    processor.setCurrentOutput(currentOutputName);
    
    // Set execution info
    processor.setExecutionInfo(executionNumber, totalExecutions);
    
    // Set parameters and process
    processor.setParamValues(params);
    processor.process();
}
```

**Test Criteria**:
- Multiple passes execute in the correct sequence
- Buffers are correctly routed between passes
- Multi-execution kernels work with the right iteration count
- Pass-specific parameters override global parameters
- Complex effects with multiple passes render correctly

### Stage 4: Enhanced GenericProcessor for Multi-Pass Support

**Goal**: Extend the GenericProcessor to handle multi-pass and multi-execution processing.

**Steps**:
1. Update processor to work with the BufferManager
2. Add support for execution counting
3. Implement dynamic input mapping
4. Update GPU kernel calls to support multi-pass

**Implementation**:
```cpp
// Update GenericProcessor class
class GenericProcessor : public OFX::ImageProcessor {
private:
    GenericEffect& _effect;
    XMLEffectDefinition& _xmlDef;
    std::map<std::string, OFX::ParamValue> _paramValues;
    
    // Multi-pass additions
    BufferManager* _bufferManager;
    std::map<std::string, std::string> _inputMapping;
    std::string _previousOutput;
    std::string _currentOutput;
    
    int _executionNumber;
    int _totalExecutions;
    
public:
    // Constructor
    GenericProcessor(GenericEffect& effect, XMLEffectDefinition& xmlDef)
        : OFX::ImageProcessor(effect), _effect(effect), _xmlDef(xmlDef), 
          _bufferManager(nullptr), _executionNumber(0), _totalExecutions(1) {}
    
    // Buffer manager
    void setBufferManager(BufferManager& bufferManager) { 
        _bufferManager = &bufferManager; 
    }
    
    // Input/output mapping
    void setInputMapping(const std::map<std::string, std::string>& mapping) {
        _inputMapping = mapping;
    }
    
    void setPreviousOutput(const std::string& name) { 
        _previousOutput = name; 
    }
    
    void setCurrentOutput(const std::string& name) { 
        _currentOutput = name; 
    }
    
    // Execution info
    void setExecutionInfo(int executionNumber, int totalExecutions) {
        _executionNumber = executionNumber;
        _totalExecutions = totalExecutions;
    }
    
    // Parameter values
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

// CUDA implementation for multi-pass
void GenericProcessor::processImagesCUDA() {
    // Validate requirements
    if (!_bufferManager) {
        Logger::getInstance().logMessage("Error: BufferManager not set");
        return;
    }
    
    // Get kernel definition
    const auto& kernelDef = _xmlDef.getKernelByType("cuda");
    if (kernelDef.file.empty() || kernelDef.function.empty()) {
        // No CUDA kernel defined, fall back to CPU
        multiThreadProcessImages(renderWindow);
        return;
    }
    
    // Get frame dimensions
    int width = _bufferManager->getWidth();
    int height = _bufferManager->getHeight();
    
    // Prepare input buffers based on mapping
    std::map<std::string, float*> inputBuffers;
    for (const auto& mapping : _inputMapping) {
        std::string inputName = mapping.first;
        std::string sourceName = mapping.second;
        
        if (sourceName == "source") {
            inputBuffers[inputName] = _bufferManager->getSourceBuffer();
        } else if (sourceName == "mask") {
            inputBuffers[inputName] = _bufferManager->getMaskBuffer();
        } else if (sourceName.substr(0, 5) == "pass:") {
            inputBuffers[inputName] = _bufferManager->getBuffer(sourceName);
        } else if (sourceName == "self.previous") {
            inputBuffers[inputName] = _bufferManager->getBuffer(_previousOutput);
        }
    }
    
    // Get output buffer
    float* outputBuffer = _bufferManager->getBuffer(_currentOutput);
    
    // Prepare kernel parameters in the order defined in XML
    std::vector<void*> kernelParams;
    std::vector<void*> kernelValues; // Hold actual values for parameters
    
    // Add standard parameters
    kernelValues.push_back(new int(width));
    kernelParams.push_back(kernelValues.back());
    
    kernelValues.push_back(new int(height));
    kernelParams.push_back(kernelValues.back());
    
    // Add execution parameters
    kernelValues.push_back(new int(_executionNumber));
    kernelParams.push_back(kernelValues.back());
    
    kernelValues.push_back(new int(_totalExecutions));
    kernelParams.push_back(kernelValues.back());
    
    // Add effect parameters based on mapping
    for (const auto& param : kernelDef.params) {
        if (!param.mapTo.empty() && _paramValues.find(param.mapTo) != _paramValues.end()) {
            // Map to parameter value
            if (param.type == "double" || param.type == "float") {
                double value = _paramValues[param.mapTo].as<double>();
                kernelValues.push_back(new float(static_cast<float>(value)));
                kernelParams.push_back(kernelValues.back());
            } else if (param.type == "int") {
                int value = _paramValues[param.mapTo].as<int>();
                kernelValues.push_back(new int(value));
                kernelParams.push_back(kernelValues.back());
            } else if (param.type == "bool") {
                bool value = _paramValues[param.mapTo].as<bool>();
                kernelValues.push_back(new bool(value));
                kernelParams.push_back(kernelValues.back());
            }
        }
    }
    
    // Add input buffer pointers
    for (const auto& param : kernelDef.params) {
        if (param.type == "image" && param.role == "input") {
            if (inputBuffers.find(param.name) != inputBuffers.end()) {
                kernelParams.push_back(&inputBuffers[param.name]);
            } else if (!param.optional) {
                Logger::getInstance().logMessage("Error: Required input %s not available", param.name.c_str());
                return;
            } else {
                kernelParams.push_back(nullptr);
            }
        }
    }
    
    // Add output buffer
    kernelParams.push_back(&outputBuffer);
    
    // Call kernel through KernelManager
    KernelManager::invokeCUDAKernel(
        kernelDef.function,
        _pCudaStream,
        kernelParams
    );
    
    // Clean up allocated parameter values
    for (auto valuePtr : kernelValues) {
        delete valuePtr;
    }
}

// Similar implementations for OpenCL and Metal
// ...
```

**Test Criteria**:
- Processor correctly handles multi-pass inputs and outputs
- Execution numbering works for multi-execution kernels
- Input mapping correctly routes buffers between passes
- Buffer management properly handles intermediate results

### Stage 5: Enhanced Kernel Management

**Goal**: Extend KernelManager to support multi-pass kernels with execution counting.

**Steps**:
1. Update KernelManager to support execution counters
2. Add support for accessing multiple input buffers
3. Enhance parameter passing for multi-pass kernels
4. Implement kernel function calling with correct parameter order

**Implementation**:
```cpp
// Update KernelManager class
class KernelManager {
public:
    // CUDA kernel invocation with multi-pass support
    static void invokeCUDAKernel(
        const std::string& function,
        void* stream,
        const std::vector<void*>& args
    ) {
        // Extract standard parameters
        int width = *static_cast<int*>(args[0]);
        int height = *static_cast<int*>(args[1]);
        int executionNumber = *static_cast<int*>(args[2]);
        int totalExecutions = *static_cast<int*>(args[3]);
        
        // Effect-specific parameters start at index 4
        int paramStart = 4;
        
        // For now, map to specific kernel functions
        // In a more advanced implementation, this would use dynamic loading
        if (function == "GaussianBlurKernel") {
            float radius = *static_cast<float*>(args[paramStart]);
            int quality = *static_cast<int*>(args[paramStart + 1]);
            float maskStrength = *static_cast<float*>(args[paramStart + 2]);
            
            float* input = *static_cast<float**>(args[paramStart + 3]);
            float* mask = args[paramStart + 4] ? *static_cast<float**>(args[paramStart + 4]) : nullptr;
            float* output = *static_cast<float**>(args[paramStart + 5]);
            
            // Call multi-execution aware kernel
            RunMultiExecCudaKernel(
                stream, 
                width, height,
                executionNumber, totalExecutions,
                radius, quality, maskStrength,
                input, mask, output
            );
        } else if (function == "EdgeDetectKernel") {
            float threshold = *static_cast<float*>(args[paramStart]);
            
            float* input = *static_cast<float**>(args[paramStart + 1]);
            float* output = *static_cast<float**>(args[paramStart + 2]);
            
            // Call edge detect kernel
            RunEdgeDetectCudaKernel(
                stream,
                width, height,
                threshold,
                input, output
            );
        } else if (function == "CompositeKernel") {
            float blendAmount = *static_cast<float*>(args[paramStart]);
            
            float* original = *static_cast<float**>(args[paramStart + 1]);
            float* edges = *static_cast<float**>(args[paramStart + 2]);
            float* blurred = *static_cast<float**>(args[paramStart + 3]);
            float* output = *static_cast<float**>(args[paramStart + 4]);
            
            // Call composite kernel
            RunCompositeCudaKernel(
                stream,
                width, height,
                blendAmount,
                original, edges, blurred, output
            );
        } else {
            Logger::getInstance().logMessage("Unknown CUDA kernel function: %s", function.c_str());
        }
    }
    
    // Similar methods for OpenCL and Metal
    // ...
};
```

**External Kernel Function Declarations**:
```cpp
// Multi-execution aware kernel functions
extern void RunMultiExecCudaKernel(
    void* stream,
    int width, int height,
    int executionNumber, int totalExecutions,
    float radius, int quality, float maskStrength,
    const float* input, const float* mask, float* output
);

extern void RunEdgeDetectCudaKernel(
    void* stream,
    int width, int height,
    float threshold,
    const float* input, float* output
);

extern void RunCompositeCudaKernel(
    void* stream,
    int width, int height,
    float blendAmount,
    const float* original, const float* edges, const float* blurred, float* output
);
```

**Test Criteria**:
- KernelManager correctly invokes kernel functions with execution info
- Parameters are passed in the right order to each kernel
- Multiple input buffers are correctly handled
- Error handling gracefully manages missing or invalid inputs

### Stage 6: Sample Multi-Pass Effects

**Goal**: Create sample effects that demonstrate multi-pass and multi-execution capabilities.

**Steps**:
1. Create EdgeEnhance effect with edge detection and blur passes
2. Implement Bloom effect with threshold, blur, and composite passes
3. Create iterative blur effect with multi-execution kernel
4. Test effects in different OFX hosts

**Sample Effect: EdgeEnhance.xml**:
```xml
<effect name="EdgeEnhance" category="Filter">
  <description>Edge detection and enhancement</description>
  
  <parameters>
    <parameter name="edgeThreshold" type="double" default="0.2" min="0.0" max="1.0">
      <label>Edge Threshold</label>
      <hint>Threshold for edge detection</hint>
    </parameter>
    <parameter name="blurRadius" type="double" default="3.0" min="0.0" max="10.0">
      <label>Blur Radius</label>
      <hint>Radius for edge smoothing</hint>
    </parameter>
    <parameter name="blendAmount" type="double" default="0.5" min="0.0" max="1.0">
      <label>Blend Amount</label>
      <hint>Strength of edge enhancement</hint>
    </parameter>
  </parameters>
  
  <passes>
    <pass name="EdgeDetect" kernel="EdgeDetect.cu" executions="1">
      <inputs>
        <input name="source" source="source" />
      </inputs>
    </pass>
    
    <pass name="Blur" kernel="GaussianBlur.cu" executions="1">
      <inputs>
        <input name="input" source="pass:EdgeDetect" />
      </inputs>
      <parameters>
        <parameter name="radius" mapTo="blurRadius" />
        <parameter name="quality" type="int" default="8" min="1" max="16" />
      </parameters>
    </pass>
    
    <pass name="Composite" kernel="Composite.cu" executions="1">
      <inputs>
        <input name="original" source="source" />
        <input name="edges" source="pass:EdgeDetect" />
        <input name="blurred" source="pass:Blur" />
      </inputs>
      <parameters>
        <parameter name="amount" mapTo="blendAmount" />
      </parameters>
    </pass>
  </passes>
  
  <kernels>
    <kernel type="cuda" file="EdgeDetect.cu" function="EdgeDetectKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" />
        <param name="totalExecutions" type="int" />
        <param name="threshold" mapTo="edgeThreshold" />
        <param name="input" type="image" role="input" />
        <param name="output" type="image" role="output" />
      </params>
    </kernel>
    
    <kernel type="cuda" file="GaussianBlur.cu" function="GaussianBlurKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" />
        <param name="totalExecutions" type="int" />
        <param name="radius" mapTo="radius" />
        <param name="quality" mapTo="quality" />
        <param name="input" type="image" role="input" />
        <param name="mask" type="image" role="mask" optional="true" />
        <param name="output" type="image" role="output" />
      </params>
    </kernel>
    
    <kernel type="cuda" file="Composite.cu" function="CompositeKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" />
        <param name="totalExecutions" type="int" />
        <param name="blendAmount" mapTo="amount" />
        <param name="original" type="image" role="input" />
        <param name="edges" type="image" role="input" />
        <param name="blurred" type="image" role="input" />
        <param name="output" type="image" role="output" />
      </params>
    </kernel>
  </kernels>
</effect>
```

**Sample Effect: IterativeBlur.xml**:
```xml
<effect name="IterativeBlur" category="Filter">
  <description>Progressive blur with multiple iterations</description>
  
  <parameters>
    <parameter name="radius" type="double" default="1.0" min="0.1" max="5.0">
      <label>Radius Per Iteration</label>
      <hint>Blur radius applied in each iteration</hint>
    </parameter>
    <parameter name="iterations" type="int" default="5" min="1" max="10">
      <label>Iterations</label>
      <hint>Number of blur iterations</hint>
    </parameter>
  </parameters>
  
  <passes>
    <pass name="Blur" kernel="GaussianBlur.cu" executions="5">
      <inputs>
        <input name="source" source="source" />
        <input name="previous" source="self.previous" />
      </inputs>
    </pass>
  </passes>
  
  <kernels>
    <kernel type="cuda" file="GaussianBlur.cu" function="IterativeBlurKernel">
      <params>
        <param name="width" type="int" />
        <param name="height" type="int" />
        <param name="executionNumber" type="int" />
        <param name="totalExecutions" type="int" />
        <param name="radius" mapTo="radius" />
        <param name="source" type="image" role="input" />
        <param name="previous" type="image" role="input" optional="true" />
        <param name="output" type="image" role="output" />
      </params>
    </kernel>
  </kernels>
</effect>
```

**Test Criteria**:
- Sample effects demonstrate multi-pass processing
- Effects render correctly with expected visual results
- Parameters correctly influence all relevant passes
- Multi-execution kernels work with different iteration counts

### Stage 7: Documentation and Tools

**Goal**: Create documentation and tools to help artists create multi-pass effects.

**Steps**:
1. Create comprehensive documentation for multi-pass XML schema
2. Implement visual tools for designing effect graphs
3. Add debugging capabilities for multi-pass effects
4. Create templates for common multi-pass effect patterns

**Documentation Topics**:
- Multi-pass XML schema reference
- Buffer management between passes
- Multi-execution kernel design
- Parameter mapping in complex effects
- Performance optimization for multi-pass effects

**Visual Tools**:
- Effect graph designer
- Parameter UI editor
- XML validator with visual feedback
- Performance analyzer for multi-pass effects

**Templates**:
- Edge enhancement
- Bloom effect
- Diffusion effects
- Multi-stage noise reduction
- Image processing pipelines

**Test Criteria**:
- Documentation is clear and comprehensive
- Tools help artists create complex effects more easily
- Templates work correctly and provide good starting points
- Validation helps identify and fix common issues

## Conclusion

This implementation plan provides a roadmap for extending the XML-based OFX framework to support multi-pass and multi-execution processing. By building on the solid foundation of Version 1, Version 2 will enable much more sophisticated image processing effects while maintaining the artist-friendly approach of the framework.

The key benefits of Version 2 will be:
1. Support for complex effects that require multiple processing steps
2. Ability to create iterative algorithms with multiple execution passes
3. Flexible buffer routing between processing stages
4. Reuse of processing building blocks across different effects

Following this plan will result in a powerful framework that allows image processing artists to create sophisticated OFX plugins with minimal C++ knowledge, similar to how Matchbox shaders work in Autodesk Flame.
