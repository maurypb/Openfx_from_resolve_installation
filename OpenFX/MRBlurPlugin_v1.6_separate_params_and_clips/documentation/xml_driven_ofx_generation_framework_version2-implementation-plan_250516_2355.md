# XML-Based OFX Framework Implementation Plan - Version 2

## Introduction

This document outlines the implementation plan for Version 2 of the XML-based OFX framework, which extends the single-kernel system to support multi-kernel and multi-execution processing. This plan assumes that Version 1 has been successfully implemented and tested, providing a solid foundation of XML parsing, parameter handling, and basic kernel execution.

> **Key Principle**: The user should only need to modify XML effect definitions and kernel code files, never the framework code itself.

## Prerequisites

Before beginning Version 2 implementation, ensure:
- Complete Version 1 implementation is working
- XML parsing and validation is robust
- Parameter and input handling is working correctly
- Single-kernel processing works with all GPU types
- Border mode handling for sources works as expected

## Phase 1: Enhanced XML Parsing for Multi-Kernel Support

### Step 1.1: Extend XML Schema (2-3 days)

**Goal**: Enhance the XML schema to support multi-kernel processing.

**Tasks**:
1. Add `<pipeline>` section to replace `<kernels>` for multi-kernel effects
2. Add `<step>` elements with name and execution count
3. Add nested kernel definitions for each step
4. Create sample multi-kernel effect XML

**Example**:
```xml
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
</pipeline>
```

**Test Criteria**:
- XML schema correctly defines pipeline structure
- Sample XML validates against schema
- XML parser can distinguish between Version 1 and Version 2 formats

### Step 1.2: Update XMLEffectDefinition (1-2 days)

**Goal**: Extend XMLEffectDefinition to parse and store multi-kernel information.

**Tasks**:
1. Add pipeline and step structures to XMLEffectDefinition
2. Implement parsing for pipeline elements
3. Modify kernel parsing to handle nested kernels in steps
4. Add accessors for pipeline information

**Implementation**:
```cpp
class XMLEffectDefinition {
public:
    // Existing code from Version 1...
    
    // New structures for pipeline
    struct StepKernelDef {
        std::string platform; // "cuda", "opencl", "metal"
        std::string file;
    };
    
    struct PipelineStep {
        std::string name;
        int executions;
        std::vector<StepKernelDef> kernels;
    };
    
    // Accessors for pipeline
    bool hasPipeline() const;
    std::vector<PipelineStep> getPipelineSteps() const;
    
private:
    // Existing members from Version 1...
    
    bool _hasPipeline;
    std::vector<PipelineStep> _pipelineSteps;
};
```

**Test Criteria**:
- Parse both Version 1 and Version 2 XML formats correctly
- Provide backward compatibility for single-kernel effects
- Correctly parse multi-step pipelines with execution counts

## Phase 2: Buffer Management

### Step 2.1: BufferManager Class (3-4 days)

**Goal**: Create a buffer management system for multi-kernel processing.

**Tasks**:
1. Implement BufferManager class to handle image buffers
2. Add support for named input/output/intermediate buffers
3. Add buffer creation and access methods
4. Implement buffer swapping for multi-execution passes

**Implementation**:
```cpp
class BufferManager {
public:
    BufferManager(int width, int height);
    ~BufferManager();
    
    // Buffer creation and management
    void addSourceBuffer(const std::string& name, float* data, const std::string& borderMode);
    float* createIntermediateBuffer(const std::string& name);
    float* getBuffer(const std::string& name) const;
    bool hasBuffer(const std::string& name) const;
    
    // Buffer operations
    void copyBuffer(const std::string& source, const std::string& destination);
    void swapBuffers(const std::string& buffer1, const std::string& buffer2);
    void clearBuffer(const std::string& name);
    
    // Special buffers
    float* getOutputBuffer() { return getBuffer("output"); }
    
    // Buffer info
    int getWidth() const { return _width; }
    int getHeight() const { return _height; }
    std::string getBorderMode(const std::string& name) const;
    
private:
    int _width, _height;
    
    struct Buffer {
        float* data;
        std::string borderMode;
        bool ownedByManager; // If true, manager must free the memory
    };
    
    std::map<std::string, Buffer> _buffers;
    void freeBuffer(const std::string& name);
};
```

**Test Criteria**:
- Buffer creation and management works correctly
- Buffer swapping correctly handles data
- Border modes are preserved with buffers
- Memory management works without leaks

### Step 2.2: Buffer Routing Logic (2-3 days)

**Goal**: Create a system to route data between pipeline steps.

**Tasks**:
1. Implement buffer routing based on step names
2. Add special handling for self.previous in multi-execution
3. Implement automatic pass-through of source buffers

**Implementation**:
```cpp
class BufferRouter {
public:
    BufferRouter(BufferManager& bufferManager);
    
    // Initialize routing for a pipeline
    void initializePipeline(const std::vector<XMLEffectDefinition::PipelineStep>& steps);
    
    // Get input buffers for a step
    std::vector<std::pair<std::string, float*>> getInputBuffersForStep(
        const std::string& stepName,
        int executionNumber
    );
    
    // Get output buffer for a step
    float* getOutputBufferForStep(
        const std::string& stepName,
        int executionNumber,
        int totalExecutions
    );
    
    // Prepare for next execution in multi-execution step
    void prepareNextExecution(
        const std::string& stepName,
        int executionNumber,
        int totalExecutions
    );
    
private:
    BufferManager& _bufferManager;
    std::vector<std::string> _stepNames;
    
    // Create special buffers for multi-execution steps
    void createMultiExecutionBuffers(const std::string& stepName, int totalExecutions);
};
```

**Test Criteria**:
- Correct buffer routing between steps
- Multi-execution buffer swapping works
- Previous execution results accessible when needed
- All source buffers automatically available to all steps

## Phase 3: Enhanced GenericEffect for Multi-Kernel Processing

### Step 3.1: Update GenericEffect (2-3 days)

**Goal**: Extend GenericEffect to support multi-kernel processing.

**Tasks**:
1. Add pipeline processing method to GenericEffect
2. Add support for determining if effect uses pipeline
3. Add buffer management for multi-kernel effects
4. Update render method to handle both single and multi-kernel effects

**Implementation**:
```cpp
class GenericEffect : public OFX::ImageEffect {
    // Existing code from Version 1...
    
public:
    // Override render
    virtual void render(const OFX::RenderArguments& args) override;
    
protected:
    // Process methods for different versions
    void processSingleKernel(const OFX::RenderArguments& args);
    void processPipeline(const OFX::RenderArguments& args);
    
    // Process a single pipeline step
    void processStep(
        const XMLEffectDefinition::PipelineStep& step,
        int executionNumber,
        int totalExecutions,
        BufferManager& bufferManager,
        BufferRouter& bufferRouter,
        const OFX::RenderArguments& args
    );
};

// Implementation of the render method
void GenericEffect::render(const OFX::RenderArguments& args) {
    if (_xmlDef.hasPipeline()) {
        processPipeline(args);
    } else {
        processSingleKernel(args);
    }
}
```

**Test Criteria**:
- Generic effect detects and processes both single-kernel and pipeline effects
- Pipeline processing correctly sequences steps
- Multi-execution steps work correctly
- Results match expected output for both single and multi-kernel effects

### Step 3.2: Pipeline Processing Implementation (3-4 days)

**Goal**: Implement pipeline processing in GenericEffect.

**Tasks**:
1. Implement processPipeline method
2. Add buffer setup for pipeline
3. Handle multi-execution steps
4. Handle final output copying

**Implementation**:
```cpp
void GenericEffect::processPipeline(const OFX::RenderArguments& args) {
    // Create buffer manager and router
    const OfxRectI& bounds = _dstClip->fetchImage(args.time)->getBounds();
    int width = bounds.x2 - bounds.x1;
    int height = bounds.y2 - bounds.y1;
    BufferManager bufferManager(width, height);
    BufferRouter bufferRouter(bufferManager);
    
    // Initialize pipeline
    bufferRouter.initializePipeline(_xmlDef.getPipelineSteps());
    
    // Set up source buffers
    for (const auto& entry : _srcClips) {
        const std::string& clipName = entry.first;
        const InputClip& inputClip = entry.second;
        
        if (inputClip.clip && inputClip.clip->isConnected()) {
            std::unique_ptr<OFX::Image> src(inputClip.clip->fetchImage(args.time));
            float* srcData = static_cast<float*>(src->getPixelData());
            bufferManager.addSourceBuffer(clipName, srcData, inputClip.borderMode);
        }
    }
    
    // Create output buffer
    std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
    float* outputData = static_cast<float*>(dst->getPixelData());
    bufferManager.addSourceBuffer("output", outputData, "clamp");
    
    // Process each step in the pipeline
    const auto& steps = _xmlDef.getPipelineSteps();
    for (const auto& step : steps) {
        for (int execution = 0; execution < step.executions; ++execution) {
            processStep(step, execution, step.executions, bufferManager, bufferRouter, args);
            
            // Prepare for next execution if needed
            if (execution < step.executions - 1) {
                bufferRouter.prepareNextExecution(step.name, execution, step.executions);
            }
        }
    }
    
    // Copy final result to output
    // This happens automatically since we set up the output buffer directly
}
```

**Test Criteria**:
- Pipeline steps execute in correct sequence
- Multi-execution steps run the correct number of times
- Intermediate buffers are correctly managed
- Output is correctly produced

## Phase 4: Enhanced Processor and Kernel Management

### Step 4.1: Update GenericProcessor (2-3 days)

**Goal**: Enhance GenericProcessor to support multi-kernel processing.

**Tasks**:
1. Add buffer manager and router integration
2. Add step-specific processing methods
3. Update GPU processing methods for pipeline steps
4. Handle dynamic input routing for kernels

**Implementation**:
```cpp
class GenericProcessor : public OFX::ImageProcessor {
    // Existing code from Version 1...
    
private:
    // New members for Version 2
    BufferManager* _bufferManager;
    BufferRouter* _bufferRouter;
    std::string _stepName;
    int _executionNumber;
    int _totalExecutions;
    float* _outputBuffer;
    
public:
    // New methods for Version 2
    void setBufferManager(BufferManager* bufferManager) { _bufferManager = bufferManager; }
    void setBufferRouter(BufferRouter* bufferRouter) { _bufferRouter = bufferRouter; }
    void setStepInfo(const std::string& stepName, int executionNumber, int totalExecutions) {
        _stepName = stepName;
        _executionNumber = executionNumber;
        _totalExecutions = totalExecutions;
    }
    void setOutputBuffer(float* outputBuffer) { _outputBuffer = outputBuffer; }
    
    // Updated GPU processing methods
    virtual void processImagesCUDA() override;
    virtual void processImagesOpenCL() override;
    virtual void processImagesMetal() override;
};
```

**Test Criteria**:
- Processor correctly handles buffer manager and router
- Step-specific information is used in processing
- Input buffers are correctly routed to kernels
- Output is written to the correct buffer

### Step 4.2: Update KernelManager (2-3 days)

**Goal**: Enhance KernelManager to support multi-kernel processing.

**Tasks**:
1. Update kernel execution to handle multiple input buffers
2. Add support for step name and execution information
3. Enhance parameter passing for pipeline steps
4. Update error handling for multi-kernel effects

**Implementation**:
```cpp
class KernelManager {
public:
    // Updated CUDA kernel execution
    static void executeCUDAKernel(
        const std::string& kernelFile,
        const std::string& stepName,
        int executionNumber,
        int totalExecutions,
        void* stream,
        int width, int height,
        const std::vector<std::pair<std::string, float*>>& inputBuffers,
        const std::map<std::string, std::string>& borderModes,
        float* outputBuffer,
        const std::map<std::string, double>& doubleParams,
        const std::map<std::string, int>& intParams,
        const std::map<std::string, bool>& boolParams
    );
    
    // Similar updates for OpenCL and Metal
};

// Implementation
void KernelManager::executeCUDAKernel(...) {
    // Load kernel
    void* kernel = loadCUDAKernel(kernelFile);
    
    // Prepare parameters
    std::vector<void*> params;
    
    // Add standard parameters
    params.push_back(&width);
    params.push_back(&height);
    params.push_back(&executionNumber);
    params.push_back(&totalExecutions);
    
    // Add input buffers
    for (const auto& input : inputBuffers) {
        const std::string& name = input.first;
        float* buffer = input.second;
        params.push_back(&buffer);
        
        // Add border mode for this input
        int borderModeValue = convertBorderModeToInt(borderModes.at(name));
        params.push_back(&borderModeValue);
    }
    
    // Add output buffer
    params.push_back(&outputBuffer);
    
    // Add all other parameters
    for (const auto& param : doubleParams) {
        double value = param.second;
        params.push_back(&value);
    }
    // Add int and bool parameters...
    
    // Execute kernel
    executeCUDAFunction(kernel, stream, params);
}
```

**Test Criteria**:
- Kernel manager correctly handles multiple input buffers
- Execution information is correctly passed to kernels
- Border modes are passed with each input buffer
- All parameters are passed to kernels correctly

## Phase 5: Testing and Validation

### Step 5.1: Integration Testing (3-4 days)

**Goal**: Test the framework with complete multi-kernel effects.

**Tasks**:
1. Create EdgeEnhance.xml with edge detection, blur, and composite steps
2. Create matching CUDA, OpenCL, and Metal kernels
3. Test in various OFX hosts
4. Compare results with expected output

**Test Criteria**:
- Multi-kernel effects load and run correctly
- Pipeline steps execute in the correct order
- Multi-execution steps work correctly
- Results match expected output

### Step 5.2: Performance Testing (2-3 days)

**Goal**: Ensure multi-kernel effects perform well.

**Tasks**:
1. Create performance tests for multi-kernel effects
2. Compare with equivalent single-kernel versions
3. Optimize buffer management if needed
4. Measure memory usage

**Test Criteria**:
- Multi-kernel effects have acceptable performance
- Memory usage is reasonable
- No memory leaks in buffer management
- Performance scales reasonably with complexity

### Step 5.3: Error Handling and Recovery (2-3 days)

**Goal**: Ensure the framework handles errors gracefully.

**Tasks**:
1. Add error handling for pipeline processing
2. Implement recovery for failed steps
3. Add detailed logging for debugging
4. Create test cases for error conditions

**Test Criteria**:
- Framework handles errors without crashing
- Meaningful error messages are provided
- Failed steps are reported correctly
- Logging provides useful debugging information

## Phase 6: Documentation and Examples

### Step 6.1: Pipeline Documentation (2 days)

**Goal**: Create documentation for multi-kernel effects.

**Tasks**:
1. Update XML schema documentation for pipeline structure
2. Document buffer routing system
3. Explain multi-execution step handling
4. Document border mode handling for intermediate buffers

### Step 6.2: Advanced Examples (3 days)

**Goal**: Create example effects demonstrating Version 2 features.

**Tasks**:
1. Create EdgeEnhance example (edge detection + blur + composite)
2. Create Bloom example (threshold + multi-pass blur + composite)
3. Create NoiseReduction example (multi-execution denoise)
4. Document examples with explanations

### Step 6.3: Troubleshooting Guide (2 days)

**Goal**: Create a guide for debugging multi-kernel effects.

**Tasks**:
1. Document common pipeline issues
2. Create guide for debugging buffer routing
3. Explain multi-execution debugging
4. Provide tips for optimizing multi-kernel effects

## Timeline Summary

**Phase 1: Enhanced XML Parsing for Multi-Kernel Support** (3-5 days)
- Extend XML Schema
- Update XMLEffectDefinition

**Phase 2: Buffer Management** (5-7 days)
- BufferManager Class
- Buffer Routing Logic

**Phase 3: Enhanced GenericEffect for Multi-Kernel Processing** (5-7 days)
- Update GenericEffect
- Pipeline Processing Implementation

**Phase 4: Enhanced Processor and Kernel Management** (4-6 days)
- Update GenericProcessor
- Update KernelManager

**Phase 5: Testing and Validation** (7-10 days)
- Integration Testing
- Performance Testing
- Error Handling and Recovery

**Phase 6: Documentation and Examples** (7 days)
- Pipeline Documentation
- Advanced Examples
- Troubleshooting Guide

**Total Estimated Time**: 31-42 days

## Conclusion

This implementation plan provides a structured approach to building Version 2 of the XML-based OFX framework, extending it to support multi-kernel and multi-execution processing. By building on the solid foundation of Version 1 and breaking the work into small, testable increments, we ensure that progress is steady and verifiable.

The plan emphasizes:
1. Enhanced XML schema for pipeline definitions
2. Robust buffer management for multi-kernel processing
3. Automatic buffer routing between pipeline steps
4. Support for multi-execution steps with intermediate buffers
5. Comprehensive testing and validation

Following this plan will result in a powerful framework that allows artists to create sophisticated multi-kernel effects simply by writing XML definitions and kernel code, without needing to understand the underlying OFX C++ infrastructure.
