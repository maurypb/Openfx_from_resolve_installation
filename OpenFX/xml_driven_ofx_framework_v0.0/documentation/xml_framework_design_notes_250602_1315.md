# XML Framework Design Notes
*Key insights and improvements discovered during BlurPlugin.cpp analysis*

## UPDATE (June 2025): Implementation Status
Most insights from this analysis have been successfully implemented:
- ✅ Dynamic parameter maps (ParameterValue system)
- ✅ Arbitrary input count support (XML-driven clips)
- ✅ Simplified kernel files (setup moved to framework)
- ✅ API naming improvements (descriptive parameter names)
- 📋 Pixel format detection (Phase 4A priority)
- 📋 CUDA boilerplate elimination (Phase 4C enhancement)

*Original analysis below remains valid for understanding design decisions*


## API Design & Naming
- **Current Issue**: Generic names like `p_Args` don't indicate content
- **Framework Solution**: Use descriptive names like `RenderContext`, `FrameRenderArgs`
- **Principle**: API names should clearly indicate purpose and contents

## Parameter Handling - Current Limitations
- **Current**: Fixed parameter order, manual function signatures
  ```cpp
  RunCudaKernel(stream, width, height, radius, quality, maskStrength, input, mask, output);
  ```
- **Framework Solution**: Dynamic parameter maps
  ```cpp
  std::map<std::string, ParameterValue> params;
  params["radius"] = 5.0f;
  params["quality"] = 8;
  RunGenericKernel(stream, width, height, params, inputBuffers, outputBuffer);
  ```
- **Benefits**: Arbitrary parameter count/names, automatic XML->kernel parameter passing

## Hard-coded Limitations to Address
1. **Pixel Format**: Currently hard-coded as `float4` (RGBA float)
   - Framework should detect format from image metadata
   - Support multiple formats dynamically

2. **Fixed Input/Output Structure**: Currently assumes source + optional mask
   - Framework should support arbitrary number of inputs from XML
   - Dynamic input naming and handling

3. **GPU Platform Selection**: Manual override methods for each platform
   - Framework should automatically select available platform
   - Unified kernel interface across CUDA/OpenCL/Metal

## Memory Management Insights
- **Current Pattern**: Manual GPU memory allocation/copying
- **Framework Opportunity**: Automate common patterns
  - Texture upload/download
  - Buffer management
  - Memory cleanup

## Race Condition Pitfalls in Legacy Code
**Real-world example**: Mask flickering bug in BlurPlugin required `cudaDeviceSynchronize()` fix
- Memory allocation timing bugs are hard to predict and reproduce
- Often work during testing but fail in production with different GPU loads
- Require expensive device-wide synchronization that eliminates stream parallelism
- Framework eliminates these by using proper RAII memory management patterns
- **Developer impact**: Effect authors shouldn't need CUDA expertise to avoid race conditions




## Object Lifecycle Understanding
- **Factory Pattern**: One factory creates multiple plugin instances
- **Processing Pattern**: Plugin creates processor per frame
- **Framework Implication**: GenericEffect must handle dynamic creation

## Parameter Types to Support
From Python/GLSL developer perspective:
- Uniform scalars (float, int, bool)
- Color values (vec3/vec4)
- Curve/animation data
- Choice parameters (enums)
- String parameters
- Resolution-dependent parameters

## Bridge Patterns Observed
Multiple layers between user input and GPU processing:
1. **Host** (provides context, calls plugin)
2. **Plugin** (manages parameters, creates processor)  
3. **Processor** (sets up GPU, calls kernel)
4. **Kernel** (actual pixel processing)

**Framework Goal**: Simplify this to:
1. **Host** calls **GenericEffect**
2. **GenericEffect** calls **XMLDefinedKernel**

## Key Developer Experience Principles
1. **Image processing artists should only touch**:
   - XML effect definitions
   - GPU kernel code (CUDA/OpenCL/Metal)

2. **Framework should automatically handle**:
   - OFX infrastructure
   - Parameter UI creation
   - Memory management  
   - Platform selection
   - Parameter passing to kernels

## Implementation Priorities
1. **Replace fixed signatures** with dynamic parameter passing
2. **Automate pixel format detection** instead of hard-coding
3. **Support arbitrary input count** from XML definitions  
4. **Unify GPU platform handling** behind single interface
5. **Improve API naming** for clarity and maintainability

## Texture and Sampling API Design Insights
- **CUDA's readMode complexity**: CUDA forces developers to choose between normalized (0-1) and native format reads via `cudaReadModeElementType` vs `cudaReadModeNormalizedFloat`
- **Framework opportunity**: Default to normalized coordinates (like GLSL) for intuitive image processing, with optional native format access for performance-critical cases
- **Developer experience**: Hide low-level texture setup complexity behind framework, provide simple unified sampling interface
- **Design principle**: Prioritize intuitive defaults (normalized 0-1 range) while allowing expert control when needed

## Kernel File Scope Insights
**Current Problem in Legacy Code**: CudaKernel.cu contains both setup and processing logic
- Memory allocation (`cudaMallocArray`, `cudaMemcpy2DToArray`)
- Texture object creation (`cudaCreateTextureObject`)
- Launch configuration (`dim3 threads/blocks`)
- AND the actual pixel processing (`__global__ GaussianBlurKernel`)

**Framework Solution**: Kernel files should contain ONLY the `__global__` function
- All GPU memory management handled by framework
- All texture setup handled by framework  
- All launch configuration calculated automatically
- Effect authors write pure pixel processing logic

**Benefit**: Effect authors never touch memory management code
- Similar to GLSL: you write pixel logic, driver handles everything else
- Framework handles the "plumbing", author focuses on the creative algorithm
- Eliminates common sources of GPU programming errors (memory leaks, incorrect setup)

**Example Transformation**:
```cpp
// Current: CudaKernel.cu has 200+ lines of setup + processing
// Framework: MyEffect.cu has ~20 lines of pure processing
__global__ void process(cudaTextureObject_t input, float* output, float radius) {
    // Only pixel processing logic here
}


## CUDA Boilerplate Elimination
This isnt a pressing issue, but here it is for completeness:
**Current Problem**: Every CUDA image processing kernel starts with identical coordinate calculation boilerplate - extracting pixel coordinates from blockIdx/threadIdx, converting to normalized UV coordinates, bounds checking, and index calculation. This repetitive code obscures the actual image processing logic and must be rewritten for every effect.

**Framework Solution**: Generate wrapper kernels automatically that handle coordinate calculation and call user-defined processing functions with pre-calculated values. This allows effect authors to write pure image processing logic with optional helper functions, similar to GLSL fragment shaders where coordinate handling is automatic.

**Example Transformation**:
```cpp
// Current: Every kernel starts with 8+ lines of coordinate boilerplate
__global__ void MyKernel(int width, int height, ...) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    // ... 6 more lines of setup
    
    // Finally: actual image processing logic
}

// Framework: Pure processing logic only
__device__ void process(int x, int y, float2 uv, int index, ...) {
    // Only image processing logic here
}



```

## Questions for Future Implementation
- How to handle parameter animation (getValueAtTime)?
- How to support different pixel formats in kernels?
- How to maintain performance with dynamic parameter systems?
- How to debug GPU kernels when parameters are passed dynamically?
- Should framework default to normalized texture coordinates (0-1) or pixel coordinates for sampling?

---
*These notes captured during BlurPlugin.cpp analysis session - integrate into formal specification and implementation plan*