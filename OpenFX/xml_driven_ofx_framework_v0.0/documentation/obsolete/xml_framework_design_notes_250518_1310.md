# XML Framework Design Notes
*Key insights and improvements discovered during BlurPlugin.cpp analysis*

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

## Questions for Future Implementation
- How to handle parameter animation (getValueAtTime)?
- How to support different pixel formats in kernels?
- How to maintain performance with dynamic parameter systems?
- How to debug GPU kernels when parameters are passed dynamically?

---
*These notes captured during BlurPlugin.cpp analysis session - integrate into formal specification and implementation plan*
