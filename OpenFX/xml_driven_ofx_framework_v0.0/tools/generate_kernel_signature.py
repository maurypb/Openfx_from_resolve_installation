#!/usr/bin/env python3
"""
Kernel Signature Generator for XML-driven OFX Framework

Reads XML effect definitions and generates CUDA kernel function signatures
with standardized wrapper interface for the framework.
"""

import xml.etree.ElementTree as ET
import sys
import os

def validate_xml_syntax(xml_file):
    """Check if XML file has valid syntax (well-formed)"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        print("XML syntax valid")
        return tree, root
    except ET.ParseError as e:
        print("✗ XML syntax error: " + str(e))
        return None, None
    except FileNotFoundError:
        print("✗ File not found: " + xml_file)
        return None, None

def validate_xml_structure(root):
    """Check if XML has required structure and valid content"""
    errors = []
    
    # Check root element
    if root.tag != 'effect':
        errors.append("Root element must be 'effect'")
        return errors  # Can't continue without proper root
    
    # Check required attributes
    if not root.get('name'):
        errors.append("Effect missing 'name' attribute")
    if not root.get('category'):
        errors.append("Effect missing 'category' attribute")
    
    # Check required sections exist
    inputs_section = root.find('inputs')
    if inputs_section is None:
        errors.append("Missing 'inputs' section")
    
    params_section = root.find('parameters')
    if params_section is None:
        errors.append("Missing 'parameters' section")
        return errors  # Can't generate signature without parameters
    
    kernels_section = root.find('kernels')
    pipeline_section = root.find('pipeline')
    if kernels_section is None and pipeline_section is None:
        errors.append("Missing 'kernels' or 'pipeline' section")
    
    # Validate parameters
    valid_param_types = ['double', 'float', 'int', 'bool', 'string', 'choice', 'color', 'vec2', 'vec3', 'vec4', 'curve']
    
    for param in params_section.findall('parameter'):
        param_name = param.get('name')
        param_type = param.get('type')
        
        if not param_name:
            errors.append("Parameter missing 'name' attribute")
        elif len(param_name.strip()) == 0:
            errors.append("Parameter has empty 'name' attribute")
        
        if not param_type:
            errors.append("Parameter '" + param_name + "' missing 'type' attribute")
        elif param_type not in valid_param_types:
            errors.append("Parameter '" + param_name + "' has invalid type '" + param_type + "'. Valid types: " + ', '.join(valid_param_types))
        
        # Check for default value
        if not param.get('default'):
            errors.append("Parameter '" + param_name + "' missing 'default' attribute")
    
    # Validate inputs
    if inputs_section is not None:
        for source in inputs_section.findall('source'):
            source_name = source.get('name')
            if not source_name:
                errors.append("Input source missing 'name' attribute")
            
            border_mode = source.get('border_mode', 'clamp')
            valid_border_modes = ['clamp', 'repeat', 'mirror', 'black']
            if border_mode not in valid_border_modes:
                errors.append("Input '" + source_name + "' has invalid border_mode '" + border_mode + "'. Valid modes: " + ', '.join(valid_border_modes))
    
    return errors

def generate_cuda_skeleton(root):
    """Generate complete CUDA file skeleton with standardized wrapper"""
    effect_name = root.get('name', 'UnknownEffect')
    
    code_lines = []
    code_lines.append("// " + effect_name + ".cu - Generated skeleton")
    code_lines.append("// Fill in the image processing logic in the kernel function")
    code_lines.append("")
    code_lines.append("#include <cuda_runtime.h>")
    code_lines.append("#include <cmath>")
    code_lines.append("#include <cstdint>  // For uintptr_t")
    code_lines.append("")
    
    # Generate the __global__ kernel signature with XML-driven parameters
    code_lines.append("// Image processing kernel - implement your algorithm here")
    code_lines.append("__global__ void " + effect_name + "Kernel(")
    
    # Build parameter list for the actual kernel
    params = []
    params.append("int width")
    params.append("int height")
    
    # Add XML-defined inputs
    inputs_section = root.find('inputs')
    if inputs_section is not None:
        for input_def in inputs_section.findall('source'):
            input_name = input_def.get('name')
            is_optional = input_def.get('optional', 'false').lower() == 'true'
            
            params.append("cudaTextureObject_t " + input_name + "Tex")
            
            if is_optional:
                params.append("bool " + input_name + "Present")
    
    params.append("float* output")
    
    # Add XML-defined parameters
    params_section = root.find('parameters')
    if params_section is not None:
        for param in params_section.findall('parameter'):
            param_name = param.get('name')
            param_type = param.get('type')
            
            if param_type in ['double', 'float']:
                c_type = 'float'
            elif param_type == 'int':
                c_type = 'int'
            elif param_type == 'bool':
                c_type = 'bool'
            else:
                c_type = 'float'
                
            params.append(c_type + " " + param_name)
    
    # Write kernel parameters
    for i, param in enumerate(params):
        comma = "," if i < len(params) - 1 else ""
        code_lines.append("    " + param + comma)
    
    code_lines.append(")")
    code_lines.append("{")
    code_lines.append("    // Standard CUDA coordinate calculation")
    code_lines.append("    const int x = blockIdx.x * blockDim.x + threadIdx.x;")
    code_lines.append("    const int y = blockIdx.y * blockDim.y + threadIdx.y;")
    code_lines.append("")
    code_lines.append("    if ((x < width) && (y < height)) {")
    code_lines.append("        // Normalize coordinates to [0,1] range for texture sampling")
    code_lines.append("        float u = (x + 0.5f) / width;")
    code_lines.append("        float v = (y + 0.5f) / height;")
    code_lines.append("        ")
    code_lines.append("        // Calculate output array index")
    code_lines.append("        const int index = ((y * width) + x) * 4;")
    code_lines.append("")
    code_lines.append("        // TODO: Implement your image processing algorithm here")
    code_lines.append("        // Sample from input textures:")
    
    # Generate texture sampling examples
    if inputs_section is not None:
        for input_def in inputs_section.findall('source'):
            input_name = input_def.get('name')
            is_optional = input_def.get('optional', 'false').lower() == 'true'
            
            if is_optional:
                code_lines.append("        // if (" + input_name + "Present) {")
                code_lines.append("        //     float4 " + input_name.lower() + "Color = tex2D<float4>(" + input_name + "Tex, u, v);")
                code_lines.append("        // }")
            else:
                code_lines.append("        // float4 " + input_name.lower() + "Color = tex2D<float4>(" + input_name + "Tex, u, v);")
    
    code_lines.append("")
    code_lines.append("        // Write to output")
    code_lines.append("        // output[index + 0] = result.x;  // Red")
    code_lines.append("        // output[index + 1] = result.y;  // Green") 
    code_lines.append("        // output[index + 2] = result.z;  // Blue")
    code_lines.append("        // output[index + 3] = result.w;  // Alpha")
    code_lines.append("    }")
    code_lines.append("}")
    code_lines.append("")
    
    # Generate the standardized wrapper function
    code_lines.append("// Standardized wrapper function - framework calls this")
    code_lines.append("extern \"C\" void call_" + effect_name.lower() + "_kernel(")
    code_lines.append("    void* stream,")
    code_lines.append("    int width,")
    code_lines.append("    int height,")
    code_lines.append("    void** textures,")
    code_lines.append("    int textureCount,")
    code_lines.append("    bool* presenceFlags,")
    code_lines.append("    float* output,")
    code_lines.append("    float* floatParams,")
    code_lines.append("    int* intParams,")
    code_lines.append("    bool* boolParams")
    code_lines.append(") {")
    code_lines.append("    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);")
    code_lines.append("")
    code_lines.append("    // Unpack texture objects")
    
    # Generate texture unpacking
    tex_index = 0
    if inputs_section is not None:
        for input_def in inputs_section.findall('source'):
            input_name = input_def.get('name')
            code_lines.append("    cudaTextureObject_t " + input_name + "Tex = (cudaTextureObject_t)(uintptr_t)textures[" + str(tex_index) + "];")
            tex_index += 1
    
    code_lines.append("")
    code_lines.append("    // Unpack parameters")
    
    # Generate parameter unpacking
    float_index = 0
    int_index = 0
    bool_index = 0
    
    if params_section is not None:
        for param in params_section.findall('parameter'):
            param_name = param.get('name')
            param_type = param.get('type')
            
            if param_type in ['double', 'float']:
                code_lines.append("    float " + param_name + " = floatParams[" + str(float_index) + "];")
                float_index += 1
            elif param_type == 'int':
                code_lines.append("    int " + param_name + " = intParams[" + str(int_index) + "];")
                int_index += 1
            elif param_type == 'bool':
                code_lines.append("    bool " + param_name + " = boolParams[" + str(bool_index) + "];")
                bool_index += 1
            else:
                code_lines.append("    float " + param_name + " = floatParams[" + str(float_index) + "];  // " + param_type + " -> float")
                float_index += 1
    
    code_lines.append("")
    code_lines.append("    // Launch configuration")
    code_lines.append("    dim3 threads(16, 16, 1);")
    code_lines.append("    dim3 blocks(((width + threads.x - 1) / threads.x), ((height + threads.y - 1) / threads.y), 1);")
    code_lines.append("")
    code_lines.append("    // Launch the kernel")
    code_lines.append("    " + effect_name + "Kernel<<<blocks, threads, 0, cudaStream>>>(")
    
    # Generate kernel call arguments
    call_args = ["width", "height"]
    
    # Add texture arguments
    if inputs_section is not None:
        for input_def in inputs_section.findall('source'):
            input_name = input_def.get('name')
            is_optional = input_def.get('optional', 'false').lower() == 'true'
            
            call_args.append(input_name + "Tex")
            if is_optional:
                presence_index = len([i for i in inputs_section.findall('source')[:inputs_section.findall('source').index(input_def)] if i.get('optional', 'false').lower() == 'true'])
                call_args.append("presenceFlags[" + str(presence_index) + "]")
    
    call_args.append("output")
    
    # Add parameter arguments
    if params_section is not None:
        for param in params_section.findall('parameter'):
            param_name = param.get('name')
            call_args.append(param_name)
    
    # Write kernel call arguments
    for i, arg in enumerate(call_args):
        comma = "," if i < len(call_args) - 1 else ""
        code_lines.append("        " + arg + comma)
    
    code_lines.append("    );")
    code_lines.append("}")
    
    return "\n".join(code_lines)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 generate_kernel_signature.py <xml_file>")
        print("Example: python3 generate_kernel_signature.py TestBlurV2.xml")
        sys.exit(1)
    
    xml_file = sys.argv[1]
    
    print("Generating standardized kernel template from: " + xml_file)
    print("=" * 50)
    
    # Step 1: Validate XML syntax
    tree, root = validate_xml_syntax(xml_file)
    if tree is None:
        sys.exit(1)
    
    # Step 2: Validate XML structure
    errors = validate_xml_structure(root)
    if errors:
        print("✗ XML structure validation failed:")
        for error in errors:
            print("  - " + error)
        sys.exit(1)
    
    print("XML structure validation passed")
    print()
    
    # Step 3: Generate kernel skeleton
    print("Generated CUDA Kernel Template:")
    print("-" * 40)
    skeleton = generate_cuda_skeleton(root)
    print(skeleton)
    print()
    
    print("Generation complete!")
    
    # Write skeleton to file
    base_name = os.path.splitext(xml_file)[0]
    
    with open(base_name + "_template.cu", 'w') as f:
        f.write(skeleton)
    print("CUDA template written to: " + base_name + "_template.cu")

if __name__ == "__main__":
    main()