#!/usr/bin/env python3
"""
Kernel Signature Generator for XML-driven OFX Framework

Reads XML effect definitions and generates CUDA kernel function signatures
with parameters in the correct order for the framework to call.
"""

import xml.etree.ElementTree as ET
import sys
import os

def validate_xml_syntax(xml_file):
    """Check if XML file has valid syntax (well-formed)"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        print(f"✓ XML syntax valid")
        return tree, root
    except ET.ParseError as e:
        print(f"✗ XML syntax error: {e}")
        return None, None
    except FileNotFoundError:
        print(f"✗ File not found: {xml_file}")
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
            errors.append(f"Parameter '{param_name}' missing 'type' attribute")
        elif param_type not in valid_param_types:
            errors.append(f"Parameter '{param_name}' has invalid type '{param_type}'. Valid types: {', '.join(valid_param_types)}")
        
        # Check for default value
        if not param.get('default'):
            errors.append(f"Parameter '{param_name}' missing 'default' attribute")
    
    # Validate inputs
    if inputs_section is not None:
        for source in inputs_section.findall('source'):
            source_name = source.get('name')
            if not source_name:
                errors.append("Input source missing 'name' attribute")
            
            border_mode = source.get('border_mode', 'clamp')
            valid_border_modes = ['clamp', 'repeat', 'mirror', 'black']
            if border_mode not in valid_border_modes:
                errors.append(f"Input '{source_name}' has invalid border_mode '{border_mode}'. Valid modes: {', '.join(valid_border_modes)}")
    
    return errors

def map_xml_type_to_cuda(xml_type):
    """Map XML parameter types to CUDA types"""
    type_mapping = {
        'double': 'float',
        'float': 'float', 
        'int': 'int',
        'bool': 'bool',
        'string': 'const char*',  # Note: strings need special handling
        'choice': 'int',
        'color': 'float3',        # RGB color
        'vec2': 'float2',
        'vec3': 'float3',
        'vec4': 'float4',
        'curve': 'float'          # Simplified for now
    }
    return type_mapping.get(xml_type, 'float')  # Default to float

def generate_kernel_signature(root):
    """Generate CUDA kernel function signature from XML - completely XML-driven"""
    effect_name = root.get('name', 'UnknownEffect')
    
    # Build complete parameter list - only from XML, nothing hard-coded
    all_params = []
    
    # Fixed parameters (always present in framework)
    all_params.append(("int", "width", ""))
    all_params.append(("int", "height", ""))
    
    # Add XML-defined inputs dynamically - no assumptions about names or count
    inputs_section = root.find('inputs')
    if inputs_section is not None:
        for input_def in inputs_section.findall('source'):
            input_name = input_def.get('name')
            is_optional = input_def.get('optional', 'false').lower() == 'true'
            border_mode = input_def.get('border_mode', 'clamp')
            label = input_def.get('label', input_name)
            
            # Add texture object parameter
            comment = f"// from <source name=\"{input_name}\" optional=\"{is_optional}\" border_mode=\"{border_mode}\">"
            all_params.append(("cudaTextureObject_t", f"{input_name}Tex", comment))
            
            # Add boolean for optional inputs only
            if is_optional:
                all_params.append(("bool", f"{input_name}Present", f"// whether {input_name} is connected"))
    
    # Add output (always present in framework)
    all_params.append(("float*", "output", ""))
    
    # Add XML-defined parameters - completely driven by XML structure
    params_section = root.find('parameters')
    if params_section is not None:
        for param in params_section.findall('parameter'):
            param_name = param.get('name')
            param_type = param.get('type')
            default_val = param.get('default', 'N/A')
            cuda_type = map_xml_type_to_cuda(param_type)
            comment = f"// from <parameter name=\"{param_name}\" type=\"{param_type}\" default=\"{default_val}\">"
            all_params.append((cuda_type, param_name, comment))
    
    # Start building signature
    signature_lines = []
    signature_lines.append(f"// Generated from {effect_name}.xml")
    signature_lines.append(f"__global__ void {effect_name}Kernel(")
    
    # Generate parameters with correct comma placement
    for i, (param_type, param_name, comment) in enumerate(all_params):
        comma = "," if i < len(all_params) - 1 else ""
        if comment:
            signature_lines.append(f"    {param_type} {param_name}{comma}  {comment}")
        else:
            signature_lines.append(f"    {param_type} {param_name}{comma}")
    
    signature_lines.append(");")
    
    return "\n".join(signature_lines)

def generate_cuda_skeleton(root):
    """Generate complete CUDA file skeleton for the kernel author"""
    effect_name = root.get('name', 'UnknownEffect')
    
    code_lines = []
    code_lines.append(f"// {effect_name}.cu - Generated skeleton")
    code_lines.append(f"// Fill in the image processing logic in the kernel function")
    code_lines.append("")
    code_lines.append("#include <cuda_runtime.h>")
    code_lines.append("#include <cmath>")
    code_lines.append("")
    
    # Generate the __global__ kernel signature
    code_lines.append("// Image processing kernel - implement your algorithm here")
    kernel_sig = generate_kernel_signature(root)
    
    # Process the signature lines properly
    kernel_lines = kernel_sig.split('\n')
    inside_signature = False
    
    for line in kernel_lines:
        if line.startswith('//') and 'Generated from' in line:
            continue  # Skip the comment line
        elif line.startswith('__global__'):
            inside_signature = True
            code_lines.append(line)
        elif inside_signature and line.endswith(');'):
            # This is the closing line - replace ); with ) {
            code_lines.append(line.replace(');', ')'))
            code_lines.append('{')
            break
        elif inside_signature:
            code_lines.append(line)
    
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
    
    # Generate texture sampling examples for each input defined in XML
    inputs_section = root.find('inputs')
    if inputs_section is not None:
        for input_def in inputs_section.findall('source'):
            input_name = input_def.get('name')
            is_optional = input_def.get('optional', 'false').lower() == 'true'
            
            if is_optional:
                code_lines.append(f"        // if ({input_name}Present) {{")
                code_lines.append(f"        //     float4 {input_name.lower()}Color = tex2D<float4>({input_name}Tex, u, v);")
                code_lines.append(f"        // }}")
            else:
                code_lines.append(f"        // float4 {input_name.lower()}Color = tex2D<float4>({input_name}Tex, u, v);")
    
    code_lines.append("")
    code_lines.append("        // Write to output")
    code_lines.append("        // output[index + 0] = result.x;  // Red")
    code_lines.append("        // output[index + 1] = result.y;  // Green") 
    code_lines.append("        // output[index + 2] = result.z;  // Blue")
    code_lines.append("        // output[index + 3] = result.w;  // Alpha")
    code_lines.append("    }")
    code_lines.append("}")  # Close the __global__ function
    code_lines.append("")
    
    # Generate the bridge function - must match kernel parameter order exactly
    code_lines.append("// Bridge function - connects framework to your kernel")
    code_lines.append(f"extern \"C\" void call_{effect_name.lower()}_kernel(")
    code_lines.append("    void* stream, int width, int height,")
    
    # Build bridge function parameters to match kernel order
    bridge_params = []
    
    # Add input textures (matching the kernel's dynamic input order)
    inputs_section = root.find('inputs')
    if inputs_section is not None:
        for input_def in inputs_section.findall('source'):
            input_name = input_def.get('name')
            bridge_params.append(f"cudaTextureObject_t {input_name}Tex")
            
            is_optional = input_def.get('optional', 'false').lower() == 'true'
            if is_optional:
                bridge_params.append(f"bool {input_name}Present")
    
    # Add output
    bridge_params.append("float* output")
    
    # Add XML parameters in same order as kernel
    params_section = root.find('parameters')
    if params_section is not None:
        parameters = params_section.findall('parameter')
        for param in parameters:
            param_name = param.get('name')
            param_type = param.get('type')
            cuda_type = map_xml_type_to_cuda(param_type)
            bridge_params.append(f"{cuda_type} {param_name}")
    
    # Write bridge function parameters
    for i, param in enumerate(bridge_params):
        comma = "," if i < len(bridge_params) - 1 else ""
        code_lines.append(f"    {param}{comma}")
    
    code_lines.append(") {")
    code_lines.append("    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);")
    code_lines.append("")
    code_lines.append("    // Launch configuration")
    code_lines.append("    dim3 threads(16, 16, 1);")
    code_lines.append("    dim3 blocks(((width + threads.x - 1) / threads.x), ((height + threads.y - 1) / threads.y), 1);")
    code_lines.append("")
    code_lines.append("    // Launch the kernel")
    code_lines.append(f"    {effect_name}Kernel<<<blocks, threads, 0, cudaStream>>>(")
    
    # Build kernel call arguments in exact same order as kernel signature
    kernel_args = ["width", "height"]
    
    # Add input textures in same order
    if inputs_section is not None:
        for input_def in inputs_section.findall('source'):
            input_name = input_def.get('name')
            kernel_args.append(f"{input_name}Tex")
            
            is_optional = input_def.get('optional', 'false').lower() == 'true'
            if is_optional:
                kernel_args.append(f"{input_name}Present")
    
    # Add output
    kernel_args.append("output")
    
    # Add XML parameters
    if params_section is not None:
        parameters = params_section.findall('parameter')
        for param in parameters:
            param_name = param.get('name')
            kernel_args.append(param_name)
    
    # Write kernel call arguments
    for i, arg in enumerate(kernel_args):
        comma = "," if i < len(kernel_args) - 1 else ""
        code_lines.append(f"        {arg}{comma}")
    
    code_lines.append("    );")
    code_lines.append("}")
    
    return "\n".join(code_lines)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 generate_kernel_signature.py <xml_file>")
        print("Example: python3 generate_kernel_signature.py TestBlurV2.xml")
        sys.exit(1)
    
    xml_file = sys.argv[1]
    
    print(f"Generating kernel signature from: {xml_file}")
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
            print(f"  - {error}")
        sys.exit(1)
    
    print("✓ XML structure validation passed")
    print()
    
    # Step 3: Generate kernel skeleton
    print("Generated CUDA Kernel Skeleton:")
    print("-" * 40)
    skeleton = generate_cuda_skeleton(root)
    print(skeleton)
    print()
    
    print("✓ Generation complete!")
    
    # Write skeleton to file
    base_name = os.path.splitext(xml_file)[0]
    
    with open(f"{base_name}_template.cu", 'w') as f:
        f.write(skeleton)
    print(f"✓ CUDA template written to: {base_name}_template.cu")
    
    # Also generate just the signature for reference
    signature = generate_kernel_signature(root)
    with open(f"{base_name}_signature.txt", 'w') as f:
        f.write(signature)
    print(f"✓ Signature reference written to: {base_name}_signature.txt")

if __name__ == "__main__":
    main()