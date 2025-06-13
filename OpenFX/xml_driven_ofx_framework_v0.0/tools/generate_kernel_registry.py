#!/usr/bin/env python3
"""
Kernel Registry Generator for XML-driven OFX Framework

Scans the effects/ folder for XML files and generates a C++ registry
of all kernel functions for dynamic dispatch.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_xml_inputs_and_params(xml_file):
    """Parse XML file and extract inputs and parameters for signature generation"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        if root.tag != 'effect':
            return None, None
        
        # Parse inputs
        inputs = []
        inputs_section = root.find('inputs')
        if inputs_section is not None:
            for source in inputs_section.findall('source'):
                input_name = source.get('name')
                is_optional = source.get('optional', 'false').lower() == 'true'
                if input_name:
                    inputs.append({
                        'name': input_name,
                        'optional': is_optional
                    })
        
        # Parse parameters
        params = []
        params_section = root.find('parameters')
        if params_section is not None:
            for param in params_section.findall('parameter'):
                param_name = param.get('name')
                param_type = param.get('type')
                if param_name and param_type:
                    params.append({
                        'name': param_name,
                        'type': param_type
                    })
        
        return inputs, params
        
    except Exception as e:
        print("  ERROR: Failed to parse " + str(xml_file) + ": " + str(e))
        return None, None

def map_xml_type_to_c(xml_type):
    """Map XML parameter types to C types for function signatures"""
    type_mapping = {
        'double': 'float',
        'float': 'float', 
        'int': 'int',
        'bool': 'bool',
        'string': 'const char*',
        'choice': 'int',
        'color': 'float',  # Simplified to float for registry
        'vec2': 'float',   # Simplified to float for registry  
        'vec3': 'float',   # Simplified to float for registry
        'vec4': 'float',   # Simplified to float for registry
        'curve': 'float'
    }
    return type_mapping.get(xml_type, 'float')

def generate_function_signature(effect_name, inputs, params):
    """Generate function signature for a specific effect"""
    sig_parts = []
    
    # Fixed parameters
    sig_parts.append("void* stream")
    sig_parts.append("int width") 
    sig_parts.append("int height")
    
    # Dynamic inputs from XML
    for input_info in inputs:
        input_name = input_info['name']
        is_optional = input_info['optional']
        
        # Add texture parameter
        tex_param = "cudaTextureObject_t " + input_name + "Tex"
        sig_parts.append(tex_param)
        
        # Add presence boolean only for optional inputs
        if is_optional:
            present_param = "bool " + input_name + "Present"
            sig_parts.append(present_param)
    
    # Output buffer
    sig_parts.append("float* output")
    
    # Dynamic parameters from XML
    for param_info in params:
        param_name = param_info['name']
        param_type = param_info['type']
        c_type = map_xml_type_to_c(param_type)
        param_decl = c_type + " " + param_name
        sig_parts.append(param_decl)
    
    return sig_parts

def scan_effects_folder(effects_dir="effects"):
    """Scan effects folder for XML files and extract effect information"""
    effects = []
    
    if not os.path.exists(effects_dir):
        print("ERROR: Effects directory '" + effects_dir + "' not found")
        return effects
    
    for xml_file in Path(effects_dir).glob("*.xml"):
        print("Processing: " + str(xml_file))
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            if root.tag != 'effect':
                print("  WARNING: Skipping " + str(xml_file) + " - not an effect definition")
                continue
            
            effect_name = root.get('name')
            if not effect_name:
                print("  WARNING: Skipping " + str(xml_file) + " - no effect name")
                continue
            
            # Check for CUDA kernel
            cuda_kernel = None
            kernels_section = root.find('kernels')
            if kernels_section is not None:
                cuda_node = kernels_section.find('cuda')
                if cuda_node is not None:
                    cuda_kernel = cuda_node.get('file')
            
            if cuda_kernel:
                # Parse inputs and parameters for this effect
                inputs, params = parse_xml_inputs_and_params(xml_file)
                if inputs is not None and params is not None:
                    function_name = "call_" + effect_name.lower() + "_kernel"
                    
                    effects.append({
                        'name': effect_name,
                        'xml_file': str(xml_file),
                        'cuda_file': cuda_kernel,
                        'function_name': function_name,
                        'inputs': inputs,
                        'params': params
                    })
                    print("  Found effect: " + effect_name + " -> " + function_name)
                    input_names = [i['name'] for i in inputs]
                    param_names = [p['name'] for p in params]
                    print("    Inputs: " + str(len(inputs)) + " (" + ', '.join(input_names) + ")")
                    print("    Parameters: " + str(len(params)) + " (" + ', '.join(param_names) + ")")
                else:
                    print("  WARNING: Failed to parse inputs/parameters for " + effect_name)
            else:
                print("  WARNING: No CUDA kernel found for " + effect_name)
                
        except ET.ParseError as e:
            print("  ERROR: XML parse error in " + str(xml_file) + ": " + str(e))
        except Exception as e:
            print("  ERROR: Failed to process " + str(xml_file) + ": " + str(e))
    
    return effects

def generate_registry_header(effects):
    """Generate KernelRegistry.h with standardized function type"""
    header_content = """#ifndef KERNEL_REGISTRY_H
#define KERNEL_REGISTRY_H

#include <string>

/**
 * @brief Auto-generated kernel registry for XML-driven OFX framework
 * 
 * This file is generated by generate_kernel_registry.py
 * DO NOT EDIT MANUALLY - changes will be overwritten
 */

// Standardized kernel function type - all kernels use this signature
typedef void (*KernelFunction)(
    void* stream, int width, int height,
    void** textures, int textureCount, bool* presenceFlags,
    float* output,
    float* floatParams, int* intParams, bool* boolParams
);

/**
 * @brief Get kernel function by effect name
 * @param effectName Name from XML effect definition
 * @return Function pointer or nullptr if not found
 */
KernelFunction getKernelFunction(const std::string& effectName);

/**
 * @brief Get number of registered effects
 * @return Number of effects in registry
 */
int getRegisteredEffectCount();

/**
 * @brief Get effect name by index
 * @param index Index in registry
 * @return Effect name or empty string if invalid index
 */
std::string getEffectName(int index);

#endif // KERNEL_REGISTRY_H
"""
    return header_content

def generate_registry_implementation(effects):
    """Generate KernelRegistry.cpp with standardized signatures"""
    
    # Generate forward declarations with standardized signatures
    forward_decls = []
    for effect in effects:
        func_decl = "extern \"C\" void " + effect['function_name'] + "("
        forward_decls.append(func_decl)
        forward_decls.append("    void* stream,")
        forward_decls.append("    int width,")
        forward_decls.append("    int height,")
        forward_decls.append("    void** textures,")
        forward_decls.append("    int textureCount,")
        forward_decls.append("    bool* presenceFlags,")
        forward_decls.append("    float* output,")
        forward_decls.append("    float* floatParams,")
        forward_decls.append("    int* intParams,")
        forward_decls.append("    bool* boolParams")
        forward_decls.append(");")
        forward_decls.append("")
    
    # Generate registry table
    registry_entries = []
    for effect in effects:
        entry = "    { \"" + effect['name'] + "\", (KernelFunction)" + effect['function_name'] + " },"
        registry_entries.append(entry)
    
    # Generate effect names array
    effect_names = []
    for effect in effects:
        name_entry = "    \"" + effect['name'] + "\","
        effect_names.append(name_entry)
    
    # Build implementation content using simple string concatenation
    impl_content = "#include \"KernelRegistry.h\"\n"
    impl_content += "#include <map>\n"
    impl_content += "#include <vector>\n\n"
    impl_content += "/**\n"
    impl_content += " * @brief Auto-generated kernel registry for XML-driven OFX framework\n"
    impl_content += " * \n"
    impl_content += " * Generated from effects folder containing " + str(len(effects)) + " effect(s):\n"
    
    for effect in effects:
        impl_content += " * - " + effect['name'] + " (" + effect['xml_file'] + ")\n"
        impl_content += " *   Inputs: " + str(len(effect['inputs'])) + " "
        input_list = ', '.join([i['name'] + ('*' if i['optional'] else '') for i in effect['inputs']])
        impl_content += "(" + input_list + ")\n"
        impl_content += " *   Parameters: " + str(len(effect['params'])) + " "
        param_list = ', '.join([p['name'] + "(" + p['type'] + ")" for p in effect['params']])
        impl_content += "(" + param_list + ")\n"
    
    impl_content += " * \n"
    impl_content += " * This file is generated by generate_kernel_registry.py\n"
    impl_content += " * DO NOT EDIT MANUALLY - changes will be overwritten\n"
    impl_content += " * \n"
    impl_content += " * Legend: * = optional input\n"
    impl_content += " */\n\n"
    impl_content += "// Forward declarations for all kernel functions\n"
    impl_content += "\n".join(forward_decls) + "\n\n"
    impl_content += "// Registry structure\n"
    impl_content += "struct KernelEntry {\n"
    impl_content += "    const char* effectName;\n"
    impl_content += "    KernelFunction function;\n"
    impl_content += "};\n\n"
    impl_content += "// Auto-generated registry table\n"
    impl_content += "static const KernelEntry kernelRegistry[] = {\n"
    impl_content += "\n".join(registry_entries) + "\n"
    impl_content += "};\n\n"
    impl_content += "static const int registrySize = " + str(len(effects)) + ";\n\n"
    impl_content += "// Effect names for index lookup\n"
    impl_content += "static const char* effectNames[] = {\n"
    impl_content += "\n".join(effect_names) + "\n"
    impl_content += "};\n\n"
    impl_content += "KernelFunction getKernelFunction(const std::string& effectName) {\n"
    impl_content += "    for (int i = 0; i < registrySize; ++i) {\n"
    impl_content += "        if (effectName == kernelRegistry[i].effectName) {\n"
    impl_content += "            return kernelRegistry[i].function;\n"
    impl_content += "        }\n"
    impl_content += "    }\n"
    impl_content += "    return nullptr;\n"
    impl_content += "}\n\n"
    impl_content += "int getRegisteredEffectCount() {\n"
    impl_content += "    return registrySize;\n"
    impl_content += "}\n\n"
    impl_content += "std::string getEffectName(int index) {\n"
    impl_content += "    if (index >= 0 && index < registrySize) {\n"
    impl_content += "        return std::string(effectNames[index]);\n"
    impl_content += "    }\n"
    impl_content += "    return std::string();\n"
    impl_content += "}\n"
    
    return impl_content

def main():
    print("Kernel Registry Generator")
    print("=" * 50)
    
    # Scan effects folder
    effects = scan_effects_folder()
    
    if not effects:
        print("No effects found - registry will be empty")
        return 1
    
    print("\nFound " + str(len(effects)) + " effect(s)")
    
    # Generate header file
    header_content = generate_registry_header(effects)
    with open("src/core/KernelRegistry.h", 'w') as f:
        f.write(header_content)
    print("Generated: src/core/KernelRegistry.h")
    
    # Generate implementation file
    impl_content = generate_registry_implementation(effects)
    with open("src/core/KernelRegistry.cpp", 'w') as f:
        f.write(impl_content)
    print("Generated: src/core/KernelRegistry.cpp")
    
    print("\nRegistry generation complete!")
    print("Add KernelRegistry.o to your Makefile to build the registry.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())