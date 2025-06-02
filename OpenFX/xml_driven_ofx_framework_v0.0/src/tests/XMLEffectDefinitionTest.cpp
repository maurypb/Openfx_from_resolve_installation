#include "../core/XMLEffectDefinition.h"
#include <cassert>
#include <iostream>
#include <fstream>

// Simple test program for XMLEffectDefinition
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <xml-file>" << std::endl;
        return 1;
    }
    
    try {
        // Parse XML file
        XMLEffectDefinition xmlDef(argv[1]);
        
        // Print basic information
        std::cout << "Effect: " << xmlDef.getName() << std::endl;
        std::cout << "Category: " << xmlDef.getCategory() << std::endl;
        std::cout << "Description: " << xmlDef.getDescription() << std::endl;
        std::cout << "Version: " << xmlDef.getVersion() << std::endl;
        std::cout << "Author: " << xmlDef.getAuthor() << std::endl;
        std::cout << "Copyright: " << xmlDef.getCopyright() << std::endl;
        std::cout << "Supports Timeline: " << (xmlDef.supportsTimeline() ? "Yes" : "No") << std::endl;
        std::cout << "Supports Matte: " << (xmlDef.supportsMatte() ? "Yes" : "No") << std::endl;
        
        // Print inputs
        std::cout << "\nInputs:" << std::endl;
        for (const auto& input : xmlDef.getInputs()) {
            std::cout << "  " << input.name << " (" << input.label << ")" 
                      << (input.optional ? " [optional]" : "")
                      << " - Border Mode: " << input.borderMode << std::endl;
        }
        
        // Print parameters
        std::cout << "\nParameters:" << std::endl;
        for (const auto& param : xmlDef.getParameters()) {
            std::cout << "  " << param.name << " (" << param.label << ") - Type: " << param.type << std::endl;
            std::cout << "    Hint: " << param.hint << std::endl;
            
            if (param.type == "double" || param.type == "int" || param.type == "float") {
                std::cout << "    Default: " << param.defaultValue << std::endl;
                std::cout << "    Range: " << param.minValue << " to " << param.maxValue << std::endl;
                std::cout << "    Display Range: " << param.displayMin << " to " << param.displayMax << std::endl;
                std::cout << "    Increment: " << param.inc << std::endl;
                std::cout << "    Resolution Dependent: " << param.resDependent << std::endl;
            } 
            else if (param.type == "bool") {
                std::cout << "    Default: " << (param.defaultBool ? "true" : "false") << std::endl;
            } 
            else if (param.type == "string") {
                std::cout << "    Default: \"" << param.defaultString << "\"" << std::endl;
            } 
            else if (param.type == "curve") {
                std::cout << "    Default Shape: " << param.defaultShape << std::endl;
                std::cout << "    Curve Background: " << param.curveBackground << std::endl;
            } 
            else if (param.type == "choice") {
                std::cout << "    Default: " << param.defaultValue << std::endl;
                std::cout << "    Options:" << std::endl;
                for (const auto& option : param.options) {
                    std::cout << "      " << option.value << ": " << option.label << std::endl;
                }
            } 
            else if (param.type == "color" || param.type == "vec2" || param.type == "vec3" || param.type == "vec4") {
                std::cout << "    Components:" << std::endl;
                for (const auto& comp : param.components) {
                    std::cout << "      " << comp.name << ": Default=" << comp.defaultValue
                              << ", Range=" << comp.minValue << " to " << comp.maxValue
                              << ", Increment=" << comp.inc << std::endl;
                }
            }
        }
        
        // Print UI organization
        std::cout << "\nUI Organization:" << std::endl;
        for (const auto& page : xmlDef.getUIPages()) {
            std::cout << "  Page: " << page.name;
            if (!page.tooltip.empty()) {
                std::cout << " - " << page.tooltip;
            }
            std::cout << std::endl;
            
            for (const auto& column : page.columns) {
                std::cout << "    Column: " << column.name;
                if (!column.tooltip.empty()) {
                    std::cout << " - " << column.tooltip;
                }
                std::cout << std::endl;
                
                for (const auto& param : column.parameters) {
                    std::cout << "      Parameter: " << param.name << std::endl;
                }
            }
        }
        
        // Print kernel information
        if (xmlDef.hasPipeline()) {
            std::cout << "\nPipeline Steps:" << std::endl;
            for (const auto& step : xmlDef.getPipelineSteps()) {
                std::cout << "  Step: " << step.name << " (Executions: " << step.executions << ")" << std::endl;
                std::cout << "    Kernels:" << std::endl;
                for (const auto& kernel : step.kernels) {
                    std::cout << "      " << kernel.platform << ": " << kernel.file << std::endl;
                }
            }
        } else {
            std::cout << "\nKernels:" << std::endl;
            for (const auto& kernel : xmlDef.getKernels()) {
                std::cout << "  " << kernel.platform << ": " << kernel.file
                          << " (Executions: " << kernel.executions << ")" << std::endl;
            }
        }
        
        // Print identity conditions
        std::cout << "\nIdentity Conditions:" << std::endl;
        for (const auto& condition : xmlDef.getIdentityConditions()) {
            std::cout << "  " << condition.paramName << " " << condition.op << " " << condition.value << std::endl;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
