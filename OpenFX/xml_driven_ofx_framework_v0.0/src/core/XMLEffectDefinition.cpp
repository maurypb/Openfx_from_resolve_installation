#include "XMLEffectDefinition.h"
#include "../../include/pugixml/pugixml.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>

// Implementation class (PIMPL pattern)
class XMLEffectDefinition::Impl {
public:
    pugi::xml_document _doc;
};

XMLEffectDefinition::XMLEffectDefinition(const std::string& filename)
    : _impl(std::make_unique<Impl>()),
      _supportsTimeline(false),
      _supportsMatte(false),
      _hasPipeline(false)
{
    parseXML(filename);
}

XMLEffectDefinition::~XMLEffectDefinition() = default;

std::string XMLEffectDefinition::getName() const {
    return _name;
}

std::string XMLEffectDefinition::getCategory() const {
    return _category;
}

std::string XMLEffectDefinition::getDescription() const {
    return _description;
}

std::string XMLEffectDefinition::getVersion() const {
    return _version;
}

std::string XMLEffectDefinition::getAuthor() const {
    return _author;
}

std::string XMLEffectDefinition::getCopyright() const {
    return _copyright;
}

bool XMLEffectDefinition::supportsTimeline() const {
    return _supportsTimeline;
}

bool XMLEffectDefinition::supportsMatte() const {
    return _supportsMatte;
}

const std::vector<XMLEffectDefinition::InputDef>& XMLEffectDefinition::getInputs() const {
    return _inputs;
}

const std::vector<XMLEffectDefinition::ParameterDef>& XMLEffectDefinition::getParameters() const {
    return _parameters;
}

const XMLEffectDefinition::ParameterDef& XMLEffectDefinition::getParameter(const std::string& name) const {
    auto it = _parameterMap.find(name);
    if (it == _parameterMap.end()) {
        throw std::out_of_range("Parameter not found: " + name);
    }
    return _parameters[it->second];
}

const std::vector<XMLEffectDefinition::UIGroupDef>& XMLEffectDefinition::getUIGroups() const {
    return _uiGroups;
}

const std::vector<XMLEffectDefinition::KernelDef>& XMLEffectDefinition::getKernels() const {
    return _kernels;
}

std::vector<XMLEffectDefinition::KernelDef> XMLEffectDefinition::getKernelsForPlatform(const std::string& platform) const {
    std::vector<KernelDef> result;
    for (const auto& kernel : _kernels) {
        if (kernel.platform == platform) {
            result.push_back(kernel);
        }
    }
    return result;
}

bool XMLEffectDefinition::hasPipeline() const {
    return _hasPipeline;
}

const std::vector<XMLEffectDefinition::PipelineStepDef>& XMLEffectDefinition::getPipelineSteps() const {
    return _pipelineSteps;
}

const std::vector<XMLEffectDefinition::IdentityConditionDef>& XMLEffectDefinition::getIdentityConditions() const {
    return _identityConditions;
}

void XMLEffectDefinition::parseXML(const std::string& filename) {
    // Load XML file
    pugi::xml_parse_result result = _impl->_doc.load_file(filename.c_str());
    if (!result) {
        throw std::runtime_error("Failed to load XML file: " + std::string(result.description()));
    }
    
    // Get root node
    pugi::xml_node effectNode = _impl->_doc.child("effect");
    if (!effectNode) {
        throw std::runtime_error("XML file does not contain an effect element");
    }
    
    // Parse effect metadata
    parseEffectMetadata(effectNode);
    
    // Parse inputs
    pugi::xml_node inputsNode = effectNode.child("inputs");
    if (inputsNode) {
        parseInputs(inputsNode);
    }
    
    // Parse parameters
    pugi::xml_node parametersNode = effectNode.child("parameters");
    if (parametersNode) {
        parseParameters(parametersNode);
    }
    
    // Parse UI
    pugi::xml_node uiNode = effectNode.child("ui");
    if (uiNode) {
        parseUI(uiNode);
    }
    
    // Parse kernels or pipeline
    pugi::xml_node kernelsNode = effectNode.child("kernels");
    pugi::xml_node pipelineNode = effectNode.child("pipeline");
    
    if (pipelineNode) {
        _hasPipeline = true;
        parsePipeline(pipelineNode);
    } else if (kernelsNode) {
        parseKernels(kernelsNode);
    }
    
    // Parse identity conditions
    pugi::xml_node conditionsNode = effectNode.child("identity_conditions");
    if (conditionsNode) {
        parseIdentityConditions(conditionsNode);
    }
}

void XMLEffectDefinition::parseEffectMetadata(const pugi::xml_node& effectNode) {
    // Required attributes
    _name = effectNode.attribute("name").as_string();
    _category = effectNode.attribute("category").as_string();
    
    if (_name.empty()) {
        throw std::runtime_error("Effect must have a name attribute");
    }
    
    if (_category.empty()) {
        throw std::runtime_error("Effect must have a category attribute");
    }
    
    // Optional attributes
    _version = effectNode.attribute("version").as_string();
    _author = effectNode.attribute("author").as_string();
    _copyright = effectNode.attribute("copyright").as_string();
    _supportsTimeline = effectNode.attribute("supports_timeline").as_bool(false);
    _supportsMatte = effectNode.attribute("supports_matte").as_bool(false);
    
    // Description element
    pugi::xml_node descNode = effectNode.child("description");
    if (descNode) {
        _description = descNode.text().as_string();
    }
}

void XMLEffectDefinition::parseInputs(const pugi::xml_node& inputsNode) {
    for (pugi::xml_node sourceNode : inputsNode.children("source")) {
        InputDef input;
        input.name = sourceNode.attribute("name").as_string();
        input.label = sourceNode.attribute("label").as_string();
        input.optional = sourceNode.attribute("optional").as_bool(false);
        input.borderMode = sourceNode.attribute("border_mode").as_string("clamp");
        
        if (input.name.empty()) {
            throw std::runtime_error("Input source must have a name attribute");
        }
        
        if (input.label.empty()) {
            // Default to name if label not provided
            input.label = input.name;
        }
        
        _inputs.push_back(std::move(input));
    }
}

void XMLEffectDefinition::parseParameters(const pugi::xml_node& parametersNode) {
    int paramIndex = 0;
    for (pugi::xml_node paramNode : parametersNode.children("parameter")) {
        ParameterDef param;
        param.name = paramNode.attribute("name").as_string();
        param.type = paramNode.attribute("type").as_string();
        param.label = paramNode.attribute("label").as_string();
        param.hint = paramNode.attribute("hint").as_string();
        
        if (param.name.empty()) {
            throw std::runtime_error("Parameter must have a name attribute");
        }
        
        if (param.type.empty()) {
            throw std::runtime_error("Parameter must have a type attribute");
        }
        
        if (param.label.empty()) {
            // Default to name if label not provided
            param.label = param.name;
        }
        
        // Parse type-specific attributes
        param.resDependent = paramNode.attribute("res_dependent").as_string("none");
        
        if (param.type == "double" || param.type == "int" || param.type == "float") {
            // Numeric parameter
            param.defaultValue = paramNode.attribute("default").as_double(0.0);
            param.minValue = paramNode.attribute("min").as_double(0.0);
            param.maxValue = paramNode.attribute("max").as_double(1.0);
            param.displayMin = paramNode.attribute("displayMin").as_double(param.minValue);
            param.displayMax = paramNode.attribute("displayMax").as_double(param.maxValue);
            param.inc = paramNode.attribute("inc").as_double(0.1);
        } 
        else if (param.type == "bool") {
            // Boolean parameter
            param.defaultBool = paramNode.attribute("default").as_bool(false);
        } 
        else if (param.type == "string") {
            // String parameter
            param.defaultString = paramNode.attribute("default").as_string("");
        } 
        else if (param.type == "curve") {
            // Curve parameter
            param.defaultShape = paramNode.attribute("default_shape").as_string("linear");
            param.curveBackground = paramNode.attribute("curve_background").as_int(0);
        } 
        else if (param.type == "choice") {
            // Choice parameter
            param.defaultValue = paramNode.attribute("default").as_double(0);
            
            // Parse options
            for (pugi::xml_node optionNode : paramNode.children("option")) {
                ParameterOptionDef option;
                option.value = optionNode.attribute("value").as_string();
                option.label = optionNode.attribute("label").as_string();
                
                if (option.value.empty()) {
                    throw std::runtime_error("Option must have a value attribute");
                }
                
                if (option.label.empty()) {
                    // Default to value if label not provided
                    option.label = option.value;
                }
                
                param.options.push_back(std::move(option));
            }
        } 
        else if (param.type == "color" || param.type == "vec2" || param.type == "vec3" || param.type == "vec4") {
            // Vector parameter
            // Parse components
            for (pugi::xml_node compNode : paramNode.children("component")) {
                ParameterComponentDef comp;
                comp.name = compNode.attribute("name").as_string();
                comp.defaultValue = compNode.attribute("default").as_double(0.0);
                comp.minValue = compNode.attribute("min").as_double(0.0);
                comp.maxValue = compNode.attribute("max").as_double(1.0);
                comp.inc = compNode.attribute("inc").as_double(0.01);
                
                if (comp.name.empty()) {
                    throw std::runtime_error("Component must have a name attribute");
                }
                
                param.components.push_back(std::move(comp));
            }
            
            // If no components defined but default attribute exists, try to parse it
            if (param.components.empty() && paramNode.attribute("default")) {
                std::string defaultStr = paramNode.attribute("default").as_string();
                std::vector<std::string> values;
                
                // Split by commas
                size_t pos = 0;
                std::string token;
                std::string s = defaultStr;
                while ((pos = s.find(',')) != std::string::npos) {
                    token = s.substr(0, pos);
                    values.push_back(token);
                    s.erase(0, pos + 1);
                }
                values.push_back(s);
                
                // Create components
                std::vector<std::string> componentNames;
                if (param.type == "color" || param.type == "vec3") {
                    componentNames = {"r", "g", "b"};
                } else if (param.type == "vec2") {
                    componentNames = {"x", "y"};
                } else if (param.type == "vec4") {
                    componentNames = {"r", "g", "b", "a"};
                }
                
                for (size_t i = 0; i < std::min(values.size(), componentNames.size()); ++i) {
                    ParameterComponentDef comp;
                    comp.name = componentNames[i];
                    comp.defaultValue = std::stod(values[i]);
                    comp.minValue = 0.0;
                    comp.maxValue = 1.0;
                    comp.inc = 0.01;
                    
                    param.components.push_back(std::move(comp));
                }
            }
        }
        
        // Store parameter
        _parameters.push_back(std::move(param));
        _parameterMap[_parameters.back().name] = paramIndex++;
    }
}

void XMLEffectDefinition::parseUI(const pugi::xml_node& uiNode) {
    for (pugi::xml_node groupNode : uiNode.children("group")) {
        UIGroupDef group;
        group.name = groupNode.attribute("name").as_string();
        group.tooltip = groupNode.attribute("tooltip").as_string();
        
        if (group.name.empty()) {
            throw std::runtime_error("group must have a name attribute");
        }
        
        // Check if this group has columns (old format) or direct parameters (new format)
        bool hasColumns = groupNode.child("column");
        bool hasDirectParams = groupNode.child("parameter");
        
        if (hasColumns) {
            // OLD FORMAT: Parse columns
            for (pugi::xml_node colNode : groupNode.children("column")) {
                UIColumnDef column;
                column.name = colNode.attribute("name").as_string();
                column.tooltip = colNode.attribute("tooltip").as_string();
                
                if (column.name.empty()) {
                    throw std::runtime_error("Column must have a name attribute");
                }
                
                // Parse parameters in column
                for (pugi::xml_node paramNode : colNode.children("parameter")) {
                    UIParameterDef param;
                    param.name = paramNode.text().as_string();
                    
                    if (param.name.empty()) {
                        throw std::runtime_error("Parameter reference cannot be empty");
                    }
                    
                    // Verify parameter exists
                    if (_parameterMap.find(param.name) == _parameterMap.end()) {
                        throw std::runtime_error("Referenced parameter does not exist: " + param.name);
                    }
                    
                    column.parameters.push_back(std::move(param));
                }
                
                group.columns.push_back(std::move(column));
            }
        }
        
        if (hasDirectParams) {
            // NEW FORMAT: Parse parameters directly
            for (pugi::xml_node paramNode : groupNode.children("parameter")) {
                UIParameterDef param;
                param.name = paramNode.text().as_string();
                
                if (param.name.empty()) {
                    throw std::runtime_error("Parameter reference cannot be empty");
                }
                
                // Verify parameter exists
                if (_parameterMap.find(param.name) == _parameterMap.end()) {
                    throw std::runtime_error("Referenced parameter does not exist: " + param.name);
                }
                
                group.parameters.push_back(std::move(param));
            }
        }
        
        if (!hasColumns && !hasDirectParams) {
            throw std::runtime_error("group must contain either columns or direct parameter references");
        }
        
        _uiGroups.push_back(std::move(group));
    }
}


void XMLEffectDefinition::parseKernels(const pugi::xml_node& kernelsNode) {
    for (pugi::xml_node kernelNode : kernelsNode.children()) {
        KernelDef kernel;
        kernel.platform = kernelNode.name();
        kernel.file = kernelNode.attribute("file").as_string();
        kernel.executions = kernelNode.attribute("executions").as_int(1);
        
        if (kernel.file.empty()) {
            throw std::runtime_error("Kernel must have a file attribute");
        }
        
        _kernels.push_back(std::move(kernel));
    }
}

void XMLEffectDefinition::parsePipeline(const pugi::xml_node& pipelineNode) {
    for (pugi::xml_node stepNode : pipelineNode.children("step")) {
        PipelineStepDef step;
        step.name = stepNode.attribute("name").as_string();
        step.executions = stepNode.attribute("executions").as_int(1);
        
        if (step.name.empty()) {
            throw std::runtime_error("Pipeline step must have a name attribute");
        }
        
        // Parse kernels
        pugi::xml_node kernelsNode = stepNode.child("kernels");
        if (kernelsNode) {
            for (pugi::xml_node kernelNode : kernelsNode.children()) {
                StepKernelDef kernel;
                kernel.platform = kernelNode.name();
                kernel.file = kernelNode.attribute("file").as_string();
                
                if (kernel.file.empty()) {
                    throw std::runtime_error("Kernel must have a file attribute");
                }
                
                step.kernels.push_back(std::move(kernel));
            }
        }
        
        _pipelineSteps.push_back(std::move(step));
    }
}

void XMLEffectDefinition::parseIdentityConditions(const pugi::xml_node& conditionsNode) {
    for (pugi::xml_node condNode : conditionsNode.children("condition")) {
        pugi::xml_node paramNode = condNode.child("parameter");
        if (!paramNode) {
            throw std::runtime_error("Condition must have a parameter element");
        }
        
        IdentityConditionDef condition;
        condition.paramName = paramNode.attribute("name").as_string();
        condition.op = paramNode.attribute("operator").as_string();
        condition.value = paramNode.attribute("value").as_double();
        
        if (condition.paramName.empty()) {
            throw std::runtime_error("Parameter in condition must have a name attribute");
        }
        
        if (condition.op.empty()) {
            throw std::runtime_error("Parameter in condition must have an operator attribute");
        }
        
        // Verify parameter exists
        if (_parameterMap.find(condition.paramName) == _parameterMap.end()) {
            throw std::runtime_error("Referenced parameter does not exist: " + condition.paramName);
        }
        
        _identityConditions.push_back(std::move(condition));
    }
}
