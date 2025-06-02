#ifndef XML_EFFECT_DEFINITION_H
#define XML_EFFECT_DEFINITION_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

// Forward declaration for the XML parsing library
namespace pugi {
    class xml_document;
    class xml_node;
}

/**
 * @class XMLEffectDefinition
 * @brief Parses and provides access to XML effect definitions
 * 
 * This class is responsible for loading XML effect definition files,
 * validating their structure, and providing access to the parsed data.
 */
class XMLEffectDefinition {
public:
    /**
     * @struct InputDef
     * @brief Defines an input source for the effect
     */
    struct InputDef {
        std::string name;        ///< Unique identifier for the input
        std::string label;       ///< Display name in the UI
        bool optional;           ///< Whether the input is optional
        std::string borderMode;  ///< Border handling mode ("clamp", "repeat", "mirror", "black")
    };
    
    /**
     * @struct ParameterComponentDef
     * @brief Defines a component of a vector parameter
     */
    struct ParameterComponentDef {
        std::string name;        ///< Component name (e.g., "r", "g", "b", "x", "y", "z")
        double defaultValue;     ///< Default value
        double minValue;         ///< Minimum value
        double maxValue;         ///< Maximum value
        double inc;              ///< Increment for UI adjustment
    };
    
    /**
     * @struct ParameterOptionDef
     * @brief Defines an option for a choice parameter
     */
    struct ParameterOptionDef {
        std::string value;       ///< Option value
        std::string label;       ///< Display label
    };
    
    /**
     * @struct ParameterDef
     * @brief Defines a parameter for the effect
     */
    struct ParameterDef {
        std::string name;        ///< Unique identifier for the parameter
        std::string type;        ///< Parameter type (double, int, bool, vec3, etc.)
        std::string label;       ///< Display name in the UI
        std::string hint;        ///< Tooltip text
        
        // Basic properties
        double defaultValue;     ///< Default value for numeric parameters
        double minValue;         ///< Minimum value for numeric parameters
        double maxValue;         ///< Maximum value for numeric parameters
        double displayMin;       ///< Minimum value for UI display
        double displayMax;       ///< Maximum value for UI display
        double inc;              ///< Increment for UI adjustment
        
        // For curve parameters
        std::string defaultShape;    ///< Default curve shape
        int curveBackground;         ///< Background display for curves
        
        // For choice parameters
        std::vector<ParameterOptionDef> options;  ///< Options for choice parameters
        
        // For vector/color parameters
        std::vector<ParameterComponentDef> components;  ///< Components for vector parameters
        
        // For string parameters
        std::string defaultString;   ///< Default string value
        
        // For boolean parameters
        bool defaultBool;            ///< Default boolean value
        
        // Resolution dependency
        std::string resDependent;    ///< Whether value scales with resolution
    };
    
    /**
     * @struct UIParameterDef
     * @brief Defines a parameter reference in the UI
     */
    struct UIParameterDef {
        std::string name;        ///< Parameter name reference
    };
    
    /**
     * @struct UIColumnDef
     * @brief Defines a column of parameters in the UI
     */
    struct UIColumnDef {
        std::string name;        ///< Column name
        std::string tooltip;     ///< Column tooltip
        std::vector<UIParameterDef> parameters;  ///< Parameters in this column
    };
    
    /**
     * @struct UIPageDef
     * @brief Defines a page of parameters in the UI
     */
    struct UIPageDef {
        std::string name;        ///< Page name
        std::string tooltip;     ///< Page tooltip
        std::vector<UIColumnDef> columns;  ///< Columns in this page
    };
    
    /**
     * @struct KernelDef
     * @brief Defines a kernel for the effect
     */
    struct KernelDef {
        std::string platform;    ///< Platform ("cuda", "opencl", "metal")
        std::string file;        ///< Kernel file path
        int executions;          ///< Number of times to execute
    };
    
    /**
     * @struct StepKernelDef
     * @brief Defines a kernel for a pipeline step
     */
    struct StepKernelDef {
        std::string platform;    ///< Platform ("cuda", "opencl", "metal")
        std::string file;        ///< Kernel file path
    };
    
    /**
     * @struct PipelineStepDef
     * @brief Defines a step in a multi-kernel pipeline
     */
    struct PipelineStepDef {
        std::string name;        ///< Step name
        int executions;          ///< Number of times to execute
        std::vector<StepKernelDef> kernels;  ///< Kernels for this step
    };
    
    /**
     * @struct IdentityConditionDef
     * @brief Defines a condition for when the effect is a pass-through
     */
    struct IdentityConditionDef {
        std::string paramName;   ///< Parameter name
        std::string op;          ///< Operator ("equal", "notEqual", "lessThan", etc.)
        double value;            ///< Value to compare against
    };
    
public:
    /**
     * @brief Constructor
     * @param filename Path to XML effect definition file
     * @throws std::runtime_error if the file cannot be loaded or parsed
     */
    explicit XMLEffectDefinition(const std::string& filename);
    
    /**
     * @brief Destructor
     */
    ~XMLEffectDefinition();
    
    /**
     * @brief Get the effect name
     * @return Effect name
     */
    std::string getName() const;
    
    /**
     * @brief Get the effect category
     * @return Effect category
     */
    std::string getCategory() const;
    
    /**
     * @brief Get the effect description
     * @return Effect description
     */
    std::string getDescription() const;
    
    /**
     * @brief Get the effect version
     * @return Effect version
     */
    std::string getVersion() const;
    
    /**
     * @brief Get the effect author
     * @return Effect author
     */
    std::string getAuthor() const;
    
    /**
     * @brief Get the effect copyright
     * @return Effect copyright
     */
    std::string getCopyright() const;
    
    /**
     * @brief Check if the effect supports timeline
     * @return True if the effect supports timeline
     */
    bool supportsTimeline() const;
    
    /**
     * @brief Check if the effect supports matte
     * @return True if the effect supports matte
     */
    bool supportsMatte() const;
    
    /**
     * @brief Get the list of input sources
     * @return Vector of input source definitions
     */
    const std::vector<InputDef>& getInputs() const;
    
    /**
     * @brief Get the list of parameters
     * @return Vector of parameter definitions
     */
    const std::vector<ParameterDef>& getParameters() const;
    
    /**
     * @brief Get a specific parameter by name
     * @param name Parameter name
     * @return Parameter definition
     * @throws std::out_of_range if the parameter does not exist
     */
    const ParameterDef& getParameter(const std::string& name) const;
    
    /**
     * @brief Get the UI organization
     * @return Vector of page definitions
     */
    const std::vector<UIPageDef>& getUIPages() const;
    
    /**
     * @brief Get the list of kernels
     * @return Vector of kernel definitions
     */
    const std::vector<KernelDef>& getKernels() const;
    
    /**
     * @brief Get kernels for a specific platform
     * @param platform Platform name ("cuda", "opencl", "metal")
     * @return Vector of kernel definitions for the platform
     */
    std::vector<KernelDef> getKernelsForPlatform(const std::string& platform) const;
    
    /**
     * @brief Check if the effect has a pipeline (multiple kernels)
     * @return True if the effect has a pipeline
     */
    bool hasPipeline() const;
    
    /**
     * @brief Get the pipeline steps
     * @return Vector of pipeline step definitions
     */
    const std::vector<PipelineStepDef>& getPipelineSteps() const;
    
    /**
     * @brief Get the identity conditions
     * @return Vector of identity condition definitions
     */
    const std::vector<IdentityConditionDef>& getIdentityConditions() const;

private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> _impl;
    
    // Parsed data
    std::string _name;
    std::string _category;
    std::string _description;
    std::string _version;
    std::string _author;
    std::string _copyright;
    bool _supportsTimeline;
    bool _supportsMatte;
    
    std::vector<InputDef> _inputs;
    std::vector<ParameterDef> _parameters;
    std::map<std::string, size_t> _parameterMap;  // Maps parameter names to indices
    std::vector<UIPageDef> _uiPages;
    std::vector<KernelDef> _kernels;
    bool _hasPipeline;
    std::vector<PipelineStepDef> _pipelineSteps;
    std::vector<IdentityConditionDef> _identityConditions;
    
    // Private methods
    void parseXML(const std::string& filename);
    void parseEffectMetadata(const pugi::xml_node& effectNode);
    void parseInputs(const pugi::xml_node& inputsNode);
    void parseParameters(const pugi::xml_node& parametersNode);
    void parseUI(const pugi::xml_node& uiNode);
    void parseKernels(const pugi::xml_node& kernelsNode);
    void parsePipeline(const pugi::xml_node& pipelineNode);
    void parseIdentityConditions(const pugi::xml_node& conditionsNode);
};

#endif // XML_EFFECT_DEFINITION_H
