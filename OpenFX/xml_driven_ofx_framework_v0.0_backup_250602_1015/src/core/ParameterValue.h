#ifndef PARAMETER_VALUE_H
#define PARAMETER_VALUE_H

#include <string>

/**
 * @class ParameterValue
 * @brief Type-safe parameter value storage for dynamic parameter passing
 * 
 * This class provides type-safe storage and conversion for parameter values
 * that can be passed dynamically to GPU kernels. It supports the main
 * parameter types used in image processing effects.
 */
class ParameterValue {
public:
    /**
     * @brief Default constructor - creates a double value of 0.0
     */
    ParameterValue();
    
    /**
     * @brief Construct from double value
     * @param value The double value to store
     */
    ParameterValue(double value);
    
    /**
     * @brief Construct from int value
     * @param value The int value to store
     */
    ParameterValue(int value);
    
    /**
     * @brief Construct from bool value
     * @param value The bool value to store
     */
    ParameterValue(bool value);
    
    /**
     * @brief Construct from string value
     * @param value The string value to store
     */
    ParameterValue(const std::string& value);
    
    /**
     * @brief Construct from C-string value
     * @param value The C-string value to store
     */
    ParameterValue(const char* value);
    
    /**
     * @brief Copy constructor
     * @param other The ParameterValue to copy
     */
    ParameterValue(const ParameterValue& other);
    
    /**
     * @brief Assignment operator
     * @param other The ParameterValue to assign from
     * @return Reference to this object
     */
    ParameterValue& operator=(const ParameterValue& other);
    
    /**
     * @brief Destructor
     */
    ~ParameterValue();
    
    /**
     * @brief Convert to double value
     * @return The value as a double
     */
    double asDouble() const;
    
    /**
     * @brief Convert to int value
     * @return The value as an int
     */
    int asInt() const;
    
    /**
     * @brief Convert to bool value
     * @return The value as a bool
     */
    bool asBool() const;
    
    /**
     * @brief Convert to float value
     * @return The value as a float
     */
    float asFloat() const;
    
    /**
     * @brief Convert to string value
     * @return The value as a string
     */
    std::string asString() const;
    
    /**
     * @brief Get the type of the stored value
     * @return The type as a string ("double", "int", "bool", "string")
     */
    std::string getType() const;

private:
    enum Type { DOUBLE, INT, BOOL, STRING };
    
    Type m_type;
    
    union {
        double m_double;
        int m_int;
        bool m_bool;
    };
    
    std::string m_string;
    
    void copyFrom(const ParameterValue& other);
};

#endif // PARAMETER_VALUE_H