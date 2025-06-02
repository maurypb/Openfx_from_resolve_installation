#include "ParameterValue.h"

ParameterValue::ParameterValue() : m_type(DOUBLE), m_double(0.0) {
}

ParameterValue::ParameterValue(double value) : m_type(DOUBLE), m_double(value) {
}

ParameterValue::ParameterValue(int value) : m_type(INT), m_int(value) {
}

ParameterValue::ParameterValue(bool value) : m_type(BOOL), m_bool(value) {
}

ParameterValue::ParameterValue(const std::string& value) : m_type(STRING), m_string(value) {
}

ParameterValue::ParameterValue(const char* value) : m_type(STRING), m_string(value) {
}

ParameterValue::ParameterValue(const ParameterValue& other) {
    copyFrom(other);
}

ParameterValue& ParameterValue::operator=(const ParameterValue& other) {
    if (this != &other) {
        copyFrom(other);
    }
    return *this;
}

ParameterValue::~ParameterValue() {
    // Destructor - string handled automatically
}

double ParameterValue::asDouble() const {
    switch (m_type) {
        case DOUBLE: 
            return m_double;
        case INT: 
            return static_cast<double>(m_int);
        case BOOL: 
            return m_bool ? 1.0 : 0.0;
        case STRING: 
            // Try to parse string as double
            try {
                return std::stod(m_string);
            } catch (...) {
                return 0.0;
            }
        default: 
            return 0.0;
    }
}

int ParameterValue::asInt() const {
    switch (m_type) {
        case INT: 
            return m_int;
        case DOUBLE: 
            return static_cast<int>(m_double);
        case BOOL: 
            return m_bool ? 1 : 0;
        case STRING: 
            // Try to parse string as int
            try {
                return std::stoi(m_string);
            } catch (...) {
                return 0;
            }
        default: 
            return 0;
    }
}

bool ParameterValue::asBool() const {
    switch (m_type) {
        case BOOL: 
            return m_bool;
        case DOUBLE: 
            return m_double != 0.0;
        case INT: 
            return m_int != 0;
        case STRING: 
            // Common boolean string representations
            if (m_string == "true" || m_string == "True" || m_string == "TRUE" || m_string == "1") {
                return true;
            } else if (m_string == "false" || m_string == "False" || m_string == "FALSE" || m_string == "0") {
                return false;
            } else {
                // Non-empty string is true, empty string is false
                return !m_string.empty();
            }
        default: 
            return false;
    }
}

float ParameterValue::asFloat() const {
    return static_cast<float>(asDouble());
}

std::string ParameterValue::asString() const {
    switch (m_type) {
        case STRING: 
            return m_string;
        case DOUBLE: 
            return std::to_string(m_double);
        case INT: 
            return std::to_string(m_int);
        case BOOL: 
            return m_bool ? "true" : "false";
        default: 
            return "";
    }
}

std::string ParameterValue::getType() const {
    switch (m_type) {
        case DOUBLE: return "double";
        case INT: return "int";
        case BOOL: return "bool";
        case STRING: return "string";
        default: return "unknown";
    }
}

void ParameterValue::copyFrom(const ParameterValue& other) {
    m_type = other.m_type;
    
    switch (m_type) {
        case DOUBLE:
            m_double = other.m_double;
            break;
        case INT:
            m_int = other.m_int;
            break;
        case BOOL:
            m_bool = other.m_bool;
            break;
        case STRING:
            m_string = other.m_string;
            break;
    }
}