#include "core/ParameterValue.h"
#include <iostream>
#include <cassert>
#include <cmath>

void testDoubleValue() {
    std::cout << "Testing double value..." << std::endl;
    
    ParameterValue value(5.7);
    
    assert(value.getType() == "double");
    assert(std::abs(value.asDouble() - 5.7) < 0.001);
    assert(value.asInt() == 5);
    assert(value.asBool() == true);
    assert(std::abs(value.asFloat() - 5.7f) < 0.001);
    assert(value.asString() == "5.700000");
    
    std::cout << "  Double value tests passed!" << std::endl;
}

void testIntValue() {
    std::cout << "Testing int value..." << std::endl;
    
    ParameterValue value(42);
    
    assert(value.getType() == "int");
    assert(value.asDouble() == 42.0);
    assert(value.asInt() == 42);
    assert(value.asBool() == true);
    assert(value.asFloat() == 42.0f);
    assert(value.asString() == "42");
    
    // Test zero int
    ParameterValue zero(0);
    assert(zero.asBool() == false);
    
    std::cout << "  Int value tests passed!" << std::endl;
}

void testBoolValue() {
    std::cout << "Testing bool value..." << std::endl;
    
    ParameterValue valueTrue(true);
    assert(valueTrue.getType() == "bool");
    assert(valueTrue.asDouble() == 1.0);
    assert(valueTrue.asInt() == 1);
    assert(valueTrue.asBool() == true);
    assert(valueTrue.asFloat() == 1.0f);
    assert(valueTrue.asString() == "true");
    
    ParameterValue valueFalse(false);
    assert(valueFalse.asDouble() == 0.0);
    assert(valueFalse.asInt() == 0);
    assert(valueFalse.asBool() == false);
    assert(valueFalse.asString() == "false");
    
    std::cout << "  Bool value tests passed!" << std::endl;
}

void testStringValue() {
    std::cout << "Testing string value..." << std::endl;
    
    std::cout << "  Creating ParameterValue with string literal..." << std::endl;
    ParameterValue value("hello");
    std::cout << "  String value type: '" << value.getType() << "'" << std::endl;
    std::cout << "  String value content: '" << value.asString() << "'" << std::endl;
    
    // Try with explicit std::string
    std::cout << "  Creating ParameterValue with std::string..." << std::endl;
    std::string hello = "world";
    ParameterValue value2(hello);
    std::cout << "  String value2 type: '" << value2.getType() << "'" << std::endl;
    
    assert(value.getType() == "string");
    assert(value.asString() == "hello");
    assert(value.asBool() == true);  // Non-empty string is true
    
    // Test numeric string
    ParameterValue numString("3.14");
    assert(std::abs(numString.asDouble() - 3.14) < 0.001);
    assert(numString.asInt() == 3);
    
    // Test boolean strings
    ParameterValue trueString("true");
    assert(trueString.asBool() == true);
    
    ParameterValue falseString("false");
    assert(falseString.asBool() == false);
    
    // Test empty string
    ParameterValue empty("");
    assert(empty.asBool() == false);
    
    std::cout << "  String value tests passed!" << std::endl;
}

void testCopyAndAssignment() {
    std::cout << "Testing copy and assignment..." << std::endl;
    
    ParameterValue original(42);
    
    // Test copy constructor
    ParameterValue copy(original);
    assert(copy.asInt() == 42);
    assert(copy.getType() == "int");
    
    // Test assignment
    ParameterValue assigned(1.5);  // Different type initially
    assigned = original;
    assert(assigned.asInt() == 42);
    assert(assigned.getType() == "int");
    
    // Test self-assignment
    assigned = assigned;
    assert(assigned.asInt() == 42);
    
    std::cout << "  Copy and assignment tests passed!" << std::endl;
}

void testDefaultConstructor() {
    std::cout << "Testing default constructor..." << std::endl;
    
    ParameterValue value;
    assert(value.getType() == "double");
    assert(value.asDouble() == 0.0);
    assert(value.asInt() == 0);
    assert(value.asBool() == false);
    
    std::cout << "  Default constructor tests passed!" << std::endl;
}

void testTypeConversions() {
    std::cout << "Testing edge case type conversions..." << std::endl;
    
    // Test invalid string to number conversions
    ParameterValue invalidNum("not_a_number");
    assert(invalidNum.asDouble() == 0.0);
    assert(invalidNum.asInt() == 0);
    
    // Test zero values
    ParameterValue zeroDouble(0.0);
    assert(zeroDouble.asBool() == false);
    
    ParameterValue zeroInt(0);
    assert(zeroInt.asBool() == false);
    
    std::cout << "  Type conversion tests passed!" << std::endl;
}

int main() {
    std::cout << "Running ParameterValue tests..." << std::endl;
    
    try {
        testDefaultConstructor();
        testDoubleValue();
        testIntValue();
        testBoolValue();
        testStringValue();
        testCopyAndAssignment();
        testTypeConversions();
        
        std::cout << std::endl << "All ParameterValue tests passed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}