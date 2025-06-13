#include "Logger.h"
#include "src/core/XMLEffectDefinition.h"
#include <iostream>

int main() {
    try {
        // Test Logger
        Logger::getInstance().logMessage("=== Framework Test Starting ===");
        
        // Test XML Effect Definition
        XMLEffectDefinition xmlDef("effects/TestBlurV2.xml");
        Logger::getInstance().logMessage("XMLEffectDefinition created and loaded successfully");
        
        // Test getting effect information
        std::cout << "Effect Name: " << xmlDef.getName() << std::endl;
        std::cout << "Effect Category: " << xmlDef.getCategory() << std::endl;
        std::cout << "Effect Description: " << xmlDef.getDescription() << std::endl;
        
        Logger::getInstance().logMessage("=== Framework Test Complete ===");
        std::cout << "Framework test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}