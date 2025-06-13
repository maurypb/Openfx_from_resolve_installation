#include <iostream>
#include <fstream>
#include <string>

int main() {
    // Test the hardcoded Linux path
    std::string linuxPath = "/mnt/tank/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/xml_driven_ofx_framework_v0.0/effects/TestBlurV2.xml";
    
    // Test the correct Windows path
    std::string windowsPath = "effects/TestBlurV2.xml";
    
    std::cout << "Testing XML file paths:\n";
    std::cout << "======================\n";
    
    // Test Linux path
    std::ifstream linuxFile(linuxPath);
    if (linuxFile.good()) {
        std::cout << "[OK] Linux path exists: " << linuxPath << "\n";
    } else {
        std::cout << "[FAIL] Linux path missing: " << linuxPath << "\n";
    }
    linuxFile.close();
    
    // Test Windows path
    std::ifstream windowsFile(windowsPath);
    if (windowsFile.good()) {
        std::cout << "[OK] Windows path exists: " << windowsPath << "\n";
    } else {
        std::cout << "[FAIL] Windows path missing: " << windowsPath << "\n";
    }
    windowsFile.close();
    
    return 0;
}