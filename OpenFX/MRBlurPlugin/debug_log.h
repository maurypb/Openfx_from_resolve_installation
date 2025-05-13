#pragma once
#include <fstream>
#include <string>

class DebugLog {
public:
    static void log(const std::string& component, const std::string& message) {
        std::ofstream logFile("/tmp/blur_debug.log", std::ios::app);
        if (logFile.is_open()) {
            logFile << "[" << component << "] " << message << std::endl;
            logFile.flush();
        }
    }

    static void logMemory(const std::string& stage, size_t width, size_t height) {
        std::ofstream logFile("/tmp/blur_memory.log", std::ios::app);
        if (logFile.is_open()) {
            logFile << "[Memory-" << stage << "] "
                   << "Width: " << width 
                   << ", Height: " << height
                   << ", Total bytes: " << (width * height * 4 * sizeof(float))
                   << std::endl;
            logFile.flush();
        }
    }

    static void logMaskStats(const std::string& stage, const float* mask, size_t width, size_t height) {
        std::ofstream logFile("/tmp/blur_mask.log", std::ios::app);
        if (logFile.is_open()) {
            float min = 1.0f, max = 0.0f, sum = 0.0f;
            size_t nonZeroPixels = 0;
            
            for(size_t i = 0; i < width * height * 4; i += 4) {
                float val = mask[i]; // Using first channel
                min = std::min(min, val);
                max = std::max(max, val);
                sum += val;
                if(val > 0.0001f) nonZeroPixels++;
            }

            logFile << "[Mask-" << stage << "] "
                   << "Min: " << min 
                   << ", Max: " << max
                   << ", Avg: " << (sum / (width * height))
                   << ", NonZero: " << nonZeroPixels
                   << std::endl;
            logFile.flush();
        }
    }
};