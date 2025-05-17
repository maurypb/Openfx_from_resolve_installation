#pragma once

#include <cstdio>  // For FILE type

class Logger {
private:
    FILE* m_LogFile;
    static Logger* m_Instance;  // Singleton pattern
    
    // Private constructor for singleton
    Logger();

public:
    // Destructor automatically closes the file
    ~Logger();
    
    // Delete copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    // Get the singleton instance
    static Logger& getInstance();
    
    // Release the singleton (call during shutdown)
    static void release();
    
    void logMessage(const char* format, ...);
};