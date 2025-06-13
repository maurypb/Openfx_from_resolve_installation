#include "Logger.h"
#include <cstdarg>  // For va_list, va_start, va_end
#include <ctime>    // For time_t, time, localtime
#include <cstddef>  // For NULL

// Initialize the static member
Logger* Logger::m_Instance = nullptr;

Logger::Logger() {
    m_LogFile = fopen("blur_plugin_log.txt", "w");
    if (m_LogFile) {
        fprintf(m_LogFile, "=== New BlurPlugin Session Started ===\n");
        fflush(m_LogFile);
        
        // Add timestamp
        time_t rawtime;
        struct tm* timeinfo;
        char buffer[80];
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
        fprintf(m_LogFile, "Session time: %s\n", buffer);
        fflush(m_LogFile);
    }
}

Logger::~Logger() {
    if (m_LogFile) {
        fprintf(m_LogFile, "=== Logger Destroyed ===\n");
        fclose(m_LogFile);
        m_LogFile = NULL;
    }
}

Logger& Logger::getInstance() {
    if (!m_Instance) {
        m_Instance = new Logger();
    }
    return *m_Instance;
}

void Logger::release() {
    delete m_Instance;
    m_Instance = nullptr;
}

void Logger::logMessage(const char* format, ...) {
    if (m_LogFile) {
        va_list args;
        va_start(args, format);
        vfprintf(m_LogFile, format, args);
        fprintf(m_LogFile, "\n");
        fflush(m_LogFile);
        va_end(args);
    }
}