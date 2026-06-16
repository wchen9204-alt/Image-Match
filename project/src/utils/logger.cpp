#include "utils/logger.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace ir {

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

void Logger::writeLine(LogLevel lv, const std::string& msg) {
#ifdef _WIN32
    const HANDLE handle = GetStdHandle(lv >= LogLevel::Warn ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (handle != INVALID_HANDLE_VALUE && handle != nullptr && GetConsoleMode(handle, &mode)) {
        const int wide_len =
            MultiByteToWideChar(CP_UTF8, 0, msg.c_str(), static_cast<int>(msg.size()), nullptr, 0);
        if (wide_len > 0) {
            std::wstring wide(static_cast<size_t>(wide_len), L'\0');
            MultiByteToWideChar(
                CP_UTF8, 0, msg.c_str(), static_cast<int>(msg.size()), wide.data(), wide_len);
            wide.push_back(L'\n');

            DWORD written = 0;
            WriteConsoleW(handle, wide.c_str(), static_cast<DWORD>(wide.size()), &written, nullptr);
            return;
        }
    }
#endif

    if (lv >= LogLevel::Warn)
        std::cerr << msg << std::endl;
    else
        std::cout << msg << std::endl;
}

} // namespace ir

