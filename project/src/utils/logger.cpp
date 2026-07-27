#include "utils/logger.h"

#include <yaml-cpp/yaml.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ir {

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

bool Logger::loadConfig(const std::filesystem::path& path, std::string* error) {
    try {
        const YAML::Node root = YAML::LoadFile(path.string());
        const YAML::Node logging = root["logging"] ? root["logging"] : root;
        if (!logging.IsMap()) {
            if (error) {
                *error = "expected a 'logging' mapping";
            }
            return false;
        }

        Options options;
        auto readBool = [&logging](const char* key, bool fallback) {
            const YAML::Node value = logging[key];
            return value ? value.as<bool>() : fallback;
        };
        options.error = readBool("error", options.error);
        options.warn = readBool("warn", options.warn);
        options.info = readBool("info", options.info);
        options.debug = readBool("debug", options.debug);
        options.trace = readBool("trace", options.trace);

        std::lock_guard<std::mutex> guard(_mu);
        _options = options;
        return true;
    } catch (const std::exception& e) {
        if (error) {
            *error = e.what();
        }
        return false;
    }
}

bool Logger::isEnabled(LogLevel lv) const {
    std::lock_guard<std::mutex> guard(_mu);
    switch (lv) {
    case LogLevel::Trace:
        return _options.trace;
    case LogLevel::Debug:
        return _options.debug;
    case LogLevel::Info:
        return _options.info;
    case LogLevel::Warn:
        return _options.warn;
    case LogLevel::Error:
        return _options.error;
    }
    return false;
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
