#pragma once

#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace ir {

// ---------------------------------------------------------------------------
// 轻量级线程安全日志工具，输出到 stdout/stderr。
// ---------------------------------------------------------------------------
enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

class Logger {
public:
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void setLevel(LogLevel lv) { level_ = lv; }
    LogLevel level() const     { return level_; }

    template <typename... Args>
    void log(LogLevel lv, Args&&... args) {
        if (lv < level_) return;
        std::ostringstream oss;
        oss << prefix(lv);
        appendAll(oss, std::forward<Args>(args)...);
        std::lock_guard<std::mutex> g(mu_);
        if (lv >= LogLevel::Warn) std::cerr << oss.str() << std::endl;
        else                       std::cout << oss.str() << std::endl;
    }

private:
    Logger() = default;

    static const char* prefix(LogLevel lv) {
        switch (lv) {
            case LogLevel::Debug: return "[DEBUG] ";
            case LogLevel::Info:  return "[INFO ] ";
            case LogLevel::Warn:  return "[WARN ] ";
            case LogLevel::Error: return "[ERROR] ";
        }
        return "[?????] ";
    }

    template <typename T>
    static void appendAll(std::ostringstream& oss, T&& t) {
        oss << std::forward<T>(t);
    }
    template <typename T, typename... Rest>
    static void appendAll(std::ostringstream& oss, T&& t, Rest&&... rest) {
        oss << std::forward<T>(t);
        appendAll(oss, std::forward<Rest>(rest)...);
    }

    LogLevel   level_ = LogLevel::Info;
    std::mutex mu_;
};

#define IR_LOG_DEBUG(...) ::ir::Logger::instance().log(::ir::LogLevel::Debug, __VA_ARGS__)
#define IR_LOG_INFO(...)  ::ir::Logger::instance().log(::ir::LogLevel::Info,  __VA_ARGS__)
#define IR_LOG_WARN(...)  ::ir::Logger::instance().log(::ir::LogLevel::Warn,  __VA_ARGS__)
#define IR_LOG_ERROR(...) ::ir::Logger::instance().log(::ir::LogLevel::Error, __VA_ARGS__)

} // namespace ir
