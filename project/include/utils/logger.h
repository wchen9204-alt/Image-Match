#pragma once

#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace ir {

/// 线程安全的轻量级日志工具。
///
/// 日志默认输出到 stdout/stderr，并提供 `Debug`、`Info`、`Warn` 和
/// `Error` 四个级别。
enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

/// 全局日志器单例。
class Logger {
public:
    /// 返回全局唯一日志器实例。
    static Logger& instance();

    /// 设置最低输出级别。
    void setLevel(LogLevel lv) { _level = lv; }

    /// 返回当前日志级别。
    LogLevel level() const { return _level; }

    /// 输出一条日志消息。
    template <typename... Args> void log(LogLevel lv, Args&&... args) {
        if (lv < _level)
            return;
        std::ostringstream oss;
        oss << prefix(lv);
        appendAll(oss, std::forward<Args>(args)...);
        std::lock_guard<std::mutex> g(_mu);
        writeLine(lv, oss.str());
    }

private:
    Logger() = default;

    /// 为日志级别生成前缀文本。
    static const char* prefix(LogLevel lv) {
        switch (lv) {
        case LogLevel::Debug:
            return "[DEBUG] ";
        case LogLevel::Info:
            return "[INFO ] ";
        case LogLevel::Warn:
            return "[WARN ] ";
        case LogLevel::Error:
            return "[ERROR] ";
        }
        return "[?????] ";
    }

    template <typename T> static void appendAll(std::ostringstream& oss, T&& t) {
        oss << std::forward<T>(t);
    }
    template <typename T, typename... Rest>
    static void appendAll(std::ostringstream& oss, T&& t, Rest&&... rest) {
        oss << std::forward<T>(t);
        appendAll(oss, std::forward<Rest>(rest)...);
    }

    static void writeLine(LogLevel lv, const std::string& msg);

    LogLevel _level = LogLevel::Info;
    std::mutex _mu;
};

#define IR_STRINGIZE_IMPL(x) #x
#define IR_STRINGIZE(x) IR_STRINGIZE_IMPL(x)
#define IR_LOG_LOCATION __FILE__ ":" IR_STRINGIZE(__LINE__)

#define IR_LOG_DEBUG(...)                                                                          \
    ::ir::Logger::instance().log(::ir::LogLevel::Debug, IR_LOG_LOCATION, " | ", __VA_ARGS__)
#define IR_LOG_INFO(...)                                                                           \
    ::ir::Logger::instance().log(::ir::LogLevel::Info, IR_LOG_LOCATION, " | ", __VA_ARGS__)
#define IR_LOG_WARN(...)                                                                           \
    ::ir::Logger::instance().log(::ir::LogLevel::Warn, IR_LOG_LOCATION, " | ", __VA_ARGS__)
#define IR_LOG_ERROR(...)                                                                          \
    ::ir::Logger::instance().log(::ir::LogLevel::Error, IR_LOG_LOCATION, " | ", __VA_ARGS__)

} // namespace ir
