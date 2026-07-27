#pragma once

#include <filesystem>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace ir {

/// 线程安全的轻量级日志工具。
///
/// 日志默认输出到 stdout/stderr，并提供五个可独立开关的级别。
enum class LogLevel { Trace = 0, Debug = 1, Info = 2, Warn = 3, Error = 4 };

/// 全局日志器单例。
class Logger {
public:
    /// 返回全局唯一日志器实例。
    static Logger& instance();

    struct Options {
        bool error = true;
        bool warn = true;
        bool info = true;
        bool debug = false;
        bool trace = false;
    };

    /// 从 logging.yaml 加载各日志级别的独立开关。配置错误时保持当前设置。
    bool loadConfig(const std::filesystem::path& path, std::string* error = nullptr);

    /// 返回指定级别当前是否启用。
    bool isEnabled(LogLevel lv) const;

    /// 输出一条日志消息。
    template <typename... Args> void log(LogLevel lv, Args&&... args) {
        if (!isEnabled(lv))
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
        case LogLevel::Trace:
            return "[TRACE] ";
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

    Options _options;
    mutable std::mutex _mu;
};

#define IR_STRINGIZE_IMPL(x) #x
#define IR_STRINGIZE(x) IR_STRINGIZE_IMPL(x)
#define IR_LOG_LOCATION __FILE__ ":" IR_STRINGIZE(__LINE__)

#define IR_LOG_IMPL(level, ...)                                                                    \
    do {                                                                                           \
        auto& ir_logger = ::ir::Logger::instance();                                                \
        if (ir_logger.isEnabled(level)) {                                                          \
            ir_logger.log(level, IR_LOG_LOCATION, " | ", __VA_ARGS__);                           \
        }                                                                                          \
    } while (false)

#define IR_LOG_TRACE(...) IR_LOG_IMPL(::ir::LogLevel::Trace, __VA_ARGS__)
#define IR_LOG_DEBUG(...) IR_LOG_IMPL(::ir::LogLevel::Debug, __VA_ARGS__)
#define IR_LOG_INFO(...) IR_LOG_IMPL(::ir::LogLevel::Info, __VA_ARGS__)
#define IR_LOG_WARN(...) IR_LOG_IMPL(::ir::LogLevel::Warn, __VA_ARGS__)
#define IR_LOG_ERROR(...) IR_LOG_IMPL(::ir::LogLevel::Error, __VA_ARGS__)

} // namespace ir
