#include "learning/python_learning_matcher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <opencv2/core.hpp>

#include "core/config.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

struct LearningMatch {
    /// 源图坐标。
    cv::Point2f p1;
    /// 目标图坐标。
    cv::Point2f p2;
    /// 模型输出的匹配置信度，越大表示越可信。
    double confidence = 1.0;
};

/// 对文件名/worker key 做保守清洗，避免方法名或路径字符进入临时目录名。
std::string sanitizeToken(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "worker" : out;
}

/// 为 request JSON 转义字符串字段。
std::string jsonEscape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"':  out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:   out.push_back(c); break;
        }
    }
    return out;
}

/// 将路径或命令参数包裹为命令行参数，避免空格路径被拆分。
std::string quoteArg(const fs::path& value) {
    std::string s = value.string();
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') {
            out += "\\\"";
        } else {
            out.push_back(c);
        }
    }
    out += "\"";
    return out;
}

/// 将普通字符串按路径参数同样转义，供 Python 可执行文件和权重预设复用。
std::string quoteArg(const std::string& value) {
    return quoteArg(fs::path(value));
}

/// 从 JSON/YAML 节点读取 double 字段；字段缺失时使用 fallback。
double readDouble(const YAML::Node& node, const std::string& key, double fallback) {
    return yaml_utils::get<double>(node, key, fallback);
}

/// 执行 Python 推理命令并返回进程退出码；Windows 下绕开 cmd.exe 的引号解析。
int runProcessCommand(const std::string& command) {
#ifdef _WIN32
    STARTUPINFOA si{};
    PROCESS_INFORMATION pi{};
    si.cb = sizeof(si);

    // 1. CreateProcessA 需要可写命令行缓冲区，因此先复制到 vector<char>。
    std::vector<char> commandLine(command.begin(), command.end());
    commandLine.push_back('\0');

    // 2. 直接启动 Python 命令行，避免 std::system/cmd.exe 错误解析带引号路径。
    if (!CreateProcessA(nullptr,
                        commandLine.data(),
                        nullptr,
                        nullptr,
                        TRUE,
                        0,
                        nullptr,
                        nullptr,
                        &si,
                        &pi)) {
        const DWORD err = GetLastError();
        IR_LOG_ERROR("Learning matcher failed to start process, Win32 error=", err);
        return -1;
    }

    // 3. 等待 Python 推理完成，并把子进程退出码传回 C++ pipeline。
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 1;
    if (!GetExitCodeProcess(pi.hProcess, &exitCode)) {
        exitCode = 1;
    }

    // 4. 回收 Win32 句柄，避免批处理长时间运行时积累系统资源。
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return static_cast<int>(exitCode);
#else
    // 非 Windows 平台保留 std::system 路径，复用 shell 自身的命令行解析。
    return std::system(command.c_str());
#endif
}

/// 常驻 Python worker 子进程；同一学习配置在 batch 内复用一个进程和一份模型权重。
class LearningWorkerProcess {
public:
    LearningWorkerProcess(std::string key, std::string command, fs::path requestDir)
        : _key(std::move(key)), _command(std::move(command)), _requestDir(std::move(requestDir)) {}

    LearningWorkerProcess(const LearningWorkerProcess&) = delete;
    LearningWorkerProcess& operator=(const LearningWorkerProcess&) = delete;

    ~LearningWorkerProcess() { stop(); }

    /// 启动 worker 进程；若已经启动则直接复用。
    bool ensureStarted() {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_started) {
#ifdef _WIN32
            // 已启动但进程提前退出时，先回收旧句柄，再允许下面重新拉起 worker。
            if (WaitForSingleObject(_pi.hProcess, 0) == WAIT_OBJECT_0) {
                closeHandles();
                _started = false;
            } else {
                return true;
            }
#else
            return true;
#endif
        }

        // 1. 清理旧 stop 标记并创建 request 目录，保证 worker 进入干净轮次。
        std::error_code ec;
        fs::create_directories(_requestDir, ec);
        fs::remove(_requestDir / "stop", ec);
        // 2. 清理上次异常退出遗留的 request/response，避免新 worker 重放旧样本。
        for (const auto& entry : fs::directory_iterator(_requestDir, ec)) {
            if (ec) {
                break;
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind("request_", 0) == 0 || name.rfind("response_", 0) == 0) {
                fs::remove(entry.path(), ec);
            }
        }

#ifdef _WIN32
        STARTUPINFOA si{};
        si.cb = sizeof(si);
        std::vector<char> commandLine(_command.begin(), _command.end());
        commandLine.push_back('\0');

        // 3. 直接启动常驻 Python backend；模型加载后进程保持存活处理多个 request。
        if (!CreateProcessA(nullptr,
                            commandLine.data(),
                            nullptr,
                            nullptr,
                            TRUE,
                            0,
                            nullptr,
                            nullptr,
                            &si,
                            &_pi)) {
            IR_LOG_ERROR("Learning worker failed to start, key=", _key, ", Win32 error=", GetLastError());
            return false;
        }
        _started = true;
        IR_LOG_INFO("Learning worker started: key=", _key);
        return true;
#else
        IR_LOG_ERROR("Learning worker mode is currently implemented for Windows only.");
        return false;
#endif
    }

    /// 写入一个 request，等待 worker 完成并返回 response 节点。
    bool request(const std::string& requestJson,
                 int timeoutMs,
                 YAML::Node& response,
                 std::string& message) {
        if (!ensureStarted()) {
            message = "failed to start learning worker";
            return false;
        }

        const uint64_t id = _nextRequestId.fetch_add(1);
        const fs::path requestPath = _requestDir / ("request_" + std::to_string(id) + ".json");
        const fs::path responsePath = _requestDir / ("response_" + std::to_string(id) + ".json");

        // 1. request 先写临时文件再 rename，避免 worker 读取到半截 JSON。
        const fs::path tmpPath = requestPath.string() + ".tmp";
        {
            std::ofstream out(tmpPath, std::ios::binary);
            out << requestJson;
        }
        std::error_code ec;
        fs::rename(tmpPath, requestPath, ec);
        if (ec) {
            message = "failed to publish worker request: " + ec.message();
            return false;
        }

        // 2. 轮询 response 文件；超时用于防止 worker 崩溃或模型卡死导致 C++ 永久阻塞。
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(std::max(1, timeoutMs));
        while (std::chrono::steady_clock::now() < deadline) {
            if (fs::exists(responsePath)) {
                try {
                    response = YAML::LoadFile(responsePath.string());
                } catch (const YAML::Exception& e) {
                    message = e.what();
                    return false;
                }
                fs::remove(responsePath, ec);
                return true;
            }

#ifdef _WIN32
            if (_started && WaitForSingleObject(_pi.hProcess, 0) == WAIT_OBJECT_0) {
                message = "learning worker exited before writing response";
                _started = false;
                return false;
            }
#endif
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        message = "learning worker request timed out";
        return false;
    }

    /// 通知 worker 退出，并在短暂等待后回收进程句柄。
    void stop() {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_started) {
            return;
        }

        // 1. stop 文件是 worker 的正常退出信号。
        std::error_code ec;
        fs::create_directories(_requestDir, ec);
        std::ofstream(_requestDir / "stop").close();

#ifdef _WIN32
        // 2. 给 Python 清理窗口；若仍未退出则终止，避免应用结束时遗留后台进程。
        const DWORD wait = WaitForSingleObject(_pi.hProcess, 5000);
        if (wait == WAIT_TIMEOUT) {
            TerminateProcess(_pi.hProcess, 1);
            WaitForSingleObject(_pi.hProcess, 1000);
        }
        closeHandles();
#endif
        _started = false;
    }

private:
#ifdef _WIN32
    /// 关闭 Win32 进程句柄；正常退出和异常退出检测都走同一处清理。
    void closeHandles() {
        if (_pi.hThread) {
            CloseHandle(_pi.hThread);
        }
        if (_pi.hProcess) {
            CloseHandle(_pi.hProcess);
        }
        _pi = PROCESS_INFORMATION{};
    }
#endif

    std::string _key;
    std::string _command;
    fs::path _requestDir;
    std::mutex _mutex;
    std::atomic<uint64_t> _nextRequestId{1};
    bool _started = false;
#ifdef _WIN32
    PROCESS_INFORMATION _pi{};
#endif
};

/// 全局 worker 池；batch 中反复创建 matcher 时仍能按配置 key 复用 Python 进程。
std::shared_ptr<LearningWorkerProcess> getWorker(const std::string& key,
                                                 const std::string& command,
                                                 const fs::path& requestDir) {
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<LearningWorkerProcess>> workers;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = workers.find(key);
    if (it != workers.end()) {
        return it->second;
    }

    auto worker = std::make_shared<LearningWorkerProcess>(key, command, requestDir);
    workers.emplace(key, worker);
    return worker;
}

} // namespace

PythonLearningMatcher::PythonLearningMatcher(const YAML::Node& cfg, const fs::path& configDir)
    : _configDir(configDir) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : YAML::Node();
    _method = string_utils::toUpperAscii(yaml_utils::getString(cfg, "method", "LEARNING"));
    _pythonExecutable =
        yaml_utils::getString(cfg, "python", yaml_utils::getString(cfg, "python_executable", "python"));
    _scriptPath = resolveConfigPath(yaml_utils::getString(cfg, "script"));
    _mode = string_utils::normalizedKey(yaml_utils::getString(cfg, "mode", "single"));
    if (_mode.empty()) {
        _mode = "SINGLE";
    }
    _weightsArg = resolveModelArgument(yaml_utils::getString(cfg, "weights"));
    _minConfidence = yaml_utils::getDouble(params, "min_confidence", 0.0);
    _maxMatches = yaml_utils::getInt(params, "max_matches", 0);
    _resize = yaml_utils::getInt(params, "resize", yaml_utils::getInt(params, "image_size", 0));
    _workerTimeoutMs = yaml_utils::getInt(params, "worker_timeout_ms", 600000);
}

fs::path PythonLearningMatcher::resolveConfigPath(const std::string& value) const {
    if (value.empty()) {
        return {};
    }
    return Config::resolvePath(_configDir, value);
}

std::string PythonLearningMatcher::resolveModelArgument(const std::string& value) const {
    if (value.empty()) {
        return {};
    }

    const fs::path rawPath(value);
    const fs::path resolved = resolveConfigPath(value);
    const bool hasSeparator = value.find('/') != std::string::npos ||
                              value.find('\\') != std::string::npos;
    const bool pathLike = rawPath.is_absolute() || hasSeparator || rawPath.has_extension();
    // 权重字段既可能是文件路径，也可能是脚本自己的预设名；只有路径语义明确时才做路径解析。
    if (pathLike || fs::exists(resolved)) {
        return resolved.string();
    }
    return value;
}

fs::path PythonLearningMatcher::outputJsonPath(const RegistrationContext& ctx) const {
    fs::path root = ctx.output_dir;
    if (root.empty()) {
        root = fs::temp_directory_path() / "image_registration_learning";
    }
    const fs::path dir = root / "learning";
    std::error_code ec;
    fs::create_directories(dir, ec);
    const std::string stem =
        ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string() + "_" + _method;
    return dir / (stem + "_matches.json");
}

std::string PythonLearningMatcher::buildCommand(const RegistrationContext& ctx,
                                                const fs::path& outputPath) const {
    std::ostringstream cmd;
    cmd << quoteArg(_pythonExecutable)
        << " " << quoteArg(_scriptPath)
        << " --mode single"
        << " --method " << quoteArg(_method)
        << " --image1 " << quoteArg(ctx.image1_path)
        << " --image2 " << quoteArg(ctx.image2_path)
        << " --output " << quoteArg(outputPath)
        << " --min-confidence " << _minConfidence;
    if (!_weightsArg.empty()) {
        cmd << " --weights " << quoteArg(_weightsArg);
    }
    if (_maxMatches > 0) {
        cmd << " --max-matches " << _maxMatches;
    }
    if (_resize > 0) {
        cmd << " --resize " << _resize;
    }
    return cmd.str();
}

std::string PythonLearningMatcher::buildWorkerCommand(const fs::path& requestDir) const {
    std::ostringstream cmd;
    cmd << quoteArg(_pythonExecutable)
        << " " << quoteArg(_scriptPath)
        << " --mode worker"
        << " --method " << quoteArg(_method)
        << " --request-dir " << quoteArg(requestDir)
        << " --min-confidence " << _minConfidence;
    if (!_weightsArg.empty()) {
        cmd << " --weights " << quoteArg(_weightsArg);
    }
    if (_maxMatches > 0) {
        cmd << " --max-matches " << _maxMatches;
    }
    if (_resize > 0) {
        cmd << " --resize " << _resize;
    }
    return cmd.str();
}

bool PythonLearningMatcher::runWorkerRequest(const RegistrationContext& ctx,
                                             const fs::path& outputPath) const {
    const std::string keySeed = _method + "_" + _scriptPath.string() + "_" + _weightsArg + "_" +
                                std::to_string(_maxMatches) + "_" + std::to_string(_resize);
    // worker key 进入临时目录名，使用短 hash 避免 Windows 路径过长。
    const std::string key = sanitizeToken(_method) + "_" +
                            std::to_string(std::hash<std::string>{}(keySeed));
    const fs::path requestDir = fs::temp_directory_path() /
                                "image_registration_learning_workers" / key;
    const std::string command = buildWorkerCommand(requestDir);
    auto worker = getWorker(key, command, requestDir);

    // 1. request 只描述单个样本；模型、权重和阈值由 worker 启动参数固定。
    std::ostringstream req;
    req << "{\n"
        << "  \"image1\": \"" << jsonEscape(ctx.image1_path.string()) << "\",\n"
        << "  \"image2\": \"" << jsonEscape(ctx.image2_path.string()) << "\",\n"
        << "  \"output\": \"" << jsonEscape(outputPath.string()) << "\",\n"
        << "  \"min_confidence\": " << _minConfidence << ",\n"
        << "  \"max_matches\": " << _maxMatches << ",\n"
        << "  \"resize\": " << _resize << "\n"
        << "}\n";

    // 2. 等待 response，并把 Python 侧失败原因写入日志。
    YAML::Node response;
    std::string message;
    if (!worker->request(req.str(), _workerTimeoutMs, response, message)) {
        IR_LOG_ERROR("Learning worker request failed: ", message);
        return false;
    }
    const bool ok = response["ok"] && response["ok"].as<bool>();
    if (!ok) {
        IR_LOG_ERROR("Learning worker returned failure: ",
                     response["message"] ? response["message"].as<std::string>() : std::string("unknown"));
        return false;
    }
    return true;
}

bool PythonLearningMatcher::loadMatchesJson(const fs::path& path, RegistrationContext& ctx) const {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path.string());
    } catch (const YAML::Exception& e) {
        IR_LOG_ERROR("Learning matcher failed to parse output JSON: ", e.what());
        return false;
    }

    const YAML::Node matchesNode = root["matches"];
    if (!matchesNode || !matchesNode.IsSequence()) {
        IR_LOG_ERROR("Learning matcher output missing 'matches' sequence: ", path.string());
        return false;
    }

    std::vector<LearningMatch> matches;
    matches.reserve(matchesNode.size());
    for (const auto& item : matchesNode) {
        const double confidence = readDouble(item, "confidence", 1.0);
        if (!std::isfinite(confidence) || confidence < _minConfidence) {
            continue;
        }

        LearningMatch match;
        match.p1.x = static_cast<float>(readDouble(item, "x1", 0.0));
        match.p1.y = static_cast<float>(readDouble(item, "y1", 0.0));
        match.p2.x = static_cast<float>(readDouble(item, "x2", 0.0));
        match.p2.y = static_cast<float>(readDouble(item, "y2", 0.0));
        match.confidence = confidence;
        if (!std::isfinite(match.p1.x) || !std::isfinite(match.p1.y) ||
            !std::isfinite(match.p2.x) || !std::isfinite(match.p2.y)) {
            continue;
        }
        matches.push_back(match);
    }

    std::stable_sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
        return a.confidence > b.confidence;
    });
    if (_maxMatches > 0 && static_cast<int>(matches.size()) > _maxMatches) {
        matches.resize(static_cast<size_t>(_maxMatches));
    }
    if (matches.empty()) {
        IR_LOG_ERROR("Learning matcher produced no usable matches after confidence filtering.");
        return false;
    }

    auto& kd = ctx.keypoint_data;
    auto& md = ctx.keypoint_match_data;
    kd.clear();
    md.clear();
    // 学习模型直接输出稀疏点对；这里显式转换为平台已有的点对容器，
    // 实际来源由 ctx.correspondence_source / CorrespondenceView 标记为 LEARNING。
    kd.type = KeypointType::UNKNOWN;
    kd.norm_type = NormType::L2;
    kd.first.keypoints.reserve(matches.size());
    kd.second.keypoints.reserve(matches.size());
    kd.first.descriptors = cv::Mat(static_cast<int>(matches.size()), 1, CV_32F);
    kd.second.descriptors = cv::Mat(static_cast<int>(matches.size()), 1, CV_32F);
    md.match_method = MatchMethod::MATCH;
    md.raw_matches.reserve(matches.size());
    md.filtered_matches.reserve(matches.size());

    for (size_t i = 0; i < matches.size(); ++i) {
        const auto& match = matches[i];
        kd.first.keypoints.emplace_back(match.p1, 1.0f);
        kd.second.keypoints.emplace_back(match.p2, 1.0f);
        kd.first.descriptors.at<float>(static_cast<int>(i), 0) =
            static_cast<float>(match.confidence);
        kd.second.descriptors.at<float>(static_cast<int>(i), 0) =
            static_cast<float>(match.confidence);

        const float distance =
            static_cast<float>(std::max(0.0, 1.0 - std::clamp(match.confidence, 0.0, 1.0)));
        cv::DMatch dmatch(static_cast<int>(i), static_cast<int>(i), distance);
        md.filtered_matches.push_back(dmatch);
        md.raw_matches.push_back(dmatch);
    }

    IR_LOG_INFO("Learning matcher loaded ",
                matches.size(),
                " matches from ",
                path.string());
    return true;
}

bool PythonLearningMatcher::match(RegistrationContext& ctx) {
    if (_scriptPath.empty() || !fs::exists(_scriptPath)) {
        IR_LOG_ERROR("Learning matcher script not found: ", _scriptPath.string());
        return false;
    }

    const fs::path outputPath = outputJsonPath(ctx);
    if (_mode == "WORKER") {
        IR_LOG_INFO("Running learning matcher through persistent worker: method=", _method);
        if (!runWorkerRequest(ctx, outputPath)) {
            return false;
        }
    } else {
        const std::string command = buildCommand(ctx, outputPath);
        IR_LOG_INFO("Running learning matcher: ", command);
        const int code = runProcessCommand(command);
        if (code != 0) {
            IR_LOG_ERROR("Learning matcher command failed with code ", code);
            return false;
        }
    }
    if (!fs::exists(outputPath)) {
        IR_LOG_ERROR("Learning matcher did not write output: ", outputPath.string());
        return false;
    }
    return loadMatchesJson(outputPath, ctx);
}

} // namespace ir


