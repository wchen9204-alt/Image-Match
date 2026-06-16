#pragma once

#include <filesystem>
#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_learning_matcher.h"

namespace ir {

/// 通过外部 Python 脚本执行深度模型推理，并读取统一 matches JSON。
class PythonLearningMatcher : public ILearningMatcher {
public:
    /// 从 learning YAML 节点读取 Python 脚本、权重参数和匹配过滤参数。
    PythonLearningMatcher(const YAML::Node& cfg, const std::filesystem::path& configDir);

    /// 返回当前深度学习匹配方法名。
    std::string name() const override { return _method; }
    /// 执行 Python 推理脚本并把输出 JSON 转换为 RegistrationContext 中的学习点对。
    bool match(RegistrationContext& ctx) override;

private:
    /// 按 learning 配置所在目录解析脚本或权重路径。
    std::filesystem::path resolveConfigPath(const std::string& value) const;
    /// 解析模型参数；既支持真实权重路径，也支持脚本预设名。
    std::string resolveModelArgument(const std::string& value) const;
    /// 根据当前样本和输出目录生成 Python matches JSON 路径。
    std::filesystem::path outputJsonPath(const RegistrationContext& ctx) const;
    /// 组装 Python 推理命令行。
    std::string buildCommand(const RegistrationContext& ctx,
                             const std::filesystem::path& outputPath) const;
    /// 组装 Python worker 启动命令行。
    std::string buildWorkerCommand(const std::filesystem::path& requestDir) const;
    /// 通过常驻 Python worker 执行一次推理请求。
    bool runWorkerRequest(const RegistrationContext& ctx,
                          const std::filesystem::path& outputPath) const;
    /// 读取 Python 输出 JSON，并写入 keypoint_data/keypoint_match_data 兼容容器。
    bool loadMatchesJson(const std::filesystem::path& path, RegistrationContext& ctx) const;

    /// 当前学习配置文件所在目录，用于解析相对脚本/权重路径。
    std::filesystem::path _configDir;
    /// 方法名，写入日志、输出文件名和结果摘要。
    std::string _method = "LEARNING";
    /// Python 可执行文件路径或命令名，默认使用系统 python。
    std::string _pythonExecutable = "python";
    /// 实际执行推理的 Python 脚本路径。
    std::filesystem::path _scriptPath;
    /// Python backend 模式；SINGLE 每次启动脚本，WORKER 复用常驻模型进程。
    std::string _mode = "SINGLE";
    /// 可选模型参数；可以是权重路径，也可以是脚本支持的预设名（如 SuperGlue 的 outdoor/indoor）。
    std::string _weightsArg;
    /// 最小置信度阈值，低于该值的学习匹配会被丢弃。
    double _minConfidence = 0.0;
    /// 最大保留匹配数；小于等于 0 表示不额外截断。
    int _maxMatches = 0;
    /// 推理前长边缩放尺寸；小于等于 0 表示脚本按原图处理。
    int _resize = 0;
    /// worker 请求等待超时；用于避免 Python worker 异常时 C++ 无限等待。
    int _workerTimeoutMs = 600000;
};

} // namespace ir

