#pragma once

#include <string>

#include "core/config.h"
#include "core/context.h"

namespace ir {

/// 配准流水线接口。
///
/// 负责串联特征提取、匹配、过滤、几何估计和变换输出等阶段。
class IPipeline {
public:
    virtual ~IPipeline() = default;

    /// 返回流水线名称。
    virtual std::string name() const = 0;

    /// 根据配置初始化流水线各阶段组件。
    virtual bool configure(const PipelineConfig& cfg) = 0;

    /// 使用本次运行的输入输出路径执行流程；省略参数时回退到 YAML 配置。
    virtual bool run(RegistrationContext& ctx,
                     const PipelineRunOptions& options = PipelineRunOptions{}) = 0;

    /// 按配置显示调试窗口，便于在运行后观察结果。
    virtual bool showWindows(RegistrationContext& ctx) = 0;
};

} // namespace ir

