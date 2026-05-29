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

    /// 执行完整的配准流程。
    virtual bool run(RegistrationContext& ctx) = 0;
};

} // namespace ir
