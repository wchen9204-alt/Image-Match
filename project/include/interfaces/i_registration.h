#pragma once

#include <string>

#include "core/config.h"
#include "core/context.h"

namespace ir {

/// 配准任务的顶层接口。
///
/// 该接口通常由 `Registration` 实现，用于向外暴露配置和执行入口。
class IRegistration {
public:
    virtual ~IRegistration() = default;

    /// 返回注册实例名称。
    virtual std::string name() const = 0;

    /// 应用 pipeline 配置。
    virtual bool configure(const PipelineConfig& cfg) = 0;

    /// 执行一次完整的配准流程。
    virtual bool run(RegistrationContext& ctx) = 0;
};

} // namespace ir
