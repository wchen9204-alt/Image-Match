#pragma once

#include <memory>
#include <string>

#include "core/config.h"
#include "core/context.h"
#include "interfaces/i_pipeline.h"
#include "interfaces/i_registration.h"

namespace ir {

/// 基于可配置流水线的图像配准入口类。
///
/// Registration 负责保存解析后的流水线配置，并把实际处理工作交给
/// IPipeline 的具体实现。
class Registration : public IRegistration {
public:
    /// 构造一个尚未配置的注册实例。
    Registration() = default;

    /// 构造一个绑定到指定流水线的注册实例。
    explicit Registration(std::shared_ptr<IPipeline> pipeline);

    /// 返回当前注册实例的名称。
    std::string name() const override;

    /// 保存流水线配置并初始化底层流水线。
    bool configure(const PipelineConfig& cfg) override;

    /// 在给定的注册上下文中执行已配置的流水线。
    bool run(RegistrationContext& ctx) override;

    /// 返回当前保存的流水线配置。
    const PipelineConfig& config() const { return _cfg; }

    /// 返回当前用于执行配准流程的流水线对象。
    std::shared_ptr<IPipeline> pipeline() const { return _pipeline; }

private:
    /// 最近一次应用到该实例的配置。
    PipelineConfig _cfg;

    /// 执行特征提取、匹配和变换的具体流水线实现。
    std::shared_ptr<IPipeline> _pipeline;
};

} // namespace ir

