#pragma once

#include <memory>
#include <string>

#include "core/config.h"
#include "interfaces/i_pipeline.h"
#include "transform/warper.h"

namespace ir {

/// 通用配准流水线骨架。
///
/// 负责组织一次完整配准流程的公共部分：配置保存、图像加载、阶段调度、
/// 变换输出、窗口显示和耗时统计。具体的提取、关联和模型估计由子类实现。
class BasePipeline : public IPipeline {
public:
    BasePipeline() = default;
    ~BasePipeline() override = default;

    /// 保存 pipeline 配置，并委托子类装配算法阶段组件。
    bool configure(const PipelineConfig& cfg) override;

    /// 执行完整配准流程，结果统一写入 `RegistrationContext`。
    bool run(RegistrationContext& ctx) override;

protected:
    /// 子类清空上一次配置产生的阶段组件。
    virtual void resetStages() {}

    /// 子类根据配置创建本方法族需要的阶段组件。
    virtual bool configureStages(const PipelineConfig& cfg) = 0;

    /// 加载输入图像，并准备显示用彩色图与计算用灰度图。
    virtual bool loadImages(RegistrationContext& ctx);

    /// 执行方法族自己的特征或结构提取阶段。
    virtual bool runExtraction(RegistrationContext& ctx) = 0;

    /// 执行候选关联阶段，例如描述子匹配、结构匹配或距离场对齐。
    virtual bool runAssociation(RegistrationContext& ctx) = 0;

    /// 根据前序关联结果估计空间模型，并统计质量信息。
    virtual bool runEstimation(RegistrationContext& ctx) = 0;

    /// 在几何模型有效时生成配准后的图像结果。
    virtual bool runWarp(RegistrationContext& ctx);

    /// 保存通用输出，子类可扩展自己的阶段可视化。
    virtual bool saveOutputs(RegistrationContext& ctx);

    /// 按配置弹出调试窗口，便于交互式观察结果。
    virtual bool showWindows(RegistrationContext& ctx);

    /// 生成通用输出文件名前缀，子类可加入算法标签。
    virtual std::string buildOutputStem(const RegistrationContext& ctx) const;

    /// 当前流水线配置，作为各阶段的统一输入来源。
    PipelineConfig _config;

    /// 变换阶段组件，负责将估计结果落到实际图像重采样。
    std::shared_ptr<IWarper> _warper;
};

} // namespace ir
