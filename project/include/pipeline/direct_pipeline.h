#pragma once

#include <memory>
#include <string>

#include "interfaces/i_direct_aligner.h"
#include "pipeline/base_pipeline.h"

namespace ir {

/// 直接法配准流水线。
class DirectPipeline : public BasePipeline {
public:
    DirectPipeline() = default;

    /// 返回流水线名称，用于日志、摘要和输出文件命名。
    std::string name() const override { return "DirectPipeline"; }

protected:
    /// 清空上一轮配置创建的直接法配准器。
    void resetStages() override;

    /// 根据 pipeline 配置创建直接法配准器。
    bool configureStages(const PipelineConfig& cfg) override;

    /// 直接法不需要独立特征提取阶段，这里作为公共流水线占位。
    bool runExtraction(RegistrationContext& ctx) override;

    /// 直接法不需要描述子关联阶段，这里作为公共流水线占位。
    bool runAssociation(RegistrationContext& ctx) override;

    /// 执行直接法配准器，并同步几何结果到通用上下文。
    bool runEstimation(RegistrationContext& ctx) override;

    /// 保存直接法专属输出，再委托基类保存通用 warp 输出。
    bool saveOutputs(RegistrationContext& ctx) override;

    /// 生成带直接法算法标签的输出文件名前缀。
    std::string buildOutputStem(const RegistrationContext& ctx) const override;

private:
    /// 当前 pipeline 配置创建出的直接法配准器实例。
    std::shared_ptr<IDirectAligner> _aligner;
};

} // namespace ir
