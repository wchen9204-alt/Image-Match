#pragma once

#include <memory>
#include <string>

#include "interfaces/i_structure_extractor.h"
#include "pipeline/base_pipeline.h"

namespace ir {

/// 结构特征配准流水线。
///
/// 负责组织边缘、直线、轮廓等结构方法的完整流程，包括结构提取、结构响应关联、
/// 几何估计，以及结构响应图可视化输出。
class StructurePipeline : public BasePipeline {
public:
    StructurePipeline() = default;

    /// 返回流水线名称，用于日志、摘要和输出文件命名。
    std::string name() const override { return "StructurePipeline"; }

protected:
    /// 清空上一次配置创建的结构特征阶段组件和估计参数。
    void resetStages() override;

    /// 根据 pipeline 配置创建结构特征提取器，并读取结构估计参数。
    bool configureStages(const PipelineConfig& cfg) override;

    /// 执行边缘、直线或轮廓提取，并回填结构数量统计。
    bool runExtraction(RegistrationContext& ctx) override;

    /// 检查结构响应图是否可用于估计，并为后续估计阶段准备统计字段。
    bool runAssociation(RegistrationContext& ctx) override;

    /// 基于结构响应图估计当前的空间变换模型。
    bool runEstimation(RegistrationContext& ctx) override;

    /// 保存结构响应图可视化输出，再委托基类保存通用 warp 输出。
    bool saveOutputs(RegistrationContext& ctx) override;

    /// 生成带有结构类型和估计策略标签的输出文件名前缀。
    std::string buildOutputStem(const RegistrationContext& ctx) const override;

private:
    /// 当前结构特征提取器，由 `structure/*.yaml` 中的 `type` 字段决定。
    std::shared_ptr<IStructureExtractor> _extractor;

    /// 相位相关响应阈值，低于该值时认为结构配准不可靠。
    double _responseThreshold = 0.01;

    /// 结构响应图进入相位相关估计前的高斯模糊核大小。
    int _phaseBlurKernel = 5;
};

} // namespace ir
