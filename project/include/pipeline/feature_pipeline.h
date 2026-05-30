#pragma once

#include <memory>
#include <string>
#include <vector>

#include "interfaces/i_feature_extractor.h"
#include "interfaces/i_filter.h"
#include "interfaces/i_geometry_estimator.h"
#include "interfaces/i_matcher.h"
#include "pipeline/base_pipeline.h"

namespace ir {

/// 点特征配准流水线。
///
/// 负责组织 keypoint/descriptor 方法族的完整流程，包括特征提取、描述子匹配、
/// 匹配过滤、几何估计，以及关键点和匹配关系可视化输出。
class FeaturePipeline : public BasePipeline {
public:
    FeaturePipeline() = default;

    /// 返回流水线名称，用于日志、摘要和输出文件命名。
    std::string name() const override { return "FeaturePipeline"; }

protected:
    /// 清空上一次配置创建的点特征阶段组件。
    void resetStages() override;

    /// 根据 pipeline 配置创建点特征提取器、匹配器、过滤器链和几何估计器。
    bool configureStages(const PipelineConfig& cfg) override;

    /// 执行关键点检测与描述子计算，并回填关键点数量统计。
    bool runExtraction(RegistrationContext& ctx) override;

    /// 执行描述子匹配和匹配过滤，生成进入几何估计的候选匹配。
    bool runAssociation(RegistrationContext& ctx) override;

    /// 基于过滤后的匹配估计几何模型，并回填内点统计。
    bool runEstimation(RegistrationContext& ctx) override;

    /// 保存点特征专属可视化输出，再委托基类保存通用 warp 输出。
    bool saveOutputs(RegistrationContext& ctx) override;

    /// 生成带有特征、几何模型和匹配器标签的输出文件名前缀。
    std::string buildOutputStem(const RegistrationContext& ctx) const override;

private:
    /// 调用匹配器生成原始匹配结果，并统计原始候选规模。
    bool runMatch(RegistrationContext& ctx);

    /// 按配置顺序执行匹配过滤器链，并统计过滤后的候选规模。
    bool runFilters(RegistrationContext& ctx);

    /// 特征提取阶段组件，负责生成关键点与描述子。
    std::shared_ptr<IFeatureExtractor> _extractor;

    /// 匹配阶段组件，负责生成原始近邻匹配结果。
    std::shared_ptr<IMatcher> _matcher;

    /// 过滤阶段组件链，按配置顺序串行收缩匹配集合。
    std::vector<std::shared_ptr<IFilter>> _filters;

    /// 几何估计阶段组件，负责求解模型参数与内点掩码。
    std::shared_ptr<IGeometryEstimator> _geometry;
};

} // namespace ir
