#pragma once

#include <memory>
#include <string>
#include <vector>

#include "interfaces/i_filter.h"
#include "interfaces/i_geometry_estimator.h"
#include "interfaces/i_keypoint_extractor.h"
#include "interfaces/i_matcher.h"
#include "pipeline/base_pipeline.h"

namespace ir {

/// 点特征配准流水线。
class KeypointPipeline : public BasePipeline {
public:
    KeypointPipeline() = default;

    /// 返回流水线名称，用于日志、摘要和输出文件命名。
    std::string name() const override { return "KeypointPipeline"; }

protected:
    /// 清空上一轮配置创建的点特征阶段组件。
    void resetStages() override;

    /// 根据 pipeline 配置创建点特征提取器、匹配器、过滤器链和几何估计器。
    bool configureStages(const PipelineConfig& cfg) override;

    /// 执行关键点检测与描述子计算，并回填关键点数量统计。
    bool runExtraction(RegistrationContext& ctx) override;

    /// 执行匹配与过滤阶段，生成进入几何估计的候选匹配。
    /// 其中 raw 来自匹配器原始输出，filtered 来自 runFilters() 的最终结果。
    bool runAssociation(RegistrationContext& ctx) override;

    /// 基于过滤后的匹配估计几何模型，并回填内点统计。
    /// 几何阶段会把最终确认的 inliers / inlier_mask 写回上下文。
    bool runEstimation(RegistrationContext& ctx) override;

    /// 保存点特征专属可视化输出，再委托基类保存通用 warp 输出。
    bool saveOutputs(RegistrationContext& ctx) override;

    /// 生成带有提取器、几何模型和匹配器标签的输出文件名前缀。
    std::string buildOutputStem(const RegistrationContext& ctx) const override;

private:
    /// 调用匹配器生成原始匹配结果，并统计原始候选规模。
    /// 结果写入 KeypointMatchData::raw_matches，并按需保留邻居列表。
    bool runMatch(RegistrationContext& ctx);

    /// 按配置顺序执行匹配过滤器链，并统计过滤后规模。
    /// 若匹配器只提供按 query 分组的候选，会先以每行 top-1 种子生成 filtered。
    bool runFilters(RegistrationContext& ctx);

    std::shared_ptr<IKeypointExtractor> _extractor;
    std::shared_ptr<IMatcher> _matcher;
    std::vector<std::shared_ptr<IFilter>> _filters;
    std::shared_ptr<IGeometryEstimator> _geometry;
};

} // namespace ir

