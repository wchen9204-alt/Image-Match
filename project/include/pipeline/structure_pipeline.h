#pragma once

#include <memory>
#include <string>
#include <vector>

#include "interfaces/i_filter.h"
#include "interfaces/i_geometry_estimator.h"
#include "interfaces/i_structure_associator.h"
#include "interfaces/i_structure_extractor.h"
#include "pipeline/base_pipeline.h"

namespace ir {

/// 结构特征配准流水线。
///
/// 负责组织边缘、直线、轮廓等结构方法的完整流程，包括结构提取、
/// 结构关联、过滤链、几何估计以及结构响应图可视化输出。
class StructurePipeline : public BasePipeline {
public:
    StructurePipeline() = default;

    /// 返回流水线名称，用于日志、摘要和输出文件命名。
    std::string name() const override { return "StructurePipeline"; }

protected:
    /// 清空上一轮配置创建的结构阶段组件。
    void resetStages() override;

    /// 根据 pipeline 配置创建结构特征提取器、关联器和过滤器链。
    bool configureStages(const PipelineConfig& cfg) override;

    /// 执行边缘、直线或轮廓提取，并回填结构数量统计。
    bool runExtraction(RegistrationContext& ctx) override;

    /// 执行结构关联/匹配 + 过滤链，将结果写入 `structure_match_data`。
    bool runAssociation(RegistrationContext& ctx) override;

    /// 根据结构匹配结果估计当前的空间变换模型。
    bool runEstimation(RegistrationContext& ctx) override;

    /// 保存结构响应图可视化输出，再委托基类保存通用 warp 输出。
    bool saveOutputs(RegistrationContext& ctx) override;

    /// 生成带有结构类型和关联策略标签的输出文件名前缀。
    std::string buildOutputStem(const RegistrationContext& ctx) const override;

private:
    /// 当前结构特征提取器，由 `structure/*.yaml` 中的 `type` 字段决定。
    std::shared_ptr<IStructureExtractor> _extractor;

    /// 当前结构关联/匹配器，由 `association.method` 字段决定。
    std::shared_ptr<IStructureAssociator> _associator;

    /// 当前几何估计器，由 pipeline YAML 中的 `geometry` 字段决定。
    std::shared_ptr<IGeometryEstimator> _geometry;

    /// 过滤链，从结构配置的 `filters:` 列表中按序创建。
    std::vector<std::shared_ptr<IFilter>> _filters;

    /// 对接 IFilter 链：将 raw_matches_knn → filtered_matches → line_matches。
    bool runFilters(RegistrationContext& ctx);
};

} // namespace ir
