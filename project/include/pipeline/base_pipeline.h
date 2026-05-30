#pragma once

#include <memory>
#include <vector>

#include "core/config.h"
#include "interfaces/i_feature_extractor.h"
#include "interfaces/i_filter.h"
#include "interfaces/i_geometry_estimator.h"
#include "interfaces/i_matcher.h"
#include "interfaces/i_pipeline.h"
#include "transform/warper.h"

namespace ir {

/// 基础配准流水线实现。
///
/// 将特征提取、匹配、过滤、几何估计、变换和结果保存这些阶段串联起来，
/// 供更具体的 pipeline 类型继承和定制。
class BasePipeline : public IPipeline {
public:
    BasePipeline() = default;

    bool configure(const PipelineConfig& cfg) override;

    bool run(RegistrationContext& ctx) override;

protected:
    /// 各阶段钩子函数，子类可按需重写。
    virtual bool loadImages(RegistrationContext& ctx);
    virtual bool runExtract(RegistrationContext& ctx);
    virtual bool runMatch(RegistrationContext& ctx);
    virtual bool runFilters(RegistrationContext& ctx);
    virtual bool runGeometry(RegistrationContext& ctx);
    virtual bool runWarp(RegistrationContext& ctx);
    virtual bool saveOutputs(RegistrationContext& ctx);
    virtual bool showWindows(RegistrationContext& ctx);

    PipelineConfig _config;
    std::shared_ptr<IFeatureExtractor> _extractor;
    std::shared_ptr<IMatcher> _matcher;
    std::vector<std::shared_ptr<IFilter>> _filters;
    std::shared_ptr<IGeometryEstimator> _geometry;
    std::shared_ptr<IWarper> _warper;
};

} // namespace ir
