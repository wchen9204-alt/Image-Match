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

// ---------------------------------------------------------------------------
// BasePipeline：串联加载、提取、匹配、过滤、估计、变换和保存阶段。
// ---------------------------------------------------------------------------
class BasePipeline : public IPipeline {
public:
    BasePipeline() = default;

    bool configure(const PipelineConfig& cfg) override;

    bool run(RegistrationContext& ctx) override;

protected:
    // 阶段钩子函数，子类可按需重写。
    virtual bool loadImages(RegistrationContext& ctx);
    virtual bool runExtract (RegistrationContext& ctx);
    virtual bool runMatch   (RegistrationContext& ctx);
    virtual bool runFilters (RegistrationContext& ctx);
    virtual bool runGeometry(RegistrationContext& ctx);
    virtual bool runWarp    (RegistrationContext& ctx);
    virtual bool saveOutputs(RegistrationContext& ctx);

    PipelineConfig                                config_;
    std::shared_ptr<IFeatureExtractor>            extractor_;
    std::shared_ptr<IMatcher>                     matcher_;
    std::vector<std::shared_ptr<IFilter>>         filters_;
    std::shared_ptr<IGeometryEstimator>           geometry_;
    std::shared_ptr<IWarper>                      warper_;
};

} // namespace ir
