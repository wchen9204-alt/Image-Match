#pragma once

#include <memory>
#include <string>

#include "core/config.h"
#include "evaluator/evaluator.h"
#include "interfaces/i_pipeline.h"
#include "transform/warper.h"

namespace ir {

/// 通用配准流水线骨架。
class BasePipeline : public IPipeline {
public:
    BasePipeline() = default;
    ~BasePipeline() override = default;

    bool configure(const PipelineConfig& cfg) override;
    bool run(RegistrationContext& ctx) override;
    bool showWindows(RegistrationContext& ctx) override;

protected:
    virtual void resetStages() {}
    virtual bool configureStages(const PipelineConfig& cfg) = 0;
    virtual bool loadImages(RegistrationContext& ctx);
    virtual bool runExtraction(RegistrationContext& ctx) = 0;
    virtual bool runAssociation(RegistrationContext& ctx) = 0;
    virtual bool runEstimation(RegistrationContext& ctx) = 0;
    virtual bool runWarp(RegistrationContext& ctx);
    virtual bool validateWarpQuality(RegistrationContext& ctx);
    virtual bool saveOutputs(RegistrationContext& ctx);
    virtual std::string buildOutputStem(const RegistrationContext& ctx) const;

    PipelineConfig _config;
    std::shared_ptr<IWarper> _warper;

    /// 评测器，在 warp 后计算 PSNR/SSIM/RMSE 等指标。
    Evaluator _evaluator;
};

} // namespace ir
