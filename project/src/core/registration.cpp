#include "core/registration.h"

#include "pipeline/feature_pipeline.h"
#include "utils/logger.h"

namespace ir {

Registration::Registration(std::shared_ptr<IPipeline> pipeline)
    : pipeline_(std::move(pipeline)) {}

std::string Registration::name() const {
    if (!cfg_.name.empty()) return cfg_.name;
    if (pipeline_)          return pipeline_->name();
    return "Registration";
}

bool Registration::configure(const PipelineConfig& cfg) {
    cfg_ = cfg;
    if (!pipeline_) {
        pipeline_ = std::make_shared<FeaturePipeline>();
    }
    if (!pipeline_->configure(cfg_)) {
        IR_LOG_ERROR("Registration::configure - pipeline configure failed for '",
                     cfg_.name, "'");
        return false;
    }
    return true;
}

bool Registration::run(RegistrationContext& ctx) {
    if (!pipeline_) {
        IR_LOG_ERROR("Registration::run - pipeline not configured.");
        return false;
    }
    return pipeline_->run(ctx);
}

} // namespace ir
