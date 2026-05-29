#include "core/registration.h"

#include "pipeline/feature_pipeline.h"
#include "utils/logger.h"

namespace ir {

Registration::Registration(std::shared_ptr<IPipeline> pipeline)
    : _pipeline(std::move(pipeline)) {}

std::string Registration::name() const {
    if (!_cfg.name.empty()) return _cfg.name;
    if (_pipeline)          return _pipeline->name();
    return "Registration";
}

bool Registration::configure(const PipelineConfig& cfg) {
    _cfg = cfg;
    if (!_pipeline) {
        _pipeline = std::make_shared<FeaturePipeline>();
    }
    if (!_pipeline->configure(_cfg)) {
        IR_LOG_ERROR("Registration::configure - pipeline configure failed for '",
                     _cfg.name, "'");
        return false;
    }
    return true;
}

bool Registration::run(RegistrationContext& ctx) {
    if (!_pipeline) {
        IR_LOG_ERROR("Registration::run - pipeline not configured.");
        return false;
    }
    return _pipeline->run(ctx);
}

} // namespace ir

