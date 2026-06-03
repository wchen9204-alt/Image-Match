#include "core/registration.h"

#include "pipeline/keypoint_pipeline.h"
#include "pipeline/structure_pipeline.h"
#include "utils/logger.h"

namespace ir {

Registration::Registration(std::shared_ptr<IPipeline> pipeline) : _pipeline(std::move(pipeline)) {}

std::string Registration::name() const {
    // 名称解析优先尊重外部配置，其次退化为底层流水线名，便于日志保持稳定语义。
    if (!_cfg.name.empty())
        return _cfg.name;
    if (_pipeline)
        return _pipeline->name();
    return "Registration";
}

bool Registration::configure(const PipelineConfig& cfg) {
    // 允许在未显式注入流水线实现时自动回退到默认特征配准流水线。
    _cfg = cfg;
    if (!_pipeline) {
        if (!_cfg.structure_path.empty()) {
            _pipeline = std::make_shared<StructurePipeline>();
        } else {
            _pipeline = std::make_shared<KeypointPipeline>();
        }
    }
    if (!_pipeline->configure(_cfg)) {
        IR_LOG_ERROR("Registration::configure - pipeline configure failed for '", _cfg.name, "'");
        return false;
    }
    return true;
}

bool Registration::run(RegistrationContext& ctx) {
    // Registration 只承担顶层执行入口职责，具体阶段调度全部下沉到流水线实现。
    if (!_pipeline) {
        IR_LOG_ERROR("Registration::run - pipeline not configured.");
        return false;
    }
    return _pipeline->run(ctx);
}

} // namespace ir
