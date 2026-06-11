#include "core/registration.h"

#include "pipeline/direct_pipeline.h"
#include "pipeline/keypoint_pipeline.h"
#include "pipeline/structure_pipeline.h"
#include "utils/logger.h"

namespace ir {

Registration::Registration(std::shared_ptr<IPipeline> pipeline) : _pipeline(std::move(pipeline)) {}

std::string Registration::name() const {
    // 优先返回显式配置名称，便于外部日志和输出命名保持稳定。
    if (!_cfg.name.empty()) {
        return _cfg.name;
    }
    if (_pipeline) {
        return _pipeline->name();
    }
    return "Registration";
}

bool Registration::configure(const PipelineConfig& cfg) {
    // 若外部未注入具体流水线，则按 method_family 选择默认实现。
    _cfg = cfg;
    if (!_pipeline) {
        switch (_cfg.method_family) {
        case MethodFamily::STRUCTURE:
            _pipeline = std::make_shared<StructurePipeline>();
            break;
        case MethodFamily::DIRECT:
            _pipeline = std::make_shared<DirectPipeline>();
            break;
        case MethodFamily::KEYPOINT:
        default:
            _pipeline = std::make_shared<KeypointPipeline>();
            break;
        }
    }
    if (!_pipeline->configure(_cfg)) {
        IR_LOG_ERROR("Registration::configure - pipeline configure failed for '", _cfg.name, "'");
        return false;
    }
    return true;
}

bool Registration::run(RegistrationContext& ctx) {
    // Registration 只负责统一入口调度，具体执行细节交给流水线实现。
    if (!_pipeline) {
        IR_LOG_ERROR("Registration::run - pipeline not configured.");
        return false;
    }
    return _pipeline->run(ctx);
}

} // namespace ir
