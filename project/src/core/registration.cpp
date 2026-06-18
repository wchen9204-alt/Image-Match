#include "core/registration.h"

#include "pipeline/direct_pipeline.h"
#include "pipeline/keypoint_pipeline.h"
#include "pipeline/learning_pipeline.h"
#include "pipeline/structure_pipeline.h"
#include "utils/logger.h"

namespace ir {

Registration::Registration(std::shared_ptr<IPipeline> pipeline) : _pipeline(std::move(pipeline)) {}

std::string Registration::name() const {
    // 优先返回显式配置的名称，保证日志和输出目录命名稳定。
    if (!_cfg.name.empty()) {
        return _cfg.name;
    }
    if (_pipeline) {
        return _pipeline->name();
    }
    return "Registration";
}

bool Registration::configure(const PipelineConfig& cfg) {
    _cfg = cfg;

    // 外部未注入具体流水线时，按 method_family 统一选择默认实现。
    if (!_pipeline) {
        switch (_cfg.method_family) {
        case MethodFamily::STRUCTURE:
            _pipeline = std::make_shared<StructurePipeline>();
            break;
        case MethodFamily::DIRECT:
            _pipeline = std::make_shared<DirectPipeline>();
            break;
        case MethodFamily::LEARNING:
            _pipeline = std::make_shared<LearningPipeline>();
            break;
        case MethodFamily::KEYPOINT:
        default:
            _pipeline = std::make_shared<KeypointPipeline>();
            break;
        }
    }

    // Registration 只负责高层配置，具体阶段配置交给流水线实现。
    if (!_pipeline->configure(_cfg)) {
        IR_LOG_ERROR("Registration::configure - pipeline configure failed for '", _cfg.name, "'");
        return false;
    }
    return true;
}

bool Registration::run(RegistrationContext& ctx) {
    // 运行前必须已经完成 configure，然后由流水线执行各阶段。
    if (!_pipeline) {
        IR_LOG_ERROR("Registration::run - pipeline not configured.");
        return false;
    }
    return _pipeline->run(ctx);
}

} // namespace ir

