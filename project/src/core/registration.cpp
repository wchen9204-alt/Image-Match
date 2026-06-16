#include "core/registration.h"

#include "pipeline/direct_pipeline.h"
#include "pipeline/keypoint_pipeline.h"
#include "pipeline/learning_pipeline.h"
#include "pipeline/structure_pipeline.h"
#include "utils/logger.h"

namespace ir {

Registration::Registration(std::shared_ptr<IPipeline> pipeline) : _pipeline(std::move(pipeline)) {}

std::string Registration::name() const {
    // ���ȷ�����ʽ�������ƣ�������־��������������ȶ���
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

    // �ⲿδע�������ˮ��ʱ���� method_family ͳһѡ��Ĭ��ʵ�֡�
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

    // Registration ֻ����������ȣ�����׶����ý�����ˮ��ʵ�֡�
    if (!_pipeline->configure(_cfg)) {
        IR_LOG_ERROR("Registration::configure - pipeline configure failed for '", _cfg.name, "'");
        return false;
    }
    return true;
}

bool Registration::run(RegistrationContext& ctx) {
    // ����ǰ��������� configure���������ˮ�߽���ִ�н׶Ρ�
    if (!_pipeline) {
        IR_LOG_ERROR("Registration::run - pipeline not configured.");
        return false;
    }
    return _pipeline->run(ctx);
}

} // namespace ir

