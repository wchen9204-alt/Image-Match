#include "core/registration.h"

#include "pipeline/keypoint_pipeline.h"
#include "pipeline/structure_pipeline.h"
#include "utils/logger.h"

namespace ir {

Registration::Registration(std::shared_ptr<IPipeline> pipeline) : _pipeline(std::move(pipeline)) {}

std::string Registration::name() const {
    // ���ƽ������������ⲿ���ã�����˻�Ϊ�ײ���ˮ������������־�����ȶ����塣
    if (!_cfg.name.empty())
        return _cfg.name;
    if (_pipeline)
        return _pipeline->name();
    return "Registration";
}

bool Registration::configure(const PipelineConfig& cfg) {
    // ������δ��ʽע����ˮ��ʵ��ʱ�Զ����˵�Ĭ��������׼��ˮ�ߡ�
    _cfg = cfg;
    if (!_pipeline) {
        switch (_cfg.method_family) {
        case MethodFamily::STRUCTURE:
            _pipeline = std::make_shared<StructurePipeline>();
            break;
        case MethodFamily::DIRECT:
            // TODO: _pipeline = std::make_shared<DirectPipeline>();
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
    // Registration ֻ�е�����ִ�����ְ�𣬾���׶ε���ȫ���³�����ˮ��ʵ�֡�
    if (!_pipeline) {
        IR_LOG_ERROR("Registration::run - pipeline not configured.");
        return false;
    }
    return _pipeline->run(ctx);
}

} // namespace ir
