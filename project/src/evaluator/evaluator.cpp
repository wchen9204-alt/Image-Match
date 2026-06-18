#include "evaluator/evaluator.h"

#include <memory>
#include <string>

#include "core/config.h"
#include "evaluator/metrics/geometric/inlier_ratio.h"
#include "evaluator/metrics/geometric/reprojection_error.h"
#include "evaluator/metrics/keypoint/repeatability.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {

// ---------------------------------------------------------------------------
// 工厂：根据名称字符串创建具体指标实例
// ---------------------------------------------------------------------------
std::shared_ptr<IMetric> Evaluator::createMetric(const std::string& name,
                                                  const YAML::Node& params) {
    // 大小写和分隔符不敏感匹配。
    const std::string key = string_utils::normalizedKey(name);

    if (key == "INLIERRATIO")
        return std::make_shared<InlierRatioMetric>(params);
    if (key == "REPROJECTIONERROR")
        return std::make_shared<ReprojectionErrorMetric>(params);
    if (key == "REPEATABILITY")
        return std::make_shared<RepeatabilityMetric>(params);

    IR_LOG_WARN("Evaluator: unknown metric '", name, "'");
    return nullptr;
}

// ---------------------------------------------------------------------------
// 从 YAML 加载指标列表
// ---------------------------------------------------------------------------
bool Evaluator::loadFromYaml(const std::filesystem::path& yaml_path) {
    const YAML::Node root = Config::load(yaml_path);
    return loadFromNode(root);
}

bool Evaluator::loadFromNode(const YAML::Node& root) {
    _metrics.clear();
    const YAML::Node list = root["metrics"];
    if (!list || !list.IsSequence()) {
        IR_LOG_WARN("Evaluator: no 'metrics' sequence in config");
        return false;
    }

    for (const auto& entry : list) {
        const bool enabled = yaml_utils::getBool(entry, "enabled", true);
        if (!enabled) continue;

        const std::string name = yaml_utils::getString(entry, "name", "");
        if (name.empty()) continue;

        const YAML::Node params = entry["params"] ? entry["params"] : YAML::Node();
        auto metric = createMetric(name, params);
        if (metric) {
            _metrics.push_back(std::move(metric));
            IR_LOG_INFO("Evaluator: loaded metric ", name);
        }
    }
    return !_metrics.empty();
}

// ---------------------------------------------------------------------------
// 执行所有指标，结果写入 ctx.evaluation
// ---------------------------------------------------------------------------
void Evaluator::evaluate(RegistrationContext& ctx, const Sample& sample) const {
    ctx.evaluation.clear();
    ctx.evaluation.metrics.reserve(_metrics.size());

    for (const auto& m : _metrics) {
        MetricResult r = m->compute(ctx, sample);
        IR_LOG_INFO("  ", r.name, " = ", r.valid ? std::to_string(r.value) : r.note);
        ctx.evaluation.metrics.push_back(std::move(r));
    }
}

} // namespace ir

