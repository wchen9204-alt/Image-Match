#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

/// 几何内点比例指标。
class InlierRatioMetric : public IMetric {
public:
    /// 该指标无需额外参数。
    explicit InlierRatioMetric(const YAML::Node& /*params*/) {}

    /// 返回指标名称。
    std::string name() const override { return "INLIER_RATIO"; }

    /// 计算内点比例。
    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;
};

} // namespace ir
