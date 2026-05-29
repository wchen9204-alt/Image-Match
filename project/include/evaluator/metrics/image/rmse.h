#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

/// RMSE 指标。
class RmseMetric : public IMetric {
public:
    /// 该指标无需额外参数。
    explicit RmseMetric(const YAML::Node& /*params*/) {}

    /// 返回指标名称。
    std::string name() const override { return "RMSE"; }

    /// 计算均方根误差。
    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;
};

} // namespace ir
