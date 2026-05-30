#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

/// 关键点重复率指标。
class RepeatabilityMetric : public IMetric {
public:
    /// 根据 YAML 参数初始化指标。
    explicit RepeatabilityMetric(const YAML::Node& params);

    /// 返回指标名称。
    std::string name() const override { return "REPEATABILITY"; }

    /// 计算关键点重复率。
    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;

private:
    double _pixelThreshold = 3.0;
};

} // namespace ir
