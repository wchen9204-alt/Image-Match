#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

// ---------------------------------------------------------------------------
// RepeatabilityMetric：计算真值变换下关键点的重复率。
// ---------------------------------------------------------------------------
class RepeatabilityMetric : public IMetric {
public:
    explicit RepeatabilityMetric(const YAML::Node& params);

    std::string name() const override { return "REPEATABILITY"; }

    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;

private:
    double pixel_threshold_ = 3.0;
};

} // namespace ir
