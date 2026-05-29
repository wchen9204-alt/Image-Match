#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

class InlierRatioMetric : public IMetric {
public:
    explicit InlierRatioMetric(const YAML::Node& /*params*/) {}

    std::string name() const override { return "INLIER_RATIO"; }

    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;
};

} // namespace ir
