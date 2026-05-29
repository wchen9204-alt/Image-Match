#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

class RmseMetric : public IMetric {
public:
    explicit RmseMetric(const YAML::Node& /*params*/) {}

    std::string name() const override { return "RMSE"; }

    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;
};

} // namespace ir
