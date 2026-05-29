#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

class PsnrMetric : public IMetric {
public:
    explicit PsnrMetric(const YAML::Node& params);

    std::string name() const override { return "PSNR"; }

    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;

private:
    double max_value_ = 255.0;
};

} // namespace ir
