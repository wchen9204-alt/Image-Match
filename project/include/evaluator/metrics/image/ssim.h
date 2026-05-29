#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

class SsimMetric : public IMetric {
public:
    explicit SsimMetric(const YAML::Node& params);

    std::string name() const override { return "SSIM"; }

    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;

private:
    int    window_ = 11;
    double sigma_  = 1.5;
};

} // namespace ir
