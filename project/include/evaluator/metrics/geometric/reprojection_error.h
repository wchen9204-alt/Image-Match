#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

// ---------------------------------------------------------------------------
// ReprojectionErrorMetric：计算内点匹配的平均重投影误差。
// ---------------------------------------------------------------------------
class ReprojectionErrorMetric : public IMetric {
public:
    explicit ReprojectionErrorMetric(const YAML::Node& params);

    std::string name() const override { return "REPROJECTION_ERROR"; }

    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;

private:
    bool symmetric_ = true;
};

} // namespace ir
