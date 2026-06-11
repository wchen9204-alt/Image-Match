#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_filter.h"

namespace ir {

/// 根据匹配距离分布自适应裁剪异常值。
class DistanceDistributionFilter : public IFilter {
public:
    enum class Mode { MeanStd = 0, Percentile };

    explicit DistanceDistributionFilter(const YAML::Node& cfg);

    std::string name() const override { return "DistanceDistribution"; }

    bool apply(RegistrationContext& ctx) override;

private:
    Mode _mode = Mode::MeanStd;
    float _stdMultiplier = 1.0f;
    float _percentile = 0.8f;
    float _minDistanceFloor = 0.0f;
};

}
