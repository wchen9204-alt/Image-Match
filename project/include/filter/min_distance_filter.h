#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_filter.h"

namespace ir {

/// 按 OpenCV 教程中的 min_dist 启发式过滤匹配。
class MinDistanceFilter : public IFilter {
public:
    explicit MinDistanceFilter(const YAML::Node& cfg);

    std::string name() const override { return "MinDistance"; }

    bool apply(RegistrationContext& ctx) override;

private:
    float _multiplier = 2.0f;
    float _minCutoff = 30.0f;
};

} // namespace ir
