#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_filter.h"

namespace ir {

/// 按描述子距离上限直接裁剪匹配。
class DistanceThresholdFilter : public IFilter {
public:
    explicit DistanceThresholdFilter(const YAML::Node& cfg);

    std::string name() const override { return "DistanceThreshold"; }

    bool apply(RegistrationContext& ctx) override;

private:
    float _maxDistance = 30.0f;
};

}

