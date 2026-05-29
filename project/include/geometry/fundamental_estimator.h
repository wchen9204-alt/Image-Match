#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_geometry_estimator.h"

namespace ir {

class FundamentalEstimator : public IGeometryEstimator {
public:
    explicit FundamentalEstimator(const YAML::Node& cfg);

    std::string  name() const override { return "Fundamental"; }
    GeometryType type() const override { return GeometryType::FUNDAMENTAL; }

    bool estimate(RegistrationContext& ctx) override;

private:
    int    method_                = 8;     // 默认使用 cv::FM_RANSAC。
    double ransacReprojThreshold_ = 3.0;
    double confidence_            = 0.99;
    int    maxIters_              = 2000;
    int    minInliers_            = 8;
};

} // namespace ir
