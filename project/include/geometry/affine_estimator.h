#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_geometry_estimator.h"

namespace ir {

class AffineEstimator : public IGeometryEstimator {
public:
    explicit AffineEstimator(const YAML::Node& cfg);

    std::string  name() const override { return "Affine2D"; }
    GeometryType type() const override { return GeometryType::AFFINE; }

    bool estimate(RegistrationContext& ctx) override;

private:
    int    method_                = 8;     // 默认使用 cv::RANSAC。
    double ransacReprojThreshold_ = 3.0;
    int    maxIters_              = 2000;
    double confidence_            = 0.99;
    int    refineIters_           = 10;
    int    minInliers_            = 6;
};

} // namespace ir
