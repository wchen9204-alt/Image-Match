#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_geometry_estimator.h"

namespace ir {

class HomographyEstimator : public IGeometryEstimator {
public:
    explicit HomographyEstimator(const YAML::Node& cfg);

    std::string  name() const override { return "Homography"; }
    GeometryType type() const override { return GeometryType::HOMOGRAPHY; }

    bool estimate(RegistrationContext& ctx) override;

private:
    int    method_                 = 8;     // 默认使用 cv::RANSAC。
    double ransacReprojThreshold_  = 3.0;
    int    maxIters_               = 2000;
    double confidence_             = 0.995;
    int    minInliers_             = 8;
};

} // namespace ir
