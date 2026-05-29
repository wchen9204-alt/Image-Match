#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv2/core.hpp>

#include "interfaces/i_geometry_estimator.h"

namespace ir {

class EssentialEstimator : public IGeometryEstimator {
public:
    explicit EssentialEstimator(const YAML::Node& cfg);

    std::string  name() const override { return "Essential"; }
    GeometryType type() const override { return GeometryType::ESSENTIAL; }

    bool estimate(RegistrationContext& ctx) override;

private:
    int     method_     = 8;       // 默认使用 cv::RANSAC。
    double  threshold_  = 1.0;
    double  prob_       = 0.999;
    int     minInliers_ = 8;

    cv::Mat K_;                    // 相机内参。
};

} // namespace ir
