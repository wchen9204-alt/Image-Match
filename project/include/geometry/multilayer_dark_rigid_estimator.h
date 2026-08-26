#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_geometry_estimator.h"

namespace ir {

/// 多层暗部专用刚体估计器，复用参考流程的投票分簇与严格刚体 RANSAC。
class MultilayerDarkRigidEstimator final : public IGeometryEstimator {
public:
    explicit MultilayerDarkRigidEstimator(const YAML::Node& cfg);

    std::string name() const override { return "Rigid2D_DARK_MULTILAYER"; }
    GeometryType type() const override { return GeometryType::RIGID; }
    bool estimate(RegistrationContext& ctx) override;

private:
    int _min_inliers = 6;
    double _rotation_bin_degrees = 5.0;
    double _translation_bin_size = 10.0;
    double _reprojection_threshold = 5.0;
    int _ransac_iterations = 1000;
    int _max_clusters = 5;
};

} // namespace ir
