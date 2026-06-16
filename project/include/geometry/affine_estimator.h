#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_geometry_estimator.h"

namespace ir {

/// 仿射几何估计器。
class AffineEstimator : public IGeometryEstimator {
public:
    /// 根据 YAML 配置初始化仿射估计参数。
    explicit AffineEstimator(const YAML::Node& cfg);

    /// 返回估计器名称。
    std::string name() const override { return "Affine2D"; }

    /// 返回当前估计器的几何类型。
    GeometryType type() const override { return GeometryType::AFFINE; }

    /// 估计 2x3 仿射矩阵并写入上下文。
    bool estimate(RegistrationContext& ctx) override;

private:
    int _method = 8;
    double _ransacReprojThreshold = 3.0;
    int _maxIters = 2000;
    double _confidence = 0.99;
    int _refineIters = 10;
    int _minInliers = 6;
};

} // namespace ir

