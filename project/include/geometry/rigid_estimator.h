#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_geometry_estimator.h"

namespace ir {

/// 刚体变换估计器。
class RigidEstimator : public IGeometryEstimator {
public:
    /// 从 YAML 配置初始化刚体估计器参数。
    explicit RigidEstimator(const YAML::Node& cfg);

    /// 返回估计器名称。
    std::string name() const override { return "Rigid2D"; }

    /// 返回当前估计器支持的几何类型。
    GeometryType type() const override { return GeometryType::RIGID; }

    /// 估计 2x3 刚体变换，并将结果写回上下文。
    bool estimate(RegistrationContext& ctx) override;

private:
    int _method = 8;
    double _ransacReprojThreshold = 3.0;
    int _maxIters = 2000;
    double _confidence = 0.99;
    int _refineIters = 10;
    int _minInliers = 3;
    std::string _estimatorBackend = "OPENCV_PARTIAL_AFFINE";
    std::string _rigidRefineMode = "SVD";
};

} // namespace ir
