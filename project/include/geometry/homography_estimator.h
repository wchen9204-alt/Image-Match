#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_geometry_estimator.h"

namespace ir {

/// 单应矩阵估计器。
class HomographyEstimator : public IGeometryEstimator {
public:
    /// 根据 YAML 配置初始化单应估计参数。
    explicit HomographyEstimator(const YAML::Node& cfg);

    /// 返回估计器名称。
    std::string name() const override { return "Homography"; }

    /// 返回当前估计器的几何类型。
    GeometryType type() const override { return GeometryType::HOMOGRAPHY; }

    /// 估计单应矩阵并写入上下文。
    bool estimate(RegistrationContext& ctx) override;

private:
    int _method = 8;
    double _ransacReprojThreshold = 3.0;
    int _maxIters = 2000;
    double _confidence = 0.995;
    int _minInliers = 8;
};

} // namespace ir

