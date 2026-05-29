#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_geometry_estimator.h"

namespace ir {

/// 相似变换估计器。
class SimilarityEstimator : public IGeometryEstimator {
public:
    /// 根据 YAML 配置初始化相似变换估计参数。
    explicit SimilarityEstimator(const YAML::Node& cfg);

    /// 返回估计器名称。
    std::string name() const override { return "Similarity2D"; }

    /// 返回当前估计器的几何类型。
    GeometryType type() const override { return GeometryType::SIMILARITY; }

    /// 估计 2x3 相似变换矩阵并写入上下文。
    bool estimate(RegistrationContext& ctx) override;

private:
    int    _method                = 8;
    double _ransacReprojThreshold = 3.0;
    int    _maxIters              = 2000;
    double _confidence            = 0.99;
    int    _refineIters           = 10;
    int    _minInliers            = 3;
};

} // namespace ir
