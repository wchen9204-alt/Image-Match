#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_filter.h"

namespace ir {

/// 基于 pairwise vote 的 rigid 几何一致性点特征预过滤器。
///
/// 该过滤器不依赖 KNN，只使用当前 filtered_matches 中的点对关系。
/// 它通过 pairwise vote 和主旋转峰，优先保留满足“旋转 + 平移、无缩放”假设的匹配。
class PairwiseRigidConsistencyFilter : public IFilter {
public:
    explicit PairwiseRigidConsistencyFilter(const YAML::Node& cfg);

    std::string name() const override { return "PairwiseRigidConsistency"; }

    bool apply(RegistrationContext& ctx) override;

private:
    double _minPairDistance = 20.0;
    double _maxDistanceDiff = 5.0;
    double _maxAngleDiffDeg = 8.0;
    int _rotationBins = 36;
    int _minVotes = 2;
    int _keepTopK = 0;
    bool _fallbackToInputIfEmpty = true;
};

} // namespace ir
