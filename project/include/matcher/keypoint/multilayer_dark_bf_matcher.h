#pragma once

#include <yaml-cpp/yaml.h>

#include "core/types.h"
#include "interfaces/i_matcher.h"

namespace ir {

/// 多层暗部专用 BF KNN 匹配器，生成扩展候选匹配。
class MultilayerDarkBfMatcher final : public IMatcher {
public:
    /// 从匹配与几何 YAML 读取 KNN、扩展和关键点预过滤参数。
    MultilayerDarkBfMatcher(const YAML::Node& matcher_cfg,
                            const YAML::Node& geometry_cfg);

    std::string name() const override { return "BFMatcher_DARK_MULTILAYER"; }
    bool match(RegistrationContext& ctx) override;

private:
    NormType _norm_type = NormType::UNKNOWN;
    int _knn_k = 64;
    float _expansion_multiplier = 2.0f;
    bool _keypoint_filter_enabled = false;
    double _keypoint_filter_response_threshold = 0.008;
    double _keypoint_filter_grid_size = 50.0;
    int _keypoint_filter_max_keypoints_per_grid = 10;
};

} // namespace ir
