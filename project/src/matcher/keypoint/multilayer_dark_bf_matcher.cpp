#include "matcher/keypoint/multilayer_dark_bf_matcher.h"

#include <algorithm>

#include <opencv2/features2d.hpp>

#include "utils/descriptor_norm_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

MultilayerDarkBfMatcher::MultilayerDarkBfMatcher(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"];
    const YAML::Node multilayer = cfg["multilayer_dark"];
    const YAML::Node options = multilayer && multilayer.IsMap() ? multilayer : params;

    _norm_type = normTypeFromString(yaml_utils::getString(
        params, "norm_type", "AUTO"));
    _knn_k = std::max(1, yaml_utils::getInt(
        options, "knn", yaml_utils::getInt(options, "knn_k", 64)));
    _expansion_multiplier = std::max(
        1.0f, yaml_utils::getFloat(options, "expansion_multiplier", 2.0f));

    IR_LOG_INFO("多层暗部 BF KNN 匹配器：knn_k=",
                _knn_k,
                ", expansion_multiplier=",
                _expansion_multiplier);
}

bool MultilayerDarkBfMatcher::match(RegistrationContext& ctx) {
    auto& features = ctx.keypoint_data;
    auto& matches = ctx.keypoint_match_data;
    matches.clear();
    matches.match_method = MatchMethod::KNN;

    if (features.first.descriptors.empty() || features.second.descriptors.empty()) {
        IR_LOG_ERROR("多层暗部 BF KNN 匹配失败：描述子为空。");
        return false;
    }

    // 1. 使用项目统一的描述子距离解析规则。
    const NormType effective = descriptor_norm_utils::resolve(
        _norm_type, features.norm_type, features.first.descriptors);
    const int cv_norm = toCvNorm(effective);
    cv::BFMatcher matcher(cv_norm, false);
    matcher.knnMatch(features.first.descriptors,
                     features.second.descriptors,
                     matches.neighbour_matches_by_query,
                     _knn_k);

    // 2. 保留每个查询点的最佳匹配，供诊断和共享对应点快照使用。
    matches.buildRawMatchesFromNeighbours();

    // 3. 参考流程按每个 query 的最佳距离扩展候选，交给后续刚体投票估计。
    for (const auto& neighbours : matches.neighbour_matches_by_query) {
        if (neighbours.empty()) {
            continue;
        }
        const float distance_limit = neighbours.front().distance * _expansion_multiplier;
        for (const auto& candidate : neighbours) {
            if (candidate.distance < distance_limit) {
                matches.filtered_matches.push_back(candidate);
            }
        }
    }

    IR_LOG_INFO("多层暗部 BF KNN 匹配完成：raw=",
                matches.raw_matches.size(),
                ", expanded=",
                matches.filtered_matches.size(),
                ", norm=",
                toString(effective));
    return !matches.filtered_matches.empty();
}

} // namespace ir
