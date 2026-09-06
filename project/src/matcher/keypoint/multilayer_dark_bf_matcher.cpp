#include "matcher/keypoint/multilayer_dark_bf_matcher.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include <opencv2/features2d.hpp>

#include "utils/descriptor_norm_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

std::vector<int> selectMultilayerKeypoints(const std::vector<cv::KeyPoint>& keypoints,
                                           double responseThreshold,
                                           double gridSize,
                                           int maxPerGrid) {
    std::vector<int> candidates;
    candidates.reserve(keypoints.size());
    for (size_t index = 0; index < keypoints.size(); ++index) {
        const cv::KeyPoint& keypoint = keypoints[index];
        if (keypoint.response <= static_cast<float>(responseThreshold)) {
            continue;
        }
        candidates.push_back(static_cast<int>(index));
    }

    std::stable_sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
        return keypoints[static_cast<size_t>(lhs)].response >
               keypoints[static_cast<size_t>(rhs)].response;
    });
    std::map<std::pair<int, int>, int> gridCounts;
    std::vector<int> selected;
    selected.reserve(candidates.size());
    for (const int index : candidates) {
        const cv::KeyPoint& keypoint = keypoints[static_cast<size_t>(index)];
        const std::pair<int, int> gridCell{
            static_cast<int>(std::floor(keypoint.pt.x / gridSize)),
            static_cast<int>(std::floor(keypoint.pt.y / gridSize))};
        int& gridCount = gridCounts[gridCell];
        if (gridCount >= maxPerGrid) {
            continue;
        }
        ++gridCount;
        selected.push_back(index);
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

bool filterMultilayerFeatures(KeypointImageData& features,
                              double responseThreshold,
                              double gridSize,
                              int maxPerGrid) {
    const bool hasDescriptors = !features.descriptors.empty();
    if (hasDescriptors &&
        features.descriptors.rows != static_cast<int>(features.keypoints.size())) {
        return false;
    }
    const std::vector<int> selected = selectMultilayerKeypoints(
        features.keypoints, responseThreshold, gridSize, maxPerGrid);
    std::vector<cv::KeyPoint> filteredKeypoints;
    filteredKeypoints.reserve(selected.size());
    cv::Mat filteredDescriptors;
    if (hasDescriptors) {
        filteredDescriptors = cv::Mat(
            static_cast<int>(selected.size()), features.descriptors.cols, features.descriptors.type());
    }
    for (size_t row = 0; row < selected.size(); ++row) {
        const int sourceIndex = selected[row];
        filteredKeypoints.push_back(features.keypoints[static_cast<size_t>(sourceIndex)]);
        if (hasDescriptors) {
            features.descriptors.row(sourceIndex).copyTo(
                filteredDescriptors.row(static_cast<int>(row)));
        }
    }
    features.keypoints = std::move(filteredKeypoints);
    features.descriptors = std::move(filteredDescriptors);
    return !features.empty();
}

} // namespace

MultilayerDarkBfMatcher::MultilayerDarkBfMatcher(const YAML::Node& cfg,
                                                 const YAML::Node& geometryCfg) {
    const YAML::Node params = cfg["params"];
    const YAML::Node multilayer = cfg["multilayer_dark"];
    const YAML::Node options = multilayer && multilayer.IsMap() ? multilayer : params;

    _norm_type = normTypeFromString(yaml_utils::getString(
        params, "norm_type", "AUTO"));
    _knn_k = std::max(1, yaml_utils::getInt(
        options, "knn", yaml_utils::getInt(options, "knn_k", 64)));
    _expansion_multiplier = std::max(
        1.0f, yaml_utils::getFloat(options, "expansion_multiplier", 2.0f));
    const YAML::Node geometryOptions = geometryCfg["params"]["multilayer_dark"];
    _keypoint_filter_enabled = yaml_utils::getBool(
        geometryOptions, "keypoint_filter_enabled", false);
    _keypoint_filter_response_threshold = std::max(
        0.0, yaml_utils::getDouble(geometryOptions, "keypoint_filter_response_threshold", 0.008));
    _keypoint_filter_grid_size = std::max(
        1.0, yaml_utils::getDouble(geometryOptions, "keypoint_filter_grid_size", 50.0));
    _keypoint_filter_max_keypoints_per_grid = std::max(
        1, yaml_utils::getInt(geometryOptions, "keypoint_filter_max_keypoints_per_grid", 10));

    IR_LOG_INFO("多层暗部 BF KNN 匹配器：knn_k=",
                _knn_k,
                ", expansion_multiplier=",
                _expansion_multiplier,
                ", keypoint_pre_filter=",
                _keypoint_filter_enabled);
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

    if (_keypoint_filter_enabled &&
        (!filterMultilayerFeatures(features.first, _keypoint_filter_response_threshold,
                                   _keypoint_filter_grid_size,
                                   _keypoint_filter_max_keypoints_per_grid) ||
         !filterMultilayerFeatures(features.second, _keypoint_filter_response_threshold,
                                   _keypoint_filter_grid_size,
                                   _keypoint_filter_max_keypoints_per_grid))) {
        IR_LOG_ERROR("多层暗部点对投票预过滤后关键点或描述子不足。");
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

    // 3. 按每个 query 的最佳距离扩展候选，交给后续刚体投票估计。
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
                ", keypoints=",
                features.first.keypoints.size(),
                " / ",
                features.second.keypoints.size(),
                ", norm=",
                toString(effective));
    return !matches.filtered_matches.empty();
}

} // namespace ir
