#include "filter/distance_threshold_filter.h"

#include <vector>

#include <opencv2/core.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

// 点特征法路径统一从当前筛选结果集合读取输入。
std::vector<cv::DMatch> collectInputMatches(const KeypointMatchData& md) {
    return md.filtered_matches;
}

} // 匿名命名空间

// 读取固定距离阈值；超过该阈值的匹配会被直接剔除。
DistanceThresholdFilter::DistanceThresholdFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _maxDistance = yaml_utils::getFloat(params, "max_distance", 30.0f);
    if (_maxDistance < 0.0f) {
        _maxDistance = 0.0f;
    }

    IR_LOG_INFO("DistanceThresholdFilter max_distance=", _maxDistance);
}

bool DistanceThresholdFilter::apply(RegistrationContext& ctx) {
    // 结构法路径：按固定距离阈值过滤结构匹配。
    auto& smd = ctx.structure_match_data;
    if (!smd.filtered_matches.empty() || !smd.raw_matches_knn.empty()) {
        const std::vector<cv::DMatch>& input = smd.filtered_matches;
        if (input.empty()) {
            IR_LOG_WARN("DistanceThresholdFilter [structure]: no matches available.");
            return false;
        }

        std::vector<cv::DMatch> kept;
        kept.reserve(input.size());

        // 步骤一：剔除距离超过固定上限的结构匹配。
        for (const auto& match : input) {
            if (match.distance <= _maxDistance) {
                kept.push_back(match);
            }
        }

        smd.filtered_matches = std::move(kept);
        IR_LOG_INFO("DistanceThresholdFilter [structure] kept ",
                    smd.filtered_matches.size(),
                    " / ",
                    input.size());
        return true;
    }

    // 点特征法路径：对当前筛选结果中的点特征匹配执行同样的固定阈值过滤。
    auto& md = ctx.keypoint_match_data;
    const std::vector<cv::DMatch> input = collectInputMatches(md);
    if (input.empty()) {
        IR_LOG_WARN("DistanceThresholdFilter: no matches available to filter.");
        md.filtered_matches.clear();
        return false;
    }

    std::vector<cv::DMatch> kept;
    kept.reserve(input.size());

    // 步骤一：对点特征匹配执行同样的固定阈值过滤。
    for (const auto& match : input) {
        if (match.distance <= _maxDistance) {
            kept.push_back(match);
        }
    }

    IR_LOG_INFO("DistanceThresholdFilter [keypoint] kept ",
                kept.size(),
                " / ",
                input.size(),
                " matches (max_distance=",
                _maxDistance,
                ")");

    // 步骤二：用阈值过滤后的结果覆盖当前筛选结果。
    md.filtered_matches = std::move(kept);
    return true;
}

}


