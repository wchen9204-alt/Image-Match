#include "filter/min_distance_filter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

// 点特征法过滤链统一从当前筛选结果集合读取输入。
std::vector<cv::DMatch> collectInputMatches(const KeypointMatchData& md) {
    return md.filtered_matches;
}

} // 匿名命名空间

// 读取自适应阈值规则：取“最小距离倍数阈值”和“固定下限阈值”中的较大者。
MinDistanceFilter::MinDistanceFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _multiplier = yaml_utils::getFloat(params, "multiplier", 2.0f);
    _minCutoff = yaml_utils::getFloat(params, "min_cutoff", 30.0f);

    if (_multiplier < 0.0f) {
        _multiplier = 0.0f;
    }
    if (_minCutoff < 0.0f) {
        _minCutoff = 0.0f;
    }

    IR_LOG_INFO("MinDistanceFilter multiplier=", _multiplier, ", min_cutoff=", _minCutoff);
}

bool MinDistanceFilter::apply(RegistrationContext& ctx) {
    // 结构法路径：基于最小距离阈值进一步筛选结构匹配。
    auto& smd = ctx.structure_match_data;
    if (!smd.filtered_matches.empty() || !smd.raw_matches_knn.empty()) {
        const std::vector<cv::DMatch>& input = smd.filtered_matches;
        if (input.empty()) {
            IR_LOG_WARN("MinDistanceFilter [structure]: no matches available.");
            return false;
        }

        // 步骤一：统计当前结构匹配集合中的最小距离。
        float minDistance = std::numeric_limits<float>::max();
        for (const auto& match : input) {
            minDistance = std::min(minDistance, match.distance);
        }
        if (!std::isfinite(minDistance)) {
            smd.filtered_matches.clear();
            return false;
        }

        const float threshold = std::max(_multiplier * minDistance, _minCutoff);
        std::vector<cv::DMatch> kept;
        kept.reserve(input.size());

        // 步骤二：仅保留距离不超过自适应阈值的结构匹配。
        for (const auto& match : input) {
            if (match.distance <= threshold) {
                kept.push_back(match);
            }
        }

        smd.filtered_matches = std::move(kept);
        IR_LOG_INFO("MinDistanceFilter [structure] kept ",
                    smd.filtered_matches.size(),
                    " / ",
                    input.size());
        return true;
    }

    // 点特征法路径：对当前筛选结果中的点特征匹配执行同样的阈值筛选。
    auto& md = ctx.keypoint_match_data;
    const std::vector<cv::DMatch> input = collectInputMatches(md);
    if (input.empty()) {
        IR_LOG_WARN("MinDistanceFilter: no matches available to filter.");
        md.filtered_matches.clear();
        return false;
    }

    // 步骤一：计算当前点特征匹配集合中的最小距离。
    float minDistance = std::numeric_limits<float>::max();
    for (const auto& match : input) {
        minDistance = std::min(minDistance, match.distance);
    }
    if (!std::isfinite(minDistance)) {
        IR_LOG_WARN("MinDistanceFilter: invalid min distance.");
        md.filtered_matches.clear();
        return false;
    }

    const float threshold = std::max(_multiplier * minDistance, _minCutoff);
    std::vector<cv::DMatch> kept;
    kept.reserve(input.size());

    // 步骤二：用自适应阈值裁剪距离过大的点特征匹配。
    for (const auto& match : input) {
        if (match.distance <= threshold) {
            kept.push_back(match);
        }
    }

    IR_LOG_INFO("MinDistanceFilter [keypoint] kept ",
                kept.size(),
                " / ",
                input.size(),
                " matches (min_dist=",
                minDistance,
                ", threshold=",
                threshold,
                ")");

    // 步骤三：将结果写回当前筛选结果集合，供后续过滤器或几何估计使用。
    md.filtered_matches = std::move(kept);
    return true;
}

}


