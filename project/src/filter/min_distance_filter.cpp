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

std::vector<cv::DMatch> collectInputMatches(const KeypointMatchData& md) {
    return md.filtered;
}

} // namespace

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

    IR_LOG_INFO(
        "MinDistanceFilter multiplier=", _multiplier, ", min_cutoff=", _minCutoff);
}

bool MinDistanceFilter::apply(RegistrationContext& ctx) {
    auto& md = ctx.keypoint_match_data;
    const std::vector<cv::DMatch> input = collectInputMatches(md);
    if (input.empty()) {
        IR_LOG_WARN("MinDistanceFilter: no matches available to filter.");
        md.filtered.clear();
        return false;
    }

    float minDistance = std::numeric_limits<float>::max();
    for (const auto& match : input) {
        minDistance = std::min(minDistance, match.distance);
    }
    if (!std::isfinite(minDistance)) {
        IR_LOG_WARN("MinDistanceFilter: invalid min distance.");
        md.filtered.clear();
        return false;
    }

    const float threshold = std::max(_multiplier * minDistance, _minCutoff);
    std::vector<cv::DMatch> kept;
    kept.reserve(input.size());
    for (const auto& match : input) {
        if (match.distance <= threshold) {
            kept.push_back(match);
        }
    }

    IR_LOG_INFO("MinDistanceFilter kept ",
                kept.size(),
                " / ",
                input.size(),
                " matches (min_dist=",
                minDistance,
                ", threshold=",
                threshold,
                ")");
    md.filtered = std::move(kept);
    return true;
}

} // namespace ir


