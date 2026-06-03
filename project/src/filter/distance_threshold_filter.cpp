#include "filter/distance_threshold_filter.h"

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

DistanceThresholdFilter::DistanceThresholdFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _maxDistance = yaml_utils::getFloat(params, "max_distance", 30.0f);
    if (_maxDistance < 0.0f) {
        _maxDistance = 0.0f;
    }

    IR_LOG_INFO("DistanceThresholdFilter max_distance=", _maxDistance);
}

bool DistanceThresholdFilter::apply(RegistrationContext& ctx) {
    auto& md = ctx.keypoint_match_data;
    const std::vector<cv::DMatch> input = collectInputMatches(md);
    if (input.empty()) {
        IR_LOG_WARN("DistanceThresholdFilter: no matches available to filter.");
        md.filtered.clear();
        return false;
    }

    std::vector<cv::DMatch> kept;
    kept.reserve(input.size());
    for (const auto& match : input) {
        if (match.distance <= _maxDistance) {
            kept.push_back(match);
        }
    }

    IR_LOG_INFO("DistanceThresholdFilter kept ",
                kept.size(),
                " / ",
                input.size(),
                " matches (max_distance=",
                _maxDistance,
                ")");
    md.filtered = std::move(kept);
    return true;
}

} // namespace ir


