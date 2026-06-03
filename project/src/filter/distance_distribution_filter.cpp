#include "filter/distance_distribution_filter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <vector>

#include <opencv2/core.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

std::string toUpperAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

std::vector<cv::DMatch> collectInputMatches(const KeypointMatchData& md) {
    return md.filtered;
}

DistanceDistributionFilter::Mode modeFromString(const std::string& s) {
    const std::string normalized = toUpperAscii(s);
    if (normalized == "PERCENTILE" || normalized == "PCTL") {
        return DistanceDistributionFilter::Mode::Percentile;
    }
    return DistanceDistributionFilter::Mode::MeanStd;
}

float percentileDistance(std::vector<float> distances, float percentile) {
    if (distances.empty()) {
        return 0.0f;
    }

    std::sort(distances.begin(), distances.end());
    const float clamped = std::clamp(percentile, 0.0f, 1.0f);
    const std::size_t index =
        static_cast<std::size_t>(std::floor(clamped * static_cast<float>(distances.size() - 1)));
    return distances[index];
}

} // namespace

DistanceDistributionFilter::DistanceDistributionFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _mode = modeFromString(yaml_utils::getString(params, "mode", "mean_std"));
    _stdMultiplier = yaml_utils::getFloat(params, "std_multiplier", 1.0f);
    _percentile = yaml_utils::getFloat(params, "percentile", 0.8f);
    _minDistanceFloor = yaml_utils::getFloat(params, "min_distance_floor", 0.0f);

    if (_stdMultiplier < 0.0f) {
        _stdMultiplier = 0.0f;
    }
    _percentile = std::clamp(_percentile, 0.0f, 1.0f);
    if (_minDistanceFloor < 0.0f) {
        _minDistanceFloor = 0.0f;
    }

    IR_LOG_INFO("DistanceDistributionFilter mode=",
                (_mode == Mode::Percentile ? "percentile" : "mean_std"),
                ", std_multiplier=",
                _stdMultiplier,
                ", percentile=",
                _percentile,
                ", min_distance_floor=",
                _minDistanceFloor);
}

bool DistanceDistributionFilter::apply(RegistrationContext& ctx) {
    auto& md = ctx.keypoint_match_data;
    const std::vector<cv::DMatch> input = collectInputMatches(md);
    if (input.empty()) {
        IR_LOG_WARN("DistanceDistributionFilter: no matches available to filter.");
        md.filtered.clear();
        return false;
    }

    std::vector<float> distances;
    distances.reserve(input.size());
    for (const auto& match : input) {
        distances.push_back(match.distance);
    }

    float threshold = _minDistanceFloor;
    if (_mode == Mode::Percentile) {
        threshold = std::max(threshold, percentileDistance(distances, _percentile));
    } else {
        const float sum = std::accumulate(distances.begin(), distances.end(), 0.0f);
        const float mean = sum / static_cast<float>(distances.size());
        float variance = 0.0f;
        for (const float distance : distances) {
            const float delta = distance - mean;
            variance += delta * delta;
        }
        variance /= static_cast<float>(distances.size());
        const float stddev = std::sqrt(variance);
        threshold = std::max(threshold, mean + _stdMultiplier * stddev);
    }

    std::vector<cv::DMatch> kept;
    kept.reserve(input.size());
    for (const auto& match : input) {
        if (match.distance <= threshold) {
            kept.push_back(match);
        }
    }

    IR_LOG_INFO("DistanceDistributionFilter kept ",
                kept.size(),
                " / ",
                input.size(),
                " matches (threshold=",
                threshold,
                ")");
    md.filtered = std::move(kept);
    return true;
}

} // namespace ir


