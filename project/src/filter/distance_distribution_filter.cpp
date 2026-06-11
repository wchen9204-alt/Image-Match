#include "filter/distance_distribution_filter.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <opencv2/core.hpp>

#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

// 点特征法路径统一从当前筛选结果集合读取输入。
std::vector<cv::DMatch> collectInputMatches(const KeypointMatchData& md) {
    return md.filtered;
}

// 将配置中的模式名称映射为具体的阈值估计策略。
DistanceDistributionFilter::Mode modeFromString(const std::string& s) {
    const std::string normalized = string_utils::toUpperAscii(s);
    if (normalized == "PERCENTILE" || normalized == "PCTL") {
        return DistanceDistributionFilter::Mode::Percentile;
    }
    return DistanceDistributionFilter::Mode::MeanStd;
}

// 计算距离分布在指定百分位上的阈值。
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

} // 匿名命名空间

// 读取基于距离分布的阈值策略及其控制参数。
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
    // 结构法路径：根据结构匹配的距离分布估计筛选阈值。
    auto& smd = ctx.structure_match_data;
    if (!smd.filtered_matches.empty() || !smd.raw_matches_knn.empty()) {
        const std::vector<cv::DMatch>& input = smd.filtered_matches;
        if (input.empty()) {
            IR_LOG_WARN("DistanceDistributionFilter [structure]: no matches available.");
            return false;
        }

        // 步骤一：收集结构匹配距离，用于估计当前批次阈值。
        std::vector<float> distances;
        distances.reserve(input.size());
        for (const auto& match : input) {
            distances.push_back(match.distance);
        }

        float threshold = _minDistanceFloor;
        if (_mode == Mode::Percentile) {
            threshold = std::max(threshold, percentileDistance(distances, _percentile));
        } else {
            // 步骤二：在均值/标准差模式下，阈值等于均值加上标准差倍数项。
            const float sum = std::accumulate(distances.begin(), distances.end(), 0.0f);
            const float mean = sum / static_cast<float>(distances.size());
            float variance = 0.0f;
            for (const float d : distances) {
                const float delta = d - mean;
                variance += delta * delta;
            }
            variance /= static_cast<float>(distances.size());
            const float stddev = std::sqrt(variance);
            threshold = std::max(threshold, mean + _stdMultiplier * stddev);
        }

        std::vector<cv::DMatch> kept;
        kept.reserve(input.size());

        // 步骤三：仅保留落在估计阈值以内的结构匹配。
        for (const auto& match : input) {
            if (match.distance <= threshold) {
                kept.push_back(match);
            }
        }

        smd.filtered_matches = std::move(kept);
        IR_LOG_INFO("DistanceDistributionFilter [structure] kept ",
                    smd.filtered_matches.size(),
                    " / ",
                    input.size());
        return true;
    }

    // 点特征法路径：对当前筛选结果中的点特征匹配执行同样的分布式阈值过滤。
    auto& md = ctx.keypoint_match_data;
    const std::vector<cv::DMatch> input = collectInputMatches(md);
    if (input.empty()) {
        IR_LOG_WARN("DistanceDistributionFilter: no matches available to filter.");
        md.filtered.clear();
        return false;
    }

    // 步骤一：收集点特征匹配距离，用于估计阈值。
    std::vector<float> distances;
    distances.reserve(input.size());
    for (const auto& match : input) {
        distances.push_back(match.distance);
    }

    float threshold = _minDistanceFloor;
    if (_mode == Mode::Percentile) {
        threshold = std::max(threshold, percentileDistance(distances, _percentile));
    } else {
        // 步骤二：根据均值和标准差估计当前阈值。
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

    // 步骤三：保留距离落在估计阈值以内的匹配。
    for (const auto& match : input) {
        if (match.distance <= threshold) {
            kept.push_back(match);
        }
    }

    IR_LOG_INFO("DistanceDistributionFilter [keypoint] kept ",
                kept.size(),
                " / ",
                input.size(),
                " matches (threshold=",
                threshold,
                ")");

    // 步骤四：将筛选结果写回，供后续阶段继续使用。
    md.filtered = std::move(kept);
    return true;
}

}
