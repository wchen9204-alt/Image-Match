#include "matcher/structure/hausdorff_associator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "matcher/structure/structure_point_set.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

// 计算 source 点集在给定平移下到 target 距离图的有序 Hausdorff 距离。
// percentile 小于 1.0 时可降低少量离群边缘点对评分的影响。
double directedHausdorff(const std::vector<cv::Point2f>& srcPoints,
                         const cv::Mat& targetDistance,
                         const cv::Point2i& shift,
                         double percentile) {
    if (srcPoints.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<float> distances;
    distances.reserve(srcPoints.size());
    for (const auto& p : srcPoints) {
        const int x = cvRound(p.x) + shift.x;
        const int y = cvRound(p.y) + shift.y;
        if (x < 0 || y < 0 || x >= targetDistance.cols || y >= targetDistance.rows) {
            distances.push_back(std::numeric_limits<float>::infinity());
            continue;
        }
        distances.push_back(targetDistance.at<float>(y, x));
    }

    if (distances.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    // 用 nth_element 只取目标分位点，避免完整排序带来的额外开销。
    percentile = std::clamp(percentile, 0.0, 1.0);
    const size_t idx = static_cast<size_t>(
        std::floor(percentile * static_cast<double>(distances.size() - 1)));
    std::nth_element(distances.begin(), distances.begin() + idx, distances.end());
    return static_cast<double>(distances[idx]);
}

} // namespace

HausdorffAssociator::HausdorffAssociator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _searchRadius = yaml_utils::getInt(params, "searchRadius", 20);
    _step = std::max(1, yaml_utils::getInt(params, "step", 1));
    _maxPoints = yaml_utils::getInt(params, "maxPoints", 2000);
    _scoreThreshold = yaml_utils::getDouble(params, "scoreThreshold", 3.0);
    _percentile = yaml_utils::getDouble(params, "percentile", 0.95);
    _bidirectional = yaml_utils::getBool(params, "bidirectional", true);
}

bool HausdorffAssociator::associate(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;
    md.clear();
    md.method = name();

    if (ctx.structure_data.first.response.empty() || ctx.structure_data.second.response.empty()) {
        md.message = "structure response images are empty";
        return false;
    }
    if (ctx.structure_data.first.response.size() != ctx.structure_data.second.response.size()) {
        md.message = "structure response images have different sizes";
        return false;
    }

    cv::Mat distDst;
    cv::Mat distSrc;
    if (!structure_points::prepareDistanceMap(ctx.structure_data.second.response, distDst) ||
        !structure_points::prepareDistanceMap(ctx.structure_data.first.response, distSrc)) {
        md.message = "structure response images are empty";
        return false;
    }

    const std::vector<cv::Point2f> pointsSrc =
        structure_points::collectPoints(ctx.structure_data.first.response, _maxPoints);
    const std::vector<cv::Point2f> pointsDst =
        structure_points::collectPoints(ctx.structure_data.second.response, _maxPoints);
    if (pointsSrc.empty() || pointsDst.empty()) {
        md.message = "no structure points found";
        return false;
    }

    double bestScore = std::numeric_limits<double>::infinity();
    cv::Point2d bestShift(0.0, 0.0);
    for (int dy = -_searchRadius; dy <= _searchRadius; dy += _step) {
        for (int dx = -_searchRadius; dx <= _searchRadius; dx += _step) {
            const double forward =
                directedHausdorff(pointsSrc, distDst, cv::Point2i(dx, dy), _percentile);
            double score = forward;
            if (_bidirectional) {
                const double backward =
                    directedHausdorff(pointsDst, distSrc, cv::Point2i(-dx, -dy), _percentile);
                score = std::max(forward, backward);
            }
            if (score < bestScore) {
                bestScore = score;
                bestShift = cv::Point2d(dx, dy);
            }
        }
    }

    md.translation = bestShift;
    md.score = bestScore;
    md.valid = bestScore <= _scoreThreshold;
    if (!md.valid) {
        md.message = "hausdorff score above threshold: " + std::to_string(bestScore);
        IR_LOG_WARN("HausdorffAssociator rejected match: ", md.message);
    }

    IR_LOG_INFO("HausdorffAssociator estimated translation dx=",
                bestShift.x,
                ", dy=",
                bestShift.y,
                ", score=",
                bestScore);
    return md.valid;
}

} // namespace ir
