#include "matcher/structure/chamfer_associator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "matcher/structure/structure_point_set.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

// 计算给定平移下 source 点到 target 距离图的平均倒角距离。
double scoreTranslation(const std::vector<cv::Point2f>& srcPoints,
                        const cv::Mat& target,
                        const cv::Point2i& shift) {
    if (srcPoints.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    double sum = 0.0;
    int count = 0;
    for (const auto& p : srcPoints) {
        const int x = cvRound(p.x) + shift.x;
        const int y = cvRound(p.y) + shift.y;
        if (x < 0 || y < 0 || x >= target.cols || y >= target.rows) {
            continue;
        }
        sum += static_cast<double>(target.at<float>(y, x));
        ++count;
    }
    if (count == 0) {
        return std::numeric_limits<double>::infinity();
    }

    return sum / static_cast<double>(count);
}

} // namespace

ChamferAssociator::ChamferAssociator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _searchRadius = yaml_utils::getInt(params, "searchRadius", 20);
    _step = std::max(1, yaml_utils::getInt(params, "step", 1));
    _maxPoints = yaml_utils::getInt(params, "maxPoints", 2000);
    _scoreThreshold = yaml_utils::getDouble(params, "scoreThreshold", 0.25);
    _bidirectional = yaml_utils::getBool(params, "bidirectional", true);
}

bool ChamferAssociator::associate(RegistrationContext& ctx) {
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

    cv::Point2d centerShift(0.0, 0.0);

    // 倒角搜索围绕零平移展开，由搜索半径直接覆盖允许的平移范围。
    double bestScore = std::numeric_limits<double>::infinity();
    cv::Point2d bestShift = centerShift;
    for (int dy = -_searchRadius; dy <= _searchRadius; dy += _step) {
        for (int dx = -_searchRadius; dx <= _searchRadius; dx += _step) {
            const cv::Point2i shift(cvRound(centerShift.x) + dx, cvRound(centerShift.y) + dy);
            const double forward = scoreTranslation(pointsSrc, distDst, shift);
            double score = forward;
            if (_bidirectional) {
                const double backward = scoreTranslation(pointsDst, distSrc, -shift);
                score = 0.5 * (forward + backward);
            }
            if (score < bestScore) {
                bestScore = score;
                bestShift = cv::Point2d(shift.x, shift.y);
            }
        }
    }

    md.translation = bestShift;
    md.score = bestScore;
    md.valid = bestScore <= _scoreThreshold;
    if (!md.valid) {
        md.message = "chamfer score above threshold: " + std::to_string(bestScore);
        IR_LOG_WARN("ChamferAssociator rejected match: ", md.message);
    }

    IR_LOG_INFO("ChamferAssociator estimated translation dx=",
                bestShift.x,
                ", dy=",
                bestShift.y,
                ", score=",
                bestScore,
                ", initial dx=",
                centerShift.x,
                ", initial dy=",
                centerShift.y);
    return md.valid;
}

} // namespace ir

