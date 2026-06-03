#include "matcher/structure/chamfer_associator.h"

#include <algorithm>
#include <cctype>
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

// 将字符串参数归一化为大写，兼容 phase_correlate / PHASE_CORRELATE 等写法。
std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

// 使用相位相关估计全局初始平移，倒角匹配再围绕该初值做局部精搜。
bool estimatePhaseShift(const cv::Mat& first,
                        const cv::Mat& second,
                        int blurKernel,
                        cv::Point2d& shift) {
    cv::Mat src = structure_points::toGray32F(first);
    cv::Mat dst = structure_points::toGray32F(second);
    if (src.empty() || dst.empty() || src.size() != dst.size() || cv::countNonZero(src) == 0 ||
        cv::countNonZero(dst) == 0) {
        return false;
    }

    if (blurKernel >= 3) {
        if (blurKernel % 2 == 0) {
            ++blurKernel;
        }
        cv::GaussianBlur(src, src, cv::Size(blurKernel, blurKernel), 0.0);
        cv::GaussianBlur(dst, dst, cv::Size(blurKernel, blurKernel), 0.0);
    }

    double response = 0.0;
    shift = cv::phaseCorrelate(src, dst, cv::noArray(), &response);
    return std::isfinite(shift.x) && std::isfinite(shift.y);
}

} // namespace

ChamferAssociator::ChamferAssociator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _searchRadius = yaml_utils::getInt(params, "searchRadius", 20);
    _step = std::max(1, yaml_utils::getInt(params, "step", 1));
    _maxPoints = yaml_utils::getInt(params, "maxPoints", 2000);
    _phaseBlurKernel = yaml_utils::getInt(params, "phaseBlurKernel", 5);
    _scoreThreshold = yaml_utils::getDouble(params, "scoreThreshold", 0.25);
    _bidirectional = yaml_utils::getBool(params, "bidirectional", true);
    _initialization = upper(yaml_utils::getString(params, "initialization", "PHASE_CORRELATE"));
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
    if (_initialization == "PHASE_CORRELATE") {
        if (!estimatePhaseShift(ctx.structure_data.first.response,
                                ctx.structure_data.second.response,
                                _phaseBlurKernel,
                                centerShift)) {
            IR_LOG_WARN("ChamferAssociator phase initialization failed; fallback to zero shift.");
        }
    }

    // 倒角搜索只在初值附近展开，避免真实位移超过 searchRadius 时被限制在原点附近。
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
