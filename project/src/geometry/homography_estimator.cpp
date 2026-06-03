#include "geometry/homography_estimator.h"

#include <opencv2/calib3d.hpp>

#include <string>
#include <vector>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

// 从 filtered 中读取匹配点，保证输入已经过前序过滤器裁剪。
void extractPoints(const RegistrationContext& ctx,
                   std::vector<cv::Point2f>& pts1,
                   std::vector<cv::Point2f>& pts2) {
    const auto& fd = ctx.keypoint_data;
    const auto& md = ctx.keypoint_match_data;
    pts1.clear();
    pts2.clear();
    pts1.reserve(md.filtered.size());
    pts2.reserve(md.filtered.size());
    for (const auto& m : md.filtered) {
        pts1.push_back(fd.first.keypoints[m.queryIdx].pt);
        pts2.push_back(fd.second.keypoints[m.trainIdx].pt);
    }
}

// 将鲁棒估计返回的掩码回写到上下文，便于后续可视化和统计模块直接消费。
void promoteInliers(RegistrationContext& ctx, const std::vector<unsigned char>& mask) {
    auto& md = ctx.keypoint_match_data;
    md.inlier_mask = mask;
    md.inliers.clear();
    md.inliers.reserve(mask.size());
    for (size_t i = 0; i < md.filtered.size() && i < mask.size(); ++i) {
        if (mask[i]) {
            md.inliers.push_back(md.filtered[i]);
        }
    }
}

} // namespace

HomographyEstimator::HomographyEstimator(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    const std::string method_str = yaml_utils::getString(params, "method", "RANSAC");
    const int m = robustMethodFromString(method_str);
    _method = (m < 0) ? cv::RANSAC : m;

    _ransacReprojThreshold = yaml_utils::getDouble(params, "ransacReprojThreshold", 3.0);
    _maxIters = yaml_utils::getInt(params, "maxIters", 2000);
    _confidence = yaml_utils::getDouble(params, "confidence", 0.995);
    _minInliers = yaml_utils::getInt(params, "minInliers", 8);

    IR_LOG_INFO("HomographyEstimator: method=",
                method_str,
                ", thr=",
                _ransacReprojThreshold,
                ", maxIters=",
                _maxIters,
                ", confidence=",
                _confidence,
                ", minInliers=",
                _minInliers);
}

bool HomographyEstimator::estimate(RegistrationContext& ctx) {
    auto& md = ctx.keypoint_match_data;
    auto& gd = ctx.geometry_data;

    gd.clear();
    gd.type = GeometryType::HOMOGRAPHY;

    if (md.filtered.size() < 4) {
        gd.message = "need at least 4 matches, got " + std::to_string(md.filtered.size());
        IR_LOG_ERROR("HomographyEstimator: need at least 4 matches, got ", md.filtered.size());
        return false;
    }

    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    extractPoints(ctx, pts1, pts2);

    std::vector<unsigned char> mask;
    cv::Mat H = cv::findHomography(
        pts1, pts2, _method, _ransacReprojThreshold, mask, _maxIters, _confidence);

    if (H.empty()) {
        gd.message = "findHomography returned an empty matrix";
        IR_LOG_ERROR("findHomography returned an empty matrix.");
        return false;
    }

    promoteInliers(ctx, mask);
    const int inliers = static_cast<int>(md.inliers.size());

    gd.H = H;
    gd.num_inliers = inliers;
    gd.inlier_ratio = md.filtered.empty() ? 0.0 : static_cast<double>(inliers) / md.filtered.size();
    gd.valid = inliers >= _minInliers;
    if (!gd.valid) {
        gd.message = "estimated homography with " + std::to_string(inliers) +
                     " inliers, below minInliers=" + std::to_string(_minInliers);
        IR_LOG_WARN("HomographyEstimator rejected model: ", gd.message);
    }

    IR_LOG_INFO("Homography inliers=",
                inliers,
                " / ",
                md.filtered.size(),
                " (ratio=",
                gd.inlier_ratio,
                ")");
    return gd.valid;
}

} // namespace ir
