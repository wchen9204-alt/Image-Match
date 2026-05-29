#include "geometry/fundamental_estimator.h"

#include <opencv2/calib3d.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

void extractPoints(const RegistrationContext& ctx,
                   std::vector<cv::Point2f>& pts1,
                   std::vector<cv::Point2f>& pts2) {
    const auto& fd = ctx.feature_data;
    const auto& md = ctx.match_data;
    pts1.clear();
    pts2.clear();
    pts1.reserve(md.filtered.size());
    pts2.reserve(md.filtered.size());
    for (const auto& m : md.filtered) {
        pts1.push_back(fd.first.keypoints [m.queryIdx].pt);
        pts2.push_back(fd.second.keypoints[m.trainIdx].pt);
    }
}

void promoteInliers(RegistrationContext& ctx,
                    const std::vector<unsigned char>& mask) {
    auto& md = ctx.match_data;
    md.inlier_mask = mask;
    md.inliers.clear();
    md.inliers.reserve(mask.size());
    for (size_t i = 0; i < md.filtered.size() && i < mask.size(); ++i) {
        if (mask[i]) md.inliers.push_back(md.filtered[i]);
    }
}

int resolveFundamentalMethod(const std::string& s) {
    int v = robustMethodFromString(s);
    if (v < 0) return cv::FM_RANSAC;
    // robustMethodFromString 已直接返回 findFundamentalMat 可用的 OpenCV 常量。
    return v;
}

} // namespace

FundamentalEstimator::FundamentalEstimator(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    const std::string method_str =
        yaml_utils::getString(params, "method", "RANSAC");
    method_ = resolveFundamentalMethod(method_str);

    ransacReprojThreshold_ =
        yaml_utils::getDouble(params, "ransacReprojThreshold", 3.0);
    confidence_ = yaml_utils::getDouble(params, "confidence", 0.99);
    maxIters_   = yaml_utils::getInt   (params, "maxIters",   2000);
    minInliers_ = yaml_utils::getInt   (params, "minInliers", 8);

    IR_LOG_INFO("FundamentalEstimator: method=", method_str,
                ", thr=",        ransacReprojThreshold_,
                ", confidence=", confidence_,
                ", maxIters=",   maxIters_,
                ", minInliers=", minInliers_);
}

bool FundamentalEstimator::estimate(RegistrationContext& ctx) {
    auto& md = ctx.match_data;
    auto& gd = ctx.geometry_data;
    gd.clear();
    gd.type = GeometryType::FUNDAMENTAL;

    if (md.filtered.size() < 8) {
        IR_LOG_ERROR("FundamentalEstimator: need at least 8 matches, got ",
                     md.filtered.size());
        return false;
    }

    std::vector<cv::Point2f> pts1, pts2;
    extractPoints(ctx, pts1, pts2);

    std::vector<unsigned char> mask;
    // 使用 OpenCV 4.x 通用的 6 参数接口，避免不同小版本的重载差异。
    cv::Mat F = cv::findFundamentalMat(pts1, pts2,
                                       method_,
                                       ransacReprojThreshold_,
                                       confidence_,
                                       mask);
    (void)maxIters_;

    if (F.empty()) {
        IR_LOG_ERROR("findFundamentalMat returned an empty matrix.");
        return false;
    }

    // 七点法可能返回多个解，这里取顶部 3x3 矩阵。
    if (F.rows > 3) F = F.rowRange(0, 3).clone();

    promoteInliers(ctx, mask);
    const int inliers = static_cast<int>(md.inliers.size());

    gd.F            = F;
    gd.num_inliers  = inliers;
    gd.inlier_ratio = md.filtered.empty()
                          ? 0.0
                          : static_cast<double>(inliers) / md.filtered.size();
    gd.valid        = inliers >= minInliers_;

    IR_LOG_INFO("Fundamental inliers=", inliers, " / ", md.filtered.size(),
                " (ratio=", gd.inlier_ratio, ")");
    return gd.valid;
}

} // namespace ir
