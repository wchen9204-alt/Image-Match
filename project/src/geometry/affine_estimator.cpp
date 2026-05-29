#include "geometry/affine_estimator.h"

#include <opencv2/calib3d.hpp>
#include <string>

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

} // namespace

AffineEstimator::AffineEstimator(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    const std::string method_str =
        yaml_utils::getString(params, "method", "RANSAC");
    int m = robustMethodFromString(method_str);
    _method = (m < 0) ? cv::RANSAC : m;

    _ransacReprojThreshold =
        yaml_utils::getDouble(params, "ransacReprojThreshold", 3.0);
    _maxIters   = yaml_utils::getInt   (params, "maxIters",    2000);
    _confidence = yaml_utils::getDouble(params, "confidence",  0.99);
    _refineIters= yaml_utils::getInt   (params, "refineIters", 10);
    _minInliers = yaml_utils::getInt   (params, "minInliers",  6);

    IR_LOG_INFO("AffineEstimator: method=", method_str,
                ", thr=",         _ransacReprojThreshold,
                ", maxIters=",    _maxIters,
                ", confidence=",  _confidence,
                ", refineIters=", _refineIters,
                ", minInliers=",  _minInliers);
}

bool AffineEstimator::estimate(RegistrationContext& ctx) {
    auto& md = ctx.match_data;
    auto& gd = ctx.geometry_data;
    gd.clear();
    gd.type = GeometryType::AFFINE;

    if (md.filtered.size() < 3) {
        gd.message = "need at least 3 matches, got " +
                     std::to_string(md.filtered.size());
        IR_LOG_ERROR("AffineEstimator: need at least 3 matches, got ",
                     md.filtered.size());
        return false;
    }

    std::vector<cv::Point2f> pts1, pts2;
    extractPoints(ctx, pts1, pts2);

    std::vector<unsigned char> mask;
    cv::Mat A = cv::estimateAffine2D(pts1,
                                     pts2,
                                     mask,
                                     _method,
                                     _ransacReprojThreshold,
                                     static_cast<size_t>(_maxIters),
                                     _confidence,
                                     static_cast<size_t>(_refineIters));

    if (A.empty()) {
        gd.message = "estimateAffine2D returned an empty matrix";
        IR_LOG_ERROR("estimateAffine2D returned an empty matrix.");
        return false;
    }

    promoteInliers(ctx, mask);
    const int inliers = static_cast<int>(md.inliers.size());

    gd.A            = A;
    gd.num_inliers  = inliers;
    gd.inlier_ratio = md.filtered.empty()
                          ? 0.0
                          : static_cast<double>(inliers) / md.filtered.size();
    gd.valid        = inliers >= _minInliers;
    if (!gd.valid) {
        gd.message = "estimated affine with " + std::to_string(inliers) +
                     " inliers, below minInliers=" +
                     std::to_string(_minInliers);
        IR_LOG_WARN("AffineEstimator rejected model: ", gd.message);
    }

    IR_LOG_INFO("Affine2D inliers=", inliers, " / ", md.filtered.size(),
                " (ratio=", gd.inlier_ratio, ")");
    return gd.valid;
}

} // namespace ir

