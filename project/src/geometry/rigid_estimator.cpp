#include "geometry/rigid_estimator.h"

#include <opencv2/calib3d.hpp>

#include <string>
#include <vector>

#include "partial_affine_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

RigidEstimator::RigidEstimator(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    const std::string method_str = yaml_utils::getString(params, "method", "RANSAC");
    const int m = robustMethodFromString(method_str);
    _method = (m < 0) ? cv::RANSAC : m;

    _ransacReprojThreshold = yaml_utils::getDouble(params, "ransacReprojThreshold", 3.0);
    _maxIters = yaml_utils::getInt(params, "maxIters", 2000);
    _confidence = yaml_utils::getDouble(params, "confidence", 0.99);
    _refineIters = yaml_utils::getInt(params, "refineIters", 10);
    _minInliers = yaml_utils::getInt(params, "minInliers", 3);

    IR_LOG_INFO("RigidEstimator: method=",
                method_str,
                ", thr=",
                _ransacReprojThreshold,
                ", maxIters=",
                _maxIters,
                ", confidence=",
                _confidence,
                ", refineIters=",
                _refineIters,
                ", minInliers=",
                _minInliers);
}

bool RigidEstimator::estimate(RegistrationContext& ctx) {
    auto& md = ctx.keypoint_match_data;
    auto& gd = ctx.geometry_data;

    gd.clear();
    gd.type = GeometryType::RIGID;

    if (md.filtered.size() < 2) {
        gd.message = "need at least 2 matches, got " + std::to_string(md.filtered.size());
        IR_LOG_ERROR("RigidEstimator: need at least 2 matches, got ", md.filtered.size());
        return false;
    }

    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    partial_affine_utils::extractPoints(ctx, pts1, pts2);

    std::vector<unsigned char> initialMask;
    cv::Mat similarityA = cv::estimateAffinePartial2D(pts1,
                                                      pts2,
                                                      initialMask,
                                                      _method,
                                                      _ransacReprojThreshold,
                                                      static_cast<size_t>(_maxIters),
                                                      _confidence,
                                                      static_cast<size_t>(_refineIters));

    if (similarityA.empty()) {
        gd.message = "estimateAffinePartial2D returned an empty matrix";
        IR_LOG_ERROR("estimateAffinePartial2D returned an empty matrix.");
        return false;
    }

    std::vector<cv::Point2f> inlierPts1;
    std::vector<cv::Point2f> inlierPts2;
    for (size_t i = 0; i < pts1.size() && i < initialMask.size(); ++i) {
        if (!initialMask[i]) {
            continue;
        }
        inlierPts1.push_back(pts1[i]);
        inlierPts2.push_back(pts2[i]);
    }
    if (inlierPts1.size() < 2) {
        gd.message = "rigid refinement needs at least 2 initial inliers, got " +
                     std::to_string(inlierPts1.size());
        IR_LOG_ERROR("RigidEstimator: rigid refinement needs at least 2 initial inliers, got ",
                     inlierPts1.size());
        return false;
    }

    cv::Mat A;
    if (!partial_affine_utils::estimateRigid2D(inlierPts1, inlierPts2, A)) {
        gd.message = "failed to fit rigid transform from inliers";
        IR_LOG_ERROR("RigidEstimator: failed to fit rigid transform from inliers.");
        return false;
    }

    std::vector<unsigned char> mask =
        partial_affine_utils::maskByReprojection(pts1, pts2, A, _ransacReprojThreshold);

    inlierPts1.clear();
    inlierPts2.clear();
    for (size_t i = 0; i < pts1.size() && i < mask.size(); ++i) {
        if (!mask[i]) {
            continue;
        }
        inlierPts1.push_back(pts1[i]);
        inlierPts2.push_back(pts2[i]);
    }
    if (inlierPts1.size() >= 2 &&
        partial_affine_utils::estimateRigid2D(inlierPts1, inlierPts2, A)) {
        mask = partial_affine_utils::maskByReprojection(pts1, pts2, A, _ransacReprojThreshold);
    }

    partial_affine_utils::promoteInliers(ctx, mask);
    const int inliers = static_cast<int>(md.inliers.size());

    gd.A = A;
    gd.num_inliers = inliers;
    gd.inlier_ratio = md.filtered.empty() ? 0.0 : static_cast<double>(inliers) / md.filtered.size();
    gd.valid = inliers >= _minInliers;
    if (!gd.valid) {
        gd.message = partial_affine_utils::rejectMessage("rigid transform", inliers, _minInliers);
        IR_LOG_WARN("RigidEstimator rejected model: ", gd.message);
    }

    IR_LOG_INFO(
        "Rigid2D inliers=", inliers, " / ", md.filtered.size(), " (ratio=", gd.inlier_ratio, ")");
    return gd.valid;
}

} // namespace ir
