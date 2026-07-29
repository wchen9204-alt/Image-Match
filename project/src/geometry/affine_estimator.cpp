#include "geometry/affine_estimator.h"

#include <opencv2/calib3d.hpp>

#include <string>
#include <vector>

#include "data/correspondence_view.h"
#include "geometry/partial_affine_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

AffineEstimator::AffineEstimator(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    const std::string method_str = yaml_utils::getString(params, "method", "RANSAC");
    const int m = robustMethodFromString(method_str);
    _method = (m < 0) ? cv::RANSAC : m;
    if (_method == cv::USAC_MAGSAC) {
        IR_LOG_WARN("AffineEstimator: estimateAffine2D does not support USAC_MAGSAC, fallback to RANSAC.");
        _method = cv::RANSAC;
    }

    _ransacReprojThreshold = yaml_utils::getDouble(params, "ransacReprojThreshold", 3.0);
    _maxIters = yaml_utils::getInt(params, "maxIters", 2000);
    _confidence = yaml_utils::getDouble(params, "confidence", 0.99);
    _refineIters = yaml_utils::getInt(params, "refineIters", 10);
    _minInliers = yaml_utils::getInt(params, "minInliers", 6);

    IR_LOG_INFO("AffineEstimator: method=",
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

bool AffineEstimator::estimate(RegistrationContext& ctx) {
    auto& gd = ctx.geometry_data;

    gd.clear();
    gd.type = GeometryType::AFFINE;

    const CorrespondenceView view = ensureCorrespondenceView(ctx);
    if (view.filtered.size() < 3) {
        gd.message = "need at least 3 correspondences, got " + std::to_string(view.filtered.size());
        IR_LOG_ERROR("AffineEstimator: need at least 3 correspondences, got ", view.filtered.size());
        return false;
    }

    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    partial_affine_utils::extractPoints(view, pts1, pts2);

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

    partial_affine_utils::promoteInliers(ctx, view, mask);
    const int inliers = partial_affine_utils::countInliers(mask);

    gd.A = A;
    gd.num_inliers = inliers;
    gd.inlier_ratio = view.filtered.empty() ? 0.0 : static_cast<double>(inliers) / view.filtered.size();
    gd.valid = inliers >= _minInliers;
    if (!gd.valid) {
        gd.message = "estimated affine with " + std::to_string(inliers) +
                     " inliers, below minInliers=" + std::to_string(_minInliers);
        IR_LOG_WARN("AffineEstimator rejected model: ", gd.message);
    }

    IR_LOG_INFO("Affine2D inliers=",
                inliers,
                " / ",
                view.filtered.size(),
                " (ratio=",
                gd.inlier_ratio,
                ", source=",
                view.source_name,
                ")");
    return gd.valid;
}

} // namespace ir
