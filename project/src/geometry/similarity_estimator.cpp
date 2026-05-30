#include "geometry/similarity_estimator.h"

#include <opencv2/calib3d.hpp>
#include <string>
#include <vector>

#include "partial_affine_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

SimilarityEstimator::SimilarityEstimator(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    const std::string method_str = yaml_utils::getString(params, "method", "RANSAC");
    int m = robustMethodFromString(method_str);
    _method = (m < 0) ? cv::RANSAC : m;

    _ransacReprojThreshold = yaml_utils::getDouble(params, "ransacReprojThreshold", 3.0);
    _maxIters = yaml_utils::getInt(params, "maxIters", 2000);
    _confidence = yaml_utils::getDouble(params, "confidence", 0.99);
    _refineIters = yaml_utils::getInt(params, "refineIters", 10);
    _minInliers = yaml_utils::getInt(params, "minInliers", 3);

    IR_LOG_INFO("SimilarityEstimator: method=",
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

bool SimilarityEstimator::estimate(RegistrationContext& ctx) {
    auto& md = ctx.match_data;
    auto& gd = ctx.geometry_data;
    gd.clear();
    gd.type = GeometryType::SIMILARITY;

    if (md.filtered.size() < 2) {
        gd.message = "need at least 2 matches, got " + std::to_string(md.filtered.size());
        IR_LOG_ERROR("SimilarityEstimator: need at least 2 matches, got ", md.filtered.size());
        return false;
    }

    std::vector<cv::Point2f> pts1, pts2;
    partial_affine_utils::extractPoints(ctx, pts1, pts2);

    std::vector<unsigned char> mask;
    cv::Mat A = cv::estimateAffinePartial2D(pts1,
                                            pts2,
                                            mask,
                                            _method,
                                            _ransacReprojThreshold,
                                            static_cast<size_t>(_maxIters),
                                            _confidence,
                                            static_cast<size_t>(_refineIters));

    if (A.empty()) {
        gd.message = "estimateAffinePartial2D returned an empty matrix";
        IR_LOG_ERROR("estimateAffinePartial2D returned an empty matrix.");
        return false;
    }

    partial_affine_utils::promoteInliers(ctx, mask);
    const int inliers = static_cast<int>(md.inliers.size());

    gd.A = A;
    gd.num_inliers = inliers;
    gd.inlier_ratio = md.filtered.empty() ? 0.0 : static_cast<double>(inliers) / md.filtered.size();
    gd.valid = inliers >= _minInliers;
    if (!gd.valid) {
        gd.message =
            partial_affine_utils::rejectMessage("similarity transform", inliers, _minInliers);
        IR_LOG_WARN("SimilarityEstimator rejected model: ", gd.message);
    }

    IR_LOG_INFO("Similarity2D inliers=",
                inliers,
                " / ",
                md.filtered.size(),
                " (ratio=",
                gd.inlier_ratio,
                ")");
    return gd.valid;
}

} // namespace ir
