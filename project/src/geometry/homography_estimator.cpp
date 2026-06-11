#include "geometry/homography_estimator.h"

#include <opencv2/calib3d.hpp>

#include <string>
#include <vector>

#include "data/correspondence_view.h"
#include "geometry/partial_affine_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

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
    auto& gd = ctx.geometry_data;

    gd.clear();
    gd.type = GeometryType::HOMOGRAPHY;

    const CorrespondenceSource source = correspondenceSourceFromContext(ctx);
    const CorrespondenceView view =
        source == CorrespondenceSource::NONE ? buildBestCorrespondenceView(ctx)
                                             : buildCorrespondenceView(ctx, source);
    if (view.filtered.size() < 4) {
        gd.message = "need at least 4 correspondences, got " + std::to_string(view.filtered.size());
        IR_LOG_ERROR("HomographyEstimator: need at least 4 correspondences, got ", view.filtered.size());
        return false;
    }

    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    partial_affine_utils::extractPoints(view, pts1, pts2);

    std::vector<unsigned char> mask;
    cv::Mat H = cv::findHomography(
        pts1, pts2, _method, _ransacReprojThreshold, mask, _maxIters, _confidence);

    if (H.empty()) {
        gd.message = "findHomography returned an empty matrix";
        IR_LOG_ERROR("findHomography returned an empty matrix.");
        return false;
    }

    partial_affine_utils::promoteInliers(ctx, view, mask);
    const int inliers = partial_affine_utils::countInliers(mask);

    gd.H = H;
    gd.num_inliers = inliers;
    gd.inlier_ratio = view.filtered.empty() ? 0.0 : static_cast<double>(inliers) / view.filtered.size();
    gd.valid = inliers >= _minInliers;
    if (!gd.valid) {
        gd.message = "estimated homography with " + std::to_string(inliers) +
                     " inliers, below minInliers=" + std::to_string(_minInliers);
        IR_LOG_WARN("HomographyEstimator rejected model: ", gd.message);
    }

    IR_LOG_INFO("Homography inliers=",
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
