#include "geometry/pose_estimator.h"

#include <opencv2/calib3d.hpp>

#include "utils/logger.h"

namespace ir {

int PoseEstimator::recoverPose(RegistrationContext& ctx) {
    const auto& fd = ctx.feature_data;
    auto&       md = ctx.match_data;
    auto&       gd = ctx.geometry_data;

    if (gd.E.empty() || gd.K.empty()) {
        IR_LOG_WARN("PoseEstimator: missing E or K.");
        return 0;
    }

    std::vector<cv::Point2f> p1, p2;
    p1.reserve(md.filtered.size());
    p2.reserve(md.filtered.size());
    for (const auto& m : md.filtered) {
        p1.push_back(fd.first.keypoints [m.queryIdx].pt);
        p2.push_back(fd.second.keypoints[m.trainIdx].pt);
    }
    if (p1.size() < 5) return 0;

    cv::Mat R, t, mask = md.inlier_mask.empty() ? cv::Mat()
                                                : cv::Mat(md.inlier_mask, true);
    int n = cv::recoverPose(gd.E, p1, p2, gd.K, R, t, mask);
    gd.R = R;
    gd.t = t;

    if (mask.rows == static_cast<int>(md.filtered.size())) {
        md.inlier_mask.assign(mask.ptr<unsigned char>(),
                              mask.ptr<unsigned char>() + mask.rows);
        md.inliers.clear();
        md.inliers.reserve(md.inlier_mask.size());
        for (size_t i = 0; i < md.filtered.size(); ++i) {
            if (md.inlier_mask[i]) md.inliers.push_back(md.filtered[i]);
        }
    }
    IR_LOG_INFO("PoseEstimator recovered pose with ", n, " chirality inliers");
    return n;
}

} // namespace ir
