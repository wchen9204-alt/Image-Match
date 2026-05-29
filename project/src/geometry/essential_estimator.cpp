#include "geometry/essential_estimator.h"

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

cv::Mat buildIntrinsicsFromYaml(const YAML::Node& cam) {
    if (!cam || !cam.IsMap()) return {};

    const auto K_node = cam["K"];
    if (K_node && K_node.IsSequence() && K_node.size() == 9) {
        cv::Mat K(3, 3, CV_64F);
        for (int i = 0; i < 9; ++i) {
            K.at<double>(i / 3, i % 3) = K_node[i].as<double>();
        }
        return K;
    }

    const double focal = yaml_utils::getDouble(cam, "focal", 1000.0);
    std::vector<double> pp = yaml_utils::getVec<double>(cam, "pp", {640.0, 360.0});
    if (pp.size() != 2) pp = {640.0, 360.0};

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = focal;
    K.at<double>(1, 1) = focal;
    K.at<double>(0, 2) = pp[0];
    K.at<double>(1, 2) = pp[1];
    return K;
}

} // namespace

EssentialEstimator::EssentialEstimator(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    const std::string method_str =
        yaml_utils::getString(params, "method", "RANSAC");
    int m = robustMethodFromString(method_str);
    method_     = (m < 0) ? cv::RANSAC : m;
    threshold_  = yaml_utils::getDouble(params, "threshold",  1.0);
    prob_       = yaml_utils::getDouble(params, "prob",       0.999);
    minInliers_ = yaml_utils::getInt   (params, "minInliers", 8);

    K_ = buildIntrinsicsFromYaml(cfg["camera"]);
    if (K_.empty()) {
        K_ = cv::Mat::eye(3, 3, CV_64F);
        K_.at<double>(0, 0) = 1000.0;
        K_.at<double>(1, 1) = 1000.0;
        K_.at<double>(0, 2) = 640.0;
        K_.at<double>(1, 2) = 360.0;
    }

    IR_LOG_INFO("EssentialEstimator: method=", method_str,
                ", threshold=",  threshold_,
                ", prob=",       prob_,
                ", minInliers=", minInliers_);
}

bool EssentialEstimator::estimate(RegistrationContext& ctx) {
    auto& md = ctx.match_data;
    auto& gd = ctx.geometry_data;
    gd.clear();
    gd.type = GeometryType::ESSENTIAL;

    if (md.filtered.size() < 5) {
        IR_LOG_ERROR("EssentialEstimator: need at least 5 matches, got ",
                     md.filtered.size());
        return false;
    }

    std::vector<cv::Point2f> pts1, pts2;
    extractPoints(ctx, pts1, pts2);

    std::vector<unsigned char> mask;
    cv::Mat E = cv::findEssentialMat(pts1, pts2,
                                     K_,
                                     method_,
                                     prob_,
                                     threshold_,
                                     mask);

    if (E.empty()) {
        IR_LOG_ERROR("findEssentialMat returned an empty matrix.");
        return false;
    }

    // 五点法可能返回多个堆叠的 3x3 候选矩阵。
    if (E.rows > 3) E = E.rowRange(0, 3).clone();

    promoteInliers(ctx, mask);
    const int inliers = static_cast<int>(md.inliers.size());

    cv::Mat R, t;
    int recovered = cv::recoverPose(E, pts1, pts2, K_, R, t, mask);

    gd.E            = E;
    gd.R            = R;
    gd.t            = t;
    gd.K            = K_.clone();
    gd.num_inliers  = inliers;
    gd.inlier_ratio = md.filtered.empty()
                          ? 0.0
                          : static_cast<double>(inliers) / md.filtered.size();
    gd.valid        = inliers >= minInliers_ && recovered >= minInliers_;

    IR_LOG_INFO("Essential inliers=", inliers, " / ", md.filtered.size(),
                " (recovered=", recovered, ", ratio=", gd.inlier_ratio, ")");
    return gd.valid;
}

} // namespace ir
