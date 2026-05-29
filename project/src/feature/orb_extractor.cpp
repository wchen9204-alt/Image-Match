#include "feature/orb_extractor.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

OrbExtractor::OrbExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    nfeatures_     = yaml_utils::getInt  (params, "nfeatures",     2000);
    scaleFactor_   = yaml_utils::getFloat(params, "scaleFactor",   1.2f);
    nlevels_       = yaml_utils::getInt  (params, "nlevels",       8);
    edgeThreshold_ = yaml_utils::getInt  (params, "edgeThreshold", 31);
    firstLevel_    = yaml_utils::getInt  (params, "firstLevel",    0);
    WTA_K_         = yaml_utils::getInt  (params, "WTA_K",         2);
    scoreType_     = yaml_utils::getInt  (params, "scoreType",
                                          static_cast<int>(cv::ORB::HARRIS_SCORE));
    patchSize_     = yaml_utils::getInt  (params, "patchSize",     31);
    fastThreshold_ = yaml_utils::getInt  (params, "fastThreshold", 20);

    impl_ = cv::ORB::create(
        nfeatures_,
        scaleFactor_,
        nlevels_,
        edgeThreshold_,
        firstLevel_,
        WTA_K_,
        static_cast<cv::ORB::ScoreType>(scoreType_),
        patchSize_,
        fastThreshold_);

    IR_LOG_INFO("ORB created: nfeatures=", nfeatures_,
                ", scaleFactor=",          scaleFactor_,
                ", nlevels=",              nlevels_,
                ", WTA_K=",                WTA_K_,
                ", patchSize=",            patchSize_);
}

bool OrbExtractor::extract(RegistrationContext& ctx) {
    if (!impl_) {
        IR_LOG_ERROR("ORB extractor not constructed.");
        return false;
    }

    auto& fd = ctx.feature_data;
    fd.type      = FeatureType::ORB;
    fd.norm_type = NormType::HAMMING;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_ERROR("ORB::extract - source images are empty.");
        return false;
    }
    if (fd.first.gray.empty()) {
        cv::cvtColor(fd.first.image, fd.first.gray, cv::COLOR_BGR2GRAY);
    }
    if (fd.second.gray.empty()) {
        cv::cvtColor(fd.second.image, fd.second.gray, cv::COLOR_BGR2GRAY);
    }

    impl_->detectAndCompute(fd.first.gray,  cv::noArray(),
                            fd.first.keypoints,  fd.first.descriptors);
    impl_->detectAndCompute(fd.second.gray, cv::noArray(),
                            fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO("ORB extracted ", fd.first.keypoints.size(),
                " / ",           fd.second.keypoints.size(), " keypoints");
    return !fd.empty();
}

} // namespace ir
